"""Prototype: circuit catalog, spectral features, shortlist + fitting + BIC.

This module does not alter existing pipelines; it offers a sandbox to explore
multiple EIS equivalent circuits and choose among them using a small heuristic
shortlist plus statistical criteria (BIC/AIC).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Element primitives
# ---------------------------------------------------------------------------


def _cpe(omega: np.ndarray, Q: float, n: float) -> np.ndarray:
    return 1.0 / (Q * (1j * omega) ** n)


def _warburg(omega: np.ndarray, sigma: float) -> np.ndarray:
    return sigma / np.sqrt(1j * omega)


def _inductor(omega: np.ndarray, L: float) -> np.ndarray:
    return 1j * omega * L


# ---------------------------------------------------------------------------
# Circuit template and catalog
# ---------------------------------------------------------------------------


@dataclass
class CircuitTemplate:
    name: str
    param_names: List[str]
    bounds: Tuple[List[float], List[float]]
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    init_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    diagram: str
    description: str = ""
    physical_meaning: Optional[Dict[str, str]] = None
    typical_systems: Optional[List[str]] = None


def _randles_cpe_warburg() -> CircuitTemplate:
    # Rs + (Rp || CPE) + W
    param_names = ["Rs", "Rp", "Q", "n", "Sigma"]
    bounds = (
        [1e-6, 1e-3, 1e-12, 0.3, 1e-10],
        [1e6, 1e8, 1.0, 1.0, 1e5],
    )

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp, Q, n, sigma = p
        Zcpe = _cpe(omega, Q, n)
        Zw = _warburg(omega, sigma)
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
        return Rs + Zpar + Zw

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = float(np.nanmin(z.real[-5:])) if z.size >= 5 else float(z.real[-1])
        rp = float(max(z.real[0] - rs, 0.1))
        return np.array([max(rs, 1e-3), rp, 1e-4, 0.85, 0.01])

    diagram = "Rs - (Rp || CPE) - W"
    return CircuitTemplate("Randles-CPE-W", param_names, bounds, model, init, diagram)


def _double_arc_cpe() -> CircuitTemplate:
    # Rs + (Rp1 || CPE1) + (Rp2 || CPE2)
    param_names = ["Rs", "Rp1", "Q1", "n1", "Rp2", "Q2", "n2"]
    bounds = (
        [1e-6, 1e-3, 1e-12, 0.3, 1e-3, 1e-12, 0.3],
        [1e6, 1e8, 1.0, 1.0, 1e8, 1.0, 1.0],
    )

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp1, Q1, n1, Rp2, Q2, n2 = p
        Zcpe1 = _cpe(omega, Q1, n1)
        Zcpe2 = _cpe(omega, Q2, n2)
        Zpar1 = 1.0 / (1.0 / Rp1 + 1.0 / Zcpe1)
        Zpar2 = 1.0 / (1.0 / Rp2 + 1.0 / Zcpe2)
        return Rs + Zpar1 + Zpar2

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = float(np.nanmin(z.real[-5:])) if z.size >= 5 else float(z.real[-1])
        span = float(max(z.real.max() - z.real.min(), 0.1))
        rp1 = span * 0.6
        rp2 = span * 0.4
        return np.array([max(rs, 1e-3), rp1, 1e-4, 0.85, rp2, 5e-5, 0.8])

    diagram = "Rs - (Rp1 || CPE1) - (Rp2 || CPE2)"
    return CircuitTemplate("Two-Arc-CPE", param_names, bounds, model, init, diagram)


def _inductive_loop() -> CircuitTemplate:
    # Rs + L + (Rp || CPE)
    param_names = ["Rs", "L", "Rp", "Q", "n"]
    bounds = (
        [1e-6, 1e-9, 1e-3, 1e-12, 0.3],
        [1e6, 1.0, 1e8, 1.0, 1.0],
    )

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, L, Rp, Q, n = p
        Zl = _inductor(omega, L)
        Zcpe = _cpe(omega, Q, n)
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
        return Rs + Zl + Zpar

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = float(np.nanmin(z.real[-5:])) if z.size >= 5 else float(z.real[-1])
        rp = float(max(z.real.max() - rs, 0.1))
        return np.array([max(rs, 1e-3), 1e-3, rp, 1e-4, 0.9])

    diagram = "Rs - L - (Rp || CPE)"
    return CircuitTemplate("Inductive-CPE", param_names, bounds, model, init, diagram)


def circuit_catalog() -> List[CircuitTemplate]:
    """Return all registered circuits (delegates to CircuitRegistry)."""
    try:
        from src.circuit_registry import CircuitRegistry
        return CircuitRegistry.all()
    except Exception:  # pragma: no cover — fallback if registry not available
        return [
            _randles_cpe_warburg(),
            _double_arc_cpe(),
            _inductive_loop(),
        ]


# ---------------------------------------------------------------------------
# Spectral feature extractor (for ML/heuristics)
# ---------------------------------------------------------------------------


def extract_eis_features_for_ml(df) -> Dict[str, float]:
    """Lightweight spectral descriptors to guide circuit shortlist."""
    freq = df["frequency"].to_numpy()
    z = df["zreal"].to_numpy() + 1j * df["zimag"].to_numpy()

    # ensure sorted by frequency ascending
    idx = np.argsort(freq)
    freq = freq[idx]
    z = z[idx]

    logf = np.log10(freq)
    mag = np.abs(z)
    phase = np.unwrap(np.angle(z, deg=True))

    def slope(x, y):
        if len(x) < 2:
            return np.nan
        return np.polyfit(x, y, 1)[0]

    n = len(freq)
    q = max(int(0.2 * n), 2)
    low_slice = slice(0, q)
    high_slice = slice(n - q, n)

    feats = {
        "logf_slope_low": slope(logf[low_slice], np.log10(mag[low_slice])) if n >= 2 else np.nan,
        "logf_slope_high": slope(logf[high_slice], np.log10(mag[high_slice])) if n >= 2 else np.nan,
        "phase_min": float(np.nanmin(phase)) if phase.size else np.nan,
        "phase_max": float(np.nanmax(phase)) if phase.size else np.nan,
        "phase_range": float(np.nanmax(phase) - np.nanmin(phase)) if phase.size else np.nan,
        "freq_at_phase_min": float(freq[np.nanargmin(phase)]) if phase.size else np.nan,
        "mag_range": float(np.nanmax(mag) - np.nanmin(mag)) if mag.size else np.nan,
        "zreal_min": float(np.nanmin(z.real)) if z.size else np.nan,
        "zreal_max": float(np.nanmax(z.real)) if z.size else np.nan,
    }
    return feats


# ---------------------------------------------------------------------------
# Shortlist heuristic (placeholder for ML classifier)
# ---------------------------------------------------------------------------


def shortlist_circuits(
    features: Dict[str, float],
    catalog: List[CircuitTemplate],
    top_n: int = 3,
    ml_ranked: Optional[List[str]] = None,
) -> List[CircuitTemplate]:
    """Pick a small set of circuits based on rules or ML predictions.

    When *ml_ranked* is a non-empty list of circuit names (from
    :class:`~src.ml_circuit_selector.CircuitMLSelector`) those are used
    directly instead of the rule-based heuristic.
    """
    catalog_map = {c.name: c for c in catalog}

    # ── ML path ──────────────────────────────────────────────────
    if ml_ranked:
        picks = [catalog_map[n] for n in ml_ranked if n in catalog_map]
        if picks:
            logger.info("shortlist_circuits: using ML-ranked %s", [c.name for c in picks[:top_n]])
            return picks[:top_n]
        logger.debug("shortlist_circuits: ML names not in catalog, falling back to heuristic.")

    # ── Heuristic path (unchanged) ───────────────────────────────
    picks: List[CircuitTemplate] = []
    names_available = {c.name for c in catalog}

    def add_by_name(name: str):
        for c in catalog:
            if c.name == name and c not in picks:
                picks.append(c)
                break

    # Always include a baseline Randles
    add_by_name("Randles-CPE-W")

    phase_min = features.get("phase_min", np.nan)
    phase_max = features.get("phase_max", np.nan)
    slope_low = features.get("logf_slope_low", np.nan)
    mag_range = features.get("mag_range", np.nan)

    # Strong capacitive arc → more likely two arcs or coating
    if np.isfinite(phase_min) and phase_min < -70:
        add_by_name("Two-Arc-CPE")
        if "Coating-CPE" in names_available:
            add_by_name("Coating-CPE")

    # Diffusion tail: slope ~ -0.5 in low freq magnitude
    if np.isfinite(slope_low) and -0.8 < slope_low < -0.2:
        add_by_name("Randles-CPE-W")  # already present, safe
        if "Warburg-Finite" in names_available:
            add_by_name("Warburg-Finite")

    # Very wide mag_range + two arcs → ZARC-ZARC-W
    if (np.isfinite(mag_range) and mag_range > 100
            and np.isfinite(phase_min) and phase_min < -50):
        if "ZARC-ZARC-W" in names_available:
            add_by_name("ZARC-ZARC-W")

    # Inductive loop if phase crosses positive or mag rises at high freq
    if np.isfinite(phase_max) and phase_max > 10:
        add_by_name("Inductive-CPE")

    # Always include Simple-RC as BIC baseline if few picks so far
    if len(picks) < top_n and "Simple-RC" in names_available:
        add_by_name("Simple-RC")

    if not picks and catalog:
        picks.append(catalog[0])

    return picks[:top_n]


# ---------------------------------------------------------------------------
# Fitting and model selection
# ---------------------------------------------------------------------------


def _bic_aic(rss: float, n: int, k: int) -> Tuple[float, float]:
    rss = float(rss)
    if n <= 0 or rss <= 0:
        return np.inf, np.inf
    bic = n * np.log(rss / n) + k * np.log(n)
    aic = n * np.log(rss / n) + 2 * k
    return bic, aic


def fit_template(template: CircuitTemplate, freq: np.ndarray, z: np.ndarray) -> Dict:
    omega = 2 * np.pi * freq

    def residuals(p: np.ndarray) -> np.ndarray:
        z_model = template.model_fn(p, omega)
        return np.concatenate([(z_model.real - z.real), (z_model.imag - z.imag)])

    p0_base = template.init_fn(omega, z)
    lb, ub = np.array(template.bounds[0]), np.array(template.bounds[1])

    # ── Adaptive bounds: widen Rs/Rp lower bound based on data ────
    # Estimate physical Rs from high-freq real intercept
    z_real_hf = np.nanmin(z.real[-5:]) if len(z) >= 5 else z.real[-1]
    if z_real_hf > 0:
        # Set Rs lower bound to 10% of observed high-freq intercept
        # (prevents optimizer from collapsing to 1e-6)
        rs_floor = max(float(z_real_hf) * 0.1, 1e-6)
        if "Rs" in template.param_names:
            idx_rs = template.param_names.index("Rs")
            lb[idx_rs] = rs_floor

    # ── Multi-start with diverse seeds ────────────────────────────
    rng = np.random.RandomState(42)
    seeds = [p0_base]

    # Scaled perturbations around base
    for scale in (0.5, 0.8, 1.2, 2.0):
        p = np.clip(p0_base * scale, lb, ub)
        seeds.append(p)

    # Log-uniform random seeds within bounds
    if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
        safe_lb = np.maximum(lb, 1e-15)
        safe_ub = np.maximum(ub, 1e-14)
        for _ in range(5):
            log_p = rng.uniform(np.log10(safe_lb), np.log10(safe_ub))
            seeds.append(np.clip(10 ** log_p, lb, ub))

    best_res = None
    for p0 in seeds:
        try:
            res = least_squares(
                residuals,
                p0,
                bounds=(lb, ub),
                max_nfev=8000,
            )
            if best_res is None or res.cost < best_res.cost:
                best_res = res
        except Exception:
            continue

    # ── Retry with relaxed bounds if parameters hit limits ────────
    if best_res is not None:
        tol = 1e-4
        hit_lb = np.abs(best_res.x - lb) < tol * (ub - lb + 1e-30)
        hit_ub = np.abs(best_res.x - ub) < tol * (ub - lb + 1e-30)
        if np.any(hit_lb) or np.any(hit_ub):
            lb2 = lb.copy()
            ub2 = ub.copy()
            lb2[hit_lb] = lb[hit_lb] / 10.0
            ub2[hit_ub] = ub[hit_ub] * 10.0
            for p0 in seeds[:3]:
                p0_clipped = np.clip(p0, lb2, ub2)
                try:
                    res2 = least_squares(
                        residuals, p0_clipped,
                        bounds=(lb2, ub2), max_nfev=8000,
                    )
                    if res2.cost < best_res.cost:
                        best_res = res2
                        lb, ub = lb2, ub2  # use relaxed for bound-hit check
                except Exception:
                    continue

    res = best_res
    rss = float(np.sum(res.fun**2))
    n_points = len(res.fun)
    k_params = len(p0_base)
    bic, aic = _bic_aic(rss, n_points, k_params)

    # Covariance and parameter std dev
    params_std = {}
    try:
        if res.jac is not None and res.jac.shape[0] > res.jac.shape[1]:
            jtj = res.jac.T @ res.jac
            inv = np.linalg.inv(jtj)
            sigma2 = rss / max(n_points - k_params, 1)
            cov = inv * sigma2
            params_std = {
                name: float(np.sqrt(abs(cov[i, i]))) for i, name in enumerate(template.param_names)
            }
    except LinAlgError:
        params_std = {}

    # Residual diagnostics
    fun = res.fun
    # Runs/structure via lag-1 autocorrelation on residuals real/imag stacked
    r_auto = np.nan
    if fun.size > 3:
        r_auto = float(np.corrcoef(fun[:-1], fun[1:])[0, 1])
    structured = bool(abs(r_auto) > 0.3) if np.isfinite(r_auto) else False

    # Bound hits
    hits = []
    for val, lo, hi in zip(res.x, lb, ub):
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            continue
        if val <= lo + 0.01 * span or val >= hi - 0.01 * span:
            hits.append(True)
    bound_hits = len(hits)

    return {
        "template": template.name,
        "diagram": template.diagram,
        "params": {k: v for k, v in zip(template.param_names, res.x)},
        "params_std": params_std,
        "success": bool(res.success),
        "message": res.message,
        "rss": rss,
        "bic": bic,
        "aic": aic,
        "n_params": k_params,
        "n_points": len(freq),
        "res_autocorr": r_auto,
        "res_structured": structured,
        "bound_hits": bound_hits,
    }


def _save_diagnostics(
    sample_name: str,
    freq: np.ndarray,
    z: np.ndarray,
    template: CircuitTemplate,
    params: Dict[str, float],
    out_dir: str = "outputs/figures/circuits",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    omega = 2 * np.pi * freq
    p = np.array([params[k] for k in template.param_names])
    z_model = template.model_fn(p, omega)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Nyquist
    axes[0].plot(z.real, -z.imag, "o", label="Dados", markersize=4)
    axes[0].plot(z_model.real, -z_model.imag, "-", label="Ajuste")
    axes[0].set_xlabel("Z' (Ohm)")
    axes[0].set_ylabel("-Z'' (Ohm)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Bode magnitude e fase
    axes[1].semilogx(freq, np.abs(z), "o", label="|Z| dados", markersize=3)
    axes[1].semilogx(freq, np.abs(z_model), "-", label="|Z| ajuste")
    axes[1].set_xlabel("Frequencia (Hz)")
    axes[1].set_ylabel("|Z| (Ohm)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    phase_data = np.unwrap(np.angle(z, deg=True))
    phase_model = np.unwrap(np.angle(z_model, deg=True))
    axes[2].semilogx(freq, phase_data, "o", label="Fase dados", markersize=3)
    axes[2].semilogx(freq, phase_model, "-", label="Fase ajuste")
    axes[2].set_xlabel("Frequencia (Hz)")
    axes[2].set_ylabel("Fase (graus)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fname = f"{sample_name}_{template.name}.png" if sample_name else f"diag_{template.name}.png"
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def run_shortlist_fit(
    df,
    sample_name: Optional[str] = None,
    save_plots: bool = False,
    plots_dir: str = "outputs/figures/circuits",
    ml_ranked: Optional[List[str]] = None,
) -> Dict:
    """End-to-end: extract features, shortlist circuits, fit each, rank by BIC.

    Parameters
    ----------
    sample_name : str | None
        Used to name diagnostic plots.
    save_plots : bool
        Save Nyquist/Bode of the best circuit.
    plots_dir : str
        Directory for diagnostic plots.
    ml_ranked : list[str] | None
        ML-predicted circuit names (best first).  When provided, the
        shortlist heuristic is bypassed.
    """
    if len(df) < 3:
        raise ValueError("Poucos pontos para fitting")

    feats = extract_eis_features_for_ml(df)
    catalog = circuit_catalog()
    catalog_map = {c.name: c for c in catalog}
    short = shortlist_circuits(feats, catalog, top_n=3, ml_ranked=ml_ranked)

    freq = df["frequency"].to_numpy()
    z = df["zreal"].to_numpy() + 1j * df["zimag"].to_numpy()

    results = []
    for tmpl in short:
        try:
            res = fit_template(tmpl, freq, z)
        except Exception as exc:
            res = {
                "template": tmpl.name,
                "diagram": tmpl.diagram,
                "success": False,
                "message": str(exc),
                "rss": np.inf,
                "bic": np.inf,
                "aic": np.inf,
                "params": {},
                "n_params": len(tmpl.param_names),
                "n_points": len(freq),
            }
        results.append(res)

    # Penalize structured residuals and bound hits in ranking
    def penalized_bic(r):
        penalty = 0.0
        if r.get("res_structured"):
            penalty += 5.0
        penalty += 0.5 * r.get("bound_hits", 0)
        bic = r.get("bic", np.inf)
        return bic + penalty

    results_sorted = sorted(results, key=lambda r: (penalized_bic(r), r.get("rss", np.inf)))

    # Confidence estimate via softmax over -0.5 * penalized BIC (relative)
    finite_bic = [penalized_bic(r) for r in results_sorted if np.isfinite(penalized_bic(r))]
    if finite_bic:
        bmin = min(finite_bic)
        scores = []
        for r in results_sorted:
            pb = penalized_bic(r)
            if np.isfinite(pb):
                scores.append(np.exp(-0.5 * (pb - bmin)))
            else:
                scores.append(0.0)
        total = sum(scores) or 1.0
        for r, s in zip(results_sorted, scores):
            r["confidence"] = s / total
            r["bic_penalized"] = penalized_bic(r)
    else:
        for r in results_sorted:
            r["confidence"] = np.nan
            r["bic_penalized"] = np.nan

    best = results_sorted[0] if results_sorted else None

    diag_path = None
    if save_plots and best and best.get("success"):
        tmpl = catalog_map.get(best.get("template"))
        if tmpl and best.get("params"):
            try:
                diag_path = _save_diagnostics(
                    sample_name or "sample",
                    freq,
                    z,
                    tmpl,
                    best["params"],
                    out_dir=plots_dir,
                )
                best["diagnostic_plot"] = diag_path
            except Exception:
                best["diagnostic_plot"] = None

    return {
        "features": feats,
        "shortlist": [t.name for t in short],
        "results": results_sorted,
        "best": best,
        "diagnostic_plot": diag_path,
    }
