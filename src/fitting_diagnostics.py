"""Fitting diagnostics — rich visual feedback for every circuit fit.

Generates publication-quality diagnostic plots for EIS circuit fitting:

1. **Nyquist overlay** — measured data + model fit + residuals (3 panels)
2. **Bode overlay** — |Z| and phase with fit curves
3. **Residual analysis** — residuals vs frequency + histogram + QQ-plot
4. **Parameter confidence** — error bars with parameter bounds
5. **Model comparison** — BIC bar chart of candidates with confidence

Each plot set includes a **traffic-light quality indicator**:

- 🟢 **Excellent**: low RSS, non-structured residuals, no bound hits
- 🟡 **Acceptable**: moderate RSS or 1–2 bound hits
- 🔴 **Problematic**: structured residuals, many bound hits, or high RSS

Day 9 of the UPGRADE_PLAN_v0.2.0 schedule.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# =====================================================================
# Traffic-light quality assessment
# =====================================================================

# Quality thresholds (can be overridden via FittingDiagnostics constructor)
_DEFAULT_THRESHOLDS = {
    "rss_excellent": 0.01,   # normalised RSS (RSS / n_points)
    "rss_acceptable": 0.10,
    "autocorr_threshold": 0.3,  # |autocorrelation| above → structured
    "bound_hits_acceptable": 2,
    "bound_hits_problematic": 4,
}


@dataclass
class QualityIndicator:
    """Traffic-light quality assessment of a fit result.

    Attributes
    ----------
    level : str
        ``"green"``, ``"yellow"`` or ``"red"``.
    emoji : str
        ``"🟢"``, ``"🟡"`` or ``"🔴"``.
    label : str
        Human-readable label (e.g. ``"Fit excelente"``).
    reasons : list[str]
        Explanation lines for the assigned level.
    """

    level: str = "green"
    emoji: str = "🟢"
    label: str = "Fit excelente"
    reasons: List[str] = field(default_factory=list)


def assess_quality(
    fit_result: Dict[str, Any],
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> QualityIndicator:
    """Evaluate a fit result and return a :class:`QualityIndicator`.

    Parameters
    ----------
    fit_result : dict
        As returned by :func:`~src.circuit_fitting.fit_template` or a
        single entry from ``run_shortlist_fit()["results"]``.
    thresholds : dict | None
        Override default quality thresholds.
    """
    th = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons: List[str] = []

    rss = fit_result.get("rss", np.inf)
    n_pts = fit_result.get("n_points", 1)
    norm_rss = rss / max(n_pts, 1)

    autocorr = fit_result.get("res_autocorr", 0.0)
    structured = fit_result.get("res_structured", False)
    bound_hits = fit_result.get("bound_hits", 0)
    success = fit_result.get("success", False)

    # Start at green, downgrade as problems are found
    level = "green"

    # ── RSS check ────────────────────────────────────────────────
    if norm_rss <= th["rss_excellent"]:
        reasons.append(f"RSS normalizado baixo ({norm_rss:.4f})")
    elif norm_rss <= th["rss_acceptable"]:
        level = _downgrade(level, "yellow")
        reasons.append(f"RSS normalizado moderado ({norm_rss:.4f})")
    else:
        level = _downgrade(level, "red")
        reasons.append(f"RSS normalizado alto ({norm_rss:.4f})")

    # ── Structured residuals ─────────────────────────────────────
    if structured or (np.isfinite(autocorr) and abs(autocorr) > th["autocorr_threshold"]):
        level = _downgrade(level, "red")
        reasons.append(
            f"Resíduos estruturados (autocorrelação lag-1 = {autocorr:.3f})"
        )
    else:
        reasons.append("Resíduos não-estruturados")

    # ── Bound hits ───────────────────────────────────────────────
    if bound_hits >= th["bound_hits_problematic"]:
        level = _downgrade(level, "red")
        reasons.append(f"{bound_hits} parâmetros nos limites (bound hits)")
    elif bound_hits >= th["bound_hits_acceptable"]:
        level = _downgrade(level, "yellow")
        reasons.append(f"{bound_hits} parâmetros próximos dos limites")
    else:
        reasons.append("Nenhum parâmetro nos limites")

    # ── Convergence ──────────────────────────────────────────────
    if not success:
        level = _downgrade(level, "yellow")
        reasons.append("Optimizador não convergiu completamente")

    emoji_map = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    label_map = {
        "green": "Fit excelente",
        "yellow": "Fit aceitável",
        "red": "Fit problemático",
    }
    return QualityIndicator(
        level=level,
        emoji=emoji_map[level],
        label=label_map[level],
        reasons=reasons,
    )


def _downgrade(current: str, to: str) -> str:
    """Downgrade quality level (green > yellow > red)."""
    order = {"green": 0, "yellow": 1, "red": 2}
    return to if order.get(to, 0) > order.get(current, 0) else current


# =====================================================================
# FittingDiagnostics — main class
# =====================================================================

class FittingDiagnostics:
    """Generate diagnostic visualisations for circuit fitting results.

    Parameters
    ----------
    out_dir : str
        Directory for saved figures.
    dpi : int
        Resolution for saved figures.
    thresholds : dict | None
        Override default quality thresholds.

    Examples
    --------
    >>> diag = FittingDiagnostics(out_dir="outputs/figures/diagnostics")
    >>> paths = diag.generate_all(
    ...     sample_name="sample_01",
    ...     freq=freq,
    ...     z_data=z,
    ...     fit_result=best,
    ...     all_results=results_sorted,
    ...     template=template,
    ... )
    """

    def __init__(
        self,
        out_dir: str = "outputs/figures/diagnostics",
        dpi: int = 150,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.out_dir = out_dir
        self.dpi = dpi
        self.thresholds = thresholds

    # ── Public: generate everything ──────────────────────────────

    def generate_all(
        self,
        sample_name: str,
        freq: np.ndarray,
        z_data: np.ndarray,
        fit_result: Dict[str, Any],
        all_results: Optional[List[Dict[str, Any]]] = None,
        template: Optional[Any] = None,
    ) -> Dict[str, Optional[str]]:
        """Generate all diagnostic plots.

        Returns
        -------
        dict
            ``{plot_name: file_path_or_None}`` for each diagnostic.
        """
        os.makedirs(self.out_dir, exist_ok=True)
        omega = 2.0 * np.pi * freq

        # Compute model impedance
        z_model = None
        if template is not None and fit_result.get("params"):
            p = np.array([
                fit_result["params"][k]
                for k in template.param_names
            ])
            z_model = template.model_fn(p, omega)

        quality = assess_quality(fit_result, thresholds=self.thresholds)

        paths: Dict[str, Optional[str]] = {}

        # 1. Nyquist overlay
        paths["nyquist"] = self._plot_nyquist_overlay(
            sample_name, freq, z_data, z_model, quality,
        )

        # 2. Bode overlay
        paths["bode"] = self._plot_bode_overlay(
            sample_name, freq, z_data, z_model, quality,
        )

        # 3. Residual analysis
        paths["residuals"] = self._plot_residual_analysis(
            sample_name, freq, z_data, z_model, quality,
        )

        # 4. Parameter confidence
        paths["param_confidence"] = self._plot_param_confidence(
            sample_name, fit_result, template, quality,
        )

        # 5. Model comparison
        paths["model_comparison"] = self._plot_model_comparison(
            sample_name, all_results, quality,
        )

        # 6. Quality indicator
        paths["quality"] = quality

        logger.info(
            "FittingDiagnostics: %s — %s %s (%d plots saved)",
            sample_name,
            quality.emoji,
            quality.label,
            sum(1 for v in paths.values() if isinstance(v, str)),
        )
        return paths

    # ── 1. Nyquist overlay ───────────────────────────────────────

    def _plot_nyquist_overlay(
        self,
        sample_name: str,
        freq: np.ndarray,
        z_data: np.ndarray,
        z_model: Optional[np.ndarray],
        quality: QualityIndicator,
    ) -> Optional[str]:
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Panel 1: Nyquist data + fit
            ax = axes[0]
            ax.plot(z_data.real, -z_data.imag, "o", color="#2196F3",
                    markersize=4, label="Dados", alpha=0.7)
            if z_model is not None:
                ax.plot(z_model.real, -z_model.imag, "-", color="#E53935",
                        linewidth=2, label="Ajuste")
            ax.set_xlabel("Z' (Ω)")
            ax.set_ylabel("-Z'' (Ω)")
            ax.set_title("Nyquist — Dados vs Ajuste")
            ax.legend(fontsize=8)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.3)

            # Panel 2: Real residuals
            ax = axes[1]
            if z_model is not None:
                res_real = z_data.real - z_model.real
                ax.plot(freq, res_real, "o-", markersize=3, color="#4CAF50")
                ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Frequência (Hz)")
            ax.set_ylabel("ΔZ' (Ω)")
            ax.set_title("Resíduo — Parte Real")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)

            # Panel 3: Imaginary residuals
            ax = axes[2]
            if z_model is not None:
                res_imag = z_data.imag - z_model.imag
                ax.plot(freq, res_imag, "o-", markersize=3, color="#FF9800")
                ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Frequência (Hz)")
            ax.set_ylabel("ΔZ'' (Ω)")
            ax.set_title("Resíduo — Parte Imaginária")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)

            fig.suptitle(
                f"{sample_name}  {quality.emoji} {quality.label}",
                fontsize=12, fontweight="bold",
            )
            plt.tight_layout()
            path = os.path.join(
                self.out_dir, f"{sample_name}_nyquist_overlay.png"
            )
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as exc:
            logger.warning("Nyquist overlay failed: %s", exc)
            plt.close("all")
            return None

    # ── 2. Bode overlay ──────────────────────────────────────────

    def _plot_bode_overlay(
        self,
        sample_name: str,
        freq: np.ndarray,
        z_data: np.ndarray,
        z_model: Optional[np.ndarray],
        quality: QualityIndicator,
    ) -> Optional[str]:
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            mag_data = np.abs(z_data)
            phase_data = np.angle(z_data, deg=True)

            # |Z| panel
            ax1.semilogx(freq, mag_data, "o", color="#2196F3",
                         markersize=4, label="|Z| dados", alpha=0.7)
            if z_model is not None:
                mag_model = np.abs(z_model)
                ax1.semilogx(freq, mag_model, "-", color="#E53935",
                             linewidth=2, label="|Z| ajuste")
            ax1.set_ylabel("|Z| (Ω)")
            ax1.set_title(f"Bode — {sample_name}  {quality.emoji}")
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)

            # Phase panel
            ax2.semilogx(freq, phase_data, "o", color="#9C27B0",
                         markersize=4, label="Fase dados", alpha=0.7)
            if z_model is not None:
                phase_model = np.angle(z_model, deg=True)
                ax2.semilogx(freq, phase_model, "-", color="#E53935",
                             linewidth=2, label="Fase ajuste")
            ax2.set_xlabel("Frequência (Hz)")
            ax2.set_ylabel("Fase (°)")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            path = os.path.join(
                self.out_dir, f"{sample_name}_bode_overlay.png"
            )
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as exc:
            logger.warning("Bode overlay failed: %s", exc)
            plt.close("all")
            return None

    # ── 3. Residual analysis ─────────────────────────────────────

    def _plot_residual_analysis(
        self,
        sample_name: str,
        freq: np.ndarray,
        z_data: np.ndarray,
        z_model: Optional[np.ndarray],
        quality: QualityIndicator,
    ) -> Optional[str]:
        if z_model is None:
            return None
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            residuals = np.concatenate([
                z_data.real - z_model.real,
                z_data.imag - z_model.imag,
            ])

            # Panel 1: Residuals vs frequency (real + imag stacked)
            ax = axes[0]
            n_half = len(freq)
            ax.plot(freq, residuals[:n_half], "o", markersize=3,
                    color="#4CAF50", label="ΔZ' (real)", alpha=0.7)
            ax.plot(freq, residuals[n_half:], "s", markersize=3,
                    color="#FF9800", label="ΔZ'' (imag)", alpha=0.7)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Frequência (Hz)")
            ax.set_ylabel("Resíduo (Ω)")
            ax.set_xscale("log")
            ax.set_title("Resíduos vs Frequência")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # Panel 2: Histogram
            ax = axes[1]
            ax.hist(residuals, bins=min(30, max(5, len(residuals) // 5)),
                    color="#42A5F5", edgecolor="white", alpha=0.8, density=True)
            # Overlay normal PDF
            mu, sigma = np.mean(residuals), np.std(residuals)
            if sigma > 0:
                x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
                pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * ((x_norm - mu) / sigma) ** 2
                )
                ax.plot(x_norm, pdf, "r-", linewidth=2, label="Normal")
            ax.set_xlabel("Resíduo (Ω)")
            ax.set_ylabel("Densidade")
            ax.set_title("Histograma de Resíduos")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # Panel 3: QQ-plot
            ax = axes[2]
            sorted_res = np.sort(residuals)
            n = len(sorted_res)
            theoretical_q = np.array([
                _norm_ppf((i - 0.5) / n) for i in range(1, n + 1)
            ])
            ax.plot(theoretical_q, sorted_res, "o", markersize=3,
                    color="#7E57C2", alpha=0.7)
            # Reference line
            if n >= 2:
                q25_idx = max(0, int(0.25 * n))
                q75_idx = min(n - 1, int(0.75 * n))
                q25_t, q75_t = theoretical_q[q25_idx], theoretical_q[q75_idx]
                q25_s, q75_s = sorted_res[q25_idx], sorted_res[q75_idx]
                if q75_t != q25_t:
                    slope = (q75_s - q25_s) / (q75_t - q25_t)
                    intercept = q25_s - slope * q25_t
                    x_line = np.array([theoretical_q[0], theoretical_q[-1]])
                    ax.plot(x_line, slope * x_line + intercept,
                            "r--", linewidth=1.5, label="Referência")
            ax.set_xlabel("Quantis Teóricos")
            ax.set_ylabel("Quantis Amostrais (Ω)")
            ax.set_title("QQ-Plot dos Resíduos")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            fig.suptitle(
                f"Análise de Resíduos — {sample_name}  {quality.emoji}",
                fontsize=12, fontweight="bold",
            )
            plt.tight_layout()
            path = os.path.join(
                self.out_dir, f"{sample_name}_residual_analysis.png"
            )
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as exc:
            logger.warning("Residual analysis failed: %s", exc)
            plt.close("all")
            return None

    # ── 4. Parameter confidence ──────────────────────────────────

    def _plot_param_confidence(
        self,
        sample_name: str,
        fit_result: Dict[str, Any],
        template: Optional[Any],
        quality: QualityIndicator,
    ) -> Optional[str]:
        params = fit_result.get("params", {})
        params_std = fit_result.get("params_std", {})
        if not params:
            return None
        try:
            names = list(params.keys())
            values = np.array([params[n] for n in names])
            stds = np.array([params_std.get(n, 0.0) for n in names])

            # Normalise values for visualisation (log scale if wide range)
            fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))

            x = np.arange(len(names))
            bars = ax.bar(x, values, color="#42A5F5", alpha=0.8,
                          edgecolor="white", zorder=3)
            # Error bars
            if np.any(stds > 0):
                ax.errorbar(x, values, yerr=stds, fmt="none",
                            ecolor="#E53935", capsize=5, linewidth=2, zorder=4)

            # Show bounds as horizontal markers
            if template is not None and hasattr(template, "bounds"):
                lb = np.array(template.bounds[0])
                ub = np.array(template.bounds[1])
                for i, (lo, hi) in enumerate(zip(lb, ub)):
                    if np.isfinite(lo) and np.isfinite(hi):
                        ax.plot([i - 0.3, i + 0.3], [lo, lo], "--",
                                color="#4CAF50", linewidth=1, alpha=0.6)
                        ax.plot([i - 0.3, i + 0.3], [hi, hi], "--",
                                color="#FF5722", linewidth=1, alpha=0.6)

            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Valor")
            ax.set_title(
                f"Parâmetros — {sample_name}  {quality.emoji} {quality.label}",
                fontsize=11,
            )
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            path = os.path.join(
                self.out_dir, f"{sample_name}_param_confidence.png"
            )
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as exc:
            logger.warning("Parameter confidence plot failed: %s", exc)
            plt.close("all")
            return None

    # ── 5. Model comparison ──────────────────────────────────────

    def _plot_model_comparison(
        self,
        sample_name: str,
        all_results: Optional[List[Dict[str, Any]]],
        quality: QualityIndicator,
    ) -> Optional[str]:
        if not all_results or len(all_results) < 1:
            return None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            names = [r.get("template", "?") for r in all_results]
            bics = [r.get("bic", np.inf) for r in all_results]
            confs = [r.get("confidence", 0.0) for r in all_results]

            # Truncate long names
            short_names = [
                n[:20] + "…" if len(n) > 20 else n for n in names
            ]

            # Filter out inf BICs for display
            valid = [(n, b, c) for n, b, c in zip(short_names, bics, confs)
                     if np.isfinite(b)]
            if not valid:
                plt.close(fig)
                return None

            v_names, v_bics, v_confs = zip(*valid)
            x = np.arange(len(v_names))

            # Panel 1: BIC bar chart
            colours = ["#4CAF50" if i == 0 else "#90CAF9" for i in range(len(v_names))]
            ax1.barh(x, v_bics, color=colours, edgecolor="white", alpha=0.85)
            ax1.set_yticks(x)
            ax1.set_yticklabels(v_names, fontsize=9)
            ax1.set_xlabel("BIC")
            ax1.set_title("Comparação de Modelos — BIC")
            ax1.invert_yaxis()
            ax1.grid(axis="x", alpha=0.3)

            # Panel 2: Confidence bar chart
            conf_arr = np.array(v_confs, dtype=float)
            conf_arr = np.where(np.isfinite(conf_arr), conf_arr, 0.0)
            ax2.barh(x, conf_arr, color=colours, edgecolor="white", alpha=0.85)
            ax2.set_yticks(x)
            ax2.set_yticklabels(v_names, fontsize=9)
            ax2.set_xlabel("Confiança (softmax)")
            ax2.set_title("Confiança Relativa")
            ax2.set_xlim(0, 1.05)
            ax2.invert_yaxis()
            ax2.grid(axis="x", alpha=0.3)

            fig.suptitle(
                f"Comparação — {sample_name}  {quality.emoji}",
                fontsize=12, fontweight="bold",
            )
            plt.tight_layout()
            path = os.path.join(
                self.out_dir, f"{sample_name}_model_comparison.png"
            )
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as exc:
            logger.warning("Model comparison plot failed: %s", exc)
            plt.close("all")
            return None


# =====================================================================
# Lightweight normal inverse CDF (avoids scipy.stats import for speed)
# =====================================================================

def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (Beasley-Springer-Moro algorithm).

    Accurate to ~1e-9 for 0.01 < p < 0.99.
    """
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0

    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    else:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
