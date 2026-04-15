"""Parameter uncertainty estimation via Monte Carlo and bootstrap methods.

Provides two complementary approaches to quantify how much trust we can place
in fitted EIS circuit parameters:

1. **Monte Carlo Error Propagation** — perturb the measured spectrum with
   Gaussian noise proportional to |Z| and re-fit *N* times.
2. **Bootstrap of Residuals** — re-sample the fit residuals, add them back to
   the model prediction, and re-fit to obtain a non-parametric confidence
   interval.

Both methods produce per-parameter distributions, from which we extract mean,
std, and percentile-based confidence intervals.  A convenience function
generates confidence-ellipse data for 2-D parameter visualisation.

Public API
----------
``MonteCarloResult``
    Dataclass with MC samples, stats, and CI.
``BootstrapResult``
    Dataclass with bootstrap samples, stats, and CI.
``UncertaintyAnalyzer``
    Main class: ``monte_carlo()``, ``bootstrap_residuals()``,
    ``confidence_ellipse()``, ``summary_table()``, ``plot_distributions()``,
    ``plot_ellipses()``.

Day 11 of the UPGRADE_PLAN_v0.2.0 schedule.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


# =====================================================================
# Result dataclasses
# =====================================================================

@dataclass
class MonteCarloResult:
    """Outcome of a Monte Carlo uncertainty analysis.

    Attributes
    ----------
    param_names : list[str]
        Parameter names in order.
    samples : np.ndarray
        Shape ``(n_iter, n_params)`` — each row is a set of fitted parameters.
    mean : dict[str, float]
        Per-parameter mean across MC iterations.
    std : dict[str, float]
        Per-parameter standard deviation.
    ci_low : dict[str, float]
        Lower bound of the confidence interval (default 2.5 %).
    ci_high : dict[str, float]
        Upper bound of the confidence interval (default 97.5 %).
    n_success : int
        How many MC iterations converged successfully.
    n_iter : int
        Total iterations attempted.
    noise_pct : float
        Noise level used (fraction of |Z|).
    """

    param_names: List[str] = field(default_factory=list)
    samples: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    ci_low: Dict[str, float] = field(default_factory=dict)
    ci_high: Dict[str, float] = field(default_factory=dict)
    n_success: int = 0
    n_iter: int = 0
    noise_pct: float = 0.0


@dataclass
class BootstrapResult:
    """Outcome of a residual-bootstrap uncertainty analysis.

    Same structure as :class:`MonteCarloResult` — field meanings are
    identical but the generation method is different.
    """

    param_names: List[str] = field(default_factory=list)
    samples: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    ci_low: Dict[str, float] = field(default_factory=dict)
    ci_high: Dict[str, float] = field(default_factory=dict)
    n_success: int = 0
    n_iter: int = 0


# =====================================================================
# UncertaintyAnalyzer
# =====================================================================

class UncertaintyAnalyzer:
    """Estimate parameter uncertainty via MC and bootstrap methods.

    Parameters
    ----------
    n_iter : int
        Number of iterations for MC / bootstrap (default 100).
    noise_pct : float
        Gaussian noise level as fraction of |Z| for MC (default 0.02 = 2 %).
    ci_level : float
        Confidence level for intervals (default 0.95 → 2.5th – 97.5th pctile).
    seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> from src.uncertainty import UncertaintyAnalyzer
    >>> ua = UncertaintyAnalyzer(n_iter=50, seed=42)
    >>> mc = ua.monte_carlo(template, freq, z_data)
    >>> mc.mean
    {'Rs': 10.1, 'Rp': 98.3, ...}
    """

    def __init__(
        self,
        n_iter: int = 100,
        noise_pct: float = 0.02,
        ci_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        self.n_iter = max(n_iter, 2)
        self.noise_pct = np.clip(noise_pct, 0.001, 0.20)
        self.ci_level = np.clip(ci_level, 0.50, 0.999)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal: single-shot refit
    # ------------------------------------------------------------------

    @staticmethod
    def _refit(
        model_fn,
        p0: np.ndarray,
        omega: np.ndarray,
        z_target: np.ndarray,
        bounds: Tuple,
    ) -> Optional[np.ndarray]:
        """Run a single ``least_squares`` fit; return params or *None*."""

        def _residuals(p: np.ndarray) -> np.ndarray:
            z_model = model_fn(p, omega)
            return np.concatenate([z_model.real - z_target.real,
                                   z_model.imag - z_target.imag])

        try:
            res = least_squares(_residuals, p0, bounds=bounds, max_nfev=3000)
            if res.success:
                return res.x.copy()
        except Exception:  # noqa: BLE001
            pass
        return None

    # ------------------------------------------------------------------
    # Helpers to build result dicts
    # ------------------------------------------------------------------

    def _build_result(
        self,
        param_names: List[str],
        all_params: List[np.ndarray],
        n_iter: int,
    ) -> Tuple[np.ndarray, Dict, Dict, Dict, Dict]:
        """Aggregate a list of param vectors into statistics."""
        if not all_params:
            n_p = len(param_names)
            empty = {k: np.nan for k in param_names}
            return np.empty((0, n_p)), empty, empty, empty, empty

        samples = np.array(all_params)  # (n_success, n_params)
        alpha = (1.0 - self.ci_level) / 2.0
        lo_pct = alpha * 100
        hi_pct = (1.0 - alpha) * 100

        mean = {k: float(np.mean(samples[:, i])) for i, k in enumerate(param_names)}
        std = {k: float(np.std(samples[:, i], ddof=1)) for i, k in enumerate(param_names)}
        ci_low = {k: float(np.percentile(samples[:, i], lo_pct)) for i, k in enumerate(param_names)}
        ci_high = {k: float(np.percentile(samples[:, i], hi_pct)) for i, k in enumerate(param_names)}
        return samples, mean, std, ci_low, ci_high

    # ------------------------------------------------------------------
    # Monte Carlo Error Propagation
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        template,
        freq: np.ndarray,
        z_data: np.ndarray,
        *,
        p0: Optional[np.ndarray] = None,
    ) -> MonteCarloResult:
        """Monte Carlo uncertainty: add noise ∝ |Z| and re-fit *N* times.

        Parameters
        ----------
        template
            A circuit template object with ``model_fn``, ``param_names``,
            ``bounds``, and ``init_fn``.
        freq : np.ndarray
            Frequency vector (Hz).
        z_data : np.ndarray
            Measured complex impedance.
        p0 : np.ndarray | None
            Starting parameters.  If *None*, ``template.init_fn`` is used.

        Returns
        -------
        MonteCarloResult
        """
        omega = 2.0 * np.pi * freq
        if p0 is None:
            p0 = template.init_fn(omega, z_data)
        p0 = np.asarray(p0, dtype=float)

        mag = np.abs(z_data)
        sigma_noise = self.noise_pct * mag  # per-point noise σ
        # Ensure sigma_noise is at least a small floor to avoid zero noise
        sigma_noise = np.maximum(sigma_noise, 1e-12)

        successful: List[np.ndarray] = []
        for _ in range(self.n_iter):
            noise_real = self._rng.normal(0, sigma_noise)
            noise_imag = self._rng.normal(0, sigma_noise)
            z_noisy = z_data + noise_real + 1j * noise_imag

            result = self._refit(template.model_fn, p0, omega, z_noisy, template.bounds)
            if result is not None:
                successful.append(result)

        samples, mean, std, ci_low, ci_high = self._build_result(
            template.param_names, successful, self.n_iter
        )

        return MonteCarloResult(
            param_names=list(template.param_names),
            samples=samples,
            mean=mean,
            std=std,
            ci_low=ci_low,
            ci_high=ci_high,
            n_success=len(successful),
            n_iter=self.n_iter,
            noise_pct=float(self.noise_pct),
        )

    # ------------------------------------------------------------------
    # Bootstrap of Residuals
    # ------------------------------------------------------------------

    def bootstrap_residuals(
        self,
        template,
        freq: np.ndarray,
        z_data: np.ndarray,
        *,
        p_fit: Optional[np.ndarray] = None,
    ) -> BootstrapResult:
        """Residual bootstrap: resample residuals and re-fit.

        Parameters
        ----------
        template
            Circuit template (``model_fn``, ``param_names``, ``bounds``,
            ``init_fn``).
        freq : np.ndarray
            Frequency vector (Hz).
        z_data : np.ndarray
            Measured complex impedance.
        p_fit : np.ndarray | None
            Parameters from the original best fit.  If *None*,
            ``template.init_fn`` is called.

        Returns
        -------
        BootstrapResult
        """
        omega = 2.0 * np.pi * freq
        if p_fit is None:
            p_fit = template.init_fn(omega, z_data)
        p_fit = np.asarray(p_fit, dtype=float)

        # Model prediction at best-fit params
        z_model = template.model_fn(p_fit, omega)
        residuals_real = z_data.real - z_model.real
        residuals_imag = z_data.imag - z_model.imag
        n_pts = len(freq)

        successful: List[np.ndarray] = []
        for _ in range(self.n_iter):
            # Resample residual indices with replacement
            idx = self._rng.integers(0, n_pts, size=n_pts)
            z_boot = z_model + residuals_real[idx] + 1j * residuals_imag[idx]

            result = self._refit(template.model_fn, p_fit, omega, z_boot, template.bounds)
            if result is not None:
                successful.append(result)

        samples, mean, std, ci_low, ci_high = self._build_result(
            template.param_names, successful, self.n_iter
        )

        return BootstrapResult(
            param_names=list(template.param_names),
            samples=samples,
            mean=mean,
            std=std,
            ci_low=ci_low,
            ci_high=ci_high,
            n_success=len(successful),
            n_iter=self.n_iter,
        )

    # ------------------------------------------------------------------
    # Confidence ellipse data
    # ------------------------------------------------------------------

    @staticmethod
    def confidence_ellipse(
        samples: np.ndarray,
        param_i: int,
        param_j: int,
        *,
        ci_level: float = 0.95,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2-D confidence ellipse from sample matrix.

        Parameters
        ----------
        samples : np.ndarray
            ``(n_samples, n_params)`` array — e.g. ``mc.samples``.
        param_i, param_j : int
            Column indices for the two parameters.
        ci_level : float
            Confidence level (default 0.95).
        n_points : int
            Number of points on the ellipse boundary.

        Returns
        -------
        x, y : np.ndarray
            Arrays of length *n_points* tracing the ellipse.
        """
        if samples.ndim != 2 or samples.shape[0] < 3:
            return np.array([]), np.array([])

        data = samples[:, [param_i, param_j]]
        mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)

        # Eigenvalue decomposition for ellipse
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-30)  # guard negative numerics

        # Chi-squared quantile for 2 DOF at ci_level
        # chi2_ppf(ci_level, 2) ≈ -2*ln(1 - ci_level)
        chi2_val = -2.0 * np.log(1.0 - ci_level)
        scale = np.sqrt(eigvals * chi2_val)

        theta = np.linspace(0, 2 * np.pi, n_points)
        unit_circle = np.column_stack([np.cos(theta), np.sin(theta)])
        ellipse = unit_circle * scale[np.newaxis, :] @ eigvecs.T + mean[np.newaxis, :]
        return ellipse[:, 0], ellipse[:, 1]

    # ------------------------------------------------------------------
    # Summary table (dict of dicts for DataFrame construction)
    # ------------------------------------------------------------------

    @staticmethod
    def summary_table(
        *results: MonteCarloResult | BootstrapResult,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Merge one or more uncertainty results into a summary table.

        Returns a dict of dicts suitable for ``pd.DataFrame(table).T``::

            {"Rs": {"MC_mean": …, "MC_std": …, "BS_mean": …, …}, …}
        """
        if labels is None:
            labels = [f"R{i}" for i in range(len(results))]

        table: Dict[str, Dict[str, Any]] = {}
        for res, label in zip(results, labels):
            for pname in res.param_names:
                if pname not in table:
                    table[pname] = {}
                table[pname][f"{label}_mean"] = res.mean.get(pname, np.nan)
                table[pname][f"{label}_std"] = res.std.get(pname, np.nan)
                table[pname][f"{label}_ci_low"] = res.ci_low.get(pname, np.nan)
                table[pname][f"{label}_ci_high"] = res.ci_high.get(pname, np.nan)
        return table

    # ------------------------------------------------------------------
    # Visualisation: parameter distributions
    # ------------------------------------------------------------------

    def plot_distributions(
        self,
        result: MonteCarloResult | BootstrapResult,
        *,
        out_path: Optional[str] = None,
        dpi: int = 150,
    ):
        """Histogram + KDE of each parameter from MC or bootstrap samples.

        Parameters
        ----------
        result
            A :class:`MonteCarloResult` or :class:`BootstrapResult`.
        out_path : str | None
            If given, save figure to this path.
        dpi : int
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        n_params = len(result.param_names)
        if n_params == 0 or result.samples.size == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No samples", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        cols = min(n_params, 3)
        rows = int(np.ceil(n_params / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.atleast_1d(axes).ravel()

        for idx, pname in enumerate(result.param_names):
            ax = axes[idx]
            data = result.samples[:, idx]
            ax.hist(data, bins="auto", density=True, alpha=0.6, color="steelblue",
                    edgecolor="white")
            ax.axvline(result.mean[pname], color="crimson", ls="--", lw=1.5,
                       label=f"μ = {result.mean[pname]:.4g}")
            ax.axvline(result.ci_low[pname], color="grey", ls=":", lw=1)
            ax.axvline(result.ci_high[pname], color="grey", ls=":", lw=1)
            ax.set_title(pname, fontsize=11)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        # Hide unused axes
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Parameter Distributions", fontsize=13, y=1.02)
        fig.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # Visualisation: confidence ellipses
    # ------------------------------------------------------------------

    def plot_ellipses(
        self,
        result: MonteCarloResult | BootstrapResult,
        *,
        pairs: Optional[List[Tuple[int, int]]] = None,
        out_path: Optional[str] = None,
        dpi: int = 150,
    ):
        """Plot 2-D confidence ellipses for selected parameter pairs.

        Parameters
        ----------
        result
            MC or bootstrap result.
        pairs : list[tuple[int, int]] | None
            Parameter index pairs to plot.  If *None*, all consecutive pairs.
        out_path : str | None
            Save figure if given.
        dpi : int
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        n_params = len(result.param_names)
        if n_params < 2 or result.samples.size == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient parameters/samples",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        if pairs is None:
            pairs = [(i, i + 1) for i in range(n_params - 1)]

        n_plots = len(pairs)
        cols = min(n_plots, 3)
        rows = int(np.ceil(n_plots / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        axes = np.atleast_1d(axes).ravel()

        for idx, (pi, pj) in enumerate(pairs):
            ax = axes[idx]
            # Scatter samples
            ax.scatter(result.samples[:, pi], result.samples[:, pj],
                       s=8, alpha=0.3, color="steelblue")
            # Confidence ellipse
            ex, ey = self.confidence_ellipse(
                result.samples, pi, pj, ci_level=self.ci_level
            )
            if ex.size > 0:
                ax.plot(ex, ey, color="crimson", lw=1.5,
                        label=f"{self.ci_level * 100:.0f}% CI")
            # Mark mean
            ax.plot(result.mean[result.param_names[pi]],
                    result.mean[result.param_names[pj]],
                    "x", color="black", ms=8, mew=2)
            ax.set_xlabel(result.param_names[pi])
            ax.set_ylabel(result.param_names[pj])
            ax.legend(fontsize=8)

        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Parameter Confidence Ellipses", fontsize=13, y=1.02)
        fig.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # Uncertainty columns for CSV export
    # ------------------------------------------------------------------

    @staticmethod
    def uncertainty_columns(
        fit_result: Dict[str, Any],
        mc: Optional[MonteCarloResult] = None,
        bs: Optional[BootstrapResult] = None,
    ) -> Dict[str, Any]:
        """Build a flat dict with ``param_fit ± param_std`` columns.

        Useful for appending to an existing ``circuit_fits.csv`` row.

        Parameters
        ----------
        fit_result : dict
            Original fit result dict from ``fit_template``.
        mc : MonteCarloResult | None
            Monte Carlo result (preferred source of uncertainty).
        bs : BootstrapResult | None
            Bootstrap result (fallback source).

        Returns
        -------
        dict
            E.g. ``{"Rs_fit": 10.1, "Rs_mc_std": 0.5, "Rs_mc_ci95_low": 9.2, …}``
        """
        cols: Dict[str, Any] = {}
        params = fit_result.get("params", {})
        params_std_fit = fit_result.get("params_std", {})

        for pname, val in params.items():
            cols[f"{pname}_fit"] = val
            cols[f"{pname}_fit_std"] = params_std_fit.get(pname, np.nan)
            if mc is not None and pname in mc.std:
                cols[f"{pname}_mc_std"] = mc.std[pname]
                cols[f"{pname}_mc_ci_low"] = mc.ci_low.get(pname, np.nan)
                cols[f"{pname}_mc_ci_high"] = mc.ci_high.get(pname, np.nan)
            if bs is not None and pname in bs.std:
                cols[f"{pname}_bs_std"] = bs.std[pname]
                cols[f"{pname}_bs_ci_low"] = bs.ci_low.get(pname, np.nan)
                cols[f"{pname}_bs_ci_high"] = bs.ci_high.get(pname, np.nan)
        return cols
