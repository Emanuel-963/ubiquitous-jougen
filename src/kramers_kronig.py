"""Kramers-Kronig validation for EIS data (linear KK test, Boukamp method).

Implements the **linear Kramers-Kronig** consistency test described by
Boukamp (1995).  The measured impedance is fitted to a chain of *M* Voigt
elements (R‖C in series) using linear least-squares.  The residuals between
the model and the data indicate whether the data satisfy the KK relations
(linearity, causality, stability, stationarity).

Residual classification (per-point |Δ|/|Z|):

* ``< 1 %`` → **excelente** — data is fully KK-compliant
* ``< 5 %`` → **aceitável** — minor deviations, usable with care
* ``≥ 5 %`` → **suspeito** — possible non-stationarity or non-linearity

Public API
----------
``KKResult``
    Dataclass with residuals, classification, and summary.
``KramersKronigValidator``
    Main class: ``validate()``, ``plot_residuals()``, ``summary_text()``.

Day 12 of the UPGRADE_PLAN_v0.2.0 schedule.

References
----------
.. [1] B. A. Boukamp, "A Linear Kronig-Kramers Transform Test for Immittance
   Data Validation", J. Electrochem. Soc. 142 (1995) 1885-1894.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# KKResult dataclass
# =====================================================================

@dataclass
class KKResult:
    """Result of a Kramers-Kronig consistency test.

    Attributes
    ----------
    freq : np.ndarray
        Frequency vector (Hz).
    z_data : np.ndarray
        Measured complex impedance.
    z_model : np.ndarray
        KK-model (Voigt chain) impedance.
    residuals_real : np.ndarray
        Relative residuals on the real part: ``(Z_re_data − Z_re_model) / |Z_data|``.
    residuals_imag : np.ndarray
        Relative residuals on the imaginary part.
    mean_residual_real : float
        Mean |residual| on the real part.
    mean_residual_imag : float
        Mean |residual| on the imaginary part.
    max_residual : float
        Maximum |residual| across both real and imaginary.
    classification : str
        ``"excelente"`` (< 1 %), ``"aceitável"`` (< 5 %), or ``"suspeito"`` (≥ 5 %).
    kk_valid : bool
        ``True`` if classification is not ``"suspeito"``.
    n_voigt : int
        Number of Voigt elements used.
    n_points : int
        Number of frequency points.
    """

    freq: np.ndarray = field(default_factory=lambda: np.array([]))
    z_data: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    z_model: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    residuals_real: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals_imag: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_residual_real: float = 0.0
    mean_residual_imag: float = 0.0
    max_residual: float = 0.0
    classification: str = "suspeito"
    kk_valid: bool = False
    n_voigt: int = 0
    n_points: int = 0


# =====================================================================
# KramersKronigValidator
# =====================================================================

class KramersKronigValidator:
    """Linear Kramers-Kronig test using a Voigt-element chain.

    The number of Voigt elements *M* is by default set to
    ``n_points // 2`` (a common heuristic in the literature).  It can be
    overridden via the constructor.

    Parameters
    ----------
    n_voigt : int | None
        Number of Voigt elements.  ``None`` → auto (``n_points // 2``).
    threshold_excellent : float
        Mean |residual| below this → ``"excelente"`` (default 0.01 = 1 %).
    threshold_acceptable : float
        Mean |residual| below this → ``"aceitável"`` (default 0.05 = 5 %).
    add_inductance : bool
        If *True* an inductance term is appended to handle high-frequency
        inductive artefacts (default *False*).

    Examples
    --------
    >>> kk = KramersKronigValidator()
    >>> result = kk.validate(freq, z_data)
    >>> result.kk_valid
    True
    """

    def __init__(
        self,
        n_voigt: Optional[int] = None,
        threshold_excellent: float = 0.01,
        threshold_acceptable: float = 0.05,
        add_inductance: bool = False,
    ) -> None:
        self._n_voigt_user = n_voigt
        self.threshold_excellent = threshold_excellent
        self.threshold_acceptable = threshold_acceptable
        self.add_inductance = add_inductance

    # ------------------------------------------------------------------
    # Voigt element impedance
    # ------------------------------------------------------------------

    @staticmethod
    def _voigt_impedance(omega: np.ndarray, R: float, tau: float) -> np.ndarray:
        """Single Voigt element: R / (1 + jωτ).

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency vector.
        R : float
            Resistance of the element.
        tau : float
            Time constant ``R·C``.
        """
        return R / (1.0 + 1j * omega * tau)

    # ------------------------------------------------------------------
    # Build linear system  A·x = b
    # ------------------------------------------------------------------

    def _build_linear_system(
        self,
        omega: np.ndarray,
        z_data: np.ndarray,
        tau_vec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the overdetermined linear system for Voigt R-values.

        For *M* Voigt elements with fixed time constants τ_k, the model is:

            Z_model(ω) = R_∞ + Σ_k  R_k / (1 + jωτ_k)   [+ jωL]

        This is *linear* in the unknowns [R_∞, R_1, …, R_M, (L)].

        We split real and imaginary parts and stack into a 2N × (M+1+?) system.
        """
        n = len(omega)
        m = len(tau_vec)
        n_cols = 1 + m  # R_inf + M resistances
        if self.add_inductance:
            n_cols += 1

        A = np.zeros((2 * n, n_cols))
        b = np.zeros(2 * n)

        # Column 0 = R_inf (pure real, contributes 1 to real part, 0 to imag)
        A[:n, 0] = 1.0  # real part
        # A[n:, 0] = 0.0  # imag part (already zero)

        for k in range(m):
            tau_k = tau_vec[k]
            denom = 1.0 + (omega * tau_k) ** 2
            # Real part of Voigt: R_k / (1 + (ωτ)²)
            A[:n, 1 + k] = 1.0 / denom
            # Imag part of Voigt: −R_k·ωτ / (1 + (ωτ)²)
            A[n:, 1 + k] = -omega * tau_k / denom

        if self.add_inductance:
            # Inductance: Z_L = jωL → real=0, imag=ωL
            A[n:, -1] = omega

        b[:n] = z_data.real
        b[n:] = z_data.imag

        return A, b

    # ------------------------------------------------------------------
    # Core: validate
    # ------------------------------------------------------------------

    def validate(
        self,
        freq: np.ndarray,
        z_data: np.ndarray,
    ) -> KKResult:
        """Run the linear Kramers-Kronig test.

        Parameters
        ----------
        freq : np.ndarray
            Frequency vector in Hz (must be > 0).
        z_data : np.ndarray
            Measured complex impedance.

        Returns
        -------
        KKResult
        """
        freq = np.asarray(freq, dtype=float)
        z_data = np.asarray(z_data, dtype=complex)

        # Sort by ascending frequency for consistency
        order = np.argsort(freq)
        freq = freq[order]
        z_data = z_data[order]

        n_pts = len(freq)
        if n_pts < 4:
            logger.warning("KK test: fewer than 4 points — skipping.")
            return KKResult(
                freq=freq, z_data=z_data, n_points=n_pts,
                classification="suspeito", kk_valid=False,
            )

        omega = 2.0 * np.pi * freq

        # Number of Voigt elements
        n_voigt = self._n_voigt_user or max(n_pts // 2, 2)
        # Cap at n_pts - 1 to keep system overdetermined
        n_voigt = min(n_voigt, n_pts - 1)

        # Distribute time constants logarithmically across the frequency range
        tau_min = 1.0 / (2.0 * np.pi * freq.max())
        tau_max = 1.0 / (2.0 * np.pi * max(freq.min(), 1e-12))
        tau_vec = np.logspace(np.log10(tau_min), np.log10(tau_max), n_voigt)

        # Build and solve linear system
        A, b = self._build_linear_system(omega, z_data, tau_vec)

        # Solve via least-squares (allow negative R — that already signals a problem)
        try:
            x, residual_norm, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            logger.warning("KK test: lstsq failed.")
            return KKResult(
                freq=freq, z_data=z_data, n_points=n_pts,
                n_voigt=n_voigt, classification="suspeito", kk_valid=False,
            )

        # Reconstruct model impedance
        z_model = self._reconstruct(omega, x, tau_vec)

        # Compute relative residuals
        mag = np.abs(z_data)
        mag_safe = np.where(mag > 1e-30, mag, 1e-30)

        res_real = (z_data.real - z_model.real) / mag_safe
        res_imag = (z_data.imag - z_model.imag) / mag_safe

        mean_re = float(np.mean(np.abs(res_real)))
        mean_im = float(np.mean(np.abs(res_imag)))
        max_res = float(np.max(np.maximum(np.abs(res_real), np.abs(res_imag))))

        # Classify
        mean_overall = (mean_re + mean_im) / 2.0
        if mean_overall < self.threshold_excellent:
            classification = "excelente"
        elif mean_overall < self.threshold_acceptable:
            classification = "aceitável"
        else:
            classification = "suspeito"

        kk_valid = classification != "suspeito"

        logger.info(
            "KK test: %d points, %d Voigt elements → %s "
            "(mean ΔRe=%.2f%%, ΔIm=%.2f%%)",
            n_pts, n_voigt, classification, mean_re * 100, mean_im * 100,
        )

        return KKResult(
            freq=freq,
            z_data=z_data,
            z_model=z_model,
            residuals_real=res_real,
            residuals_imag=res_imag,
            mean_residual_real=mean_re,
            mean_residual_imag=mean_im,
            max_residual=max_res,
            classification=classification,
            kk_valid=kk_valid,
            n_voigt=n_voigt,
            n_points=n_pts,
        )

    # ------------------------------------------------------------------
    # Reconstruct model from solved coefficients
    # ------------------------------------------------------------------

    def _reconstruct(
        self,
        omega: np.ndarray,
        x: np.ndarray,
        tau_vec: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct Z_model from the least-squares solution vector."""
        r_inf = x[0]
        z_model = np.full_like(omega, r_inf, dtype=complex)
        for k, tau_k in enumerate(tau_vec):
            R_k = x[1 + k]
            z_model += self._voigt_impedance(omega, R_k, tau_k)
        if self.add_inductance:
            L = x[-1]
            z_model += 1j * omega * L
        return z_model

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------

    @staticmethod
    def summary_text(result: KKResult) -> str:
        """Generate a human-readable summary of the KK test.

        Parameters
        ----------
        result : KKResult
            Output from :meth:`validate`.

        Returns
        -------
        str
            Multi-line summary in Portuguese.
        """
        emoji = {"excelente": "🟢", "aceitável": "🟡", "suspeito": "🔴"}
        e = emoji.get(result.classification, "⚪")

        lines = [
            f"{e} **Teste Kramers-Kronig — {result.classification.upper()}**",
            "",
            f"- Pontos: {result.n_points}",
            f"- Elementos Voigt: {result.n_voigt}",
            f"- Resíduo médio Re: {result.mean_residual_real * 100:.2f} %",
            f"- Resíduo médio Im: {result.mean_residual_imag * 100:.2f} %",
            f"- Resíduo máximo: {result.max_residual * 100:.2f} %",
            f"- KK válido: {'Sim' if result.kk_valid else 'Não'}",
        ]

        if result.classification == "excelente":
            lines.append(
                "\nOs dados satisfazem as relações de Kramers-Kronig. "
                "Nenhuma evidência de não-linearidade ou instabilidade."
            )
        elif result.classification == "aceitável":
            lines.append(
                "\nDesvios menores detectados. Os dados são utilizáveis, "
                "mas recomenda-se verificar condições de medição."
            )
        else:
            lines.append(
                "\n⚠️ Desvios significativos detectados! Possíveis causas: "
                "não-linearidade, instabilidade do sistema, ou artefactos "
                "de medição. Recomenda-se repetir a medição."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Visualisation: residuals vs frequency
    # ------------------------------------------------------------------

    def plot_residuals(
        self,
        result: KKResult,
        *,
        out_path: Optional[str] = None,
        dpi: int = 150,
    ):
        """Plot KK residuals vs frequency.

        Generates a 2-panel figure:
        - Top: ΔRe(%) and ΔIm(%) vs log(f)
        - Bottom: Nyquist overlay (data vs KK model)

        Parameters
        ----------
        result : KKResult
            Output from :meth:`validate`.
        out_path : str | None
            If given, save figure to this path.
        dpi : int
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if result.n_points == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # ── Panel 1: Residuals ──
        ax1 = axes[0]
        logf = np.log10(result.freq)
        ax1.plot(logf, result.residuals_real * 100, "o-", ms=3,
                 color="steelblue", label="ΔRe (%)", alpha=0.8)
        ax1.plot(logf, result.residuals_imag * 100, "s-", ms=3,
                 color="coral", label="ΔIm (%)", alpha=0.8)
        ax1.axhline(0, color="grey", ls="--", lw=0.8)
        # Threshold bands
        for thr, color, lbl in [
            (self.threshold_excellent * 100, "green", "1 %"),
            (self.threshold_acceptable * 100, "orange", "5 %"),
        ]:
            ax1.axhspan(-thr, thr, alpha=0.08, color=color)
            ax1.axhline(thr, color=color, ls=":", lw=0.7, label=lbl)
            ax1.axhline(-thr, color=color, ls=":", lw=0.7)

        emoji = {"excelente": "🟢", "aceitável": "🟡", "suspeito": "🔴"}
        e = emoji.get(result.classification, "")
        ax1.set_title(f"Resíduos Kramers-Kronig  {e} {result.classification}",
                       fontsize=12)
        ax1.set_xlabel("log₁₀(f / Hz)")
        ax1.set_ylabel("Resíduo relativo (%)")
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: Nyquist overlay ──
        ax2 = axes[1]
        ax2.plot(result.z_data.real, -result.z_data.imag, "o", ms=4,
                 color="steelblue", label="Dados", alpha=0.7)
        ax2.plot(result.z_model.real, -result.z_model.imag, "-",
                 color="crimson", lw=1.5, label="Modelo KK")
        ax2.set_xlabel("Z' (Ω)")
        ax2.set_ylabel("−Z'' (Ω)")
        ax2.set_title("Nyquist — Dados vs Modelo KK", fontsize=12)
        ax2.legend(fontsize=8)
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # Bode-residual plot
    # ------------------------------------------------------------------

    def plot_bode_residuals(
        self,
        result: KKResult,
        *,
        out_path: Optional[str] = None,
        dpi: int = 150,
    ):
        """Plot Bode-style residual diagram.

        Two panels: |Z| overlay and phase overlay with residual bands.

        Parameters
        ----------
        result : KKResult
            Output from :meth:`validate`.
        out_path : str | None
            Save path.
        dpi : int
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if result.n_points == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        logf = np.log10(result.freq)

        # |Z| overlay
        ax1.plot(logf, np.abs(result.z_data), "o", ms=3, color="steelblue",
                 label="|Z| dados")
        ax1.plot(logf, np.abs(result.z_model), "-", color="crimson",
                 label="|Z| modelo KK")
        ax1.set_ylabel("|Z| (Ω)")
        ax1.set_yscale("log")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Bode — Dados vs Modelo KK", fontsize=12)

        # Phase overlay
        phase_data = np.degrees(np.angle(result.z_data))
        phase_model = np.degrees(np.angle(result.z_model))
        ax2.plot(logf, phase_data, "o", ms=3, color="steelblue",
                 label="θ dados")
        ax2.plot(logf, phase_model, "-", color="crimson",
                 label="θ modelo KK")
        ax2.set_xlabel("log₁₀(f / Hz)")
        ax2.set_ylabel("Fase (°)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # CSV-ready dict
    # ------------------------------------------------------------------

    @staticmethod
    def to_dict(result: KKResult) -> Dict[str, Any]:
        """Convert KK result to a flat dict for CSV/DataFrame integration.

        Returns keys: ``KK_valid``, ``KK_class``, ``KK_mean_re``,
        ``KK_mean_im``, ``KK_max``, ``KK_n_voigt``.
        """
        return {
            "KK_valid": result.kk_valid,
            "KK_class": result.classification,
            "KK_mean_re_pct": round(result.mean_residual_real * 100, 3),
            "KK_mean_im_pct": round(result.mean_residual_imag * 100, 3),
            "KK_max_pct": round(result.max_residual * 100, 3),
            "KK_n_voigt": result.n_voigt,
        }
