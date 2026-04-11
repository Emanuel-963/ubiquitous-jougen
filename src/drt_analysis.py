"""Distribution of Relaxation Times (DRT) via Tikhonov regularization.

Primary entry-point
-------------------
    compute_drt(freq, z_real, z_imag, *, n_taus, lambda_reg) -> DRTResult

DRTResult  (TypedDict)
-----------
    tau        : np.ndarray  shape (n_taus,)  — relaxation times [s]
    gamma      : np.ndarray  shape (n_taus,)  — DRT distribution γ(τ) [Ω]
    r_inf      : float                        — high-frequency resistance [Ω]
    peaks      : list[dict]                   — [{tau_peak, gamma_peak, width_decades}]
    residuals  : np.ndarray  shape (n_freq,)  — |Z_imag_fit − Z_imag_meas|
    lambda_reg : float                        — λ used (provenance)
    n_taus     : int                          — τ-grid resolution used

Mathematical background
-----------------------
The EIS imaginary part is related to the DRT by:

    −Z″(ω) = ∫ γ(τ) · (ωτ)/(1 + (ωτ)²) d(ln τ)

Discretised on a log-uniform τ-grid this becomes:

    b ≈ A · γ        where  A[i,j] = Δln(τ) · (ω_i τ_j)/(1 + (ω_i τ_j)²)
    b = −Z″(ω)       (positive for typical EIS; loader stores Z″ < 0)

The regularised problem (Tikhonov, second-order smoothness):

    min  ‖A·γ − b‖² + λ ‖L·γ‖²
    s.t. γ ≥ 0

Solved via normal equations:
    (AᵀA + λ LᵀL) γ = Aᵀ b
then clipped to 0.
"""

import logging
from typing import TypedDict

import numpy as np
from scipy import linalg, signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public type contract
# ---------------------------------------------------------------------------

class DRTResult(TypedDict):
    """Return type of :func:`compute_drt`."""

    tau: np.ndarray        # relaxation times [s],         shape (n_taus,)
    gamma: np.ndarray      # DRT distribution γ(τ) [Ω],   shape (n_taus,)
    r_inf: float           # high-frequency resistance [Ω]
    peaks: list            # [{tau_peak, gamma_peak, width_decades}]
    residuals: np.ndarray  # |Z_imag_fit − Z_imag_meas|,  shape (n_freq,)
    lambda_reg: float      # λ used (provenance)
    n_taus: int            # τ-grid resolution used


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_kernel_matrix(omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Build the imaginary-part DRT kernel matrix A  (n_freq × n_taus).

    A[i, j] = Δln(τ) · (ω_i τ_j) / (1 + (ω_i τ_j)²)

    Maps γ → −Z″:  b ≈ A · γ
    """
    delta_ln_tau = np.diff(np.log(tau)).mean()  # uniform log-spacing step
    x = omega[:, np.newaxis] * tau[np.newaxis, :]  # broadcast (n_freq, n_taus)
    return delta_ln_tau * x / (1.0 + x ** 2)


def _build_regularization_matrix(n: int) -> np.ndarray:
    """Build second-order finite-difference Tikhonov matrix L  ((n−2) × n).

    Penalises curvature in γ, yielding smooth spectra.
    """
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def _detect_peaks(
    tau: np.ndarray,
    gamma: np.ndarray,
    n_taus: int,
    tau_min: float,
    tau_max: float,
) -> list:
    """Return list of peak dicts from the γ(τ) vector."""
    height_threshold = 0.01 * gamma.max() if gamma.max() > 0 else 0.0
    prominence_threshold = height_threshold * 0.5

    peak_indices, _ = signal.find_peaks(
        gamma,
        height=height_threshold,
        prominence=prominence_threshold,
    )

    peaks = []
    ln_tau_per_idx = (np.log10(tau_max) - np.log10(tau_min)) / max(n_taus - 1, 1)

    for idx in peak_indices:
        try:
            widths, *_ = signal.peak_widths(gamma, [idx], rel_height=0.5)
            width_decades = float(widths[0]) * ln_tau_per_idx
        except Exception:
            width_decades = float("nan")

        peaks.append(
            {
                "tau_peak": float(tau[idx]),
                "gamma_peak": float(gamma[idx]),
                "width_decades": width_decades,
            }
        )

    # Sort by descending prominence (gamma_peak)
    peaks.sort(key=lambda p: p["gamma_peak"], reverse=True)
    return peaks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_drt(
    freq: np.ndarray,
    z_real: np.ndarray,
    z_imag: np.ndarray,
    *,
    n_taus: int = 50,
    lambda_reg: float = 1e-3,
) -> DRTResult:
    """Compute the DRT spectrum via Tikhonov regularization (imaginary part).

    Parameters
    ----------
    freq : array-like
        Measurement frequencies [Hz].  Must all be positive.
    z_real : array-like
        Real part of impedance Z′ [Ω].
    z_imag : array-like
        Imaginary part Z″ [Ω].  **Loader convention: stored as negative**
        (zimag = −|Z″|).  This function negates internally to get −Z″ > 0.
    n_taus : int, optional
        Number of log-uniform discretisation points for τ.  Default 50.
    lambda_reg : float, optional
        Tikhonov regularisation parameter λ.  Larger → smoother γ.
        Typical range: 1e-5 … 1e-1.  Default 1e-3.

    Returns
    -------
    DRTResult
        See module docstring for key contract.

    Raises
    ------
    ValueError
        If fewer than 4 valid data points remain after cleaning.
    """
    freq = np.asarray(freq, dtype=float)
    z_real = np.asarray(z_real, dtype=float)
    z_imag = np.asarray(z_imag, dtype=float)

    # Drop NaN / non-positive frequencies
    valid = np.isfinite(freq) & np.isfinite(z_real) & np.isfinite(z_imag) & (freq > 0)
    freq, z_real, z_imag = freq[valid], z_real[valid], z_imag[valid]

    if len(freq) < 4:
        raise ValueError(
            f"compute_drt: only {len(freq)} valid data points — need ≥ 4."
        )

    # Sort ascending by frequency
    idx_sort = np.argsort(freq)
    freq = freq[idx_sort]
    z_real = z_real[idx_sort]
    z_imag = z_imag[idx_sort]

    omega = 2.0 * np.pi * freq

    # Loader stores Z″ < 0 (Nyquist convention). Flip to get positive side.
    z_imag_pos = -z_imag

    # R_inf: real part at highest measured frequency (ω → ∞ limit)
    r_inf = float(z_real[-1])

    # τ grid: spans the reciprocal frequency range
    tau_min = 1.0 / (2.0 * np.pi * freq[-1])
    tau_max = 1.0 / (2.0 * np.pi * freq[0])
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_taus)

    # Build matrices
    A = _build_kernel_matrix(omega, tau)            # (n_freq, n_taus)
    L = _build_regularization_matrix(n_taus)        # (n_taus-2, n_taus)

    # Normal equations: (AᵀA + λ LᵀL) γ = Aᵀ b
    M = A.T @ A + lambda_reg * (L.T @ L)
    rhs = A.T @ z_imag_pos
    gamma, _, _, _ = linalg.lstsq(M, rhs)

    # Non-negativity constraint
    gamma = np.maximum(gamma, 0.0)

    # Residuals
    z_fit = A @ gamma
    residuals = np.abs(z_fit - z_imag_pos)

    # Peak detection
    peaks = _detect_peaks(tau, gamma, n_taus, tau_min, tau_max)

    logger.debug(
        "DRT: n_freq=%d  n_taus=%d  λ=%.2e  R_inf=%.4f Ω  peaks=%d",
        len(freq), n_taus, lambda_reg, r_inf, len(peaks),
    )

    return DRTResult(
        tau=tau,
        gamma=gamma,
        r_inf=r_inf,
        peaks=peaks,
        residuals=residuals,
        lambda_reg=lambda_reg,
        n_taus=n_taus,
    )
