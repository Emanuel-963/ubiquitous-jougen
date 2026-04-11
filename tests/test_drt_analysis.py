"""Tests for src.drt_analysis — smoke tests and regression guards.

Covers:
  - Output shapes
  - Non-negativity of γ
  - R_inf estimate accuracy
  - Peak detection on a clean synthetic RC spectrum
  - Peak τ location within one decade of truth
  - Provenance fields (lambda_reg, n_taus)
  - Noisy spectrum: no crash
  - λ smoothing: higher λ → fewer or equal peaks
  - Edge case: minimally valid input (4 points)
"""

import numpy as np
import pytest

from src.drt_analysis import compute_drt


# ---------------------------------------------------------------------------
# Synthetic test fixture
# ---------------------------------------------------------------------------

def _synthetic_rc(n: int = 40, noise_scale: float = 0.0, seed: int = 42):
    """Single-RC EIS spectrum for unit testing.

    Ground truth:
        R_inf = 1 Ω   (high-frequency resistance)
        R_ct  = 10 Ω  (charge-transfer resistance)
        τ     = R_ct × C = 1e-3 s  → DRT peak near τ = 1 ms (f ~ 160 Hz)

    Returns
    -------
    freq, z_real, z_imag
        z_imag follows loader convention: stored as −|Z″| (negative).
    """
    freq = np.logspace(-1, 5, n)
    omega = 2.0 * np.pi * freq
    R_inf, R_ct, tau0 = 1.0, 10.0, 1e-3

    denom = 1.0 + (omega * tau0) ** 2
    z_real = R_inf + R_ct / denom
    z_imag_positive = R_ct * omega * tau0 / denom   # −Z″ (positive physics)
    z_imag = -z_imag_positive                       # loader convention

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        z_real = z_real + rng.normal(0, noise_scale, n)
        z_imag = z_imag + rng.normal(0, noise_scale, n)

    return freq, z_real, z_imag


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_tau_shape(self):
        freq, z_real, z_imag = _synthetic_rc()
        res = compute_drt(freq, z_real, z_imag, n_taus=40)
        assert res["tau"].shape == (40,), "tau must have shape (n_taus,)"

    def test_gamma_shape(self):
        freq, z_real, z_imag = _synthetic_rc()
        res = compute_drt(freq, z_real, z_imag, n_taus=40)
        assert res["gamma"].shape == (40,), "gamma must have shape (n_taus,)"

    def test_residuals_shape(self):
        freq, z_real, z_imag = _synthetic_rc(n=30)
        res = compute_drt(freq, z_real, z_imag, n_taus=40)
        assert res["residuals"].shape == (30,), "residuals must have shape (n_freq,)"

    def test_peaks_is_list(self):
        freq, z_real, z_imag = _synthetic_rc()
        res = compute_drt(freq, z_real, z_imag)
        assert isinstance(res["peaks"], list)

    def test_peak_dict_keys(self):
        freq, z_real, z_imag = _synthetic_rc(n=60)
        res = compute_drt(freq, z_real, z_imag, n_taus=80, lambda_reg=1e-4)
        if res["peaks"]:
            pk = res["peaks"][0]
            assert "tau_peak" in pk
            assert "gamma_peak" in pk
            assert "width_decades" in pk


# ---------------------------------------------------------------------------
# Physical correctness
# ---------------------------------------------------------------------------

class TestPhysics:
    def test_gamma_non_negative(self):
        freq, z_real, z_imag = _synthetic_rc()
        res = compute_drt(freq, z_real, z_imag)
        assert (res["gamma"] >= 0).all(), "DRT γ must be non-negative"

    def test_r_inf_close_to_truth(self):
        """R_inf estimate (min Z′) should be within 0.5 Ω of ground truth 1 Ω."""
        freq, z_real, z_imag = _synthetic_rc()
        res = compute_drt(freq, z_real, z_imag)
        assert abs(res["r_inf"] - 1.0) < 0.5, (
            f"R_inf = {res['r_inf']:.4f} Ω, expected ~1.0 Ω"
        )

    def test_tau_grid_monotone(self):
        freq, z_real, z_imag = _synthetic_rc()
        res = compute_drt(freq, z_real, z_imag)
        assert (np.diff(res["tau"]) > 0).all(), "τ grid must be strictly increasing"

    def test_tau_grid_log_uniform(self):
        freq, z_real, z_imag = _synthetic_rc()
        res = compute_drt(freq, z_real, z_imag, n_taus=50)
        log_diffs = np.diff(np.log10(res["tau"]))
        # All steps should be nearly equal (within 1 %)
        assert np.std(log_diffs) / np.mean(log_diffs) < 0.01


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

class TestPeakDetection:
    def test_at_least_one_peak_clean(self):
        """Clean single-RC spectrum should yield at least one peak."""
        freq, z_real, z_imag = _synthetic_rc(n=60)
        res = compute_drt(freq, z_real, z_imag, n_taus=80, lambda_reg=1e-4)
        assert len(res["peaks"]) >= 1, (
            "Expected at least one DRT peak for single-RC spectrum"
        )

    def test_peak_tau_within_one_decade(self):
        """Dominant peak τ should be within one decade of ground truth τ=1e-3 s."""
        freq, z_real, z_imag = _synthetic_rc(n=60)
        res = compute_drt(freq, z_real, z_imag, n_taus=80, lambda_reg=1e-4)
        if not res["peaks"]:
            pytest.skip("No peaks detected — skip location test")
        tau_peaks = [p["tau_peak"] for p in res["peaks"]]
        closest = min(tau_peaks, key=lambda t: abs(np.log10(t) - np.log10(1e-3)))
        assert abs(np.log10(closest) - np.log10(1e-3)) < 1.0, (
            f"Closest peak at τ={closest:.2e} s, expected ~1e-3 s"
        )

    def test_peaks_sorted_by_prominence(self):
        """Peaks list must be sorted descending by gamma_peak."""
        freq, z_real, z_imag = _synthetic_rc(n=60)
        res = compute_drt(freq, z_real, z_imag, n_taus=80, lambda_reg=1e-4)
        if len(res["peaks"]) >= 2:
            gammas = [p["gamma_peak"] for p in res["peaks"]]
            assert gammas == sorted(gammas, reverse=True), (
                "Peaks must be sorted by descending gamma_peak"
            )


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_lambda_provenance(self):
        freq, z_real, z_imag = _synthetic_rc()
        lam = 5e-4
        res = compute_drt(freq, z_real, z_imag, lambda_reg=lam)
        assert res["lambda_reg"] == lam

    def test_n_taus_provenance(self):
        freq, z_real, z_imag = _synthetic_rc()
        n = 35
        res = compute_drt(freq, z_real, z_imag, n_taus=n)
        assert res["n_taus"] == n


# ---------------------------------------------------------------------------
# Robustness / edge cases
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_noisy_spectrum_no_crash(self):
        freq, z_real, z_imag = _synthetic_rc(noise_scale=0.05)
        res = compute_drt(freq, z_real, z_imag)
        assert "tau" in res and "gamma" in res

    def test_lambda_smoothing(self):
        """Higher λ should produce a smoother γ (fewer or equal peaks)."""
        freq, z_real, z_imag = _synthetic_rc(n=50)
        res_sharp = compute_drt(freq, z_real, z_imag, lambda_reg=1e-6)
        res_smooth = compute_drt(freq, z_real, z_imag, lambda_reg=1.0)
        assert len(res_sharp["peaks"]) >= len(res_smooth["peaks"]), (
            "Sharp (low λ) should have at least as many peaks as smooth (high λ)"
        )

    def test_minimum_valid_input(self):
        """4-point spectrum (minimum) must not raise."""
        freq, z_real, z_imag = _synthetic_rc(n=4)
        res = compute_drt(freq, z_real, z_imag, n_taus=10)
        assert res["tau"].shape == (10,)

    def test_raises_on_too_few_points(self):
        """Fewer than 4 valid points must raise ValueError."""
        freq = np.array([100.0, 1000.0, 10000.0])
        z_real = np.array([1.5, 1.2, 1.0])
        z_imag = np.array([-0.5, -0.3, -0.1])
        with pytest.raises(ValueError, match="valid data points"):
            compute_drt(freq, z_real, z_imag, n_taus=10)

    def test_nan_rows_filtered(self):
        """NaN rows should be silently dropped; result is still valid."""
        freq, z_real, z_imag = _synthetic_rc(n=30)
        freq[5] = np.nan
        z_real[10] = np.nan
        res = compute_drt(freq, z_real, z_imag)
        assert (res["gamma"] >= 0).all()

    def test_large_n_taus(self):
        """n_taus=200 must complete without error."""
        freq, z_real, z_imag = _synthetic_rc(n=50)
        res = compute_drt(freq, z_real, z_imag, n_taus=200)
        assert res["tau"].shape == (200,)
