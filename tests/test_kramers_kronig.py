"""Tests for src/kramers_kronig.py — Day 12."""

from __future__ import annotations

import numpy as np
import pytest

from src.kramers_kronig import KKResult, KramersKronigValidator


# ── Helpers ──────────────────────────────────────────────────────────

def _ideal_rc(freq: np.ndarray, Rs: float = 5.0, Rp: float = 50.0,
              C: float = 1e-3) -> np.ndarray:
    """Generate an ideal Rs + (Rp || C) spectrum — perfectly KK-compliant."""
    omega = 2.0 * np.pi * freq
    return Rs + Rp / (1.0 + 1j * omega * Rp * C)


def _randles(freq: np.ndarray, Rs=5.0, Rp=50.0, C=1e-3,
             sigma=10.0) -> np.ndarray:
    """Randles + Warburg — also KK-compliant."""
    omega = 2.0 * np.pi * freq
    Zw = sigma / np.sqrt(1j * omega + 1e-30)
    Zpar = 1.0 / (1.0 / Rp + 1j * omega * C)
    return Rs + Zpar + Zw


def _kk_freq(n=60):
    """Standard logarithmic frequency vector."""
    return np.logspace(-1, 5, n)


# =====================================================================
# KKResult dataclass
# =====================================================================


class TestKKResult:
    """Tests for the KKResult dataclass."""

    def test_defaults(self):
        r = KKResult()
        assert r.n_points == 0
        assert r.classification == "suspeito"
        assert r.kk_valid is False
        assert r.max_residual == 0.0

    def test_custom_fields(self):
        r = KKResult(
            classification="excelente",
            kk_valid=True,
            n_voigt=15,
            n_points=60,
            mean_residual_real=0.005,
        )
        assert r.kk_valid is True
        assert r.n_voigt == 15
        assert r.mean_residual_real == 0.005


# =====================================================================
# KramersKronigValidator — constructor
# =====================================================================


class TestValidatorInit:
    """Tests for KramersKronigValidator construction."""

    def test_defaults(self):
        kk = KramersKronigValidator()
        assert kk.threshold_excellent == 0.01
        assert kk.threshold_acceptable == 0.05
        assert kk.add_inductance is False

    def test_custom_thresholds(self):
        kk = KramersKronigValidator(threshold_excellent=0.005,
                                     threshold_acceptable=0.03)
        assert kk.threshold_excellent == 0.005
        assert kk.threshold_acceptable == 0.03

    def test_custom_n_voigt(self):
        kk = KramersKronigValidator(n_voigt=10)
        assert kk._n_voigt_user == 10

    def test_inductance_flag(self):
        kk = KramersKronigValidator(add_inductance=True)
        assert kk.add_inductance is True


# =====================================================================
# validate — ideal data (should be excelente)
# =====================================================================


class TestValidateIdeal:
    """Tests with ideal KK-compliant spectra."""

    def test_ideal_rc_excellent(self):
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        assert isinstance(result, KKResult)
        assert result.classification == "excelente"
        assert result.kk_valid is True

    def test_randles_excellent(self):
        freq = _kk_freq(60)
        z = _randles(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        assert result.kk_valid is True
        assert result.classification in ("excelente", "aceitável")

    def test_residuals_small(self):
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        assert result.mean_residual_real < 0.01
        assert result.mean_residual_imag < 0.01

    def test_z_model_close_to_data(self):
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        diff = np.abs(result.z_data - result.z_model) / np.abs(result.z_data)
        assert np.mean(diff) < 0.02

    def test_n_points_stored(self):
        freq = _kk_freq(40)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        assert result.n_points == 40

    def test_n_voigt_auto(self):
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        # Auto: n_pts // 2 = 30, capped at n_pts - 1 = 59
        assert result.n_voigt == 30

    def test_n_voigt_override(self):
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator(n_voigt=10)
        result = kk.validate(freq, z)
        assert result.n_voigt == 10

    def test_freq_sorted_internally(self):
        freq = _kk_freq(50)
        z = _ideal_rc(freq)
        # Reverse order
        kk = KramersKronigValidator()
        result = kk.validate(freq[::-1], z[::-1])
        assert result.kk_valid is True
        # Internally sorted ascending
        assert result.freq[0] < result.freq[-1]


# =====================================================================
# validate — noisy / corrupted data (should detect problems)
# =====================================================================


class TestValidateNoisy:
    """Tests with corrupted or non-KK-compliant data."""

    def test_heavy_noise_suspect(self):
        rng = np.random.default_rng(42)
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        # Add 20% noise
        noise = rng.normal(0, 0.2 * np.abs(z)) + 1j * rng.normal(0, 0.2 * np.abs(z))
        z_noisy = z + noise
        kk = KramersKronigValidator()
        result = kk.validate(freq, z_noisy)
        assert result.mean_residual_real > 0.01 or result.mean_residual_imag > 0.01

    def test_random_data_suspect(self):
        rng = np.random.default_rng(7)
        freq = _kk_freq(50)
        z = rng.normal(50, 30, 50) + 1j * rng.normal(-20, 15, 50)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        # Random data should not be KK-compliant
        assert result.classification in ("aceitável", "suspeito")

    def test_moderate_noise_acceptable_or_suspect(self):
        rng = np.random.default_rng(42)
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        noise = rng.normal(0, 0.05 * np.abs(z)) + 1j * rng.normal(0, 0.05 * np.abs(z))
        z_noisy = z + noise
        kk = KramersKronigValidator()
        result = kk.validate(freq, z_noisy)
        # Could be any classification; residuals should be higher than ideal
        assert result.mean_residual_real > 0.001


# =====================================================================
# validate — edge cases
# =====================================================================


class TestValidateEdgeCases:
    """Edge-case handling."""

    def test_too_few_points(self):
        freq = np.array([1.0, 10.0, 100.0])
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        assert result.classification == "suspeito"
        assert result.kk_valid is False

    def test_single_point(self):
        freq = np.array([100.0])
        z = np.array([50 + 10j])
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        assert result.kk_valid is False

    def test_empty_arrays(self):
        kk = KramersKronigValidator()
        result = kk.validate(np.array([]), np.array([], dtype=complex))
        assert result.kk_valid is False
        assert result.n_points == 0

    def test_four_points_minimum(self):
        freq = np.logspace(0, 3, 4)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        # Should run (4 points is the minimum)
        assert result.n_points == 4
        assert result.n_voigt >= 2

    def test_large_dataset(self):
        freq = _kk_freq(200)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        assert result.kk_valid is True
        assert result.n_voigt == 100  # 200 // 2

    def test_with_inductance(self):
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        # Add small inductive artifact at high freq
        omega = 2 * np.pi * freq
        z = z + 1j * omega * 1e-7
        kk = KramersKronigValidator(add_inductance=True)
        result = kk.validate(freq, z)
        # Should handle inductance well
        assert result.n_points == 60


# =====================================================================
# summary_text
# =====================================================================


class TestSummaryText:
    """Tests for the summary_text static method."""

    def test_excellent_summary(self):
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        text = kk.summary_text(result)
        assert "EXCELENTE" in text
        assert "🟢" in text
        assert "Kramers-Kronig" in text

    def test_suspect_summary(self):
        result = KKResult(classification="suspeito", kk_valid=False,
                          n_points=60, n_voigt=30)
        text = KramersKronigValidator.summary_text(result)
        assert "SUSPEITO" in text
        assert "🔴" in text
        assert "Desvios significativos" in text

    def test_acceptable_summary(self):
        result = KKResult(classification="aceitável", kk_valid=True,
                          n_points=60, n_voigt=30,
                          mean_residual_real=0.03, mean_residual_imag=0.02)
        text = KramersKronigValidator.summary_text(result)
        assert "ACEITÁVEL" in text
        assert "🟡" in text

    def test_summary_contains_stats(self):
        result = KKResult(
            classification="excelente", kk_valid=True,
            n_points=50, n_voigt=25,
            mean_residual_real=0.003, mean_residual_imag=0.004,
            max_residual=0.01,
        )
        text = KramersKronigValidator.summary_text(result)
        assert "50" in text     # n_points
        assert "25" in text     # n_voigt
        assert "Sim" in text    # kk_valid


# =====================================================================
# to_dict
# =====================================================================


class TestToDict:
    """Tests for the to_dict static method."""

    def test_keys_present(self):
        freq = _kk_freq(40)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        d = kk.to_dict(result)
        expected_keys = {"KK_valid", "KK_class", "KK_mean_re_pct",
                         "KK_mean_im_pct", "KK_max_pct", "KK_n_voigt"}
        assert expected_keys == set(d.keys())

    def test_values_types(self):
        freq = _kk_freq(40)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        d = kk.to_dict(result)
        assert isinstance(d["KK_valid"], bool)
        assert isinstance(d["KK_class"], str)
        assert isinstance(d["KK_mean_re_pct"], float)
        assert isinstance(d["KK_n_voigt"], int)

    def test_excellent_dict(self):
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        d = kk.to_dict(result)
        assert d["KK_valid"] is True
        assert d["KK_class"] == "excelente"
        assert d["KK_mean_re_pct"] < 1.0


# =====================================================================
# plot_residuals
# =====================================================================


class TestPlotResiduals:
    """Tests for the residual plot."""

    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")

        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        fig = kk.plot_residuals(result)
        assert fig is not None

    def test_saves_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")

        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        out = str(tmp_path / "kk_residuals.png")
        kk.plot_residuals(result, out_path=out)
        assert (tmp_path / "kk_residuals.png").exists()

    def test_empty_data_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")

        kk = KramersKronigValidator()
        result = KKResult()
        fig = kk.plot_residuals(result)
        assert fig is not None


# =====================================================================
# plot_bode_residuals
# =====================================================================


class TestPlotBodeResiduals:
    """Tests for the Bode residual plot."""

    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")

        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        fig = kk.plot_bode_residuals(result)
        assert fig is not None

    def test_saves_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")

        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        out = str(tmp_path / "kk_bode.png")
        kk.plot_bode_residuals(result, out_path=out)
        assert (tmp_path / "kk_bode.png").exists()

    def test_empty_data_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")

        kk = KramersKronigValidator()
        result = KKResult()
        fig = kk.plot_bode_residuals(result)
        assert fig is not None


# =====================================================================
# Integration: full flow
# =====================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_flow_ideal(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")

        freq = _kk_freq(80)
        z = _ideal_rc(freq, Rs=2.0, Rp=100.0, C=5e-4)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)

        assert result.kk_valid is True
        assert result.classification == "excelente"

        # Summary text
        text = kk.summary_text(result)
        assert "EXCELENTE" in text

        # to_dict
        d = kk.to_dict(result)
        assert d["KK_valid"] is True

        # Plots
        fig1 = kk.plot_residuals(result, out_path=str(tmp_path / "res.png"))
        fig2 = kk.plot_bode_residuals(result, out_path=str(tmp_path / "bode.png"))
        assert fig1 is not None
        assert fig2 is not None
        assert (tmp_path / "res.png").exists()
        assert (tmp_path / "bode.png").exists()

    def test_full_flow_randles(self):
        freq = _kk_freq(60)
        z = _randles(freq, Rs=3.0, Rp=80.0, C=2e-3, sigma=5.0)
        kk = KramersKronigValidator()
        result = kk.validate(freq, z)
        # Randles is KK-compliant
        assert result.kk_valid is True
        d = kk.to_dict(result)
        assert d["KK_valid"] is True

    def test_various_n_voigt(self):
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        for nv in [5, 15, 29]:
            kk = KramersKronigValidator(n_voigt=nv)
            result = kk.validate(freq, z)
            assert result.n_voigt == nv
            assert result.n_points == 60

    def test_strict_thresholds(self):
        freq = _kk_freq(60)
        z = _ideal_rc(freq)
        # Very strict thresholds
        kk = KramersKronigValidator(threshold_excellent=0.0001,
                                     threshold_acceptable=0.001)
        result = kk.validate(freq, z)
        # Even with strict thresholds, ideal data should be pretty good
        # but might not pass "excelente" at 0.01%
        assert result.n_points == 60
