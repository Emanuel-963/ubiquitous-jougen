"""Tests for src/fitting_diagnostics.py — Day 9."""

from __future__ import annotations

import os
import numpy as np
import pytest

from src.fitting_diagnostics import (
    FittingDiagnostics,
    QualityIndicator,
    assess_quality,
    _downgrade,
    _norm_ppf,
    _DEFAULT_THRESHOLDS,
)
from src.circuit_fitting import CircuitTemplate


# ── Helpers ──────────────────────────────────────────────────────────

FREQ = np.logspace(-1, 5, 60)
OMEGA = 2.0 * np.pi * FREQ


def _simple_template() -> CircuitTemplate:
    """Rs + (Rp || C) template for testing."""
    param_names = ["Rs", "Rp", "C"]
    bounds = ([1e-6, 1e-3, 1e-15], [1e6, 1e8, 1e-1])

    def model(p, omega):
        Rs, Rp, C = p
        Zc = 1.0 / (1j * omega * C)
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zc)
        return Rs + Zpar

    def init(omega, z):
        return np.array([10.0, 100.0, 1e-5])

    return CircuitTemplate("Simple-RC", param_names, bounds, model, init,
                           "Rs − (Rp ‖ C)")


_SENTINEL = object()


def _make_fit_result(*, rss=0.5, n_points=120, success=True,
                     res_autocorr=0.1, res_structured=False,
                     bound_hits=0, params=_SENTINEL, params_std=_SENTINEL,
                     bic=-50.0, aic=-55.0, confidence=0.9,
                     template="Simple-RC", diagram="Rs − (Rp ‖ C)"):
    """Build a mock fit_result dict."""
    return {
        "template": template,
        "diagram": diagram,
        "params": {"Rs": 10.0, "Rp": 100.0, "C": 1e-5} if params is _SENTINEL else params,
        "params_std": {"Rs": 0.5, "Rp": 5.0, "C": 1e-7} if params_std is _SENTINEL else params_std,
        "success": success,
        "rss": rss,
        "bic": bic,
        "aic": aic,
        "n_params": 3,
        "n_points": n_points,
        "res_autocorr": res_autocorr,
        "res_structured": res_structured,
        "bound_hits": bound_hits,
        "confidence": confidence,
    }


def _synth_data():
    """Generate synthetic Randles data + model for testing."""
    tpl = _simple_template()
    p_true = np.array([10.0, 100.0, 1e-5])
    z_data = tpl.model_fn(p_true, OMEGA)
    rng = np.random.default_rng(42)
    z_data = z_data + rng.normal(0, 0.1, z_data.shape) + 1j * rng.normal(0, 0.1, z_data.shape)
    return FREQ, z_data, tpl


# =====================================================================
# QualityIndicator tests
# =====================================================================


class TestQualityIndicator:
    """Tests for the QualityIndicator dataclass."""

    def test_default_is_green(self):
        qi = QualityIndicator()
        assert qi.level == "green"
        assert qi.emoji == "🟢"

    def test_fields_set(self):
        qi = QualityIndicator(level="red", emoji="🔴",
                              label="Fit problemático",
                              reasons=["High RSS"])
        assert qi.level == "red"
        assert "High RSS" in qi.reasons


# =====================================================================
# assess_quality tests
# =====================================================================


class TestAssessQuality:
    """Tests for the traffic-light quality assessment."""

    def test_excellent_fit_green(self):
        result = _make_fit_result(rss=0.5, n_points=120,
                                 res_autocorr=0.05, bound_hits=0)
        qi = assess_quality(result)
        assert qi.level == "green"
        assert qi.emoji == "🟢"

    def test_moderate_rss_yellow(self):
        result = _make_fit_result(rss=6.0, n_points=120,
                                 res_autocorr=0.05, bound_hits=0)
        qi = assess_quality(result)
        assert qi.level == "yellow"

    def test_high_rss_red(self):
        result = _make_fit_result(rss=50.0, n_points=120,
                                 res_autocorr=0.05, bound_hits=0)
        qi = assess_quality(result)
        assert qi.level == "red"

    def test_structured_residuals_red(self):
        result = _make_fit_result(rss=0.5, n_points=120,
                                 res_autocorr=0.8, res_structured=True,
                                 bound_hits=0)
        qi = assess_quality(result)
        assert qi.level == "red"
        assert any("estruturados" in r for r in qi.reasons)

    def test_many_bound_hits_red(self):
        result = _make_fit_result(rss=0.5, n_points=120,
                                 res_autocorr=0.05, bound_hits=5)
        qi = assess_quality(result)
        assert qi.level == "red"

    def test_few_bound_hits_yellow(self):
        result = _make_fit_result(rss=0.5, n_points=120,
                                 res_autocorr=0.05, bound_hits=2)
        qi = assess_quality(result)
        # At least yellow
        assert qi.level in ("yellow", "red")

    def test_no_convergence_at_least_yellow(self):
        result = _make_fit_result(rss=0.5, n_points=120,
                                 success=False,
                                 res_autocorr=0.05, bound_hits=0)
        qi = assess_quality(result)
        assert qi.level in ("yellow", "red")

    def test_reasons_populated(self):
        result = _make_fit_result()
        qi = assess_quality(result)
        assert len(qi.reasons) >= 3

    def test_custom_thresholds(self):
        result = _make_fit_result(rss=5.0, n_points=120)
        # With strict threshold, even moderate RSS → red
        qi = assess_quality(result, thresholds={"rss_acceptable": 0.001})
        assert qi.level == "red"

    def test_inf_rss_red(self):
        result = _make_fit_result(rss=np.inf, n_points=120)
        qi = assess_quality(result)
        assert qi.level == "red"

    def test_label_matches_level(self):
        for rss, expected_level in [(0.5, "green"), (50.0, "red")]:
            result = _make_fit_result(rss=rss, n_points=120,
                                     res_autocorr=0.05, bound_hits=0)
            qi = assess_quality(result)
            if expected_level == "green":
                assert "excelente" in qi.label
            elif expected_level == "red":
                assert "problemático" in qi.label


# =====================================================================
# _downgrade tests
# =====================================================================


class TestDowngrade:
    """Tests for the _downgrade helper."""

    def test_green_to_yellow(self):
        assert _downgrade("green", "yellow") == "yellow"

    def test_green_to_red(self):
        assert _downgrade("green", "red") == "red"

    def test_yellow_to_red(self):
        assert _downgrade("yellow", "red") == "red"

    def test_red_stays_red(self):
        assert _downgrade("red", "yellow") == "red"

    def test_same_level(self):
        assert _downgrade("yellow", "yellow") == "yellow"

    def test_green_stays_when_downgrade_to_green(self):
        assert _downgrade("green", "green") == "green"


# =====================================================================
# _norm_ppf tests
# =====================================================================


class TestNormPPF:
    """Tests for the lightweight inverse-normal CDF."""

    def test_median_is_zero(self):
        assert abs(_norm_ppf(0.5)) < 1e-6

    def test_symmetry(self):
        assert abs(_norm_ppf(0.25) + _norm_ppf(0.75)) < 1e-6

    def test_extreme_low(self):
        assert _norm_ppf(0.0) == -6.0

    def test_extreme_high(self):
        assert _norm_ppf(1.0) == 6.0

    def test_typical_values(self):
        # Φ⁻¹(0.975) ≈ 1.96
        assert abs(_norm_ppf(0.975) - 1.96) < 0.01

    def test_monotonic(self):
        vals = [_norm_ppf(p) for p in np.linspace(0.01, 0.99, 50)]
        assert all(a < b for a, b in zip(vals, vals[1:]))


# =====================================================================
# FittingDiagnostics — plot generation
# =====================================================================


class TestFittingDiagnosticsPlots:
    """Test actual plot generation (files written to tmp dir)."""

    @pytest.fixture()
    def diag_env(self, tmp_path):
        """Set up diagnostics env with synthetic data."""
        freq, z_data, tpl = _synth_data()
        out_dir = str(tmp_path / "diagnostics")
        diag = FittingDiagnostics(out_dir=out_dir, dpi=72)
        fit_result = _make_fit_result(
            params={"Rs": 10.0, "Rp": 100.0, "C": 1e-5},
            params_std={"Rs": 0.5, "Rp": 5.0, "C": 1e-7},
        )
        all_results = [
            _make_fit_result(template="Simple-RC", bic=-50, confidence=0.7),
            _make_fit_result(template="Randles-CPE-W", bic=-30, confidence=0.2),
            _make_fit_result(template="Inductive-CPE", bic=-20, confidence=0.1),
        ]
        return {
            "diag": diag,
            "freq": freq,
            "z_data": z_data,
            "tpl": tpl,
            "fit_result": fit_result,
            "all_results": all_results,
            "out_dir": out_dir,
        }

    def test_generate_all_returns_dict(self, diag_env):
        env = diag_env
        paths = env["diag"].generate_all(
            sample_name="test_01",
            freq=env["freq"],
            z_data=env["z_data"],
            fit_result=env["fit_result"],
            all_results=env["all_results"],
            template=env["tpl"],
        )
        assert isinstance(paths, dict)
        assert "nyquist" in paths
        assert "bode" in paths
        assert "residuals" in paths
        assert "param_confidence" in paths
        assert "model_comparison" in paths
        assert "quality" in paths

    def test_nyquist_file_created(self, diag_env):
        env = diag_env
        paths = env["diag"].generate_all(
            sample_name="test_02",
            freq=env["freq"],
            z_data=env["z_data"],
            fit_result=env["fit_result"],
            template=env["tpl"],
        )
        assert paths["nyquist"] is not None
        assert os.path.isfile(paths["nyquist"])
        assert paths["nyquist"].endswith(".png")

    def test_bode_file_created(self, diag_env):
        env = diag_env
        paths = env["diag"].generate_all(
            sample_name="test_03",
            freq=env["freq"],
            z_data=env["z_data"],
            fit_result=env["fit_result"],
            template=env["tpl"],
        )
        assert paths["bode"] is not None
        assert os.path.isfile(paths["bode"])

    def test_residuals_file_created(self, diag_env):
        env = diag_env
        paths = env["diag"].generate_all(
            sample_name="test_04",
            freq=env["freq"],
            z_data=env["z_data"],
            fit_result=env["fit_result"],
            template=env["tpl"],
        )
        assert paths["residuals"] is not None
        assert os.path.isfile(paths["residuals"])

    def test_param_confidence_file_created(self, diag_env):
        env = diag_env
        paths = env["diag"].generate_all(
            sample_name="test_05",
            freq=env["freq"],
            z_data=env["z_data"],
            fit_result=env["fit_result"],
            template=env["tpl"],
        )
        assert paths["param_confidence"] is not None
        assert os.path.isfile(paths["param_confidence"])

    def test_model_comparison_file_created(self, diag_env):
        env = diag_env
        paths = env["diag"].generate_all(
            sample_name="test_06",
            freq=env["freq"],
            z_data=env["z_data"],
            fit_result=env["fit_result"],
            all_results=env["all_results"],
            template=env["tpl"],
        )
        assert paths["model_comparison"] is not None
        assert os.path.isfile(paths["model_comparison"])

    def test_quality_indicator_in_result(self, diag_env):
        env = diag_env
        paths = env["diag"].generate_all(
            sample_name="test_07",
            freq=env["freq"],
            z_data=env["z_data"],
            fit_result=env["fit_result"],
            template=env["tpl"],
        )
        assert isinstance(paths["quality"], QualityIndicator)

    def test_out_dir_created(self, diag_env):
        env = diag_env
        env["diag"].generate_all(
            sample_name="test_08",
            freq=env["freq"],
            z_data=env["z_data"],
            fit_result=env["fit_result"],
            template=env["tpl"],
        )
        assert os.path.isdir(env["out_dir"])


class TestFittingDiagnosticsEdgeCases:
    """Edge cases and robustness."""

    def test_no_template_no_model_plots(self, tmp_path):
        """Without template, z_model is None — residuals should be None."""
        freq, z_data, _ = _synth_data()
        diag = FittingDiagnostics(out_dir=str(tmp_path / "diag"), dpi=72)
        fit_result = _make_fit_result()
        paths = diag.generate_all(
            sample_name="notemplate",
            freq=freq,
            z_data=z_data,
            fit_result=fit_result,
            template=None,
        )
        # Residual analysis requires z_model → None
        assert paths["residuals"] is None

    def test_no_all_results_model_comparison_none(self, tmp_path):
        freq, z_data, tpl = _synth_data()
        diag = FittingDiagnostics(out_dir=str(tmp_path / "diag"), dpi=72)
        fit_result = _make_fit_result()
        paths = diag.generate_all(
            sample_name="nocomp",
            freq=freq,
            z_data=z_data,
            fit_result=fit_result,
            all_results=None,
            template=tpl,
        )
        assert paths["model_comparison"] is None

    def test_empty_params_no_param_confidence(self, tmp_path):
        freq, z_data, tpl = _synth_data()
        diag = FittingDiagnostics(out_dir=str(tmp_path / "diag"), dpi=72)
        fit_result = _make_fit_result(params={})
        paths = diag.generate_all(
            sample_name="noparams",
            freq=freq,
            z_data=z_data,
            fit_result=fit_result,
            template=tpl,
        )
        assert paths["param_confidence"] is None

    def test_all_inf_bic_model_comparison_none(self, tmp_path):
        freq, z_data, tpl = _synth_data()
        diag = FittingDiagnostics(out_dir=str(tmp_path / "diag"), dpi=72)
        fit_result = _make_fit_result()
        all_results = [_make_fit_result(bic=np.inf)]
        paths = diag.generate_all(
            sample_name="infbic",
            freq=freq,
            z_data=z_data,
            fit_result=fit_result,
            all_results=all_results,
            template=tpl,
        )
        assert paths["model_comparison"] is None

    def test_single_candidate_model_comparison(self, tmp_path):
        freq, z_data, tpl = _synth_data()
        diag = FittingDiagnostics(out_dir=str(tmp_path / "diag"), dpi=72)
        fit_result = _make_fit_result()
        all_results = [_make_fit_result(bic=-40, confidence=1.0)]
        paths = diag.generate_all(
            sample_name="single",
            freq=freq,
            z_data=z_data,
            fit_result=fit_result,
            all_results=all_results,
            template=tpl,
        )
        assert paths["model_comparison"] is not None

    def test_custom_dpi(self, tmp_path):
        diag = FittingDiagnostics(out_dir=str(tmp_path / "diag"), dpi=300)
        assert diag.dpi == 300

    def test_custom_thresholds_passed(self, tmp_path):
        diag = FittingDiagnostics(
            out_dir=str(tmp_path / "diag"),
            thresholds={"rss_excellent": 0.001},
        )
        assert diag.thresholds["rss_excellent"] == 0.001

    def test_five_png_files_generated(self, tmp_path):
        """All 5 plot types should create distinct PNG files."""
        freq, z_data, tpl = _synth_data()
        out_dir = str(tmp_path / "diag")
        diag = FittingDiagnostics(out_dir=out_dir, dpi=72)
        fit_result = _make_fit_result()
        all_results = [
            _make_fit_result(template="A", bic=-50, confidence=0.8),
            _make_fit_result(template="B", bic=-30, confidence=0.2),
        ]
        paths = diag.generate_all(
            sample_name="allplots",
            freq=freq,
            z_data=z_data,
            fit_result=fit_result,
            all_results=all_results,
            template=tpl,
        )
        png_paths = [v for k, v in paths.items() if isinstance(v, str)]
        assert len(png_paths) == 5
        for p in png_paths:
            assert os.path.isfile(p)
            assert os.path.getsize(p) > 0


class TestDefaultThresholds:
    """Ensure default thresholds are sensible."""

    def test_thresholds_keys(self):
        expected = {"rss_excellent", "rss_acceptable",
                    "autocorr_threshold",
                    "bound_hits_acceptable", "bound_hits_problematic"}
        assert set(_DEFAULT_THRESHOLDS.keys()) == expected

    def test_excellent_stricter_than_acceptable(self):
        assert _DEFAULT_THRESHOLDS["rss_excellent"] < _DEFAULT_THRESHOLDS["rss_acceptable"]

    def test_bound_hits_ordering(self):
        assert _DEFAULT_THRESHOLDS["bound_hits_acceptable"] < _DEFAULT_THRESHOLDS["bound_hits_problematic"]
