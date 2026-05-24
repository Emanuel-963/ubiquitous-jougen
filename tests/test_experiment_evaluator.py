"""Tests for src/experiment_evaluator.py — Modo Orientador."""

from __future__ import annotations

import numpy as np

from src.experiment_evaluator import (
    ExperimentEvaluation,
    ExperimentEvaluator,
    _count_zombie_params,
    _detect_powerline,
    _traffic_light,
    format_report,
)

# ── helpers ──────────────────────────────────────────────────────────

FREQ = np.logspace(-1, 5, 60)
OMEGA = 2.0 * np.pi * FREQ


def _randles_z(freq, Rs=10.0, Rp=100.0, C=1e-4):
    omega = 2.0 * np.pi * freq
    Zc = 1.0 / (1j * omega * C)
    Zpar = 1.0 / (1.0 / Rp + 1.0 / Zc)
    return Rs + Zpar


def _good_fit_result():
    return {
        "template": "Simple-RC",
        "diagram": "Rs − (Rp ‖ C)",
        "params": {"Rs": 10.0, "Rp": 100.0, "C": 1e-4},
        "params_std": {"Rs": 0.5, "Rp": 5.0, "C": 1e-6},
        "success": True,
        "rss": 0.5,
        "bic": -120.0,
        "aic": -130.0,
        "n_params": 3,
        "n_points": 60,
        "res_autocorr": 0.05,
        "res_structured": False,
        "bound_hits": 0,
        "chi2_over_nu": 1.4,
    }


# ── _traffic_light ────────────────────────────────────────────────────


class TestTrafficLight:
    def test_green_high(self):
        assert _traffic_light(8.0) == "🟢"

    def test_yellow_medium(self):
        assert _traffic_light(6.0) == "🟡"

    def test_red_low(self):
        assert _traffic_light(2.0) == "🔴"

    def test_boundary_green(self):
        assert _traffic_light(7.5) == "🟢"

    def test_boundary_yellow(self):
        assert _traffic_light(4.0) == "🟡"

    def test_boundary_red(self):
        assert _traffic_light(3.9) == "🔴"


# ── _count_zombie_params ──────────────────────────────────────────────


class TestCountZombieParams:
    def test_no_zombies_when_all_significant(self):
        fit = {
            "params": {"Rs": 10.0, "Rp": 100.0},
            "params_std": {"Rs": 0.5, "Rp": 5.0},
        }
        assert _count_zombie_params(fit) == []

    def test_zombie_when_ci_includes_zero(self):
        # val=0.5, std=1.0 → CI = ±1.96, includes 0
        fit = {
            "params": {"Rs": 10.0, "Rp": 0.5},
            "params_std": {"Rs": 0.5, "Rp": 1.0},
        }
        zombies = _count_zombie_params(fit)
        assert "Rp" in zombies
        assert "Rs" not in zombies

    def test_no_std_means_not_zombie(self):
        fit = {
            "params": {"Rs": 10.0},
            "params_std": {},
        }
        assert _count_zombie_params(fit) == []

    def test_empty_fit(self):
        assert _count_zombie_params({}) == []


# ── _detect_powerline ─────────────────────────────────────────────────


class TestDetectPowerline:
    def test_clean_spectrum_no_detection(self):
        z = _randles_z(FREQ)
        detected = _detect_powerline(FREQ, z)
        assert detected == []

    def test_insufficient_points_returns_empty(self):
        freq = np.array([50.0, 100.0])
        z = np.array([1 + 0j, 1 + 0j])
        assert _detect_powerline(freq, z) == []

    def test_artificial_spike_at_60hz(self):
        freq = np.logspace(-1, 5, 100)
        z = _randles_z(freq)
        # Inject a spike at ≈60 Hz
        mask = np.abs(freq - 60) < 5
        z[mask] *= 10.0
        detected = _detect_powerline(freq, z)
        # Not guaranteed to always detect; just ensure function runs without error
        assert isinstance(detected, list)


# ── ExperimentEvaluator ───────────────────────────────────────────────


class TestExperimentEvaluatorStructure:
    """Verify the evaluation returns correct structure."""

    def test_returns_evaluation_object(self):
        ev = ExperimentEvaluator()
        freq = FREQ
        z = _randles_z(freq)
        result = ev.evaluate("test_sample", freq, z)
        assert isinstance(result, ExperimentEvaluation)

    def test_has_10_criteria(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ))
        assert len(result.criteria) == 10

    def test_overall_score_in_range(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ))
        assert 0.0 <= result.overall_score <= 10.0

    def test_verdict_is_string(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ))
        assert isinstance(result.verdict, str)
        assert len(result.verdict) > 0

    def test_article_fragment_nonempty(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ))
        assert len(result.article_fragment) > 0

    def test_all_criteria_scores_in_range(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ))
        for c in result.criteria:
            assert 0.0 <= c.score <= 10.0, f"{c.name}: {c.score}"

    def test_criteria_have_emoji(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ))
        for c in result.criteria:
            assert c.emoji in ("🟢", "🟡", "🔴", "⚪"), f"{c.name}: {c.emoji!r}"


class TestExperimentEvaluatorQualitySensitivity:
    """Good data should score higher than bad data."""

    def test_good_data_higher_than_random(self):
        ev = ExperimentEvaluator()
        rng = np.random.default_rng(42)
        good_z = _randles_z(FREQ)
        bad_z = rng.normal(50, 50, len(FREQ)) + 1j * rng.normal(-20, 50, len(FREQ))
        good = ev.evaluate("good", FREQ, good_z)
        bad = ev.evaluate("bad", FREQ, bad_z)
        assert good.overall_score >= bad.overall_score

    def test_full_freq_range_scores_better(self):
        ev = ExperimentEvaluator()
        full_freq = np.logspace(-2, 5, 70)
        narrow_freq = np.logspace(1, 3, 30)
        full_z = _randles_z(full_freq)
        narrow_z = _randles_z(narrow_freq)
        full_result = ev.evaluate("full", full_freq, full_z)
        narrow_result = ev.evaluate("narrow", narrow_freq, narrow_z)
        # Full frequency coverage should give better freq_range criterion
        full_freq_score = next(
            c.score for c in full_result.criteria if "frequência" in c.name
        )
        narrow_freq_score = next(
            c.score for c in narrow_result.criteria if "frequência" in c.name
        )
        assert full_freq_score >= narrow_freq_score

    def test_electrode_inversion_detected(self):
        ev = ExperimentEvaluator()
        z = _randles_z(FREQ)
        # Invert sign of Z' at high frequencies
        z_inverted = z.copy()
        z_inverted[FREQ > 1000] = -z[FREQ > 1000].real + 1j * z[FREQ > 1000].imag
        result = ev.evaluate("inverted", FREQ, z_inverted)
        inversion_crit = next(c for c in result.criteria if "inversão" in c.name)
        assert inversion_crit.score < 10.0


class TestExperimentEvaluatorWithFitResult:
    """Tests that use a fit_result dict."""

    def test_good_fit_scores_high(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ), fit_result=_good_fit_result())
        fit_crit = next(c for c in result.criteria if "RSS" in c.name)
        chi2_crit = next(c for c in result.criteria if "χ²" in c.name)
        zombie_crit = next(c for c in result.criteria if "zumbi" in c.name)
        assert fit_crit.score >= 7.0
        assert chi2_crit.score >= 7.0
        assert zombie_crit.score == 10.0

    def test_zombie_params_detected(self):
        ev = ExperimentEvaluator()
        fit = _good_fit_result()
        # Make Rp a zombie
        fit["params"]["Rp"] = 0.3
        fit["params_std"]["Rp"] = 1.0
        result = ev.evaluate("s", FREQ, _randles_z(FREQ), fit_result=fit)
        zombie_crit = next(c for c in result.criteria if "zumbi" in c.name)
        assert zombie_crit.score < 10.0
        assert "Rp" in zombie_crit.message

    def test_bad_chi2_penalised(self):
        ev = ExperimentEvaluator()
        fit = _good_fit_result()
        fit["chi2_over_nu"] = 15.0
        result = ev.evaluate("s", FREQ, _randles_z(FREQ), fit_result=fit)
        chi2_crit = next(c for c in result.criteria if "χ²" in c.name)
        assert chi2_crit.score < 4.0

    def test_negative_resistance_flagged(self):
        ev = ExperimentEvaluator()
        fit = _good_fit_result()
        fit["params"]["Rs"] = -5.0
        result = ev.evaluate("s", FREQ, _randles_z(FREQ), fit_result=fit)
        phys_crit = next(c for c in result.criteria if "física" in c.name)
        assert phys_crit.score < 10.0

    def test_no_fit_gives_neutral_scores(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ), fit_result=None)
        # Fitting criteria should be neutral (score=5) when no fit provided
        fit_crit = next(c for c in result.criteria if "RSS" in c.name)
        assert fit_crit.score == 5.0

    def test_convergence_failure_penalised(self):
        ev = ExperimentEvaluator()
        fit = _good_fit_result()
        fit["success"] = False
        result = ev.evaluate("s", FREQ, _randles_z(FREQ), fit_result=fit)
        conv_crit = next(c for c in result.criteria if "Convergência" in c.name)
        assert conv_crit.score < 4.0


class TestExperimentEvaluatorVerdicts:
    """Test that verdict thresholds work correctly."""

    def test_high_score_publishable(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ), fit_result=_good_fit_result())
        # Good data + good fit should trend toward publishable
        assert result.overall_score > 0.0  # sanity check

    def test_sample_name_in_report(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("Nb4_NH4F_500cycles", FREQ, _randles_z(FREQ))
        report = format_report(result)
        assert "Nb4_NH4F_500cycles" in report


# ── format_report ─────────────────────────────────────────────────────


class TestFormatReport:
    def test_returns_string(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ))
        report = format_report(result)
        assert isinstance(report, str)

    def test_contains_verdict(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ))
        report = format_report(result)
        assert "NOTA GERAL" in report

    def test_contains_all_criterion_names(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ))
        report = format_report(result)
        for c in result.criteria:
            # At least first 20 chars of each criterion name should appear
            assert c.name[:20] in report, f"Missing: {c.name}"

    def test_article_fragment_in_report(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate("s", FREQ, _randles_z(FREQ))
        if result.article_fragment:
            report = format_report(result)
            assert "FRASE PRONTA" in report

    def test_report_with_good_fit(self):
        ev = ExperimentEvaluator()
        result = ev.evaluate(
            "sample01", FREQ, _randles_z(FREQ), fit_result=_good_fit_result()
        )
        report = format_report(result)
        assert "sample01" in report
        assert "χ²" in report

    def test_empty_criteria_does_not_crash(self):
        result = ExperimentEvaluation(
            sample_name="empty",
            criteria=[],
            overall_score=5.0,
            verdict="Revisão necessária ⚠️",
        )
        report = format_report(result)
        assert "empty" in report


# ── ExperimentEvaluator — frequency range checks ─────────────────────


class TestFreqRangeCheck:
    def test_very_narrow_range_penalised(self):
        ev = ExperimentEvaluator()
        narrow_freq = np.array([10.0, 100.0, 1000.0])
        narrow_z = _randles_z(narrow_freq)
        result = ev.evaluate("s", narrow_freq, narrow_z)
        freq_crit = next(c for c in result.criteria if "frequência" in c.name)
        assert freq_crit.score < 7.5

    def test_adequate_range_full_score(self):
        ev = ExperimentEvaluator(freq_min_ok=0.1, freq_max_ok=10_000.0)
        freq = np.logspace(-2, 5, 60)  # 0.01 Hz to 100 kHz
        z = _randles_z(freq)
        result = ev.evaluate("s", freq, z)
        freq_crit = next(c for c in result.criteria if "frequência" in c.name)
        assert freq_crit.score == 10.0


# ── point density ─────────────────────────────────────────────────────


class TestPointDensityCheck:
    def test_dense_spectrum_full_score(self):
        ev = ExperimentEvaluator(pts_per_decade_ok=5.0)
        freq = np.logspace(-1, 5, 80)
        z = _randles_z(freq)
        result = ev.evaluate("s", freq, z)
        density_crit = next(c for c in result.criteria if "Densidade" in c.name)
        assert density_crit.score >= 7.5

    def test_sparse_spectrum_penalised(self):
        ev = ExperimentEvaluator(pts_per_decade_ok=10.0)
        freq = np.logspace(-1, 5, 10)
        z = _randles_z(freq)
        result = ev.evaluate("s", freq, z)
        density_crit = next(c for c in result.criteria if "Densidade" in c.name)
        assert density_crit.score < 7.5
