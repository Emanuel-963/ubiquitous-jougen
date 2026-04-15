"""Tests for src/fitting_report.py — Day 10."""

from __future__ import annotations

import numpy as np
import pytest

from src.fitting_report import (
    FittingReport,
    FittingReportGenerator,
    _PARAM_TEMPLATES,
    _CANONICAL_MAP,
    _canonical,
)
from src.feature_store import FeatureStore, FittingHistory


# ── Helpers ──────────────────────────────────────────────────────────

def _make_fit_result(**overrides):
    """Build a mock fit_result dict with sensible defaults."""
    base = {
        "template": "Randles-CPE-W",
        "diagram": "Rs − (Rp ‖ CPE) − W",
        "params": {"Rs": 10.0, "Rp": 100.0, "Q": 1e-4, "n": 0.85, "Sigma": 0.5},
        "params_std": {"Rs": 0.5, "Rp": 5.0, "Q": 1e-6, "n": 0.02, "Sigma": 0.05},
        "success": True,
        "rss": 0.5,
        "bic": -120.0,
        "aic": -130.0,
        "n_params": 5,
        "n_points": 120,
        "res_autocorr": 0.1,
        "res_structured": False,
        "bound_hits": 0,
        "confidence": 0.85,
    }
    base.update(overrides)
    return base


def _make_all_results():
    """Build a list of candidate results for model comparison context."""
    return [
        _make_fit_result(template="Randles-CPE-W", bic=-120.0, confidence=0.7),
        _make_fit_result(template="Two-Arc-CPE", bic=-100.0, confidence=0.2),
        _make_fit_result(template="Inductive-CPE", bic=-80.0, confidence=0.1),
    ]


class _MockRegistryTemplate:
    """Mock CircuitTemplate with registry metadata."""

    def __init__(self):
        self.name = "Randles-CPE-W"
        self.description = (
            "Modified Randles circuit with CPE and semi-infinite Warburg."
        )
        self.physical_meaning = {
            "Rs": "Ohmic resistance of the electrolyte (Ω)",
            "Rp": "Charge-transfer resistance (Ω)",
            "Q": "CPE pseudo-capacitance (F·s^(n-1))",
            "n": "CPE exponent — 1=ideal capacitor",
            "Sigma": "Warburg coefficient (Ω·s^−½)",
        }
        self.typical_systems = [
            "Li-ion batteries",
            "Supercapacitors",
            "Corrosion cells",
        ]
        self.param_names = ["Rs", "Rp", "Q", "n", "Sigma"]
        self.bounds = (
            [1e-6, 1e-3, 1e-12, 0.3, 1e-10],
            [1e6, 1e8, 1.0, 1.0, 1e5],
        )


def _populated_history(tmp_path) -> FittingHistory:
    """Create a FittingHistory with a few records for comparison tests."""
    store = FeatureStore(path=str(tmp_path / "history.json"))
    for i in range(5):
        store.add_record({
            "sample_id": f"sample_{i:03d}.txt",
            "circuit_name": "Randles-CPE-W",
            "spectral_features": {
                "logf_slope_low": -0.4 + i * 0.01,
                "logf_slope_high": 0.1,
                "phase_min": -60.0,
                "phase_max": 5.0,
                "phase_range": 65.0,
                "freq_at_phase_min": 100.0,
                "mag_range": 80.0,
                "zreal_min": 8.0 + i,
                "zreal_max": 120.0 + i * 5,
            },
            "params": {
                "Rs": 8.0 + i * 0.5,
                "Rp": 90.0 + i * 5,
                "Q": 1e-4,
                "n": 0.84,
                "Sigma": 0.4 + i * 0.02,
            },
            "bic": -115.0 + i,
            "confidence": 0.80,
        })
    return FittingHistory(store)


# =====================================================================
# FittingReport dataclass
# =====================================================================


class TestFittingReport:
    """Tests for the FittingReport dataclass."""

    def test_default_empty_report(self):
        r = FittingReport()
        assert r.summary == ""
        assert r.parameter_interpretation == {}
        assert r.recommendations == []

    def test_to_text_all_sections(self):
        r = FittingReport(
            summary="Test summary",
            circuit_justification="Test justification",
            parameter_interpretation={"Rs": "10 Ω"},
            quality_assessment="Green",
            recommendations=["Try X"],
            comparison_with_similar="Better than average",
        )
        text = r.to_text()
        assert "Resumo" in text
        assert "Test summary" in text
        assert "Justificação" in text
        assert "Parâmetros" in text
        assert "Rs" in text
        assert "Qualidade" in text
        assert "Recomendações" in text
        assert "Comparação" in text

    def test_to_text_empty_report(self):
        r = FittingReport()
        assert r.to_text() == ""

    def test_to_text_partial_report(self):
        r = FittingReport(summary="Only summary")
        text = r.to_text()
        assert "Resumo" in text
        assert "Only summary" in text
        # Other sections absent
        assert "Justificação" not in text


# =====================================================================
# _canonical mapping
# =====================================================================


class TestCanonical:
    """Tests for parameter name canonicalization."""

    def test_direct_match(self):
        assert _canonical("Rs") == "Rs"
        assert _canonical("Rp") == "Rp"
        assert _canonical("Sigma") == "Sigma"

    def test_canonical_map(self):
        assert _canonical("Rp1") == "Rp"
        assert _canonical("Q1") == "Q"
        assert _canonical("ncoat") == "n"

    def test_suffixed_with_block_prefix(self):
        # ZARC_Rp splits to ["ZARC", "Rp"] → "Rp" in templates
        assert _canonical("ZARC_Rp") == "Rp"
        # BLOCK_Q1 splits to ["BLOCK", "Q1"] → "Q1" in _CANONICAL_MAP
        assert _canonical("BLOCK_Q1") == "Q"
        # Pure unknown → returns itself
        result = _canonical("ZARC_R")
        assert result == "ZARC_R"

    def test_unknown_returns_itself(self):
        assert _canonical("XYZZY") == "XYZZY"


# =====================================================================
# _PARAM_TEMPLATES inventory
# =====================================================================


class TestParamTemplates:
    """Tests for the parameter interpretation templates."""

    def test_core_params_present(self):
        for key in ("Rs", "Rp", "Q", "n", "Sigma", "C", "L", "Rd", "Td"):
            assert key in _PARAM_TEMPLATES

    def test_template_structure(self):
        for key, tpl in _PARAM_TEMPLATES.items():
            assert "unit" in tpl
            assert "desc" in tpl
            assert "low" in tpl
            assert "mid" in tpl
            assert "high" in tpl
            assert "th_low" in tpl
            assert "th_high" in tpl

    def test_thresholds_ordered(self):
        for key, tpl in _PARAM_TEMPLATES.items():
            assert tpl["th_low"] < tpl["th_high"], f"{key}: th_low >= th_high"


# =====================================================================
# FittingReportGenerator — core generation
# =====================================================================


class TestFittingReportGeneratorBasic:
    """Basic tests for report generation."""

    def test_generate_returns_report(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result())
        assert isinstance(report, FittingReport)

    def test_summary_contains_circuit_name(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result())
        assert "Randles-CPE-W" in report.summary

    def test_summary_contains_quality_emoji(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result())
        # Good fit → green emoji
        assert "🟢" in report.summary or "🟡" in report.summary

    def test_summary_contains_bic(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(bic=-120.0))
        assert "-120" in report.summary

    def test_summary_contains_confidence(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(confidence=0.85))
        assert "85" in report.summary

    def test_quality_assessment_populated(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result())
        assert len(report.quality_assessment) > 0


# =====================================================================
# Parameter interpretation
# =====================================================================


class TestParameterInterpretation:
    """Tests for parameter interpretation logic."""

    def test_all_params_interpreted(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result())
        for pname in ("Rs", "Rp", "Q", "n", "Sigma"):
            assert pname in report.parameter_interpretation

    def test_low_Rs_interpretation(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(
            params={"Rs": 1.0, "Rp": 100.0, "Q": 1e-4, "n": 0.85, "Sigma": 0.5},
        ))
        interp = report.parameter_interpretation["Rs"]
        assert "baixo" in interp.lower() or "low" in interp.lower()

    def test_high_Rp_interpretation(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(
            params={"Rs": 10.0, "Rp": 1000.0, "Q": 1e-4, "n": 0.85, "Sigma": 0.5},
        ))
        interp = report.parameter_interpretation["Rp"]
        assert "elevado" in interp.lower() or "lenta" in interp.lower()

    def test_mid_n_interpretation(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(
            params={"Rs": 10.0, "Rp": 100.0, "Q": 1e-4, "n": 0.8, "Sigma": 0.5},
        ))
        interp = report.parameter_interpretation["n"]
        assert "moderada" in interp.lower() or "rugosidade" in interp.lower()

    def test_uncertainty_shown(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(
            params_std={"Rs": 0.5, "Rp": 5.0, "Q": 1e-6, "n": 0.02, "Sigma": 0.05},
        ))
        interp = report.parameter_interpretation["Rs"]
        assert "±" in interp

    def test_no_uncertainty_no_plusminus(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(params_std={}))
        interp = report.parameter_interpretation["Rs"]
        assert "±" not in interp

    def test_registry_meaning_used(self):
        gen = FittingReportGenerator()
        reg = _MockRegistryTemplate()
        report = gen.generate(
            _make_fit_result(),
            registry_template=reg,
        )
        interp = report.parameter_interpretation["Rs"]
        # Should contain registry description
        assert "Ohmic" in interp or "electrolyte" in interp

    def test_empty_params_empty_interpretation(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(params={}))
        assert report.parameter_interpretation == {}


# =====================================================================
# Circuit justification
# =====================================================================


class TestCircuitJustification:
    """Tests for the circuit justification section."""

    def test_justification_with_registry(self):
        gen = FittingReportGenerator()
        reg = _MockRegistryTemplate()
        report = gen.generate(
            _make_fit_result(),
            registry_template=reg,
        )
        assert "Warburg" in report.circuit_justification or "Randles" in report.circuit_justification

    def test_justification_with_typical_systems(self):
        gen = FittingReportGenerator()
        reg = _MockRegistryTemplate()
        report = gen.generate(
            _make_fit_result(),
            registry_template=reg,
        )
        assert "Li-ion" in report.circuit_justification or "Sistemas" in report.circuit_justification

    def test_justification_with_ranking(self):
        gen = FittingReportGenerator()
        all_results = _make_all_results()
        report = gen.generate(
            _make_fit_result(),
            all_results=all_results,
        )
        assert "1º" in report.circuit_justification or "candidato" in report.circuit_justification

    def test_justification_delta_bic(self):
        gen = FittingReportGenerator()
        all_results = _make_all_results()
        report = gen.generate(
            _make_fit_result(),
            all_results=all_results,
        )
        assert "ΔBIC" in report.circuit_justification

    def test_justification_no_context(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result())
        assert len(report.circuit_justification) > 0


# =====================================================================
# Recommendations
# =====================================================================


class TestRecommendations:
    """Tests for the recommendation engine."""

    def test_good_fit_no_critical_recs(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result())
        assert len(report.recommendations) >= 1
        # Should contain "excelente" or similar positive
        text = " ".join(report.recommendations)
        assert "excelente" in text.lower() or "nenhuma" in text.lower()

    def test_structured_residuals_recommend_complex(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(
            res_structured=True, res_autocorr=0.8,
        ))
        text = " ".join(report.recommendations)
        assert "estruturado" in text.lower() or "complexo" in text.lower()

    def test_bound_hits_recommend_check_bounds(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(bound_hits=3))
        text = " ".join(report.recommendations)
        assert "bound" in text.lower() or "limite" in text.lower()

    def test_high_rss_recommend_check_data(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(rss=50.0, n_points=120))
        text = " ".join(report.recommendations)
        assert "rss" in text.lower() or "dados" in text.lower()

    def test_no_convergence_recommend_nfev(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result(success=False))
        text = " ".join(report.recommendations)
        assert "convergiu" in text.lower() or "nfev" in text.lower()


# =====================================================================
# Comparison with history
# =====================================================================


class TestComparison:
    """Tests for comparison with FittingHistory."""

    def test_no_history_message(self):
        gen = FittingReportGenerator()
        report = gen.generate(_make_fit_result())
        assert "Sem dados" in report.comparison_with_similar or \
               "histórico" in report.comparison_with_similar.lower()

    def test_with_history(self, tmp_path):
        gen = FittingReportGenerator()
        history = _populated_history(tmp_path)
        features = {
            "logf_slope_low": -0.4,
            "logf_slope_high": 0.1,
            "phase_min": -60.0,
            "phase_max": 5.0,
            "phase_range": 65.0,
            "freq_at_phase_min": 100.0,
            "mag_range": 80.0,
            "zreal_min": 10.0,
            "zreal_max": 130.0,
        }
        report = gen.generate(
            _make_fit_result(),
            history=history,
            spectral_features=features,
        )
        assert "Randles-CPE-W" in report.comparison_with_similar or \
               "amostra" in report.comparison_with_similar.lower()

    def test_comparison_with_param_diff(self, tmp_path):
        gen = FittingReportGenerator()
        history = _populated_history(tmp_path)
        features = {
            "logf_slope_low": -0.4,
            "logf_slope_high": 0.1,
            "phase_min": -60.0,
            "phase_max": 5.0,
            "phase_range": 65.0,
            "freq_at_phase_min": 100.0,
            "mag_range": 80.0,
            "zreal_min": 10.0,
            "zreal_max": 130.0,
        }
        report = gen.generate(
            _make_fit_result(params={
                "Rs": 20.0,  # much higher than historical ~10
                "Rp": 100.0,
                "Q": 1e-4,
                "n": 0.85,
                "Sigma": 0.5,
            }),
            history=history,
            spectral_features=features,
        )
        # Should mention percentage difference
        assert "%" in report.comparison_with_similar or \
               "maior" in report.comparison_with_similar

    def test_comparison_empty_history(self, tmp_path):
        gen = FittingReportGenerator()
        store = FeatureStore(path=str(tmp_path / "empty.json"))
        history = FittingHistory(store)
        features = {"logf_slope_low": -0.4}
        report = gen.generate(
            _make_fit_result(),
            history=history,
            spectral_features=features,
        )
        assert "Sem histórico" in report.comparison_with_similar or \
               "heurística" in report.comparison_with_similar


# =====================================================================
# Full report flow
# =====================================================================


class TestFullReportFlow:
    """End-to-end tests with all context provided."""

    def test_full_report_all_sections(self, tmp_path):
        gen = FittingReportGenerator()
        reg = _MockRegistryTemplate()
        history = _populated_history(tmp_path)
        features = {
            "logf_slope_low": -0.4,
            "logf_slope_high": 0.1,
            "phase_min": -60.0,
            "phase_max": 5.0,
            "phase_range": 65.0,
            "freq_at_phase_min": 100.0,
            "mag_range": 80.0,
            "zreal_min": 10.0,
            "zreal_max": 130.0,
        }
        report = gen.generate(
            _make_fit_result(),
            history=history,
            spectral_features=features,
            all_results=_make_all_results(),
            registry_template=reg,
        )
        text = report.to_text()
        assert "Resumo" in text
        assert "Justificação" in text
        assert "Parâmetros" in text
        assert "Qualidade" in text
        assert "Recomendações" in text
        assert "Comparação" in text

    def test_full_report_red_quality(self, tmp_path):
        gen = FittingReportGenerator()
        report = gen.generate(
            _make_fit_result(
                rss=100.0,
                res_structured=True,
                res_autocorr=0.9,
                bound_hits=5,
                success=False,
            ),
        )
        assert "🔴" in report.quality_assessment
        assert len(report.recommendations) >= 2

    def test_custom_thresholds(self):
        gen = FittingReportGenerator(thresholds={"rss_excellent": 0.001})
        report = gen.generate(_make_fit_result(rss=0.5, n_points=120))
        # With very strict threshold, rss=0.5 / 120 = 0.004 > 0.001
        assert "🟡" in report.quality_assessment or "🔴" in report.quality_assessment
