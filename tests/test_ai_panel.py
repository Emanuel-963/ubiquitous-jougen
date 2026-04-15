"""Tests for src.gui.tabs.ai_panel (Day 19).

All tests are headless — no tkinter / customtkinter required.
The AI panel module is a pure-logic layer: data in → text out.

Coverage targets
----------------
* AIPanelConfig — defaults, is_summary, is_full
* AIPanelResult — defaults, has_predictions, has_process_report
* format_findings_text — empty, normal, summary truncation
* format_anomalies_text — empty, normal, summary truncation
* format_recommendations_text — empty, normal, summary truncation
* format_predictions_text — None prediction, with prediction, summary
* format_process_text — None, with report, summary
* build_executive_summary — various combos
* _has_eis / _has_cycling / _has_drt — state checks
* _build_eis_proxy / _build_cycling_proxy / _build_drt_proxy
* _build_process_entries — from ranked_df index, Arquivo column, fallback
* run_ai_analysis — empty state, EIS-only, all-pipeline, scope filtering,
  summary vs full, exception tolerance
* _assemble_full_report — section headers present
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.ai.inference_engine import (
    AnalysisReport,
    Anomaly,
    Finding,
    Priority,
    Recommendation,
)
from src.ai.performance_predictor import (
    CyclingPrediction,
    Improvement,
    ImprovementArea,
)
from src.ai.process_advisor import ProcessReport, ProductionRec, RecommendationArea
from src.gui.tabs.ai_panel import (
    AIPanelConfig,
    AIPanelResult,
    _assemble_full_report,
    _build_cycling_proxy,
    _build_drt_proxy,
    _build_eis_proxy,
    _build_process_entries,
    _has_cycling,
    _has_drt,
    _has_eis,
    _SimpleProxy,
    build_executive_summary,
    format_anomalies_text,
    format_findings_text,
    format_predictions_text,
    format_process_text,
    format_recommendations_text,
    run_ai_analysis,
)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers — fake AppState
# ═══════════════════════════════════════════════════════════════════════

def _make_ranked_df(**overrides: float) -> pd.DataFrame:
    base = {
        "Rs_fit": [1.0],
        "Rp_fit": [100.0],
        "Q": [1e-4],
        "n": [0.85],
        "Sigma": [20.0],
        "C_mean": [5e-5],
        "Tau": [0.01],
        "Dispersion": [0.3],
        "Energy_mean": [12.0],
        "Score": [0.75],
    }
    for k, v in overrides.items():
        base[k] = [v]
    df = pd.DataFrame(base, index=["H2SO4_1A_GCT.csv"])
    return df


def _make_state(*, eis=True, cycling=True, drt=True) -> SimpleNamespace:
    """Create a fake AppState with optional pipeline data."""
    state = SimpleNamespace(
        rank_df=None,
        eis_df=None,
        raw_eis={},
        cic_df=None,
        cic_results={},
        drt_df=None,
        drt_peaks_df=None,
        drt_summary_df=None,
        drt_results={},
    )
    if eis:
        state.rank_df = _make_ranked_df()
        state.eis_df = state.rank_df.copy()
    if cycling:
        state.cic_df = pd.DataFrame({
            "Retenção (%)": [92.0],
            "Energia (µJ)": [15.0],
            "Potência (µW)": [3.0],
        })
    if drt:
        state.drt_df = pd.DataFrame({"Sample": ["s1"], "tau_peak1": [0.01]})
        state.drt_summary_df = pd.DataFrame({
            "n_peaks": [2], "dominant_tau": [0.01],
        })
    return state


# ═══════════════════════════════════════════════════════════════════════
#  AIPanelConfig
# ═══════════════════════════════════════════════════════════════════════

class TestAIPanelConfig:
    def test_defaults(self):
        cfg = AIPanelConfig()
        assert cfg.scope_eis is True
        assert cfg.scope_cycling is True
        assert cfg.scope_drt is True
        assert cfg.detail == "full"

    def test_is_summary(self):
        cfg = AIPanelConfig(detail="summary")
        assert cfg.is_summary is True
        assert cfg.is_full is False

    def test_is_full(self):
        cfg = AIPanelConfig(detail="full")
        assert cfg.is_full is True
        assert cfg.is_summary is False

    def test_custom_scope(self):
        cfg = AIPanelConfig(scope_eis=False, scope_cycling=True, scope_drt=False)
        assert cfg.scope_eis is False
        assert cfg.scope_drt is False


# ═══════════════════════════════════════════════════════════════════════
#  AIPanelResult
# ═══════════════════════════════════════════════════════════════════════

class TestAIPanelResult:
    def test_defaults(self):
        r = AIPanelResult()
        assert r.executive_summary == ""
        assert r.has_predictions is False
        assert r.has_process_report is False
        assert r.quality_score == 0.0
        assert r.n_findings == 0

    def test_has_predictions(self):
        r = AIPanelResult(cycling_prediction=CyclingPrediction(energy=10.0))
        assert r.has_predictions is True

    def test_has_process_report(self):
        r = AIPanelResult(process_report=ProcessReport(n_conditions=1))
        assert r.has_process_report is True


# ═══════════════════════════════════════════════════════════════════════
#  format_findings_text
# ═══════════════════════════════════════════════════════════════════════

class TestFormatFindingsText:
    def test_empty(self):
        assert "No findings" in format_findings_text([])

    def test_normal(self):
        findings = [
            Finding(parameter="Rs", value=2.0, description="low"),
            Finding(parameter="Rp", value=100.0, description="normal"),
        ]
        text = format_findings_text(findings)
        assert "Rs" in text
        assert "Rp" in text

    def test_summary_truncation(self):
        findings = [
            Finding(parameter=f"P{i}", value=float(i), description="x")
            for i in range(10)
        ]
        text = format_findings_text(findings, summary=True)
        assert "5 more" in text

    def test_summary_no_truncation_if_5(self):
        findings = [
            Finding(parameter=f"P{i}", value=float(i), description="x")
            for i in range(5)
        ]
        text = format_findings_text(findings, summary=True)
        assert "more" not in text


# ═══════════════════════════════════════════════════════════════════════
#  format_anomalies_text
# ═══════════════════════════════════════════════════════════════════════

class TestFormatAnomaliesText:
    def test_empty(self):
        assert "No anomalies" in format_anomalies_text([])

    def test_normal(self):
        anomalies = [Anomaly(parameter="Rp", value=-5.0, description="negative")]
        text = format_anomalies_text(anomalies)
        assert "Rp" in text

    def test_summary_truncation(self):
        anomalies = [
            Anomaly(parameter=f"A{i}", description="x") for i in range(8)
        ]
        text = format_anomalies_text(anomalies, summary=True)
        assert "3 more" in text


# ═══════════════════════════════════════════════════════════════════════
#  format_recommendations_text
# ═══════════════════════════════════════════════════════════════════════

class TestFormatRecommendationsText:
    def test_empty(self):
        assert "No recommendations" in format_recommendations_text([])

    def test_numbered(self):
        recs = [
            Recommendation(text="Do A", priority=Priority.HIGH),
            Recommendation(text="Do B", priority=Priority.MEDIUM),
        ]
        text = format_recommendations_text(recs)
        assert "1." in text
        assert "2." in text

    def test_summary_truncation(self):
        recs = [Recommendation(text=f"R{i}") for i in range(10)]
        text = format_recommendations_text(recs, summary=True)
        assert "5 more" in text


# ═══════════════════════════════════════════════════════════════════════
#  format_predictions_text
# ═══════════════════════════════════════════════════════════════════════

class TestFormatPredictionsText:
    def test_no_prediction(self):
        text = format_predictions_text(None, [])
        assert "No predictions" in text

    def test_with_prediction(self):
        pred = CyclingPrediction(energy=12.0, power=3.0, retention=85.0, confidence=0.7)
        text = format_predictions_text(pred, [])
        assert "12.0" in text
        assert "85.0" in text
        assert "70%" in text

    def test_with_improvements(self):
        pred = CyclingPrediction(energy=10.0, confidence=0.5)
        imps = [Improvement(area=ImprovementArea.OHMIC_RESISTANCE, action="Polish", expected_impact="lower Rs")]
        text = format_predictions_text(pred, imps)
        assert "improvement" in text.lower()

    def test_summary_limits_improvements(self):
        pred = CyclingPrediction(energy=10.0, confidence=0.5)
        imps = [Improvement(action=f"Action {i}") for i in range(10)]
        text = format_predictions_text(pred, imps, summary=True)
        assert "more" in text


# ═══════════════════════════════════════════════════════════════════════
#  format_process_text
# ═══════════════════════════════════════════════════════════════════════

class TestFormatProcessText:
    def test_none(self):
        text = format_process_text(None)
        assert "not available" in text.lower()

    def test_with_report(self):
        report = ProcessReport(
            material_assessment="Good material",
            best_conditions={"electrolyte": "H2SO4"},
            bottleneck_analysis="Rs is high",
            production_recommendations=[
                ProductionRec(recommendation="Use H2SO4"),
            ],
            next_experiments=["Test KOH"],
            n_conditions=2,
        )
        text = format_process_text(report)
        assert "Good material" in text
        assert "H2SO4" in text
        assert "Test KOH" in text

    def test_summary_hides_experiments(self):
        report = ProcessReport(
            material_assessment="OK",
            next_experiments=["Exp 1", "Exp 2"],
            n_conditions=1,
        )
        text = format_process_text(report, summary=True)
        assert "Exp 1" not in text  # summary hides experiments

    def test_summary_truncates_recommendations(self):
        report = ProcessReport(
            material_assessment="OK",
            production_recommendations=[
                ProductionRec(recommendation=f"Rec {i}") for i in range(10)
            ],
            n_conditions=1,
        )
        text = format_process_text(report, summary=True)
        assert "more" in text


# ═══════════════════════════════════════════════════════════════════════
#  build_executive_summary
# ═══════════════════════════════════════════════════════════════════════

class TestBuildExecutiveSummary:
    def test_with_summary(self):
        report = AnalysisReport(summary="Everything is fine.")
        text = build_executive_summary(report)
        assert "Everything is fine" in text

    def test_without_summary_builds_one(self):
        report = AnalysisReport(
            sample_count=5,
            pipelines_used=["eis", "cycling"],
            quality_score=80.0,
            findings=[Finding(parameter="Rs")],
            anomalies=[],
            recommendations=[],
        )
        text = build_executive_summary(report)
        assert "5 sample" in text
        assert "80" in text

    def test_with_prediction(self):
        report = AnalysisReport(summary="OK.")
        pred = CyclingPrediction(retention=90.0, confidence=0.8)
        text = build_executive_summary(report, prediction=pred)
        assert "90.0" in text

    def test_with_process(self):
        report = AnalysisReport(summary="OK.")
        proc = ProcessReport(best_conditions={"electrolyte": "H2SO4"})
        text = build_executive_summary(report, process=proc)
        assert "H2SO4" in text


# ═══════════════════════════════════════════════════════════════════════
#  _assemble_full_report
# ═══════════════════════════════════════════════════════════════════════

class TestAssembleFullReport:
    def test_all_sections_present(self):
        text = _assemble_full_report(
            executive="Summary here",
            findings="Finding 1",
            anomalies="Anomaly 1",
            recommendations="Rec 1",
            predictions="Pred 1",
            process="Process 1",
            quality_score=75.0,
            summary_mode=False,
        )
        assert "Executive Summary" in text
        assert "Findings" in text
        assert "Anomalies" in text
        assert "Recommendations" in text
        assert "Predictions" in text
        assert "Process" in text
        assert "75" in text
        assert "Summary here" in text


# ═══════════════════════════════════════════════════════════════════════
#  State introspection helpers
# ═══════════════════════════════════════════════════════════════════════

class TestHasEis:
    def test_true(self):
        state = _make_state(eis=True, cycling=False, drt=False)
        assert _has_eis(state) is True

    def test_false_none(self):
        state = _make_state(eis=False, cycling=False, drt=False)
        assert _has_eis(state) is False

    def test_false_empty(self):
        state = SimpleNamespace(rank_df=pd.DataFrame())
        assert _has_eis(state) is False


class TestHasCycling:
    def test_true(self):
        state = _make_state(eis=False, cycling=True, drt=False)
        assert _has_cycling(state) is True

    def test_false(self):
        state = _make_state(eis=False, cycling=False, drt=False)
        assert _has_cycling(state) is False


class TestHasDrt:
    def test_true(self):
        state = _make_state(eis=False, cycling=False, drt=True)
        assert _has_drt(state) is True

    def test_false(self):
        state = _make_state(eis=False, cycling=False, drt=False)
        assert _has_drt(state) is False


# ═══════════════════════════════════════════════════════════════════════
#  Proxy builders
# ═══════════════════════════════════════════════════════════════════════

class TestSimpleProxy:
    def test_attrs(self):
        p = _SimpleProxy(a=1, b="hello")
        assert p.a == 1
        assert p.b == "hello"


class TestBuildEisProxy:
    def test_has_ranked_df(self):
        state = _make_state(eis=True, cycling=False, drt=False)
        proxy = _build_eis_proxy(state)
        assert proxy.ranked_df is not None
        assert not proxy.ranked_df.empty


class TestBuildCyclingProxy:
    def test_has_merged_table(self):
        state = _make_state(eis=False, cycling=True, drt=False)
        proxy = _build_cycling_proxy(state)
        assert proxy.merged_table is not None


class TestBuildDrtProxy:
    def test_has_tables(self):
        state = _make_state(eis=False, cycling=False, drt=True)
        proxy = _build_drt_proxy(state)
        assert proxy.drt_table is not None
        assert proxy.drt_summary_table is not None


# ═══════════════════════════════════════════════════════════════════════
#  _build_process_entries
# ═══════════════════════════════════════════════════════════════════════

class TestBuildProcessEntries:
    def test_from_index(self):
        state = _make_state(eis=True, cycling=False, drt=False)
        cfg = AIPanelConfig()
        entries = _build_process_entries(state, cfg)
        assert len(entries) == 1
        assert entries[0]["label"] == "H2SO4_1A_GCT.csv"
        assert "eis" in entries[0]

    def test_from_arquivo_column(self):
        df = pd.DataFrame({
            "Arquivo": ["s1.csv", "s2.csv"],
            "Rs_fit": [1.0, 2.0],
            "Score": [0.8, 0.6],
        })
        state = SimpleNamespace(rank_df=df)
        cfg = AIPanelConfig()
        entries = _build_process_entries(state, cfg)
        assert len(entries) == 2

    def test_empty_state(self):
        state = SimpleNamespace(rank_df=None)
        cfg = AIPanelConfig()
        entries = _build_process_entries(state, cfg)
        assert entries == []

    def test_scope_eis_off(self):
        state = _make_state(eis=True, cycling=False, drt=False)
        cfg = AIPanelConfig(scope_eis=False)
        entries = _build_process_entries(state, cfg)
        # Entries are still built, but without eis data
        assert len(entries) == 1
        assert "eis" not in entries[0]

    def test_numeric_index_fallback(self):
        """When index is numeric, should fallback to 'all_samples'."""
        df = pd.DataFrame({
            "Rs_fit": [1.0, 2.0],
            "Score": [0.8, 0.6],
        })
        # default integer index
        state = SimpleNamespace(rank_df=df)
        cfg = AIPanelConfig()
        entries = _build_process_entries(state, cfg)
        assert len(entries) == 1
        assert entries[0]["label"] == "all_samples"


# ═══════════════════════════════════════════════════════════════════════
#  run_ai_analysis — integration
# ═══════════════════════════════════════════════════════════════════════

class TestRunAIAnalysisEmpty:
    def test_empty_state(self):
        state = _make_state(eis=False, cycling=False, drt=False)
        result = run_ai_analysis(state)
        assert "No data" in result.executive_summary
        assert result.formatted_report != ""

    def test_empty_with_config(self):
        state = _make_state(eis=False, cycling=False, drt=False)
        cfg = AIPanelConfig(scope_eis=False, scope_cycling=False, scope_drt=False)
        result = run_ai_analysis(state, cfg)
        assert "No data" in result.executive_summary


class TestRunAIAnalysisEISOnly:
    def test_eis_only(self):
        state = _make_state(eis=True, cycling=False, drt=False)
        result = run_ai_analysis(state)
        assert "eis" in result.pipelines_used
        assert result.n_findings >= 0
        assert result.formatted_report != ""
        assert "Executive Summary" in result.formatted_report

    def test_has_predictions(self):
        state = _make_state(eis=True, cycling=False, drt=False)
        result = run_ai_analysis(state)
        assert result.has_predictions is True

    def test_has_process_report(self):
        state = _make_state(eis=True, cycling=False, drt=False)
        result = run_ai_analysis(state)
        assert result.has_process_report is True


class TestRunAIAnalysisAllPipelines:
    def test_all_pipelines(self):
        state = _make_state(eis=True, cycling=True, drt=True)
        result = run_ai_analysis(state)
        assert "eis" in result.pipelines_used
        assert result.formatted_report != ""
        assert "Findings" in result.formatted_report
        assert "Predictions" in result.formatted_report

    def test_quality_score(self):
        state = _make_state(eis=True, cycling=True, drt=True)
        result = run_ai_analysis(state)
        assert 0 <= result.quality_score <= 100


class TestRunAIAnalysisScopeFiltering:
    def test_eis_scope_off(self):
        state = _make_state(eis=True, cycling=True, drt=True)
        cfg = AIPanelConfig(scope_eis=False)
        result = run_ai_analysis(state, cfg)
        # Should still have cycling data
        assert result.formatted_report != ""
        # No predictions since EIS is off
        assert result.has_predictions is False

    def test_cycling_scope_off(self):
        state = _make_state(eis=True, cycling=True, drt=True)
        cfg = AIPanelConfig(scope_cycling=False)
        result = run_ai_analysis(state, cfg)
        assert result.has_predictions is True  # EIS still provides predictions


class TestRunAIAnalysisSummaryVsFull:
    def test_summary_mode(self):
        state = _make_state(eis=True, cycling=False, drt=False)
        cfg = AIPanelConfig(detail="summary")
        result = run_ai_analysis(state, cfg)
        assert result.formatted_report != ""

    def test_full_mode(self):
        state = _make_state(eis=True, cycling=False, drt=False)
        cfg = AIPanelConfig(detail="full")
        result = run_ai_analysis(state, cfg)
        assert result.formatted_report != ""


class TestRunAIAnalysisMultipleSamples:
    def test_multiple_samples(self):
        """Multiple samples in ranked_df → multiple process entries."""
        df = pd.DataFrame(
            {
                "Rs_fit": [1.0, 5.0],
                "Rp_fit": [100.0, 500.0],
                "Q": [1e-4, 1e-4],
                "n": [0.85, 0.7],
                "Sigma": [20.0, 50.0],
                "C_mean": [5e-5, 3e-5],
                "Tau": [0.01, 0.05],
                "Dispersion": [0.3, 0.5],
                "Energy_mean": [12.0, 8.0],
                "Score": [0.9, 0.4],
            },
            index=["H2SO4_1A_GCT.csv", "Li2SO4_1A_GC.csv"],
        )
        state = SimpleNamespace(
            rank_df=df,
            eis_df=df.copy(),
            raw_eis={},
            cic_df=None,
            cic_results={},
            drt_df=None,
            drt_peaks_df=None,
            drt_summary_df=None,
            drt_results={},
        )
        result = run_ai_analysis(state)
        assert result.has_process_report is True
        assert result.process_report.n_conditions >= 1


class TestRunAIAnalysisExceptionTolerance:
    def test_broken_eis_still_runs(self):
        """If the ranked_df has missing columns, analysis should still complete."""
        df = pd.DataFrame({"Score": [0.5]}, index=["sample.csv"])
        state = SimpleNamespace(
            rank_df=df,
            eis_df=df.copy(),
            raw_eis={},
            cic_df=None,
            cic_results={},
            drt_df=None,
            drt_peaks_df=None,
            drt_summary_df=None,
            drt_results={},
        )
        result = run_ai_analysis(state)
        assert result.formatted_report != ""
