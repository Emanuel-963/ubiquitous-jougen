"""Tests for the rule-based inference engine (Day 16).

Covers:
* Data classes (Finding, Anomaly, Recommendation, AnalysisReport, Priority)
* Measurement extractors (_extract_eis_measurements, _extract_cycling_measurements,
  _extract_drt_measurements)
* Anomaly detectors (_detect_anomalies_eis, _detect_anomalies_drt)
* Cross-pipeline reasoning (_cross_pipeline_findings)
* Quality score calculator
* Summary generator
* InferenceEngine.analyze() — EIS-only, EIS+cycling, EIS+DRT, all pipelines
* InferenceEngine.analyze_sample() — lightweight single-sample
* Edge cases: empty DataFrames, None results, NaN values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.ai.inference_engine import (
    AnalysisReport,
    Anomaly,
    Finding,
    InferenceEngine,
    Priority,
    Recommendation,
    _build_findings_from_matches,
    _build_recommendations,
    _compute_quality_score,
    _cross_pipeline_findings,
    _detect_anomalies_drt,
    _detect_anomalies_eis,
    _extract_cycling_measurements,
    _extract_drt_measurements,
    _extract_eis_measurements,
    _generate_summary,
    _safe_mean,
    _safe_median,
)
from src.ai.knowledge_base import (
    ElectrochemicalRule,
    KnowledgeBase,
    RuleMatch,
    Severity,
)
from src.config import PipelineConfig


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture()
def ranked_df_good() -> pd.DataFrame:
    """Healthy EIS ranked DataFrame with 4 samples."""
    return pd.DataFrame({
        "Rs_fit": [2.5, 2.8, 3.0, 2.6],
        "Rp_fit": [50.0, 55.0, 48.0, 52.0],
        "Q": [1e-5, 1.1e-5, 9.5e-6, 1.05e-5],
        "n": [0.85, 0.87, 0.83, 0.86],
        "Sigma": [10.0, 12.0, 11.0, 10.5],
        "C_mean": [1.2, 1.3, 1.1, 1.25],
        "Energy_mean": [15.0, 16.0, 14.5, 15.5],
        "Tau": [0.01, 0.012, 0.009, 0.011],
        "Dispersion": [0.2, 0.22, 0.19, 0.21],
        "Score": [80.0, 85.0, 78.0, 82.0],
        "Rank": [2, 1, 4, 3],
    }, index=["S1", "S2", "S3", "S4"])


@pytest.fixture()
def ranked_df_bad() -> pd.DataFrame:
    """EIS ranked DataFrame with anomalies."""
    return pd.DataFrame({
        "Rs_fit": [-1.0, 2.5, 30.0, 2.0],
        "Rp_fit": [-50.0, 55.0, 48.0, 52.0],
        "Q": [1e-5, 1.1e-5, 9.5e-6, 1.05e-5],
        "n": [1.2, 0.87, -0.1, 0.86],
        "Sigma": [10.0, 12.0, 11.0, 10.5],
    }, index=["Bad1", "S2", "Bad3", "S4"])


@pytest.fixture()
def cycling_result_good():
    """Stub CyclingResult with decent values."""
    @dataclass
    class _CyclingStub:
        merged_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    return _CyclingStub(merged_table=pd.DataFrame({
        "Retenção (%)": [92.0, 88.0, 90.0],
        "Energia (µJ)": [20.0, 19.0, 21.0],
        "Potência (µW)": [10.0, 9.5, 10.5],
    }))


@pytest.fixture()
def cycling_result_degraded():
    """Stub CyclingResult with low retention and power."""
    @dataclass
    class _CyclingStub:
        merged_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    return _CyclingStub(merged_table=pd.DataFrame({
        "Retenção (%)": [55.0, 50.0, 60.0],
        "Energia (µJ)": [3.0, 2.5, 3.5],
        "Potência (µW)": [2.0, 1.5, 2.5],
    }))


@pytest.fixture()
def drt_result_good():
    """Stub DRTPipelineResult."""
    @dataclass
    class _DRTStub:
        drt_table: pd.DataFrame = field(default_factory=pd.DataFrame)
        errors: Dict = field(default_factory=dict)
    return _DRTStub(drt_table=pd.DataFrame({
        "tau_peak_1": [0.01, 0.012, 0.009],
        "gamma_peak_1": [5.0, 4.8, 5.2],
        "tau_peak_2": [0.5, 0.45, 0.55],
    }))


@pytest.fixture()
def drt_result_with_errors():
    """Stub DRTPipelineResult with errors."""
    @dataclass
    class _DRTStub:
        drt_table: pd.DataFrame = field(default_factory=pd.DataFrame)
        errors: Dict = field(default_factory=dict)
    return _DRTStub(
        drt_table=pd.DataFrame({"tau_peak_1": [0.01]}),
        errors={"file_bad.csv": "DRT inversion failed"},
    )


@pytest.fixture()
def eis_result_good(ranked_df_good):
    """Stub EISResult wrapping good ranked_df."""
    @dataclass
    class _EISStub:
        ranked_df: pd.DataFrame = field(default_factory=pd.DataFrame)
        features_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    return _EISStub(ranked_df=ranked_df_good)


@pytest.fixture()
def eis_result_bad(ranked_df_bad):
    """Stub EISResult wrapping bad ranked_df."""
    @dataclass
    class _EISStub:
        ranked_df: pd.DataFrame = field(default_factory=pd.DataFrame)
        features_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    return _EISStub(ranked_df=ranked_df_bad)


# ═══════════════════════════════════════════════════════════════════════
#  Priority enum
# ═══════════════════════════════════════════════════════════════════════

class TestPriority:
    def test_values(self):
        assert Priority.HIGH.value == "high"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.LOW.value == "low"

    def test_str(self):
        assert str(Priority.HIGH) == "high"

    def test_is_string_enum(self):
        assert isinstance(Priority.HIGH, str)
        assert Priority.HIGH == "high"


# ═══════════════════════════════════════════════════════════════════════
#  Finding
# ═══════════════════════════════════════════════════════════════════════

class TestFinding:
    def test_defaults(self):
        f = Finding()
        assert f.parameter == ""
        assert f.value is None
        assert f.description == ""
        assert f.category == "general"

    def test_str_with_value(self):
        f = Finding(parameter="Rs", value=2.5, description="normal")
        assert "Rs = 2.5" in str(f)
        assert "normal" in str(f)

    def test_str_without_value(self):
        f = Finding(parameter="X", description="note")
        assert "X — note" == str(f)


# ═══════════════════════════════════════════════════════════════════════
#  Anomaly
# ═══════════════════════════════════════════════════════════════════════

class TestAnomaly:
    def test_defaults(self):
        a = Anomaly()
        assert a.severity == Severity.WARNING

    def test_str_with_value(self):
        a = Anomaly(parameter="Rp", value=-50.0, description="neg",
                     severity=Severity.CRITICAL)
        s = str(a)
        assert "[CRITICAL]" in s
        assert "Rp = -50" in s

    def test_str_without_value(self):
        a = Anomaly(parameter="DRT", description="err")
        assert "[WARNING]" in str(a)
        assert "DRT — err" in str(a)


# ═══════════════════════════════════════════════════════════════════════
#  Recommendation
# ═══════════════════════════════════════════════════════════════════════

class TestRecommendation:
    def test_defaults(self):
        r = Recommendation()
        assert r.priority == Priority.MEDIUM
        assert r.source_rule == ""

    def test_str(self):
        r = Recommendation(text="Fix electrode", priority=Priority.HIGH, source_rule="R1")
        assert "[HIGH] Fix electrode" == str(r)


# ═══════════════════════════════════════════════════════════════════════
#  AnalysisReport
# ═══════════════════════════════════════════════════════════════════════

class TestAnalysisReport:
    def test_defaults(self):
        r = AnalysisReport()
        assert r.findings == []
        assert r.anomalies == []
        assert r.recommendations == []
        assert r.quality_score == 0.0
        assert r.summary == ""
        assert r.sample_count == 0
        assert r.pipelines_used == []


# ═══════════════════════════════════════════════════════════════════════
#  _safe_median / _safe_mean
# ═══════════════════════════════════════════════════════════════════════

class TestSafeAggregates:
    def test_safe_median_normal(self):
        s = pd.Series([1.0, 3.0, 5.0])
        assert _safe_median(s) == 3.0

    def test_safe_median_with_nan(self):
        s = pd.Series([1.0, np.nan, 5.0])
        assert _safe_median(s) == 3.0

    def test_safe_median_empty(self):
        assert _safe_median(pd.Series([], dtype=float)) is None

    def test_safe_median_all_nan(self):
        assert _safe_median(pd.Series([np.nan, np.nan])) is None

    def test_safe_mean_normal(self):
        s = pd.Series([2.0, 4.0])
        assert _safe_mean(s) == 3.0

    def test_safe_mean_empty(self):
        assert _safe_mean(pd.Series([], dtype=float)) is None


# ═══════════════════════════════════════════════════════════════════════
#  _extract_eis_measurements
# ═══════════════════════════════════════════════════════════════════════

class TestExtractEIS:
    def test_good_df(self, ranked_df_good):
        m = _extract_eis_measurements(ranked_df_good)
        assert "Rs" in m
        assert "Rp" in m
        assert "n" in m
        assert "Score" in m
        assert 2.5 <= m["Rs"] <= 3.0
        assert 48.0 <= m["Rp"] <= 55.0

    def test_empty_df(self):
        m = _extract_eis_measurements(pd.DataFrame())
        assert m == {}

    def test_partial_cols(self):
        df = pd.DataFrame({"Rs_fit": [1.0, 2.0]})
        m = _extract_eis_measurements(df)
        assert "Rs" in m
        assert "Rp" not in m


# ═══════════════════════════════════════════════════════════════════════
#  _extract_cycling_measurements
# ═══════════════════════════════════════════════════════════════════════

class TestExtractCycling:
    def test_good_result(self, cycling_result_good):
        m = _extract_cycling_measurements(cycling_result_good)
        assert "retention" in m
        assert "Energy_mean" in m
        assert "Power_mean" in m
        assert 88 <= m["retention"] <= 92

    def test_none_result(self):
        m = _extract_cycling_measurements(None)
        assert m == {}

    def test_empty_table(self):
        @dataclass
        class _Stub:
            merged_table: pd.DataFrame = field(default_factory=pd.DataFrame)
        m = _extract_cycling_measurements(_Stub())
        assert m == {}


# ═══════════════════════════════════════════════════════════════════════
#  _extract_drt_measurements
# ═══════════════════════════════════════════════════════════════════════

class TestExtractDRT:
    def test_good_result(self, drt_result_good):
        m = _extract_drt_measurements(drt_result_good)
        assert "n_peaks" in m
        assert m["n_peaks"] == 2.0  # 2 peak columns
        assert "tau_peak_main" in m
        assert "gamma_peak_main" in m

    def test_none_result(self):
        m = _extract_drt_measurements(None)
        assert m == {}

    def test_empty_table(self):
        @dataclass
        class _Stub:
            drt_table: pd.DataFrame = field(default_factory=pd.DataFrame)
        m = _extract_drt_measurements(_Stub())
        assert m == {}


# ═══════════════════════════════════════════════════════════════════════
#  _detect_anomalies_eis
# ═══════════════════════════════════════════════════════════════════════

class TestDetectAnomaliesEIS:
    def test_good_df_no_anomalies(self, ranked_df_good):
        anomalies = _detect_anomalies_eis(ranked_df_good)
        assert len(anomalies) == 0

    def test_negative_rp(self, ranked_df_bad):
        anomalies = _detect_anomalies_eis(ranked_df_bad)
        rp_anoms = [a for a in anomalies if a.parameter == "Rp_fit"]
        assert len(rp_anoms) >= 1
        assert rp_anoms[0].severity == Severity.CRITICAL

    def test_negative_rs(self, ranked_df_bad):
        anomalies = _detect_anomalies_eis(ranked_df_bad)
        rs_anoms = [a for a in anomalies if a.parameter == "Rs_fit"]
        assert len(rs_anoms) >= 1
        assert rs_anoms[0].severity == Severity.CRITICAL

    def test_n_out_of_range(self, ranked_df_bad):
        anomalies = _detect_anomalies_eis(ranked_df_bad)
        n_anoms = [a for a in anomalies if a.parameter == "n"]
        assert len(n_anoms) >= 2  # n=1.2 and n=-0.1

    def test_empty_df(self):
        anomalies = _detect_anomalies_eis(pd.DataFrame())
        assert anomalies == []

    def test_high_cv_rs(self):
        """Rs with very high variability should generate warning."""
        df = pd.DataFrame({
            "Rs_fit": [1.0, 10.0, 50.0, 100.0],
            "Rp_fit": [50.0, 55.0, 48.0, 52.0],
        })
        anomalies = _detect_anomalies_eis(df)
        cv_anoms = [a for a in anomalies if "CV" in a.description]
        assert len(cv_anoms) == 1
        assert cv_anoms[0].severity == Severity.WARNING


# ═══════════════════════════════════════════════════════════════════════
#  _detect_anomalies_drt
# ═══════════════════════════════════════════════════════════════════════

class TestDetectAnomaliesDRT:
    def test_no_errors(self, drt_result_good):
        anomalies = _detect_anomalies_drt(drt_result_good)
        assert anomalies == []

    def test_with_errors(self, drt_result_with_errors):
        anomalies = _detect_anomalies_drt(drt_result_with_errors)
        assert len(anomalies) == 1
        assert "file_bad.csv" in anomalies[0].description


# ═══════════════════════════════════════════════════════════════════════
#  _cross_pipeline_findings
# ═══════════════════════════════════════════════════════════════════════

class TestCrossPipeline:
    def test_diffusion_bottleneck(self):
        eis_m = {"Rs": 2.0}
        cyc_m = {"Power_mean": 2.0}
        findings = _cross_pipeline_findings(eis_m, cyc_m, {})
        descs = [f.description for f in findings]
        assert any("gargalo" in d.lower() or "difusão" in d.lower() for d in descs)

    def test_mechanical_degradation(self):
        eis_m = {"Rp": 30.0}
        cyc_m = {"retention": 50.0}
        findings = _cross_pipeline_findings(eis_m, cyc_m, {})
        descs = [f.description for f in findings]
        assert any("mecânica" in d.lower() for d in descs)

    def test_heterogeneous_interface(self):
        eis_m = {"n": 0.5}
        drt_m = {"n_peaks": 4.0}
        findings = _cross_pipeline_findings(eis_m, {}, drt_m)
        descs = [f.description for f in findings]
        assert any("heterogénea" in d.lower() or "heterogenea" in d.lower() for d in descs)

    def test_diffusion_dominated(self):
        eis_m = {"Sigma": 80.0}
        drt_m = {"tau_peak_main": 0.5}
        findings = _cross_pipeline_findings(eis_m, {}, drt_m)
        descs = [f.description for f in findings]
        assert any("difusão" in d.lower() for d in descs)

    def test_rs_energy_bottleneck(self):
        eis_m = {"Rs": 15.0, "Energy_mean": 2.0}
        findings = _cross_pipeline_findings(eis_m, {}, {})
        descs = [f.description for f in findings]
        assert any("resistência" in d.lower() or "limitante" in d.lower() for d in descs)

    def test_no_findings_when_data_missing(self):
        findings = _cross_pipeline_findings({}, {}, {})
        assert findings == []


# ═══════════════════════════════════════════════════════════════════════
#  _compute_quality_score
# ═══════════════════════════════════════════════════════════════════════

class TestQualityScore:
    def test_perfect_score(self):
        assert _compute_quality_score([], []) == 100.0

    def test_critical_anomaly_deduction(self):
        anomalies = [Anomaly(severity=Severity.CRITICAL)]
        score = _compute_quality_score(anomalies, [])
        assert score == 85.0

    def test_warning_anomaly_deduction(self):
        anomalies = [Anomaly(severity=Severity.WARNING)]
        score = _compute_quality_score(anomalies, [])
        assert score == 95.0

    def test_rule_match_deduction(self):
        rule = ElectrochemicalRule(
            rule_id="T1", category="test", condition="x > 1",
            parameter="x", severity=Severity.CRITICAL,
        )
        matches = [RuleMatch(rule=rule, actual_value=2.0)]
        score = _compute_quality_score([], matches)
        assert score == 92.0

    def test_floor_at_zero(self):
        anomalies = [Anomaly(severity=Severity.CRITICAL)] * 10
        score = _compute_quality_score(anomalies, [])
        assert score == 0.0

    def test_combined(self):
        anomalies = [
            Anomaly(severity=Severity.CRITICAL),
            Anomaly(severity=Severity.WARNING),
        ]
        rule = ElectrochemicalRule(
            rule_id="T1", category="test", condition="x > 1",
            parameter="x", severity=Severity.WARNING,
        )
        matches = [RuleMatch(rule=rule, actual_value=2.0)]
        # 100 - 15 - 5 - 3 = 77
        score = _compute_quality_score(anomalies, matches)
        assert score == 77.0


# ═══════════════════════════════════════════════════════════════════════
#  _build_recommendations
# ═══════════════════════════════════════════════════════════════════════

class TestBuildRecommendations:
    def test_empty(self):
        assert _build_recommendations([]) == []

    def test_dedup(self):
        """Duplicate recommendation text should be de-duplicated."""
        rule = ElectrochemicalRule(
            rule_id="R1", category="test", condition="x > 1",
            parameter="x", severity=Severity.WARNING,
            recommendations=["Do A", "Do B"],
        )
        matches = [
            RuleMatch(rule=rule, actual_value=2.0),
            RuleMatch(rule=rule, actual_value=3.0),
        ]
        recs = _build_recommendations(matches)
        assert len(recs) == 2

    def test_sorted_by_priority(self):
        r_crit = ElectrochemicalRule(
            rule_id="R1", category="test", condition="x > 1",
            parameter="x", severity=Severity.CRITICAL,
            recommendations=["Fix urgently"],
        )
        r_info = ElectrochemicalRule(
            rule_id="R2", category="test", condition="y > 1",
            parameter="y", severity=Severity.INFO,
            recommendations=["Note this"],
        )
        matches = [
            RuleMatch(rule=r_info, actual_value=2.0),
            RuleMatch(rule=r_crit, actual_value=3.0),
        ]
        recs = _build_recommendations(matches)
        assert recs[0].priority == Priority.HIGH
        assert recs[1].priority == Priority.LOW


# ═══════════════════════════════════════════════════════════════════════
#  _build_findings_from_matches
# ═══════════════════════════════════════════════════════════════════════

class TestBuildFindings:
    def test_empty(self):
        assert _build_findings_from_matches([]) == []

    def test_conversion(self):
        rule = ElectrochemicalRule(
            rule_id="F1", category="impedance", condition="Rs > 10",
            parameter="Rs", severity=Severity.WARNING,
            interpretation="Rs is high",
        )
        match = RuleMatch(rule=rule, actual_value=15.0)
        findings = _build_findings_from_matches([match])
        assert len(findings) == 1
        assert findings[0].parameter == "Rs"
        assert findings[0].value == 15.0
        assert findings[0].description == "Rs is high"
        assert findings[0].category == "impedance"


# ═══════════════════════════════════════════════════════════════════════
#  _generate_summary
# ═══════════════════════════════════════════════════════════════════════

class TestSummary:
    def test_excellent(self):
        report = AnalysisReport(
            quality_score=90.0,
            sample_count=4,
            pipelines_used=["eis"],
        )
        summary = _generate_summary(report)
        assert "EXCELENTE" in summary
        assert "4" in summary

    def test_critical(self):
        report = AnalysisReport(
            quality_score=20.0,
            sample_count=1,
            pipelines_used=["eis"],
            anomalies=[Anomaly(severity=Severity.CRITICAL)] * 3,
        )
        summary = _generate_summary(report)
        assert "CRÍTICA" in summary
        assert "3 crítica(s)" in summary

    def test_recommendations_count(self):
        report = AnalysisReport(
            quality_score=70.0,
            sample_count=2,
            pipelines_used=["eis", "cycling"],
            recommendations=[
                Recommendation(text="A", priority=Priority.HIGH),
                Recommendation(text="B", priority=Priority.MEDIUM),
            ],
        )
        summary = _generate_summary(report)
        assert "2 recomendação" in summary
        assert "1 de alta prioridade" in summary

    def test_good_quality(self):
        report = AnalysisReport(quality_score=65.0, sample_count=1,
                                pipelines_used=["eis"])
        summary = _generate_summary(report)
        assert "BOA" in summary

    def test_moderate_quality(self):
        report = AnalysisReport(quality_score=45.0, sample_count=1,
                                pipelines_used=["eis"])
        summary = _generate_summary(report)
        assert "MODERADA" in summary


# ═══════════════════════════════════════════════════════════════════════
#  InferenceEngine — construction
# ═══════════════════════════════════════════════════════════════════════

class TestInferenceEngineInit:
    def test_default_construction(self):
        eng = InferenceEngine()
        assert eng.knowledge_base is not None
        assert len(eng.knowledge_base._rules) > 0

    def test_custom_kb(self):
        kb = KnowledgeBase()
        eng = InferenceEngine(knowledge_base=kb)
        assert eng.knowledge_base is kb

    def test_custom_config(self):
        cfg = PipelineConfig.default()
        eng = InferenceEngine(config=cfg)
        assert eng._config is cfg


# ═══════════════════════════════════════════════════════════════════════
#  InferenceEngine.analyze — EIS only
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeEISOnly:
    def test_good_eis(self, eis_result_good):
        eng = InferenceEngine()
        report = eng.analyze(eis_result=eis_result_good)
        assert report.pipelines_used == ["eis"]
        assert report.sample_count == 4
        assert 0.0 <= report.quality_score <= 100.0
        assert report.summary != ""

    def test_bad_eis_detects_anomalies(self, eis_result_bad):
        eng = InferenceEngine()
        report = eng.analyze(eis_result=eis_result_bad)
        assert len(report.anomalies) > 0
        # Should detect negative Rp, negative Rs, n out of range
        params = {a.parameter for a in report.anomalies}
        assert "Rp_fit" in params
        assert "Rs_fit" in params
        assert "n" in params
        assert report.quality_score < 100.0


# ═══════════════════════════════════════════════════════════════════════
#  InferenceEngine.analyze — EIS + Cycling
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeEISCycling:
    def test_good(self, eis_result_good, cycling_result_good):
        eng = InferenceEngine()
        report = eng.analyze(
            eis_result=eis_result_good,
            cycling_result=cycling_result_good,
        )
        assert "eis" in report.pipelines_used
        assert "cycling" in report.pipelines_used

    def test_cross_pipeline_findings(self, eis_result_good, cycling_result_degraded):
        """Low retention + low power should trigger cross-pipeline findings."""
        eng = InferenceEngine()
        report = eng.analyze(
            eis_result=eis_result_good,
            cycling_result=cycling_result_degraded,
        )
        # Should have at least one cross-pipeline finding
        cross = [f for f in report.findings if f.category == "cross-pipeline"]
        assert len(cross) >= 1


# ═══════════════════════════════════════════════════════════════════════
#  InferenceEngine.analyze — EIS + DRT
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeEISDRT:
    def test_good(self, eis_result_good, drt_result_good):
        eng = InferenceEngine()
        report = eng.analyze(
            eis_result=eis_result_good,
            drt_result=drt_result_good,
        )
        assert "drt" in report.pipelines_used
        assert report.sample_count == 4

    def test_drt_errors(self, eis_result_good, drt_result_with_errors):
        eng = InferenceEngine()
        report = eng.analyze(
            eis_result=eis_result_good,
            drt_result=drt_result_with_errors,
        )
        drt_anoms = [a for a in report.anomalies if a.parameter == "DRT"]
        assert len(drt_anoms) >= 1


# ═══════════════════════════════════════════════════════════════════════
#  InferenceEngine.analyze — all pipelines
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeAll:
    def test_all_pipelines(self, eis_result_good, cycling_result_good, drt_result_good):
        eng = InferenceEngine()
        report = eng.analyze(
            eis_result=eis_result_good,
            cycling_result=cycling_result_good,
            drt_result=drt_result_good,
        )
        assert set(report.pipelines_used) == {"eis", "cycling", "drt"}
        assert report.summary != ""
        assert report.quality_score >= 0.0


# ═══════════════════════════════════════════════════════════════════════
#  InferenceEngine.analyze — edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeEdgeCases:
    def test_no_results(self):
        eng = InferenceEngine()
        report = eng.analyze()
        assert report.pipelines_used == []
        assert report.sample_count == 0
        assert report.quality_score == 100.0

    def test_eis_none_ranked(self):
        """EISResult with ranked_df = None."""
        @dataclass
        class _EISStub:
            ranked_df = None
        eng = InferenceEngine()
        report = eng.analyze(eis_result=_EISStub())
        assert "eis" in report.pipelines_used
        assert report.sample_count == 0

    def test_cycling_empty_table(self):
        @dataclass
        class _CycStub:
            merged_table: pd.DataFrame = field(default_factory=pd.DataFrame)
        eng = InferenceEngine()
        report = eng.analyze(cycling_result=_CycStub())
        assert "cycling" in report.pipelines_used

    def test_drt_empty_table(self):
        @dataclass
        class _DRTStub:
            drt_table: pd.DataFrame = field(default_factory=pd.DataFrame)
            errors: Dict = field(default_factory=dict)
        eng = InferenceEngine()
        report = eng.analyze(drt_result=_DRTStub())
        assert "drt" in report.pipelines_used


# ═══════════════════════════════════════════════════════════════════════
#  InferenceEngine.analyze_sample
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeSample:
    def test_basic(self):
        eng = InferenceEngine()
        measurements = {"Rs": 50.0, "Rp": 500.0, "n": 0.5}
        report = eng.analyze_sample(measurements)
        assert report.sample_count == 1
        assert report.summary != ""

    def test_with_categories(self):
        eng = InferenceEngine()
        measurements = {"Rs": 50.0}
        report = eng.analyze_sample(measurements, categories=["impedance"])
        assert "impedance" in report.pipelines_used

    def test_empty_measurements(self):
        eng = InferenceEngine()
        report = eng.analyze_sample({})
        assert report.quality_score == 100.0
        assert report.findings == []

    def test_high_rs_triggers_finding(self):
        eng = InferenceEngine()
        measurements = {"Rs": 100.0}
        report = eng.analyze_sample(measurements, categories=["impedance"])
        # Should trigger rules about high Rs
        rs_findings = [f for f in report.findings if f.parameter == "Rs"]
        assert len(rs_findings) >= 1


# ═══════════════════════════════════════════════════════════════════════
#  Re-export checks
# ═══════════════════════════════════════════════════════════════════════

class TestReExports:
    def test_ai_package(self):
        from src.ai import (
            AnalysisReport,
            Anomaly,
            Finding,
            InferenceEngine,
            Priority,
            Recommendation,
        )
        assert AnalysisReport is not None
        assert InferenceEngine is not None

    def test_src_package(self):
        from src import (
            AnalysisReport,
            Anomaly,
            Finding,
            InferenceEngine,
            Priority,
            Recommendation,
        )
        assert InferenceEngine is not None


# ═══════════════════════════════════════════════════════════════════════
#  NaN handling
# ═══════════════════════════════════════════════════════════════════════

class TestNaNHandling:
    def test_eis_with_nans(self):
        df = pd.DataFrame({
            "Rs_fit": [np.nan, 2.0, np.nan],
            "Rp_fit": [50.0, np.nan, 45.0],
            "n": [0.8, 0.9, np.nan],
        })
        m = _extract_eis_measurements(df)
        assert m.get("Rs") == 2.0
        assert 45.0 <= m.get("Rp", 0) <= 50.0

    def test_engine_with_nan_df(self):
        @dataclass
        class _EISStub:
            ranked_df: pd.DataFrame = field(default_factory=pd.DataFrame)
        eis = _EISStub(ranked_df=pd.DataFrame({
            "Rs_fit": [np.nan, np.nan],
            "Rp_fit": [np.nan, np.nan],
        }))
        eng = InferenceEngine()
        report = eng.analyze(eis_result=eis)
        assert report.sample_count == 2
        assert report.quality_score >= 0.0
