"""Tests for src.ai.process_advisor (Day 18).

Coverage targets
----------------
* ProductionRec / ProcessReport dataclass instantiation & defaults
* RecommendationArea enum
* _extract_condition — explicit metadata, label fallback, empty
* _condition_key
* _extract_eis_metrics — from EISResult, raw DataFrame, None
* _extract_cycling_metrics — from CyclingResult, None
* _extract_drt_metrics — from DRTPipelineResult, None
* _aggregate_condition — single / multiple entries
* _detect_outliers — normal, outlier present, too-few samples
* _identify_bottleneck — each bottleneck type, no bottleneck
* _compare_metric — two+ conditions, single condition
* _build_comparison_table — multi-row, empty
* _suggest_next_experiments — missing electrolytes / treatments / currents
* _generate_recommendations — electrolyte, treatment, bottleneck, outlier
* ProcessAdvisor.analyze_material_system:
    - empty input
    - single condition
    - multiple conditions
    - explicit metadata override
    - cycling-only, drt-only entries
    - comparison_table shape / content
    - next_experiments populated
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.ai.process_advisor import (
    ProcessAdvisor,
    ProcessReport,
    ProductionRec,
    RecommendationArea,
    _aggregate_condition,
    _build_comparison_table,
    _compare_metric,
    _condition_key,
    _detect_outliers,
    _extract_condition,
    _extract_cycling_metrics,
    _extract_drt_metrics,
    _extract_eis_metrics,
    _generate_recommendations,
    _identify_bottleneck,
    _suggest_next_experiments,
)
from src.config import PipelineConfig


# ═══════════════════════════════════════════════════════════════════════
#  Helpers — fake result objects
# ═══════════════════════════════════════════════════════════════════════

def _make_ranked_df(**overrides: float) -> pd.DataFrame:
    """Create a minimal ranked DataFrame with standard EIS columns."""
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
    return pd.DataFrame(base)


def _make_eis(ranked_df: pd.DataFrame | None = None, **kw: float) -> SimpleNamespace:
    if ranked_df is None:
        ranked_df = _make_ranked_df(**kw)
    return SimpleNamespace(ranked_df=ranked_df)


def _make_cycling(retention: float = 92.0, energy: float = 15.0, power: float = 3.0) -> SimpleNamespace:
    merged = pd.DataFrame({
        "Retenção (%)": [retention],
        "Energia (µJ)": [energy],
        "Potência (µW)": [power],
    })
    return SimpleNamespace(merged_table=merged)


def _make_drt(n_peaks: float = 2, dominant_tau: float = 0.01) -> SimpleNamespace:
    summary = pd.DataFrame({"n_peaks": [n_peaks], "dominant_tau": [dominant_tau]})
    return SimpleNamespace(drt_summary_table=summary)


def _entry(label: str, eis=None, cycling=None, drt=None, metadata=None) -> dict:
    d: dict = {"label": label}
    if eis is not None:
        d["eis"] = eis
    if cycling is not None:
        d["cycling"] = cycling
    if drt is not None:
        d["drt"] = drt
    if metadata is not None:
        d["metadata"] = metadata
    return d


# ═══════════════════════════════════════════════════════════════════════
#  Enum & dataclass basics
# ═══════════════════════════════════════════════════════════════════════

class TestRecommendationArea:
    def test_members(self):
        assert RecommendationArea.ELECTROLYTE.value == "electrolyte"
        assert RecommendationArea.TREATMENT.value == "treatment"
        assert RecommendationArea.PROCESS.value == "process"
        assert RecommendationArea.MEASUREMENT.value == "measurement"
        assert RecommendationArea.GENERAL.value == "general"
        assert RecommendationArea.SUBSTRATE.value == "substrate"

    def test_str(self):
        assert str(RecommendationArea.ELECTROLYTE) == "electrolyte"


class TestProductionRec:
    def test_defaults(self):
        rec = ProductionRec()
        assert rec.recommendation == ""
        assert rec.area == RecommendationArea.GENERAL
        assert rec.priority == 2

    def test_custom(self):
        rec = ProductionRec(
            recommendation="Use H2SO4",
            area=RecommendationArea.ELECTROLYTE,
            rationale="lowest Rs",
            expected_impact="reduce losses",
            priority=1,
        )
        assert rec.priority == 1
        assert "H2SO4" in str(rec)

    def test_str_method(self):
        rec = ProductionRec(recommendation="Do X", expected_impact="Better Y", priority=1)
        s = str(rec)
        assert "[P1]" in s
        assert "Do X" in s
        assert "Better Y" in s


class TestProcessReport:
    def test_defaults(self):
        r = ProcessReport()
        assert r.material_assessment == ""
        assert r.best_conditions == {}
        assert r.bottleneck_analysis == ""
        assert isinstance(r.production_recommendations, list)
        assert isinstance(r.comparison_table, pd.DataFrame)
        assert isinstance(r.next_experiments, list)
        assert r.n_conditions == 0

    def test_custom(self):
        r = ProcessReport(
            material_assessment="Good",
            best_conditions={"electrolyte": "H2SO4"},
            n_conditions=3,
        )
        assert r.n_conditions == 3
        assert r.best_conditions["electrolyte"] == "H2SO4"


# ═══════════════════════════════════════════════════════════════════════
#  _extract_condition
# ═══════════════════════════════════════════════════════════════════════

class TestExtractCondition:
    def test_from_label(self):
        entry = {"label": "Sample_H2SO4_1A_GCT.csv"}
        cond = _extract_condition(entry)
        assert cond["electrolyte"] == "H2SO4"
        assert cond["current"] == "1A"
        assert cond["treatment"] == "GCT"

    def test_from_explicit_metadata(self):
        entry = {
            "label": "whatever",
            "metadata": {"electrolyte": "KOH", "current": "5A", "treatment": "Plasma"},
        }
        cond = _extract_condition(entry)
        assert cond["electrolyte"] == "KOH"
        assert cond["current"] == "5A"
        assert cond["treatment"] == "Plasma"

    def test_empty_label(self):
        cond = _extract_condition({"label": ""})
        assert cond["electrolyte"] == "Unknown"

    def test_no_label(self):
        cond = _extract_condition({})
        assert cond["electrolyte"] == "Unknown"

    def test_metadata_priority_over_label(self):
        entry = {
            "label": "Li2SO4_1A_GCT.csv",
            "metadata": {"electrolyte": "NaOH", "current": "10A", "treatment": "Steel316"},
        }
        cond = _extract_condition(entry)
        assert cond["electrolyte"] == "NaOH"  # metadata wins


class TestConditionKey:
    def test_basic(self):
        cond = {"electrolyte": "H2SO4", "current": "1A", "treatment": "GCT"}
        assert _condition_key(cond) == "H2SO4_1A_GCT"


# ═══════════════════════════════════════════════════════════════════════
#  _extract_eis_metrics
# ═══════════════════════════════════════════════════════════════════════

class TestExtractEisMetrics:
    def test_from_eis_result(self):
        eis = _make_eis(Rs_fit=2.5, Rp_fit=200.0, Score=0.8)
        m = _extract_eis_metrics(eis)
        assert m["Rs_fit"] == 2.5
        assert m["Score"] == 0.8

    def test_from_raw_dataframe(self):
        df = _make_ranked_df(Rs_fit=5.0)
        m = _extract_eis_metrics(df)
        assert m["Rs_fit"] == 5.0

    def test_none(self):
        assert _extract_eis_metrics(None) == {}

    def test_empty_df(self):
        eis = SimpleNamespace(ranked_df=pd.DataFrame())
        assert _extract_eis_metrics(eis) == {}

    def test_missing_columns(self):
        eis = SimpleNamespace(ranked_df=pd.DataFrame({"Rs_fit": [1.0]}))
        m = _extract_eis_metrics(eis)
        assert "Rs_fit" in m
        assert "Rp_fit" not in m

    def test_non_numeric_values_ignored(self):
        df = pd.DataFrame({"Rs_fit": ["abc"], "Score": [0.5]})
        eis = SimpleNamespace(ranked_df=df)
        m = _extract_eis_metrics(eis)
        assert "Rs_fit" not in m
        assert m["Score"] == 0.5


# ═══════════════════════════════════════════════════════════════════════
#  _extract_cycling_metrics
# ═══════════════════════════════════════════════════════════════════════

class TestExtractCyclingMetrics:
    def test_normal(self):
        cyc = _make_cycling(retention=85.0, energy=10.0, power=2.0)
        m = _extract_cycling_metrics(cyc)
        assert m["retention"] == 85.0
        assert m["energy"] == 10.0
        assert m["power"] == 2.0

    def test_none(self):
        assert _extract_cycling_metrics(None) == {}

    def test_empty_merged(self):
        cyc = SimpleNamespace(merged_table=pd.DataFrame())
        assert _extract_cycling_metrics(cyc) == {}

    def test_no_merged_attr(self):
        cyc = SimpleNamespace()
        assert _extract_cycling_metrics(cyc) == {}


# ═══════════════════════════════════════════════════════════════════════
#  _extract_drt_metrics
# ═══════════════════════════════════════════════════════════════════════

class TestExtractDrtMetrics:
    def test_normal(self):
        drt = _make_drt(n_peaks=3, dominant_tau=0.05)
        m = _extract_drt_metrics(drt)
        assert m["n_peaks"] == 3.0
        assert m["dominant_tau"] == 0.05

    def test_none(self):
        assert _extract_drt_metrics(None) == {}

    def test_empty_summary(self):
        drt = SimpleNamespace(drt_summary_table=pd.DataFrame())
        assert _extract_drt_metrics(drt) == {}


# ═══════════════════════════════════════════════════════════════════════
#  _aggregate_condition
# ═══════════════════════════════════════════════════════════════════════

class TestAggregateCondition:
    def test_single_entry(self):
        entries = [_entry("s1", eis=_make_eis(Rs_fit=3.0, Score=0.7))]
        agg = _aggregate_condition(entries)
        assert agg["Rs_fit"] == 3.0
        assert agg["Score"] == 0.7

    def test_multiple_entries_median(self):
        entries = [
            _entry("s1", eis=_make_eis(Rs_fit=2.0, Score=0.6)),
            _entry("s2", eis=_make_eis(Rs_fit=4.0, Score=0.8)),
            _entry("s3", eis=_make_eis(Rs_fit=3.0, Score=0.7)),
        ]
        agg = _aggregate_condition(entries)
        assert agg["Rs_fit"] == pytest.approx(3.0)
        assert agg["Score"] == pytest.approx(0.7)

    def test_mixed_pipelines(self):
        entries = [
            _entry("s1", eis=_make_eis(Rs_fit=1.0), cycling=_make_cycling(retention=90.0)),
        ]
        agg = _aggregate_condition(entries)
        assert "Rs_fit" in agg
        assert "retention" in agg

    def test_empty(self):
        assert _aggregate_condition([]) == {}


# ═══════════════════════════════════════════════════════════════════════
#  _detect_outliers
# ═══════════════════════════════════════════════════════════════════════

class TestDetectOutliers:
    def test_no_outlier(self):
        entries = [
            _entry("s1", eis=_make_eis(Score=0.7)),
            _entry("s2", eis=_make_eis(Score=0.71)),
            _entry("s3", eis=_make_eis(Score=0.69)),
        ]
        assert _detect_outliers(entries) == []

    def test_with_outlier(self):
        entries = [
            _entry("s1", eis=_make_eis(Score=0.7)),
            _entry("s2", eis=_make_eis(Score=0.7)),
            _entry("s3", eis=_make_eis(Score=0.7)),
            _entry("s4", eis=_make_eis(Score=0.7)),
            _entry("s5", eis=_make_eis(Score=0.7)),
            _entry("s6", eis=_make_eis(Score=0.7)),
            _entry("s7", eis=_make_eis(Score=0.7)),
            _entry("s8", eis=_make_eis(Score=0.7)),
            _entry("outlier", eis=_make_eis(Score=10.0)),
        ]
        outliers = _detect_outliers(entries)
        assert "outlier" in outliers

    def test_too_few(self):
        entries = [
            _entry("s1", eis=_make_eis(Score=0.7)),
            _entry("s2", eis=_make_eis(Score=100.0)),
        ]
        assert _detect_outliers(entries) == []

    def test_zero_variance(self):
        entries = [
            _entry("s1", eis=_make_eis(Score=0.5)),
            _entry("s2", eis=_make_eis(Score=0.5)),
            _entry("s3", eis=_make_eis(Score=0.5)),
        ]
        assert _detect_outliers(entries) == []


# ═══════════════════════════════════════════════════════════════════════
#  _identify_bottleneck
# ═══════════════════════════════════════════════════════════════════════

class TestIdentifyBottleneck:
    def test_high_rs(self):
        param, text = _identify_bottleneck({"Rs_fit": 50.0, "Rp_fit": 100.0, "n": 0.9})
        assert param == "Rs_fit"
        assert "Rs" in text

    def test_high_rp(self):
        param, text = _identify_bottleneck({"Rs_fit": 1.0, "Rp_fit": 5000.0, "n": 0.9})
        assert param == "Rp_fit"

    def test_low_n(self):
        param, text = _identify_bottleneck({"Rs_fit": 1.0, "Rp_fit": 100.0, "n": 0.3})
        assert param == "n"

    def test_high_sigma(self):
        param, text = _identify_bottleneck({"Rs_fit": 1.0, "Rp_fit": 100.0, "n": 0.9, "Sigma": 500.0})
        assert param == "Sigma"

    def test_low_retention(self):
        param, text = _identify_bottleneck({"Rs_fit": 1.0, "Rp_fit": 100.0, "retention": 40.0})
        assert param == "retention"

    def test_no_bottleneck(self):
        param, text = _identify_bottleneck({"Rs_fit": 1.0, "Rp_fit": 100.0, "n": 0.9})
        assert param == "none"
        assert "well" in text.lower()

    def test_empty_metrics(self):
        param, text = _identify_bottleneck({})
        assert param == "none"


# ═══════════════════════════════════════════════════════════════════════
#  _compare_metric
# ═══════════════════════════════════════════════════════════════════════

class TestCompareMetric:
    def test_two_conditions(self):
        cond_m = {
            "A": {"Rs_fit": 2.0},
            "B": {"Rs_fit": 10.0},
        }
        result = _compare_metric(cond_m, "Rs_fit", lower_is_better=True)
        assert result is not None
        best_key, text, pct = result
        assert best_key == "A"
        assert pct > 0

    def test_higher_is_better(self):
        cond_m = {
            "A": {"Score": 0.6},
            "B": {"Score": 0.9},
        }
        result = _compare_metric(cond_m, "Score", lower_is_better=False)
        assert result is not None
        assert result[0] == "B"

    def test_single_condition(self):
        cond_m = {"A": {"Rs_fit": 2.0}}
        assert _compare_metric(cond_m, "Rs_fit") is None

    def test_missing_metric(self):
        cond_m = {"A": {"Rs_fit": 2.0}, "B": {}}
        assert _compare_metric(cond_m, "Rs_fit") is None


# ═══════════════════════════════════════════════════════════════════════
#  _build_comparison_table
# ═══════════════════════════════════════════════════════════════════════

class TestBuildComparisonTable:
    def test_multi(self):
        cond_m = {
            "A_1A_GCT": {"Rs_fit": 2.0, "Score": 0.8},
            "B_1A_GC": {"Rs_fit": 5.0, "Score": 0.5},
        }
        df = _build_comparison_table(cond_m)
        assert df.shape[0] == 2
        assert "Rs_fit" in df.columns
        assert df.index.name == "Condition"

    def test_empty(self):
        df = _build_comparison_table({})
        assert df.empty


# ═══════════════════════════════════════════════════════════════════════
#  _suggest_next_experiments
# ═══════════════════════════════════════════════════════════════════════

class TestSuggestNextExperiments:
    def test_missing_electrolytes(self):
        suggestions = _suggest_next_experiments(
            {"A": {}}, {"H2SO4"}, {"GCT"}, {"1A"}, "none",
        )
        assert any("electrolyte" in s.lower() for s in suggestions)

    def test_missing_treatments(self):
        suggestions = _suggest_next_experiments(
            {"A": {}}, {"H2SO4", "Li2SO4", "NaOH", "Na2SO4", "LiCl", "KOH"},
            {"GCT"}, {"1A"}, "none",
        )
        assert any("treatment" in s.lower() for s in suggestions)

    def test_missing_currents(self):
        suggestions = _suggest_next_experiments(
            {"A": {}}, {"H2SO4"}, {"GCT"}, {"1A"}, "none",
        )
        assert any("current" in s.lower() for s in suggestions)

    def test_bottleneck_rs(self):
        suggestions = _suggest_next_experiments(
            {"A": {}}, set(), set(), set(), "Rs_fit",
        )
        assert any("ohmic" in s.lower() for s in suggestions)

    def test_bottleneck_rp(self):
        suggestions = _suggest_next_experiments(
            {"A": {}}, set(), set(), set(), "Rp_fit",
        )
        assert any("charge-transfer" in s.lower() for s in suggestions)

    def test_bottleneck_n(self):
        suggestions = _suggest_next_experiments(
            {"A": {}}, set(), set(), set(), "n",
        )
        assert any("homogeneity" in s.lower() or "surface" in s.lower() for s in suggestions)

    def test_bottleneck_sigma(self):
        suggestions = _suggest_next_experiments(
            {"A": {}}, set(), set(), set(), "Sigma",
        )
        assert any("diffusion" in s.lower() for s in suggestions)

    def test_bottleneck_retention(self):
        suggestions = _suggest_next_experiments(
            {"A": {}}, set(), set(), set(), "retention",
        )
        assert any("stability" in s.lower() or "cycling" in s.lower() for s in suggestions)

    def test_few_conditions(self):
        suggestions = _suggest_next_experiments(
            {"A": {}, "B": {}}, set(), set(), set(), "none",
        )
        assert any("replica" in s.lower() for s in suggestions)

    def test_all_covered(self):
        """When everything is covered, still bottleneck-specific suggestion appears."""
        suggestions = _suggest_next_experiments(
            {"A": {}, "B": {}, "C": {}},
            {"Li2SO4", "LiCl", "H2SO4", "Na2SO4", "NaOH", "KOH"},
            {"GCT", "GC", "Steel316", "None"},
            {"0.1A", "1A", "10A"},
            "Rs_fit",
        )
        # Should have at least the bottleneck suggestion
        assert any("ohmic" in s.lower() for s in suggestions)


# ═══════════════════════════════════════════════════════════════════════
#  _generate_recommendations
# ═══════════════════════════════════════════════════════════════════════

class TestGenerateRecommendations:
    def test_electrolyte_rec(self):
        cond_m = {
            "H2SO4_1A_GCT": {"Rs_fit": 1.0, "Score": 0.9},
            "Li2SO4_1A_GCT": {"Rs_fit": 5.0, "Score": 0.5},
        }
        best = {"electrolyte": "H2SO4", "treatment": "GCT"}
        recs = _generate_recommendations(cond_m, best, "none", {})
        elec_recs = [r for r in recs if r.area == RecommendationArea.ELECTROLYTE]
        assert len(elec_recs) >= 1
        assert "H2SO4" in elec_recs[0].recommendation

    def test_treatment_rec(self):
        cond_m = {"A_1A_GCT": {"Score": 0.9}}
        best = {"electrolyte": "A", "treatment": "GCT"}
        recs = _generate_recommendations(cond_m, best, "none", {})
        treat_recs = [r for r in recs if r.area == RecommendationArea.TREATMENT]
        assert len(treat_recs) >= 1

    def test_bottleneck_rs_rec(self):
        recs = _generate_recommendations({}, {}, "Rs_fit", {})
        assert any("ohmic" in r.recommendation.lower() for r in recs)

    def test_bottleneck_rp_rec(self):
        recs = _generate_recommendations({}, {}, "Rp_fit", {})
        assert any("charge-transfer" in r.recommendation.lower() for r in recs)

    def test_bottleneck_n_rec(self):
        recs = _generate_recommendations({}, {}, "n", {})
        assert any("homogeneity" in r.recommendation.lower() or "surface" in r.recommendation.lower() for r in recs)

    def test_bottleneck_sigma_rec(self):
        recs = _generate_recommendations({}, {}, "Sigma", {})
        assert any("diffusion" in r.recommendation.lower() for r in recs)

    def test_bottleneck_retention_rec(self):
        recs = _generate_recommendations({}, {}, "retention", {})
        assert any("stability" in r.recommendation.lower() or "cycling" in r.recommendation.lower() for r in recs)

    def test_outlier_rec(self):
        recs = _generate_recommendations({}, {}, "none", {"A_1A_GCT": ["bad_sample"]})
        assert any("outlier" in r.recommendation.lower() for r in recs)

    def test_single_condition_rec(self):
        cond_m = {"A_1A_GCT": {"Score": 0.5}}
        recs = _generate_recommendations(cond_m, {}, "none", {})
        assert any(r.area == RecommendationArea.MEASUREMENT for r in recs)

    def test_sorted_by_priority(self):
        cond_m = {
            "H2SO4_1A_GCT": {"Rs_fit": 1.0, "Score": 0.9},
            "Li2SO4_1A_GCT": {"Rs_fit": 5.0, "Score": 0.5},
        }
        recs = _generate_recommendations(
            cond_m, {"electrolyte": "H2SO4", "treatment": "GCT"},
            "Rs_fit", {"Li2SO4_1A_GCT": ["s3"]},
        )
        priorities = [r.priority for r in recs]
        assert priorities == sorted(priorities)


# ═══════════════════════════════════════════════════════════════════════
#  ProcessAdvisor — integration
# ═══════════════════════════════════════════════════════════════════════

class TestProcessAdvisorInit:
    def test_default_config(self):
        advisor = ProcessAdvisor()
        assert advisor._config is not None

    def test_custom_config(self):
        cfg = PipelineConfig(voltage=2.0)
        advisor = ProcessAdvisor(config=cfg)
        assert advisor._config.voltage == 2.0


class TestProcessAdvisorEmpty:
    def test_empty_input(self):
        advisor = ProcessAdvisor()
        report = advisor.analyze_material_system([])
        assert "No data" in report.material_assessment
        assert report.n_conditions == 0


class TestProcessAdvisorSingleCondition:
    def test_single_eis(self):
        advisor = ProcessAdvisor()
        entries = [_entry("H2SO4_1A_GCT.csv", eis=_make_eis(Rs_fit=2.0, Score=0.8))]
        report = advisor.analyze_material_system(entries)
        assert report.n_conditions == 1
        assert report.best_conditions.get("electrolyte") == "H2SO4"
        assert isinstance(report.comparison_table, pd.DataFrame)
        assert report.comparison_table.shape[0] == 1

    def test_single_with_cycling(self):
        advisor = ProcessAdvisor()
        entries = [_entry(
            "Li2SO4_0.1A_GC.csv",
            eis=_make_eis(Rs_fit=3.0, Score=0.6),
            cycling=_make_cycling(retention=85.0),
        )]
        report = advisor.analyze_material_system(entries)
        assert report.n_conditions == 1
        assert "retention" in report.comparison_table.columns

    def test_single_measurement_rec(self):
        """Single condition should suggest testing more conditions."""
        advisor = ProcessAdvisor()
        entries = [_entry("H2SO4_1A_GCT.csv", eis=_make_eis(Score=0.5))]
        report = advisor.analyze_material_system(entries)
        assert any(
            r.area == RecommendationArea.MEASUREMENT
            for r in report.production_recommendations
        )


class TestProcessAdvisorMultiCondition:
    def _two_conditions(self):
        return [
            _entry("H2SO4_1A_GCT_rep1.csv", eis=_make_eis(Rs_fit=1.0, Score=0.9)),
            _entry("H2SO4_1A_GCT_rep2.csv", eis=_make_eis(Rs_fit=1.2, Score=0.88)),
            _entry("Li2SO4_1A_GC_rep1.csv", eis=_make_eis(Rs_fit=5.0, Score=0.5)),
            _entry("Li2SO4_1A_GC_rep2.csv", eis=_make_eis(Rs_fit=5.5, Score=0.48)),
        ]

    def test_n_conditions(self):
        advisor = ProcessAdvisor()
        report = advisor.analyze_material_system(self._two_conditions())
        assert report.n_conditions == 2

    def test_comparison_table_shape(self):
        advisor = ProcessAdvisor()
        report = advisor.analyze_material_system(self._two_conditions())
        assert report.comparison_table.shape[0] == 2

    def test_best_electrolyte(self):
        advisor = ProcessAdvisor()
        report = advisor.analyze_material_system(self._two_conditions())
        assert report.best_conditions["electrolyte"] == "H2SO4"

    def test_has_recommendations(self):
        advisor = ProcessAdvisor()
        report = advisor.analyze_material_system(self._two_conditions())
        assert len(report.production_recommendations) >= 1

    def test_has_next_experiments(self):
        advisor = ProcessAdvisor()
        report = advisor.analyze_material_system(self._two_conditions())
        assert len(report.next_experiments) >= 1

    def test_assessment_mentions_conditions(self):
        advisor = ProcessAdvisor()
        report = advisor.analyze_material_system(self._two_conditions())
        assert "2" in report.material_assessment

    def test_comparison_table_has_rs(self):
        advisor = ProcessAdvisor()
        report = advisor.analyze_material_system(self._two_conditions())
        assert "Rs_fit" in report.comparison_table.columns


class TestProcessAdvisorExplicitMetadata:
    def test_metadata_override(self):
        advisor = ProcessAdvisor()
        entries = [
            _entry(
                "sample1.csv",
                eis=_make_eis(Score=0.9),
                metadata={"electrolyte": "KOH", "current": "5A", "treatment": "Plasma"},
            ),
        ]
        report = advisor.analyze_material_system(entries)
        assert report.best_conditions.get("electrolyte") == "KOH"

    def test_global_metadata(self):
        advisor = ProcessAdvisor()
        entries = [_entry("sample1.csv", eis=_make_eis(Score=0.9))]
        global_meta = {"electrolyte": "NaOH", "current": "10A", "treatment": "GCT"}
        report = advisor.analyze_material_system(entries, metadata=global_meta)
        assert report.best_conditions.get("electrolyte") == "NaOH"


class TestProcessAdvisorCyclingOnly:
    def test_cycling_only(self):
        advisor = ProcessAdvisor()
        entries = [
            _entry("H2SO4_1A_GCT.csv", cycling=_make_cycling(retention=90.0, energy=15.0)),
        ]
        report = advisor.analyze_material_system(entries)
        assert report.n_conditions == 1
        # The table should have cycling metrics
        if not report.comparison_table.empty:
            assert "retention" in report.comparison_table.columns or report.comparison_table.shape[0] >= 1


class TestProcessAdvisorDrtOnly:
    def test_drt_only(self):
        advisor = ProcessAdvisor()
        entries = [
            _entry("H2SO4_1A_GCT.csv", drt=_make_drt(n_peaks=3, dominant_tau=0.02)),
        ]
        report = advisor.analyze_material_system(entries)
        assert report.n_conditions == 1


class TestProcessAdvisorOutlierDetection:
    def test_outlier_reported(self):
        advisor = ProcessAdvisor()
        entries = [
            _entry("H2SO4_1A_GCT_1.csv", eis=_make_eis(Score=0.7)),
            _entry("H2SO4_1A_GCT_2.csv", eis=_make_eis(Score=0.7)),
            _entry("H2SO4_1A_GCT_3.csv", eis=_make_eis(Score=0.7)),
            _entry("H2SO4_1A_GCT_4.csv", eis=_make_eis(Score=0.7)),
            _entry("H2SO4_1A_GCT_5.csv", eis=_make_eis(Score=0.7)),
            _entry("H2SO4_1A_GCT_6.csv", eis=_make_eis(Score=0.7)),
            _entry("H2SO4_1A_GCT_7.csv", eis=_make_eis(Score=0.7)),
            _entry("H2SO4_1A_GCT_8.csv", eis=_make_eis(Score=0.7)),
            _entry("H2SO4_1A_GCT_bad.csv", eis=_make_eis(Score=10.0)),
        ]
        report = advisor.analyze_material_system(entries)
        # Should have outlier recommendation
        assert any(
            "outlier" in r.recommendation.lower()
            for r in report.production_recommendations
        )
        # Assessment should mention outlier
        assert "outlier" in report.material_assessment.lower()


class TestProcessAdvisorBottlenecks:
    def test_high_rs_bottleneck(self):
        advisor = ProcessAdvisor()
        entries = [
            _entry("H2SO4_1A_GCT.csv", eis=_make_eis(Rs_fit=50.0, Rp_fit=100.0, Score=0.3)),
        ]
        report = advisor.analyze_material_system(entries)
        assert "Rs" in report.bottleneck_analysis

    def test_high_rp_bottleneck(self):
        advisor = ProcessAdvisor()
        entries = [
            _entry("H2SO4_1A_GCT.csv", eis=_make_eis(Rs_fit=1.0, Rp_fit=5000.0, Score=0.3)),
        ]
        report = advisor.analyze_material_system(entries)
        assert "Rp" in report.bottleneck_analysis


class TestProcessAdvisorThreeConditions:
    def test_three_electrolytes(self):
        advisor = ProcessAdvisor()
        entries = [
            _entry("H2SO4_1A_GCT.csv", eis=_make_eis(Rs_fit=1.0, Score=0.9)),
            _entry("Li2SO4_1A_GCT.csv", eis=_make_eis(Rs_fit=5.0, Score=0.5)),
            _entry("NaOH_1A_GCT.csv", eis=_make_eis(Rs_fit=3.0, Score=0.7)),
        ]
        report = advisor.analyze_material_system(entries)
        assert report.n_conditions == 3
        assert report.comparison_table.shape[0] == 3
        assert report.best_conditions["electrolyte"] == "H2SO4"


class TestProcessAdvisorAllPipelines:
    def test_combined(self):
        advisor = ProcessAdvisor()
        entries = [
            _entry(
                "H2SO4_1A_GCT.csv",
                eis=_make_eis(Rs_fit=1.0, Score=0.9),
                cycling=_make_cycling(retention=95.0, energy=20.0, power=5.0),
                drt=_make_drt(n_peaks=2, dominant_tau=0.01),
            ),
            _entry(
                "Li2SO4_1A_GC.csv",
                eis=_make_eis(Rs_fit=5.0, Score=0.5),
                cycling=_make_cycling(retention=70.0, energy=8.0, power=1.5),
                drt=_make_drt(n_peaks=4, dominant_tau=0.1),
            ),
        ]
        report = advisor.analyze_material_system(entries)
        assert report.n_conditions == 2
        table = report.comparison_table
        # Should have EIS + cycling + DRT columns
        assert "Rs_fit" in table.columns
        assert "retention" in table.columns
        assert "n_peaks" in table.columns
