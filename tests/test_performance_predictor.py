"""Tests for the performance predictor (Day 17).

Covers:
* Dataclasses (CyclingPrediction, DegradationPrediction, Improvement)
* Enums (DegradationMechanism, ImprovementArea)
* Helper functions (_extract_eis_vector, _safe_pct_change, _build_training_data)
* Heuristic cycling prediction
* Degradation classifier with different scenarios
* Improvement recommender
* ML predictor with synthetic training data
* PerformancePredictor full API (predict_cycling_from_eis, predict_degradation,
  recommend_improvements, *_from_result variants)
* Edge cases: empty data, missing keys, NaN values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from src.ai.performance_predictor import (
    CyclingPrediction,
    DegradationMechanism,
    DegradationPrediction,
    Improvement,
    ImprovementArea,
    PerformancePredictor,
    _build_training_data,
    _classify_degradation,
    _extract_eis_vector,
    _extract_median_params,
    _extract_cycling_targets,
    _heuristic_cycling_prediction,
    _recommend_improvements,
    _safe_pct_change,
    _EIS_FEATURE_KEYS,
    _CYCLING_TARGET_KEYS,
    _MIN_RECORDS_FOR_ML,
)
from src.config import PipelineConfig


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture()
def good_eis_params() -> Dict[str, float]:
    """Typical healthy EIS parameters."""
    return {
        "Rs_fit": 2.5,
        "Rp_fit": 50.0,
        "Q": 1e-5,
        "n": 0.85,
        "Sigma": 12.0,
        "C_mean": 1.2,
        "Tau": 0.01,
        "Dispersion": 0.2,
        "Energy_mean": 15.0,
    }


@pytest.fixture()
def bad_eis_params() -> Dict[str, float]:
    """Problematic EIS parameters."""
    return {
        "Rs_fit": 25.0,
        "Rp_fit": 300.0,
        "Q": 1e-5,
        "n": 0.45,
        "Sigma": 150.0,
        "C_mean": 0.3,
        "Tau": 0.5,
        "Dispersion": 0.8,
        "Energy_mean": 2.0,
    }


@pytest.fixture()
def ranked_df_good() -> pd.DataFrame:
    """Healthy EIS ranked DataFrame."""
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
    }, index=["S1", "S2", "S3", "S4"])


@pytest.fixture()
def eis_result_good(ranked_df_good):
    @dataclass
    class _EISStub:
        ranked_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    return _EISStub(ranked_df=ranked_df_good)


@pytest.fixture()
def cycling_result_good():
    @dataclass
    class _CycStub:
        merged_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    return _CycStub(merged_table=pd.DataFrame({
        "Retenção (%)": [92.0, 88.0, 90.0],
        "Energia (µJ)": [20.0, 19.0, 21.0],
        "Potência (µW)": [10.0, 9.5, 10.5],
    }))


def _make_training_records(n: int = 50) -> List[Dict[str, Any]]:
    """Generate synthetic training records for ML predictor."""
    rng = np.random.RandomState(42)
    records = []
    for i in range(n):
        rs = rng.uniform(1, 20)
        rp = rng.uniform(10, 300)
        q = rng.uniform(1e-6, 1e-4)
        n_val = rng.uniform(0.5, 1.0)
        sigma = rng.uniform(5, 100)
        c_mean = rng.uniform(0.1, 5.0)
        tau = rng.uniform(0.001, 0.5)
        disp = rng.uniform(0.1, 1.0)
        energy_eis = rng.uniform(1, 30)

        # Simulate cycling targets with some correlation
        energy = c_mean * 10 + rng.normal(0, 1)
        power = 250 / max(rs, 0.1) + rng.normal(0, 2)
        retention = 70 + n_val * 20 - rp * 0.05 + rng.normal(0, 3)
        retention = max(30, min(99, retention))

        records.append({
            "eis_params": {
                "Rs_fit": rs,
                "Rp_fit": rp,
                "Q": q,
                "n": n_val,
                "Sigma": sigma,
                "C_mean": c_mean,
                "Tau": tau,
                "Dispersion": disp,
                "Energy_mean": energy_eis,
            },
            "cycling_targets": {
                "energy": energy,
                "power": power,
                "retention": retention,
            },
        })
    return records


# ═══════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════

class TestDegradationMechanism:
    def test_values(self):
        assert DegradationMechanism.FILM_GROWTH.value == "film_growth"
        assert DegradationMechanism.NONE.value == "none"

    def test_str(self):
        assert str(DegradationMechanism.MIXED) == "mixed"

    def test_is_str(self):
        assert isinstance(DegradationMechanism.FILM_GROWTH, str)


class TestImprovementArea:
    def test_values(self):
        assert ImprovementArea.OHMIC_RESISTANCE.value == "ohmic_resistance"
        assert ImprovementArea.GENERAL.value == "general"

    def test_str(self):
        assert str(ImprovementArea.DIFFUSION) == "diffusion"


# ═══════════════════════════════════════════════════════════════════════
#  CyclingPrediction
# ═══════════════════════════════════════════════════════════════════════

class TestCyclingPrediction:
    def test_defaults(self):
        p = CyclingPrediction()
        assert p.energy is None
        assert p.power is None
        assert p.retention is None
        assert p.confidence == 0.0
        assert p.method == "heuristic"
        assert p.explanation == ""

    def test_with_values(self):
        p = CyclingPrediction(energy=12.5, power=8.0, retention=85.0, confidence=0.7)
        assert p.energy == 12.5
        assert p.retention == 85.0


# ═══════════════════════════════════════════════════════════════════════
#  DegradationPrediction
# ═══════════════════════════════════════════════════════════════════════

class TestDegradationPrediction:
    def test_defaults(self):
        d = DegradationPrediction()
        assert d.mechanism == DegradationMechanism.NONE
        assert d.severity == 0.0
        assert d.delta == {}
        assert d.secondary_mechanisms == []

    def test_with_values(self):
        d = DegradationPrediction(
            mechanism=DegradationMechanism.FILM_GROWTH,
            severity=0.4,
            delta={"Rs_fit": 5.0},
        )
        assert d.mechanism == DegradationMechanism.FILM_GROWTH


# ═══════════════════════════════════════════════════════════════════════
#  Improvement
# ═══════════════════════════════════════════════════════════════════════

class TestImprovement:
    def test_defaults(self):
        imp = Improvement()
        assert imp.area == ImprovementArea.GENERAL
        assert imp.priority == 2

    def test_str(self):
        imp = Improvement(action="Polish electrode", expected_impact="Better contact", priority=1)
        assert "[P1]" in str(imp)
        assert "Polish electrode" in str(imp)


# ═══════════════════════════════════════════════════════════════════════
#  _extract_eis_vector
# ═══════════════════════════════════════════════════════════════════════

class TestExtractEISVector:
    def test_complete_params(self, good_eis_params):
        vec = _extract_eis_vector(good_eis_params)
        assert vec is not None
        assert len(vec) == len(_EIS_FEATURE_KEYS)
        assert vec[0] == 2.5  # Rs_fit

    def test_missing_key(self):
        params = {"Rs_fit": 2.5}  # incomplete
        vec = _extract_eis_vector(params)
        assert vec is None

    def test_nan_value(self, good_eis_params):
        good_eis_params["Rp_fit"] = float("nan")
        vec = _extract_eis_vector(good_eis_params)
        assert vec is None

    def test_inf_value(self, good_eis_params):
        good_eis_params["Sigma"] = float("inf")
        vec = _extract_eis_vector(good_eis_params)
        assert vec is None

    def test_non_numeric(self):
        params = {k: "abc" for k in _EIS_FEATURE_KEYS}
        vec = _extract_eis_vector(params)
        assert vec is None


# ═══════════════════════════════════════════════════════════════════════
#  _safe_pct_change
# ═══════════════════════════════════════════════════════════════════════

class TestSafePctChange:
    def test_normal(self):
        assert _safe_pct_change(100, 120) == 20.0

    def test_decrease(self):
        assert _safe_pct_change(100, 80) == -20.0

    def test_zero_before_zero_after(self):
        assert _safe_pct_change(0, 0) == 0.0

    def test_zero_before_nonzero_after(self):
        assert _safe_pct_change(0, 10) == 100.0

    def test_negative_before(self):
        result = _safe_pct_change(-100, -80)
        assert result == pytest.approx(20.0)


# ═══════════════════════════════════════════════════════════════════════
#  _build_training_data
# ═══════════════════════════════════════════════════════════════════════

class TestBuildTrainingData:
    def test_sufficient_data(self):
        records = _make_training_records(50)
        X, Y = _build_training_data(records)
        assert X is not None
        assert Y is not None
        assert X.shape[0] == 50
        assert X.shape[1] == len(_EIS_FEATURE_KEYS)
        assert "energy" in Y
        assert "power" in Y
        assert "retention" in Y

    def test_insufficient_data(self):
        records = _make_training_records(10)
        X, Y = _build_training_data(records)
        assert X is None
        assert Y is None

    def test_empty(self):
        X, Y = _build_training_data([])
        assert X is None
        assert Y is None

    def test_missing_fields(self):
        records = [{"sample_id": "x"}] * 50
        X, Y = _build_training_data(records)
        assert X is None

    def test_partial_targets(self):
        """Records with only some cycling targets should still count."""
        records = _make_training_records(50)
        # Remove power from half the records
        for r in records[:25]:
            del r["cycling_targets"]["power"]
        X, Y = _build_training_data(records)
        assert X is not None
        assert X.shape[0] == 50


# ═══════════════════════════════════════════════════════════════════════
#  _heuristic_cycling_prediction
# ═══════════════════════════════════════════════════════════════════════

class TestHeuristicPrediction:
    def test_good_params(self, good_eis_params):
        pred = _heuristic_cycling_prediction(good_eis_params)
        assert pred.method == "heuristic"
        assert pred.confidence == 0.3
        assert pred.energy is not None
        assert pred.power is not None
        assert pred.retention is not None
        assert pred.explanation != ""

    def test_minimal_params(self):
        pred = _heuristic_cycling_prediction({"Rs_fit": 5.0, "Rp_fit": 30.0, "n": 0.8})
        assert pred.power is not None
        assert pred.retention is not None
        # No C_mean → no energy from that branch
        assert pred.method == "heuristic"

    def test_empty_params(self):
        pred = _heuristic_cycling_prediction({})
        assert pred.method == "heuristic"
        assert pred.energy is None
        assert pred.power is None
        assert pred.retention is None

    def test_energy_from_eis(self):
        """When C_mean is missing but Energy_mean is present, use Energy_mean."""
        pred = _heuristic_cycling_prediction({"Energy_mean": 10.0})
        assert pred.energy == 10.0

    def test_retention_bounds(self):
        """Retention should be clamped between 30 and 99."""
        # Very good n, very low Rp → high retention
        pred = _heuristic_cycling_prediction({"Rp_fit": 0.1, "n": 1.0})
        assert 30.0 <= pred.retention <= 99.0

        # Very bad n, very high Rp → low retention
        pred = _heuristic_cycling_prediction({"Rp_fit": 500.0, "n": 0.3})
        assert 30.0 <= pred.retention <= 99.0


# ═══════════════════════════════════════════════════════════════════════
#  _classify_degradation
# ═══════════════════════════════════════════════════════════════════════

class TestClassifyDegradation:
    def test_no_degradation(self, good_eis_params):
        pred = _classify_degradation(good_eis_params, good_eis_params)
        assert pred.mechanism == DegradationMechanism.NONE
        assert pred.severity < 0.05
        assert all(v == 0 for v in pred.delta.values())

    def test_film_growth(self, good_eis_params):
        after = dict(good_eis_params)
        after["Rs_fit"] = good_eis_params["Rs_fit"] * 2.0  # 100% increase
        pred = _classify_degradation(good_eis_params, after)
        assert pred.mechanism == DegradationMechanism.FILM_GROWTH
        assert pred.delta_pct["Rs_fit"] > 0
        assert "filme" in pred.explanation.lower() or "Rs" in pred.explanation

    def test_active_material_loss(self, good_eis_params):
        after = dict(good_eis_params)
        after["C_mean"] = good_eis_params["C_mean"] * 0.5  # 50% decrease
        pred = _classify_degradation(good_eis_params, after)
        assert pred.mechanism == DegradationMechanism.ACTIVE_MATERIAL_LOSS
        assert "material" in pred.explanation.lower() or "C_mean" in pred.explanation

    def test_contact_degradation(self, good_eis_params):
        after = dict(good_eis_params)
        after["Rs_fit"] = good_eis_params["Rs_fit"] * 1.5  # 50% increase
        after["n"] = good_eis_params["n"] * 0.8  # ~6% decrease
        pred = _classify_degradation(good_eis_params, after)
        assert pred.mechanism == DegradationMechanism.CONTACT_DEGRADATION
        assert "contato" in pred.explanation.lower() or "contact" in pred.explanation.lower()

    def test_electrolyte_degradation(self, good_eis_params):
        after = dict(good_eis_params)
        after["Sigma"] = good_eis_params["Sigma"] * 3.0  # 200% increase
        pred = _classify_degradation(good_eis_params, after)
        assert pred.mechanism == DegradationMechanism.ELECTROLYTE_DEGRADATION
        assert "eletrólito" in pred.explanation.lower() or "sigma" in pred.explanation.lower()

    def test_mixed_degradation(self, good_eis_params):
        after = dict(good_eis_params)
        after["Rs_fit"] = good_eis_params["Rs_fit"] * 2.0
        after["C_mean"] = good_eis_params["C_mean"] * 0.5
        after["n"] = good_eis_params["n"] * 0.7
        pred = _classify_degradation(good_eis_params, after)
        # Should have secondary mechanisms
        assert len(pred.secondary_mechanisms) >= 1

    def test_empty_params(self):
        pred = _classify_degradation({}, {})
        assert pred.severity == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  _recommend_improvements
# ═══════════════════════════════════════════════════════════════════════

class TestRecommendImprovements:
    def test_good_params_few_recommendations(self, good_eis_params):
        imps = _recommend_improvements(good_eis_params)
        # Good params → few or moderate recommendations
        assert isinstance(imps, list)

    def test_bad_params_many_recommendations(self, bad_eis_params):
        imps = _recommend_improvements(bad_eis_params)
        assert len(imps) >= 3
        # Should include high priority items
        priorities = [imp.priority for imp in imps]
        assert 1 in priorities

    def test_high_rs(self):
        imps = _recommend_improvements({"Rs_fit": 25.0})
        areas = [imp.area for imp in imps]
        assert ImprovementArea.OHMIC_RESISTANCE in areas

    def test_high_rp(self):
        imps = _recommend_improvements({"Rp_fit": 250.0})
        areas = [imp.area for imp in imps]
        assert ImprovementArea.CHARGE_TRANSFER in areas

    def test_low_n(self):
        imps = _recommend_improvements({"n": 0.4})
        areas = [imp.area for imp in imps]
        assert ImprovementArea.SURFACE_MORPHOLOGY in areas

    def test_high_sigma(self):
        imps = _recommend_improvements({"Sigma": 120.0})
        areas = [imp.area for imp in imps]
        assert ImprovementArea.DIFFUSION in areas

    def test_low_retention(self):
        imps = _recommend_improvements({"retention": 55.0})
        areas = [imp.area for imp in imps]
        assert ImprovementArea.CYCLING_STABILITY in areas

    def test_sorted_by_priority(self, bad_eis_params):
        imps = _recommend_improvements(bad_eis_params)
        for i in range(len(imps) - 1):
            assert imps[i].priority <= imps[i + 1].priority

    def test_empty_params(self):
        imps = _recommend_improvements({})
        assert imps == []


# ═══════════════════════════════════════════════════════════════════════
#  _extract_median_params
# ═══════════════════════════════════════════════════════════════════════

class TestExtractMedianParams:
    def test_good_df(self, ranked_df_good):
        params = _extract_median_params(ranked_df_good)
        assert "Rs_fit" in params
        assert "Rp_fit" in params
        assert 2.5 <= params["Rs_fit"] <= 3.0

    def test_empty_df(self):
        params = _extract_median_params(pd.DataFrame())
        assert params == {}

    def test_with_retention(self):
        df = pd.DataFrame({
            "Rs_fit": [2.0, 3.0],
            "Retenção (%)": [90.0, 85.0],
        })
        params = _extract_median_params(df)
        assert "retention" in params
        assert params["retention"] == 87.5

    def test_nan_handling(self):
        df = pd.DataFrame({
            "Rs_fit": [np.nan, 3.0, np.nan],
            "Rp_fit": [50.0, np.nan, 45.0],
        })
        params = _extract_median_params(df)
        assert params["Rs_fit"] == 3.0
        assert 45.0 <= params["Rp_fit"] <= 50.0


# ═══════════════════════════════════════════════════════════════════════
#  _extract_cycling_targets
# ═══════════════════════════════════════════════════════════════════════

class TestExtractCyclingTargets:
    def test_good_result(self, cycling_result_good):
        targets = _extract_cycling_targets(cycling_result_good)
        assert "retention" in targets
        assert "energy" in targets
        assert "power" in targets

    def test_none_result(self):
        targets = _extract_cycling_targets(None)
        assert targets == {}

    def test_empty_table(self):
        @dataclass
        class _Stub:
            merged_table: pd.DataFrame = field(default_factory=pd.DataFrame)
        targets = _extract_cycling_targets(_Stub())
        assert targets == {}


# ═══════════════════════════════════════════════════════════════════════
#  PerformancePredictor — construction
# ═══════════════════════════════════════════════════════════════════════

class TestPredictorInit:
    def test_default_construction(self):
        pred = PerformancePredictor()
        assert not pred.is_ml_trained

    def test_custom_config(self):
        cfg = PipelineConfig.default()
        pred = PerformancePredictor(config=cfg)
        assert pred._config is cfg

    def test_with_empty_store(self):
        @dataclass
        class _StoreStub:
            records: list = field(default_factory=list)
        pred = PerformancePredictor(feature_store=_StoreStub())
        assert not pred.is_ml_trained


# ═══════════════════════════════════════════════════════════════════════
#  PerformancePredictor.predict_cycling_from_eis (heuristic)
# ═══════════════════════════════════════════════════════════════════════

class TestPredictCyclingHeuristic:
    def test_good_params(self, good_eis_params):
        pred = PerformancePredictor()
        result = pred.predict_cycling_from_eis(good_eis_params)
        assert result.method == "heuristic"
        assert result.energy is not None
        assert result.power is not None
        assert result.retention is not None

    def test_empty_params(self):
        pred = PerformancePredictor()
        result = pred.predict_cycling_from_eis({})
        assert result.method == "heuristic"
        assert result.explanation != ""


# ═══════════════════════════════════════════════════════════════════════
#  PerformancePredictor.predict_cycling_from_result
# ═══════════════════════════════════════════════════════════════════════

class TestPredictCyclingFromResult:
    def test_good_result(self, eis_result_good):
        pred = PerformancePredictor()
        result = pred.predict_cycling_from_result(eis_result_good)
        assert result.method == "heuristic"
        assert result.energy is not None

    def test_none_result(self):
        pred = PerformancePredictor()
        result = pred.predict_cycling_from_result(None)
        assert "Sem dados" in result.explanation

    def test_empty_ranked_df(self):
        @dataclass
        class _EISStub:
            ranked_df: pd.DataFrame = field(default_factory=pd.DataFrame)
        pred = PerformancePredictor()
        result = pred.predict_cycling_from_result(_EISStub())
        assert "Sem dados" in result.explanation


# ═══════════════════════════════════════════════════════════════════════
#  PerformancePredictor.predict_degradation
# ═══════════════════════════════════════════════════════════════════════

class TestPredictDegradation:
    def test_no_change(self, good_eis_params):
        pred = PerformancePredictor()
        result = pred.predict_degradation(good_eis_params, good_eis_params)
        assert result.mechanism == DegradationMechanism.NONE

    def test_film_growth(self, good_eis_params):
        after = dict(good_eis_params)
        after["Rs_fit"] *= 3
        pred = PerformancePredictor()
        result = pred.predict_degradation(good_eis_params, after)
        assert result.mechanism == DegradationMechanism.FILM_GROWTH


# ═══════════════════════════════════════════════════════════════════════
#  PerformancePredictor.predict_degradation_from_results
# ═══════════════════════════════════════════════════════════════════════

class TestPredictDegradationFromResults:
    def test_good_results(self, ranked_df_good):
        @dataclass
        class _EISStub:
            ranked_df: pd.DataFrame = field(default_factory=pd.DataFrame)

        before = _EISStub(ranked_df=ranked_df_good)
        after_df = ranked_df_good.copy()
        after_df["Rs_fit"] = after_df["Rs_fit"] * 3.0
        after = _EISStub(ranked_df=after_df)

        pred = PerformancePredictor()
        result = pred.predict_degradation_from_results(before, after)
        assert result.mechanism != DegradationMechanism.NONE

    def test_none_before(self):
        pred = PerformancePredictor()
        result = pred.predict_degradation_from_results(None, None)
        assert "insuficientes" in result.explanation.lower()


# ═══════════════════════════════════════════════════════════════════════
#  PerformancePredictor.recommend_improvements
# ═══════════════════════════════════════════════════════════════════════

class TestRecommendImprovementsPredictor:
    def test_bad_params(self, bad_eis_params):
        pred = PerformancePredictor()
        imps = pred.recommend_improvements(bad_eis_params)
        assert len(imps) >= 3

    def test_empty_params(self):
        pred = PerformancePredictor()
        imps = pred.recommend_improvements({})
        assert imps == []


# ═══════════════════════════════════════════════════════════════════════
#  PerformancePredictor.recommend_improvements_from_result
# ═══════════════════════════════════════════════════════════════════════

class TestRecommendFromResult:
    def test_good_result(self, eis_result_good):
        pred = PerformancePredictor()
        imps = pred.recommend_improvements_from_result(eis_result_good)
        assert isinstance(imps, list)

    def test_none_result(self):
        pred = PerformancePredictor()
        imps = pred.recommend_improvements_from_result(None)
        assert imps == []

    def test_empty_result(self):
        @dataclass
        class _EISStub:
            ranked_df: pd.DataFrame = field(default_factory=pd.DataFrame)
        pred = PerformancePredictor()
        imps = pred.recommend_improvements_from_result(_EISStub())
        assert imps == []


# ═══════════════════════════════════════════════════════════════════════
#  ML predictor (train + predict)
# ═══════════════════════════════════════════════════════════════════════

class TestMLPredictor:
    def test_train_and_predict(self, good_eis_params):
        records = _make_training_records(50)
        pred = PerformancePredictor()
        pred.train(records)
        assert pred.is_ml_trained

        result = pred.predict_cycling_from_eis(good_eis_params)
        assert result.method == "ml"
        assert result.confidence > 0.3
        assert result.energy is not None
        assert result.power is not None
        assert result.retention is not None
        assert result.feature_importances != {}
        assert result.explanation != ""

    def test_train_insufficient_data(self):
        records = _make_training_records(10)
        pred = PerformancePredictor()
        pred.train(records)
        assert not pred.is_ml_trained

    def test_ml_fallback_on_missing_features(self):
        records = _make_training_records(50)
        pred = PerformancePredictor()
        pred.train(records)
        assert pred.is_ml_trained

        # Incomplete params → fallback to heuristic
        result = pred.predict_cycling_from_eis({"Rs_fit": 3.0})
        assert result.method == "heuristic"

    def test_ml_retention_clamped(self, good_eis_params):
        records = _make_training_records(50)
        pred = PerformancePredictor()
        pred.train(records)
        result = pred.predict_cycling_from_eis(good_eis_params)
        if result.retention is not None:
            assert 0.0 <= result.retention <= 100.0

    def test_feature_importances_sum(self, good_eis_params):
        records = _make_training_records(50)
        pred = PerformancePredictor()
        pred.train(records)
        imps = pred._ml._feature_importances
        total = sum(imps.values())
        assert 0.9 <= total <= 1.1  # Should sum to ~1


# ═══════════════════════════════════════════════════════════════════════
#  Auto-training from store
# ═══════════════════════════════════════════════════════════════════════

class TestAutoTrain:
    def test_auto_train_with_sufficient_store(self):
        records = _make_training_records(50)

        @dataclass
        class _StoreStub:
            records: list = field(default_factory=list)

        store = _StoreStub(records=records)
        pred = PerformancePredictor(feature_store=store)
        assert pred.is_ml_trained

    def test_no_auto_train_without_store(self):
        pred = PerformancePredictor()
        assert not pred.is_ml_trained


# ═══════════════════════════════════════════════════════════════════════
#  Re-export checks
# ═══════════════════════════════════════════════════════════════════════

class TestReExports:
    def test_ai_package(self):
        from src.ai import (
            CyclingPrediction,
            DegradationMechanism,
            DegradationPrediction,
            Improvement,
            ImprovementArea,
            PerformancePredictor,
        )
        assert PerformancePredictor is not None

    def test_src_package(self):
        from src import (
            CyclingPrediction,
            DegradationMechanism,
            DegradationPrediction,
            Improvement,
            ImprovementArea,
            PerformancePredictor,
        )
        assert PerformancePredictor is not None


# ═══════════════════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_predict_with_nan_params(self):
        pred = PerformancePredictor()
        result = pred.predict_cycling_from_eis({"Rs_fit": float("nan")})
        assert result.method == "heuristic"

    def test_degradation_single_key(self):
        pred = PerformancePredictor()
        result = pred.predict_degradation({"Rs_fit": 2.0}, {"Rs_fit": 20.0})
        assert result.mechanism != DegradationMechanism.NONE or result.severity > 0

    def test_all_methods_return_types(self, good_eis_params, eis_result_good):
        pred = PerformancePredictor()
        assert isinstance(pred.predict_cycling_from_eis(good_eis_params), CyclingPrediction)
        assert isinstance(pred.predict_cycling_from_result(eis_result_good), CyclingPrediction)
        assert isinstance(
            pred.predict_degradation(good_eis_params, good_eis_params),
            DegradationPrediction,
        )
        assert isinstance(pred.recommend_improvements(good_eis_params), list)
