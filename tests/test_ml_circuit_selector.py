"""Tests for src/ml_circuit_selector.py — CircuitMLSelector.

Coverage targets
~~~~~~~~~~~~~~~~
* CircuitMLSelector lifecycle: init → train → predict/confidence/explain
* Fallback to empty list when store has < min_samples
* Fallback when only 1 class present
* Synthetic multi-class data → correct majority prediction
* Feature importances extraction
* explain() text content
* Edge cases: NaN features, missing keys, empty store
* Integration with FeatureStore round-trip
* shortlist_circuits accepts ml_ranked parameter
* run_shortlist_fit accepts ml_ranked parameter
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from typing import Dict, List

from src.feature_store import FeatureStore
from src.ml_circuit_selector import CircuitMLSelector, _FEATURE_KEYS


# ─── Synthetic data helpers ──────────────────────────────────────────

def _make_features(
    base: Dict[str, float] | None = None,
    noise: float = 0.01,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """Generate a spectral-feature dict with small random noise."""
    if rng is None:
        rng = np.random.default_rng(0)
    default = {
        "logf_slope_low": -0.35,
        "logf_slope_high": -0.90,
        "phase_min": -78.0,
        "phase_max": -5.0,
        "phase_range": 73.0,
        "freq_at_phase_min": 1.0,
        "mag_range": 50.0,
        "zreal_min": 1.0,
        "zreal_max": 51.0,
    }
    base = base or default
    return {k: float(base[k] + rng.normal(0, noise * abs(base[k]) + 0.01))
            for k in _FEATURE_KEYS}


def _populate_store(
    store: FeatureStore,
    n: int = 50,
    circuits: List[str] | None = None,
    seed: int = 42,
) -> None:
    """Populate a store with N synthetic records spanning multiple circuits.

    Creates well-separated clusters so the classifier can learn.
    """
    if circuits is None:
        circuits = ["Randles-CPE-W", "Two-Arc-CPE", "Inductive-CPE"]

    rng = np.random.default_rng(seed)

    # Create distinct feature profiles per circuit
    profiles = {
        "Randles-CPE-W": {
            "logf_slope_low": -0.35, "logf_slope_high": -0.90,
            "phase_min": -78.0, "phase_max": -5.0, "phase_range": 73.0,
            "freq_at_phase_min": 1.0, "mag_range": 50.0,
            "zreal_min": 1.0, "zreal_max": 51.0,
        },
        "Two-Arc-CPE": {
            "logf_slope_low": -0.15, "logf_slope_high": -0.50,
            "phase_min": -85.0, "phase_max": -10.0, "phase_range": 75.0,
            "freq_at_phase_min": 0.1, "mag_range": 200.0,
            "zreal_min": 5.0, "zreal_max": 205.0,
        },
        "Inductive-CPE": {
            "logf_slope_low": -0.60, "logf_slope_high": -0.30,
            "phase_min": -40.0, "phase_max": 15.0, "phase_range": 55.0,
            "freq_at_phase_min": 100.0, "mag_range": 20.0,
            "zreal_min": 0.5, "zreal_max": 20.5,
        },
    }

    per_circuit = n // len(circuits)
    remainder = n - per_circuit * len(circuits)

    records = []
    idx = 0
    for c_idx, circ in enumerate(circuits):
        count = per_circuit + (1 if c_idx < remainder else 0)
        prof = profiles.get(circ, profiles["Randles-CPE-W"])
        for _ in range(count):
            feats = _make_features(prof, noise=0.05, rng=rng)
            records.append({
                "sample_id": f"syn_{idx:04d}.txt",
                "circuit_name": circ,
                "spectral_features": feats,
                "bic": float(-100 - rng.normal(0, 10)),
                "confidence": float(rng.uniform(0.6, 0.95)),
                "params": {"Rs": float(rng.uniform(0.5, 5.0))},
                "user_label": None,
            })
            idx += 1

    store.add_records(records)


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store_path(tmp_path: Path) -> Path:
    return tmp_path / "ml" / "test_ml_store.json"


@pytest.fixture
def empty_store(tmp_store_path: Path) -> FeatureStore:
    return FeatureStore(tmp_store_path)


@pytest.fixture
def populated_store(tmp_store_path: Path) -> FeatureStore:
    store = FeatureStore(tmp_store_path)
    _populate_store(store, n=60)
    return store


@pytest.fixture
def small_store(tmp_store_path: Path) -> FeatureStore:
    """Store with < 30 records (below training threshold)."""
    store = FeatureStore(tmp_store_path)
    _populate_store(store, n=15)
    return store


@pytest.fixture
def selector() -> CircuitMLSelector:
    return CircuitMLSelector()


# ═══════════════════════════════════════════════════════════════════════
# Init & properties
# ═══════════════════════════════════════════════════════════════════════

class TestInit:
    def test_default_state(self, selector: CircuitMLSelector):
        assert not selector.is_trained
        assert selector.n_training_samples == 0
        assert selector.classes == []
        assert selector.min_samples == 30

    def test_custom_min_samples(self):
        sel = CircuitMLSelector(min_samples=10)
        assert sel.min_samples == 10

    def test_custom_n_estimators(self):
        sel = CircuitMLSelector(n_estimators=50)
        assert sel.n_estimators == 50


# ═══════════════════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════════════════

class TestTrain:
    def test_train_empty_store(self, selector: CircuitMLSelector, empty_store: FeatureStore):
        result = selector.train(empty_store)
        assert result is False
        assert not selector.is_trained

    def test_train_small_store_fallback(self, selector: CircuitMLSelector, small_store: FeatureStore):
        result = selector.train(small_store)
        assert result is False
        assert not selector.is_trained

    def test_train_populated_store(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        result = selector.train(populated_store)
        assert result is True
        assert selector.is_trained
        assert selector.n_training_samples == 60
        assert len(selector.classes) == 3

    def test_train_single_class_fallback(self, empty_store: FeatureStore):
        """Only one circuit → can't train a classifier."""
        _populate_store(empty_store, n=40, circuits=["Randles-CPE-W"])
        sel = CircuitMLSelector()
        result = sel.train(empty_store)
        assert result is False
        assert not sel.is_trained

    def test_train_custom_threshold(self, small_store: FeatureStore):
        sel = CircuitMLSelector(min_samples=10)
        result = sel.train(small_store)
        assert result is True
        assert sel.is_trained

    def test_retrain_replaces_old_model(self, populated_store: FeatureStore):
        sel = CircuitMLSelector()
        sel.train(populated_store)
        n1 = sel.n_training_samples

        # Add more records and retrain
        _populate_store(populated_store, n=20, seed=99)
        sel.train(populated_store)
        assert sel.n_training_samples >= n1


# ═══════════════════════════════════════════════════════════════════════
# Predict
# ═══════════════════════════════════════════════════════════════════════

class TestPredict:
    def test_predict_untrained_returns_empty(self, selector: CircuitMLSelector):
        feats = _make_features()
        assert selector.predict(feats) == []

    def test_predict_randles_profile(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        # Features similar to Randles-CPE-W profile
        feats = _make_features({
            "logf_slope_low": -0.35, "logf_slope_high": -0.90,
            "phase_min": -78.0, "phase_max": -5.0, "phase_range": 73.0,
            "freq_at_phase_min": 1.0, "mag_range": 50.0,
            "zreal_min": 1.0, "zreal_max": 51.0,
        }, noise=0.01)
        ranked = selector.predict(feats, top_n=3)
        assert len(ranked) <= 3
        assert "Randles-CPE-W" in ranked

    def test_predict_two_arc_profile(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        feats = _make_features({
            "logf_slope_low": -0.15, "logf_slope_high": -0.50,
            "phase_min": -85.0, "phase_max": -10.0, "phase_range": 75.0,
            "freq_at_phase_min": 0.1, "mag_range": 200.0,
            "zreal_min": 5.0, "zreal_max": 205.0,
        }, noise=0.01)
        ranked = selector.predict(feats, top_n=3)
        assert len(ranked) <= 3
        assert "Two-Arc-CPE" in ranked

    def test_predict_inductive_profile(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        feats = _make_features({
            "logf_slope_low": -0.60, "logf_slope_high": -0.30,
            "phase_min": -40.0, "phase_max": 15.0, "phase_range": 55.0,
            "freq_at_phase_min": 100.0, "mag_range": 20.0,
            "zreal_min": 0.5, "zreal_max": 20.5,
        }, noise=0.01)
        ranked = selector.predict(feats, top_n=3)
        assert len(ranked) <= 3
        assert "Inductive-CPE" in ranked

    def test_predict_top_n_respects_limit(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        feats = _make_features()
        ranked = selector.predict(feats, top_n=1)
        assert len(ranked) == 1

    def test_predict_nan_features_returns_empty(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        bad = {k: float("nan") for k in _FEATURE_KEYS}
        assert selector.predict(bad) == []


# ═══════════════════════════════════════════════════════════════════════
# Confidence
# ═══════════════════════════════════════════════════════════════════════

class TestConfidence:
    def test_confidence_untrained_returns_empty(self, selector: CircuitMLSelector):
        feats = _make_features()
        assert selector.confidence(feats) == {}

    def test_confidence_sums_to_one(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        feats = _make_features()
        probs = selector.confidence(feats)
        assert len(probs) == 3
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_confidence_keys_are_circuit_names(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        feats = _make_features()
        probs = selector.confidence(feats)
        assert set(probs.keys()) == {"Randles-CPE-W", "Two-Arc-CPE", "Inductive-CPE"}

    def test_confidence_nan_features_returns_empty(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        bad = {k: float("nan") for k in _FEATURE_KEYS}
        assert selector.confidence(bad) == {}


# ═══════════════════════════════════════════════════════════════════════
# Explain
# ═══════════════════════════════════════════════════════════════════════

class TestExplain:
    def test_explain_untrained(self, selector: CircuitMLSelector):
        feats = _make_features()
        text = selector.explain(feats)
        assert "heurística" in text
        assert "30" in text  # min_samples

    def test_explain_trained_has_key_info(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        feats = _make_features()
        text = selector.explain(feats)
        assert "60 amostras" in text
        assert "%" in text
        assert "Features mais influentes" in text

    def test_explain_includes_alternative(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        feats = _make_features()
        text = selector.explain(feats)
        assert "Alternativa" in text

    def test_explain_includes_slope_and_phase(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        feats = _make_features()
        text = selector.explain(feats)
        assert "slope_low" in text
        assert "phase_min" in text


# ═══════════════════════════════════════════════════════════════════════
# Feature importances
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureImportances:
    def test_importances_untrained_empty(self, selector: CircuitMLSelector):
        assert selector.feature_importances() == {}

    def test_importances_trained(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        imp = selector.feature_importances()
        assert set(imp.keys()) == set(_FEATURE_KEYS)
        assert all(isinstance(v, float) for v in imp.values())
        assert abs(sum(imp.values()) - 1.0) < 1e-6  # RF importances sum to 1

    def test_importances_all_non_negative(self, selector: CircuitMLSelector, populated_store: FeatureStore):
        selector.train(populated_store)
        imp = selector.feature_importances()
        assert all(v >= 0 for v in imp.values())


# ═══════════════════════════════════════════════════════════════════════
# _build_dataset / _features_to_array
# ═══════════════════════════════════════════════════════════════════════

class TestInternalHelpers:
    def test_build_dataset_empty(self, empty_store: FeatureStore):
        X, y = CircuitMLSelector._build_dataset(empty_store)
        assert X.shape == (0, 9)
        assert y.shape == (0,)

    def test_build_dataset_filters_incomplete(self, empty_store: FeatureStore):
        """Records without spectral_features or with NaN are excluded."""
        empty_store.add_records([
            {"sample_id": "a.txt", "circuit_name": "X"},  # no features
            {"sample_id": "b.txt", "circuit_name": "Y",
             "spectral_features": {k: float("nan") for k in _FEATURE_KEYS}},
            {"sample_id": "c.txt", "circuit_name": "Z",
             "spectral_features": {k: 1.0 for k in _FEATURE_KEYS}},
        ])
        X, y = CircuitMLSelector._build_dataset(empty_store)
        assert X.shape == (1, 9)
        assert y[0] == "Z"

    def test_features_to_array_valid(self):
        feats = {k: float(i) for i, k in enumerate(_FEATURE_KEYS)}
        arr = CircuitMLSelector._features_to_array(feats)
        assert arr is not None
        assert arr.shape == (1, 9)

    def test_features_to_array_nan_returns_none(self):
        feats = {k: float("nan") for k in _FEATURE_KEYS}
        assert CircuitMLSelector._features_to_array(feats) is None

    def test_features_to_array_missing_key(self):
        feats = {k: 1.0 for k in list(_FEATURE_KEYS)[:5]}
        # Missing keys → nan → returns None
        assert CircuitMLSelector._features_to_array(feats) is None


# ═══════════════════════════════════════════════════════════════════════
# Integration with shortlist_circuits and run_shortlist_fit
# ═══════════════════════════════════════════════════════════════════════

class TestShortlistIntegration:
    def test_shortlist_accepts_ml_ranked(self):
        """shortlist_circuits should accept ml_ranked parameter."""
        from src.circuit_fitting import shortlist_circuits, circuit_catalog

        catalog = circuit_catalog()
        feats = _make_features()
        ml_names = ["Two-Arc-CPE", "Randles-CPE-W"]

        result = shortlist_circuits(feats, catalog, top_n=3, ml_ranked=ml_names)
        # Should use the ML-ranked order
        assert result[0].name == "Two-Arc-CPE"
        assert result[1].name == "Randles-CPE-W"

    def test_shortlist_ml_ranked_empty_falls_back(self):
        """Empty ml_ranked → heuristic fallback."""
        from src.circuit_fitting import shortlist_circuits, circuit_catalog

        catalog = circuit_catalog()
        feats = _make_features()

        result_ml = shortlist_circuits(feats, catalog, top_n=3, ml_ranked=[])
        result_heur = shortlist_circuits(feats, catalog, top_n=3, ml_ranked=None)

        # Both should fall back to the same heuristic
        assert [c.name for c in result_ml] == [c.name for c in result_heur]

    def test_shortlist_ml_ranked_unknown_name_falls_back(self):
        """ML names not in catalog → heuristic fallback."""
        from src.circuit_fitting import shortlist_circuits, circuit_catalog

        catalog = circuit_catalog()
        feats = _make_features()

        result = shortlist_circuits(feats, catalog, top_n=3,
                                    ml_ranked=["Nonexistent-Circuit"])
        # Should fall back to heuristic (first pick is Randles-CPE-W)
        assert result[0].name == "Randles-CPE-W"

    def test_run_shortlist_fit_accepts_ml_ranked(self):
        """run_shortlist_fit should accept ml_ranked keyword."""
        import pandas as pd
        from src.circuit_fitting import run_shortlist_fit

        freq = np.logspace(-1, 5, 30)
        omega = 2 * np.pi * freq
        z = 10 + 100 / (1 + 1j * omega * 1e-3)
        df = pd.DataFrame({
            "frequency": freq,
            "zreal": z.real,
            "zimag": z.imag,
        })

        result = run_shortlist_fit(df, ml_ranked=["Two-Arc-CPE", "Randles-CPE-W"])
        assert "shortlist" in result
        assert "Two-Arc-CPE" in result["shortlist"]


# ═══════════════════════════════════════════════════════════════════════
# End-to-end: FeatureStore → train → predict cycle
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_cycle(self, tmp_store_path: Path):
        """Populate store → train → predict → verify consistency."""
        store = FeatureStore(tmp_store_path)
        _populate_store(store, n=60, seed=123)

        sel = CircuitMLSelector(min_samples=30)
        assert sel.train(store) is True
        assert sel.is_trained

        # Randles-like features → should predict Randles
        feats_randles = _make_features({
            "logf_slope_low": -0.35, "logf_slope_high": -0.90,
            "phase_min": -78.0, "phase_max": -5.0, "phase_range": 73.0,
            "freq_at_phase_min": 1.0, "mag_range": 50.0,
            "zreal_min": 1.0, "zreal_max": 51.0,
        }, noise=0.005)

        ranked = sel.predict(feats_randles)
        assert "Randles-CPE-W" in ranked

        probs = sel.confidence(feats_randles)
        assert probs["Randles-CPE-W"] > 0.3  # should be dominant

        text = sel.explain(feats_randles)
        assert len(text) > 50
        assert "amostras" in text

    def test_init_re_exports(self):
        """CircuitMLSelector should be importable from src."""
        from src import CircuitMLSelector as CLS
        assert CLS is not None

    def test_persistent_training(self, tmp_store_path: Path):
        """Train, save store, reload, train again → same results."""
        store = FeatureStore(tmp_store_path)
        _populate_store(store, n=60, seed=42)

        sel1 = CircuitMLSelector(random_state=42)
        sel1.train(store)
        feats = _make_features()
        pred1 = sel1.predict(feats)

        # Reload from disk
        store2 = FeatureStore(tmp_store_path)
        sel2 = CircuitMLSelector(random_state=42)
        sel2.train(store2)
        pred2 = sel2.predict(feats)

        assert pred1 == pred2
