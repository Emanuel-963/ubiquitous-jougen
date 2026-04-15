"""Tests for src/feature_store.py — FeatureStore & FittingHistory.

Coverage targets
~~~~~~~~~~~~~~~~
* FeatureStore CRUD: add, add_records, query, unique_*, clear, reload
* JSON round-trip persistence (write → reload → verify)
* Missing required keys → ValueError
* Auto-timestamp when absent
* FittingHistory.similar_samples with normalised Euclidean distance
* FittingHistory.circuit_stats aggregates
* FittingHistory.best_circuit_for_features majority voting
* FittingHistory.summary_text formatting
* record_from_shortlist_result extraction helper
* Edge cases: empty store, NaN features, corrupted JSON
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from src.feature_store import (
    FeatureStore,
    FittingHistory,
    _SPECTRAL_KEYS,
    _safe_float,
    _safe_params,
    record_from_shortlist_result,
)


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store_path(tmp_path: Path) -> Path:
    """Return a fresh JSON path inside pytest's tmp_path."""
    return tmp_path / "ml" / "test_store.json"


@pytest.fixture
def store(tmp_store_path: Path) -> FeatureStore:
    """Return an empty FeatureStore writing to a temp file."""
    return FeatureStore(tmp_store_path)


@pytest.fixture
def sample_features() -> dict:
    """A realistic 9-element spectral feature dict."""
    return {
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


def _make_record(
    sample_id: str = "sample_001.txt",
    circuit_name: str = "Randles-CPE-W",
    bic: float = -120.0,
    confidence: float = 0.80,
    features: dict | None = None,
) -> dict:
    """Helper to create a valid record."""
    return {
        "sample_id": sample_id,
        "circuit_name": circuit_name,
        "spectral_features": features or {
            "logf_slope_low": -0.35,
            "logf_slope_high": -0.90,
            "phase_min": -78.0,
            "phase_max": -5.0,
            "phase_range": 73.0,
            "freq_at_phase_min": 1.0,
            "mag_range": 50.0,
            "zreal_min": 1.0,
            "zreal_max": 51.0,
        },
        "params": {"Rs": 1.2, "Rp": 50.0},
        "bic": bic,
        "confidence": confidence,
        "user_label": None,
    }


# ═══════════════════════════════════════════════════════════════════════
# FeatureStore — CRUD
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureStoreCRUD:
    """Basic add / query / len / bool operations."""

    def test_empty_store(self, store: FeatureStore):
        assert len(store) == 0
        assert not store
        assert store.records == []

    def test_add_record(self, store: FeatureStore):
        rec = _make_record()
        store.add_record(rec)
        assert len(store) == 1
        assert store

    def test_add_records_bulk(self, store: FeatureStore):
        recs = [_make_record(f"s{i}.txt") for i in range(5)]
        store.add_records(recs)
        assert len(store) == 5

    def test_auto_timestamp(self, store: FeatureStore):
        rec = {"sample_id": "x.txt", "circuit_name": "A"}
        store.add_record(rec)
        saved = store.records[0]
        assert "timestamp" in saved
        # ISO format: 2024-01-01T12:00:00
        assert "T" in saved["timestamp"]

    def test_missing_required_key_raises(self, store: FeatureStore):
        with pytest.raises(ValueError, match="sample_id"):
            store.add_record({"circuit_name": "A"})
        with pytest.raises(ValueError, match="circuit_name"):
            store.add_record({"sample_id": "x.txt"})

    def test_missing_required_key_bulk_raises(self, store: FeatureStore):
        with pytest.raises(ValueError, match="sample_id"):
            store.add_records([{"circuit_name": "A"}])

    def test_query_by_circuit_name(self, store: FeatureStore):
        store.add_records([
            _make_record("a.txt", "Randles-CPE-W"),
            _make_record("b.txt", "Two-Arc-CPE"),
            _make_record("c.txt", "Randles-CPE-W"),
        ])
        randles = store.query(circuit_name="Randles-CPE-W")
        assert len(randles) == 2

    def test_query_by_sample_id(self, store: FeatureStore):
        store.add_records([
            _make_record("a.txt"),
            _make_record("b.txt"),
        ])
        result = store.query(sample_id="a.txt")
        assert len(result) == 1
        assert result[0]["sample_id"] == "a.txt"

    def test_query_combined(self, store: FeatureStore):
        store.add_records([
            _make_record("a.txt", "Randles-CPE-W"),
            _make_record("a.txt", "Two-Arc-CPE"),
        ])
        result = store.query(sample_id="a.txt", circuit_name="Two-Arc-CPE")
        assert len(result) == 1

    def test_unique_circuits(self, store: FeatureStore):
        store.add_records([
            _make_record("a.txt", "B"),
            _make_record("b.txt", "A"),
            _make_record("c.txt", "B"),
        ])
        assert store.unique_circuits() == ["A", "B"]

    def test_unique_samples(self, store: FeatureStore):
        store.add_records([
            _make_record("b.txt"),
            _make_record("a.txt"),
        ])
        assert store.unique_samples() == ["a.txt", "b.txt"]

    def test_clear(self, store: FeatureStore, tmp_store_path: Path):
        store.add_record(_make_record())
        assert tmp_store_path.exists()
        store.clear()
        assert len(store) == 0
        assert not tmp_store_path.exists()


# ═══════════════════════════════════════════════════════════════════════
# FeatureStore — Persistence
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureStorePersistence:
    """JSON serialisation round-trips."""

    def test_save_creates_file(self, store: FeatureStore, tmp_store_path: Path):
        store.add_record(_make_record())
        assert tmp_store_path.exists()

    def test_round_trip(self, tmp_store_path: Path):
        # Write
        s1 = FeatureStore(tmp_store_path)
        s1.add_records([_make_record("a.txt"), _make_record("b.txt")])

        # Read
        s2 = FeatureStore(tmp_store_path)
        assert len(s2) == 2
        assert s2.records[0]["sample_id"] == "a.txt"
        assert s2.records[1]["sample_id"] == "b.txt"

    def test_reload(self, store: FeatureStore, tmp_store_path: Path):
        store.add_record(_make_record())
        # Externally append via raw JSON
        data = json.loads(tmp_store_path.read_text("utf-8"))
        data.append({"sample_id": "ext.txt", "circuit_name": "X"})
        tmp_store_path.write_text(json.dumps(data), encoding="utf-8")

        store.reload()
        assert len(store) == 2

    def test_corrupted_json_resets_gracefully(self, tmp_store_path: Path):
        tmp_store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_store_path.write_text("NOT VALID JSON", encoding="utf-8")
        store = FeatureStore(tmp_store_path)
        assert len(store) == 0

    def test_non_list_json_resets(self, tmp_store_path: Path):
        tmp_store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_store_path.write_text('{"key": "value"}', encoding="utf-8")
        store = FeatureStore(tmp_store_path)
        assert len(store) == 0

    def test_numpy_types_serialize(self, store: FeatureStore):
        """Ensure numpy scalars don't crash json.dump."""
        rec = {
            "sample_id": "np.txt",
            "circuit_name": "C",
            "bic": np.float64(-100.5),
            "confidence": np.float32(0.9),
        }
        store.add_record(rec)
        # Re-load from disk — should parse cleanly
        s2 = FeatureStore(store.path)
        assert len(s2) == 1

    def test_mkdir_parents(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "store.json"
        store = FeatureStore(deep_path)
        store.add_record(_make_record())
        assert deep_path.exists()


# ═══════════════════════════════════════════════════════════════════════
# FittingHistory — similar_samples
# ═══════════════════════════════════════════════════════════════════════

class TestFittingHistorySimilarSamples:
    """Normalised Euclidean nearest-neighbour lookups."""

    def test_empty_store_returns_empty(self, store: FeatureStore, sample_features):
        history = FittingHistory(store)
        assert history.similar_samples(sample_features) == []

    def test_basic_nearest(self, store: FeatureStore, sample_features):
        # Identical features → distance ≈ 0
        store.add_record(_make_record("same.txt", features=dict(sample_features)))
        # Quite different features
        diff = {k: v * 10 for k, v in sample_features.items()}
        store.add_record(_make_record("diff.txt", features=diff))

        history = FittingHistory(store)
        result = history.similar_samples(sample_features, n=2)
        assert len(result) == 2
        # The identical record should come first (smallest distance)
        assert result[0]["sample_id"] == "same.txt"
        assert result[0]["_distance"] < result[1]["_distance"]

    def test_n_limits_output(self, store: FeatureStore, sample_features):
        for i in range(10):
            feat = {k: v + i * 0.01 for k, v in sample_features.items()}
            store.add_record(_make_record(f"s{i}.txt", features=feat))

        history = FittingHistory(store)
        result = history.similar_samples(sample_features, n=3)
        assert len(result) == 3

    def test_records_without_features_are_skipped(self, store: FeatureStore, sample_features):
        store.add_record({"sample_id": "no_feat.txt", "circuit_name": "X"})
        store.add_record(_make_record("with_feat.txt", features=dict(sample_features)))

        history = FittingHistory(store)
        result = history.similar_samples(sample_features, n=5)
        assert len(result) == 1
        assert result[0]["sample_id"] == "with_feat.txt"

    def test_nan_query_returns_empty(self, store: FeatureStore, sample_features):
        store.add_record(_make_record("a.txt", features=dict(sample_features)))
        bad_query = {k: float("nan") for k in _SPECTRAL_KEYS}

        history = FittingHistory(store)
        assert history.similar_samples(bad_query) == []

    def test_single_record(self, store: FeatureStore, sample_features):
        store.add_record(_make_record("only.txt", features=dict(sample_features)))
        history = FittingHistory(store)
        result = history.similar_samples(sample_features, n=5)
        assert len(result) == 1
        # Distance to itself should be 0
        assert result[0]["_distance"] == pytest.approx(0.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# FittingHistory — circuit_stats
# ═══════════════════════════════════════════════════════════════════════

class TestFittingHistoryCircuitStats:
    """Aggregate statistics per circuit name."""

    def test_empty(self, store: FeatureStore):
        history = FittingHistory(store)
        assert history.circuit_stats() == {}

    def test_basic_stats(self, store: FeatureStore):
        store.add_records([
            _make_record("a.txt", "Randles-CPE-W", bic=-100, confidence=0.8),
            _make_record("b.txt", "Randles-CPE-W", bic=-120, confidence=0.9),
            _make_record("c.txt", "Two-Arc-CPE", bic=-90, confidence=0.7),
        ])
        history = FittingHistory(store)
        stats = history.circuit_stats()

        assert set(stats.keys()) == {"Randles-CPE-W", "Two-Arc-CPE"}

        r = stats["Randles-CPE-W"]
        assert r["count"] == 2
        assert r["mean_bic"] == pytest.approx(-110.0)
        assert r["mean_confidence"] == pytest.approx(0.85)
        assert r["pct"] == pytest.approx(200 / 3)

        t = stats["Two-Arc-CPE"]
        assert t["count"] == 1
        assert t["pct"] == pytest.approx(100 / 3)

    def test_nan_bic_excluded_from_mean(self, store: FeatureStore):
        store.add_records([
            _make_record("a.txt", "X", bic=-100, confidence=0.8),
            _make_record("b.txt", "X", bic=float("nan"), confidence=0.9),
        ])
        history = FittingHistory(store)
        stats = history.circuit_stats()
        # Only the valid BIC should count
        assert stats["X"]["mean_bic"] == pytest.approx(-100.0)

    def test_none_bic_confidence(self, store: FeatureStore):
        rec = {"sample_id": "x.txt", "circuit_name": "Z"}
        store.add_record(rec)
        history = FittingHistory(store)
        stats = history.circuit_stats()
        assert stats["Z"]["mean_bic"] is None
        assert stats["Z"]["mean_confidence"] is None


# ═══════════════════════════════════════════════════════════════════════
# FittingHistory — best_circuit_for_features
# ═══════════════════════════════════════════════════════════════════════

class TestFittingHistoryBestCircuit:
    """Majority voting from similar samples."""

    def test_empty(self, store: FeatureStore, sample_features):
        history = FittingHistory(store)
        assert history.best_circuit_for_features(sample_features) is None

    def test_majority_wins(self, store: FeatureStore, sample_features):
        # 3 Randles, 1 Two-Arc — all very similar features
        for i in range(3):
            f = {k: v + i * 0.001 for k, v in sample_features.items()}
            store.add_record(_make_record(f"r{i}.txt", "Randles-CPE-W", features=f))
        f = {k: v + 0.005 for k, v in sample_features.items()}
        store.add_record(_make_record("t0.txt", "Two-Arc-CPE", features=f))

        history = FittingHistory(store)
        best = history.best_circuit_for_features(sample_features, n=4)
        assert best == "Randles-CPE-W"


# ═══════════════════════════════════════════════════════════════════════
# FittingHistory — summary_text
# ═══════════════════════════════════════════════════════════════════════

class TestFittingHistorySummaryText:
    """Human-readable summary string."""

    def test_empty_store(self, store: FeatureStore, sample_features):
        history = FittingHistory(store)
        text = history.summary_text(sample_features)
        assert "Sem histórico" in text

    def test_with_data(self, store: FeatureStore, sample_features):
        for i in range(5):
            f = {k: v + i * 0.001 for k, v in sample_features.items()}
            store.add_record(_make_record(f"s{i}.txt", "Randles-CPE-W", features=f))

        history = FittingHistory(store)
        text = history.summary_text(sample_features)
        assert "5 amostras" in text
        assert "Randles-CPE-W" in text


# ═══════════════════════════════════════════════════════════════════════
# record_from_shortlist_result
# ═══════════════════════════════════════════════════════════════════════

class TestRecordFromShortlistResult:
    """Build a FeatureStore record from run_shortlist_fit() output."""

    def test_successful_fit(self, sample_features):
        circ_res = {
            "features": sample_features,
            "shortlist": ["Randles-CPE-W", "Two-Arc-CPE"],
            "best": {
                "template": "Randles-CPE-W",
                "success": True,
                "bic": -120.5,
                "confidence": 0.82,
                "params": {"Rs": 1.2, "Rp": 50.0},
            },
        }
        rec = record_from_shortlist_result("file_01.txt", circ_res)
        assert rec is not None
        assert rec["sample_id"] == "file_01.txt"
        assert rec["circuit_name"] == "Randles-CPE-W"
        assert rec["bic"] == pytest.approx(-120.5)
        assert rec["confidence"] == pytest.approx(0.82)
        assert "timestamp" in rec
        # Spectral features should be present
        for k in _SPECTRAL_KEYS:
            assert k in rec["spectral_features"]

    def test_failed_fit_returns_none(self):
        circ_res = {"features": {}, "best": {"success": False}}
        assert record_from_shortlist_result("x.txt", circ_res) is None

    def test_no_best_returns_none(self):
        circ_res = {"features": {}, "best": None}
        assert record_from_shortlist_result("x.txt", circ_res) is None

    def test_missing_best_key_returns_none(self):
        circ_res = {"features": {}}
        assert record_from_shortlist_result("x.txt", circ_res) is None

    def test_nan_in_params(self, sample_features):
        circ_res = {
            "features": sample_features,
            "best": {
                "template": "T",
                "success": True,
                "bic": float("nan"),
                "confidence": float("inf"),
                "params": {"Rs": float("nan"), "Rp": 10.0},
            },
        }
        rec = record_from_shortlist_result("nan.txt", circ_res)
        assert rec is not None
        assert rec["bic"] is None  # NaN → None
        assert rec["confidence"] is None  # Inf → None
        assert rec["params"]["Rs"] is None
        assert rec["params"]["Rp"] == pytest.approx(10.0)


# ═══════════════════════════════════════════════════════════════════════
# _safe_float / _safe_params helpers
# ═══════════════════════════════════════════════════════════════════════

class TestSafeHelpers:
    """JSON-safety utilities."""

    def test_safe_float_normal(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) is None

    def test_safe_float_inf(self):
        assert _safe_float(float("inf")) is None

    def test_safe_float_string(self):
        assert _safe_float("not_a_number") is None

    def test_safe_float_numpy(self):
        assert _safe_float(np.float64(2.5)) == pytest.approx(2.5)

    def test_safe_params_dict(self):
        result = _safe_params({"a": 1.0, "b": np.nan, "c": None})
        assert result == {"a": 1.0, "b": None, "c": None}

    def test_safe_params_not_dict(self):
        assert _safe_params(None) == {}
        assert _safe_params("string") == {}


# ═══════════════════════════════════════════════════════════════════════
# Integration — FeatureStore + FittingHistory end-to-end
# ═══════════════════════════════════════════════════════════════════════

class TestIntegrationEndToEnd:
    """Full workflow: add records, query history, persist, reload."""

    def test_full_workflow(self, tmp_store_path: Path, sample_features):
        # 1) Create store and populate
        store = FeatureStore(tmp_store_path)
        for i in range(20):
            circuit = "Randles-CPE-W" if i < 14 else "Two-Arc-CPE"
            feat = {k: v + i * 0.01 for k, v in sample_features.items()}
            store.add_record(_make_record(
                f"sample_{i:03d}.txt",
                circuit,
                bic=-100 - i,
                confidence=0.7 + i * 0.01,
                features=feat,
            ))
        assert len(store) == 20

        # 2) History queries
        history = FittingHistory(store)
        best = history.best_circuit_for_features(sample_features, n=10)
        assert best == "Randles-CPE-W"

        stats = history.circuit_stats()
        assert stats["Randles-CPE-W"]["count"] == 14
        assert stats["Two-Arc-CPE"]["count"] == 6

        similar = history.similar_samples(sample_features, n=3)
        assert len(similar) == 3
        assert all("_distance" in s for s in similar)

        # 3) Persist and reload
        store2 = FeatureStore(tmp_store_path)
        assert len(store2) == 20

        history2 = FittingHistory(store2)
        best2 = history2.best_circuit_for_features(sample_features, n=10)
        assert best2 == best

    def test_config_has_feature_store_path(self):
        """PipelineConfig must expose the store path."""
        from src.config import PipelineConfig
        cfg = PipelineConfig.default()
        assert hasattr(cfg, "feature_store_path")
        assert "fitting_history.json" in cfg.feature_store_path

    def test_init_re_exports(self):
        """FeatureStore and FittingHistory should be importable from src."""
        from src import FeatureStore as FS, FittingHistory as FH
        assert FS is not None
        assert FH is not None
