"""Tests for Phase 7 — SQLite backend (src/db/)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.db.feature_store_v2 import FeatureStoreV2
from src.db.migrations import run_migrations
from src.db.repository import IonFlowRepository
from src.db.schema import init_db

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_ionflow.db"


@pytest.fixture
def repo(db_path):
    return IonFlowRepository(db_path)


@pytest.fixture
def store(db_path):
    return FeatureStoreV2(db_path)


# ── init_db ───────────────────────────────────────────────────────────


def test_init_db_creates_file(tmp_path):
    path = tmp_path / "new.db"
    conn = init_db(path)
    conn.close()
    assert path.exists()


def test_init_db_creates_tables(tmp_path):
    path = tmp_path / "schema.db"
    conn = init_db(path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    expected = {
        "samples",
        "eis_results",
        "cycling_results",
        "drt_results",
        "parameters",
        "fitting_history",
        "_migrations",
    }
    assert expected.issubset(tables)


def test_init_db_idempotent(tmp_path):
    path = tmp_path / "idem.db"
    conn1 = init_db(path)
    conn1.close()
    # Second call must not raise
    conn2 = init_db(path)
    conn2.close()


def test_init_db_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "ionflow.db"
    conn = init_db(path)
    conn.close()
    assert path.exists()


# ── run_migrations ────────────────────────────────────────────────────


def test_run_migrations_returns_zero_on_empty_registry(tmp_path):
    conn = init_db(tmp_path / "m.db")
    applied = run_migrations(conn)
    conn.close()
    assert applied == 0


def test_run_migrations_idempotent(tmp_path):
    conn = init_db(tmp_path / "m2.db")
    run_migrations(conn)
    # Second call should also return 0
    assert run_migrations(conn) == 0
    conn.close()


# ── IonFlowRepository — samples ───────────────────────────────────────


def test_add_and_get_sample(repo):
    sid = repo.add_sample("run01", "eis")
    assert isinstance(sid, int)
    df = repo.get_all_samples()
    assert len(df) == 1
    assert df.iloc[0]["name"] == "run01"
    assert df.iloc[0]["type"] == "eis"


def test_get_all_samples_empty(repo):
    df = repo.get_all_samples()
    assert df.empty


def test_get_sample_by_id(repo):
    sid = repo.add_sample("myrun", "cycling", file_path="/tmp/a.csv")
    rec = repo.get_sample_by_id(sid)
    assert rec is not None
    assert rec["name"] == "myrun"
    assert rec["file_path"] == "/tmp/a.csv"


def test_get_sample_by_id_missing(repo):
    assert repo.get_sample_by_id(9999) is None


def test_delete_sample_removes_row(repo):
    sid = repo.add_sample("toDelete", "eis")
    repo.delete_sample(sid)
    df = repo.get_all_samples()
    assert df.empty


def test_delete_sample_cascades_eis(repo):
    sid = repo.add_sample("cascade_eis", "eis")
    eis_df = pd.DataFrame(
        [{"Rs_fit": 1.0, "Rp_fit": 50.0, "circuit_name": "R(RC)"}], index=["f1"]
    )
    repo.save_eis_results(sid, eis_df)
    repo.delete_sample(sid)
    assert repo.get_eis_results(sid).empty


# ── IonFlowRepository — EIS results ──────────────────────────────────


def test_save_and_get_eis_results(repo):
    sid = repo.add_sample("eis_run", "eis")
    eis_df = pd.DataFrame(
        [
            {
                "Rs_fit": 1.2,
                "Rp_fit": 45.0,
                "circuit_name": "Randles-CPE",
                "BIC": -110.0,
                "confidence": 0.85,
                "Score": 0.72,
                "Rank": 1,
                "category": "Interface eficiente",
            },
            {
                "Rs_fit": 2.0,
                "Rp_fit": 30.0,
                "circuit_name": "RC",
                "BIC": -90.0,
                "confidence": 0.70,
                "Score": 0.55,
                "Rank": 2,
                "category": "Genérica estável",
            },
        ],
        index=["sample_A", "sample_B"],
    )
    repo.save_eis_results(sid, eis_df)
    result = repo.get_eis_results(sid)
    assert len(result) == 2
    assert set(result["circuit_name"]) == {"Randles-CPE", "RC"}


def test_get_eis_results_all(repo):
    sid1 = repo.add_sample("r1", "eis")
    sid2 = repo.add_sample("r2", "eis")
    for sid in [sid1, sid2]:
        repo.save_eis_results(
            sid,
            pd.DataFrame([{"Rs_fit": 1.0, "circuit_name": "RC"}], index=["f1"]),
        )
    df = repo.get_eis_results()
    assert len(df) == 2


def test_get_eis_results_empty(repo):
    assert repo.get_eis_results().empty


# ── IonFlowRepository — cycling results ──────────────────────────────


def test_save_and_get_cycling_results(repo):
    sid = repo.add_sample("cyc_run", "cycling")
    cyc_df = pd.DataFrame(
        {
            "cycle": [1, 2, 3],
            "Energy_Wh_kg": [50.0, 48.5, 47.0],
            "Power_W_kg": [200.0, 200.0, 200.0],
            "retention": [100.0, 97.0, 94.0],
        }
    )
    repo.save_cycling_results(sid, cyc_df)
    result = repo.get_cycling_results(sid)
    assert len(result) == 3
    assert list(result["cycle_number"]) == [1, 2, 3]


def test_get_cycling_results_all(repo):
    for i in range(3):
        sid = repo.add_sample(f"c{i}", "cycling")
        repo.save_cycling_results(
            sid,
            pd.DataFrame([{"cycle": 1, "Energy_Wh_kg": 50.0}]),
        )
    df = repo.get_cycling_results()
    assert len(df) == 3


# ── IonFlowRepository — DRT results ──────────────────────────────────


def test_save_and_get_drt_results(repo):
    sid = repo.add_sample("drt_run", "drt")
    drt_df = pd.DataFrame(
        [
            {
                "tau_peak_1": 1e-4,
                "gamma_peak_1": 0.8,
                "tau_peak_2": 1e-2,
                "gamma_peak_2": 0.3,
            }
        ],
        index=["sample_X"],
    )
    repo.save_drt_results(sid, drt_df)
    result = repo.get_drt_results(sid)
    assert len(result) == 1
    assert abs(result.iloc[0]["tau_peak1"] - 1e-4) < 1e-10


# ── IonFlowRepository — parameters ───────────────────────────────────


def test_save_and_get_parameter(repo):
    sid = repo.add_sample("param_run", "eis")
    repo.save_parameter(sid, "Rs_mean", 1.5, "Ω")
    df = repo.get_parameters(sid)
    assert len(df) == 1
    assert df.iloc[0]["param_name"] == "Rs_mean"
    assert df.iloc[0]["param_value"] == pytest.approx(1.5)


# ── IonFlowRepository — stats ─────────────────────────────────────────


def test_stats_returns_counts(repo):
    sid = repo.add_sample("s", "eis")
    repo.save_eis_results(
        sid,
        pd.DataFrame([{"Rs_fit": 1.0}], index=["f1"]),
    )
    s = repo.stats()
    assert s["samples"] == 1
    assert s["eis_results"] == 1
    assert "cycling_results" in s
    assert "fitting_history" in s


def test_stats_empty_db(repo):
    s = repo.stats()
    assert all(v == 0 for v in s.values())


# ── FeatureStoreV2 — add + records ────────────────────────────────────


def _make_record(sample_id="s1", circuit="RC", bic=-100.0, conf=0.8, **kwargs):
    rec = {
        "sample_id": sample_id,
        "circuit_name": circuit,
        "bic": bic,
        "confidence": conf,
    }
    rec.update(kwargs)
    return rec


def test_add_record_increments_len(store):
    assert len(store) == 0
    store.add_record(_make_record())
    assert len(store) == 1


def test_records_returns_all(store):
    for i in range(5):
        store.add_record(_make_record(sample_id=f"s{i}"))
    assert len(store.records) == 5


def test_add_record_missing_required_skipped(store):
    store.add_record({"bic": -1.0})  # no sample_id / circuit_name
    assert len(store) == 0


def test_bool_empty_store(store):
    assert not store


def test_bool_nonempty_store(store):
    store.add_record(_make_record())
    assert store


# ── FeatureStoreV2 — query ────────────────────────────────────────────


def test_query_by_circuit_name(store):
    store.add_record(_make_record(circuit="RC"))
    store.add_record(_make_record(circuit="Randles"))
    result = store.query(circuit_name="RC")
    assert len(result) == 1
    assert result[0]["circuit_name"] == "RC"


def test_query_by_sample_id(store):
    store.add_record(_make_record(sample_id="alpha"))
    store.add_record(_make_record(sample_id="beta"))
    result = store.query(sample_id="alpha")
    assert len(result) == 1


def test_query_no_filter_returns_all(store):
    for i in range(3):
        store.add_record(_make_record(sample_id=f"s{i}"))
    assert len(store.query()) == 3


# ── FeatureStoreV2 — unique_* ─────────────────────────────────────────


def test_unique_circuits(store):
    for name in ["RC", "RC", "Randles"]:
        store.add_record(_make_record(circuit=name))
    assert sorted(store.unique_circuits()) == ["RC", "Randles"]


def test_unique_samples(store):
    for s in ["a", "a", "b"]:
        store.add_record(_make_record(sample_id=s))
    assert sorted(store.unique_samples()) == ["a", "b"]


# ── FeatureStoreV2 — circuit_stats ────────────────────────────────────


def test_circuit_stats_counts(store):
    for _ in range(3):
        store.add_record(_make_record(circuit="RC"))
    store.add_record(_make_record(circuit="Randles"))
    stats = store.circuit_stats()
    assert stats["RC"]["count"] == 3
    assert stats["Randles"]["count"] == 1
    assert stats["RC"]["pct"] == pytest.approx(75.0)


def test_circuit_stats_empty_store(store):
    assert store.circuit_stats() == {}


# ── FeatureStoreV2 — similar_samples ─────────────────────────────────


def test_similar_samples_returns_neighbours(store):
    for i in range(5):
        store.add_record(
            _make_record(
                sample_id=f"s{i}",
                logf_slope_low=float(i),
                logf_slope_high=float(i) * 0.5,
            )
        )
    results = store.similar_samples(
        {"logf_slope_low": 0.1, "logf_slope_high": 0.05}, n=2
    )
    assert len(results) <= 2


def test_similar_samples_empty_store(store):
    assert store.similar_samples({"logf_slope_low": 1.0}) == []


def test_similar_samples_no_overlap(store):
    store.add_record(_make_record(sample_id="x", phase_min=-30.0))
    # query has no common keys with stored record
    results = store.similar_samples({"logf_slope_low": 0.5})
    assert results == []


# ── FeatureStoreV2 — best_circuit_for_features ───────────────────────


def test_best_circuit_for_features(store):
    for _ in range(3):
        store.add_record(
            _make_record(circuit="RC", logf_slope_low=0.1, logf_slope_high=0.2)
        )
    store.add_record(
        _make_record(circuit="Randles", logf_slope_low=0.15, logf_slope_high=0.25)
    )
    best = store.best_circuit_for_features(
        {"logf_slope_low": 0.1, "logf_slope_high": 0.2}, n=10
    )
    assert best == "RC"


def test_best_circuit_for_features_empty(store):
    assert store.best_circuit_for_features({"logf_slope_low": 1.0}) is None


# ── FeatureStoreV2 — summary_text ────────────────────────────────────


def test_summary_text_empty(store):
    assert "vazio" in store.summary_text().lower()


def test_summary_text_nonempty(store):
    store.add_record(_make_record(circuit="RC"))
    text = store.summary_text()
    assert "RC" in text
    assert "1 registos" in text


# ── FeatureStoreV2 — clear ────────────────────────────────────────────


def test_clear_empties_store(store):
    for _ in range(5):
        store.add_record(_make_record())
    store.clear()
    assert len(store) == 0


def test_clear_then_add(store):
    store.add_record(_make_record())
    store.clear()
    store.add_record(_make_record(circuit="NewCircuit"))
    assert len(store) == 1
    assert store.records[0]["circuit_name"] == "NewCircuit"


# ── FeatureStoreV2 — add_records ─────────────────────────────────────


def test_add_records_batch(store):
    batch = [_make_record(sample_id=f"s{i}") for i in range(10)]
    store.add_records(batch)
    assert len(store) == 10


# ── FeatureStoreV2 — spectral features round-trip ────────────────────


def test_spectral_features_stored_and_retrieved(store):
    store.add_record(
        _make_record(
            sample_id="spec",
            logf_slope_low=-0.42,
            phase_min=-85.0,
            mag_range=200.0,
        )
    )
    rec = store.records[0]
    sf = rec.get("spectral_features") or {}
    assert sf.get("logf_slope_low") == pytest.approx(-0.42)
    assert sf.get("phase_min") == pytest.approx(-85.0)


def test_nested_spectral_features_dict(store):
    store.add_record(
        {
            "sample_id": "nested",
            "circuit_name": "RC",
            "spectral_features": {"logf_slope_low": -1.5, "phase_max": -10.0},
        }
    )
    rec = store.records[0]
    assert rec["spectral_features"]["logf_slope_low"] == pytest.approx(-1.5)
