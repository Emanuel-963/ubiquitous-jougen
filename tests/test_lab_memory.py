"""Tests for src/lab_memory.py."""

from __future__ import annotations

import pandas as pd

from src.lab_memory import ExperimentalMemory, similarity_report


def _bench() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": ["A", "B", "C"],
            "rs": [10.0, 5.0, 8.0],
            "rp": [80.0, 120.0, 100.0],
            "c_mean": [1e-4, 2e-4, 1.5e-4],
            "chi2_over_nu": [3.0, 1.5, 2.1],
            "confidence": [0.7, 0.9, 0.8],
            "kk_valid": [1.0, 1.0, 0.0],
        }
    )


def test_insert_and_query_similarity(tmp_path):
    db = tmp_path / "lab_memory.db"
    mem = ExperimentalMemory(str(db))

    inserted = mem.add_from_benchmark_table(_bench(), notes="batch1")
    assert inserted == 3

    hits = mem.find_similar(
        {
            "rs": 5.2,
            "rp": 118.0,
            "c_mean": 2.1e-4,
            "chi2_over_nu": 1.6,
            "confidence": 0.92,
            "kk_valid": 1.0,
        },
        top_k=2,
    )
    assert len(hits) == 2
    assert hits[0].similarity >= hits[1].similarity


def test_similarity_report_text(tmp_path):
    db = tmp_path / "lab_memory.db"
    mem = ExperimentalMemory(str(db))
    mem.add_from_benchmark_table(_bench(), notes="batch1")

    text = similarity_report(
        mem,
        query_name="Q",
        query_signature={
            "rs": 10.0,
            "rp": 80.0,
            "c_mean": 1e-4,
            "chi2_over_nu": 3.0,
            "confidence": 0.7,
            "kk_valid": 1.0,
        },
        top_k=2,
    )
    assert "MEMORIA EXPERIMENTAL" in text
    assert "similaridade" in text
