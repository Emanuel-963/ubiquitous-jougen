"""Tests for src/comparison/auto_benchmark.py."""

from __future__ import annotations

import pandas as pd

from src.comparison.auto_benchmark import (
    benchmark_report,
    prepare_benchmark_table,
    recommend_best_configuration,
    score_objective,
)


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": ["Nb2_H2SO4_500", "Nb4_NH4F_500", "Nb4_LiCl_1000"],
            "best_circuit": ["Randles", "Randles-CPE", "Porous-TLM"],
            "Rs_fit": [12.0, 6.0, 9.5],
            "Rp_fit": [80.0, 140.0, 110.0],
            "C_mean": [1e-4, 2.8e-4, 2.0e-4],
            "chi2_over_nu": [3.2, 1.4, 2.2],
            "kk_valid": [1.0, 1.0, 0.0],
            "confidence": [0.75, 0.9, 0.8],
        }
    )


def test_prepare_table_columns():
    bench = prepare_benchmark_table(_table())
    assert set(
        [
            "sample",
            "circuit",
            "rs",
            "rp",
            "c_mean",
            "chi2_over_nu",
            "kk_valid",
            "confidence",
        ]
    ).issubset(bench.columns)


def test_low_rs_picks_expected_sample():
    rec = recommend_best_configuration(_table(), objective="low_rs", top_k=2)
    assert rec.sample == "Nb4_NH4F_500"
    assert len(rec.top_candidates) == 2


def test_high_rp_picks_expected_sample():
    rec = recommend_best_configuration(_table(), objective="high_rp", top_k=2)
    assert rec.sample == "Nb4_NH4F_500"


def test_balanced_scores_size_matches_rows():
    bench = prepare_benchmark_table(_table())
    s = score_objective(bench, "balanced")
    assert len(s) == len(bench)


def test_report_contains_objective():
    text = benchmark_report(_table(), objectives=["balanced"], top_k=2)
    assert "Objective: balanced" in text
    assert "Melhor configuracao" in text
