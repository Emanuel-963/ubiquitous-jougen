"""Tests for src/paper_first_export.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.paper_first_export import build_paper_first_package


@dataclass
class _FakeEIS:
    ranked_df: pd.DataFrame
    circuit_table: pd.DataFrame
    pca: object = None
    raw_eis: dict = None


@dataclass
class _FakeCycling:
    merged_table: pd.DataFrame
    energy_power_paths: list
    results: dict


@dataclass
class _FakeDRT:
    drt_summary_table: pd.DataFrame
    drt_peaks_table: pd.DataFrame
    plot_paths: list
    run_meta: dict


def test_paper_first_creates_files(tmp_path):
    eis = _FakeEIS(
        ranked_df=pd.DataFrame(
            {
                "Sample": ["S1"],
                "Rs_fit": [5.0],
                "Rp_fit": [120.0],
                "Score": [88.0],
                "Rank": [1],
            }
        ),
        circuit_table=pd.DataFrame({"Circuito": ["Randles-CPE"]}),
        pca=None,
        raw_eis={"S1": pd.DataFrame()},
    )

    cyc = _FakeCycling(
        merged_table=pd.DataFrame({"sample": ["S1"], "Energy_mean": [12.3]}),
        energy_power_paths=[],
        results={"S1": {}},
    )

    drt = _FakeDRT(
        drt_summary_table=pd.DataFrame({"sample": ["S1"], "n_peaks": [2]}),
        drt_peaks_table=pd.DataFrame({"sample": ["S1"], "tau": [0.1]}),
        plot_paths=[],
        run_meta={"n_success": 1, "n_failed": 0},
    )

    out = tmp_path / "paper_first"
    result = build_paper_first_package(
        {"eis": eis, "cycling": cyc, "drt": drt},
        output_dir=str(out),
        title="Draft",
        author="Tester",
    )

    assert Path(result.manuscript_path).exists()
    assert Path(result.figures_index_path).exists()
    assert Path(result.tables_dir).exists()

    txt = Path(result.manuscript_path).read_text(encoding="utf-8")
    assert "## Methods (Draft)" in txt
    assert "## Results (Draft)" in txt
    assert "## Draft Discussion" in txt
