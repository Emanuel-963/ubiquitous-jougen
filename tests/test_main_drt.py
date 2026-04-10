"""Contract tests for main_drt.run_drt_pipeline."""

from pathlib import Path

import pandas as pd

from main_drt import run_drt_pipeline


def _write_synthetic_eis(path: Path):
    # Loader accepts headers containing freq, Z', Z''
    content = "\n".join(
        [
            "freq\tZ'\tZ''",
            "10000\t1.05\t-0.05",
            "1000\t1.30\t-0.30",
            "100\t3.20\t-2.20",
            "10\t8.00\t-4.50",
            "1\t10.00\t-1.00",
        ]
    )
    path.write_text(content, encoding="utf-8")


def test_run_drt_pipeline_contract(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_eis(raw_dir / "sample_a.txt")

    result = run_drt_pipeline(
        lambda_reg=1e-3,
        n_taus=40,
        data_dir=str(raw_dir),
        show_plots=False,
    )

    required_keys = {
        "drt_table",
        "drt_peaks_table",
        "drt_summary_table",
        "per_file_results",
        "plot_paths",
        "errors",
        "run_meta",
    }
    assert required_keys.issubset(result.keys())

    assert isinstance(result["drt_table"], pd.DataFrame)
    assert isinstance(result["drt_peaks_table"], pd.DataFrame)
    assert isinstance(result["drt_summary_table"], pd.DataFrame)
    assert isinstance(result["per_file_results"], dict)
    assert isinstance(result["plot_paths"], list)
    assert isinstance(result["errors"], dict)
    assert isinstance(result["run_meta"], dict)

    for key in [
        "lambda_reg",
        "n_taus",
        "data_dir",
        "n_files",
        "n_success",
        "n_failed",
        "generated_at",
    ]:
        assert key in result["run_meta"]

    assert result["run_meta"]["n_files"] >= 1
    assert result["run_meta"]["n_success"] >= 1

    assert len(result["errors"]) == 0
    assert len(result["drt_table"]) >= 1
    assert len(result["drt_summary_table"]) >= 1


def test_run_drt_pipeline_peaks_table_columns(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_eis(raw_dir / "sample_b.txt")

    result = run_drt_pipeline(data_dir=str(raw_dir), show_plots=False)
    peaks_df = result["drt_peaks_table"]

    expected_cols = {
        "Arquivo",
        "Sample",
        "peak_order",
        "tau_peak",
        "gamma_peak",
        "width_decades",
    }
    assert expected_cols.issubset(peaks_df.columns)
