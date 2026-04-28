"""Tests for the scientific EIS data exporters (Phase 3).

All fixtures are in-memory — no external files required.
Each test checks:
    1. The output file is created.
    2. Key structural content is present (headers, signatures, data values).
    3. Round-trip: the exported data can be re-read and contains the right
       numeric values (within floating-point tolerance).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.export import EXPORTERS as _EXPORTERS
from src.export import (
    ExportError,
    LaTeXExporter,
    MEISPExporter,
    OriginCSVExporter,
    ZViewExporter,
    export_circuit_table_latex,
    export_eis,
    export_ranking_latex,
)
from src.export.base import EISExporter

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal 3-point EIS spectrum."""
    return pd.DataFrame(
        {
            "frequency": [10000.0, 1000.0, 100.0],
            "zreal": [12.345, 13.100, 15.200],
            "zimag": [-3.210, -5.500, -8.900],
        }
    )


@pytest.fixture()
def sample_df_extra_cols() -> pd.DataFrame:
    """EIS spectrum with extra (non-standard) columns — should be ignored."""
    return pd.DataFrame(
        {
            "frequency": [10000.0, 1000.0, 100.0],
            "zreal": [12.345, 13.100, 15.200],
            "zimag": [-3.210, -5.500, -8.900],
            "zmag": [12.76, 14.21, 17.66],
            "phase": [-14.6, -22.8, -30.3],
        }
    )


@pytest.fixture()
def raw_eis(sample_df: pd.DataFrame) -> dict:
    return {
        "sample_A.txt": sample_df.copy(),
        "sample_B.txt": sample_df.copy(),
    }


@pytest.fixture()
def circuit_table_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sample": ["A", "B", "C"],
            "Rs (Ohm)": [1.23, 1.45, 1.67],
            "Rct (Ohm)": [34.5, 45.6, 56.7],
            "Cdl (F)": [1.2e-5, 1.4e-5, 1.6e-5],
            "Chi2": [1e-4, 2e-4, 3e-4],
        }
    ).set_index("Sample")


@pytest.fixture()
def ranked_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sample": ["A", "B", "C"],
            "Rank": [1, 2, 3],
            "Score": [85.0, 72.0, 60.0],
            "Rs": [1.23, 1.45, 1.67],
            "Rp": [34.5, 45.6, 56.7],
        }
    ).set_index("Sample")


# ---------------------------------------------------------------------------
# ZViewExporter tests
# ---------------------------------------------------------------------------


class TestZViewExporter:
    def test_creates_file(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.z"
        ZViewExporter().export_dataframe(sample_df, out, sample_name="TestSample")
        assert out.exists()

    def test_first_line_is_zahnprog(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.z"
        ZViewExporter().export_dataframe(sample_df, out)
        lines = out.read_text().splitlines()
        assert lines[0].startswith("ZAHNPROG")

    def test_correct_row_count(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.z"
        ZViewExporter().export_dataframe(sample_df, out)
        lines = out.read_text().splitlines()
        # Line 3 should be the count (index 2)
        assert int(lines[2]) == len(sample_df)

    def test_data_rows_count(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.z"
        ZViewExporter().export_dataframe(sample_df, out)
        lines = out.read_text().splitlines()
        data_lines = lines[3:]  # skip header (3 lines)
        assert len(data_lines) == len(sample_df)

    def test_zimag_sign_inverted(self, tmp_path: Path, sample_df: pd.DataFrame):
        """ZView expects -Im(Z) convention (positive for capacitive)."""
        out = tmp_path / "eis.z"
        ZViewExporter().export_dataframe(sample_df, out)
        lines = out.read_text().splitlines()
        first_data = lines[3].split("\t")
        zimag_in_file = float(first_data[2])
        # original zimag = -3.210 → stored as +3.210
        assert zimag_in_file > 0

    def test_frequency_value_correct(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.z"
        ZViewExporter().export_dataframe(sample_df, out)
        lines = out.read_text().splitlines()
        freq_in_file = float(lines[3].split("\t")[0])
        assert math.isclose(freq_in_file, 10000.0, rel_tol=1e-4)

    def test_extra_cols_ignored(
        self, tmp_path: Path, sample_df_extra_cols: pd.DataFrame
    ):
        out = tmp_path / "eis.z"
        ZViewExporter().export_dataframe(sample_df_extra_cols, out)
        lines = out.read_text().splitlines()
        # Each data line should have exactly 3 tab-separated columns
        assert len(lines[3].split("\t")) == 3

    def test_default_extension(self):
        assert ZViewExporter.DEFAULT_EXTENSION == ".z"

    def test_export_all_creates_multiple_files(self, tmp_path: Path, raw_eis: dict):
        paths = ZViewExporter().export_all(raw_eis, tmp_path)
        assert len(paths) == 2
        for p in paths:
            assert p.suffix == ".z"
            assert p.exists()

    def test_export_eis_shortcut(self, tmp_path: Path, raw_eis: dict):
        paths = export_eis(raw_eis, fmt="zview", out_dir=tmp_path)
        assert len(paths) == 2

    def test_raises_on_missing_columns(self, tmp_path: Path):
        bad_df = pd.DataFrame({"frequency": [1000.0], "zreal": [10.0]})
        with pytest.raises(ExportError, match="zimag"):
            ZViewExporter().export_dataframe(bad_df, tmp_path / "x.z")

    def test_raises_on_empty_df(self, tmp_path: Path):
        empty_df = pd.DataFrame(columns=["frequency", "zreal", "zimag"])
        with pytest.raises(ExportError):
            ZViewExporter().export_dataframe(empty_df, tmp_path / "x.z")


# ---------------------------------------------------------------------------
# MEISPExporter tests
# ---------------------------------------------------------------------------


class TestMEISPExporter:
    def test_creates_file(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.mps"
        MEISPExporter().export_dataframe(sample_df, out, sample_name="TestSample")
        assert out.exists()

    def test_first_line_is_measurement_file(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ):
        out = tmp_path / "eis.mps"
        MEISPExporter().export_dataframe(sample_df, out)
        lines = out.read_text().splitlines()
        assert lines[0] == "Measurement file"

    def test_fourth_line_is_count(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.mps"
        MEISPExporter().export_dataframe(sample_df, out)
        lines = out.read_text().splitlines()
        assert int(lines[3]) == len(sample_df)

    def test_zimag_sign_preserved(self, tmp_path: Path, sample_df: pd.DataFrame):
        """MEISP stores natural Im(Z) — negative for capacitive."""
        out = tmp_path / "eis.mps"
        MEISPExporter().export_dataframe(sample_df, out)
        lines = out.read_text().splitlines()
        first_data = lines[4].split("\t")
        zimag_in_file = float(first_data[2])
        assert zimag_in_file < 0

    def test_frequency_value_correct(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.mps"
        MEISPExporter().export_dataframe(sample_df, out)
        lines = out.read_text().splitlines()
        freq_in_file = float(lines[4].split("\t")[0])
        assert math.isclose(freq_in_file, 10000.0, rel_tol=1e-4)

    def test_default_extension(self):
        assert MEISPExporter.DEFAULT_EXTENSION == ".mps"

    def test_export_all_creates_files(self, tmp_path: Path, raw_eis: dict):
        paths = MEISPExporter().export_all(raw_eis, tmp_path)
        assert len(paths) == 2

    def test_export_eis_shortcut(self, tmp_path: Path, raw_eis: dict):
        paths = export_eis(raw_eis, fmt="meisp", out_dir=tmp_path)
        assert len(paths) == 2


# ---------------------------------------------------------------------------
# OriginCSVExporter tests
# ---------------------------------------------------------------------------


class TestOriginCSVExporter:
    def test_creates_file(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.csv"
        OriginCSVExporter().export_dataframe(sample_df, out, sample_name="TestSample")
        assert out.exists()

    def test_comment_block_present(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.csv"
        OriginCSVExporter().export_dataframe(sample_df, out, sample_name="MySample")
        content = out.read_text()
        assert "# IonFlow Pipeline Export" in content
        assert "# Sample: MySample" in content

    def test_column_header_present(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.csv"
        OriginCSVExporter().export_dataframe(sample_df, out)
        content = out.read_text()
        assert "Frequency(Hz)" in content
        assert "Z'(Ohm)" in content
        assert "-Z''(Ohm)" in content

    def test_data_row_count(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.csv"
        OriginCSVExporter().export_dataframe(sample_df, out)
        lines = [
            line
            for line in out.read_text().splitlines()
            if line and not line.startswith("#") and "Frequency" not in line
        ]
        assert len(lines) == len(sample_df)

    def test_zimag_sign_inverted(self, tmp_path: Path, sample_df: pd.DataFrame):
        """Origin stores -Im(Z) convention (positive for capacitive)."""
        out = tmp_path / "eis.csv"
        OriginCSVExporter().export_dataframe(sample_df, out)
        lines = [
            line
            for line in out.read_text().splitlines()
            if line and not line.startswith("#") and "Frequency" not in line
        ]
        neg_zimag = float(lines[0].split(",")[2])
        assert neg_zimag > 0

    def test_extra_meta_in_comments(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.csv"
        OriginCSVExporter().export_dataframe(
            sample_df, out, extra_meta={"Electrolyte": "H2SO4"}
        )
        assert "# Electrolyte: H2SO4" in out.read_text()

    def test_default_extension(self):
        assert OriginCSVExporter.DEFAULT_EXTENSION == ".csv"

    def test_export_eis_shortcut(self, tmp_path: Path, raw_eis: dict):
        paths = export_eis(raw_eis, fmt="origin", out_dir=tmp_path)
        assert len(paths) == 2


# ---------------------------------------------------------------------------
# LaTeXExporter tests
# ---------------------------------------------------------------------------


class TestLaTeXExporter:
    def test_creates_file(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.tex"
        LaTeXExporter().export_dataframe(sample_df, out, sample_name="TestSample")
        assert out.exists()

    def test_contains_booktabs_macros(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.tex"
        LaTeXExporter().export_dataframe(sample_df, out)
        content = out.read_text()
        assert "\\toprule" in content
        assert "\\midrule" in content
        assert "\\bottomrule" in content

    def test_contains_table_environment(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.tex"
        LaTeXExporter().export_dataframe(sample_df, out)
        content = out.read_text()
        assert "\\begin{table}" in content
        assert "\\end{table}" in content

    def test_data_rows_count(self, tmp_path: Path, sample_df: pd.DataFrame):
        out = tmp_path / "eis.tex"
        LaTeXExporter().export_dataframe(sample_df, out)
        content = out.read_text()
        # Each data row ends with \\ and starts with whitespace + a digit/minus
        import re

        data_rows = [
            line
            for line in content.splitlines()
            if line.strip().endswith("\\\\") and re.match(r"\s+[\d\-]", line)
        ]
        assert len(data_rows) == len(sample_df)

    def test_default_extension(self):
        assert LaTeXExporter.DEFAULT_EXTENSION == ".tex"

    def test_export_eis_shortcut(self, tmp_path: Path, raw_eis: dict):
        paths = export_eis(raw_eis, fmt="latex", out_dir=tmp_path)
        assert len(paths) == 2


class TestExportCircuitTableLatex:
    def test_creates_file(self, tmp_path: Path, circuit_table_df: pd.DataFrame):
        out = tmp_path / "circuit.tex"
        export_circuit_table_latex(circuit_table_df, out)
        assert out.exists()

    def test_contains_caption(self, tmp_path: Path, circuit_table_df: pd.DataFrame):
        out = tmp_path / "circuit.tex"
        export_circuit_table_latex(circuit_table_df, out, caption="My caption")
        assert "My caption" in out.read_text()

    def test_all_sample_names_present(
        self, tmp_path: Path, circuit_table_df: pd.DataFrame
    ):
        out = tmp_path / "circuit.tex"
        export_circuit_table_latex(circuit_table_df, out)
        content = out.read_text()
        for sample in circuit_table_df.index:
            assert str(sample) in content

    def test_raises_on_empty(self, tmp_path: Path):
        empty_df = pd.DataFrame()
        with pytest.raises(ExportError):
            export_circuit_table_latex(empty_df, tmp_path / "x.tex")


class TestExportRankingLatex:
    def test_creates_file(self, tmp_path: Path, ranked_df: pd.DataFrame):
        out = tmp_path / "ranking.tex"
        export_ranking_latex(ranked_df, out)
        assert out.exists()

    def test_rank_column_present(self, tmp_path: Path, ranked_df: pd.DataFrame):
        out = tmp_path / "ranking.tex"
        export_ranking_latex(ranked_df, out)
        assert "Rank" in out.read_text()

    def test_sample_names_present(self, tmp_path: Path, ranked_df: pd.DataFrame):
        out = tmp_path / "ranking.tex"
        export_ranking_latex(ranked_df, out)
        content = out.read_text()
        for sample in ranked_df.index:
            assert str(sample) in content

    def test_raises_on_empty(self, tmp_path: Path):
        empty_df = pd.DataFrame()
        with pytest.raises(ExportError):
            export_ranking_latex(empty_df, tmp_path / "x.tex")


# ---------------------------------------------------------------------------
# export_eis() integration tests
# ---------------------------------------------------------------------------


class TestExportEIS:
    def test_unknown_fmt_raises(self, tmp_path: Path, raw_eis: dict):
        with pytest.raises(ValueError, match="Unknown export format"):
            export_eis(raw_eis, fmt="unknown_fmt", out_dir=tmp_path)

    def test_all_registered_formats_work(self, tmp_path: Path, raw_eis: dict):
        for fmt in _EXPORTERS:
            sub_dir = tmp_path / fmt
            paths = export_eis(raw_eis, fmt=fmt, out_dir=sub_dir)
            assert len(paths) == len(raw_eis), f"Format '{fmt}' wrote wrong count"
            for p in paths:
                assert p.exists(), f"Format '{fmt}': file not created: {p}"

    def test_creates_output_directory(self, tmp_path: Path, raw_eis: dict):
        new_dir = tmp_path / "nested" / "subdir"
        assert not new_dir.exists()
        export_eis(raw_eis, fmt="zview", out_dir=new_dir)
        assert new_dir.exists()

    def test_returns_list_of_paths(self, tmp_path: Path, raw_eis: dict):
        result = export_eis(raw_eis, fmt="meisp", out_dir=tmp_path)
        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)


# ---------------------------------------------------------------------------
# EISExporter base / registry tests
# ---------------------------------------------------------------------------


class TestExporterRegistry:
    def test_all_exporters_are_subclasses(self):
        for key, cls in _EXPORTERS.items():
            assert issubclass(
                cls, EISExporter
            ), f"EXPORTERS['{key}'] is not a subclass of EISExporter"

    def test_all_exporters_have_format_name(self):
        for key, cls in _EXPORTERS.items():
            assert cls.FORMAT_NAME, f"EXPORTERS['{key}'] has empty FORMAT_NAME"

    def test_all_exporters_have_default_extension(self):
        for key, cls in _EXPORTERS.items():
            assert cls.DEFAULT_EXTENSION.startswith(
                "."
            ), f"EXPORTERS['{key}'].DEFAULT_EXTENSION should start with '.'"
