"""Tests for the GUI tab modules and widgets (Day 14).

All tests are headless — no tkinter / customtkinter required.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

# ── widgets ──────────────────────────────────────────────────────────
from src.gui.widgets import (
    ChartExporter,
    ExportResult,
    FilterableTableManager,
    LogRedirector,
    StyledOptionMenuHelper,
    TableState,
)

# ── tabs ─────────────────────────────────────────────────────────────
from src.gui.tabs.advanced_charts import (
    build_fig_corr,
    build_fig_drt_eis,
    build_fig_pca,
    build_fig_pca_metric,
    build_fig_rank,
    build_fig_series,
    _find_matching_index,
    _split_arquivo,
)
from src.gui.tabs.tables import TableColumnConfig, table_column_configs


# ═══════════════════════════════════════════════════════════════════════
#  LogRedirector
# ═══════════════════════════════════════════════════════════════════════


class TestLogRedirector:
    def test_write_non_empty(self):
        msgs: List[str] = []
        lr = LogRedirector(msgs.append)
        lr.write("hello")
        assert msgs == ["hello"]

    def test_write_blank_ignored(self):
        msgs: List[str] = []
        lr = LogRedirector(msgs.append)
        lr.write("   ")
        lr.write("")
        lr.write("\n")
        assert msgs == []

    def test_flush_returns_none(self):
        lr = LogRedirector(lambda m: None)
        assert lr.flush() is None


# ═══════════════════════════════════════════════════════════════════════
#  ChartExporter
# ═══════════════════════════════════════════════════════════════════════


class TestChartExporter:
    def test_save_figure(self, tmp_path):
        fig = Figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        dest = str(tmp_path / "test_plot.png")
        result = ChartExporter.save_figure(fig, dest)
        assert result.success is True
        assert os.path.isfile(result.path)

    def test_save_figure_svg(self, tmp_path):
        fig = Figure(figsize=(3, 2))
        fig.add_subplot(111)
        dest = str(tmp_path / "test_plot.svg")
        result = ChartExporter.save_figure(fig, dest)
        assert result.success is True
        assert dest.endswith(".svg")

    def test_save_figure_creates_dirs(self, tmp_path):
        fig = Figure(figsize=(2, 2))
        fig.add_subplot(111)
        dest = str(tmp_path / "subdir" / "deep" / "plot.png")
        result = ChartExporter.save_figure(fig, dest)
        assert result.success is True

    def test_copy_image(self, tmp_path):
        src = tmp_path / "source.png"
        src.write_text("fake image")
        dest_dir = str(tmp_path / "dest")
        result = ChartExporter.copy_image(str(src), dest_dir)
        assert result.success is True
        assert os.path.isfile(os.path.join(dest_dir, "source.png"))

    def test_copy_image_missing(self, tmp_path):
        result = ChartExporter.copy_image(
            str(tmp_path / "missing.png"),
            str(tmp_path / "dest"),
        )
        assert result.success is False
        assert "não encontrado" in result.error

    def test_export_formats(self):
        fmts = ChartExporter.export_formats()
        assert ".png" in fmts
        assert ".svg" in fmts
        assert ".pdf" in fmts


class TestExportResult:
    def test_default_values(self):
        r = ExportResult(success=True)
        assert r.path == ""
        assert r.error == ""


# ═══════════════════════════════════════════════════════════════════════
#  FilterableTableManager
# ═══════════════════════════════════════════════════════════════════════


class TestFilterableTableManager:
    def _sample_df(self):
        return pd.DataFrame({
            "Name": ["Alpha", "Beta", "Gamma", "Delta"],
            "Value": [10, 20, 30, 40],
            "Score": [1.1, 2.2, 3.3, 4.4],
        })

    def test_register_and_keys(self):
        mgr = FilterableTableManager()
        mgr.register("eis")
        mgr.register("cic")
        assert set(mgr.keys) == {"eis", "cic"}

    def test_set_data(self):
        mgr = FilterableTableManager()
        mgr.register("eis")
        df = self._sample_df()
        mgr.set_data("eis", df)
        state = mgr.get_state("eis")
        assert state is not None
        assert state.source_df is not None
        assert len(state.source_df) == 4

    def test_filter_basic(self):
        mgr = FilterableTableManager()
        mgr.register("t")
        mgr.set_data("t", self._sample_df())
        result = mgr.apply_filter("t", "alpha")
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["Name"] == "Alpha"

    def test_filter_empty_text_returns_all(self):
        mgr = FilterableTableManager()
        mgr.register("t")
        mgr.set_data("t", self._sample_df())
        result = mgr.apply_filter("t", "")
        assert len(result) == 4

    def test_filter_no_match(self):
        mgr = FilterableTableManager()
        mgr.register("t")
        mgr.set_data("t", self._sample_df())
        result = mgr.apply_filter("t", "zzzzz")
        assert len(result) == 0

    def test_filter_numeric_value(self):
        mgr = FilterableTableManager()
        mgr.register("t")
        mgr.set_data("t", self._sample_df())
        result = mgr.apply_filter("t", "20")
        assert len(result) == 1

    def test_toggle_sort_asc(self):
        mgr = FilterableTableManager()
        mgr.register("t")
        mgr.set_data("t", self._sample_df())
        result = mgr.toggle_sort("t", "Value")
        assert result is not None
        assert list(result["Value"]) == [10, 20, 30, 40]

    def test_toggle_sort_desc(self):
        mgr = FilterableTableManager()
        mgr.register("t")
        mgr.set_data("t", self._sample_df())
        mgr.toggle_sort("t", "Value")
        result = mgr.toggle_sort("t", "Value")
        assert list(result["Value"]) == [40, 30, 20, 10]

    def test_toggle_sort_new_column_resets_asc(self):
        mgr = FilterableTableManager()
        mgr.register("t")
        mgr.set_data("t", self._sample_df())
        mgr.toggle_sort("t", "Value")
        mgr.toggle_sort("t", "Value")
        result = mgr.toggle_sort("t", "Name")
        assert result.iloc[0]["Name"] == "Alpha"

    def test_toggle_sort_nonexistent_column(self):
        mgr = FilterableTableManager()
        mgr.register("t")
        mgr.set_data("t", self._sample_df())
        result = mgr.toggle_sort("t", "NonExistent")
        assert len(result) == 4

    def test_row_counts(self):
        mgr = FilterableTableManager()
        mgr.register("t")
        mgr.set_data("t", self._sample_df())
        mgr.apply_filter("t", "alpha")
        visible, total = mgr.row_counts("t")
        assert visible == 1
        assert total == 4

    def test_row_counts_unregistered(self):
        mgr = FilterableTableManager()
        assert mgr.row_counts("nope") == (0, 0)

    def test_estimate_column_width(self):
        w = FilterableTableManager.estimate_column_width(
            "Name", ["Alpha", "Beta", "Gamma"]
        )
        assert 70 <= w <= 350

    def test_estimate_column_width_long_values(self):
        w = FilterableTableManager.estimate_column_width(
            "Description",
            ["A very very very long description text that exceeds max"],
        )
        assert w == 350

    def test_get_state_unregistered(self):
        mgr = FilterableTableManager()
        assert mgr.get_state("nope") is None

    def test_set_data_unregistered_noop(self):
        mgr = FilterableTableManager()
        mgr.set_data("nope", pd.DataFrame())  # should not raise

    def test_apply_filter_unregistered(self):
        mgr = FilterableTableManager()
        assert mgr.apply_filter("nope", "x") is None

    def test_apply_filter_no_data(self):
        mgr = FilterableTableManager()
        mgr.register("t")
        assert mgr.apply_filter("t", "x") is None


# ═══════════════════════════════════════════════════════════════════════
#  StyledOptionMenuHelper
# ═══════════════════════════════════════════════════════════════════════


class TestStyledOptionMenuHelper:
    def test_numeric_columns(self):
        df = pd.DataFrame({"a": [1], "b": ["x"], "c": [2.0]})
        cols = StyledOptionMenuHelper.numeric_columns(df)
        assert "a" in cols
        assert "c" in cols
        assert "b" not in cols

    def test_numeric_columns_none(self):
        assert StyledOptionMenuHelper.numeric_columns(None) == []

    def test_numeric_columns_empty(self):
        assert StyledOptionMenuHelper.numeric_columns(pd.DataFrame()) == []

    def test_sample_names_sorted(self):
        data = {"c": None, "a": None, "b": None}
        assert StyledOptionMenuHelper.sample_names(data) == ["a", "b", "c"]

    def test_sample_names_unsorted(self):
        data = {"c": None, "a": None}
        names = StyledOptionMenuHelper.sample_names(data, sort=False)
        assert set(names) == {"a", "c"}

    def test_unique_series_bases(self):
        df = pd.DataFrame({
            "Arquivo": ["1 NF H2SO4", "2 NF H2SO4", "1 GCD NaOH", "2 GCD NaOH"]
        })
        bases = StyledOptionMenuHelper.unique_series_bases(df)
        assert "GCD NaOH" in bases
        assert "NF H2SO4" in bases

    def test_unique_series_bases_none(self):
        assert StyledOptionMenuHelper.unique_series_bases(None) == []

    def test_unique_series_bases_no_column(self):
        df = pd.DataFrame({"X": [1]})
        assert StyledOptionMenuHelper.unique_series_bases(df) == []

    def test_unique_series_bases_no_numeric_prefix(self):
        df = pd.DataFrame({"Arquivo": ["SampleA", "SampleB"]})
        bases = StyledOptionMenuHelper.unique_series_bases(df)
        assert "SampleA" in bases
        assert "SampleB" in bases


# ═══════════════════════════════════════════════════════════════════════
#  TableState
# ═══════════════════════════════════════════════════════════════════════


class TestTableState:
    def test_defaults(self):
        ts = TableState()
        assert ts.source_df is None
        assert ts.view_df is None
        assert ts.sort_col is None
        assert ts.sort_asc is True
        assert ts.filter_text == ""


# ═══════════════════════════════════════════════════════════════════════
#  advanced_charts helpers
# ═══════════════════════════════════════════════════════════════════════


class TestAdvancedHelpers:
    def test_find_matching_index_exact(self):
        assert _find_matching_index(["abc", "def"], "abc") == 0

    def test_find_matching_index_normalized(self):
        assert _find_matching_index(["ABC.txt", "DEF"], "abc") == 0

    def test_find_matching_index_suffix(self):
        assert _find_matching_index(["1 nf h2so4", "2 nf h2so4"], "nf h2so4") == 0

    def test_find_matching_index_none(self):
        assert _find_matching_index(["abc"], None) is None

    def test_find_matching_index_not_found(self):
        assert _find_matching_index(["abc"], "xyz") is None

    def test_split_arquivo_normal(self):
        lead, base = _split_arquivo("1 NF H2SO4")
        assert lead == 1.0
        assert base == "NF H2SO4"

    def test_split_arquivo_no_number(self):
        lead, base = _split_arquivo("SampleA")
        assert lead == 0.0
        assert base == "SampleA"

    def test_split_arquivo_with_txt(self):
        lead, base = _split_arquivo("3 GCD.txt")
        assert lead == 3.0
        assert base == "GCD"


# ═══════════════════════════════════════════════════════════════════════
#  advanced_charts — build_fig_rank
# ═══════════════════════════════════════════════════════════════════════


class TestBuildFigRank:
    def _rank_df(self):
        return pd.DataFrame(
            {"Rank": [1, 2, 3], "Retenção (%)": [95.0, 80.0, 70.0]},
            index=["s1", "s2", "s3"],
        )

    def test_returns_figure(self):
        fig = build_fig_rank(self._rank_df())
        assert isinstance(fig, Figure)

    def test_returns_none_for_empty(self):
        assert build_fig_rank(None) is None
        assert build_fig_rank(pd.DataFrame()) is None

    def test_returns_none_missing_columns(self):
        df = pd.DataFrame({"X": [1]})
        assert build_fig_rank(df) is None

    def test_highlight_sample(self):
        fig = build_fig_rank(self._rank_df(), highlight_sample="s2")
        assert isinstance(fig, Figure)

    def test_highlight_not_found(self):
        fig = build_fig_rank(self._rank_df(), highlight_sample="missing")
        assert isinstance(fig, Figure)


# ═══════════════════════════════════════════════════════════════════════
#  advanced_charts — build_fig_pca
# ═══════════════════════════════════════════════════════════════════════


class TestBuildFigPCA:
    def _pca_df(self):
        return pd.DataFrame(
            {"PC1": [0.1, 0.2, 0.3], "PC2": [0.4, 0.5, 0.6]},
            index=["a", "b", "c"],
        )

    def test_returns_figure(self):
        fig = build_fig_pca(self._pca_df())
        assert isinstance(fig, Figure)

    def test_returns_none(self):
        assert build_fig_pca(None) is None
        assert build_fig_pca(pd.DataFrame()) is None

    def test_highlight(self):
        fig = build_fig_pca(self._pca_df(), highlight_sample="b")
        assert isinstance(fig, Figure)


# ═══════════════════════════════════════════════════════════════════════
#  advanced_charts — build_fig_pca_metric
# ═══════════════════════════════════════════════════════════════════════


class TestBuildFigPCAMetric:
    def _data(self):
        pca = pd.DataFrame(
            {"PC1": [0.1, 0.2], "PC2": [0.3, 0.4]},
            index=["a", "b"],
        )
        rank = pd.DataFrame(
            {"Retenção (%)": [90.0, 80.0]},
            index=["a", "b"],
        )
        return pca, rank

    def test_with_retention(self):
        pca, rank = self._data()
        fig = build_fig_pca_metric(pca, rank)
        assert isinstance(fig, Figure)

    def test_without_retention(self):
        pca, _ = self._data()
        fig = build_fig_pca_metric(pca)
        assert isinstance(fig, Figure)

    def test_none(self):
        assert build_fig_pca_metric(None) is None


# ═══════════════════════════════════════════════════════════════════════
#  advanced_charts — build_fig_corr
# ═══════════════════════════════════════════════════════════════════════


class TestBuildFigCorr:
    def test_returns_figure(self):
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [3.0, 2.0, 1.0],
            "C": [1.5, 2.5, 3.5],
        })
        fig = build_fig_corr(df)
        assert isinstance(fig, Figure)

    def test_returns_none_single_col(self):
        df = pd.DataFrame({"A": [1, 2]})
        assert build_fig_corr(df) is None

    def test_returns_none_empty(self):
        assert build_fig_corr(None) is None


# ═══════════════════════════════════════════════════════════════════════
#  advanced_charts — build_fig_drt_eis
# ═══════════════════════════════════════════════════════════════════════


class TestBuildFigDRTEIS:
    def _df(self):
        return pd.DataFrame({
            "Sample": ["s1", "s2"],
            "gamma_peak_main": [0.5, 0.3],
            "Retenção (%)": [90.0, 80.0],
        })

    def test_returns_figure(self):
        fig = build_fig_drt_eis(self._df())
        assert isinstance(fig, Figure)

    def test_highlight(self):
        fig = build_fig_drt_eis(self._df(), highlight_sample="s1")
        assert isinstance(fig, Figure)

    def test_missing_columns(self):
        df = pd.DataFrame({"X": [1]})
        assert build_fig_drt_eis(df) is None

    def test_all_nan(self):
        df = pd.DataFrame({
            "gamma_peak_main": [np.nan],
            "Retenção (%)": [np.nan],
        })
        assert build_fig_drt_eis(df) is None


# ═══════════════════════════════════════════════════════════════════════
#  advanced_charts — build_fig_series
# ═══════════════════════════════════════════════════════════════════════


class TestBuildFigSeries:
    def _df(self):
        return pd.DataFrame({
            "Arquivo": ["1 NF H2SO4", "2 NF H2SO4", "3 NF H2SO4"],
            "Rs": [1.0, 1.5, 2.0],
        })

    def test_returns_figure(self):
        fig = build_fig_series(self._df(), "Rs", "NF H2SO4")
        assert isinstance(fig, Figure)

    def test_returns_none_wrong_base(self):
        assert build_fig_series(self._df(), "Rs", "Wrong Base") is None

    def test_returns_none_wrong_col(self):
        assert build_fig_series(self._df(), "NonExistent", "NF H2SO4") is None

    def test_returns_none_empty(self):
        assert build_fig_series(None, "Rs", "X") is None

    def test_returns_none_no_arquivo(self):
        df = pd.DataFrame({"Rs": [1, 2]})
        assert build_fig_series(df, "Rs", "X") is None


# ═══════════════════════════════════════════════════════════════════════
#  Table configs
# ═══════════════════════════════════════════════════════════════════════


class TestTableColumnConfig:
    def test_all_six_keys(self):
        configs = table_column_configs()
        assert set(configs.keys()) == {
            "eis", "cic", "circuit", "drt", "drt_peaks", "drt_eis"
        }

    def test_has_required_fields(self):
        for key, cfg in table_column_configs().items():
            assert isinstance(cfg.key, str)
            assert isinstance(cfg.label, str)
            assert isinstance(cfg.default_name, str)
            assert cfg.key == key

    def test_default_names_have_extension(self):
        for cfg in table_column_configs().values():
            assert cfg.default_name.endswith(".csv")


# ═══════════════════════════════════════════════════════════════════════
#  Package imports
# ═══════════════════════════════════════════════════════════════════════


class TestTabsPackageImports:
    """Verify that the tabs __init__ re-exports work."""

    def test_import_eis_charts(self):
        from src.gui.tabs import build_fig_nyquist, build_fig_bode, build_fig_impedance_heatmap
        assert callable(build_fig_nyquist)
        assert callable(build_fig_bode)
        assert callable(build_fig_impedance_heatmap)

    def test_import_cycling_charts(self):
        from src.gui.tabs import (
            build_fig_energy_power, build_fig_energy_cycle,
            build_fig_retention_cycle, build_fig_ragone,
        )
        assert callable(build_fig_energy_power)
        assert callable(build_fig_ragone)

    def test_import_drt_charts(self):
        from src.gui.tabs import (
            build_fig_drt_spectrum, build_fig_drt_overlay, build_fig_drt_heatmap,
        )
        assert callable(build_fig_drt_spectrum)

    def test_import_advanced_charts(self):
        from src.gui.tabs import (
            build_fig_rank, build_fig_pca, build_fig_pca_metric,
            build_fig_corr, build_fig_drt_eis, build_fig_series,
        )
        assert callable(build_fig_rank)
        assert callable(build_fig_series)

    def test_import_tables(self):
        from src.gui.tabs import TableColumnConfig, table_column_configs
        assert callable(table_column_configs)
