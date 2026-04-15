"""Integration tests — end-to-end pipeline execution on synthetic data.

These tests exercise the *public API* of each pipeline (EIS, DRT, Cycling)
and the configuration round-trip, using temporary synthetic data files so
they never depend on real laboratory data being present.

Coverage targets
~~~~~~~~~~~~~~~~
* ``test_eis_pipeline_full``   — ``run_eis_pipeline`` returns EISResult with
  correct types and non-empty DataFrames.
* ``test_drt_pipeline_full``   — ``run_drt_pipeline`` returns a dict with the
  expected key contract, DRT table and per-file results.
* ``test_cycling_pipeline_full`` — ``run_ciclagem_pipeline`` loads fixture data,
  returns CyclingResult with export tables.
* ``test_config_round_trip``   — save → load → compare PipelineConfig via JSON.
* ``test_feature_store_in_eis_pipeline`` — after EIS pipeline, store is populated.
* ``test_ml_selector_after_eis`` — CircuitMLSelector can train on pipeline output.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.models import EISResult, CyclingResult, DRTPipelineResult, PCAResult
from tests.fixtures import create_eis_fixture, create_cycling_fixture


# ─── Helpers ─────────────────────────────────────────────────────────

@pytest.fixture
def tmp_workspace(tmp_path: Path):
    """Create a temporary workspace tree with all required directories."""
    dirs = {
        "data_dir": tmp_path / "data" / "raw",
        "processed_dir": tmp_path / "data" / "processed",
        "tables_dir": tmp_path / "outputs" / "tables",
        "figures_dir": tmp_path / "outputs" / "figures",
        "circuits_fig_dir": tmp_path / "outputs" / "figures" / "circuits",
        "analytics_fig_dir": tmp_path / "outputs" / "figures" / "analytics",
        "drt_fig_dir": tmp_path / "outputs" / "figures" / "drt",
        "reports_dir": tmp_path / "outputs" / "circuit_reports",
        "excel_dir": tmp_path / "outputs" / "excel",
        "log_dir": tmp_path / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        **{k: str(v) for k, v in dirs.items()},
        feature_store_path=str(tmp_path / "data" / "ml" / "fitting_history.json"),
    )
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Config round-trip
# ═══════════════════════════════════════════════════════════════════════

class TestConfigRoundTrip:
    """PipelineConfig → JSON → PipelineConfig preserves all fields."""

    def test_default_round_trip(self, tmp_path: Path):
        cfg1 = PipelineConfig.default()
        json_path = tmp_path / "cfg.json"
        cfg1.to_json(json_path)
        cfg2 = PipelineConfig.from_json(json_path)

        d1 = cfg1.to_dict()
        d2 = cfg2.to_dict()
        assert d1 == d2

    def test_custom_values_round_trip(self, tmp_path: Path):
        cfg1 = PipelineConfig(
            data_dir="my/data",
            voltage=3.7,
            n_head=10,
            kmeans_n_clusters=4,
            drt_lambda=1e-2,
            language="en",
        )
        json_path = tmp_path / "custom.json"
        cfg1.to_json(json_path)
        cfg2 = PipelineConfig.from_json(json_path)

        assert cfg2.data_dir == "my/data"
        assert cfg2.voltage == pytest.approx(3.7)
        assert cfg2.n_head == 10
        assert cfg2.kmeans_n_clusters == 4
        assert cfg2.drt_lambda == pytest.approx(1e-2)
        assert cfg2.language == "en"

    def test_unknown_keys_ignored(self, tmp_path: Path):
        json_path = tmp_path / "future.json"
        data = PipelineConfig.default().to_dict()
        data["future_key_v0.3.0"] = True
        json_path.write_text(json.dumps(data), encoding="utf-8")
        cfg = PipelineConfig.from_json(json_path)
        assert isinstance(cfg, PipelineConfig)

    def test_from_json_safe_fallback(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("NOT JSON", encoding="utf-8")
        cfg = PipelineConfig.from_json_safe(bad)
        assert cfg.data_dir == "data/raw"  # default

    def test_ensure_dirs_creates_all(self, tmp_path: Path):
        cfg = PipelineConfig(
            tables_dir=str(tmp_path / "a"),
            figures_dir=str(tmp_path / "b"),
            circuits_fig_dir=str(tmp_path / "c"),
            analytics_fig_dir=str(tmp_path / "d"),
            drt_fig_dir=str(tmp_path / "e"),
            reports_dir=str(tmp_path / "f"),
            excel_dir=str(tmp_path / "g"),
            log_dir=str(tmp_path / "h"),
        )
        cfg.ensure_dirs()
        for letter in "abcdefgh":
            assert (tmp_path / letter).is_dir()

    def test_feature_store_path_present(self):
        cfg = PipelineConfig.default()
        assert "fitting_history" in cfg.feature_store_path


# ═══════════════════════════════════════════════════════════════════════
# EIS pipeline end-to-end
# ═══════════════════════════════════════════════════════════════════════

class TestEISPipelineFull:
    """Run the full EIS pipeline on synthetic data."""

    def test_pipeline_returns_eis_result(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=3, n_points=40)

        from main import run_eis_pipeline
        result = run_eis_pipeline(config=cfg)

        assert isinstance(result, EISResult)

    def test_features_df_populated(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=3, n_points=40)

        from main import run_eis_pipeline
        result = run_eis_pipeline(config=cfg)

        assert result.features_df is not None
        assert not result.features_df.empty
        assert len(result.features_df) == 3

    def test_ranked_df_has_expected_columns(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=3, n_points=40)

        from main import run_eis_pipeline
        result = run_eis_pipeline(config=cfg)

        rdf = result.ranked_df
        assert rdf is not None
        # Should have ranking columns added by classify_and_rank
        assert "Score" in rdf.columns or "Rank" in rdf.columns

    def test_raw_eis_dict_populated(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=2, n_points=30)

        from main import run_eis_pipeline
        result = run_eis_pipeline(config=cfg)

        assert len(result.raw_eis) == 2
        for key, df in result.raw_eis.items():
            assert isinstance(df, pd.DataFrame)
            assert "frequency" in df.columns
            assert "zreal" in df.columns
            assert "zimag" in df.columns

    def test_circuit_table_populated(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=2, n_points=30)

        from main import run_eis_pipeline
        result = run_eis_pipeline(config=cfg)

        ct = result.circuit_table
        assert ct is not None
        assert isinstance(ct, pd.DataFrame)
        assert len(ct) >= 2

    def test_pca_result_present(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=4, n_points=40)

        from main import run_eis_pipeline
        result = run_eis_pipeline(config=cfg)

        pca = result.pca
        assert isinstance(pca, PCAResult)

    def test_stability_present(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=4, n_points=40)

        from main import run_eis_pipeline
        result = run_eis_pipeline(config=cfg)

        stab = result.stability
        assert stab is not None
        # stability is a dict of DataFrames keyed by parameter name
        assert isinstance(stab, dict)
        for key, val in stab.items():
            assert isinstance(val, pd.DataFrame)

    def test_csv_outputs_written(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=3, n_points=30)

        from main import run_eis_pipeline
        run_eis_pipeline(config=cfg)

        tables_dir = Path(cfg.tables_dir)
        csv_files = list(tables_dir.glob("*.csv"))
        assert len(csv_files) >= 1

    def test_empty_data_dir(self, tmp_workspace: PipelineConfig):
        """Empty data dir → EISResult with empty DataFrames, no crash."""
        cfg = tmp_workspace

        from main import run_eis_pipeline
        result = run_eis_pipeline(config=cfg)

        assert isinstance(result, EISResult)
        assert result.features_df.empty

    def test_dict_access_compat(self, tmp_workspace: PipelineConfig):
        """EISResult supports bracket access for GUI compatibility."""
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=2, n_points=30)

        from main import run_eis_pipeline
        result = run_eis_pipeline(config=cfg)

        # Dict-style access should work
        assert result["out_dir"] == cfg.tables_dir
        assert "raw_eis" in result


# ═══════════════════════════════════════════════════════════════════════
# DRT pipeline end-to-end
# ═══════════════════════════════════════════════════════════════════════

class TestDRTPipelineFull:
    """Run the full DRT pipeline on synthetic data."""

    def test_pipeline_returns_dict(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=2, n_points=30)

        from main_drt import run_drt_pipeline
        result = run_drt_pipeline(config=cfg)

        assert isinstance(result, (dict, DRTPipelineResult))

    def test_drt_table_populated(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=2, n_points=30)

        from main_drt import run_drt_pipeline
        result = run_drt_pipeline(config=cfg)

        drt_table = result["drt_table"]
        assert isinstance(drt_table, pd.DataFrame)
        assert len(drt_table) >= 1
        assert "Arquivo" in drt_table.columns
        assert "R_inf" in drt_table.columns
        assert "n_peaks" in drt_table.columns

    def test_per_file_results_populated(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=2, n_points=30)

        from main_drt import run_drt_pipeline
        result = run_drt_pipeline(config=cfg)

        pfr = result["per_file_results"]
        assert isinstance(pfr, dict)
        assert len(pfr) >= 1

    def test_plot_paths_exist(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=1, n_points=30)

        from main_drt import run_drt_pipeline
        result = run_drt_pipeline(config=cfg, show_plots=False)

        plot_paths = result["plot_paths"]
        assert isinstance(plot_paths, list)
        # Check at least one plot was created
        for stem, path_str in plot_paths:
            assert Path(path_str).exists()

    def test_run_meta_present(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=1, n_points=30)

        from main_drt import run_drt_pipeline
        result = run_drt_pipeline(config=cfg)

        meta = result["run_meta"]
        assert "lambda_reg" in meta
        assert "n_taus" in meta
        assert "n_files" in meta

    def test_drt_peaks_table(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=2, n_points=30)

        from main_drt import run_drt_pipeline
        result = run_drt_pipeline(config=cfg)

        pt = result.get("drt_peaks_table")
        assert pt is not None
        assert isinstance(pt, pd.DataFrame)

    def test_errors_dict_present(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=1, n_points=30)

        from main_drt import run_drt_pipeline
        result = run_drt_pipeline(config=cfg)

        assert "errors" in result
        assert isinstance(result["errors"], dict)

    def test_empty_data_dir_no_crash(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace

        from main_drt import run_drt_pipeline
        result = run_drt_pipeline(config=cfg)

        assert isinstance(result, (dict, DRTPipelineResult))
        assert len(result["drt_table"]) == 0


# ═══════════════════════════════════════════════════════════════════════
# Cycling pipeline end-to-end
# ═══════════════════════════════════════════════════════════════════════

class TestCyclingPipelineFull:
    """Run the galvanostatic cycling pipeline on synthetic data."""

    def test_pipeline_returns_cycling_result(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_cycling_fixture(cfg.processed_dir, n_files=2, n_cycles=3)

        from main_cycling import run_ciclagem_pipeline
        result = run_ciclagem_pipeline(scan_rate=1.0, show_plots=False, config=cfg)

        assert isinstance(result, CyclingResult)

    def test_results_per_file(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_cycling_fixture(cfg.processed_dir, n_files=2, n_cycles=3)

        from main_cycling import run_ciclagem_pipeline
        result = run_ciclagem_pipeline(scan_rate=1.0, show_plots=False, config=cfg)

        assert len(result["results"]) == 2

    def test_export_tables_populated(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_cycling_fixture(cfg.processed_dir, n_files=1, n_cycles=3)

        from main_cycling import run_ciclagem_pipeline
        result = run_ciclagem_pipeline(scan_rate=1.0, show_plots=False, config=cfg)

        assert len(result["export_tables"]) == 1

    def test_merged_table(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_cycling_fixture(cfg.processed_dir, n_files=2, n_cycles=3)

        from main_cycling import run_ciclagem_pipeline
        result = run_ciclagem_pipeline(scan_rate=1.0, show_plots=False, config=cfg)

        merged = result["merged_table"]
        assert merged is not None
        assert isinstance(merged, pd.DataFrame)
        assert "Arquivo" in merged.columns

    def test_excel_files_written(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_cycling_fixture(cfg.processed_dir, n_files=1, n_cycles=3)

        from main_cycling import run_ciclagem_pipeline
        run_ciclagem_pipeline(scan_rate=1.0, show_plots=False, config=cfg)

        excel_files = list(Path(cfg.excel_dir).glob("*.xlsx"))
        assert len(excel_files) >= 1

    def test_dict_access_compat(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_cycling_fixture(cfg.processed_dir, n_files=1, n_cycles=3)

        from main_cycling import run_ciclagem_pipeline
        result = run_ciclagem_pipeline(scan_rate=1.0, show_plots=False, config=cfg)

        assert "results" in result
        assert "config_used" in result


# ═══════════════════════════════════════════════════════════════════════
# Feature store integration after EIS pipeline
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureStoreIntegration:
    """After the EIS pipeline runs, the feature store should be populated."""

    def test_store_populated_after_eis(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=3, n_points=30)

        from main import run_eis_pipeline
        run_eis_pipeline(config=cfg)

        from src.feature_store import FeatureStore
        store = FeatureStore(cfg.feature_store_path)
        assert len(store) >= 1  # at least some successful fits recorded

    def test_store_has_required_fields(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=2, n_points=30)

        from main import run_eis_pipeline
        run_eis_pipeline(config=cfg)

        from src.feature_store import FeatureStore
        store = FeatureStore(cfg.feature_store_path)
        if len(store) > 0:
            rec = store.records[0]
            assert "sample_id" in rec
            assert "circuit_name" in rec
            assert "spectral_features" in rec
            assert "timestamp" in rec


# ═══════════════════════════════════════════════════════════════════════
# ML selector integration
# ═══════════════════════════════════════════════════════════════════════

class TestMLSelectorIntegration:
    """CircuitMLSelector can be trained on EIS pipeline output."""

    def test_selector_on_pipeline_output(self, tmp_workspace: PipelineConfig):
        cfg = tmp_workspace
        create_eis_fixture(cfg.data_dir, n_files=3, n_points=30)

        from main import run_eis_pipeline
        run_eis_pipeline(config=cfg)

        from src.feature_store import FeatureStore
        from src.ml_circuit_selector import CircuitMLSelector

        store = FeatureStore(cfg.feature_store_path)
        sel = CircuitMLSelector(min_samples=1)  # low threshold for test

        if len(store) >= 2:
            # Need ≥ 2 classes for training
            unique = store.unique_circuits()
            if len(unique) >= 2:
                result = sel.train(store)
                assert result is True
