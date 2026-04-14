"""Tests for the decomposed EIS pipeline stages in main.py."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.models import EISResult, PCAResult
from main import (
    build_features_df,
    classify_and_rank,
    compute_stability,
    compute_pca_stage,
    build_cap_energy,
    build_circuit_tables,
)


# ── Fixtures ────────────────────────────────────────────────────────────

def _make_features_records(n: int = 5) -> dict:
    """Create mock per-file feature records as returned by load_and_extract."""
    rng = np.random.RandomState(42)
    records = {}
    for i in range(n):
        fname = f"1 H2SO4 0.1A GCT sample{i}.txt"
        records[fname] = {
            "Rs": rng.uniform(0.1, 2.0),
            "Rp": rng.uniform(5.0, 50.0),
            "C_mean": rng.uniform(1e-6, 1e-3),
            "C_max": rng.uniform(1e-5, 1e-2),
            "C_lowfreq": rng.uniform(1e-6, 1e-3),
            "Energy_mean": rng.uniform(1e-8, 1e-5),
            "Tau": rng.uniform(1e-4, 1.0),
            "Dispersion": rng.uniform(0.0, 2.0),
            "Rs_fit": rng.uniform(0.1, 2.0),
            "Rp_fit": rng.uniform(5.0, 50.0),
            "Q": rng.uniform(1e-5, 1e-2),
            "n": rng.uniform(0.7, 1.0),
            "Sigma": rng.uniform(1.0, 100.0),
        }
    return records


@pytest.fixture
def features_records():
    return _make_features_records()


@pytest.fixture
def features_df(features_records):
    return build_features_df(features_records)


@pytest.fixture
def ranked_df(features_df):
    return classify_and_rank(features_df)


# ── build_features_df ──────────────────────────────────────────────────

class TestBuildFeaturesDf:
    def test_returns_dataframe(self, features_records):
        df = build_features_df(features_records)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_adds_metadata_columns(self, features_df):
        assert "Electrolyte" in features_df.columns
        assert "Current" in features_df.columns
        assert "Treatment" in features_df.columns
        assert "Sample" in features_df.columns

    def test_empty_records(self):
        df = build_features_df({})
        assert df.empty


# ── classify_and_rank ──────────────────────────────────────────────────

class TestClassifyAndRank:
    def test_adds_subclass_and_rank(self, ranked_df):
        assert "Subclass" in ranked_df.columns
        assert "Score" in ranked_df.columns
        assert "Rank" in ranked_df.columns

    def test_subclass_values_are_known(self, ranked_df):
        known = {"Interface eficiente", "Genérica estável", "Indefinida (dados insuficientes)"}
        for v in ranked_df["Subclass"].unique():
            assert v in known


# ── compute_stability ──────────────────────────────────────────────────

class TestComputeStability:
    def test_returns_dict(self, ranked_df, tmp_path):
        cfg = PipelineConfig(stability_columns=["Rs_fit", "Rp_fit"])
        stab = compute_stability(ranked_df, cfg, str(tmp_path))
        assert isinstance(stab, dict)
        assert "Rs_fit" in stab or "Rp_fit" in stab

    def test_saves_csv(self, ranked_df, tmp_path):
        cfg = PipelineConfig(stability_columns=["Rs_fit"])
        compute_stability(ranked_df, cfg, str(tmp_path))
        assert (tmp_path / "stability_Rs_fit.csv").exists()


# ── compute_pca_stage ─────────────────────────────────────────────────

class TestComputePcaStage:
    def test_returns_pca_result(self, ranked_df, tmp_path):
        cfg = PipelineConfig.default()
        pca = compute_pca_stage(ranked_df, cfg, str(tmp_path))
        assert isinstance(pca, PCAResult)

    def test_pca_with_sufficient_data(self, ranked_df, tmp_path):
        cfg = PipelineConfig.default()
        pca = compute_pca_stage(ranked_df, cfg, str(tmp_path))
        # 5 samples with all columns present → PCA should succeed
        assert pca.df_pca is not None
        assert pca.loadings is not None

    def test_pca_skipped_with_few_rows(self, tmp_path):
        df = pd.DataFrame({
            "Rs_fit": [1.0], "Rp_fit": [10.0],
            "Q": [1e-4], "n": [0.9], "Sigma": [50.0],
            "Subclass": ["test"],
        })
        cfg = PipelineConfig.default()
        pca = compute_pca_stage(df, cfg, str(tmp_path))
        assert pca.df_pca is None


# ── build_cap_energy ──────────────────────────────────────────────────

class TestBuildCapEnergy:
    def test_returns_dataframe_with_retention(self, ranked_df):
        cap = build_cap_energy(ranked_df)
        assert isinstance(cap, pd.DataFrame)
        assert "Retenção (%)" in cap.columns
        assert "C média (F)" in cap.columns
        assert "C_espec (F/g)" in cap.columns

    def test_column_rename(self, ranked_df):
        cap = build_cap_energy(ranked_df)
        # Old column names should not be present
        assert "C_mean" not in cap.columns
        assert "Energy_mean" not in cap.columns


# ── build_circuit_tables ──────────────────────────────────────────────

class TestBuildCircuitTables:
    def test_empty_rows(self, tmp_path):
        table, summary = build_circuit_tables([], str(tmp_path))
        assert table is None
        assert summary is None

    def test_with_rows(self, tmp_path):
        rows = [
            {
                "Arquivo": "a.txt",
                "Circuito": "Randles-CPE-W",
                "Representacao": "Rs - (Rp || CPE) - W",
                "BIC": 10.0,
                "BIC_penalizado": 10.5,
                "AIC": 8.0,
                "RSS": 0.01,
                "Confianca": 0.8,
                "Sucesso": True,
                "Res_estruturado": False,
                "Bound_hits": 0,
            },
        ]
        table, summary = build_circuit_tables(rows, str(tmp_path))
        assert table is not None
        assert len(table) == 1
        assert (tmp_path / "circuit_fits.csv").exists()


# ── run_eis_pipeline (integration, empty data) ─────────────────────────

class TestRunEisPipelineEmpty:
    def test_empty_dir_returns_eis_result(self, tmp_path):
        from main import run_eis_pipeline

        data_dir = tmp_path / "raw"
        data_dir.mkdir()
        cfg = PipelineConfig(
            data_dir=str(data_dir),
            tables_dir=str(tmp_path / "tables"),
            reports_dir=str(tmp_path / "reports"),
            circuits_fig_dir=str(tmp_path / "circ"),
        )
        result = run_eis_pipeline(config=cfg)
        assert isinstance(result, EISResult)
        assert result.features_df.empty
        # Legacy dict access still works
        assert result["df"].empty
