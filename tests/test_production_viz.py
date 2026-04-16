"""Tests for production heatmap, enhanced Ragone, metadata extraction, and gap analysis."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.metadata import (
    extract_full_metadata,
    extract_material_type,
    extract_metadata,
    extract_synthesis_process,
)
from src.visualization import production_heatmap
from src.eis_plots import plot_ragone, ragone_gap_analysis, RagoneGapResult


# ====================================================================
# Metadata extraction
# ====================================================================

class TestExtractMaterialType:
    def test_nb2(self):
        assert extract_material_type("1 Nb2 H2SO4 0.1A") == "Nb2"

    def test_nb4(self):
        assert extract_material_type("3 Nb4 Na2SO4 1A GCT") == "Nb4"

    def test_nf(self):
        assert extract_material_type("5 NF H2SO4 0.1A") == "NF"

    def test_unknown(self):
        assert extract_material_type("random_file.txt") == "Unknown"

    def test_case_insensitive(self):
        assert extract_material_type("2 nb2 H2SO4") == "Nb2"


class TestExtractSynthesisProcess:
    def test_prisca(self):
        assert extract_synthesis_process("1 Nb2 Prisca H2SO4") == "Prisca"

    def test_standard(self):
        assert extract_synthesis_process("1 Nb2 H2SO4 0.1A") == "Standard"

    def test_alcohol(self):
        assert extract_synthesis_process("2 Nb4 alcool Na2SO4") == "Alcohol"

    def test_gcd(self):
        assert extract_synthesis_process("3 Nb2 GCD H2SO4") == "GCD"

    def test_case_insensitive(self):
        assert extract_synthesis_process("1 Nb2 prisca H2SO4") == "Prisca"


class TestExtractFullMetadata:
    def test_complete(self):
        m = extract_full_metadata("1 Nb2 H2SO4 0.1A GCT")
        assert m["Material_Type"] == "Nb2"
        assert m["Synthesis"] == "Standard"
        assert m["Electrolyte"] == "H2SO4"
        assert m["Current"] == "0.1A"
        assert m["Treatment"] == "GCT"

    def test_prisca_nb4(self):
        m = extract_full_metadata("3 Nb4 Prisca Na2SO4 1A")
        assert m["Material_Type"] == "Nb4"
        assert m["Synthesis"] == "Prisca"


# ====================================================================
# Production heatmap
# ====================================================================

@pytest.fixture()
def sample_df():
    """DataFrame with production variables and metrics."""
    return pd.DataFrame({
        "Material_Type": ["Nb2", "Nb2", "Nb4", "Nb4", "Nb2", "Nb4"],
        "Synthesis": ["Standard", "Standard", "Prisca", "Prisca", "Prisca", "Standard"],
        "Rs_fit": [2.5, 3.0, 5.0, 4.5, 2.8, 6.0],
        "Rp_fit": [100, 120, 200, 180, 110, 250],
        "Score": [85, 80, 60, 65, 82, 55],
        "C_espec (F/g)": [150, 140, 90, 95, 145, 80],
    })


class TestProductionHeatmap:
    def test_basic(self, sample_df, tmp_path):
        path = production_heatmap(
            sample_df,
            ["Rs_fit", "Rp_fit", "Score", "C_espec (F/g)"],
            out_dir=str(tmp_path),
        )
        assert path is not None
        assert os.path.exists(path)

    def test_csv_exported(self, sample_df, tmp_path):
        production_heatmap(
            sample_df,
            ["Rs_fit", "Score"],
            out_dir=str(tmp_path),
        )
        assert os.path.exists(os.path.join(str(tmp_path), "production_means.csv"))

    def test_no_group_cols(self, tmp_path):
        df = pd.DataFrame({"Rs_fit": [1, 2, 3]})
        assert production_heatmap(df, ["Rs_fit"], out_dir=str(tmp_path)) is None

    def test_no_metrics(self, sample_df, tmp_path):
        assert production_heatmap(sample_df, ["nonexistent"], out_dir=str(tmp_path)) is None

    def test_single_group_returns_none(self, tmp_path):
        df = pd.DataFrame({
            "Material_Type": ["Nb2", "Nb2"],
            "Synthesis": ["Standard", "Standard"],
            "Rs_fit": [1, 2],
        })
        assert production_heatmap(df, ["Rs_fit"], out_dir=str(tmp_path)) is None

    def test_primary_only(self, tmp_path):
        df = pd.DataFrame({
            "Material_Type": ["Nb2", "Nb2", "Nb4", "Nb4"],
            "Rs_fit": [1, 2, 3, 4],
        })
        path = production_heatmap(
            df, ["Rs_fit"],
            group_col="Material_Type",
            secondary_group="nonexistent",
            out_dir=str(tmp_path),
        )
        assert path is not None


# ====================================================================
# Ragone plot with target
# ====================================================================

@pytest.fixture()
def cycling_tables():
    return {
        "sample_a": pd.DataFrame({
            "Energia (Wh/kg)": [10, 12, 11],
            "Potência (W/kg)": [500, 520, 510],
        }),
        "sample_b": pd.DataFrame({
            "Energia (Wh/kg)": [20, 22, 21],
            "Potência (W/kg)": [800, 850, 820],
        }),
    }


class TestRagoneWithTarget:
    def test_no_target(self, cycling_tables, tmp_path):
        """Existing API still works without target."""
        path = plot_ragone(cycling_tables, out_dir=str(tmp_path), show=False, save=True)
        assert path is not None

    def test_with_target(self, cycling_tables, tmp_path):
        path = plot_ragone(
            cycling_tables,
            target_energy=300.0,
            target_power=3000.0,
            out_dir=str(tmp_path),
            show=False,
            save=True,
        )
        assert path is not None
        assert os.path.exists(path)

    def test_empty_tables(self, tmp_path):
        assert plot_ragone({}, out_dir=str(tmp_path), show=False) is None


# ====================================================================
# Gap analysis
# ====================================================================

class TestRagoneGapAnalysis:
    def test_basic(self, cycling_tables):
        result = ragone_gap_analysis(cycling_tables, target_energy=300, target_power=3000)
        assert result is not None
        assert isinstance(result, RagoneGapResult)
        assert result.best_sample in ("sample_a", "sample_b")
        assert result.energy_error_pct > 0
        assert result.power_error_pct > 0
        assert result.energy_factor > 1
        assert result.power_factor > 1
        assert len(result.recommendations) >= 2

    def test_target_achieved(self):
        tables = {
            "good": pd.DataFrame({
                "Energia (Wh/kg)": [300, 310, 295],
                "Potência (W/kg)": [3000, 3100, 2950],
            }),
        }
        result = ragone_gap_analysis(tables, target_energy=300, target_power=3000)
        assert result is not None
        assert result.energy_error_pct < 5
        assert result.power_error_pct < 5

    def test_empty_tables(self):
        assert ragone_gap_analysis({}) is None

    def test_no_valid_columns(self):
        tables = {"x": pd.DataFrame({"a": [1]})}
        assert ragone_gap_analysis(tables) is None

    def test_recommendations_content(self, cycling_tables):
        result = ragone_gap_analysis(cycling_tables, target_energy=300, target_power=3000)
        recs_text = " ".join(result.recommendations)
        assert "improvement" in recs_text.lower() or "target" in recs_text.lower()

    def test_snake_case_columns(self):
        tables = {
            "s1": pd.DataFrame({
                "energia_wh_kg": [50, 55],
                "potencia_w_kg": [1000, 1100],
            }),
        }
        result = ragone_gap_analysis(tables, target_energy=300, target_power=3000)
        assert result is not None
        assert result.best_sample == "s1"

    def test_sample_medians(self, cycling_tables):
        result = ragone_gap_analysis(cycling_tables)
        assert "sample_a" in result.sample_medians
        assert "sample_b" in result.sample_medians
        e, p = result.sample_medians["sample_b"]
        assert 20 <= e <= 22
        assert 800 <= p <= 850

    def test_energy_power_factors(self, cycling_tables):
        result = ragone_gap_analysis(cycling_tables, target_energy=100, target_power=2000)
        assert result.energy_factor >= 1
        assert result.power_factor >= 1

    def test_best_is_closest_in_log_space(self):
        tables = {
            "far": pd.DataFrame({
                "Energia (Wh/kg)": [1],
                "Potência (W/kg)": [10],
            }),
            "close": pd.DataFrame({
                "Energia (Wh/kg)": [100],
                "Potência (W/kg)": [1000],
            }),
        }
        result = ragone_gap_analysis(tables, target_energy=300, target_power=3000)
        assert result.best_sample == "close"
