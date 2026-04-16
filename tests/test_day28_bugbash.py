"""Day 28 — Bug bash & comprehensive edge-case tests.

Expands coverage for under-tested modules:
- preprocessing (edge cases: NaN, zero freq, empty, string cols)
- loader (column detection, separators, encoding)
- stability (edge cases, single sample, NaN)
- metadata (full metadata, new extractors already in test_production_viz)
- visualization (correlation_heatmap, boxplot_param, series_by_prefix)
- eis_plots (new Ragone target, reference zones, gap analysis extras)
- cycling_calculator (zero mass, single cycle, edge cases)
- cross-module integration stress tests
"""

from __future__ import annotations

import os
import textwrap

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import preprocess
from src.stability import extract_sample_id, stability_metrics
from src.metadata import (
    extract_full_metadata,
    extract_material_type,
    extract_metadata,
    extract_synthesis_process,
)
from src.cycling_calculator import calculate_energy_power, calculate_mass, identify_cycles
from src.eis_plots import (
    RagoneGapResult,
    plot_bode,
    plot_impedance_heatmap,
    plot_nyquist,
    plot_ragone,
    ragone_gap_analysis,
)
from src.visualization import (
    correlation_heatmap,
    production_heatmap,
    series_by_prefix,
)


# ====================================================================
# preprocessing.py — expanded
# ====================================================================

class TestPreprocessEdgeCases:
    def test_nan_rows_dropped(self):
        df = pd.DataFrame({
            "frequency": [100, np.nan, 10],
            "zreal": [5, 3, 1],
            "zimag": [-0.5, -0.3, -0.1],
        })
        out = preprocess(df)
        assert len(out) == 2

    def test_zero_frequency_removed(self):
        df = pd.DataFrame({
            "frequency": [0, 10, 100],
            "zreal": [5, 3, 1],
            "zimag": [-0.5, -0.3, -0.1],
        })
        out = preprocess(df)
        assert (out["frequency"] > 0).all()
        assert len(out) == 2

    def test_negative_frequency_removed(self):
        df = pd.DataFrame({
            "frequency": [-5, 10, 100],
            "zreal": [5, 3, 1],
            "zimag": [-0.5, -0.3, -0.1],
        })
        out = preprocess(df)
        assert len(out) == 2

    def test_sorted_descending(self):
        df = pd.DataFrame({
            "frequency": [1, 1000, 100, 10],
            "zreal": [10, 1, 3, 5],
            "zimag": [-1, -0.01, -0.1, -0.5],
        })
        out = preprocess(df)
        assert list(out["frequency"]) == [1000, 100, 10, 1]

    def test_omega_computed(self):
        df = pd.DataFrame({
            "frequency": [1.0],
            "zreal": [10.0],
            "zimag": [-1.0],
        })
        out = preprocess(df)
        assert abs(out["omega"].iloc[0] - 2 * np.pi) < 1e-10

    def test_string_frequency_coerced(self):
        df = pd.DataFrame({
            "frequency": ["100", "abc", "10"],
            "zreal": [5, 3, 1],
            "zimag": [-0.5, -0.3, -0.1],
        })
        out = preprocess(df)
        # "abc" → NaN → dropped
        assert len(out) == 2

    def test_empty_after_filtering(self):
        df = pd.DataFrame({
            "frequency": [0, -1],
            "zreal": [1, 2],
            "zimag": [-0.1, -0.2],
        })
        out = preprocess(df)
        assert len(out) == 0

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="Colunas faltando"):
            preprocess(pd.DataFrame({"frequency": [1], "zreal": [1]}))

    def test_extra_columns_preserved(self):
        df = pd.DataFrame({
            "frequency": [100, 10],
            "zreal": [5, 1],
            "zimag": [-0.5, -0.1],
            "extra": ["a", "b"],
        })
        out = preprocess(df)
        assert "extra" in out.columns

    def test_single_row(self):
        df = pd.DataFrame({
            "frequency": [50.0],
            "zreal": [10.0],
            "zimag": [-1.0],
        })
        out = preprocess(df)
        assert len(out) == 1


# ====================================================================
# stability.py — expanded
# ====================================================================

class TestStabilityExpanded:
    def test_extract_sample_id_pattern(self):
        assert extract_sample_id("1 Nb2 GCT H2SO4.txt") == "1Nb2"

    def test_extract_sample_id_complex(self):
        assert extract_sample_id("10 Nb4 Prisca Na2SO4.txt") == "10Nb4"

    def test_extract_sample_id_fallback(self):
        assert extract_sample_id("random_file.txt") == "random_file"

    def test_extract_sample_id_no_extension(self):
        assert extract_sample_id("no_extension") == "no_extension"

    def test_stability_metrics_basic(self):
        df = pd.DataFrame({
            "Sample": ["A", "A", "A"],
            "Rs_fit": [10.0, 12.0, 11.0],
        })
        result = stability_metrics(df, "Rs_fit")
        assert abs(result.loc["A", "Mean"] - 11.0) < 1e-10

    def test_stability_single_sample(self):
        df = pd.DataFrame({"Sample": ["X"], "val": [5.0]})
        result = stability_metrics(df, "val")
        assert "CV" in result.columns

    def test_stability_multiple_groups(self):
        df = pd.DataFrame({
            "Sample": ["A", "A", "B", "B"],
            "val": [10, 12, 100, 100],
        })
        result = stability_metrics(df, "val")
        assert len(result) == 2
        assert result.loc["B", "Std"] == 0

    def test_stability_zero_mean_safe(self):
        df = pd.DataFrame({"Sample": ["A", "A"], "val": [0.0, 0.0]})
        result = stability_metrics(df, "val")
        # CV should be NaN (0/0), not inf
        assert pd.isna(result.loc["A", "CV"])


# ====================================================================
# metadata.py — new extractors expanded
# ====================================================================

class TestMetadataExpanded:
    def test_full_metadata_all_fields(self):
        m = extract_full_metadata("1 Nb2 Prisca H2SO4 0.1A GCT")
        assert m["Material_Type"] == "Nb2"
        assert m["Synthesis"] == "Prisca"
        assert m["Electrolyte"] == "H2SO4"
        assert m["Current"] == "0.1A"
        assert m["Treatment"] == "GCT"

    def test_material_nb4_case(self):
        assert extract_material_type("3 NB4 H2SO4") == "Nb4"

    def test_synthesis_gcd(self):
        assert extract_synthesis_process("2 Nb2 GCD H2SO4") == "GCD"

    def test_unknown_everything(self):
        m = extract_full_metadata("random_file.txt")
        assert m["Material_Type"] == "Unknown"
        assert m["Synthesis"] == "Standard"
        assert m["Electrolyte"] == "Unknown"

    def test_treatment_steel316(self):
        _, _, treatment = extract_metadata("1 Nb2 S316 H2SO4")
        assert treatment == "Steel316"


# ====================================================================
# cycling_calculator.py — expanded
# ====================================================================

class TestCyclingCalculatorExpanded:
    def _make_cycling_df(self, n_cycles=3, points_per_cycle=20):
        rows = []
        t = 0.0
        for c in range(1, n_cycles + 1):
            for i in range(points_per_cycle):
                phase = 2 * np.pi * i / points_per_cycle
                rows.append({
                    "tempo": t,
                    "potencial": 0.5 * np.sin(phase),
                    "corrente": 0.001,
                    "ciclo": c,
                })
                t += 0.1
        return pd.DataFrame(rows)

    def test_calculate_mass(self):
        assert calculate_mass(0.001, 1.0) == pytest.approx(0.001)

    def test_calculate_mass_negative_current(self):
        assert calculate_mass(-0.005, 1.0) == pytest.approx(0.005)

    def test_identify_cycles_from_column(self):
        df = self._make_cycling_df()
        cycles = identify_cycles(df)
        assert set(cycles.unique()) == {1, 2, 3}

    def test_energy_power_basic(self):
        df = self._make_cycling_df()
        result = calculate_energy_power(df, scan_rate=1.0)
        assert len(result) == 3
        assert "energia_wh_kg" in result.columns
        assert "potencia_w_kg" in result.columns
        assert (result["energia_wh_kg"] >= 0).all()

    def test_single_cycle(self):
        df = self._make_cycling_df(n_cycles=1)
        result = calculate_energy_power(df, scan_rate=1.0)
        assert len(result) == 1

    def test_zero_duration_cycle(self):
        df = pd.DataFrame({
            "tempo": [0.0],
            "potencial": [0.5],
            "corrente": [0.001],
            "ciclo": [1],
        })
        result = calculate_energy_power(df, scan_rate=1.0)
        # Should handle gracefully (0 or NaN)
        assert len(result) <= 1


# ====================================================================
# eis_plots — edge cases for Ragone and other plots
# ====================================================================

class TestEISPlotsEdgeCases:
    def test_nyquist_missing_columns(self, tmp_path):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert plot_nyquist(df, "test", out_dir=str(tmp_path), show=False) is None

    def test_bode_missing_columns(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        assert plot_bode(df, "test", out_dir=str(tmp_path), show=False) is None

    def test_nyquist_basic(self, tmp_path):
        df = pd.DataFrame({
            "frequency": [100, 10, 1],
            "zreal": [10, 20, 30],
            "zimag": [-5, -10, -15],
        })
        path = plot_nyquist(df, "sample", out_dir=str(tmp_path), show=False, save=True)
        assert path is not None and os.path.exists(path)

    def test_bode_basic(self, tmp_path):
        df = pd.DataFrame({
            "frequency": [100, 10, 1],
            "zreal": [10, 20, 30],
            "zimag": [-5, -10, -15],
        })
        path = plot_bode(df, "sample", out_dir=str(tmp_path), show=False, save=True)
        assert path is not None and os.path.exists(path)

    def test_impedance_heatmap_empty(self, tmp_path):
        assert plot_impedance_heatmap({}, out_dir=str(tmp_path), show=False) is None

    def test_impedance_heatmap_basic(self, tmp_path):
        raw = {
            "s1": pd.DataFrame({
                "frequency": [100, 10, 1],
                "zreal": [10, 20, 30],
                "zimag": [-5, -10, -15],
            }),
            "s2": pd.DataFrame({
                "frequency": [100, 10, 1],
                "zreal": [8, 18, 28],
                "zimag": [-4, -9, -14],
            }),
        }
        path = plot_impedance_heatmap(raw, n_bands=5, out_dir=str(tmp_path), show=False, save=True)
        assert path is not None

    def test_ragone_with_highlight(self, tmp_path):
        tables = {
            "sample_a": pd.DataFrame({
                "Energia (Wh/kg)": [10, 12],
                "Potência (W/kg)": [500, 520],
            }),
        }
        path = plot_ragone(
            tables,
            highlight_sample="sample_a",
            out_dir=str(tmp_path),
            show=False,
            save=True,
        )
        assert path is not None

    def test_ragone_with_target_and_zones(self, tmp_path):
        tables = {
            "s1": pd.DataFrame({
                "Energia (Wh/kg)": [10],
                "Potência (W/kg)": [500],
            }),
        }
        path = plot_ragone(
            tables,
            target_energy=300.0,
            target_power=3000.0,
            out_dir=str(tmp_path),
            show=False,
            save=True,
        )
        assert path is not None and os.path.exists(path)


# ====================================================================
# visualization.py — correlation_heatmap + series_by_prefix
# ====================================================================

class TestVisualizationExpanded:
    def test_correlation_heatmap_basic(self, tmp_path):
        df = pd.DataFrame({
            "Rs_fit": np.random.randn(20),
            "Rp_fit": np.random.randn(20),
            "Score": np.random.randn(20),
        })
        path = correlation_heatmap(df, ["Rs_fit", "Rp_fit", "Score"], out_dir=str(tmp_path))
        assert path is not None
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(str(tmp_path), "correlation_matrix.csv"))

    def test_correlation_heatmap_too_few_cols(self, tmp_path):
        df = pd.DataFrame({"Rs_fit": [1, 2, 3]})
        assert correlation_heatmap(df, ["Rs_fit"], out_dir=str(tmp_path)) is None

    def test_correlation_heatmap_missing_cols(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        assert correlation_heatmap(df, ["nonexistent1", "nonexistent2"], out_dir=str(tmp_path)) is None

    def test_correlation_heatmap_with_nan(self, tmp_path):
        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5],
            "b": [5, np.nan, 3, 2, 1],
        })
        path = correlation_heatmap(df, ["a", "b"], out_dir=str(tmp_path))
        assert path is not None

    def test_series_by_prefix_basic(self, tmp_path):
        df = pd.DataFrame(
            {"Score": [10, 20, 30]},
            index=["1 Nb2 H2SO4", "2 Nb2 H2SO4", "3 Nb2 H2SO4"],
        )
        paths = series_by_prefix(df, "Score", out_dir=str(tmp_path))
        assert paths is not None and len(paths) > 0

    def test_series_by_prefix_missing_col(self, tmp_path):
        df = pd.DataFrame({"x": [1]}, index=["1 A"])
        assert series_by_prefix(df, "nonexistent", out_dir=str(tmp_path)) is None


# ====================================================================
# Gap analysis — additional edge cases
# ====================================================================

class TestGapAnalysisExpanded:
    def test_single_sample(self):
        tables = {
            "s1": pd.DataFrame({
                "Energia (Wh/kg)": [50],
                "Potência (W/kg)": [1000],
            }),
        }
        result = ragone_gap_analysis(tables, target_energy=300, target_power=3000)
        assert result is not None
        assert result.best_sample == "s1"
        assert result.energy_gap > 0
        assert result.power_gap > 0

    def test_all_zero_energy(self):
        tables = {
            "s1": pd.DataFrame({
                "Energia (Wh/kg)": [0, 0],
                "Potência (W/kg)": [0, 0],
            }),
        }
        result = ragone_gap_analysis(tables)
        assert result is None  # no valid data

    def test_mixed_valid_invalid(self):
        tables = {
            "bad": pd.DataFrame({
                "Energia (Wh/kg)": [0, -1],
                "Potência (W/kg)": [0, -1],
            }),
            "good": pd.DataFrame({
                "Energia (Wh/kg)": [50, 60],
                "Potência (W/kg)": [500, 600],
            }),
        }
        result = ragone_gap_analysis(tables)
        assert result is not None
        assert result.best_sample == "good"
        assert len(result.sample_medians) == 1  # only good has valid data

    def test_gap_result_fields(self):
        tables = {
            "s1": pd.DataFrame({
                "Energia (Wh/kg)": [30],
                "Potência (W/kg)": [300],
            }),
        }
        r = ragone_gap_analysis(tables, target_energy=300, target_power=3000)
        assert r.target_energy == 300
        assert r.target_power == 3000
        assert r.energy_factor == pytest.approx(10.0)
        assert r.power_factor == pytest.approx(10.0)
        assert r.energy_error_pct == pytest.approx(90.0)
        assert r.power_error_pct == pytest.approx(90.0)


# ====================================================================
# loader.py — file-based tests
# ====================================================================

class TestLoaderEdgeCases:
    def test_semicolon_separated(self, tmp_path):
        from src.loader import load_eis_file
        f = tmp_path / "test.csv"
        f.write_text("freq;z';z''\n100;10;5\n10;20;10\n1;30;15\n")
        df = load_eis_file(str(f))
        assert len(df) == 3
        assert "frequency" in df.columns

    def test_comma_decimal(self, tmp_path):
        from src.loader import load_eis_file
        f = tmp_path / "test.csv"
        f.write_text("freq;z';z''\n100,5;10,2;5,3\n10,1;20,4;10,6\n")
        df = load_eis_file(str(f))
        assert len(df) == 2
        assert df["frequency"].iloc[0] > 10  # 100.5 or 10.1

    def test_tab_separated(self, tmp_path):
        from src.loader import load_eis_file
        f = tmp_path / "test.txt"
        f.write_text("freq\tz'\tz''\n100\t10\t5\n10\t20\t10\n")
        df = load_eis_file(str(f))
        assert len(df) == 2

    def test_empty_file_raises(self, tmp_path):
        from src.loader import load_eis_file
        f = tmp_path / "empty.csv"
        f.write_text("")
        with pytest.raises((ValueError, Exception)):
            load_eis_file(str(f))

    def test_single_column_raises(self, tmp_path):
        from src.loader import load_eis_file
        f = tmp_path / "bad.csv"
        f.write_text("a\n1\n2\n")
        with pytest.raises(ValueError):
            load_eis_file(str(f))

    def test_all_nan_data_raises(self, tmp_path):
        from src.loader import load_eis_file
        f = tmp_path / "nan.csv"
        f.write_text("freq;z';z''\nabc;def;ghi\nxyz;foo;bar\n")
        with pytest.raises(ValueError):
            load_eis_file(str(f))

    def test_positional_fallback(self, tmp_path):
        from src.loader import load_eis_file
        f = tmp_path / "test.csv"
        f.write_text("col1;col2;col3\n100;10;5\n10;20;10\n")
        df = load_eis_file(str(f))
        assert len(df) == 2


# ====================================================================
# Cross-module integration stress
# ====================================================================

class TestCrossModuleStress:
    def test_preprocess_then_features(self):
        """Pipeline: preprocess → extract_features works end-to-end."""
        from src.physics_metrics import extract_features
        df = pd.DataFrame({
            "frequency": [10000, 1000, 100, 10, 1, 0.1],
            "zreal": [5, 10, 20, 50, 100, 200],
            "zimag": [-1, -5, -20, -30, -20, -10],
        })
        out = preprocess(df)
        features = extract_features(out)
        assert isinstance(features, dict)
        assert "Rs" in features or "Z_real_min" in features or len(features) > 0

    def test_metadata_from_sample_names(self):
        """All metadata extractors work together."""
        names = [
            "1 Nb2 H2SO4 0.1A GCT",
            "2 Nb4 Prisca Na2SO4 1A",
            "3 NF Alcohol H2SO4",
        ]
        for name in names:
            m = extract_full_metadata(name)
            assert m["Material_Type"] != ""
            assert m["Synthesis"] != ""
            assert m["Electrolyte"] != ""

    def test_production_heatmap_with_real_cols(self, tmp_path):
        """Production heatmap works with typical pipeline output columns."""
        df = pd.DataFrame({
            "Material_Type": ["Nb2"] * 5 + ["Nb4"] * 5,
            "Synthesis": ["Standard"] * 3 + ["Prisca"] * 4 + ["Standard"] * 3,
            "Rs_fit": np.random.uniform(1, 10, 10),
            "Rp_fit": np.random.uniform(50, 300, 10),
            "n": np.random.uniform(0.5, 1.0, 10),
            "C_espec (F/g)": np.random.uniform(50, 200, 10),
            "Score": np.random.uniform(40, 100, 10),
            "Retenção (%)": np.random.uniform(60, 100, 10),
        })
        path = production_heatmap(
            df,
            ["Rs_fit", "Rp_fit", "n", "C_espec (F/g)", "Score", "Retenção (%)"],
            out_dir=str(tmp_path),
        )
        assert path is not None
