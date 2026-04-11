"""Tests for src.eis_plots — smoke tests for all public plot functions.

Each test creates minimal synthetic data, calls the function in
non-interactive mode (show=False) and verifies that a file was saved
or that the function returns None for invalid input.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src.eis_plots import (  # noqa: E402
    plot_bode,
    plot_boxplot_metrics,
    plot_energy_cycle,
    plot_impedance_heatmap,
    plot_nyquist,
    plot_radar,
    plot_ragone,
)


# ---------------------------------------------------------------------------
#  Synthetic data helpers
# ---------------------------------------------------------------------------

def _eis_df(n: int = 30) -> pd.DataFrame:
    """Fake EIS DataFrame with frequency, zreal, zimag."""
    freq = np.logspace(0, 5, n)
    zr = 5.0 + 10.0 / (1 + (2 * np.pi * freq * 1e-3) ** 2)
    zi = -(10.0 * 2 * np.pi * freq * 1e-3 / (1 + (2 * np.pi * freq * 1e-3) ** 2))
    return pd.DataFrame({"frequency": freq, "zreal": zr, "zimag": zi})


def _cycling_tables(n_samples: int = 3, n_cycles: int = 10):
    """Dict of cycling DataFrames with Ciclos, Energia, Potência."""
    tables = {}
    for i in range(n_samples):
        cycles = list(range(1, n_cycles + 1))
        energy = [50.0 - 0.3 * c + np.random.default_rng(i).normal(0, 0.5) for c in cycles]
        power = [100.0 + 2 * c + np.random.default_rng(i + 10).normal(0, 1) for c in cycles]
        tables[f"sample_{i}"] = pd.DataFrame({
            "Ciclos": cycles,
            "Energia (Wh/kg)": energy,
            "Potência (W/kg)": power,
        })
    return tables


def _metrics_df(n: int = 8) -> pd.DataFrame:
    """Fake EIS metrics DataFrame with Arquivo and numeric columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Arquivo": [f"sample_{i}" for i in range(n)],
        "Rs": rng.uniform(0.5, 5, n),
        "Rp": rng.uniform(5, 50, n),
        "C_mean": rng.uniform(1e-6, 1e-4, n),
        "C_max": rng.uniform(1e-5, 1e-3, n),
        "Energy_mean": rng.uniform(1e-8, 1e-6, n),
        "Score": rng.uniform(-2, 2, n),
        "Subclass": ["Interface eficiente"] * 4 + ["Genérica estável"] * 4,
    })


# ---------------------------------------------------------------------------
#  Nyquist
# ---------------------------------------------------------------------------

class TestNyquist:
    def test_saves_png(self, tmp_path: Path):
        df = _eis_df()
        out = plot_nyquist(df, "test_sample", out_dir=str(tmp_path), show=False, save=True)
        assert out is not None
        assert Path(out).exists()

    def test_returns_none_missing_cols(self, tmp_path: Path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert plot_nyquist(df, "bad", out_dir=str(tmp_path), show=False) is None


# ---------------------------------------------------------------------------
#  Bode
# ---------------------------------------------------------------------------

class TestBode:
    def test_saves_png(self, tmp_path: Path):
        df = _eis_df()
        out = plot_bode(df, "test_sample", out_dir=str(tmp_path), show=False, save=True)
        assert out is not None
        assert Path(out).exists()

    def test_returns_none_missing_cols(self, tmp_path: Path):
        df = pd.DataFrame({"x": [1]})
        assert plot_bode(df, "bad", out_dir=str(tmp_path), show=False) is None


# ---------------------------------------------------------------------------
#  Ragone
# ---------------------------------------------------------------------------

class TestRagone:
    def test_saves_png(self, tmp_path: Path):
        tables = _cycling_tables()
        out = plot_ragone(tables, out_dir=str(tmp_path), show=False, save=True)
        assert out is not None
        assert Path(out).exists()

    def test_empty_tables(self, tmp_path: Path):
        assert plot_ragone({}, out_dir=str(tmp_path), show=False) is None


# ---------------------------------------------------------------------------
#  Energy vs Cycle
# ---------------------------------------------------------------------------

class TestEnergyCycle:
    def test_saves_png(self, tmp_path: Path):
        tables = _cycling_tables()
        out = plot_energy_cycle(tables, out_dir=str(tmp_path), show=False, save=True)
        assert out is not None
        assert Path(out).exists()


# ---------------------------------------------------------------------------
#  Impedance Heatmap
# ---------------------------------------------------------------------------

class TestImpedanceHeatmap:
    def test_saves_png(self, tmp_path: Path):
        raw = {f"s{i}": _eis_df(20) for i in range(3)}
        out = plot_impedance_heatmap(raw, out_dir=str(tmp_path), show=False, save=True)
        assert out is not None
        assert Path(out).exists()

    def test_empty_dict(self, tmp_path: Path):
        assert plot_impedance_heatmap({}, out_dir=str(tmp_path), show=False) is None


# ---------------------------------------------------------------------------
#  Box-plot
# ---------------------------------------------------------------------------

class TestBoxplot:
    def test_saves_png(self, tmp_path: Path):
        df = _metrics_df()
        out = plot_boxplot_metrics(df, metric="Rs", out_dir=str(tmp_path), show=False, save=True)
        assert out is not None
        assert Path(out).exists()

    def test_missing_metric(self, tmp_path: Path):
        df = _metrics_df()
        assert plot_boxplot_metrics(df, metric="NonExistent", out_dir=str(tmp_path), show=False) is None


# ---------------------------------------------------------------------------
#  Radar
# ---------------------------------------------------------------------------

class TestRadar:
    def test_saves_png(self, tmp_path: Path):
        df = _metrics_df()
        samples = ["sample_0", "sample_1", "sample_2"]
        out = plot_radar(df, samples, out_dir=str(tmp_path), show=False, save=True)
        assert out is not None
        assert Path(out).exists()

    def test_too_few_metrics(self, tmp_path: Path):
        df = pd.DataFrame({"Arquivo": ["a", "b"], "Rs": [1.0, 2.0]})
        assert plot_radar(df, ["a", "b"], metrics=["Rs"], out_dir=str(tmp_path), show=False) is None
