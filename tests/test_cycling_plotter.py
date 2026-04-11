"""Tests for src.cycling_plotter — cycling visualisation helpers."""

import matplotlib
matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from src.cycling_plotter import plot_time_potential_with_integral  # noqa: E402


class TestPlotTimePotentialWithIntegral:
    def test_creates_figure_file(self, tmp_path):
        df = pd.DataFrame({
            "tempo": [0.0, 1.0, 2.0, 3.0],
            "potencial": [0.5, 1.0, 0.8, 0.3],
        })
        plot_time_potential_with_integral(
            df, "sample", out_dir=str(tmp_path), show=False,
        )
        assert (tmp_path / "sample_integral.png").exists()
