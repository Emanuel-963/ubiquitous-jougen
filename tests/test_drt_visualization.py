"""Tests for src.drt_visualization plotting helpers."""

from pathlib import Path

import matplotlib
import numpy as np

from src.drt_analysis import compute_drt
from src.drt_visualization import plot_drt_heatmap, plot_drt_overlay, plot_drt_spectrum

matplotlib.use("Agg")


def _sample_result(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    freq = np.logspace(-1, 5, 40)
    omega = 2.0 * np.pi * freq
    r_inf = 1.0
    r_ct = 8.0
    tau0 = 1e-3
    denom = 1.0 + (omega * tau0) ** 2
    z_real = r_inf + r_ct / denom + rng.normal(0, 0.01, len(freq))
    z_imag = -(r_ct * omega * tau0 / denom) + rng.normal(0, 0.01, len(freq))
    return compute_drt(freq, z_real, z_imag, n_taus=60, lambda_reg=1e-3)


def test_plot_drt_spectrum_saves_png(tmp_path: Path):
    result = _sample_result(1)
    out = plot_drt_spectrum(result, "sample_a", out_dir=tmp_path, save=True, show=False)
    assert out
    assert Path(out).exists()


def test_plot_drt_overlay_saves_png(tmp_path: Path):
    results = {
        "sample_a": _sample_result(2),
        "sample_b": _sample_result(3),
    }
    out_path = tmp_path / "overlay.png"
    out = plot_drt_overlay(results, out_path=out_path, show=False)
    assert out
    assert out_path.exists()


def test_plot_drt_heatmap_saves_png(tmp_path: Path):
    results = {
        "sample_a": _sample_result(4),
        "sample_b": _sample_result(5),
        "sample_c": _sample_result(6),
    }
    out_path = tmp_path / "heatmap.png"
    out = plot_drt_heatmap(results, out_path=out_path, show=False)
    assert out
    assert out_path.exists()
