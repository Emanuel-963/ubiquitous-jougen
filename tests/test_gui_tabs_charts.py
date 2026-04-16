"""Tests for low-coverage GUI tab chart builders (eis_charts, cycling_charts, drt_charts).

Targets Rec 4 — raise coverage from ~24% → 80%+ for these modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture()
def eis_df():
    """Minimal EIS DataFrame with frequency, zreal, zimag."""
    freq = np.logspace(-1, 5, 30)
    zr = 10 + 5 / (1 + (2 * np.pi * freq * 1e-3) ** 2)
    zi = -5 * (2 * np.pi * freq * 1e-3) / (1 + (2 * np.pi * freq * 1e-3) ** 2)
    return pd.DataFrame({"frequency": freq, "zreal": zr, "zimag": zi})


@pytest.fixture()
def raw_eis(eis_df):
    return {"sample_A.csv": eis_df, "SAMPLE_B.csv": eis_df.copy()}


@pytest.fixture()
def cic_df():
    """Minimal cycling DataFrame."""
    return pd.DataFrame({
        "Ciclo": [1, 2, 3, 4, 5],
        "Energia (Wh/kg)": [10.1, 9.8, 9.6, 9.5, 9.3],
        "Potência (W/kg)": [50, 49, 48, 47, 46],
        "Retenção (%)": [100, 97, 95, 94, 92],
    })


@pytest.fixture()
def cic_results(cic_df):
    return {"sampleX": cic_df, "sampleY": cic_df.copy()}


@pytest.fixture()
def drt_result():
    """Minimal DRT result dict."""
    tau = np.logspace(-5, 1, 50)
    gamma = np.exp(-((np.log10(tau) + 2) ** 2) / 0.5)
    return {"tau": tau, "gamma": gamma, "peaks": [{"tau": 0.01, "gamma": 0.8}]}


@pytest.fixture()
def drt_results(drt_result):
    return {"drt_A": drt_result, "drt_B": dict(drt_result)}


# ═══════════════════════════════════════════════════════════════════════
#  EIS Charts
# ═══════════════════════════════════════════════════════════════════════

class TestEISChartBuilders:
    """Tests for src.gui.tabs.eis_charts."""

    def test_build_fig_nyquist_returns_figure(self, raw_eis):
        from src.gui.tabs.eis_charts import build_fig_nyquist
        fig = build_fig_nyquist(raw_eis, "sample_A.csv")
        assert isinstance(fig, Figure)

    def test_build_fig_nyquist_normalized_match(self, raw_eis):
        from src.gui.tabs.eis_charts import build_fig_nyquist
        fig = build_fig_nyquist(raw_eis, "sample_a")
        assert isinstance(fig, Figure)

    def test_build_fig_nyquist_empty_dict(self):
        from src.gui.tabs.eis_charts import build_fig_nyquist
        assert build_fig_nyquist({}, "x") is None

    def test_build_fig_nyquist_missing_sample(self, raw_eis):
        from src.gui.tabs.eis_charts import build_fig_nyquist
        assert build_fig_nyquist(raw_eis, "nonexistent") is None

    def test_build_fig_nyquist_empty_df(self):
        from src.gui.tabs.eis_charts import build_fig_nyquist
        raw = {"empty.csv": pd.DataFrame()}
        assert build_fig_nyquist(raw, "empty.csv") is None

    def test_build_fig_bode_returns_figure(self, raw_eis):
        from src.gui.tabs.eis_charts import build_fig_bode
        fig = build_fig_bode(raw_eis, "sample_A.csv")
        assert isinstance(fig, Figure)

    def test_build_fig_bode_normalized(self, raw_eis):
        from src.gui.tabs.eis_charts import build_fig_bode
        fig = build_fig_bode(raw_eis, "SAMPLE_B")
        assert isinstance(fig, Figure)

    def test_build_fig_bode_empty(self):
        from src.gui.tabs.eis_charts import build_fig_bode
        assert build_fig_bode({}, "x") is None

    def test_build_fig_bode_missing(self, raw_eis):
        from src.gui.tabs.eis_charts import build_fig_bode
        assert build_fig_bode(raw_eis, "nope") is None

    def test_build_fig_impedance_heatmap_returns_figure(self, raw_eis):
        from src.gui.tabs.eis_charts import build_fig_impedance_heatmap
        fig = build_fig_impedance_heatmap(raw_eis)
        assert isinstance(fig, Figure)

    def test_build_fig_impedance_heatmap_empty(self):
        from src.gui.tabs.eis_charts import build_fig_impedance_heatmap
        assert build_fig_impedance_heatmap({}) is None

    def test_resolve_sample_df_exact(self, raw_eis):
        from src.gui.tabs.eis_charts import _resolve_sample_df
        df = _resolve_sample_df(raw_eis, "sample_A.csv")
        assert df is not None

    def test_resolve_sample_df_normalized(self, raw_eis):
        from src.gui.tabs.eis_charts import _resolve_sample_df
        df = _resolve_sample_df(raw_eis, "sample_a")
        assert df is not None

    def test_resolve_sample_df_missing(self, raw_eis):
        from src.gui.tabs.eis_charts import _resolve_sample_df
        assert _resolve_sample_df(raw_eis, "nope") is None


# ═══════════════════════════════════════════════════════════════════════
#  Cycling Charts
# ═══════════════════════════════════════════════════════════════════════

class TestCyclingChartBuilders:
    """Tests for src.gui.tabs.cycling_charts."""

    def test_build_fig_energy_power_returns_figure(self, cic_results):
        from src.gui.tabs.cycling_charts import build_fig_energy_power
        fig = build_fig_energy_power(cic_results, "sampleX")
        assert isinstance(fig, Figure)

    def test_build_fig_energy_power_empty(self):
        from src.gui.tabs.cycling_charts import build_fig_energy_power
        assert build_fig_energy_power({}, "x") is None

    def test_build_fig_energy_power_missing(self, cic_results):
        from src.gui.tabs.cycling_charts import build_fig_energy_power
        assert build_fig_energy_power(cic_results, "nope") is None

    def test_build_fig_energy_power_normalized(self, cic_df):
        from src.gui.tabs.cycling_charts import build_fig_energy_power
        results = {"SampleX.csv": cic_df}
        fig = build_fig_energy_power(results, "samplex")
        assert isinstance(fig, Figure)

    def test_build_fig_energy_cycle_returns_figure(self, cic_results):
        from src.gui.tabs.cycling_charts import build_fig_energy_cycle
        fig = build_fig_energy_cycle(cic_results)
        assert isinstance(fig, Figure)

    def test_build_fig_energy_cycle_empty(self):
        from src.gui.tabs.cycling_charts import build_fig_energy_cycle
        assert build_fig_energy_cycle({}) is None

    def test_build_fig_energy_cycle_with_metric(self, cic_results):
        from src.gui.tabs.cycling_charts import build_fig_energy_cycle
        fig = build_fig_energy_cycle(cic_results, metric="Potência (W/kg)")
        assert isinstance(fig, Figure)

    def test_build_fig_retention_cycle_returns_figure(self, cic_results):
        from src.gui.tabs.cycling_charts import build_fig_retention_cycle
        fig = build_fig_retention_cycle(cic_results)
        assert isinstance(fig, Figure)

    def test_build_fig_retention_cycle_empty(self):
        from src.gui.tabs.cycling_charts import build_fig_retention_cycle
        assert build_fig_retention_cycle({}) is None

    def test_build_fig_ragone_returns_figure(self, cic_results):
        from src.gui.tabs.cycling_charts import build_fig_ragone
        fig = build_fig_ragone(cic_results)
        assert isinstance(fig, Figure)

    def test_build_fig_ragone_empty(self):
        from src.gui.tabs.cycling_charts import build_fig_ragone
        assert build_fig_ragone({}) is None

    def test_build_fig_ragone_with_highlight(self, cic_results):
        from src.gui.tabs.cycling_charts import build_fig_ragone
        fig = build_fig_ragone(cic_results, highlight_sample="sampleX")
        assert isinstance(fig, Figure)


# ═══════════════════════════════════════════════════════════════════════
#  DRT Charts
# ═══════════════════════════════════════════════════════════════════════

class TestDRTChartBuilders:
    """Tests for src.gui.tabs.drt_charts."""

    def test_build_fig_drt_spectrum_returns_figure(self, drt_results):
        from src.gui.tabs.drt_charts import build_fig_drt_spectrum
        fig = build_fig_drt_spectrum(drt_results, "drt_A")
        assert isinstance(fig, Figure)

    def test_build_fig_drt_spectrum_empty(self):
        from src.gui.tabs.drt_charts import build_fig_drt_spectrum
        assert build_fig_drt_spectrum({}, "x") is None

    def test_build_fig_drt_spectrum_missing(self, drt_results):
        from src.gui.tabs.drt_charts import build_fig_drt_spectrum
        assert build_fig_drt_spectrum(drt_results, "nope") is None

    def test_build_fig_drt_overlay_returns_figure(self, drt_results):
        from src.gui.tabs.drt_charts import build_fig_drt_overlay
        fig = build_fig_drt_overlay(drt_results)
        assert isinstance(fig, Figure)

    def test_build_fig_drt_overlay_empty(self):
        from src.gui.tabs.drt_charts import build_fig_drt_overlay
        assert build_fig_drt_overlay({}) is None

    def test_build_fig_drt_overlay_selected(self, drt_results):
        from src.gui.tabs.drt_charts import build_fig_drt_overlay
        fig = build_fig_drt_overlay(drt_results, sample_names=["drt_A"])
        assert isinstance(fig, Figure)

    def test_build_fig_drt_heatmap_returns_figure(self, drt_results):
        from src.gui.tabs.drt_charts import build_fig_drt_heatmap
        fig = build_fig_drt_heatmap(drt_results)
        assert isinstance(fig, Figure)

    def test_build_fig_drt_heatmap_empty(self):
        from src.gui.tabs.drt_charts import build_fig_drt_heatmap
        assert build_fig_drt_heatmap({}) is None

    def test_build_fig_drt_heatmap_selected(self, drt_results):
        from src.gui.tabs.drt_charts import build_fig_drt_heatmap
        fig = build_fig_drt_heatmap(drt_results, sample_names=["drt_B"])
        assert isinstance(fig, Figure)


# ═══════════════════════════════════════════════════════════════════════
#  Normalize helper
# ═══════════════════════════════════════════════════════════════════════

class TestNormalize:
    def test_eis_normalize(self):
        from src.gui.tabs.eis_charts import _normalize
        assert _normalize("Sample_A.csv") == "sample_a"
        assert _normalize("  TEST.CSV  ") == "test"

    def test_cycling_normalize(self):
        from src.gui.tabs.cycling_charts import _normalize
        assert _normalize("Sample.csv") == "sample"
