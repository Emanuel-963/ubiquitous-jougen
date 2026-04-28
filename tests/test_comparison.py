"""Tests for Phase 4 — Comparative Analysis (src/comparison/).

Covers:
- Health Score computation, labels and colours
- Nyquist and Bode overlay plots (output type, axes content)
- Parameter timeline plot (dual-axis, missing columns)
- Public __init__ exports

The Agg (non-interactive) matplotlib backend is activated globally via
tests/conftest.py so no Tk window is needed when running tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.comparison import (
    DEFAULT_WEIGHTS,
    METRIC_LABELS,
    available_timeline_params,
    compute_health_score,
    health_score_color,
    health_score_label,
    plot_bode_overlay,
    plot_nyquist_overlay,
    plot_parameter_timeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_features(n: int = 4) -> pd.DataFrame:
    """Return a synthetic features_df with typical EIS columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Rs_fit": rng.uniform(0.1, 2.0, n),
            "Rp_fit": rng.uniform(10.0, 200.0, n),
            "C_mean": rng.uniform(1e-4, 1e-2, n),
            "Energy_mean": rng.uniform(0.5, 50.0, n),
        },
        index=[f"sample_{i}" for i in range(n)],
    )


def _make_raw_eis(n: int = 3) -> dict:
    """Return a synthetic raw_eis dict {name: DataFrame}."""
    rng = np.random.default_rng(1)
    raw = {}
    for i in range(n):
        freq = np.logspace(-1, 5, 30)
        zr = 5.0 + rng.uniform(-0.5, 0.5, 30).cumsum()
        zi = -rng.uniform(0.1, 3.0, 30)
        raw[f"sample_{i}"] = pd.DataFrame({"frequency": freq, "zreal": zr, "zimag": zi})
    return raw


# ============================================================================
# TestHealthScore
# ============================================================================


class TestHealthScore:
    def test_returns_series_same_index(self):
        df = _make_features()
        scores = compute_health_score(df)
        assert list(scores.index) == list(df.index)

    def test_values_in_range(self):
        df = _make_features(10)
        scores = compute_health_score(df)
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_empty_df_returns_50(self):
        df = pd.DataFrame({"Rs_fit": [], "Rp_fit": []})
        scores = compute_health_score(df)
        assert scores.empty

    def test_no_matching_columns_returns_50(self):
        df = pd.DataFrame({"unknown_col": [1.0, 2.0]}, index=["a", "b"])
        scores = compute_health_score(df)
        assert (scores == 50.0).all()

    def test_custom_weights_change_ranking(self):
        df = pd.DataFrame(
            {"Rs_fit": [0.1, 5.0], "Rp_fit": [100.0, 10.0]},
            index=["good", "bad"],
        )
        # Only Rs_fit: lower is better → 'good' should score higher
        scores = compute_health_score(df, weights={"Rs_fit": -1.0})
        assert scores["good"] > scores["bad"]

    def test_single_sample_is_50(self):
        df = _make_features(1)
        scores = compute_health_score(df)
        assert float(scores.iloc[0]) == 50.0

    def test_constant_column_does_not_crash(self):
        df = pd.DataFrame(
            {"Rs_fit": [2.0, 2.0, 2.0], "Rp_fit": [10.0, 20.0, 30.0]},
            index=["a", "b", "c"],
        )
        scores = compute_health_score(df)
        assert not scores.isna().any()

    def test_series_name_is_health_score(self):
        scores = compute_health_score(_make_features())
        assert scores.name == "Health Score"


class TestHealthScoreLabels:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (90.0, "Excelente"),
            (75.0, "Excelente"),
            (60.0, "Bom"),
            (50.0, "Bom"),
            (40.0, "Regular"),
            (25.0, "Regular"),
            (10.0, "Degradado"),
            (0.0, "Degradado"),
        ],
    )
    def test_label(self, score, expected):
        assert health_score_label(score) == expected

    @pytest.mark.parametrize(
        "score,expected_prefix",
        [
            (80.0, "#22c55e"),
            (55.0, "#eab308"),
            (30.0, "#f97316"),
            (5.0, "#ef4444"),
        ],
    )
    def test_color_hex(self, score, expected_prefix):
        assert health_score_color(score) == expected_prefix


# ============================================================================
# TestNyquistOverlay
# ============================================================================


class TestNyquistOverlay:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(3)
        fig = plot_nyquist_overlay(raw, list(raw.keys()))
        assert fig is not None
        plt.close(fig)

    def test_legend_entries_match_selected(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(3)
        selected = ["sample_0", "sample_1"]
        fig = plot_nyquist_overlay(raw, selected)
        ax = fig.axes[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert len(legend_texts) == 2
        plt.close(fig)

    def test_empty_selection_creates_empty_axes(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(2)
        fig = plot_nyquist_overlay(raw, [])
        ax = fig.axes[0]
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_missing_sample_is_skipped(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(2)
        fig = plot_nyquist_overlay(raw, ["sample_0", "nonexistent"])
        ax = fig.axes[0]
        # Only one valid sample should produce one line
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_embedded_mode_uses_provided_axes(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(2)
        fig, ax = plt.subplots()
        result = plot_nyquist_overlay(raw, list(raw.keys()), fig=fig, ax=ax)
        assert result is fig
        plt.close(fig)

    def test_axes_labels(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(2)
        fig = plot_nyquist_overlay(raw, ["sample_0"])
        ax = fig.axes[0]
        assert "Z′" in ax.get_xlabel()
        assert "Z″" in ax.get_ylabel()
        plt.close(fig)


# ============================================================================
# TestBodeOverlay
# ============================================================================


class TestBodeOverlay:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(3)
        fig = plot_bode_overlay(raw, list(raw.keys()))
        assert fig is not None
        plt.close(fig)

    def test_two_axes(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(2)
        fig = plot_bode_overlay(raw, list(raw.keys()))
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_magnitude_axis_ylabel(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(2)
        fig = plot_bode_overlay(raw, ["sample_0"])
        assert "|Z|" in fig.axes[0].get_ylabel()
        plt.close(fig)

    def test_phase_axis_ylabel(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(2)
        fig = plot_bode_overlay(raw, ["sample_0"])
        assert "Fase" in fig.axes[1].get_ylabel()
        plt.close(fig)

    def test_embedded_mode_reuses_figure(self):
        import matplotlib.pyplot as plt

        raw = _make_raw_eis(2)
        fig, axes = plt.subplots(2, 1)
        result = plot_bode_overlay(raw, ["sample_0"], fig=fig, axes=axes)
        assert result is fig
        plt.close(fig)

    def test_df_missing_frequency_column_skipped(self):
        import matplotlib.pyplot as plt

        raw = {
            "ok": pd.DataFrame({"frequency": [100.0], "zreal": [5.0], "zimag": [-1.0]}),
            "bad": pd.DataFrame({"zreal": [5.0], "zimag": [-1.0]}),
        }
        fig = plot_bode_overlay(raw, ["ok", "bad"])
        # Only one valid sample plotted on magnitude axis
        assert len(fig.axes[0].lines) == 1
        plt.close(fig)


# ============================================================================
# TestParameterTimeline
# ============================================================================


class TestParameterTimeline:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        df = _make_features()
        fig = plot_parameter_timeline(df)
        assert fig is not None
        plt.close(fig)

    def test_empty_df_shows_placeholder_text(self):
        import matplotlib.pyplot as plt

        df = pd.DataFrame({"Rs_fit": []})
        fig = plot_parameter_timeline(df)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("dados" in t.lower() for t in texts)
        plt.close(fig)

    def test_single_param_no_twin_axis(self):
        import matplotlib.pyplot as plt

        df = _make_features()
        fig = plot_parameter_timeline(df, params=["Rs_fit"])
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_two_params_creates_twin_axis(self):
        import matplotlib.pyplot as plt

        df = _make_features()
        fig = plot_parameter_timeline(df, params=["Rs_fit", "Rp_fit"])
        # twinx creates a second axes sharing x
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_xtick_count_matches_samples(self):
        import matplotlib.pyplot as plt

        df = _make_features(5)
        fig = plot_parameter_timeline(df, params=["Rs_fit"])
        ax = fig.axes[0]
        assert len(ax.get_xticks()) == 5
        plt.close(fig)

    def test_params_filter_works(self):
        import matplotlib.pyplot as plt

        df = _make_features()
        fig = plot_parameter_timeline(df, params=["Rs_fit"])
        ax = fig.axes[0]
        # Should have exactly one line
        assert len(ax.lines) == 1
        plt.close(fig)


# ============================================================================
# TestAvailableTimelineParams
# ============================================================================


class TestAvailableTimelineParams:
    def test_returns_only_present_cols(self):
        df = pd.DataFrame({"Rs_fit": [1.0], "not_a_col": [2.0]})
        params = available_timeline_params(df)
        cols = [c for c, _ in params]
        assert "Rs_fit" in cols
        assert "not_a_col" not in cols

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame()
        assert available_timeline_params(df) == []

    def test_all_defaults_present(self):
        df = _make_features()  # has Rs_fit, Rp_fit, C_mean, Energy_mean
        params = available_timeline_params(df)
        assert len(params) == 4


# ============================================================================
# TestComparisonPublicAPI
# ============================================================================


class TestComparisonPublicAPI:
    def test_default_weights_has_four_keys(self):
        assert len(DEFAULT_WEIGHTS) == 4

    def test_metric_labels_all_strings(self):
        assert all(isinstance(v, str) for v in METRIC_LABELS.values())

    def test_all_exports_callable(self):
        for fn in (
            compute_health_score,
            health_score_label,
            health_score_color,
            plot_nyquist_overlay,
            plot_bode_overlay,
            available_timeline_params,
            plot_parameter_timeline,
        ):
            assert callable(fn)
