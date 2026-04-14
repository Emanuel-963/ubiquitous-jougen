"""Tests for src.validation — data validators."""

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    ValidationResult,
    validate_eis_dataframe,
    validate_cycling_dataframe,
    validate_frequency_range,
    validate_impedance_quality,
    validate_eis_full,
)


# ── Helpers / fixtures ───────────────────────────────────────────────
def _good_eis_df(n: int = 50) -> pd.DataFrame:
    """Generate a realistic-looking EIS DataFrame."""
    freq = np.logspace(-1, 5, n)  # 0.1 Hz → 100 kHz = 6 decades
    zreal = 100 + 50 * np.random.randn(n) * 0.01
    zimag = -20 + 10 * np.random.randn(n) * 0.01
    return pd.DataFrame({"frequency": freq, "zreal": zreal, "zimag": zimag})


def _good_cycling_df(n_cycles: int = 5, pts_per_cycle: int = 20) -> pd.DataFrame:
    """Generate a realistic cycling DataFrame."""
    rows = []
    for c in range(1, n_cycles + 1):
        t = np.linspace((c - 1) * 10, c * 10, pts_per_cycle)
        rows.append(pd.DataFrame({
            "tempo": t,
            "corrente": np.sin(t) * 0.01,
            "potencial": np.cos(t) * 0.5 + 0.5,
            "ciclo": c,
        }))
    return pd.concat(rows, ignore_index=True)


# ── ValidationResult ─────────────────────────────────────────────────
class TestValidationResult:
    def test_default_is_ok(self):
        vr = ValidationResult()
        assert vr.ok is True
        assert vr.warnings == []
        assert vr.errors == []

    def test_add_warning_keeps_ok(self):
        vr = ValidationResult()
        vr.add_warning("minor issue")
        assert vr.ok is True
        assert len(vr.warnings) == 1

    def test_add_error_sets_not_ok(self):
        vr = ValidationResult()
        vr.add_error("big problem")
        assert vr.ok is False
        assert len(vr.errors) == 1

    def test_merge_combines_messages(self):
        a = ValidationResult()
        a.add_warning("w1")
        b = ValidationResult()
        b.add_error("e1")
        a.merge(b)
        assert not a.ok
        assert "w1" in a.warnings
        assert "e1" in a.errors

    def test_log_all_does_not_raise(self):
        vr = ValidationResult()
        vr.add_warning("w")
        vr.add_error("e")
        vr.log_all()  # smoke — should not raise


# ── validate_eis_dataframe ───────────────────────────────────────────
class TestValidateEIS:
    def test_valid_df(self):
        vr = validate_eis_dataframe(_good_eis_df())
        assert vr.ok

    def test_none_input(self):
        vr = validate_eis_dataframe(None)
        assert not vr.ok

    def test_missing_column(self):
        df = _good_eis_df().drop(columns=["zimag"])
        vr = validate_eis_dataframe(df)
        assert not vr.ok
        assert any("zimag" in e for e in vr.errors)

    def test_non_numeric_column(self):
        df = _good_eis_df()
        df["frequency"] = df["frequency"].astype(str)
        vr = validate_eis_dataframe(df)
        assert not vr.ok

    def test_nan_produces_warning(self):
        df = _good_eis_df()
        df.loc[0, "zreal"] = np.nan
        vr = validate_eis_dataframe(df)
        assert vr.ok  # NaN is a warning, not error
        assert len(vr.warnings) >= 1

    def test_inf_produces_error(self):
        df = _good_eis_df()
        df.loc[0, "zreal"] = np.inf
        vr = validate_eis_dataframe(df)
        assert not vr.ok

    def test_negative_frequency_produces_error(self):
        df = _good_eis_df()
        df.loc[0, "frequency"] = -1.0
        vr = validate_eis_dataframe(df)
        assert not vr.ok

    def test_few_points_produces_warning(self):
        df = _good_eis_df(n=3)
        vr = validate_eis_dataframe(df)
        assert vr.ok
        assert len(vr.warnings) >= 1


# ── validate_cycling_dataframe ───────────────────────────────────────
class TestValidateCycling:
    def test_valid_df(self):
        vr = validate_cycling_dataframe(_good_cycling_df())
        assert vr.ok

    def test_none_input(self):
        vr = validate_cycling_dataframe(None)
        assert not vr.ok

    def test_missing_column(self):
        df = _good_cycling_df().drop(columns=["corrente"])
        vr = validate_cycling_dataframe(df)
        assert not vr.ok

    def test_non_numeric(self):
        df = _good_cycling_df()
        df["tempo"] = "abc"
        vr = validate_cycling_dataframe(df)
        assert not vr.ok

    def test_single_cycle_warns(self):
        df = _good_cycling_df(n_cycles=1)
        vr = validate_cycling_dataframe(df)
        assert vr.ok
        assert any("1 cycle" in w for w in vr.warnings)

    def test_non_monotonic_time_warns(self):
        df = _good_cycling_df(n_cycles=1, pts_per_cycle=10)
        # Reverse time in cycle 1
        df["tempo"] = df["tempo"].values[::-1]
        vr = validate_cycling_dataframe(df)
        assert any("monoton" in w.lower() for w in vr.warnings)

    def test_nan_produces_warning(self):
        df = _good_cycling_df()
        df.loc[0, "potencial"] = np.nan
        vr = validate_cycling_dataframe(df)
        assert len(vr.warnings) >= 1


# ── validate_frequency_range ─────────────────────────────────────────
class TestValidateFrequencyRange:
    def test_good_range(self):
        df = _good_eis_df(50)  # 6 decades
        vr = validate_frequency_range(df)
        assert vr.ok
        assert len(vr.warnings) == 0

    def test_narrow_range_warns(self):
        freq = np.linspace(100, 200, 20)  # < 1 decade
        df = pd.DataFrame({
            "frequency": freq,
            "zreal": np.ones(20),
            "zimag": np.ones(20),
        })
        vr = validate_frequency_range(df)
        assert any("decade" in w for w in vr.warnings)

    def test_missing_column(self):
        df = pd.DataFrame({"zreal": [1], "zimag": [1]})
        vr = validate_frequency_range(df)
        assert not vr.ok

    def test_no_positive_freq(self):
        df = pd.DataFrame({"frequency": [-1, -2], "zreal": [1, 2], "zimag": [1, 2]})
        vr = validate_frequency_range(df)
        assert not vr.ok

    def test_custom_min_decades(self):
        # 2 decades is fine for min_decades=1
        freq = np.logspace(0, 2, 20)
        df = pd.DataFrame({
            "frequency": freq,
            "zreal": np.ones(20),
            "zimag": np.ones(20),
        })
        vr = validate_frequency_range(df, min_decades=1.0)
        assert vr.ok


# ── validate_impedance_quality ───────────────────────────────────────
class TestValidateImpedanceQuality:
    def test_clean_semicircle(self):
        # Perfect semicircle — low residuals
        theta = np.linspace(0, np.pi, 60)
        zr = 50 + 50 * np.cos(theta)
        zi = -50 * np.sin(theta)
        df = pd.DataFrame({"zreal": zr, "zimag": zi})
        vr = validate_impedance_quality(df)
        assert vr.ok

    def test_noisy_data_warns(self):
        np.random.seed(42)
        zr = np.random.randn(60) * 100
        zi = np.random.randn(60) * 100
        df = pd.DataFrame({"zreal": zr, "zimag": zi})
        vr = validate_impedance_quality(df, residual_threshold=0.01)
        # With random data, residuals should be large
        assert len(vr.warnings) >= 1

    def test_too_few_points_skips(self):
        df = pd.DataFrame({"zreal": [1, 2, 3], "zimag": [1, 2, 3]})
        vr = validate_impedance_quality(df)
        assert vr.ok
        assert any("skip" in w.lower() for w in vr.warnings)

    def test_missing_columns(self):
        df = pd.DataFrame({"frequency": [1, 2]})
        vr = validate_impedance_quality(df)
        assert not vr.ok


# ── validate_eis_full (integration) ──────────────────────────────────
class TestValidateEISFull:
    def test_valid_data(self):
        vr = validate_eis_full(_good_eis_df())
        assert vr.ok

    def test_bad_data_stops_early(self):
        vr = validate_eis_full(pd.DataFrame({"x": [1]}))
        assert not vr.ok

    def test_merges_all_checks(self):
        # Narrow frequency range → should get frequency warning
        freq = np.linspace(100, 200, 20)
        df = pd.DataFrame({
            "frequency": freq,
            "zreal": np.ones(20) * 50,
            "zimag": -np.ones(20) * 10,
        })
        vr = validate_eis_full(df)
        assert any("decade" in w for w in vr.warnings)
