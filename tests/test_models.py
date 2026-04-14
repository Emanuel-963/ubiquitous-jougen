"""Tests for src/models.py — typed result objects."""

import pandas as pd
import pytest

from src.config import PipelineConfig
from src.models import (
    CyclingResult,
    DRTPipelineResult,
    EISResult,
    PCAResult,
    _DictAccessMixin,
)


# ── DictAccessMixin ─────────────────────────────────────────────────────

class TestDictAccessMixin:
    """Bracket access and .get() for migration."""

    def test_getitem(self):
        r = EISResult()
        assert r["out_dir"] == "outputs/tables"

    def test_getitem_raises_keyerror(self):
        r = EISResult()
        with pytest.raises(KeyError):
            _ = r["nonexistent_field"]

    def test_get_with_default(self):
        r = EISResult()
        assert r.get("nonexistent", 42) == 42

    def test_get_existing(self):
        r = EISResult(out_dir="/tmp")
        assert r.get("out_dir") == "/tmp"

    def test_contains(self):
        r = EISResult()
        assert "out_dir" in r
        assert "bogus" not in r


# ── PCAResult ───────────────────────────────────────────────────────────

class TestPCAResult:
    """PCA result defaults and access."""

    def test_defaults(self):
        p = PCAResult()
        assert p.df_pca is None
        assert p.loadings is None
        assert p.evr is None
        assert p.figure_paths == []

    def test_with_data(self):
        df = pd.DataFrame({"PC1": [1, 2], "PC2": [3, 4]})
        p = PCAResult(df_pca=df, figure_paths=["a.png", "b.png"])
        assert len(p.figure_paths) == 2
        assert p.df_pca is df


# ── EISResult ───────────────────────────────────────────────────────────

class TestEISResult:
    """EIS result defaults, legacy aliases and to_dict."""

    def test_defaults_empty(self):
        r = EISResult()
        assert r.features_df.empty
        assert r.ranked_df.empty
        assert r.circuit_table is None
        assert r.raw_eis == {}
        assert isinstance(r.config_used, PipelineConfig)

    def test_legacy_alias_df(self):
        feats = pd.DataFrame({"a": [1]})
        r = EISResult(features_df=feats)
        assert r.df is feats
        assert r["df"] is feats

    def test_legacy_alias_df_ranked(self):
        ranked = pd.DataFrame({"Score": [0.5]})
        r = EISResult(ranked_df=ranked)
        assert r.df_ranked is ranked
        assert r["df_ranked"] is ranked

    def test_legacy_alias_cap_energy(self):
        cap = pd.DataFrame({"C": [1e-6]})
        r = EISResult(cap_energy_df=cap)
        assert r.cap_energy is cap

    def test_legacy_alias_pca(self):
        pca = PCAResult(figure_paths=["x.png"])
        r = EISResult(pca=pca)
        assert r.df_pca is None
        assert r.pca_loadings is None
        assert r.pca_evr is None
        assert r.pca_paths == ["x.png"]

    def test_to_dict_keys(self):
        r = EISResult()
        d = r.to_dict()
        expected = {
            "df", "df_ranked", "cap_energy", "df_pca",
            "pca_loadings", "pca_evr", "pca_paths",
            "out_dir", "circuit_table", "raw_eis",
        }
        assert set(d.keys()) == expected

    def test_config_used_default(self):
        r = EISResult()
        assert r.config_used.data_dir == "data/raw"


# ── CyclingResult ───────────────────────────────────────────────────────

class TestCyclingResult:
    """Cycling result defaults and to_dict."""

    def test_defaults(self):
        r = CyclingResult()
        assert r.results == {}
        assert r.merged_table is None
        assert r.plot_paths == []
        assert r.energy_power_paths == []

    def test_to_dict(self):
        r = CyclingResult(plot_paths=[("f.txt", "/tmp/f.png")])
        d = r.to_dict()
        assert "plot_paths" in d
        assert len(d["plot_paths"]) == 1

    def test_bracket_access(self):
        r = CyclingResult()
        assert r["results"] == {}
        assert r.get("merged_table") is None


# ── DRTPipelineResult ──────────────────────────────────────────────────

class TestDRTPipelineResult:
    """DRT pipeline result defaults and to_dict."""

    def test_defaults(self):
        r = DRTPipelineResult()
        assert r.drt_table.empty
        assert r.drt_peaks_table.empty
        assert r.errors == {}
        assert r.run_meta == {}

    def test_to_dict(self):
        r = DRTPipelineResult(errors={"file.txt": "bad data"})
        d = r.to_dict()
        expected_keys = {
            "drt_table", "drt_peaks_table", "drt_summary_table",
            "per_file_results", "plot_paths", "errors", "run_meta",
        }
        assert set(d.keys()) == expected_keys
        assert d["errors"]["file.txt"] == "bad data"

    def test_bracket_access(self):
        r = DRTPipelineResult()
        assert r["plot_paths"] == []
        assert r.get("errors") == {}
