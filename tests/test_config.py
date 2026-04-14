"""Tests for src/config.py — PipelineConfig dataclass."""

import json
import tempfile
from pathlib import Path

import pytest

from src.config import PipelineConfig


# ── Defaults ────────────────────────────────────────────────────────────

class TestDefaults:
    """PipelineConfig.default() returns sane values."""

    def test_default_creates_instance(self):
        cfg = PipelineConfig.default()
        assert isinstance(cfg, PipelineConfig)

    def test_data_dir(self):
        assert PipelineConfig.default().data_dir == "data/raw"

    def test_processed_dir(self):
        assert PipelineConfig.default().processed_dir == "data/processed"

    def test_tables_dir(self):
        assert PipelineConfig.default().tables_dir == "outputs/tables"

    def test_voltage(self):
        assert PipelineConfig.default().voltage == 1.0

    def test_n_head(self):
        assert PipelineConfig.default().n_head == 5

    def test_capacitance_filter_range(self):
        cfg = PipelineConfig.default()
        assert cfg.capacitance_filter_range == (1e-15, 1e-2)

    def test_kmeans_n_clusters(self):
        assert PipelineConfig.default().kmeans_n_clusters == 2

    def test_score_weights_keys(self):
        w = PipelineConfig.default().score_weights
        assert set(w) == {"Rp_fit", "Rs_fit", "C_mean", "Energy_mean"}

    def test_pca_columns_length(self):
        assert len(PipelineConfig.default().pca_columns) == 5

    def test_stability_columns_length(self):
        assert len(PipelineConfig.default().stability_columns) == 4

    def test_drt_lambda(self):
        assert PipelineConfig.default().drt_lambda == 1e-3

    def test_drt_n_taus(self):
        assert PipelineConfig.default().drt_n_taus == 50

    def test_drt_max_peaks_exported(self):
        assert PipelineConfig.default().drt_max_peaks_exported == 3

    def test_language_default(self):
        assert PipelineConfig.default().language == "pt"

    def test_circuit_max_nfev(self):
        assert PipelineConfig.default().circuit_max_nfev == 5000

    def test_shortlist_top_n(self):
        assert PipelineConfig.default().circuit_shortlist_top_n == 3

    def test_bic_penalty_structured(self):
        assert PipelineConfig.default().bic_penalty_structured == 5.0

    def test_bic_penalty_per_bound_hit(self):
        assert PipelineConfig.default().bic_penalty_per_bound_hit == 0.5

    def test_dpi_diagnostics(self):
        assert PipelineConfig.default().dpi_diagnostics == 300

    def test_required_columns_is_tuple(self):
        assert isinstance(PipelineConfig.default().required_columns, tuple)


# ── Serialisation ───────────────────────────────────────────────────────

class TestSerialisation:
    """JSON round-trip preserves every field."""

    def test_roundtrip_defaults(self, tmp_path):
        original = PipelineConfig.default()
        path = tmp_path / "cfg.json"
        original.to_json(path)
        restored = PipelineConfig.from_json(path)

        assert restored.data_dir == original.data_dir
        assert restored.voltage == original.voltage
        assert restored.score_weights == original.score_weights
        assert restored.pca_columns == original.pca_columns
        assert restored.required_columns == original.required_columns
        assert restored.circuit_multi_seed_scales == original.circuit_multi_seed_scales

    def test_roundtrip_custom_values(self, tmp_path):
        cfg = PipelineConfig(
            voltage=3.7,
            kmeans_n_clusters=4,
            language="en",
            drt_lambda=1e-4,
            scan_rate=5.0,
        )
        path = tmp_path / "custom.json"
        cfg.to_json(path)
        loaded = PipelineConfig.from_json(path)

        assert loaded.voltage == 3.7
        assert loaded.kmeans_n_clusters == 4
        assert loaded.language == "en"
        assert loaded.drt_lambda == 1e-4
        assert loaded.scan_rate == 5.0

    def test_to_dict_returns_plain_dict(self):
        d = PipelineConfig.default().to_dict()
        assert isinstance(d, dict)
        # Tuples must become lists for JSON compatibility
        assert isinstance(d["required_columns"], list)
        assert isinstance(d["circuit_multi_seed_scales"], list)

    def test_json_file_is_valid_json(self, tmp_path):
        path = tmp_path / "test.json"
        PipelineConfig.default().to_json(path)
        with open(path) as fh:
            data = json.load(fh)
        assert "data_dir" in data

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "cfg.json"
        PipelineConfig.default().to_json(path)
        assert path.exists()


# ── Forward compatibility ───────────────────────────────────────────────

class TestForwardCompat:
    """Unknown keys are ignored gracefully."""

    def test_unknown_keys_are_ignored(self, tmp_path):
        path = tmp_path / "future.json"
        data = PipelineConfig.default().to_dict()
        data["brand_new_field_v99"] = "hello"
        with open(path, "w") as fh:
            json.dump(data, fh)

        cfg = PipelineConfig.from_json(path)
        assert cfg.data_dir == "data/raw"  # known fields still load

    def test_missing_keys_use_defaults(self, tmp_path):
        path = tmp_path / "minimal.json"
        with open(path, "w") as fh:
            json.dump({"voltage": 2.5}, fh)

        cfg = PipelineConfig.from_json(path)
        assert cfg.voltage == 2.5
        assert cfg.data_dir == "data/raw"  # default

    def test_from_json_safe_on_bad_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all!!!")
        cfg = PipelineConfig.from_json_safe(path)
        assert cfg.voltage == 1.0  # defaults

    def test_from_json_safe_on_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        cfg = PipelineConfig.from_json_safe(path)
        assert cfg.voltage == 1.0


# ── ensure_dirs ─────────────────────────────────────────────────────────

class TestEnsureDirs:
    """ensure_dirs() creates all output directories."""

    def test_creates_all_output_dirs(self, tmp_path):
        cfg = PipelineConfig(
            tables_dir=str(tmp_path / "t"),
            figures_dir=str(tmp_path / "f"),
            circuits_fig_dir=str(tmp_path / "fc"),
            analytics_fig_dir=str(tmp_path / "fa"),
            drt_fig_dir=str(tmp_path / "fd"),
            reports_dir=str(tmp_path / "r"),
            excel_dir=str(tmp_path / "e"),
            log_dir=str(tmp_path / "l"),
        )
        cfg.ensure_dirs()
        for d in ("t", "f", "fc", "fa", "fd", "r", "e", "l"):
            assert (tmp_path / d).is_dir()


# ── Custom field values ─────────────────────────────────────────────────

class TestCustomValues:
    """Fields can be overridden at construction."""

    def test_override_score_weights(self):
        custom = {"Rp_fit": 1.0}
        cfg = PipelineConfig(score_weights=custom)
        assert cfg.score_weights == {"Rp_fit": 1.0}

    def test_override_pca_columns(self):
        cfg = PipelineConfig(pca_columns=["A", "B"])
        assert cfg.pca_columns == ["A", "B"]

    def test_override_required_columns(self):
        cfg = PipelineConfig(required_columns=("freq", "z_re", "z_im"))
        assert cfg.required_columns == ("freq", "z_re", "z_im")
