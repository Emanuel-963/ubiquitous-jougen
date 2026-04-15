"""Comprehensive tests for the IonFlow CLI (src/cli.py).

Tests all subcommands (eis, cycling, drt, analyze, config, validate, version),
JSON output mode, return codes, error handling, and argument parsing.
All pipeline calls are mocked to isolate CLI logic.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.cli import (
    RC_ERROR,
    RC_OK,
    RC_WARNING,
    _ProgressReporter,
    _load_config,
    _print_json,
    _use_json,
    build_parser,
    cmd_analyze,
    cmd_config,
    cmd_cycling,
    cmd_drt,
    cmd_eis,
    cmd_validate,
    cmd_version,
    main,
)
from src.config import PipelineConfig


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory with some fake .txt files."""
    for i in range(3):
        (tmp_path / f"sample_{i}.txt").write_text(f"data {i}")
    return tmp_path


@pytest.fixture()
def config_file(tmp_path):
    """Create a temporary config.json."""
    cfg = PipelineConfig.default()
    path = tmp_path / "config.json"
    cfg.to_json(str(path))
    return str(path)


def _make_eis_result():
    """Build a mock EIS result."""
    r = MagicMock()
    r.ranked_df = pd.DataFrame({"Rs_fit": [1.0], "Rp_fit": [10.0]})
    r.circuit_table = pd.DataFrame({"Circuito": ["Randles-CPE"]})
    r.pca = MagicMock()
    r.pca.df_pca = pd.DataFrame({"PC1": [0.1], "PC2": [0.2]})
    r.get = lambda k, d=None: getattr(r, k, d)
    r.__getitem__ = lambda s, k: getattr(s, k)
    return r


def _make_cycling_result():
    """Build a mock cycling result."""
    r = MagicMock()
    r.results = {"file1.txt": pd.DataFrame({"cycle": [1, 2]})}
    r.merged_table = pd.DataFrame({"Arquivo": ["f1"], "Energia (Wh/kg)": [10.0]})
    r.get = lambda k, d=None: getattr(r, k, d)
    r.__getitem__ = lambda s, k: getattr(s, k)
    return r


def _make_drt_result():
    """Build a mock DRT result."""
    r = MagicMock()
    r.run_meta = {"n_success": 3, "n_failed": 0}
    r.drt_table = pd.DataFrame({"Arquivo": ["a.txt"], "n_peaks": [2]})
    r.drt_peaks_table = pd.DataFrame({"Arquivo": ["a.txt"], "peak_order": [1]})
    r.drt_summary_table = pd.DataFrame({"Arquivo": ["a.txt"], "n_peaks": [2]})
    r.get = lambda k, d=None: getattr(r, k, d)
    r.__getitem__ = lambda s, k: getattr(s, k)
    return r


# ═══════════════════════════════════════════════════════════════════════
# Test build_parser
# ═══════════════════════════════════════════════════════════════════════

class TestBuildParser:
    """Tests for the argument parser construction."""

    def test_parser_is_created(self):
        parser = build_parser()
        assert parser is not None
        assert parser.prog == "ionflow"

    def test_subcommands_exist(self):
        parser = build_parser()
        # Parsing each subcommand should not raise
        for cmd in ["eis", "cycling", "drt", "analyze", "config", "validate", "version"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_global_flags(self):
        parser = build_parser()
        args = parser.parse_args(["--json", "--verbose", "--config", "my.json", "version"])
        assert args.json is True
        assert args.verbose is True
        assert args.config == "my.json"

    def test_eis_args(self):
        parser = build_parser()
        args = parser.parse_args(["eis", "--data-dir", "mydata", "--output", "myout"])
        assert args.data_dir == "mydata"
        assert args.output == "myout"

    def test_cycling_args(self):
        parser = build_parser()
        args = parser.parse_args(["cycling", "--scan-rate", "0.5"])
        assert args.scan_rate == 0.5

    def test_drt_args(self):
        parser = build_parser()
        args = parser.parse_args(["drt", "--lambda", "1e-4"])
        assert args.lambda_reg == 1e-4

    def test_analyze_args(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--all", "--ai", "--export-pdf", "rpt.pdf"])
        assert args.all is True
        assert args.ai is True
        assert args.export_pdf == "rpt.pdf"

    def test_config_init_args(self):
        parser = build_parser()
        args = parser.parse_args(["config", "--init", "--output", "c.json"])
        assert args.init is True
        assert args.output == "c.json"

    def test_config_show_args(self):
        parser = build_parser()
        args = parser.parse_args(["config", "--show"])
        assert args.show is True

    def test_validate_args(self):
        parser = build_parser()
        args = parser.parse_args(["validate", "--data-dir", "d"])
        assert args.data_dir == "d"

    def test_language_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--language", "en", "version"])
        assert args.language == "en"


# ═══════════════════════════════════════════════════════════════════════
# Test _load_config
# ═══════════════════════════════════════════════════════════════════════

class TestLoadConfig:
    """Tests for config loading from CLI args."""

    def test_default_config(self):
        args = build_parser().parse_args(["version"])
        cfg = _load_config(args)
        assert isinstance(cfg, PipelineConfig)
        assert cfg.data_dir == "data/raw"

    def test_config_from_json(self, config_file):
        args = build_parser().parse_args(["--config", config_file, "version"])
        cfg = _load_config(args)
        assert isinstance(cfg, PipelineConfig)

    def test_override_data_dir(self):
        args = build_parser().parse_args(["eis", "--data-dir", "/custom/path"])
        cfg = _load_config(args)
        assert cfg.data_dir == "/custom/path"

    def test_override_output(self):
        args = build_parser().parse_args(["eis", "--output", "/my/output"])
        cfg = _load_config(args)
        assert cfg.output_dir == "/my/output"
        assert "tables" in cfg.tables_dir
        assert "figures" in cfg.figures_dir

    def test_override_language(self):
        args = build_parser().parse_args(["--language", "es", "eis"])
        cfg = _load_config(args)
        assert cfg.language == "es"


# ═══════════════════════════════════════════════════════════════════════
# Test _use_json / _print_json
# ═══════════════════════════════════════════════════════════════════════

class TestJSONHelpers:
    """Tests for JSON mode utilities."""

    def test_use_json_true(self):
        args = build_parser().parse_args(["--json", "version"])
        assert _use_json(args) is True

    def test_use_json_false(self):
        args = build_parser().parse_args(["version"])
        assert _use_json(args) is False

    def test_print_json(self, capsys):
        _print_json({"key": "value", "num": 42})
        captured = capsys.readouterr()
        obj = json.loads(captured.out)
        assert obj["key"] == "value"
        assert obj["num"] == 42


# ═══════════════════════════════════════════════════════════════════════
# Test ProgressReporter
# ═══════════════════════════════════════════════════════════════════════

class TestProgressReporter:
    """Tests for the progress reporting wrapper."""

    def test_json_mode_suppresses_output(self, capsys):
        pr = _ProgressReporter(3, "Test", use_json=True)
        pr.update(msg="step1")
        pr.update(msg="step2")
        pr.close()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_fallback_without_tqdm(self, capsys):
        # Force tqdm import to fail
        with patch.dict("sys.modules", {"tqdm": None}):
            pr = _ProgressReporter(2, "Test", use_json=False)
            pr.update(msg="step1")
            pr.close()
            captured = capsys.readouterr()
            # stderr should have the fallback output
            assert "step1" in captured.err

    def test_with_tqdm(self):
        """If tqdm is available, no crash."""
        pr = _ProgressReporter(3, "Test", use_json=False)
        pr.update(msg="a")
        pr.update(msg="b")
        pr.close()
        # No assertion needed — just verifying no crash


# ═══════════════════════════════════════════════════════════════════════
# Test cmd_version
# ═══════════════════════════════════════════════════════════════════════

class TestCmdVersion:
    """Tests for the version subcommand."""

    def test_version_text(self, capsys):
        args = build_parser().parse_args(["version"])
        rc = cmd_version(args)
        assert rc == RC_OK
        captured = capsys.readouterr()
        assert "IonFlow Pipeline" in captured.out

    def test_version_json(self, capsys):
        args = build_parser().parse_args(["--json", "version"])
        rc = cmd_version(args)
        assert rc == RC_OK
        obj = json.loads(capsys.readouterr().out)
        assert "version" in obj


# ═══════════════════════════════════════════════════════════════════════
# Test cmd_eis
# ═══════════════════════════════════════════════════════════════════════

class TestCmdEIS:
    """Tests for the EIS subcommand."""

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main.run_eis_pipeline")
    def test_eis_success(self, mock_pipeline, mock_dirs, capsys):
        mock_result = _make_eis_result()
        mock_result.raw_eis = {"a.txt": pd.DataFrame(), "b.txt": pd.DataFrame()}
        mock_result.out_dir = "outputs/tables"
        mock_pipeline.return_value = mock_result
        args = build_parser().parse_args(["eis"])
        rc = cmd_eis(args)
        assert rc == RC_OK
        assert "2 files processed" in capsys.readouterr().out

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main.run_eis_pipeline")
    def test_eis_success_json(self, mock_pipeline, mock_dirs, capsys):
        mock_result = _make_eis_result()
        mock_result.raw_eis = {"a.txt": pd.DataFrame()}
        mock_result.out_dir = "outputs/tables"
        mock_pipeline.return_value = mock_result
        args = build_parser().parse_args(["--json", "eis"])
        rc = cmd_eis(args)
        assert rc == RC_OK
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "success"
        assert obj["pipeline"] == "eis"
        assert obj["files_processed"] == 1

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main.run_eis_pipeline", side_effect=FileNotFoundError("no data"))
    def test_eis_file_not_found(self, mock_pipeline, mock_dirs, capsys):
        args = build_parser().parse_args(["eis"])
        rc = cmd_eis(args)
        assert rc == RC_ERROR
        assert "no data" in capsys.readouterr().err

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main.run_eis_pipeline", side_effect=FileNotFoundError("no data"))
    def test_eis_file_not_found_json(self, mock_pipeline, mock_dirs, capsys):
        args = build_parser().parse_args(["--json", "eis"])
        rc = cmd_eis(args)
        assert rc == RC_ERROR
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "error"

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main.run_eis_pipeline", side_effect=RuntimeError("boom"))
    def test_eis_generic_error(self, mock_pipeline, mock_dirs, capsys):
        args = build_parser().parse_args(["eis"])
        rc = cmd_eis(args)
        assert rc == RC_ERROR

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main.run_eis_pipeline", side_effect=RuntimeError("boom"))
    def test_eis_generic_error_json(self, mock_pipeline, mock_dirs, capsys):
        args = build_parser().parse_args(["--json", "eis"])
        rc = cmd_eis(args)
        assert rc == RC_ERROR
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "error"
        assert "boom" in obj["error"]

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main.run_eis_pipeline")
    def test_eis_with_data_dir(self, mock_pipeline, mock_dirs):
        mock_pipeline.return_value = _make_eis_result()
        mock_pipeline.return_value.raw_eis = {}
        mock_pipeline.return_value.out_dir = "outputs/tables"
        args = build_parser().parse_args(["eis", "--data-dir", "/custom"])
        cmd_eis(args)
        call_args = mock_pipeline.call_args
        cfg_used = call_args[1].get("config") or call_args[0][0] if call_args[0] else call_args[1]["config"]
        assert cfg_used.data_dir == "/custom"


# ═══════════════════════════════════════════════════════════════════════
# Test cmd_cycling
# ═══════════════════════════════════════════════════════════════════════

class TestCmdCycling:
    """Tests for the cycling subcommand."""

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_cycling.run_ciclagem_pipeline")
    def test_cycling_success(self, mock_pipeline, mock_dirs, capsys):
        mock_pipeline.return_value = _make_cycling_result()
        args = build_parser().parse_args(["cycling"])
        rc = cmd_cycling(args)
        assert rc == RC_OK
        assert "1 files processed" in capsys.readouterr().out

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_cycling.run_ciclagem_pipeline")
    def test_cycling_success_json(self, mock_pipeline, mock_dirs, capsys):
        mock_pipeline.return_value = _make_cycling_result()
        args = build_parser().parse_args(["--json", "cycling"])
        rc = cmd_cycling(args)
        assert rc == RC_OK
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "success"

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_cycling.run_ciclagem_pipeline")
    def test_cycling_with_scan_rate(self, mock_pipeline, mock_dirs):
        mock_pipeline.return_value = _make_cycling_result()
        args = build_parser().parse_args(["cycling", "--scan-rate", "0.5"])
        cmd_cycling(args)
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["scan_rate"] == 0.5

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_cycling.run_ciclagem_pipeline", side_effect=FileNotFoundError("missing"))
    def test_cycling_error(self, mock_pipeline, mock_dirs, capsys):
        args = build_parser().parse_args(["cycling"])
        rc = cmd_cycling(args)
        assert rc == RC_ERROR

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_cycling.run_ciclagem_pipeline", side_effect=FileNotFoundError("missing"))
    def test_cycling_error_json(self, mock_pipeline, mock_dirs, capsys):
        args = build_parser().parse_args(["--json", "cycling"])
        rc = cmd_cycling(args)
        assert rc == RC_ERROR
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "error"

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_cycling.run_ciclagem_pipeline", side_effect=RuntimeError("crash"))
    def test_cycling_generic_error(self, mock_pipeline, mock_dirs, capsys):
        args = build_parser().parse_args(["cycling"])
        rc = cmd_cycling(args)
        assert rc == RC_ERROR


# ═══════════════════════════════════════════════════════════════════════
# Test cmd_drt
# ═══════════════════════════════════════════════════════════════════════

class TestCmdDRT:
    """Tests for the DRT subcommand."""

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline")
    def test_drt_success(self, mock_pipeline, mock_dirs, capsys):
        mock_pipeline.return_value = _make_drt_result()
        args = build_parser().parse_args(["drt"])
        rc = cmd_drt(args)
        assert rc == RC_OK
        assert "3 OK" in capsys.readouterr().out

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline")
    def test_drt_with_warnings(self, mock_pipeline, mock_dirs, capsys):
        result = _make_drt_result()
        result.run_meta = {"n_success": 2, "n_failed": 1}
        mock_pipeline.return_value = result
        args = build_parser().parse_args(["drt"])
        rc = cmd_drt(args)
        assert rc == RC_WARNING

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline")
    def test_drt_success_json(self, mock_pipeline, mock_dirs, capsys):
        mock_pipeline.return_value = _make_drt_result()
        args = build_parser().parse_args(["--json", "drt"])
        rc = cmd_drt(args)
        assert rc == RC_OK
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "success"
        assert obj["files_success"] == 3

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline")
    def test_drt_with_lambda(self, mock_pipeline, mock_dirs):
        mock_pipeline.return_value = _make_drt_result()
        args = build_parser().parse_args(["drt", "--lambda", "1e-4"])
        cmd_drt(args)
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["lambda_reg"] == 1e-4

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline", side_effect=FileNotFoundError("nodir"))
    def test_drt_not_found(self, mock_pipeline, mock_dirs, capsys):
        args = build_parser().parse_args(["drt"])
        rc = cmd_drt(args)
        assert rc == RC_ERROR

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline", side_effect=FileNotFoundError("nodir"))
    def test_drt_not_found_json(self, mock_pipeline, mock_dirs, capsys):
        args = build_parser().parse_args(["--json", "drt"])
        rc = cmd_drt(args)
        assert rc == RC_ERROR
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "error"

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline", side_effect=RuntimeError("crash"))
    def test_drt_generic_error(self, mock_pipeline, mock_dirs):
        args = build_parser().parse_args(["drt"])
        rc = cmd_drt(args)
        assert rc == RC_ERROR


# ═══════════════════════════════════════════════════════════════════════
# Test cmd_analyze
# ═══════════════════════════════════════════════════════════════════════

class TestCmdAnalyze:
    """Tests for the analyze subcommand."""

    @patch("src.cli.PipelineConfig.ensure_dirs")
    def test_analyze_no_flags(self, mock_dirs, capsys):
        """Analyze with no --all runs nothing, still returns ok."""
        args = build_parser().parse_args(["analyze"])
        rc = cmd_analyze(args)
        assert rc == RC_OK

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline")
    @patch("main_cycling.run_ciclagem_pipeline")
    @patch("main.run_eis_pipeline")
    def test_analyze_all(self, mock_eis, mock_cyc, mock_drt, mock_dirs, capsys):
        mock_eis.return_value = _make_eis_result()
        mock_cyc.return_value = _make_cycling_result()
        mock_drt.return_value = _make_drt_result()
        args = build_parser().parse_args(["analyze", "--all"])
        rc = cmd_analyze(args)
        assert rc == RC_OK
        out = capsys.readouterr().out
        assert "eis" in out
        assert "cycling" in out
        assert "drt" in out

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline")
    @patch("main_cycling.run_ciclagem_pipeline")
    @patch("main.run_eis_pipeline")
    def test_analyze_all_json(self, mock_eis, mock_cyc, mock_drt, mock_dirs, capsys):
        mock_eis.return_value = _make_eis_result()
        mock_cyc.return_value = _make_cycling_result()
        mock_drt.return_value = _make_drt_result()
        args = build_parser().parse_args(["--json", "analyze", "--all"])
        rc = cmd_analyze(args)
        assert rc == RC_OK
        obj = json.loads(capsys.readouterr().out)
        assert set(obj["pipelines_run"]) == {"eis", "cycling", "drt"}

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline", side_effect=RuntimeError("drt fail"))
    @patch("main_cycling.run_ciclagem_pipeline")
    @patch("main.run_eis_pipeline")
    def test_analyze_partial_failure(self, mock_eis, mock_cyc, mock_drt, mock_dirs, capsys):
        mock_eis.return_value = _make_eis_result()
        mock_cyc.return_value = _make_cycling_result()
        args = build_parser().parse_args(["analyze", "--all"])
        rc = cmd_analyze(args)
        assert rc == RC_WARNING  # partial failure → warning

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main.run_eis_pipeline")
    def test_analyze_with_ai(self, mock_eis, mock_dirs, capsys):
        mock_eis.return_value = _make_eis_result()
        args = build_parser().parse_args(["analyze", "--all", "--ai"])
        # AI analysis may fail on mock data — that's ok, should still not crash
        rc = cmd_analyze(args)
        assert rc in (RC_OK, RC_WARNING)

    @patch("src.cli.PipelineConfig.ensure_dirs")
    def test_analyze_export_pdf_no_module(self, mock_dirs, capsys):
        """If report_generator is not yet available, graceful warning."""
        args = build_parser().parse_args(["analyze", "--export-pdf", "test.pdf"])
        rc = cmd_analyze(args)
        assert rc in (RC_OK, RC_WARNING)

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline")
    @patch("main_cycling.run_ciclagem_pipeline")
    @patch("main.run_eis_pipeline")
    def test_analyze_all_pipelines_fail(self, mock_eis, mock_cyc, mock_drt, mock_dirs):
        mock_eis.side_effect = RuntimeError("eis fail")
        mock_cyc.side_effect = RuntimeError("cyc fail")
        mock_drt.side_effect = RuntimeError("drt fail")
        args = build_parser().parse_args(["analyze", "--all"])
        rc = cmd_analyze(args)
        assert rc == RC_WARNING  # all pipeline failures = warnings, not errors

    @patch("src.cli.PipelineConfig.ensure_dirs", side_effect=RuntimeError("outer"))
    def test_analyze_outer_exception(self, mock_dirs, capsys):
        args = build_parser().parse_args(["analyze"])
        rc = cmd_analyze(args)
        assert rc == RC_ERROR

    @patch("src.cli.PipelineConfig.ensure_dirs", side_effect=RuntimeError("outer"))
    def test_analyze_outer_exception_json(self, mock_dirs, capsys):
        args = build_parser().parse_args(["--json", "analyze"])
        rc = cmd_analyze(args)
        assert rc == RC_ERROR
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "error"


# ═══════════════════════════════════════════════════════════════════════
# Test cmd_config
# ═══════════════════════════════════════════════════════════════════════

class TestCmdConfig:
    """Tests for the config subcommand."""

    def test_config_init(self, tmp_path, capsys):
        out_path = str(tmp_path / "new_config.json")
        args = build_parser().parse_args(["config", "--init", "--output", out_path])
        rc = cmd_config(args)
        assert rc == RC_OK
        assert Path(out_path).exists()
        data = json.loads(Path(out_path).read_text())
        assert "data_dir" in data

    def test_config_init_json(self, tmp_path, capsys):
        out_path = str(tmp_path / "new_config.json")
        args = build_parser().parse_args(["--json", "config", "--init", "--output", out_path])
        rc = cmd_config(args)
        assert rc == RC_OK
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "success"
        assert obj["action"] == "config_init"

    def test_config_show_default(self, capsys):
        args = build_parser().parse_args(["config", "--show"])
        rc = cmd_config(args)
        assert rc == RC_OK
        out = capsys.readouterr().out
        assert "data_dir" in out

    def test_config_show_json(self, capsys):
        args = build_parser().parse_args(["--json", "config", "--show"])
        rc = cmd_config(args)
        assert rc == RC_OK
        obj = json.loads(capsys.readouterr().out)
        assert "data_dir" in obj

    def test_config_show_from_file(self, config_file, capsys):
        args = build_parser().parse_args(["--config", config_file, "config", "--show"])
        rc = cmd_config(args)
        assert rc == RC_OK

    def test_config_no_action(self, capsys):
        args = build_parser().parse_args(["config"])
        rc = cmd_config(args)
        assert rc == RC_ERROR

    def test_config_no_action_json(self, capsys):
        args = build_parser().parse_args(["--json", "config"])
        rc = cmd_config(args)
        assert rc == RC_ERROR
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "error"

    def test_config_init_default_path(self, tmp_path, monkeypatch, capsys):
        """--init without --output writes to config.json in cwd."""
        monkeypatch.chdir(tmp_path)
        args = build_parser().parse_args(["config", "--init"])
        rc = cmd_config(args)
        assert rc == RC_OK
        assert (tmp_path / "config.json").exists()

    def test_config_roundtrip(self, tmp_path):
        """Config init → show from file should be consistent."""
        out_path = str(tmp_path / "rt.json")
        args1 = build_parser().parse_args(["config", "--init", "--output", out_path])
        cmd_config(args1)
        loaded = PipelineConfig.from_json(out_path)
        default = PipelineConfig.default()
        assert loaded.data_dir == default.data_dir
        assert loaded.drt_lambda == default.drt_lambda


# ═══════════════════════════════════════════════════════════════════════
# Test cmd_validate
# ═══════════════════════════════════════════════════════════════════════

class TestCmdValidate:
    """Tests for the validate subcommand."""

    def test_validate_dir_not_found(self, capsys):
        args = build_parser().parse_args(["validate", "--data-dir", "/nonexistent/path"])
        rc = cmd_validate(args)
        assert rc == RC_ERROR

    def test_validate_dir_not_found_json(self, capsys):
        args = build_parser().parse_args(["--json", "validate", "--data-dir", "/nonexistent/path"])
        rc = cmd_validate(args)
        assert rc == RC_ERROR
        obj = json.loads(capsys.readouterr().out)
        assert obj["status"] == "error"

    def test_validate_no_txt_files(self, tmp_path, capsys):
        args = build_parser().parse_args(["validate", "--data-dir", str(tmp_path)])
        rc = cmd_validate(args)
        assert rc == RC_ERROR

    def test_validate_no_txt_files_json(self, tmp_path, capsys):
        args = build_parser().parse_args(["--json", "validate", "--data-dir", str(tmp_path)])
        rc = cmd_validate(args)
        assert rc == RC_ERROR
        obj = json.loads(capsys.readouterr().out)
        assert "No .txt" in obj["error"]

    @patch("src.kramers_kronig.KramersKronigValidator")
    @patch("src.validation.validate_eis_full")
    @patch("src.preprocessing.preprocess")
    @patch("src.loader.load_eis_file")
    def test_validate_all_valid(self, mock_load, mock_preprocess, mock_validate, mock_kk_cls, tmp_dir, capsys):
        mock_load.return_value = pd.DataFrame({"frequency": [1.0], "zreal": [1.0], "zimag": [-1.0]})
        mock_preprocess.return_value = pd.DataFrame({"frequency": [1.0], "zreal": [1.0], "zimag": [-1.0]})
        vr = MagicMock()
        vr.ok = True
        vr.errors = []
        vr.warnings = []
        mock_validate.return_value = vr

        kk_result = MagicMock()
        kk_result.kk_valid = True
        kk_result.classification = "excelente"
        kk_result.mean_residual = 0.001
        kk_result.max_residual = 0.005
        kk_instance = MagicMock()
        kk_instance.validate.return_value = kk_result
        mock_kk_cls.return_value = kk_instance

        args = build_parser().parse_args(["validate", "--data-dir", str(tmp_dir)])
        rc = cmd_validate(args)
        assert rc == RC_OK

    @patch("src.kramers_kronig.KramersKronigValidator")
    @patch("src.validation.validate_eis_full")
    @patch("src.preprocessing.preprocess")
    @patch("src.loader.load_eis_file")
    def test_validate_all_valid_json(self, mock_load, mock_preprocess, mock_validate, mock_kk_cls, tmp_dir, capsys):
        mock_load.return_value = pd.DataFrame({"frequency": [1.0], "zreal": [1.0], "zimag": [-1.0]})
        mock_preprocess.return_value = pd.DataFrame({"frequency": [1.0], "zreal": [1.0], "zimag": [-1.0]})
        vr = MagicMock()
        vr.ok = True
        vr.errors = []
        vr.warnings = []
        mock_validate.return_value = vr

        kk_result = MagicMock()
        kk_result.kk_valid = True
        kk_result.classification = "excelente"
        kk_result.mean_residual = 0.001
        kk_result.max_residual = 0.005
        kk_instance = MagicMock()
        kk_instance.validate.return_value = kk_result
        mock_kk_cls.return_value = kk_instance

        args = build_parser().parse_args(["--json", "validate", "--data-dir", str(tmp_dir)])
        rc = cmd_validate(args)
        assert rc == RC_OK
        obj = json.loads(capsys.readouterr().out)
        assert obj["valid"] == 3
        assert obj["invalid"] == 0

    @patch("src.kramers_kronig.KramersKronigValidator")
    @patch("src.validation.validate_eis_full")
    @patch("src.preprocessing.preprocess")
    @patch("src.loader.load_eis_file")
    def test_validate_some_invalid(self, mock_load, mock_preprocess, mock_validate, mock_kk_cls, tmp_dir, capsys):
        mock_load.return_value = pd.DataFrame({"frequency": [1.0], "zreal": [1.0], "zimag": [-1.0]})
        mock_preprocess.return_value = pd.DataFrame({"frequency": [1.0], "zreal": [1.0], "zimag": [-1.0]})
        vr = MagicMock()
        vr.ok = False
        vr.errors = ["bad col"]
        vr.warnings = []
        mock_validate.return_value = vr

        kk_result = MagicMock()
        kk_result.kk_valid = False
        kk_result.classification = "suspeito"
        kk_result.mean_residual = 0.1
        kk_result.max_residual = 0.3
        kk_instance = MagicMock()
        kk_instance.validate.return_value = kk_result
        mock_kk_cls.return_value = kk_instance

        args = build_parser().parse_args(["validate", "--data-dir", str(tmp_dir)])
        rc = cmd_validate(args)
        assert rc == RC_WARNING

    @patch("src.loader.load_eis_file", side_effect=Exception("parse error"))
    def test_validate_file_errors(self, mock_load, tmp_dir, capsys):
        args = build_parser().parse_args(["validate", "--data-dir", str(tmp_dir)])
        rc = cmd_validate(args)
        assert rc == RC_ERROR

    @patch("src.loader.load_eis_file", side_effect=Exception("parse error"))
    def test_validate_file_errors_json(self, mock_load, tmp_dir, capsys):
        args = build_parser().parse_args(["--json", "validate", "--data-dir", str(tmp_dir)])
        rc = cmd_validate(args)
        assert rc == RC_ERROR
        obj = json.loads(capsys.readouterr().out)
        assert obj["errors"] == 3


# ═══════════════════════════════════════════════════════════════════════
# Test main() entry-point
# ═══════════════════════════════════════════════════════════════════════

class TestMain:
    """Tests for the main() dispatching function."""

    def test_no_command(self, capsys):
        rc = main([])
        assert rc == RC_ERROR

    def test_version_command(self, capsys):
        rc = main(["version"])
        assert rc == RC_OK
        assert "IonFlow" in capsys.readouterr().out

    def test_json_version(self, capsys):
        rc = main(["--json", "version"])
        assert rc == RC_OK
        obj = json.loads(capsys.readouterr().out)
        assert "version" in obj

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main.run_eis_pipeline")
    def test_main_eis(self, mock_pipeline, mock_dirs, capsys):
        result = _make_eis_result()
        result.raw_eis = {}
        result.out_dir = "outputs/tables"
        mock_pipeline.return_value = result
        rc = main(["eis"])
        assert rc == RC_OK

    def test_main_config_init(self, tmp_path, capsys):
        out = str(tmp_path / "test_cfg.json")
        rc = main(["config", "--init", "--output", out])
        assert rc == RC_OK
        assert Path(out).exists()

    def test_verbose_flag(self, capsys):
        """Verbose flag doesn't crash."""
        rc = main(["--verbose", "version"])
        assert rc == RC_OK


# ═══════════════════════════════════════════════════════════════════════
# Test return codes
# ═══════════════════════════════════════════════════════════════════════

class TestReturnCodes:
    """Verify return code constants and semantics."""

    def test_rc_values(self):
        assert RC_OK == 0
        assert RC_ERROR == 1
        assert RC_WARNING == 2

    def test_rc_distinct(self):
        assert len({RC_OK, RC_ERROR, RC_WARNING}) == 3


# ═══════════════════════════════════════════════════════════════════════
# Test edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_parser_epilog_has_examples(self):
        parser = build_parser()
        assert "ionflow eis" in parser.epilog
        assert "ionflow cycling" in parser.epilog
        assert "ionflow config --init" in parser.epilog

    def test_config_file_not_found_uses_default(self):
        args = build_parser().parse_args(["--config", "/nonexistent/config.json", "version"])
        cfg = _load_config(args)
        # Should fall back to default
        assert cfg.data_dir == "data/raw"

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main.run_eis_pipeline")
    def test_eis_none_pca(self, mock_pipeline, mock_dirs, capsys):
        """EIS result with no PCA shouldn't crash JSON output."""
        result = _make_eis_result()
        result.raw_eis = {"a.txt": pd.DataFrame()}
        result.out_dir = "outputs/tables"
        result.pca = None
        mock_pipeline.return_value = result
        args = build_parser().parse_args(["--json", "eis"])
        rc = cmd_eis(args)
        assert rc == RC_OK
        obj = json.loads(capsys.readouterr().out)
        assert obj["has_pca"] is False

    def test_validate_with_mixed_files(self, tmp_path):
        """Directory with both .txt and non-.txt files."""
        (tmp_path / "good.txt").write_text("data")
        (tmp_path / "readme.md").write_text("docs")
        (tmp_path / "data.csv").write_text("csv")
        args = build_parser().parse_args(["validate", "--data-dir", str(tmp_path)])
        # Will fail on parsing but should only try .txt files
        with patch("src.loader.load_eis_file", side_effect=Exception("parse")):
            rc = cmd_validate(args)
        assert rc == RC_ERROR  # 1 file errored

    @patch("src.cli.PipelineConfig.ensure_dirs")
    @patch("main_drt.run_drt_pipeline")
    @patch("main_cycling.run_ciclagem_pipeline")
    @patch("main.run_eis_pipeline")
    def test_analyze_json_output_structure(self, mock_eis, mock_cyc, mock_drt, mock_dirs, capsys):
        mock_eis.return_value = _make_eis_result()
        mock_cyc.return_value = _make_cycling_result()
        mock_drt.return_value = _make_drt_result()
        args = build_parser().parse_args(["--json", "analyze", "--all"])
        cmd_analyze(args)
        obj = json.loads(capsys.readouterr().out)
        assert "status" in obj
        assert "pipelines_run" in obj
        assert "warnings" in obj
        assert "errors" in obj
        assert "ai_summary" in obj
        assert "export_pdf" in obj
