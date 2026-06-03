"""Tests for v0.4.11 new modules."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════════
# Tests for VAL-03: Auto-detect encoding and separator
# ═══════════════════════════════════════════════════════════════════════


class TestEncodingDetection:
    """Test _detect_encoding from src.loader."""

    def test_utf8_file(self, tmp_path):
        from src.loader import _detect_encoding

        f = tmp_path / "test.csv"
        f.write_text("frequency;zreal;zimag\n100;5.0;-2.0\n", encoding="utf-8")
        assert _detect_encoding(str(f)) == "utf-8"

    def test_latin1_file(self, tmp_path):
        from src.loader import _detect_encoding

        f = tmp_path / "test.csv"
        # Latin-1 specific char (ã)
        f.write_bytes("frequência;zreal;zimag\n100;5.0;-2.0\n".encode("latin-1"))
        enc = _detect_encoding(str(f))
        assert enc in ("latin-1", "iso-8859-1")

    def test_utf8_bom(self, tmp_path):
        from src.loader import _detect_encoding

        f = tmp_path / "test.csv"
        content = "\ufefffrequency;zreal;zimag\n100;5.0;-2.0\n"
        f.write_text(content, encoding="utf-8-sig")
        assert _detect_encoding(str(f)) in ("utf-8-sig", "utf-8")


class TestDelimiterSniffing:
    """Test _sniff_delimiter from src.loader."""

    def test_semicolon(self, tmp_path):
        from src.loader import _sniff_delimiter

        f = tmp_path / "test.csv"
        f.write_text("frequency;zreal;zimag\n100;5.0;-2.0\n50;4.0;-1.5\n")
        assert _sniff_delimiter(str(f), "utf-8") == ";"

    def test_comma(self, tmp_path):
        from src.loader import _sniff_delimiter

        f = tmp_path / "test.csv"
        f.write_text("frequency,zreal,zimag\n100,5.0,-2.0\n50,4.0,-1.5\n")
        assert _sniff_delimiter(str(f), "utf-8") == ","

    def test_tab(self, tmp_path):
        from src.loader import _sniff_delimiter

        f = tmp_path / "test.csv"
        f.write_text("frequency\tzreal\tzimag\n100\t5.0\t-2.0\n50\t4.0\t-1.5\n")
        assert _sniff_delimiter(str(f), "utf-8") == "\t"

    def test_comment_lines_skipped(self, tmp_path):
        from src.loader import _sniff_delimiter

        f = tmp_path / "test.csv"
        f.write_text("# header comment\n# another\nfrequency;zreal;zimag\n100;5.0;-2.0\n")
        assert _sniff_delimiter(str(f), "utf-8") == ";"


# ═══════════════════════════════════════════════════════════════════════
# Tests for VAL-02: EISLoadError
# ═══════════════════════════════════════════════════════════════════════


class TestEISLoadError:
    """Test the researcher-friendly error class."""

    def test_is_valueerror(self):
        from src.loader import EISLoadError

        err = EISLoadError("test message", path="/test.csv")
        assert isinstance(err, ValueError)
        assert err.path == "/test.csv"
        assert "test message" in str(err)

    def test_attributes(self):
        from src.loader import EISLoadError

        err = EISLoadError(
            "msg",
            path="/data/test.csv",
            detected_encoding="latin-1",
            detected_separator=";",
            columns_found=2,
        )
        assert err.detected_encoding == "latin-1"
        assert err.detected_separator == ";"
        assert err.columns_found == 2

    def test_raised_on_bad_file(self, tmp_path):
        from src.loader import load_eis_file, EISLoadError

        f = tmp_path / "bad.csv"
        f.write_text("only one column\nvalue1\nvalue2\n")
        with pytest.raises((ValueError, EISLoadError)):
            load_eis_file(str(f))


# ═══════════════════════════════════════════════════════════════════════
# Tests for UX-03: Material Presets
# ═══════════════════════════════════════════════════════════════════════


class TestMaterialPresets:
    """Test PipelineConfig material presets."""

    def test_all_presets_exist(self):
        from src.config import PipelineConfig

        cfg = PipelineConfig.default()
        expected = {"supercapacitor", "li_ion", "corrosion_coating", "fuel_cell", "generic"}
        assert set(cfg.MATERIAL_PRESETS.keys()) == expected

    def test_apply_preset_changes_drt(self):
        from src.config import PipelineConfig

        cfg = PipelineConfig.default()
        original_lambda = cfg.drt_lambda
        cfg.apply_material_preset("li_ion")
        assert cfg.drt_lambda == 5e-4
        assert cfg.drt_n_taus == 80
        assert cfg.material_preset == "li_ion"

    def test_apply_unknown_preset_is_noop(self):
        from src.config import PipelineConfig

        cfg = PipelineConfig.default()
        original = cfg.drt_lambda
        cfg.apply_material_preset("nonexistent")
        assert cfg.drt_lambda == original

    def test_preset_has_required_keys(self):
        from src.config import PipelineConfig

        cfg = PipelineConfig.default()
        required_keys = {"label", "description", "drt_lambda", "drt_n_taus",
                         "preferred_circuits", "quality_thresholds",
                         "expected_rs_range", "expected_rp_range", "expected_n_range"}
        for name, preset in cfg.MATERIAL_PRESETS.items():
            assert required_keys.issubset(set(preset.keys())), f"Missing keys in {name}"


# ═══════════════════════════════════════════════════════════════════════
# Tests for VIZ-04: Journal Styles
# ═══════════════════════════════════════════════════════════════════════


class TestJournalStyles:
    """Test journal figure style presets."""

    def test_get_available_styles(self):
        from src.journal_styles import get_available_styles

        styles = get_available_styles()
        assert "acs" in styles
        assert "nature" in styles
        assert "elsevier" in styles
        assert "rsc" in styles
        assert "ionflow" in styles

    def test_apply_style(self):
        from src.journal_styles import apply_journal_style
        import matplotlib.pyplot as plt

        apply_journal_style("acs")
        assert plt.rcParams["font.size"] == 8

    def test_invalid_style_raises(self):
        from src.journal_styles import apply_journal_style

        with pytest.raises(ValueError, match="não reconhecido"):
            apply_journal_style("nonexistent_journal")

    def test_context_manager_restores(self):
        from src.journal_styles import journal_style_context
        import matplotlib.pyplot as plt

        original_size = plt.rcParams["font.size"]
        with journal_style_context("nature"):
            assert plt.rcParams["font.size"] == 7
        assert plt.rcParams["font.size"] == original_size

    def test_style_labels(self):
        from src.journal_styles import STYLE_LABELS, JOURNAL_STYLES

        # Every style should have a label
        for key in JOURNAL_STYLES:
            assert key in STYLE_LABELS


# ═══════════════════════════════════════════════════════════════════════
# Tests for VIZ-03: Figure Pack Export
# ═══════════════════════════════════════════════════════════════════════


class TestFigurePack:
    """Test figure pack export."""

    def test_export_creates_files(self, tmp_path):
        from src.figure_pack import export_figure_pack
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        data = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})

        result = export_figure_pack(
            figures={"test_plot": (fig, data)},
            output_dir=tmp_path / "pack",
        )
        assert "test_plot" in result
        files = result["test_plot"]
        assert any(f.endswith(".png") for f in files)
        assert any(f.endswith(".svg") for f in files)
        assert any(f.endswith("_data.csv") for f in files)

    def test_export_no_data(self, tmp_path):
        from src.figure_pack import export_figure_pack
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])

        result = export_figure_pack(
            figures={"no_data": (fig, None)},
            output_dir=tmp_path / "pack",
        )
        assert "no_data" in result
        assert len(result["no_data"]) == 2  # png + svg only

    def test_custom_formats(self, tmp_path):
        from src.figure_pack import export_figure_pack
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1], [1])

        result = export_figure_pack(
            figures={"minimal": (fig, None)},
            output_dir=tmp_path / "pack",
            formats=["png"],
        )
        assert len(result["minimal"]) == 1


# ═══════════════════════════════════════════════════════════════════════
# Tests for AI-01: Auto Summary
# ═══════════════════════════════════════════════════════════════════════


class TestAutoSummary:
    """Test automatic executive summary generation."""

    def test_empty_results(self):
        from src.ai.auto_summary import generate_auto_summary

        summary = generate_auto_summary()
        assert "Resumo automático" in summary
        assert "Nenhum resultado" in summary

    def test_eis_summary(self):
        from src.ai.auto_summary import generate_auto_summary

        eis_result = {
            "results_df": pd.DataFrame({
                "Rs_fit": [2.5, 3.0],
                "Rp_fit": [100.0, 120.0],
                "chi2_over_nu": [0.005, 0.008],
            }),
            "best_circuit": "Randles_CPE",
        }
        summary = generate_auto_summary(eis_result=eis_result)
        assert "EIS" in summary
        assert "Randles_CPE" in summary
        assert "excelente" in summary  # chi2 < 0.01

    def test_cycling_summary(self):
        from src.ai.auto_summary import generate_auto_summary

        cycling_result = {
            "results_df": pd.DataFrame({
                "cycle": [1, 2, 3, 100],
                "energy": [10.0, 9.8, 9.5, 8.0],
                "retention": [100, 98, 95, 80],
            }),
        }
        summary = generate_auto_summary(cycling_result=cycling_result)
        assert "Ciclagem" in summary


# ═══════════════════════════════════════════════════════════════════════
# Tests for AI-02: Next-Step Suggestions
# ═══════════════════════════════════════════════════════════════════════


class TestNextSteps:
    """Test automatic next-step suggestions."""

    def test_high_rs_suggestion(self):
        from src.ai.auto_summary import generate_next_steps

        result = {
            "results_df": pd.DataFrame({
                "Rs_fit": [15.0, 20.0],
                "n": [0.85, 0.90],
                "chi2_over_nu": [0.01, 0.02],
            }),
        }
        steps = generate_next_steps(eis_result=result)
        assert any("Rs" in s for s in steps)

    def test_low_n_suggestion(self):
        from src.ai.auto_summary import generate_next_steps

        result = {
            "results_df": pd.DataFrame({
                "Rs_fit": [2.0],
                "n": [0.65],
                "chi2_over_nu": [0.01],
            }),
        }
        steps = generate_next_steps(eis_result=result)
        assert any("arco deprimido" in s or "CPE" in s for s in steps)

    def test_no_results_gives_general_suggestion(self):
        from src.ai.auto_summary import generate_next_steps

        steps = generate_next_steps()
        assert len(steps) >= 1
        assert any("Análise IA" in s for s in steps)


# ═══════════════════════════════════════════════════════════════════════
# Tests for DEV-04: Log Level Configuration
# ═══════════════════════════════════════════════════════════════════════


class TestLogLevelConfig:
    """Test configurable log levels."""

    def test_presets_exist(self):
        from src.logger import LOG_LEVEL_PRESETS

        assert "silent" in LOG_LEVEL_PRESETS
        assert "normal" in LOG_LEVEL_PRESETS
        assert "debug" in LOG_LEVEL_PRESETS

    def test_set_log_level_no_crash(self):
        from src.logger import set_log_level, setup_logging

        setup_logging(force=True)
        # Should not raise
        set_log_level("silent")
        set_log_level("normal")
        set_log_level("debug")
        set_log_level("unknown_falls_back_to_normal")


# ═══════════════════════════════════════════════════════════════════════
# Tests for UX-01: Wizard utilities
# ═══════════════════════════════════════════════════════════════════════


class TestWizardUtils:
    """Test wizard helper functions (non-GUI)."""

    def test_should_show_wizard_no_file(self, tmp_path):
        from src.gui.wizard import should_show_wizard

        assert should_show_wizard(str(tmp_path / "nonexistent.json")) is True

    def test_should_show_wizard_after_completion(self, tmp_path):
        from src.gui.wizard import should_show_wizard, mark_wizard_completed

        settings = str(tmp_path / "settings.json")
        mark_wizard_completed(settings)
        assert should_show_wizard(settings) is False

    def test_wizard_result_defaults(self):
        from src.gui.wizard import WizardResult

        r = WizardResult()
        assert r.language == "pt"
        assert r.completed is False
        assert r.material_preset == "generic"


# ═══════════════════════════════════════════════════════════════════════
# Tests for loader with auto-detection integration
# ═══════════════════════════════════════════════════════════════════════


class TestLoaderAutoDetect:
    """Integration tests for the improved loader."""

    def test_semicolon_utf8(self, tmp_path):
        from src.loader import load_eis_file

        f = tmp_path / "test.csv"
        f.write_text(
            "Frequency (Hz);Z' (Ohm);-Z'' (Ohm)\n"
            "1000;5.0;2.0\n500;6.0;3.0\n100;8.0;5.0\n"
        )
        df = load_eis_file(str(f))
        assert len(df) == 3
        assert list(df.columns) == ["frequency", "zreal", "zimag"]

    def test_comma_separated(self, tmp_path):
        from src.loader import load_eis_file

        f = tmp_path / "test.csv"
        f.write_text(
            "Frequency (Hz),Z' (Ohm),-Z'' (Ohm)\n"
            "1000,5.0,2.0\n500,6.0,3.0\n"
        )
        df = load_eis_file(str(f))
        assert len(df) == 2

    def test_tab_separated(self, tmp_path):
        from src.loader import load_eis_file

        f = tmp_path / "test.csv"
        f.write_text(
            "Frequency (Hz)\tZ' (Ohm)\t-Z'' (Ohm)\n"
            "1000\t5.0\t2.0\n500\t6.0\t3.0\n"
        )
        df = load_eis_file(str(f))
        assert len(df) == 2

    def test_latin1_encoding(self, tmp_path):
        from src.loader import load_eis_file

        f = tmp_path / "test.csv"
        # Write with Latin-1 encoding (common in European labs)
        content = "Frequência (Hz);Z' (Ohm);-Z'' (Ohm)\n1000;5.0;2.0\n500;6.0;3.0\n"
        f.write_bytes(content.encode("latin-1"))
        df = load_eis_file(str(f))
        assert len(df) == 2

    def test_decimal_comma(self, tmp_path):
        from src.loader import load_eis_file

        f = tmp_path / "test.csv"
        f.write_text(
            "Frequency (Hz);Z' (Ohm);-Z'' (Ohm)\n"
            "1000;5,0;2,0\n500;6,0;3,0\n"
        )
        df = load_eis_file(str(f))
        assert len(df) == 2
        assert df["zreal"].iloc[0] == 5.0
