"""Tests for the ReportGenerator (Day 23).

Covers:
- ReportConfig defaults and customisation
- GenerationHistory CRUD
- Section builders (EIS, cycling, DRT, correlations)
- Markdown generation
- PDF generation with fpdf2
- LaTeX/DOCX stubs
- Multi-format dispatch
- AI text enhancement path
- _clean_text helper
- _safe_str helper
- _IonFlowPDF internals
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ═══════════════════════════════════════════════════════════════════
# Fixtures & helpers
# ═══════════════════════════════════════════════════════════════════


@dataclass
class _FakePCA:
    figure_paths: List[str] = field(default_factory=list)


@dataclass
class _FakeEIS:
    features_df: Optional[pd.DataFrame] = None
    ranked_df: Optional[pd.DataFrame] = None
    cap_energy_df: Optional[pd.DataFrame] = None
    circuit_table: Optional[pd.DataFrame] = None
    circuit_summary: Optional[str] = None
    pca: Optional[_FakePCA] = None
    stability: Optional[Any] = None
    raw_eis: Optional[Dict] = None
    out_dir: str = ""
    config_used: Optional[Any] = None


@dataclass
class _FakeCycling:
    results: Optional[Dict] = None
    export_tables: Optional[Dict] = None
    merged_table: Optional[pd.DataFrame] = None
    plot_paths: Optional[List] = None
    energy_power_paths: Optional[List] = None


@dataclass
class _FakeDRT:
    drt_table: Optional[pd.DataFrame] = None
    drt_peaks_table: Optional[pd.DataFrame] = None
    drt_summary_table: Optional[pd.DataFrame] = None
    per_file_results: Optional[Dict] = None
    plot_paths: Optional[List] = None
    errors: Optional[List] = None
    run_meta: Optional[Dict] = None


def _make_ranked_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "Sample": [f"sample_{i}" for i in range(n)],
            "Rs_fit": rng.uniform(0.1, 10, n),
            "Rp_fit": rng.uniform(10, 100, n),
            "Q": rng.uniform(1e-6, 1e-3, n),
            "n": rng.uniform(0.7, 1.0, n),
            "Score": rng.uniform(0, 1, n),
            "Rank": list(range(1, n + 1)),
        }
    )


def _make_circuit_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Circuito": ["R(RC)", "R(RC)", "R(RQ)", "R(RC)", "R(RQ)"],
            "R_s": [1.0, 1.1, 0.9, 1.2, 0.8],
            "R_ct": [50, 55, 48, 60, 45],
        }
    )


def _make_merged_table() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "File": [f"cell_{i}" for i in range(5)],
            "Energy_Wh": rng.uniform(10, 100, 5),
            "Power_W": rng.uniform(5, 50, 5),
        }
    )


def _make_drt_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "File": ["drt_1", "drt_2", "drt_3"],
            "N_peaks": [3, 2, 4],
            "R_total": [100, 120, 95],
        }
    )


def _make_drt_peaks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "File": ["drt_1", "drt_1", "drt_2"],
            "tau": [1e-3, 1e-1, 1e-2],
            "R_peak": [30, 40, 50],
        }
    )


@pytest.fixture
def full_pipeline_results():
    """Pipeline results with all three analyses present."""
    ranked = _make_ranked_df()
    eis = _FakeEIS(
        ranked_df=ranked,
        circuit_table=_make_circuit_table(),
        raw_eis={"file1": {}, "file2": {}, "file3": {}},
        pca=_FakePCA(figure_paths=[]),
    )
    cyc = _FakeCycling(
        merged_table=_make_merged_table(),
        results={"c1": {}, "c2": {}},
        energy_power_paths=[],
    )
    drt = _FakeDRT(
        drt_summary_table=_make_drt_summary(),
        drt_peaks_table=_make_drt_peaks(),
        plot_paths=[],
        run_meta={"n_success": 3, "n_failed": 0},
    )
    return {"eis": eis, "cycling": cyc, "drt": drt}


@pytest.fixture
def eis_only_results():
    """Pipeline results with only EIS."""
    eis = _FakeEIS(
        ranked_df=_make_ranked_df(5),
        circuit_table=_make_circuit_table(),
        raw_eis={"f1": {}, "f2": {}},
    )
    return {"eis": eis}


@pytest.fixture
def empty_results():
    """Pipeline results with no analyses."""
    return {}


@pytest.fixture
def tmp_out(tmp_path):
    """Temporary output directory."""
    return str(tmp_path / "report")


# ═══════════════════════════════════════════════════════════════════
# ReportConfig
# ═══════════════════════════════════════════════════════════════════


class TestReportConfig:
    def test_defaults(self):
        from src.report_generator import ReportConfig

        cfg = ReportConfig()
        assert cfg.title == "IonFlow Pipeline - Analysis Report"
        assert cfg.page_size == "A4"
        assert cfg.margin == 15.0
        assert cfg.include_eis is True
        assert cfg.output_formats == ["pdf"]

    def test_custom_values(self):
        from src.report_generator import ReportConfig

        cfg = ReportConfig(
            title="Custom Title",
            author="Dr. Test",
            page_size="Letter",
            include_cycling=False,
            max_table_rows=50,
        )
        assert cfg.title == "Custom Title"
        assert cfg.author == "Dr. Test"
        assert cfg.page_size == "Letter"
        assert cfg.include_cycling is False
        assert cfg.max_table_rows == 50

    def test_color_tuples(self):
        from src.report_generator import ReportConfig

        cfg = ReportConfig(color_primary=(255, 0, 0))
        assert cfg.color_primary == (255, 0, 0)

    def test_output_formats_list(self):
        from src.report_generator import ReportConfig

        cfg = ReportConfig(output_formats=["pdf", "markdown", "latex"])
        assert "latex" in cfg.output_formats


# ═══════════════════════════════════════════════════════════════════
# GenerationHistory
# ═══════════════════════════════════════════════════════════════════


class TestGenerationHistory:
    def test_empty_history(self, tmp_path):
        from src.report_generator import GenerationHistory

        h = GenerationHistory(str(tmp_path / "hist.json"))
        assert len(h) == 0
        assert h.records == []

    def test_add_record(self, tmp_path):
        from src.report_generator import GenerationHistory, GenerationRecord

        h = GenerationHistory(str(tmp_path / "hist.json"))
        rec = GenerationRecord(
            timestamp="2024-01-01T00:00:00",
            output_path="/tmp/report.pdf",
            output_format="pdf",
            sections=["eis", "cycling"],
            config_snapshot={"title": "test"},
        )
        version = h.add(rec)
        assert version == 1
        assert len(h) == 1

    def test_versioning_same_path(self, tmp_path):
        from src.report_generator import GenerationHistory, GenerationRecord

        h = GenerationHistory(str(tmp_path / "hist.json"))
        for _ in range(3):
            rec = GenerationRecord(
                timestamp="2024-01-01T00:00:00",
                output_path="/tmp/report.pdf",
                output_format="pdf",
                sections=["eis"],
                config_snapshot={},
            )
            v = h.add(rec)
        assert v == 3
        assert len(h) == 3

    def test_versioning_different_paths(self, tmp_path):
        from src.report_generator import GenerationHistory, GenerationRecord

        h = GenerationHistory(str(tmp_path / "hist.json"))
        for i in range(2):
            rec = GenerationRecord(
                timestamp="2024-01-01T00:00:00",
                output_path=f"/tmp/report_{i}.pdf",
                output_format="pdf",
                sections=["eis"],
                config_snapshot={},
            )
            v = h.add(rec)
        # Each path is version 1
        assert v == 1

    def test_persistence(self, tmp_path):
        from src.report_generator import GenerationHistory, GenerationRecord

        path = str(tmp_path / "hist.json")
        h1 = GenerationHistory(path)
        rec = GenerationRecord(
            timestamp="2024-01-01T00:00:00",
            output_path="/tmp/report.pdf",
            output_format="pdf",
            sections=["eis"],
            config_snapshot={},
        )
        h1.add(rec)

        # Reload
        h2 = GenerationHistory(path)
        assert len(h2) == 1
        assert h2.records[0]["output_path"] == "/tmp/report.pdf"

    def test_corrupt_file_recovery(self, tmp_path):
        from src.report_generator import GenerationHistory

        path = tmp_path / "hist.json"
        path.write_text("INVALID JSON!!!", encoding="utf-8")
        h = GenerationHistory(str(path))
        assert len(h) == 0  # gracefully recovered


# ═══════════════════════════════════════════════════════════════════
# _safe_str & _clean_text
# ═══════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_safe_str_none(self):
        from src.report_generator import _safe_str

        assert _safe_str(None) == "—"

    def test_safe_str_nan(self):
        from src.report_generator import _safe_str

        assert _safe_str(float("nan")) == "—"

    def test_safe_str_inf(self):
        from src.report_generator import _safe_str

        assert _safe_str(float("inf")) == "—"

    def test_safe_str_small_float(self):
        from src.report_generator import _safe_str

        result = _safe_str(1.23e-5)
        assert "e" in result.lower()

    def test_safe_str_normal_float(self):
        from src.report_generator import _safe_str

        result = _safe_str(3.14159)
        assert result == "3.1416"

    def test_safe_str_string(self):
        from src.report_generator import _safe_str

        assert _safe_str("hello") == "hello"

    def test_clean_text_markdown(self):
        from src.report_generator import _clean_text

        assert _clean_text("**bold** and *italic*") == "bold and italic"

    def test_clean_text_unicode_replacements(self):
        from src.report_generator import _clean_text

        text = _clean_text("\u2022 bullet \u2013 dash \u00b1 pm")
        assert "bullet" in text
        assert "-" in text

    def test_clean_text_omega(self):
        from src.report_generator import _clean_text

        assert "Ohm" in _clean_text("50 \u03a9")


# ═══════════════════════════════════════════════════════════════════
# Section builders
# ═══════════════════════════════════════════════════════════════════


class TestBuildEISSection:
    def test_with_data(self, full_pipeline_results):
        from src.report_generator import build_eis_section

        data = build_eis_section(full_pipeline_results)
        assert "processed 3 files" in data["text"]
        assert data["best_circuit"] == "R(RC)"
        assert data["ranking_table"] is not None
        assert isinstance(data["circuit_stats"], dict)
        assert data["circuit_stats"]["total_fits"] == 5

    def test_without_eis(self, empty_results):
        from src.report_generator import build_eis_section

        data = build_eis_section(empty_results)
        assert "not performed" in data["text"]
        assert data["ranking_table"] is None

    def test_empty_circuit_table(self):
        from src.report_generator import build_eis_section

        eis = _FakeEIS(
            ranked_df=_make_ranked_df(3),
            circuit_table=pd.DataFrame(),
            raw_eis={"f1": {}},
        )
        data = build_eis_section({"eis": eis})
        assert data["best_circuit"] == "N/A"

    def test_image_paths_filtering(self, tmp_path):
        from src.report_generator import build_eis_section

        # Create a real file so it passes exists()
        real = tmp_path / "pca.png"
        real.write_text("fake")
        eis = _FakeEIS(
            ranked_df=_make_ranked_df(2),
            circuit_table=_make_circuit_table(),
            raw_eis={},
            pca=_FakePCA(figure_paths=[str(real), "/nonexistent/img.png"]),
        )
        data = build_eis_section({"eis": eis})
        assert len(data["image_paths"]) == 1


class TestBuildCyclingSection:
    def test_with_data(self, full_pipeline_results):
        from src.report_generator import build_cycling_section

        data = build_cycling_section(full_pipeline_results)
        assert "processed 2 files" in data["text"]
        assert data["table"] is not None

    def test_without_cycling(self, empty_results):
        from src.report_generator import build_cycling_section

        data = build_cycling_section(empty_results)
        assert "not performed" in data["text"]
        assert data["table"] is None


class TestBuildDRTSection:
    def test_with_data(self, full_pipeline_results):
        from src.report_generator import build_drt_section

        data = build_drt_section(full_pipeline_results)
        assert "3 files succeeded" in data["text"]
        assert data["peaks_table"] is not None
        assert data["summary_table"] is not None

    def test_without_drt(self, empty_results):
        from src.report_generator import build_drt_section

        data = build_drt_section(empty_results)
        assert "not performed" in data["text"]

    def test_run_meta_defaults(self):
        from src.report_generator import build_drt_section

        drt = _FakeDRT(run_meta=None)
        data = build_drt_section({"drt": drt})
        assert "0 files succeeded" in data["text"]


class TestBuildCorrelationText:
    def test_with_ranked_df(self, full_pipeline_results):
        from src.report_generator import build_correlation_text

        text = build_correlation_text(full_pipeline_results)
        assert "correlations" in text.lower() or "Spearman" in text

    def test_without_eis(self, empty_results):
        from src.report_generator import build_correlation_text

        text = build_correlation_text(empty_results)
        assert "requires EIS" in text

    def test_too_few_columns(self):
        from src.report_generator import build_correlation_text

        eis = _FakeEIS(
            ranked_df=pd.DataFrame({"A": [1, 2, 3]}),
        )
        text = build_correlation_text({"eis": eis})
        assert "Not enough" in text


# ═══════════════════════════════════════════════════════════════════
# Markdown generation
# ═══════════════════════════════════════════════════════════════════


class TestGenerateMarkdown:
    def test_full_report(self, full_pipeline_results):
        from src.report_generator import generate_markdown

        md = generate_markdown(full_pipeline_results, ai_summary="AI says hello.")
        assert "# IonFlow Pipeline" in md
        assert "EIS Impedance Analysis" in md
        assert "Cycling Analysis" in md
        assert "DRT Analysis" in md
        assert "Correlations" in md
        assert "AI Interpretation" in md
        assert "References" in md
        assert "AI says hello." in md

    def test_eis_only(self, eis_only_results):
        from src.report_generator import generate_markdown

        md = generate_markdown(eis_only_results)
        assert "EIS Impedance Analysis" in md
        # Cycling and DRT sections should still appear (config enables them)
        # but their content says "not performed"
        assert "Cycling" in md

    def test_custom_config(self, full_pipeline_results):
        from src.report_generator import ReportConfig, generate_markdown

        cfg = ReportConfig(
            title="My Custom Title",
            author="Dr. Smith",
            include_cycling=False,
            include_drt=False,
            include_ai=False,
        )
        md = generate_markdown(full_pipeline_results, config=cfg)
        assert "My Custom Title" in md
        assert "Dr. Smith" in md
        assert "Cycling Analysis" not in md
        assert "DRT Analysis" not in md

    def test_no_ai_summary(self, full_pipeline_results):
        from src.report_generator import generate_markdown

        md = generate_markdown(full_pipeline_results, ai_summary=None)
        assert "Executive Summary" not in md
        assert "AI Interpretation" not in md

    def test_footer_present(self, full_pipeline_results):
        from src.report_generator import generate_markdown

        md = generate_markdown(full_pipeline_results)
        assert "Generated by IonFlow Pipeline" in md


# ═══════════════════════════════════════════════════════════════════
# PDF generation
# ═══════════════════════════════════════════════════════════════════


class TestPDFGeneration:
    def test_generate_pdf_full(self, full_pipeline_results, tmp_out):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        paths = gen.generate(
            tmp_out,
            full_pipeline_results,
            ai_summary="Test AI summary.",
            formats=["pdf"],
        )
        assert len(paths) == 1
        assert paths[0].endswith(".pdf")
        assert os.path.exists(paths[0])
        assert os.path.getsize(paths[0]) > 1000  # Not empty

    def test_generate_pdf_eis_only(self, eis_only_results, tmp_out):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        paths = gen.generate(tmp_out, eis_only_results, formats=["pdf"])
        assert len(paths) == 1
        assert os.path.exists(paths[0])

    def test_generate_pdf_empty(self, empty_results, tmp_out):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        paths = gen.generate(tmp_out, empty_results, formats=["pdf"])
        assert len(paths) == 1
        # Even with empty results, a cover page is generated
        assert os.path.exists(paths[0])

    def test_pdf_sections_disabled(self, full_pipeline_results, tmp_out):
        from src.report_generator import ReportConfig, ReportGenerator

        cfg = ReportConfig(
            include_eis=False,
            include_cycling=False,
            include_drt=False,
            include_correlations=False,
            include_ai=False,
            include_references=False,
        )
        gen = ReportGenerator(report_config=cfg)
        paths = gen.generate(tmp_out, full_pipeline_results, formats=["pdf"])
        assert len(paths) == 1
        # Minimal PDF (just cover)
        assert os.path.exists(paths[0])


class TestPDFInternals:
    def test_ionflow_pdf_creation(self):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        assert doc.pdf is not None

    def test_add_cover_page(self):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        doc.add_cover_page(ai_summary="Test summary")
        assert doc.pdf.page > 0

    def test_add_section_headers(self):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        doc.add_section_header("Level 1", level=1)
        doc.add_section_header("Level 2", level=2)
        doc.add_section_header("Level 3", level=3)
        assert doc.pdf.page >= 2  # level 1 adds a page

    def test_add_text(self):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        doc.add_text("Hello, world!")
        # No exception = success

    def test_add_dataframe_table(self):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        df = _make_ranked_df(5)
        doc.add_dataframe_table(df, columns=["Sample", "Score"])
        # No exception

    def test_add_empty_dataframe(self):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        doc.add_dataframe_table(pd.DataFrame())
        # Should handle gracefully

    def test_add_none_dataframe(self):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        doc.add_dataframe_table(None)
        # Should handle gracefully

    def test_add_bullet_list(self):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        doc.add_bullet_list(["Item 1", "Item 2", "Item 3"])
        # No exception

    def test_add_key_value(self):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        doc.add_key_value("Key", "Value")
        # No exception

    def test_add_image_nonexistent(self):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        doc.add_image("/nonexistent/image.png")
        # Should skip gracefully

    def test_save(self, tmp_path):
        from src.report_generator import ReportConfig, _IonFlowPDF

        doc = _IonFlowPDF(ReportConfig())
        doc.add_text("Test content")
        path = doc.save(str(tmp_path / "test.pdf"))
        assert os.path.exists(path)


# ═══════════════════════════════════════════════════════════════════
# Multi-format generation
# ═══════════════════════════════════════════════════════════════════


class TestMultiFormat:
    def test_pdf_and_markdown(self, full_pipeline_results, tmp_out):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        paths = gen.generate(
            tmp_out,
            full_pipeline_results,
            formats=["pdf", "markdown"],
        )
        assert len(paths) == 2
        exts = {Path(p).suffix for p in paths}
        assert ".pdf" in exts
        assert ".md" in exts

    def test_latex_stub(self, full_pipeline_results, tmp_out):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        paths = gen.generate(
            tmp_out,
            full_pipeline_results,
            formats=["latex"],
        )
        assert len(paths) == 1
        assert paths[0].endswith(".tex")
        content = Path(paths[0]).read_text(encoding="utf-8")
        assert "\\documentclass" in content
        assert "EIS Impedance Analysis" in content

    def test_docx_stub(self, full_pipeline_results, tmp_out):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        paths = gen.generate(
            tmp_out,
            full_pipeline_results,
            formats=["docx"],
        )
        assert len(paths) == 1
        content = Path(paths[0]).read_text(encoding="utf-8")
        assert "DOCX Export Placeholder" in content

    def test_all_formats(self, full_pipeline_results, tmp_out):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        paths = gen.generate(
            tmp_out,
            full_pipeline_results,
            formats=["pdf", "markdown", "latex", "docx"],
        )
        assert len(paths) == 4

    def test_unknown_format_ignored(self, full_pipeline_results, tmp_out):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        paths = gen.generate(
            tmp_out,
            full_pipeline_results,
            formats=["pdf", "unknown_format"],
        )
        assert len(paths) == 1  # only PDF succeeds


# ═══════════════════════════════════════════════════════════════════
# History tracking
# ═══════════════════════════════════════════════════════════════════


class TestHistoryIntegration:
    def test_history_recorded_after_generation(self, full_pipeline_results, tmp_path):
        from src.report_generator import ReportConfig, ReportGenerator

        hist_path = str(tmp_path / "history.json")
        cfg = ReportConfig(history_file=hist_path)
        gen = ReportGenerator(report_config=cfg)
        gen.generate(
            str(tmp_path / "report"),
            full_pipeline_results,
            formats=["pdf"],
        )
        assert len(gen.history) == 1
        rec = gen.history.records[0]
        assert rec["output_format"] == "pdf"
        assert rec["version"] == 1
        assert rec["file_size_bytes"] > 0

    def test_history_versions_increment(self, full_pipeline_results, tmp_path):
        from src.report_generator import ReportConfig, ReportGenerator

        hist_path = str(tmp_path / "history.json")
        cfg = ReportConfig(history_file=hist_path)
        gen = ReportGenerator(report_config=cfg)
        for _ in range(3):
            gen.generate(
                str(tmp_path / "report"),
                full_pipeline_results,
                formats=["pdf"],
            )
        assert len(gen.history) == 3
        versions = [r["version"] for r in gen.history.records]
        assert versions == [1, 2, 3]

    def test_multiformat_history(self, full_pipeline_results, tmp_path):
        from src.report_generator import ReportConfig, ReportGenerator

        hist_path = str(tmp_path / "history.json")
        cfg = ReportConfig(history_file=hist_path)
        gen = ReportGenerator(report_config=cfg)
        gen.generate(
            str(tmp_path / "report"),
            full_pipeline_results,
            formats=["pdf", "markdown"],
        )
        assert len(gen.history) == 2


# ═══════════════════════════════════════════════════════════════════
# AI text enhancement path
# ═══════════════════════════════════════════════════════════════════


class TestAIEnhancement:
    def test_enhance_text_no_llm(self):
        from src.report_generator import _enhance_text

        original = "Test text for enhancement."
        result = _enhance_text(original, "context")
        assert result == original  # Falls back when no LLM

    def test_enhance_text_empty(self):
        from src.report_generator import _enhance_text

        assert _enhance_text("", "context") == ""
        assert _enhance_text("  ", "context") == "  "

    @patch("src.ai.llm_adapter.create_adapter_from_config")
    def test_enhance_text_with_mock_llm(self, mock_create):
        from src.report_generator import _enhance_text

        mock_adapter = MagicMock()
        mock_adapter.interpret.return_value = "Enhanced scientific text with clarity."
        mock_create.return_value = mock_adapter

        result = _enhance_text(
            "Original text.", "test context", pipeline_config=MagicMock()
        )
        assert result == "Enhanced scientific text with clarity."
        mock_adapter.interpret.assert_called_once()

    @patch("src.ai.llm_adapter.create_adapter_from_config")
    def test_enhance_text_llm_returns_short(self, mock_create):
        from src.report_generator import _enhance_text

        mock_adapter = MagicMock()
        mock_adapter.interpret.return_value = "short"  # < 10 chars
        mock_create.return_value = mock_adapter

        result = _enhance_text("Original text.", "ctx", pipeline_config=MagicMock())
        assert result == "Original text."  # Falls back

    @patch("src.ai.llm_adapter.create_adapter_from_config")
    def test_enhance_text_llm_error(self, mock_create):
        from src.report_generator import _enhance_text

        mock_create.side_effect = RuntimeError("No LLM module")

        result = _enhance_text("Original text.", "ctx", pipeline_config=MagicMock())
        assert result == "Original text."

    def test_report_ai_enhancement_flag(self, full_pipeline_results, tmp_path):
        from src.report_generator import ReportConfig, ReportGenerator

        cfg = ReportConfig(
            enable_ai_enhancement=True,
            history_file=str(tmp_path / "h.json"),
        )
        gen = ReportGenerator(report_config=cfg)
        # Should not crash even with enhancement enabled but no LLM
        paths = gen.generate(
            str(tmp_path / "report"),
            full_pipeline_results,
            ai_summary="AI summary text here.",
            formats=["pdf"],
        )
        assert len(paths) == 1


# ═══════════════════════════════════════════════════════════════════
# ReportGenerator class interface
# ═══════════════════════════════════════════════════════════════════


class TestReportGeneratorInterface:
    def test_default_construction(self):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        assert gen.report_config is not None
        assert gen.history is not None

    def test_custom_config(self):
        from src.report_generator import ReportConfig, ReportGenerator

        cfg = ReportConfig(title="Custom")
        gen = ReportGenerator(report_config=cfg)
        assert gen.report_config.title == "Custom"

    def test_generate_returns_list(self, full_pipeline_results, tmp_out):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        result = gen.generate(tmp_out, full_pipeline_results, formats=["pdf"])
        assert isinstance(result, list)
        assert all(isinstance(p, str) for p in result)

    def test_generate_creates_directories(self, full_pipeline_results, tmp_path):
        from src.report_generator import ReportGenerator

        deep_path = str(tmp_path / "a" / "b" / "c" / "report")
        gen = ReportGenerator()
        paths = gen.generate(deep_path, full_pipeline_results, formats=["pdf"])
        assert len(paths) == 1
        assert os.path.exists(paths[0])


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_dict_based_results(self, tmp_out):
        """Pipeline results as plain dicts (instead of dataclasses)."""
        from src.report_generator import build_eis_section

        results = {
            "eis": {"ranked_df": None, "circuit_table": None, "raw_eis": {}},
        }
        data = build_eis_section(results)
        assert "not performed" not in data["text"]  # eis key exists

    def test_large_table_truncation(self, tmp_out):
        from src.report_generator import ReportConfig, ReportGenerator

        cfg = ReportConfig(max_table_rows=5)
        gen = ReportGenerator(report_config=cfg)

        eis = _FakeEIS(
            ranked_df=_make_ranked_df(50),
            circuit_table=_make_circuit_table(),
            raw_eis={},
        )
        paths = gen.generate(tmp_out, {"eis": eis}, formats=["pdf"])
        assert len(paths) == 1

    def test_nan_in_dataframe(self, tmp_out):
        from src.report_generator import ReportGenerator

        df = pd.DataFrame(
            {
                "Sample": ["a", "b"],
                "Score": [1.0, float("nan")],
            }
        )
        eis = _FakeEIS(ranked_df=df, circuit_table=pd.DataFrame(), raw_eis={})
        gen = ReportGenerator()
        paths = gen.generate(tmp_out, {"eis": eis}, formats=["pdf"])
        assert len(paths) == 1

    def test_unicode_in_summary(self, full_pipeline_results, tmp_out):
        from src.report_generator import ReportGenerator

        gen = ReportGenerator()
        summary = "Analysis shows τ = 1.5ms, R = 50Ω ± 2Ω. Status: 🟢"
        paths = gen.generate(
            tmp_out,
            full_pipeline_results,
            ai_summary=summary,
            formats=["pdf"],
        )
        assert len(paths) == 1

    def test_markdown_table_rendering(self, full_pipeline_results):
        from src.report_generator import generate_markdown

        md = generate_markdown(full_pipeline_results)
        # Should contain table separators
        assert "|" in md
