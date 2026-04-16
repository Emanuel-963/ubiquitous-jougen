"""Comprehensive tests for the refactored i18n system (Day 25).

Covers:
- JSON loading (pt, en, es)
- Dotted-key lookup (tr("section.key"))
- Legacy Portuguese-key lookup (tr("Portuguese string"))
- tr_section convenience
- available_keys, get_section, missing_keys, translation_coverage
- Fallback behaviour for missing keys
- Thread safety
- reload_strings
- All sections populated
- Spanish translations complete
- LANGUAGES / SECTIONS constants
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.i18n import (
    LANGUAGES,
    SECTIONS,
    available_keys,
    get_language,
    get_languages,
    get_section,
    missing_keys,
    reload_strings,
    set_language,
    tr,
    tr_section,
    translation_coverage,
)

# Path to JSON string files
_STRINGS_DIR = Path(__file__).resolve().parent.parent / "src" / "i18n_strings"


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _restore_pt():
    """Ensure language is reset to 'pt' after every test."""
    set_language("pt")
    yield
    set_language("pt")


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════


class TestConstants:
    def test_languages_tuple(self):
        assert LANGUAGES == ("pt", "en", "es")

    def test_sections_tuple(self):
        assert "ui" in SECTIONS
        assert "pipeline" in SECTIONS
        assert "ai" in SECTIONS
        assert "reports" in SECTIONS
        assert "columns" in SECTIONS

    def test_get_languages(self):
        assert get_languages() == LANGUAGES


# ═══════════════════════════════════════════════════════════════════
# JSON files exist and are valid
# ═══════════════════════════════════════════════════════════════════


class TestJSONFiles:
    @pytest.mark.parametrize("lang", LANGUAGES)
    def test_file_exists(self, lang):
        path = _STRINGS_DIR / f"{lang}.json"
        assert path.exists(), f"{lang}.json not found"

    @pytest.mark.parametrize("lang", LANGUAGES)
    def test_file_valid_json(self, lang):
        path = _STRINGS_DIR / f"{lang}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert len(data) > 0

    @pytest.mark.parametrize("lang", LANGUAGES)
    def test_has_all_sections(self, lang):
        path = _STRINGS_DIR / f"{lang}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for section in SECTIONS:
            assert section in data, f"Section '{section}' missing from {lang}.json"

    def test_all_languages_same_sections(self):
        """All language files must have the same top-level sections."""
        sections_per_lang = {}
        for lang in LANGUAGES:
            path = _STRINGS_DIR / f"{lang}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            sections_per_lang[lang] = set(data.keys())
        pt_sections = sections_per_lang["pt"]
        for lang in ("en", "es"):
            assert sections_per_lang[lang] == pt_sections, (
                f"{lang}.json sections differ from pt.json"
            )


# ═══════════════════════════════════════════════════════════════════
# set_language / get_language
# ═══════════════════════════════════════════════════════════════════


class TestSetLanguage:
    def test_default_is_pt(self):
        set_language("pt")
        assert get_language() == "pt"

    def test_switch_to_en(self):
        set_language("en")
        assert get_language() == "en"

    def test_switch_to_es(self):
        set_language("es")
        assert get_language() == "es"

    def test_invalid_falls_back_to_pt(self):
        set_language("xx")
        assert get_language() == "pt"

    def test_case_insensitive(self):
        set_language("EN")
        assert get_language() == "en"

    def test_whitespace_stripped(self):
        set_language("  es  ")
        assert get_language() == "es"

    def test_long_code_truncated(self):
        set_language("english")
        assert get_language() == "en"


# ═══════════════════════════════════════════════════════════════════
# tr — legacy Portuguese-key mode
# ═══════════════════════════════════════════════════════════════════


class TestTrLegacy:
    """Backward compatibility: tr("Portuguese string") works."""

    def test_pt_returns_key_unchanged(self):
        set_language("pt")
        assert tr("Rodar Pipeline EIS") == "Rodar Pipeline EIS"

    def test_en_translates_known_key(self):
        set_language("en")
        assert tr("Rodar Pipeline EIS") == "Run EIS Pipeline"

    def test_es_translates_known_key(self):
        set_language("es")
        assert tr("Rodar Pipeline EIS") == "Ejecutar Pipeline EIS"

    def test_en_returns_key_for_missing(self):
        set_language("en")
        unknown = "chave_que_nao_existe_12345"
        assert tr(unknown) == unknown

    def test_es_returns_key_for_missing(self):
        set_language("es")
        assert tr("nonexistent") == "nonexistent"

    # Spot-check several existing translations
    def test_en_interactive_charts(self):
        set_language("en")
        assert tr("Gráficos Interativos") == "Interactive Charts"

    def test_en_run_both(self):
        set_language("en")
        assert tr("Rodar Ambos") == "Run Both"

    def test_en_status_ready(self):
        set_language("en")
        assert tr("Status: pronto") == "Status: ready"

    def test_en_save_image(self):
        set_language("en")
        assert tr("Salvar imagem") == "Save image"

    def test_en_cycling(self):
        set_language("en")
        assert tr("Ciclagem") == "Cycling"

    def test_en_theme_dark(self):
        set_language("en")
        assert tr("Escuro") == "Dark"

    def test_en_error_eis(self):
        set_language("en")
        assert tr("erro no EIS") == "EIS error"

    def test_en_empty_state(self):
        set_language("en")
        assert tr("Nenhum dado disponível.") == "No data available."

    def test_en_log_message(self):
        set_language("en")
        result = tr("Tabela não encontrada.")
        assert result == "Table not found."

    def test_es_theme(self):
        set_language("es")
        assert tr("Tema") == "Tema"  # Same word in Spanish

    def test_es_cycling(self):
        set_language("es")
        assert tr("Ciclagem") == "Ciclaje"


# ═══════════════════════════════════════════════════════════════════
# tr — dotted key mode
# ═══════════════════════════════════════════════════════════════════


class TestTrDotted:
    """New section-based key lookup: tr("section.key")."""

    def test_pt_dotted_key(self):
        set_language("pt")
        assert tr("ui.run_eis") == "Rodar Pipeline EIS"

    def test_en_dotted_key(self):
        set_language("en")
        assert tr("ui.run_eis") == "Run EIS Pipeline"

    def test_es_dotted_key(self):
        set_language("es")
        assert tr("ui.run_eis") == "Ejecutar Pipeline EIS"

    def test_pipeline_section(self):
        set_language("en")
        assert tr("pipeline.eis_complete") == "EIS completed"

    def test_ai_section(self):
        set_language("en")
        assert tr("ai.executive_summary") == "Executive Summary"

    def test_reports_section(self):
        set_language("en")
        assert tr("reports.eis_analysis") == "EIS Impedance Analysis"

    def test_columns_section(self):
        set_language("en")
        assert tr("columns.sample") == "Sample"

    def test_plots_section(self):
        set_language("en")
        assert tr("plots.frequency_axis") == "Frequency (Hz)"

    def test_diagnostics_section(self):
        set_language("en")
        assert tr("diagnostics.fit_excellent") == "Excellent fit"

    def test_cli_section(self):
        set_language("en")
        assert tr("cli.description") == "IonFlow Pipeline — EIS analytics toolkit"

    def test_empty_section(self):
        set_language("en")
        assert tr("empty.no_data") == "No data available."

    def test_log_section(self):
        set_language("en")
        assert tr("log.table_not_found") == "Table not found."

    def test_knowledge_section(self):
        set_language("en")
        result = tr("knowledge.rs_very_low")
        assert "excellent" in result.lower()

    def test_missing_dotted_key(self):
        set_language("en")
        assert tr("nonexistent.key") == "nonexistent.key"

    def test_pt_dotted_returns_portuguese(self):
        set_language("pt")
        assert tr("plots.phase_axis") == "Fase (°)"


# ═══════════════════════════════════════════════════════════════════
# tr_section
# ═══════════════════════════════════════════════════════════════════


class TestTrSection:
    def test_basic(self):
        set_language("en")
        assert tr_section("ui", "run_eis") == "Run EIS Pipeline"

    def test_pt(self):
        set_language("pt")
        assert tr_section("ui", "run_eis") == "Rodar Pipeline EIS"

    def test_es(self):
        set_language("es")
        assert tr_section("ui", "run_eis") == "Ejecutar Pipeline EIS"

    def test_missing_key(self):
        set_language("en")
        result = tr_section("ui", "nonexistent_key_xyz")
        assert result == "ui.nonexistent_key_xyz"


# ═══════════════════════════════════════════════════════════════════
# available_keys
# ═══════════════════════════════════════════════════════════════════


class TestAvailableKeys:
    def test_all_keys_non_empty(self):
        keys = available_keys()
        assert len(keys) > 100  # We have 250+ keys

    def test_all_keys_sorted(self):
        keys = available_keys()
        assert keys == sorted(keys)

    def test_section_filter(self):
        keys = available_keys("ui")
        assert all(k.startswith("ui.") for k in keys)
        assert len(keys) > 20

    def test_unknown_section_empty(self):
        keys = available_keys("nonexistent_section")
        assert keys == []

    @pytest.mark.parametrize("section", SECTIONS)
    def test_each_section_has_keys(self, section):
        keys = available_keys(section)
        assert len(keys) > 0, f"Section '{section}' has no keys"


# ═══════════════════════════════════════════════════════════════════
# get_section
# ═══════════════════════════════════════════════════════════════════


class TestGetSection:
    def test_returns_dict(self):
        data = get_section("ui")
        assert isinstance(data, dict)
        assert len(data) > 20

    def test_keys_are_short(self):
        data = get_section("ui")
        # Keys should NOT start with "ui."
        for k in data:
            assert not k.startswith("ui."), f"Key '{k}' should be short"

    def test_specific_language(self):
        data = get_section("ui", lang="en")
        assert data["run_eis"] == "Run EIS Pipeline"

    def test_specific_language_es(self):
        data = get_section("ui", lang="es")
        assert data["run_eis"] == "Ejecutar Pipeline EIS"

    def test_follows_current_language(self):
        set_language("en")
        data = get_section("plots")
        assert data["frequency_axis"] == "Frequency (Hz)"


# ═══════════════════════════════════════════════════════════════════
# missing_keys / translation_coverage
# ═══════════════════════════════════════════════════════════════════


class TestCoverage:
    def test_en_coverage_100(self):
        coverage = translation_coverage("en")
        assert coverage == 1.0, f"en coverage = {coverage:.1%}"

    def test_es_coverage_100(self):
        coverage = translation_coverage("es")
        assert coverage == 1.0, f"es coverage = {coverage:.1%}"

    def test_pt_coverage_100(self):
        coverage = translation_coverage("pt")
        assert coverage == 1.0

    def test_en_no_missing_keys(self):
        m = missing_keys("en")
        assert m == [], f"Missing en keys: {m[:5]}..."

    def test_es_no_missing_keys(self):
        m = missing_keys("es")
        assert m == [], f"Missing es keys: {m[:5]}..."

    def test_unknown_lang_all_missing(self):
        coverage = translation_coverage("zz")
        assert coverage == 0.0


# ═══════════════════════════════════════════════════════════════════
# reload_strings
# ═══════════════════════════════════════════════════════════════════


class TestReloadStrings:
    def test_reload_preserves_function(self):
        reload_strings()
        set_language("en")
        assert tr("ui.run_eis") == "Run EIS Pipeline"

    def test_reload_twice(self):
        reload_strings()
        reload_strings()
        assert tr("ui.run_eis") == "Rodar Pipeline EIS"  # pt is default


# ═══════════════════════════════════════════════════════════════════
# Spanish translations — spot checks
# ═══════════════════════════════════════════════════════════════════


class TestSpanish:
    def test_ui_run_cycling(self):
        set_language("es")
        assert tr("ui.run_cycling") == "Ejecutar Pipeline Ciclaje"

    def test_pipeline_eis_complete(self):
        set_language("es")
        assert tr("pipeline.eis_complete") == "EIS completado"

    def test_ai_quality_excellent(self):
        set_language("es")
        assert tr("ai.quality_excellent") == "Calidad general: EXCELENTE"

    def test_reports_fit_quality(self):
        set_language("es")
        assert tr("reports.fit_quality") == "Calidad del Ajuste"

    def test_columns_sample(self):
        set_language("es")
        assert tr("columns.sample") == "Muestra"

    def test_plots_phase_axis(self):
        set_language("es")
        assert tr("plots.phase_axis") == "Fase (°)"

    def test_diagnostics_fit_problematic(self):
        set_language("es")
        assert tr("diagnostics.fit_problematic") == "Ajuste problemático"

    def test_cli_description(self):
        set_language("es")
        result = tr("cli.description")
        assert "herramientas" in result

    def test_empty_no_data(self):
        set_language("es")
        assert tr("empty.no_data") == "No hay datos disponibles."

    def test_knowledge_rs_high(self):
        set_language("es")
        result = tr("knowledge.rs_high")
        assert "elevada" in result


# ═══════════════════════════════════════════════════════════════════
# K-Means labels
# ═══════════════════════════════════════════════════════════════════


class TestKMeansLabels:
    def test_pt_efficient(self):
        set_language("pt")
        assert tr("pipeline.kmeans_efficient") == "Interface eficiente"

    def test_en_efficient(self):
        set_language("en")
        assert tr("pipeline.kmeans_efficient") == "Efficient interface"

    def test_es_efficient(self):
        set_language("es")
        assert tr("pipeline.kmeans_efficient") == "Interfaz eficiente"

    def test_pt_stable(self):
        set_language("pt")
        assert tr("pipeline.kmeans_stable") == "Genérica estável"

    def test_en_stable(self):
        set_language("en")
        assert tr("pipeline.kmeans_stable") == "Generic stable"

    def test_es_stable(self):
        set_language("es")
        assert tr("pipeline.kmeans_stable") == "Genérica estable"


# ═══════════════════════════════════════════════════════════════════
# Thread safety
# ═══════════════════════════════════════════════════════════════════


class TestThreadSafety:
    def test_concurrent_set_language(self):
        """Multiple threads setting language concurrently should not crash."""
        errors = []

        def worker(lang):
            try:
                for _ in range(50):
                    set_language(lang)
                    _ = get_language()
                    _ = tr("ui.run_eis")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(lang,))
            for lang in ("pt", "en", "es") * 3
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_key(self):
        assert tr("") == ""

    def test_key_with_dots_not_in_json(self):
        assert tr("a.b.c.d.e") == "a.b.c.d.e"

    def test_pt_never_looks_up(self):
        """Portuguese legacy key should return as-is without any lookup."""
        set_language("pt")
        assert tr("anything at all") == "anything at all"

    def test_special_characters_in_value(self):
        set_language("en")
        # The ≥ symbol should pass through correctly
        result = tr("empty.need_2_samples_radar")
        assert "≥2" in result

    def test_newline_in_value(self):
        set_language("en")
        result = tr("empty.insuf_radar")
        assert "\n" in result

    def test_all_values_are_strings(self):
        keys = available_keys()
        set_language("pt")
        for k in keys:
            v = tr(k)
            assert isinstance(v, str), f"Key '{k}' returned {type(v)}"
