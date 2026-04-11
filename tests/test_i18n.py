"""Tests for src/i18n.py — translation infrastructure."""
from src.i18n import get_language, set_language, tr


class TestSetLanguage:
    def test_default_is_pt(self):
        set_language("pt")
        assert get_language() == "pt"

    def test_switch_to_en(self):
        set_language("en")
        assert get_language() == "en"
        set_language("pt")  # restore

    def test_invalid_falls_back_to_pt(self):
        set_language("xx")
        assert get_language() == "pt"


class TestTr:
    def test_pt_returns_key_unchanged(self):
        set_language("pt")
        assert tr("Rodar Pipeline EIS") == "Rodar Pipeline EIS"

    def test_en_translates_known_key(self):
        set_language("en")
        result = tr("Rodar Pipeline EIS")
        assert result == "Run EIS Pipeline"
        set_language("pt")

    def test_en_returns_key_for_missing_translation(self):
        set_language("en")
        unknown = "chave_que_nao_existe_12345"
        assert tr(unknown) == unknown
        set_language("pt")
