"""Comprehensive tests for src/gui/shortcuts.py — Day 26.

Covers:
- ShortcutAction enum
- ShortcutBinding dataclass
- DEFAULT_BINDINGS completeness
- ShortcutManager: register, dispatch, enable/disable, rebind, help
- StatusBarState: states, format_bar, is_running
- TooltipRegistry: register, get, translator, bulk
- AccessibilitySettings: clamping, increase/decrease, serialisation
"""

from __future__ import annotations

import time

import pytest

from src.gui.shortcuts import (
    DEFAULT_BINDINGS,
    DEFAULT_FONT_SIZE,
    DEFAULT_TOOLTIPS,
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
    AccessibilitySettings,
    ShortcutAction,
    ShortcutBinding,
    ShortcutManager,
    StatusBarState,
    TooltipRegistry,
)


# ====================================================================
# ShortcutAction enum
# ====================================================================

class TestShortcutAction:
    def test_all_nine_actions(self):
        assert len(ShortcutAction) == 9

    def test_values_unique(self):
        vals = [a.value for a in ShortcutAction]
        assert len(vals) == len(set(vals))

    @pytest.mark.parametrize(
        "member",
        [
            "PIPELINE_EIS", "PIPELINE_CYCLING", "PIPELINE_DRT",
            "AI_ANALYSIS", "EXPORT_PDF", "OPEN_CHARTS",
            "SAVE_CONFIG", "RERUN_LAST", "CANCEL_PIPELINE",
        ],
    )
    def test_member_exists(self, member):
        assert hasattr(ShortcutAction, member)


# ====================================================================
# ShortcutBinding
# ====================================================================

class TestShortcutBinding:
    def test_frozen(self):
        b = DEFAULT_BINDINGS[0]
        with pytest.raises(AttributeError):
            b.key_combo = "X"  # type: ignore[misc]

    def test_fields(self):
        b = ShortcutBinding(ShortcutAction.EXPORT_PDF, "Ctrl+E", "<Control-Key-e>", "Export")
        assert b.action == ShortcutAction.EXPORT_PDF
        assert b.key_combo == "Ctrl+E"
        assert b.tk_sequence == "<Control-Key-e>"
        assert b.description == "Export"


# ====================================================================
# DEFAULT_BINDINGS
# ====================================================================

class TestDefaultBindings:
    def test_count(self):
        assert len(DEFAULT_BINDINGS) == 9

    def test_all_actions_covered(self):
        actions = {b.action for b in DEFAULT_BINDINGS}
        assert actions == set(ShortcutAction)

    def test_no_duplicate_tk_sequences(self):
        seqs = [b.tk_sequence for b in DEFAULT_BINDINGS]
        assert len(seqs) == len(set(seqs))

    def test_ctrl_1_is_eis(self):
        eis = [b for b in DEFAULT_BINDINGS if b.action == ShortcutAction.PIPELINE_EIS]
        assert eis[0].key_combo == "Ctrl+1"

    def test_escape_is_cancel(self):
        cancel = [b for b in DEFAULT_BINDINGS if b.action == ShortcutAction.CANCEL_PIPELINE]
        assert cancel[0].tk_sequence == "<Escape>"

    def test_f5_is_rerun(self):
        rerun = [b for b in DEFAULT_BINDINGS if b.action == ShortcutAction.RERUN_LAST]
        assert rerun[0].key_combo == "F5"


# ====================================================================
# ShortcutManager
# ====================================================================

class TestShortcutManager:
    def test_default_bindings_loaded(self):
        mgr = ShortcutManager()
        assert len(mgr.bindings) == 9

    def test_register_and_dispatch(self):
        mgr = ShortcutManager()
        called = []
        mgr.register_handler(ShortcutAction.PIPELINE_EIS, lambda: called.append("eis"))
        assert mgr.dispatch(ShortcutAction.PIPELINE_EIS)
        assert called == ["eis"]

    def test_dispatch_unregistered_returns_false(self):
        mgr = ShortcutManager()
        assert not mgr.dispatch(ShortcutAction.PIPELINE_EIS)

    def test_unregister_handler(self):
        mgr = ShortcutManager()
        mgr.register_handler(ShortcutAction.EXPORT_PDF, lambda: None)
        mgr.unregister_handler(ShortcutAction.EXPORT_PDF)
        assert not mgr.dispatch(ShortcutAction.EXPORT_PDF)

    def test_unregister_absent_no_error(self):
        mgr = ShortcutManager()
        mgr.unregister_handler(ShortcutAction.EXPORT_PDF)  # no-op

    def test_dispatch_by_tk_sequence(self):
        mgr = ShortcutManager()
        called = []
        mgr.register_handler(ShortcutAction.SAVE_CONFIG, lambda: called.append("save"))
        assert mgr.dispatch_by_tk_sequence("<Control-Key-s>")
        assert called == ["save"]

    def test_dispatch_by_unknown_tk_sequence(self):
        mgr = ShortcutManager()
        assert not mgr.dispatch_by_tk_sequence("<Alt-F4>")

    def test_enabled_disable(self):
        mgr = ShortcutManager()
        called = []
        mgr.register_handler(ShortcutAction.RERUN_LAST, lambda: called.append(1))
        mgr.enabled = False
        assert not mgr.dispatch(ShortcutAction.RERUN_LAST)
        assert called == []
        mgr.enabled = True
        assert mgr.dispatch(ShortcutAction.RERUN_LAST)
        assert called == [1]

    def test_get_binding(self):
        mgr = ShortcutManager()
        b = mgr.get_binding(ShortcutAction.AI_ANALYSIS)
        assert b is not None
        assert b.key_combo == "Ctrl+Shift+A"

    def test_get_binding_missing(self):
        mgr = ShortcutManager(bindings=[])
        assert mgr.get_binding(ShortcutAction.AI_ANALYSIS) is None

    def test_get_action_for_tk(self):
        mgr = ShortcutManager()
        action = mgr.get_action_for_tk("<F5>")
        assert action == ShortcutAction.RERUN_LAST

    def test_get_action_for_tk_unknown(self):
        mgr = ShortcutManager()
        assert mgr.get_action_for_tk("<F12>") is None

    def test_registered_actions(self):
        mgr = ShortcutManager()
        mgr.register_handler(ShortcutAction.PIPELINE_EIS, lambda: None)
        mgr.register_handler(ShortcutAction.PIPELINE_DRT, lambda: None)
        assert set(mgr.registered_actions) == {
            ShortcutAction.PIPELINE_EIS,
            ShortcutAction.PIPELINE_DRT,
        }

    def test_rebind(self):
        mgr = ShortcutManager()
        new_b = ShortcutBinding(ShortcutAction.EXPORT_PDF, "Ctrl+P", "<Control-Key-p>", "Print")
        mgr.rebind(ShortcutAction.EXPORT_PDF, new_b)
        assert mgr.get_binding(ShortcutAction.EXPORT_PDF).key_combo == "Ctrl+P"

    def test_rebind_mismatched_action_raises(self):
        mgr = ShortcutManager()
        bad = ShortcutBinding(ShortcutAction.SAVE_CONFIG, "X", "x", "x")
        with pytest.raises(ValueError):
            mgr.rebind(ShortcutAction.EXPORT_PDF, bad)

    def test_help_text(self):
        mgr = ShortcutManager()
        txt = mgr.help_text()
        assert "Ctrl+1" in txt
        assert "Escape" in txt
        assert len(txt.strip().split("\n")) == 9

    def test_custom_bindings(self):
        custom = (
            ShortcutBinding(ShortcutAction.PIPELINE_EIS, "A", "a", "eis"),
        )
        mgr = ShortcutManager(bindings=custom)
        assert len(mgr.bindings) == 1


# ====================================================================
# StatusBarState
# ====================================================================

class TestStatusBarState:
    def test_defaults(self):
        s = StatusBarState()
        assert s.pipeline_status == "idle"
        assert s.samples_loaded == 0
        assert not s.is_running

    def test_set_running(self):
        s = StatusBarState()
        s.set_running("EIS")
        assert s.is_running
        assert "EIS" in s.pipeline_status

    def test_set_idle(self):
        s = StatusBarState()
        s.set_running("DRT")
        s.set_idle()
        assert not s.is_running
        assert s.pipeline_status == "idle"

    def test_set_error(self):
        s = StatusBarState()
        s.set_error("timeout")
        assert "timeout" in s.pipeline_status
        assert not s.is_running

    def test_format_bar_minimal(self):
        s = StatusBarState()
        bar = s.format_bar()
        assert "idle" in bar
        assert "Samples: 0" in bar

    def test_format_bar_full(self):
        s = StatusBarState(
            pipeline_status="idle",
            samples_loaded=42,
            last_ai_analysis="2026-04-15",
            version="0.2.0",
        )
        bar = s.format_bar()
        assert "42" in bar
        assert "2026-04-15" in bar
        assert "0.2.0" in bar

    def test_as_dict(self):
        s = StatusBarState(version="1.0")
        d = s.as_dict()
        assert d["version"] == "1.0"
        assert "last_updated" not in d

    def test_last_updated_changes(self):
        s = StatusBarState()
        t1 = s.last_updated
        time.sleep(0.01)
        s.set_running("X")
        assert s.last_updated >= t1


# ====================================================================
# TooltipRegistry
# ====================================================================

class TestTooltipRegistry:
    def test_register_and_get(self):
        reg = TooltipRegistry()
        reg.register("btn_x", "Click me")
        assert reg.get("btn_x") == "Click me"

    def test_get_missing(self):
        reg = TooltipRegistry()
        assert reg.get("no_such") == ""

    def test_register_many(self):
        reg = TooltipRegistry()
        reg.register_many({"a": "A", "b": "B"})
        assert reg.get("a") == "A"
        assert len(reg) == 2

    def test_remove(self):
        reg = TooltipRegistry()
        reg.register("x", "X")
        reg.remove("x")
        assert not reg.has("x")

    def test_remove_missing_no_error(self):
        reg = TooltipRegistry()
        reg.remove("nope")

    def test_has(self):
        reg = TooltipRegistry()
        reg.register("y", "Y")
        assert reg.has("y")
        assert not reg.has("z")

    def test_all_ids(self):
        reg = TooltipRegistry()
        reg.register_many({"a": "A", "b": "B", "c": "C"})
        assert sorted(reg.all_ids) == ["a", "b", "c"]

    def test_translator_applied(self):
        def fake_tr(key: str) -> str:
            return {"ui.run_eis": "Executar EIS"}.get(key, key)

        reg = TooltipRegistry(translator=fake_tr)
        reg.register("btn", "ui.run_eis")
        assert reg.get("btn") == "Executar EIS"

    def test_translator_fallback_to_raw(self):
        reg = TooltipRegistry(translator=lambda k: k)  # identity → no translation
        reg.register("btn", "some raw text")
        assert reg.get("btn") == "some raw text"

    def test_len(self):
        reg = TooltipRegistry()
        assert len(reg) == 0
        reg.register("a", "A")
        assert len(reg) == 1


# ====================================================================
# DEFAULT_TOOLTIPS
# ====================================================================

class TestDefaultTooltips:
    def test_non_empty(self):
        assert len(DEFAULT_TOOLTIPS) >= 10

    def test_all_values_are_strings(self):
        for k, v in DEFAULT_TOOLTIPS.items():
            assert isinstance(k, str) and isinstance(v, str)


# ====================================================================
# AccessibilitySettings
# ====================================================================

class TestAccessibilitySettings:
    def test_defaults(self):
        a = AccessibilitySettings()
        assert a.font_size == DEFAULT_FONT_SIZE
        assert not a.high_contrast

    def test_clamp_low(self):
        a = AccessibilitySettings(font_size=5)
        assert a.font_size == MIN_FONT_SIZE

    def test_clamp_high(self):
        a = AccessibilitySettings(font_size=99)
        assert a.font_size == MAX_FONT_SIZE

    def test_set_font_size(self):
        a = AccessibilitySettings()
        a.set_font_size(18)
        assert a.font_size == 18

    def test_set_font_size_clamped(self):
        a = AccessibilitySettings()
        a.set_font_size(0)
        assert a.font_size == MIN_FONT_SIZE

    def test_increase_font(self):
        a = AccessibilitySettings(font_size=14)
        new = a.increase_font(2)
        assert new == 16
        assert a.font_size == 16

    def test_increase_font_capped(self):
        a = AccessibilitySettings(font_size=MAX_FONT_SIZE)
        new = a.increase_font(5)
        assert new == MAX_FONT_SIZE

    def test_decrease_font(self):
        a = AccessibilitySettings(font_size=16)
        new = a.decrease_font(3)
        assert new == 13

    def test_decrease_font_capped(self):
        a = AccessibilitySettings(font_size=MIN_FONT_SIZE)
        new = a.decrease_font(5)
        assert new == MIN_FONT_SIZE

    def test_as_dict(self):
        a = AccessibilitySettings(font_size=18, high_contrast=True)
        d = a.as_dict()
        assert d == {"font_size": 18, "high_contrast": True}

    def test_from_dict(self):
        a = AccessibilitySettings.from_dict({"font_size": 16, "high_contrast": True})
        assert a.font_size == 16
        assert a.high_contrast

    def test_from_dict_defaults(self):
        a = AccessibilitySettings.from_dict({})
        assert a.font_size == DEFAULT_FONT_SIZE
        assert not a.high_contrast

    def test_round_trip(self):
        a = AccessibilitySettings(font_size=18, high_contrast=True)
        b = AccessibilitySettings.from_dict(a.as_dict())
        assert a.font_size == b.font_size
        assert a.high_contrast == b.high_contrast
