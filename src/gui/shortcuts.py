"""Keyboard shortcuts, status bar, tooltips, and accessibility helpers.

Day 26 — provides a fully testable, widget-free layer that any GUI frontend
can consume.  The module defines *what* each shortcut does (symbolic action),
leaving the actual tkinter/CTk binding to the view layer.

Public API
----------
ShortcutAction          – Enum of every shortcut action.
ShortcutBinding         – Dataclass tying a key combo to an action.
ShortcutManager         – Registry + dispatcher (no GUI dependency).
StatusBarState          – Dataclass for the bottom status bar.
TooltipRegistry         – Maps widget-id → tooltip text (i18n-aware).
AccessibilitySettings   – Font size, high-contrast, etc.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Shortcut actions
# ---------------------------------------------------------------------------

class ShortcutAction(enum.Enum):
    """Symbolic names for every keyboard shortcut in IonFlow."""

    PIPELINE_EIS = "pipeline_eis"
    PIPELINE_CYCLING = "pipeline_cycling"
    PIPELINE_DRT = "pipeline_drt"
    AI_ANALYSIS = "ai_analysis"
    EXPORT_PDF = "export_pdf"
    OPEN_CHARTS = "open_charts"
    SAVE_CONFIG = "save_config"
    RERUN_LAST = "rerun_last"
    CANCEL_PIPELINE = "cancel_pipeline"


# ---------------------------------------------------------------------------
# Key binding dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShortcutBinding:
    """Associates a keyboard combination with a :class:`ShortcutAction`.

    Parameters
    ----------
    action : ShortcutAction
        The symbolic action.
    key_combo : str
        Human-readable key combination, e.g. ``"Ctrl+1"``.
    tk_sequence : str
        Tkinter event sequence, e.g. ``"<Control-Key-1>"``.
    description : str
        Short tooltip / help text.
    """

    action: ShortcutAction
    key_combo: str
    tk_sequence: str
    description: str


# ---------------------------------------------------------------------------
# Default bindings
# ---------------------------------------------------------------------------

DEFAULT_BINDINGS: Tuple[ShortcutBinding, ...] = (
    ShortcutBinding(
        ShortcutAction.PIPELINE_EIS,
        "Ctrl+1",
        "<Control-Key-1>",
        "Run EIS pipeline",
    ),
    ShortcutBinding(
        ShortcutAction.PIPELINE_CYCLING,
        "Ctrl+2",
        "<Control-Key-2>",
        "Run Cycling pipeline",
    ),
    ShortcutBinding(
        ShortcutAction.PIPELINE_DRT,
        "Ctrl+3",
        "<Control-Key-3>",
        "Run DRT pipeline",
    ),
    ShortcutBinding(
        ShortcutAction.AI_ANALYSIS,
        "Ctrl+Shift+A",
        "<Control-Shift-Key-A>",
        "Run AI analysis",
    ),
    ShortcutBinding(
        ShortcutAction.EXPORT_PDF,
        "Ctrl+E",
        "<Control-Key-e>",
        "Export PDF report",
    ),
    ShortcutBinding(
        ShortcutAction.OPEN_CHARTS,
        "Ctrl+G",
        "<Control-Key-g>",
        "Open interactive charts",
    ),
    ShortcutBinding(
        ShortcutAction.SAVE_CONFIG,
        "Ctrl+S",
        "<Control-Key-s>",
        "Save configuration",
    ),
    ShortcutBinding(
        ShortcutAction.RERUN_LAST,
        "F5",
        "<F5>",
        "Re-run last pipeline",
    ),
    ShortcutBinding(
        ShortcutAction.CANCEL_PIPELINE,
        "Escape",
        "<Escape>",
        "Cancel running pipeline",
    ),
)


# ---------------------------------------------------------------------------
# ShortcutManager
# ---------------------------------------------------------------------------

class ShortcutManager:
    """Widget-free shortcut registry and dispatcher.

    Usage::

        mgr = ShortcutManager()
        mgr.register_handler(ShortcutAction.PIPELINE_EIS, run_eis)
        mgr.dispatch(ShortcutAction.PIPELINE_EIS)  # calls run_eis()
    """

    def __init__(
        self,
        bindings: Sequence[ShortcutBinding] = DEFAULT_BINDINGS,
    ) -> None:
        self._bindings: Dict[ShortcutAction, ShortcutBinding] = {
            b.action: b for b in bindings
        }
        self._handlers: Dict[ShortcutAction, Callable[[], Any]] = {}
        self._enabled: bool = True

    # -- registration --------------------------------------------------------

    def register_handler(
        self,
        action: ShortcutAction,
        handler: Callable[[], Any],
    ) -> None:
        """Bind *handler* to *action*."""
        self._handlers[action] = handler

    def unregister_handler(self, action: ShortcutAction) -> None:
        """Remove handler for *action* (no-op if absent)."""
        self._handlers.pop(action, None)

    # -- dispatch ------------------------------------------------------------

    def dispatch(self, action: ShortcutAction) -> bool:
        """Invoke the handler for *action*.

        Returns ``True`` if a handler was found and called, ``False`` otherwise.
        """
        if not self._enabled:
            return False
        handler = self._handlers.get(action)
        if handler is None:
            return False
        handler()
        return True

    def dispatch_by_tk_sequence(self, tk_sequence: str) -> bool:
        """Look up action by tk event string, then dispatch."""
        for binding in self._bindings.values():
            if binding.tk_sequence == tk_sequence:
                return self.dispatch(binding.action)
        return False

    # -- query ---------------------------------------------------------------

    @property
    def bindings(self) -> Dict[ShortcutAction, ShortcutBinding]:
        """Return a copy of the current binding map."""
        return dict(self._bindings)

    def get_binding(self, action: ShortcutAction) -> Optional[ShortcutBinding]:
        """Return the :class:`ShortcutBinding` for *action*, or ``None`` if not mapped."""
        return self._bindings.get(action)

    def get_action_for_tk(self, tk_sequence: str) -> Optional[ShortcutAction]:
        """Resolve a tkinter event *tk_sequence* to its :class:`ShortcutAction`, or ``None``."""
        for b in self._bindings.values():
            if b.tk_sequence == tk_sequence:
                return b.action
        return None

    @property
    def registered_actions(self) -> List[ShortcutAction]:
        """Return a list of all actions that currently have a registered handler."""
        return list(self._handlers.keys())

    # -- enable / disable ----------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether shortcut dispatching is currently enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable shortcut dispatching globally."""
        self._enabled = bool(value)

    # -- customisation -------------------------------------------------------

    def rebind(self, action: ShortcutAction, new_binding: ShortcutBinding) -> None:
        """Replace the binding for *action*."""
        if new_binding.action != action:
            raise ValueError("Binding action must match the target action")
        self._bindings[action] = new_binding

    def help_text(self) -> str:
        """Return a multi-line help string listing all shortcuts."""
        lines: List[str] = []
        for b in self._bindings.values():
            lines.append(f"{b.key_combo:<20s} {b.description}")
        return "\n".join(sorted(lines))


# ---------------------------------------------------------------------------
# StatusBarState
# ---------------------------------------------------------------------------

@dataclass
class StatusBarState:
    """Data model for the GUI status bar.

    The view layer reads this dataclass and renders it; the controller
    updates it and notifies the view.
    """

    pipeline_status: str = "idle"
    samples_loaded: int = 0
    last_ai_analysis: str = ""
    version: str = ""
    last_updated: float = field(default_factory=time.time)

    # -- convenience ---------------------------------------------------------

    def set_running(self, pipeline_name: str) -> None:
        """Transition status to *running* for the given *pipeline_name* and update timestamp."""
        self.pipeline_status = f"running: {pipeline_name}"
        self.last_updated = time.time()

    def set_idle(self) -> None:
        """Reset status to *idle* and update the timestamp."""
        self.pipeline_status = "idle"
        self.last_updated = time.time()

    def set_error(self, message: str = "error") -> None:
        """Transition status to *error* with an optional diagnostic *message* and update timestamp."""
        self.pipeline_status = f"error: {message}"
        self.last_updated = time.time()

    @property
    def is_running(self) -> bool:
        """Return ``True`` if a pipeline is currently running."""
        return self.pipeline_status.startswith("running")

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the status bar state to a plain dictionary (excludes ``last_updated``)."""
        return {
            "pipeline_status": self.pipeline_status,
            "samples_loaded": self.samples_loaded,
            "last_ai_analysis": self.last_ai_analysis,
            "version": self.version,
        }

    def format_bar(self) -> str:
        """Return a single-line summary suitable for a status bar widget."""
        parts = [
            f"Status: {self.pipeline_status}",
            f"Samples: {self.samples_loaded}",
        ]
        if self.last_ai_analysis:
            parts.append(f"AI: {self.last_ai_analysis}")
        if self.version:
            parts.append(f"v{self.version}")
        return "  |  ".join(parts)


# ---------------------------------------------------------------------------
# TooltipRegistry
# ---------------------------------------------------------------------------

class TooltipRegistry:
    """Maps widget identifiers to tooltip strings.

    Tooltip texts can be registered individually or in bulk.  The registry
    is i18n-aware: pass a *translator* callable (e.g. ``src.i18n.tr``) and
    keys will be translated on lookup.
    """

    def __init__(
        self,
        translator: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._tips: Dict[str, str] = {}
        self._translator = translator

    def register(self, widget_id: str, text: str) -> None:
        """Register a tooltip *text* for *widget_id*."""
        self._tips[widget_id] = text

    def register_many(self, mapping: Dict[str, str]) -> None:
        """Bulk-register tooltips from a dictionary."""
        self._tips.update(mapping)

    def get(self, widget_id: str) -> str:
        """Return tooltip text (translated if translator is set)."""
        raw = self._tips.get(widget_id, "")
        if raw and self._translator is not None:
            translated = self._translator(raw)
            # If translator returns the key itself, fall back to raw
            if translated != raw:
                return translated
        return raw

    def remove(self, widget_id: str) -> None:
        """Remove the tooltip for *widget_id* (no-op if absent)."""
        self._tips.pop(widget_id, None)

    def has(self, widget_id: str) -> bool:
        """Return ``True`` if a tooltip is registered for *widget_id*."""
        return widget_id in self._tips

    @property
    def all_ids(self) -> List[str]:
        """Return a list of all widget identifiers that have registered tooltips."""
        return list(self._tips.keys())

    def __len__(self) -> int:
        return len(self._tips)


# Default tooltips for common GUI widgets
DEFAULT_TOOLTIPS: Dict[str, str] = {
    "btn_run_eis": "ui.run_eis",
    "btn_run_cycling": "ui.run_cycling",
    "btn_run_drt": "ui.run_drt",
    "btn_run_both": "ui.run_both",
    "btn_ai_analysis": "ui.ai_analysis",
    "btn_export_pdf": "ui.export_pdf",
    "btn_open_charts": "ui.open_interactive",
    "btn_save_config": "ui.save_settings",
    "btn_cancel": "ui.cancel",
    "combo_language": "ui.language",
    "combo_circuit": "ui.select_circuit",
    "entry_data_dir": "ui.data_dir",
    "entry_output_dir": "ui.output_dir",
    "slider_font_size": "ui.font_size",
}


# ---------------------------------------------------------------------------
# AccessibilitySettings
# ---------------------------------------------------------------------------

MIN_FONT_SIZE = 12
MAX_FONT_SIZE = 20
DEFAULT_FONT_SIZE = 14


@dataclass
class AccessibilitySettings:
    """User-configurable accessibility preferences."""

    font_size: int = DEFAULT_FONT_SIZE
    high_contrast: bool = False

    def __post_init__(self) -> None:
        self.font_size = self._clamp(self.font_size)

    @staticmethod
    def _clamp(size: int) -> int:
        return max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, size))

    def set_font_size(self, size: int) -> None:
        """Set font size, clamped to [MIN_FONT_SIZE, MAX_FONT_SIZE]."""
        self.font_size = self._clamp(size)

    def increase_font(self, step: int = 1) -> int:
        """Increase font size by *step*, return new size."""
        self.font_size = self._clamp(self.font_size + step)
        return self.font_size

    def decrease_font(self, step: int = 1) -> int:
        """Decrease font size by *step*, return new size."""
        self.font_size = self._clamp(self.font_size - step)
        return self.font_size

    def as_dict(self) -> Dict[str, Any]:
        """Serialize accessibility preferences to a plain dictionary for persistence."""
        return {"font_size": self.font_size, "high_contrast": self.high_contrast}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccessibilitySettings":
        """Construct an :class:`AccessibilitySettings` instance from a saved dictionary."""
        return cls(
            font_size=data.get("font_size", DEFAULT_FONT_SIZE),
            high_contrast=data.get("high_contrast", False),
        )
