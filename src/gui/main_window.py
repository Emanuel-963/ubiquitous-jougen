"""MVC View entry-point — MainWindow composes AppState + PipelineController.

In **Day 14** this class will subclass ``customtkinter.CTk`` and own the
full layout.  For Day 13 it is a plain class that creates the MVC triad
and exposes convenience methods for the legacy ``PipelineApp`` to consume.
"""

from __future__ import annotations

from typing import Any, Callable

from src.gui.controller import PipelineController
from src.gui.models import AppState


class MainWindow:
    """High-level MVC wiring that owns the *Model* and *Controller*.

    Parameters
    ----------
    settings_path:
        Absolute path to the ``.ionflow_gui_settings.json`` file.
        If empty, no settings are persisted.
    """

    def __init__(self, *, settings_path: str = "") -> None:
        self.state: AppState = AppState()
        self.controller: PipelineController = PipelineController(
            self.state, settings_path=settings_path
        )
        # Load persisted settings into state on construction
        self.controller.load_settings()

    # ── Convenience delegators ──────────────────────────────────

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Subscribe to a controller event."""
        self.controller.on(event, callback)

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        """Unsubscribe from a controller event."""
        self.controller.off(event, callback)

    @property
    def is_running(self) -> bool:
        """Whether a pipeline is currently executing."""
        return self.state.is_running

    @property
    def status(self) -> str:
        """Current pipeline status text."""
        return self.state.status
