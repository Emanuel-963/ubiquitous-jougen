"""IonFlow GUI — MVC package.

Re-exports the core MVC triad so callers can do::

    from src.gui import AppState, PipelineController, MainWindow
"""

from src.gui.controller import PipelineController  # noqa: F401
from src.gui.main_window import MainWindow  # noqa: F401
from src.gui.models import (  # noqa: F401
    AppState,
    DRT_DEFAULT_PRESET,
    DRT_PRESETS,
    PlotItem,
)
