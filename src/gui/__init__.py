"""IonFlow GUI — MVC package.

Re-exports the core MVC triad so callers can do::

    from src.gui import AppState, PipelineController, MainWindow

Day 14 adds the widget helpers and tab-level figure builders.
"""

from src.gui.controller import PipelineController  # noqa: F401
from src.gui.main_window import MainWindow  # noqa: F401
from src.gui.models import (  # noqa: F401
    AppState,
    DRT_DEFAULT_PRESET,
    DRT_PRESETS,
    PlotItem,
)

# ── Day 26: keyboard shortcuts + accessibility ──────────────────────
from src.gui.shortcuts import (  # noqa: F401
    AccessibilitySettings,
    DEFAULT_BINDINGS,
    DEFAULT_TOOLTIPS,
    ShortcutAction,
    ShortcutBinding,
    ShortcutManager,
    StatusBarState,
    TooltipRegistry,
)

# ── Day 14: reusable widget logic ────────────────────────────────────
from src.gui.widgets import (  # noqa: F401
    ChartExporter,
    ExportResult,
    FilterableTableManager,
    LogRedirector,
    StyledOptionMenuHelper,
    TableState,
)

# ── Day 14: tab-level figure builders ────────────────────────────────
from src.gui.tabs import (  # noqa: F401
    AIPanelConfig,
    AIPanelResult,
    TableColumnConfig,
    build_executive_summary,
    build_fig_bode,
    build_fig_corr,
    build_fig_drt_eis,
    build_fig_drt_heatmap,
    build_fig_drt_overlay,
    build_fig_drt_spectrum,
    build_fig_energy_cycle,
    build_fig_energy_power,
    build_fig_impedance_heatmap,
    build_fig_nyquist,
    build_fig_pca,
    build_fig_pca_metric,
    build_fig_ragone,
    build_fig_rank,
    build_fig_retention_cycle,
    build_fig_series,
    run_ai_analysis,
    table_column_configs,
)
