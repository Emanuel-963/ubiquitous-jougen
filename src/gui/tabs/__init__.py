"""IonFlow GUI — Tab modules package.

Each module exposes **pure figure-builder functions** that accept data
(DataFrames / dicts) and return ``matplotlib.figure.Figure`` objects.
This makes them testable without any tkinter dependency.
"""

from src.gui.tabs.eis_charts import (  # noqa: F401
    build_fig_nyquist,
    build_fig_bode,
    build_fig_impedance_heatmap,
)
from src.gui.tabs.cycling_charts import (  # noqa: F401
    build_fig_energy_power,
    build_fig_energy_cycle,
    build_fig_retention_cycle,
    build_fig_ragone,
)
from src.gui.tabs.drt_charts import (  # noqa: F401
    build_fig_drt_spectrum,
    build_fig_drt_overlay,
    build_fig_drt_heatmap,
)
from src.gui.tabs.advanced_charts import (  # noqa: F401
    build_fig_rank,
    build_fig_pca,
    build_fig_pca_metric,
    build_fig_corr,
    build_fig_drt_eis,
    build_fig_series,
)
from src.gui.tabs.tables import TableColumnConfig, table_column_configs  # noqa: F401
from src.gui.tabs.ai_panel import (  # noqa: F401
    AIPanelConfig,
    AIPanelResult,
    build_executive_summary,
    format_anomalies_text,
    format_findings_text,
    format_predictions_text,
    format_process_text,
    format_recommendations_text,
    run_ai_analysis,
)
