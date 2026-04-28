"""Comparison and health-score utilities — Phase 4 of IonFlow v0.3.0.

Public API
----------
compute_health_score   — Electrode Health Score 0-100
health_score_label     — Category label ("Excelente", "Bom", …)
health_score_color     — Hex colour for UI badges
plot_nyquist_overlay   — Overlay Nyquist for N samples
plot_bode_overlay      — Overlay Bode (|Z| + phase) for N samples
available_timeline_params — Which default params exist in a DataFrame
plot_parameter_timeline   — Evolution of EIS params vs sample order
"""

from src.comparison.health_score import (
    DEFAULT_WEIGHTS,
    METRIC_LABELS,
    compute_health_score,
    health_score_color,
    health_score_label,
)
from src.comparison.overlay_plots import plot_bode_overlay, plot_nyquist_overlay
from src.comparison.parameter_timeline import (
    available_timeline_params,
    plot_parameter_timeline,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "METRIC_LABELS",
    "compute_health_score",
    "health_score_label",
    "health_score_color",
    "plot_nyquist_overlay",
    "plot_bode_overlay",
    "available_timeline_params",
    "plot_parameter_timeline",
]
