"""Electrode Health Score (0–100) for EIS spectra.

The score aggregates multiple EIS-derived metrics into a single index
using weighted min-max normalisation.  All weights are configurable
so that domain experts can tune the formula without touching code.

Sign convention
---------------
- Positive weight → higher column value = healthier electrode.
- Negative weight → lower column value = healthier electrode.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Default weights (mirrors config.py score_weights but extended)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "Rs_fit": -0.25,  # solution resistance — lower is better
    "Rp_fit": 0.35,  # polarisation resistance — higher is better
    "C_mean": 0.25,  # mean capacitance — higher is better
    "Energy_mean": 0.15,  # stored energy — higher is better
}

# Human-readable metric labels (Portuguese)
METRIC_LABELS: Dict[str, str] = {
    "Rs_fit": "Rs (Ω)",
    "Rp_fit": "Rp (Ω)",
    "C_mean": "C média (F)",
    "Energy_mean": "Energia (Wh/kg)",
    "Score": "Score EIS",
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_health_score(
    features_df: pd.DataFrame,
    *,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Compute Electrode Health Score (0–100) for each sample.

    Parameters
    ----------
    features_df:
        DataFrame with samples as index.  At least one of the weight keys
        must be a column.
    weights:
        Signed weight mapping.  Positive = higher is healthier; negative =
        lower is healthier.  Defaults to :data:`DEFAULT_WEIGHTS`.

    Returns
    -------
    pd.Series
        Float values in [0, 100], indexed like *features_df*.
        Samples where no weighted column is available receive 50.0.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    available = {col: w for col, w in weights.items() if col in features_df.columns}

    if not available or features_df.empty:
        return pd.Series(50.0, index=features_df.index, name="Health Score")

    total_abs_weight = sum(abs(w) for w in available.values())
    accumulator = pd.Series(0.0, index=features_df.index)

    for col, w in available.items():
        series = pd.to_numeric(features_df[col], errors="coerce").fillna(0.0)
        vmin, vmax = float(series.min()), float(series.max())

        if vmax == vmin:
            normalised = pd.Series(0.5, index=series.index)
        else:
            normalised = (series - vmin) / (vmax - vmin)

        if w < 0:
            normalised = 1.0 - normalised

        accumulator += abs(w) * normalised

    return (accumulator / total_abs_weight * 100).round(1).rename("Health Score")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def health_score_label(score: float) -> str:
    """Return a Portuguese category label for *score*."""
    if score >= 75:
        return "Excelente"
    elif score >= 50:
        return "Bom"
    elif score >= 25:
        return "Regular"
    return "Degradado"


def health_score_color(score: float) -> str:
    """Return a hex color for the health badge."""
    if score >= 75:
        return "#22c55e"  # green
    elif score >= 50:
        return "#eab308"  # yellow
    elif score >= 25:
        return "#f97316"  # orange
    return "#ef4444"  # red
