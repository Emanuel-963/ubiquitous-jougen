"""Timeline plot of EIS parameters across samples / experiments.

Useful for tracking electrode degradation over repeated measurements or
comparing parameter evolution across a sample series.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Default parameters to show (column, Portuguese label)
_DEFAULT_PARAMS: List[Tuple[str, str]] = [
    ("Rs_fit", "Rs (Ω)"),
    ("Rp_fit", "Rp (Ω)"),
    ("C_mean", "C (F)"),
    ("Energy_mean", "Energia (Wh/kg)"),
]

_PALETTE = ["#1b4f72", "#c0392b", "#27ae60", "#8e44ad"]


def available_timeline_params(features_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return default params that are actually present in *features_df*."""
    return [(col, lbl) for col, lbl in _DEFAULT_PARAMS if col in features_df.columns]


def plot_parameter_timeline(
    features_df: pd.DataFrame,
    *,
    params: Optional[List[str]] = None,
    fig: Optional[Figure] = None,
    ax=None,
) -> Figure:
    """Plot selected EIS parameters vs sample order (index).

    Parameters
    ----------
    features_df:
        DataFrame indexed by sample name with numeric parameter columns.
        Rows are displayed in their natural order (index order).
    params:
        Column names to include.  If *None*, uses all available defaults.
    fig, ax:
        Embedded mode when provided; standalone mode when both are *None*.

    Notes
    -----
    When exactly two parameters are plotted they share the x-axis but use
    independent y-axes (left / right) so different units do not overlap.
    For three or more params only the first two use dual axes; additional
    params are overlaid on the left axis with rescaling (best-effort).
    """
    all_params = available_timeline_params(features_df)
    if params is not None:
        all_params = [(c, lbl) for c, lbl in all_params if c in params]

    standalone = fig is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=110)

    if features_df.empty or not all_params:
        ax.text(
            0.5,
            0.5,
            "Sem dados para exibir",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            color="gray",
        )
        if standalone:
            fig.tight_layout()
        return fig

    n = len(features_df)
    x = np.arange(n)
    short_names = [
        s if len(s) <= 16 else f"\u2026{s[-13:]}" for s in features_df.index.astype(str)
    ]

    # Secondary axis only when we have exactly two params (clean dual-axis)
    ax2 = ax.twinx() if len(all_params) == 2 else None

    for idx, (col, label) in enumerate(all_params):
        target_ax = ax2 if (idx == 1 and ax2 is not None) else ax
        series = pd.to_numeric(features_df[col], errors="coerce").values
        color = _PALETTE[idx % len(_PALETTE)]
        target_ax.plot(
            x,
            series,
            "o-",
            color=color,
            label=label,
            linewidth=1.6,
            markersize=6,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )
        target_ax.set_ylabel(label, color=color, fontsize=10)
        target_ax.tick_params(axis="y", labelcolor=color)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=32, ha="right", fontsize=9)
    ax.set_xlabel("Amostra", fontsize=11)
    ax.set_title("Timeline de Parâmetros EIS", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    # Merged legend from both axes
    lines, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    if lines:
        ax.legend(lines, labels, fontsize=9, framealpha=0.85, loc="best")

    if standalone:
        fig.tight_layout()
    return fig
