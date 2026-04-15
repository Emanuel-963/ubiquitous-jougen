"""EIS chart builders — Nyquist, Bode, impedance heatmap.

Every function is a **pure builder**: data in → ``Figure`` out.
No tkinter, no side-effects, fully testable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.eis_plots import plot_bode, plot_impedance_heatmap, plot_nyquist


# ── Helpers ─────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return os.path.splitext(str(text).strip().lower())[0]


def _resolve_sample_df(
    raw_eis: Dict[str, pd.DataFrame],
    sample_name: str,
) -> Optional[pd.DataFrame]:
    """Find the DataFrame for *sample_name* using exact or normalised match."""
    df = raw_eis.get(sample_name)
    if df is not None:
        return df
    norm = _normalize(sample_name)
    for key, val in raw_eis.items():
        if _normalize(key) == norm:
            return val
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Nyquist
# ═══════════════════════════════════════════════════════════════════════

def build_fig_nyquist(
    raw_eis: Dict[str, pd.DataFrame],
    sample_name: str,
    *,
    figsize: tuple = (5.8, 5.2),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build an embeddable Nyquist plot for *sample_name*.

    Parameters
    ----------
    raw_eis:
        ``{filename: DataFrame}`` dict with ``frequency``, ``zreal``,
        ``zimag`` columns.
    sample_name:
        Key into *raw_eis* (exact or normalised match).

    Returns ``None`` when data is unavailable.
    """
    if not raw_eis:
        return None
    df = _resolve_sample_df(raw_eis, sample_name)
    if df is None or df.empty:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_nyquist(df, sample_name, ax=ax, fig=fig, save=False, show=False)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Bode
# ═══════════════════════════════════════════════════════════════════════

def build_fig_bode(
    raw_eis: Dict[str, pd.DataFrame],
    sample_name: str,
    *,
    figsize: tuple = (6.2, 4.6),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build an embeddable Bode plot for *sample_name*."""
    if not raw_eis:
        return None
    df = _resolve_sample_df(raw_eis, sample_name)
    if df is None or df.empty:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_bode(df, sample_name, ax=ax, fig=fig, save=False, show=False)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Impedance heatmap
# ═══════════════════════════════════════════════════════════════════════

def build_fig_impedance_heatmap(
    raw_eis: Dict[str, pd.DataFrame],
    *,
    figsize: tuple = (8, 5.6),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build a heatmap of log₁₀|Z| across samples and frequencies."""
    if not raw_eis:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_impedance_heatmap(raw_eis, ax=ax, fig=fig, save=False, show=False)
    fig.tight_layout()
    return fig
