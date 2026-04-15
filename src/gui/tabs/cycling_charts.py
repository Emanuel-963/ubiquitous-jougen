"""Cycling chart builders — energy×power, energy/cycle, retention, Ragone.

Pure figure builders: data in → ``Figure`` out.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from matplotlib.figure import Figure

from src.cycling_plotter import plot_energy_power_vs_cycle
from src.eis_plots import (
    plot_energy_cycle,
    plot_ragone,
    plot_retention_cycle,
)


def _normalize(text: str) -> str:
    return os.path.splitext(str(text).strip().lower())[0]


# ═══════════════════════════════════════════════════════════════════════
#  Energy × Power vs Cycle
# ═══════════════════════════════════════════════════════════════════════

def build_fig_energy_power(
    cic_results: Dict[str, pd.DataFrame],
    sample_name: str,
    *,
    figsize: tuple = (5.8, 4.4),
    dpi: int = 100,
) -> Optional[Figure]:
    """Dual-axis Energy × Power vs Cycle for *sample_name*."""
    if not cic_results:
        return None
    cycle_df = cic_results.get(sample_name)
    if cycle_df is None:
        norm = _normalize(sample_name)
        for key, df in cic_results.items():
            if _normalize(key) == norm:
                cycle_df = df
                break
    if cycle_df is None or cycle_df.empty:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_energy_power_vs_cycle(
        cycle_df, sample_name,
        ax=ax, fig=fig, save=False, show=False,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Energy per Cycle overlay
# ═══════════════════════════════════════════════════════════════════════

def build_fig_energy_cycle(
    cic_results: Dict[str, pd.DataFrame],
    *,
    metric: str = "Energia (Wh/kg)",
    highlight_samples: Optional[List[str]] = None,
    figsize: tuple = (7.5, 4.8),
    dpi: int = 100,
) -> Optional[Figure]:
    """Overlay of *metric* vs cycle for all samples."""
    if not cic_results:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_energy_cycle(
        cic_results,
        metric=metric,
        highlight_samples=highlight_samples,
        ax=ax, fig=fig, save=False, show=False,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Retention vs Cycle
# ═══════════════════════════════════════════════════════════════════════

def build_fig_retention_cycle(
    cic_results: Dict[str, pd.DataFrame],
    *,
    figsize: tuple = (7.5, 4.8),
    dpi: int = 100,
) -> Optional[Figure]:
    """Retention (%) vs Cycle overlay."""
    if not cic_results:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_retention_cycle(
        cic_results,
        ax=ax, fig=fig, save=False, show=False,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Ragone plot
# ═══════════════════════════════════════════════════════════════════════

def build_fig_ragone(
    cic_results: Dict[str, pd.DataFrame],
    *,
    highlight_sample: Optional[str] = None,
    figsize: tuple = (5.8, 4.8),
    dpi: int = 100,
) -> Optional[Figure]:
    """Ragone plot (Energy vs Power) for all cycling samples."""
    if not cic_results:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_ragone(
        cic_results,
        highlight_sample=highlight_sample,
        ax=ax, fig=fig, save=False, show=False,
    )
    fig.tight_layout()
    return fig
