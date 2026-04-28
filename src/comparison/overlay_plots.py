"""Overlay Nyquist and Bode plots for multi-sample comparison.

Both functions follow the same embedded/standalone dual-mode API as
``src/eis_plots.py``:  pass *fig*/*ax* to embed in a GUI canvas, or
leave them *None* to get a standalone ``Figure``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Colour palette for up to 10 samples
_PALETTE = [
    "#1b4f72",
    "#c0392b",
    "#27ae60",
    "#8e44ad",
    "#d35400",
    "#2980b9",
    "#16a085",
    "#f39c12",
    "#2c3e50",
    "#e74c3c",
]

_MAX_LABEL_LEN = 25


def _short_label(name: str) -> str:
    return (
        name if len(name) <= _MAX_LABEL_LEN else f"\u2026{name[-(_MAX_LABEL_LEN - 1):]}"
    )


# ---------------------------------------------------------------------------
# Nyquist overlay
# ---------------------------------------------------------------------------


def plot_nyquist_overlay(
    raw_eis: Dict[str, pd.DataFrame],
    selected: List[str],
    *,
    fig: Optional[Figure] = None,
    ax=None,
) -> Figure:
    """Overlay Nyquist (Z′ vs −Z″) for several samples on one axes.

    Parameters
    ----------
    raw_eis:
        Mapping ``{filename: DataFrame}`` from ``EISResult.raw_eis``.
    selected:
        Ordered list of keys to include.
    fig, ax:
        If provided, render into this axes (embedded mode).
        If ``None``, create a new ``Figure`` (standalone mode).
    """
    standalone = fig is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=110)

    plotted = 0
    for i, name in enumerate(selected):
        df = raw_eis.get(name)
        if df is None or not {"zreal", "zimag"}.issubset(df.columns):
            continue
        d = (
            df.sort_values("frequency", ascending=False)
            if "frequency" in df.columns
            else df
        )
        zr = d["zreal"].values
        zi = -d["zimag"].values
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(
            zr,
            zi,
            "o-",
            color=color,
            label=_short_label(name),
            linewidth=1.4,
            markersize=4,
            markeredgewidth=0.4,
            markeredgecolor="white",
        )
        plotted += 1

    ax.set_xlabel("Z′ (Ω)", fontsize=11)
    ax.set_ylabel("−Z″ (Ω)", fontsize=11)
    ax.set_title(
        f"Nyquist — Overlay ({plotted} amostras)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    if plotted:
        ax.legend(fontsize=8, framealpha=0.85, loc="best")

    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bode overlay
# ---------------------------------------------------------------------------


def plot_bode_overlay(
    raw_eis: Dict[str, pd.DataFrame],
    selected: List[str],
    *,
    fig: Optional[Figure] = None,
    axes=None,
) -> Figure:
    """Overlay Bode (|Z| magnitude and phase) for several samples.

    Parameters
    ----------
    raw_eis, selected:
        Same as :func:`plot_nyquist_overlay`.
    fig:
        If provided, must contain **two** stacked axes (magnitude, phase).
    axes:
        Tuple/list of two ``Axes`` objects matching *fig*.
    """
    standalone = fig is None
    if standalone:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7, 6),
            sharex=True,
            dpi=110,
        )

    ax_mag, ax_phase = axes[0], axes[1]

    plotted = 0
    for i, name in enumerate(selected):
        df = raw_eis.get(name)
        required = {"frequency", "zreal", "zimag"}
        if df is None or not required.issubset(df.columns):
            continue
        d = df.sort_values("frequency").copy()
        freq = d["frequency"].values
        zr = d["zreal"].values
        zi = d["zimag"].values
        zmag = np.sqrt(zr**2 + zi**2)
        phase = np.degrees(np.arctan2(-zi, zr))
        color = _PALETTE[i % len(_PALETTE)]
        kw = dict(
            linewidth=1.4,
            markersize=4,
            markeredgewidth=0.4,
            markeredgecolor="white",
        )
        ax_mag.loglog(freq, zmag, "o-", color=color, label=_short_label(name), **kw)
        ax_phase.semilogx(freq, phase, "o-", color=color, **kw)
        plotted += 1

    ax_mag.set_ylabel("|Z| (Ω)", fontsize=11)
    ax_mag.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax_mag.set_title(
        f"Bode — Overlay ({plotted} amostras)",
        fontsize=12,
        fontweight="bold",
    )
    if plotted:
        ax_mag.legend(fontsize=8, framealpha=0.85, loc="best")

    ax_phase.set_xlabel("Frequência (Hz)", fontsize=11)
    ax_phase.set_ylabel("Fase (°)", fontsize=11)
    ax_phase.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    if standalone:
        fig.subplots_adjust(hspace=0.1)
    return fig
