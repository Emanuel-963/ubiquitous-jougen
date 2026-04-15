"""DRT chart builders — spectrum, overlay, heatmap.

Pure figure builders: data in → ``Figure`` out.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from matplotlib.figure import Figure

from src.drt_visualization import (
    plot_drt_heatmap,
    plot_drt_overlay,
    plot_drt_spectrum,
)


# ═══════════════════════════════════════════════════════════════════════
#  DRT Spectrum (single sample)
# ═══════════════════════════════════════════════════════════════════════

def build_fig_drt_spectrum(
    drt_results: Dict[str, dict],
    sample_name: str,
    *,
    figsize: tuple = (5.4, 4.2),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build a DRT γ(τ) spectrum for *sample_name*."""
    if not drt_results:
        return None
    result = drt_results.get(sample_name)
    if not result:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_drt_spectrum(result, sample_name, ax=ax, save=False, show=False)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  DRT Overlay (multiple samples)
# ═══════════════════════════════════════════════════════════════════════

def build_fig_drt_overlay(
    drt_results: Dict[str, dict],
    sample_names: Optional[List[str]] = None,
    *,
    max_default: int = 6,
    figsize: tuple = (5.8, 4.2),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build an overlay of DRT spectra for selected samples.

    When *sample_names* is empty or ``None``, takes the first
    *max_default* sorted keys.
    """
    if not drt_results:
        return None
    if not sample_names:
        sample_names = sorted(drt_results.keys())[:max_default]

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_drt_overlay(drt_results, selected=sample_names, ax=ax, show=False)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  DRT Heatmap
# ═══════════════════════════════════════════════════════════════════════

def build_fig_drt_heatmap(
    drt_results: Dict[str, dict],
    sample_names: Optional[List[str]] = None,
    *,
    figsize: tuple = (6.2, 4.4),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build a heatmap of DRT γ(τ) across samples."""
    if not drt_results:
        return None
    if not sample_names:
        sample_names = sorted(drt_results.keys())

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    plot_drt_heatmap(drt_results, stems=sample_names, ax=ax, show=False)
    fig.tight_layout()
    return fig
