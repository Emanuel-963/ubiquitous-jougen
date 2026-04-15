"""Advanced / analytical chart builders — Rank, PCA, Correlation, DRT×EIS, Series.

Pure figure builders: data in → ``Figure`` out.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


# ── Helpers ─────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return os.path.splitext(str(text).strip().lower())[0]


def _find_matching_index(
    names: List[str],
    sample_name: Optional[str],
) -> Optional[int]:
    """Find the index of *sample_name* in *names* using normalised match."""
    if not sample_name:
        return None
    target = _normalize(sample_name)
    norm = [_normalize(n) for n in names]
    if target in norm:
        return norm.index(target)
    for idx, item in enumerate(norm):
        if item.endswith(target) or target.endswith(item):
            return idx
    return None


def _highlight_scatter(
    ax: Any,
    names: List[str],
    x_data: Any,
    y_data: Any,
    sample_name: Optional[str],
) -> None:
    """Add a red-star highlight for *sample_name* on *ax*."""
    idx = _find_matching_index(names, sample_name)
    if idx is not None:
        ax.scatter(
            [x_data.iloc[idx] if hasattr(x_data, "iloc") else x_data[idx]],
            [y_data.iloc[idx] if hasattr(y_data, "iloc") else y_data[idx]],
            s=180, marker="*", c="#ef4444",
            edgecolors="black", linewidths=0.8, zorder=6,
            label="Amostra selecionada",
        )
        ax.legend(loc="best", fontsize=8)
    elif sample_name:
        ax.text(
            0.02, 0.98,
            f"Amostra não encontrada:\n{sample_name}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox={"facecolor": "#fff3cd", "edgecolor": "#f59e0b", "alpha": 0.9},
        )


# ═══════════════════════════════════════════════════════════════════════
#  Rank vs Retention
# ═══════════════════════════════════════════════════════════════════════

def build_fig_rank(
    rank_df: Optional[pd.DataFrame],
    *,
    highlight_sample: Optional[str] = None,
    figsize: tuple = (5, 4),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build a Rank vs Retention scatter plot."""
    if rank_df is None or rank_df.empty:
        return None
    if "Rank" not in rank_df.columns or "Retenção (%)" not in rank_df.columns:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(
        rank_df["Rank"], rank_df["Retenção (%)"],
        c="#1f77b4", alpha=0.85, edgecolors="black", linewidths=0.4,
    )
    ax.set_xlabel("Rank")
    ax.set_ylabel("Retenção (%)")
    ax.set_title("Rank vs Retenção")
    ax.grid(True, alpha=0.3)

    _highlight_scatter(
        ax,
        rank_df.index.astype(str).tolist(),
        rank_df["Rank"],
        rank_df["Retenção (%)"],
        highlight_sample,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  PCA 2D
# ═══════════════════════════════════════════════════════════════════════

def build_fig_pca(
    df_pca: Optional[pd.DataFrame],
    *,
    highlight_sample: Optional[str] = None,
    figsize: tuple = (5, 4),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build a PCA 2D scatter (PC1 vs PC2)."""
    if df_pca is None or df_pca.empty:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(
        df_pca.get("PC1", []), df_pca.get("PC2", []),
        c="tab:blue", alpha=0.85, edgecolors="black", linewidths=0.4,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2D")
    ax.grid(True, alpha=0.3)

    if "PC1" in df_pca.columns and "PC2" in df_pca.columns:
        _highlight_scatter(
            ax,
            df_pca.index.astype(str).tolist(),
            df_pca["PC1"],
            df_pca["PC2"],
            highlight_sample,
        )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  PCA 2D with metric colouring
# ═══════════════════════════════════════════════════════════════════════

def build_fig_pca_metric(
    df_pca: Optional[pd.DataFrame],
    rank_df: Optional[pd.DataFrame] = None,
    *,
    highlight_sample: Optional[str] = None,
    figsize: tuple = (5, 4),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build PCA 2D coloured by Retention (%) from *rank_df*."""
    if df_pca is None or df_pca.empty:
        return None
    retention = None
    if rank_df is not None and "Retenção (%)" in rank_df.columns:
        retention = rank_df["Retenção (%)"].reindex(df_pca.index)

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(
        df_pca.get("PC1", []),
        df_pca.get("PC2", []),
        c=retention if retention is not None else "tab:blue",
        cmap="viridis" if retention is not None else None,
        alpha=0.85, edgecolors="black", linewidths=0.4,
    )
    if retention is not None:
        fig.colorbar(scatter, ax=ax, label="Retenção (%)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2D - Retenção")
    ax.grid(True, alpha=0.3)

    if "PC1" in df_pca.columns and "PC2" in df_pca.columns:
        _highlight_scatter(
            ax,
            df_pca.index.astype(str).tolist(),
            df_pca["PC1"],
            df_pca["PC2"],
            highlight_sample,
        )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Correlation heatmap
# ═══════════════════════════════════════════════════════════════════════

def build_fig_corr(
    rank_df: Optional[pd.DataFrame],
    *,
    figsize: tuple = (5.2, 4.4),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build a Spearman correlation heatmap from numeric columns."""
    if rank_df is None or rank_df.empty:
        return None
    data = rank_df.select_dtypes(include=["number"]).dropna(how="all")
    if data.shape[1] < 2:
        return None

    corr = data.corr(method="spearman")
    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  DRT × EIS scatter (gamma_peak vs retention)
# ═══════════════════════════════════════════════════════════════════════

def build_fig_drt_eis(
    drt_eis_df: Optional[pd.DataFrame],
    *,
    highlight_sample: Optional[str] = None,
    figsize: tuple = (5.3, 4.2),
    dpi: int = 100,
) -> Optional[Figure]:
    """Scatter γ peak main vs Retention (%) from the DRT×EIS join."""
    if drt_eis_df is None or drt_eis_df.empty:
        return None
    if "gamma_peak_main" not in drt_eis_df.columns:
        return None
    if "Retenção (%)" not in drt_eis_df.columns:
        return None

    df = drt_eis_df.copy()
    x = pd.to_numeric(df["gamma_peak_main"], errors="coerce")
    y = pd.to_numeric(df["Retenção (%)"], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() == 0:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(
        x[mask], y[mask],
        c="#0ea5e9", alpha=0.85, edgecolors="black", linewidths=0.4,
    )
    ax.set_xlabel("γ pico principal")
    ax.set_ylabel("Retenção (%)")
    ax.set_title("DRT × EIS")
    ax.grid(True, alpha=0.3)

    names = (
        df["Sample"].astype(str).tolist()
        if "Sample" in df.columns
        else df.index.astype(str).tolist()
    )
    valid_names = [names[i] for i, ok in enumerate(mask.tolist()) if ok]
    _highlight_scatter(ax, valid_names, x[mask], y[mask], highlight_sample)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Series plot (metric vs numeric prefix)
# ═══════════════════════════════════════════════════════════════════════

def _split_arquivo(text: str) -> Tuple[float, str]:
    """Parse ``'1 NF H2SO4'`` → ``(1.0, 'NF H2SO4')``."""
    parts = str(text).replace(".txt", "").split()
    if not parts:
        return (0.0, str(text))
    try:
        lead = float(parts[0])
        base = " ".join(parts[1:]) if len(parts) > 1 else str(text)
        return (lead, base)
    except ValueError:
        return (0.0, str(text))


def build_fig_series(
    eis_df: Optional[pd.DataFrame],
    value_col: str,
    base_name: str,
    *,
    figsize: tuple = (5, 4),
    dpi: int = 100,
) -> Optional[Figure]:
    """Build a series plot of *value_col* grouped by *base_name*."""
    if eis_df is None or eis_df.empty:
        return None
    if value_col not in eis_df.columns:
        return None
    if "Arquivo" not in eis_df.columns:
        return None

    df = eis_df.copy()
    info = df["Arquivo"].apply(_split_arquivo)
    df["_lead"] = info.apply(lambda x: x[0])
    df["_base"] = info.apply(lambda x: x[1])
    grp = df[df["_base"] == base_name].sort_values("_lead")
    if grp.empty or grp[value_col].dropna().empty:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(grp["_lead"], grp[value_col], "o-", color="#1f77b4")
    ax.set_xlabel("Prefixo numérico")
    ax.set_ylabel(value_col)
    ax.set_title(f"{value_col} - {base_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
