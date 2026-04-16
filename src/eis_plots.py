"""Publication-quality EIS plots: Nyquist, Bode, and Ragone.

All plot functions support both standalone (save/show) and embeddable
(pass *ax*/*fig*) modes so the GUI can render them on a Canvas.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ── colour palette ──────────────────────────────────────────────────
_C_Z = "#1b4f72"          # impedance magnitude
_C_PHASE = "#c0392b"      # phase angle
_C_SCATTER = "#0ea5e9"    # Ragone scatter
_C_HIGHLIGHT = "#ef4444"  # selected sample marker


# =====================================================================
#  NYQUIST  (Z_real  vs  -Z_imag)
# =====================================================================

def plot_nyquist(
    df: pd.DataFrame,
    sample_name: str,
    *,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    out_dir: str = "outputs/figures",
    show: bool = True,
    save: bool = True,
) -> Optional[str]:
    """Nyquist diagram with equal axes and frequency colour-map.

    Parameters
    ----------
    df : DataFrame with *frequency*, *zreal*, *zimag* columns.
         zimag is already in negative-convention (loader default).
    """
    for col in ("frequency", "zreal", "zimag"):
        if col not in df.columns:
            return None

    d = df.sort_values("frequency", ascending=False).copy()
    freq = d["frequency"].values
    zr = d["zreal"].values
    zi = -d["zimag"].values          # plot as positive -Z''

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5.4), dpi=120)

    sc = ax.scatter(
        zr, zi, c=np.log10(freq + 1e-30),
        cmap="viridis", s=26, edgecolors="black",
        linewidths=0.3, zorder=3,
    )
    ax.plot(zr, zi, "-", color="#adb5bd", linewidth=0.8, zorder=2)

    ax.set_xlabel("Z′ (Ω)", fontsize=11)
    ax.set_ylabel("−Z″ (Ω)", fontsize=11)
    ax.set_title(f"Nyquist — {sample_name}", fontsize=12, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    if fig is not None:
        fig.colorbar(sc, ax=ax, label="log₁₀ f (Hz)", pad=0.02)
        fig.tight_layout()

    filepath: Optional[str] = None
    if standalone:
        if save:
            os.makedirs(out_dir, exist_ok=True)
            stem = Path(sample_name).stem
            filepath = os.path.join(out_dir, f"{stem}_nyquist.png")
            fig.savefig(filepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return filepath


# =====================================================================
#  BODE  ( |Z| and phase  vs  frequency )
# =====================================================================

def plot_bode(
    df: pd.DataFrame,
    sample_name: str,
    *,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    out_dir: str = "outputs/figures",
    show: bool = True,
    save: bool = True,
) -> Optional[str]:
    """Bode plot with dual y-axis (|Z| left, phase right)."""
    for col in ("frequency", "zreal", "zimag"):
        if col not in df.columns:
            return None

    d = df.sort_values("frequency").copy()
    freq = d["frequency"].values
    z_mag = np.sqrt(d["zreal"].values ** 2 + d["zimag"].values ** 2)
    phase = np.degrees(
        np.arctan2(-d["zimag"].values, d["zreal"].values)
    )

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.8), dpi=120)

    ax2 = ax.twinx()

    ln1 = ax.loglog(
        freq, z_mag,
        "o-", color=_C_Z, markersize=3.2,
        linewidth=1.4, label="|Z| (Ω)",
    )
    ln2 = ax2.semilogx(
        freq, phase,
        "s-", color=_C_PHASE, markersize=3.2,
        linewidth=1.4, label="Fase (°)",
    )

    ax.set_xlabel("Frequência (Hz)", fontsize=11)
    ax.set_ylabel("|Z| (Ω)", color=_C_Z, fontsize=11)
    ax2.set_ylabel("Fase (°)", color=_C_PHASE, fontsize=11)
    ax.tick_params(axis="y", labelcolor=_C_Z)
    ax2.tick_params(axis="y", labelcolor=_C_PHASE)
    ax.grid(True, which="both", alpha=0.2, linestyle="--", linewidth=0.5)

    lines = ln1 + ln2
    labels = [ln.get_label() for ln in lines]
    ax.legend(
        lines, labels, loc="upper right",
        fontsize=9, framealpha=0.85, edgecolor="#cccccc",
    )

    ax.set_title(
        f"Bode — {sample_name}",
        fontsize=12, fontweight="bold",
    )

    if fig is not None:
        fig.tight_layout()

    filepath: Optional[str] = None
    if standalone:
        if save:
            os.makedirs(out_dir, exist_ok=True)
            stem = Path(sample_name).stem
            filepath = os.path.join(out_dir, f"{stem}_bode.png")
            fig.savefig(filepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return filepath


# =====================================================================
#  RAGONE  ( Energy  vs  Power,  log-log )
# =====================================================================

def plot_ragone(
    export_tables: Dict[str, pd.DataFrame],
    *,
    highlight_sample: Optional[str] = None,
    target_energy: Optional[float] = None,
    target_power: Optional[float] = None,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    out_dir: str = "outputs/figures",
    show: bool = True,
    save: bool = True,
) -> Optional[str]:
    """Ragone plot (log-log Potência vs Energia) for all samples.

    Parameters
    ----------
    export_tables : ``{sample_name: DataFrame}`` — each DataFrame must
        contain *Energia (Wh/kg)* and *Potência (W/kg)* columns (or
        the snake_case variants).
    """
    all_e: List[float] = []
    all_p: List[float] = []
    all_names: List[str] = []

    col_e = None
    col_p = None

    for name, tbl in export_tables.items():
        # Detect column names (supports both display and internal)
        if col_e is None:
            for c in ("Energia (Wh/kg)", "energia_wh_kg"):
                if c in tbl.columns:
                    col_e = c
                    break
        if col_p is None:
            for c in ("Potência (W/kg)", "potencia_w_kg"):
                if c in tbl.columns:
                    col_p = c
                    break
        if col_e is None or col_p is None:
            return None

        energy = pd.to_numeric(tbl[col_e], errors="coerce")
        power = pd.to_numeric(tbl[col_p], errors="coerce")
        mask = energy.notna() & power.notna() & (energy > 0) & (power > 0)

        if mask.sum() == 0:
            continue

        # Use *median* per sample for a cleaner scatter
        e_med = float(energy[mask].median())
        p_med = float(power[mask].median())
        all_e.append(e_med)
        all_p.append(p_med)
        all_names.append(name)

    if len(all_e) < 1:
        return None

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.4, 5), dpi=120)

    ax.scatter(
        all_e, all_p,
        c=_C_SCATTER, s=64, edgecolors="black",
        linewidths=0.4, alpha=0.85, zorder=3,
    )

    # Highlight specific sample
    if highlight_sample:
        norm_h = _norm(highlight_sample)
        for idx, nm in enumerate(all_names):
            if _norm(nm) == norm_h:
                ax.scatter(
                    [all_e[idx]], [all_p[idx]],
                    s=180, marker="*", c=_C_HIGHLIGHT,
                    edgecolors="black", linewidths=0.8,
                    zorder=5, label="Selecionada",
                )
                ax.legend(loc="best", fontsize=9)
                break

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energia (Wh/kg)", fontsize=11)
    ax.set_ylabel("Potência (W/kg)", fontsize=11)
    ax.set_title("Ragone Plot", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.2, linestyle="--", linewidth=0.5)

    # ── Target point & reference zones ──────────────────────────
    if target_energy is not None and target_power is not None:
        ax.scatter(
            [target_energy], [target_power],
            s=220, marker="X", c="#e74c3c",
            edgecolors="black", linewidths=1.0,
            zorder=6, label="Target",
        )
        # Reference technology zones (approximate, log-log)
        _draw_reference_zones(ax)

        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    if fig is not None:
        fig.tight_layout()

    filepath: Optional[str] = None
    if standalone:
        if save:
            os.makedirs(out_dir, exist_ok=True)
            filepath = os.path.join(out_dir, "ragone_plot.png")
            fig.savefig(filepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return filepath


# ── helpers ─────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return os.path.splitext(str(text).strip().lower())[0]


# ── Reference technology zones for Ragone plot ─────────────────────

def _draw_reference_zones(ax: Axes) -> None:
    """Draw approximate reference technology zones on a Ragone plot.

    Regions based on literature values for supercapacitors, batteries,
    and fuel cells on a log-log Energy (Wh/kg) vs Power (W/kg) plot.
    """
    zones = [
        # (x_min, x_max, y_min, y_max, label, colour)
        (0.01, 0.1, 1e3, 1e5, "Capacitors", "#bdc3c7"),
        (1, 30, 100, 1e4, "Supercapacitors", "#aed6f1"),
        (20, 300, 10, 1e3, "Li-ion Batteries", "#a9dfbf"),
        (200, 2000, 1, 500, "Fuel Cells", "#f9e79f"),
    ]
    for x0, x1, y0, y1, label, colour in zones:
        ax.fill_between(
            [x0, x1], y0, y1,
            alpha=0.12, color=colour, zorder=0,
        )
        # Place label at geometric center
        cx = np.sqrt(x0 * x1)
        cy = np.sqrt(y0 * y1)
        ax.text(
            cx, cy, label,
            ha="center", va="center", fontsize=7,
            color="#555555", fontstyle="italic", zorder=1,
        )


# ── Gap analysis to target ──────────────────────────────────────────


@dataclass
class RagoneGapResult:
    """Result of gap analysis between measured samples and a Ragone target."""

    target_energy: float
    target_power: float
    sample_medians: Dict[str, Tuple[float, float]]  # name → (energy, power)
    best_sample: str
    best_energy: float
    best_power: float
    energy_error_pct: float      # relative error (%) to target energy
    power_error_pct: float       # relative error (%) to target power
    energy_gap: float            # absolute gap (Wh/kg)
    power_gap: float             # absolute gap (W/kg)
    energy_factor: float         # how many × improvement needed
    power_factor: float          # how many × improvement needed
    recommendations: List[str]


def ragone_gap_analysis(
    export_tables: Dict[str, pd.DataFrame],
    target_energy: float = 300.0,
    target_power: float = 3000.0,
) -> Optional[RagoneGapResult]:
    """Compute relative error and gap to a Ragone target point.

    Parameters
    ----------
    export_tables : ``{sample: DataFrame}`` with energy/power columns.
    target_energy : Target energy density in Wh/kg (default 300).
    target_power : Target power density in W/kg (default 3000).

    Returns
    -------
    RagoneGapResult or None if no valid data.
    """
    medians: Dict[str, Tuple[float, float]] = {}

    col_e = col_p = None
    for tbl in export_tables.values():
        if col_e is None:
            for c in ("Energia (Wh/kg)", "energia_wh_kg"):
                if c in tbl.columns:
                    col_e = c
                    break
        if col_p is None:
            for c in ("Potência (W/kg)", "potencia_w_kg"):
                if c in tbl.columns:
                    col_p = c
                    break
        if col_e and col_p:
            break

    if col_e is None or col_p is None:
        return None

    for name, tbl in export_tables.items():
        if col_e not in tbl.columns or col_p not in tbl.columns:
            continue
        energy = pd.to_numeric(tbl[col_e], errors="coerce")
        power = pd.to_numeric(tbl[col_p], errors="coerce")
        mask = energy.notna() & power.notna() & (energy > 0) & (power > 0)
        if mask.sum() == 0:
            continue
        medians[name] = (float(energy[mask].median()), float(power[mask].median()))

    if not medians:
        return None

    # Find best sample (closest in Euclidean log-space to target)
    best_name = ""
    best_dist = float("inf")
    for name, (e, p) in medians.items():
        dist = np.sqrt(
            (np.log10(e) - np.log10(target_energy)) ** 2
            + (np.log10(p) - np.log10(target_power)) ** 2
        )
        if dist < best_dist:
            best_dist = dist
            best_name = name

    best_e, best_p = medians[best_name]

    energy_error_pct = abs(target_energy - best_e) / target_energy * 100
    power_error_pct = abs(target_power - best_p) / target_power * 100
    energy_gap = target_energy - best_e
    power_gap = target_power - best_p
    energy_factor = target_energy / best_e if best_e > 0 else float("inf")
    power_factor = target_power / best_p if best_p > 0 else float("inf")

    recs: List[str] = []
    if energy_factor > 1.05:
        recs.append(
            f"Energy density needs {energy_factor:.1f}× improvement "
            f"({best_e:.2f} → {target_energy:.0f} Wh/kg). "
            "Consider: increase active material loading, optimise electrolyte "
            "concentration, extend potential window."
        )
    if power_factor > 1.05:
        recs.append(
            f"Power density needs {power_factor:.1f}× improvement "
            f"({best_p:.1f} → {target_power:.0f} W/kg). "
            "Consider: reduce Rs (better contacts), increase conductivity, "
            "optimise electrode porosity for fast ion transport."
        )
    if energy_factor <= 1.05 and power_factor <= 1.05:
        recs.append("Target achieved! Current performance meets the goal.")

    # Material-specific recommendations
    if energy_gap > 0:
        recs.append(
            "Electrode design: increase specific capacitance via "
            "nanostructuring or surface activation to boost energy."
        )
    if power_gap > 0:
        recs.append(
            "Kinetics: minimise charge-transfer resistance (Rp) and "
            "diffusion impedance to improve rate capability and power."
        )

    return RagoneGapResult(
        target_energy=target_energy,
        target_power=target_power,
        sample_medians=medians,
        best_sample=best_name,
        best_energy=best_e,
        best_power=best_p,
        energy_error_pct=energy_error_pct,
        power_error_pct=power_error_pct,
        energy_gap=energy_gap,
        power_gap=power_gap,
        energy_factor=energy_factor,
        power_factor=power_factor,
        recommendations=recs,
    )


# =====================================================================
#  ENERGY / POWER  vs  CYCLE  (multi-sample overlay)
# =====================================================================

_PALETTE = [
    "#0ea5e9", "#ef4444", "#22c55e", "#f59e0b",
    "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
    "#6366f1", "#84cc16", "#a855f7", "#06b6d4",
]


def plot_energy_cycle(
    export_tables: Dict[str, pd.DataFrame],
    *,
    metric: str = "Energia (Wh/kg)",
    highlight_samples: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    out_dir: str = "outputs/figures",
    show: bool = True,
    save: bool = True,
) -> Optional[str]:
    """Overlay of *metric* vs cycle for every sample.

    Parameters
    ----------
    export_tables : ``{sample: DataFrame}`` from cycling calculator.
    metric : column to plot on the Y-axis (default ``Energia (Wh/kg)``).
    highlight_samples : optional list of sample names to render with
        thicker lines / markers; the rest are shown as thin grey.
    """
    # Detect cycle column
    col_cycle: Optional[str] = None
    col_y: Optional[str] = None
    for _tbl in export_tables.values():
        for c in ("Ciclos", "ciclo", "Cycle"):
            if c in _tbl.columns:
                col_cycle = c
                break
        for c in (metric, metric.lower().replace(" ", "_")):
            if c in _tbl.columns:
                col_y = c
                break
        if col_cycle and col_y:
            break

    if col_cycle is None or col_y is None:
        return None

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=120)

    norm_hl = set()
    if highlight_samples:
        norm_hl = {_norm(s) for s in highlight_samples}

    names = sorted(export_tables.keys())
    for idx, name in enumerate(names):
        tbl = export_tables[name]
        if col_cycle not in tbl.columns or col_y not in tbl.columns:
            continue
        x = tbl[col_cycle].values
        y = pd.to_numeric(tbl[col_y], errors="coerce").values
        colour = _PALETTE[idx % len(_PALETTE)]

        if norm_hl and _norm(name) not in norm_hl:
            ax.plot(
                x, y, "-",
                color="#d1d5db", linewidth=0.7,
                alpha=0.55, zorder=1,
            )
        else:
            ax.plot(
                x, y, "o-",
                color=colour, markersize=3.6,
                linewidth=1.5, label=name, zorder=3,
            )

    ax.set_xlabel("Ciclo", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(
        f"{metric} vs Ciclo",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    handles, labels = ax.get_legend_handles_labels()
    if 0 < len(labels) <= 12:
        ax.legend(
            fontsize=7.5, loc="best",
            framealpha=0.85, edgecolor="#cccccc",
            ncol=max(1, len(labels) // 6),
        )

    if fig is not None:
        fig.tight_layout()

    filepath: Optional[str] = None
    if standalone:
        if save:
            os.makedirs(out_dir, exist_ok=True)
            safe = metric.replace(" ", "_").replace("/", "-")
            filepath = os.path.join(
                out_dir, f"{safe}_vs_cycle.png",
            )
            fig.savefig(filepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return filepath


# =====================================================================
#  IMPEDANCE HEATMAP  ( |Z| across samples × frequency )
# =====================================================================

def plot_impedance_heatmap(
    raw_eis: Dict[str, pd.DataFrame],
    *,
    n_bands: int = 40,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    out_dir: str = "outputs/figures",
    show: bool = True,
    save: bool = True,
) -> Optional[str]:
    """2-D heatmap of log₁₀|Z| with samples on Y and frequency on X.

    Parameters
    ----------
    raw_eis : ``{sample: DataFrame}`` — each DataFrame must contain
        *frequency*, *zreal*, *zimag*.
    n_bands : number of frequency bins (log-spaced).
    """
    if not raw_eis:
        return None

    # Collect global frequency range
    f_min, f_max = np.inf, -np.inf
    for tbl in raw_eis.values():
        if "frequency" not in tbl.columns:
            continue
        freqs = tbl["frequency"].dropna()
        if len(freqs) == 0:
            continue
        f_min = min(f_min, freqs.min())
        f_max = max(f_max, freqs.max())

    if np.isinf(f_min) or f_min <= 0:
        return None

    freq_edges = np.logspace(
        np.log10(f_min), np.log10(f_max), n_bands + 1,
    )

    names = sorted(raw_eis.keys())
    matrix = np.full((len(names), n_bands), np.nan)

    for row, name in enumerate(names):
        tbl = raw_eis[name]
        if "frequency" not in tbl.columns:
            continue
        freq = tbl["frequency"].values
        zr = tbl["zreal"].values
        zi = tbl["zimag"].values
        z_mag = np.sqrt(zr ** 2 + zi ** 2)

        for col_idx in range(n_bands):
            mask = (freq >= freq_edges[col_idx]) & (
                freq < freq_edges[col_idx + 1]
            )
            if mask.sum() > 0:
                matrix[row, col_idx] = np.log10(
                    np.median(z_mag[mask]) + 1e-30,
                )

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5.6), dpi=120)

    im = ax.imshow(
        matrix, aspect="auto", origin="lower",
        cmap="inferno",
        extent=[
            np.log10(f_min), np.log10(f_max),
            -0.5, len(names) - 0.5,
        ],
    )
    ax.set_yticks(range(len(names)))
    # Truncate long names for readability
    short = [
        (n[:22] + "…") if len(n) > 24 else n for n in names
    ]
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("log₁₀ f (Hz)", fontsize=11)
    ax.set_ylabel("Amostra", fontsize=11)
    ax.set_title(
        "Heatmap de Impedância — log₁₀|Z|",
        fontsize=12, fontweight="bold",
    )

    if fig is not None:
        fig.colorbar(im, ax=ax, label="log₁₀|Z| (Ω)", pad=0.02)
        fig.tight_layout()

    filepath: Optional[str] = None
    if standalone:
        if save:
            os.makedirs(out_dir, exist_ok=True)
            filepath = os.path.join(
                out_dir, "impedance_heatmap.png",
            )
            fig.savefig(filepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return filepath


# =====================================================================
#  BOX-PLOT COMPARATIVO  ( selected metric across samples )
# =====================================================================

_BOXPLOT_COLS = [
    "Rs", "Rp", "C_mean", "C_max",
    "Energy_mean", "Score",
    "C_espec (F/g)", "Retenção (%)",
]


def plot_boxplot_metrics(
    eis_df: pd.DataFrame,
    *,
    metric: str = "Rs",
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    out_dir: str = "outputs/figures",
    show: bool = True,
    save: bool = True,
) -> Optional[str]:
    """Box-plot of *metric* grouped by ``Subclass`` (or all samples).

    Parameters
    ----------
    eis_df : main EIS DataFrame (from ``run_eis_pipeline``).
    metric : column name to plot.
    """
    if metric not in eis_df.columns:
        return None

    vals = pd.to_numeric(eis_df[metric], errors="coerce")
    valid = vals.dropna()
    if len(valid) < 2:
        return None

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.8), dpi=120)

    has_subclass = (
        "Subclass" in eis_df.columns
        and eis_df["Subclass"].nunique() > 1
    )

    if has_subclass:
        groups = []
        labels_list: List[str] = []
        for cls, grp in eis_df.groupby("Subclass"):
            v = pd.to_numeric(
                grp[metric], errors="coerce",
            ).dropna()
            if len(v) == 0:
                continue
            groups.append(v.values)
            labels_list.append(str(cls))

        if not groups:
            return None

        bp = ax.boxplot(
            groups,
            tick_labels=labels_list,
            patch_artist=True,
            showfliers=True,
            widths=0.55,
        )
        for patch, colour in zip(
            bp["boxes"],
            _PALETTE[: len(groups)],
        ):
            patch.set_facecolor(colour)
            patch.set_alpha(0.65)
        ax.set_xlabel("Subclasse", fontsize=11)
    else:
        bp = ax.boxplot(
            [valid.values],
            tick_labels=["Todas"],
            patch_artist=True,
            showfliers=True,
            widths=0.55,
        )
        bp["boxes"][0].set_facecolor(_C_SCATTER)
        bp["boxes"][0].set_alpha(0.65)
        ax.set_xlabel("Amostras", fontsize=11)

    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(
        f"Box-plot — {metric}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

    if fig is not None:
        fig.tight_layout()

    filepath: Optional[str] = None
    if standalone:
        if save:
            os.makedirs(out_dir, exist_ok=True)
            safe = metric.replace(" ", "_").replace("/", "-")
            filepath = os.path.join(
                out_dir, f"boxplot_{safe}.png",
            )
            fig.savefig(filepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return filepath


# =====================================================================
#  RADAR / SPIDER  CHART  ( multi-metric normalised comparison )
# =====================================================================

_RADAR_COLS = [
    "Rs", "Rp", "C_mean", "C_max",
    "Energy_mean", "Score",
    "C_espec (F/g)", "Retenção (%)",
]


def plot_radar(
    eis_df: pd.DataFrame,
    samples: List[str],
    *,
    metrics: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    out_dir: str = "outputs/figures",
    show: bool = True,
    save: bool = True,
) -> Optional[str]:
    """Radar (spider) chart comparing *samples* across normalised metrics.

    Parameters
    ----------
    eis_df : main EIS DataFrame.
    samples : list of sample identifiers (matched against ``Arquivo``
        column or index).
    metrics : columns to use as radar axes.  Falls back to
        ``_RADAR_COLS`` filtered by availability.
    """
    if eis_df is None or eis_df.empty:
        return None

    # Resolve sample rows
    id_col: Optional[str] = None
    for c in ("Arquivo", "Sample"):
        if c in eis_df.columns:
            id_col = c
            break
    if id_col is None:
        return None

    sub = eis_df[
        eis_df[id_col].apply(lambda v: _norm(str(v))).isin(
            [_norm(s) for s in samples]
        )
    ]
    if sub.empty:
        return None

    # Determine axes
    if metrics is None:
        metrics = [
            c for c in _RADAR_COLS if c in eis_df.columns
        ]
    cols = [
        c for c in metrics
        if c in sub.columns
        and pd.to_numeric(sub[c], errors="coerce").notna().sum() > 0
    ]
    if len(cols) < 3:
        return None

    # Normalise to 0-1 (per column across ALL samples)
    num = eis_df[cols].apply(pd.to_numeric, errors="coerce")
    mn = num.min()
    mx = num.max()
    rng = mx - mn
    rng[rng == 0] = 1.0

    norm_all = (num - mn) / rng  # full DataFrame normalised

    # Extract rows for selected samples
    selected_indices = sub.index
    norm_sel = norm_all.loc[selected_indices]

    # Radar angles
    n_cols = len(cols)
    angles = np.linspace(0, 2 * np.pi, n_cols, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # close

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(6.4, 6.4), dpi=120)
        ax = fig.add_subplot(111, polar=True)

    for row_idx, (i, row) in enumerate(norm_sel.iterrows()):
        vals = row[cols].values.astype(float)
        vals = np.concatenate([vals, [vals[0]]])
        colour = _PALETTE[row_idx % len(_PALETTE)]
        label = str(sub.loc[i, id_col]) if id_col else str(i)
        ax.plot(
            angles, vals,
            "o-", color=colour,
            linewidth=1.8, markersize=5,
            label=label, zorder=3,
        )
        ax.fill(angles, vals, color=colour, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_title(
        "Radar — Comparação de Métricas",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.25, 1.12),
        fontsize=8, framealpha=0.85, edgecolor="#cccccc",
    )

    if fig is not None:
        fig.tight_layout()

    filepath: Optional[str] = None
    if standalone:
        if save:
            os.makedirs(out_dir, exist_ok=True)
            filepath = os.path.join(out_dir, "radar_metrics.png")
            fig.savefig(filepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return filepath


# =====================================================================
#  RETENTION vs CYCLE  ( per-sample overlay )
# =====================================================================

def plot_retention_cycle(
    export_tables: Dict[str, pd.DataFrame],
    *,
    energy_col: str = "Energia (Wh/kg)",
    highlight_samples: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    out_dir: str = "outputs/figures",
    show: bool = True,
    save: bool = True,
) -> Optional[str]:
    """Retention (%) vs Cycle for every sample.

    Retention is computed as ``E_cycle / E_cycle_1 * 100``.

    Parameters
    ----------
    export_tables : ``{sample: DataFrame}`` from cycling calculator.
    energy_col : column used to compute retention (default
        ``Energia (Wh/kg)``).
    """
    # Detect columns
    col_cycle: Optional[str] = None
    col_e: Optional[str] = None
    for _tbl in export_tables.values():
        for c in ("Ciclos", "ciclo", "Cycle"):
            if c in _tbl.columns:
                col_cycle = c
                break
        for c in (
            energy_col,
            energy_col.lower().replace(" ", "_"),
        ):
            if c in _tbl.columns:
                col_e = c
                break
        if col_cycle and col_e:
            break

    if col_cycle is None or col_e is None:
        return None

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=120)

    norm_hl = set()
    if highlight_samples:
        norm_hl = {_norm(s) for s in highlight_samples}

    names = sorted(export_tables.keys())
    for idx, name in enumerate(names):
        tbl = export_tables[name]
        if col_cycle not in tbl.columns:
            continue
        if col_e not in tbl.columns:
            continue
        x = tbl[col_cycle].values
        e = pd.to_numeric(tbl[col_e], errors="coerce").values
        if len(e) == 0 or np.isnan(e[0]) or e[0] == 0:
            continue
        retention = (e / e[0]) * 100.0

        colour = _PALETTE[idx % len(_PALETTE)]
        if norm_hl and _norm(name) not in norm_hl:
            ax.plot(
                x, retention, "-",
                color="#d1d5db", linewidth=0.7,
                alpha=0.55, zorder=1,
            )
        else:
            ax.plot(
                x, retention, "o-",
                color=colour, markersize=3.6,
                linewidth=1.5, label=name, zorder=3,
            )

    # Reference line at 100 %
    ax.axhline(
        100, color="#6b7280", linestyle="--",
        linewidth=0.9, alpha=0.6, zorder=2,
    )

    ax.set_xlabel("Ciclo", fontsize=11)
    ax.set_ylabel("Retenção (%)", fontsize=11)
    ax.set_title(
        "Retenção vs Ciclo",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    handles, labels = ax.get_legend_handles_labels()
    if 0 < len(labels) <= 12:
        ax.legend(
            fontsize=7.5, loc="best",
            framealpha=0.85, edgecolor="#cccccc",
            ncol=max(1, len(labels) // 6),
        )

    if fig is not None:
        fig.tight_layout()

    filepath: Optional[str] = None
    if standalone:
        if save:
            os.makedirs(out_dir, exist_ok=True)
            filepath = os.path.join(
                out_dir, "retention_vs_cycle.png",
            )
            fig.savefig(filepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return filepath
