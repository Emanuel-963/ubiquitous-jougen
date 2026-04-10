"""DRT visualization helpers used by CLI and GUI flows."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _ensure_out_dir(out_dir: Optional[str | Path]) -> Path:
    target = Path(out_dir) if out_dir else Path("outputs/figures/drt")
    target.mkdir(parents=True, exist_ok=True)
    return target


def plot_drt_spectrum(result, stem, out_dir=None, *, ax=None, show=False, save=True):
    """Plot γ(τ) for one sample and optionally save as PNG."""
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    tau = np.asarray(result.get("tau", []), dtype=float)
    gamma = np.asarray(result.get("gamma", []), dtype=float)
    peaks = result.get("peaks", []) or []

    if len(tau) == 0 or len(gamma) == 0:
        ax.text(0.5, 0.5, "Sem dados DRT", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.plot(tau, gamma, color="#2563eb", linewidth=1.8)
        ax.fill_between(tau, 0, gamma, color="#60a5fa", alpha=0.18)
        ax.set_xscale("log")
        ax.set_xlabel("τ [s]")
        ax.set_ylabel("γ(τ) [Ω]")
        ax.set_title(f"DRT — {stem}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)

        for pk in peaks:
            tau_peak = pk.get("tau_peak")
            gamma_peak = pk.get("gamma_peak")
            if tau_peak is None or gamma_peak is None:
                continue
            ax.scatter(
                [tau_peak],
                [gamma_peak],
                s=42,
                c="#ef4444",
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            ax.axvline(
                tau_peak,
                linestyle=":",
                linewidth=1.0,
                color="#ef4444",
                alpha=0.7,
            )

        r_inf = result.get("r_inf", float("nan"))
        lambda_reg = result.get("lambda_reg", float("nan"))
        ax.text(
            0.02,
            0.98,
            f"R∞ = {r_inf:.3f} Ω\nλ = {lambda_reg:.0e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.9},
        )

    fig.tight_layout()

    saved_path = ""
    if save:
        target_dir = _ensure_out_dir(out_dir)
        saved_path = str((target_dir / f"{stem}_drt.png").resolve())
        fig.savefig(saved_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    if owns_fig:
        plt.close(fig)

    return saved_path


def plot_drt_overlay(
    results_dict,
    selected=None,
    out_path=None,
    *,
    ax=None,
    show=False,
):
    """Plot multiple γ(τ) spectra in one axis."""
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(7, 4.4))
    else:
        fig = ax.figure

    names = selected if selected else sorted(results_dict.keys())
    plotted = 0
    for name in names:
        result = results_dict.get(name)
        if not result:
            continue
        tau = np.asarray(result.get("tau", []), dtype=float)
        gamma = np.asarray(result.get("gamma", []), dtype=float)
        if len(tau) == 0 or len(gamma) == 0:
            continue
        ax.plot(tau, gamma, linewidth=1.5, alpha=0.9, label=name)
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "Sem dados DRT", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.set_xscale("log")
        ax.set_xlabel("τ [s]")
        ax.set_ylabel("γ(τ) [Ω]")
        ax.set_title("DRT Overlay")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    saved_path = ""
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved_path = str(path.resolve())
        fig.savefig(saved_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    if owns_fig:
        plt.close(fig)

    return saved_path


def plot_drt_heatmap(
    results_dict,
    stems=None,
    out_path=None,
    *,
    n_taus=80,
    ax=None,
    show=False,
):
    """Plot DRT heatmap (samples × log10(τ))."""
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(7, 4.8))
    else:
        fig = ax.figure

    names = stems if stems else sorted(results_dict.keys())
    valid_names = []
    all_tau = []
    for name in names:
        res = results_dict.get(name)
        if not res:
            continue
        tau = np.asarray(res.get("tau", []), dtype=float)
        gamma = np.asarray(res.get("gamma", []), dtype=float)
        if len(tau) == 0 or len(gamma) == 0:
            continue
        valid_names.append(name)
        all_tau.append(tau)

    if not valid_names:
        ax.text(0.5, 0.5, "Sem dados DRT", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        if owns_fig:
            plt.close(fig)
        return ""

    tau_min = min(float(np.min(t)) for t in all_tau)
    tau_max = max(float(np.max(t)) for t in all_tau)
    common_tau = np.logspace(np.log10(tau_min), np.log10(tau_max), int(n_taus))

    heat_rows = []
    for name in valid_names:
        res = results_dict[name]
        tau = np.asarray(res.get("tau", []), dtype=float)
        gamma = np.asarray(res.get("gamma", []), dtype=float)
        order = np.argsort(tau)
        tau_sorted = tau[order]
        gamma_sorted = gamma[order]
        interp = np.interp(
            np.log10(common_tau),
            np.log10(tau_sorted),
            gamma_sorted,
            left=np.nan,
            right=np.nan,
        )
        heat_rows.append(interp)

    matrix = np.asarray(heat_rows)
    extent = [
        np.log10(common_tau[0]),
        np.log10(common_tau[-1]),
        -0.5,
        len(valid_names) - 0.5,
    ]
    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap="viridis",
        origin="lower",
    )
    ax.set_xlabel("log10(τ [s])")
    ax.set_ylabel("Amostras")
    ax.set_title("DRT Heatmap")
    ax.set_yticks(range(len(valid_names)))
    ax.set_yticklabels(valid_names, fontsize=8)
    fig.colorbar(im, ax=ax, label="γ(τ) [Ω]")
    fig.tight_layout()

    saved_path = ""
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved_path = str(path.resolve())
        fig.savefig(saved_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    if owns_fig:
        plt.close(fig)

    return saved_path
