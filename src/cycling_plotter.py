"""Plotter for cycling data: Time vs Potential with integral as legend."""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_time_potential_with_integral(
    df: pd.DataFrame,
    filename: str,
    out_dir: str = "outputs/figures",
    show: bool = True,
):
    """Plot Time vs Potential, add integral as legend.

    Returns the saved figure path.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df["tempo"], df["potencial"], label="Potential")

    # Calculate integral (area under curve, approximate energy proxy)
    integral = np.trapezoid(df["potencial"], df["tempo"])
    plt.title(f"{filename}: Time vs Potential\nIntegral: {integral:.2f}")
    plt.xlabel("Time")
    plt.ylabel("Potential")
    plt.legend()
    plt.grid(True)

    stem = Path(filename).stem
    filepath = os.path.join(out_dir, f"{stem}_integral.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return filepath


# ---------------------------------------------------------------------------
# Energia × Potência vs Ciclo  (dual Y-axis)
# ---------------------------------------------------------------------------

_COLOR_POWER = "#1b4f72"       # azul escuro (eixo esquerdo)
_COLOR_ENERGY = "#e74c3c"      # vermelho vivo (eixo direito)
_FILL_ALPHA = 0.18


def plot_energy_power_vs_cycle(
    cycle_df: pd.DataFrame,
    filename: str,
    *,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    out_dir: str = "outputs/figures",
    show: bool = True,
    save: bool = True,
) -> Optional[str]:
    """Gera gráfico dual-axis de Potência (W/kg) e Energia (Wh/kg) vs Ciclo.

    Parameters
    ----------
    cycle_df : DataFrame com colunas ``Ciclos``, ``Potência (W/kg)``
               e ``Energia (Wh/kg)``.
    filename : nome da amostra (usado no título e nome de arquivo).
    ax / fig : se fornecidos, desenha no *ax* existente (modo embeddable).
               Quando ``ax`` é passado, **save** e **show** são ignorados.
    out_dir  : pasta de saída para imagem salva.
    show     : exibe com ``plt.show()`` (modo standalone).
    save     : salva PNG em *out_dir*.

    Returns
    -------
    Caminho do arquivo salvo ou ``None``.
    """
    # ---- validação de colunas -----------------------------------------------
    col_ciclo = None
    for candidate in ("Ciclos", "Ciclos (numero)", "ciclo"):
        if candidate in cycle_df.columns:
            col_ciclo = candidate
            break
    col_power = (
        "Potência (W/kg)" if "Potência (W/kg)" in cycle_df.columns
        else "potencia_w_kg" if "potencia_w_kg" in cycle_df.columns
        else None
    )
    col_energy = (
        "Energia (Wh/kg)" if "Energia (Wh/kg)" in cycle_df.columns
        else "energia_wh_kg" if "energia_wh_kg" in cycle_df.columns
        else None
    )

    if col_ciclo is None or col_power is None or col_energy is None:
        return None

    df = cycle_df.copy().sort_values(col_ciclo)
    cycles = pd.to_numeric(df[col_ciclo], errors="coerce")
    power = pd.to_numeric(df[col_power], errors="coerce")
    energy = pd.to_numeric(df[col_energy], errors="coerce")
    mask = cycles.notna() & power.notna() & energy.notna()
    if mask.sum() < 2:
        return None

    cycles = cycles[mask].values
    power = power[mask].values
    energy = energy[mask].values

    # ---- modo embeddable vs standalone --------------------------------------
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)

    ax2 = ax.twinx()

    # ---- Potência (eixo esquerdo) -------------------------------------------
    ln1 = ax.plot(
        cycles, power,
        color=_COLOR_POWER, linewidth=1.6, marker="o",
        markersize=2.4, markerfacecolor=_COLOR_POWER,
        label="Potência (W/kg)", zorder=3,
    )
    ax.fill_between(
        cycles, power, alpha=_FILL_ALPHA,
        color=_COLOR_POWER, zorder=2,
    )

    # ---- Energia (eixo direito) ---------------------------------------------
    ln2 = ax2.plot(
        cycles, energy,
        color=_COLOR_ENERGY, linewidth=1.6, marker="s",
        markersize=2.4, markerfacecolor=_COLOR_ENERGY,
        label="Energia (Wh/kg)", zorder=3,
    )
    ax2.fill_between(
        cycles, energy, alpha=_FILL_ALPHA,
        color=_COLOR_ENERGY, zorder=2,
    )

    # ---- labels e estilo ----------------------------------------------------
    ax.set_xlabel("Número de Ciclos", fontsize=11, fontweight="medium")
    ax.set_ylabel("Potência (W/kg)", color=_COLOR_POWER, fontsize=11)
    ax2.set_ylabel("Energia (Wh/kg)", color=_COLOR_ENERGY, fontsize=11)

    ax.tick_params(axis="y", labelcolor=_COLOR_POWER)
    ax2.tick_params(axis="y", labelcolor=_COLOR_ENERGY)
    ax.tick_params(axis="x", rotation=45)

    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

    # legenda combinada
    lines = ln1 + ln2
    labels_leg = [ln.get_label() for ln in lines]
    ax.legend(
        lines, labels_leg,
        loc="upper center", ncol=2,
        fontsize=9, framealpha=0.85,
        edgecolor="#cccccc",
    )

    # anotações de retenção
    if len(power) >= 2:
        p_ret = power[-1] / power[0] * 100 if power[0] != 0 else 0
        e_ret = energy[-1] / energy[0] * 100 if energy[0] != 0 else 0
        ax.annotate(
            f"Ret. Pot.: {p_ret:.1f}%  |  Ret. En.: {e_ret:.1f}%",
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#f8f9fa", edgecolor="#adb5bd", alpha=0.92,
            ),
        )

    ax.set_title(
        f"{filename}\nEnergia × Potência vs Ciclo",
        fontsize=12, fontweight="bold", pad=10,
    )

    if fig is not None:
        fig.tight_layout()

    # ---- salvar / exibir ----------------------------------------------------
    filepath: Optional[str] = None
    if standalone:
        if save:
            os.makedirs(out_dir, exist_ok=True)
            stem = Path(filename).stem
            filepath = os.path.join(out_dir, f"{stem}_energy_power.png")
            fig.savefig(filepath, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    return filepath
