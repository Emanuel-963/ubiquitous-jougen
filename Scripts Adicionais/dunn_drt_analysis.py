"""
Analise Dunn (contribuicao capacitiva x difusional) para dados de
voltametria ciclica exportados do BT-Lab (EC-Lab).

Requisitos: numpy, scipy, matplotlib  (todos ja disponiveis)

------------------------------------------------------------------
COMO USAR
------------------------------------------------------------------
1) Dunn (b-value + separacao capacitiva/difusional):
   - Preencha o dicionario CV_FILES no final do script com o caminho de
     cada arquivo de CV (um por velocidade de varredura, em V/s).
     - Preencha ELECTRODES com a massa ativa (g) de cada eletrodo (WE e CE)
         e com a fracao da janela de potencial de cada um.
   - Preencha CELL_AREA_CM2 com a area geometrica total da celula Swagelok
     (cm2); cada eletrodo usa metade dessa area.
    - No Windows, rode: ./.venv/Scripts/python.exe dunn_drt_analysis.py --dunn

------------------------------------------------------------------
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# =====================================================================
# Leitura de arquivos de CV (.txt BT-Lab e CV staircase, decimal com virgula)
# =====================================================================
def read_bt(path, encoding="latin-1"):
    with open(path, encoding=encoding) as f:
        first_line = f.readline()
        delimiter = ";" if first_line.count(";") > first_line.count("\t") else "\t"
        header = [
            h.strip().lstrip("\ufeff")
            for h in first_line.split(delimiter)
            if h.strip() != ""
        ]
        reader = csv.reader(f, delimiter=delimiter)
        rows = []
        for r in reader:
            if not any(c.strip() for c in r):
                continue
            if len(r) < len(header):
                continue
            rows.append([c.strip() for c in r[: len(header)]])
    return header, rows


def col(header, rows, name):
    """Devolve uma coluna como array de float (aceita decimal com virgula)."""
    idx = header.index(name)
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        out[i] = float(r[idx].replace(",", "."))
    return out


# =====================================================================
# 1) ANALISE DE DUNN
# =====================================================================
def resolve_cv_files(cv_files):
    """Mantem apenas velocidades cujo arquivo existe; avisa sobre ausentes."""
    available = {}
    missing = []
    for v, path in cv_files.items():
        p = Path(path)
        if p.is_file():
            available[v] = p
        else:
            missing.append((v, p))

    if missing:
        print("Arquivos CV ausentes (ignorados):")
        for v, p in missing:
            print(f"  {v * 1000:.1f} mV/s -> {p}")

    return available


def load_cv(path, which_cycle="last"):
    """
    Le um CV do BT-Lab e devolve (Ecell [V], I [A]) de UM UNICO ciclo completo.

    Muitos arquivos do BT-Lab gravam mais de 1 ciclo de CV na mesma tecnica
    (a coluna 'cycle number' as vezes fica constante mesmo havendo 2+ voltas
    fisicas). Para nao contar area de dois ciclos como se fosse 1 (dobrando
    Cs), o ciclo e delimitado prioritariamente pela coluna explicita 'Cycle'
    (arquivos CV staircase) ou pelas mudancas de 'counter inc.' (BT-Lab). Por
    padrao pega-se o ULTIMO ciclo completo (which_cycle='last'). Se nenhum
    marcador estiver disponivel, usam-se minimos locais consecutivos de Ecell.
    Use which_cycle='first' ou um inteiro (0-indexado) para escolher outro.
    """
    header, rows = read_bt(path)
    potential_name = next(
        (name for name in ("Ecell/V", "WE(1).Potential (V)") if name in header),
        None,
    )
    current_name = next(
        (name for name in ("<I>/mA", "I/mA", "WE(1).Current (A)") if name in header),
        None,
    )
    if potential_name is None or current_name is None:
        raise ValueError(
            f"{Path(path).name}: colunas de potencial/corrente nao reconhecidas. "
            f'Encontradas: {", ".join(header)}'
        )

    E = col(header, rows, potential_name)
    I = col(header, rows, current_name)
    if current_name.endswith("/mA"):
        I /= 1000.0  # mA -> A

    def choose_cycle(cycle_values):
        cycle_values = np.asarray(cycle_values)
        starts = np.r_[0, np.flatnonzero(cycle_values[1:] != cycle_values[:-1]) + 1]
        ends = np.r_[starts[1:], len(cycle_values)]
        if which_cycle == "last":
            cycle_index = len(starts) - 1
        elif which_cycle == "first":
            cycle_index = 0
        else:
            cycle_index = int(which_cycle)
        if not 0 <= cycle_index < len(starts):
            raise ValueError(
                f"Ciclo {cycle_index} invalido: o arquivo contem "
                f"{len(starts)} ciclo(s) marcado(s)."
            )
        return starts[cycle_index], ends[cycle_index]

    # Arquivos CV staircase trazem o numero do ciclo diretamente.
    if "Cycle" in header:
        cycle_idx = header.index("Cycle")
        lo, hi = choose_cycle([row[cycle_idx] for row in rows])
        return E[lo:hi], I[lo:hi]

    # O contador do BT-Lab delimita o ciclo sem depender da forma, dos minimos
    # ou dos maximos da curva. Cada intervalo entre duas mudancas consecutivas
    # e um ciclo completo; um eventual trecho posterior e descartado.
    if "counter inc." in header:
        counter_idx = header.index("counter inc.")
        counter = np.asarray([row[counter_idx].strip() for row in rows])
        boundaries = np.flatnonzero(counter[1:] != counter[:-1]) + 1
        if len(boundaries) >= 2:
            n_cycles = len(boundaries) - 1
            if which_cycle == "last":
                c = n_cycles - 1
            elif which_cycle == "first":
                c = 0
            else:
                c = int(which_cycle)
            if not 0 <= c < n_cycles:
                raise ValueError(
                    f"Ciclo {c} invalido: o arquivo contem "
                    f"{n_cycles} ciclo(s) completo(s)."
                )
            lo, hi = boundaries[c], boundaries[c + 1]
            return E[lo:hi], I[lo:hi]

    # Alternativa para arquivos sem pelo menos dois marcadores de contador:
    # usa os pontos de minimo de potencial como fronteiras dos ciclos.
    from scipy.signal import find_peaks

    dE_full = E.max() - E.min()
    peaks, _ = find_peaks(-E, prominence=0.1 * dE_full, distance=max(3, len(E) // 5))
    # Um ciclo completo vai de um minimo de potencial ao minimo seguinte.
    # O arquivo frequentemente termina durante a varredura seguinte; portanto,
    # o trecho do ultimo minimo ate o fim nao e tratado como ciclo completo.
    if len(peaks) < 2:
        return E, I  # nao ha dois limites para isolar um ciclo completo

    n_cycles = len(peaks) - 1
    if which_cycle == "last":
        c = n_cycles - 1
    elif which_cycle == "first":
        c = 0
    else:
        c = int(which_cycle)

    if not 0 <= c < n_cycles:
        raise ValueError(
            f"Ciclo {c} invalido: o arquivo contem {n_cycles} ciclo(s) completo(s)."
        )

    lo, hi = peaks[c], peaks[c + 1]
    return E[lo : hi + 1], I[lo : hi + 1]


def dunn_analysis(
    cv_files, electrodes, cell_area_cm2, n_points=200, out_prefix="dunn", dpi=300
):
    """
    cv_files: dict {scan_rate_V_s: caminho_do_arquivo}
    electrodes: dict {nome: {'mass_g': ..., 'potential_fraction': ...}}.
        A carga Q e a mesma nos dois eletrodos. A capacitancia de cada um e
        C_e = Q / DeltaV_e, em que DeltaV_e = potential_fraction * DeltaV.
        As fracoes devem somar 1; use 0.5 para cada eletrodo quando nao houver
        medidas individuais de potencial (hipotese de divisao simetrica).
    cell_area_cm2: area geometrica total da celula Swagelok (cm2); cada
        eletrodo usa metade (A_e = cell_area_cm2 / 2).

    Metodo:
    - Interpola todas as curvas de CV para uma grade comum de potencial.
    - Para cada potencial E, ajusta i(E) = k1(E)*v + k2(E)*v^0.5 (Dunn 1990),
      via regressao linear de i/sqrt(v) vs sqrt(v).
    - Contribuicao capacitiva a cada v: i_cap = k1*v ; fracao capacitiva =
      integral(i_cap dE) / integral(i_total dE) no ciclo completo.
    - b-value: log(i_peak) = b*log(v) + log(a) no potencial de maior corrente.
    """
    cv_files = resolve_cv_files(cv_files)
    if not cv_files:
        raise ValueError("Nenhum arquivo CV encontrado em CV_FILES.")

    rates = np.array(sorted(cv_files.keys()))
    if len(rates) < 2:
        raise ValueError(
            f"Dunn precisa de pelo menos 2 velocidades com arquivo encontrado "
            f"(encontradas: {len(rates)})."
        )

    print(f"\nUsando {len(rates)} velocidade(s) de varredura com arquivo encontrado.")

    curves = {}
    emin, emax = -np.inf, np.inf
    for v in rates:
        E, I = load_cv(cv_files[v])
        curves[v] = (E, I)
        emin = max(emin, E.min())
        emax = min(emax, E.max())

    E_grid = np.linspace(emin, emax, n_points)

    # separa cada curva em ramo de ida (oxidacao) e volta (reducao) para
    # interpolar sem ambiguidade (E nao e monotonico numa CV completa)
    I_grid = np.zeros((len(rates), n_points))
    for k, v in enumerate(rates):
        E, I = curves[v]
        mid = np.argmax(E)  # pico do potencial separa ida/volta
        E_up, I_up = E[: mid + 1], I[: mid + 1]
        E_dn, I_dn = E[mid:], I[mid:]
        order = np.argsort(E_up)
        Iu = np.interp(E_grid, E_up[order], I_up[order])
        order2 = np.argsort(E_dn)
        Id = np.interp(E_grid, E_dn[order2], I_dn[order2])
        I_grid[k] = (Iu + Id) / 2.0  # media dos dois ramos p/ 1 valor por E

    sqrt_v = np.sqrt(rates)

    k1 = np.zeros(n_points)
    k2 = np.zeros(n_points)
    r2 = np.zeros(n_points)
    for j in range(n_points):
        y = I_grid[:, j] / sqrt_v  # i/sqrt(v)
        x = sqrt_v  # sqrt(v)
        slope, intercept, r, p, se = stats.linregress(x, y)
        k1[j] = slope
        k2[j] = intercept
        r2[j] = r**2

    # Fracao capacitiva por velocidade. Pela definicao fisica, em cada
    # potencial vale |i_cap| <= |i_total|. A regressao linear sem restricao
    # pode ultrapassar esse limite em dados ruidosos; nesse caso, limita-se a
    # componente capacitiva ao valor medido, preservando o sinal da corrente.
    frac_cap = {}
    I_cap_grid = np.zeros_like(I_grid)
    for k, v in enumerate(rates):
        i_total = I_grid[k]
        i_cap_magnitude = np.minimum(np.abs(k1 * v), np.abs(i_total))
        i_cap = np.sign(i_total) * i_cap_magnitude
        I_cap_grid[k] = i_cap
        q_total = np.trapezoid(np.abs(i_total), E_grid)
        q_cap = np.trapezoid(np.abs(i_cap), E_grid)
        frac_cap[v] = 100 * q_cap / q_total if q_total > 0 else np.nan

    # b-value no ponto de maior |corrente| media entre as velocidades
    j_peak = np.argmax(np.mean(np.abs(I_grid), axis=0))
    i_peak = np.abs(I_grid[:, j_peak])
    b_slope, b_intercept, b_r, b_p, b_se = stats.linregress(
        np.log(rates), np.log(i_peak)
    )

    # A carga por ciclo em cada eletrodo e Q = integral(|I| dE) / (2 v).
    # Em uma celula de dois eletrodos, ambos conduzem a mesma carga, mas suas
    # capacitancias especificas e capacidades diferem por massa. A area de
    # cada eletrodo e metade da area geometrica total, salvo configuracao
    # explicita de 'area_cm2' no dicionario do eletrodo.
    if not electrodes:
        raise ValueError("Informe pelo menos um eletrodo em ELECTRODES.")

    fraction_sum = sum(e["potential_fraction"] for e in electrodes.values())
    if not np.isclose(fraction_sum, 1.0):
        raise ValueError(
            "As potential_fraction de ELECTRODES devem somar 1.0; "
            f"valor atual: {fraction_sum:.3f}."
        )

    for name, electrode in electrodes.items():
        if electrode["mass_g"] <= 0:
            raise ValueError(f"A massa de {name} deve ser maior que zero.")
        if electrode["potential_fraction"] <= 0:
            raise ValueError(f"A potential_fraction de {name} deve ser maior que zero.")

    Cs = {name: {} for name in electrodes}
    Ca_cm2 = {name: {} for name in electrodes}
    cap_mAh_g = {name: {} for name in electrodes}
    cell_Cs = {}
    cell_Ca_cm2 = {}
    cell_cap_mAh_g = {}
    total_mass_g = sum(electrode["mass_g"] for electrode in electrodes.values())
    for v in rates:
        E, I = curves[v]
        area = np.sum(np.abs((I[1:] + I[:-1]) / 2) * np.abs(np.diff(E)))
        dE = E.max() - E.min()
        charge_c = area / (2 * v)
        cell_capacitance_f = charge_c / dE
        cell_Cs[v] = cell_capacitance_f / total_mass_g
        cell_Ca_cm2[v] = cell_capacitance_f / cell_area_cm2
        cell_cap_mAh_g[v] = charge_c / (3.6 * total_mass_g)
        for name, electrode in electrodes.items():
            mass_g = electrode["mass_g"]
            area_cm2 = electrode.get("area_cm2", cell_area_cm2 / 2)
            delta_electrode = dE * electrode["potential_fraction"]
            capacitance_f = charge_c / delta_electrode
            Cs[name][v] = capacitance_f / mass_g
            Ca_cm2[name][v] = capacitance_f / area_cm2
            cap_mAh_g[name][v] = charge_c / (3.6 * mass_g)

    # ---------------- resultados em texto ----------------
    print("\n=== ANALISE DE DUNN ===")
    for name, electrode in electrodes.items():
        area_cm2 = electrode.get("area_cm2", cell_area_cm2 / 2)
        print(
            f'\n--- {name}: massa = {electrode["mass_g"]:.6g} g, '
            f"area = {area_cm2:.4f} cm2, "
            f'janela = {electrode["potential_fraction"] * 100:.1f}% de DeltaV ---'
        )
        print(
            f'{"v (mV/s)":>10} | {"Cs (F/g)":>10} | {"Ca (F/cm2)":>11} | {"Cap. (mAh/g)":>13} | {"% capacitivo*":>14}'
        )
        for v in rates:
            print(
                f"{v*1000:>10.1f} | {Cs[name][v]:>10.2f} | {Ca_cm2[name][v]:>11.4f} | "
                f"{cap_mAh_g[name][v]:>13.2f} | {frac_cap[v]:>14.1f}"
            )
    print(
        f"\n--- Celula inteira: massa total = {total_mass_g:.6g} g, "
        f"area total = {cell_area_cm2:.4f} cm2, janela = 100.0% de DeltaV ---"
    )
    print(
        f'{"v (mV/s)":>10} | {"Cs (F/g)":>10} | {"Ca (F/cm2)":>11} | '
        f'{"Cap. (mAh/g)":>13} | {"% capacitivo*":>14}'
    )
    for v in rates:
        print(
            f"{v*1000:>10.1f} | {cell_Cs[v]:>10.2f} | "
            f"{cell_Ca_cm2[v]:>11.4f} | {cell_cap_mAh_g[v]:>13.2f} | "
            f"{frac_cap[v]:>14.1f}"
        )
    print("\n* A fracao capacitiva e uma propriedade da resposta da celula completa;")
    print(
        "  ela nao pode ser separada por eletrodo sem medidas individuais de potencial."
    )
    print(f"\nb-value (no pico de corrente): b = {b_slope:.3f}  (R2 = {b_r**2:.4f})")
    print("  b ~ 1.0  -> processo capacitivo/superficial")
    print("  b ~ 0.5  -> processo controlado por difusao")

    # erro da regressao do b-value (info estatistica real, sem precisar de replicatas)
    print(f"  erro padrao do slope (b): {b_se:.3f}")

    # ---------------- graficos ----------------
    # Figura 1: voltamogramas completos e capacitancia especifica.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rates)))

    for color, v in zip(colors, rates):
        E, I = curves[v]
        axes[0].plot(
            E,
            I * 1000,
            color=color,
            linewidth=1.4,
            label=(
                f"{v * 1000:.0f} mV/s - "
                f"{cell_Cs[v]:.2f} F/g - {cell_Ca_cm2[v] * 1000:.2f} mF/cm²"
            ),
        )
    axes[0].set_xlabel("Ecell (V)")
    axes[0].set_ylabel("I (mA)")
    axes[0].set_title("1- Voltamogramas cíclicos")
    axes[0].legend(
        title="Velocidade - Cs (celula) - Ca (celula)",
        loc="lower right",
        fontsize=7.5,
        title_fontsize=8,
    )
    axes[0].grid(alpha=0.3)

    for name in electrodes:
        axes[1].plot(rates * 1000, [Cs[name][v] for v in rates], "o-", label=name)
    axes[1].plot(
        rates * 1000,
        [cell_Cs[v] for v in rates],
        "s--",
        color="black",
        label="Célula inteira",
    )
    axes[1].set_xlabel("Velocidade de varredura (mV/s)")
    axes[1].set_ylabel("Cs (F/g)")
    axes[1].set_title("2- Capacitância específica por velocidade")
    axes[1].legend(title="Amostra")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_resultado.png", dpi=dpi)
    plt.close(fig)
    print(f"\nGrafico salvo em {out_prefix}_resultado.png")

    # Figura 2: contribuicoes de Dunn, b-value e separacao na maior velocidade.
    v_top = rates.max()
    idx = list(rates).index(v_top)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))

    cap_percent = np.array([frac_cap[v] for v in rates])
    faradaic_percent = 100 - cap_percent
    rate_labels = [f"{v * 1000:.0f}" for v in rates]
    y_pos = np.arange(len(rates))
    cap_color = "#2ca02c"
    faradaic_color = "#c13a82"
    axes2[1].barh(y_pos, cap_percent, color=cap_color, label="Contribuição capacitiva")
    axes2[1].barh(
        y_pos,
        faradaic_percent,
        left=cap_percent,
        color=faradaic_color,
        label="Contribuição faradaica",
    )
    for y, cap, far in zip(y_pos, cap_percent, faradaic_percent):
        axes2[1].text(
            cap / 2,
            y,
            f"{cap:.0f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
        axes2[1].text(
            cap + far / 2,
            y,
            f"{far:.0f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    axes2[1].set_yticks(y_pos, rate_labels)
    axes2[1].set_xlabel("Contribuição (%)")
    axes2[1].set_ylabel("Velocidade de varredura (mV/s)")
    axes2[1].set_xlim(0, 100)
    axes2[1].set_ylim(len(rates) - 0.5, -1.15)
    axes2[1].set_title("2- Contribuições capacitiva e faradaica")
    axes2[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        fontsize=8,
        frameon=True,
    )
    axes2[1].grid(alpha=0.25, axis="x")

    axes2[0].plot(np.log(rates), np.log(i_peak), "o", color="#d62728")
    xx = np.linspace(np.log(rates).min(), np.log(rates).max(), 50)
    axes2[0].plot(
        xx,
        b_slope * xx + b_intercept,
        "--",
        color="gray",
        label=f"b = {b_slope:.2f} (R²={b_r**2:.3f})",
    )
    axes2[0].set_xlabel("ln(v)")
    axes2[0].set_ylabel("ln(i_pico)")
    axes2[0].set_title("1- b-value")
    axes2[0].legend()
    axes2[0].text(
        0.98,
        0.04,
        "b ≈ 1: predominantemente capacitivo\n" "b ≈ 0,5: controlado por difusão",
        transform=axes2[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.85
        ),
    )
    axes2[0].grid(alpha=0.3)

    i_total_top = I_grid[idx] * 1000
    i_cap_top = I_cap_grid[idx] * 1000
    axes2[2].plot(
        E_grid,
        i_total_top,
        color="black",
        linewidth=1.7,
        label=f"i total ({v_top * 1000:.0f} mV/s)",
    )
    axes2[2].fill_between(
        E_grid,
        0,
        i_cap_top,
        color=cap_color,
        alpha=0.45,
        label="Contribuição capacitiva",
    )
    axes2[2].fill_between(
        E_grid,
        i_cap_top,
        i_total_top,
        color=faradaic_color,
        alpha=0.45,
        label="Contribuição faradaica",
    )
    axes2[2].set_xlabel("Ecell (V)")
    axes2[2].set_ylabel("I (mA)")
    axes2[2].set_title(f"3- Separação capacitiva/faradaica a {v_top*1000:.0f} mV/s")
    axes2[2].legend()
    axes2[2].grid(alpha=0.3)
    # `tight_layout` pode comprimir excessivamente os paineis quando mudam o
    # numero de velocidades, o tamanho das legendas ou os valores exibidos.
    # Margens fixas preservam tres paineis de largura equivalente.
    fig2.subplots_adjust(left=0.055, right=0.985, bottom=0.14, top=0.90, wspace=0.32)
    fig2.savefig(f"{out_prefix}_separacao.png", dpi=dpi)
    plt.close(fig2)
    print(f"Grafico salvo em {out_prefix}_separacao.png")

    return dict(
        Cs=Cs,
        Ca_cm2=Ca_cm2,
        cap_mAh_g=cap_mAh_g,
        cell_Cs=cell_Cs,
        cell_Ca_cm2=cell_Ca_cm2,
        cell_cap_mAh_g=cell_cap_mAh_g,
        frac_cap=frac_cap,
        b_value=b_slope,
        b_r2=b_r**2,
        b_stderr=b_se,
    )


# =====================================================================
# CONFIGURACAO - preencha aqui com seus arquivos
# =====================================================================
_DATA_DIR = Path(__file__).resolve().parent
_CV_DIR = (
    _DATA_DIR / "Nb2CTx Lote 3 + Nego  de Fumo"
)  # pasta com os arquivos de CV exportados do BT-Lab

CV_FILES = {
    # velocidade (V/s): caminho do arquivo
    0.005: _CV_DIR / "FulltestC2_02_CV_CA2.txt",
    0.010: _CV_DIR / "FulltestC2_03_CV_CA2.txt",
    0.020: _CV_DIR / "FulltestC2_04_CV_CA2.txt",
    0.025: _CV_DIR / "",
    0.050: _CV_DIR / "FulltestC2_05_CV_CA2.txt",
    0.075: _CV_DIR / "",
    0.100: _CV_DIR / "FulltestC2_06_CV_CA2.txt",
    0.150: _CV_DIR / "",
    0.200: _CV_DIR / "FulltestC2_07_CV_CA2.txt",
}
CELL_AREA_CM2 = 2.4544  # cm2; cada eletrodo ocupa, por padrao, metade desta area
ELECTRODES = {
    # Ajuste as massas ativas. potential_fraction deve refletir a janela de
    # potencial de cada eletrodo e as fracoes devem somar 1. Sem dados de
    # potencial individual, 0.5/0.5 e a hipotese simetrica usual.
    "WE": {"mass_g": 0.0032, "potential_fraction": 0.5},
    "CE": {"mass_g": 0.0033, "potential_fraction": 0.5},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dunn",
        action="store_true",
        help="Executa a analise Dunn (opcional; e o padrao).",
    )
    args = parser.parse_args()

    try:
        dunn_analysis(CV_FILES, ELECTRODES, CELL_AREA_CM2)
    except ValueError as e:
        print(f"\n[Dunn nao executado] {e}")
        print("Preencha CV_FILES com pelo menos 2 arquivos de CV existentes.")
