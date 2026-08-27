"""
Figura 4 (GCD reduzida) a partir de dados brutos do BT-Lab (EC-Lab).

Gera a figura de 3 paineis:
  (a) curvas de carga/descarga em cada corrente aplicada
  (b) retencao de capacitancia (Cs) ao longo dos ~2100 ciclos, com
      comparacao explicita entre o bloco inicial de 1 A/g e o bloco de
      retorno a 1 A/g no final (recuperacao apos alta corrente)
  (c) grafico de Ragone (energia x potencia especificas), 1 ponto por
      corrente aplicada

Segue a mesma convencao do dunn_drt_analysis.py: parser read_bt/col,
matplotlib Agg, viridis, titulos numerados, config no final do arquivo.

------------------------------------------------------------------
COMO USAR
------------------------------------------------------------------
1) Preencha GCD_FILE com o caminho do .txt exportado do BT-Lab (a tecnica
   de GCD reduzida inteira, nao precisa recortar).
2) Preencha MASS_G com a massa ativa TOTAL da celula (soma dos dois
   eletrodos), em gramas -- e a mesma usada no calculo de Cs da celula
   inteira no seu script de Dunn.
3) Confira CURRENT_SEQUENCE_A_G -- e so' a sequencia nominal esperada
   (em A/g) na ordem em que os blocos aparecem no arquivo; serve para
   rotular a legenda e para o aviso de conferencia de massa. Nao precisa
   bater 100% se a celula tiver menos blocos (ex.: arquivo truncado).
4) Rode:  python3 gcd_figure4.py
   Gera <OUT_PREFIX>_fig4.png e <OUT_PREFIX>_ciclos.csv (dados por ciclo,
   utilizavel na Tabela 2 do relatorio).
------------------------------------------------------------------
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# Leitura de arquivos BT-Lab (.txt, separados por TAB, decimal com virgula)
# (identica ao dunn_drt_analysis.py, para manter os dois scripts consistentes)
# =====================================================================
def read_bt(path, encoding="latin-1"):
    with open(path, encoding=encoding) as f:
        reader = csv.reader(f, delimiter="\t")
        header = [h for h in next(reader) if h.strip() != ""]
        rows = []
        for r in reader:
            if not any(c.strip() for c in r):
                continue
            rows.append(r[: len(header)])
    return header, rows


def col(header, rows, name):
    """Devolve uma coluna como array de float (aceita decimal com virgula)."""
    idx = header.index(name)
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        out[i] = float(r[idx].replace(",", "."))
    return out


# =====================================================================
# Segmentacao automatica em blocos de corrente
# =====================================================================
def detect_current_blocks(ctrl_mA, round_decimals=4):
    """
    Agrupa linhas consecutivas com o mesmo |control/mA| (arredondado) em
    blocos. Cada bloco corresponde a uma corrente aplicada da GCD reduzida
    (ex.: 0.1, 0.5, 1, 5, 10 e o retorno a 1 A/g), na ORDEM em que aparecem
    no arquivo -- inclusive repetindo o mesmo nivel de corrente 2x, como
    no bloco de retorno.

    Devolve lista de (level_mA, idx_inicio, idx_fim) com idx_fim exclusivo.
    """
    level = np.round(np.abs(ctrl_mA), round_decimals)
    change_idx = np.where(np.diff(level) != 0)[0] + 1
    bounds = np.concatenate(([0], change_idx, [len(level)]))
    blocks = []
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        blocks.append((level[lo], lo, hi))
    return blocks


def infer_mass_g(blocks, current_sequence_a_g):
    """
    Estima a massa ativa total (g) comparando o |control/mA| de cada bloco
    com a densidade de corrente nominal esperada (A/g), na mesma ordem.
    So' serve como conferencia cruzada -- confirme sempre contra a massa
    pesada na fabricacao do eletrodo.
    """
    n = min(len(blocks), len(current_sequence_a_g))
    if n == 0:
        return None
    estimates = []
    for (level_mA, _, _), j_a_g in zip(blocks[:n], current_sequence_a_g[:n]):
        if j_a_g > 0:
            estimates.append((level_mA / 1000.0) / j_a_g)
    if not estimates:
        return None
    return float(np.median(estimates))


# =====================================================================
# Metricas por ciclo (dentro de cada bloco de corrente)
# =====================================================================
def cycle_metrics_for_block(header, rows, lo, hi, mass_g, j_label_a_g=None):
    """
    Para o intervalo de linhas [lo, hi) de UM bloco de corrente, calcula,
    para cada ciclo completo (com trecho de carga E de descarga presentes):
      t_charge (s), t_discharge (s), V no inicio e no fim da descarga,
      IR drop (queda imediata ao trocar de carga p/ descarga),
      Cs (F/g) -- janela de potencial completa da descarga (com IR drop),
      E (Wh/kg) e P (W/kg) a partir de Cs, deltaV e t_discharge.
    Ciclos incompletos nas bordas do bloco (ex.: arquivo truncado) sao
    ignorados.
    """
    cyc = col(header, rows[lo:hi], "cycle number")
    oxred = col(header, rows[lo:hi], "ox/red")
    t = col(header, rows[lo:hi], "time/s")
    E = col(header, rows[lo:hi], "Ecell/V")
    I = col(header, rows[lo:hi], "I/mA") / 1000.0  # mA -> A

    out = []
    for c in np.unique(cyc):
        m = cyc == c
        m_chg = m & (oxred == 1)
        m_dis = m & (oxred == 0)
        if not m_chg.any() or not m_dis.any():
            continue  # ciclo incompleto (borda do bloco/arquivo)

        t_charge = t[m_chg].max() - t[m_chg].min()
        t_discharge = t[m_dis].max() - t[m_dis].min()
        if t_charge <= 0 or t_discharge <= 0:
            continue

        v_before_discharge = E[m_chg][np.argmax(t[m_chg])]  # ultimo ponto da carga
        v_discharge_start = E[m_dis][np.argmin(t[m_dis])]
        v_discharge_end = E[m_dis][np.argmax(t[m_dis])]
        ir_drop = v_before_discharge - v_discharge_start
        delta_v = v_discharge_start - v_discharge_end
        if delta_v <= 0:
            continue

        i_charge = np.abs(I[m_chg]).mean()
        i_discharge = np.abs(I[m_dis]).mean()
        q_charge_c = i_charge * t_charge
        q_discharge_c = i_discharge * t_discharge
        coulombic_efficiency_pct = 100.0 * q_discharge_c / q_charge_c
        Cs = (i_discharge * t_discharge) / (mass_g * delta_v)  # F/g
        E_wh_kg = Cs * delta_v**2 / (2 * 3600.0) * 1000.0  # Wh/kg
        P_w_kg = E_wh_kg * 3600.0 / t_discharge  # W/kg

        out.append(
            dict(
                cycle=c,
                j_a_g=j_label_a_g,
                t_charge_s=t_charge,
                t_discharge_s=t_discharge,
                v_discharge_start=v_discharge_start,
                v_discharge_end=v_discharge_end,
                delta_v=delta_v,
                ir_drop=ir_drop,
                Cs_F_g=Cs,
                E_Wh_kg=E_wh_kg,
                P_W_kg=P_w_kg,
                coulombic_efficiency_pct=coulombic_efficiency_pct,
            )
        )
    return out


# =====================================================================
# Funcao principal
# =====================================================================
def build_figure4(gcd_file, mass_g, current_sequence_a_g, out_prefix="gcd", dpi=300):
    header, rows = read_bt(gcd_file)
    ctrl = col(header, rows, "control/mA")
    blocks = detect_current_blocks(ctrl)

    print(f"\n{len(blocks)} bloco(s) de corrente detectado(s) no arquivo:")
    inferred_mass = infer_mass_g(blocks, current_sequence_a_g)
    for i, (level_mA, lo, hi) in enumerate(blocks):
        label = (
            f"{current_sequence_a_g[i]:.2g} A/g"
            if i < len(current_sequence_a_g)
            else "(sem rotulo nominal)"
        )
        print(
            f"  bloco {i}: |control| = {level_mA:.4g} mA -> {label}  "
            f"(linhas {lo}:{hi}, n={hi - lo})"
        )
    if inferred_mass is not None:
        print(
            f"\nMassa ativa inferida a partir dos niveis de corrente: "
            f"{inferred_mass * 1000:.4f} mg"
        )
        print(f"Massa configurada (MASS_G): {mass_g * 1000:.4f} mg")
        if abs(inferred_mass - mass_g) / mass_g > 0.05:
            print(
                "  [!] Diferenca > 5% entre massa inferida e configurada -- "
                "confira MASS_G ou a sequencia CURRENT_SEQUENCE_A_G."
            )

    # ---------------- metricas por ciclo, bloco a bloco ----------------
    all_cycles = []
    block_labels = []
    for i, (level_mA, lo, hi) in enumerate(blocks):
        j_label = current_sequence_a_g[i] if i < len(current_sequence_a_g) else None
        cycles = cycle_metrics_for_block(header, rows, lo, hi, mass_g, j_label)
        if not cycles:
            print(f"  [aviso] bloco {i} nao rendeu nenhum ciclo completo (ignorado).")
            continue
        for cy in cycles:
            cy["block"] = i
        all_cycles.extend(cycles)
        block_labels.append(i)

    if not all_cycles:
        raise ValueError("Nenhum ciclo completo encontrado em nenhum bloco.")

    print("\nEficiencia coulombica por bloco:")
    for i in sorted(set(cy["block"] for cy in all_cycles)):
        efficiencies = [
            cy["coulombic_efficiency_pct"] for cy in all_cycles if cy["block"] == i
        ]
        label = (
            f"{current_sequence_a_g[i]:.2g} A/g"
            if i < len(current_sequence_a_g)
            else f"bloco {i}"
        )
        print(
            f"  {label}: {np.mean(efficiencies):.2f}% "
            f"(min {np.min(efficiencies):.2f}%, "
            f"max {np.max(efficiencies):.2f}%, n={len(efficiencies)})"
        )

    # ---------------- export CSV (para a Tabela 2 do relatorio) ----------------
    csv_path = f"{out_prefix}_ciclos.csv"
    fieldnames = [
        "block",
        "j_a_g",
        "cycle",
        "t_charge_s",
        "t_discharge_s",
        "v_discharge_start",
        "v_discharge_end",
        "delta_v",
        "ir_drop",
        "Cs_F_g",
        "E_Wh_kg",
        "P_W_kg",
        "coulombic_efficiency_pct",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cy in all_cycles:
            writer.writerow({k: cy.get(k) for k in fieldnames})
    print(f"\nDados por ciclo salvos em {csv_path}")

    # ---------------- (a) curvas de carga/descarga por corrente ----------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(blocks)))

    for color, i, (level_mA, lo, hi) in zip(colors, range(len(blocks)), blocks):
        cycles_i = [cy for cy in all_cycles if cy["block"] == i]
        if not cycles_i:
            continue
        # ciclo representativo: o do meio da lista de ciclos completos do bloco
        cy_mid = cycles_i[len(cycles_i) // 2]
        c = cy_mid["cycle"]

        cyc_col = col(header, rows[lo:hi], "cycle number")
        oxred_col = col(header, rows[lo:hi], "ox/red")
        t_col = col(header, rows[lo:hi], "time/s")
        E_col = col(header, rows[lo:hi], "Ecell/V")
        m = cyc_col == c
        t0 = t_col[m].min()
        order = np.argsort(t_col[m])
        t_rel = t_col[m][order] - t0
        E_rel = E_col[m][order]

        label = (
            f"{current_sequence_a_g[i]:.2g} A/g"
            if i < len(current_sequence_a_g)
            else f"bloco {i}"
        )
        axes[0].plot(t_rel, E_rel, color=color, linewidth=1.3, label=label)

    axes[0].set_xlabel("Tempo (s)")
    axes[0].set_ylabel("Ecell (V)")
    axes[0].set_title("1- Curvas de carga/descarga por corrente")
    axes[0].legend(title="Densidade de corrente", fontsize=8, title_fontsize=8)
    axes[0].grid(alpha=0.3)

    # ---------------- (b) retencao ao longo dos ciclos ----------------
    cycles_sorted = sorted(all_cycles, key=lambda cy: cy["cycle"])
    cyc_arr = np.array([cy["cycle"] for cy in cycles_sorted])
    Cs_arr = np.array([cy["Cs_F_g"] for cy in cycles_sorted])
    block_arr = np.array([cy["block"] for cy in cycles_sorted])

    for i in sorted(set(block_arr)):
        m = block_arr == i
        label = (
            f"{current_sequence_a_g[i]:.2g} A/g"
            if i < len(current_sequence_a_g)
            else f"bloco {i}"
        )
        axes[1].plot(
            cyc_arr[m], Cs_arr[m], ".", color=colors[i], markersize=3, label=label
        )

    # recuperacao: compara o 1o bloco de 1 A/g com o bloco de retorno a 1 A/g
    # (identificados por current_sequence_a_g == valor repetido)
    j_arr = np.array(current_sequence_a_g[: len(blocks)])
    idx_1ag = np.where(np.isclose(j_arr, j_arr[j_arr > 0].min() if False else np.nan))[
        0
    ]
    repeated = {}
    for i, j in enumerate(j_arr):
        repeated.setdefault(j, []).append(i)
    recovery_text = None
    for j, idxs in repeated.items():
        if len(idxs) >= 2:
            first_block, last_block = idxs[0], idxs[-1]
            Cs_first = Cs_arr[block_arr == first_block]
            Cs_last = Cs_arr[block_arr == last_block]
            if len(Cs_first) and len(Cs_last):
                recovery_pct = 100 * np.mean(Cs_last) / np.mean(Cs_first)
                recovery_text = (
                    f"Recuperacao em {j:.2g} A/g: {recovery_pct:.1f}%\n"
                    f"({np.mean(Cs_first):.1f} -> {np.mean(Cs_last):.1f} F/g)"
                )
                print(f"\n{recovery_text}")

    axes[1].set_xlabel("Numero do ciclo")
    axes[1].set_ylabel("Cs (F/g)")
    axes[1].set_title("2- Retencao de capacitancia")
    axes[1].set_yscale("log")
    axes[1].legend(title="Densidade de corrente", fontsize=8, title_fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    if recovery_text:
        axes[1].text(
            0.02,
            0.04,
            recovery_text,
            transform=axes[1].transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.85,
            ),
        )

    # ---------------- (c) Ragone ----------------
    E_means, P_means, labels_r = [], [], []
    for i in sorted(set(block_arr)):
        m = block_arr == i
        if not m.any():
            continue
        E_means.append(
            np.mean([cy["E_Wh_kg"] for cy in cycles_sorted if cy["block"] == i])
        )
        P_means.append(
            np.mean([cy["P_W_kg"] for cy in cycles_sorted if cy["block"] == i])
        )
        labels_r.append(
            f"{current_sequence_a_g[i]:.2g} A/g"
            if i < len(current_sequence_a_g)
            else f"bloco {i}"
        )

    for i, (E_i, P_i, lab) in enumerate(zip(E_means, P_means, labels_r)):
        axes[2].plot(P_i, E_i, "o", color=colors[i], markersize=9, label=lab)

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Potencia especifica (W/kg)")
    axes[2].set_ylabel("Energia especifica (Wh/kg)")
    axes[2].set_title("3- Ragone")
    axes[2].legend(
        title="Densidade de corrente",
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )
    axes[2].grid(alpha=0.3, which="both")

    # Usa a posição real do ponto no painel para manter cada anotação próxima
    # dele. O deslocamento é em fração do eixo log, não em pontos fixos.
    x_min, x_max = axes[2].get_xlim()
    y_min, y_max = axes[2].get_ylim()
    log_x_range = np.log10(x_max) - np.log10(x_min)
    log_y_range = np.log10(y_max) - np.log10(y_min)
    point_fractions = []
    for i, (E_i, P_i, _lab) in enumerate(zip(E_means, P_means, labels_r)):
        point_fractions.append(
            (
                i,
                E_i,
                P_i,
                (np.log10(P_i) - np.log10(x_min)) / log_x_range,
                (np.log10(E_i) - np.log10(y_min)) / log_y_range,
            )
        )

    # Cada caixa ocupa aproximadamente 22% x 7% do painel. A busca gulosa
    # testa posições locais acima/abaixo e nos dois lados, evitando que pontos
    # próximos no Ragone tenham textos sobrepostos.
    occupied = []
    point_boxes = {
        item[0]: (
            max(0.0, item[3] - 0.025),
            min(1.0, item[3] + 0.025),
            max(0.0, item[4] - 0.035),
            min(1.0, item[4] + 0.035),
        )
        for item in point_fractions
    }
    for i, E_i, P_i, point_x, point_y in sorted(
        point_fractions, key=lambda item: item[4], reverse=True
    ):
        candidates = []
        for distance in (0.055, 0.11, 0.18, 0.27):
            for horizontal in (-1, 1):
                for vertical in (1, -1):
                    candidates.append(
                        (
                            point_x + horizontal * distance,
                            point_y + vertical * (0.065 + distance * 0.18),
                        )
                    )
        best = None
        for candidate_x, candidate_y in candidates:
            center_x = min(0.86, max(0.14, candidate_x))
            center_y = min(0.76, max(0.10, candidate_y))
            box = (
                center_x - 0.115,
                center_x + 0.115,
                center_y - 0.038,
                center_y + 0.038,
            )
            overlap = sum(
                max(0.0, min(box[1], other[1]) - max(box[0], other[0]))
                * max(0.0, min(box[3], other[3]) - max(box[2], other[2]))
                for other in occupied
            )
            point_overlap = sum(
                max(0.0, min(box[1], point_box[1]) - max(box[0], point_box[0]))
                * max(0.0, min(box[3], point_box[3]) - max(box[2], point_box[2]))
                for point_index, point_box in point_boxes.items()
                if point_index != i
            )
            distance_from_point = abs(center_x - point_x) + abs(center_y - point_y)
            score = (overlap + point_overlap) * 1000.0 + distance_from_point
            if best is None or score < best[0]:
                best = (score, center_x, center_y, box)

        _score, target_x, target_y, box = best
        occupied.append(box)
        axes[2].annotate(
            f"P = {P_i:.1f} W/kg\nE = {E_i:.3g} Wh/kg",
            xy=(P_i, E_i),
            xycoords="data",
            xytext=(target_x, target_y),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-",
                color=colors[i],
                linewidth=0.8,
                shrinkA=0,
                shrinkB=4,
            ),
            fontsize=7,
            color=colors[i],
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=colors[i],
                alpha=0.8,
            ),
        )

    fig.tight_layout()
    out_png = f"{out_prefix}_fig4.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"\nFigura salva em {out_png}")

    return dict(cycles=all_cycles, blocks=blocks, inferred_mass_g=inferred_mass)


# =====================================================================
# CONFIGURACAO -- preencha aqui
# =====================================================================
_DATA_DIR = Path(__file__).resolve().parent

GCD_FILE = _DATA_DIR / "-Nb2CTx L3 - Na2SO4/GCDrEDpROG_02_MB_CA4.txt"
MASS_G = 0.0056  # massa ativa TOTAL da celula (g) -- confira contra a massa pesada

# sequencia nominal esperada, na ordem em que os blocos aparecem no arquivo
CURRENT_SEQUENCE_A_G = [1, 10, 7.5, 5, 2.5, 1]  # A/g

OUT_PREFIX = "gcd"

if __name__ == "__main__":
    build_figure4(GCD_FILE, MASS_G, CURRENT_SEQUENCE_A_G, OUT_PREFIX)
