"""
Figura 5 (EIS + circuito equivalente) a partir de dados brutos de PEIS do
BT-Lab (EC-Lab).

Ajusta, para cada estado (inicial / pos-CV / final), um circuito de Randles
modificado:

    Z(w) = Rs + (Rct // CPE) + Zw

  Rs       -- resistencia em serie (eletrolito + contatos)
  Rct      -- resistencia de transferencia de carga / interfacial
  CPE      -- elemento de fase constante em paralelo com Rct (Q, n)
              Z_CPE = 1 / (Q * (jw)^n)
  Zw       -- Warburg de comprimento finito (fronteira refletora, tipico
              de poros/particulas finitas): Zw = Rw * tanh(sqrt(jw*tau)) /
              sqrt(jw*tau) -- Rw = resistencia difusiva, tau = B^2/D
              (tempo caracteristico de difusao)

O ajuste e' feito por minimos quadrados nao lineares (scipy), com residuo
ponderado por 1/|Z| (peso de modulo -- pratica padrao em EIS para nao deixar
os pontos de alta frequencia, que tem |Z| pequeno, dominarem o ajuste).

Segue a mesma convencao dos outros scripts: parser read_bt/col, matplotlib
Agg, titulos numerados, config no final do arquivo.

------------------------------------------------------------------
COMO USAR
------------------------------------------------------------------
1) Preencha EIS_FILES com o caminho de cada estado (inicial, pos-CV, final).
   Se um estado nao tiver arquivo valido (ex.: EIS final que falhou), deixe
   None -- o script pula esse estado e avisa.
2) Rode: python3 eis_figure5.py
   Gera <OUT_PREFIX>_fig5.png e <OUT_PREFIX>_parametros.csv (para a Tabela 3).
------------------------------------------------------------------
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# Leitura de arquivos BT-Lab (.txt, separados por TAB, decimal com virgula)
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
    idx = header.index(name)
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        out[i] = float(r[idx].replace(",", "."))
    return out


def load_eis(path, exclude_last_n=0):
    """
    Devolve (freq [Hz], Z [complexo]) de um arquivo PEIS do BT-Lab.
    A coluna '-Im(Z)/Ohm' ja' vem com o sinal invertido (convencao do
    Nyquist com eixo Y positivo para cima), entao Im(Z) real = -coluna.

    exclude_last_n: descarta os N pontos de MENOR frequencia (cauda de
    baixa frequencia). Uteis quando a medida fica instavel no fim da
    varredura (Re(Z) negativo, inversao de sinal de Im(Z) etc.) -- confira
    sempre visualmente antes de excluir, e prefira investigar a causa da
    instabilidade (tempo de medida longo demais, deriva da celula) a so'
    cortar os pontos.
    """
    header, rows = read_bt(path)
    freq = col(header, rows, "freq/Hz")
    Re = col(header, rows, "Re(Z)/Ohm")
    negIm = col(header, rows, "-Im(Z)/Ohm")
    Z = Re - 1j * negIm
    # ordena por frequencia decrescente -> crescente para os calculos
    order = np.argsort(freq)[::-1]
    freq, Z = freq[order], Z[order]
    if exclude_last_n > 0:
        freq, Z = freq[:-exclude_last_n], Z[:-exclude_last_n]
    return freq, Z


# =====================================================================
# Modelo de circuito: Rs + (Rct // CPE) + Warburg de comprimento finito
# =====================================================================
def z_model(params, freq):
    Rs, Rct, Q, n, Rw, tau = params
    w = 2 * np.pi * freq
    z_cpe = 1.0 / (Q * (1j * w) ** n)
    z_par = 1.0 / (1.0 / Rct + 1.0 / z_cpe)
    arg = np.sqrt(1j * w * tau)
    z_w = Rw * np.tanh(arg) / arg
    return Rs + z_par + z_w


def residuals(params, freq, Z_exp):
    Z_calc = z_model(params, freq)
    diff = Z_calc - Z_exp
    weight = 1.0 / np.abs(Z_exp)  # peso de modulo
    return np.concatenate([diff.real * weight, diff.imag * weight])


def initial_guess(freq, Z):
    Re, negIm = Z.real, -Z.imag
    Rs0 = Re[np.argmax(freq)]  # Re(Z) na maior frequencia
    idx_peak = np.argmax(negIm)
    w_peak = 2 * np.pi * freq[idx_peak]
    Rct0 = max(2 * (Re[idx_peak] - Rs0), 1e-3)
    Q0 = 1.0 / (Rct0 * w_peak) if Rct0 * w_peak > 0 else 1e-4
    n0 = 0.8
    Rw0 = max(Re.max() - Rs0 - Rct0, 1e-3)
    tau0 = 1.0 / (2 * np.pi * freq.min())
    return np.array([Rs0, Rct0, Q0, n0, Rw0, tau0])


def fit_circuit(freq, Z):
    p0 = initial_guess(freq, Z)
    lb = [0, 0, 1e-9, 0.3, 0, 1e-6]
    ub = [np.inf, np.inf, 1.0, 1.0, np.inf, 1e6]
    result = least_squares(
        residuals, p0, args=(freq, Z), bounds=(lb, ub), max_nfev=20000
    )
    Rs, Rct, Q, n, Rw, tau = result.x
    Z_fit = z_model(result.x, freq)
    ss_res = np.sum(np.abs(Z - Z_fit) ** 2)
    ss_tot = np.sum(np.abs(Z - np.mean(Z)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(
        Rs=Rs,
        Rct=Rct,
        Q=Q,
        n=n,
        Rw=Rw,
        tau=tau,
        r2=r2,
        cost=result.cost,
        success=result.success,
    )


# =====================================================================
# Funcao principal
# =====================================================================
def build_figure5(eis_files, out_prefix="eis", exclude_last_n=None, dpi=300):
    exclude_last_n = exclude_last_n or {}
    states = [
        s
        for s in eis_files
        if eis_files[s] is not None and Path(eis_files[s]).is_file()
    ]
    skipped = [s for s in eis_files if s not in states]
    if skipped:
        print(f"Estados sem arquivo valido (ignorados): {skipped}")
    if not states:
        raise ValueError("Nenhum arquivo de EIS valido em EIS_FILES.")

    data, fits = {}, {}
    for s in states:
        n_excl = exclude_last_n.get(s, 0)
        freq, Z = load_eis(eis_files[s], exclude_last_n=n_excl)
        if len(freq) < 8:
            print(f"  [aviso] {s}: poucos pontos ({len(freq)}), fit pode ficar ruim.")
        data[s] = (freq, Z)
        fits[s] = fit_circuit(freq, Z)

    # ---------------- resultados em texto ----------------
    print("\n=== PARAMETROS DO CIRCUITO EQUIVALENTE ===")
    print(
        f'{"Estado":>10} | {"Rs (Ω)":>8} | {"Rct (Ω)":>9} | {"Q (F·s^(n-1))":>14} | '
        f'{"n":>5} | {"Rw (Ω)":>8} | {"tau (s)":>9} | {"R²":>6}'
    )
    for s in states:
        p = fits[s]
        print(
            f'{s:>10} | {p["Rs"]:>8.3f} | {p["Rct"]:>9.3f} | {p["Q"]:>14.3e} | '
            f'{p["n"]:>5.3f} | {p["Rw"]:>8.3f} | {p["tau"]:>9.3g} | {p["r2"]:>6.4f}'
        )

    comparison_text = None
    if len(states) >= 2:
        base = states[0]
        lines = [f'Variação vs. "{base}":']
        print(f'\nVariacao percentual relativa a "{base}":')
        for s in states[1:]:
            dRs = 100 * (fits[s]["Rs"] - fits[base]["Rs"]) / fits[base]["Rs"]
            dRct = 100 * (fits[s]["Rct"] - fits[base]["Rct"]) / fits[base]["Rct"]
            print(f"  {s}: Rs {dRs:+.1f}% | Rct {dRct:+.1f}%")
            lines.append(f"{s}: Rs {dRs:+.0f}%, Rct {dRct:+.0f}%")
        comparison_text = "\n".join(lines)

    # ---------------- CSV (Tabela 3) ----------------
    csv_path = f"{out_prefix}_parametros.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["estado", "Rs", "Rct", "Q", "n", "Rw", "tau", "r2"]
        )
        writer.writeheader()
        for s in states:
            p = fits[s]
            writer.writerow(
                dict(
                    estado=s,
                    Rs=p["Rs"],
                    Rct=p["Rct"],
                    Q=p["Q"],
                    n=p["n"],
                    Rw=p["Rw"],
                    tau=p["tau"],
                    r2=p["r2"],
                )
            )
    print(f"\nParametros salvos em {csv_path}")

    # ---------------- figuras ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(states)))

    for color, s in zip(colors, states):
        freq, Z = data[s]
        axes[0].plot(
            Z.real, -Z.imag, "o", color=color, markersize=4, label=f"{s} (dados)"
        )
        f_dense = np.logspace(np.log10(freq.min()), np.log10(freq.max()), 300)
        Z_fit_dense = z_model(
            [
                fits[s]["Rs"],
                fits[s]["Rct"],
                fits[s]["Q"],
                fits[s]["n"],
                fits[s]["Rw"],
                fits[s]["tau"],
            ],
            f_dense,
        )
        axes[0].plot(
            Z_fit_dense.real,
            -Z_fit_dense.imag,
            "--",
            color=color,
            linewidth=1.3,
            label=f'{s} (ajuste, R²={fits[s]["r2"]:.3f})',
        )

    axes[0].set_xlabel("Re(Z) (Ω)")
    axes[0].set_ylabel("-Im(Z) (Ω)")
    axes[0].set_title("1- Nyquist completo")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    # zoom na regiao de alta frequencia (semicirculo Rs/Rct)
    zoom_max = max(fits[s]["Rs"] + fits[s]["Rct"] for s in states) * 1.6
    for color, s in zip(colors, states):
        freq, Z = data[s]
        m = Z.real <= zoom_max
        axes[1].plot(
            Z.real[m], -Z.imag[m], "o", color=color, markersize=4, label=f"{s} (dados)"
        )
        f_dense = np.logspace(np.log10(freq.min()), np.log10(freq.max()), 300)
        Z_fit_dense = z_model(
            [
                fits[s]["Rs"],
                fits[s]["Rct"],
                fits[s]["Q"],
                fits[s]["n"],
                fits[s]["Rw"],
                fits[s]["tau"],
            ],
            f_dense,
        )
        m2 = Z_fit_dense.real <= zoom_max
        axes[1].plot(
            Z_fit_dense.real[m2],
            -Z_fit_dense.imag[m2],
            "--",
            color=color,
            linewidth=1.3,
        )

    axes[1].set_xlabel("Re(Z) (Ω)")
    axes[1].set_ylabel("-Im(Z) (Ω)")
    axes[1].set_title("2- Zoom em alta frequência (Rs/Rct)")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(alpha=0.3)
    if comparison_text:
        axes[1].text(
            0.98,
            0.04,
            comparison_text,
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.85,
            ),
        )

    fig.tight_layout()
    out_png = f"{out_prefix}_fig5.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"\nFigura salva em {out_png}")

    return dict(data=data, fits=fits)


# =====================================================================
# CONFIGURACAO -- preencha aqui
# =====================================================================
_DATA_DIR = Path(__file__).resolve().parent

EIS_FILES = {
    "inicial": _DATA_DIR / "Nb2CTx Lote 3 + Nego  de Fumo/FulltestC2_01_PEIS_CA2.txt",
    "pos-CV": _DATA_DIR / "Nb2CTx Lote 3 + Nego  de Fumo/FulltestC2_08_PEIS_CA2.txt",
    "final": _DATA_DIR
    / "Nb2CTx Lote 3 + Nego  de Fumo/FulltestC2-2_CA2.txt",  # substituto, se o EIS final original falhou
}

OUT_PREFIX = "eis"

# Descarte de pontos instaveis de baixa frequencia, por estado (opcional).
# Confira SEMPRE visualmente o Nyquist antes de definir isso -- nao use
# como padrao, so' quando ver Re(Z) negativo, inversao de sinal de Im(Z)
# ou saltos abruptos no fim da varredura (tipico de instabilidade/deriva
# durante uma medida muito longa em baixa frequencia).
EXCLUDE_LAST_N = {
    # 'inicial': 3,
}

if __name__ == "__main__":
    build_figure5(EIS_FILES, OUT_PREFIX, EXCLUDE_LAST_N)
