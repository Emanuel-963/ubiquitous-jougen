"""
Figura 6 (DRT - Distribution of Relaxation Times) a partir de dados brutos
de PEIS do BT-Lab (EC-Lab).

Metodo: regularizacao de Tikhonov (ridge) com matriz de suavizacao de
2a derivada, resolvida como minimos quadrados nao-negativos (NNLS) no
sistema aumentado -- e a formulacao padrao usada por ferramentas como o
DRTtools (Ciucci & Chen, 2015; Wan et al., 2015).

Modelo:
    Z(w) - Rs = integral( gamma(ln tau) * 1/(1+j*w*tau) d(ln tau) )

Discretizando ln(tau) numa grade e aproximando a integral por soma
trapezoidal, isso vira um sistema linear A*gamma = b (b = [Re(Z)-Rs ;
-Im(Z)] empilhados), com gamma >= 0. Para evitar picos espurios (o
problema e' mal-posto), adiciona-se regularizacao de 2a derivada:

    min ||A*gamma - b||^2 + lambda^2 * ||L*gamma||^2 ,  gamma >= 0

O lambda e' escolhido automaticamente pelo "corner" da curva-L (o ponto
de maior curvatura entre o erro de ajuste e a norma de regularizacao).

Segue a mesma convencao dos outros scripts: parser read_bt/col,
matplotlib Agg, titulos numerados, config no final do arquivo.

------------------------------------------------------------------
COMO USAR
------------------------------------------------------------------
1) Preencha EIS_FILES com os mesmos arquivos usados na Figura 5 (inicial,
   pos-CV, final).
2) Rode: python3 drt_figure6.py
   Gera <OUT_PREFIX>_fig6.png e <OUT_PREFIX>_picos.csv (posicao/amplitude
   dos picos por estado).

ATENCAO -- leia antes de reportar "areas" de pico:
   A amplitude gamma(tau) no pico e' confiavel. A AREA de um pico so' e'
   confiavel quando ele esta' bem separado dos vizinhos (minimo local
   claro dos dois lados). Quando picos se sobrepoem (ombro em vez de vale),
   o script marca a area como "nao confiavel" em vez de reportar um
   numero que pareceria preciso sem ser.
------------------------------------------------------------------
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# Leitura de arquivos BT-Lab (identica aos outros scripts)
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
    header, rows = read_bt(path)
    freq = col(header, rows, "freq/Hz")
    Re = col(header, rows, "Re(Z)/Ohm")
    negIm = col(header, rows, "-Im(Z)/Ohm")
    Z = Re - 1j * negIm
    order = np.argsort(freq)[::-1]
    freq, Z = freq[order], Z[order]
    if exclude_last_n > 0:
        freq, Z = freq[:-exclude_last_n], Z[:-exclude_last_n]
    return freq, Z


# =====================================================================
# Nucleo do DRT
# =====================================================================
def build_tau_grid(freq, points_per_decade=10, pad_decades=0.5):
    tau_min = 1.0 / (2 * np.pi * freq.max()) / (10**pad_decades)
    tau_max = 1.0 / (2 * np.pi * freq.min()) * (10**pad_decades)
    n_dec = np.log10(tau_max / tau_min)
    n_points = max(int(n_dec * points_per_decade), 20)
    return np.logspace(np.log10(tau_min), np.log10(tau_max), n_points)


def second_derivative_matrix(n):
    """Matriz (n-2) x n de 2a derivada discreta, para regularizacao de suavidade."""
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def build_kernel(freq, tau):
    """
    Devolve A_re, A_im (n_freq x n_tau), com peso trapezoidal em ln(tau)
    ja' incorporado, para: Re(Z)-Rs = A_re @ gamma ; -Im(Z) = A_im @ gamma
    """
    w = 2 * np.pi * freq
    ln_tau = np.log(tau)
    d_ln_tau = np.gradient(ln_tau)  # peso trapezoidal aproximado
    wt = w[:, None] * tau[None, :]
    A_re = (1.0 / (1.0 + wt**2)) * d_ln_tau[None, :]
    A_im = (wt / (1.0 + wt**2)) * d_ln_tau[None, :]
    return A_re, A_im


def solve_drt(A, b, L, lam):
    n = A.shape[1]
    A_aug = np.vstack([A, lam * L])
    b_aug = np.concatenate([b, np.zeros(L.shape[0])])
    gamma, _ = nnls(A_aug, b_aug, maxiter=5000)
    return gamma


def l_curve_lambda(A, b, L, lambdas):
    etas, xis = [], []
    for lam in lambdas:
        gamma = solve_drt(A, b, L, lam)
        etas.append(np.linalg.norm(A @ gamma - b))
        xis.append(np.linalg.norm(L @ gamma))
    etas, xis = np.array(etas), np.array(xis)
    # corner da curva-L em escala log-log via curvatura discreta
    log_eta, log_xi = np.log(etas + 1e-30), np.log(xis + 1e-30)
    d1_eta = np.gradient(log_eta)
    d1_xi = np.gradient(log_xi)
    d2_eta = np.gradient(d1_eta)
    d2_xi = np.gradient(d1_xi)
    curvature = (
        np.abs(d1_eta * d2_xi - d1_xi * d2_eta)
        / (d1_eta**2 + d1_xi**2 + 1e-30) ** 1.5
    )
    # ignora as pontas do grid (curvatura numerica instavel ali)
    curvature[:2] = -np.inf
    curvature[-2:] = -np.inf
    idx = np.argmax(curvature)
    return lambdas[idx], etas, xis, idx


def compute_drt(freq, Z, points_per_decade=10, n_lambda=25):
    Rs = Z.real[np.argmax(freq)]  # Re(Z) na maior frequencia
    tau = build_tau_grid(freq, points_per_decade)
    A_re, A_im = build_kernel(freq, tau)
    A = np.vstack([A_re, A_im])
    b = np.concatenate([Z.real - Rs, -Z.imag])
    L = second_derivative_matrix(len(tau))

    lambdas = np.logspace(-6, 1, n_lambda)
    lam_opt, etas, xis, idx = l_curve_lambda(A, b, L, lambdas)
    gamma = solve_drt(A, b, L, lam_opt)

    return dict(
        tau=tau,
        gamma=gamma,
        Rs=Rs,
        lam=lam_opt,
        lambdas=lambdas,
        etas=etas,
        xis=xis,
        l_curve_idx=idx,
    )


def find_drt_peaks(tau, gamma, prominence_frac=0.03):
    if gamma.max() <= 0:
        return []
    prom = prominence_frac * gamma.max()
    peaks, props = find_peaks(gamma, prominence=prom)
    out = []
    for k, p in enumerate(peaks):
        # tenta achar minimos locais dos dois lados para julgar separacao/area
        left = p
        while left > 0 and gamma[left - 1] < gamma[left]:
            left -= 1
        right = p
        while right < len(gamma) - 1 and gamma[right + 1] < gamma[right]:
            right += 1
        well_separated = (left > 0) and (right < len(gamma) - 1)
        if well_separated:
            _trapz = getattr(np, "trapezoid", None) or np.trapz
            area = _trapz(gamma[left : right + 1], np.log(tau[left : right + 1]))
        else:
            area = None
        out.append(
            dict(
                tau_peak=tau[p],
                freq_peak=1.0 / (2 * np.pi * tau[p]),
                amplitude=gamma[p],
                area=area,
                well_separated=well_separated,
            )
        )
    return out


def slow_drt_area(peaks, tau_threshold_s=0.1):
    """
    Soma as areas CONFIAVEIS dos picos DRT lentos, definidos por tau >= tau_threshold_s.

    Retorna:
        area_sum: soma das areas confiaveis [Ohm]
        n_slow: numero total de picos lentos detectados
        n_reliable: numero de picos lentos com area confiavel
        n_unreliable: numero de picos lentos sobrepostos/sem area confiavel
    """
    slow_peaks = [pk for pk in peaks if pk["tau_peak"] >= tau_threshold_s]
    reliable = [
        pk for pk in slow_peaks if pk["well_separated"] and pk["area"] is not None
    ]
    unreliable = [
        pk for pk in slow_peaks if not pk["well_separated"] or pk["area"] is None
    ]

    area_sum = float(sum(pk["area"] for pk in reliable))
    return dict(
        area_sum=area_sum,
        n_slow=len(slow_peaks),
        n_reliable=len(reliable),
        n_unreliable=len(unreliable),
    )


# =====================================================================
# Funcao principal
# =====================================================================
def build_figure6(
    eis_files, out_prefix="drt", exclude_last_n=None, slow_tau_threshold_s=0.1, dpi=300
):
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

    results, peaks_all = {}, {}
    for s in states:
        n_excl = exclude_last_n.get(s, 0)
        freq, Z = load_eis(eis_files[s], exclude_last_n=n_excl)
        if freq.size == 0:
            print(
                f"Estado '{s}' ignorado: {eis_files[s]} nao contem pontos EIS validos."
            )
            continue
        res = compute_drt(freq, Z)
        peaks = find_drt_peaks(res["tau"], res["gamma"])
        results[s] = res
        peaks_all[s] = peaks
        print(f'\n=== {s}: lambda otimo = {res["lam"]:.3e}, Rs = {res["Rs"]:.3f} Ω ===')
        if not peaks:
            print("  nenhum pico detectado (prominencia abaixo do limiar).")
        for i, pk in enumerate(peaks):
            area_txt = (
                f'{pk["area"]:.3g} Ω'
                if pk["well_separated"]
                else "não confiável (picos sobrepostos)"
            )
            print(
                f'  pico {i + 1}: tau = {pk["tau_peak"]:.3e} s '
                f'(f = {pk["freq_peak"]:.3g} Hz), amplitude = {pk["amplitude"]:.3g} Ω, '
                f"area = {area_txt}"
            )

        slow = slow_drt_area(peaks, tau_threshold_s=slow_tau_threshold_s)
        if slow["n_slow"] == 0:
            print(
                f"  >> AREA DRT DE PROCESSOS LENTOS (tau >= {slow_tau_threshold_s:g} s): "
                f"0.000 Ω — nenhum pico lento detectado"
            )
        elif slow["n_reliable"] == 0:
            print(
                f"  >> AREA DRT DE PROCESSOS LENTOS (tau >= {slow_tau_threshold_s:g} s): "
                f'N/D — {slow["n_slow"]} pico(s) lento(s), mas nenhuma area e confiavel'
            )
        else:
            warning = ""
            if slow["n_unreliable"] > 0:
                warning = (
                    f' | ATENCAO: {slow["n_unreliable"]} pico(s) lento(s) '
                    f"sobreposto(s) nao incluido(s)"
                )
            print(
                f"  >> AREA DRT DE PROCESSOS LENTOS (tau >= {slow_tau_threshold_s:g} s): "
                f'{slow["area_sum"]:.6g} Ω '
                f'({slow["n_reliable"]}/{slow["n_slow"]} picos com area confiavel){warning}'
            )

    states = list(results)
    if not states:
        raise ValueError(
            "Nenhum arquivo de EIS contem pontos validos para calcular o DRT."
        )

    # ---------------- CSV ----------------
    csv_path = f"{out_prefix}_picos.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "estado",
                "pico",
                "tau_s",
                "freq_hz",
                "amplitude",
                "area",
                "bem_separado",
            ],
        )
        writer.writeheader()
        for s in states:
            for i, pk in enumerate(peaks_all[s]):
                writer.writerow(
                    dict(
                        estado=s,
                        pico=i + 1,
                        tau_s=pk["tau_peak"],
                        freq_hz=pk["freq_peak"],
                        amplitude=pk["amplitude"],
                        area=pk["area"] if pk["well_separated"] else "",
                        bem_separado=pk["well_separated"],
                    )
                )
    print(f"\nPicos salvos em {csv_path}")

    # Valor recomendado para a planilha: usa o estado 'final' quando disponivel;
    # caso contrario, usa o ultimo estado processado.
    score_state = "final" if "final" in states else states[-1]
    slow_final = slow_drt_area(
        peaks_all[score_state], tau_threshold_s=slow_tau_threshold_s
    )
    print("\n" + "=" * 72)
    print(f"INDICADOR PARA PLANILHA — estado: {score_state}")
    if slow_final["n_slow"] == 0:
        print(
            f"Area DRT de processos lentos (tau >= {slow_tau_threshold_s:g} s) = 0.000 Ω"
        )
        print("Observacao: nenhum pico lento foi detectado acima do limiar.")
    elif slow_final["n_reliable"] == 0:
        print(f"Area DRT de processos lentos (tau >= {slow_tau_threshold_s:g} s) = N/D")
        print("Observacao: ha pico(s) lento(s), mas as areas nao sao confiaveis.")
    else:
        print(
            f"Area DRT de processos lentos (tau >= {slow_tau_threshold_s:g} s) = "
            f'{slow_final["area_sum"]:.6g} Ω'
        )
        print(
            f'Picos lentos considerados = {slow_final["n_reliable"]} '
            f'de {slow_final["n_slow"]} detectados'
        )
        if slow_final["n_unreliable"] > 0:
            print(
                f'ATENCAO: {slow_final["n_unreliable"]} pico(s) lento(s) sobreposto(s) '
                "nao entrou/entraram na soma."
            )
    print("=" * 72)

    # ---------------- figura ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(states)))

    for color, s in zip(colors, states):
        res = results[s]
        axes[0].plot(res["tau"], res["gamma"], color=color, linewidth=1.6, label=s)
        for pk in peaks_all[s]:
            marker = "o" if pk["well_separated"] else "x"
            axes[0].plot(
                pk["tau_peak"],
                pk["amplitude"],
                marker,
                color=color,
                markersize=7,
                markeredgewidth=1.5,
                markerfacecolor="none" if pk["well_separated"] else color,
            )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("τ (s)")
    axes[0].set_ylabel("γ(τ) (Ω)")
    axes[0].set_title("1- Distribuição dos tempos de relaxação")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].text(
        0.02,
        0.97,
        "o = pico bem separado (área confiável)\nx = ombro/sobreposto (área não reportada)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.85
        ),
    )

    # curva-L do ULTIMO estado processado, como referencia de que o lambda
    # foi escolhido de forma nao arbitraria (nao e' o foco principal da figura,
    # mas ajuda a justificar a escolha na secao de metodos)
    s_ref = states[-1]
    res_ref = results[s_ref]
    axes[1].loglog(res_ref["etas"], res_ref["xis"], "o-", color="gray", markersize=4)
    axes[1].loglog(
        res_ref["etas"][res_ref["l_curve_idx"]],
        res_ref["xis"][res_ref["l_curve_idx"]],
        "o",
        color="red",
        markersize=10,
        label=f'λ ótimo ({s_ref}) = {res_ref["lam"]:.2e}',
    )
    axes[1].set_xlabel("||A·γ - b||  (erro de ajuste)")
    axes[1].set_ylabel("||L·γ||  (norma de regularização)")
    axes[1].set_title(f"2- Curva-L para escolha de λ ({s_ref})")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_png = f"{out_prefix}_fig6.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"\nFigura salva em {out_png}")

    return dict(results=results, peaks=peaks_all)


# =====================================================================
# CONFIGURACAO -- preencha aqui (mesmos arquivos da Figura 5)
# =====================================================================
_DATA_DIR = Path(__file__).resolve().parent

EIS_FILES = {
    "inicial": _DATA_DIR / "--Nb2L-LiCl Wis Tripli/Fulltestcell3_01_PEIS_CA4.txt",
    "pos-CV": _DATA_DIR / "--Nb2L-LiCl Wis Tripli/Fulltestcell3_08_PEIS_CA4.txt",
    "final": _DATA_DIR / "--Nb2L-LiCl Wis Tripli/Fulltestcell3_10_PEIS_CA4.txt",
}

# mesmo mecanismo de corte de pontos instaveis de baixa frequencia usado
# na Figura 5 -- confira visualmente antes de usar
EXCLUDE_LAST_N = {
    # 'inicial': 3,
}

# Limiar adotado para o indicador da planilha:
# processos lentos = picos DRT com tau >= 0,1 s.
SLOW_TAU_THRESHOLD_S = 0.1

OUT_PREFIX = "drt"

if __name__ == "__main__":
    build_figure6(
        EIS_FILES,
        OUT_PREFIX,
        EXCLUDE_LAST_N,
        slow_tau_threshold_s=SLOW_TAU_THRESHOLD_S,
    )
