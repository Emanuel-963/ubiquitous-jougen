"""Gerador de espectros EIS sintéticos para bootstrapping do ML.

Cria arquivos .txt no mesmo formato do potenciostato (separador ';') para cada
classe de circuito registrada no CircuitRegistry.  Os dados são usados para
treinar o RandomForestClassifier em ``ml_circuit_selector.py`` quando ainda não
há amostras reais suficientes (mínimo = 30).

Uso
---
::

    python scripts/gen_synthetic_eis.py                 # 20 arquivos × 11 circuitos
    python scripts/gen_synthetic_eis.py --n 40          # 40 por classe
    python scripts/gen_synthetic_eis.py --out data/raw  # pasta de destino
    python scripts/gen_synthetic_eis.py --seed 0        # semente aleatória fixa
    python scripts/gen_synthetic_eis.py --noise 0.03    # 3 % de ruído (padrão)
    python scripts/gen_synthetic_eis.py --list-circuits # listar circuitos disponíveis
    python scripts/gen_synthetic_eis.py --circuits Randles-CPE-W Two-Arc-CPE

Os arquivos gerados recebem o prefixo ``SYN_`` para diferenciar de medições reais
e podem ser excluídos com ``--clean``.

Circuitos suportados (11)
--------------------------
 1. Randles-CPE-W    — Rs − (Rp ‖ CPE) − W
 2. Two-Arc-CPE      — Rs − (Rp1 ‖ CPE1) − (Rp2 ‖ CPE2)
 3. Inductive-CPE    — Rs − L − (Rp ‖ CPE)
 4. Coating-CPE      — Rs − (Rcoat ‖ CPEcoat) − (Rct ‖ CPEdl)
 5. Warburg-Finite   — Rs − (Rp ‖ CPE) − Wfinite  [transmissive / tanh]
 6. ZARC-ZARC-W      — Rs − ZARC₁ − ZARC₂ − W
 7. Simple-RC        — Rs − (Rp ‖ C)
 8. CPE-Simple       — Rs − CPE                    [EDLC / blocking electrode]
 9. Warburg-Short    — Rs − (Rp ‖ CPE) − Wo        [reflective / coth]
10. Gerischer        — Rs − (Rp ‖ CPE) − Z_Ger     [SOFC / mixed conductors]
11. Three-ZARC       — Rs − ZARC₁ − ZARC₂ − ZARC₃ [solid electrolytes]

Parâmetros físicos (realistas para supercapacitores/células eletroquímicas)
---------------------------------------------------------------------------
Os intervalos foram calibrados a partir da literatura (ver
tutoriais/08_referencias_bibliograficas.txt, refs EIS-1–5, FIT-1–6, CPE-1–3).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── Ensure project root is on sys.path ─────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.circuit_fitting import _cpe, _inductor, _warburg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Frequency axis (50 pontos, 100 kHz → 0.1 Hz) ──────────────────────────
_FREQS_HZ: np.ndarray = np.logspace(5, -1, 50)
_OMEGA: np.ndarray = 2 * np.pi * _FREQS_HZ

# ── Intervalos de parâmetros físicos por circuito ─────────────────────────
# Formato: {nome_circuito: [(low, high), ...]} — um par por parâmetro
_PARAM_RANGES: dict[str, list[tuple[float, float]]] = {
    # Rs(Ω)         Rp(Ω)         Q(F·s^n-1)       n             Sigma
    "Randles-CPE-W": [
        (0.05, 4.0),  # Rs
        (1.0, 150.0),  # Rp
        (2e-5, 5e-3),  # Q
        (0.72, 0.97),  # n
        (0.005, 3.0),  # Sigma
    ],
    # Rs             Rp1           Q1               n1            Rp2           Q2             n2
    "Two-Arc-CPE": [
        (0.05, 3.0),  # Rs
        (1.0, 60.0),  # Rp1
        (1e-5, 2e-3),  # Q1
        (0.75, 0.97),  # n1
        (5.0, 250.0),  # Rp2
        (1e-6, 5e-4),  # Q2
        (0.65, 0.92),  # n2
    ],
    # Rs             L(H)          Rp            Q               n
    "Inductive-CPE": [
        (0.05, 2.5),  # Rs
        (5e-8, 8e-5),  # L
        (1.0, 100.0),  # Rp
        (1e-5, 8e-3),  # Q
        (0.78, 0.99),  # n
    ],
    # Rs             Rcoat         Qcoat            ncoat         Rct           Qdl            ndl
    "Coating-CPE": [
        (0.05, 2.0),  # Rs
        (50.0, 8000.0),  # Rcoat
        (5e-10, 5e-7),  # Qcoat
        (0.82, 0.99),  # ncoat
        (10.0, 800.0),  # Rct
        (1e-6, 8e-4),  # Qdl
        (0.68, 0.92),  # ndl
    ],
    # Rs             Rp            Q                n             Rd            Td(s)
    "Warburg-Finite": [
        (0.05, 3.0),  # Rs
        (1.0, 100.0),  # Rp
        (2e-5, 5e-3),  # Q
        (0.72, 0.97),  # n
        (0.5, 80.0),  # Rd
        (0.05, 30.0),  # Td
    ],
    # Rs             R1            Q1               n1            R2            Q2             n2            Sigma
    "ZARC-ZARC-W": [
        (0.05, 3.0),  # Rs
        (1.0, 60.0),  # R1
        (1e-5, 2e-3),  # Q1
        (0.75, 0.97),  # n1
        (3.0, 120.0),  # R2
        (1e-6, 5e-4),  # Q2
        (0.65, 0.92),  # n2
        (0.003, 2.0),  # Sigma
    ],
    # Rs             Rp            C(F)
    "Simple-RC": [
        (0.05, 3.0),  # Rs
        (1.0, 120.0),  # Rp
        (5e-5, 1e-2),  # C (ideal capacitor)
    ],
    # Rs             Q               n
    "CPE-Simple": [
        (0.01, 5.0),  # Rs
        (1e-4, 2.0),  # Q  (cobre desde células pequenas até EDLC)
        (0.85, 1.0),  # n  (próximo de 1 = capacitor ideal)
    ],
    # Rs             Rp            Q                n             Rd            Td(s)
    "Warburg-Short": [
        (0.05, 3.0),  # Rs
        (1.0, 100.0),  # Rp
        (2e-5, 5e-3),  # Q
        (0.72, 0.97),  # n
        (0.5, 80.0),  # Rd
        (0.05, 30.0),  # Td
    ],
    # Rs             Rp            Q                n             Rg            Tg(s)
    "Gerischer": [
        (0.01, 2.0),  # Rs
        (0.1, 20.0),  # Rp
        (1e-4, 5e-3),  # Q
        (0.70, 0.95),  # n
        (0.5, 100.0),  # Rg
        (1e-4, 1.0),  # Tg
    ],
    # Rs             R1            Q1              n1            R2             Q2            n2            R3             Q3            n3
    "Three-ZARC": [
        (0.01, 1.0),  # Rs
        (1.0, 100.0),  # R1  — arco de alta freq (bulk)
        (1e-12, 1e-9),  # Q1
        (0.85, 1.0),  # n1
        (10.0, 5000.0),  # R2  — arco de freq média (contorno de grão)
        (1e-10, 1e-7),  # Q2
        (0.75, 0.95),  # n2
        (50.0, 20000.0),  # R3  — arco de baixa freq (interface eletrodo)
        (1e-8, 1e-5),  # Q3
        (0.65, 0.90),  # n3
    ],
}


# ── Model functions (mirrors circuit_registry, avoids re-import of closures) ─
def _model_randles_cpe_w(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, Rp, Q, n, sigma = p
    Zcpe = _cpe(omega, Q, n)
    Zw = _warburg(omega, sigma)
    Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
    return Rs + Zpar + Zw


def _model_two_arc_cpe(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, Rp1, Q1, n1, Rp2, Q2, n2 = p
    Zpar1 = 1.0 / (1.0 / Rp1 + 1.0 / _cpe(omega, Q1, n1))
    Zpar2 = 1.0 / (1.0 / Rp2 + 1.0 / _cpe(omega, Q2, n2))
    return Rs + Zpar1 + Zpar2


def _model_inductive_cpe(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, L, Rp, Q, n = p
    Zl = _inductor(omega, L)
    Zpar = 1.0 / (1.0 / Rp + 1.0 / _cpe(omega, Q, n))
    return Rs + Zl + Zpar


def _model_coating_cpe(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, Rcoat, Qcoat, ncoat, Rct, Qdl, ndl = p
    Zcoat = 1.0 / (1.0 / Rcoat + 1.0 / _cpe(omega, Qcoat, ncoat))
    Zct = 1.0 / (1.0 / Rct + 1.0 / _cpe(omega, Qdl, ndl))
    return Rs + Zcoat + Zct


def _model_warburg_finite(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, Rp, Q, n, Rd, Td = p
    Zcpe = _cpe(omega, Q, n)
    s = np.sqrt(1j * omega * Td)
    # Clamp |s| to avoid tanh overflow at very low frequencies / large Td
    s_abs = np.abs(s)
    s_safe = np.where(
        s_abs < 1e-30, 1e-30 + 0j, np.where(s_abs > 20, s / s_abs * 20, s)
    )
    Zw_finite = Rd * np.tanh(s_safe) / s_safe
    Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
    return Rs + Zpar + Zw_finite


def _model_zarc_zarc_w(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, R1, Q1, n1, R2, Q2, n2, sigma = p
    Zarc1 = 1.0 / (1.0 / R1 + 1.0 / _cpe(omega, Q1, n1))
    Zarc2 = 1.0 / (1.0 / R2 + 1.0 / _cpe(omega, Q2, n2))
    Zw = _warburg(omega, sigma)
    return Rs + Zarc1 + Zarc2 + Zw


def _model_simple_rc(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, Rp, C = p
    Zc = 1.0 / (1j * omega * C)
    Zpar = 1.0 / (1.0 / Rp + 1.0 / Zc)
    return Rs + Zpar


def _model_cpe_simple(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, Q, n = p
    return Rs + _cpe(omega, Q, n)


def _model_warburg_short(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, Rp, Q, n, Rd, Td = p
    Zcpe = _cpe(omega, Q, n)
    Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
    # Warburg open (reflective): Rd * coth(s) / s,  s = sqrt(j*omega*Td)
    s = np.sqrt(1j * omega * Td)
    s_abs = np.abs(s)
    safe_s = np.where(s_abs < 1e-10, (1e-10 + 0j) * np.ones_like(s), s)
    # Clamp |s| ≤ 20 before cosh/sinh to avoid overflow (np.where evaluates both
    # branches eagerly).  At |s| = 20, coth(20) ≈ 1.0 to 10 sig-figs.
    s_clamped = np.where(
        s_abs > 20, safe_s / np.where(s_abs > 0, s_abs, 1.0) * 20, safe_s
    )
    coth_s = np.cosh(s_clamped) / np.sinh(s_clamped)
    Zw_open = Rd * coth_s / safe_s
    return Rs + Zpar + Zw_open


def _model_gerischer(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, Rp, Q, n, Rg, Tg = p
    Zcpe = _cpe(omega, Q, n)
    Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
    # Gerischer element: Rg / sqrt(1 + j*omega*Tg)
    Zg = Rg / np.sqrt(1.0 + 1j * omega * Tg)
    return Rs + Zpar + Zg


def _model_three_zarc(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, R1, Q1, n1, R2, Q2, n2, R3, Q3, n3 = p
    Zarc1 = 1.0 / (1.0 / R1 + 1.0 / _cpe(omega, Q1, n1))
    Zarc2 = 1.0 / (1.0 / R2 + 1.0 / _cpe(omega, Q2, n2))
    Zarc3 = 1.0 / (1.0 / R3 + 1.0 / _cpe(omega, Q3, n3))
    return Rs + Zarc1 + Zarc2 + Zarc3


_MODELS: dict[str, Any] = {
    "Randles-CPE-W": _model_randles_cpe_w,
    "Two-Arc-CPE": _model_two_arc_cpe,
    "Inductive-CPE": _model_inductive_cpe,
    "Coating-CPE": _model_coating_cpe,
    "Warburg-Finite": _model_warburg_finite,
    "ZARC-ZARC-W": _model_zarc_zarc_w,
    "Simple-RC": _model_simple_rc,
    "CPE-Simple": _model_cpe_simple,
    "Warburg-Short": _model_warburg_short,
    "Gerischer": _model_gerischer,
    "Three-ZARC": _model_three_zarc,
}


# ── Core generator ─────────────────────────────────────────────────────────


def _random_params(circuit_name: str, rng: np.random.Generator) -> np.ndarray:
    """Amostra parâmetros log-uniformes dentro dos intervalos físicos."""
    ranges = _PARAM_RANGES[circuit_name]
    params = []
    for lo, hi in ranges:
        # Log-uniform para grandezas que variam em ordens de magnitude
        if lo > 0 and hi / lo > 10:
            val = np.exp(rng.uniform(np.log(lo), np.log(hi)))
        else:
            val = rng.uniform(lo, hi)
        params.append(val)
    return np.array(params)


def simulate_spectrum(
    circuit_name: str,
    rng: np.random.Generator,
    noise_level: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simula um espectro EIS com ruído Gaussiano.

    Returns
    -------
    freqs : ndarray (Hz)
    zreal : ndarray (Ω)
    zimag : ndarray (Ω)  — sinal positivo da parte imaginária (como -Z'')
    """
    model_fn = _MODELS[circuit_name]
    params = _random_params(circuit_name, rng)

    Z: np.ndarray = model_fn(params, _OMEGA)

    # Adicionar ruído proporcional à magnitude
    Zmag = np.abs(Z)
    noise_r = rng.normal(0, noise_level * Zmag)
    noise_i = rng.normal(0, noise_level * Zmag)

    zreal = Z.real + noise_r
    zimag_raw = Z.imag + noise_i  # geralmente negativo (capacitivo)

    return _FREQS_HZ, zreal, -zimag_raw  # -Z'' ≥ 0 na convenção do arquivo


def write_eis_file(
    path: Path,
    freqs: np.ndarray,
    zreal: np.ndarray,
    zimag_neg: np.ndarray,  # -Z'' (positivo)
) -> None:
    """Escreve arquivo .txt no formato do potenciostato (sep=';')."""
    header = "Filename;Index;DataIndex;Frequency (Hz);Time (s);-Z'' (Ω);Z (Ω);-Phase (°);Z' (Ω)"

    lines = [header]
    for i, (f, zr, zi) in enumerate(zip(freqs, zreal, zimag_neg), start=1):
        # Z (módulo) e Phase
        z_complex = zr - 1j * zi  # reconstrução: Z'' negativo → capacitivo
        z_mod = abs(z_complex)
        phase = -np.degrees(np.angle(z_complex))  # -Phase (°)
        t_fake = 0.0 + i * 2.5  # tempo fictício crescente
        fname_col = str(path) if i == 1 else ""
        lines.append(
            f"{fname_col};{i};{i};{f:.10g};{t_fake:.6f};"
            f"{zi:.12g};{z_mod:.12g};{phase:.10g};{zr:.12g}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def generate(
    circuits: list[str],
    n_per_class: int,
    out_dir: Path,
    noise_level: float,
    seed: int | None,
) -> list[Path]:
    """Gera os arquivos e retorna lista de caminhos criados."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for circuit_name in circuits:
        if circuit_name not in _MODELS:
            log.warning("Circuito desconhecido '%s' — ignorado.", circuit_name)
            continue
        for k in range(1, n_per_class + 1):
            safe_name = circuit_name.replace(" ", "_")
            filename = f"SYN_{safe_name}_{k:03d}.txt"
            dest = out_dir / filename

            freqs, zreal, zimag = simulate_spectrum(circuit_name, rng, noise_level)
            write_eis_file(dest, freqs, zreal, zimag)
            created.append(dest)

        log.info("  [%s] %d arquivos gerados", circuit_name, n_per_class)

    return created


def clean_synthetic(out_dir: Path) -> int:
    """Remove todos os arquivos com prefixo SYN_ da pasta destino."""
    removed = 0
    for f in out_dir.glob("SYN_*.txt"):
        f.unlink()
        removed += 1
    return removed


# ── CLI ────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera espectros EIS sintéticos para bootstrapping do ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        metavar="N",
        help="Arquivos por classe de circuito (padrão: 20)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        metavar="DIR",
        help="Diretório de saída (padrão: data/raw)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        metavar="FRAC",
        help="Nível de ruído Gaussiano proporcional a |Z| (padrão: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Semente aleatória para reprodutibilidade (padrão: aleatória)",
    )
    parser.add_argument(
        "--circuits",
        nargs="+",
        metavar="NOME",
        default=None,
        help="Gerar apenas para estas classes (padrão: todas as 7)",
    )
    parser.add_argument(
        "--list-circuits",
        action="store_true",
        help="Listar circuitos disponíveis e sair",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remover arquivos SYN_* existentes em --out antes de gerar",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.list_circuits:
        print("Circuitos disponíveis:")
        for name in _MODELS:
            nparams = len(_PARAM_RANGES[name])
            print(f"  {name:<22} ({nparams} parâmetros)")
        return

    circuits = args.circuits if args.circuits else list(_MODELS.keys())
    out_dir: Path = args.out.resolve()

    if args.clean:
        n_removed = clean_synthetic(out_dir)
        log.info("Removidos %d arquivos SYN_* de %s", n_removed, out_dir)

    log.info(
        "Gerando %d arquivo(s) × %d circuito(s) → %s",
        args.n,
        len(circuits),
        out_dir,
    )
    created = generate(
        circuits=circuits,
        n_per_class=args.n,
        out_dir=out_dir,
        noise_level=args.noise,
        seed=args.seed,
    )
    total = len(created)
    log.info("Total criado: %d arquivos SYN_*.txt", total)
    log.info(
        "Para treinar o ML: rode o Pipeline EIS na GUI e o modelo detectará "
        "automaticamente ≥ 30 amostras no FeatureStore."
    )


if __name__ == "__main__":
    main()
