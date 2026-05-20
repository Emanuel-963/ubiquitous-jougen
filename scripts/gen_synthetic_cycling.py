"""Gerador de dados de ciclagem galvanostática sintéticos para bootstrapping do ML.

Produz arquivos .txt no mesmo formato semicolon + decimal europeu (vírgula)
esperado pelo ``cycling_loader.py``.  Os arquivos recebem o prefixo ``SYN_CIC_``
para fácil identificação e limpeza.

Colunas geradas
---------------
``Time (s);WE(1).Current (A);WE(1).Potential (V);Cycle``

Cada arquivo simula um supercapacitor/bateria ciclado em corrente constante
(protocolo GCPL) com perfil de tensão realista e degradação gradual ao longo
dos ciclos.

Uso
---
::

    python scripts/gen_synthetic_cycling.py               # 10 arquivos em data/processed
    python scripts/gen_synthetic_cycling.py --n 20        # 20 arquivos
    python scripts/gen_synthetic_cycling.py --seed 42     # semente fixa
    python scripts/gen_synthetic_cycling.py --clean       # remover SYN_CIC_* existentes
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# ── Ensure project root is on sys.path ─────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SYN_PREFIX = "SYN_CIC_"
_DEFAULT_N_FILES = 10
_POINTS_PER_HALFCYCLE = 50  # 50 pts/half → 100 pts/ciclo completo


# ── Physics helpers ────────────────────────────────────────────────────────


def _simulate_gcpl(
    n_cycles: int,
    current_a: float,
    v_max: float,
    v_min: float,
    capacitance_f: float,
    esr_ohm: float,
    degradation_rate: float,
    noise_sigma_v: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simula ciclagem galvanostática em corrente constante (GCPL).

    Returns
    -------
    times : ndarray  — tempo em segundos
    currents : ndarray  — corrente em A (+charge / -discharge)
    potentials : ndarray  — potencial em V
    cycles : ndarray[int]  — número do ciclo (1-indexed)
    """
    # dt derivado da capacitância e corrente: ΔV = I/C · Δt
    v_range = v_max - v_min
    if v_range <= 0 or capacitance_f <= 0 or current_a <= 0:
        v_range = 1.0
        capacitance_f = 0.1
        current_a = 0.001
    dt = v_range * capacitance_f / current_a / _POINTS_PER_HALFCYCLE

    times_list = []
    currents_list = []
    potentials_list = []
    cycles_list = []

    t = 0.0
    for cyc in range(1, n_cycles + 1):
        # Degradação suave: capacidade reduz gradualmente
        cap = capacitance_f * max(0.5, 1.0 - degradation_rate * (cyc - 1))
        esr_drop = esr_ohm * current_a

        # ── Carga (corrente positiva) ──────────────────────────────────────
        for pt in range(_POINTS_PER_HALFCYCLE):
            frac = pt / (_POINTS_PER_HALFCYCLE - 1)
            v = (v_min + esr_drop) + frac * (v_range - 2 * esr_drop)
            v += rng.normal(0.0, noise_sigma_v)
            times_list.append(t)
            currents_list.append(current_a)
            potentials_list.append(float(v))
            cycles_list.append(cyc)
            t += dt

        # ── Descarga (corrente negativa) ───────────────────────────────────
        for pt in range(_POINTS_PER_HALFCYCLE):
            frac = pt / (_POINTS_PER_HALFCYCLE - 1)
            v = (v_max - esr_drop) - frac * (v_range - 2 * esr_drop)
            v += rng.normal(0.0, noise_sigma_v)
            times_list.append(t)
            currents_list.append(-current_a)
            potentials_list.append(float(v))
            cycles_list.append(cyc)
            t += dt

    return (
        np.array(times_list, dtype=float),
        np.array(currents_list, dtype=float),
        np.array(potentials_list, dtype=float),
        np.array(cycles_list, dtype=int),
    )


# ── File writer ────────────────────────────────────────────────────────────


def write_cycling_file(
    path: Path,
    times: np.ndarray,
    currents: np.ndarray,
    potentials: np.ndarray,
    cycles: np.ndarray,
) -> None:
    """Escreve arquivo .txt no formato do potenciostato (sep=';', decimal=',')."""
    header = "Time (s);WE(1).Current (A);WE(1).Potential (V);Cycle"
    lines = [header]
    for t, i, v, c in zip(times, currents, potentials, cycles):
        t_str = f"{t:.6f}".replace(".", ",")
        i_str = f"{i:.8e}".replace(".", ",")
        v_str = f"{v:.6f}".replace(".", ",")
        lines.append(f"{t_str};{i_str};{v_str};{c}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Core generator ─────────────────────────────────────────────────────────


def generate(
    n_files: int,
    out_dir: Path,
    seed: int | None = None,
) -> list[Path]:
    """Gera *n_files* arquivos de ciclagem sintéticos e retorna caminhos criados."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for k in range(1, n_files + 1):
        n_cycles = int(rng.integers(50, 500))
        current_ma = float(rng.uniform(1.0, 10.0))
        v_max = float(rng.uniform(0.8, 1.0))
        v_min = 0.0
        capacitance_f = float(rng.uniform(0.05, 2.0))
        esr_ohm = float(rng.uniform(0.01, 2.0))
        degradation_rate = float(rng.uniform(1e-5, 5e-4))
        noise_sigma_v = float(rng.uniform(0.001, 0.010))

        times, currents, potentials, cycles = _simulate_gcpl(
            n_cycles=n_cycles,
            current_a=current_ma * 1e-3,
            v_max=v_max,
            v_min=v_min,
            capacitance_f=capacitance_f,
            esr_ohm=esr_ohm,
            degradation_rate=degradation_rate,
            noise_sigma_v=noise_sigma_v,
            rng=rng,
        )

        filename = f"{SYN_PREFIX}{k:03d}.txt"
        dest = out_dir / filename
        write_cycling_file(dest, times, currents, potentials, cycles)
        created.append(dest)
        log.info("  [CIC] %s — %d ciclos, I=%.2f mA", filename, n_cycles, current_ma)

    return created


def clean_synthetic(out_dir: Path) -> int:
    """Remove todos os arquivos SYN_CIC_* da pasta destino."""
    removed = 0
    for f in out_dir.glob(f"{SYN_PREFIX}*.txt"):
        f.unlink()
        removed += 1
    return removed


# ── CLI ────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera dados de ciclagem sintéticos para bootstrapping do ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_DEFAULT_N_FILES,
        metavar="N",
        help=f"Número de arquivos a gerar (padrão: {_DEFAULT_N_FILES})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed"),
        metavar="DIR",
        help="Diretório de saída (padrão: data/processed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Semente aleatória para reprodutibilidade (padrão: aleatória)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remover arquivos SYN_CIC_* existentes em --out antes de gerar",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir: Path = args.out.resolve()

    if args.clean:
        n_removed = clean_synthetic(out_dir)
        log.info("Removidos %d arquivos SYN_CIC_* de %s", n_removed, out_dir)

    log.info("Gerando %d arquivo(s) de ciclagem sintéticos → %s", args.n, out_dir)
    created = generate(n_files=args.n, out_dir=out_dir, seed=args.seed)
    log.info("Total criado: %d arquivos SYN_CIC_*.txt", len(created))
    log.info(
        "Para treinar o preditor ML: rode o Pipeline Ciclagem na GUI."
    )


if __name__ == "__main__":
    main()
