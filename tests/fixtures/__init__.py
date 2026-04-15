"""Synthetic fixture generator for integration tests.

Creates temporary data files that the EIS, DRT and cycling pipelines can load,
without depending on real laboratory data.

Public helpers
--------------
``create_eis_fixture(directory, n_files, n_points)``
``create_cycling_fixture(directory, n_files, n_cycles)``

All files follow the exact same format (semicolon-separated, comma decimal)
that the real IonFlow loader expects.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# EIS fixture — format identical to data/raw/*.txt
# ═══════════════════════════════════════════════════════════════════════

def create_eis_fixture(
    directory: str | Path,
    n_files: int = 3,
    n_points: int = 40,
    seed: int = 42,
) -> list[str]:
    """Write synthetic EIS files that ``load_eis_file`` can parse.

    Each file has the header:
        Filename;Index;DataIndex;Time (s);Frequency (Hz);-Z'' (Ω);Z (Ω);-Phase (°);Z' (Ω)

    Returns the list of written file paths.
    """
    rng = np.random.default_rng(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for i in range(n_files):
        # Randles-like impedance: Rs + Rp/(1 + jωRpC)
        Rs = rng.uniform(0.5, 3.0)
        Rp = rng.uniform(10.0, 100.0)
        C = rng.uniform(1e-5, 1e-3)

        freq = np.logspace(-1, 5, n_points)
        omega = 2 * np.pi * freq
        Z = Rs + Rp / (1 + 1j * omega * Rp * C)
        # Add small noise
        noise = rng.normal(0, 0.01 * np.abs(Z)) + 1j * rng.normal(0, 0.01 * np.abs(Z))
        Z = Z + noise

        zreal = Z.real
        zimag_neg = -Z.imag  # file stores -Z''
        zmag = np.abs(Z)
        phase = -np.angle(Z, deg=True)  # -Phase (°)

        fname = f"{i+1} Syn H2SO4 Am{i+1}.txt"
        fpath = directory / fname
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("Filename;Index;DataIndex;Time (s);Frequency (Hz);-Z'' (Ω);Z (Ω);-Phase (°);Z' (Ω)\n")
            for j in range(n_points):
                t = 300 + j * 1.1
                line = (
                    f";{j+1};{j+1};"
                    f"{_eu(t)};{_eu(freq[j])};"
                    f"{_eu(zimag_neg[j])};{_eu(zmag[j])};"
                    f"{_eu(phase[j])};{_eu(zreal[j])}\n"
                )
                fh.write(line)
        paths.append(str(fpath))

    return paths


# ═══════════════════════════════════════════════════════════════════════
# Cycling fixture — format identical to data/processed/*.txt
# ═══════════════════════════════════════════════════════════════════════

def create_cycling_fixture(
    directory: str | Path,
    n_files: int = 2,
    n_cycles: int = 3,
    points_per_cycle: int = 50,
    seed: int = 42,
) -> list[str]:
    """Write synthetic galvanostatic cycling files.

    Header:
        Filename;Index;DataIndex;Time (s);Current applied (A);WE(1).Current (A);WE(1).Potential (V);
        Discharge capacity (Ah);Σ Capacity (Ah);Normalized WE.Current;Normalized Discharge capacity;
        Normalized Σ Capacity;Cycle

    Returns the list of written file paths.
    """
    rng = np.random.default_rng(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for fi in range(n_files):
        fname = f"GCD_syn_Am{fi+1}.txt"
        fpath = directory / fname
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write(
                "Filename;Index;DataIndex;Time (s);Current applied (A);"
                "WE(1).Current (A);WE(1).Potential (V);"
                "Discharge capacity (Ah);Σ Capacity (Ah);"
                "Normalized WE.Current;Normalized Discharge capacity;"
                "Normalized Σ Capacity;Cycle\n"
            )
            idx = 1
            for cyc in range(n_cycles):
                t0 = cyc * points_per_cycle * 0.01
                for j in range(points_per_cycle):
                    t = t0 + j * 0.01
                    current = -6.2e-5
                    we_current = current + rng.normal(0, 1e-7)
                    potential = 0.55 - 0.15 * (j / points_per_cycle) + rng.normal(0, 0.005)
                    dc = idx * 1e-10
                    sc = -idx * 2e-11
                    nwc = we_current / 1.24e-4
                    ndc = dc * 8e3
                    nsc = sc * 8e3
                    line = (
                        f";{idx};{idx};"
                        f"{_eu(t)};{_eu(current)};"
                        f"{_eu(we_current)};{_eu(potential)};"
                        f"{_eu(dc)};{_eu(sc)};"
                        f"{_eu(nwc)};{_eu(ndc)};"
                        f"{_eu(nsc)};{cyc}\n"
                    )
                    fh.write(line)
                    idx += 1
        paths.append(str(fpath))

    return paths


# ── Helper ───────────────────────────────────────────────────────────

def _eu(val: float) -> str:
    """Format a float using European comma-decimal notation."""
    return f"{val}".replace(".", ",")
