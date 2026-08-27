"""GCD adapter around ``Scripts Adicionais/gcd_figure4.py``."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from ._legacy import load_script


def analyze_gcd(
    gcd_file: str | Path,
    output_dir: str | Path,
    *,
    mass_g: float,
    current_sequence_a_g: Sequence[float],
    dpi: int = 300,
) -> dict:
    """Run Figure 4 logic and aggregate cycle metrics without inventing N/D."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    module = load_script("gcd_figure4.py")
    prefix = output / "gcd"
    raw = module.build_figure4(
        gcd_file, mass_g, list(current_sequence_a_g), str(prefix), dpi=dpi
    )
    csv_path = prefix.with_name(prefix.name + "_ciclos.csv")
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        return float(np.mean(values)) if values else np.nan

    retention = np.nan
    by_block = {}
    for row in rows:
        block = row.get("block")
        current = row.get("j_a_g")
        if block not in (None, "") and current not in (None, ""):
            by_block.setdefault((int(float(block)), float(current)), []).append(
                float(row["Cs_F_g"])
            )
    block_groups = {}
    for (block, current), values in by_block.items():
        block_groups.setdefault(current, []).append((block, values))
    repeated = [sorted(groups) for groups in block_groups.values() if len(groups) >= 2]
    if repeated:
        first = float(np.mean(repeated[0][0][1]))
        last = float(np.mean(repeated[0][-1][1]))
        retention = 100.0 * last / first if first else np.nan

    metrics = {
        "specific_capacitance": mean("Cs_F_g"),
        "coulombic_efficiency": mean("coulombic_efficiency_pct"),
        "energy_density": mean("E_Wh_kg"),
        "power_density": mean("P_W_kg"),
        "capacitance_retention": retention,
        "ir_drop": mean("ir_drop"),
    }
    figures = [str(p) for p in output.glob("gcd_*.png")]
    return {"result": raw, "metrics": metrics, "figures": figures, "table": rows}
