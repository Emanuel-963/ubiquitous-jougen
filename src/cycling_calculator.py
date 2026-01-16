"""Calculator for energy and power from cycling data.

Calculates energy (Wh/kg) and power (W/kg) for groups of 5 cycles.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def calculate_mass(current: float, scan_rate: float) -> float:
    """Calculate mass in grams: mass = current / scan_rate (A/g)."""
    return current / scan_rate


def calculate_energy_power(df: pd.DataFrame, scan_rate: float) -> pd.DataFrame:
    """Calculate energy and power for groups of 5 cycles.

    Groups: cycles 1-5, 6-10, etc.
    For each group: energy = integral(V * I * dt) / mass, power = mean(V * I) / mass.
    """
    # Assume dt is constant, or calculate from tempo
    df = df.sort_values("tempo").reset_index(drop=True)
    dt = np.diff(df["tempo"])
    dt = np.append(dt, dt[-1])  # Approximate last dt

    # Power = V * I
    df["power"] = df["potencial"] * df["corrente"]

    # Energy cumulative
    df["energy_cum"] = np.cumsum(df["power"] * dt)

    # Group by cycle groups
    df["cycle_group"] = ((df["ciclo"] - 1) // 5) + 1

    results = []
    for group, gdf in df.groupby("cycle_group"):
        # Mass: use mean current for group
        mean_current = gdf["corrente"].mean()
        mass = calculate_mass(mean_current, scan_rate)

        # Energy: total energy / mass (Wh/kg)
        total_energy = gdf["energy_cum"].iloc[-1] - gdf["energy_cum"].iloc[0] if len(gdf) > 1 else 0
        energy_wh_kg = total_energy / mass if mass > 0 else 0

        # Power: mean power / mass (W/kg)
        mean_power = gdf["power"].mean()
        power_w_kg = mean_power / mass if mass > 0 else 0

        results.append({
            "cycle_group": group,
            "energy_wh_kg": energy_wh_kg,
            "power_w_kg": power_w_kg,
            "mass_g": mass
        })

    return pd.DataFrame(results)


def process_all_files(data: Dict[str, pd.DataFrame], scan_rate: float) -> Dict[str, pd.DataFrame]:
    """Process all files and return results."""
    results = {}
    for name, df in data.items():
        results[name] = calculate_energy_power(df, scan_rate)
    return results