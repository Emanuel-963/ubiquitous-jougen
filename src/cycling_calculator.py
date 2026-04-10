"""Calculator for energy and power from cycling data.

Calculates energy (Wh/kg) and power (W/kg) for each cycle.
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy.signal import find_peaks


def calculate_mass(current: float, scan_rate: float) -> float:
    """Calculate mass in grams: mass = |current| / scan_rate (A/g)."""
    return abs(current) / scan_rate


def identify_cycles(df: pd.DataFrame) -> pd.Series:
    """Identify cycle numbers from data.
    
    First tries to find a column containing 'ciclo' or 'cycle'.
    If not found or all values are the same, uses potential to detect cycles.
    """
    # Try to find cycle column
    cycle_col = None
    for col in df.columns:
        if 'ciclo' in col.lower() or 'cycle' in col.lower():
            cycle_col = col
            break
    
    if cycle_col is not None:
        unique_vals = df[cycle_col].unique()
        if len(unique_vals) > 1 and not all(v == 0 or pd.isna(v) for v in unique_vals):
            return df[cycle_col].astype(int)
    
    # Fallback: use potential to detect cycles
    potential = df['potencial'].values
    # Find local minima (inverted peaks)
    minima, _ = find_peaks(-potential)
    # Find local maxima
    maxima, _ = find_peaks(potential)
    
    # Combine extrema indices
    extrema = np.sort(np.concatenate([minima, maxima]))
    
    if len(extrema) < 2:
        # If not enough extrema, treat as one cycle
        return pd.Series([1] * len(df), index=df.index)
    
    # Assign cycles: each full cycle between consecutive minima
    cycle_nums = np.zeros(len(df), dtype=int)
    current_cycle = 1
    prev_min_idx = None
    
    for idx in extrema:
        if idx in minima:
            if prev_min_idx is not None:
                # End previous cycle at this min
                cycle_nums[prev_min_idx:idx] = current_cycle
                current_cycle += 1
            prev_min_idx = idx
    
    # Last segment
    if prev_min_idx is not None:
        cycle_nums[prev_min_idx:] = current_cycle
    
    return pd.Series(cycle_nums, index=df.index)


def calculate_energy_power(df: pd.DataFrame, scan_rate: float) -> pd.DataFrame:
    """Calculate energy and power for each cycle.

    For each cycle: energy = trapezoidal integral of |V * I| dt / mass (Wh/kg), power = energy / time / mass (W/kg).
    """
    # Identify cycles
    df = df.copy()
    df['ciclo'] = identify_cycles(df)
    
    results = []
    for cycle, gdf in df.groupby("ciclo"):
        if cycle == 0:
            continue  # Skip if cycle 0
        # Mass: use mean current for cycle
        mean_current = gdf["corrente"].mean()
        mass_g = calculate_mass(mean_current, scan_rate)
        mass_kg = mass_g / 1000  # Convert to kg
        
        # Time duration
        time_duration_s = gdf["tempo"].iloc[-1] - gdf["tempo"].iloc[0] if len(gdf) > 1 else 0
        
        # Energy: trapezoidal integral of |V * I| dt (Joules)
        total_energy_J = np.trapezoid(np.abs(gdf["potencial"] * gdf["corrente"]), gdf["tempo"])
        
        # Convert to Wh/kg
        energy_wh_kg = (total_energy_J / 3600) / mass_kg if mass_kg > 0 else 0
        
        # Power: average power W/kg
        average_power_W = total_energy_J / time_duration_s if time_duration_s > 0 else 0
        power_w_kg = average_power_W / mass_kg if mass_kg > 0 else 0

        results.append({
            "ciclo": cycle,
            "duracao_s": time_duration_s,
            "integral_vs": total_energy_J,  # Assuming this is what they mean by V/s
            "energia_wh_kg": energy_wh_kg,
            "potencia_w_kg": power_w_kg,
            "mass_g": mass_g
        })

    return pd.DataFrame(results)


def process_all_files(data: Dict[str, pd.DataFrame], scan_rate: float) -> Dict[str, pd.DataFrame]:
    """Process all files and return results."""
    import os
    excel_dir = "outputs/excel"
    os.makedirs(excel_dir, exist_ok=True)
    
    results = {}
    for name, df in data.items():
        df_result = calculate_energy_power(df, scan_rate)
        # Rename columns for output
        df_result = df_result.rename(columns={
            "ciclo": "Ciclos",
            "duracao_s": "Duração dos Ciclos (s)",
            "integral_vs": "Valor da Integral (V/s)",
            "energia_wh_kg": "Energia (Wh/kg)",
            "potencia_w_kg": "Potência (W/kg)"
        })
        # Select only desired columns
        df_result = df_result[["Ciclos", "Duração dos Ciclos (s)", "Valor da Integral (V/s)", "Energia (Wh/kg)", "Potência (W/kg)"]]
        results[name] = df_result
        
        # Save to Excel
        excel_path = os.path.join(excel_dir, f"{name}.xlsx")
        df_result.to_excel(excel_path, index=False)
        print(f"Exported to {excel_path}")

    return results
