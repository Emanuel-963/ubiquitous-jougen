"""Main script for cycling data analysis.

Loads cycling .txt files from data/processed, calculates energy and power per 5-cycle groups,
displays table, and plots Time vs Potential with integral.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from src.cycling_loader import load_cycling_files
from src.cycling_calculator import process_all_files
from src.cycling_plotter import (
    plot_energy_power_vs_cycle,
    plot_time_potential_with_integral,
)


def run_ciclagem_pipeline(scan_rate: float, show_plots: bool = True) -> dict:
    # Directory
    data_dir = Path("data/processed")
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    # Load data
    data = load_cycling_files(data_dir)
    if not data:
        raise FileNotFoundError("No .txt files found in data/processed.")

    # Process
    results = process_all_files(data, scan_rate)

    # Prepare Excel output directory
    excel_dir = Path("outputs/excel")
    excel_dir.mkdir(parents=True, exist_ok=True)

    export_tables = {}
    plot_paths = []
    energy_power_paths = []

    # Display tables and export to Excel
    for filename, df in results.items():
        # Use processed ciclo column for grouping
        ciclo_col = "ciclo" if "ciclo" in df.columns else df.columns[0]
        ciclo_values = df[ciclo_col].values
        # For each cycle, get duration and integral from original_df
        duration_list = []
        integral_list = []
        for cycle in ciclo_values:
            gdf = data[filename][
                (data[filename]["tempo"] >= data[filename]["tempo"].min())
                & (data[filename]["tempo"] <= data[filename]["tempo"].max())
            ]
            gdf = (
                data[filename][data[filename][ciclo_col] == cycle]
                if ciclo_col in data[filename].columns
                else data[filename]
            )
            duration = (
                gdf["tempo"].iloc[-1] - gdf["tempo"].iloc[0]
                if len(gdf) > 1
                else 0
            )
            integral = np.trapezoid(
                np.abs(gdf["potencial"] * gdf["corrente"]), gdf["tempo"]
            )
            duration_list.append(duration)
            integral_list.append(integral)

        # Build export DataFrame
        export_df = pd.DataFrame(
            {
                "Ciclos (numero)": ciclo_values,
                "Duração dos Ciclos (s)": duration_list,
                "Valor da Integral (V/s)": integral_list,
                "Energia (Wh/kg)": df["Energia (Wh/kg)"].values,
                "Potência (W/kg)": df["Potência (W/kg)"].values,
            }
        )

        # Add header row for units
        units_row = ["numero", "s", "V/s", "Wh/kg", "W/kg"]
        export_df_units = pd.DataFrame([units_row], columns=export_df.columns)
        export_df_final = pd.concat([export_df_units, export_df], ignore_index=True)

        # Export to Excel
        excel_path = excel_dir / f"{filename}.xlsx"
        export_df_final.to_excel(excel_path, index=False)

        export_tables[filename] = export_df

        # Plot for each file
        plot_path = plot_time_potential_with_integral(
            data[filename], filename, show=show_plots
        )
        plot_paths.append((filename, plot_path))

        # Energy/Power vs Cycle chart
        ep_path = plot_energy_power_vs_cycle(
            export_df, filename, show=show_plots,
        )
        if ep_path:
            energy_power_paths.append((filename, ep_path))

    merged_table = None
    if export_tables:
        merged_table = pd.concat(
            [
                tbl.assign(Arquivo=filename)
                for filename, tbl in export_tables.items()
            ],
            ignore_index=True,
        )
        cols = ["Arquivo"] + [c for c in merged_table.columns if c != "Arquivo"]
        merged_table = merged_table[cols]

    return {
        "results": results,
        "export_tables": export_tables,
        "merged_table": merged_table,
        "plot_paths": plot_paths,
        "energy_power_paths": energy_power_paths,
    }


def main():
    # Input scan rate
    try:
        scan_rate = float(input("Enter scan rate (A/g, e.g., 0.1, 1, 10): "))
    except ValueError:
        print("Invalid scan rate. Exiting.")
        sys.exit(1)

    try:
        run_result = run_ciclagem_pipeline(scan_rate, show_plots=True)
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

    for filename, df in run_result["results"].items():
        print(f"\nResults for {filename}:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
