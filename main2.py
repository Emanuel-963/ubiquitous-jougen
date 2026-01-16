"""Main script for cycling data analysis.

Loads cycling .txt files from data/processed, calculates energy and power per 5-cycle groups,
displays table, and plots Time vs Potential with integral.
"""

import sys
from pathlib import Path
from src.cycling_loader import load_cycling_files
from src.cycling_calculator import process_all_files
from src.cycling_plotter import plot_time_potential_with_integral


def main():
    # Input scan rate
    try:
        scan_rate = float(input("Enter scan rate (A/g, e.g., 0.1, 1, 10): "))
    except ValueError:
        print("Invalid scan rate. Exiting.")
        sys.exit(1)

    # Directory
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print(f"Directory {data_dir} does not exist. Exiting.")
        sys.exit(1)

    # Load data
    data = load_cycling_files(data_dir)
    if not data:
        print("No .txt files found in data/processed. Exiting.")
        sys.exit(1)

    # Process
    results = process_all_files(data, scan_rate)

    # Display tables
    for filename, df in results.items():
        print(f"\nResults for {filename}:")
        print(df.to_string(index=False))

        # Plot for each file
        original_df = data[filename]
        plot_time_potential_with_integral(original_df, filename)


if __name__ == "__main__":
    main()