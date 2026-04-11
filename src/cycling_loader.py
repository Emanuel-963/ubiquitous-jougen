"""Loader for cycling data files (.txt) in data/processed.

Loads files with columns: tempo, corrente, potencial, ciclo.
"""

import pandas as pd
from pathlib import Path


def load_cycling_files(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all .txt files from directory, handling real file format with semicolon separator and European decimals."""
    files = list(directory.glob("*.txt"))
    data = {}
    for file in files:
        # Read with semicolon separator and European decimal format
        df = pd.read_csv(file, sep=";", decimal=",")
        # Rename columns to expected names
        column_mapping = {
            "Time (s)": "tempo",
            "WE(1).Current (A)": "corrente",
            "WE(1).Potential (V)": "potencial",
            "Cycle": "ciclo"
        }
        df = df.rename(columns=column_mapping)
        # Ensure required columns exist
        required_cols = ["tempo", "corrente", "potencial", "ciclo"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"File {file} missing required columns after mapping: {required_cols}")
        # Select only required columns
        df = df[required_cols]
        df["ciclo"] = df["ciclo"].astype(int)
        data[file.stem] = df
    return data
