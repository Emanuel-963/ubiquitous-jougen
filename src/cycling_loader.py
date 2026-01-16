"""Loader for cycling data files (.txt) in data/processed.

Loads files with columns: tempo, corrente, potencial, ciclo.
"""

import pandas as pd
from pathlib import Path


def load_cycling_files(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all .txt files from directory, assuming columns: tempo, corrente, potencial, ciclo."""
    files = list(directory.glob("*.txt"))
    data = {}
    for file in files:
        df = pd.read_csv(file, sep="\t")  # Assume tab-separated
        # Ensure columns exist
        required_cols = ["tempo", "corrente", "potencial", "ciclo"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"File {file} missing required columns: {required_cols}")
        df["ciclo"] = df["ciclo"].astype(int)
        data[file.stem] = df
    return data