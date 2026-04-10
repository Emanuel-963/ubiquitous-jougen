"""Minimal script to demonstrate a reproducible figure generation flow.

Usage:
    python scripts/regenerate_figures.py

It creates a tiny synthetic dataset, runs PCA, and generates a PCA 2D figure under
`outputs/figures` (or a temporary folder if desired).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src import visualization as viz
from src.pca_analysis import run_pca


def main(out_dir: str = "outputs/figures") -> int:
    os.makedirs(out_dir, exist_ok=True)

    # Synthetic example features
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, size=20),
            "feature2": rng.normal(1, 0.5, size=20),
            "feature3": rng.normal(-1, 2, size=20),
        },
        index=[f"s{i}" for i in range(20)],
    )

    pca, scores = run_pca(df, n_components=2)
    labels = pd.Series(
        [
            "Interface eficiente" if i % 2 == 0 else "Genérica estável"
            for i in range(20)
        ],
        index=scores.index,
    )

    path = viz.pca_2d(scores, labels, out_dir=out_dir)
    print("Generated:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
