"""CV/Dunn adapter around ``Scripts Adicionais/dunn_drt_analysis.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ._legacy import load_script


def analyze_cv(
    cv_files: Mapping[float, str | Path],
    output_dir: str | Path,
    *,
    electrodes: dict | None = None,
    cell_area_cm2: float = 1.0,
    dpi: int = 300,
) -> dict:
    """Run the validated Dunn analysis and return ranking-ready metrics."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not cv_files:
        return {"figures": [], "metrics": {}}
    module = load_script("dunn_drt_analysis.py")
    electrodes = electrodes or {"cell": {"mass_g": 1.0, "potential_fraction": 1.0}}
    prefix = output / "cv_dunn"
    result = module.dunn_analysis(
        dict(cv_files),
        electrodes,
        float(cell_area_cm2),
        out_prefix=str(prefix),
        dpi=dpi,
    )
    figures = [str(p) for p in output.glob("cv_dunn_*.png")]
    rates = sorted(result.get("cell_Cs", {}))
    metrics = {
        "specific_capacitance": float(np.nanmean([result["cell_Cs"][v] for v in rates]))
        if rates
        else np.nan,
        "capacitive_contribution": float(
            np.nanmean(list(result.get("frac_cap", {}).values()))
        )
        if result.get("frac_cap")
        else np.nan,
        "b_value": float(result.get("b_value", np.nan)),
        "b_r2": float(result.get("b_r2", np.nan)),
    }
    pd.DataFrame([metrics]).to_csv(output / "cv_metrics.csv", index=False)
    return {"result": result, "metrics": metrics, "figures": figures}
