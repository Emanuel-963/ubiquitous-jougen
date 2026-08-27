"""EIS and DRT adapters around the supplied Figure 5 and Figure 6 scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ._legacy import load_script


def analyze_eis_drt(
    eis_files: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    exclude_last_n: Mapping[str, int] | None = None,
    slow_tau_threshold_s: float = 0.1,
    dpi: int = 300,
) -> dict:
    """Run validated EIS fitting and DRT, returning final-state metrics."""
    output = Path(output_dir)
    eis_dir, drt_dir = output / "eis", output / "drt"
    eis_dir.mkdir(parents=True, exist_ok=True)
    drt_dir.mkdir(parents=True, exist_ok=True)
    valid = {k: Path(v) for k, v in eis_files.items() if v and Path(v).is_file()}
    if not valid:
        return {"metrics": {}, "figures": [], "tables": {}}

    eis_module = load_script("eis_figure5 (1).py")
    drt_module = load_script("drt_figure6_area_lenta.py")
    eis_raw = eis_module.build_figure5(
        valid, str(eis_dir / "eis"), dict(exclude_last_n or {}), dpi=dpi
    )
    drt_raw = drt_module.build_figure6(
        valid,
        str(drt_dir / "drt"),
        dict(exclude_last_n or {}),
        slow_tau_threshold_s,
        dpi=dpi,
    )
    final_state = "final" if "final" in eis_raw["fits"] else list(eis_raw["fits"])[-1]
    fit = eis_raw["fits"][final_state]
    slow_state = "final" if "final" in drt_raw["peaks"] else list(drt_raw["peaks"])[-1]
    slow = drt_module.slow_drt_area(drt_raw["peaks"][slow_state], slow_tau_threshold_s)
    metrics = {
        "rs": float(fit.get("Rs", np.nan)),
        "rct_rp": float(fit.get("Rct", np.nan)),
        "area_drt_lenta": float(slow["area_sum"]) if slow["n_reliable"] else np.nan,
    }
    figures = [str(p) for p in output.rglob("*.png")]
    tables = {"eis_parameters": eis_raw["fits"], "drt_peaks": drt_raw["peaks"]}
    return {
        "eis": eis_raw,
        "drt": drt_raw,
        "metrics": metrics,
        "figures": figures,
        "tables": tables,
    }


def slow_drt_area(peaks: list[dict], tau_threshold_s: float = 0.1) -> dict:
    """Expose the reference script's slow-process area calculation."""
    module = load_script("drt_figure6_area_lenta.py")
    return module.slow_drt_area(peaks, tau_threshold_s)
