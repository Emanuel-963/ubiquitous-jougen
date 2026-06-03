"""Figure pack exporter — export all plots as SVG + PNG + CSV data.

VIZ-03: Provides a single function to export all generated figures in
publication-ready formats (300 dpi PNG, SVG vector, and .csv data files
with the underlying plot data).

Usage
-----
    from src.figure_pack import export_figure_pack

    export_figure_pack(
        figures={"nyquist": (fig, data_df), "bode": (fig2, data_df2)},
        output_dir="outputs/figure_pack",
        style="acs",
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def export_figure_pack(
    figures: Dict[str, Tuple[Any, Optional[pd.DataFrame]]],
    output_dir: str | Path,
    *,
    dpi: int = 300,
    style: Optional[str] = None,
    formats: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Export all figures as a publication-ready figure pack.

    Parameters
    ----------
    figures : dict
        Mapping of figure name → (matplotlib Figure, optional DataFrame).
        The DataFrame, if provided, contains the plot data and will be
        exported as a .csv file alongside the image.
    output_dir : str or Path
        Directory where the figure pack will be saved.
    dpi : int
        Resolution for raster images (PNG). Default: 300.
    style : str or None
        If provided, applies a journal style before saving (see
        ``src.journal_styles``).
    formats : list of str or None
        Image formats to export. Default: ``["png", "svg"]``.

    Returns
    -------
    dict
        Mapping of figure name → list of exported file paths.
    """
    import matplotlib.pyplot as plt

    if formats is None:
        formats = ["png", "svg"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally apply journal style
    if style:
        from src.journal_styles import apply_journal_style
        apply_journal_style(style)

    exported: Dict[str, List[str]] = {}

    for name, (fig, data_df) in figures.items():
        exported[name] = []
        safe_name = name.replace(" ", "_").replace("/", "_")

        for fmt in formats:
            fpath = output_dir / f"{safe_name}.{fmt}"
            try:
                fig.savefig(
                    str(fpath),
                    dpi=dpi if fmt == "png" else None,
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                exported[name].append(str(fpath))
                logger.debug("Exported %s → %s", name, fpath)
            except Exception as exc:
                logger.warning("Failed to export %s as %s: %s", name, fmt, exc)

        # Export underlying data as CSV
        if data_df is not None and not data_df.empty:
            csv_path = output_dir / f"{safe_name}_data.csv"
            try:
                data_df.to_csv(str(csv_path), index=False, float_format="%.6g")
                exported[name].append(str(csv_path))
            except Exception as exc:
                logger.warning("Failed to export data for %s: %s", name, exc)

        plt.close(fig)

    n_files = sum(len(v) for v in exported.values())
    logger.info(
        "Figure pack exported: %d figures → %d files in %s",
        len(figures), n_files, output_dir,
    )
    return exported
