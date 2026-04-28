"""Scientific EIS data export package (Phase 3).

This package provides publication-ready exporters for EIS data in formats
accepted by leading electrochemical impedance software and LaTeX manuscripts.

Quick start
-----------
::

    from src.export import export_eis

    # raw_eis = {"sample_A.txt": DataFrame, "sample_B.txt": DataFrame, ...}
    paths = export_eis(raw_eis, fmt="zview", out_dir="outputs/export")

Supported formats
-----------------

+----------+---------------------------+------------------------------------+
| Key      | Class                     | Description                        |
+==========+===========================+====================================+
| zview    | :class:`ZViewExporter`    | ZView / ZPlot (.z)                 |
+----------+---------------------------+------------------------------------+
| latex    | :class:`LaTeXExporter`    | LaTeX booktabs table (.tex)        |
+----------+---------------------------+------------------------------------+
| origin   | :class:`OriginCSVExporter`| OriginPro annotated CSV (.csv)     |
+----------+---------------------------+------------------------------------+
| meisp    | :class:`MEISPExporter`    | MEISP / EIS Spectrum Analyser (.mps)|
+----------+---------------------------+------------------------------------+

Advanced usage
--------------
Circuit fitting table → LaTeX::

    from src.export import export_circuit_table_latex

    export_circuit_table_latex(
        eis_result.circuit_table,
        "outputs/export/table_circuit.tex",
        caption="Equivalent circuit parameters",
        label="tab:params",
    )

Ranking table → LaTeX::

    from src.export import export_ranking_latex

    export_ranking_latex(
        eis_result.ranked_df,
        "outputs/export/table_ranking.tex",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Type

import pandas as pd

from .base import EISExporter, ExportError
from .latex import LaTeXExporter, export_circuit_table_latex, export_ranking_latex
from .meisp import MEISPExporter
from .origin import OriginCSVExporter
from .zview import ZViewExporter

__all__ = [
    "ZViewExporter",
    "LaTeXExporter",
    "OriginCSVExporter",
    "MEISPExporter",
    "EISExporter",
    "ExportError",
    "export_eis",
    "export_circuit_table_latex",
    "export_ranking_latex",
    "EXPORTERS",
]

# Registry: short key → exporter class
EXPORTERS: Dict[str, Type[EISExporter]] = {
    "zview": ZViewExporter,
    "latex": LaTeXExporter,
    "origin": OriginCSVExporter,
    "meisp": MEISPExporter,
}


def export_eis(
    raw_eis: Dict[str, pd.DataFrame],
    fmt: str,
    out_dir: str | Path,
    *,
    extra_meta: dict | None = None,
) -> List[Path]:
    """Export a collection of EIS measurements to the requested format.

    Parameters
    ----------
    raw_eis : dict
        ``{original_filename: DataFrame}`` mapping from ``EISResult.raw_eis``.
        Each DataFrame must contain ``frequency``, ``zreal``, ``zimag``.
    fmt : str
        One of: ``"zview"``, ``"latex"``, ``"origin"``, ``"meisp"``.
    out_dir : str or Path
        Destination directory.  Created automatically if absent.
    extra_meta : dict, optional
        Metadata forwarded to each exporter (e.g. ``{"date": "2024-01-01"}``).

    Returns
    -------
    list of Path
        Paths of all files written.

    Raises
    ------
    ValueError
        If *fmt* is not a recognised exporter key.
    ExportError
        If writing any individual file fails.
    """
    fmt = fmt.lower().strip()
    if fmt not in EXPORTERS:
        raise ValueError(
            f"Unknown export format: '{fmt}'. "
            f"Available: {', '.join(sorted(EXPORTERS))}"
        )

    exporter = EXPORTERS[fmt]()
    return exporter.export_all(raw_eis, out_dir, extra_meta=extra_meta)
