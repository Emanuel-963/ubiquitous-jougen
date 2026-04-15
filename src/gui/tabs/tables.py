"""Table configuration metadata for the six GUI tables.

Pure data — no widgets.  The configuration is consumed by both
the ``FilterableTableManager`` and the actual tkinter table builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TableColumnConfig:
    """Display configuration for a single table."""

    key: str
    label: str
    default_sort: str = ""
    default_name: str = "table.csv"
    editable: bool = False


def table_column_configs() -> Dict[str, TableColumnConfig]:
    """Return the canonical table configuration map."""
    return {
        "eis": TableColumnConfig(
            key="eis",
            label="EIS",
            default_sort="Arquivo",
            default_name="eis_results.csv",
        ),
        "cic": TableColumnConfig(
            key="cic",
            label="Ciclagem",
            default_sort="",
            default_name="ciclagem_results.csv",
        ),
        "circuit": TableColumnConfig(
            key="circuit",
            label="Circuitos",
            default_sort="",
            default_name="circuit_fits.csv",
        ),
        "drt": TableColumnConfig(
            key="drt",
            label="DRT",
            default_sort="",
            default_name="drt_results.csv",
        ),
        "drt_peaks": TableColumnConfig(
            key="drt_peaks",
            label="DRT Peaks",
            default_sort="",
            default_name="drt_peaks.csv",
        ),
        "drt_eis": TableColumnConfig(
            key="drt_eis",
            label="DRT + EIS",
            default_sort="",
            default_name="drt_eis_join.csv",
        ),
    }
