"""ZView / ZPlot compatible EIS exporter (.z format).

The ``.z`` text format is used by Scribner Associates' ZView and ZPlot
impedance analysis software.  It is widely accepted as an interchange
format among electrochemical impedance software.

File structure
--------------
::

    ZAHNPROG    1   0   0   1   1   0   0  0.000E+000  0.000E+000
    <Comment / sample name>
    <N data points>
    <Freq Hz>  <Z' Ohm>  <-Z'' Ohm>  <...optional columns>

The first line is a fixed signature that ZView uses to identify the file.
The second line is a free-text comment (sample name / date).
The third line contains the number of data points.
Data lines use scientific notation, TAB-separated.

Columns written: ``Freq``, ``Zreal``, ``-Zimag`` (ZView convention: the
imaginary part is stored as **-Im(Z)**, i.e. positive for capacitive arcs).

References
----------
* Scribner Associates ZView manual, §A.2 "ASCII File Format"
* https://www.scribner.com/software/68-zview/
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import EISExporter, ExportError

logger = logging.getLogger(__name__)

__all__ = ["ZViewExporter"]


# Fixed first-line signature expected by ZView
_ZAHNPROG_HEADER = "ZAHNPROG    1   0   0   1   1   0   0  0.000E+000  0.000E+000"


class ZViewExporter(EISExporter):
    """Export EIS data in ZView / ZPlot compatible ASCII format (.z).

    The ``-Im(Z)`` convention is applied: if ``zimag`` values are negative
    (the pipeline's internal sign), they are negated so ZView receives
    positive imaginary parts for capacitive arcs, which matches ZView's
    own display convention.
    """

    DEFAULT_EXTENSION = ".z"
    FORMAT_NAME = "ZView / ZPlot (.z)"

    def export_dataframe(
        self,
        df: pd.DataFrame,
        out_path: Path,
        *,
        sample_name: str = "",
        extra_meta: dict | None = None,
    ) -> Path:
        """Write EIS DataFrame to a ZView-compatible ``.z`` ASCII file.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``frequency``, ``zreal``, ``zimag`` columns.
        out_path : Path
            Destination file path.
        sample_name : str
            Label written to the comment line (line 2).
        extra_meta : dict, optional
            Unused for ZView (format has no free-form metadata section),
            but preserved for API compatibility.

        Returns
        -------
        Path
            *out_path* after successful write.
        """
        data = self._validate_df(df).copy()

        # Remove rows where any required value is NaN
        data.dropna(subset=["frequency", "zreal", "zimag"], inplace=True)
        if data.empty:
            raise ExportError(f"No valid data rows to export for '{sample_name}'.")

        # ZView stores -Im(Z): negate zimag if values are negative (cap. convention)
        neg_zimag = -data["zimag"]

        n = len(data)
        comment = sample_name or "IonFlow Pipeline Export"
        if extra_meta and "date" in extra_meta:
            comment += f"  [{extra_meta['date']}]"
        elif not sample_name:
            comment += f"  [{self._now_str()}]"

        lines: list[str] = [
            _ZAHNPROG_HEADER,
            comment,
            str(n),
        ]

        for freq, zr, zi_neg in zip(data["frequency"], data["zreal"], neg_zimag):
            lines.append(f"{freq:.6E}\t{zr:.6E}\t{zi_neg:.6E}")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path
