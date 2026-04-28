"""OriginPro / Origin CSV exporter for EIS data.

Origin (OriginLab) accepts CSV files with metadata stored in ``#``-prefixed
comment lines before the data.  When such a file is drag-dropped into
Origin or imported via *File → Import → CSV*, Origin 2021+ automatically
reads the inline comments as *long-name*, *units*, and *comment* row
annotations.

File structure
--------------
::

    # IonFlow Pipeline Export
    # Sample: <sample_name>
    # Date: <timestamp>
    # Instrument: IonFlow Pipeline v0.2.0
    # Columns: Frequency(Hz), Z'(Ohm), -Z''(Ohm), |Z|(Ohm), Phase(deg)
    #
    Frequency(Hz),Z'(Ohm),-Z''(Ohm),|Z|(Ohm),Phase(deg)
    1.000000E+04,1.234500E+01,3.210000E+00,1.275600E+01,-1.463000E+01
    ...

The ``#``-comment block is parsed by Origin as:
* ``IonFlow Pipeline Export`` → long name for the column group
* ``Sample: ...`` → project comment

References
----------
* OriginLab documentation: "Import ASCII with Embedded Notes"
  https://www.originlab.com/doc/Origin-Help/Import-ASCII
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from .base import EISExporter, ExportError

logger = logging.getLogger(__name__)

__all__ = ["OriginCSVExporter"]


class OriginCSVExporter(EISExporter):
    """Export EIS data as an Origin-compatible CSV file.

    Adds ``#``-prefixed metadata comments before the column header so
    Origin can automatically annotate the imported columns.  The file is
    plain UTF-8 CSV and can also be opened in Excel or any text editor.
    """

    DEFAULT_EXTENSION = ".csv"
    FORMAT_NAME = "Origin CSV (.csv)"

    def export_dataframe(
        self,
        df: pd.DataFrame,
        out_path: Path,
        *,
        sample_name: str = "",
        extra_meta: dict | None = None,
    ) -> Path:
        """Write EIS DataFrame to an Origin-compatible CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``frequency``, ``zreal``, ``zimag`` columns.
        out_path : Path
            Destination file path.
        sample_name : str
            Written to the ``# Sample:`` comment line.
        extra_meta : dict, optional
            Additional key-value pairs appended as ``# <key>: <value>``
            comment lines.

        Returns
        -------
        Path
            *out_path* after successful write.
        """
        data = self._validate_df(df).copy()
        data.dropna(subset=["frequency", "zreal", "zimag"], inplace=True)
        if data.empty:
            raise ExportError(f"No valid data rows for '{sample_name}'.")

        # Derived columns
        neg_zimag = -data["zimag"]
        zmag = (data["zreal"] ** 2 + data["zimag"] ** 2) ** 0.5
        phase = data.apply(
            lambda r: math.degrees(math.atan2(-r["zimag"], r["zreal"])), axis=1
        )

        meta = extra_meta or {}
        comment_lines = [
            "# IonFlow Pipeline Export",
            f"# Sample: {sample_name or 'Unknown'}",
            f"# Date: {self._now_str()}",
            "# Software: IonFlow Pipeline (github.com/Emanuel-963/ubiquitous-jougen)",
        ]
        for k, v in meta.items():
            comment_lines.append(f"# {k}: {v}")
        comment_lines.append("#")
        comment_lines.append(
            "# Columns: Frequency(Hz), Z'(Ohm), -Z''(Ohm), |Z|(Ohm), Phase(deg)"
        )
        comment_lines.append("#")

        # Origin long-name row (Origin reads first non-comment line as header)
        col_header = "Frequency(Hz),Z'(Ohm),-Z''(Ohm),|Z|(Ohm),Phase(deg)"

        data_lines = []
        for freq, zr, zi_neg, zm, ph in zip(
            data["frequency"], data["zreal"], neg_zimag, zmag, phase
        ):
            data_lines.append(f"{freq:.6E},{zr:.6E},{zi_neg:.6E},{zm:.6E},{ph:.4f}")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(comment_lines + [col_header] + data_lines) + "\n"
        out_path.write_text(content, encoding="utf-8")
        return out_path
