"""MEISP (Multiple Electrochemical Impedance Spectra Parameterization) exporter.

MEISP is a widely-used freeware for fitting EIS data, developed at the
University of Tartu (Estonia).  Its text-based ``.mps`` format is also
understood by EIS fitting software such as EIS Spectrum Analyser, RelaxIS,
and custom MATLAB routines.

File structure
--------------
::

    Measurement file
    <title / sample name>
    <date>
    <n data points>
    <Freq Hz> <Z' Ohm> <Z'' Ohm>
    ...

* The imaginary part is stored with its **natural sign** (negative for
  capacitive arcs) in MEISP.
* Columns are TAB-separated, using scientific notation.
* The first four header lines are mandatory; the order must be preserved.

References
----------
* Jukka Juuti, Kari Lanu, "MEISP 3.0 User's Guide", University of Oulu, 2005
* http://www.abc.chemistry.bsu.by/vi/analyser/

Note: MEISP does not have a publicly documented formal specification.
The format documented here is based on files produced by MEISP 3.0 and
accepted by RelaxIS 3 and EIS Spectrum Analyser 1.0.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import EISExporter, ExportError

logger = logging.getLogger(__name__)

__all__ = ["MEISPExporter"]


class MEISPExporter(EISExporter):
    """Export EIS data in MEISP ``.mps`` text format.

    The MEISP format stores impedance with the **physical sign convention**:
    Im(Z) is negative for capacitive arcs.  Because the IonFlow pipeline
    uses the same negative convention internally, the ``zimag`` values are
    written as-is without any sign transformation.
    """

    DEFAULT_EXTENSION = ".mps"
    FORMAT_NAME = "MEISP (.mps)"

    def export_dataframe(
        self,
        df: pd.DataFrame,
        out_path: Path,
        *,
        sample_name: str = "",
        extra_meta: dict | None = None,
    ) -> Path:
        """Write EIS DataFrame to a MEISP-compatible ``.mps`` ASCII file.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``frequency``, ``zreal``, ``zimag`` columns.
        out_path : Path
            Destination file path.
        sample_name : str
            Written to the title line (line 2).
        extra_meta : dict, optional
            Not used by the MEISP format; kept for API compatibility.

        Returns
        -------
        Path
            *out_path* after successful write.
        """
        data = self._validate_df(df).copy()
        data.dropna(subset=["frequency", "zreal", "zimag"], inplace=True)
        if data.empty:
            raise ExportError(f"No valid data rows for '{sample_name}'.")

        title = sample_name or "IonFlow Pipeline Export"
        date_str = self._now_str()
        n = len(data)

        lines: list[str] = [
            "Measurement file",
            title,
            date_str,
            str(n),
        ]

        for _, row in data.iterrows():
            # MEISP: Freq  Zreal  Zimag (negative for capacitive)
            lines.append(
                f"{row['frequency']:.6E}\t{row['zreal']:.6E}\t{row['zimag']:.6E}"
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path
