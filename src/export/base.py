"""Base class for scientific EIS data exporters (Phase 3).

All exporters follow the same interface so they can be called uniformly
from the GUI and from scripts.  Each exporter writes one or more files
for a *collection* of EIS measurements (the standard ``raw_eis`` dict).

Usage example
-------------
::

    from src.export import export_eis

    # raw_eis = {filename: DataFrame(frequency, zreal, zimag), ...}
    paths = export_eis(raw_eis, fmt="zview", out_dir="outputs/export")

Subclass contract
-----------------
Subclasses must implement :meth:`export_dataframe` and set
:attr:`DEFAULT_EXTENSION`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["EISExporter", "ExportError"]


class ExportError(RuntimeError):
    """Raised when an export operation fails."""


class EISExporter(ABC):
    """Abstract base for single-file EIS exporters.

    Each subclass handles one output format.  The main public method is
    :meth:`export_all`, which iterates over a ``raw_eis`` dict and writes
    one file per measurement.
    """

    #: File extension produced by this exporter, e.g. ``.z``
    DEFAULT_EXTENSION: str = ".txt"
    #: Human-readable format name shown in the GUI
    FORMAT_NAME: str = "Generic"

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    def export_dataframe(
        self,
        df: pd.DataFrame,
        out_path: Path,
        *,
        sample_name: str = "",
        extra_meta: dict | None = None,
    ) -> Path:
        """Write *df* (with columns ``frequency``, ``zreal``, ``zimag``) to
        *out_path* in the format specific to this exporter.

        Parameters
        ----------
        df : pd.DataFrame
            EIS data with at least ``frequency``, ``zreal``, ``zimag`` columns.
        out_path : Path
            Full destination path (directory + filename).
        sample_name : str, optional
            Label used inside the file header.
        extra_meta : dict, optional
            Additional metadata to embed in the header (if the format
            supports free-form comments).

        Returns
        -------
        Path
            The path that was written (same as *out_path*).
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def export_all(
        self,
        raw_eis: Dict[str, pd.DataFrame],
        out_dir: str | Path,
        *,
        extra_meta: dict | None = None,
    ) -> List[Path]:
        """Export every measurement in *raw_eis* to *out_dir*.

        Parameters
        ----------
        raw_eis : dict
            ``{original_filename: DataFrame}`` mapping produced by the EIS
            pipeline (``EISResult.raw_eis``).
        out_dir : str or Path
            Target directory.  Created if it does not exist.
        extra_meta : dict, optional
            Metadata forwarded to :meth:`export_dataframe`.

        Returns
        -------
        list of Path
            Paths of written files.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        written: List[Path] = []
        for fname, df in raw_eis.items():
            try:
                stem = Path(fname).stem
                dest = out_dir / (stem + self.DEFAULT_EXTENSION)
                result = self.export_dataframe(
                    df,
                    dest,
                    sample_name=stem,
                    extra_meta=extra_meta or {},
                )
                written.append(result)
                logger.debug("Exported %s → %s", fname, dest)
            except Exception as exc:
                logger.warning("Failed to export %s: %s", fname, exc)
        return written

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns are present and numeric."""
        required = {"frequency", "zreal", "zimag"}
        missing = required - set(df.columns)
        if missing:
            raise ExportError(
                f"DataFrame missing required columns: {', '.join(sorted(missing))}"
            )
        return df[["frequency", "zreal", "zimag"]].copy()

    @staticmethod
    def _now_str() -> str:
        """Return current timestamp as ``YYYY-MM-DD HH:MM:SS``."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
