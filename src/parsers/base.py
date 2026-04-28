"""Base interface for potentiostat file parsers.

All parsers must produce a DataFrame with exactly these columns:

    frequency  : float  — frequency in Hz (positive)
    zreal      : float  — real part of impedance in Ohm
    zimag      : float  — imaginary part in Ohm, sign convention -Z''
                          (negative for capacitive arcs)

Additional metadata columns may be present but are optional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class ParsedEIS:
    """Container returned by every parser.

    Attributes
    ----------
    data : pd.DataFrame
        Columns: ``frequency``, ``zreal``, ``zimag`` (required).
        May contain extra columns such as ``|Z|``, ``phase_deg``.
    source_file : str
        Absolute path to the parsed file.
    instrument : str
        Human-readable instrument / software name, e.g. "Gamry Framework".
    extra_meta : dict
        Any key-value pairs extracted from the file header (optional).
    """

    data: pd.DataFrame
    source_file: str
    instrument: str = ""
    extra_meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError if required columns are missing or data is empty."""
        required = {"frequency", "zreal", "zimag"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Parser result is missing required columns: {missing}")
        if len(self.data) == 0:
            raise ValueError(f"Parser returned empty DataFrame for {self.source_file}")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class PotentiostatParser(ABC):
    """Abstract base class for potentiostat EIS file parsers."""

    #: File extensions handled by this parser (lowercase, with dot).
    EXTENSIONS: List[str] = []

    @classmethod
    def can_parse(cls, path: str | Path) -> bool:
        """Return True if this parser can handle the given file.

        The default implementation checks the file extension.
        Override to add magic-byte or header inspection.
        """
        return Path(path).suffix.lower() in cls.EXTENSIONS

    @abstractmethod
    def parse(self, path: str | Path) -> ParsedEIS:
        """Parse a potentiostat EIS file and return a :class:`ParsedEIS`.

        Parameters
        ----------
        path : str or Path
            Path to the file to parse.

        Returns
        -------
        ParsedEIS
            Validated container with ``frequency``, ``zreal``, ``zimag``
            columns and optional metadata.

        Raises
        ------
        ValueError
            If the file cannot be parsed or is not a valid EIS file.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        """Convert a string Series to float, handling comma decimals."""
        return pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )

    @staticmethod
    def _enforce_zimag_convention(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure zimag follows the -Z'' convention (negative for capacitive).

        Most EIS data uses negative imaginary part for capacitive systems.
        If the majority of zimag values are positive, negate the column.
        """
        df = df.copy()
        if df["zimag"].dropna().gt(0).mean() > 0.5:
            df["zimag"] = -df["zimag"].abs()
        return df
