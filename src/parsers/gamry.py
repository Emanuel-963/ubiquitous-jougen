"""Gamry Framework EIS file parser (.dta).

Gamry `.dta` files are structured plain-text files with a header section
followed by a ``ZCURVE`` (or ``CORPOTENTIALSWEEP``) data table.

Typical structure::

    EXPLAIN
    TAG     EISPOT
    TITLE   EIS Measurement
    ...
    ZCURVE  TABLE   ...
    Pt      Time    Freq    Zreal   Zimag   Zsig    ...
    #       s       Hz      ohm     ohm     V       ...
    1       0.0     100000  4.56    -0.54   ...
    ...

The column order can vary; this parser identifies columns by name.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import ParsedEIS, PotentiostatParser

logger = logging.getLogger(__name__)

# Magic string present in all Gamry .dta files
_GAMRY_MAGIC = "EXPLAIN"
# Table section markers (most common)
_TABLE_MARKERS = ("ZCURVE", "CORPOTENTIALSWEEP", "ZCHROM", "IMPEDANCE")


class GamryParser(PotentiostatParser):
    """Parser for Gamry Framework EIS files (.dta)."""

    EXTENSIONS = [".dta"]

    @classmethod
    def can_parse(cls, path: str | Path) -> bool:
        p = Path(path)
        if p.suffix.lower() not in cls.EXTENSIONS:
            return False
        # Confirm magic string in first ~10 lines
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                header = "".join(fh.readline() for _ in range(10))
            return _GAMRY_MAGIC in header
        except OSError:
            return False

    def parse(self, path: str | Path) -> ParsedEIS:
        """Parse a Gamry .dta EIS file."""
        path = Path(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        # --- Extract header metadata ---
        meta: dict = {}
        table_start = None
        col_names: list = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Key-value pairs in header (TAB separated)
            parts = stripped.split("\t")
            if (
                len(parts) >= 2
                and parts[0].isupper()
                and not any(stripped.startswith(m) for m in _TABLE_MARKERS)
            ):
                meta[parts[0]] = parts[1] if len(parts) > 1 else ""

            # Detect start of the impedance data table
            for marker in _TABLE_MARKERS:
                if stripped.startswith(marker):
                    # Next two lines are column names and units
                    if i + 2 < len(lines):
                        col_names = lines[i + 1].strip().split("\t")
                        table_start = i + 3
                    break

            if table_start is not None:
                break
            i += 1

        if table_start is None:
            raise ValueError(
                f"Could not locate impedance data table in Gamry file: {path.name}"
            )

        # --- Parse data rows ---
        data_lines = []
        for line in lines[table_start:]:
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("CURVE")
                or stripped.startswith("EXPLAIN")
            ):
                break
            data_lines.append(stripped.split("\t"))

        if not data_lines:
            raise ValueError(f"No data rows found in Gamry file: {path.name}")

        df_raw = pd.DataFrame(data_lines, columns=col_names[: len(data_lines[0])])

        # --- Map columns to standard names ---
        col_map = _build_gamry_column_map(df_raw.columns.tolist())
        if not col_map:
            raise ValueError(
                f"Could not identify Freq/Zreal/Zimag columns in: {path.name}"
            )

        df = pd.DataFrame()
        df["frequency"] = self._to_numeric(df_raw[col_map["freq"]])
        df["zreal"] = self._to_numeric(df_raw[col_map["zreal"]])
        df["zimag"] = self._to_numeric(df_raw[col_map["zimag"]])

        # Optional extras
        if col_map.get("zmag"):
            df["|Z|"] = self._to_numeric(df_raw[col_map["zmag"]])
        if col_map.get("phase"):
            df["phase_deg"] = self._to_numeric(df_raw[col_map["phase"]])

        df = df.dropna(subset=["frequency", "zreal", "zimag"])
        df = self._enforce_zimag_convention(df)
        df = df.reset_index(drop=True)

        result = ParsedEIS(
            data=df,
            source_file=str(path.resolve()),
            instrument=f"Gamry Framework ({meta.get('TAG', 'EIS')})",
            extra_meta=meta,
        )
        result.validate()
        return result


# ---------------------------------------------------------------------------
# Column mapping helpers
# ---------------------------------------------------------------------------


def _build_gamry_column_map(columns: list) -> dict:
    """Map Gamry column names to standard keys.

    Returns a dict with keys ``freq``, ``zreal``, ``zimag`` (required)
    and optionally ``zmag``, ``phase``.
    """
    col_map: dict = {}
    lower = [c.lower() for c in columns]

    for i, c in enumerate(lower):
        if col_map.get("freq") is None and c in ("freq", "frequency", "f"):
            col_map["freq"] = columns[i]
        elif col_map.get("zreal") is None and c in ("zreal", "z_real", "zre", "z'"):
            col_map["zreal"] = columns[i]
        elif col_map.get("zimag") is None and c in (
            "zimag",
            "z_imag",
            "zim",
            "z''",
            "-zimag",
            'z"',
        ):
            col_map["zimag"] = columns[i]
        elif col_map.get("zmag") is None and c in ("zmag", "|z|", "z", "zmod"):
            col_map["zmag"] = columns[i]
        elif col_map.get("phase") is None and c in ("phase", "phz", "theta", "zphz"):
            col_map["phase"] = columns[i]

    # Fallback: use positional guesses (Pt, Time, Freq, Zreal, Zimag, ...)
    if "freq" not in col_map and len(columns) >= 5:
        col_map.setdefault("freq", columns[2])
        col_map.setdefault("zreal", columns[3])
        col_map.setdefault("zimag", columns[4])

    return col_map
