"""Autolab NOVA / Metrohm EIS file parser (.csv and .txt exports).

Autolab instruments running NOVA software export EIS data as CSV files
with a characteristic header block.

**NOVA 2.x export** (most common)::

    Autolab NOVA 2.1.4
    Procedure: EIS Measurement
    Date: 2024-01-15
    ...
    Frequency (Hz),Z' (Ohm),Z'' (Ohm),|Z| (Ohm),Phase angle (deg),...
    100000,4.563,-0.543,4.595,6.79,...

**Older NOVA / GPES export** may use tab separators and slightly different
column names.  The parser tries both.

The distinguishing signature is the ``Autolab NOVA`` or ``GPES`` string
in the first few lines of the file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import ParsedEIS, PotentiostatParser

logger = logging.getLogger(__name__)

# Magic strings found at top of Autolab exports
_AUTOLAB_MAGIC = ("autolab nova", "gpes", "metrohm autolab", "autolab b.v.")
# FRA / impedance section marker in GPES exports
_GPES_SECTION_MARKERS = ("Frequency(Hz)", "Freq.(Hz)", "Frequency (Hz)")


class AutolabParser(PotentiostatParser):
    """Parser for Autolab NOVA / Metrohm EIS CSV exports."""

    EXTENSIONS = [".csv", ".txt"]

    @classmethod
    def can_parse(cls, path: str | Path) -> bool:
        p = Path(path)
        if p.suffix.lower() not in cls.EXTENSIONS:
            return False
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                header = "".join(fh.readline() for _ in range(8)).lower()
            return any(magic in header for magic in _AUTOLAB_MAGIC)
        except OSError:
            return False

    def parse(self, path: str | Path) -> ParsedEIS:
        """Parse an Autolab NOVA EIS export file."""
        path = Path(path)
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        meta = {}
        col_line_idx: int | None = None
        separator = ","

        # --- Scan header for metadata and locate the column line ---
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Collect metadata key-value pairs
            if ":" in stripped and col_line_idx is None:
                k, _, v = stripped.partition(":")
                meta[k.strip()] = v.strip()

            # Detect column header line (contains "Frequency" or "Freq")
            lower = stripped.lower()
            if (
                col_line_idx is None
                and ("frequency" in lower or "freq" in lower)
                and (
                    "z'" in lower
                    or "zreal" in lower
                    or "re(z)" in lower
                    or "ohm" in lower
                )
            ):
                col_line_idx = i
                # Detect separator
                if "\t" in stripped:
                    separator = "\t"
                elif ";" in stripped:
                    separator = ";"
                else:
                    separator = ","
                break

        if col_line_idx is None:
            raise ValueError(
                f"Could not locate EIS column header in Autolab file: {path.name}"
            )

        # --- Parse column names and data ---
        col_names = [c.strip() for c in lines[col_line_idx].split(separator)]

        data_rows = []
        for line in lines[col_line_idx + 1 :]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(separator)
            if len(parts) >= 3:
                data_rows.append(parts)

        if not data_rows:
            raise ValueError(f"No data rows found in Autolab file: {path.name}")

        df_raw = pd.DataFrame(data_rows, columns=col_names[: len(data_rows[0])])
        col_map = _autolab_column_map(df_raw.columns.tolist())

        if "freq" not in col_map or "zreal" not in col_map or "zimag" not in col_map:
            raise ValueError(
                f"Could not identify Freq/Zreal/Zimag columns in {path.name}.\n"
                f"Found columns: {list(df_raw.columns)}"
            )

        df = pd.DataFrame()
        df["frequency"] = self._to_numeric(df_raw[col_map["freq"]])
        df["zreal"] = self._to_numeric(df_raw[col_map["zreal"]])
        df["zimag"] = self._to_numeric(df_raw[col_map["zimag"]])

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
            instrument=f"Autolab NOVA ({meta.get('Autolab NOVA', 'EIS')})",
            extra_meta=meta,
        )
        result.validate()
        return result


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------


def _autolab_column_map(columns: list) -> dict:
    """Map Autolab NOVA column names to standard keys."""
    col_map: dict = {}
    lower = [
        c.lower().replace(" ", "").replace("(", "").replace(")", "") for c in columns
    ]

    for i, c in enumerate(lower):
        if col_map.get("freq") is None and c in (
            "frequencyhz",
            "freq.hz",
            "freqhz",
            "frequency",
            "freq",
            "f/hz",
            "fhz",
        ):
            col_map["freq"] = columns[i]
        elif col_map.get("zreal") is None and c in (
            "z'ohm",
            "z'",
            "zrealohm",
            "zreal",
            "re(z)ohm",
            "re(z)",
            "z'(ohm)",
            "zreohm",
        ):
            col_map["zreal"] = columns[i]
        elif col_map.get("zimag") is None and c in (
            "z''ohm",
            "z''",
            "-z''ohm",
            "-z''",
            "zimagohm",
            "zimag",
            "-im(z)ohm",
            "-im(z)",
            "z''(ohm)",
            "-z''(ohm)",
        ):
            col_map["zimag"] = columns[i]
        elif col_map.get("zmag") is None and c in (
            "|z|ohm",
            "|z|",
            "zohm",
            "z",
            "modulus",
        ):
            col_map["zmag"] = columns[i]
        elif col_map.get("phase") is None and c in (
            "phaseangledeg",
            "phaseangle",
            "phasedeg",
            "phase",
            "theta",
        ):
            col_map["phase"] = columns[i]

    return col_map
