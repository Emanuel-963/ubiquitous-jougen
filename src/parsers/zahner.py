"""Zahner Elektrik EIS file parser (.ism, .isc, .txt exports).

Zahner instruments (IM6, Zennium, PP series) produce several file formats:

1. **ISM / ISC** — Zahner proprietary text-based impedance format::

       [IMPEDANCE SPECTRUM]
       EXCITATION FREQUENCY: 100000 Hz
       EXCITATION AC VOLTAGE: 5 mV rms
       TEMPERATURE: 25 °C
       MEASUREMENT DATE: 2024-01-15
       BIAS VOLTAGE: 0 mV
       NUMBER OF SAMPLES: 61
       FREQUENCY[Hz]  REAL_PART[Ohm]  IMAGINARY_PART[Ohm]
       1.00000E+05    4.563           -0.543
       ...

2. **Text export from Thales / Zahner Analysis** — tab-separated with
   keyword header lines.

The parser detects both variants.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import ParsedEIS, PotentiostatParser

logger = logging.getLogger(__name__)

# Magic strings in Zahner ISM/ISC files
_ZAHNER_MAGIC = ("[impedance spectrum]", "zahner", "im6", "zennium", "thales")
# Column header patterns
_COL_KEYWORDS = ("frequency[hz]", "freq.[hz]", "f/hz", "frequency (hz)")


class ZahnerParser(PotentiostatParser):
    """Parser for Zahner Elektrik EIS files (.ism, .isc, .txt)."""

    EXTENSIONS = [".ism", ".isc", ".txt"]

    @classmethod
    def can_parse(cls, path: str | Path) -> bool:
        p = Path(path)
        if p.suffix.lower() not in cls.EXTENSIONS:
            return False
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                header = "".join(fh.readline() for _ in range(12)).lower()
            return any(magic in header for magic in _ZAHNER_MAGIC)
        except OSError:
            return False

    def parse(self, path: str | Path) -> ParsedEIS:
        """Parse a Zahner EIS file."""
        path = Path(path)
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        meta: dict = {}
        col_line_idx: int | None = None
        separator = "\t"

        # --- Scan header ---
        for i, line in enumerate(lines):
            stripped = line.strip()
            lower = stripped.lower()

            # Metadata key-value (colon or equals separated)
            for sep_char in (":", "="):
                if sep_char in stripped and col_line_idx is None:
                    k, _, v = stripped.partition(sep_char)
                    if k.strip() and not k.strip().startswith("["):
                        meta[k.strip()] = v.strip()

            # Detect column header
            if col_line_idx is None and (
                any(kw in lower for kw in _COL_KEYWORDS)
                or (
                    ("freq" in lower or "f/" in lower)
                    and ("real" in lower or "imag" in lower or "ohm" in lower)
                )
            ):
                col_line_idx = i
                if "," in stripped:
                    separator = ","
                elif "\t" in stripped:
                    separator = "\t"
                else:
                    separator = None  # type: ignore[assignment]

        if col_line_idx is None:
            # Last resort: find first line that looks like float data with 3+ columns
            col_line_idx = _detect_data_start(lines)
            if col_line_idx is None:
                raise ValueError(
                    f"Could not locate EIS data table in Zahner file: {path.name}"
                )
            # Use the line before as synthetic header
            col_names = ["frequency", "zreal", "zimag"]
            data_rows = _parse_data_rows(lines[col_line_idx:], sep=separator)
        else:
            raw_header = lines[col_line_idx].strip()
            if separator:
                col_names = [c.strip() for c in raw_header.split(separator)]
            else:
                col_names = raw_header.split()
            data_rows = _parse_data_rows(lines[col_line_idx + 1 :], sep=separator)

        if not data_rows:
            raise ValueError(f"No data rows found in Zahner file: {path.name}")

        max_cols = max(len(r) for r in data_rows)
        # Pad col_names if needed
        while len(col_names) < max_cols:
            col_names.append(f"col_{len(col_names)}")

        df_raw = pd.DataFrame(data_rows, columns=col_names[:max_cols])
        col_map = _zahner_column_map(df_raw.columns.tolist())

        if "freq" not in col_map:
            # Positional fallback
            col_map["freq"] = df_raw.columns[0]
            col_map["zreal"] = df_raw.columns[1]
            col_map["zimag"] = df_raw.columns[2]

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
            instrument=f"Zahner Elektrik ({meta.get('INSTRUMENT', 'EIS')})",
            extra_meta=meta,
        )
        result.validate()
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_data_start(lines: list) -> int | None:
    """Find the first line that looks like numeric EIS data."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("[") or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 3:
            try:
                float(parts[0].replace(",", "."))
                float(parts[1].replace(",", "."))
                float(parts[2].replace(",", "."))
                return i
            except ValueError:
                pass
    return None


def _parse_data_rows(lines: list, sep: str | None) -> list:
    """Parse data rows from lines after the header."""
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("[") or stripped.startswith("#"):
            continue
        if sep:
            parts = stripped.split(sep)
        else:
            parts = stripped.split()
        if len(parts) >= 3:
            # Check first part is numeric
            try:
                float(parts[0].replace(",", "."))
                rows.append(parts)
            except ValueError:
                pass
    return rows


def _zahner_column_map(columns: list) -> dict:
    """Map Zahner column names to standard keys."""
    col_map: dict = {}
    lower = [
        c.lower()
        .replace(" ", "")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        for c in columns
    ]

    for i, c in enumerate(lower):
        if col_map.get("freq") is None and c in (
            "frequencyhz",
            "freq.hz",
            "fhz",
            "frequency",
            "freq",
            "f/hz",
        ):
            col_map["freq"] = columns[i]
        elif col_map.get("zreal") is None and c in (
            "real_partohm",
            "realpartohm",
            "zrealohm",
            "zreal",
            "re_partohm",
            "re(z)ohm",
            "re(z)",
            "z'ohm",
            "z'",
            "realohm",
            "rezohm",
        ):
            col_map["zreal"] = columns[i]
        elif col_map.get("zimag") is None and c in (
            "imaginary_partohm",
            "imaginarypartohm",
            "zimagohm",
            "zimag",
            "-im_partohm",
            "im(z)ohm",
            "im(z)",
            "z''ohm",
            "z''",
            "imagohm",
            "-imagohm",
            "imzohm",
            "-imzohm",
        ):
            col_map["zimag"] = columns[i]
        elif col_map.get("zmag") is None and c in (
            "|z|ohm",
            "|z|",
            "modulus",
            "zmodohm",
        ):
            col_map["zmag"] = columns[i]
        elif col_map.get("phase") is None and c in (
            "phasedeg",
            "phase_angledeg",
            "phaseangledeg",
            "theta",
            "phideg",
        ):
            col_map["phase"] = columns[i]

    return col_map
