"""Solartron / AMETEK Scientific Instruments EIS file parser.

Handles two file formats produced by Solartron instruments (SI 1260,
1287, 1470, Modulab):

1. **IDF** — "Instrument Data File" (``.idf``), the native text export
   from Solartron SmartSoft / ZPlot::

       SOLARTRON INSTRUMENT 1260
       File  : experiment.idf
       Date  : 01/01/2024  Time : 10:00:00
       ...
       BEGIN
       Frequency         Z'              -Z''     |Z|    Phase
       1.000000E+006     1.2346E+000     5.679E-001 ...
       ...
       END

2. **DFR** — "Data File Results" (``.dfr``), a simpler tab-separated
   export from Corrware / Scribner ZView when saving Solartron data::

       Freq (Hz)  Zreal (Ohm)  Zimag (Ohm)
       100000     1.234        -0.568
       ...

The parser auto-detects which variant it is handling by inspecting the
file header.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .base import ParsedEIS, PotentiostatParser

logger = logging.getLogger(__name__)

_IDF_MAGIC = ("solartron", "smartsoft", "zplot", "modulab", "si 1260", "si 1287")
_DFR_MAGIC = ("freq (hz)", "zreal", "zimag", "z-real", "z-imag")

# Regex for scientific notation numbers (handles both 1.23E+003 and 1.23e+3)
_NUM = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"


class SolartronParser(PotentiostatParser):
    """Parser for Solartron / AMETEK EIS files (.idf, .dfr)."""

    EXTENSIONS = [".idf", ".dfr"]

    @classmethod
    def can_parse(cls, path: str | Path) -> bool:
        p = Path(path)
        if p.suffix.lower() not in cls.EXTENSIONS:
            return False
        # Additionally sniff the first 20 lines for magic keywords
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                header = fh.read(2048).lower()
            if p.suffix.lower() == ".idf":
                return any(kw in header for kw in _IDF_MAGIC)
            # .dfr: check column keywords
            return any(kw in header for kw in _DFR_MAGIC)
        except OSError:
            return False

    def parse(self, path: str | Path) -> ParsedEIS:
        p = Path(path)
        if p.suffix.lower() == ".idf":
            return self._parse_idf(p)
        return self._parse_dfr(p)

    # ── IDF parser ──────────────────────────────────────────────────────────

    def _parse_idf(self, path: Path) -> ParsedEIS:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        extra_meta: dict = {}
        data_lines: list[str] = []
        in_data = False
        col_indices: dict[str, int] = {}

        for line in lines:
            stripped = line.strip()

            # Extract metadata from header
            if not in_data:
                for key in ("date", "time", "file", "operator", "temp"):
                    if stripped.lower().startswith(key):
                        parts = stripped.split(":", 1)
                        if len(parts) == 2:
                            extra_meta[key] = parts[1].strip()

            # Look for data section marker
            if stripped.upper() == "BEGIN":
                in_data = True
                continue
            if stripped.upper() == "END":
                break

            if not in_data:
                continue

            # First non-empty line after BEGIN is the column header
            if not col_indices and stripped:
                headers = re.split(r"\s+", stripped)
                for i, h in enumerate(headers):
                    h_lower = h.lower()
                    if "freq" in h_lower:
                        col_indices["freq"] = i
                    elif h_lower in ("z'", "zreal", "z-real", "re(z)"):
                        col_indices["zreal"] = i
                    elif h_lower in (
                        "-z''",
                        "zimag",
                        "-zimag",
                        "z-imag",
                        "-im(z)",
                        "im(z)",
                    ):
                        col_indices["zimag"] = i
                continue

            if re.match(r"^\s*" + _NUM, stripped):
                data_lines.append(stripped)

        if not col_indices:
            raise ValueError(f"Could not detect column layout in IDF file: {path}")
        if "freq" not in col_indices:
            raise ValueError(f"Frequency column not found in IDF file: {path}")

        return self._build_result(
            path, data_lines, col_indices, extra_meta, "Solartron IDF"
        )

    # ── DFR parser ──────────────────────────────────────────────────────────

    def _parse_dfr(self, path: Path) -> ParsedEIS:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        col_indices: dict[str, int] = {}
        data_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", ";")):
                continue

            # Detect header row
            if not col_indices:
                tokens = re.split(r"[\t,;]+|\s{2,}", stripped)
                for i, tok in enumerate(tokens):
                    t = tok.lower().replace(" ", "").replace("(", "").replace(")", "")
                    if "freq" in t:
                        col_indices["freq"] = i
                    elif t in ("zreal", "z-real", "z'", "re(z)", "zrealohm"):
                        col_indices["zreal"] = i
                    elif t in (
                        "zimag",
                        "-zimag",
                        "z-imag",
                        "-z''",
                        "-im(z)",
                        "zimag(ohm)",
                    ):
                        col_indices["zimag"] = i
                if col_indices:
                    continue

            if re.match(r"^\s*" + _NUM, stripped):
                data_lines.append(stripped)

        if not col_indices:
            raise ValueError(f"Could not detect column layout in DFR file: {path}")

        return self._build_result(path, data_lines, col_indices, {}, "Solartron DFR")

    # ── shared helper ────────────────────────────────────────────────────────

    def _build_result(
        self,
        path: Path,
        data_lines: list[str],
        col_indices: dict[str, int],
        extra_meta: dict,
        instrument: str,
    ) -> ParsedEIS:
        if not data_lines:
            raise ValueError(f"No numeric data rows found in {path}")

        records: list[dict] = []
        for line in data_lines:
            tokens = re.split(r"\s+", line.strip())
            try:
                freq = float(tokens[col_indices["freq"]])
                zreal = float(tokens[col_indices.get("zreal", 1)])
                # Solartron convention: column is "−Z''" (already negative imaginary)
                # We want zimag = −Z'' = negative for capacitive data, so keep sign.
                raw_imag = float(tokens[col_indices.get("zimag", 2)])
                zimag = raw_imag  # already in correct sign convention
                records.append({"frequency": freq, "zreal": zreal, "zimag": zimag})
            except (IndexError, ValueError):
                continue

        if not records:
            raise ValueError(f"No valid data rows parsed from {path}")

        df = pd.DataFrame(records)
        df = df[df["frequency"] > 0].copy()
        df = df.sort_values("frequency", ascending=False).reset_index(drop=True)

        result = ParsedEIS(
            data=df,
            source_file=str(path.resolve()),
            instrument=instrument,
            extra_meta=extra_meta,
        )
        result.validate()
        return result
