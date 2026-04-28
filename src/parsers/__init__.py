"""Potentiostat file parser package.

Usage
-----
The simplest entry-point is :func:`parse_eis_file`, which auto-detects
the file format and returns a standardised DataFrame::

    from src.parsers import parse_eis_file

    result = parse_eis_file("data/raw/sample.dta")
    df = result.data  # columns: frequency, zreal, zimag
    print(result.instrument)  # "Gamry Framework (EIS)"

For direct control, use the individual parsers::

    from src.parsers import GamryParser, BioLogicParser, AutolabParser, ZahnerParser

Supported formats
-----------------
+------------+-------------------+---------------------------------------+
| Parser     | Extensions        | Instrument                            |
+============+===================+=======================================+
| Gamry      | .dta              | Gamry Framework EIS                   |
+------------+-------------------+---------------------------------------+
| BioLogic   | .mpr, .mpt, .txt  | BioLogic EC-Lab (binary + text)       |
+------------+-------------------+---------------------------------------+
| Autolab    | .csv, .txt        | Autolab NOVA / Metrohm                |
+------------+-------------------+---------------------------------------+
| Zahner     | .ism, .isc, .txt  | Zahner Elektrik (Thales / IM6)        |
+------------+-------------------+---------------------------------------+

Auto-detection order
--------------------
Each parser exposes a ``can_parse(path)`` classmethod that checks both
the file extension *and* a magic string / magic bytes in the file header.
:func:`parse_eis_file` tries all registered parsers in sequence and uses
the first that reports it can handle the file.

If no specialised parser matches, the fallback :class:`GenericCSVParser`
wraps the existing ``src.loader.load_eis_file`` logic so that all
previously supported CSV/TXT files continue to work unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Type

from .autolab import AutolabParser
from .base import ParsedEIS, PotentiostatParser
from .biologic import BioLogicParser
from .gamry import GamryParser
from .zahner import ZahnerParser

logger = logging.getLogger(__name__)

__all__ = [
    "GamryParser",
    "BioLogicParser",
    "AutolabParser",
    "ZahnerParser",
    "GenericCSVParser",
    "parse_eis_file",
    "detect_parser",
    "ParsedEIS",
    "PotentiostatParser",
    "REGISTERED_PARSERS",
]

# ---------------------------------------------------------------------------
# Generic CSV fallback
# ---------------------------------------------------------------------------


class GenericCSVParser(PotentiostatParser):
    """Fallback parser that wraps :func:`src.loader.load_eis_file`.

    Handles all plain CSV / TXT files that do not match any specialised
    potentiostat format (the formats used in the project's raw data folder).
    """

    EXTENSIONS = [".txt", ".csv", ".dat", ".asc"]

    @classmethod
    def can_parse(cls, path: str | Path) -> bool:
        return Path(path).suffix.lower() in cls.EXTENSIONS

    def parse(self, path: str | Path) -> ParsedEIS:
        from src.loader import load_eis_file  # avoid circular at import time

        path = Path(path)
        df = load_eis_file(str(path))

        result = ParsedEIS(
            data=df,
            source_file=str(path.resolve()),
            instrument="Generic CSV/TXT",
            extra_meta={},
        )
        result.validate()
        return result


# ---------------------------------------------------------------------------
# Registry and auto-detection
# ---------------------------------------------------------------------------

# Order matters: more specific parsers first, generic fallback last.
REGISTERED_PARSERS: List[Type[PotentiostatParser]] = [
    GamryParser,
    BioLogicParser,
    AutolabParser,
    ZahnerParser,
    GenericCSVParser,
]


def detect_parser(path: str | Path) -> Optional[Type[PotentiostatParser]]:
    """Return the first registered parser class that can handle *path*.

    Returns ``None`` if no parser matches (should not happen in practice
    since ``GenericCSVParser`` is the last fallback).

    Parameters
    ----------
    path : str or Path
        File to probe.

    Returns
    -------
    Type[PotentiostatParser] or None
        The parser class (not an instance).
    """
    for parser_cls in REGISTERED_PARSERS:
        try:
            if parser_cls.can_parse(path):
                logger.debug(
                    "Auto-detected %s for file: %s",
                    parser_cls.__name__,
                    Path(path).name,
                )
                return parser_cls
        except Exception as exc:
            logger.debug("Parser probe failed for %s: %s", parser_cls.__name__, exc)
    return None


def parse_eis_file(path: str | Path) -> ParsedEIS:
    """Parse any EIS file by auto-detecting its format.

    This is the main public entry-point.  It probes each registered parser
    in order, uses the first match, and returns a validated :class:`ParsedEIS`.

    Parameters
    ----------
    path : str or Path
        Path to the EIS file.  Supported: .dta, .mpr, .mpt, .ism, .isc,
        .csv, .txt, .dat, .asc.

    Returns
    -------
    ParsedEIS
        Container with ``data`` (DataFrame with ``frequency``, ``zreal``,
        ``zimag``), ``instrument``, ``source_file``, and ``extra_meta``.

    Raises
    ------
    ValueError
        If no parser can handle the file or the file contains no valid data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EIS file not found: {path}")

    parser_cls = detect_parser(path)
    if parser_cls is None:
        raise ValueError(
            f"No parser found for file: {path.name}\n"
            f"Supported extensions: "
            + ", ".join(ext for cls in REGISTERED_PARSERS for ext in cls.EXTENSIONS)
        )

    parser = parser_cls()
    result = parser.parse(path)
    result.validate()
    return result
