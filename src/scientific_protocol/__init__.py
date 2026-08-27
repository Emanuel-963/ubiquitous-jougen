"""Independent scientific protocol for multicriteria electrochemical analysis."""

from .config import DEFAULT_CRITERIA, Criterion
from .protocol import run_scientific_protocol
from .ranking import normalize_series, rank_multicriteria

__all__ = [
    "Criterion",
    "DEFAULT_CRITERIA",
    "normalize_series",
    "rank_multicriteria",
    "run_scientific_protocol",
]
