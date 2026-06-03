"""Journal-specific matplotlib style presets for publication-quality figures.

VIZ-04: Provides ready-to-use style dictionaries for major scientific
publishers (ACS, RSC, Elsevier, Nature/Springer, and a generic IonFlow
style). Researchers can apply these with a single function call before
generating plots.

Usage
-----
    from src.journal_styles import apply_journal_style, JOURNAL_STYLES

    apply_journal_style("acs")    # Apply ACS style globally
    # ... generate plots ...

    # Or use as context manager:
    with journal_style_context("nature"):
        fig, ax = plt.subplots()
        ...
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Generator, List

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Style definitions
# ═══════════════════════════════════════════════════════════════════════

_BASE_STYLE: Dict[str, Any] = {
    "font.family": "sans-serif",
    "mathtext.default": "regular",
    "axes.linewidth": 0.8,
    "axes.labelpad": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

JOURNAL_STYLES: Dict[str, Dict[str, Any]] = {
    "acs": {
        **_BASE_STYLE,
        "figure.figsize": (3.25, 2.5),  # ACS single-column width: 3.25 in
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "legend.fontsize": 7,
    },
    "rsc": {
        **_BASE_STYLE,
        "figure.figsize": (3.5, 2.7),  # RSC single-column: ~8.5 cm
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "legend.fontsize": 7,
    },
    "elsevier": {
        **_BASE_STYLE,
        "figure.figsize": (3.54, 2.75),  # Elsevier single-column: 90 mm
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.sans-serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.family": "serif",
        "legend.fontsize": 8,
    },
    "nature": {
        **_BASE_STYLE,
        "figure.figsize": (3.5, 2.6),  # Nature single-column: 89 mm
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
    },
    "ionflow": {
        **_BASE_STYLE,
        "figure.figsize": (5.0, 3.8),
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    },
}
"""Available journal style presets."""

STYLE_LABELS: Dict[str, str] = {
    "acs": "ACS (J. Am. Chem. Soc., ACS Energy Lett., etc.)",
    "rsc": "RSC (J. Mater. Chem., Energy Environ. Sci., etc.)",
    "elsevier": "Elsevier (Electrochimica Acta, J. Power Sources, etc.)",
    "nature": "Nature / Springer (Nature Energy, Sci. Reports, etc.)",
    "ionflow": "IonFlow (estilo padrão para tela / apresentações)",
}
"""Human-readable labels for each journal style."""


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def get_available_styles() -> List[str]:
    """Return list of available journal style names.

    Returns
    -------
    List[str]
        Style identifiers that can be passed to ``apply_journal_style``.
    """
    return list(JOURNAL_STYLES.keys())


def apply_journal_style(style_name: str) -> None:
    """Apply a journal style globally to matplotlib.

    Parameters
    ----------
    style_name : str
        One of the keys in ``JOURNAL_STYLES`` (e.g. ``'acs'``, ``'nature'``).

    Raises
    ------
    ValueError
        If *style_name* is not recognised.
    """
    import matplotlib.pyplot as plt

    if style_name not in JOURNAL_STYLES:
        available = ", ".join(JOURNAL_STYLES.keys())
        raise ValueError(
            f"Estilo '{style_name}' não reconhecido. Disponíveis: {available}"
        )
    plt.rcParams.update(JOURNAL_STYLES[style_name])
    logger.info("Applied journal style: %s", style_name)


@contextlib.contextmanager
def journal_style_context(style_name: str) -> Generator[None, None, None]:
    """Context manager that temporarily applies a journal style.

    Restores previous rcParams on exit.

    Parameters
    ----------
    style_name : str
        Style identifier.

    Yields
    ------
    None

    Example
    -------
    >>> with journal_style_context("nature"):
    ...     fig, ax = plt.subplots()
    ...     ax.plot(x, y)
    ...     fig.savefig("plot.svg")
    """
    import matplotlib.pyplot as plt

    if style_name not in JOURNAL_STYLES:
        available = ", ".join(JOURNAL_STYLES.keys())
        raise ValueError(
            f"Estilo '{style_name}' não reconhecido. Disponíveis: {available}"
        )

    old_params = {k: plt.rcParams.get(k) for k in JOURNAL_STYLES[style_name]}
    plt.rcParams.update(JOURNAL_STYLES[style_name])
    try:
        yield
    finally:
        # Restore only keys we changed
        for k, v in old_params.items():
            if v is not None:
                plt.rcParams[k] = v
