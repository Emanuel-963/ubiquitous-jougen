"""IonFlow Dashboard — Streamlit helper utilities.

Shared helpers used by the dashboard pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def fmt_float(v: Any, decimals: int = 4) -> str:
    """Format a value as a rounded float string, or '—' if None/NaN."""
    try:
        f = float(v)
        if f != f:  # NaN check
            return "—"
        return f"{f:.{decimals}g}"
    except (TypeError, ValueError):
        return "—"


def dataframe_or_empty(df: Optional[pd.DataFrame], msg: str = "Sem dados.") -> str:
    """Return *df* if non-empty, else the placeholder message string."""
    if df is None or df.empty:
        return msg
    return df


def friendly_size(path: str | Path) -> str:
    """Human-readable file size for the given path."""
    try:
        size = Path(path).stat().st_size
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except OSError:
        return "?"


def parse_uploaded_file(
    uploaded_file,
) -> Optional[pd.DataFrame]:
    """Try to parse a Streamlit UploadedFile into an EIS DataFrame.

    Attempts the IonFlow parser chain first (Gamry, BioLogic, Autolab,
    Zahner, generic CSV).  Falls back to pandas ``read_csv`` /
    ``read_excel`` for plain tabular files.

    Returns a DataFrame with at least ``frequency``, ``zreal``, ``zimag``
    columns, or ``None`` on failure.
    """
    import tempfile

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        from src.parsers import parse_eis_file

        result = parse_eis_file(tmp_path)
        df = result.data
    except Exception:
        # Fallback to pandas
        try:
            if suffix in (".xlsx", ".xls"):
                df = pd.read_excel(tmp_path)
            else:
                df = pd.read_csv(tmp_path, sep=None, engine="python")
        except Exception:
            return None
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    # Normalise column names
    rename: Dict[str, str] = {}
    lower = {c.lower().strip(): c for c in df.columns}
    for std in ("frequency", "zreal", "zimag"):
        candidates = [k for k in lower if std in k]
        if candidates:
            rename[lower[candidates[0]]] = std
    if rename:
        df = df.rename(columns=rename)

    required = {"frequency", "zreal", "zimag"}
    if not required.issubset(df.columns):
        return None

    # Ensure numeric types
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=list(required)).reset_index(drop=True)


def nyquist_fig(df: pd.DataFrame, title: str = ""):  # type: ignore[return]
    """Return a simple Nyquist matplotlib figure for *df*."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["zreal"], -df["zimag"], s=20, alpha=0.7, color="#1b4f72")
    ax.set_xlabel("Z' (Ω)")
    ax.set_ylabel("−Z'' (Ω)")
    ax.set_title(title or "Nyquist")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def bode_fig(df: pd.DataFrame, title: str = ""):  # type: ignore[return]
    """Return a Bode plot (|Z| and phase) for *df*."""
    import matplotlib.pyplot as plt
    import numpy as np

    freq = df["frequency"].values
    zmag = np.sqrt(df["zreal"] ** 2 + df["zimag"] ** 2).values
    phase = np.degrees(np.arctan2(-df["zimag"].values, df["zreal"].values))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    ax1.semilogx(freq, zmag, color="#1b4f72")
    ax1.set_ylabel("|Z| (Ω)")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogx(freq, phase, color="#c0392b")
    ax2.set_ylabel("Fase (°)")
    ax2.set_xlabel("Frequência (Hz)")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(title or "Bode")
    fig.tight_layout()
    return fig
