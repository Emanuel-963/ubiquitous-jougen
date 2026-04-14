"""Data validation for the IonFlow Pipeline.

Every validator returns a :class:`ValidationResult` — a lightweight
container with ``ok`` (bool), ``warnings`` (list of human-readable
strings) and ``errors`` (list of hard-stop messages).

Pipeline code is expected to call the validators right after loading
data and decide whether to abort or merely log the warnings.

Public functions
----------------
* ``validate_eis_dataframe(df)``        — columns, types, ranges, NaN
* ``validate_cycling_dataframe(df)``    — ciclo, tempo, potencial, corrente
* ``validate_frequency_range(df)``      — alert if < 3 decades
* ``validate_impedance_quality(df)``    — simplified Kramers-Kronig residual
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Result container ─────────────────────────────────────────────────
@dataclass
class ValidationResult:
    """Outcome of a validation check.

    Attributes
    ----------
    ok : bool
        ``True`` when there are **no errors** (warnings are tolerated).
    warnings : list[str]
        Non-fatal issues that the user should be aware of.
    errors : list[str]
        Hard failures — the data should not be processed further.
    """

    ok: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Convenience helpers
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def merge(self, other: "ValidationResult") -> None:
        """Absorb all messages from *other* into this result."""
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        if not other.ok:
            self.ok = False

    def log_all(self) -> None:
        """Emit every message to the module logger."""
        for w in self.warnings:
            logger.warning("Validation: %s", w)
        for e in self.errors:
            logger.error("Validation: %s", e)


# ── EIS DataFrame validator ─────────────────────────────────────────
_EIS_REQUIRED_COLS: Sequence[str] = ("frequency", "zreal", "zimag")


def validate_eis_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Validate a preprocessed EIS DataFrame.

    Checks
    ------
    * Required columns: ``frequency``, ``zreal``, ``zimag``.
    * Numeric types on all three columns.
    * No NaN / Inf values.
    * ``frequency`` strictly positive.
    * At least 5 data points.
    """
    vr = ValidationResult()

    # -- existence --
    if df is None or not isinstance(df, pd.DataFrame):
        vr.add_error("Input is not a DataFrame.")
        return vr

    missing = [c for c in _EIS_REQUIRED_COLS if c not in df.columns]
    if missing:
        vr.add_error(f"Missing required columns: {missing}")
        return vr

    # -- types --
    for col in _EIS_REQUIRED_COLS:
        if not np.issubdtype(df[col].dtype, np.number):
            vr.add_error(f"Column '{col}' is not numeric (dtype={df[col].dtype}).")

    if not vr.ok:
        return vr

    # -- NaN / Inf --
    for col in _EIS_REQUIRED_COLS:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            vr.add_warning(f"Column '{col}' has {n_nan} NaN value(s).")
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            vr.add_error(f"Column '{col}' has {n_inf} Inf value(s).")

    # -- frequency > 0 --
    if (df["frequency"] <= 0).any():
        n_bad = int((df["frequency"] <= 0).sum())
        vr.add_error(f"frequency has {n_bad} non-positive value(s).")

    # -- minimum rows --
    if len(df) < 5:
        vr.add_warning(f"Only {len(df)} data point(s) — results may be unreliable.")

    return vr


# ── Cycling DataFrame validator ──────────────────────────────────────
_CYCLING_REQUIRED_COLS: Sequence[str] = ("tempo", "corrente", "potencial", "ciclo")


def validate_cycling_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Validate a cycling DataFrame.

    Checks
    ------
    * Required columns: ``tempo``, ``corrente``, ``potencial``, ``ciclo``.
    * Numeric types.
    * No NaN in essential columns.
    * ``tempo`` monotonically non-decreasing (per cycle).
    * At least 1 cycle present.
    """
    vr = ValidationResult()

    if df is None or not isinstance(df, pd.DataFrame):
        vr.add_error("Input is not a DataFrame.")
        return vr

    missing = [c for c in _CYCLING_REQUIRED_COLS if c not in df.columns]
    if missing:
        vr.add_error(f"Missing required columns: {missing}")
        return vr

    for col in _CYCLING_REQUIRED_COLS:
        if not np.issubdtype(df[col].dtype, np.number):
            vr.add_error(f"Column '{col}' is not numeric (dtype={df[col].dtype}).")

    if not vr.ok:
        return vr

    # NaN check
    for col in _CYCLING_REQUIRED_COLS:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            vr.add_warning(f"Column '{col}' has {n_nan} NaN value(s).")

    # At least 1 cycle
    n_cycles = df["ciclo"].nunique()
    if n_cycles == 0:
        vr.add_error("No cycles found in 'ciclo' column.")
    elif n_cycles < 3:
        vr.add_warning(f"Only {n_cycles} cycle(s) — statistics may be unreliable.")

    # Time monotonicity per cycle
    for cycle, grp in df.groupby("ciclo"):
        diffs = grp["tempo"].diff().dropna()
        if (diffs < 0).any():
            vr.add_warning(
                f"Cycle {cycle}: 'tempo' is not monotonically non-decreasing."
            )
            break  # one warning is enough

    return vr


# ── Frequency-range check ───────────────────────────────────────────
def validate_frequency_range(
    df: pd.DataFrame, min_decades: float = 3.0
) -> ValidationResult:
    """Warn if the frequency span is less than *min_decades* decades.

    A good EIS measurement typically covers ≥ 4–6 decades
    (e.g. 10 mHz → 100 kHz).  Less than 3 decades makes fitting
    unreliable.
    """
    vr = ValidationResult()

    if "frequency" not in df.columns:
        vr.add_error("Column 'frequency' not found.")
        return vr

    freq = df["frequency"].dropna()
    pos = freq[freq > 0]
    if pos.empty:
        vr.add_error("No positive frequency values.")
        return vr

    decades = np.log10(pos.max()) - np.log10(pos.min())
    if decades < min_decades:
        vr.add_warning(
            f"Frequency range spans only {decades:.1f} decade(s) "
            f"(minimum recommended: {min_decades})."
        )

    return vr


# ── Simplified Kramers-Kronig residual check ─────────────────────────
def validate_impedance_quality(
    df: pd.DataFrame, residual_threshold: float = 0.05
) -> ValidationResult:
    """Simplified impedance quality check via residuals.

    This is **not** a full Kramers-Kronig transform — that would require
    an external library or heavy maths.  Instead we compute:

    1. Fit Z'' vs Z' to a polynomial (degree 3).
    2. Compute the relative residuals.
    3. If > *residual_threshold* of points have |residual| > 5 % of the
       range, flag the data as noisy.

    This catches obviously noisy / corrupted spectra without heavy deps.
    """
    vr = ValidationResult()

    if "zreal" not in df.columns or "zimag" not in df.columns:
        vr.add_error("Columns 'zreal' and/or 'zimag' not found.")
        return vr

    zr = df["zreal"].dropna().values
    zi = df["zimag"].dropna().values

    n = min(len(zr), len(zi))
    if n < 10:
        vr.add_warning(
            f"Only {n} impedance points — quality check skipped."
        )
        return vr

    zr = zr[:n].astype(float)
    zi = zi[:n].astype(float)

    # Polynomial fit Z''(Z')
    try:
        coeffs = np.polyfit(zr, zi, deg=min(3, n - 1))
        zi_fit = np.polyval(coeffs, zr)
    except Exception:
        vr.add_warning("Polynomial fit failed — quality check skipped.")
        return vr

    residuals = np.abs(zi - zi_fit)
    z_range = max(np.ptp(zi), 1e-30)  # avoid division by zero
    relative_residuals = residuals / z_range

    fraction_bad = np.mean(relative_residuals > 0.05)
    if fraction_bad > residual_threshold:
        vr.add_warning(
            f"{fraction_bad:.0%} of points exceed 5 % residual — "
            f"data may be noisy or corrupted."
        )

    return vr


# ── Convenience: run all EIS validators at once ──────────────────────
def validate_eis_full(df: pd.DataFrame) -> ValidationResult:
    """Run all EIS-related validators and merge into one result."""
    vr = validate_eis_dataframe(df)
    if vr.ok:
        vr.merge(validate_frequency_range(df))
        vr.merge(validate_impedance_quality(df))
    return vr
