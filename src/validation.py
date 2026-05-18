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
        """Append a non-fatal warning message.

        Parameters
        ----------
        msg : str
            Human-readable warning description.
        """
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        """Append a hard-stop error and mark the result as failed.

        Parameters
        ----------
        msg : str
            Human-readable error description.
        """
        self.errors.append(msg)
        self.ok = False

    def merge(self, other: "ValidationResult") -> None:
        """Absorb all messages from *other* into this result.

        Parameters
        ----------
        other : ValidationResult
            Another validation result whose warnings and errors
            will be appended to this instance.  If *other* is not
            ``ok``, this result is also marked as failed.
        """
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        if not other.ok:
            self.ok = False

    def log_all(self) -> None:
        """Emit every message to the module logger.

        Warnings are logged at ``WARNING`` level and errors at
        ``ERROR`` level via the module-level :data:`logger`.
        """
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
        vr.add_warning(f"Only {n} impedance points — quality check skipped.")
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


# ── Metrological preprocessing helpers ───────────────────────────────


def detect_powerline_noise(
    freq: np.ndarray,
    *,
    targets: tuple[float, ...] = (50.0, 100.0),
    tolerance: float = 3.0,
) -> np.ndarray:
    """Return a boolean mask of points contaminated by power-line harmonics.

    Following Orazem & Tribollet (2017) §2.2.2: impedance values in the
    range ``target ± tolerance`` Hz should be removed before regression
    because 50/100 Hz noise from the power line is inadequately filtered
    by most FRA instruments.

    Parameters
    ----------
    freq : np.ndarray
        Frequency vector (Hz).
    targets : tuple of float
        Power-line frequencies to flag (default: 50 and 100 Hz).
    tolerance : float
        Half-width in Hz around each target (default: ±3 Hz).

    Returns
    -------
    np.ndarray
        Boolean mask, ``True`` where frequency is within a noise band.
        Pass ``~mask`` to keep only clean points.
    """
    freq = np.asarray(freq, dtype=float)
    mask = np.zeros(len(freq), dtype=bool)
    for f0 in targets:
        mask |= np.abs(freq - f0) <= tolerance
    return mask


def remove_highest_frequency_point(
    freq: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the single highest-frequency point to eliminate transient artefacts.

    When the instrument switches from DC to its first AC frequency a brief
    transient corrupts the first measured point.  Orazem & Tribollet (2026)
    §2.2.2 explicitly remove it before KK and model analysis.

    Parameters
    ----------
    freq, z : np.ndarray
        Frequency vector and complex impedance.

    Returns
    -------
    freq_clean, z_clean : np.ndarray
        Arrays with the highest-frequency point removed (sorted descending
        to highest first, so index 0 is removed).
    """
    freq = np.asarray(freq, dtype=float)
    z = np.asarray(z, dtype=complex)
    if len(freq) <= 1:
        return freq, z
    idx_max = int(np.argmax(freq))
    keep = np.ones(len(freq), dtype=bool)
    keep[idx_max] = False
    return freq[keep], z[keep]


def estimate_critical_frequency(
    re_ohm: float,
    c_inf_farad: float,
) -> float:
    """Estimate the characteristic frequency *f_c* above which the ohmic
    impedance dominates the spectrum.

    .. math::
        f_c = \\frac{1}{2\\pi R_e C_\\infty}

    Above *f_c* the ohmic drop confounds capacitive and faradaic features.
    The frequency range should be truncated to ``f ≤ f_c`` before regression
    of process models (Orazem & Tribollet, Electrochimica Acta 568, 2026).

    Parameters
    ----------
    re_ohm : float
        Ohmic (solution) resistance in Ω (or Ω·cm²).
    c_inf_farad : float
        High-frequency capacitance in F (or F/cm²) — typically obtained
        from the measurement model or the high-frequency limit of Re(C).

    Returns
    -------
    float
        Critical frequency *f_c* in Hz.
    """
    if re_ohm <= 0 or c_inf_farad <= 0:
        return np.inf
    return 1.0 / (2.0 * np.pi * re_ohm * c_inf_farad)


def truncate_above_fc(
    freq: np.ndarray,
    z: np.ndarray,
    fc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove all frequency points above the critical frequency *f_c*.

    Parameters
    ----------
    freq, z : np.ndarray
        Frequency vector (Hz) and complex impedance.
    fc : float
        Critical frequency in Hz (from :func:`estimate_critical_frequency`).

    Returns
    -------
    freq_clean, z_clean : np.ndarray
        Only the points with ``freq <= fc``.
    """
    freq = np.asarray(freq, dtype=float)
    z = np.asarray(z, dtype=complex)
    keep = freq <= fc
    if not np.any(keep):
        logger.warning(
            "truncate_above_fc: fc=%.3g Hz removed all points — returning original", fc
        )
        return freq, z
    n_removed = int(np.sum(~keep))
    if n_removed:
        logger.debug(
            "truncate_above_fc: removed %d points above fc=%.3g Hz", n_removed, fc
        )
    return freq[keep], z[keep]
