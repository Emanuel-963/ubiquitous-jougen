"""
IonFlow Pipeline — Headless SDK
================================
Clean programmatic API for embedding IonFlow analytics in third-party
software (Manufacturer SDK / OEM tier).

This module intentionally has **no GUI dependencies** — it only imports
from the analytical core so it can be used in scripts, notebooks, REST
servers, and C extension wrappers.

Typical usage
-------------
>>> from src.sdk import EISAnalyzer, DRTAnalyzer, ReportBuilder
>>>
>>> analyzer = EISAnalyzer(license_key="IONFLOW-OEM-ABCD1234-XXXXXXXXXX")
>>> result   = analyzer.analyze(freq, zreal, zimag)
>>>
>>> drt = DRTAnalyzer()
>>> drt_result = drt.analyze(freq, zreal, zimag)
>>>
>>> report = ReportBuilder(logo="logo.png", institution="Acme Labs")
>>> report.from_result(result)
>>> report.save("analysis.pdf")

License tiers
-------------
- No key / free : single-file analysis only (batch raises LicenseLimitError)
- Pro key        : unlimited single-file analysis, branded reports
- Lab key        : same as Pro + multi-user context (up to 5 seats)
- OEM key        : all features, white-label branding, no seat limit
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import __version__
from src.circuit_fitting import fit_circuit
from src.circuit_registry import get_circuit_names
from src.drt_analysis import run_drt
from src.kramers_kronig import kramers_kronig_test
from src.license_manager import LicenseLimitError, LicenseManager  # noqa: F401
from src.loader import load_eis_file
from src.report_generator import generate_report

# ---------------------------------------------------------------------------
# Public result containers
# ---------------------------------------------------------------------------


@dataclass
class EISResult:
    """Container for EIS pipeline results returned by :class:`EISAnalyzer`."""

    frequency: np.ndarray
    zreal: np.ndarray
    zimag: np.ndarray
    circuit_name: str = ""
    params: dict = field(default_factory=dict)
    kk_score: float = float("nan")
    kk_passed: bool = False
    source_file: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def impedance(self) -> np.ndarray:
        """Complex impedance array Z = Zreal + j*Zimag."""
        return self.zreal + 1j * self.zimag

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with columns frequency, Zreal, Zimag."""
        return pd.DataFrame(
            {"frequency": self.frequency, "Zreal": self.zreal, "Zimag": self.zimag}
        )


@dataclass
class DRTResult:
    """Container for DRT inversion results returned by :class:`DRTAnalyzer`."""

    tau: np.ndarray
    gamma: np.ndarray
    peaks: pd.DataFrame = field(default_factory=pd.DataFrame)
    lambda_reg: float = float("nan")
    n_taus: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Return DRT spectrum as a DataFrame (tau, gamma)."""
        return pd.DataFrame({"tau": self.tau, "gamma": self.gamma})


# ---------------------------------------------------------------------------
# EISAnalyzer
# ---------------------------------------------------------------------------


class EISAnalyzer:
    """
    High-level EIS analysis API.

    Parameters
    ----------
    license_key : str | None
        Optional IonFlow license key.  When not provided the free-tier limit
        of 5 files per session applies.
    language : str
        Output language for messages and reports (``"pt"``, ``"en"``, ``"es"``).
    """

    def __init__(
        self,
        license_key: Optional[str] = None,
        language: str = "en",
    ) -> None:
        from src.i18n import set_language

        set_language(language)
        self._mgr = LicenseManager.get()
        if license_key:
            self._mgr.activate(license_key)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        freq: "np.ndarray | list",
        zreal: "np.ndarray | list",
        zimag: "np.ndarray | list",
        circuit: Optional[str] = None,
        source_file: str = "",
    ) -> EISResult:
        """
        Analyze a single EIS spectrum.

        Parameters
        ----------
        freq : array-like
            Frequency values in Hz.
        zreal : array-like
            Real part of impedance (Ω).
        zimag : array-like
            Imaginary part of impedance (Ω).  Convention: positive values
            for capacitive systems (i.e. ``-Z''``).
        circuit : str | None
            Circuit model name (e.g. ``"R-RC"``).  When ``None`` the best
            circuit is chosen automatically.
        source_file : str
            Optional label for the source data file.

        Returns
        -------
        EISResult
        """
        freq = np.asarray(freq, dtype=float)
        zreal = np.asarray(zreal, dtype=float)
        zimag = np.asarray(zimag, dtype=float)

        # Kramers-Kronig validation
        kk_passed, kk_score = False, float("nan")
        try:
            kk_result = kramers_kronig_test(freq, zreal, zimag)
            kk_passed = bool(kk_result.get("passed", False))
            kk_score = float(kk_result.get("score", float("nan")))
        except Exception:
            pass

        # Circuit fitting
        params: dict = {}
        circuit_name = circuit or ""
        try:
            fit = fit_circuit(
                freq,
                zreal,
                zimag,
                circuit_name=circuit_name or None,
            )
            params = fit.get("params", {})
            circuit_name = fit.get("circuit_name", circuit_name)
        except Exception:
            pass

        return EISResult(
            frequency=freq,
            zreal=zreal,
            zimag=zimag,
            circuit_name=circuit_name,
            params=params,
            kk_score=kk_score,
            kk_passed=kk_passed,
            source_file=source_file,
        )

    def analyze_file(self, path: "str | Path") -> EISResult:
        """
        Load *path* and run :meth:`analyze`.

        Supports all formats recognised by the IonFlow parser registry
        (Gamry, BioLogic, Zahner, Autolab, Solartron, generic CSV/TSV).
        """
        parsed = load_eis_file(str(path))
        df = parsed.data
        return self.analyze(
            freq=df["frequency"].values,
            zreal=df["Zreal"].values,
            zimag=df["Zimag"].values,
            source_file=os.path.basename(str(path)),
        )

    def analyze_batch(
        self,
        paths: "list[str | Path]",
        circuit: Optional[str] = None,
    ) -> "list[EISResult]":
        """
        Analyze a list of files.

        Raises :class:`LicenseLimitError` when the free-tier file limit is
        exceeded without an active Pro/Lab/OEM license.
        """
        self._mgr.check_file_limit(len(paths))
        return [self.analyze_file(p) for p in paths]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def available_circuits(self) -> list[str]:
        """Return the list of circuit model names supported by the fitting engine."""
        return get_circuit_names()

    @property
    def tier(self) -> str:
        """Current license tier: ``"free"``, ``"pro"``, ``"lab"``, or ``"oem"``."""
        return self._mgr.tier


# ---------------------------------------------------------------------------
# DRTAnalyzer
# ---------------------------------------------------------------------------


class DRTAnalyzer:
    """
    Distribution of Relaxation Times (DRT) analysis API.

    Parameters
    ----------
    lambda_reg : float
        Tikhonov regularisation parameter (default 1e-3).
    n_taus : int
        Number of time constants in the DRT grid (default 50).
    """

    def __init__(self, lambda_reg: float = 1e-3, n_taus: int = 50) -> None:
        self.lambda_reg = lambda_reg
        self.n_taus = n_taus

    def analyze(
        self,
        freq: "np.ndarray | list",
        zreal: "np.ndarray | list",
        zimag: "np.ndarray | list",
    ) -> DRTResult:
        """
        Compute DRT spectrum from an EIS dataset.

        Parameters
        ----------
        freq, zreal, zimag : array-like
            Same conventions as :meth:`EISAnalyzer.analyze`.

        Returns
        -------
        DRTResult
        """
        freq = np.asarray(freq, dtype=float)
        zreal = np.asarray(zreal, dtype=float)
        zimag = np.asarray(zimag, dtype=float)

        raw = run_drt(
            freq,
            zreal,
            zimag,
            lambda_reg=self.lambda_reg,
            n_taus=self.n_taus,
        )
        tau = np.asarray(raw.get("tau", []))
        gamma = np.asarray(raw.get("gamma", []))
        peaks_df = raw.get("peaks", pd.DataFrame())
        if not isinstance(peaks_df, pd.DataFrame):
            peaks_df = pd.DataFrame()

        return DRTResult(
            tau=tau,
            gamma=gamma,
            peaks=peaks_df,
            lambda_reg=self.lambda_reg,
            n_taus=self.n_taus,
        )


# ---------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------


class ReportBuilder:
    """
    Thin wrapper around :func:`src.report_generator.generate_report`.

    Parameters
    ----------
    logo : str | Path | None
        Path to an image file to use as the report logo.
    institution : str
        Institution name shown in the report header.
    author : str
        Author name shown in the report header.
    language : str
        Report output language (``"pt"``, ``"en"``, ``"es"``).
    """

    def __init__(
        self,
        logo: "Optional[str | Path]" = None,
        institution: str = "",
        author: str = "",
        language: str = "en",
    ) -> None:
        self.logo = str(logo) if logo else None
        self.institution = institution
        self.author = author
        self.language = language
        self._eis_result: Optional[EISResult] = None
        self._drt_result: Optional[DRTResult] = None

    def from_result(
        self,
        eis_result: EISResult,
        drt_result: Optional[DRTResult] = None,
    ) -> "ReportBuilder":
        """Attach analysis results to include in the report."""
        self._eis_result = eis_result
        self._drt_result = drt_result
        return self

    def save(self, path: "str | Path") -> None:
        """
        Generate the PDF report and write it to *path*.

        Parameters
        ----------
        path : str | Path
            Destination PDF file path.
        """
        from src.i18n import set_language

        set_language(self.language)

        eis_df = self._eis_result.to_dataframe() if self._eis_result else None
        drt_df = self._drt_result.to_dataframe() if self._drt_result else None

        generate_report(
            output_path=str(path),
            eis_df=eis_df,
            drt_df=drt_df,
            logo_path=self.logo,
            institution=self.institution,
            author=self.author,
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def version() -> str:
    """Return the IonFlow Pipeline version string."""
    return __version__
