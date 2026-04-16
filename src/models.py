"""Typed result objects returned by each IonFlow pipeline.

Every pipeline entry-point returns one of the dataclasses defined here instead
of an untyped ``dict``.  Each result carries **exactly** the fields that the
GUI and downstream code consume, plus a ``config_used`` reference for
provenance.

Backward compatibility
----------------------
All result classes expose a ``to_dict()`` helper and support ``result["key"]``
bracket access via ``__getitem__`` so that old code using the dict interface
continues to work without changes during the migration period.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import PipelineConfig


# ---------------------------------------------------------------------------
# Mixin for dict-like access (migration aid)
# ---------------------------------------------------------------------------

class _DictAccessMixin:
    """Allow ``result["key"]``, ``result.get("key")`` and ``result.keys()`` for migration."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if it does not exist.

        Mimics ``dict.get()`` for backward-compatible bracket access.

        Parameters
        ----------
        key : str
            Attribute / field name to look up.
        default : Any, optional
            Value returned when *key* is not found (default ``None``).

        Returns
        -------
        Any
            Field value or *default*.
        """
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        """Return field names, mimicking ``dict.keys()``.

        Returns
        -------
        List[str]
            Names of all dataclass fields.
        """
        return [f.name for f in self.__dataclass_fields__.values()]


# ---------------------------------------------------------------------------
# PCA result (nested inside EISResult)
# ---------------------------------------------------------------------------

@dataclass
class PCAResult(_DictAccessMixin):
    """Result of a PCA computation."""

    df_pca: Optional[pd.DataFrame] = None
    """PC score matrix (rows=samples, cols=PC1, PC2, …)."""

    loadings: Optional[pd.DataFrame] = None
    """PCA loadings (original features × PCs)."""

    evr: Optional[pd.Series] = None
    """Explained variance ratio per component."""

    figure_paths: List[str] = field(default_factory=list)
    """Paths to PCA figures (2D, 3D, scree, biplot, metric)."""


# ---------------------------------------------------------------------------
# EIS pipeline result
# ---------------------------------------------------------------------------

@dataclass
class EISResult(_DictAccessMixin):
    """Typed result of ``run_eis_pipeline()``."""

    # ── Core DataFrames ──────────────────────────────────────────────
    features_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Per-file features before ranking (index = filename)."""

    ranked_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Features after classification + ranking (Score, Rank columns)."""

    cap_energy_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Capacitance / energy / retention table."""

    circuit_table: Optional[pd.DataFrame] = None
    """Per-file circuit fitting results (None if no circuits fitted)."""

    circuit_summary: Optional[pd.DataFrame] = None
    """Aggregated circuit statistics grouped by circuit name."""

    # ── PCA ──────────────────────────────────────────────────────────
    pca: PCAResult = field(default_factory=PCAResult)
    """PCA scores, loadings, explained variance and figure paths."""

    # ── Stability ────────────────────────────────────────────────────
    stability: Dict[str, pd.DataFrame] = field(default_factory=dict)
    """``{parameter_name: stability_df}`` — CV per Sample for each column."""

    # ── Raw EIS data ─────────────────────────────────────────────────
    raw_eis: Dict[str, pd.DataFrame] = field(default_factory=dict)
    """``{filename: preprocessed_eis_df}`` — kept for interactive plots."""

    # ── Provenance ───────────────────────────────────────────────────
    out_dir: str = "outputs/tables"
    """Directory where CSVs were written."""

    config_used: PipelineConfig = field(default_factory=PipelineConfig.default)
    """Configuration snapshot at time of execution."""

    # ── Legacy dict aliases (read-only properties for migration) ─────
    @property
    def df(self) -> pd.DataFrame:
        """Legacy alias: ``result["df"]`` → ``result.features_df``.

        Returns
        -------
        pd.DataFrame
            Per-file features before ranking.
        """
        return self.features_df

    @property
    def df_ranked(self) -> pd.DataFrame:
        """Legacy alias: ``result["df_ranked"]`` → ``result.ranked_df``.

        Returns
        -------
        pd.DataFrame
            Features after classification and ranking.
        """
        return self.ranked_df

    @property
    def cap_energy(self) -> pd.DataFrame:
        """Legacy alias: ``result["cap_energy"]`` → ``result.cap_energy_df``.

        Returns
        -------
        pd.DataFrame
            Capacitance, energy and retention table.
        """
        return self.cap_energy_df

    @property
    def df_pca(self) -> Optional[pd.DataFrame]:
        """Legacy alias: ``result["df_pca"]`` → ``result.pca.df_pca``.

        Returns
        -------
        pd.DataFrame or None
            PCA score matrix, or ``None`` if PCA was not computed.
        """
        return self.pca.df_pca

    @property
    def pca_loadings(self) -> Optional[pd.DataFrame]:
        """Legacy alias."""
        return self.pca.loadings

    @property
    def pca_evr(self) -> Optional[pd.Series]:
        """Legacy alias."""
        return self.pca.evr

    @property
    def pca_paths(self) -> List[str]:
        """Legacy alias."""
        return self.pca.figure_paths

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict matching the legacy return contract.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys expected by legacy callers
            (``df``, ``df_ranked``, ``cap_energy``, etc.).
        """
        return {
            "df": self.features_df,
            "df_ranked": self.ranked_df,
            "cap_energy": self.cap_energy_df,
            "df_pca": self.pca.df_pca,
            "pca_loadings": self.pca.loadings,
            "pca_evr": self.pca.evr,
            "pca_paths": self.pca.figure_paths,
            "out_dir": self.out_dir,
            "circuit_table": self.circuit_table,
            "raw_eis": self.raw_eis,
        }


# ---------------------------------------------------------------------------
# Cycling pipeline result
# ---------------------------------------------------------------------------

@dataclass
class CyclingResult(_DictAccessMixin):
    """Typed result of ``run_ciclagem_pipeline()``."""

    results: Dict[str, pd.DataFrame] = field(default_factory=dict)
    """``{filename: per-cycle energy/power DataFrame}``."""

    export_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    """``{filename: export DataFrame with units row}``."""

    merged_table: Optional[pd.DataFrame] = None
    """All files merged into one DataFrame (``None`` if empty)."""

    plot_paths: List[Tuple[str, str]] = field(default_factory=list)
    """``[(filename, abs_path_to_png)]`` — time-vs-potential plots."""

    energy_power_paths: List[Tuple[str, str]] = field(default_factory=list)
    """``[(filename, abs_path_to_png)]`` — energy/power vs cycle plots."""

    config_used: PipelineConfig = field(default_factory=PipelineConfig.default)
    """Configuration snapshot."""

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict matching the legacy return contract."""
        return {
            "results": self.results,
            "export_tables": self.export_tables,
            "merged_table": self.merged_table,
            "plot_paths": self.plot_paths,
            "energy_power_paths": self.energy_power_paths,
        }


# ---------------------------------------------------------------------------
# DRT pipeline result
# ---------------------------------------------------------------------------

@dataclass
class DRTPipelineResult(_DictAccessMixin):
    """Typed result of ``run_drt_pipeline()``.

    Note: named ``DRTPipelineResult`` to avoid collision with
    ``src.drt_analysis.DRTResult`` (the per-file TypedDict).
    """

    drt_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    """One row per file — up to N peak columns."""

    drt_peaks_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    """One row per peak across all files."""

    drt_summary_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Per-sample aggregated metrics."""

    per_file_results: Dict[str, Any] = field(default_factory=dict)
    """``{stem: DRTResult}`` — raw arrays/peaks for programmatic use."""

    plot_paths: List[Tuple[str, str]] = field(default_factory=list)
    """``[(stem, abs_path_to_png)]``."""

    errors: Dict[str, str] = field(default_factory=dict)
    """``{filename: error_message}`` for files that failed."""

    run_meta: Dict[str, Any] = field(default_factory=dict)
    """Execution metadata (λ, n_τ, counts, timestamp)."""

    config_used: PipelineConfig = field(default_factory=PipelineConfig.default)
    """Configuration snapshot."""

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict matching the legacy return contract."""
        return {
            "drt_table": self.drt_table,
            "drt_peaks_table": self.drt_peaks_table,
            "drt_summary_table": self.drt_summary_table,
            "per_file_results": self.per_file_results,
            "plot_paths": self.plot_paths,
            "errors": self.errors,
            "run_meta": self.run_meta,
        }
