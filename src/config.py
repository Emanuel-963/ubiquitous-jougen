"""Centralised configuration for the IonFlow Pipeline.

Every magic number, path, threshold and weight that was previously scattered
across the code-base lives here in a single typed dataclass.  All pipeline
functions accept an *optional* ``PipelineConfig``; when ``None`` is passed
they fall back to ``PipelineConfig.default()``.

Persistence
-----------
``PipelineConfig.to_json(path)``   — write current config to disk
``PipelineConfig.from_json(path)`` — restore from a previously saved file
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sentinel used internally – not exported
_UNSET = object()


# ---------------------------------------------------------------------------
# Main dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Single source of truth for every tuneable parameter in IonFlow."""

    # ── Directories ──────────────────────────────────────────────────
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    output_dir: str = "outputs"
    tables_dir: str = "outputs/tables"
    figures_dir: str = "outputs/figures"
    circuits_fig_dir: str = "outputs/figures/circuits"
    analytics_fig_dir: str = "outputs/figures/analytics"
    drt_fig_dir: str = "outputs/figures/drt"
    reports_dir: str = "outputs/circuit_reports"
    excel_dir: str = "outputs/excel"
    log_dir: str = "logs"

    # ── Physics / Metrics (src/physics_metrics.py) ───────────────────
    voltage: float = 1.0
    """Default voltage (V) for energy calculation  E = ½CV²."""

    n_head: int = 5
    """Number of high/low frequency points for Rs/Rp estimation."""

    capacitance_filter_min: float = 1e-15
    """Lower bound for physically reasonable capacitance (F)."""

    capacitance_filter_max: float = 1e-2
    """Upper bound for physically reasonable capacitance (F)."""

    # ── Preprocessing (src/preprocessing.py) ─────────────────────────
    required_columns: Tuple[str, ...] = ("frequency", "zreal", "zimag")
    """Column names that every raw EIS file must contain."""

    # ── CPE + Warburg fit (src/cpe_fit.py) ───────────────────────────
    cpe_max_nfev: int = 5000
    """Maximum function evaluations for CPE+Warburg fitting."""

    # ── Circuit fitting (src/circuit_fitting.py) ─────────────────────
    circuit_max_nfev: int = 5000
    """Maximum function evaluations per circuit fit attempt."""

    circuit_multi_seed_scales: Tuple[float, ...] = (0.8, 1.2)
    """Multipliers applied to the base initial guess to create seeds."""

    circuit_shortlist_top_n: int = 3
    """How many candidate circuits the shortlist selects."""

    bic_penalty_structured: float = 5.0
    """BIC penalty added when residuals show autocorrelation structure."""

    bic_penalty_per_bound_hit: float = 0.5
    """BIC penalty added for each parameter that hit its bound."""

    residual_autocorr_threshold: float = 0.3
    """Lag-1 autocorrelation threshold above which residuals are 'structured'."""

    bound_proximity_fraction: float = 0.01
    """Fraction of param range within which a value counts as 'hitting' its bound."""

    # ── Classification / Ranking (src/ranking.py) ────────────────────
    kmeans_n_clusters: int = 2
    """Number of K-Means clusters for sample classification."""

    kmeans_min_rows: int = 3
    """Minimum number of samples required for K-Means (else fallback)."""

    kmeans_variance_threshold: float = 1e-12
    """Minimum total variance in (Rs, Rp) to proceed with clustering."""

    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "Rp_fit": 0.35,
        "Rs_fit": -0.25,
        "C_mean": 0.25,
        "Energy_mean": 0.15,
    })
    """Weights for the composite ranking score (sign encodes direction)."""

    # ── PCA (src/pca_analysis.py) ────────────────────────────────────
    pca_columns: List[str] = field(default_factory=lambda: [
        "Rs_fit", "Rp_fit", "Q", "n", "Sigma",
    ])
    """Columns used as input to PCA."""

    pca_min_rows: int = 3
    """Minimum samples for PCA to be meaningful."""

    # ── Stability (src/stability.py) ─────────────────────────────────
    stability_columns: List[str] = field(default_factory=lambda: [
        "Rs_fit", "Rp_fit", "Q", "n",
    ])
    """Parameters evaluated for inter-replica stability (CV)."""

    # ── Correlation (src/visualization.py) ───────────────────────────
    correlation_columns: List[str] = field(default_factory=lambda: [
        "Rs_fit", "Rp_fit", "Q", "n", "Sigma",
        "C_mean", "C_lowfreq", "Energy_mean",
        "Tau", "Dispersion", "Score", "Rank",
    ])
    """Columns included in the Spearman correlation heatmap."""

    # ── DRT (src/drt_analysis.py) ────────────────────────────────────
    drt_lambda: float = 1e-3
    """Tikhonov regularisation parameter λ."""

    drt_n_taus: int = 50
    """Number of log-uniform τ discretisation points."""

    drt_tau_min: float = 1e-7
    """Lower bound of the τ grid (s)."""

    drt_tau_max: float = 1e2
    """Upper bound of the τ grid (s)."""

    drt_max_peaks_exported: int = 3
    """Maximum number of DRT peaks exported per sample in the table."""

    # ── Cycling (main_cycling.py) ────────────────────────────────────
    scan_rate: float = 1.0
    """Default galvanostatic scan rate (A/g)."""

    electrode_mass: Optional[float] = None
    """Electrode active mass (g).  None = normalise per-gram by default."""

    electrode_area: Optional[float] = None
    """Electrode geometric area (cm²).  None = not used."""

    # ── GUI / Display ────────────────────────────────────────────────
    language: str = "pt"
    """Interface language ('pt', 'en', 'es')."""

    theme: str = "blue"
    """CustomTkinter colour theme."""

    appearance: str = "dark"
    """CustomTkinter appearance mode ('dark', 'light', 'system')."""

    dpi_screen: int = 100
    """DPI for on-screen matplotlib figures."""

    dpi_save: int = 160
    """DPI for saved PNG figures."""

    dpi_diagnostics: int = 300
    """DPI for circuit diagnostic plots."""

    # ── Misc ─────────────────────────────────────────────────────────
    settings_filename: str = "ionflow_settings.json"
    """Filename for GUI settings persistence."""

    # ==================================================================
    # Factory & I/O
    # ==================================================================

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Return a config with all default values."""
        return cls()

    # ── JSON persistence ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe types)."""
        d = asdict(self)
        # Convert tuples to lists for JSON round-trip
        for key, value in d.items():
            if isinstance(value, tuple):
                d[key] = list(value)
        return d

    def to_json(self, path: str | Path) -> None:
        """Write config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Config saved → %s", path)

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        """Load config from a JSON file.

        Unknown keys are silently ignored (forward-compat).
        Missing keys use the default value.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)

        # Filter to known fields only
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for key, value in raw.items():
            if key not in known_fields:
                logger.warning("Config: unknown key '%s' ignored", key)
                continue
            # Restore tuples where the field type expects them
            field_obj = cls.__dataclass_fields__[key]
            if "Tuple" in str(field_obj.type) and isinstance(value, list):
                value = tuple(value)
            filtered[key] = value

        return cls(**filtered)

    @classmethod
    def from_json_safe(cls, path: str | Path) -> "PipelineConfig":
        """Load config from JSON, falling back to defaults on any error."""
        try:
            return cls.from_json(path)
        except Exception as exc:
            logger.warning("Failed to load config from %s: %s — using defaults", path, exc)
            return cls.default()

    # ── Helpers ───────────────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create all output directories defined in this config."""
        for attr in (
            "tables_dir",
            "figures_dir",
            "circuits_fig_dir",
            "analytics_fig_dir",
            "drt_fig_dir",
            "reports_dir",
            "excel_dir",
            "log_dir",
        ):
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)

    @property
    def capacitance_filter_range(self) -> Tuple[float, float]:
        """Convenience accessor for the capacitance filter bounds."""
        return (self.capacitance_filter_min, self.capacitance_filter_max)
