"""MVC Model — pure data containers for the IonFlow GUI.

``AppState`` is the single source of truth for every DataFrame, config
value and runtime flag that the application tracks.  It is the *Model*
layer: **no UI, no I/O, no threads**.  The ``PipelineController`` mutates
it; the ``MainWindow`` reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


# ── Plot gallery item ───────────────────────────────────────────────────
@dataclass
class PlotItem:
    """Represents a generated plot thumbnail in the gallery."""

    title: str
    path: str


# ── DRT analysis presets ────────────────────────────────────────────────
DRT_PRESETS: Dict[str, Dict[str, str]] = {
    "Rápido": {"lambda_reg": "5e-3", "n_taus": "30"},
    "Balanceado": {"lambda_reg": "1e-3", "n_taus": "50"},
    "Alta resolução": {"lambda_reg": "5e-4", "n_taus": "80"},
}

DRT_DEFAULT_PRESET: str = "Balanceado"


# ── Application state ──────────────────────────────────────────────────
@dataclass
class AppState:
    """Application state — holds every DataFrame and config value.

    This is the *Model* layer: no UI, no I/O, no threads.
    The ``PipelineController`` mutates it; the ``MainWindow`` reads it.
    """

    # ── EIS pipeline ────────────────────────────────────────────
    eis_df: Optional[pd.DataFrame] = None
    rank_df: Optional[pd.DataFrame] = None
    df_pca: Optional[pd.DataFrame] = None
    circuit_df: Optional[pd.DataFrame] = None
    raw_eis: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # ── Cycling pipeline ────────────────────────────────────────
    cic_df: Optional[pd.DataFrame] = None
    cic_results: Dict[str, pd.DataFrame] = field(default_factory=dict)
    cic_plot_map: Dict[str, str] = field(default_factory=dict)

    # ── DRT pipeline ────────────────────────────────────────────
    drt_df: Optional[pd.DataFrame] = None
    drt_peaks_df: Optional[pd.DataFrame] = None
    drt_summary_df: Optional[pd.DataFrame] = None
    drt_eis_df: Optional[pd.DataFrame] = None
    drt_results: Dict[str, dict] = field(default_factory=dict)
    drt_plot_map: Dict[str, str] = field(default_factory=dict)

    # ── Plots gallery ───────────────────────────────────────────
    plot_items: List[PlotItem] = field(default_factory=list)

    # ── GUI preferences ─────────────────────────────────────────
    gui_settings: Dict[str, Any] = field(default_factory=dict)
    drt_ui_prefs: Dict[str, str] = field(
        default_factory=lambda: {
            "sample": "",
            "mode": "Espectro",
            "overlay_text": "",
        }
    )

    # ── Runtime status ──────────────────────────────────────────
    status: str = "pronto"
    progress_text: str = "Pronto"
    is_running: bool = False

    # ── Convenience queries ─────────────────────────────────────

    def has_eis_data(self) -> bool:
        """Return *True* if EIS results are loaded."""
        return self.eis_df is not None

    def has_cycling_data(self) -> bool:
        """Return *True* if cycling results are loaded."""
        return self.cic_df is not None

    def has_drt_data(self) -> bool:
        """Return *True* if DRT results are loaded."""
        return self.drt_df is not None

    # ── Reset helpers ───────────────────────────────────────────

    def clear_eis(self) -> None:
        """Reset all EIS-related fields."""
        self.eis_df = None
        self.rank_df = None
        self.df_pca = None
        self.circuit_df = None
        self.raw_eis = {}

    def clear_cycling(self) -> None:
        """Reset all cycling-related fields."""
        self.cic_df = None
        self.cic_results = {}
        self.cic_plot_map = {}

    def clear_drt(self) -> None:
        """Reset all DRT-related fields."""
        self.drt_df = None
        self.drt_peaks_df = None
        self.drt_summary_df = None
        self.drt_eis_df = None
        self.drt_results = {}
        self.drt_plot_map = {}

    def clear_plots(self) -> None:
        """Remove all plot gallery items."""
        self.plot_items.clear()

    def clear_all(self) -> None:
        """Reset the entire application state."""
        self.clear_eis()
        self.clear_cycling()
        self.clear_drt()
        self.clear_plots()

    # ── Serialisation ───────────────────────────────────────────

    def to_summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of the current state."""
        return {
            "has_eis": self.has_eis_data(),
            "has_cycling": self.has_cycling_data(),
            "has_drt": self.has_drt_data(),
            "n_plots": len(self.plot_items),
            "n_raw_eis": len(self.raw_eis),
            "n_cic_results": len(self.cic_results),
            "n_drt_results": len(self.drt_results),
            "status": self.status,
            "is_running": self.is_running,
        }
