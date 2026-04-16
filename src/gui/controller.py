"""MVC Controller — pipeline orchestration and event bus.

``PipelineController`` sits between the *Model* (``AppState``) and the
*View* (``MainWindow``).  It owns **no widgets** — it manipulates
``AppState`` and emits named events that any view can subscribe to.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.gui.models import (
    AppState,
    DRT_DEFAULT_PRESET,
    DRT_PRESETS,
    PlotItem,
)


class PipelineController:
    """Orchestrates pipeline execution and result processing.

    Events emitted
    ──────────────
    ``status_changed``    *(status_text: str)*
    ``log``               *(message: str)*
    ``progress_start``    *(text: str)*
    ``progress_update``   *(text: str)*
    ``progress_stop``     *(text: str)*
    ``eis_completed``     *(success: bool)*
    ``cycling_completed`` *(success: bool)*
    ``both_completed``    *(success: bool)*
    ``drt_completed``     *(success: bool)*
    ``plots_added``       *(items: List[PlotItem])*
    ``table_updated``     *(table_key: str)*
    ``buttons_disable``
    ``buttons_enable``
    ``state_reset``
    """

    def __init__(
        self,
        state: Optional[AppState] = None,
        *,
        settings_path: str = "",
    ) -> None:
        self.state: AppState = state if state is not None else AppState()
        self.settings_path: str = settings_path
        self._listeners: Dict[str, List[Callable[..., Any]]] = {}

    # ── Event bus ───────────────────────────────────────────────

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Register *callback* to be invoked whenever *event* is emitted.

        Multiple callbacks may be registered for the same event; they are
        called in registration order.
        """
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        """Remove *callback* from the listener list for *event*.

        Does nothing if *callback* was never registered for *event*.
        """
        cbs = self._listeners.get(event, [])
        if callback in cbs:
            cbs.remove(callback)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire all callbacks registered for *event*, forwarding *args* and *kwargs*.

        Listeners are invoked synchronously in registration order.  A copy
        of the listener list is iterated so that callbacks may safely
        unregister themselves during dispatch.
        """
        for cb in list(self._listeners.get(event, [])):
            cb(*args, **kwargs)

    def listener_count(self, event: str) -> int:
        """Return number of listeners currently registered for *event*."""
        return len(self._listeners.get(event, []))

    # ── Pipeline result processors ─────────────────────────────

    def process_eis_result(self, result: Optional[dict]) -> bool:
        """Ingest raw EIS pipeline output into *AppState*.

        Returns ``True`` on success, ``False`` on failure.
        """
        if result is None:
            self.state.status = "erro no EIS"
            self.state.is_running = False
            self.emit("status_changed", self.state.status)
            self.emit("progress_stop", "Erro")
            self.emit("buttons_enable")
            self.emit("eis_completed", False)
            return False

        cap_df = result.get("cap_energy")
        if cap_df is not None:
            self.state.eis_df = cap_df.reset_index().rename(
                columns={"index": "Arquivo"}
            )
        else:
            self.state.eis_df = None

        self.state.rank_df = result.get("df_ranked")
        self.state.df_pca = result.get("df_pca")
        self.state.raw_eis = result.get("raw_eis") or {}
        self.state.circuit_df = result.get("circuit_table")

        new_plots: List[PlotItem] = []
        for path in result.get("pca_paths", []):
            title = "PCA" if "pca" in path.lower() else "Gráfico"
            new_plots.append(PlotItem(title=title, path=path))
        self.state.plot_items.extend(new_plots)

        self.state.status = "EIS concluído"
        self.state.is_running = False

        self.emit("table_updated", "eis")
        self.emit("table_updated", "circuit")
        if new_plots:
            self.emit("plots_added", new_plots)
        self.emit("status_changed", self.state.status)
        self.emit("progress_stop", "EIS concluído")
        self.emit("buttons_enable")
        self.emit("eis_completed", True)
        return True

    def process_cycling_result(self, result: Optional[dict]) -> bool:
        """Ingest raw cycling pipeline output into *AppState*."""
        if result is None:
            self.state.status = "erro na Ciclagem"
            self.state.is_running = False
            self.emit("status_changed", self.state.status)
            self.emit("progress_stop", "Erro")
            self.emit("buttons_enable")
            self.emit("cycling_completed", False)
            return False

        self.state.cic_df = result.get("merged_table")
        self.state.cic_results = result.get("results") or {}
        self.state.cic_plot_map = {}

        new_plots: List[PlotItem] = []
        for filename, path in result.get("plot_paths", []):
            key = self.normalize_sample_name(filename)
            self.state.cic_plot_map[key] = os.path.abspath(path)
            new_plots.append(
                PlotItem(title=f"{filename} - Integral", path=path)
            )
        for filename, path in result.get("energy_power_paths", []):
            new_plots.append(
                PlotItem(title=f"{filename} - Energia×Potência", path=path)
            )
        self.state.plot_items.extend(new_plots)

        n_files = len(self.state.cic_results)
        self.state.status = "Ciclagem concluída"
        self.state.is_running = False

        self.emit("table_updated", "cic")
        self.emit("log", f"Ciclagem: {n_files} arquivo(s) processado(s).")
        if new_plots:
            self.emit("plots_added", new_plots)
        self.emit("status_changed", self.state.status)
        self.emit("progress_stop", "Ciclagem concluída")
        self.emit("buttons_enable")
        self.emit("cycling_completed", True)
        return True

    def process_both_result(
        self, result: Optional[Tuple[dict, dict]]
    ) -> bool:
        """Ingest combined EIS + Cycling pipeline output."""
        if result is None:
            self.state.status = "erro ao rodar ambos"
            self.state.is_running = False
            self.emit("status_changed", self.state.status)
            self.emit("progress_stop", "Erro")
            self.emit("buttons_enable")
            self.emit("both_completed", False)
            return False

        eis_result, cic_result = result

        # EIS portion
        cap_df = eis_result.get("cap_energy")
        if cap_df is not None:
            self.state.eis_df = cap_df.reset_index().rename(
                columns={"index": "Arquivo"}
            )
        else:
            self.state.eis_df = None

        self.state.rank_df = eis_result.get("df_ranked")
        self.state.df_pca = eis_result.get("df_pca")
        self.state.raw_eis = eis_result.get("raw_eis") or {}
        self.state.circuit_df = eis_result.get("circuit_table")

        # Cycling portion
        self.state.cic_df = cic_result.get("merged_table")
        self.state.cic_results = cic_result.get("results") or {}
        self.state.cic_plot_map = {}

        new_plots: List[PlotItem] = []
        for path in eis_result.get("pca_paths", []):
            title = "PCA" if "pca" in path.lower() else "Gráfico"
            new_plots.append(PlotItem(title=title, path=path))
        for filename, path in cic_result.get("plot_paths", []):
            key = self.normalize_sample_name(filename)
            self.state.cic_plot_map[key] = os.path.abspath(path)
            new_plots.append(
                PlotItem(title=f"{filename} - Integral", path=path)
            )
        for filename, path in cic_result.get("energy_power_paths", []):
            new_plots.append(
                PlotItem(title=f"{filename} - Energia×Potência", path=path)
            )
        self.state.plot_items.extend(new_plots)

        self.state.status = "Ambos concluídos"
        self.state.is_running = False

        self.emit("table_updated", "eis")
        self.emit("table_updated", "cic")
        self.emit("table_updated", "circuit")
        if new_plots:
            self.emit("plots_added", new_plots)
        self.emit("status_changed", self.state.status)
        self.emit("progress_stop", "Ambos concluídos")
        self.emit("buttons_enable")
        self.emit("both_completed", True)
        return True

    def process_drt_result(self, result: Optional[dict]) -> bool:
        """Ingest raw DRT pipeline output into *AppState*."""
        if result is None:
            self.state.status = "erro no DRT"
            self.state.is_running = False
            self.emit("status_changed", self.state.status)
            self.emit("progress_stop", "Erro")
            self.emit("buttons_enable")
            self.emit("drt_completed", False)
            return False

        self.state.drt_df = result.get("drt_table")
        self.state.drt_peaks_df = result.get("drt_peaks_table")
        self.state.drt_summary_df = result.get("drt_summary_table")
        self.state.drt_results = result.get("per_file_results", {}) or {}
        self.state.drt_plot_map = {}

        new_plots: List[PlotItem] = []
        for filename, path in result.get("plot_paths", []):
            key = self.normalize_sample_name(filename)
            self.state.drt_plot_map[key] = os.path.abspath(path)
            new_plots.append(
                PlotItem(title=f"{filename} - DRT", path=path)
            )
        self.state.plot_items.extend(new_plots)

        # Log errors and metadata
        errors = result.get("errors", {}) or {}
        run_meta = result.get("run_meta", {}) or {}
        if errors:
            self.emit(
                "log",
                f"DRT concluído com {len(errors)} falha(s): "
                + ", ".join(errors.keys()),
            )
        if run_meta:
            self.emit(
                "log",
                "DRT meta: "
                f"arquivos={run_meta.get('n_files', 0)}, "
                f"ok={run_meta.get('n_success', 0)}, "
                f"falhas={run_meta.get('n_failed', 0)}, "
                f"λ={run_meta.get('lambda_reg', float('nan')):.2e}, "
                f"n_taus={run_meta.get('n_taus', 0)}",
            )

        # Summary logging
        with contextlib.suppress(Exception):
            if (
                self.state.drt_summary_df is not None
                and not self.state.drt_summary_df.empty
                and "gamma_peak_main" in self.state.drt_summary_df.columns
            ):
                best_row = self.state.drt_summary_df.sort_values(
                    by="gamma_peak_main",
                    ascending=False,
                    na_position="last",
                ).iloc[0]
                self.emit(
                    "log",
                    "DRT resumo: "
                    f"amostras={len(self.state.drt_summary_df)}, "
                    f"maior pico={best_row.get('Sample', '')} "
                    f"(γ={best_row.get('gamma_peak_main', float('nan')):.3f})",
                )

        self.state.status = "DRT concluído"
        self.state.is_running = False

        self.emit("table_updated", "drt")
        self.emit("table_updated", "drt_peaks")
        if new_plots:
            self.emit("plots_added", new_plots)
        self.emit("status_changed", self.state.status)
        self.emit("progress_stop", "DRT concluído")
        self.emit("buttons_enable")
        self.emit("drt_completed", True)
        return True

    # ── Pipeline lifecycle ─────────────────────────────────────

    def start_pipeline(self, pipeline_name: str) -> None:
        """Prepare application state for a new pipeline run.

        Sets ``is_running``, clears previous plot items, disables UI
        buttons, and emits ``progress_start`` so the view can show a
        spinner or progress indicator.  Threading is handled by the caller.
        """
        self.state.is_running = True
        self.state.status = f"rodando {pipeline_name}"
        self.state.plot_items.clear()
        self.emit("buttons_disable")
        self.emit("status_changed", self.state.status)
        self.emit("progress_start", f"Identificando {pipeline_name}...")

    # ── Settings management ────────────────────────────────────

    def load_settings(self) -> Dict[str, Any]:
        """Load persisted JSON settings from disk into ``state.gui_settings``.

        Returns the loaded dictionary, or an empty dict if the file is
        missing, unreadable, or contains non-dict JSON.
        """
        if not self.settings_path or not os.path.exists(self.settings_path):
            return {}
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = data if isinstance(data, dict) else {}
            self.state.gui_settings = result
            return result
        except Exception:
            return {}

    def save_settings(self) -> bool:
        """Persist ``state.gui_settings`` to the JSON file on disk.

        Returns ``True`` on success, ``False`` if no path is configured
        or the write fails.
        """
        if not self.settings_path:
            return False
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.state.gui_settings, f, ensure_ascii=False, indent=2
                )
            return True
        except Exception:
            return False

    def remember_dialog_dir(self, key: str, path_value: str) -> None:
        """Persist the last-used directory for a given file dialog."""
        if not path_value:
            return
        folder = path_value
        if os.path.isfile(path_value):
            folder = os.path.dirname(path_value)
        if not os.path.isdir(folder):
            return
        self.state.gui_settings[f"last_dir_{key}"] = folder
        self.save_settings()

    def get_initial_dialog_dir(self, key: str) -> Optional[str]:
        """Return the last-used directory for *key*, or ``None``."""
        folder = self.state.gui_settings.get(f"last_dir_{key}")
        if isinstance(folder, str) and os.path.isdir(folder):
            return folder
        return None

    # ── DRT preset helpers ─────────────────────────────────────

    @staticmethod
    def get_drt_preset(name: str) -> Optional[Dict[str, str]]:
        """Return the DRT preset dict for *name*, or ``None``."""
        return DRT_PRESETS.get(name)

    @staticmethod
    def validate_drt_params(
        lambda_str: str, n_taus_str: str
    ) -> Tuple[float, int]:
        """Parse and validate DRT parameters from string inputs.

        Raises ``ValueError`` on invalid input.
        """
        lambda_reg = float(lambda_str.strip())
        n_taus = int(float(n_taus_str.strip()))
        if lambda_reg <= 0:
            raise ValueError("λ DRT deve ser > 0")
        if n_taus < 10:
            raise ValueError("n_taus deve ser >= 10")
        return lambda_reg, n_taus

    def reset_drt_defaults(self) -> Dict[str, str]:
        """Reset DRT params to the default preset; return preset dict."""
        preset = DRT_PRESETS[DRT_DEFAULT_PRESET]
        self.state.drt_ui_prefs.update(
            {"sample": "", "mode": "Espectro", "overlay_text": ""}
        )
        self.emit("log", "DRT resetado para preset padrão (Balanceado).")
        return preset

    # ── Utility ────────────────────────────────────────────────

    @staticmethod
    def normalize_sample_name(text: str) -> str:
        """Canonical form for sample name look-ups."""
        return text.strip().lower().replace(" ", "_")
