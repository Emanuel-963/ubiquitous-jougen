#!/usr/bin/env python
import contextlib
import json
import os
import queue
import shutil
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
from tkinter import filedialog, ttk

from main import run_eis_pipeline
from main_cycling import run_ciclagem_pipeline
from main_drt import run_drt_pipeline
from src.drt_visualization import (
    plot_drt_heatmap,
    plot_drt_overlay,
    plot_drt_spectrum,
)
from src.cycling_plotter import plot_energy_power_vs_cycle
from src.eis_plots import (
    plot_bode,
    plot_boxplot_metrics,
    plot_energy_cycle,
    plot_impedance_heatmap,
    plot_nyquist,
    plot_radar,
    plot_ragone,
    plot_retention_cycle,
)
from src.i18n import get_language, set_language, tr
from src.updater import check_for_updates


@dataclass
class PlotItem:
    title: str
    path: str


class QueueWriter:
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, msg: str):
        if msg.strip():
            self.q.put(("log", msg))

    def flush(self):
        return None


class PipelineApp(ctk.CTk):
    REQUIRED_THEME_KEYS = {
        "CTk",
        "CTkButton",
        "CTkEntry",
        "CTkFrame",
        "CTkLabel",
        "CTkOptionMenu",
        "CTkProgressBar",
        "CTkScrollableFrame",
        "CTkSegmentedButton",
        "CTkScrollbar",
        "CTkTabview",
        "CTkTextbox",
        "CTkToplevel",
        "DropdownMenu",
    }
    DRT_PRESETS = {
        "Rápido": {"lambda_reg": "5e-3", "n_taus": "30"},
        "Balanceado": {"lambda_reg": "1e-3", "n_taus": "50"},
        "Alta resolução": {"lambda_reg": "5e-4", "n_taus": "80"},
    }
    DRT_DEFAULT_PRESET = "Balanceado"

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        theme_path = os.path.join(os.path.dirname(__file__), "themes", "ionflow.json")
        self._load_theme_with_fallback(theme_path)

        self.title("IonFlow Pipeline")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        self.log_queue: queue.Queue = queue.Queue()
        self.image_refs: List[ctk.CTkImage] = []
        self.eis_df: Optional[pd.DataFrame] = None
        self.cic_df: Optional[pd.DataFrame] = None
        self.circuit_df: Optional[pd.DataFrame] = None
        self.drt_df: Optional[pd.DataFrame] = None
        self.drt_peaks_df: Optional[pd.DataFrame] = None
        self.drt_summary_df: Optional[pd.DataFrame] = None
        self.drt_eis_df: Optional[pd.DataFrame] = None
        self.drt_results: Dict[str, dict] = {}
        self.rank_df: Optional[pd.DataFrame] = None
        self.df_pca: Optional[pd.DataFrame] = None
        self.interactive_win: Optional[ctk.CTkToplevel] = None
        self.table_states: Dict[str, dict] = {}
        self.cic_plot_map: Dict[str, str] = {}
        self.cic_results: Dict[str, pd.DataFrame] = {}
        self.raw_eis: Dict[str, pd.DataFrame] = {}
        self.drt_plot_map: Dict[str, str] = {}
        self.settings_path = os.path.join(
            os.path.dirname(__file__),
            ".ionflow_gui_settings.json",
        )
        self.gui_settings = self._load_gui_settings()
        self.drt_ui_prefs = {
            "sample": "",
            "mode": "Espectro",
            "overlay_text": "",
        }

        self._build_layout()
        self._restore_ui_preferences()
        self._restore_language()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._process_queue)
        self.after(2000, self._check_for_updates_async)

    def _load_theme_with_fallback(self, theme_path: str):
        """Carrega tema customizado; se inválido/incompleto, cai para tema padrão."""
        if not os.path.exists(theme_path):
            ctk.set_default_color_theme("blue")
            return

        try:
            with open(theme_path, "r", encoding="utf-8") as f:
                theme_data = json.load(f)
            if not isinstance(theme_data, dict):
                raise ValueError("Tema inválido: conteúdo raiz não é objeto JSON.")

            missing = sorted(self.REQUIRED_THEME_KEYS - set(theme_data.keys()))
            if missing:
                raise KeyError(f"Chaves de tema ausentes: {', '.join(missing)}")

            ctk.set_default_color_theme(theme_path)
        except Exception as exc:
            ctk.set_default_color_theme("blue")
            with contextlib.suppress(Exception):
                self._append_log(
                    f"Tema IonFlow inválido/incompleto ({exc}). Usando tema padrão."
                )

    def _label_from_appearance_mode(self, mode: str) -> str:
        mapping = {
            "light": "Claro",
            "dark": "Escuro",
            "system": "Sistema",
        }
        return mapping.get(str(mode).lower(), "Escuro")

    def _mode_from_appearance_label(self, label: str) -> str:
        mapping = {
            "Claro": "light",
            "Escuro": "dark",
            "Sistema": "system",
        }
        return mapping.get(label, "dark")

    def _set_appearance_mode(self, mode: str, persist: bool = True):
        normalized = str(mode).lower()
        if normalized not in {"light", "dark", "system"}:
            normalized = "dark"

        ctk.set_appearance_mode(normalized)

        if hasattr(self, "appearance_mode_selector"):
            with contextlib.suppress(Exception):
                self.appearance_mode_selector.set(
                    self._label_from_appearance_mode(normalized)
                )

        if persist:
            self.gui_settings["appearance_mode"] = normalized
            self._save_gui_settings()

    def _on_appearance_mode_change(self, value: str):
        try:
            mode = self._mode_from_appearance_label(value)
            self._set_appearance_mode(mode, persist=True)
        except Exception as exc:
            self._append_log(f"Falha ao trocar tema: {exc}")
            self._set_appearance_mode("dark", persist=True)

    def _on_language_change(self, value: str):
        """Switch the application language and persist the preference."""
        lang = "en" if value == "English" else "pt"
        set_language(lang)
        self.gui_settings["language"] = lang
        self._save_gui_settings()
        self._append_log(
            f"Language set to {'English' if lang == 'en' else 'Português'}. "
            "Restart for full effect."
        )

    def _check_for_updates_async(self):
        """Run version check in background thread so it never blocks the GUI."""
        def _worker():
            result = check_for_updates()
            if result:
                self.log_queue.put(("log", result))

        threading.Thread(target=_worker, daemon=True).start()

    def _log_interactive_unavailable(self):
        self._append_log("Interativo não disponível para este gráfico.")

    def _mark_drt_preset_custom(self, _event=None):
        if getattr(self, "drt_preset_selector", None) is None:
            return
        with contextlib.suppress(Exception):
            if self.drt_preset_selector.get() != "Custom":
                self.drt_preset_selector.set("Custom")

    def _apply_drt_preset(self, preset_name: str, persist: bool = True):
        if preset_name == "Custom":
            if persist:
                self.gui_settings["drt_preset"] = "Custom"
                self._save_gui_settings()
            return

        preset = self.DRT_PRESETS.get(preset_name)
        if not preset:
            return

        self.drt_lambda_entry.delete(0, "end")
        self.drt_lambda_entry.insert(0, preset["lambda_reg"])
        self.drt_n_taus_entry.delete(0, "end")
        self.drt_n_taus_entry.insert(0, preset["n_taus"])

        with contextlib.suppress(Exception):
            self.drt_preset_selector.set(preset_name)

        if persist:
            self.gui_settings["drt_preset"] = preset_name
            self._save_gui_settings()

    def _read_drt_params(self) -> Tuple[float, int]:
        lambda_reg = float(self.drt_lambda_entry.get().strip())
        n_taus = int(float(self.drt_n_taus_entry.get().strip()))
        if lambda_reg <= 0:
            raise ValueError("λ DRT deve ser > 0")
        if n_taus < 10:
            raise ValueError("n_taus deve ser >= 10")
        return (lambda_reg, n_taus)

    def _reset_drt_defaults(self):
        self._apply_drt_preset(self.DRT_DEFAULT_PRESET, persist=True)
        self.drt_ui_prefs.update(
            {
                "sample": "",
                "mode": "Espectro",
                "overlay_text": "",
            }
        )
        self._append_log("DRT resetado para preset padrão (Balanceado).")

    def _load_gui_settings(self) -> dict:
        if not os.path.exists(self.settings_path):
            return {}
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_gui_settings(self):
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.gui_settings, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._append_log(f"Falha ao salvar configurações da GUI: {exc}")

    def _remember_dialog_dir(self, key: str, path_value: str):
        if not path_value:
            return
        folder = path_value
        if os.path.isfile(path_value):
            folder = os.path.dirname(path_value)
        if not os.path.isdir(folder):
            return
        self.gui_settings[f"last_dir_{key}"] = folder
        self._save_gui_settings()

    def _get_initial_dialog_dir(self, key: str) -> Optional[str]:
        folder = self.gui_settings.get(f"last_dir_{key}")
        if isinstance(folder, str) and os.path.isdir(folder):
            return folder
        return None

    def _restore_language(self):
        """Restore persisted language preference."""
        lang = self.gui_settings.get("language", "pt")
        if lang not in ("pt", "en"):
            lang = "pt"
        set_language(lang)
        label = "English" if lang == "en" else "Português"
        with contextlib.suppress(Exception):
            self.language_selector.set(label)

    def _restore_ui_preferences(self):
        appearance_mode = self.gui_settings.get("appearance_mode", "dark")
        self._set_appearance_mode(str(appearance_mode), persist=False)

        with contextlib.suppress(Exception):
            window_geometry = self.gui_settings.get("window_geometry")
            if isinstance(window_geometry, str) and window_geometry.strip():
                self.geometry(window_geometry)

        scan_rate = self.gui_settings.get("scan_rate")
        if isinstance(scan_rate, str) and scan_rate.strip():
            self.scan_rate_entry.delete(0, "end")
            self.scan_rate_entry.insert(0, scan_rate)

        drt_lambda_reg = self.gui_settings.get("drt_lambda_reg")
        if isinstance(drt_lambda_reg, str) and drt_lambda_reg.strip():
            self.drt_lambda_entry.delete(0, "end")
            self.drt_lambda_entry.insert(0, drt_lambda_reg)

        drt_n_taus = self.gui_settings.get("drt_n_taus")
        if isinstance(drt_n_taus, str) and drt_n_taus.strip():
            self.drt_n_taus_entry.delete(0, "end")
            self.drt_n_taus_entry.insert(0, drt_n_taus)

        drt_preset = self.gui_settings.get("drt_preset", self.DRT_DEFAULT_PRESET)
        if isinstance(drt_preset, str):
            valid = ["Custom", "Rápido", "Balanceado", "Alta resolução"]
            self.drt_preset_selector.set(
                drt_preset if drt_preset in valid else self.DRT_DEFAULT_PRESET
            )

        with contextlib.suppress(Exception):
            main_tab = self.gui_settings.get("main_tab")
            if isinstance(main_tab, str):
                self.tabs.set(main_tab)

        with contextlib.suppress(Exception):
            table_tab = self.gui_settings.get("table_tab")
            if isinstance(table_tab, str):
                self.tables_tabs.set(table_tab)

        table_filters = self.gui_settings.get("table_filters", {})
        if isinstance(table_filters, dict):
            for key, state in self.table_states.items():
                text = table_filters.get(key)
                if isinstance(text, str) and text:
                    state["filter_entry"].delete(0, "end")
                    state["filter_entry"].insert(0, text)

        drt_ui = self.gui_settings.get("drt_ui", {})
        if isinstance(drt_ui, dict):
            sample = drt_ui.get("sample", "")
            mode = drt_ui.get("mode", "Espectro")
            overlay_text = drt_ui.get("overlay_text", "")
            self.drt_ui_prefs = {
                "sample": sample if isinstance(sample, str) else "",
                "mode": mode if isinstance(mode, str) else "Espectro",
                "overlay_text": (
                    overlay_text if isinstance(overlay_text, str) else ""
                ),
            }

    def _on_close(self):
        with contextlib.suppress(Exception):
            self.gui_settings["window_geometry"] = self.geometry()
        with contextlib.suppress(Exception):
            self.gui_settings["scan_rate"] = self.scan_rate_entry.get().strip()
        with contextlib.suppress(Exception):
            self.gui_settings["drt_lambda_reg"] = self.drt_lambda_entry.get().strip()
        with contextlib.suppress(Exception):
            self.gui_settings["drt_n_taus"] = self.drt_n_taus_entry.get().strip()
        with contextlib.suppress(Exception):
            self.gui_settings["drt_preset"] = self.drt_preset_selector.get()
        with contextlib.suppress(Exception):
            self.gui_settings["main_tab"] = self.tabs.get()
        with contextlib.suppress(Exception):
            self.gui_settings["table_tab"] = self.tables_tabs.get()
        with contextlib.suppress(Exception):
            self.gui_settings["appearance_mode"] = self._mode_from_appearance_label(
                self.appearance_mode_selector.get()
            )

        filters = {}
        for key, state in self.table_states.items():
            with contextlib.suppress(Exception):
                filters[key] = state["filter_entry"].get().strip()
        self.gui_settings["table_filters"] = filters
        self.gui_settings["drt_ui"] = dict(self.drt_ui_prefs)
        with contextlib.suppress(Exception):
            self.gui_settings["language"] = get_language()

        self._save_gui_settings()
        self.destroy()

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, corner_radius=12)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=16, pady=16)
        sidebar.grid_rowconfigure(16, weight=1)

        ctk.CTkLabel(
            sidebar,
            text=tr("Pipelines"),
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        ctk.CTkLabel(sidebar, text=tr("Scan rate (A/g)")).grid(
            row=1, column=0, padx=16, pady=(8, 4), sticky="w"
        )
        self.scan_rate_entry = ctk.CTkEntry(sidebar)
        self.scan_rate_entry.insert(0, "0.1")
        self.scan_rate_entry.grid(row=2, column=0, padx=16, pady=(0, 12), sticky="ew")

        self.btn_import_raw = ctk.CTkButton(
            sidebar,
            text=tr("Importar EIS para raw"),
            command=lambda: self._import_files(
                target_dir="data/raw",
                label="Selecione arquivos EIS",
            ),
        )
        self.btn_import_raw.grid(row=3, column=0, padx=16, pady=(0, 8), sticky="ew")

        self.btn_import_processed = ctk.CTkButton(
            sidebar,
            text=tr("Importar Ciclagem para processed"),
            command=lambda: self._import_files(
                target_dir="data/processed",
                label="Selecione arquivos de Ciclagem",
            ),
        )
        self.btn_import_processed.grid(
            row=4,
            column=0,
            padx=16,
            pady=(0, 12),
            sticky="ew",
        )

        self.btn_interactive = ctk.CTkButton(
            sidebar,
            text=tr("Gráficos Interativos"),
            command=self._open_interactive_window,
        )
        self.btn_interactive.grid(row=5, column=0, padx=16, pady=(0, 12), sticky="ew")

        self.btn_eis = ctk.CTkButton(
            sidebar,
            text=tr("Rodar Pipeline EIS"),
            command=self._run_eis_clicked,
        )
        self.btn_eis.grid(row=6, column=0, padx=16, pady=8, sticky="ew")

        self.btn_ciclagem = ctk.CTkButton(
            sidebar,
            text=tr("Rodar Pipeline Ciclagem"),
            command=self._run_ciclagem_clicked,
        )
        self.btn_ciclagem.grid(row=7, column=0, padx=16, pady=8, sticky="ew")

        self.btn_both = ctk.CTkButton(
            sidebar,
            text=tr("Rodar Ambos"),
            command=self._run_both_clicked,
        )
        self.btn_both.grid(row=8, column=0, padx=16, pady=8, sticky="ew")

        self.btn_drt = ctk.CTkButton(
            sidebar,
            text=tr("Rodar Pipeline DRT"),
            command=self._run_drt_clicked,
        )
        self.btn_drt.grid(row=9, column=0, padx=16, pady=8, sticky="ew")

        drt_param_frame = ctk.CTkFrame(sidebar)
        drt_param_frame.grid(row=10, column=0, padx=16, pady=(6, 8), sticky="ew")
        drt_param_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(drt_param_frame, text=tr("λ DRT")).grid(
            row=0, column=0, padx=(8, 4), pady=(6, 2), sticky="w"
        )
        ctk.CTkLabel(drt_param_frame, text=tr("n_taus")).grid(
            row=0, column=1, padx=(4, 8), pady=(6, 2), sticky="w"
        )

        self.drt_lambda_entry = ctk.CTkEntry(drt_param_frame)
        self.drt_lambda_entry.insert(0, "1e-3")
        self.drt_lambda_entry.grid(
            row=1,
            column=0,
            padx=(8, 4),
            pady=(0, 8),
            sticky="ew",
        )

        self.drt_n_taus_entry = ctk.CTkEntry(drt_param_frame)
        self.drt_n_taus_entry.insert(0, "50")
        self.drt_n_taus_entry.grid(
            row=1,
            column=1,
            padx=(4, 8),
            pady=(0, 8),
            sticky="ew",
        )
        self.drt_lambda_entry.bind("<KeyRelease>", self._mark_drt_preset_custom)
        self.drt_n_taus_entry.bind("<KeyRelease>", self._mark_drt_preset_custom)

        ctk.CTkLabel(drt_param_frame, text=tr("Preset DRT")).grid(
            row=2,
            column=0,
            padx=(8, 4),
            pady=(0, 2),
            sticky="w",
        )
        self.drt_preset_selector = ctk.CTkOptionMenu(
            drt_param_frame,
            values=["Custom", "Rápido", "Balanceado", "Alta resolução"],
            fg_color="#e2e8f0",
            button_color="#0b84ff",
            button_hover_color="#0c76e0",
            text_color="#0f172a",
            dropdown_fg_color="#ffffff",
            dropdown_hover_color="#e2e8f0",
            dropdown_text_color="#0f172a",
        )
        self.drt_preset_selector.set("Balanceado")
        self.drt_preset_selector.grid(
            row=3,
            column=0,
            padx=(8, 4),
            pady=(0, 8),
            sticky="ew",
        )

        ctk.CTkButton(
            drt_param_frame,
            text=tr("Aplicar preset"),
            width=120,
            command=lambda: self._apply_drt_preset(
                self.drt_preset_selector.get(),
                persist=True,
            ),
        ).grid(row=3, column=1, padx=(4, 8), pady=(0, 8), sticky="ew")

        ctk.CTkButton(
            drt_param_frame,
            text=tr("Reset DRT"),
            width=120,
            command=self._reset_drt_defaults,
        ).grid(row=4, column=1, padx=(4, 8), pady=(0, 8), sticky="ew")

        ctk.CTkLabel(
            drt_param_frame,
            text=tr("Rápido:30 | Balanceado:50 | Alta:80"),
            font=ctk.CTkFont(size=10),
            anchor="w",
        ).grid(row=4, column=0, padx=(8, 4), pady=(0, 8), sticky="ew")

        self.status_label = ctk.CTkLabel(
            sidebar, text=tr("Status: pronto"), anchor="w"
        )
        self.status_label.grid(row=11, column=0, padx=16, pady=8, sticky="ew")

        self.progress_label = ctk.CTkLabel(sidebar, text=tr("Pronto"), anchor="w")
        self.progress_label.grid(row=12, column=0, padx=16, pady=(0, 4), sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(sidebar, mode="indeterminate")
        self.progress_bar.grid(row=13, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.progress_bar.set(0)

        ctk.CTkLabel(sidebar, text=tr("Tema"), anchor="w").grid(
            row=14, column=0, padx=16, pady=(0, 4), sticky="ew"
        )
        self.appearance_mode_selector = ctk.CTkSegmentedButton(
            sidebar,
            values=[tr("Claro"), tr("Escuro"), tr("Sistema")],
            command=self._on_appearance_mode_change,
        )
        self.appearance_mode_selector.grid(
            row=15, column=0, padx=16, pady=(0, 12), sticky="ew"
        )
        self.appearance_mode_selector.set(tr("Escuro"))

        ctk.CTkLabel(sidebar, text=tr("Idioma"), anchor="w").grid(
            row=16, column=0, padx=16, pady=(0, 4), sticky="ew"
        )
        self.language_selector = ctk.CTkSegmentedButton(
            sidebar,
            values=["Português", "English"],
            command=self._on_language_change,
        )
        self.language_selector.grid(
            row=17, column=0, padx=16, pady=(0, 12), sticky="ew"
        )
        self.language_selector.set("Português")

        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)

        self.tab_plots = self.tabs.add(tr("Gráficos"))
        self.tab_tables = self.tabs.add(tr("Tabelas"))
        self.tab_logs = self.tabs.add(tr("Logs"))

        # Plots area
        self.plots_frame = ctk.CTkScrollableFrame(self.tab_plots)
        self.plots_frame.pack(fill="both", expand=True, padx=12, pady=12)

        # Tables area
        self.tables_tabs = ctk.CTkTabview(self.tab_tables)
        self.tables_tabs.pack(fill="both", expand=True, padx=8, pady=8)
        self.tab_table_eis = self.tables_tabs.add(tr("EIS"))
        self.tab_table_cic = self.tables_tabs.add(tr("Ciclagem"))
        self.tab_table_circuit = self.tables_tabs.add(tr("Circuitos"))
        self.tab_table_drt = self.tables_tabs.add(tr("DRT"))
        self.tab_table_drt_peaks = self.tables_tabs.add(tr("DRT Peaks"))
        self.tab_table_drt_eis = self.tables_tabs.add(tr("DRT + EIS"))

        self.eis_table = self._create_table(self.tab_table_eis, "eis")
        self.cic_table = self._create_table(self.tab_table_cic, "cic")
        self.circuit_table = self._create_table(self.tab_table_circuit, "circuit")
        self.drt_table = self._create_table(self.tab_table_drt, "drt")
        self.drt_peaks_table = self._create_table(
            self.tab_table_drt_peaks,
            "drt_peaks",
        )
        self.drt_eis_table = self._create_table(self.tab_table_drt_eis, "drt_eis")

        # Logs area
        self.log_text = ctk.CTkTextbox(self.tab_logs, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=12, pady=12)

    def _create_table(self, parent, table_key: str) -> ttk.Treeview:
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        frame.grid_rowconfigure(0, weight=0)  # botão
        frame.grid_rowconfigure(1, weight=1)  # tabela
        frame.grid_rowconfigure(2, weight=0)  # status
        frame.grid_rowconfigure(3, weight=0)  # scrollbar horizontal
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0)

        controls = ctk.CTkFrame(frame)
        controls.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=(0, 2),
            pady=(2, 4),
        )
        controls.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(controls, text="Filtro:").grid(
            row=0,
            column=0,
            padx=(8, 6),
            pady=6,
            sticky="w",
        )
        filter_entry = ctk.CTkEntry(
            controls,
            placeholder_text="Buscar em todas as colunas",
        )
        filter_entry.grid(row=0, column=1, padx=(0, 8), pady=6, sticky="ew")

        btn_save_filtered = ctk.CTkButton(
            controls,
            text="Salvar filtrado",
            width=130,
            command=lambda k=table_key: self._save_table_by_key(k, filtered=True),
        )
        btn_save_filtered.grid(row=0, column=2, padx=(0, 8), pady=6)

        btn_save_all = ctk.CTkButton(
            controls,
            text="Salvar tudo",
            width=110,
            command=lambda k=table_key: self._save_table_by_key(k, filtered=False),
        )
        btn_save_all.grid(row=0, column=3, padx=(0, 8), pady=6)

        tree = ttk.Treeview(frame, show="headings")
        tree.grid(row=1, column=0, sticky="nsew")

        scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scrollbar_y.grid(row=1, column=1, sticky="ns")
        scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        scrollbar_x.grid(row=3, column=0, sticky="ew")

        tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        status_label = ctk.CTkLabel(frame, text="Linhas: 0/0", anchor="w")
        status_label.grid(row=2, column=0, sticky="ew", pady=(4, 2))

        self.table_states[table_key] = {
            "tree": tree,
            "filter_entry": filter_entry,
            "status_label": status_label,
            "source_df": None,
            "view_df": None,
            "sort_col": None,
            "sort_asc": True,
        }
        filter_entry.bind(
            "<KeyRelease>",
            lambda _e, k=table_key: self._refresh_table(k),
        )
        tree.bind("<Double-1>", lambda _e, k=table_key: self._on_table_row_activate(k))
        tree.bind("<Return>", lambda _e, k=table_key: self._on_table_row_activate(k))
        return tree

    def _split_series_name(self, name: str) -> Tuple[float, str]:
        text = str(name).replace(".txt", "")
        parts = text.split()
        if not parts:
            return (0.0, text)
        try:
            lead = float(parts[0])
            base = " ".join(parts[1:]) if len(parts) > 1 else text
            return (lead, base)
        except ValueError:
            return (0.0, text)

    def _normalize_sample_name(self, text: str) -> str:
        return os.path.splitext(str(text).strip().lower())[0]

    def _find_matching_index(
        self, names: List[str], sample_name: Optional[str]
    ) -> Optional[int]:
        if not sample_name:
            return None
        target = self._normalize_sample_name(sample_name)
        norm_names = [self._normalize_sample_name(n) for n in names]

        if target in norm_names:
            return norm_names.index(target)

        for idx, item in enumerate(norm_names):
            if item.endswith(target) or target.endswith(item):
                return idx
        return None

    def _find_cic_plot_path(self, sample_name: str) -> Optional[str]:
        key = self._normalize_sample_name(sample_name)
        if key in self.cic_plot_map:
            return self.cic_plot_map[key]

        # fallback: busca por sufixo para nomes próximos
        for map_key, map_path in self.cic_plot_map.items():
            if map_key.endswith(key) or key.endswith(map_key):
                return map_path
        return None

    def _parse_drt_sample_list(
        self, text: str, available: List[str]
    ) -> Tuple[List[str], List[str]]:
        if not text:
            return ([], [])

        normalized_available = [self._normalize_sample_name(v) for v in available]
        selected: List[str] = []
        not_found: List[str] = []
        seen = set()

        for token in [t.strip() for t in text.split(",") if t.strip()]:
            target = self._normalize_sample_name(token)
            match_name = None

            if target in normalized_available:
                match_name = available[normalized_available.index(target)]
            else:
                for idx, item in enumerate(normalized_available):
                    if item.endswith(target) or target.endswith(item):
                        match_name = available[idx]
                        break

            if match_name and match_name not in seen:
                selected.append(match_name)
                seen.add(match_name)
            elif not match_name:
                not_found.append(token)

        return (selected, not_found)

    def _open_image_preview(self, title: str, path: str):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            self._append_log(f"Imagem não encontrada: {abs_path}")
            return

        try:
            img = Image.open(abs_path)
        except Exception as exc:
            self._append_log(f"Falha ao abrir imagem {abs_path}: {exc}")
            return

        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry("1200x820")
        win.minsize(900, 650)

        top = ctk.CTkFrame(win)
        top.pack(fill="x", padx=12, pady=(12, 6))

        ctk.CTkLabel(top, text=title, font=ctk.CTkFont(size=15, weight="bold")).pack(
            side="left", padx=(8, 12)
        )
        ctk.CTkButton(
            top,
            text="Salvar imagem",
            command=lambda p=abs_path: self._save_plot(p),
            width=140,
        ).pack(side="right", padx=8)

        body = ctk.CTkScrollableFrame(win)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        max_w, max_h = 1500, 1500
        img.thumbnail((max_w, max_h))
        ctk_img = ctk.CTkImage(img, size=(img.width, img.height))
        self.image_refs.append(ctk_img)

        ctk.CTkLabel(body, image=ctk_img, text="", compound="top").pack(
            pady=8, padx=8
        )

    def _on_table_row_activate(self, table_key: str):
        state = self.table_states.get(table_key)
        if not state:
            return

        tree = state["tree"]
        selection = tree.selection()
        if not selection:
            return

        columns = list(tree["columns"])
        if columns == ["info"]:
            return

        values = tree.item(selection[0], "values")
        row = {columns[i]: values[i] for i in range(min(len(columns), len(values)))}

        try:
            if table_key == "eis":
                arquivo = row.get("Arquivo")
                if not arquivo:
                    self._append_log("Linha EIS sem coluna 'Arquivo'.")
                    return

                _lead, base_name = self._split_series_name(str(arquivo))
                preferred_col = None
                if self.eis_df is not None:
                    available_cols = list(self.eis_df.columns)
                else:
                    available_cols = []
                for col in ["Energia média (J)", "C_espec (F/g)"]:
                    if col in available_cols:
                        preferred_col = col
                        break

                self._open_interactive_window(
                    preferred_tab="Séries",
                    preferred_series_col=preferred_col,
                    preferred_series_base=base_name,
                    preferred_sample=str(arquivo),
                )
                return

            if table_key == "circuit":
                sample_name = None
                for key in ["Arquivo", "Sample", "Amostra", "Nome"]:
                    value = row.get(key)
                    if value:
                        sample_name = str(value)
                        break
                self._open_interactive_window(
                    preferred_tab="PCA 2D",
                    preferred_sample=sample_name,
                )
                return

            if table_key == "cic":
                arquivo = row.get("Arquivo")
                if not arquivo:
                    self._append_log("Linha de Ciclagem sem coluna 'Arquivo'.")
                    return

                self._open_interactive_window(
                    preferred_tab="Energia × Potência",
                    preferred_sample=str(arquivo),
                )
                return

            if table_key == "drt":
                arquivo = row.get("Arquivo")
                if not arquivo:
                    self._append_log("Linha DRT sem coluna 'Arquivo'.")
                    return

                if self.drt_results:
                    self._open_interactive_window(
                        preferred_tab="DRT",
                        preferred_sample=str(arquivo),
                    )
                    return

                drt_path = self.drt_plot_map.get(
                    self._normalize_sample_name(str(arquivo))
                )
                if drt_path:
                    self._open_image_preview(f"{arquivo} - DRT", drt_path)
                else:
                    self.tabs.set("Gráficos")
                    self._append_log(f"Gráfico DRT não encontrado para: {arquivo}")
                return

            if table_key == "drt_peaks":
                sample_name = row.get("Sample") or row.get("Arquivo")
                if not sample_name:
                    self._append_log("Linha DRT Peaks sem coluna Sample/Arquivo.")
                    return
                self._open_interactive_window(
                    preferred_tab="DRT",
                    preferred_sample=str(sample_name),
                )
                return

            if table_key == "drt_eis":
                sample_name = row.get("Sample") or row.get("Arquivo")
                if not sample_name:
                    self._append_log("Linha DRT+EIS sem coluna Sample/Arquivo.")
                    return
                self._open_interactive_window(
                    preferred_tab="DRT × EIS",
                    preferred_sample=str(sample_name),
                )
                return
        except Exception as exc:
            self._append_log(f"Erro ao abrir ação contextual: {exc}")

    def _save_table_by_key(self, table_key: str, filtered: bool):
        state = self.table_states.get(table_key)
        if not state:
            self._append_log("Tabela não encontrada.")
            return
        df = state.get("view_df") if filtered else state.get("source_df")
        default_name = {
            "eis": "tabela_eis.csv",
            "cic": "tabela_ciclagem.csv",
            "circuit": "tabela_circuitos.csv",
            "drt": "tabela_drt.csv",
            "drt_peaks": "tabela_drt_peaks.csv",
            "drt_eis": "tabela_drt_eis.csv",
        }.get(table_key, "tabela.csv")
        self._save_table(df, default_name)

    def _set_table_data(self, table_key: str, df: Optional[pd.DataFrame]):
        state = self.table_states.get(table_key)
        if not state:
            return
        state["source_df"] = df.copy() if df is not None else None
        state["sort_col"] = None
        state["sort_asc"] = True
        self._refresh_table(table_key)

    def _update_drt_eis_join_table(self):
        if (
            self.drt_summary_df is None
            or self.drt_summary_df.empty
            or self.rank_df is None
            or self.rank_df.empty
        ):
            self.drt_eis_df = None
            self._set_table_data("drt_eis", None)
            return

        drt_df = self.drt_summary_df.copy()
        if "Arquivo" not in drt_df.columns and "Sample" in drt_df.columns:
            drt_df["Arquivo"] = drt_df["Sample"]

        drt_df["_norm"] = drt_df["Arquivo"].astype(str).apply(
            self._normalize_sample_name
        )

        rank_df = self.rank_df.copy().reset_index().rename(columns={"index": "Arquivo"})
        rank_df["_norm"] = rank_df["Arquivo"].astype(str).apply(
            self._normalize_sample_name
        )

        rank_cols = [
            "_norm",
            "Arquivo",
            "Rank",
            "Retenção (%)",
            "Subclasse",
        ]
        rank_cols = [c for c in rank_cols if c in rank_df.columns]

        merged = drt_df.merge(
            rank_df[rank_cols],
            on="_norm",
            how="left",
            suffixes=("_drt", "_eis"),
        )

        if "Arquivo_drt" not in merged.columns and "Arquivo" in merged.columns:
            merged["Arquivo_drt"] = merged["Arquivo"]
        if "Arquivo_eis" not in merged.columns:
            merged["Arquivo_eis"] = merged.get("Arquivo", "")

        ordered_cols = [
            "Sample",
            "Arquivo_drt",
            "Arquivo_eis",
            "n_peaks",
            "R_inf",
            "tau_peak_main",
            "gamma_peak_main",
            "residual_mean",
            "residual_max",
            "Rank",
            "Retenção (%)",
            "Subclasse",
        ]

        score_cols = ["gamma_peak_main", "Retenção (%)"]
        if all(col in merged.columns for col in score_cols):
            gamma = pd.to_numeric(merged["gamma_peak_main"], errors="coerce")
            retention = pd.to_numeric(merged["Retenção (%)"], errors="coerce")

            g_min, g_max = gamma.min(skipna=True), gamma.max(skipna=True)
            r_min, r_max = retention.min(skipna=True), retention.max(skipna=True)

            gamma_norm = (
                (gamma - g_min) / (g_max - g_min)
                if pd.notna(g_min) and pd.notna(g_max) and g_max > g_min
                else pd.Series(0.0, index=merged.index)
            )
            retention_norm = (
                (retention - r_min) / (r_max - r_min)
                if pd.notna(r_min) and pd.notna(r_max) and r_max > r_min
                else pd.Series(0.0, index=merged.index)
            )

            merged["Score_DRT_EIS"] = (
                0.5 * gamma_norm.fillna(0.0)
                + 0.5 * retention_norm.fillna(0.0)
            )
            ordered_cols.append("Score_DRT_EIS")

        ordered_cols = [c for c in ordered_cols if c in merged.columns]
        merged = merged[ordered_cols]

        self.drt_eis_df = merged
        self._set_table_data("drt_eis", self.drt_eis_df)

        # Ordenação padrão: maior score combinado primeiro
        state = self.table_states.get("drt_eis")
        if state:
            if "Score_DRT_EIS" in merged.columns:
                state["sort_col"] = "Score_DRT_EIS"
                state["sort_asc"] = False
            elif "Rank" in merged.columns:
                state["sort_col"] = "Rank"
                state["sort_asc"] = True
            self._refresh_table("drt_eis")

        # Export consolidado para relatórios
        out_dir = os.path.join("outputs", "tables")
        with contextlib.suppress(Exception):
            os.makedirs(out_dir, exist_ok=True)
            self.drt_eis_df.to_csv(
                os.path.join(out_dir, "drt_eis_summary.csv"),
                index=False,
            )

    def _on_table_heading_click(self, table_key: str, col: str):
        state = self.table_states.get(table_key)
        if not state:
            return
        if state.get("sort_col") == col:
            state["sort_asc"] = not state.get("sort_asc", True)
        else:
            state["sort_col"] = col
            state["sort_asc"] = True
        self._refresh_table(table_key)

    def _estimate_column_width(
        self, df: pd.DataFrame, col: str, *, min_w: int = 120, max_w: int = 520
    ) -> int:
        """Estima largura da coluna com base no cabeçalho e amostra de valores."""
        if col not in df.columns:
            return min_w

        sample = df[col].astype(str).fillna("").head(120)
        max_len = max([len(str(col))] + [len(v) for v in sample.tolist()])

        # aproximação simples: ~7 px por caractere + margem
        est = int(max_len * 7 + 28)
        return max(min_w, min(max_w, est))

    def _refresh_table(self, table_key: str):
        state = self.table_states.get(table_key)
        if not state:
            return
        tree = state["tree"]
        source_df = state.get("source_df")
        query = state["filter_entry"].get().strip().lower()
        sort_col = state.get("sort_col")
        sort_asc = state.get("sort_asc", True)

        for row_id in tree.get_children():
            tree.delete(row_id)

        if source_df is None or source_df.empty:
            tree["columns"] = ["info"]
            tree.heading("info", text="info")
            tree.column("info", width=280, anchor="w")
            tree.insert("", "end", values=["Nenhum dado disponível."])
            state["view_df"] = pd.DataFrame()
            state["status_label"].configure(text="Linhas: 0/0")
            return

        df = source_df.copy()
        total_rows = len(df)

        if query:
            mask = (
                df.astype(str)
                .apply(lambda s: s.str.contains(query, case=False, na=False))
                .any(axis=1)
            )
            df = df[mask]

        if sort_col in df.columns:
            series = df[sort_col]
            if pd.api.types.is_numeric_dtype(series):
                df = df.sort_values(by=sort_col, ascending=sort_asc, na_position="last")
            else:
                numeric_try = pd.to_numeric(series, errors="coerce")
                if numeric_try.notna().sum() > 0:
                    df = df.assign(_sort_col=numeric_try).sort_values(
                        by="_sort_col", ascending=sort_asc, na_position="last"
                    ).drop(columns=["_sort_col"])
                else:
                    df = df.sort_values(
                        by=sort_col,
                        key=lambda s: s.astype(str).str.lower(),
                        ascending=sort_asc,
                        na_position="last",
                    )

        state["view_df"] = df.copy()

        tree["columns"] = list(df.columns)
        for col in df.columns:
            arrow = ""
            if col == sort_col:
                arrow = " ▲" if sort_asc else " ▼"
            col_width = self._estimate_column_width(df, col)
            tree.heading(
                col,
                text=f"{col}{arrow}",
                command=lambda c=col, k=table_key: self._on_table_heading_click(k, c),
            )
            tree.column(col, width=col_width, anchor="center", stretch=True)

        for row in df.itertuples(index=False, name=None):
            tree.insert("", "end", values=list(row))

        state["status_label"].configure(text=f"Linhas: {len(df)}/{total_rows}")

    def _set_status(self, text: str):
        self.status_label.configure(text=f"Status: {text}")

    def _disable_buttons(self):
        self.btn_eis.configure(state="disabled")
        self.btn_ciclagem.configure(state="disabled")
        self.btn_both.configure(state="disabled")
        self.btn_drt.configure(state="disabled")

    def _enable_buttons(self):
        self.btn_eis.configure(state="normal")
        self.btn_ciclagem.configure(state="normal")
        self.btn_both.configure(state="normal")
        self.btn_drt.configure(state="normal")

    def _append_log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def _start_progress(self, text: str):
        self.progress_label.configure(text=text)
        self.progress_bar.start()

    def _update_progress(self, text: str):
        self.progress_label.configure(text=text)

    def _stop_progress(self, text: str = "Pronto"):
        self.progress_bar.stop()
        self.progress_label.configure(text=text)

    def _clear_plots(self):
        for widget in self.plots_frame.winfo_children():
            widget.destroy()
        self.image_refs.clear()

    def _destroy_interactive(self):
        if self.interactive_win is not None and self.interactive_win.winfo_exists():
            with contextlib.suppress(Exception):
                self.interactive_win.destroy()
        self.interactive_win = None

    def _save_plot(self, path: str):
        folder = filedialog.askdirectory(
            title="Escolha a pasta para salvar",
            initialdir=self._get_initial_dialog_dir("plot_save"),
        )
        if not folder:
            return
        self._remember_dialog_dir("plot_save", folder)
        dest = os.path.join(folder, os.path.basename(path))
        try:
            shutil.copy(path, dest)
            self._append_log(f"Imagem salva em {dest}")
        except Exception as exc:
            self._append_log(f"Falha ao salvar imagem: {exc}")

    def _save_table(self, df: Optional[pd.DataFrame], default_name: str):
        if df is None or df.empty:
            self._append_log("Nenhum dado para salvar.")
            return
        dest = filedialog.asksaveasfilename(
            title="Salvar tabela",
            initialfile=default_name,
            initialdir=self._get_initial_dialog_dir("table_save"),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")],
        )
        if not dest:
            return
        self._remember_dialog_dir("table_save", dest)
        try:
            ext = os.path.splitext(dest)[1].lower()
            if ext == ".xlsx":
                df.to_excel(dest, index=False)
            else:
                df.to_csv(dest, index=False)
            self._append_log(f"Tabela salva em {dest}")
        except Exception as exc:
            self._append_log(f"Falha ao salvar tabela: {exc}")

    def _render_fig_on_frame(self, frame, fig: Figure):
        for child in frame.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_fig_rank(
        self, highlight_sample: Optional[str] = None
    ) -> Optional[Figure]:
        if self.rank_df is None or self.rank_df.empty:
            return None
        if (
            "Rank" not in self.rank_df.columns
            or "Retenção (%)" not in self.rank_df.columns
        ):
            return None
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(
            self.rank_df["Rank"],
            self.rank_df["Retenção (%)"],
            c="#1f77b4",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.set_xlabel("Rank")
        ax.set_ylabel("Retenção (%)")
        ax.set_title("Rank vs Retenção")
        ax.grid(True, alpha=0.3)

        idx = self._find_matching_index(
            self.rank_df.index.astype(str).tolist(), highlight_sample
        )
        if idx is not None:
            x_val = self.rank_df["Rank"].iloc[idx]
            y_val = self.rank_df["Retenção (%)"].iloc[idx]
            ax.scatter(
                [x_val],
                [y_val],
                s=180,
                marker="*",
                c="#ef4444",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
                label="Amostra selecionada",
            )
            ax.legend(loc="best", fontsize=8)
        elif highlight_sample:
            ax.text(
                0.02,
                0.98,
                f"Amostra não encontrada:\n{highlight_sample}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "#fff3cd", "edgecolor": "#f59e0b", "alpha": 0.9},
            )

        try:
            import mplcursors

            cursor = mplcursors.cursor(ax.collections[0], hover=True)
            labels = list(self.rank_df.index.astype(str))

            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                label = labels[idx] if idx < len(labels) else ""
                x, y = sel.target
                sel.annotation.set_text(f"{label}\nRank: {x:.2f}\nRetenção: {y:.2f}%")
        except ImportError:
            pass
        fig.tight_layout()
        return fig

    def _build_fig_pca(
        self, highlight_sample: Optional[str] = None
    ) -> Optional[Figure]:
        if self.df_pca is None or self.df_pca.empty:
            return None
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(
            self.df_pca.get("PC1", []),
            self.df_pca.get("PC2", []),
            c="tab:blue",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA 2D")
        ax.grid(True, alpha=0.3)

        idx = self._find_matching_index(
            self.df_pca.index.astype(str).tolist(), highlight_sample
        )
        if idx is not None:
            x_val = self.df_pca["PC1"].iloc[idx]
            y_val = self.df_pca["PC2"].iloc[idx]
            ax.scatter(
                [x_val],
                [y_val],
                s=180,
                marker="*",
                c="#ef4444",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label="Amostra selecionada",
            )
            ax.legend(loc="best", fontsize=8)
        elif highlight_sample:
            ax.text(
                0.02,
                0.98,
                f"Amostra não encontrada:\n{highlight_sample}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "#fff3cd", "edgecolor": "#f59e0b", "alpha": 0.9},
            )

        try:
            import mplcursors

            cursor = mplcursors.cursor(ax.collections[0], hover=True)
            names = list(self.df_pca.index.astype(str))

            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                name = names[idx] if idx < len(names) else ""
                x, y = sel.target
                sel.annotation.set_text(f"{name}\nPC1: {x:.2f}\nPC2: {y:.2f}")
        except ImportError:
            pass
        fig.tight_layout()
        return fig

    def _build_fig_pca_metric(
        self, highlight_sample: Optional[str] = None
    ) -> Optional[Figure]:
        if self.df_pca is None or self.df_pca.empty:
            return None
        retention = None
        if self.rank_df is not None and "Retenção (%)" in self.rank_df.columns:
            retention = self.rank_df["Retenção (%)"].reindex(self.df_pca.index)
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            self.df_pca.get("PC1", []),
            self.df_pca.get("PC2", []),
            c=retention if retention is not None else "tab:blue",
            cmap="viridis" if retention is not None else None,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
        )
        if retention is not None:
            fig.colorbar(scatter, ax=ax, label="Retenção (%)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA 2D - Retenção")
        ax.grid(True, alpha=0.3)

        idx = self._find_matching_index(
            self.df_pca.index.astype(str).tolist(), highlight_sample
        )
        if idx is not None:
            x_val = self.df_pca["PC1"].iloc[idx]
            y_val = self.df_pca["PC2"].iloc[idx]
            ax.scatter(
                [x_val],
                [y_val],
                s=180,
                marker="*",
                c="#ef4444",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label="Amostra selecionada",
            )
            ax.legend(loc="best", fontsize=8)
        elif highlight_sample:
            ax.text(
                0.02,
                0.98,
                f"Amostra não encontrada:\n{highlight_sample}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "#fff3cd", "edgecolor": "#f59e0b", "alpha": 0.9},
            )

        try:
            import mplcursors

            cursor = mplcursors.cursor(scatter, hover=True)
            names = list(self.df_pca.index.astype(str))
            vals = list(retention if retention is not None else [None for _ in names])

            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                name = names[idx] if idx < len(names) else ""
                x, y = sel.target
                val = vals[idx] if idx < len(vals) else None
                metric = (
                    f"\nRetenção: {val:.2f}%"
                    if val is not None and not pd.isna(val)
                    else ""
                )
                sel.annotation.set_text(f"{name}\nPC1: {x:.2f}\nPC2: {y:.2f}{metric}")
        except ImportError:
            pass
        fig.tight_layout()
        return fig

    def _build_fig_corr(self) -> Optional[Figure]:
        if self.rank_df is None or self.rank_df.empty:
            return None
        data = self.rank_df.select_dtypes(include=["number"]).dropna(how="all")
        if data.shape[1] < 2:
            return None
        corr = data.corr(method="spearman")
        fig = Figure(figsize=(5.2, 4.4), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        fig.colorbar(im, ax=ax, label="Spearman ρ")
        try:
            import mplcursors

            cursor = mplcursors.cursor(im, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                j, i = int(round(sel.target[0])), int(round(sel.target[1]))
                if 0 <= i < corr.shape[0] and 0 <= j < corr.shape[1]:
                    val = corr.iat[i, j]
                    row = corr.index[i]
                    col = corr.columns[j]
                    sel.annotation.set_text(f"{row} x {col}\nρ = {val:.2f}")
        except ImportError:
            pass
        fig.tight_layout()
        return fig

    def _build_fig_drt_eis(
        self, highlight_sample: Optional[str] = None
    ) -> Optional[Figure]:
        if self.drt_eis_df is None or self.drt_eis_df.empty:
            return None
        if "gamma_peak_main" not in self.drt_eis_df.columns:
            return None
        if "Retenção (%)" not in self.drt_eis_df.columns:
            return None

        df = self.drt_eis_df.copy()
        x = pd.to_numeric(df["gamma_peak_main"], errors="coerce")
        y = pd.to_numeric(df["Retenção (%)"], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            return None

        fig = Figure(figsize=(5.3, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        pts = ax.scatter(
            x[mask],
            y[mask],
            c="#0ea5e9",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.set_xlabel("γ pico principal")
        ax.set_ylabel("Retenção (%)")
        ax.set_title("DRT × EIS")
        ax.grid(True, alpha=0.3)

        names = (
            df["Sample"].astype(str).tolist()
            if "Sample" in df.columns
            else df.index.astype(str).tolist()
        )
        valid_names = [names[i] for i, ok in enumerate(mask.tolist()) if ok]

        idx = self._find_matching_index(valid_names, highlight_sample)
        if idx is not None:
            x_val = x[mask].iloc[idx]
            y_val = y[mask].iloc[idx]
            ax.scatter(
                [x_val],
                [y_val],
                s=180,
                marker="*",
                c="#ef4444",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label="Amostra selecionada",
            )
            ax.legend(loc="best", fontsize=8)

        try:
            import mplcursors

            cursor = mplcursors.cursor(pts, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                i = sel.index
                name = valid_names[i] if i < len(valid_names) else ""
                x_val, y_val = sel.target
                sel.annotation.set_text(
                    f"{name}\nγ: {x_val:.3f}\nRetenção: {y_val:.2f}%"
                )
        except ImportError:
            pass

        fig.tight_layout()
        return fig

    def _build_fig_series(self, value_col: str, base_name: str) -> Optional[Figure]:
        if self.eis_df is None or self.eis_df.empty:
            return None
        if value_col not in self.eis_df.columns:
            return None

        def _split(name: str):
            parts_local = str(name).replace(".txt", "").split()
            if not parts_local:
                return (0.0, name)
            try:
                return (
                    float(parts_local[0]),
                    " ".join(parts_local[1:]) if len(parts_local) > 1 else name,
                )
            except ValueError:
                return (0.0, name)

        df = self.eis_df.copy()
        info = df["Arquivo"].apply(_split)
        df["_lead"] = info.apply(lambda x: x[0])
        df["_base"] = info.apply(lambda x: x[1])
        grp = df[df["_base"] == base_name].sort_values("_lead")
        if grp.empty or grp[value_col].dropna().empty:
            return None
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(grp["_lead"], grp[value_col], "o-", color="#1f77b4")
        ax.set_xlabel("Prefixo numérico")
        ax.set_ylabel(value_col)
        ax.set_title(f"{value_col} - {base_name}")
        ax.grid(True, alpha=0.3)
        try:
            import mplcursors

            cursor = mplcursors.cursor(ax.lines[0], hover=True)

            @cursor.connect("add")
            def on_add(sel):
                x, y = sel.target
                sel.annotation.set_text(f"Prefixo: {x:.2f}\nValor: {y:.2f}")
        except ImportError:
            pass
        fig.tight_layout()
        return fig

    def _build_fig_energy_power(
        self, sample_name: str
    ) -> Optional[Figure]:
        """Gráfico dual-axis Energia × Potência vs Ciclo (embeddable)."""
        if not self.cic_results:
            return None

        # Busca por nome normalizado
        cycle_df = self.cic_results.get(sample_name)
        if cycle_df is None:
            norm = self._normalize_sample_name(sample_name)
            for key, df in self.cic_results.items():
                if self._normalize_sample_name(key) == norm:
                    cycle_df = df
                    break
        if cycle_df is None or cycle_df.empty:
            return None

        fig = Figure(figsize=(5.8, 4.4), dpi=100)
        ax = fig.add_subplot(111)
        plot_energy_power_vs_cycle(
            cycle_df, sample_name,
            ax=ax, fig=fig, save=False, show=False,
        )

        # mplcursors para as duas linhas
        try:
            import mplcursors

            for line in ax.lines:
                cursor = mplcursors.cursor(line, hover=True)
                lbl = line.get_label()

                @cursor.connect("add")
                def on_add(sel, label=lbl):
                    x, y = sel.target
                    sel.annotation.set_text(
                        f"{label}\n"
                        f"Ciclo: {x:.0f}\n"
                        f"Valor: {y:.4f}"
                    )
        except ImportError:
            pass

        return fig

    # ---- Nyquist / Bode / Ragone builders --------------------------------

    def _build_fig_nyquist(
        self, sample_name: str
    ) -> Optional[Figure]:
        """Nyquist embeddable com hover de frequência."""
        if not self.raw_eis:
            return None
        df = self.raw_eis.get(sample_name)
        if df is None:
            norm = self._normalize_sample_name(sample_name)
            for key, val in self.raw_eis.items():
                if self._normalize_sample_name(key) == norm:
                    df = val
                    break
        if df is None or df.empty:
            return None

        fig = Figure(figsize=(5.8, 5.2), dpi=100)
        ax = fig.add_subplot(111)
        plot_nyquist(df, sample_name, ax=ax, fig=fig, save=False, show=False)

        try:
            import mplcursors

            sc = ax.collections[0] if ax.collections else None
            if sc is not None:
                d = df.sort_values("frequency", ascending=False)
                freqs = d["frequency"].values
                cursor = mplcursors.cursor(sc, hover=True)

                @cursor.connect("add")
                def on_add(sel, _f=freqs):
                    i = sel.index
                    x, y = sel.target
                    f = _f[i] if i < len(_f) else 0
                    sel.annotation.set_text(
                        f"Z′={x:.2f} Ω\n"
                        f"−Z″={y:.2f} Ω\n"
                        f"f={f:.2f} Hz"
                    )
        except ImportError:
            pass
        return fig

    def _build_fig_bode(
        self, sample_name: str
    ) -> Optional[Figure]:
        """Bode embeddable com hover."""
        if not self.raw_eis:
            return None
        df = self.raw_eis.get(sample_name)
        if df is None:
            norm = self._normalize_sample_name(sample_name)
            for key, val in self.raw_eis.items():
                if self._normalize_sample_name(key) == norm:
                    df = val
                    break
        if df is None or df.empty:
            return None

        fig = Figure(figsize=(6.2, 4.6), dpi=100)
        ax = fig.add_subplot(111)
        plot_bode(df, sample_name, ax=ax, fig=fig, save=False, show=False)

        try:
            import mplcursors
            import numpy as _np

            d = df.sort_values("frequency")
            freqs = d["frequency"].values
            z_mag = _np.sqrt(
                d["zreal"].values ** 2 + d["zimag"].values ** 2
            )
            phase = _np.degrees(
                _np.arctan2(-d["zimag"].values, d["zreal"].values)
            )

            for line in ax.lines:
                lbl = line.get_label()
                cur = mplcursors.cursor(line, hover=True)

                @cur.connect("add")
                def on_add(
                    sel, label=lbl, _f=freqs, _z=z_mag, _p=phase
                ):
                    i = sel.index
                    f = _f[i] if i < len(_f) else 0
                    z = _z[i] if i < len(_z) else 0
                    p = _p[i] if i < len(_p) else 0
                    sel.annotation.set_text(
                        f"{label}\n"
                        f"f={f:.2f} Hz\n"
                        f"|Z|={z:.2f} Ω\n"
                        f"Fase={p:.1f}°"
                    )
        except ImportError:
            pass
        return fig

    # ------------------------------------------------------------------
    # Medium-priority builders: Energy×Cycle, Heatmap, Box-plot
    # ------------------------------------------------------------------
    def _build_fig_energy_cycle(
        self,
        metric: str = "Energia (Wh/kg)",
        highlight_samples: Optional[List[str]] = None,
    ) -> Optional[Figure]:
        """Overlay de *metric* vs ciclo para todas as amostras."""
        if not self.cic_results:
            return None

        fig = Figure(figsize=(7.5, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        plot_energy_cycle(
            self.cic_results,
            metric=metric,
            highlight_samples=highlight_samples,
            ax=ax, fig=fig, save=False, show=False,
        )

        try:
            import mplcursors

            lines = [
                ln for ln in ax.get_lines()
                if ln.get_label() and not ln.get_label().startswith("_")
            ]
            if lines:
                cursor = mplcursors.cursor(lines, hover=True)

                @cursor.connect("add")
                def on_add(sel):
                    nm = sel.artist.get_label()
                    x, y = sel.target
                    sel.annotation.set_text(
                        f"{nm}\nCiclo {x:.0f}: {y:.4f}"
                    )
        except ImportError:
            pass
        return fig

    def _build_fig_impedance_heatmap(
        self,
    ) -> Optional[Figure]:
        """Heatmap log₁₀|Z| (amostras × frequência)."""
        if not self.raw_eis:
            return None

        fig = Figure(figsize=(8, 5.6), dpi=100)
        ax = fig.add_subplot(111)
        plot_impedance_heatmap(
            self.raw_eis,
            ax=ax, fig=fig, save=False, show=False,
        )

        try:
            import mplcursors

            imgs = [c for c in ax.get_children()
                    if hasattr(c, "get_array")]
            if imgs:
                names = sorted(self.raw_eis.keys())
                cursor = mplcursors.cursor(
                    imgs, hover=True,
                )

                @cursor.connect("add")
                def on_add(sel, _names=names):
                    x, y = sel.target
                    row = int(round(y))
                    nm = (
                        _names[row]
                        if 0 <= row < len(_names)
                        else "?"
                    )
                    sel.annotation.set_text(
                        f"{nm}\nlog₁₀f={x:.2f}"
                    )
        except ImportError:
            pass
        return fig

    def _build_fig_boxplot(
        self,
        metric: str = "Rs",
    ) -> Optional[Figure]:
        """Box-plot de *metric* agrupado por Subclass."""
        if self.eis_df is None or self.eis_df.empty:
            return None

        fig = Figure(figsize=(7, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        result = plot_boxplot_metrics(
            self.eis_df,
            metric=metric,
            ax=ax, fig=fig, save=False, show=False,
        )
        if result is None and ax is not None:
            # plot_boxplot_metrics retornou None → métrica inválida
            ax.text(
                0.5, 0.5,
                f"Sem dados para '{metric}'",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=12,
            )
        return fig

    def _build_fig_radar(
        self,
        samples: List[str],
    ) -> Optional[Figure]:
        """Radar/spider chart comparando amostras."""
        if self.eis_df is None or self.eis_df.empty:
            return None
        if not samples:
            return None

        fig = Figure(figsize=(6.4, 6.4), dpi=100)
        ax = fig.add_subplot(111, polar=True)
        result = plot_radar(
            self.eis_df, samples,
            ax=ax, fig=fig, save=False, show=False,
        )
        if result is None:
            ax.text(
                0, 0,
                "Dados insuficientes\npara radar",
                ha="center", va="center",
                fontsize=12,
            )
        return fig

    def _build_fig_retention_cycle(
        self,
    ) -> Optional[Figure]:
        """Retenção (%) vs Ciclo overlay."""
        if not self.cic_results:
            return None

        fig = Figure(figsize=(7.5, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        plot_retention_cycle(
            self.cic_results,
            ax=ax, fig=fig, save=False, show=False,
        )

        try:
            import mplcursors

            lines = [
                ln for ln in ax.get_lines()
                if ln.get_label()
                and not ln.get_label().startswith("_")
            ]
            if lines:
                cursor = mplcursors.cursor(
                    lines, hover=True,
                )

                @cursor.connect("add")
                def on_add(sel):
                    nm = sel.artist.get_label()
                    x, y = sel.target
                    sel.annotation.set_text(
                        f"{nm}\n"
                        f"Ciclo {x:.0f}: {y:.1f}%"
                    )
        except ImportError:
            pass
        return fig

    def _build_fig_ragone(
        self, highlight_sample: Optional[str] = None
    ) -> Optional[Figure]:
        """Ragone plot embeddable com hover."""
        if not self.cic_results:
            return None

        fig = Figure(figsize=(5.8, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        plot_ragone(
            self.cic_results,
            highlight_sample=highlight_sample,
            ax=ax, fig=fig, save=False, show=False,
        )

        try:
            import mplcursors

            sc = ax.collections[0] if ax.collections else None
            if sc is not None:
                names = sorted(self.cic_results.keys())
                cursor = mplcursors.cursor(sc, hover=True)

                @cursor.connect("add")
                def on_add(sel, _n=names):
                    i = sel.index
                    x, y = sel.target
                    nm = _n[i] if i < len(_n) else ""
                    sel.annotation.set_text(
                        f"{nm}\n"
                        f"Energia: {x:.4f} Wh/kg\n"
                        f"Potência: {y:.2f} W/kg"
                    )
        except ImportError:
            pass
        return fig

    def _build_fig_drt_spectrum(self, sample_name: str) -> Optional[Figure]:
        if not self.drt_results:
            return None
        result = self.drt_results.get(sample_name)
        if not result:
            return None
        fig = Figure(figsize=(5.4, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        plot_drt_spectrum(result, sample_name, ax=ax, save=False, show=False)
        fig.tight_layout()
        return fig

    def _build_fig_drt_overlay(self, sample_names: List[str]) -> Optional[Figure]:
        if not self.drt_results:
            return None
        if not sample_names:
            sample_names = sorted(self.drt_results.keys())[:6]
        fig = Figure(figsize=(5.8, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        plot_drt_overlay(
            self.drt_results,
            selected=sample_names,
            ax=ax,
            show=False,
        )
        fig.tight_layout()
        return fig

    def _build_fig_drt_heatmap(self, sample_names: List[str]) -> Optional[Figure]:
        if not self.drt_results:
            return None
        if not sample_names:
            sample_names = sorted(self.drt_results.keys())
        fig = Figure(figsize=(6.2, 4.4), dpi=100)
        ax = fig.add_subplot(111)
        plot_drt_heatmap(
            self.drt_results,
            stems=sample_names,
            ax=ax,
            show=False,
        )
        fig.tight_layout()
        return fig

    def _update_drt_plot(
        self,
        tab_drt,
        sample_name: str,
        mode: str,
        overlay_samples: Optional[List[str]] = None,
    ):
        prev_state = getattr(tab_drt, "_drt_state", {})
        selected_overlay = (
            overlay_samples
            if overlay_samples is not None
            else prev_state.get("overlay_samples", [])
        )
        tab_drt._drt_state = {
            "sample": sample_name,
            "mode": mode,
            "overlay_samples": selected_overlay,
        }
        self.drt_ui_prefs["sample"] = str(sample_name or "")
        self.drt_ui_prefs["mode"] = str(mode or "Espectro")
        self.drt_ui_prefs["overlay_text"] = ", ".join(selected_overlay)
        container = getattr(tab_drt, "_drt_container", None)
        if container is None:
            container = ctk.CTkFrame(tab_drt)
            container.pack(fill="both", expand=True, padx=8, pady=8)
            tab_drt._drt_container = container

        names = sorted(self.drt_results.keys())
        if mode == "Heatmap":
            fig = self._build_fig_drt_heatmap(names)
        elif mode == "Overlay":
            selected = selected_overlay if selected_overlay else []
            if not selected and sample_name and sample_name in self.drt_results:
                selected = [sample_name]
            fig = self._build_fig_drt_overlay(selected)
        else:
            fig = self._build_fig_drt_spectrum(sample_name)

        if fig:
            self._render_fig_on_frame(container, fig)
        else:
            for child in container.winfo_children():
                child.destroy()
            ctk.CTkLabel(container, text="Sem dados DRT disponíveis").pack(pady=20)

    def _save_drt_current_view(self, tab_drt):
        state = getattr(tab_drt, "_drt_state", {})
        mode = state.get("mode", "Espectro")
        sample_name = state.get("sample", "drt")
        overlay_samples = state.get("overlay_samples", [])

        default_name = {
            "Espectro": f"{sample_name}_drt_interativo.png",
            "Overlay": "drt_overlay_interativo.png",
            "Heatmap": "drt_heatmap_interativo.png",
        }.get(mode, "drt_interativo.png")

        dest = filedialog.asksaveasfilename(
            title="Salvar visual DRT",
            initialfile=default_name,
            initialdir=self._get_initial_dialog_dir("plot_save"),
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not dest:
            return
        self._remember_dialog_dir("plot_save", dest)

        fig = None
        try:
            if mode == "Heatmap":
                fig = self._build_fig_drt_heatmap(sorted(self.drt_results.keys()))
            elif mode == "Overlay":
                fig = self._build_fig_drt_overlay(overlay_samples)
            else:
                fig = self._build_fig_drt_spectrum(sample_name)

            if fig is None:
                self._append_log("Sem visual DRT para salvar.")
                return

            fig.savefig(dest, dpi=160, bbox_inches="tight")
            self._append_log(f"Visual DRT salvo em {dest}")
        except Exception as exc:
            self._append_log(f"Falha ao salvar visual DRT: {exc}")
        finally:
            if fig is not None:
                plt.close(fig)

    # ------------------------------------------------------------------
    # Helper: tab genérica com seletor de amostra + plot renderizado
    # ------------------------------------------------------------------
    def _build_sample_selector_tab(
        self,
        tab,
        data_dict: Dict[str, Any],
        builder,  # callable(sample_name) -> Optional[Figure]
        empty_msg: str = "Sem dados disponíveis",
        preferred_sample: Optional[str] = None,
        tab_label: str = "Plot",
    ):
        """Popula *tab* com dropdown de amostra + gráfico interativo."""
        if not data_dict:
            ctk.CTkLabel(tab, text=empty_msg).pack(pady=20)
            return

        names = sorted(data_dict.keys())
        initial = names[0]
        if preferred_sample:
            norm_pref = self._normalize_sample_name(preferred_sample)
            for n in names:
                if self._normalize_sample_name(n) == norm_pref:
                    initial = n
                    break

        control = ctk.CTkFrame(tab)
        control.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(control, text="Amostra:").pack(
            side="left", padx=(0, 6)
        )

        def _update(sample: str, _tab=tab):
            container = getattr(_tab, "_sel_container", None)
            if container is None:
                container = ctk.CTkFrame(_tab)
                container.pack(fill="both", expand=True, padx=8, pady=8)
                _tab._sel_container = container
            fig = builder(sample)
            if fig:
                self._render_fig_on_frame(container, fig)
            else:
                for ch in container.winfo_children():
                    ch.destroy()
                ctk.CTkLabel(
                    container,
                    text=f"Sem dados para {sample}",
                ).pack(pady=20)

        menu = ctk.CTkOptionMenu(
            control,
            values=names,
            command=_update,
            fg_color="#e2e8f0",
            button_color="#0b84ff",
            button_hover_color="#0c76e0",
            text_color="#0f172a",
            dropdown_fg_color="#ffffff",
            dropdown_hover_color="#e2e8f0",
            dropdown_text_color="#0f172a",
        )
        menu.set(initial)
        menu.pack(side="left", padx=(0, 12))

        def _save(_tab=tab):
            container = getattr(_tab, "_sel_container", None)
            if container is None:
                self._append_log(f"Nenhum gráfico {tab_label} para salvar.")
                return
            sample = menu.get()
            dest = filedialog.asksaveasfilename(
                title=f"Salvar gráfico {tab_label}",
                initialfile=f"{sample}_{tab_label.lower()}.png",
                initialdir=self._get_initial_dialog_dir("plot_save"),
                defaultextension=".png",
                filetypes=[("PNG", "*.png")],
            )
            if not dest:
                return
            self._remember_dialog_dir("plot_save", dest)
            fig = builder(sample)
            if fig:
                fig.savefig(dest, dpi=160, bbox_inches="tight")
                plt.close(fig)
                self._append_log(f"Gráfico {tab_label} salvo em {dest}")

        ctk.CTkButton(
            control,
            text="Salvar gráfico",
            width=120,
            command=_save,
        ).pack(side="left", padx=(0, 8))

        container = ctk.CTkFrame(tab)
        container.pack(fill="both", expand=True, padx=8, pady=8)
        tab._sel_container = container
        _update(initial)

    def _open_interactive_window(
        self,
        preferred_tab: Optional[str] = None,
        preferred_series_col: Optional[str] = None,
        preferred_series_base: Optional[str] = None,
        preferred_sample: Optional[str] = None,
    ):
        try:
            self._destroy_interactive()

            rank_idx = None
            pca_idx = None
            if preferred_sample:
                if self.rank_df is not None and not self.rank_df.empty:
                    rank_idx = self._find_matching_index(
                        self.rank_df.index.astype(str).tolist(), preferred_sample
                    )
                if self.df_pca is not None and not self.df_pca.empty:
                    pca_idx = self._find_matching_index(
                        self.df_pca.index.astype(str).tolist(), preferred_sample
                    )

            win = ctk.CTkToplevel(self)
            win.title("Gráficos Interativos")
            win.geometry("1100x750")
            win.minsize(900, 600)
            self.interactive_win = win

            tabs = ctk.CTkTabview(win)
            tabs.pack(fill="both", expand=True, padx=12, pady=12)

            tab_rank = tabs.add("Rank vs Retenção")
            tab_nyquist = tabs.add("Nyquist")
            tab_bode = tabs.add("Bode")
            tab_pca = tabs.add("PCA 2D")
            tab_pca_metric = tabs.add("PCA Retenção")
            tab_corr = tabs.add("Correlação")
            tab_series = tabs.add("Séries")
            tab_ep = tabs.add("Energia × Potência")
            tab_ragone = tabs.add("Ragone")
            tab_ecycle = tabs.add("Energia vs Ciclo")
            tab_retention = tabs.add("Retenção vs Ciclo")
            tab_heatmap = tabs.add("Heatmap |Z|")
            tab_boxplot = tabs.add("Box-plot")
            tab_radar = tabs.add("Radar")
            tab_drt = tabs.add("DRT")
            tab_drt_eis = tabs.add("DRT × EIS")

            rank_fig = self._build_fig_rank(highlight_sample=preferred_sample)
            if rank_fig:
                frame = ctk.CTkFrame(tab_rank)
                frame.pack(fill="both", expand=True, padx=8, pady=8)
                self._render_fig_on_frame(frame, rank_fig)
            else:
                ctk.CTkLabel(
                    tab_rank, text="Sem dados para Rank vs Retenção"
                ).pack(pady=20)

            # ---- Nyquist com seletor de amostra -------------------------
            self._build_sample_selector_tab(
                tab_nyquist, self.raw_eis,
                builder=self._build_fig_nyquist,
                empty_msg="Sem dados EIS disponíveis",
                preferred_sample=preferred_sample,
                tab_label="Nyquist",
            )

            # ---- Bode com seletor de amostra ----------------------------
            self._build_sample_selector_tab(
                tab_bode, self.raw_eis,
                builder=self._build_fig_bode,
                empty_msg="Sem dados EIS disponíveis",
                preferred_sample=preferred_sample,
                tab_label="Bode",
            )

            pca_fig = self._build_fig_pca(highlight_sample=preferred_sample)
            if pca_fig:
                frame = ctk.CTkFrame(tab_pca)
                frame.pack(fill="both", expand=True, padx=8, pady=8)
                self._render_fig_on_frame(frame, pca_fig)
            else:
                ctk.CTkLabel(tab_pca, text="Sem dados de PCA").pack(pady=20)

            pca_metric_fig = self._build_fig_pca_metric(
                highlight_sample=preferred_sample
            )
            if pca_metric_fig:
                frame = ctk.CTkFrame(tab_pca_metric)
                frame.pack(fill="both", expand=True, padx=8, pady=8)
                self._render_fig_on_frame(frame, pca_metric_fig)
            else:
                ctk.CTkLabel(
                    tab_pca_metric, text="Sem dados de PCA/Retenção"
                ).pack(pady=20)

            corr_fig = self._build_fig_corr()
            if corr_fig:
                frame = ctk.CTkFrame(tab_corr)
                frame.pack(fill="both", expand=True, padx=8, pady=8)
                self._render_fig_on_frame(frame, corr_fig)
            else:
                ctk.CTkLabel(
                    tab_corr, text="Sem dados suficientes para correlação"
                ).pack(pady=20)

            drt_eis_fig = self._build_fig_drt_eis(highlight_sample=preferred_sample)
            if drt_eis_fig:
                frame = ctk.CTkFrame(tab_drt_eis)
                frame.pack(fill="both", expand=True, padx=8, pady=8)
                self._render_fig_on_frame(frame, drt_eis_fig)
            else:
                ctk.CTkLabel(
                    tab_drt_eis,
                    text="Sem dados combinados DRT+EIS",
                ).pack(pady=20)

            # ---- Energia × Potência com seleção de arquivo ------------------
            if self.cic_results:
                ep_names = sorted(self.cic_results.keys())
                initial_ep = ep_names[0]
                if preferred_sample:
                    norm_pref = self._normalize_sample_name(
                        preferred_sample
                    )
                    for name in ep_names:
                        if self._normalize_sample_name(name) == norm_pref:
                            initial_ep = name
                            break

                ep_control = ctk.CTkFrame(tab_ep)
                ep_control.pack(fill="x", padx=8, pady=(8, 4))
                ctk.CTkLabel(ep_control, text="Arquivo:").pack(
                    side="left", padx=(0, 6)
                )

                def _update_ep_plot(
                    sample: str,
                    _tab=tab_ep,
                ):
                    container = getattr(_tab, "_ep_container", None)
                    if container is None:
                        container = ctk.CTkFrame(_tab)
                        container.pack(
                            fill="both", expand=True, padx=8, pady=8,
                        )
                        _tab._ep_container = container
                    fig = self._build_fig_energy_power(sample)
                    if fig:
                        self._render_fig_on_frame(container, fig)
                    else:
                        for child in container.winfo_children():
                            child.destroy()
                        ctk.CTkLabel(
                            container,
                            text="Sem dados de ciclos para esta amostra",
                        ).pack(pady=20)

                ep_menu = ctk.CTkOptionMenu(
                    ep_control,
                    values=ep_names,
                    command=_update_ep_plot,
                    fg_color="#e2e8f0",
                    button_color="#0b84ff",
                    button_hover_color="#0c76e0",
                    text_color="#0f172a",
                    dropdown_fg_color="#ffffff",
                    dropdown_hover_color="#e2e8f0",
                    dropdown_text_color="#0f172a",
                )
                ep_menu.set(initial_ep)
                ep_menu.pack(side="left", padx=(0, 12))

                def _save_ep_view(
                    _tab=tab_ep,
                ):
                    container = getattr(_tab, "_ep_container", None)
                    if container is None:
                        self._append_log("Nenhum gráfico EP para salvar.")
                        return
                    sample = ep_menu.get()
                    dest = filedialog.asksaveasfilename(
                        title="Salvar gráfico Energia×Potência",
                        initialfile=f"{sample}_energy_power.png",
                        initialdir=self._get_initial_dialog_dir(
                            "plot_save"
                        ),
                        defaultextension=".png",
                        filetypes=[("PNG", "*.png")],
                    )
                    if not dest:
                        return
                    self._remember_dialog_dir("plot_save", dest)
                    fig = self._build_fig_energy_power(sample)
                    if fig:
                        fig.savefig(
                            dest, dpi=160, bbox_inches="tight",
                        )
                        plt.close(fig)
                        self._append_log(
                            f"Gráfico Energia×Potência salvo em {dest}"
                        )

                ctk.CTkButton(
                    ep_control,
                    text="Salvar gráfico",
                    width=120,
                    command=_save_ep_view,
                ).pack(side="left", padx=(0, 8))

                ep_container = ctk.CTkFrame(tab_ep)
                ep_container.pack(
                    fill="both", expand=True, padx=8, pady=8,
                )
                tab_ep._ep_container = ep_container
                _update_ep_plot(initial_ep)
            else:
                ctk.CTkLabel(
                    tab_ep,
                    text="Sem dados de ciclagem disponíveis",
                ).pack(pady=20)

            # ---- Ragone (Energia vs Potência por amostra) -------------------
            if self.cic_results:
                ragone_names = ["— nenhum —"] + sorted(
                    self.cic_results.keys()
                )
                initial_hl = "— nenhum —"
                if preferred_sample:
                    norm_pref = self._normalize_sample_name(
                        preferred_sample
                    )
                    for n in ragone_names:
                        if self._normalize_sample_name(n) == norm_pref:
                            initial_hl = n
                            break

                rag_control = ctk.CTkFrame(tab_ragone)
                rag_control.pack(fill="x", padx=8, pady=(8, 4))
                ctk.CTkLabel(rag_control, text="Destaque:").pack(
                    side="left", padx=(0, 6)
                )

                def _update_ragone(
                    sample: str,
                    _tab=tab_ragone,
                ):
                    hl = None if sample == "— nenhum —" else sample
                    container = getattr(_tab, "_rag_container", None)
                    if container is None:
                        container = ctk.CTkFrame(_tab)
                        container.pack(
                            fill="both", expand=True, padx=8, pady=8,
                        )
                        _tab._rag_container = container
                    fig = self._build_fig_ragone(highlight_sample=hl)
                    if fig:
                        self._render_fig_on_frame(container, fig)
                    else:
                        for ch in container.winfo_children():
                            ch.destroy()
                        ctk.CTkLabel(
                            container,
                            text="Sem dados para Ragone",
                        ).pack(pady=20)

                rag_menu = ctk.CTkOptionMenu(
                    rag_control,
                    values=ragone_names,
                    command=_update_ragone,
                    fg_color="#e2e8f0",
                    button_color="#0b84ff",
                    button_hover_color="#0c76e0",
                    text_color="#0f172a",
                    dropdown_fg_color="#ffffff",
                    dropdown_hover_color="#e2e8f0",
                    dropdown_text_color="#0f172a",
                )
                rag_menu.set(initial_hl)
                rag_menu.pack(side="left", padx=(0, 12))

                def _save_ragone(_tab=tab_ragone):
                    hl_name = rag_menu.get()
                    hl = None if hl_name == "— nenhum —" else hl_name
                    dest = filedialog.asksaveasfilename(
                        title="Salvar gráfico Ragone",
                        initialfile="ragone_plot.png",
                        initialdir=self._get_initial_dialog_dir(
                            "plot_save"
                        ),
                        defaultextension=".png",
                        filetypes=[("PNG", "*.png")],
                    )
                    if not dest:
                        return
                    self._remember_dialog_dir("plot_save", dest)
                    fig = self._build_fig_ragone(highlight_sample=hl)
                    if fig:
                        fig.savefig(
                            dest, dpi=160, bbox_inches="tight",
                        )
                        plt.close(fig)
                        self._append_log(
                            f"Gráfico Ragone salvo em {dest}"
                        )

                ctk.CTkButton(
                    rag_control,
                    text="Salvar gráfico",
                    width=120,
                    command=_save_ragone,
                ).pack(side="left", padx=(0, 8))

                rag_container = ctk.CTkFrame(tab_ragone)
                rag_container.pack(
                    fill="both", expand=True, padx=8, pady=8,
                )
                tab_ragone._rag_container = rag_container
                _update_ragone(initial_hl)
            else:
                ctk.CTkLabel(
                    tab_ragone,
                    text="Sem dados de ciclagem para Ragone",
                ).pack(pady=20)

            # ---- Energia vs Ciclo (overlay multi-amostra) ---------------
            if self.cic_results:
                ecyc_metrics: List[str] = []
                for _tbl in self.cic_results.values():
                    for c in (
                        "Energia (Wh/kg)", "Potência (W/kg)",
                        "Duração dos Ciclos (s)",
                    ):
                        if c in _tbl.columns and c not in ecyc_metrics:
                            ecyc_metrics.append(c)
                    break

                if ecyc_metrics:
                    initial_metric = ecyc_metrics[0]

                    ecyc_ctrl = ctk.CTkFrame(tab_ecycle)
                    ecyc_ctrl.pack(fill="x", padx=8, pady=(8, 4))
                    ctk.CTkLabel(
                        ecyc_ctrl, text="Métrica:"
                    ).pack(side="left", padx=(0, 6))

                    def _update_ecycle(
                        metric: str, _tab=tab_ecycle,
                    ):
                        container = getattr(
                            _tab, "_ecyc_container", None,
                        )
                        if container is None:
                            container = ctk.CTkFrame(_tab)
                            container.pack(
                                fill="both", expand=True,
                                padx=8, pady=8,
                            )
                            _tab._ecyc_container = container
                        fig = self._build_fig_energy_cycle(
                            metric=metric,
                        )
                        if fig:
                            self._render_fig_on_frame(
                                container, fig,
                            )

                    ecyc_menu = ctk.CTkOptionMenu(
                        ecyc_ctrl,
                        values=ecyc_metrics,
                        command=_update_ecycle,
                        fg_color="#e2e8f0",
                        button_color="#0b84ff",
                        button_hover_color="#0c76e0",
                        text_color="#0f172a",
                        dropdown_fg_color="#ffffff",
                        dropdown_hover_color="#e2e8f0",
                        dropdown_text_color="#0f172a",
                    )
                    ecyc_menu.set(initial_metric)
                    ecyc_menu.pack(side="left", padx=(0, 12))

                    def _save_ecycle(_tab=tab_ecycle):
                        m = ecyc_menu.get()
                        dest = filedialog.asksaveasfilename(
                            title="Salvar Energia vs Ciclo",
                            initialfile="energia_vs_ciclo.png",
                            initialdir=self._get_initial_dialog_dir(
                                "plot_save",
                            ),
                            defaultextension=".png",
                            filetypes=[("PNG", "*.png")],
                        )
                        if not dest:
                            return
                        self._remember_dialog_dir(
                            "plot_save", dest,
                        )
                        fig = self._build_fig_energy_cycle(
                            metric=m,
                        )
                        if fig:
                            fig.savefig(
                                dest, dpi=160,
                                bbox_inches="tight",
                            )
                            plt.close(fig)
                            self._append_log(
                                f"Gráfico Energia vs Ciclo salvo"
                                f" em {dest}"
                            )

                    ctk.CTkButton(
                        ecyc_ctrl,
                        text="Salvar gráfico",
                        width=120,
                        command=_save_ecycle,
                    ).pack(side="left", padx=(0, 8))

                    ec_container = ctk.CTkFrame(tab_ecycle)
                    ec_container.pack(
                        fill="both", expand=True, padx=8, pady=8,
                    )
                    tab_ecycle._ecyc_container = ec_container
                    _update_ecycle(initial_metric)
                else:
                    ctk.CTkLabel(
                        tab_ecycle,
                        text="Sem métricas de ciclagem",
                    ).pack(pady=20)
            else:
                ctk.CTkLabel(
                    tab_ecycle,
                    text="Sem dados de ciclagem disponíveis",
                ).pack(pady=20)

            # ---- Heatmap de Impedância -----------------------------------
            if self.raw_eis:
                hm_fig = self._build_fig_impedance_heatmap()
                if hm_fig:
                    hm_frame = ctk.CTkFrame(tab_heatmap)
                    hm_frame.pack(
                        fill="both", expand=True, padx=8, pady=8,
                    )
                    self._render_fig_on_frame(hm_frame, hm_fig)

                    def _save_heatmap():
                        dest = filedialog.asksaveasfilename(
                            title="Salvar Heatmap |Z|",
                            initialfile="impedance_heatmap.png",
                            initialdir=self._get_initial_dialog_dir(
                                "plot_save",
                            ),
                            defaultextension=".png",
                            filetypes=[("PNG", "*.png")],
                        )
                        if not dest:
                            return
                        self._remember_dialog_dir(
                            "plot_save", dest,
                        )
                        fig = self._build_fig_impedance_heatmap()
                        if fig:
                            fig.savefig(
                                dest, dpi=160,
                                bbox_inches="tight",
                            )
                            plt.close(fig)
                            self._append_log(
                                f"Heatmap |Z| salvo em {dest}"
                            )

                    ctk.CTkButton(
                        tab_heatmap,
                        text="Salvar gráfico",
                        width=120,
                        command=_save_heatmap,
                    ).pack(anchor="e", padx=8, pady=(0, 4))
                else:
                    ctk.CTkLabel(
                        tab_heatmap,
                        text="Não foi possível gerar heatmap",
                    ).pack(pady=20)
            else:
                ctk.CTkLabel(
                    tab_heatmap,
                    text="Sem dados EIS para heatmap",
                ).pack(pady=20)

            # ---- Box-plot Comparativo ------------------------------------
            if (
                self.eis_df is not None
                and not self.eis_df.empty
            ):
                from src.eis_plots import _BOXPLOT_COLS

                avail_metrics = [
                    c for c in _BOXPLOT_COLS
                    if c in self.eis_df.columns
                ]
                if not avail_metrics:
                    # Fallback: use any numeric column
                    avail_metrics = [
                        c for c in self.eis_df.columns
                        if pd.api.types.is_numeric_dtype(
                            self.eis_df[c]
                        )
                    ]

                if avail_metrics:
                    initial_bp = avail_metrics[0]
                    bp_ctrl = ctk.CTkFrame(tab_boxplot)
                    bp_ctrl.pack(fill="x", padx=8, pady=(8, 4))
                    ctk.CTkLabel(
                        bp_ctrl, text="Métrica:"
                    ).pack(side="left", padx=(0, 6))

                    def _update_boxplot(
                        metric: str, _tab=tab_boxplot,
                    ):
                        container = getattr(
                            _tab, "_bp_container", None,
                        )
                        if container is None:
                            container = ctk.CTkFrame(_tab)
                            container.pack(
                                fill="both", expand=True,
                                padx=8, pady=8,
                            )
                            _tab._bp_container = container
                        fig = self._build_fig_boxplot(
                            metric=metric,
                        )
                        if fig:
                            self._render_fig_on_frame(
                                container, fig,
                            )

                    bp_menu = ctk.CTkOptionMenu(
                        bp_ctrl,
                        values=avail_metrics,
                        command=_update_boxplot,
                        fg_color="#e2e8f0",
                        button_color="#0b84ff",
                        button_hover_color="#0c76e0",
                        text_color="#0f172a",
                        dropdown_fg_color="#ffffff",
                        dropdown_hover_color="#e2e8f0",
                        dropdown_text_color="#0f172a",
                    )
                    bp_menu.set(initial_bp)
                    bp_menu.pack(side="left", padx=(0, 12))

                    def _save_boxplot(_tab=tab_boxplot):
                        m = bp_menu.get()
                        dest = filedialog.asksaveasfilename(
                            title="Salvar Box-plot",
                            initialfile=f"boxplot_{m}.png",
                            initialdir=self._get_initial_dialog_dir(
                                "plot_save",
                            ),
                            defaultextension=".png",
                            filetypes=[("PNG", "*.png")],
                        )
                        if not dest:
                            return
                        self._remember_dialog_dir(
                            "plot_save", dest,
                        )
                        fig = self._build_fig_boxplot(
                            metric=m,
                        )
                        if fig:
                            fig.savefig(
                                dest, dpi=160,
                                bbox_inches="tight",
                            )
                            plt.close(fig)
                            self._append_log(
                                f"Box-plot {m} salvo em {dest}"
                            )

                    ctk.CTkButton(
                        bp_ctrl,
                        text="Salvar gráfico",
                        width=120,
                        command=_save_boxplot,
                    ).pack(side="left", padx=(0, 8))

                    bp_container = ctk.CTkFrame(tab_boxplot)
                    bp_container.pack(
                        fill="both", expand=True, padx=8, pady=8,
                    )
                    tab_boxplot._bp_container = bp_container
                    _update_boxplot(initial_bp)
                else:
                    ctk.CTkLabel(
                        tab_boxplot,
                        text="Sem métricas numéricas para box-plot",
                    ).pack(pady=20)
            else:
                ctk.CTkLabel(
                    tab_boxplot,
                    text="Sem dados EIS para box-plot",
                ).pack(pady=20)

            # ---- Radar / Spider chart ------------------------------------
            if (
                self.eis_df is not None
                and not self.eis_df.empty
            ):
                id_col = None
                for c in ("Arquivo", "Sample"):
                    if c in self.eis_df.columns:
                        id_col = c
                        break
                all_names = (
                    sorted(self.eis_df[id_col].astype(str).unique())
                    if id_col else []
                )

                if len(all_names) >= 2:
                    radar_ctrl = ctk.CTkFrame(tab_radar)
                    radar_ctrl.pack(
                        fill="x", padx=8, pady=(8, 4),
                    )
                    ctk.CTkLabel(
                        radar_ctrl,
                        text="Amostras (separe por ';'):",
                    ).pack(side="left", padx=(0, 6))

                    # Default: first 3 (or fewer)
                    default_sel = all_names[: min(3, len(all_names))]
                    radar_entry = ctk.CTkEntry(
                        radar_ctrl, width=400,
                    )
                    radar_entry.insert(
                        0, "; ".join(default_sel),
                    )
                    radar_entry.pack(
                        side="left", fill="x",
                        expand=True, padx=(0, 8),
                    )

                    def _update_radar(
                        _tab=tab_radar,
                        _entry=radar_entry,
                        _id_col=id_col,
                    ):
                        raw = _entry.get()
                        chosen = [
                            s.strip() for s in raw.split(";")
                            if s.strip()
                        ]
                        container = getattr(
                            _tab, "_radar_container", None,
                        )
                        if container is None:
                            container = ctk.CTkFrame(_tab)
                            container.pack(
                                fill="both", expand=True,
                                padx=8, pady=8,
                            )
                            _tab._radar_container = container
                        fig = self._build_fig_radar(
                            samples=chosen,
                        )
                        if fig:
                            self._render_fig_on_frame(
                                container, fig,
                            )
                        else:
                            for ch in container.winfo_children():
                                ch.destroy()
                            ctk.CTkLabel(
                                container,
                                text=(
                                    "Selecione ao menos 2 amostras"
                                    " com ≥3 métricas numéricas"
                                ),
                            ).pack(pady=20)

                    ctk.CTkButton(
                        radar_ctrl,
                        text="Atualizar",
                        width=100,
                        command=_update_radar,
                    ).pack(side="left", padx=(0, 6))

                    def _save_radar(_tab=tab_radar):
                        raw = radar_entry.get()
                        chosen = [
                            s.strip() for s in raw.split(";")
                            if s.strip()
                        ]
                        dest = filedialog.asksaveasfilename(
                            title="Salvar Radar",
                            initialfile="radar_metrics.png",
                            initialdir=self._get_initial_dialog_dir(
                                "plot_save",
                            ),
                            defaultextension=".png",
                            filetypes=[("PNG", "*.png")],
                        )
                        if not dest:
                            return
                        self._remember_dialog_dir(
                            "plot_save", dest,
                        )
                        fig = self._build_fig_radar(
                            samples=chosen,
                        )
                        if fig:
                            fig.savefig(
                                dest, dpi=160,
                                bbox_inches="tight",
                            )
                            plt.close(fig)
                            self._append_log(
                                f"Radar salvo em {dest}"
                            )

                    ctk.CTkButton(
                        radar_ctrl,
                        text="Salvar gráfico",
                        width=120,
                        command=_save_radar,
                    ).pack(side="left", padx=(0, 8))

                    # Initial render
                    radar_container = ctk.CTkFrame(tab_radar)
                    radar_container.pack(
                        fill="both", expand=True,
                        padx=8, pady=8,
                    )
                    tab_radar._radar_container = radar_container
                    _update_radar()
                else:
                    ctk.CTkLabel(
                        tab_radar,
                        text="Precisa de ≥2 amostras para radar",
                    ).pack(pady=20)
            else:
                ctk.CTkLabel(
                    tab_radar,
                    text="Sem dados EIS para radar",
                ).pack(pady=20)

            # ---- Retenção vs Ciclo ---------------------------------------
            if self.cic_results:
                ret_fig = self._build_fig_retention_cycle()
                if ret_fig:
                    ret_frame = ctk.CTkFrame(tab_retention)
                    ret_frame.pack(
                        fill="both", expand=True,
                        padx=8, pady=8,
                    )
                    self._render_fig_on_frame(
                        ret_frame, ret_fig,
                    )

                    def _save_retention():
                        dest = filedialog.asksaveasfilename(
                            title="Salvar Retenção vs Ciclo",
                            initialfile="retention_vs_cycle.png",
                            initialdir=self._get_initial_dialog_dir(
                                "plot_save",
                            ),
                            defaultextension=".png",
                            filetypes=[("PNG", "*.png")],
                        )
                        if not dest:
                            return
                        self._remember_dialog_dir(
                            "plot_save", dest,
                        )
                        fig = (
                            self._build_fig_retention_cycle()
                        )
                        if fig:
                            fig.savefig(
                                dest, dpi=160,
                                bbox_inches="tight",
                            )
                            plt.close(fig)
                            self._append_log(
                                "Retenção vs Ciclo salvo"
                                f" em {dest}"
                            )

                    ctk.CTkButton(
                        tab_retention,
                        text="Salvar gráfico",
                        width=120,
                        command=_save_retention,
                    ).pack(
                        anchor="e", padx=8, pady=(0, 4),
                    )
                else:
                    ctk.CTkLabel(
                        tab_retention,
                        text="Sem dados suficientes para retenção",
                    ).pack(pady=20)
            else:
                ctk.CTkLabel(
                    tab_retention,
                    text="Sem dados de ciclagem disponíveis",
                ).pack(pady=20)

            # Séries com seleção
            if (
                self.eis_df is not None
                and not self.eis_df.empty
                and "Arquivo" in self.eis_df.columns
            ):
                cols = [
                    c
                    for c in ["Energia média (J)", "C_espec (F/g)"]
                    if c in self.eis_df.columns
                ]
                if not cols:
                    ctk.CTkLabel(
                        tab_series, text="Sem colunas de série disponíveis"
                    ).pack(pady=20)
                else:
                    bases = [
                        b
                        for b in self.eis_df["Arquivo"]
                        .apply(lambda x: self._split_series_name(x)[1])
                        .unique()
                        .tolist()
                        if b
                    ]
                    if not bases:
                        ctk.CTkLabel(
                            tab_series, text="Sem séries identificadas"
                        ).pack(pady=20)
                    else:
                        initial_col = cols[0]
                        if preferred_series_col in cols:
                            initial_col = preferred_series_col

                        initial_base = bases[0]
                        if preferred_series_base in bases:
                            initial_base = preferred_series_base

                        series_state = {
                            "col": initial_col,
                            "base": initial_base,
                        }

                        control = ctk.CTkFrame(tab_series)
                        control.pack(fill="x", padx=8, pady=(8, 4))
                        ctk.CTkLabel(control, text="Coluna:").pack(
                            side="left", padx=(0, 6)
                        )
                        col_menu = ctk.CTkOptionMenu(
                            control,
                            values=cols,
                            command=lambda v: self._update_series_plot(
                                tab_series, v, series_state["base"]
                            ),
                            fg_color="#e2e8f0",
                            button_color="#0b84ff",
                            button_hover_color="#0c76e0",
                            text_color="#0f172a",
                            dropdown_fg_color="#ffffff",
                            dropdown_hover_color="#e2e8f0",
                            dropdown_text_color="#0f172a",
                        )
                        col_menu.set(initial_col)
                        col_menu.pack(side="left", padx=(0, 12))
                        ctk.CTkLabel(control, text="Série:").pack(
                            side="left", padx=(0, 6)
                        )
                        base_menu = ctk.CTkOptionMenu(
                            control,
                            values=bases,
                            command=lambda v: self._update_series_plot(
                                tab_series, series_state["col"], v
                            ),
                            fg_color="#e2e8f0",
                            button_color="#0b84ff",
                            button_hover_color="#0c76e0",
                            text_color="#0f172a",
                            dropdown_fg_color="#ffffff",
                            dropdown_hover_color="#e2e8f0",
                            dropdown_text_color="#0f172a",
                        )
                        base_menu.set(initial_base)
                        base_menu.pack(side="left")

                        plot_container = ctk.CTkFrame(tab_series)
                        plot_container.pack(fill="both", expand=True, padx=8, pady=8)
                        tab_series._series_state = series_state
                        tab_series._series_container = plot_container
                        self._update_series_plot(
                            tab_series,
                            series_state["col"],
                            series_state["base"],
                        )
            else:
                ctk.CTkLabel(
                    tab_series, text="Sem dados de séries disponíveis"
                ).pack(pady=20)

            # DRT com seleção de amostra/modo
            if self.drt_results:
                drt_names = sorted(self.drt_results.keys())
                mode_values = ["Espectro", "Overlay", "Heatmap"]
                pref_mode = self.drt_ui_prefs.get("mode", "Espectro")
                initial_mode = (
                    pref_mode if pref_mode in mode_values else "Espectro"
                )

                initial_drt = drt_names[0]
                pref_sample = self.drt_ui_prefs.get("sample", "")
                if isinstance(pref_sample, str) and pref_sample:
                    normalized_pref = self._normalize_sample_name(pref_sample)
                    for name in drt_names:
                        if self._normalize_sample_name(name) == normalized_pref:
                            initial_drt = name
                            break

                if preferred_sample:
                    normalized_preferred = self._normalize_sample_name(preferred_sample)
                    for name in drt_names:
                        if self._normalize_sample_name(name) == normalized_preferred:
                            initial_drt = name
                            break

                overlay_default_text = self.drt_ui_prefs.get("overlay_text", "")
                if (
                    not isinstance(overlay_default_text, str)
                    or not overlay_default_text
                ):
                    overlay_default_text = initial_drt
                parsed_overlay, _not_found = self._parse_drt_sample_list(
                    overlay_default_text,
                    drt_names,
                )
                if not parsed_overlay:
                    parsed_overlay = [initial_drt]

                drt_state = {
                    "sample": initial_drt,
                    "mode": initial_mode,
                    "overlay_samples": parsed_overlay,
                }

                drt_control = ctk.CTkFrame(tab_drt)
                drt_control.pack(fill="x", padx=8, pady=(8, 4))
                ctk.CTkLabel(drt_control, text="Amostra:").pack(
                    side="left", padx=(0, 6)
                )
                drt_sample_menu = ctk.CTkOptionMenu(
                    drt_control,
                    values=drt_names,
                    command=lambda v: self._update_drt_plot(
                        tab_drt,
                        v,
                        drt_state["mode"],
                    ),
                    fg_color="#e2e8f0",
                    button_color="#0b84ff",
                    button_hover_color="#0c76e0",
                    text_color="#0f172a",
                    dropdown_fg_color="#ffffff",
                    dropdown_hover_color="#e2e8f0",
                    dropdown_text_color="#0f172a",
                )
                drt_sample_menu.set(initial_drt)
                drt_sample_menu.pack(side="left", padx=(0, 12))

                ctk.CTkLabel(drt_control, text="Modo:").pack(
                    side="left", padx=(0, 6)
                )
                drt_mode_menu = ctk.CTkOptionMenu(
                    drt_control,
                    values=mode_values,
                    command=lambda v: self._update_drt_plot(
                        tab_drt,
                        drt_state["sample"],
                        v,
                    ),
                    fg_color="#e2e8f0",
                    button_color="#0b84ff",
                    button_hover_color="#0c76e0",
                    text_color="#0f172a",
                    dropdown_fg_color="#ffffff",
                    dropdown_hover_color="#e2e8f0",
                    dropdown_text_color="#0f172a",
                )
                drt_mode_menu.set(initial_mode)
                drt_mode_menu.pack(side="left")

                ctk.CTkLabel(drt_control, text="Overlay (vírgula):").pack(
                    side="left", padx=(12, 6)
                )
                overlay_entry = ctk.CTkEntry(
                    drt_control,
                    width=260,
                    placeholder_text="sample1, sample2",
                )
                overlay_entry.insert(0, ", ".join(parsed_overlay))
                overlay_entry.pack(side="left", padx=(0, 8))

                def _apply_overlay_selection():
                    parsed, not_found = self._parse_drt_sample_list(
                        overlay_entry.get(),
                        drt_names,
                    )
                    if not parsed:
                        parsed = [drt_state["sample"]]
                        overlay_entry.delete(0, "end")
                        overlay_entry.insert(0, drt_state["sample"])
                        self._append_log(
                            "DRT overlay vazio; usando amostra atual como fallback."
                        )

                    drt_state["overlay_samples"] = parsed
                    self._update_drt_plot(
                        tab_drt,
                        drt_state["sample"],
                        drt_state["mode"],
                        overlay_samples=parsed,
                    )
                    self.drt_ui_prefs["overlay_text"] = ", ".join(parsed)
                    if parsed:
                        self._append_log(
                            f"DRT overlay: {len(parsed)} amostra(s) selecionada(s)."
                        )
                    if not_found:
                        self._append_log(
                            "DRT overlay não encontrou: " + ", ".join(not_found)
                        )

                ctk.CTkButton(
                    drt_control,
                    text="Aplicar overlay",
                    width=120,
                    command=_apply_overlay_selection,
                ).pack(side="left", padx=(0, 8))

                def _set_overlay_top_n(n: int):
                    chosen = drt_names[:n]
                    overlay_entry.delete(0, "end")
                    overlay_entry.insert(0, ", ".join(chosen))
                    drt_state["overlay_samples"] = chosen
                    self.drt_ui_prefs["overlay_text"] = ", ".join(chosen)
                    self._update_drt_plot(
                        tab_drt,
                        drt_state["sample"],
                        drt_state["mode"],
                        overlay_samples=chosen,
                    )

                ctk.CTkButton(
                    drt_control,
                    text="Top 5",
                    width=64,
                    command=lambda: _set_overlay_top_n(5),
                ).pack(side="left", padx=(0, 6))

                ctk.CTkButton(
                    drt_control,
                    text="Todos",
                    width=64,
                    command=lambda: _set_overlay_top_n(len(drt_names)),
                ).pack(side="left", padx=(0, 8))

                ctk.CTkButton(
                    drt_control,
                    text="Salvar visual",
                    width=110,
                    command=lambda: self._save_drt_current_view(tab_drt),
                ).pack(side="left")

                tab_drt._drt_state = drt_state

                def _on_drt_sample_change(value: str):
                    drt_state["sample"] = value
                    self.drt_ui_prefs["sample"] = value
                    self._update_drt_plot(
                        tab_drt,
                        value,
                        drt_state["mode"],
                        overlay_samples=drt_state["overlay_samples"],
                    )

                def _on_drt_mode_change(value: str):
                    drt_state["mode"] = value
                    self.drt_ui_prefs["mode"] = value
                    self._update_drt_plot(
                        tab_drt,
                        drt_state["sample"],
                        value,
                        overlay_samples=drt_state["overlay_samples"],
                    )

                drt_sample_menu.configure(command=_on_drt_sample_change)
                drt_mode_menu.configure(command=_on_drt_mode_change)

                drt_container = ctk.CTkFrame(tab_drt)
                drt_container.pack(fill="both", expand=True, padx=8, pady=8)
                tab_drt._drt_container = drt_container
                self._update_drt_plot(
                    tab_drt,
                    drt_state["sample"],
                    drt_state["mode"],
                    overlay_samples=drt_state["overlay_samples"],
                )
            else:
                ctk.CTkLabel(tab_drt, text="Sem resultados DRT disponíveis").pack(
                    pady=20
                )

            if isinstance(preferred_tab, str):
                with contextlib.suppress(Exception):
                    tabs.set(preferred_tab)

            if preferred_sample and rank_idx is None and pca_idx is None:
                self._append_log(
                    "Amostra selecionada não encontrada em Rank/PCA; "
                    "verifique o nome da amostra ou use a aba Séries."
                )

            win.after(50, lambda: (win.lift(), win.focus_force()))
        except Exception as exc:
            self._append_log(f"Erro ao abrir janela interativa: {exc}")
            traceback.print_exc()
            self._destroy_interactive()

    def _update_series_plot(self, tab_series, col_value: str, base_value: str):
        tab_series._series_state = {"col": col_value, "base": base_value}
        fig = self._build_fig_series(col_value, base_value)
        container = getattr(tab_series, "_series_container", None)
        if container is None:
            container = ctk.CTkFrame(tab_series)
            container.pack(fill="both", expand=True, padx=8, pady=8)
            tab_series._series_container = container
        if fig:
            self._render_fig_on_frame(container, fig)
        else:
            for child in container.winfo_children():
                child.destroy()
            ctk.CTkLabel(container, text="Sem dados para esta série").pack(pady=20)

    def _import_files(self, target_dir: str, label: str):
        paths = filedialog.askopenfilenames(
            title=label,
            initialdir=self._get_initial_dialog_dir("import"),
        )
        if not paths:
            return
        self._remember_dialog_dir("import", paths[0])
        os.makedirs(target_dir, exist_ok=True)
        copied = 0
        for src in paths:
            try:
                shutil.copy(src, target_dir)
                copied += 1
            except Exception as exc:
                self._append_log(f"Falha ao copiar {src}: {exc}")
        self._append_log(f"{copied} arquivo(s) copiado(s) para {target_dir}")

    def _open_rank_interactive(self):
        if self.rank_df is None or self.rank_df.empty:
            self._append_log("Sem dados de rank vs retenção para exibir.")
            return
        if (
            "Rank" not in self.rank_df.columns
            or "Retenção (%)" not in self.rank_df.columns
        ):
            self._append_log("Dados de Rank/Retenção ausentes.")
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        points = ax.scatter(
            self.rank_df["Rank"],
            self.rank_df["Retenção (%)"],
            c="#1f77b4",
            alpha=0.8,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.set_xlabel("Rank")
        ax.set_ylabel("Retenção (%)")
        ax.set_title("Rank vs Retenção (interativo)")
        ax.grid(True, alpha=0.3)
        try:
            import mplcursors

            cursor = mplcursors.cursor(points, hover=True)
            labels = list(self.rank_df.index.astype(str))

            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                label = labels[idx] if idx < len(labels) else ""
                x, y = sel.target
                sel.annotation.set_text(f"{label}\nRank: {x:.2f}\nRetenção: {y:.2f}%")
        except ImportError:
            self._append_log("mplcursors não encontrado; hover desabilitado.")
        fig.tight_layout()
        fig.show()

    def _open_pca_interactive(self):
        if self.df_pca is None or self.df_pca.empty:
            self._append_log("Sem dados de PCA para exibir.")
            return
        labels = None
        if self.rank_df is not None and "Subclasse" in self.rank_df.columns:
            labels = self.rank_df["Subclasse"].reindex(self.df_pca.index)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            self.df_pca.get("PC1", []),
            self.df_pca.get("PC2", []),
            c="tab:blue" if labels is None else None,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA 2D (interativo)")
        ax.grid(True, alpha=0.3)
        try:
            import mplcursors

            cursor = mplcursors.cursor(scatter, hover=True)
            names = list(self.df_pca.index.astype(str))
            lbls = list(
                labels.fillna("") if labels is not None else ["" for _ in names]
            )

            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                name = names[idx] if idx < len(names) else ""
                lbl = lbls[idx] if idx < len(lbls) else ""
                x, y = sel.target
                extra = f"\nSubclasse: {lbl}" if lbl else ""
                sel.annotation.set_text(f"{name}\nPC1: {x:.2f}\nPC2: {y:.2f}{extra}")
        except ImportError:
            self._append_log("mplcursors não encontrado; hover desabilitado.")
        fig.tight_layout()
        fig.show()

    def _open_pca_metric_interactive(self):
        if self.df_pca is None or self.df_pca.empty:
            self._append_log("Sem dados de PCA para exibir.")
            return
        retention = None
        if self.rank_df is not None and "Retenção (%)" in self.rank_df.columns:
            retention = self.rank_df["Retenção (%)"].reindex(self.df_pca.index)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            self.df_pca.get("PC1", []),
            self.df_pca.get("PC2", []),
            c=retention if retention is not None else "tab:blue",
            cmap="viridis" if retention is not None else None,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
        )
        if retention is not None:
            fig.colorbar(scatter, ax=ax, label="Retenção (%)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA 2D - métrica (interativo)")
        ax.grid(True, alpha=0.3)
        try:
            import mplcursors

            cursor = mplcursors.cursor(scatter, hover=True)
            names = list(self.df_pca.index.astype(str))
            vals = list(retention if retention is not None else [None for _ in names])

            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                name = names[idx] if idx < len(names) else ""
                x, y = sel.target
                val = vals[idx] if idx < len(vals) else None
                metric = (
                    f"\nRetenção: {val:.2f}%"
                    if val is not None and not pd.isna(val)
                    else ""
                )
                sel.annotation.set_text(f"{name}\nPC1: {x:.2f}\nPC2: {y:.2f}{metric}")
        except ImportError:
            self._append_log("mplcursors não encontrado; hover desabilitado.")
        fig.tight_layout()
        fig.show()

    def _open_corr_interactive(self):
        if self.rank_df is None or self.rank_df.empty:
            self._append_log("Sem dados para correlação.")
            return
        df = self.rank_df.select_dtypes(include=["number"]).copy()
        df = df.dropna(how="all")
        if df.empty or df.shape[1] < 2:
            self._append_log("Dados insuficientes para correlação.")
            return
        corr = df.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        fig.colorbar(im, ax=ax, label="Spearman ρ")
        try:
            import mplcursors

            cursor = mplcursors.cursor(im, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                j, i = int(round(sel.target[0])), int(round(sel.target[1]))
                if 0 <= i < corr.shape[0] and 0 <= j < corr.shape[1]:
                    val = corr.iat[i, j]
                    row = corr.index[i]
                    col = corr.columns[j]
                    sel.annotation.set_text(f"{row} x {col}\nρ = {val:.2f}")
        except ImportError:
            self._append_log("mplcursors não encontrado; hover desabilitado.")
        fig.tight_layout()
        fig.show()

    def _open_series_interactive(self, path: str):
        if self.eis_df is None or self.eis_df.empty:
            self._append_log("Sem dados de séries para exibir.")
            return
        fname = os.path.basename(path)
        if not fname.startswith("series_"):
            self._append_log("Formato de série não reconhecido.")
            return
        parts = fname.replace("series_", "", 1).rsplit(".", 1)[0].split("_")
        if len(parts) < 2:
            self._append_log("Formato de série não reconhecido.")
            return
        value_col = parts[0].replace("_", " ")
        base_name = "_".join(parts[1:]).replace("_", " ")
        if value_col not in self.eis_df.columns:
            self._append_log(f"Coluna {value_col} não encontrada na tabela.")
            return

        def _split(name: str):
            parts_local = str(name).replace(".txt", "").split()
            if not parts_local:
                return (0.0, name)
            try:
                return (
                    float(parts_local[0]),
                    " ".join(parts_local[1:]) if len(parts_local) > 1 else name,
                )
            except ValueError:
                return (0.0, name)

        df = self.eis_df.copy()
        info = df["Arquivo"].apply(_split)
        df["_lead"] = info.apply(lambda x: x[0])
        df["_base"] = info.apply(lambda x: x[1])
        grp = df[df["_base"] == base_name].sort_values("_lead")
        if grp.empty or grp[value_col].dropna().empty:
            self._append_log("Sem pontos para esta série.")
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot(grp["_lead"], grp[value_col], "o-", color="#1f77b4")
        ax.set_xlabel("Prefixo numérico")
        ax.set_ylabel(value_col)
        ax.set_title(f"{value_col} - {base_name} (interativo)")
        ax.grid(True, alpha=0.3)
        try:
            import mplcursors

            cursor = mplcursors.cursor(line, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                x, y = sel.target
                sel.annotation.set_text(f"Prefixo: {x:.2f}\nValor: {y:.2f}")
        except ImportError:
            self._append_log("mplcursors não encontrado; hover desabilitado.")
        fig.tight_layout()
        fig.show()

    def _add_plot(self, title: str, path: str):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            self._append_log(f"Imagem não encontrada: {abs_path}")
            return
        try:
            img = Image.open(abs_path)
        except Exception as exc:
            self._append_log(f"Falha ao abrir imagem {abs_path}: {exc}")
            return

        max_w, max_h = 900, 600
        img.thumbnail((max_w, max_h))
        ctk_img = ctk.CTkImage(img, size=(img.width, img.height))
        self.image_refs.append(ctk_img)

        container = ctk.CTkFrame(self.plots_frame)
        container.pack(fill="both", expand=False, pady=10, padx=4)

        label = ctk.CTkLabel(
            container,
            text=title,
            image=ctk_img,
            compound="top",
            font=ctk.CTkFont(size=14, weight="bold"),
            padx=8,
            pady=12,
        )
        label.pack(fill="both", expand=False)

        btn_frame = ctk.CTkFrame(container)
        btn_frame.pack(fill="x", pady=(6, 4))

        save_btn = ctk.CTkButton(
            btn_frame,
            text="Salvar imagem",
            command=lambda p=abs_path: self._save_plot(p),
        )
        save_btn.pack(side="left", padx=4)

        inter_cmd = None
        name = os.path.basename(path).lower()
        if "rank_vs_retencao" in name:
            inter_cmd = self._open_rank_interactive
        elif "pca_2d_metric" in name:
            inter_cmd = self._open_pca_metric_interactive
        elif "pca_2d" in name:
            inter_cmd = self._open_pca_interactive
        elif "correlation_heatmap" in name:
            inter_cmd = self._open_corr_interactive
        elif name.endswith("_drt.png"):
            stem = name.replace("_drt.png", "")

            def _open_drt_tab(sample=stem):
                self._open_interactive_window(
                    preferred_tab="DRT",
                    preferred_sample=sample,
                )

            inter_cmd = _open_drt_tab
        elif name.startswith("series_"):
            def _open_series_plot(p=path):
                self._open_series_interactive(p)

            inter_cmd = _open_series_plot
        elif "energy_power" in name:
            stem = name.replace("_energy_power.png", "")

            def _open_ep_tab(sample=stem):
                self._open_interactive_window(
                    preferred_tab="Energia × Potência",
                    preferred_sample=sample,
                )

            inter_cmd = _open_ep_tab

        inter_btn = ctk.CTkButton(
            btn_frame,
            text="Ver interativo",
            command=(
                inter_cmd
                if inter_cmd is not None
                else self._log_interactive_unavailable
            ),
        )
        inter_btn.pack(side="left", padx=4)

    def _display_dataframe(self, tree: ttk.Treeview, df: Optional[pd.DataFrame]):
        # Mantido por compatibilidade; preferir _set_table_data().
        for row_id in tree.get_children():
            tree.delete(row_id)
        if df is None or df.empty:
            tree["columns"] = ["info"]
            tree.heading("info", text="info")
            tree.column("info", width=280, anchor="w")
            tree.insert("", "end", values=["Nenhum dado disponível."])
            return
        tree["columns"] = list(df.columns)
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=160, anchor="center", stretch=True)
        for row in df.itertuples(index=False, name=None):
            tree.insert("", "end", values=list(row))

    def _run_eis_clicked(self):
        self._disable_buttons()
        self._set_status("rodando EIS")
        self._start_progress("Identificando amostras EIS...")
        self._clear_plots()

        def worker():
            qwriter = QueueWriter(self.log_queue)
            try:
                self.log_queue.put(("stage", "Calculando valores EIS..."))
                with contextlib.redirect_stdout(
                    qwriter
                ), contextlib.redirect_stderr(qwriter):
                    result = run_eis_pipeline()
                self.log_queue.put(("stage", "Gerando tabelas e gráficos..."))
                self.log_queue.put(("eis_done", result))
            except Exception:
                self.log_queue.put(("log", traceback.format_exc()))
                self.log_queue.put(("eis_done", None))

        threading.Thread(target=worker, daemon=True).start()

    def _run_ciclagem_clicked(self):
        self._disable_buttons()
        self._set_status("rodando Ciclagem")
        self._start_progress("Identificando ciclos...")
        self._clear_plots()

        def worker():
            qwriter = QueueWriter(self.log_queue)
            try:
                scan_rate = float(self.scan_rate_entry.get().strip())
                self.log_queue.put(("stage", "Calculando valores de energia..."))
                with contextlib.redirect_stdout(
                    qwriter
                ), contextlib.redirect_stderr(qwriter):
                    result = run_ciclagem_pipeline(scan_rate, show_plots=False)
                self.log_queue.put(("stage", "Gerando gráficos e tabelas..."))
                self.log_queue.put(("cic_done", result))
            except Exception:
                self.log_queue.put(("log", traceback.format_exc()))
                self.log_queue.put(("cic_done", None))

        threading.Thread(target=worker, daemon=True).start()

    def _run_both_clicked(self):
        self._disable_buttons()
        self._set_status("rodando ambos")
        self._start_progress("Identificando dados EIS e ciclos...")
        self._clear_plots()

        def worker():
            qwriter = QueueWriter(self.log_queue)
            eis_result = None
            cic_result = None
            try:
                scan_rate = float(self.scan_rate_entry.get().strip())
                self.log_queue.put(("stage", "Calculando EIS..."))
                with contextlib.redirect_stdout(
                    qwriter
                ), contextlib.redirect_stderr(qwriter):
                    eis_result = run_eis_pipeline()
                self.log_queue.put(("stage", "Calculando ciclagem..."))
                with contextlib.redirect_stdout(
                    qwriter
                ), contextlib.redirect_stderr(qwriter):
                    cic_result = run_ciclagem_pipeline(scan_rate, show_plots=False)
                self.log_queue.put(("stage", "Gerando gráficos e tabelas..."))
            except Exception:
                self.log_queue.put(("log", traceback.format_exc()))
            finally:
                if eis_result is not None and cic_result is not None:
                    self.log_queue.put(("both_done", (eis_result, cic_result)))
                else:
                    self.log_queue.put(("both_done", None))

        threading.Thread(target=worker, daemon=True).start()

    def _run_drt_clicked(self):
        self._disable_buttons()
        self._set_status("rodando DRT")
        self._start_progress("Calculando DRT...")
        self._clear_plots()

        def worker():
            qwriter = QueueWriter(self.log_queue)
            try:
                lambda_reg, n_taus = self._read_drt_params()
                preset_name = self.drt_preset_selector.get()

                self.log_queue.put(("stage", "Executando inversão DRT..."))
                self.log_queue.put(
                    (
                        "log",
                        "DRT parâmetros: "
                        f"preset={preset_name}, λ={lambda_reg:.2e}, "
                        f"n_taus={n_taus}",
                    )
                )
                with contextlib.redirect_stdout(
                    qwriter
                ), contextlib.redirect_stderr(qwriter):
                    result = run_drt_pipeline(
                        lambda_reg=lambda_reg,
                        n_taus=n_taus,
                        show_plots=False,
                    )
                self.log_queue.put(("stage", "Organizando resultados DRT..."))
                self.log_queue.put(("drt_done", result))
            except Exception:
                self.log_queue.put(("log", traceback.format_exc()))
                self.log_queue.put(("drt_done", None))

        threading.Thread(target=worker, daemon=True).start()

    def _process_queue(self):
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break

            try:
                msg_type = item[0]
                if msg_type == "log":
                    self._append_log(item[1])
                elif msg_type == "stage":
                    self._update_progress(item[1])
                elif msg_type == "eis_done":
                    self._handle_eis_done(item[1])
                elif msg_type == "cic_done":
                    self._handle_cic_done(item[1])
                elif msg_type == "both_done":
                    self._handle_both_done(item[1])
                elif msg_type == "drt_done":
                    self._handle_drt_done(item[1])
            except Exception as exc:
                # Não deixar o loop quebrar; logar e continuar
                self._append_log(f"Erro no processamento da fila: {exc}")
                traceback.print_exc()

        self.after(100, self._process_queue)

    def _handle_eis_done(self, result: Optional[dict]):
        if result is None:
            self._set_status("erro no EIS")
            self._stop_progress("Erro")
            self._enable_buttons()
            return

        cap_df = result.get("cap_energy")
        cap_display = None
        if cap_df is not None:
            cap_display = cap_df.reset_index().rename(columns={"index": "Arquivo"})
        self.eis_df = cap_display
        self.rank_df = result.get("df_ranked")
        self.df_pca = result.get("df_pca")
        self.raw_eis = result.get("raw_eis") or {}
        self._set_table_data("eis", self.eis_df)

        self.circuit_df = result.get("circuit_table")
        self._set_table_data("circuit", self.circuit_df)
        self._update_drt_eis_join_table()

        for path in result.get("pca_paths", []):
            title = "PCA" if "pca" in path.lower() else "Gráfico"
            self._add_plot(title, path)

        self._set_status("EIS concluído")
        self._stop_progress("EIS concluído")
        self._enable_buttons()
        # Atualiza janela interativa se estiver aberta
        if self.interactive_win is not None and self.interactive_win.winfo_exists():
            self._open_interactive_window()

    def _handle_cic_done(self, result: Optional[dict]):
        if result is None:
            self._set_status("erro na Ciclagem")
            self._stop_progress("Erro")
            self._enable_buttons()
            return

        self.cic_df = result.get("merged_table")
        self.cic_results = result.get("results") or {}
        self._set_table_data("cic", self.cic_df)
        self.cic_plot_map = {}

        for filename, path in result.get("plot_paths", []):
            key = self._normalize_sample_name(filename)
            self.cic_plot_map[key] = os.path.abspath(path)
            self._add_plot(f"{filename} - Integral", path)

        for filename, path in result.get("energy_power_paths", []):
            self._add_plot(f"{filename} - Energia×Potência", path)

        n_files = len(self.cic_results)
        self._append_log(
            f"Ciclagem: {n_files} arquivo(s) processado(s)."
        )

        self._set_status("Ciclagem concluída")
        self._stop_progress("Ciclagem concluída")
        self._enable_buttons()
        if (
            self.interactive_win is not None
            and self.interactive_win.winfo_exists()
        ):
            self._open_interactive_window(
                preferred_tab="Energia × Potência",
            )

    def _handle_both_done(self, result: Optional[Tuple[dict, dict]]):
        if result is None:
            self._set_status("erro ao rodar ambos")
            self._stop_progress("Erro")
            self._enable_buttons()
            return

        eis_result, cic_result = result
        cap_df = eis_result.get("cap_energy")
        cap_display = None
        if cap_df is not None:
            cap_display = cap_df.reset_index().rename(columns={"index": "Arquivo"})
        self.eis_df = cap_display
        self.rank_df = eis_result.get("df_ranked")
        self.df_pca = eis_result.get("df_pca")
        self.raw_eis = eis_result.get("raw_eis") or {}
        self.cic_df = cic_result.get("merged_table")
        self.cic_results = cic_result.get("results") or {}
        self.circuit_df = eis_result.get("circuit_table")

        self._set_table_data("eis", self.eis_df)
        self._set_table_data("cic", self.cic_df)
        self._set_table_data("circuit", self.circuit_df)
        self._update_drt_eis_join_table()
        self.cic_plot_map = {}

        for path in eis_result.get("pca_paths", []):
            title = "PCA" if "pca" in path.lower() else "Gráfico"
            self._add_plot(title, path)
        for filename, path in cic_result.get("plot_paths", []):
            key = self._normalize_sample_name(filename)
            self.cic_plot_map[key] = os.path.abspath(path)
            self._add_plot(f"{filename} - Integral", path)
        for filename, path in cic_result.get("energy_power_paths", []):
            self._add_plot(f"{filename} - Energia×Potência", path)

        self._set_status("Ambos concluídos")
        self._stop_progress("Ambos concluídos")
        self._enable_buttons()
        if self.interactive_win is not None and self.interactive_win.winfo_exists():
            self._open_interactive_window()

    def _handle_drt_done(self, result: Optional[dict]):
        if result is None:
            self._set_status("erro no DRT")
            self._stop_progress("Erro")
            self._enable_buttons()
            return

        self.drt_df = result.get("drt_table")
        self.drt_peaks_df = result.get("drt_peaks_table")
        self.drt_summary_df = result.get("drt_summary_table")
        self.drt_results = result.get("per_file_results", {}) or {}
        self._set_table_data("drt", self.drt_df)
        self._set_table_data("drt_peaks", self.drt_peaks_df)
        self._update_drt_eis_join_table()
        self.drt_plot_map = {}

        for filename, path in result.get("plot_paths", []):
            key = self._normalize_sample_name(filename)
            self.drt_plot_map[key] = os.path.abspath(path)
            self._add_plot(f"{filename} - DRT", path)

        errors = result.get("errors", {}) or {}
        run_meta = result.get("run_meta", {}) or {}
        if errors:
            self._append_log(
                f"DRT concluído com {len(errors)} falha(s): "
                + ", ".join(errors.keys())
            )

        if run_meta:
            self._append_log(
                "DRT meta: "
                f"arquivos={run_meta.get('n_files', 0)}, "
                f"ok={run_meta.get('n_success', 0)}, "
                f"falhas={run_meta.get('n_failed', 0)}, "
                f"λ={run_meta.get('lambda_reg', float('nan')):.2e}, "
                f"n_taus={run_meta.get('n_taus', 0)}"
            )

        if self.drt_summary_df is not None and not self.drt_summary_df.empty:
            best_row = self.drt_summary_df.sort_values(
                by="gamma_peak_main",
                ascending=False,
                na_position="last",
            ).iloc[0]
            self._append_log(
                "DRT resumo: "
                f"amostras={len(self.drt_summary_df)}, "
                f"maior pico={best_row.get('Sample', '')} "
                f"(γ={best_row.get('gamma_peak_main', float('nan')):.3f})"
            )

        self._set_status("DRT concluído")
        self._stop_progress("DRT concluído")
        self._enable_buttons()
        if self.interactive_win is not None and self.interactive_win.winfo_exists():
            self._open_interactive_window(preferred_tab="DRT")


def main():
    """Entry-point for the GUI application."""
    app = PipelineApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
