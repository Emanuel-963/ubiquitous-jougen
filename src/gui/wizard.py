"""Quick Start wizard shown on first launch.

UX-01: A 3-step guided dialog that helps new researchers configure the
essential settings without reading documentation:

1. Choose interface language (PT / EN / ES)
2. Select or confirm data directory
3. Choose analysis type (EIS / Cycling / Both) and material preset

The wizard stores a flag in ``ionflow_settings.json`` so it only appears
once (or can be reset from the Settings menu).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_WIZARD_FLAG = "wizard_completed"


@dataclass
class WizardResult:
    """Outcome of the Quick Start wizard.

    Attributes
    ----------
    language : str
        Selected language code ('pt', 'en', 'es').
    data_dir : str
        Chosen data directory path.
    pipeline : str
        Selected pipeline ('eis', 'cycling', 'both').
    material_preset : str
        Selected material preset.
    completed : bool
        Whether the user completed the wizard (False if cancelled).
    """

    language: str = "pt"
    data_dir: str = "data/raw"
    pipeline: str = "eis"
    material_preset: str = "generic"
    completed: bool = False


def should_show_wizard(settings_path: str = "ionflow_settings.json") -> bool:
    """Check whether the wizard should be shown.

    Returns True if the wizard has never been completed (flag not in settings).

    Parameters
    ----------
    settings_path : str
        Path to the settings JSON file.

    Returns
    -------
    bool
        True if wizard should be displayed.
    """
    try:
        if os.path.exists(settings_path):
            with open(settings_path, encoding="utf-8") as f:
                data = json.load(f)
            return not data.get(_WIZARD_FLAG, False)
    except (json.JSONDecodeError, OSError):
        pass
    return True


def mark_wizard_completed(settings_path: str = "ionflow_settings.json") -> None:
    """Mark the wizard as completed in settings.

    Parameters
    ----------
    settings_path : str
        Path to the settings JSON file.
    """
    data = {}
    try:
        if os.path.exists(settings_path):
            with open(settings_path, encoding="utf-8") as f:
                data = json.load(f)
    except (json.JSONDecodeError, OSError):
        pass

    data[_WIZARD_FLAG] = True
    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.warning("Could not save wizard flag: %s", exc)


def run_wizard_gui(parent) -> WizardResult:
    """Display the Quick Start wizard as a modal dialog.

    Parameters
    ----------
    parent : ctk.CTk or ctk.CTkToplevel
        Parent window for the dialog.

    Returns
    -------
    WizardResult
        The user's selections, or a default result with ``completed=False``
        if the wizard was cancelled.
    """
    import customtkinter as ctk
    from tkinter import filedialog

    result = WizardResult()

    dlg = ctk.CTkToplevel(parent)
    dlg.title("🚀 IonFlow — Quick Start")
    dlg.geometry("550x480")
    dlg.resizable(False, False)
    dlg.grab_set()

    # ── State ────────────────────────────────────────────────────────
    current_step = [0]
    frames: list = []

    container = ctk.CTkFrame(dlg)
    container.pack(fill="both", expand=True, padx=16, pady=16)

    # ── Header ───────────────────────────────────────────────────────
    header_label = ctk.CTkLabel(
        container,
        text="Bem-vindo ao IonFlow Pipeline!",
        font=ctk.CTkFont(size=18, weight="bold"),
    )
    header_label.pack(pady=(8, 4))

    subtitle_label = ctk.CTkLabel(
        container,
        text="Configure o essencial em 3 passos simples.",
        font=ctk.CTkFont(size=12),
        text_color="gray60",
    )
    subtitle_label.pack(pady=(0, 12))

    # Progress indicator
    progress_frame = ctk.CTkFrame(container, fg_color="transparent")
    progress_frame.pack(fill="x", padx=20, pady=(0, 12))
    step_labels = []
    for i, txt in enumerate(["Idioma", "Dados", "Análise"]):
        lbl = ctk.CTkLabel(
            progress_frame, text=f"{'●' if i == 0 else '○'} {txt}",
            font=ctk.CTkFont(size=11),
        )
        lbl.pack(side="left", expand=True)
        step_labels.append(lbl)

    # ── Content area ─────────────────────────────────────────────────
    content_frame = ctk.CTkFrame(container)
    content_frame.pack(fill="both", expand=True, padx=8, pady=8)

    # Step 1: Language
    step1 = ctk.CTkFrame(content_frame, fg_color="transparent")
    ctk.CTkLabel(step1, text="Escolha o idioma da interface:", font=ctk.CTkFont(size=13)).pack(pady=(20, 10))
    lang_var = ctk.StringVar(value="pt")
    for code, label in [("pt", "🇧🇷 Português"), ("en", "🇬🇧 English"), ("es", "🇪🇸 Español")]:
        ctk.CTkRadioButton(step1, text=label, variable=lang_var, value=code).pack(pady=4)
    frames.append(step1)

    # Step 2: Data directory
    step2 = ctk.CTkFrame(content_frame, fg_color="transparent")
    ctk.CTkLabel(step2, text="Onde estão seus dados EIS?", font=ctk.CTkFont(size=13)).pack(pady=(20, 10))
    dir_var = ctk.StringVar(value=str(Path("data/raw").resolve()))
    dir_entry = ctk.CTkEntry(step2, textvariable=dir_var, width=380)
    dir_entry.pack(pady=4)

    def _browse():
        d = filedialog.askdirectory(title="Selecionar pasta de dados")
        if d:
            dir_var.set(d)

    ctk.CTkButton(step2, text="📁 Procurar...", command=_browse, width=140).pack(pady=8)
    ctk.CTkLabel(
        step2, text="(Coloque seus arquivos .csv/.txt/.dta/.mpr nesta pasta)",
        font=ctk.CTkFont(size=10), text_color="gray60",
    ).pack(pady=4)
    frames.append(step2)

    # Step 3: Pipeline + preset
    step3 = ctk.CTkFrame(content_frame, fg_color="transparent")
    ctk.CTkLabel(step3, text="Tipo de análise:", font=ctk.CTkFont(size=13)).pack(pady=(20, 8))
    pipeline_var = ctk.StringVar(value="eis")
    for val, label in [("eis", "⚡ EIS (Impedância)"), ("cycling", "🔄 Ciclagem"), ("both", "📊 Ambos")]:
        ctk.CTkRadioButton(step3, text=label, variable=pipeline_var, value=val).pack(pady=3)

    ctk.CTkLabel(step3, text="\nTipo de material:", font=ctk.CTkFont(size=13)).pack(pady=(8, 6))
    preset_var = ctk.StringVar(value="generic")
    preset_menu = ctk.CTkOptionMenu(
        step3, variable=preset_var, width=280,
        values=["generic", "supercapacitor", "li_ion", "corrosion_coating", "fuel_cell"],
    )
    preset_menu.pack(pady=4)
    ctk.CTkLabel(
        step3, text="(Define parâmetros otimizados para seu tipo de sistema)",
        font=ctk.CTkFont(size=10), text_color="gray60",
    ).pack(pady=2)
    frames.append(step3)

    # Show first step
    frames[0].pack(fill="both", expand=True)

    # ── Navigation buttons ───────────────────────────────────────────
    btn_frame = ctk.CTkFrame(container, fg_color="transparent")
    btn_frame.pack(fill="x", pady=(8, 0))

    def _update_step():
        for i, f in enumerate(frames):
            f.pack_forget()
        frames[current_step[0]].pack(fill="both", expand=True)
        for i, lbl in enumerate(step_labels):
            marker = "●" if i == current_step[0] else "✓" if i < current_step[0] else "○"
            lbl.configure(text=f"{marker} {['Idioma', 'Dados', 'Análise'][i]}")
        back_btn.configure(state="normal" if current_step[0] > 0 else "disabled")
        next_btn.configure(text="Começar! 🚀" if current_step[0] == 2 else "Próximo →")

    def _next():
        if current_step[0] < 2:
            current_step[0] += 1
            _update_step()
        else:
            # Finish
            result.language = lang_var.get()
            result.data_dir = dir_var.get()
            result.pipeline = pipeline_var.get()
            result.material_preset = preset_var.get()
            result.completed = True
            dlg.destroy()

    def _back():
        if current_step[0] > 0:
            current_step[0] -= 1
            _update_step()

    def _skip():
        result.completed = False
        dlg.destroy()

    back_btn = ctk.CTkButton(btn_frame, text="← Voltar", command=_back, width=100, state="disabled")
    back_btn.pack(side="left", padx=4)

    ctk.CTkButton(btn_frame, text="Pular", command=_skip, width=80, fg_color="gray40").pack(side="left", padx=4)

    next_btn = ctk.CTkButton(btn_frame, text="Próximo →", command=_next, width=120)
    next_btn.pack(side="right", padx=4)

    # Wait for dialog to close
    dlg.wait_window()
    return result
