"""CustomTkinter wizard for configuring Scientific Protocol projects."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from typing import Callable

import customtkinter as ctk

from .project_store import (
    add_cell,
    duplicate_cell,
    new_project,
    save_project,
    validate_project,
)


class ScientificProtocolWizard(ctk.CTkToplevel):
    """Guided project editor that persists the protocol JSON transparently."""

    def __init__(
        self,
        parent,
        *,
        project_root: str | Path,
        project: dict | None = None,
        on_saved: Callable | None = None,
    ):
        super().__init__(parent)
        self.parent = parent
        self.project_root = Path(project_root).resolve()
        self.project = copy.deepcopy(project) if project else None
        self.on_saved = on_saved
        self.current_electrolyte = (
            next(iter(self.project.get("electrolytes", {})), "") if self.project else ""
        )
        self.current_cell = next(iter(self._cells()), "") if self.project else ""
        self.page_index = 0
        self.cv_rows: list[tuple[ctk.CTkEntry, ctk.CTkLabel]] = []
        self.eis_vars = {}
        self.sequence_entries: list[ctk.CTkEntry] = []
        self.title("Scientific Protocol | Nova análise")
        self.geometry("940x700")
        self.minsize(780, 580)
        self.grab_set()
        self._build_shell()
        if self.project:
            self._load_vars_from_project()
        else:
            self._show_page(0)

    def _cells(self):
        if not self.project or not self.current_electrolyte:
            return {}
        return (
            self.project.get("electrolytes", {})
            .get(self.current_electrolyte, {})
            .get("cells", {})
        )

    def _cell(self):
        return self._cells().get(self.current_cell, {})

    def _build_shell(self):
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=18, pady=(14, 4))
        self.body.grid_columnconfigure(0, weight=1)
        self.body.grid_rowconfigure(1, weight=1)
        self.heading = ctk.CTkLabel(
            self.body,
            text="Nova análise científica",
            font=ctk.CTkFont(size=23, weight="bold"),
            anchor="w",
        )
        self.heading.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.pages = ctk.CTkFrame(self.body)
        self.pages.grid(row=1, column=0, sticky="nsew")
        self.pages.grid_rowconfigure(0, weight=1)
        self.pages.grid_columnconfigure(0, weight=1)
        self.page_frames = [self._make_page() for _ in range(8)]
        self._build_identification(self.page_frames[0])
        self._build_cv(self.page_frames[1])
        self._build_single_file(
            self.page_frames[2], "gcd", "3. Arquivo GCD", "Selecionar arquivo GCD"
        )
        self._build_eis(self.page_frames[3])
        self._build_physics(self.page_frames[4])
        self._build_electrodes(self.page_frames[5])
        self._build_sequence(self.page_frames[6])
        self._build_review(self.page_frames[7])
        self.footer = ctk.CTkFrame(self, fg_color="transparent")
        self.footer.pack(fill="x", padx=18, pady=(4, 16))
        self.step_label = ctk.CTkLabel(
            self.footer, text="Etapa 1 de 8", text_color="gray"
        )
        self.step_label.pack(side="left")
        self.execution_status = ctk.CTkLabel(
            self.footer, text="Pronto", text_color="gray", anchor="w"
        )
        self.execution_status.pack(side="left", padx=(16, 0))
        self.cancel_button = ctk.CTkButton(
            self.footer,
            text="Cancelar",
            width=110,
            fg_color="gray40",
            command=self.destroy,
        )
        self.cancel_button.pack(side="right", padx=4)
        self.next_button = ctk.CTkButton(
            self.footer, text="Próximo", width=125, command=self._next
        )
        self.next_button.pack(side="right", padx=4)
        self.save_button = ctk.CTkButton(
            self.footer,
            text="Salvar sem executar",
            width=150,
            command=lambda: self._save(execute=False),
        )
        self.save_button.pack(side="right", padx=4)
        self.back_button = ctk.CTkButton(
            self.footer, text="Voltar", width=110, state="disabled", command=self._back
        )
        self.back_button.pack(side="right", padx=4)

    def _make_page(self):
        page = ctk.CTkScrollableFrame(self.pages)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=1)
        return page

    def _label(self, parent, text, row, *, size=13):
        ctk.CTkLabel(
            parent, text=text, font=ctk.CTkFont(size=size, weight="bold"), anchor="w"
        ).grid(row=row, column=0, sticky="ew", padx=12, pady=(12, 5))

    def _entry(self, parent, row, value="", placeholder=""):
        entry = ctk.CTkEntry(parent, placeholder_text=placeholder)
        entry.grid(row=row, column=0, sticky="ew", padx=12, pady=5)
        if value:
            entry.insert(0, value)
        return entry

    def _build_identification(self, page):
        self._label(page, "1. Identificação", 0, size=17)
        ctk.CTkLabel(
            page,
            text="Defina o projeto, material, eletrólito e a célula que será analisada.",
            text_color="gray",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=12)
        self.project_name = self._entry(
            page,
            2,
            placeholder="Nome do projeto, por exemplo: Estudo Nb2L - Eletrólitos",
        )
        self.material = self._entry(page, 3, placeholder="Material: Nb2, Nb4, Nb2L...")
        self.electrolyte = self._entry(
            page, 4, placeholder="Eletrólito: H2SO4 1M, NaCl 1M..."
        )
        self.cell_id = self._entry(
            page, 5, placeholder="ID amigável da célula: cell_001"
        )
        self.replicate_label = ctk.CTkLabel(
            page, text="Replicata 1", text_color="gray", anchor="w"
        )
        self.replicate_label.grid(row=6, column=0, sticky="ew", padx=12, pady=5)
        self.electrolyte_menu = ctk.CTkOptionMenu(
            page, values=["Nenhum projeto"], command=self._select_electrolyte
        )
        self.electrolyte_menu.grid(row=7, column=0, sticky="w", padx=12, pady=5)
        self.cell_menu = ctk.CTkOptionMenu(
            page, values=["Nenhuma célula"], command=self._select_cell
        )
        self.cell_menu.grid(row=8, column=0, sticky="w", padx=12, pady=5)
        row = ctk.CTkFrame(page, fg_color="transparent")
        row.grid(row=9, column=0, sticky="ew", padx=8, pady=10)
        ctk.CTkButton(
            row, text="Adicionar eletrólito", command=self._add_electrolyte
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            row, text="Adicionar célula/replicata", command=self._add_cell
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            row, text="Duplicar célula sem arquivos", command=self._duplicate_cell
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            row,
            text="Excluir célula",
            command=self._delete_cell,
            fg_color="#a33",
            hover_color="#822",
        ).pack(side="left", padx=4)

    def _build_cv(self, page):
        self._label(page, "2. Arquivos de CV", 0, size=17)
        ctk.CTkLabel(
            page,
            text="Adicione quantas velocidades forem necessárias. A unidade exibida é mV/s; o protocolo recebe V/s.",
            text_color="gray",
            anchor="w",
            wraplength=800,
        ).grid(row=1, column=0, sticky="ew", padx=12)
        self.cv_frame = ctk.CTkFrame(page)
        self.cv_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=12)
        self.cv_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.cv_frame, text="Velocidade").grid(
            row=0, column=0, padx=8, pady=8
        )
        ctk.CTkLabel(self.cv_frame, text="Arquivo").grid(
            row=0, column=1, padx=8, pady=8, sticky="w"
        )
        ctk.CTkButton(page, text="+ Adicionar CV", command=self._add_cv).grid(
            row=3, column=0, sticky="w", padx=12, pady=8
        )
        self.cv_hint = ctk.CTkLabel(
            page, text="Nenhum CV selecionado.", text_color="gray", anchor="w"
        )
        self.cv_hint.grid(row=4, column=0, sticky="ew", padx=12)

    def _build_single_file(self, page, key, title, button_text):
        self._label(page, title, 0, size=17)
        ctk.CTkLabel(
            page,
            text="Selecione explicitamente o arquivo experimental. O caminho será salvo relativo ao projeto quando possível.",
            text_color="gray",
            anchor="w",
            wraplength=800,
        ).grid(row=1, column=0, sticky="ew", padx=12)
        var = ctk.StringVar()
        setattr(self, f"{key}_var", var)
        ctk.CTkEntry(page, textvariable=var).grid(
            row=2, column=0, sticky="ew", padx=12, pady=10
        )
        actions = ctk.CTkFrame(page, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="w", padx=8)
        ctk.CTkButton(
            actions, text=button_text, command=lambda: self._choose_file(var)
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            actions, text="Remover", fg_color="gray40", command=lambda: var.set("")
        ).pack(side="left", padx=4)

    def _build_eis(self, page):
        self._label(page, "4. Arquivos de EIS", 0, size=17)
        ctk.CTkLabel(
            page,
            text="Associe cada estado manualmente. O wizard não infere inicial, pós-CV ou final.",
            text_color="gray",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=12)
        for row, (state, label) in enumerate(
            (
                ("initial", "EIS inicial"),
                ("post_cv", "EIS pós-CV"),
                ("final", "EIS final"),
            ),
            start=2,
        ):
            frame = ctk.CTkFrame(page)
            frame.grid(row=row, column=0, sticky="ew", padx=8, pady=5)
            frame.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(frame, text=label, width=120, anchor="w").grid(
                row=0, column=0, padx=8
            )
            var = ctk.StringVar()
            self.eis_vars[state] = var
            ctk.CTkEntry(frame, textvariable=var).grid(
                row=0, column=1, sticky="ew", padx=5
            )
            ctk.CTkButton(
                frame,
                text="Selecionar",
                width=105,
                command=lambda v=var: self._choose_file(v),
            ).grid(row=0, column=2, padx=5)
            ctk.CTkButton(
                frame,
                text="Remover",
                width=90,
                fg_color="gray40",
                command=lambda v=var: v.set(""),
            ).grid(row=0, column=3, padx=8)

    def _build_physics(self, page):
        self._label(page, "5. Configuração física da célula", 0, size=17)
        ctk.CTkLabel(
            page,
            text="Use unidades de laboratório. As massas em mg são convertidas para gramas automaticamente.",
            text_color="gray",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=12)
        self.mass_mg = self._entry(
            page, 2, placeholder="Massa total ativa (mg), por exemplo: 5.6"
        )
        self.area_cm2 = self._entry(
            page, 3, placeholder="Área total da célula/eletrodo (cm²), por exemplo: 1.0"
        )

    def _build_electrodes(self, page):
        self._label(page, "6. Configuração dos eletrodos", 0, size=17)
        ctk.CTkLabel(
            page,
            text="Informe a massa ativa em mg e a fração de potencial de cada eletrodo.",
            text_color="gray",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=12)
        grid = ctk.CTkFrame(page)
        grid.grid(row=2, column=0, sticky="ew", padx=8, pady=12)
        for col, text in enumerate(("Eletrodo", "Massa (mg)", "Fração de potencial")):
            ctk.CTkLabel(grid, text=text, font=ctk.CTkFont(weight="bold")).grid(
                row=0, column=col, padx=10, pady=8
            )
        self.we_mass = ctk.CTkEntry(grid, placeholder_text="2.8")
        self.we_fraction = ctk.CTkEntry(grid, placeholder_text="0.5")
        self.ce_mass = ctk.CTkEntry(grid, placeholder_text="2.8")
        self.ce_fraction = ctk.CTkEntry(grid, placeholder_text="0.5")
        for row, name, mass, fraction in (
            (1, "WE", self.we_mass, self.we_fraction),
            (2, "CE", self.ce_mass, self.ce_fraction),
        ):
            ctk.CTkLabel(grid, text=name).grid(row=row, column=0, padx=10, pady=6)
            mass.grid(row=row, column=1, padx=10, pady=6)
            fraction.grid(row=row, column=2, padx=10, pady=6)
        self.electrode_hint = ctk.CTkLabel(
            page,
            text="As frações devem somar aproximadamente 1.0.",
            text_color="gray",
            anchor="w",
        )
        self.electrode_hint.grid(row=3, column=0, sticky="ew", padx=12)

    def _build_sequence(self, page):
        self._label(page, "7. Sequência GCD", 0, size=17)
        ctk.CTkLabel(
            page,
            text="Edite visualmente as densidades de corrente; não é necessário escrever uma lista Python.",
            text_color="gray",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=12)
        self.sequence_frame = ctk.CTkFrame(page)
        self.sequence_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=12)
        self.sequence_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            page,
            text="+ Adicionar etapa",
            command=lambda: self._add_sequence_value("1"),
        ).grid(row=3, column=0, sticky="w", padx=12, pady=8)

    def _build_review(self, page):
        self._label(page, "8. Revisão", 0, size=17)
        self.review = ctk.CTkTextbox(page, height=440, wrap="word")
        self.review.grid(row=1, column=0, sticky="nsew", padx=12, pady=10)
        dpi_row = ctk.CTkFrame(page, fg_color="transparent")
        dpi_row.grid(row=2, column=0, sticky="w", padx=8, pady=(8, 4))
        ctk.CTkLabel(dpi_row, text="Resolução das figuras:").pack(
            side="left", padx=(4, 8)
        )
        self.dpi_var = ctk.StringVar(value="300 DPI")
        self.dpi_menu = ctk.CTkOptionMenu(
            dpi_row,
            values=["150 DPI", "200 DPI", "300 DPI", "600 DPI"],
            variable=self.dpi_var,
        )
        self.dpi_menu.pack(side="left")
        ctk.CTkLabel(
            page,
            text="300 DPI é o padrão recomendado para relatórios e artigos.",
            text_color="gray",
            anchor="w",
        ).grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 6))
        ctk.CTkLabel(
            page,
            text="Salvar e executar protocolo gera o project.json e entrega o projeto ao executor científico existente.",
            text_color="gray",
            anchor="w",
            wraplength=800,
        ).grid(row=4, column=0, sticky="ew", padx=12, pady=(4, 8))

    def _choose_file(self, var):
        path = filedialog.askopenfilename(
            parent=self,
            title="Selecionar arquivo experimental",
            filetypes=[
                ("Arquivos de texto", "*.txt *.csv *.dat *.asc"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if path:
            var.set(path)

    def _add_electrolyte(self):
        name = simpledialog.askstring(
            "Novo eletrólito", "Nome do eletrólito:", parent=self
        )
        if not name or not name.strip():
            return
        name = name.strip()
        self.project = self.project or new_project(
            self.project_name.get() or "Novo projeto",
            self.material.get(),
            self.electrolyte.get() or name,
        )
        self.project.setdefault("electrolytes", {}).setdefault(name, {"cells": {}})
        add_cell(self.project, name)
        self.current_electrolyte = name
        self.current_cell = next(iter(self._cells()))
        self.electrolyte.delete(0, "end")
        self.electrolyte.insert(0, name)
        self._load_vars_from_project()

    def _refresh_selectors(self):
        electrolytes = (
            list(self.project.get("electrolytes", {})) if self.project else []
        )
        if electrolytes:
            self.electrolyte_menu.configure(values=electrolytes)
            self.electrolyte_menu.set(self.current_electrolyte)
            cells = list(self._cells())
            self.cell_menu.configure(values=cells or ["Nenhuma célula"])
            self.cell_menu.set(
                self.current_cell
                if self.current_cell in cells
                else (cells[0] if cells else "Nenhuma célula")
            )
        else:
            self.electrolyte_menu.configure(values=["Nenhum projeto"])
            self.electrolyte_menu.set("Nenhum projeto")
            self.cell_menu.configure(values=["Nenhuma célula"])
            self.cell_menu.set("Nenhuma célula")

    def _select_electrolyte(self, value):
        if not self.project or value not in self.project.get("electrolytes", {}):
            return
        self._capture_current()
        self.current_electrolyte = value
        self.current_cell = next(iter(self._cells()), "")
        self._load_vars_from_project()

    def _select_cell(self, value):
        if not self.project or value not in self._cells():
            return
        self._capture_current()
        self.current_cell = value
        self._load_vars_from_project()

    def _add_cell(self):
        self._capture_current()
        if not self.project:
            self._ensure_project()
        if not self.current_electrolyte:
            return
        self.current_cell = add_cell(self.project, self.current_electrolyte)
        self._load_vars_from_project()

    def _duplicate_cell(self):
        self._capture_current()
        if not self.project or not self.current_cell:
            return
        self.current_cell = duplicate_cell(
            self.project, self.current_electrolyte, self.current_cell
        )
        self._load_vars_from_project()

    def _delete_cell(self):
        self._capture_current()
        cells = self._cells()
        if not self.project or not self.current_cell or self.current_cell not in cells:
            return
        if len(cells) <= 1:
            messagebox.showwarning(
                "Célula necessária",
                "Mantenha pelo menos uma célula neste eletrólito.",
                parent=self,
            )
            return
        confirmed = messagebox.askyesno(
            "Excluir célula",
            f"Excluir a célula {self.current_cell}? Esta ação remove apenas a configuração do projeto.",
            parent=self,
        )
        if not confirmed:
            return
        del cells[self.current_cell]
        self.current_cell = next(iter(cells))
        self._load_vars_from_project()

    def _add_cv(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Selecionar arquivo CV",
            filetypes=[("Texto", "*.txt"), ("Todos", "*.*")],
        )
        if not path:
            return
        rate = simpledialog.askfloat(
            "Velocidade de varredura",
            "Informe a velocidade em mV/s:",
            parent=self,
            minvalue=0.000001,
        )
        if rate is None:
            return
        entry = ctk.CTkEntry(self.cv_frame, width=120)
        entry.insert(0, f"{rate:g}")
        label = ctk.CTkLabel(self.cv_frame, text=path, anchor="w")
        row = len(self.cv_rows) + 1
        entry.grid(row=row, column=0, padx=8, pady=5, sticky="w")
        label.grid(row=row, column=1, padx=8, pady=5, sticky="ew")
        ctk.CTkButton(
            self.cv_frame,
            text="Remover",
            width=90,
            fg_color="gray40",
            command=lambda: self._remove_cv(entry, label),
        ).grid(row=row, column=2, padx=8, pady=5)
        self.cv_rows.append((entry, label))
        self.cv_hint.configure(text=f"{len(self.cv_rows)} CV(s) selecionado(s).")

    def _remove_cv(self, entry, label):
        entry.destroy()
        label.destroy()
        for widget in self.cv_frame.winfo_children():
            if isinstance(widget, ctk.CTkButton) and widget.cget("text") == "Remover":
                widget.destroy()
        self.cv_rows = [
            (item, text) for item, text in self.cv_rows if item is not entry
        ]
        self._rebuild_cv_buttons()
        self.cv_hint.configure(
            text=f"{len(self.cv_rows)} CV(s) selecionado(s)."
            if self.cv_rows
            else "Nenhum CV selecionado."
        )

    def _rebuild_cv_buttons(self):
        for row, (entry, label) in enumerate(self.cv_rows, start=1):
            entry.grid(row=row, column=0)
            label.grid(row=row, column=1)
            ctk.CTkButton(
                self.cv_frame,
                text="Remover",
                width=90,
                fg_color="gray40",
                command=lambda e=entry, label_widget=label: self._remove_cv(
                    e, label_widget
                ),
            ).grid(row=row, column=2, padx=8, pady=5)

    def _add_sequence_value(self, value="1"):
        row = len(self.sequence_entries) + 1
        entry = ctk.CTkEntry(self.sequence_frame, width=140)
        entry.insert(0, value)
        entry.grid(row=row, column=1, padx=8, pady=4, sticky="ew")
        ctk.CTkLabel(self.sequence_frame, text="A/g").grid(row=row, column=2, padx=4)
        ctk.CTkButton(
            self.sequence_frame,
            text="↑",
            width=32,
            command=lambda e=entry: self._move_sequence(e, -1),
        ).grid(row=row, column=3, padx=2)
        ctk.CTkButton(
            self.sequence_frame,
            text="↓",
            width=32,
            command=lambda e=entry: self._move_sequence(e, 1),
        ).grid(row=row, column=4, padx=2)
        ctk.CTkButton(
            self.sequence_frame,
            text="Remover",
            width=85,
            fg_color="gray40",
            command=lambda e=entry: self._remove_sequence(e),
        ).grid(row=row, column=5, padx=6)
        self.sequence_entries.append(entry)

    def _redraw_sequence(self):
        values = [entry.get() for entry in self.sequence_entries]
        for widget in self.sequence_frame.winfo_children():
            widget.destroy()
        self.sequence_entries = []
        for value in values:
            self._add_sequence_value(value)

    def _remove_sequence(self, entry):
        self.sequence_entries = [
            item for item in self.sequence_entries if item is not entry
        ]
        self._redraw_sequence()

    def _move_sequence(self, entry, offset):
        index = self.sequence_entries.index(entry)
        target = index + offset
        if 0 <= target < len(self.sequence_entries):
            self.sequence_entries[index], self.sequence_entries[target] = (
                self.sequence_entries[target],
                self.sequence_entries[index],
            )
            self._redraw_sequence()

    def _ensure_project(self):
        name = self.project_name.get().strip()
        material = self.material.get().strip()
        electrolyte = self.electrolyte.get().strip()
        self.project = new_project(
            name or "Novo projeto", material, electrolyte or "Novo eletrólito"
        )
        self.current_electrolyte = electrolyte or "Novo eletrólito"
        self.current_cell = next(iter(self._cells()))

    def _capture_current(self):
        if not self.project:
            self._ensure_project()
        cell = self._cell()
        if not cell:
            return
        try:
            cell["mass_g"] = float(self.mass_mg.get()) / 1000.0
        except ValueError:
            cell["mass_g"] = None
        try:
            cell["cell_area_cm2"] = float(self.area_cm2.get())
        except ValueError:
            cell["cell_area_cm2"] = None
        cell["gcd"] = self.gcd_var.get().strip()
        cell["eis"] = {
            state: var.get().strip()
            for state, var in self.eis_vars.items()
            if var.get().strip()
        }
        cv = {}
        for entry, label in self.cv_rows:
            try:
                cv[f"{float(entry.get()) / 1000.0:g}"] = label.cget("text")
            except ValueError:
                pass
        cell["cv"] = cv
        sequence = []
        for entry in self.sequence_entries:
            try:
                sequence.append(float(entry.get()))
            except ValueError:
                pass
        cell["current_sequence_a_g"] = sequence
        electrodes = {}
        for name, mass, fraction in (
            ("WE", self.we_mass, self.we_fraction),
            ("CE", self.ce_mass, self.ce_fraction),
        ):
            try:
                electrodes[name] = {
                    "mass_g": float(mass.get()) / 1000.0,
                    "potential_fraction": float(fraction.get()),
                }
            except ValueError:
                pass
        cell["electrodes"] = electrodes
        self.project["project"] = {
            "name": self.project_name.get().strip(),
            "material": self.material.get().strip(),
        }
        try:
            self.project["protocol_config"] = {
                "dpi": int(self.dpi_var.get().split()[0])
            }
        except (AttributeError, ValueError):
            self.project["protocol_config"] = {"dpi": 300}

    def _load_vars_from_project(self):
        if not self.project:
            return
        self.project_name.delete(0, "end")
        self.project_name.insert(0, self.project.get("project", {}).get("name", ""))
        self.material.delete(0, "end")
        self.material.insert(0, self.project.get("project", {}).get("material", ""))
        self.electrolyte.delete(0, "end")
        self.electrolyte.insert(0, self.current_electrolyte)
        self._refresh_selectors()
        cell = self._cell()
        self.cell_id.delete(0, "end")
        self.cell_id.insert(0, self.current_cell)
        self.replicate_label.configure(text=f"Replicata {cell.get('replicate', 1)}")
        self.mass_mg.delete(0, "end")
        if cell.get("mass_g") is not None:
            self.mass_mg.insert(0, f"{float(cell['mass_g']) * 1000:g}")
        self.area_cm2.delete(0, "end")
        if cell.get("cell_area_cm2") is not None:
            self.area_cm2.insert(0, f"{float(cell['cell_area_cm2']):g}")
        self.gcd_var.set(cell.get("gcd", ""))
        for state, var in self.eis_vars.items():
            var.set(cell.get("eis", {}).get(state, ""))
        for entry, label in self.cv_rows:
            entry.destroy()
            label.destroy()
        self.cv_rows = []
        for rate, path in cell.get("cv", {}).items():
            entry = ctk.CTkEntry(self.cv_frame, width=120)
            entry.insert(0, f"{float(rate) * 1000:g}")
            label = ctk.CTkLabel(self.cv_frame, text=path, anchor="w")
            self.cv_rows.append((entry, label))
        self._rebuild_cv_buttons()
        self.cv_hint.configure(
            text=f"{len(self.cv_rows)} CV(s) selecionado(s)."
            if self.cv_rows
            else "Nenhum CV selecionado."
        )
        for entry in self.sequence_entries:
            entry.destroy()
        self.sequence_entries = []
        for value in cell.get("current_sequence_a_g", []):
            self._add_sequence_value(str(value))
        for widget, values in (
            (self.we_mass, ("WE", "mass_g")),
            (self.we_fraction, ("WE", "potential_fraction")),
            (self.ce_mass, ("CE", "mass_g")),
            (self.ce_fraction, ("CE", "potential_fraction")),
        ):
            name, key = values
            widget.delete(0, "end")
            if (
                name in cell.get("electrodes", {})
                and cell["electrodes"][name].get(key) is not None
            ):
                value = cell["electrodes"][name][key]
                widget.insert(
                    0,
                    f"{float(value) * 1000:g}"
                    if key == "mass_g"
                    else f"{float(value):g}",
                )
        dpi = self.project.get("protocol_config", {}).get("dpi", 300)
        if dpi in {150, 200, 300, 600}:
            self.dpi_var.set(f"{dpi} DPI")

    def _validate_page(self):
        if self.page_index == 0:
            if (
                not self.project_name.get().strip()
                or not self.material.get().strip()
                or not self.electrolyte.get().strip()
            ):
                return "Informe nome do projeto, material e eletrólito."
            self._ensure_project()
        self._capture_current()
        if self.page_index == 5:
            fractions = self._cell().get("electrodes", {})
            if (
                fractions
                and abs(
                    sum(item["potential_fraction"] for item in fractions.values()) - 1.0
                )
                > 1e-6
            ):
                return "As frações de potencial de WE e CE devem somar aproximadamente 1,0."
        errors = validate_project(self.project) if self.page_index == 7 else []
        return errors[0] if errors else None

    def _show_page(self, index):
        self.page_index = index
        self.page_frames[index].lift()
        self.page_frames[index].tkraise()
        self.step_label.configure(text=f"Etapa {index + 1} de 8")
        self.back_button.configure(state="normal" if index else "disabled")
        self.next_button.configure(
            text="Salvar e executar" if index == 7 else "Próximo"
        )
        if index == 7:
            self._capture_current()
            self.review.configure(state="normal")
            self.review.delete("1.0", "end")
            self.review.insert("1.0", self._review_text())
            self.review.configure(state="disabled")
            self.save_button.configure(state="normal")
        else:
            self.save_button.configure(state="disabled")
        self.update_idletasks()

    def _next(self):
        error = self._validate_page()
        if error:
            messagebox.showwarning("Revise os dados", error, parent=self)
            return
        if self.page_index < 7:
            self._show_page(self.page_index + 1)
        else:
            self._save(execute=True)

    def _back(self):
        if self.page_index > 0:
            self._capture_current()
            self._show_page(self.page_index - 1)

    def _review_text(self):
        cell = self._cell()
        lines = [
            f"PROJETO\nNome: {self.project.get('project', {}).get('name', '')}",
            f"Material: {self.project.get('project', {}).get('material', '')}",
            f"Eletrólito: {self.current_electrolyte}",
            f"Célula: {self.current_cell}",
            f"Replicata: {cell.get('replicate', 1)}",
            "",
            "CV",
        ]
        lines.extend(
            f"{float(rate) * 1000:g} mV/s -> {path}"
            for rate, path in cell.get("cv", {}).items()
        ) or lines.append("N/D")
        lines += ["", f"GCD\n{cell.get('gcd') or 'N/D'}", "", "EIS"]
        lines.extend(
            f"{state} -> {path or 'N/D'}"
            for state, path in (
                ("initial", cell.get("eis", {}).get("initial", "")),
                ("post_cv", cell.get("eis", {}).get("post_cv", "")),
                ("final", cell.get("eis", {}).get("final", "")),
            )
        )
        lines += [
            "",
            f"CÉLULA\nMassa total: {cell.get('mass_g', 'N/D')} g",
            f"Área: {cell.get('cell_area_cm2', 'N/D')} cm²",
            f"Eletrodos: {json.dumps(cell.get('electrodes', {}), ensure_ascii=False)}",
            f"Sequência GCD (A/g): {cell.get('current_sequence_a_g', [])}",
            f"Resolução das figuras: {self.dpi_var.get()}",
        ]
        return "\n".join(lines)

    def _save(self, execute=False):
        self._capture_current()
        errors = validate_project(self.project)
        if errors:
            messagebox.showerror(
                "Não foi possível salvar", "\n".join(errors[:8]), parent=self
            )
            return
        slug = "_".join(self.project["project"]["name"].lower().split()) or "projeto"
        path = (
            self.project_root / "scientific_protocol_projects" / slug / "project.json"
        )
        save_project(self.project, path)
        if self.on_saved:
            dpi = int(self.dpi_var.get().split()[0])
            self.on_saved(path, execute, self._set_execution_status, dpi)
        if execute:
            self.execution_status.configure(
                text="Projeto salvo · executando protocolo..."
            )
            self.next_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
        else:
            self.destroy()

    def _set_execution_status(self, text: str):
        """Update status safely when a background protocol run changes state."""
        try:
            if self.winfo_exists():
                self.execution_status.configure(text=text)
        except Exception:
            return
