"""Project and cell persistence for the Scientific Protocol wizard."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import __version__ as IONFLOW_VERSION

SCHEMA_VERSION = "2.0"
PROJECTS_DIRNAME = "scientific_protocol_projects"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def empty_cell(replicate: int = 1, cell_id: str | None = None) -> dict:
    """Create a new cell with no experimental files selected."""
    return {
        "replicate": int(replicate),
        "cell_id": cell_id or f"cell_{int(replicate):03d}",
        "cv": {},
        "gcd": "",
        "mass_g": None,
        "cell_area_cm2": None,
        "current_sequence_a_g": [],
        "electrodes": {},
        "eis": {},
    }


def new_project(name: str, material: str, electrolyte: str) -> dict:
    """Create the schema 2.0 project envelope."""
    project = {
        "schema_version": SCHEMA_VERSION,
        "project": {"name": name.strip(), "material": material.strip()},
        "electrolytes": {electrolyte.strip(): {"cells": {}}},
        "metadata": {},
    }
    add_cell(project, electrolyte.strip())
    return project


def add_cell(project: dict, electrolyte: str, cell: dict | None = None) -> str:
    """Add a cell and return its generated internal ID."""
    group = project.setdefault("electrolytes", {}).setdefault(
        electrolyte, {"cells": {}}
    )
    cells = group.setdefault("cells", {})
    replicate = (
        max([int(item.get("replicate", 0)) for item in cells.values()] or [0]) + 1
    )
    item = dict(cell or empty_cell(replicate))
    item["replicate"] = replicate
    base = f"cell_{replicate:03d}"
    cell_id = base
    suffix = 2
    while cell_id in cells:
        cell_id = f"{base}_{suffix}"
        suffix += 1
    item["cell_id"] = cell_id
    cells[cell_id] = item
    return cell_id


def duplicate_cell(project: dict, electrolyte: str, source_id: str) -> str:
    """Duplicate physical settings but deliberately clear all file paths."""
    source = project["electrolytes"][electrolyte]["cells"][source_id]
    copied = {
        key: value
        for key, value in source.items()
        if key not in {"cell_id", "replicate", "cv", "gcd", "eis"}
    }
    copied.update({"cv": {}, "gcd": "", "eis": {}})
    return add_cell(project, electrolyte, copied)


def validate_cell(cell: Mapping[str, Any]) -> list[str]:
    """Return friendly validation messages for one cell."""
    errors = []
    for key, label in (
        ("mass_g", "massa total ativa"),
        ("cell_area_cm2", "área da célula"),
    ):
        value = cell.get(key)
        try:
            if value is None or float(value) <= 0:
                errors.append(f"Informe uma {label} maior que zero.")
        except (TypeError, ValueError):
            errors.append(f"A {label} precisa ser numérica.")
    cv = cell.get("cv", {}) or {}
    rates = []
    for rate, path in cv.items():
        try:
            numeric = float(rate)
            if numeric <= 0:
                errors.append("As velocidades de CV devem ser positivas.")
            rates.append(numeric)
        except (TypeError, ValueError):
            errors.append("Há uma velocidade de CV inválida.")
        if not path or not Path(path).is_file():
            errors.append(f"Arquivo de CV não encontrado: {path or 'não selecionado'}")
    if len(rates) != len(set(rates)):
        errors.append("Não existem duas velocidades de CV iguais.")
    gcd = cell.get("gcd")
    if gcd and not Path(gcd).is_file():
        errors.append(f"Arquivo GCD não encontrado: {gcd}")
    for state, path in (cell.get("eis", {}) or {}).items():
        if state not in {"initial", "post_cv", "final"}:
            errors.append(f"Estado EIS inválido: {state}")
        elif path and not Path(path).is_file():
            errors.append(f"Arquivo de EIS {state} não encontrado: {path}")
    sequence = cell.get("current_sequence_a_g", []) or []
    try:
        if any(float(value) <= 0 for value in sequence):
            errors.append("A sequência GCD deve conter correntes maiores que zero.")
    except (TypeError, ValueError):
        errors.append("Há um valor inválido na sequência GCD.")
    fractions = cell.get("electrodes", {}) or {}
    if fractions:
        try:
            total = sum(
                float(item.get("potential_fraction")) for item in fractions.values()
            )
            if abs(total - 1.0) > 1e-6:
                errors.append("As frações de potencial dos eletrodos devem somar 1,0.")
            if any(float(item.get("mass_g")) <= 0 for item in fractions.values()):
                errors.append("As massas dos eletrodos devem ser maiores que zero.")
        except (TypeError, ValueError):
            errors.append("Revise massa e fração de potencial dos eletrodos.")
    return errors


def validate_project(project: Mapping[str, Any]) -> list[str]:
    """Validate schema 2.0 project data without rejecting optional protocols."""
    errors = []
    if project.get("schema_version") != SCHEMA_VERSION:
        errors.append("O projeto precisa usar o schema científico 2.0.")
    if not str(project.get("project", {}).get("name", "")).strip():
        errors.append("Informe o nome do projeto.")
    electrolytes = project.get("electrolytes", {}) or {}
    if not electrolytes:
        errors.append("Adicione pelo menos um eletrólito.")
    for name, group in electrolytes.items():
        if not str(name).strip():
            errors.append("Há um eletrólito sem nome.")
        cells = group.get("cells", {}) if isinstance(group, Mapping) else {}
        if not cells:
            errors.append(f"O eletrólito {name} não possui células.")
        for cell_id, cell in cells.items():
            errors.extend(
                f"{name} / {cell_id}: {message}" for message in validate_cell(cell)
            )
    return errors


def _relative_paths(value: Any, root: Path) -> Any:
    if isinstance(value, str) and value:
        path = Path(value)
        return os.path.relpath(path.resolve(), root.resolve()).replace(os.sep, "/")
    if isinstance(value, dict):
        return {key: _relative_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relative_paths(item, root) for item in value]
    return value


def prepare_for_save(project: Mapping[str, Any], project_root: str | Path) -> dict:
    """Add reproducibility metadata and convert paths where possible."""
    root = Path(project_root).resolve()
    result = json.loads(json.dumps(project))
    metadata = result.setdefault("metadata", {})
    timestamp = now_iso()
    metadata.setdefault("created_at", timestamp)
    metadata["updated_at"] = timestamp
    metadata["ionflow_version"] = IONFLOW_VERSION
    metadata["scientific_protocol_version"] = SCHEMA_VERSION
    metadata["schema_version"] = SCHEMA_VERSION
    commit = _git_commit()
    if commit:
        metadata["git_commit"] = commit
    for group in result.get("electrolytes", {}).values():
        for cell in group.get("cells", {}).values():
            for key in ("cv", "gcd", "eis"):
                cell[key] = _relative_paths(cell.get(key), root)
    return result


def resolve_paths(project: Mapping[str, Any], project_root: str | Path) -> dict:
    """Return a copy with relative experimental paths resolved."""
    root = Path(project_root).resolve()
    result = json.loads(json.dumps(project))
    for group in result.get("electrolytes", {}).values():
        for cell in group.get("cells", {}).values():
            for key in ("cv", "gcd", "eis"):
                value = cell.get(key)
                if isinstance(value, str) and value:
                    cell[key] = (
                        str((root / value).resolve())
                        if not Path(value).is_absolute()
                        else value
                    )
                elif isinstance(value, dict):
                    cell[key] = {
                        name: str((root / path).resolve())
                        if isinstance(path, str) and not Path(path).is_absolute()
                        else path
                        for name, path in value.items()
                    }
    return result


def save_project(project: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = prepare_for_save(project, path.parent)
    path.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_project(path: str | Path) -> dict:
    """Load schema 2.0 or migrate one legacy electrolyte into a cell."""
    path = Path(path)
    project = json.loads(path.read_text(encoding="utf-8"))
    if project.get("schema_version") == SCHEMA_VERSION:
        return resolve_paths(project, path.parent)
    electrolytes = {}
    for electrolyte, legacy in project.get("electrolytes", {}).items():
        cell = dict(legacy)
        cell["replicate"] = 1
        cell["cell_id"] = "cell_001"
        electrolytes[electrolyte] = {"cells": {"cell_001": cell}}
    migrated = {
        "schema_version": SCHEMA_VERSION,
        "project": {"name": path.stem, "material": ""},
        "electrolytes": electrolytes,
        "metadata": {"migrated_from_legacy": True},
    }
    return resolve_paths(migrated, path.parent)
