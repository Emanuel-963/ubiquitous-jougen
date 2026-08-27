"""Load the supplied scientific scripts without importing their CLI blocks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "Scripts Adicionais"


def load_script(filename: str):
    path = SCRIPT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"scientific reference script not found: {path}")
    name = f"ionflow_legacy_{path.stem.replace(' ', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load scientific reference script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
