#!/usr/bin/env python
"""Build a standalone .exe of EIS Analytics using PyInstaller.

Usage
-----
    python build_exe.py          # one-folder bundle (faster builds)
    python build_exe.py --onefile  # single .exe (larger, slower start)

The resulting executable will be placed in the ``dist/`` folder.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ICON = ROOT / "data" / "ionflow.ico"
MAIN_SCRIPT = ROOT / "gui_app.py"
APP_NAME = "EIS_Analytics"

# Extra data the app needs at runtime
DATA_ITEMS = [
    (ROOT / "data",   "data"),
    (ROOT / "themes", "themes"),
]

# Modules that PyInstaller may fail to detect automatically
HIDDEN_IMPORTS = [
    "src",
    "src.circuit_fitting",
    "src.cpe_fit",
    "src.cycling_calculator",
    "src.cycling_loader",
    "src.cycling_plotter",
    "src.drt_analysis",
    "src.drt_visualization",
    "src.eis_plots",
    "src.loader",
    "src.metadata",
    "src.pca_analysis",
    "src.physics_metrics",
    "src.preprocessing",
    "src.ranking",
    "src.stability",
    "src.visualization",
    "main",
    "main_cycling",
    "main_drt",
    # Packages whose sub-modules are lazy-imported
    "sklearn.utils._typedefs",
    "sklearn.neighbors._partition_nodes",
    "PIL._tkinter_finder",
    "customtkinter",
    "mplcursors",
]


def build(onefile: bool = False) -> None:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",                       # no console window
        f"--name={APP_NAME}",
        f"--icon={ICON}",
    ]

    if onefile:
        cmd.append("--onefile")

    # --add-data src;dst  (Windows uses ';')
    for src, dst in DATA_ITEMS:
        cmd.append(f"--add-data={src};{dst}")

    for mod in HIDDEN_IMPORTS:
        cmd.append(f"--hidden-import={mod}")

    # Exclude heavy packages the app doesn't need
    for exc in ("IPython", "notebook", "jupyter", "sphinx"):
        cmd.append(f"--exclude-module={exc}")

    cmd.append(str(MAIN_SCRIPT))

    print("▶ Running:", " ".join(cmd), "\n")
    subprocess.check_call(cmd)
    print(f"\n✅  Build complete!  → dist/{APP_NAME}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build EIS Analytics .exe")
    parser.add_argument(
        "--onefile", action="store_true",
        help="Package as a single .exe (slower startup)",
    )
    args = parser.parse_args()
    build(onefile=args.onefile)
