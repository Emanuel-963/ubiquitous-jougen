#!/usr/bin/env python3
"""Create a clean source-code ZIP for distribution.

Excludes folders that should NOT be shared:
  venv/, __pycache__/, dist/, build/, .git/, *.egg-info/, .mypy_cache/,
  .pytest_cache/, .vscode/, .precommit_home/, python311_embed/, outputs/

Usage:
    python scripts/create_source_zip.py          # -> dist/IonFlow_Pipeline_src_v0.1.0.zip
    python scripts/create_source_zip.py -o out.zip
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

# Folders to skip entirely
EXCLUDE_DIRS = {
    "venv",
    ".venv",
    "__pycache__",
    ".git",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".precommit_home",
    ".vscode",
    "python311_embed",
    "eis_analytics.egg-info",
    "outputs",
    "node_modules",
}

# File patterns to skip
EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".spec",
    ".egg",
}

EXCLUDE_FILES = {
    ".ionflow_gui_settings.json",
    "python-3.11-embed-amd64.zip",
}


def should_exclude(path: Path, root: Path) -> bool:
    """Return True if the path should be excluded from the zip."""
    rel = path.relative_to(root)

    # Check each part of the relative path against excluded dirs
    for part in rel.parts:
        if part in EXCLUDE_DIRS:
            return True

    if path.is_file():
        if path.suffix in EXCLUDE_EXTENSIONS:
            return True
        if path.name in EXCLUDE_FILES:
            return True

    return False


def create_zip(root: Path, output: Path) -> None:
    """Create a zip file from *root*, excluding unwanted files."""
    root = root.resolve()
    output = output.resolve()

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    prefix = "IonFlow_Pipeline_src"  # top-level folder inside the zip

    count = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for dirpath, dirnames, filenames in os.walk(root):
            dp = Path(dirpath)

            # Prune excluded directories (modifying dirnames in-place)
            dirnames[:] = [
                d for d in dirnames
                if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")
            ]

            for fname in sorted(filenames):
                fpath = dp / fname
                if should_exclude(fpath, root):
                    continue
                arcname = f"{prefix}/{fpath.relative_to(root).as_posix()}"
                zf.write(fpath, arcname)
                count += 1

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"✔ {count} files → {output.name}  ({size_mb:.1f} MB)")


def main() -> None:
    root = Path(__file__).resolve().parent.parent  # project root

    try:
        from src import __version__
    except ImportError:
        __version__ = "0.1.0"

    default_name = f"IonFlow_Pipeline_src_v{__version__}.zip"
    default_output = root / "dist" / default_name

    parser = argparse.ArgumentParser(description="Create clean source zip")
    parser.add_argument("-o", "--output", type=Path, default=default_output)
    args = parser.parse_args()

    create_zip(root, args.output)


if __name__ == "__main__":
    main()
