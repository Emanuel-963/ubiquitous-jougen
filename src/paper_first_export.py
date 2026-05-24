"""Paper-first export package builder.

Generates a publication-oriented package with:
- manuscript draft sections (markdown)
- tables (csv + markdown)
- figures index and copied key figures
- discussion starter focused on scientific claims and caveats
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.report_generator import (
    build_cycling_section,
    build_drt_section,
    build_eis_section,
)


@dataclass
class PaperFirstResult:
    """Artifacts produced by paper-first export."""

    manuscript_path: str
    figures_index_path: str
    tables_dir: str
    figures_dir: str


def _safe_df(obj: Any) -> Optional[pd.DataFrame]:
    return obj if isinstance(obj, pd.DataFrame) and not obj.empty else None


def _copy_figures(paths: List[str], dst_dir: Path, prefix: str) -> List[str]:
    copied: List[str] = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths, start=1):
        src = Path(p)
        if not src.exists() or not src.is_file():
            continue
        ext = src.suffix.lower() or ".png"
        name = f"{prefix}_{i:02d}{ext}"
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _export_table(df: pd.DataFrame, base: Path) -> List[str]:
    paths: List[str] = []
    base.parent.mkdir(parents=True, exist_ok=True)

    csv_path = base.with_suffix(".csv")
    md_path = base.with_suffix(".md")

    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")

    paths.append(str(csv_path))
    paths.append(str(md_path))
    return paths


def _discussion_draft(eis_best: str, kk_hint: str, drt_hint: str, cyc_hint: str) -> str:
    return (
        "## Draft Discussion\n\n"
        "The impedance results indicate that the selected equivalent-circuit "
        f"representation ({eis_best}) captures the dominant electrochemical "
        "processes under the tested conditions. "
        f"{kk_hint} {drt_hint} {cyc_hint}\n\n"
        "A key implication is that optimization should prioritize configurations "
        "that simultaneously reduce ohmic contribution and preserve interfacial "
        "stability over cycling. Future work should include replicate experiments, "
        "independent surface characterization (e.g., SEM/XPS), and robustness checks "
        "across electrolyte composition and temperature windows.\n"
    )


def build_paper_first_package(
    pipeline_results: Dict[str, Any],
    output_dir: str,
    *,
    title: str = "IonFlow Paper-First Draft",
    author: str = "IonFlow Pipeline",
    institution: str = "",
) -> PaperFirstResult:
    """Build manuscript-oriented export package from pipeline results."""
    out = Path(output_dir)
    figs_dir = out / "figures"
    tables_dir = out / "tables"
    out.mkdir(parents=True, exist_ok=True)

    # Reuse existing section extractors
    eis = build_eis_section(pipeline_results)
    cyc = build_cycling_section(pipeline_results)
    drt = build_drt_section(pipeline_results)

    # Export tables
    exported_tables: List[str] = []
    rank_df = _safe_df(eis.get("ranking_table"))
    if rank_df is not None:
        exported_tables += _export_table(rank_df.head(50), tables_dir / "eis_ranking")

    cyc_df = _safe_df(cyc.get("table"))
    if cyc_df is not None:
        exported_tables += _export_table(
            cyc_df.head(50), tables_dir / "cycling_summary"
        )

    drt_df = _safe_df(drt.get("summary_table"))
    if drt_df is not None:
        exported_tables += _export_table(drt_df.head(50), tables_dir / "drt_summary")

    # Copy core figures
    copied_figs: List[str] = []
    copied_figs += _copy_figures(eis.get("image_paths", []), figs_dir, "eis")
    copied_figs += _copy_figures(cyc.get("image_paths", []), figs_dir, "cycling")
    copied_figs += _copy_figures(drt.get("image_paths", []), figs_dir, "drt")

    # Figure index
    fig_index = out / "figures_index.md"
    idx_lines: List[str] = ["# Figures Index", ""]
    for i, p in enumerate(copied_figs, start=1):
        rel = Path(p).relative_to(out)
        idx_lines.append(f"- Figure {i}: {rel.as_posix()}")
    fig_index.write_text("\n".join(idx_lines), encoding="utf-8")

    # Manuscript draft
    best_circuit = str(eis.get("best_circuit", "N/A"))
    kk_hint = (
        "Kramers-Kronig checks should be explicitly reported to support "
        "linearity/stationarity claims."
    )
    drt_hint = (
        "DRT outputs provide complementary evidence for process separation, "
        "especially in overlapping time-constant regimes."
        if drt_df is not None
        else "DRT analysis was not available in this dataset."
    )
    cyc_hint = (
        "Cycling trends support the interpretation of impedance evolution over use."
        if cyc_df is not None
        else "Cycling data were not included in this export."
    )

    manuscript = out / "paper_first_draft.md"
    body: List[str] = []
    body.append(f"# {title}")
    body.append("")
    body.append(f"Author: {author}")
    if institution:
        body.append(f"Institution: {institution}")
    body.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    body.append("")
    body.append("## Abstract (Draft)")
    body.append(
        "This draft summarizes electrochemical results processed with IonFlow, "
        "including impedance fitting, optional DRT decomposition, and cycling-derived "
        "performance indicators."
    )
    body.append("")
    body.append("## Methods (Draft)")
    body.append(
        "EIS datasets were processed using a reproducible Python pipeline. Equivalent "
        "circuits were fitted through weighted non-linear least squares, and candidate "
        "models were ranked by fit-quality metrics (including information criteria)."
    )
    body.append("")
    body.append("## Results (Draft)")
    body.append(f"- Best representative circuit: {best_circuit}")
    body.append(
        f"- EIS ranking table exported: {'yes' if rank_df is not None else 'no'}"
    )
    body.append(f"- Cycling summary exported: {'yes' if cyc_df is not None else 'no'}")
    body.append(f"- DRT summary exported: {'yes' if drt_df is not None else 'no'}")
    body.append("")
    body.append(_discussion_draft(best_circuit, kk_hint, drt_hint, cyc_hint))
    body.append("## Figures")
    body.append("See figures_index.md for the curated figure list.")
    body.append("")
    body.append("## Tables")
    body.append("Tables are available in CSV and Markdown under the tables/ folder.")

    manuscript.write_text("\n".join(body), encoding="utf-8")

    return PaperFirstResult(
        manuscript_path=str(manuscript),
        figures_index_path=str(fig_index),
        tables_dir=str(tables_dir),
        figures_dir=str(figs_dir),
    )
