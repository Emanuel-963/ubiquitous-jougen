"""Scientific report generator for the IonFlow Pipeline.

Generates professional PDF reports from pipeline results, integrating
patterns from the auto_relatorios_cientificos project.  Also supports
Markdown export with LaTeX/DOCX placeholders for future expansion.

Architecture (inspired by auto_relatorios_cientificos)
------------------------------------------------------
- **Template-based**: Sections are composable building blocks
- **AI-enhanced**: Optional LLM enrichment of text sections
- **Multi-format**: PDF (primary), Markdown, LaTeX stub, DOCX stub
- **History tracking**: JSON log of all generations with versioning
- **Configurable**: Colours, fonts, layout all driven by ``ReportConfig``

Sections
--------
1. Cover page (logo, title, date, author)
2. Executive Summary (AI-generated if available)
3. EIS Analysis (ranking table, best circuit, PCA plot)
4. Cycling Analysis (energy/power, Ragone-style)
5. DRT Analysis (spectra, peak table)
6. Correlations (heatmap, top-5)
7. AI Interpretation (findings, anomalies, recommendations, predictions)
8. References

Dependencies
------------
``fpdf2>=2.7``
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ReportConfig:
    """Tuneable parameters for report generation.

    Inspired by ``auto_relatorios_cientificos`` config pattern.
    """

    # ── Identity ─────────────────────────────────────────────────
    title: str = "IonFlow Pipeline - Analysis Report"
    subtitle: str = "Electrochemical Impedance Spectroscopy"
    author: str = "IonFlow Pipeline"
    institution: str = ""
    logo_path: str = ""  # Optional path to a logo image

    # ── Layout ───────────────────────────────────────────────────
    page_size: str = "A4"  # A4 or Letter
    margin: float = 15.0  # mm
    font_family: str = "Helvetica"
    font_size_body: int = 10
    font_size_h1: int = 16
    font_size_h2: int = 13
    font_size_h3: int = 11
    font_size_small: int = 8

    # ── Colours (RGB tuples) ─────────────────────────────────────
    color_primary: Tuple[int, int, int] = (41, 98, 168)   # blue
    color_secondary: Tuple[int, int, int] = (68, 68, 68)  # dark grey
    color_accent: Tuple[int, int, int] = (0, 150, 80)     # green
    color_warning: Tuple[int, int, int] = (200, 120, 0)   # orange
    color_error: Tuple[int, int, int] = (200, 30, 30)     # red
    color_table_header: Tuple[int, int, int] = (41, 98, 168)
    color_table_alt_row: Tuple[int, int, int] = (240, 245, 255)

    # ── Content control ──────────────────────────────────────────
    include_eis: bool = True
    include_cycling: bool = True
    include_drt: bool = True
    include_correlations: bool = True
    include_ai: bool = True
    include_references: bool = True
    max_table_rows: int = 30
    max_images_per_section: int = 3
    dpi: int = 150

    # ── AI enhancement (from auto_relatorios pattern) ────────────
    enable_ai_enhancement: bool = False
    """If True and LLM is configured, sections are AI-enhanced."""

    # ── History / versioning ─────────────────────────────────────
    history_file: str = "outputs/report_history.json"
    """JSON file tracking all generated reports."""

    # ── Output formats ───────────────────────────────────────────
    output_formats: List[str] = field(
        default_factory=lambda: ["pdf"],
    )
    """Supported: 'pdf', 'markdown', 'latex', 'docx'."""


# ═══════════════════════════════════════════════════════════════════════
# Generation history (version control for reports)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GenerationRecord:
    """Single entry in the generation history."""

    timestamp: str
    output_path: str
    output_format: str
    sections: List[str]
    config_snapshot: Dict[str, Any]
    version: int = 1
    file_size_bytes: int = 0
    generation_time_s: float = 0.0
    ai_enhanced: bool = False


class GenerationHistory:
    """Tracks all report generations with version control.

    Inspired by auto_relatorios_cientificos Sprint 3 plan for
    generation history with version management.
    """

    def __init__(self, path: str = "outputs/report_history.json"):
        self._path = Path(path)
        self._records: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load existing history from disk."""
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as fh:
                    self._records = json.load(fh)
            except Exception as exc:
                logger.warning("Failed to load report history: %s", exc)
                self._records = []

    def _save(self) -> None:
        """Persist history to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._records, fh, indent=2, ensure_ascii=False, default=str)

    def add(self, record: GenerationRecord) -> int:
        """Add a generation record and return the version number."""
        # Calculate version based on same output_path
        same_path = [
            r for r in self._records
            if r.get("output_path") == record.output_path
        ]
        version = len(same_path) + 1
        record.version = version

        self._records.append(asdict(record))
        self._save()
        return version

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Return all history records."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)


# ═══════════════════════════════════════════════════════════════════════
# Text Enhancement (inspired by auto_relatorios ai.integrator)
# ═══════════════════════════════════════════════════════════════════════


def _enhance_text(text: str, context: str, pipeline_config: Any = None) -> str:
    """Optionally enhance text via LLM adapter.

    Mirrors ``auto_relatorios_cientificos.app.ai.integrator.enhance_section_text``.
    Falls back to original text if LLM is not configured.
    """
    if not text or not text.strip():
        return text

    try:
        from src.ai.llm_adapter import NullAdapter, create_adapter_from_config

        adapter = create_adapter_from_config(pipeline_config)

        # NullAdapter means no LLM is configured — skip enhancement
        if isinstance(adapter, NullAdapter):
            return text

        # Use interpret to enhance the section
        enhanced = adapter.interpret(
            context=f"Scientific report section:\n{text}",
            question=(
                "Enhance this scientific text: improve clarity, grammar, and "
                "professionalism while maintaining technical accuracy. "
                "Return only the enhanced text."
            ),
        )
        if enhanced and enhanced.strip() and len(enhanced.strip()) > 10:
            return enhanced.strip()
    except Exception as exc:
        logger.debug("AI text enhancement skipped: %s", exc)

    return text


# ═══════════════════════════════════════════════════════════════════════
# Section builders (composable, template-based)
# ═══════════════════════════════════════════════════════════════════════


def _safe_str(value: Any) -> str:
    """Convert value to string, handling NaN/None."""
    if value is None:
        return "—"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "—"
    if isinstance(value, float):
        if abs(value) < 0.01 or abs(value) > 1e6:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def build_eis_section(
    pipeline_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract EIS section data from pipeline results.

    Returns
    -------
    dict
        Keys: 'ranking_table' (DataFrame), 'best_circuit' (str),
        'circuit_stats' (dict), 'image_paths' (list), 'text' (str).
    """
    eis = pipeline_results.get("eis")
    if eis is None:
        return {"text": "EIS analysis was not performed.", "ranking_table": None,
                "best_circuit": None, "circuit_stats": {}, "image_paths": []}

    ranked = getattr(eis, "ranked_df", None)
    if ranked is None:
        ranked = eis.get("ranked_df") if hasattr(eis, "get") else None

    circuit_table = getattr(eis, "circuit_table", None)
    if circuit_table is None:
        circuit_table = eis.get("circuit_table") if hasattr(eis, "get") else None

    # Best circuit
    best_circuit = "N/A"
    circuit_stats: Dict[str, Any] = {}
    if circuit_table is not None and not circuit_table.empty:
        if "Circuito" in circuit_table.columns:
            counts = circuit_table["Circuito"].value_counts()
            best_circuit = str(counts.index[0]) if len(counts) > 0 else "N/A"
            circuit_stats = {
                "total_fits": len(circuit_table),
                "best_model": best_circuit,
                "model_distribution": counts.to_dict(),
            }

    # PCA figure paths
    pca = getattr(eis, "pca", None)
    image_paths: List[str] = []
    if pca is not None:
        pca_paths = getattr(pca, "figure_paths", [])
        image_paths = [p for p in pca_paths if p and os.path.exists(p)]

    # Summary text
    n_files = len(getattr(eis, "raw_eis", {}) or {})
    text_lines = [
        f"The EIS pipeline processed {n_files} files.",
        f"Best-fit equivalent circuit model: **{best_circuit}**.",
    ]
    if ranked is not None and not ranked.empty:
        text_lines.append(
            f"Ranked {len(ranked)} samples by composite score."
        )

    return {
        "ranking_table": ranked,
        "best_circuit": best_circuit,
        "circuit_stats": circuit_stats,
        "image_paths": image_paths[:3],
        "text": "\n".join(text_lines),
    }


def build_cycling_section(
    pipeline_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract cycling section data."""
    cyc = pipeline_results.get("cycling")
    if cyc is None:
        return {"text": "Cycling analysis was not performed.",
                "table": None, "image_paths": []}

    merged = getattr(cyc, "merged_table", None)
    if merged is None:
        merged = cyc.get("merged_table") if hasattr(cyc, "get") else None

    # Collect plot paths
    plot_paths = getattr(cyc, "energy_power_paths", []) or []
    image_paths = [p[1] if isinstance(p, (list, tuple)) else str(p)
                   for p in plot_paths]
    image_paths = [p for p in image_paths if p and os.path.exists(p)]

    n_files = len(getattr(cyc, "results", {}) or {})
    text = f"The cycling pipeline processed {n_files} files."

    return {
        "table": merged,
        "image_paths": image_paths[:3],
        "text": text,
    }


def build_drt_section(
    pipeline_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract DRT section data."""
    drt = pipeline_results.get("drt")
    if drt is None:
        return {"text": "DRT analysis was not performed.",
                "peaks_table": None, "summary_table": None, "image_paths": []}

    peaks = getattr(drt, "drt_peaks_table", None)
    if peaks is None:
        peaks = drt.get("drt_peaks_table") if hasattr(drt, "get") else None

    summary = getattr(drt, "drt_summary_table", None)
    if summary is None:
        summary = drt.get("drt_summary_table") if hasattr(drt, "get") else None

    # Plot paths
    plot_paths = getattr(drt, "plot_paths", []) or []
    image_paths = [p[1] if isinstance(p, (list, tuple)) else str(p)
                   for p in plot_paths]
    image_paths = [p for p in image_paths if p and os.path.exists(p)]

    meta = getattr(drt, "run_meta", {}) or {}
    n_success = meta.get("n_success", 0)
    n_failed = meta.get("n_failed", 0)
    text = f"DRT analysis: {n_success} files succeeded, {n_failed} failed."

    return {
        "peaks_table": peaks,
        "summary_table": summary,
        "image_paths": image_paths[:3],
        "text": text,
    }


def build_correlation_text(
    pipeline_results: Dict[str, Any],
) -> str:
    """Build correlation section text from EIS ranked_df."""
    eis = pipeline_results.get("eis")
    if eis is None:
        return "Correlation analysis requires EIS data."

    ranked = getattr(eis, "ranked_df", None)
    if ranked is None or ranked.empty:
        return "Insufficient data for correlation analysis."

    # Find numeric columns
    numeric = ranked.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return "Not enough numeric columns for correlation."

    from scipy.stats import spearmanr
    cols = list(numeric.columns)[:12]  # limit
    lines = ["Top Spearman correlations (|ρ| > 0.5):"]
    pairs: List[Tuple[str, str, float]] = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            try:
                rho, _ = spearmanr(
                    numeric[c1].dropna(), numeric[c2].dropna(),
                    nan_policy="omit",
                )
                if np.isfinite(rho) and abs(rho) > 0.5:
                    pairs.append((c1, c2, rho))
            except Exception:
                pass
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for c1, c2, rho in pairs[:5]:
        lines.append(f"  • {c1} vs {c2}: ρ = {rho:.3f}")

    if len(pairs) == 0:
        lines.append("  No strong correlations found (|ρ| > 0.5).")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Markdown exporter (template-based, mirrors auto_relatorios pattern)
# ═══════════════════════════════════════════════════════════════════════


def generate_markdown(
    pipeline_results: Dict[str, Any],
    ai_summary: Optional[str] = None,
    config: Optional[ReportConfig] = None,
) -> str:
    """Generate a complete Markdown report.

    Uses the template pattern from ``auto_relatorios_cientificos``
    (placeholder replacement in structured templates).

    Parameters
    ----------
    pipeline_results : dict
        Keys: 'eis' (EISResult), 'cycling' (CyclingResult), 'drt' (DRTPipelineResult).
    ai_summary : str | None
        Optional AI executive summary text.
    config : ReportConfig | None
        Report configuration.

    Returns
    -------
    str
        Complete Markdown document.
    """
    cfg = config or ReportConfig()
    sections: List[str] = []

    # Title
    sections.append(f"# {cfg.title}\n")
    sections.append(f"**{cfg.subtitle}**\n")
    sections.append(f"Author: {cfg.author}  ")
    if cfg.institution:
        sections.append(f"Institution: {cfg.institution}  ")
    sections.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    sections.append("---\n")

    # Executive summary
    if ai_summary:
        sections.append("## Executive Summary\n")
        sections.append(f"{ai_summary}\n")
        sections.append("---\n")

    # EIS
    if cfg.include_eis:
        eis_data = build_eis_section(pipeline_results)
        sections.append("## EIS Impedance Analysis\n")
        sections.append(f"{eis_data['text']}\n")
        if eis_data["ranking_table"] is not None:
            sections.append("### Ranking Table\n")
            df = eis_data["ranking_table"]
            cols = [c for c in ["Sample", "Rs_fit", "Rp_fit", "Score", "Rank"]
                    if c in df.columns]
            if cols:
                sections.append(
                    df[cols].head(cfg.max_table_rows).to_markdown(index=False)
                )
                sections.append("")

    # Cycling
    if cfg.include_cycling:
        cyc_data = build_cycling_section(pipeline_results)
        sections.append("## Cycling Analysis\n")
        sections.append(f"{cyc_data['text']}\n")
        if cyc_data["table"] is not None:
            sections.append("### Energy / Power Table\n")
            sections.append(
                cyc_data["table"].head(cfg.max_table_rows).to_markdown(index=False)
            )
            sections.append("")

    # DRT
    if cfg.include_drt:
        drt_data = build_drt_section(pipeline_results)
        sections.append("## DRT Analysis\n")
        sections.append(f"{drt_data['text']}\n")
        if drt_data.get("summary_table") is not None:
            sections.append("### DRT Summary\n")
            sections.append(
                drt_data["summary_table"].head(cfg.max_table_rows).to_markdown(
                    index=False
                )
            )
            sections.append("")

    # Correlations
    if cfg.include_correlations:
        corr_text = build_correlation_text(pipeline_results)
        sections.append("## Correlations\n")
        sections.append(f"{corr_text}\n")

    # AI section
    if cfg.include_ai and ai_summary:
        sections.append("## AI Interpretation\n")
        sections.append(f"{ai_summary}\n")

    # References
    if cfg.include_references:
        sections.append("## References\n")
        sections.append(
            "1. Boukamp, B.A. (1995). A Linear Kronig-Kramers Transform Test "
            "for Immittance Data Validation. *J. Electrochem. Soc.*, 142(6), 1885.\n"
            "2. Wan, T.H. et al. (2015). Influence of the Discretization Methods "
            "on the Distribution of Relaxation Times. *Electrochimica Acta*, 184, 483.\n"
            "3. Lasia, A. (2014). *Electrochemical Impedance Spectroscopy and its "
            "Applications*. Springer.\n"
            "4. Barsoukov, E. & Macdonald, J.R. (2018). *Impedance Spectroscopy: "
            "Theory, Experiment, and Applications*. Wiley.\n"
        )

    sections.append(
        f"\n---\n*Generated by IonFlow Pipeline on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    )

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════
# PDF generator (fpdf2)
# ═══════════════════════════════════════════════════════════════════════


class _IonFlowPDF:
    """Internal fpdf2 wrapper for IonFlow reports.

    Wraps ``fpdf.FPDF`` to provide styled helpers for headers, tables,
    and section rendering.
    """

    def __init__(self, config: ReportConfig):
        from fpdf import FPDF

        self.cfg = config
        self.pdf = FPDF(format=config.page_size)
        self.pdf.set_auto_page_break(auto=True, margin=config.margin)
        self.pdf.set_margins(config.margin, config.margin, config.margin)
        self.pdf.add_page()
        self._setup_fonts()

    def _setup_fonts(self) -> None:
        """Configure font family."""
        self.pdf.set_font(self.cfg.font_family, size=self.cfg.font_size_body)

    # ── Text helpers ─────────────────────────────────────────────

    def _set_color(self, rgb: Tuple[int, int, int]) -> None:
        self.pdf.set_text_color(*rgb)

    def add_cover_page(self, ai_summary: Optional[str] = None) -> None:
        """Add a professional cover page."""
        cfg = self.cfg
        pw = self.pdf.w - 2 * cfg.margin  # printable width

        # Logo
        if cfg.logo_path and os.path.exists(cfg.logo_path):
            try:
                self.pdf.image(cfg.logo_path, x=cfg.margin, y=cfg.margin, w=30)
                self.pdf.ln(35)
            except Exception:
                self.pdf.ln(10)
        else:
            self.pdf.ln(40)

        # Title
        self._set_color(cfg.color_primary)
        self.pdf.set_font(cfg.font_family, "B", cfg.font_size_h1 + 4)
        self.pdf.multi_cell(pw, 12, _clean_text(cfg.title), align="C")
        self.pdf.ln(4)

        # Subtitle
        self._set_color(cfg.color_secondary)
        self.pdf.set_font(cfg.font_family, "I", cfg.font_size_h2)
        self.pdf.multi_cell(pw, 8, _clean_text(cfg.subtitle), align="C")
        self.pdf.ln(10)

        # Horizontal line
        y = self.pdf.get_y()
        self.pdf.set_draw_color(*cfg.color_primary)
        self.pdf.set_line_width(0.5)
        self.pdf.line(cfg.margin, y, cfg.margin + pw, y)
        self.pdf.ln(10)

        # Author / institution / date
        self._set_color(cfg.color_secondary)
        self.pdf.set_font(cfg.font_family, "", cfg.font_size_body + 1)
        self.pdf.cell(pw, 7, _clean_text(f"Author: {cfg.author}"), align="C", new_x="LMARGIN", new_y="NEXT")
        if cfg.institution:
            self.pdf.cell(pw, 7, _clean_text(cfg.institution), align="C", new_x="LMARGIN", new_y="NEXT")
        self.pdf.cell(
            pw, 7,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            align="C", new_x="LMARGIN", new_y="NEXT",
        )
        self.pdf.ln(10)

        # Executive summary on cover
        if ai_summary:
            self.pdf.set_font(cfg.font_family, "B", cfg.font_size_h3)
            self._set_color(cfg.color_primary)
            self.pdf.cell(pw, 8, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
            self.pdf.ln(2)
            self.pdf.set_font(cfg.font_family, "", cfg.font_size_body)
            self._set_color(cfg.color_secondary)
            self.pdf.multi_cell(pw, 5, _clean_text(ai_summary))
            self.pdf.ln(5)

    def add_section_header(self, title: str, level: int = 1) -> None:
        """Add a styled section header."""
        cfg = self.cfg
        pw = self.pdf.w - 2 * cfg.margin

        if level == 1:
            self.pdf.add_page()
            self._set_color(cfg.color_primary)
            self.pdf.set_font(cfg.font_family, "B", cfg.font_size_h1)
            self.pdf.cell(pw, 10, _clean_text(title), new_x="LMARGIN", new_y="NEXT")
            y = self.pdf.get_y()
            self.pdf.set_draw_color(*cfg.color_primary)
            self.pdf.set_line_width(0.3)
            self.pdf.line(cfg.margin, y, cfg.margin + pw, y)
            self.pdf.ln(6)
        elif level == 2:
            self._set_color(cfg.color_primary)
            self.pdf.set_font(cfg.font_family, "B", cfg.font_size_h2)
            self.pdf.cell(pw, 8, _clean_text(title), new_x="LMARGIN", new_y="NEXT")
            self.pdf.ln(3)
        else:
            self._set_color(cfg.color_secondary)
            self.pdf.set_font(cfg.font_family, "B", cfg.font_size_h3)
            self.pdf.cell(pw, 7, _clean_text(title), new_x="LMARGIN", new_y="NEXT")
            self.pdf.ln(2)

    def add_text(self, text: str) -> None:
        """Add body text."""
        cfg = self.cfg
        pw = self.pdf.w - 2 * cfg.margin
        self._set_color(cfg.color_secondary)
        self.pdf.set_font(cfg.font_family, "", cfg.font_size_body)
        self.pdf.multi_cell(pw, 5, _clean_text(text))
        self.pdf.ln(3)

    def add_dataframe_table(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        max_rows: int = 30,
    ) -> None:
        """Render a DataFrame as a styled table."""
        if df is None or df.empty:
            self.add_text("No data available.")
            return

        cfg = self.cfg
        pw = self.pdf.w - 2 * cfg.margin

        if columns:
            cols = [c for c in columns if c in df.columns]
        else:
            cols = list(df.columns)[:8]  # limit columns

        if not cols:
            self.add_text("No displayable columns.")
            return

        df_show = df[cols].head(max_rows)
        col_w = pw / len(cols)

        # Header row
        self.pdf.set_font(cfg.font_family, "B", cfg.font_size_small)
        self.pdf.set_fill_color(*cfg.color_table_header)
        self.pdf.set_text_color(255, 255, 255)
        for col in cols:
            label = _clean_text(str(col)[:15])
            self.pdf.cell(col_w, 6, label, border=1, fill=True, align="C")
        self.pdf.ln()

        # Data rows
        self.pdf.set_font(cfg.font_family, "", cfg.font_size_small)
        self._set_color(cfg.color_secondary)
        for i, (_, row) in enumerate(df_show.iterrows()):
            if i % 2 == 1:
                self.pdf.set_fill_color(*cfg.color_table_alt_row)
                fill = True
            else:
                self.pdf.set_fill_color(255, 255, 255)
                fill = True
            for col in cols:
                val = _clean_text(_safe_str(row.get(col))[:15])
                self.pdf.cell(col_w, 5, val, border=1, fill=fill, align="C")
            self.pdf.ln()

        self.pdf.ln(3)
        if len(df) > max_rows:
            self.pdf.set_font(cfg.font_family, "I", cfg.font_size_small)
            self.add_text(f"Showing {max_rows} of {len(df)} rows.")

    def add_image(self, path: str, caption: str = "", width: float = 0) -> None:
        """Add an image with optional caption."""
        if not path or not os.path.exists(path):
            return

        cfg = self.cfg
        pw = self.pdf.w - 2 * cfg.margin
        w = width or pw * 0.8

        # Check if we need a new page
        if self.pdf.get_y() > self.pdf.h - 80:
            self.pdf.add_page()

        try:
            x = cfg.margin + (pw - w) / 2
            self.pdf.image(path, x=x, w=w)
            self.pdf.ln(3)
            if caption:
                self.pdf.set_font(cfg.font_family, "I", cfg.font_size_small)
                self._set_color(cfg.color_secondary)
                self.pdf.multi_cell(pw, 4, caption, align="C")
                self.pdf.ln(3)
        except Exception as exc:
            logger.warning("Failed to add image %s: %s", path, exc)

    def add_bullet_list(self, items: List[str]) -> None:
        """Add a bullet-point list."""
        cfg = self.cfg
        pw = self.pdf.w - 2 * cfg.margin
        self.pdf.set_font(cfg.font_family, "", cfg.font_size_body)
        self._set_color(cfg.color_secondary)
        for item in items:
            text = _clean_text(f"  \u2022  {item}")
            self.pdf.multi_cell(pw, 5, text)
            self.pdf.ln(1)
        self.pdf.ln(2)

    def add_key_value(self, key: str, value: str) -> None:
        """Add a bold key: value pair."""
        cfg = self.cfg
        self.pdf.set_font(cfg.font_family, "B", cfg.font_size_body)
        self._set_color(cfg.color_primary)
        self.pdf.write(5, _clean_text(f"{key}: "))
        self.pdf.set_font(cfg.font_family, "", cfg.font_size_body)
        self._set_color(cfg.color_secondary)
        self.pdf.write(5, _clean_text(value))
        self.pdf.ln(6)

    def save(self, path: str) -> str:
        """Write PDF to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.pdf.output(path)
        return path


def _clean_text(text: str) -> str:
    """Remove markdown formatting and non-Latin1 chars for fpdf2."""
    # Remove markdown bold/italic
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    # Replace common unicode that fpdf2 Helvetica can't render
    replacements = {
        "\u2022": "-",   # bullet
        "\u2013": "-",   # en-dash
        "\u2014": "--",  # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00b1": "+/-", # plus-minus
        "\u03a9": "Ohm", # Omega
        "\u03c4": "tau", # tau
        "\u03bb": "lambda", # lambda
        "\u03b3": "gamma", # gamma
        "\u2192": "->",  # arrow
        "\u2264": "<=",  # less-equal
        "\u2265": ">=",  # greater-equal
        "\u00b2": "^2",  # superscript 2
        "\u00b3": "^3",  # superscript 3
        "\u2070": "^0",  # superscript 0
        "\u2071": "^i",
        "\u207b": "^-",
        "\U0001f7e2": "[OK]",    # green circle
        "\U0001f7e1": "[WARN]",  # yellow circle
        "\U0001f534": "[ERR]",   # red circle
        "\U0001f916": "[AI]",    # robot
        "\U0001f4a1": "[TIP]",   # bulb
        "\U0001f52e": "[PRED]",  # crystal ball
        "\u26a0\ufe0f": "[!]",   # warning
        "\u2705": "[v]",         # check
        "\u274c": "[x]",         # cross
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Encode to latin-1, dropping anything that doesn't fit
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


# ═══════════════════════════════════════════════════════════════════════
# Main ReportGenerator class
# ═══════════════════════════════════════════════════════════════════════


class ReportGenerator:
    """Professional report generator for IonFlow Pipeline results.

    Integrates the template-based approach from auto_relatorios_cientificos
    with IonFlow's typed pipeline results to produce multi-format reports.

    Parameters
    ----------
    config : PipelineConfig | None
        IonFlow pipeline config (for LLM settings, paths, etc.).
    report_config : ReportConfig | None
        Report-specific config (layout, colours, sections).

    Examples
    --------
    >>> gen = ReportGenerator()
    >>> gen.generate("report.pdf", {"eis": eis_result})
    'report.pdf'
    """

    def __init__(
        self,
        config: Any = None,
        report_config: Optional[ReportConfig] = None,
    ):
        self._pipeline_config = config
        self._report_config = report_config or ReportConfig()
        self._history = GenerationHistory(self._report_config.history_file)

    @property
    def report_config(self) -> ReportConfig:
        """Current report configuration."""
        return self._report_config

    @property
    def history(self) -> GenerationHistory:
        """Generation history tracker."""
        return self._history

    # ── Main entry-point ─────────────────────────────────────────

    def generate(
        self,
        output_path: str,
        pipeline_results: Dict[str, Any],
        *,
        ai_summary: Optional[str] = None,
        formats: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate reports in one or more formats.

        Parameters
        ----------
        output_path : str
            Base path for the output file.  Extension is adjusted per format.
        pipeline_results : dict
            Keys: 'eis', 'cycling', 'drt' mapping to pipeline result objects.
        ai_summary : str | None
            Optional AI-generated executive summary.
        formats : list[str] | None
            Override ``ReportConfig.output_formats``.  Supported: pdf, markdown.

        Returns
        -------
        list[str]
            Paths to generated files.
        """
        import time as _time
        t0 = _time.time()
        cfg = self._report_config
        fmts = formats or cfg.output_formats
        generated: List[str] = []
        sections_included: List[str] = []

        # Determine which sections are present
        if cfg.include_eis and "eis" in pipeline_results:
            sections_included.append("eis")
        if cfg.include_cycling and "cycling" in pipeline_results:
            sections_included.append("cycling")
        if cfg.include_drt and "drt" in pipeline_results:
            sections_included.append("drt")
        if cfg.include_correlations:
            sections_included.append("correlations")
        if cfg.include_ai and ai_summary:
            sections_included.append("ai")
        if cfg.include_references:
            sections_included.append("references")

        # AI enhancement
        if cfg.enable_ai_enhancement and ai_summary and self._pipeline_config:
            ai_summary = _enhance_text(
                ai_summary, "Executive summary for a scientific report",
                self._pipeline_config,
            )

        for fmt in fmts:
            fmt = fmt.lower().strip()
            try:
                if fmt == "pdf":
                    path = self._generate_pdf(
                        output_path, pipeline_results, ai_summary,
                    )
                    generated.append(path)
                elif fmt == "markdown":
                    base = Path(output_path).with_suffix(".md")
                    md = generate_markdown(
                        pipeline_results, ai_summary, cfg,
                    )
                    base.parent.mkdir(parents=True, exist_ok=True)
                    base.write_text(md, encoding="utf-8")
                    generated.append(str(base))
                elif fmt == "latex":
                    path = self._generate_latex_stub(output_path, pipeline_results)
                    generated.append(path)
                elif fmt == "docx":
                    path = self._generate_docx_stub(output_path)
                    generated.append(path)
                else:
                    logger.warning("Unsupported format: %s", fmt)
            except Exception as exc:
                logger.error("Failed to generate %s: %s", fmt, exc)

        elapsed = _time.time() - t0

        # Record in history
        for path in generated:
            ext = Path(path).suffix.lstrip(".")
            size = Path(path).stat().st_size if Path(path).exists() else 0
            record = GenerationRecord(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                output_path=str(path),
                output_format=ext,
                sections=sections_included,
                config_snapshot=asdict(cfg),
                file_size_bytes=size,
                generation_time_s=round(elapsed, 2),
                ai_enhanced=cfg.enable_ai_enhancement,
            )
            version = self._history.add(record)
            logger.info(
                "Report generated: %s (v%d, %.1f KB, %.1fs)",
                path, version, size / 1024, elapsed,
            )

        return generated

    # ── PDF ──────────────────────────────────────────────────────

    def _generate_pdf(
        self,
        output_path: str,
        pipeline_results: Dict[str, Any],
        ai_summary: Optional[str],
    ) -> str:
        """Build the full PDF document."""
        path = str(Path(output_path).with_suffix(".pdf"))
        cfg = self._report_config
        doc = _IonFlowPDF(cfg)

        # 1. Cover page
        doc.add_cover_page(ai_summary)

        # 2. EIS section
        if cfg.include_eis and "eis" in pipeline_results:
            eis_data = build_eis_section(pipeline_results)
            doc.add_section_header("EIS Impedance Analysis")
            doc.add_text(eis_data["text"])

            if eis_data.get("circuit_stats"):
                doc.add_section_header("Circuit Fitting Summary", level=2)
                stats = eis_data["circuit_stats"]
                doc.add_key_value("Total fits", str(stats.get("total_fits", 0)))
                doc.add_key_value("Best model", str(stats.get("best_model", "N/A")))
                dist = stats.get("model_distribution", {})
                if dist:
                    items = [f"{k}: {v} fits" for k, v in dist.items()]
                    doc.add_bullet_list(items)

            if eis_data.get("ranking_table") is not None:
                doc.add_section_header("Sample Ranking", level=2)
                rank_cols = ["Sample", "Rs_fit", "Rp_fit", "Q", "n", "Score", "Rank"]
                doc.add_dataframe_table(
                    eis_data["ranking_table"],
                    columns=rank_cols,
                    max_rows=cfg.max_table_rows,
                )

            for i, img_path in enumerate(eis_data.get("image_paths", [])):
                doc.add_image(img_path, caption=f"EIS Figure {i + 1}")

        # 3. Cycling section
        if cfg.include_cycling and "cycling" in pipeline_results:
            cyc_data = build_cycling_section(pipeline_results)
            doc.add_section_header("Cycling Analysis")
            doc.add_text(cyc_data["text"])

            if cyc_data.get("table") is not None:
                doc.add_section_header("Energy / Power Data", level=2)
                doc.add_dataframe_table(
                    cyc_data["table"],
                    max_rows=cfg.max_table_rows,
                )

            for i, img_path in enumerate(cyc_data.get("image_paths", [])):
                doc.add_image(img_path, caption=f"Cycling Figure {i + 1}")

        # 4. DRT section
        if cfg.include_drt and "drt" in pipeline_results:
            drt_data = build_drt_section(pipeline_results)
            doc.add_section_header("DRT Analysis")
            doc.add_text(drt_data["text"])

            if drt_data.get("summary_table") is not None:
                doc.add_section_header("DRT Summary", level=2)
                doc.add_dataframe_table(
                    drt_data["summary_table"],
                    max_rows=cfg.max_table_rows,
                )

            if drt_data.get("peaks_table") is not None:
                doc.add_section_header("Peak Table", level=2)
                doc.add_dataframe_table(
                    drt_data["peaks_table"],
                    max_rows=cfg.max_table_rows,
                )

            for i, img_path in enumerate(drt_data.get("image_paths", [])):
                doc.add_image(img_path, caption=f"DRT Figure {i + 1}")

        # 5. Correlations
        if cfg.include_correlations and "eis" in pipeline_results:
            doc.add_section_header("Correlations")
            corr_text = build_correlation_text(pipeline_results)
            doc.add_text(corr_text)

        # 6. AI Interpretation
        if cfg.include_ai and ai_summary:
            doc.add_section_header("AI Interpretation")
            doc.add_text(ai_summary)

        # 7. References
        if cfg.include_references:
            doc.add_section_header("References")
            refs = [
                "Boukamp, B.A. (1995). A Linear Kronig-Kramers Transform Test for "
                "Immittance Data Validation. J. Electrochem. Soc., 142(6), 1885.",
                "Wan, T.H. et al. (2015). Influence of the Discretization Methods on "
                "the Distribution of Relaxation Times. Electrochimica Acta, 184, 483.",
                "Lasia, A. (2014). Electrochemical Impedance Spectroscopy and its "
                "Applications. Springer.",
                "Barsoukov, E. & Macdonald, J.R. (2018). Impedance Spectroscopy: "
                "Theory, Experiment, and Applications. Wiley.",
            ]
            for i, ref in enumerate(refs, 1):
                doc.add_text(f"[{i}] {ref}")

        # Footer info
        doc.add_text(
            f"\nGenerated by IonFlow Pipeline on "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        return doc.save(path)

    # ── LaTeX stub ───────────────────────────────────────────────

    def _generate_latex_stub(
        self,
        output_path: str,
        pipeline_results: Dict[str, Any],
    ) -> str:
        """Generate a LaTeX document stub.

        Sprint 3 placeholder — full LaTeX support will use jinja2 templates.
        """
        path = str(Path(output_path).with_suffix(".tex"))
        cfg = self._report_config

        sections_list = []
        if "eis" in pipeline_results:
            sections_list.append("EIS Analysis")
        if "cycling" in pipeline_results:
            sections_list.append("Cycling Analysis")
        if "drt" in pipeline_results:
            sections_list.append("DRT Analysis")

        sections_tex = "\n".join(
            f"\\section{{{s}}}\n% TODO: Add content for {s}\n" for s in sections_list
        )

        latex = (
            "\\documentclass[12pt,a4paper]{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{hyperref}\n\n"
            f"\\title{{{cfg.title}}}\n"
            f"\\author{{{cfg.author}}}\n"
            "\\date{\\today}\n\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
            "\\begin{abstract}\n"
            "% TODO: Add executive summary here\n"
            "\\end{abstract}\n\n"
            f"{sections_tex}\n"
            "\\section{References}\n"
            "% TODO: Use BibTeX for references\n\n"
            "\\end{document}\n"
        )

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(latex, encoding="utf-8")
        return path

    # ── DOCX stub ────────────────────────────────────────────────

    def _generate_docx_stub(self, output_path: str) -> str:
        """Generate a DOCX placeholder.

        Sprint 3 placeholder — full DOCX support requires python-docx.
        """
        path = str(Path(output_path).with_suffix(".docx.txt"))
        cfg = self._report_config

        content = (
            f"DOCX Export Placeholder\n"
            f"=======================\n"
            f"Title: {cfg.title}\n"
            f"Author: {cfg.author}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"Full DOCX generation requires python-docx.\n"
            f"Install with: pip install python-docx\n"
        )

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return path
