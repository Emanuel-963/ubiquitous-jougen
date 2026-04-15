"""AI Analysis Panel — pure-logic builder for the "🤖 Análise IA" tab.

Like all other tab modules in ``src/gui/tabs/``, this module exposes
**pure functions and dataclasses** that accept data and return
structured results.  No tkinter, no side-effects, fully testable.

The actual GUI widget will call these functions and render the results
inside a CustomTkinter frame.

Architecture
------------

1. :func:`run_ai_analysis` — orchestrates InferenceEngine,
   PerformancePredictor and ProcessAdvisor into a single
   :class:`AIPanelResult`.
2. :func:`format_findings_text` / :func:`format_anomalies_text` /
   :func:`format_recommendations_text` / :func:`format_predictions_text` /
   :func:`format_process_text` — render each section as Markdown-ish
   plain text ready for a ``CTkTextbox``.
3. :func:`build_executive_summary` — one-paragraph executive summary.
4. :class:`AIPanelConfig` — scope / detail-level knobs the user
   toggles in the side panel.

Usage (inside GUI code)::

    from src.gui.tabs.ai_panel import run_ai_analysis, AIPanelConfig
    cfg = AIPanelConfig(scope_eis=True, scope_cycling=True, detail="full")
    result = run_ai_analysis(state, cfg)
    textbox.insert("1.0", result.formatted_report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.ai.inference_engine import (
    AnalysisReport,
    Anomaly,
    Finding,
    InferenceEngine,
    Priority,
    Recommendation,
)
from src.ai.performance_predictor import (
    CyclingPrediction,
    Improvement,
    PerformancePredictor,
)
from src.ai.process_advisor import (
    ProcessAdvisor,
    ProcessReport,
    ProductionRec,
)
from src.config import PipelineConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Configuration for the panel
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AIPanelConfig:
    """User-facing settings for the AI analysis panel.

    Attributes
    ----------
    scope_eis : bool
        Include EIS data in the analysis.
    scope_cycling : bool
        Include cycling data.
    scope_drt : bool
        Include DRT data.
    detail : str
        ``"summary"`` for concise output, ``"full"`` for complete report.
    """

    scope_eis: bool = True
    scope_cycling: bool = True
    scope_drt: bool = True
    detail: str = "full"

    @property
    def is_summary(self) -> bool:
        """Return True when the user selected summary mode."""
        return self.detail == "summary"

    @property
    def is_full(self) -> bool:
        """Return True when the user selected full-detail mode."""
        return self.detail == "full"


# ═══════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AIPanelResult:
    """Complete output of a single AI analysis run.

    This is the only object the View needs to render the panel.
    """

    # ── Raw AI results ──────────────────────────────────────────
    inference_report: AnalysisReport = field(default_factory=AnalysisReport)
    cycling_prediction: Optional[CyclingPrediction] = None
    improvements: List[Improvement] = field(default_factory=list)
    process_report: Optional[ProcessReport] = None

    # ── Pre-formatted text sections ─────────────────────────────
    executive_summary: str = ""
    findings_text: str = ""
    anomalies_text: str = ""
    recommendations_text: str = ""
    predictions_text: str = ""
    process_text: str = ""
    formatted_report: str = ""

    # ── Metadata ────────────────────────────────────────────────
    pipelines_used: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    n_findings: int = 0
    n_anomalies: int = 0
    n_recommendations: int = 0

    @property
    def has_predictions(self) -> bool:
        """True if a cycling prediction was generated."""
        return self.cycling_prediction is not None

    @property
    def has_process_report(self) -> bool:
        """True if a process report was generated."""
        return self.process_report is not None


# ═══════════════════════════════════════════════════════════════════════
#  Text formatters
# ═══════════════════════════════════════════════════════════════════════

def format_findings_text(findings: Sequence[Finding], *, summary: bool = False) -> str:
    """Render findings as plain text.

    Parameters
    ----------
    findings : sequence of Finding
    summary : bool
        When True, limit to 5 items + count.
    """
    if not findings:
        return "No findings to report."
    lines: List[str] = []
    limit = 5 if summary else len(findings)
    for i, f in enumerate(findings[:limit]):
        lines.append(f"  • {f}")
    if summary and len(findings) > limit:
        lines.append(f"  … and {len(findings) - limit} more.")
    return "\n".join(lines)


def format_anomalies_text(anomalies: Sequence[Anomaly], *, summary: bool = False) -> str:
    """Render anomalies as plain text."""
    if not anomalies:
        return "No anomalies detected."
    lines: List[str] = []
    limit = 5 if summary else len(anomalies)
    for a in anomalies[:limit]:
        lines.append(f"  • {a}")
    if summary and len(anomalies) > limit:
        lines.append(f"  … and {len(anomalies) - limit} more.")
    return "\n".join(lines)


def format_recommendations_text(
    recommendations: Sequence[Recommendation],
    *,
    summary: bool = False,
) -> str:
    """Render recommendations as a numbered list."""
    if not recommendations:
        return "No recommendations at this time."
    lines: List[str] = []
    limit = 5 if summary else len(recommendations)
    for i, r in enumerate(recommendations[:limit], 1):
        lines.append(f"  {i}. {r}")
    if summary and len(recommendations) > limit:
        lines.append(f"  … and {len(recommendations) - limit} more.")
    return "\n".join(lines)


def format_predictions_text(
    prediction: Optional[CyclingPrediction],
    improvements: Sequence[Improvement],
    *,
    summary: bool = False,
) -> str:
    """Render cycling prediction and improvements as text."""
    if prediction is None:
        return "No predictions available (EIS data required)."
    parts: List[str] = []

    # Main predictions
    if prediction.energy is not None:
        parts.append(f"  • Estimated energy: {prediction.energy:.1f} µJ")
    if prediction.power is not None:
        parts.append(f"  • Estimated power: {prediction.power:.1f} µW")
    if prediction.retention is not None:
        parts.append(f"  • Estimated retention: {prediction.retention:.1f} %")
    parts.append(f"  • Confidence: {prediction.confidence:.0%} ({prediction.method})")
    if prediction.explanation and not summary:
        parts.append(f"  • {prediction.explanation}")

    # Improvements
    if improvements:
        parts.append("")
        parts.append("  Suggested improvements:")
        limit = 3 if summary else len(improvements)
        for imp in improvements[:limit]:
            parts.append(f"    – {imp}")
        if summary and len(improvements) > limit:
            parts.append(f"    … and {len(improvements) - limit} more.")

    return "\n".join(parts) if parts else "No predictions available."


def format_process_text(
    report: Optional[ProcessReport],
    *,
    summary: bool = False,
) -> str:
    """Render process advisor report as text."""
    if report is None:
        return "Process analysis not available."
    parts: List[str] = []

    parts.append(f"  {report.material_assessment}")

    if report.best_conditions:
        bc = ", ".join(f"{k}={v}" for k, v in report.best_conditions.items())
        parts.append(f"  Best conditions: {bc}")

    if report.bottleneck_analysis:
        parts.append(f"  Bottleneck: {report.bottleneck_analysis}")

    if report.production_recommendations:
        parts.append("")
        parts.append("  Production recommendations:")
        limit = 3 if summary else len(report.production_recommendations)
        for rec in report.production_recommendations[:limit]:
            parts.append(f"    – {rec}")
        if summary and len(report.production_recommendations) > limit:
            parts.append(
                f"    … and {len(report.production_recommendations) - limit} more."
            )

    if report.next_experiments and not summary:
        parts.append("")
        parts.append("  Suggested next experiments:")
        for exp in report.next_experiments:
            parts.append(f"    – {exp}")

    return "\n".join(parts) if parts else "Process analysis not available."


def build_executive_summary(
    report: AnalysisReport,
    prediction: Optional[CyclingPrediction] = None,
    process: Optional[ProcessReport] = None,
) -> str:
    """Build a concise executive summary paragraph."""
    parts: List[str] = []

    # From inference engine
    if report.summary:
        parts.append(report.summary)
    else:
        parts.append(
            f"Analysis of {report.sample_count} sample(s) "
            f"using {', '.join(report.pipelines_used) or 'no'} pipeline(s)."
        )
        parts.append(f"Quality score: {report.quality_score:.0f}/100.")
        parts.append(
            f"Found {len(report.findings)} observation(s), "
            f"{len(report.anomalies)} anomaly(ies) and "
            f"{len(report.recommendations)} recommendation(s)."
        )

    # From performance predictor
    if prediction is not None and prediction.retention is not None:
        parts.append(
            f"Predicted cycling retention: {prediction.retention:.1f}% "
            f"(confidence {prediction.confidence:.0%})."
        )

    # From process advisor
    if process is not None and process.best_conditions:
        bc = process.best_conditions
        best_elec = bc.get("electrolyte", "")
        if best_elec:
            parts.append(f"Best electrolyte: {best_elec}.")

    return " ".join(parts)


def _assemble_full_report(
    *,
    executive: str,
    findings: str,
    anomalies: str,
    recommendations: str,
    predictions: str,
    process: str,
    quality_score: float,
    summary_mode: bool,
) -> str:
    """Concatenate all sections into a single text block."""
    sep = "─" * 48
    sections: List[str] = []

    sections.append("📊 Executive Summary")
    sections.append(sep)
    sections.append(executive)
    sections.append("")

    sections.append(f"⚙️ Quality Score: {quality_score:.0f}/100")
    sections.append("")

    sections.append("🔍 Findings")
    sections.append(sep)
    sections.append(findings)
    sections.append("")

    sections.append("⚠️ Anomalies")
    sections.append(sep)
    sections.append(anomalies)
    sections.append("")

    sections.append("💡 Recommendations")
    sections.append(sep)
    sections.append(recommendations)
    sections.append("")

    sections.append("🔮 Predictions")
    sections.append(sep)
    sections.append(predictions)
    sections.append("")

    sections.append("🏭 Process Analysis")
    sections.append(sep)
    sections.append(process)

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════

def run_ai_analysis(
    state: Any,
    panel_config: Optional[AIPanelConfig] = None,
    *,
    pipeline_config: Optional[PipelineConfig] = None,
    feature_store: Any = None,
) -> AIPanelResult:
    """Run a complete AI analysis using whatever data is available.

    Parameters
    ----------
    state : AppState
        The current application state containing EIS, cycling, DRT data.
    panel_config : AIPanelConfig | None
        UI scope / detail settings.  Defaults to all-scope, full-detail.
    pipeline_config : PipelineConfig | None
        Pipeline configuration for the AI modules.
    feature_store : FeatureStore | None
        Historical fitting data for ML predictions.

    Returns
    -------
    AIPanelResult
        Complete result with raw objects and pre-formatted text.
    """
    if panel_config is None:
        panel_config = AIPanelConfig()
    if pipeline_config is None:
        pipeline_config = PipelineConfig.default()

    result = AIPanelResult()
    summary_mode = panel_config.is_summary

    # ── Determine what data to include ─────────────────────────

    eis_result = None
    cycling_result = None
    drt_result = None

    if panel_config.scope_eis and _has_eis(state):
        eis_result = _build_eis_proxy(state)
    if panel_config.scope_cycling and _has_cycling(state):
        cycling_result = _build_cycling_proxy(state)
    if panel_config.scope_drt and _has_drt(state):
        drt_result = _build_drt_proxy(state)

    if eis_result is None and cycling_result is None and drt_result is None:
        result.executive_summary = "No data available for AI analysis."
        result.formatted_report = result.executive_summary
        return result

    # ── 1. Inference engine ────────────────────────────────────

    try:
        engine = InferenceEngine(config=pipeline_config)
        report = engine.analyze(eis_result, cycling_result, drt_result)
    except Exception as exc:
        logger.warning("InferenceEngine failed: %s", exc)
        report = AnalysisReport(summary=f"Inference failed: {exc}")

    result.inference_report = report
    result.pipelines_used = list(report.pipelines_used)
    result.quality_score = report.quality_score
    result.n_findings = len(report.findings)
    result.n_anomalies = len(report.anomalies)
    result.n_recommendations = len(report.recommendations)

    # ── 2. Performance predictor ───────────────────────────────

    if eis_result is not None:
        try:
            predictor = PerformancePredictor(
                feature_store=feature_store,
                config=pipeline_config,
            )
            result.cycling_prediction = predictor.predict_cycling_from_result(
                eis_result,
            )
            result.improvements = predictor.recommend_improvements_from_result(
                eis_result,
            )
        except Exception as exc:
            logger.warning("PerformancePredictor failed: %s", exc)

    # ── 3. Process advisor ─────────────────────────────────────

    if eis_result is not None:
        try:
            advisor = ProcessAdvisor(config=pipeline_config)
            entries = _build_process_entries(state, panel_config)
            if entries:
                result.process_report = advisor.analyze_material_system(entries)
        except Exception as exc:
            logger.warning("ProcessAdvisor failed: %s", exc)

    # ── 4. Format text ─────────────────────────────────────────

    result.findings_text = format_findings_text(
        report.findings, summary=summary_mode,
    )
    result.anomalies_text = format_anomalies_text(
        report.anomalies, summary=summary_mode,
    )
    result.recommendations_text = format_recommendations_text(
        report.recommendations, summary=summary_mode,
    )
    result.predictions_text = format_predictions_text(
        result.cycling_prediction, result.improvements, summary=summary_mode,
    )
    result.process_text = format_process_text(
        result.process_report, summary=summary_mode,
    )
    result.executive_summary = build_executive_summary(
        report, result.cycling_prediction, result.process_report,
    )
    result.formatted_report = _assemble_full_report(
        executive=result.executive_summary,
        findings=result.findings_text,
        anomalies=result.anomalies_text,
        recommendations=result.recommendations_text,
        predictions=result.predictions_text,
        process=result.process_text,
        quality_score=result.quality_score,
        summary_mode=summary_mode,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
#  State introspection helpers  (AppState → AI module inputs)
# ═══════════════════════════════════════════════════════════════════════

def _has_eis(state: Any) -> bool:
    """Check if state has EIS data."""
    rank = getattr(state, "rank_df", None)
    return rank is not None and hasattr(rank, "empty") and not rank.empty


def _has_cycling(state: Any) -> bool:
    cic = getattr(state, "cic_df", None)
    return cic is not None and hasattr(cic, "empty") and not cic.empty


def _has_drt(state: Any) -> bool:
    drt = getattr(state, "drt_df", None)
    return drt is not None and hasattr(drt, "empty") and not drt.empty


class _SimpleProxy:
    """Lightweight proxy that quacks like EISResult / CyclingResult / DRTPipelineResult."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _build_eis_proxy(state: Any) -> Any:
    """Build a minimal EIS-result-like object from AppState."""
    return _SimpleProxy(
        ranked_df=getattr(state, "rank_df", None),
        features_df=getattr(state, "eis_df", None),
        raw_eis=getattr(state, "raw_eis", {}),
    )


def _build_cycling_proxy(state: Any) -> Any:
    """Build a minimal cycling-result-like object from AppState."""
    return _SimpleProxy(
        merged_table=getattr(state, "cic_df", None),
        results=getattr(state, "cic_results", {}),
    )


def _build_drt_proxy(state: Any) -> Any:
    """Build a minimal DRT-result-like object from AppState."""
    return _SimpleProxy(
        drt_table=getattr(state, "drt_df", None),
        drt_peaks_table=getattr(state, "drt_peaks_df", None),
        drt_summary_table=getattr(state, "drt_summary_df", None),
        per_file_results=getattr(state, "drt_results", {}),
    )


def _build_process_entries(
    state: Any,
    panel_config: AIPanelConfig,
) -> List[Dict[str, Any]]:
    """Build entries list for ProcessAdvisor from the AppState.

    Each unique sample filename becomes one entry with its EIS row as
    a tiny ranked_df.
    """
    import pandas as pd

    entries: List[Dict[str, Any]] = []

    ranked = getattr(state, "rank_df", None)
    if ranked is None or not isinstance(ranked, pd.DataFrame) or ranked.empty:
        return entries

    # If the index contains filenames, use it; otherwise check 'Arquivo'
    if ranked.index.dtype == object and ranked.index.name != "Condition":
        for label in ranked.index:
            row_df = ranked.loc[[label]]
            entry: Dict[str, Any] = {"label": str(label)}
            if panel_config.scope_eis:
                entry["eis"] = _SimpleProxy(ranked_df=row_df)
            entries.append(entry)
    elif "Arquivo" in ranked.columns:
        for label in ranked["Arquivo"].unique():
            row_df = ranked[ranked["Arquivo"] == label]
            entry = {"label": str(label)}
            if panel_config.scope_eis:
                entry["eis"] = _SimpleProxy(ranked_df=row_df)
            entries.append(entry)
    else:
        # Fallback: treat the whole DataFrame as a single entry
        entries.append({
            "label": "all_samples",
            "eis": _SimpleProxy(ranked_df=ranked),
        })

    return entries
