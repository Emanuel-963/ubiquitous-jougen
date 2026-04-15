"""Rule-based inference engine for IonFlow AI.

The :class:`InferenceEngine` consumes typed pipeline results
(:class:`~src.models.EISResult`, :class:`~src.models.CyclingResult`,
:class:`~src.models.DRTPipelineResult`) and the knowledge base to produce
an :class:`AnalysisReport` containing:

* **Findings** — factual observations (e.g. "Rs = 2.66 Ω, classified as low")
* **Anomalies** — unexpected deviations (e.g. "Rp negative → fit did not converge")
* **Recommendations** — prioritised action items
* **Quality score** — 0–100 overall assessment
* **Executive summary** — natural-language paragraph

Cross-pipeline reasoning is supported: when both EIS and cycling data are
provided the engine can correlate impedance parameters with energy / retention.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.ai.knowledge_base import (
    KnowledgeBase,
    RuleMatch,
    Severity,
)
from src.config import PipelineConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Priority enum
# ═══════════════════════════════════════════════════════════════════════

class Priority(str, Enum):
    """Priority level for recommendations."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def __str__(self) -> str:  # noqa: D105
        return self.value


# ═══════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Finding:
    """A factual observation extracted from the data.

    Example: "Rs = 2.66 Ω, classificado como baixo."
    """

    parameter: str = ""
    value: Optional[float] = None
    description: str = ""
    category: str = "general"

    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.parameter} = {self.value:.4g} — {self.description}"
        return f"{self.parameter} — {self.description}"


@dataclass
class Anomaly:
    """A deviation from expected behaviour.

    Example: "Rp negativo → fitting não convergiu."
    """

    parameter: str = ""
    value: Optional[float] = None
    description: str = ""
    severity: Severity = Severity.WARNING

    def __str__(self) -> str:
        tag = f"[{self.severity.value.upper()}]"
        if self.value is not None:
            return f"{tag} {self.parameter} = {self.value:.4g} — {self.description}"
        return f"{tag} {self.parameter} — {self.description}"


@dataclass
class Recommendation:
    """A prioritised action suggestion.

    Example: "Reduzir resistência de contato: polir eletrodo."
    """

    text: str = ""
    priority: Priority = Priority.MEDIUM
    source_rule: str = ""
    """Rule ID that originated this recommendation (traceability)."""

    def __str__(self) -> str:
        return f"[{self.priority.value.upper()}] {self.text}"


@dataclass
class AnalysisReport:
    """Full report produced by :meth:`InferenceEngine.analyze`.

    Attributes
    ----------
    findings : list[Finding]
        Factual observations.
    anomalies : list[Anomaly]
        Unexpected deviations.
    recommendations : list[Recommendation]
        Sorted by priority (HIGH → MEDIUM → LOW).
    quality_score : float
        Overall quality 0–100 (higher = better).
    summary : str
        Executive summary paragraph.
    sample_count : int
        Number of samples analysed.
    pipelines_used : list[str]
        Which pipelines contributed data (``"eis"``, ``"cycling"``, ``"drt"``).
    """

    findings: List[Finding] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    quality_score: float = 0.0
    summary: str = ""
    sample_count: int = 0
    pipelines_used: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
#  Measurement extractor helpers
# ═══════════════════════════════════════════════════════════════════════

def _safe_median(series: pd.Series) -> Optional[float]:
    """Median of a numeric series, ignoring NaN.  Returns None if empty."""
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.median())


def _safe_mean(series: pd.Series) -> Optional[float]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _extract_eis_measurements(ranked_df: pd.DataFrame) -> Dict[str, float]:
    """Extract aggregate measurements from the ranked EIS DataFrame."""
    m: Dict[str, float] = {}
    col_map = {
        "Rs_fit": "Rs",
        "Rp_fit": "Rp",
        "Q": "Q",
        "n": "n",
        "Sigma": "Sigma",
        "C_mean": "C_mean",
        "C_lowfreq": "C_lowfreq",
        "Energy_mean": "Energy_mean",
        "Tau": "Tau",
        "Dispersion": "Dispersion",
        "Score": "Score",
        "Rank": "Rank",
        "Retenção (%)": "retention",
    }
    for col, key in col_map.items():
        if col in ranked_df.columns:
            val = _safe_median(ranked_df[col])
            if val is not None:
                m[key] = val
    return m


def _extract_cycling_measurements(
    cycling_result: Any,
) -> Dict[str, float]:
    """Extract aggregate measurements from a CyclingResult."""
    m: Dict[str, float] = {}
    merged = getattr(cycling_result, "merged_table", None)
    if merged is None or not isinstance(merged, pd.DataFrame) or merged.empty:
        return m

    for col, key in [
        ("Retenção (%)", "retention"),
        ("Energia (µJ)", "Energy_mean"),
        ("Potência (µW)", "Power_mean"),
    ]:
        if col in merged.columns:
            val = _safe_median(merged[col])
            if val is not None:
                m[key] = val
    return m


def _extract_drt_measurements(drt_result: Any) -> Dict[str, float]:
    """Extract aggregate measurements from a DRTPipelineResult."""
    m: Dict[str, float] = {}
    table = getattr(drt_result, "drt_table", None)
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return m

    # Count peaks (columns named tau_peak_1, tau_peak_2, ...)
    peak_cols = [c for c in table.columns if c.startswith("tau_peak_")]
    if peak_cols:
        # Count how many peaks on average per sample
        counts = table[peak_cols].notna().sum(axis=1)
        m["n_peaks"] = float(counts.median()) if not counts.empty else 0.0

    # Main peak position and amplitude
    if "tau_peak_1" in table.columns:
        val = _safe_median(table["tau_peak_1"])
        if val is not None:
            m["tau_peak_main"] = val
    if "gamma_peak_1" in table.columns:
        val = _safe_median(table["gamma_peak_1"])
        if val is not None:
            m["gamma_peak_main"] = val

    return m


# ═══════════════════════════════════════════════════════════════════════
#  Anomaly detectors
# ═══════════════════════════════════════════════════════════════════════

def _detect_anomalies_eis(ranked_df: pd.DataFrame) -> List[Anomaly]:
    """Detect per-sample anomalies in the EIS data."""
    anomalies: List[Anomaly] = []
    if ranked_df.empty:
        return anomalies

    # Negative Rp (fit didn't converge properly)
    if "Rp_fit" in ranked_df.columns:
        neg = ranked_df[ranked_df["Rp_fit"] < 0]
        for idx in neg.index:
            anomalies.append(Anomaly(
                parameter="Rp_fit",
                value=float(neg.loc[idx, "Rp_fit"]),
                description=f"Rp negativo na amostra '{idx}' — fitting pode não ter convergido.",
                severity=Severity.CRITICAL,
            ))

    # Negative Rs
    if "Rs_fit" in ranked_df.columns:
        neg = ranked_df[ranked_df["Rs_fit"] < 0]
        for idx in neg.index:
            anomalies.append(Anomaly(
                parameter="Rs_fit",
                value=float(neg.loc[idx, "Rs_fit"]),
                description=f"Rs negativo na amostra '{idx}' — possível artefato ou fitting falho.",
                severity=Severity.CRITICAL,
            ))

    # n outside [0, 1]
    if "n" in ranked_df.columns:
        bad = ranked_df[(ranked_df["n"] < 0) | (ranked_df["n"] > 1.05)]
        for idx in bad.index:
            anomalies.append(Anomaly(
                parameter="n",
                value=float(bad.loc[idx, "n"]),
                description=f"CPE exponent fora de [0, 1] na amostra '{idx}' — fitting não-físico.",
                severity=Severity.CRITICAL,
            ))

    # Very high CV in Rs (inter-replica instability)
    if "Rs_fit" in ranked_df.columns and len(ranked_df) >= 3:
        vals = pd.to_numeric(ranked_df["Rs_fit"], errors="coerce").dropna()
        if len(vals) >= 3 and vals.mean() != 0:
            cv = vals.std() / abs(vals.mean())
            if cv > 0.5:
                anomalies.append(Anomaly(
                    parameter="Rs_fit",
                    value=round(cv, 3),
                    description=f"CV de Rs = {cv:.0%} — alta variabilidade entre amostras.",
                    severity=Severity.WARNING,
                ))

    return anomalies


def _detect_anomalies_drt(drt_result: Any) -> List[Anomaly]:
    """Detect anomalies in DRT results."""
    anomalies: List[Anomaly] = []
    errors = getattr(drt_result, "errors", {})
    if errors:
        for fname, msg in errors.items():
            anomalies.append(Anomaly(
                parameter="DRT",
                description=f"Erro no DRT da amostra '{fname}': {msg}",
                severity=Severity.WARNING,
            ))
    return anomalies


# ═══════════════════════════════════════════════════════════════════════
#  Cross-pipeline reasoning
# ═══════════════════════════════════════════════════════════════════════

def _cross_pipeline_findings(
    eis_m: Dict[str, float],
    cyc_m: Dict[str, float],
    drt_m: Dict[str, float],
) -> List[Finding]:
    """Generate findings that combine data from multiple pipelines."""
    findings: List[Finding] = []

    # Rs low + Power low → diffusion bottleneck
    rs = eis_m.get("Rs")
    power = cyc_m.get("Power_mean")
    if rs is not None and power is not None:
        if rs < 5.0 and power < 5.0:
            findings.append(Finding(
                parameter="Rs + Power",
                description=(
                    f"Rs baixo ({rs:.2f} Ω) mas potência baixa ({power:.2f} µW) "
                    "— gargalo pode ser difusão; verificar DRT."
                ),
                category="cross-pipeline",
            ))

    # Retention low + Rp stable → mechanical degradation
    retention = cyc_m.get("retention") or eis_m.get("retention")
    rp = eis_m.get("Rp")
    if retention is not None and rp is not None:
        if retention < 70.0 and rp < 100.0:
            findings.append(Finding(
                parameter="retention + Rp",
                description=(
                    f"Retenção baixa ({retention:.1f}%) com Rp moderado ({rp:.1f} Ω) "
                    "— sugere degradação mecânica, não eletroquímica."
                ),
                category="cross-pipeline",
            ))

    # DRT many peaks + n low → heterogeneous interface
    n_peaks = drt_m.get("n_peaks")
    n_val = eis_m.get("n")
    if n_peaks is not None and n_val is not None:
        if n_peaks >= 3 and n_val < 0.7:
            findings.append(Finding(
                parameter="n + DRT peaks",
                description=(
                    f"n baixo ({n_val:.2f}) com {int(n_peaks)} picos DRT "
                    "— interface muito heterogénea com processos sobrepostos."
                ),
                category="cross-pipeline",
            ))

    # High sigma + slow DRT peak → diffusion dominated
    sigma = eis_m.get("Sigma")
    tau_main = drt_m.get("tau_peak_main")
    if sigma is not None and tau_main is not None:
        if sigma > 50.0 and tau_main > 0.1:
            findings.append(Finding(
                parameter="Sigma + tau_peak",
                description=(
                    f"Sigma alto ({sigma:.1f}) e pico DRT lento (τ = {tau_main:.3g} s) "
                    "— sistema dominado por difusão."
                ),
                category="cross-pipeline",
            ))

    # Rs comparison across electrolytes (if multiple samples)
    rs_val = eis_m.get("Rs")
    energy = eis_m.get("Energy_mean") or cyc_m.get("Energy_mean")
    if rs_val is not None and energy is not None:
        if rs_val > 10.0 and energy < 5.0:
            findings.append(Finding(
                parameter="Rs + Energy",
                description=(
                    f"Rs elevado ({rs_val:.1f} Ω) com energia baixa ({energy:.2f}) "
                    "— a resistência ôhmica é provavelmente o principal limitante."
                ),
                category="cross-pipeline",
            ))

    return findings


# ═══════════════════════════════════════════════════════════════════════
#  Quality score calculator
# ═══════════════════════════════════════════════════════════════════════

def _compute_quality_score(
    anomalies: List[Anomaly],
    matches: List[RuleMatch],
) -> float:
    """Compute 0–100 quality score.

    Starts at 100 and deducts points for anomalies and rule matches:
    * Critical anomaly: −15
    * Warning anomaly: −5
    * Critical rule match: −8
    * Warning rule match: −3
    """
    score = 100.0
    for a in anomalies:
        if a.severity == Severity.CRITICAL:
            score -= 15.0
        elif a.severity == Severity.WARNING:
            score -= 5.0
    for m in matches:
        if m.rule.severity == Severity.CRITICAL:
            score -= 8.0
        elif m.rule.severity == Severity.WARNING:
            score -= 3.0
    return max(0.0, min(100.0, score))


# ═══════════════════════════════════════════════════════════════════════
#  Summary generator
# ═══════════════════════════════════════════════════════════════════════

_PRIORITY_FROM_SEVERITY = {
    Severity.CRITICAL: Priority.HIGH,
    Severity.WARNING: Priority.MEDIUM,
    Severity.INFO: Priority.LOW,
}


def _build_recommendations(matches: List[RuleMatch]) -> List[Recommendation]:
    """Convert rule matches into de-duplicated, sorted recommendations."""
    seen: set = set()
    recs: List[Recommendation] = []
    for m in matches:
        for text in m.rule.recommendations:
            if text in seen:
                continue
            seen.add(text)
            recs.append(Recommendation(
                text=text,
                priority=_PRIORITY_FROM_SEVERITY.get(m.rule.severity, Priority.MEDIUM),
                source_rule=m.rule.rule_id,
            ))
    prio_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    recs.sort(key=lambda r: prio_order.get(r.priority, 9))
    return recs


def _build_findings_from_matches(matches: List[RuleMatch]) -> List[Finding]:
    """Convert rule matches into findings."""
    findings: List[Finding] = []
    for m in matches:
        findings.append(Finding(
            parameter=m.rule.parameter,
            value=m.actual_value,
            description=m.rule.interpretation,
            category=m.rule.category,
        ))
    return findings


def _generate_summary(
    report: AnalysisReport,
) -> str:
    """Generate an executive summary paragraph for the report."""
    parts: List[str] = []

    n = report.sample_count
    pipes = ", ".join(report.pipelines_used) or "nenhum"
    parts.append(
        f"Análise realizada sobre {n} amostra(s) "
        f"utilizando os pipelines: {pipes}."
    )

    # Quality
    qs = report.quality_score
    if qs >= 80:
        parts.append(f"Qualidade geral: EXCELENTE ({qs:.0f}/100).")
    elif qs >= 60:
        parts.append(f"Qualidade geral: BOA ({qs:.0f}/100).")
    elif qs >= 40:
        parts.append(f"Qualidade geral: MODERADA ({qs:.0f}/100).")
    else:
        parts.append(f"Qualidade geral: CRÍTICA ({qs:.0f}/100) — atenção necessária.")

    # Anomalies
    n_crit = sum(1 for a in report.anomalies if a.severity == Severity.CRITICAL)
    n_warn = sum(1 for a in report.anomalies if a.severity == Severity.WARNING)
    if n_crit or n_warn:
        parts.append(
            f"Anomalias detectadas: {n_crit} crítica(s), {n_warn} aviso(s)."
        )

    # Recommendations
    n_high = sum(1 for r in report.recommendations if r.priority == Priority.HIGH)
    n_total = len(report.recommendations)
    if n_total:
        parts.append(
            f"Total de {n_total} recomendação(ões), "
            f"sendo {n_high} de alta prioridade."
        )

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  InferenceEngine
# ═══════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """Rule-based inference engine for electrochemical analysis.

    Parameters
    ----------
    knowledge_base : KnowledgeBase | None
        If ``None``, :meth:`KnowledgeBase.default` is used.
    config : PipelineConfig | None
        Used to extract thresholds for rule evaluation.
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self._kb = knowledge_base if knowledge_base is not None else KnowledgeBase.default()
        self._config = config if config is not None else PipelineConfig.default()
        self._thresholds = KnowledgeBase.thresholds_from_config(self._config)

    # ── Public API ────────────────────────────────────────────────────

    @property
    def knowledge_base(self) -> KnowledgeBase:
        """The knowledge base in use."""
        return self._kb

    def analyze(
        self,
        eis_result: Any = None,
        cycling_result: Any = None,
        drt_result: Any = None,
    ) -> AnalysisReport:
        """Run the full inference pipeline.

        Parameters
        ----------
        eis_result : EISResult | None
        cycling_result : CyclingResult | None
        drt_result : DRTPipelineResult | None

        Returns
        -------
        AnalysisReport
        """
        report = AnalysisReport()

        # ── Identify available pipelines ──────────────────────────────
        pipelines: List[str] = []
        ranked_df = pd.DataFrame()
        if eis_result is not None:
            pipelines.append("eis")
            ranked_df = getattr(eis_result, "ranked_df", pd.DataFrame())
            if ranked_df is None:
                ranked_df = pd.DataFrame()
        if cycling_result is not None:
            pipelines.append("cycling")
        if drt_result is not None:
            pipelines.append("drt")
        report.pipelines_used = pipelines

        # Count samples
        report.sample_count = len(ranked_df)

        # ── Extract aggregate measurements ────────────────────────────
        eis_m = _extract_eis_measurements(ranked_df) if not ranked_df.empty else {}
        cyc_m = _extract_cycling_measurements(cycling_result) if cycling_result else {}
        drt_m = _extract_drt_measurements(drt_result) if drt_result else {}

        # Merge all measurements for rule evaluation
        all_measurements: Dict[str, float] = {}
        all_measurements.update(eis_m)
        all_measurements.update(cyc_m)
        all_measurements.update(drt_m)

        # ── Evaluate knowledge base rules ─────────────────────────────
        matches = self._kb.evaluate(all_measurements, self._thresholds)

        # ── Build findings ────────────────────────────────────────────
        report.findings = _build_findings_from_matches(matches)

        # Add cross-pipeline findings
        cross = _cross_pipeline_findings(eis_m, cyc_m, drt_m)
        report.findings.extend(cross)

        # ── Detect anomalies ──────────────────────────────────────────
        report.anomalies = _detect_anomalies_eis(ranked_df)
        if drt_result is not None:
            report.anomalies.extend(_detect_anomalies_drt(drt_result))

        # ── Build recommendations ─────────────────────────────────────
        report.recommendations = _build_recommendations(matches)

        # ── Compute quality score ─────────────────────────────────────
        report.quality_score = _compute_quality_score(report.anomalies, matches)

        # ── Generate summary ──────────────────────────────────────────
        report.summary = _generate_summary(report)

        logger.info(
            "InferenceEngine analysis complete: %d findings, %d anomalies, "
            "%d recommendations, quality=%.0f",
            len(report.findings),
            len(report.anomalies),
            len(report.recommendations),
            report.quality_score,
        )
        return report

    def analyze_sample(
        self,
        measurements: Dict[str, float],
        *,
        categories: Optional[Sequence[str]] = None,
    ) -> AnalysisReport:
        """Lightweight analysis from a flat measurements dict.

        Useful for single-sample quick checks without building full
        pipeline result objects.
        """
        matches = self._kb.evaluate(
            measurements, self._thresholds, categories=categories,
        )
        report = AnalysisReport(
            findings=_build_findings_from_matches(matches),
            recommendations=_build_recommendations(matches),
            quality_score=_compute_quality_score([], matches),
            sample_count=1,
            pipelines_used=list(categories) if categories else ["manual"],
        )
        report.summary = _generate_summary(report)
        return report
