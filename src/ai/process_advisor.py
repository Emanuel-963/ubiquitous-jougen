"""Process advisor for IonFlow AI (Day 18).

Analyses multiple EIS / Cycling / DRT results **across experimental
conditions** (electrolyte, treatment, substrate) and produces actionable
production recommendations.

Core capabilities
-----------------
1. **Material system assessment** — aggregate metrics per condition and
   rank them.
2. **Bottleneck analysis** — identify the single biggest limiting factor
   for each condition.
3. **Production recommendations** — concrete, metadata-aware suggestions
   (e.g. "H₂SO₄ yields Rs 78 % lower → prioritise as electrolyte").
4. **Comparison table** — side-by-side DataFrame of all conditions.
5. **Next experiments** — gap-driven suggestions for unexplored parameter
   space.

Usage
-----
>>> from src.ai import ProcessAdvisor
>>> advisor = ProcessAdvisor()
>>> report = advisor.analyze_material_system(
...     all_results=[
...         {"label": "Li2SO4_0.1A_GCT", "eis": eis1},
...         {"label": "H2SO4_1A_GC",     "eis": eis2},
...     ],
... )
>>> print(report.material_assessment)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.metadata import extract_metadata

logger = logging.getLogger(__name__)

# EIS columns we aggregate per condition
_AGG_EIS_COLS: Tuple[str, ...] = (
    "Rs_fit",
    "Rp_fit",
    "Q",
    "n",
    "Sigma",
    "C_mean",
    "Tau",
    "Dispersion",
    "Energy_mean",
    "Score",
)


# ═══════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════

class RecommendationArea(str, Enum):
    """Category of a production recommendation."""

    ELECTROLYTE = "electrolyte"
    TREATMENT = "treatment"
    SUBSTRATE = "substrate"
    PROCESS = "process"
    MEASUREMENT = "measurement"
    GENERAL = "general"

    def __str__(self) -> str:  # noqa: D105
        return self.value


# ═══════════════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ProductionRec:
    """A single production recommendation.

    Attributes
    ----------
    recommendation : str
        Actionable text (e.g. "Prioritise H₂SO₄ as electrolyte").
    area : RecommendationArea
        Category.
    rationale : str
        Why this is recommended.
    expected_impact : str
        Expected quantitative / qualitative impact.
    priority : int
        1 = highest, 3 = lowest.
    """

    recommendation: str = ""
    area: RecommendationArea = RecommendationArea.GENERAL
    rationale: str = ""
    expected_impact: str = ""
    priority: int = 2

    def __str__(self) -> str:
        return f"[P{self.priority}] {self.recommendation} — {self.expected_impact}"


@dataclass
class ProcessReport:
    """Full report produced by :class:`ProcessAdvisor`.

    Attributes
    ----------
    material_assessment : str
        High-level text assessment of the material system.
    best_conditions : dict
        ``{"electrolyte": "H2SO4", "treatment": "GCT", ...}``.
    bottleneck_analysis : str
        Plain-text description of the primary limiting factor.
    production_recommendations : list[ProductionRec]
        Ordered recommendations.
    comparison_table : pd.DataFrame
        One row per condition; columns = aggregated metrics.
    next_experiments : list[str]
        Suggestions for experiments not yet explored.
    n_conditions : int
        How many distinct conditions were analysed.
    """

    material_assessment: str = ""
    best_conditions: Dict[str, str] = field(default_factory=dict)
    bottleneck_analysis: str = ""
    production_recommendations: List[ProductionRec] = field(default_factory=list)
    comparison_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    next_experiments: List[str] = field(default_factory=list)
    n_conditions: int = 0


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _extract_condition(entry: Dict[str, Any]) -> Dict[str, str]:
    """Derive condition metadata from a result entry.

    The caller can supply explicit ``metadata`` or we fall back to
    parsing the ``label`` field with :func:`extract_metadata`.
    """
    meta = entry.get("metadata")
    if isinstance(meta, dict) and meta:
        return {
            "electrolyte": str(meta.get("electrolyte", "Unknown")),
            "current": str(meta.get("current", "Unknown")),
            "treatment": str(meta.get("treatment", "None")),
        }
    label = entry.get("label", "")
    if label:
        elec, cur, treat = extract_metadata(label)
        return {"electrolyte": elec, "current": cur, "treatment": treat}
    return {"electrolyte": "Unknown", "current": "Unknown", "treatment": "None"}


def _condition_key(cond: Dict[str, str]) -> str:
    """Build a hashable key for a condition dict."""
    return f"{cond['electrolyte']}_{cond['current']}_{cond['treatment']}"


def _extract_eis_metrics(eis_result: Any) -> Dict[str, float]:
    """Extract median EIS metrics from an EISResult or ranked DataFrame."""
    if eis_result is None:
        return {}
    ranked = getattr(eis_result, "ranked_df", None)
    if ranked is None:
        if isinstance(eis_result, pd.DataFrame):
            ranked = eis_result
        else:
            return {}
    if ranked.empty:
        return {}
    params: Dict[str, float] = {}
    for col in _AGG_EIS_COLS:
        if col in ranked.columns:
            vals = pd.to_numeric(ranked[col], errors="coerce").dropna()
            if not vals.empty:
                params[col] = float(vals.median())
    return params


def _extract_cycling_metrics(cycling_result: Any) -> Dict[str, float]:
    """Extract median cycling metrics from a CyclingResult."""
    if cycling_result is None:
        return {}
    merged = getattr(cycling_result, "merged_table", None)
    if merged is None or not isinstance(merged, pd.DataFrame) or merged.empty:
        return {}
    col_map = {
        "Retenção (%)": "retention",
        "Energia (µJ)": "energy",
        "Potência (µW)": "power",
    }
    params: Dict[str, float] = {}
    for col, key in col_map.items():
        if col in merged.columns:
            vals = pd.to_numeric(merged[col], errors="coerce").dropna()
            if not vals.empty:
                params[key] = float(vals.median())
    return params


def _extract_drt_metrics(drt_result: Any) -> Dict[str, float]:
    """Extract summary DRT metrics from a DRTPipelineResult."""
    if drt_result is None:
        return {}
    summary = getattr(drt_result, "drt_summary_table", None)
    if summary is None or not isinstance(summary, pd.DataFrame) or summary.empty:
        return {}
    params: Dict[str, float] = {}
    # Number of peaks
    if "n_peaks" in summary.columns:
        vals = pd.to_numeric(summary["n_peaks"], errors="coerce").dropna()
        if not vals.empty:
            params["n_peaks"] = float(vals.median())
    # Dominant tau
    if "dominant_tau" in summary.columns:
        vals = pd.to_numeric(summary["dominant_tau"], errors="coerce").dropna()
        if not vals.empty:
            params["dominant_tau"] = float(vals.median())
    return params


def _aggregate_condition(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate metrics across all entries that share the same condition."""
    all_metrics: Dict[str, List[float]] = {}
    for entry in entries:
        eis_m = _extract_eis_metrics(entry.get("eis"))
        cyc_m = _extract_cycling_metrics(entry.get("cycling"))
        drt_m = _extract_drt_metrics(entry.get("drt"))
        combined = {**eis_m, **cyc_m, **drt_m}
        for k, v in combined.items():
            all_metrics.setdefault(k, []).append(v)
    result: Dict[str, float] = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        result[k] = float(np.nanmedian(arr))
    return result


def _detect_outliers(
    entries: List[Dict[str, Any]],
    key: str = "Score",
) -> List[str]:
    """Detect labels whose *key* metric is > 2 σ from the condition mean."""
    values: List[Tuple[str, float]] = []
    for entry in entries:
        m = _extract_eis_metrics(entry.get("eis"))
        v = m.get(key)
        if v is not None:
            values.append((entry.get("label", "?"), v))
    if len(values) < 3:
        return []
    arr = np.array([v for _, v in values])
    mu, sigma = float(arr.mean()), float(arr.std(ddof=1))
    if sigma < 1e-15:
        return []
    outliers: List[str] = []
    for label, v in values:
        if abs(v - mu) > 2 * sigma:
            outliers.append(label)
    return outliers


def _identify_bottleneck(metrics: Dict[str, float]) -> Tuple[str, str]:
    """Identify the primary bottleneck from aggregated metrics.

    Returns ``(parameter_name, description)``.
    """
    rs = metrics.get("Rs_fit")
    rp = metrics.get("Rp_fit")
    n = metrics.get("n")
    sigma = metrics.get("Sigma")
    retention = metrics.get("retention")

    bottlenecks: List[Tuple[float, str, str]] = []

    if rs is not None and rs > 10:
        bottlenecks.append((rs / 10, "Rs_fit", f"High ohmic resistance (Rs = {rs:.2f} Ω)"))
    if rp is not None and rp > 500:
        bottlenecks.append((rp / 500, "Rp_fit", f"High charge-transfer resistance (Rp = {rp:.1f} Ω)"))
    if n is not None and n < 0.7:
        bottlenecks.append(((0.7 - n) / 0.7 * 5, "n", f"Low CPE exponent (n = {n:.3f}) — rough or porous surface"))
    if sigma is not None and sigma > 50:
        bottlenecks.append((sigma / 50, "Sigma", f"High Warburg coefficient (σ = {sigma:.1f}) — diffusion limited"))
    if retention is not None and retention < 80:
        bottlenecks.append(((80 - retention) / 80 * 5, "retention", f"Low cycling retention ({retention:.1f} %)"))

    if not bottlenecks:
        return ("none", "No significant bottleneck identified — system performs well overall.")

    bottlenecks.sort(key=lambda t: t[0], reverse=True)
    return (bottlenecks[0][1], bottlenecks[0][2])


def _compare_metric(
    cond_metrics: Dict[str, Dict[str, float]],
    metric: str,
    lower_is_better: bool = True,
) -> Optional[Tuple[str, str, float]]:
    """Find the best and worst condition for a given metric.

    Returns ``(best_cond_key, comparison_text, pct_diff)`` or ``None``.
    """
    vals: List[Tuple[str, float]] = []
    for ckey, m in cond_metrics.items():
        v = m.get(metric)
        if v is not None:
            vals.append((ckey, v))
    if len(vals) < 2:
        return None
    vals.sort(key=lambda t: t[1], reverse=not lower_is_better)
    best_key, best_val = vals[0]
    worst_key, worst_val = vals[-1]
    if abs(worst_val) < 1e-15:
        pct = 0.0
    else:
        pct = abs(best_val - worst_val) / abs(worst_val) * 100
    return (best_key, f"{metric}: best={best_key} ({best_val:.4g}), worst={worst_key} ({worst_val:.4g}), diff={pct:.0f}%", pct)


def _build_comparison_table(
    cond_metrics: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Build a DataFrame comparing all conditions side-by-side."""
    if not cond_metrics:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for ckey, metrics in cond_metrics.items():
        row: Dict[str, Any] = {"Condition": ckey}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "Condition" in df.columns:
        df = df.set_index("Condition")
    return df


def _suggest_next_experiments(
    cond_metrics: Dict[str, Dict[str, float]],
    seen_electrolytes: set,
    seen_treatments: set,
    seen_currents: set,
    bottleneck_param: str,
) -> List[str]:
    """Generate experiment suggestions based on gaps and bottlenecks."""
    all_electrolytes = {"Li2SO4", "LiCl", "H2SO4", "Na2SO4", "NaOH", "KOH"}
    all_treatments = {"GCT", "GC", "Steel316", "None"}
    all_currents = {"0.1A", "1A", "10A"}

    suggestions: List[str] = []

    # Untested electrolytes
    missing_elec = all_electrolytes - seen_electrolytes
    if missing_elec:
        suggestions.append(
            f"Test with untried electrolytes: {', '.join(sorted(missing_elec))}"
        )

    # Untested treatments
    missing_treat = all_treatments - seen_treatments
    if missing_treat:
        suggestions.append(
            f"Evaluate additional surface treatments: {', '.join(sorted(missing_treat))}"
        )

    # Untested currents
    missing_cur = all_currents - seen_currents
    if missing_cur:
        suggestions.append(
            f"Measure at additional current densities: {', '.join(sorted(missing_cur))}"
        )

    # Bottleneck-specific suggestions
    if bottleneck_param == "Rs_fit":
        suggestions.append(
            "Optimise ohmic resistance: test polished substrates or higher-conductivity electrolytes"
        )
    elif bottleneck_param == "Rp_fit":
        suggestions.append(
            "Reduce charge-transfer resistance: explore catalytic surface coatings or higher electrolyte concentrations"
        )
    elif bottleneck_param == "n":
        suggestions.append(
            "Improve surface homogeneity: try different deposition parameters or annealing temperatures"
        )
    elif bottleneck_param == "Sigma":
        suggestions.append(
            "Address diffusion limitation: extend EIS to lower frequencies (< 10 mHz) and test thinner electrodes"
        )
    elif bottleneck_param == "retention":
        suggestions.append(
            "Improve cycling stability: repeat with ≥ 5 000 cycles and vary charge/discharge rate"
        )

    # General suggestions when we have very few conditions
    if len(cond_metrics) < 3:
        suggestions.append(
            "Increase statistical power by testing more replicas per condition (n ≥ 3)"
        )

    return suggestions


def _generate_recommendations(
    cond_metrics: Dict[str, Dict[str, float]],
    best_conditions: Dict[str, str],
    bottleneck_param: str,
    outlier_map: Dict[str, List[str]],
) -> List[ProductionRec]:
    """Build prioritised production recommendations."""
    recs: List[ProductionRec] = []

    # --- Electrolyte recommendation ---
    best_elec = best_conditions.get("electrolyte")
    if best_elec and best_elec != "Unknown":
        # Compute Rs comparison
        elec_metrics: Dict[str, Dict[str, float]] = {}
        for ckey, m in cond_metrics.items():
            elec = ckey.split("_")[0]
            elec_metrics.setdefault(elec, {})
            for k, v in m.items():
                elec_metrics[elec].setdefault(k, [])  # type: ignore[arg-type]
                elec_metrics[elec][k].append(v)  # type: ignore[union-attr]
        # Merge lists to medians
        elec_agg: Dict[str, Dict[str, float]] = {}
        for elec, kv in elec_metrics.items():
            elec_agg[elec] = {}
            for k, vals in kv.items():
                if isinstance(vals, list) and vals:
                    elec_agg[elec][k] = float(np.nanmedian(vals))

        best_rs = elec_agg.get(best_elec, {}).get("Rs_fit")
        other_rs = [m.get("Rs_fit") for e, m in elec_agg.items() if e != best_elec and m.get("Rs_fit") is not None]
        if best_rs is not None and other_rs:
            avg_other = float(np.mean(other_rs))
            if avg_other > 0:
                pct = (avg_other - best_rs) / avg_other * 100
                recs.append(ProductionRec(
                    recommendation=f"Prioritise {best_elec} as electrolyte",
                    area=RecommendationArea.ELECTROLYTE,
                    rationale=f"{best_elec} yields Rs {pct:.0f}% lower than alternatives",
                    expected_impact="Lower ohmic losses and improved energy efficiency",
                    priority=1,
                ))
            else:
                recs.append(ProductionRec(
                    recommendation=f"Prioritise {best_elec} as electrolyte",
                    area=RecommendationArea.ELECTROLYTE,
                    rationale=f"{best_elec} shows best overall Score",
                    expected_impact="Improved electrochemical performance",
                    priority=1,
                ))
        else:
            recs.append(ProductionRec(
                recommendation=f"Prioritise {best_elec} as electrolyte",
                area=RecommendationArea.ELECTROLYTE,
                rationale=f"{best_elec} shows best overall Score",
                expected_impact="Improved electrochemical performance",
                priority=1,
            ))

    # --- Treatment recommendation ---
    best_treat = best_conditions.get("treatment")
    if best_treat and best_treat != "None" and best_treat != "Unknown":
        recs.append(ProductionRec(
            recommendation=f"Apply {best_treat} surface treatment in production",
            area=RecommendationArea.TREATMENT,
            rationale=f"{best_treat} associated with best-performing condition",
            expected_impact="Enhanced surface interface and charge transfer",
            priority=1,
        ))

    # --- Bottleneck-specific rec ---
    if bottleneck_param == "Rs_fit":
        recs.append(ProductionRec(
            recommendation="Reduce ohmic resistance — polish substrate or increase electrolyte concentration",
            area=RecommendationArea.PROCESS,
            rationale="Rs is the primary performance limiter",
            expected_impact="Significant improvement in energy efficiency and power capability",
            priority=1,
        ))
    elif bottleneck_param == "Rp_fit":
        recs.append(ProductionRec(
            recommendation="Reduce charge-transfer resistance via surface activation or catalyst deposition",
            area=RecommendationArea.PROCESS,
            rationale="Rp is the primary performance limiter",
            expected_impact="Faster charge transfer and higher capacitance",
            priority=1,
        ))
    elif bottleneck_param == "n":
        recs.append(ProductionRec(
            recommendation="Improve surface homogeneity — consider thermal treatment or electropolishing",
            area=RecommendationArea.TREATMENT,
            rationale="Low CPE exponent indicates surface irregularity",
            expected_impact="More uniform current distribution and reduced dispersion",
            priority=2,
        ))
    elif bottleneck_param == "Sigma":
        recs.append(ProductionRec(
            recommendation="Address diffusion limitation — reduce electrode thickness or increase porosity",
            area=RecommendationArea.PROCESS,
            rationale="High Warburg coefficient indicates sluggish mass transport",
            expected_impact="Improved power performance at high rates",
            priority=2,
        ))
    elif bottleneck_param == "retention":
        recs.append(ProductionRec(
            recommendation="Improve cycling stability — investigate binder, current collector adhesion",
            area=RecommendationArea.PROCESS,
            rationale="Low retention indicates material degradation during cycling",
            expected_impact="Extended device lifetime",
            priority=1,
        ))

    # --- Outlier recommendations ---
    for ckey, labels in outlier_map.items():
        if labels:
            recs.append(ProductionRec(
                recommendation=f"Investigate outlier sample(s) in {ckey}: {', '.join(labels)}",
                area=RecommendationArea.PROCESS,
                rationale="Outlier score suggests fabrication inconsistency",
                expected_impact="Improved batch reproducibility",
                priority=2,
            ))

    # --- Measurement recommendation if few conditions ---
    if len(cond_metrics) == 1:
        recs.append(ProductionRec(
            recommendation="Test additional conditions to enable comparative analysis",
            area=RecommendationArea.MEASUREMENT,
            rationale="Only one condition found — cannot rank alternatives",
            expected_impact="Data-driven optimisation of production parameters",
            priority=2,
        ))

    # Sort by priority
    recs.sort(key=lambda r: r.priority)
    return recs


# ═══════════════════════════════════════════════════════════════════════
#  ProcessAdvisor
# ═══════════════════════════════════════════════════════════════════════

class ProcessAdvisor:
    """Analyse a material system and recommend production improvements.

    Parameters
    ----------
    config : PipelineConfig | None
        Pipeline configuration (uses defaults when ``None``).
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self._config = config if config is not None else PipelineConfig.default()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_material_system(
        self,
        all_results: Sequence[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessReport:
        """Analyse results across conditions and produce a process report.

        Parameters
        ----------
        all_results : sequence of dict
            Each dict **must** have a ``"label"`` key (filename / sample id)
            and at least one of ``"eis"``, ``"cycling"`` or ``"drt"`` keys
            holding the corresponding result objects.

            Optionally, a ``"metadata"`` key may hold
            ``{"electrolyte": ..., "current": ..., "treatment": ...}``
            to override automatic filename-based extraction.

        metadata : dict | None
            Global metadata that applies to all entries when individual
            entries do not supply their own.

        Returns
        -------
        ProcessReport
        """
        if not all_results:
            logger.warning("ProcessAdvisor: no results provided.")
            return ProcessReport(
                material_assessment="No data provided for analysis.",
                bottleneck_analysis="Cannot identify bottleneck — no data.",
            )

        # ── 1. Group entries by condition ────────────────────────────
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for entry in all_results:
            if metadata and "metadata" not in entry:
                entry = {**entry, "metadata": metadata}
            cond = _extract_condition(entry)
            key = _condition_key(cond)
            grouped.setdefault(key, []).append(entry)

        logger.info("ProcessAdvisor: %d condition(s) identified: %s",
                     len(grouped), list(grouped.keys()))

        # ── 2. Aggregate metrics per condition ───────────────────────
        cond_metrics: Dict[str, Dict[str, float]] = {}
        for ckey, entries in grouped.items():
            cond_metrics[ckey] = _aggregate_condition(entries)

        # ── 3. Best conditions ───────────────────────────────────────
        best_conditions = self._find_best_conditions(cond_metrics)

        # ── 4. Bottleneck analysis ───────────────────────────────────
        # Use overall aggregated metrics
        all_vals: Dict[str, List[float]] = {}
        for m in cond_metrics.values():
            for k, v in m.items():
                all_vals.setdefault(k, []).append(v)
        overall = {k: float(np.nanmedian(v)) for k, v in all_vals.items()}
        bottleneck_param, bottleneck_text = _identify_bottleneck(overall)

        # ── 5. Outlier detection ─────────────────────────────────────
        outlier_map: Dict[str, List[str]] = {}
        for ckey, entries in grouped.items():
            outliers = _detect_outliers(entries)
            if outliers:
                outlier_map[ckey] = outliers

        # ── 6. Seen parameters ───────────────────────────────────────
        seen_electrolytes: set = set()
        seen_treatments: set = set()
        seen_currents: set = set()
        for entry in all_results:
            if metadata and "metadata" not in entry:
                entry = {**entry, "metadata": metadata}
            cond = _extract_condition(entry)
            seen_electrolytes.add(cond["electrolyte"])
            seen_treatments.add(cond["treatment"])
            seen_currents.add(cond["current"])

        # ── 7. Next experiments ──────────────────────────────────────
        next_experiments = _suggest_next_experiments(
            cond_metrics, seen_electrolytes, seen_treatments,
            seen_currents, bottleneck_param,
        )

        # ── 8. Production recommendations ────────────────────────────
        recommendations = _generate_recommendations(
            cond_metrics, best_conditions, bottleneck_param, outlier_map,
        )

        # ── 9. Comparison table ──────────────────────────────────────
        comparison = _build_comparison_table(cond_metrics)

        # ── 10. Material assessment text ─────────────────────────────
        assessment = self._build_assessment(
            cond_metrics, best_conditions, bottleneck_text, outlier_map,
        )

        return ProcessReport(
            material_assessment=assessment,
            best_conditions=best_conditions,
            bottleneck_analysis=bottleneck_text,
            production_recommendations=recommendations,
            comparison_table=comparison,
            next_experiments=next_experiments,
            n_conditions=len(grouped),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_best_conditions(
        self,
        cond_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, str]:
        """Determine the best electrolyte, treatment and current."""
        best: Dict[str, str] = {}

        # Group by electrolyte
        elec_scores: Dict[str, List[float]] = {}
        treat_scores: Dict[str, List[float]] = {}
        cur_scores: Dict[str, List[float]] = {}

        for ckey, m in cond_metrics.items():
            parts = ckey.split("_")
            score = m.get("Score", 0.0)
            if len(parts) >= 1:
                elec_scores.setdefault(parts[0], []).append(score)
            if len(parts) >= 3:
                treat_scores.setdefault(parts[2], []).append(score)
            if len(parts) >= 2:
                cur_scores.setdefault(parts[1], []).append(score)

        if elec_scores:
            best["electrolyte"] = max(
                elec_scores,
                key=lambda e: float(np.nanmedian(elec_scores[e])),
            )
        if treat_scores:
            best["treatment"] = max(
                treat_scores,
                key=lambda t: float(np.nanmedian(treat_scores[t])),
            )
        if cur_scores:
            best["current"] = max(
                cur_scores,
                key=lambda c: float(np.nanmedian(cur_scores[c])),
            )

        return best

    def _build_assessment(
        self,
        cond_metrics: Dict[str, Dict[str, float]],
        best_conditions: Dict[str, str],
        bottleneck_text: str,
        outlier_map: Dict[str, List[str]],
    ) -> str:
        """Compose the material assessment paragraph."""
        parts: List[str] = []

        n_cond = len(cond_metrics)
        parts.append(f"Analysed {n_cond} experimental condition(s).")

        if best_conditions:
            bc = ", ".join(f"{k}={v}" for k, v in best_conditions.items())
            parts.append(f"Best conditions: {bc}.")

        parts.append(f"Primary bottleneck: {bottleneck_text}")

        if outlier_map:
            total = sum(len(v) for v in outlier_map.values())
            parts.append(
                f"Detected {total} outlier sample(s) across {len(outlier_map)} condition(s) "
                "— possible fabrication inconsistency."
            )

        # Add metric highlights
        for ckey, m in cond_metrics.items():
            score = m.get("Score")
            rs = m.get("Rs_fit")
            if score is not None and rs is not None:
                parts.append(f"{ckey}: Score={score:.3f}, Rs={rs:.2f} Ω.")

        return " ".join(parts)
