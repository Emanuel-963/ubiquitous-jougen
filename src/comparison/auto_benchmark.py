"""Automatic sample benchmarking with objective-driven recommendations.

This module turns a table of fitted EIS results into actionable guidance,
for example:

- best setup for low Rs
- best setup for high Rp
- best setup for high capacitance with stability guardrails

It is designed for lab decision support, not only ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BenchmarkRecommendation:
    """Single recommendation output for an optimization objective."""

    objective: str
    sample: str
    score: float
    rationale: str
    top_candidates: List[Tuple[str, float]]


def _coalesce(
    df: pd.DataFrame, candidates: List[str], default: float = np.nan
) -> pd.Series:
    """Return first present column among candidates, as numeric series."""
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Min-max normalize to [0, 1] with direction handling."""
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return pd.Series(0.5, index=s.index)

    lo = float(s.min(skipna=True))
    hi = float(s.max(skipna=True))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = pd.Series(0.5, index=s.index)
    else:
        out = (s - lo) / (hi - lo)

    if not higher_is_better:
        out = 1.0 - out

    return out.fillna(0.5)


def prepare_benchmark_table(circuit_table: pd.DataFrame) -> pd.DataFrame:
    """Create a canonical benchmark table from heterogeneous circuit output columns."""
    if circuit_table is None or circuit_table.empty:
        return pd.DataFrame(
            columns=[
                "sample",
                "circuit",
                "rs",
                "rp",
                "c_mean",
                "chi2_over_nu",
                "kk_valid",
                "confidence",
            ]
        )

    df = circuit_table.copy()

    sample = (
        df["sample"].astype(str)
        if "sample" in df.columns
        else (
            df["Arquivo"].astype(str)
            if "Arquivo" in df.columns
            else pd.Series([f"sample_{i}" for i in range(len(df))], index=df.index)
        )
    )

    circuit = (
        df["best_circuit"].astype(str)
        if "best_circuit" in df.columns
        else (
            df["Circuito"].astype(str)
            if "Circuito" in df.columns
            else pd.Series(["unknown"] * len(df), index=df.index)
        )
    )

    rs = _coalesce(df, ["Rs_fit", "Rs", "rs"])
    rp = _coalesce(df, ["Rp_fit", "Rp", "rp"])
    c_mean = _coalesce(df, ["C_mean", "C", "Q", "c_mean"])
    chi2 = _coalesce(df, ["chi2_over_nu", "chi2", "Chi2_nu", "chi2_red"])
    confidence = _coalesce(df, ["confidence", "Confianca"])

    if "kk_valid" in df.columns:
        kk_valid = pd.to_numeric(df["kk_valid"], errors="coerce").fillna(0.0)
    elif "KK_valid" in df.columns:
        kk_valid = pd.to_numeric(df["KK_valid"], errors="coerce").fillna(0.0)
    else:
        kk_valid = pd.Series([np.nan] * len(df), index=df.index)

    out = pd.DataFrame(
        {
            "sample": sample,
            "circuit": circuit,
            "rs": rs,
            "rp": rp,
            "c_mean": c_mean,
            "chi2_over_nu": chi2,
            "kk_valid": kk_valid,
            "confidence": confidence,
        }
    )

    return out


def score_objective(
    bench: pd.DataFrame,
    objective: str,
) -> pd.Series:
    """Score rows for a named objective; higher score is better."""
    objective = objective.strip().lower()

    rs_good = _normalize(bench["rs"], higher_is_better=False)
    rp_good = _normalize(bench["rp"], higher_is_better=True)
    c_good = _normalize(bench["c_mean"], higher_is_better=True)

    # Lower chi2 is better; unknown chi2 gets neutral 0.5
    chi2_raw = bench["chi2_over_nu"].copy()
    chi2_raw = chi2_raw.where(np.isfinite(chi2_raw), np.nan)
    chi2_good = _normalize(
        chi2_raw.fillna(chi2_raw.median(skipna=True)), higher_is_better=False
    )
    chi2_good = chi2_good.where(chi2_raw.notna(), 0.5)

    conf_good = _normalize(bench["confidence"], higher_is_better=True)
    kk_good = bench["kk_valid"].copy()
    kk_good = kk_good.where(np.isfinite(kk_good), 0.5)
    kk_good = kk_good.clip(0.0, 1.0)

    if objective in {"low_rs", "min_rs", "conductivity"}:
        score = 0.55 * rs_good + 0.2 * chi2_good + 0.15 * kk_good + 0.1 * conf_good
    elif objective in {"high_rp", "stability", "corrosion_resistance"}:
        score = 0.5 * rp_good + 0.2 * chi2_good + 0.2 * kk_good + 0.1 * conf_good
    elif objective in {"high_capacitance", "capacitance", "energy_storage"}:
        score = 0.45 * c_good + 0.2 * rp_good + 0.2 * chi2_good + 0.15 * kk_good
    elif objective in {"balanced", "overall", "health"}:
        score = (
            0.23 * rs_good
            + 0.23 * rp_good
            + 0.19 * c_good
            + 0.15 * chi2_good
            + 0.1 * kk_good
            + 0.1 * conf_good
        )
    else:
        raise ValueError(
            "Unknown objective. Use: low_rs, high_rp, high_capacitance, balanced"
        )

    return score.fillna(0.0)


def recommend_best_configuration(
    circuit_table: pd.DataFrame,
    objective: str = "balanced",
    top_k: int = 3,
) -> BenchmarkRecommendation:
    """Compute objective-driven recommendation from circuit table."""
    bench = prepare_benchmark_table(circuit_table)
    if bench.empty:
        return BenchmarkRecommendation(
            objective=objective,
            sample="N/A",
            score=0.0,
            rationale="No benchmarkable rows available.",
            top_candidates=[],
        )

    score = score_objective(bench, objective)
    ranked = bench.assign(objective_score=score).sort_values(
        "objective_score", ascending=False
    )

    top = ranked.head(max(1, top_k))
    best = top.iloc[0]

    rationale = (
        f"Objective '{objective}' optimized with multi-criteria score "
        f"(electrical target + fit quality + KK consistency)."
    )

    top_candidates = [
        (str(row["sample"]), float(row["objective_score"])) for _, row in top.iterrows()
    ]

    return BenchmarkRecommendation(
        objective=objective,
        sample=str(best["sample"]),
        score=float(best["objective_score"]),
        rationale=rationale,
        top_candidates=top_candidates,
    )


def benchmark_report(
    circuit_table: pd.DataFrame,
    objectives: Optional[List[str]] = None,
    top_k: int = 3,
) -> str:
    """Generate a human-readable benchmark recommendation report."""
    if objectives is None:
        objectives = ["low_rs", "high_rp", "high_capacitance", "balanced"]

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  BENCHMARK AUTOMATICO ENTRE AMOSTRAS")
    lines.append("=" * 72)

    for obj in objectives:
        rec = recommend_best_configuration(circuit_table, objective=obj, top_k=top_k)
        lines.append("")
        lines.append(f"Objective: {rec.objective}")
        lines.append(f"  Melhor configuracao: {rec.sample}  (score={rec.score:.3f})")
        lines.append(f"  Racional: {rec.rationale}")
        if rec.top_candidates:
            lines.append("  Top candidatos:")
            for i, (name, sc) in enumerate(rec.top_candidates, start=1):
                lines.append(f"    {i}. {name}  score={sc:.3f}")

    lines.append("")
    lines.append(
        "Sugestao: validar a melhor configuracao com replicatas experimentais."
    )
    lines.append("=" * 72)
    return "\n".join(lines)
