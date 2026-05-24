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
from typing import Dict, List, Optional, Tuple

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


DEFAULT_OBJECTIVE_PROFILES: Dict[str, Dict[str, float]] = {
    "low_rs": {
        "rs": -0.55,
        "chi2_over_nu": -0.20,
        "kk_valid": 0.15,
        "confidence": 0.10,
    },
    "high_rp": {
        "rp": 0.50,
        "chi2_over_nu": -0.20,
        "kk_valid": 0.20,
        "confidence": 0.10,
    },
    "high_capacitance": {
        "c_mean": 0.45,
        "rp": 0.20,
        "chi2_over_nu": -0.20,
        "kk_valid": 0.15,
    },
    "balanced": {
        "rs": -0.23,
        "rp": 0.23,
        "c_mean": 0.19,
        "chi2_over_nu": -0.15,
        "kk_valid": 0.10,
        "confidence": 0.10,
    },
}

_OBJECTIVE_ALIASES = {
    "min_rs": "low_rs",
    "conductivity": "low_rs",
    "stability": "high_rp",
    "corrosion_resistance": "high_rp",
    "capacitance": "high_capacitance",
    "energy_storage": "high_capacitance",
    "overall": "balanced",
    "health": "balanced",
}


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
    objective_profiles: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.Series:
    """Score rows for a named objective; higher score is better."""
    objective = objective.strip().lower()

    profiles = objective_profiles or DEFAULT_OBJECTIVE_PROFILES
    objective = _OBJECTIVE_ALIASES.get(objective, objective)

    if objective not in profiles:
        valid = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown objective '{objective}'. Available: {valid}")

    profile = profiles[objective]

    # Cache normalized metrics only once.
    metric_norm: Dict[str, pd.Series] = {}
    for metric in profile.keys():
        if metric == "kk_valid":
            s = (
                bench[metric].copy()
                if metric in bench.columns
                else pd.Series(0.5, index=bench.index)
            )
            s = s.where(np.isfinite(s), 0.5).clip(0.0, 1.0)
            metric_norm[metric] = s
            continue

        raw = (
            bench[metric].copy()
            if metric in bench.columns
            else pd.Series(np.nan, index=bench.index)
        )
        raw = pd.to_numeric(raw, errors="coerce")
        metric_norm[metric] = raw

    score = pd.Series(0.0, index=bench.index, dtype=float)
    total = 0.0
    for metric, weight in profile.items():
        if abs(weight) <= 1e-15:
            continue
        s = metric_norm[metric]
        if metric != "kk_valid":
            norm = _normalize(s, higher_is_better=(weight > 0.0))
        else:
            norm = s if weight > 0.0 else (1.0 - s)
        w = abs(weight)
        score += w * norm
        total += w

    if total <= 1e-15:
        return pd.Series(0.0, index=bench.index, dtype=float)
    return (score / total).fillna(0.0)


def recommend_best_configuration(
    circuit_table: pd.DataFrame,
    objective: str = "balanced",
    top_k: int = 3,
    objective_profiles: Optional[Dict[str, Dict[str, float]]] = None,
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

    score = score_objective(bench, objective, objective_profiles=objective_profiles)
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
    objective_profiles: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """Generate a human-readable benchmark recommendation report."""
    if objectives is None:
        objectives = ["low_rs", "high_rp", "high_capacitance", "balanced"]

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  BENCHMARK AUTOMATICO ENTRE AMOSTRAS")
    lines.append("=" * 72)

    for obj in objectives:
        rec = recommend_best_configuration(
            circuit_table,
            objective=obj,
            top_k=top_k,
            objective_profiles=objective_profiles,
        )
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
