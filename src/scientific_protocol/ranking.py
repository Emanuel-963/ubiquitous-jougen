"""Workbook-compatible multicriteria ranking."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CRITERIA, Criterion, validate_criteria


def normalize_series(values: pd.Series, direction: str) -> pd.Series:
    """Min-max normalize numeric values, preserving unavailable data as NaN."""
    numeric = pd.to_numeric(values, errors="coerce")
    present = numeric.dropna()
    if present.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    low, high = present.min(), present.max()
    if high == low:
        normalized = numeric.where(numeric.isna(), 1.0).astype(float)
    elif direction == "max":
        normalized = (numeric - low) / (high - low)
    elif direction == "min":
        normalized = (high - numeric) / (high - low)
    else:
        raise ValueError("direction must be 'min' or 'max'")
    return normalized


def rank_multicriteria(
    metrics: pd.DataFrame,
    criteria: Iterable[Criterion] = DEFAULT_CRITERIA,
) -> pd.DataFrame:
    """Rank rows using only available metrics and renormalized weights."""
    criteria = validate_criteria(criteria)
    result = metrics.copy()
    weighted = []
    available_weight = pd.Series(0.0, index=result.index)
    for criterion in criteria:
        values = result.get(criterion.key, pd.Series(np.nan, index=result.index))
        normalized = normalize_series(values, criterion.direction)
        result[f"norm_{criterion.key}"] = normalized
        present = normalized.notna()
        available_weight = available_weight + present.astype(float) * criterion.weight
        weighted.append(normalized.fillna(0.0) * criterion.weight)

    numerator = sum(weighted, pd.Series(0.0, index=result.index))
    result["score_available"] = np.where(
        available_weight > 0,
        numerator / available_weight * 100.0,
        np.nan,
    )
    result["coverage_pct"] = available_weight
    result["protocol_status"] = np.where(
        result["coverage_pct"] >= 100.0 - 1e-9, "Completo", "Parcial"
    )
    result["rank"] = result["score_available"].rank(
        ascending=False, method="dense", na_option="bottom"
    )
    return result.sort_values(
        ["score_available", "coverage_pct"],
        ascending=[False, False],
        na_position="last",
    )


def workbook_criteria() -> Tuple[Criterion, ...]:
    """Return the exact nine criteria from the supplied workbook."""
    return DEFAULT_CRITERIA
