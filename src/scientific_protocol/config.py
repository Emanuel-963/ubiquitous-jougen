"""Configuration for the independent multicriteria scientific protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


@dataclass(frozen=True)
class Criterion:
    """One ranking criterion, following the supplied workbook."""

    key: str
    label: str
    unit: str
    direction: str
    weight: float


DEFAULT_CRITERIA = (
    Criterion("specific_capacitance", "Capacitância específica", "F/g", "max", 20.0),
    Criterion("capacitance_retention", "Retenção de capacitância", "%", "max", 15.0),
    Criterion("coulombic_efficiency", "Eficiência coulômbica", "%", "max", 10.0),
    Criterion("energy_density", "Densidade de energia", "Wh/kg", "max", 15.0),
    Criterion("power_density", "Densidade de potência", "W/kg", "max", 10.0),
    Criterion("capacitive_contribution", "Contribuição capacitiva", "%", "max", 5.0),
    Criterion("rs", "Resistência série (Rs)", "Ohm", "min", 10.0),
    Criterion("rct_rp", "Resistência de transferência/polarização", "Ohm", "min", 10.0),
    Criterion("area_drt_lenta", "Área DRT de processos lentos", "Ohm", "min", 5.0),
)


def criteria_from_config(config: object = "default") -> tuple[Criterion, ...]:
    """Return criteria from defaults, a mapping, or a JSON-like list."""
    if config in (None, "default"):
        return DEFAULT_CRITERIA
    if isinstance(config, Mapping):
        overrides = config.get("criteria", config)
        if not isinstance(overrides, Mapping):
            raise TypeError("criteria config must be a mapping")
        result = []
        for criterion in DEFAULT_CRITERIA:
            value = overrides.get(criterion.key, {})
            if isinstance(value, Mapping):
                result.append(
                    Criterion(
                        criterion.key,
                        str(value.get("label", criterion.label)),
                        str(value.get("unit", criterion.unit)),
                        str(value.get("direction", criterion.direction)).lower(),
                        float(value.get("weight", criterion.weight)),
                    )
                )
            else:
                result.append(criterion)
        return tuple(result)
    if isinstance(config, Iterable) and not isinstance(config, (str, bytes)):
        return tuple(
            item if isinstance(item, Criterion) else Criterion(**item)
            for item in config
        )
    raise TypeError("config must be 'default', a mapping, or criteria iterable")


def validate_criteria(criteria: Iterable[Criterion]) -> tuple[Criterion, ...]:
    """Validate directions, positive weights, and the 100-point total."""
    result = tuple(criteria)
    if not result:
        raise ValueError("at least one ranking criterion is required")
    if any(c.direction not in {"min", "max"} for c in result):
        raise ValueError("criterion direction must be 'min' or 'max'")
    if any(c.weight < 0 for c in result):
        raise ValueError("criterion weights cannot be negative")
    total = sum(c.weight for c in result)
    if abs(total - 100.0) > 1e-9:
        raise ValueError(f"criterion weights must sum to 100, got {total}")
    return result


def criteria_as_dict(criteria: Iterable[Criterion]) -> Dict[str, dict]:
    """Serialize criteria for JSON reports."""
    return {
        c.key: {
            "label": c.label,
            "unit": c.unit,
            "direction": c.direction,
            "weight": c.weight,
        }
        for c in criteria
    }
