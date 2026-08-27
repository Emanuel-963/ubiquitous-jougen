import numpy as np
import pandas as pd
import pytest

from src.scientific_protocol import (
    DEFAULT_CRITERIA,
    normalize_series,
    rank_multicriteria,
)
from src.scientific_protocol.config import Criterion, validate_criteria
from src.scientific_protocol.eis_drt import slow_drt_area
from src.scientific_protocol.report import _json_default, write_outputs


def test_default_weights_sum_to_100():
    assert sum(item.weight for item in DEFAULT_CRITERIA) == pytest.approx(100.0)


def test_normalization_respects_direction():
    values = pd.Series([1.0, 2.0, 3.0])
    assert normalize_series(values, "max").tolist() == [0.0, 0.5, 1.0]
    assert normalize_series(values, "min").tolist() == [1.0, 0.5, 0.0]


def test_missing_values_are_nan_and_score_is_renormalized():
    frame = pd.DataFrame({"electrolyte": ["A", "B"], "rs": [2.0, 1.0]})
    ranked = rank_multicriteria(frame)
    assert np.isnan(
        ranked.loc[ranked["electrolyte"] == "A", "norm_area_drt_lenta"].iloc[0]
    )
    assert ranked["coverage_pct"].tolist() == [10.0, 10.0]
    assert ranked["protocol_status"].eq("Parcial").all()


def test_invalid_weight_total_is_rejected():
    with pytest.raises(ValueError):
        validate_criteria([Criterion("x", "X", "", "max", 99.0)])


def test_slow_drt_area_uses_reliable_peaks_only():
    peaks = [
        {"tau_peak": 0.2, "area": 0.4, "well_separated": True},
        {"tau_peak": 0.3, "area": 0.8, "well_separated": False},
        {"tau_peak": 0.01, "area": 9.0, "well_separated": True},
    ]
    result = slow_drt_area(peaks)
    assert result["area_sum"] == pytest.approx(0.4)
    assert result["n_slow"] == 2
    assert result["n_reliable"] == 1


def test_numpy_boolean_is_json_serializable():
    assert _json_default(np.bool_(True)) is True


def test_ranking_export_uses_workbook_columns(tmp_path):
    ranking = pd.DataFrame(
        {
            "electrolyte": ["KCl 1M [cell_001]"],
            "specific_capacitance": [3.5],
            "capacitance_retention": [np.nan],
            "coulombic_efficiency": [86.2],
            "energy_density": [0.04],
            "power_density": [720.0],
            "capacitive_contribution": [45.4],
            "rs": [6.1],
            "rct_rp": [12528.0],
            "area_drt_lenta": [8648.0],
            "score_available": [100.0],
            "coverage_pct": [85.0],
            "protocol_status": ["Parcial"],
            "rank": [1.0],
        }
    )
    outputs = write_outputs(
        tmp_path,
        ranking=ranking,
        metrics={},
        figures=[],
        tables={},
        summary="ok",
        criteria=DEFAULT_CRITERIA,
    )
    csv_text = open(outputs["ranking_csv"], encoding="utf-8-sig").readline()
    assert csv_text.startswith("Eletrólito;")
    assert "Capacitância" in csv_text
    assert "norm_" not in csv_text
    assert (tmp_path / "ranking" / "criterios_classificacao.csv").is_file()
