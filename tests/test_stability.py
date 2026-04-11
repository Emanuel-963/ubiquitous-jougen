import pandas as pd

from src.stability import extract_sample_id, stability_metrics


def test_extract_sample_id():
    assert extract_sample_id("1 Nb2 GCT H2SO4.txt") == "1Nb2"
    assert extract_sample_id("no_match_name.txt") == "no_match_name"


def test_stability_cv_safe():
    df = pd.DataFrame({"Sample": ["A", "A", "B"], "Rs_fit": [1.0, 1.0, 0.0]})
    out = stability_metrics(df, "Rs_fit")
    assert "CV" in out.columns
    assert not (out.loc["B", "CV"] == float("inf"))
