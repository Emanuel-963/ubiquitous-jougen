import pandas as pd
import pytest

from src.preprocessing import preprocess


def make_df():
    return pd.DataFrame(
        {"frequency": [1, 10, 100], "zreal": [10, 5, 1], "zimag": [-1, -0.5, -0.1]}
    )


def test_preprocess_basic():
    df = make_df()
    out = preprocess(df)
    assert "omega" in out.columns
    assert out.iloc[0]["frequency"] == 100  # sorted desc


def test_preprocess_missing_col():
    df = pd.DataFrame({"freq": [1]})
    with pytest.raises(ValueError):
        preprocess(df)
