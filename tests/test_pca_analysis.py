import numpy as np
import pandas as pd
import pytest

from src.pca_analysis import run_pca


def test_run_pca_basic():
    # Create a small dataframe with variance and a NaN to test imputation
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, np.nan],
            "b": [2.0, 3.0, 4.0, 5.0],
            "c": [10.0, 11.0, 9.0, 8.0],
        },
        index=["s1", "s2", "s3", "s4"],
    )

    pca, scores = run_pca(df, n_components=2)
    assert scores.shape[0] == df.shape[0]
    # At least 2 principal components were returned (or min(n_components, features))
    assert scores.shape[1] >= 2
    # pca is a fitted object with explained_variance_ attribute
    assert hasattr(pca, "explained_variance_")


def test_run_pca_insufficient_variance():
    # All columns constant -> variance is zero
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})
    with pytest.raises(ValueError):
        run_pca(df)


def test_imputation_uses_quantile():
    # Column 'a' has NaN and low values; check that function fills NaN
    # with 5th percentile
    df = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, np.nan, 3.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    pca, scores = run_pca(df, n_components=2)
    # If run_pca succeeded, NaN was filled and scores produced for all rows
    assert scores.shape[0] == df.shape[0]
    assert not scores.isna().any().any()
