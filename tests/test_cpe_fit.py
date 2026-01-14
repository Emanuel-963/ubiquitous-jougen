import numpy as np
import pandas as pd
import pytest

from src.cpe_fit import fit_cpe_warburg


def make_sample_df(n=10):
    f = np.logspace(0, 3, n)
    zreal = 10 - 0.01 * f
    zimag = -0.1 / f
    return pd.DataFrame(
        {"frequency": f, "zreal": zreal, "zimag": zimag, "omega": 2 * np.pi * f}
    )


def test_fit_insufficient_points():
    df = make_sample_df(n=2)
    with pytest.raises(ValueError):
        fit_cpe_warburg(df)


def test_fit_returns_keys():
    df = make_sample_df(n=30)
    res = fit_cpe_warburg(df)
    assert set(["Rs_fit", "Rp_fit", "Q", "n", "Sigma"]).issubset(res.keys())
    assert not np.isnan(res["Rs_fit"])
