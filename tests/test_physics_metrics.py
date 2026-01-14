import numpy as np
import pandas as pd

from src.physics_metrics import (
    effective_capacitance,
    polarization_resistance,
    series_resistance,
)


def make_df():
    f = np.array([1000, 100, 10, 1])
    zreal = np.array([1.0, 2.0, 5.0, 10.0])
    zimag = -np.array([0.1, 0.2, 0.5, 1.0])
    return pd.DataFrame(
        {"frequency": f, "omega": 2 * np.pi * f, "zreal": zreal, "zimag": zimag}
    )


def test_effective_cap():
    df = make_df()
    c = effective_capacitance(df)
    assert isinstance(c, np.ndarray)


def test_series_and_polarization():
    df = make_df()
    rs = series_resistance(df, n=2)
    rp = polarization_resistance(df, n=2)
    assert rs >= 0
    assert rp >= 0
