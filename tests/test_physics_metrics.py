import math

import numpy as np
import pandas as pd

from src.physics_metrics import (
    dispersion_index,
    dominant_tau,
    effective_capacitance,
    extract_features,
    polarization_resistance,
    series_resistance,
    stored_energy,
)


def make_df():
    f = np.array([1000, 100, 10, 1])
    zreal = np.array([1.0, 2.0, 5.0, 10.0])
    zimag = -np.array([0.1, 0.2, 0.5, 1.0])
    return pd.DataFrame(
        {"frequency": f, "omega": 2 * np.pi * f, "zreal": zreal, "zimag": zimag}
    )


def test_effective_cap_basic():
    df = make_df()
    c = effective_capacitance(df)
    assert isinstance(c, np.ndarray)
    assert c.size > 0
    assert (c > 1e-15).all() and (c < 1e-2).all()


def test_effective_cap_empty_or_zero():
    df = pd.DataFrame({"zreal": [1.0], "zimag": [0.0], "omega": [0.0]})
    c = effective_capacitance(df)
    assert c.size == 0

    df2 = pd.DataFrame({"zreal": [1e-9], "zimag": [1e-9], "omega": [1000.0]})
    c2 = effective_capacitance(df2)
    assert c2.size == 0


def test_series_resistance_and_polarization():
    df = make_df()
    rs = series_resistance(df, n=2)
    assert rs == np.median(df.head(2)["zreal"].values)

    # Check polarization returns non-negative
    rp = polarization_resistance(df, n=2)
    assert rp >= 0


def test_stored_energy_and_tau_and_dispersion():
    c_eff = np.array([1e-6, 2e-6, 3e-6])
    energy = stored_energy(c_eff, voltage=2.0)
    assert math.isclose(energy[0], 0.5 * 1e-6 * 4.0)

    rp = 0.5
    tau = dominant_tau(rp, c_eff)
    assert math.isclose(tau, rp * np.median(c_eff))

    disp = dispersion_index(c_eff)
    assert not math.isnan(disp)

    assert math.isnan(dispersion_index(np.array([1e-6])))


def test_extract_features_integration():
    df = pd.DataFrame(
        {
            "zreal": [10, 9, 8, 7, 6, 5],
            "zimag": [0.1, 0.12, 0.11, 0.09, 0.08, 0.07],
            "omega": [1000, 800, 600, 400, 200, 100],
        }
    )
    feats = extract_features(df)
    assert "Rs" in feats and "Rp" in feats and "C_mean" in feats
    assert not math.isnan(feats["Rs"])
    assert feats["Rp"] >= 0
