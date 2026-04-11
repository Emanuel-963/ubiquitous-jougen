"""Tests for src.cycling_calculator — energy/power calculations."""

import numpy as np
import pandas as pd
import pytest

from src.cycling_calculator import calculate_mass, identify_cycles


class TestCalculateMass:
    def test_positive_current(self):
        assert calculate_mass(0.5, 1.0) == pytest.approx(0.5)

    def test_negative_current(self):
        assert calculate_mass(-0.5, 1.0) == pytest.approx(0.5)

    def test_zero_current(self):
        assert calculate_mass(0.0, 1.0) == pytest.approx(0.0)


class TestIdentifyCycles:
    def test_uses_cycle_column(self):
        df = pd.DataFrame({
            "ciclo": [1, 1, 2, 2, 3, 3],
            "potencial": [0.1, 0.5, 0.1, 0.5, 0.1, 0.5],
        })
        result = identify_cycles(df)
        assert list(result) == [1, 1, 2, 2, 3, 3]

    def test_fallback_when_no_cycle_column(self):
        # Oscillating potential → should detect cycles via peaks
        t = np.linspace(0, 4 * np.pi, 200)
        df = pd.DataFrame({"potencial": np.sin(t)})
        result = identify_cycles(df)
        assert len(result) == len(df)
