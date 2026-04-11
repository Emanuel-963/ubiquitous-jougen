"""Tests for src.ranking — classification and ranking logic."""

import numpy as np
import pandas as pd

from src.ranking import apply_classification


class TestApplyClassification:
    def test_missing_columns_returns_indefinida(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = apply_classification(df)
        assert "Subclass" in result.columns
        assert (result["Subclass"].str.contains("Indefinida")).all()

    def test_with_valid_data_assigns_labels(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Rs_fit": rng.uniform(0.1, 10, size=20),
            "Rp_fit": rng.uniform(10, 1000, size=20),
        })
        result = apply_classification(df)
        assert "Subclass" in result.columns
        # Should have at most 2 distinct labels (+ possibly "Indefinida")
        assert result["Subclass"].nunique() <= 3

    def test_single_row_fallback(self):
        df = pd.DataFrame({"Rs_fit": [1.0], "Rp_fit": [100.0]})
        result = apply_classification(df)
        assert "Subclass" in result.columns

    def test_nan_values_handled(self):
        df = pd.DataFrame({
            "Rs_fit": [np.nan, 1.0, 2.0],
            "Rp_fit": [100.0, np.nan, 200.0],
        })
        result = apply_classification(df)
        assert "Subclass" in result.columns
