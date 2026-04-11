"""Tests for src.circuit_fitting — element primitives and catalog."""

import numpy as np

from src.circuit_fitting import _cpe, _warburg, _inductor


class TestElementPrimitives:
    def test_cpe_returns_complex(self):
        omega = np.array([1.0, 10.0, 100.0])
        result = _cpe(omega, Q=1e-6, n=0.9)
        assert np.iscomplexobj(result)
        assert result.shape == omega.shape

    def test_warburg_returns_complex(self):
        omega = np.array([1.0, 10.0, 100.0])
        result = _warburg(omega, sigma=50.0)
        assert np.iscomplexobj(result)
        assert result.shape == omega.shape

    def test_inductor_returns_complex(self):
        omega = np.array([1.0, 10.0, 100.0])
        result = _inductor(omega, L=1e-6)
        assert np.iscomplexobj(result)
        # Pure inductance → real part ≈ 0
        assert np.allclose(result.real, 0, atol=1e-12)

    def test_cpe_q_zero_raises_or_inf(self):
        omega = np.array([10.0])
        # Q=0 → division by zero → inf
        result = _cpe(omega, Q=0.0, n=0.9)
        assert np.any(np.isinf(result))
