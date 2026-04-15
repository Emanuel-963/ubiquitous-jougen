"""Tests for src/circuit_composer.py — Day 8."""

from __future__ import annotations

import numpy as np
import pytest

from src.circuit_composer import (
    CircuitBlock,
    CircuitComposer,
    BUILTIN_BLOCKS,
    MAX_CANDIDATES,
    get_builtin_blocks,
    _block_R,
    _block_C,
    _block_CPE,
    _block_W,
    _block_W_finite,
    _block_L,
    _block_ZARC,
    _series_impedance,
    _parallel_impedance,
    _series_parallel_impedance,
    _param_slices,
)
from src.circuit_fitting import CircuitTemplate


# ── Helpers ──────────────────────────────────────────────────────────

OMEGA = 2.0 * np.pi * np.logspace(-1, 5, 50)
FREQ = OMEGA / (2.0 * np.pi)


def _synth_randles(freq: np.ndarray, Rs=10, Rp=100, C=1e-5) -> np.ndarray:
    """Simple Randles impedance for testing (R + R||C)."""
    omega = 2.0 * np.pi * freq
    Zc = 1.0 / (1j * omega * C)
    Zpar = 1.0 / (1.0 / Rp + 1.0 / Zc)
    return Rs + Zpar


# =====================================================================
# CircuitBlock tests
# =====================================================================


class TestCircuitBlockBasics:
    """Basic properties of CircuitBlock."""

    def test_block_R_returns_real(self):
        blk = _block_R()
        p = np.array([50.0])
        z = blk.impedance(p, OMEGA)
        np.testing.assert_allclose(z.real, 50.0)
        np.testing.assert_allclose(z.imag, 0.0, atol=1e-15)

    def test_block_C_pure_imaginary(self):
        blk = _block_C()
        p = np.array([1e-6])
        z = blk.impedance(p, OMEGA)
        assert np.all(z.real < 1e-10)
        assert np.all(z.imag < 0)  # capacitive → negative imag

    def test_block_CPE_shape(self):
        blk = _block_CPE()
        p = np.array([1e-4, 0.9])
        z = blk.impedance(p, OMEGA)
        assert z.shape == OMEGA.shape
        assert np.all(np.isfinite(z))

    def test_block_CPE_n1_approximates_capacitor(self):
        """CPE with n=1 should behave like 1/(Q·jω)."""
        blk = _block_CPE()
        Q = 1e-5
        p = np.array([Q, 1.0])
        z_cpe = blk.impedance(p, OMEGA)
        z_cap = 1.0 / (1j * OMEGA * Q)
        np.testing.assert_allclose(z_cpe, z_cap, rtol=1e-6)

    def test_block_W_shape(self):
        blk = _block_W()
        p = np.array([0.01])
        z = blk.impedance(p, OMEGA)
        assert z.shape == OMEGA.shape

    def test_block_W_finite_shape(self):
        blk = _block_W_finite()
        p = np.array([100.0, 1.0])
        z = blk.impedance(p, OMEGA)
        assert z.shape == OMEGA.shape
        assert np.all(np.isfinite(z))

    def test_block_L_positive_imaginary(self):
        blk = _block_L()
        p = np.array([1e-3])
        z = blk.impedance(p, OMEGA)
        assert np.all(z.imag > 0)  # inductive
        np.testing.assert_allclose(z.real, 0.0, atol=1e-15)

    def test_block_ZARC_shape(self):
        blk = _block_ZARC()
        p = np.array([100.0, 1e-4, 0.85])
        z = blk.impedance(p, OMEGA)
        assert z.shape == OMEGA.shape
        assert np.all(np.isfinite(z))

    def test_block_ZARC_low_freq_limit(self):
        """At very low ω the ZARC → R (CPE impedance → ∞)."""
        blk = _block_ZARC()
        R = 200.0
        p = np.array([R, 1e-4, 0.85])
        omega_low = np.array([0.001])
        z = blk.impedance(p, omega_low)
        # Should be close to R at DC
        assert abs(z[0].real - R) / R < 0.15

    def test_default_p0_within_bounds(self):
        for maker in BUILTIN_BLOCKS.values():
            blk = maker()
            p0 = blk.default_p0()
            assert len(p0) == blk.n_params
            for val, (lo, hi) in zip(p0, blk.bounds):
                assert lo <= val <= hi, f"{blk.name}: {val} not in [{lo}, {hi}]"


class TestBuiltinBlocks:
    """Inventory of built-in blocks."""

    def test_builtin_count(self):
        assert len(BUILTIN_BLOCKS) == 7

    def test_builtin_names(self):
        expected = {"R", "C", "CPE", "W", "W_finite", "L", "ZARC"}
        assert set(BUILTIN_BLOCKS.keys()) == expected

    def test_get_builtin_blocks_instantiates_all(self):
        blocks = get_builtin_blocks()
        assert len(blocks) == 7
        assert all(isinstance(b, CircuitBlock) for b in blocks)


# =====================================================================
# Topology helpers
# =====================================================================


class TestTopologyHelpers:
    """Direct tests of series / parallel / series-parallel composition."""

    def test_param_slices_correct(self):
        blks = [_block_R(), _block_CPE(), _block_W()]
        slices = _param_slices(blks)
        assert slices[0] == slice(0, 1)
        assert slices[1] == slice(1, 3)
        assert slices[2] == slice(3, 4)

    def test_series_is_sum(self):
        """R1 + R2 in series = R1 + R2."""
        r1, r2 = _block_R(), _block_R()
        fn = _series_impedance([r1, r2])
        p = np.array([30.0, 70.0])
        z = fn(p, OMEGA)
        np.testing.assert_allclose(z.real, 100.0)
        np.testing.assert_allclose(z.imag, 0.0, atol=1e-15)

    def test_parallel_two_resistors(self):
        """R1 || R2 = R1·R2 / (R1 + R2)."""
        r1, r2 = _block_R(), _block_R()
        fn = _parallel_impedance([r1, r2])
        R1, R2 = 60.0, 40.0
        p = np.array([R1, R2])
        z = fn(p, OMEGA)
        expected = (R1 * R2) / (R1 + R2)
        np.testing.assert_allclose(z.real, expected, rtol=1e-10)

    def test_series_parallel_R_plus_RC(self):
        """Rs + (Rp || C) — basic Randles-like."""
        r_ser = _block_R()
        r_par = _block_R()
        c_par = _block_C()
        fn = _series_parallel_impedance(r_ser, [r_par, c_par])
        Rs, Rp, C = 10.0, 100.0, 1e-5
        p = np.array([Rs, Rp, C])
        z = fn(p, OMEGA)
        # Compare against direct calculation
        Zc = 1.0 / (1j * OMEGA * C)
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zc)
        z_expected = Rs + Zpar
        np.testing.assert_allclose(z, z_expected, rtol=1e-10)


# =====================================================================
# CircuitComposer.compose
# =====================================================================


class TestCompose:
    """Tests for CircuitComposer.compose()."""

    def setup_method(self):
        self.composer = CircuitComposer()

    def test_compose_series_returns_template(self):
        tpl = self.composer.compose(["R", "ZARC", "W"], topology="series")
        assert isinstance(tpl, CircuitTemplate)
        assert "series" in tpl.name

    def test_compose_parallel_returns_template(self):
        tpl = self.composer.compose(["R", "CPE"], topology="parallel")
        assert isinstance(tpl, CircuitTemplate)
        assert "parallel" in tpl.name

    def test_compose_series_parallel_returns_template(self):
        tpl = self.composer.compose(["R", "ZARC"], topology="series-parallel")
        assert isinstance(tpl, CircuitTemplate)
        assert "series-parallel" in tpl.name

    def test_compose_series_diagram(self):
        tpl = self.composer.compose(["R", "CPE", "W"], topology="series")
        assert "R" in tpl.diagram and "CPE" in tpl.diagram and "W" in tpl.diagram

    def test_compose_parallel_diagram(self):
        tpl = self.composer.compose(["R", "CPE"], topology="parallel")
        assert "‖" in tpl.diagram

    def test_compose_series_parallel_diagram(self):
        tpl = self.composer.compose(["R", "ZARC", "W"], topology="series-parallel")
        assert "(" in tpl.diagram

    def test_compose_param_names_unique(self):
        tpl = self.composer.compose(["R", "R"], topology="series")
        # Two R blocks → R_R and R_R_1 (or similar unique names)
        assert len(set(tpl.param_names)) == len(tpl.param_names)

    def test_compose_bounds_length(self):
        tpl = self.composer.compose(["R", "CPE", "L"], topology="series")
        # R(1) + CPE(2) + L(1) = 4 params
        assert len(tpl.bounds[0]) == 4
        assert len(tpl.bounds[1]) == 4

    def test_compose_model_callable(self):
        tpl = self.composer.compose(["R", "CPE"], topology="series")
        p0 = tpl.init_fn(OMEGA, np.ones_like(OMEGA, dtype=complex))
        z = tpl.model_fn(p0, OMEGA)
        assert z.shape == OMEGA.shape
        assert np.all(np.isfinite(z))

    def test_compose_unknown_block_raises(self):
        with pytest.raises(ValueError, match="Unknown block"):
            self.composer.compose(["R", "NONEXISTENT"], topology="series")

    def test_compose_unknown_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            self.composer.compose(["R", "CPE"], topology="banana")

    def test_compose_series_parallel_needs_two(self):
        with pytest.raises(ValueError, match="at least 2 blocks"):
            self.composer.compose(["R"], topology="series-parallel")

    def test_compose_init_fn_returns_correct_length(self):
        tpl = self.composer.compose(["R", "ZARC", "W"], topology="series")
        # R(1) + ZARC(3) + W(1) = 5
        p0 = tpl.init_fn(OMEGA, np.ones_like(OMEGA, dtype=complex))
        assert len(p0) == 5

    def test_compose_init_fn_within_bounds(self):
        tpl = self.composer.compose(["R", "CPE", "L"], topology="series")
        p0 = tpl.init_fn(OMEGA, np.ones_like(OMEGA, dtype=complex))
        for val, lo, hi in zip(p0, tpl.bounds[0], tpl.bounds[1]):
            assert lo <= val <= hi


# =====================================================================
# CircuitComposer.enumerate_candidates
# =====================================================================


class TestEnumerateCandidates:
    """Tests for candidate enumeration."""

    def setup_method(self):
        self.composer = CircuitComposer()

    def test_enumerate_returns_list(self):
        candidates = self.composer.enumerate_candidates(max_elements=2)
        assert isinstance(candidates, list)
        assert all(isinstance(c, CircuitTemplate) for c in candidates)

    def test_enumerate_max_elements_1(self):
        candidates = self.composer.enumerate_candidates(max_elements=1)
        assert len(candidates) >= 1
        # All should be single-block series
        for c in candidates:
            assert "series" in c.name

    def test_enumerate_no_duplicates(self):
        candidates = self.composer.enumerate_candidates(max_elements=3)
        names = [c.name for c in candidates]
        assert len(names) == len(set(names))

    def test_enumerate_respects_max_candidates(self):
        candidates = self.composer.enumerate_candidates(max_elements=4)
        assert len(candidates) <= MAX_CANDIDATES

    def test_enumerate_must_include_filter(self):
        candidates = self.composer.enumerate_candidates(
            max_elements=2, must_include=["R"]
        )
        for c in candidates:
            # Every candidate name should contain "R"
            assert "R" in c.name

    def test_enumerate_custom_topologies(self):
        candidates = self.composer.enumerate_candidates(
            max_elements=2, topologies=["series"]
        )
        for c in candidates:
            assert "series" in c.name

    def test_enumerate_grows_with_max_elements(self):
        c2 = self.composer.enumerate_candidates(max_elements=2)
        c3 = self.composer.enumerate_candidates(max_elements=3)
        assert len(c3) >= len(c2)


# =====================================================================
# CircuitComposer.auto_select
# =====================================================================


class TestAutoSelect:
    """Tests for auto_select (fast BIC-based screening)."""

    @pytest.fixture()
    def synth_data(self):
        """Simple Randles-like synthetic data."""
        z = _synth_randles(FREQ, Rs=10, Rp=100, C=1e-5)
        # Add small noise
        rng = np.random.default_rng(42)
        z = z + rng.normal(0, 0.1, z.shape) + 1j * rng.normal(0, 0.1, z.shape)
        return FREQ, z

    def test_auto_select_returns_list(self, synth_data):
        freq, z = synth_data
        composer = CircuitComposer()
        results = composer.auto_select(freq, z, max_elements=2, top_n=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_auto_select_result_keys(self, synth_data):
        freq, z = synth_data
        composer = CircuitComposer()
        results = composer.auto_select(freq, z, max_elements=2, top_n=3)
        for r in results:
            assert "template" in r
            assert "bic" in r
            assert "rss" in r
            assert "params" in r
            assert "n_params" in r

    def test_auto_select_sorted_by_bic(self, synth_data):
        freq, z = synth_data
        composer = CircuitComposer()
        results = composer.auto_select(freq, z, max_elements=2, top_n=5)
        bics = [r["bic"] for r in results]
        assert bics == sorted(bics), "Results should be sorted by BIC"

    def test_auto_select_best_has_finite_bic(self, synth_data):
        freq, z = synth_data
        composer = CircuitComposer()
        results = composer.auto_select(freq, z, max_elements=2, top_n=3)
        assert len(results) > 0
        assert np.isfinite(results[0]["bic"])

    def test_auto_select_top_n_limits_output(self, synth_data):
        freq, z = synth_data
        composer = CircuitComposer()
        results = composer.auto_select(freq, z, max_elements=2, top_n=2)
        assert len(results) <= 2

    def test_auto_select_must_include(self, synth_data):
        freq, z = synth_data
        composer = CircuitComposer()
        results = composer.auto_select(
            freq, z, max_elements=2, top_n=5, must_include=["R"]
        )
        for r in results:
            assert "R" in r["template"]

    def test_auto_select_empty_composer(self, synth_data):
        """Composer with no blocks → empty result."""
        freq, z = synth_data
        composer = CircuitComposer(blocks=[])
        results = composer.auto_select(freq, z, max_elements=2)
        assert results == []

    def test_auto_select_max_nfev_respected(self, synth_data):
        """With very low nfev the fit may fail but shouldn't crash."""
        freq, z = synth_data
        composer = CircuitComposer()
        # Very restrictive nfev — some fits will fail, but code must not crash
        results = composer.auto_select(
            freq, z, max_elements=2, top_n=3, max_nfev=5
        )
        assert isinstance(results, list)


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    """Edge cases and robustness checks."""

    def test_custom_block_accepted(self):
        """CircuitComposer works with user-defined blocks."""
        custom = CircuitBlock(
            name="MyR",
            impedance=lambda p, omega: np.full_like(omega, p[0], dtype=complex),
            n_params=1,
            param_names=["Rx"],
            bounds=[(0.1, 1e6)],
        )
        composer = CircuitComposer(blocks=[custom])
        assert composer.block_names == ["MyR"]
        tpl = composer.compose(["MyR"], topology="series")
        assert "MyR" in tpl.name

    def test_single_block_series(self):
        composer = CircuitComposer()
        tpl = composer.compose(["R"], topology="series")
        p0 = tpl.init_fn(OMEGA, np.ones_like(OMEGA, dtype=complex))
        z = tpl.model_fn(p0, OMEGA)
        assert z.shape == OMEGA.shape

    def test_repeated_blocks_series(self):
        """Three R blocks in series → 3 params, all unique names."""
        composer = CircuitComposer()
        tpl = composer.compose(["R", "R", "R"], topology="series")
        assert tpl.n_params == 3 if hasattr(tpl, "n_params") else len(tpl.param_names) == 3
        assert len(set(tpl.param_names)) == 3

    def test_zarc_block_params(self):
        blk = _block_ZARC()
        assert blk.n_params == 3
        assert blk.param_names == ["R", "Q", "n"]

    def test_w_finite_block_params(self):
        blk = _block_W_finite()
        assert blk.n_params == 2
        assert blk.param_names == ["Rd", "Td"]

    def test_compose_complex_topology(self):
        """R + (ZARC || W) — mixed topology."""
        composer = CircuitComposer()
        tpl = composer.compose(["R", "ZARC", "W"], topology="series-parallel")
        # R(1) + ZARC(3) + W(1) = 5 params
        assert len(tpl.param_names) == 5
        p0 = tpl.init_fn(OMEGA, np.ones_like(OMEGA, dtype=complex))
        z = tpl.model_fn(p0, OMEGA)
        assert np.all(np.isfinite(z))

    def test_max_candidates_constant(self):
        assert MAX_CANDIDATES == 50
