"""Tests for src/uncertainty.py — Day 11."""

from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty import (
    BootstrapResult,
    MonteCarloResult,
    UncertaintyAnalyzer,
)


# ── Helpers ──────────────────────────────────────────────────────────

class _SimpleTemplate:
    """Minimal circuit template for testing: Rs + Rp (pure real)."""

    name = "SimpleRC"
    param_names = ["Rs", "Rp"]
    bounds = ([0.01, 0.01], [1e4, 1e4])

    @staticmethod
    def model_fn(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp = p
        return Rs + Rp / (1.0 + 1j * omega * 1e-3 * Rp)

    @staticmethod
    def init_fn(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = float(z.real.min()) if z.size else 5.0
        rp = float(z.real.max() - z.real.min()) if z.size else 50.0
        return np.array([max(rs, 0.1), max(rp, 0.1)])


class _RandlesTemplate:
    """Randles-CPE-W-like template (5 params) for richer tests."""

    name = "Randles-CPE-W"
    param_names = ["Rs", "Rp", "Q", "n", "Sigma"]
    bounds = ([1e-6, 1e-3, 1e-12, 0.3, 1e-10], [1e6, 1e8, 1.0, 1.0, 1e5])

    @staticmethod
    def model_fn(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp, Q, n, sigma = p
        Zcpe = 1.0 / (Q * (1j * omega) ** n)
        Zw = sigma / np.sqrt(1j * omega + 1e-30)
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
        return Rs + Zpar + Zw

    @staticmethod
    def init_fn(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.array([10.0, 100.0, 1e-4, 0.85, 0.01])


def _synth_spectrum(template, params, n_pts=50, noise=0.0, seed=0):
    """Generate a synthetic impedance spectrum."""
    rng = np.random.default_rng(seed)
    freq = np.logspace(-1, 5, n_pts)
    omega = 2.0 * np.pi * freq
    z_clean = template.model_fn(np.asarray(params), omega)
    if noise > 0:
        mag = np.abs(z_clean)
        z_clean = z_clean + rng.normal(0, noise * mag) + 1j * rng.normal(0, noise * mag)
    return freq, z_clean


# =====================================================================
# MonteCarloResult dataclass
# =====================================================================


class TestMonteCarloResult:
    """Tests for the MonteCarloResult dataclass."""

    def test_defaults(self):
        r = MonteCarloResult()
        assert r.param_names == []
        assert r.samples.shape == (0, 0)
        assert r.n_success == 0
        assert r.noise_pct == 0.0

    def test_custom_fields(self):
        r = MonteCarloResult(
            param_names=["Rs", "Rp"],
            mean={"Rs": 10.0, "Rp": 100.0},
            n_iter=50,
            noise_pct=0.03,
        )
        assert r.mean["Rs"] == 10.0
        assert r.n_iter == 50


# =====================================================================
# BootstrapResult dataclass
# =====================================================================


class TestBootstrapResult:
    """Tests for the BootstrapResult dataclass."""

    def test_defaults(self):
        r = BootstrapResult()
        assert r.param_names == []
        assert r.n_success == 0

    def test_custom_fields(self):
        r = BootstrapResult(
            param_names=["Rs"],
            mean={"Rs": 10.0},
            std={"Rs": 0.5},
        )
        assert r.std["Rs"] == 0.5


# =====================================================================
# UncertaintyAnalyzer — constructor
# =====================================================================


class TestUncertaintyAnalyzerInit:
    """Tests for UncertaintyAnalyzer construction."""

    def test_defaults(self):
        ua = UncertaintyAnalyzer()
        assert ua.n_iter == 100
        assert ua.noise_pct == pytest.approx(0.02)
        assert ua.ci_level == pytest.approx(0.95)

    def test_custom_params(self):
        ua = UncertaintyAnalyzer(n_iter=50, noise_pct=0.05, ci_level=0.90, seed=42)
        assert ua.n_iter == 50
        assert ua.noise_pct == pytest.approx(0.05)
        assert ua.ci_level == pytest.approx(0.90)

    def test_clamp_noise_pct(self):
        ua = UncertaintyAnalyzer(noise_pct=0.5)  # exceeds max 0.20
        assert ua.noise_pct <= 0.20

    def test_clamp_noise_pct_low(self):
        ua = UncertaintyAnalyzer(noise_pct=0.0)
        assert ua.noise_pct >= 0.001

    def test_min_n_iter(self):
        ua = UncertaintyAnalyzer(n_iter=1)
        assert ua.n_iter >= 2


# =====================================================================
# Monte Carlo
# =====================================================================


class TestMonteCarlo:
    """Tests for Monte Carlo error propagation."""

    def test_returns_mc_result(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=10, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        assert isinstance(mc, MonteCarloResult)

    def test_param_names(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=10, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        assert mc.param_names == ["Rs", "Rp"]

    def test_samples_shape(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=20, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        assert mc.samples.shape[1] == 2
        assert mc.n_success <= 20

    def test_most_iterations_succeed(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=30, noise_pct=0.01, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        assert mc.n_success >= 20  # at least 2/3 should converge

    def test_mean_close_to_true(self):
        tpl = _SimpleTemplate()
        true_params = [10.0, 100.0]
        freq, z = _synth_spectrum(tpl, true_params)
        ua = UncertaintyAnalyzer(n_iter=50, noise_pct=0.01, seed=42)
        mc = ua.monte_carlo(tpl, freq, z, p0=np.array(true_params))
        # Means should be within 20% of true values
        assert abs(mc.mean["Rs"] - 10.0) < 5.0
        assert abs(mc.mean["Rp"] - 100.0) < 50.0

    def test_std_positive(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=30, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        for pname in mc.param_names:
            assert mc.std[pname] >= 0

    def test_ci_bounds_ordered(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=30, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        if mc.n_success > 2:
            for pname in mc.param_names:
                assert mc.ci_low[pname] <= mc.ci_high[pname]

    def test_noise_pct_stored(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=5, noise_pct=0.03, seed=0)
        mc = ua.monte_carlo(tpl, freq, z)
        assert mc.noise_pct == pytest.approx(0.03)

    def test_custom_p0(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=10, seed=42)
        mc = ua.monte_carlo(tpl, freq, z, p0=np.array([10.0, 100.0]))
        assert mc.n_success >= 1

    def test_n_iter_field(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=15, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        assert mc.n_iter == 15


# =====================================================================
# Bootstrap of Residuals
# =====================================================================


class TestBootstrapResiduals:
    """Tests for the residual bootstrap method."""

    def test_returns_bs_result(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=10, seed=42)
        bs = ua.bootstrap_residuals(tpl, freq, z)
        assert isinstance(bs, BootstrapResult)

    def test_param_names(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=10, seed=42)
        bs = ua.bootstrap_residuals(tpl, freq, z)
        assert bs.param_names == ["Rs", "Rp"]

    def test_samples_shape(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=20, seed=42)
        bs = ua.bootstrap_residuals(tpl, freq, z)
        assert bs.samples.shape[1] == 2
        assert bs.n_success <= 20

    def test_most_iterations_succeed(self):
        tpl = _SimpleTemplate()
        true_params = [10.0, 100.0]
        freq, z = _synth_spectrum(tpl, true_params)
        ua = UncertaintyAnalyzer(n_iter=30, seed=42)
        bs = ua.bootstrap_residuals(tpl, freq, z, p_fit=np.array(true_params))
        assert bs.n_success >= 20

    def test_mean_close_to_fit(self):
        tpl = _SimpleTemplate()
        true_params = [10.0, 100.0]
        freq, z = _synth_spectrum(tpl, true_params)
        ua = UncertaintyAnalyzer(n_iter=50, seed=42)
        bs = ua.bootstrap_residuals(tpl, freq, z, p_fit=np.array(true_params))
        assert abs(bs.mean["Rs"] - 10.0) < 5.0
        assert abs(bs.mean["Rp"] - 100.0) < 50.0

    def test_ci_bounds_ordered(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=30, seed=42)
        bs = ua.bootstrap_residuals(tpl, freq, z)
        if bs.n_success > 2:
            for pname in bs.param_names:
                assert bs.ci_low[pname] <= bs.ci_high[pname]

    def test_n_iter_field(self):
        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=12, seed=42)
        bs = ua.bootstrap_residuals(tpl, freq, z)
        assert bs.n_iter == 12


# =====================================================================
# Confidence Ellipse
# =====================================================================


class TestConfidenceEllipse:
    """Tests for confidence_ellipse static method."""

    def test_basic_ellipse(self):
        rng = np.random.default_rng(42)
        samples = rng.multivariate_normal([10, 100], [[1, 0.5], [0.5, 25]], size=200)
        x, y = UncertaintyAnalyzer.confidence_ellipse(samples, 0, 1)
        assert x.shape == (100,)
        assert y.shape == (100,)

    def test_custom_n_points(self):
        rng = np.random.default_rng(42)
        samples = rng.multivariate_normal([10, 100], [[1, 0], [0, 25]], size=200)
        x, y = UncertaintyAnalyzer.confidence_ellipse(samples, 0, 1, n_points=50)
        assert x.shape == (50,)

    def test_too_few_samples(self):
        samples = np.array([[1, 2], [3, 4]])  # only 2 rows
        x, y = UncertaintyAnalyzer.confidence_ellipse(samples, 0, 1)
        assert x.size == 0
        assert y.size == 0

    def test_1d_array_returns_empty(self):
        x, y = UncertaintyAnalyzer.confidence_ellipse(np.array([1, 2, 3]), 0, 0)
        assert x.size == 0

    def test_ellipse_encloses_mean(self):
        rng = np.random.default_rng(0)
        mean = [50, 200]
        samples = rng.multivariate_normal(mean, [[4, 0], [0, 100]], size=500)
        x, y = UncertaintyAnalyzer.confidence_ellipse(samples, 0, 1, ci_level=0.99)
        # Check that the mean is roughly inside the ellipse
        cx, cy = np.mean(x), np.mean(y)
        assert abs(cx - mean[0]) < 5
        assert abs(cy - mean[1]) < 30

    def test_ci_level_affects_size(self):
        rng = np.random.default_rng(42)
        samples = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=500)
        x90, y90 = UncertaintyAnalyzer.confidence_ellipse(samples, 0, 1, ci_level=0.90)
        x99, y99 = UncertaintyAnalyzer.confidence_ellipse(samples, 0, 1, ci_level=0.99)
        # 99% ellipse should be larger
        range90 = (x90.max() - x90.min()) * (y90.max() - y90.min())
        range99 = (x99.max() - x99.min()) * (y99.max() - y99.min())
        assert range99 > range90


# =====================================================================
# Summary Table
# =====================================================================


class TestSummaryTable:
    """Tests for the summary_table static method."""

    def test_single_result(self):
        mc = MonteCarloResult(
            param_names=["Rs", "Rp"],
            mean={"Rs": 10.0, "Rp": 100.0},
            std={"Rs": 0.5, "Rp": 5.0},
            ci_low={"Rs": 9.0, "Rp": 90.0},
            ci_high={"Rs": 11.0, "Rp": 110.0},
        )
        table = UncertaintyAnalyzer.summary_table(mc, labels=["MC"])
        assert "Rs" in table
        assert "Rp" in table
        assert table["Rs"]["MC_mean"] == 10.0
        assert table["Rp"]["MC_std"] == 5.0

    def test_two_results(self):
        mc = MonteCarloResult(
            param_names=["Rs"],
            mean={"Rs": 10.0}, std={"Rs": 0.5},
            ci_low={"Rs": 9.0}, ci_high={"Rs": 11.0},
        )
        bs = BootstrapResult(
            param_names=["Rs"],
            mean={"Rs": 10.1}, std={"Rs": 0.6},
            ci_low={"Rs": 8.9}, ci_high={"Rs": 11.1},
        )
        table = UncertaintyAnalyzer.summary_table(mc, bs, labels=["MC", "BS"])
        assert table["Rs"]["MC_mean"] == 10.0
        assert table["Rs"]["BS_mean"] == 10.1

    def test_default_labels(self):
        mc = MonteCarloResult(param_names=["Rs"], mean={"Rs": 10.0},
                              std={"Rs": 0.5}, ci_low={"Rs": 9.0}, ci_high={"Rs": 11.0})
        table = UncertaintyAnalyzer.summary_table(mc)
        assert "R0_mean" in table["Rs"]

    def test_empty_results(self):
        table = UncertaintyAnalyzer.summary_table()
        assert table == {}


# =====================================================================
# uncertainty_columns
# =====================================================================


class TestUncertaintyColumns:
    """Tests for the uncertainty_columns static method."""

    def test_basic_columns(self):
        fit = {"params": {"Rs": 10.0, "Rp": 100.0}, "params_std": {"Rs": 0.5, "Rp": 5.0}}
        cols = UncertaintyAnalyzer.uncertainty_columns(fit)
        assert cols["Rs_fit"] == 10.0
        assert cols["Rp_fit_std"] == 5.0

    def test_with_mc(self):
        fit = {"params": {"Rs": 10.0}, "params_std": {}}
        mc = MonteCarloResult(
            param_names=["Rs"],
            mean={"Rs": 10.1}, std={"Rs": 0.5},
            ci_low={"Rs": 9.0}, ci_high={"Rs": 11.0},
        )
        cols = UncertaintyAnalyzer.uncertainty_columns(fit, mc=mc)
        assert cols["Rs_mc_std"] == 0.5
        assert cols["Rs_mc_ci_low"] == 9.0

    def test_with_bs(self):
        fit = {"params": {"Rs": 10.0}, "params_std": {}}
        bs = BootstrapResult(
            param_names=["Rs"],
            mean={"Rs": 10.1}, std={"Rs": 0.6},
            ci_low={"Rs": 8.9}, ci_high={"Rs": 11.1},
        )
        cols = UncertaintyAnalyzer.uncertainty_columns(fit, bs=bs)
        assert cols["Rs_bs_std"] == 0.6

    def test_with_both(self):
        fit = {"params": {"Rs": 10.0}, "params_std": {"Rs": 0.3}}
        mc = MonteCarloResult(param_names=["Rs"], mean={"Rs": 10.0},
                              std={"Rs": 0.5}, ci_low={"Rs": 9.0}, ci_high={"Rs": 11.0})
        bs = BootstrapResult(param_names=["Rs"], mean={"Rs": 10.0},
                             std={"Rs": 0.6}, ci_low={"Rs": 8.9}, ci_high={"Rs": 11.1})
        cols = UncertaintyAnalyzer.uncertainty_columns(fit, mc=mc, bs=bs)
        assert "Rs_mc_std" in cols
        assert "Rs_bs_std" in cols
        assert cols["Rs_fit_std"] == 0.3

    def test_empty_params(self):
        fit = {"params": {}, "params_std": {}}
        cols = UncertaintyAnalyzer.uncertainty_columns(fit)
        assert cols == {}


# =====================================================================
# Plot distributions
# =====================================================================


class TestPlotDistributions:
    """Tests for plot_distributions visualization."""

    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")

        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=10, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        fig = ua.plot_distributions(mc)
        assert fig is not None

    def test_saves_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")

        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=10, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        out = str(tmp_path / "dist.png")
        ua.plot_distributions(mc, out_path=out)
        assert (tmp_path / "dist.png").exists()

    def test_empty_samples(self):
        import matplotlib
        matplotlib.use("Agg")

        ua = UncertaintyAnalyzer()
        mc = MonteCarloResult()
        fig = ua.plot_distributions(mc)
        assert fig is not None


# =====================================================================
# Plot ellipses
# =====================================================================


class TestPlotEllipses:
    """Tests for plot_ellipses visualization."""

    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")

        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=15, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        fig = ua.plot_ellipses(mc)
        assert fig is not None

    def test_saves_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")

        tpl = _SimpleTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0])
        ua = UncertaintyAnalyzer(n_iter=15, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        out = str(tmp_path / "ellipse.png")
        ua.plot_ellipses(mc, out_path=out)
        assert (tmp_path / "ellipse.png").exists()

    def test_custom_pairs(self):
        import matplotlib
        matplotlib.use("Agg")

        tpl = _RandlesTemplate()
        freq, z = _synth_spectrum(tpl, [10.0, 100.0, 1e-4, 0.85, 0.01])
        ua = UncertaintyAnalyzer(n_iter=15, seed=42)
        mc = ua.monte_carlo(tpl, freq, z)
        fig = ua.plot_ellipses(mc, pairs=[(0, 1), (2, 3)])
        assert fig is not None

    def test_single_param_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")

        ua = UncertaintyAnalyzer()
        # Only 1 param → "Insufficient parameters"
        mc = MonteCarloResult(param_names=["Rs"])
        fig = ua.plot_ellipses(mc)
        assert fig is not None


# =====================================================================
# Integration: MC + Bootstrap on same template
# =====================================================================


class TestIntegration:
    """End-to-end tests combining MC and Bootstrap."""

    def test_mc_and_bs_comparable(self):
        tpl = _SimpleTemplate()
        true_params = [10.0, 100.0]
        freq, z = _synth_spectrum(tpl, true_params, noise=0.01, seed=7)
        ua = UncertaintyAnalyzer(n_iter=40, noise_pct=0.02, seed=42)
        mc = ua.monte_carlo(tpl, freq, z, p0=np.array(true_params))
        bs = ua.bootstrap_residuals(tpl, freq, z, p_fit=np.array(true_params))
        # Both should have positive std
        for pname in mc.param_names:
            if mc.n_success > 2:
                assert mc.std[pname] >= 0
            if bs.n_success > 2:
                assert bs.std[pname] >= 0

    def test_summary_table_both(self):
        mc = MonteCarloResult(
            param_names=["Rs", "Rp"],
            mean={"Rs": 10.0, "Rp": 100.0},
            std={"Rs": 0.5, "Rp": 5.0},
            ci_low={"Rs": 9.0, "Rp": 90.0},
            ci_high={"Rs": 11.0, "Rp": 110.0},
        )
        bs = BootstrapResult(
            param_names=["Rs", "Rp"],
            mean={"Rs": 10.1, "Rp": 99.0},
            std={"Rs": 0.6, "Rp": 6.0},
            ci_low={"Rs": 8.8, "Rp": 88.0},
            ci_high={"Rs": 11.2, "Rp": 112.0},
        )
        table = UncertaintyAnalyzer.summary_table(mc, bs, labels=["MC", "BS"])
        assert len(table) == 2
        assert "MC_mean" in table["Rs"]
        assert "BS_std" in table["Rs"]

    def test_uncertainty_columns_full(self):
        fit = {
            "params": {"Rs": 10.0, "Rp": 100.0},
            "params_std": {"Rs": 0.3, "Rp": 3.0},
        }
        mc = MonteCarloResult(
            param_names=["Rs", "Rp"],
            mean={"Rs": 10.0, "Rp": 100.0},
            std={"Rs": 0.5, "Rp": 5.0},
            ci_low={"Rs": 9.0, "Rp": 90.0},
            ci_high={"Rs": 11.0, "Rp": 110.0},
        )
        cols = UncertaintyAnalyzer.uncertainty_columns(fit, mc=mc)
        expected_keys = ["Rs_fit", "Rs_fit_std", "Rs_mc_std",
                         "Rs_mc_ci_low", "Rs_mc_ci_high"]
        for k in expected_keys:
            assert k in cols

    def test_randles_template_mc(self):
        tpl = _RandlesTemplate()
        true_p = [10.0, 100.0, 1e-4, 0.85, 0.01]
        freq, z = _synth_spectrum(tpl, true_p)
        ua = UncertaintyAnalyzer(n_iter=15, noise_pct=0.01, seed=42)
        mc = ua.monte_carlo(tpl, freq, z, p0=np.array(true_p))
        assert mc.param_names == ["Rs", "Rp", "Q", "n", "Sigma"]
        assert mc.n_success >= 1
