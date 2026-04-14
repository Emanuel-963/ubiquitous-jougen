"""Tests for src.circuit_registry — CircuitRegistry and all built-in circuits."""

import numpy as np
import pytest

from src.circuit_fitting import CircuitTemplate as BaseTemplate
from src.circuit_registry import CircuitRegistry, CircuitTemplate


# ── Fixture: reset the registry between tests ────────────────────────
@pytest.fixture(autouse=True)
def _reset_registry():
    """Save and restore registry state so tests are isolated."""
    saved = dict(CircuitRegistry._circuits)
    yield
    CircuitRegistry._circuits = saved


# ══════════════════════════════════════════════════════════════════════
# Registry CRUD
# ══════════════════════════════════════════════════════════════════════

class TestRegistryCRUD:
    def test_count_at_import(self):
        assert CircuitRegistry.count() == 7

    def test_names_at_import(self):
        expected = {
            "Randles-CPE-W", "Two-Arc-CPE", "Inductive-CPE",
            "Coating-CPE", "Warburg-Finite", "ZARC-ZARC-W", "Simple-RC",
        }
        assert set(CircuitRegistry.names()) == expected

    def test_all_returns_list(self):
        all_circuits = CircuitRegistry.all()
        assert isinstance(all_circuits, list)
        assert len(all_circuits) == 7

    def test_get_known_circuit(self):
        tmpl = CircuitRegistry.get("Randles-CPE-W")
        assert tmpl.name == "Randles-CPE-W"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            CircuitRegistry.get("NonExistent-Circuit")

    def test_register_custom(self):
        custom = CircuitTemplate(
            name="My-Test",
            param_names=["R"],
            bounds=([0.0], [1e6]),
            model_fn=lambda p, w: p[0] * np.ones_like(w, dtype=complex),
            init_fn=lambda w, z: np.array([100.0]),
            diagram="R",
            description="Test resistor",
        )
        CircuitRegistry.register(custom)
        assert "My-Test" in CircuitRegistry.names()
        assert CircuitRegistry.count() == 8
        assert CircuitRegistry.get("My-Test") is custom

    def test_register_overwrite(self):
        orig = CircuitRegistry.get("Simple-RC")
        custom = CircuitTemplate(
            name="Simple-RC",
            param_names=["R"],
            bounds=([0.0], [1e6]),
            model_fn=lambda p, w: p[0] * np.ones_like(w, dtype=complex),
            init_fn=lambda w, z: np.array([100.0]),
            diagram="R",
        )
        CircuitRegistry.register(custom)
        assert CircuitRegistry.get("Simple-RC") is custom
        # Count unchanged
        assert CircuitRegistry.count() == 7

    def test_clear(self):
        CircuitRegistry.clear()
        assert CircuitRegistry.count() == 0

    def test_from_config_all(self):
        result = CircuitRegistry.from_config(None)
        assert len(result) == 7

    def test_from_config_filtered(self):
        result = CircuitRegistry.from_config(None, names=["Randles-CPE-W", "Simple-RC"])
        assert len(result) == 2
        assert result[0].name == "Randles-CPE-W"
        assert result[1].name == "Simple-RC"

    def test_from_config_unknown_skipped(self):
        result = CircuitRegistry.from_config(None, names=["Randles-CPE-W", "Bogus"])
        assert len(result) == 1


# ══════════════════════════════════════════════════════════════════════
# CircuitTemplate metadata
# ══════════════════════════════════════════════════════════════════════

class TestCircuitMetadata:
    @pytest.mark.parametrize("name", CircuitRegistry.names())
    def test_has_description(self, name):
        tmpl = CircuitRegistry.get(name)
        assert isinstance(tmpl.description, str)
        assert len(tmpl.description) > 10, f"{name} needs a description"

    @pytest.mark.parametrize("name", CircuitRegistry.names())
    def test_has_physical_meaning(self, name):
        tmpl = CircuitRegistry.get(name)
        assert isinstance(tmpl.physical_meaning, dict)
        # Every param should have a meaning entry
        for p in tmpl.param_names:
            assert p in tmpl.physical_meaning, (
                f"{name}: missing physical_meaning for param '{p}'"
            )

    @pytest.mark.parametrize("name", CircuitRegistry.names())
    def test_has_typical_systems(self, name):
        tmpl = CircuitRegistry.get(name)
        assert isinstance(tmpl.typical_systems, list)
        assert len(tmpl.typical_systems) >= 1

    @pytest.mark.parametrize("name", CircuitRegistry.names())
    def test_has_diagram(self, name):
        tmpl = CircuitRegistry.get(name)
        assert isinstance(tmpl.diagram, str)
        assert len(tmpl.diagram) >= 3


# ══════════════════════════════════════════════════════════════════════
# Model functions produce valid impedance
# ══════════════════════════════════════════════════════════════════════

_OMEGA = 2 * np.pi * np.logspace(-1, 5, 30)


class TestCircuitModels:
    @pytest.mark.parametrize("name", CircuitRegistry.names())
    def test_model_fn_returns_complex(self, name):
        tmpl = CircuitRegistry.get(name)
        # Use the middle of bounds as params
        lb = np.array(tmpl.bounds[0])
        ub = np.array(tmpl.bounds[1])
        p0 = np.sqrt(lb * ub)  # geometric mean
        z = tmpl.model_fn(p0, _OMEGA)
        assert np.iscomplexobj(z)
        assert z.shape == _OMEGA.shape

    @pytest.mark.parametrize("name", CircuitRegistry.names())
    def test_init_fn_returns_array(self, name):
        tmpl = CircuitRegistry.get(name)
        z_dummy = (100 + 50j) * np.ones_like(_OMEGA)
        p0 = tmpl.init_fn(_OMEGA, z_dummy)
        assert isinstance(p0, np.ndarray)
        assert len(p0) == len(tmpl.param_names)

    @pytest.mark.parametrize("name", CircuitRegistry.names())
    def test_init_within_bounds(self, name):
        tmpl = CircuitRegistry.get(name)
        z_dummy = (100 + 50j) * np.ones(len(_OMEGA), dtype=complex)
        z_dummy.real += np.linspace(0, 200, len(_OMEGA))
        p0 = tmpl.init_fn(_OMEGA, z_dummy)
        lb = np.array(tmpl.bounds[0])
        ub = np.array(tmpl.bounds[1])
        assert np.all(p0 >= lb), f"{name}: init below lower bound"
        assert np.all(p0 <= ub), f"{name}: init above upper bound"


# ══════════════════════════════════════════════════════════════════════
# circuit_catalog() delegates to registry
# ══════════════════════════════════════════════════════════════════════

class TestCatalogDelegation:
    def test_circuit_catalog_returns_all(self):
        from src.circuit_fitting import circuit_catalog
        cat = circuit_catalog()
        assert len(cat) == 7
        names = {c.name for c in cat}
        assert "Simple-RC" in names
        assert "ZARC-ZARC-W" in names


# ══════════════════════════════════════════════════════════════════════
# Shortlist heuristic with expanded catalog
# ══════════════════════════════════════════════════════════════════════

class TestShortlistExpanded:
    def _catalog(self):
        return CircuitRegistry.all()

    def test_always_includes_randles(self):
        from src.circuit_fitting import shortlist_circuits
        features = {"phase_min": -30, "phase_max": 5, "logf_slope_low": 0.0}
        picks = shortlist_circuits(features, self._catalog())
        names = [p.name for p in picks]
        assert "Randles-CPE-W" in names

    def test_strong_capacitive_arc_adds_coating(self):
        from src.circuit_fitting import shortlist_circuits
        features = {"phase_min": -80, "phase_max": 5, "logf_slope_low": 0.0}
        picks = shortlist_circuits(features, self._catalog(), top_n=5)
        names = [p.name for p in picks]
        assert "Two-Arc-CPE" in names
        assert "Coating-CPE" in names

    def test_diffusion_tail_adds_warburg_finite(self):
        from src.circuit_fitting import shortlist_circuits
        features = {"phase_min": -40, "phase_max": 5, "logf_slope_low": -0.5}
        picks = shortlist_circuits(features, self._catalog(), top_n=5)
        names = [p.name for p in picks]
        assert "Warburg-Finite" in names

    def test_inductive_loop_detected(self):
        from src.circuit_fitting import shortlist_circuits
        features = {"phase_min": -20, "phase_max": 15, "logf_slope_low": 0.0}
        picks = shortlist_circuits(features, self._catalog(), top_n=5)
        names = [p.name for p in picks]
        assert "Inductive-CPE" in names

    def test_wide_range_adds_zarc_zarc_w(self):
        from src.circuit_fitting import shortlist_circuits
        features = {
            "phase_min": -60, "phase_max": 5,
            "logf_slope_low": -0.1, "mag_range": 200,
        }
        picks = shortlist_circuits(features, self._catalog(), top_n=5)
        names = [p.name for p in picks]
        assert "ZARC-ZARC-W" in names

    def test_simple_rc_as_baseline(self):
        from src.circuit_fitting import shortlist_circuits
        features = {"phase_min": -30, "phase_max": 5, "logf_slope_low": 0.0}
        picks = shortlist_circuits(features, self._catalog(), top_n=3)
        names = [p.name for p in picks]
        assert "Simple-RC" in names

    def test_top_n_respected(self):
        from src.circuit_fitting import shortlist_circuits
        features = {
            "phase_min": -80, "phase_max": 15,
            "logf_slope_low": -0.5, "mag_range": 200,
        }
        picks = shortlist_circuits(features, self._catalog(), top_n=3)
        assert len(picks) <= 3


# ══════════════════════════════════════════════════════════════════════
# Inheritance from base CircuitTemplate
# ══════════════════════════════════════════════════════════════════════

class TestInheritance:
    def test_is_subclass(self):
        assert issubclass(CircuitTemplate, BaseTemplate)

    def test_registry_instances_have_base_fields(self):
        tmpl = CircuitRegistry.get("Randles-CPE-W")
        assert hasattr(tmpl, "name")
        assert hasattr(tmpl, "param_names")
        assert hasattr(tmpl, "bounds")
        assert hasattr(tmpl, "model_fn")
        assert hasattr(tmpl, "init_fn")
        assert hasattr(tmpl, "diagram")
