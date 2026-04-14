"""Circuit Registry — pattern Registry for EIS equivalent circuits.

Every circuit is registered as a :class:`CircuitTemplate` with rich metadata
(description, physical meaning per parameter, typical electrochemical systems).

Built-in circuits (registered at import time)
----------------------------------------------
1. ``Randles-CPE-W``    — Rs − (Rp ‖ CPE) − W
2. ``Two-Arc-CPE``      — Rs − (Rp1 ‖ CPE1) − (Rp2 ‖ CPE2)
3. ``Inductive-CPE``    — Rs − L − (Rp ‖ CPE)
4. ``Coating-CPE``      — Rs − (Rcoat ‖ CPEcoat) − (Rct ‖ CPEdl)
5. ``Warburg-Finite``   — Rs − (Rp ‖ CPE) − Wfinite
6. ``ZARC-ZARC-W``      — Rs − ZARC₁ − ZARC₂ − W
7. ``Simple-RC``        — Rs − (Rp ‖ C)

Public API
----------
``CircuitRegistry.register(template)``
``CircuitRegistry.get(name) -> CircuitTemplate``
``CircuitRegistry.all() -> list[CircuitTemplate]``
``CircuitRegistry.names() -> list[str]``
``CircuitRegistry.from_config(config) -> list[CircuitTemplate]``
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.circuit_fitting import (
    CircuitTemplate as _BaseTemplate,
    _cpe,
    _warburg,
    _inductor,
)

logger = logging.getLogger(__name__)


# ── Extended CircuitTemplate ─────────────────────────────────────────
class CircuitTemplate(_BaseTemplate):
    """CircuitTemplate with rich metadata for interpretability.

    Inherits all fields from the base dataclass and adds three optional
    attributes that are populated for registered circuits.
    """

    description: str = ""
    physical_meaning: Dict[str, str] = None  # type: ignore[assignment]
    typical_systems: List[str] = None  # type: ignore[assignment]

    def __init__(
        self,
        name: str,
        param_names: List[str],
        bounds: Tuple[List[float], List[float]],
        model_fn: Callable,
        init_fn: Callable,
        diagram: str,
        *,
        description: str = "",
        physical_meaning: Optional[Dict[str, str]] = None,
        typical_systems: Optional[List[str]] = None,
    ):
        super().__init__(
            name=name,
            param_names=param_names,
            bounds=bounds,
            model_fn=model_fn,
            init_fn=init_fn,
            diagram=diagram,
        )
        self.description = description
        self.physical_meaning = physical_meaning or {}
        self.typical_systems = typical_systems or []


# ── Registry singleton ───────────────────────────────────────────────
class CircuitRegistry:
    """Global registry of circuit templates (class-level singleton)."""

    _circuits: Dict[str, CircuitTemplate] = {}

    @classmethod
    def register(cls, template: CircuitTemplate) -> None:
        """Register (or overwrite) a :class:`CircuitTemplate`."""
        cls._circuits[template.name] = template
        logger.debug("Registered circuit '%s'", template.name)

    @classmethod
    def get(cls, name: str) -> CircuitTemplate:
        """Return a registered circuit by name.

        Raises ``KeyError`` if not found.
        """
        return cls._circuits[name]

    @classmethod
    def all(cls) -> List[CircuitTemplate]:
        """Return all registered circuits in insertion order."""
        return list(cls._circuits.values())

    @classmethod
    def names(cls) -> List[str]:
        """Return the names of all registered circuits."""
        return list(cls._circuits.keys())

    @classmethod
    def from_config(
        cls,
        config,
        *,
        names: Optional[Sequence[str]] = None,
    ) -> List[CircuitTemplate]:
        """Return circuits filtered by *names*.

        If *names* is ``None``, return all.  Unknown names are silently
        skipped with a warning.
        """
        if names is None:
            return cls.all()
        result: List[CircuitTemplate] = []
        for n in names:
            if n in cls._circuits:
                result.append(cls._circuits[n])
            else:
                logger.warning("Circuit '%s' not in registry — skipped.", n)
        return result

    @classmethod
    def clear(cls) -> None:
        """Remove all registered circuits (mostly for testing)."""
        cls._circuits.clear()

    @classmethod
    def count(cls) -> int:
        """Number of registered circuits."""
        return len(cls._circuits)


# ══════════════════════════════════════════════════════════════════════
# Built-in circuit definitions
# ══════════════════════════════════════════════════════════════════════

# Helper for typical init of Rs from high-freq real part
def _rs_from_z(z: np.ndarray) -> float:
    return float(max(np.nanmin(z.real[-5:]) if z.size >= 5 else z.real[-1], 1e-3))


# ---------- 1. Randles-CPE-W ------------------------------------------
def _make_randles_cpe_w() -> CircuitTemplate:
    param_names = ["Rs", "Rp", "Q", "n", "Sigma"]
    bounds = ([1e-6, 1e-3, 1e-12, 0.3, 1e-10],
              [1e6,  1e8,  1.0,   1.0, 1e5])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp, Q, n, sigma = p
        Zcpe = _cpe(omega, Q, n)
        Zw = _warburg(omega, sigma)
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
        return Rs + Zpar + Zw

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        rp = float(max(z.real[0] - rs, 0.1))
        return np.array([rs, rp, 1e-4, 0.85, 0.01])

    return CircuitTemplate(
        name="Randles-CPE-W",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − (Rp ‖ CPE) − W",
        description=(
            "Modified Randles circuit with constant-phase element (CPE) "
            "replacing an ideal capacitor and a semi-infinite Warburg "
            "diffusion element.  Captures charge-transfer kinetics and "
            "mass-transport limitations."
        ),
        physical_meaning={
            "Rs": "Ohmic resistance of the electrolyte (Ω)",
            "Rp": "Charge-transfer (polarization) resistance (Ω)",
            "Q": "CPE pseudo-capacitance parameter (F·s^(n-1))",
            "n": "CPE exponent — 1 = ideal capacitor, 0.5 = Warburg-like",
            "Sigma": "Warburg coefficient related to diffusion (Ω·s^−½)",
        },
        typical_systems=[
            "Li-ion batteries",
            "Supercapacitors (with diffusion)",
            "Corrosion cells with mass-transport control",
            "Fuel-cell electrodes",
        ],
    )


# ---------- 2. Two-Arc-CPE -------------------------------------------
def _make_two_arc_cpe() -> CircuitTemplate:
    param_names = ["Rs", "Rp1", "Q1", "n1", "Rp2", "Q2", "n2"]
    bounds = ([1e-6, 1e-3, 1e-12, 0.3, 1e-3, 1e-12, 0.3],
              [1e6,  1e8,  1.0,   1.0, 1e8,  1.0,   1.0])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp1, Q1, n1, Rp2, Q2, n2 = p
        Zcpe1 = _cpe(omega, Q1, n1)
        Zcpe2 = _cpe(omega, Q2, n2)
        Zpar1 = 1.0 / (1.0 / Rp1 + 1.0 / Zcpe1)
        Zpar2 = 1.0 / (1.0 / Rp2 + 1.0 / Zcpe2)
        return Rs + Zpar1 + Zpar2

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        span = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, span * 0.6, 1e-4, 0.85, span * 0.4, 5e-5, 0.8])

    return CircuitTemplate(
        name="Two-Arc-CPE",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − (Rp1 ‖ CPE1) − (Rp2 ‖ CPE2)",
        description=(
            "Two-ZARC circuit with two distinct R‖CPE arcs.  Models systems "
            "with two separable time constants — e.g. grain/grain-boundary, "
            "anode/cathode, or SEI film + charge-transfer."
        ),
        physical_meaning={
            "Rs": "Ohmic / electrolyte resistance (Ω)",
            "Rp1": "Resistance of the first process (Ω)",
            "Q1": "CPE parameter for the first process (F·s^(n-1))",
            "n1": "CPE exponent of the first process",
            "Rp2": "Resistance of the second process (Ω)",
            "Q2": "CPE parameter for the second process (F·s^(n-1))",
            "n2": "CPE exponent of the second process",
        },
        typical_systems=[
            "Solid-oxide fuel cells (SOFC)",
            "Solid electrolytes (grain + grain boundary)",
            "Coated metals with two interfaces",
            "Lithium-ion cells (SEI + charge-transfer)",
        ],
    )


# ---------- 3. Inductive-CPE -----------------------------------------
def _make_inductive_cpe() -> CircuitTemplate:
    param_names = ["Rs", "L", "Rp", "Q", "n"]
    bounds = ([1e-6, 1e-9, 1e-3, 1e-12, 0.3],
              [1e6,  1.0,  1e8,  1.0,   1.0])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, L, Rp, Q, n = p
        Zl = _inductor(omega, L)
        Zcpe = _cpe(omega, Q, n)
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
        return Rs + Zl + Zpar

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        rp = float(max(z.real.max() - rs, 0.1))
        return np.array([rs, 1e-3, rp, 1e-4, 0.9])

    return CircuitTemplate(
        name="Inductive-CPE",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − L − (Rp ‖ CPE)",
        description=(
            "Randles variant with a series inductance to capture "
            "high-frequency inductive loops caused by cable/connector "
            "artefacts or adsorption processes."
        ),
        physical_meaning={
            "Rs": "Ohmic resistance (Ω)",
            "L": "Series inductance (H) — cable artefact or adsorption",
            "Rp": "Charge-transfer resistance (Ω)",
            "Q": "CPE pseudo-capacitance (F·s^(n-1))",
            "n": "CPE exponent",
        },
        typical_systems=[
            "Corroding metals with adsorbed intermediates",
            "Systems with long measurement cables",
            "PEM fuel cells at high frequency",
        ],
    )


# ---------- 4. Coating-CPE (NEW) -------------------------------------
def _make_coating_cpe() -> CircuitTemplate:
    param_names = ["Rs", "Rcoat", "Qcoat", "ncoat", "Rct", "Qdl", "ndl"]
    bounds = ([1e-6, 1e-3, 1e-12, 0.3, 1e-3, 1e-12, 0.3],
              [1e6,  1e8,  1.0,   1.0, 1e8,  1.0,   1.0])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rcoat, Qcoat, ncoat, Rct, Qdl, ndl = p
        Zcpe_coat = _cpe(omega, Qcoat, ncoat)
        Zcpe_dl = _cpe(omega, Qdl, ndl)
        Zcoat = 1.0 / (1.0 / Rcoat + 1.0 / Zcpe_coat)
        Zct = 1.0 / (1.0 / Rct + 1.0 / Zcpe_dl)
        return Rs + Zcoat + Zct

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        span = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, span * 0.3, 1e-6, 0.9, span * 0.7, 1e-4, 0.85])

    return CircuitTemplate(
        name="Coating-CPE",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − (Rcoat ‖ CPEcoat) − (Rct ‖ CPEdl)",
        description=(
            "Two-layer model for coated electrodes.  The first R‖CPE "
            "represents the protective coating (pore resistance + coating "
            "capacitance), the second represents the metal/electrolyte "
            "interface (charge-transfer + double-layer)."
        ),
        physical_meaning={
            "Rs": "Electrolyte resistance (Ω)",
            "Rcoat": "Coating (pore) resistance (Ω)",
            "Qcoat": "Coating CPE parameter (F·s^(n-1))",
            "ncoat": "Coating CPE exponent (closer to 1 = intact coating)",
            "Rct": "Charge-transfer resistance at metal surface (Ω)",
            "Qdl": "Double-layer CPE parameter (F·s^(n-1))",
            "ndl": "Double-layer CPE exponent",
        },
        typical_systems=[
            "Organic coatings on steel (EIS corrosion protection)",
            "Anodized aluminium",
            "Painted metal structures",
            "Biomedical implant coatings",
        ],
    )


# ---------- 5. Warburg-Finite (NEW) ----------------------------------
def _make_warburg_finite() -> CircuitTemplate:
    param_names = ["Rs", "Rp", "Q", "n", "Rd", "Td"]
    bounds = ([1e-6, 1e-3, 1e-12, 0.3, 1e-6, 1e-6],
              [1e6,  1e8,  1.0,   1.0, 1e8,  1e4])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp, Q, n, Rd, Td = p
        Zcpe = _cpe(omega, Q, n)
        # Finite-length Warburg: Rd * tanh(sqrt(j*omega*Td)) / sqrt(j*omega*Td)
        s = np.sqrt(1j * omega * Td)
        # Avoid division by zero at very low freq
        s_safe = np.where(np.abs(s) < 1e-30, 1e-30, s)
        Zw_finite = Rd * np.tanh(s_safe) / s_safe
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
        return Rs + Zpar + Zw_finite

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        rp = float(max(z.real[0] - rs, 0.1))
        return np.array([rs, rp, 1e-4, 0.85, rp * 0.5, 1.0])

    return CircuitTemplate(
        name="Warburg-Finite",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − (Rp ‖ CPE) − Wfinite",
        description=(
            "Modified Randles with a finite-length Warburg element.  "
            "Unlike the semi-infinite W, this element accounts for a "
            "bounded diffusion layer (thin film, porous electrode, "
            "or symmetric cell)."
        ),
        physical_meaning={
            "Rs": "Electrolyte resistance (Ω)",
            "Rp": "Charge-transfer resistance (Ω)",
            "Q": "CPE pseudo-capacitance (F·s^(n-1))",
            "n": "CPE exponent",
            "Rd": "Diffusion resistance (Ω) — proportional to layer thickness",
            "Td": "Diffusion time constant (s) = L²/D",
        },
        typical_systems=[
            "Thin-film batteries",
            "Porous electrode supercapacitors",
            "Symmetric cells for electrolyte studies",
            "Fuel-cell gas-diffusion layers",
        ],
    )


# ---------- 6. ZARC-ZARC-W (NEW) ------------------------------------
def _make_zarc_zarc_w() -> CircuitTemplate:
    param_names = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "Sigma"]
    bounds = ([1e-6, 1e-3, 1e-12, 0.3, 1e-3, 1e-12, 0.3, 1e-10],
              [1e6,  1e8,  1.0,   1.0, 1e8,  1.0,   1.0, 1e5])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, R1, Q1, n1, R2, Q2, n2, sigma = p
        Zcpe1 = _cpe(omega, Q1, n1)
        Zcpe2 = _cpe(omega, Q2, n2)
        Zarc1 = 1.0 / (1.0 / R1 + 1.0 / Zcpe1)
        Zarc2 = 1.0 / (1.0 / R2 + 1.0 / Zcpe2)
        Zw = _warburg(omega, sigma)
        return Rs + Zarc1 + Zarc2 + Zw

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        span = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, span * 0.4, 1e-4, 0.85, span * 0.3, 5e-5, 0.8, 0.01])

    return CircuitTemplate(
        name="ZARC-ZARC-W",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − ZARC₂ − W",
        description=(
            "Two ZARC elements (R‖CPE) in series followed by a "
            "semi-infinite Warburg.  Suitable for systems with two "
            "charge-transfer processes and diffusion at low frequency."
        ),
        physical_meaning={
            "Rs": "Electrolyte resistance (Ω)",
            "R1": "Resistance of the first interface (Ω)",
            "Q1": "CPE of the first interface (F·s^(n-1))",
            "n1": "CPE exponent of the first interface",
            "R2": "Resistance of the second interface (Ω)",
            "Q2": "CPE of the second interface (F·s^(n-1))",
            "n2": "CPE exponent of the second interface",
            "Sigma": "Warburg coefficient (Ω·s^−½)",
        },
        typical_systems=[
            "Multi-layer battery electrodes (SEI + CT + diffusion)",
            "Mixed-conducting ceramics (SOFC cathodes)",
            "Complex corrosion with multiple interfaces",
        ],
    )


# ---------- 7. Simple-RC (NEW) — baseline ----------------------------
def _make_simple_rc() -> CircuitTemplate:
    param_names = ["Rs", "Rp", "C"]
    bounds = ([1e-6, 1e-3, 1e-15],
              [1e6,  1e8,  1e-1])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp, C = p
        Zc = 1.0 / (1j * omega * C)
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zc)
        return Rs + Zpar

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        rp = float(max(z.real[0] - rs, 0.1))
        return np.array([rs, rp, 1e-6])

    return CircuitTemplate(
        name="Simple-RC",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − (Rp ‖ C)",
        description=(
            "Simplest possible R-C model — one ideal resistor in series "
            "with a parallel R‖C.  Serves as a BIC baseline: if a more "
            "complex circuit does not beat Simple-RC, the extra parameters "
            "are not justified."
        ),
        physical_meaning={
            "Rs": "Series (electrolyte) resistance (Ω)",
            "Rp": "Polarization resistance (Ω)",
            "C": "Ideal double-layer capacitance (F)",
        },
        typical_systems=[
            "Ideal blocking electrodes",
            "BIC baseline / null model",
            "Simple aqueous electrolytes",
        ],
    )


# ── Auto-register all built-in circuits at import time ───────────────
_BUILTIN_MAKERS = [
    _make_randles_cpe_w,
    _make_two_arc_cpe,
    _make_inductive_cpe,
    _make_coating_cpe,
    _make_warburg_finite,
    _make_zarc_zarc_w,
    _make_simple_rc,
]


def _register_builtins() -> None:
    for maker in _BUILTIN_MAKERS:
        CircuitRegistry.register(maker())


_register_builtins()
