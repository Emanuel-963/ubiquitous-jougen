"""Circuit Registry — pattern Registry for EIS equivalent circuits.

Every circuit is registered as a :class:`CircuitTemplate` with rich metadata
(description, physical meaning per parameter, typical electrochemical systems).

Built-in circuits (registered at import time)
----------------------------------------------
 1. ``Randles-CPE-W``       — Rs − (Rp ‖ CPE) − W
 2. ``Two-Arc-CPE``         — Rs − (Rp1 ‖ CPE1) − (Rp2 ‖ CPE2)
 3. ``Inductive-CPE``       — Rs − L − (Rp ‖ CPE)
 4. ``Coating-CPE``         — Rs − (Rcoat ‖ CPEcoat) − (Rct ‖ CPEdl)
 5. ``Warburg-Finite``      — Rs − (Rp ‖ CPE) − Wfinite  [transmissive / tanh]
 6. ``ZARC-ZARC-W``         — Rs − ZARC₁ − ZARC₂ − W
 7. ``Simple-RC``           — Rs − (Rp ‖ C)
 8. ``CPE-Simple``          — Rs − CPE                    [EDLC / blocking electrode]
 9. ``Warburg-Short``       — Rs − (Rp ‖ CPE) − Wo        [reflective / coth]
10. ``Gerischer``           — Rs − (Rp ‖ CPE) − Z_Ger     [SOFC / mixed conductors]
11. ``Three-ZARC``          — Rs − ZARC₁ − ZARC₂ − ZARC₃ [solid electrolytes]
12. ``Porous-Coating-TLM``  — Rs − (Rcoat ‖ Cpore) − (Rct ‖ CPEdl)
13. ``MXene-Intercalation`` — Rs − (Rsei ‖ CPEsei) − (Rct ‖ CPEdl) − Wfinite
14. ``De-Levie-TLM``        — Rs + sqrt(Ri/Ydl)·coth(L·sqrt(Ri·Ydl))
15. ``Pseudo-Capacitance-CPE`` — Rs − (Rct ‖ CPEdl) − (Rads ‖ Cads)

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

from src.circuit_fitting import CircuitTemplate as _BaseTemplate
from src.circuit_fitting import _cpe, _inductor, _warburg

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


# ── Private element helpers (used by EXT circuit factory functions) ───
def _zarc_h(omega: np.ndarray, R: float, Q: float, n: float) -> np.ndarray:
    """R ‖ CPE arc."""
    return 1.0 / (1.0 / R + 1.0 / _cpe(omega, Q, n))


def _rc_par_h(omega: np.ndarray, R: float, C: float) -> np.ndarray:
    """R ‖ C arc with ideal capacitor."""
    return 1.0 / (1.0 / R + 1j * omega * C)


def _wfin_h(omega: np.ndarray, Rd: float, Td: float) -> np.ndarray:
    """Finite-length Warburg (tanh / transmissive boundary)."""
    s = np.sqrt(1j * omega * Td)
    sa = np.abs(s)
    ss = np.where(sa < 1e-30, 1e-30 + 0j, np.where(sa > 20, s / sa * 20, s))
    return Rd * np.tanh(ss) / ss


def _wsho_h(omega: np.ndarray, Rd: float, Td: float) -> np.ndarray:
    """Finite-length Warburg (coth / reflective boundary)."""
    s = np.sqrt(1j * omega * Td)
    sa = np.abs(s)
    safe = np.where(sa < 1e-10, 1e-10 + 0j, s)
    sc = np.where(sa > 20, safe / np.where(sa > 0, sa, 1.0) * 20, safe)
    return Rd * (np.cosh(sc) / np.sinh(sc)) / safe


def _ger_h(omega: np.ndarray, Rg: float, Tg: float) -> np.ndarray:
    """Gerischer element: Rg / √(1 + j·ω·Tg)."""
    return Rg / np.sqrt(1.0 + 1j * omega * Tg)


def _tlm_h(omega: np.ndarray, Ri: float, Ydl: float, nt: float) -> np.ndarray:
    """Semi-infinite De Levie TLM: √(Ri / (Ydl·(j·ω)^nt))."""
    jw_n = (1j * omega) ** nt
    safe = np.where(np.abs(jw_n) < 1e-30, 1e-30 + 0j, jw_n)
    return np.sqrt(Ri / (Ydl * safe))


# ---------- 1. Randles-CPE-W ------------------------------------------
def _make_randles_cpe_w() -> CircuitTemplate:
    param_names = ["Rs", "Rp", "Q", "n", "Sigma"]
    bounds = ([1e-6, 1e-3, 1e-12, 0.3, 1e-10], [1e6, 1e8, 1.0, 1.0, 1e5])

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
    bounds = (
        [1e-6, 1e-3, 1e-12, 0.3, 1e-3, 1e-12, 0.3],
        [1e6, 1e8, 1.0, 1.0, 1e8, 1.0, 1.0],
    )

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
    bounds = ([1e-6, 1e-9, 1e-3, 1e-12, 0.3], [1e6, 1.0, 1e8, 1.0, 1.0])

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
    bounds = (
        [1e-6, 1e-3, 1e-12, 0.3, 1e-3, 1e-12, 0.3],
        [1e6, 1e8, 1.0, 1.0, 1e8, 1.0, 1.0],
    )

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
    bounds = ([1e-6, 1e-3, 1e-12, 0.3, 1e-6, 1e-6], [1e6, 1e8, 1.0, 1.0, 1e8, 1e4])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp, Q, n, Rd, Td = p
        Zcpe = _cpe(omega, Q, n)
        # Finite-length Warburg: Rd * tanh(sqrt(j*omega*Td)) / sqrt(j*omega*Td)
        s = np.sqrt(1j * omega * Td)
        # Avoid division by zero at very low freq
        s_safe = np.where(np.abs(s) < 1e-30, 1e-30, s)
        abs_s = np.abs(s_safe)
        large = abs_s > 20.0
        tanh_over_s = np.empty_like(s_safe, dtype=complex)
        tanh_over_s[large] = 1.0 / s_safe[large]
        tanh_over_s[~large] = np.tanh(s_safe[~large]) / s_safe[~large]
        Zw_finite = Rd * tanh_over_s
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
    bounds = (
        [1e-6, 1e-3, 1e-12, 0.3, 1e-3, 1e-12, 0.3, 1e-10],
        [1e6, 1e8, 1.0, 1.0, 1e8, 1.0, 1.0, 1e5],
    )

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
    bounds = ([1e-6, 1e-3, 1e-15], [1e6, 1e8, 1e-1])

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


# ---------- 8. CPE-Simple -------------------------------------------
def _make_cpe_simple() -> CircuitTemplate:
    """Rs − CPE: pure constant-phase element (EDLC, blocking electrodes)."""
    param_names = ["Rs", "Q", "n"]
    bounds = ([1e-6, 1e-12, 0.5], [1e3, 1e3, 1.0])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Q, n = p
        return Rs + _cpe(omega, Q, n)

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        return np.array([rs, 1e-3, 0.95])

    return CircuitTemplate(
        name="CPE-Simple",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − CPE",
        description=(
            "Simplest capacitive model: electrolyte resistance in series with "
            "a single constant-phase element.  Models ideal to near-ideal "
            "electric double-layer capacitors (EDLC) and blocking electrodes "
            "where no charge-transfer occurs within the measurement window."
        ),
        physical_meaning={
            "Rs": "Equivalent series resistance — ESR (Ω)",
            "Q": "CPE pseudo-capacitance (F·s^(n-1)); n=1 → ideal capacitor",
            "n": "CPE exponent — n→1 ideal EDLC, n=0.85–0.95 real carbon EDLC",
        },
        typical_systems=[
            "Electric double-layer capacitors (EDLC / supercapacitors)",
            "Blocking platinum or glassy-carbon electrodes in acid",
            "Ion-blocking electrodes in solid-state cells",
            "Polymer electrolyte films at high frequency",
        ],
    )


# ---------- 9. Warburg-Short (reflective / open boundary) -----------
def _make_warburg_short() -> CircuitTemplate:
    """Rs − (Rp ‖ CPE) − Wo  [coth Warburg, reflective boundary]."""
    param_names = ["Rs", "Rp", "Q", "n", "Rd", "Td"]
    bounds = ([1e-6, 1e-3, 1e-12, 0.3, 1e-6, 1e-6], [1e6, 1e8, 1.0, 1.0, 1e8, 1e4])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp, Q, n, Rd, Td = p
        Zcpe = _cpe(omega, Q, n)
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
        # Warburg open (reflective): Rd * coth(s) / s,  s = sqrt(j*omega*Td)
        s = np.sqrt(1j * omega * Td)
        s_abs = np.abs(s)
        safe_s = np.where(s_abs < 1e-10, (1e-10 + 0j) * np.ones_like(s), s)
        # Clamp |s| ≤ 20 before cosh/sinh to avoid overflow (np.where evaluates both
        # branches eagerly).  At |s| = 20, coth(20) ≈ 1.0 to 10 sig-figs.
        s_clamped = np.where(
            s_abs > 20, safe_s / np.where(s_abs > 0, s_abs, 1.0) * 20, safe_s
        )
        coth_s = np.cosh(s_clamped) / np.sinh(s_clamped)
        Zw_open = Rd * coth_s / safe_s
        return Rs + Zpar + Zw_open

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        rp = float(max(z.real[0] - rs, 0.1))
        return np.array([rs, rp, 1e-4, 0.85, rp * 0.5, 1.0])

    return CircuitTemplate(
        name="Warburg-Short",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − (Rp ‖ CPE) − Wo",
        description=(
            "Modified Randles with a finite-length Warburg with reflective "
            "(open/blocking) boundary condition — coth form.  At low "
            "frequencies the impedance becomes purely capacitive, unlike the "
            "transmissive Warburg-Finite (tanh) which flattens to a real "
            "resistance.  Distinguishable by the upturn toward −Im(Z) axis "
            "at the lowest measured frequencies."
        ),
        physical_meaning={
            "Rs": "Electrolyte resistance (Ω)",
            "Rp": "Charge-transfer resistance (Ω)",
            "Q": "CPE pseudo-capacitance (F·s^(n-1))",
            "n": "CPE exponent",
            "Rd": "Diffusion resistance — ohmic equivalent of diffusion layer (Ω)",
            "Td": "Diffusion time constant (s) = L²/D",
        },
        typical_systems=[
            "Thin-layer cells with blocking counter electrode",
            "Ion-selective electrodes and polymer membranes",
            "Porous electrodes with blocked pore tips",
            "Solid polymer electrolytes (lithium conductors)",
        ],
    )


# ---------- 10. Gerischer -------------------------------------------
def _make_gerischer() -> CircuitTemplate:
    """Rs − (Rp ‖ CPE) − Z_Gerischer  [distributed reaction impedance]."""
    param_names = ["Rs", "Rp", "Q", "n", "Rg", "Tg"]
    bounds = ([1e-6, 1e-3, 1e-12, 0.3, 1e-3, 1e-8], [1e6, 1e8, 1.0, 1.0, 1e8, 1e4])

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rp, Q, n, Rg, Tg = p
        Zcpe = _cpe(omega, Q, n)
        Zpar = 1.0 / (1.0 / Rp + 1.0 / Zcpe)
        # Gerischer element: Rg / sqrt(1 + j*omega*Tg)
        Zg = Rg / np.sqrt(1.0 + 1j * omega * Tg)
        return Rs + Zpar + Zg

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        rp = float(max(z.real[0] - rs, 0.1))
        return np.array([rs, rp * 0.5, 1e-4, 0.85, rp * 0.3, 1e-3])

    return CircuitTemplate(
        name="Gerischer",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − (Rp ‖ CPE) − Z_Gerischer",
        description=(
            "Gerischer impedance element models a distributed electrochemical "
            "reaction coupled to a homogeneous chemical step.  Produces a "
            "characteristic arc that starts at ~45° and curves toward the "
            "real axis at low frequency — distinct from the Warburg 45° "
            "line which remains straight."
        ),
        physical_meaning={
            "Rs": "Electrolyte / series resistance (Ω)",
            "Rp": "Charge-transfer resistance (Ω)",
            "Q": "CPE pseudo-capacitance (F·s^(n-1))",
            "n": "CPE exponent",
            "Rg": "Gerischer resistance — amplitude of distributed reaction (Ω)",
            "Tg": "Gerischer time constant = k_f/D (s); k_f = rate constant, D = diffusivity",
        },
        typical_systems=[
            "SOFC cathodes (LSC, LSCF, LSM) — oxygen reduction reaction",
            "Mixed ionic-electronic conductors (MIEC)",
            "Oxygen reduction reaction (ORR) in alkaline media",
            "Porous gas-diffusion electrodes with coupled kinetics",
        ],
    )


# ---------- 11. Three-ZARC ------------------------------------------
def _make_three_zarc() -> CircuitTemplate:
    """Rs − ZARC1 − ZARC2 − ZARC3  [solid electrolytes, ceramics]."""
    param_names = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "R3", "Q3", "n3"]
    bounds = (
        [1e-6, 1e-3, 1e-15, 0.5, 1e-3, 1e-15, 0.5, 1e-3, 1e-15, 0.4],
        [1e4, 1e6, 1e-6, 1.0, 1e7, 1e-5, 1.0, 1e8, 1e-3, 1.0],
    )

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, R1, Q1, n1, R2, Q2, n2, R3, Q3, n3 = p
        Zarc1 = 1.0 / (1.0 / R1 + 1.0 / _cpe(omega, Q1, n1))
        Zarc2 = 1.0 / (1.0 / R2 + 1.0 / _cpe(omega, Q2, n2))
        Zarc3 = 1.0 / (1.0 / R3 + 1.0 / _cpe(omega, Q3, n3))
        return Rs + Zarc1 + Zarc2 + Zarc3

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        span = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array(
            [
                rs,
                span * 0.2,
                1e-12,
                0.95,
                span * 0.3,
                1e-10,
                0.85,
                span * 0.5,
                1e-7,
                0.75,
            ]
        )

    return CircuitTemplate(
        name="Three-ZARC",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − ZARC₂ − ZARC₃",
        description=(
            "Three ZARC elements (R‖CPE) in series.  Models solid-state "
            "systems with three distinct time-constant processes: bulk "
            "conduction, grain-boundary resistance, and electrode/interface "
            "polarization — each visible as a separate arc in a different "
            "frequency decade."
        ),
        physical_meaning={
            "Rs": "Geometrical / instrumental series resistance (Ω)",
            "R1": "Bulk (grain) resistance (Ω) — high-frequency arc",
            "Q1": "Bulk CPE parameter (F·s^(n-1))",
            "n1": "Bulk CPE exponent (close to 1 for ideal ceramics)",
            "R2": "Grain-boundary resistance (Ω) — mid-frequency arc",
            "Q2": "Grain-boundary CPE parameter (F·s^(n-1))",
            "n2": "Grain-boundary CPE exponent",
            "R3": "Electrode / interface resistance (Ω) — low-frequency arc",
            "Q3": "Electrode CPE parameter (F·s^(n-1))",
            "n3": "Electrode CPE exponent",
        },
        typical_systems=[
            "Garnet solid electrolytes (LLZO, Li6.5La3Zr1.5Ta0.5O12)",
            "NASICON-type conductors (LAGP, LATP)",
            "Yttria-stabilized zirconia (YSZ) — SOFC electrolyte",
            "Li-rich glass-ceramics and glass electrolytes",
            "Polycrystalline ceramics — bulk + grain boundary + electrode",
        ],
    )


# ---------- 12. Porous-Coating-TLM -----------------------------------
def _make_porous_coating_tlm() -> CircuitTemplate:
    """Transmission-Line Model for a porous coating with double-layer.

    Circuit topology
    ~~~~~~~~~~~~~~~~
    Rs − [Rcoat ‖ Cpore] − [Rct ‖ CPEdl]

    This is a simplified two-element TLM (De Levie, Tribollet/Orazem §15):
    - The outer RC branch (Rcoat‖Cpore) models the resistive/capacitive
      response of the coating itself at mid frequencies.
    - The inner ZARC branch (Rct‖CPEdl) models the porous interface charge-
      transfer at lower frequencies.

    The key feature that distinguishes this from ``Coating-CPE`` is that
    the coating capacitance ``Cpore`` is a pure capacitor (CPE exponent = 1)
    while the double-layer is a CPE (n_dl < 1), capturing the physical fact
    that pore geometry distorts the double-layer uniformity far more than
    the dielectric of the intact film.

    Parameters
    ----------
    Rs      — solution resistance (Ω)
    Rcoat   — coating / pore resistance (Ω)
    Cpore   — coating capacitance (F); ideal capacitor, not CPE
    Rct     — charge-transfer resistance at pore bottom (Ω)
    Qdl     — double-layer CPE prefactor (F·s^(n-1))
    n_dl    — double-layer CPE exponent  (0.5–1.0)
    """
    param_names = ["Rs", "Rcoat", "Cpore", "Rct", "Qdl", "n_dl"]
    bounds = (
        [1e-3, 1e-3, 1e-12, 1e-3, 1e-14, 0.50],
        [1e4, 1e6, 1e-4, 1e7, 1e-3, 1.00],
    )

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rcoat, Cpore, Rct, Qdl, n_dl = p
        # Coating branch: R ‖ C  (ideal capacitor)
        Zcoating = 1.0 / (1.0 / Rcoat + 1j * omega * Cpore)
        # Double-layer branch: Rct ‖ CPEdl
        Zdl = 1.0 / (1.0 / Rct + 1.0 / _cpe(omega, Qdl, n_dl))
        return Rs + Zcoating + Zdl

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        span = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, span * 0.3, 1e-9, span * 0.7, 1e-11, 0.85])

    return CircuitTemplate(
        name="Porous-Coating-TLM",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − (Rcoat ‖ Cpore) − (Rct ‖ CPEdl)",
        description=(
            "Transmission-line model for a porous organic or oxide coating "
            "over a metal substrate.  Separates the dielectric capacitance "
            "of the coating from the double-layer CPE at the pore tips, "
            "giving physically distinct time constants visible at mid and "
            "low frequencies respectively.  Preferred over Coating-CPE when "
            "both arcs are clearly resolved."
        ),
        physical_meaning={
            "Rs": "Electrolyte / uncompensated resistance (Ω)",
            "Rcoat": "Ionic resistance through pore electrolyte (Ω)",
            "Cpore": "Geometric (dielectric) capacitance of intact coating (F)",
            "Rct": "Charge-transfer resistance at pore base (Ω)",
            "Qdl": "Double-layer CPE prefactor at pore surface (F·s^(n-1))",
            "n_dl": "DL CPE exponent — deviation from ideal capacitance (0.5–1)",
        },
        typical_systems=[
            "Epoxy / polyurethane coatings on steel after immersion",
            "Anodised aluminium (Type II/III) with pore-filling",
            "Phosphated steel — dual porosity corrosion product",
            "Li-ion SEI on graphite anode (simplified 2-layer model)",
            "Cerium-conversion coatings on AA2024",
        ],
    )


# ---------- 13. MXene-Intercalation -----------------------------------
def _make_mxene_intercalation() -> CircuitTemplate:
    """EIS model for MXene intercalation pseudocapacitance in aqueous acid.

    Circuit topology
    ~~~~~~~~~~~~~~~~
    Rs − (RSEI ‖ CPSEI) − (Rct ‖ CPEdl) − Wfinite

    Physical description
    ~~~~~~~~~~~~~~~~~~~~
    - **RSEI ‖ CPSEI** (high frequency): surface termination passivation
      layer on Nb₂CTₓ/Ti₃C₂Tₓ.  The *termination groups* (=O, −OH, −F)
      form a disordered thin film whose EIS response is a depressed
      semicircle at high ω.
    - **Rct ‖ CPEdl** (mid frequency): charge-transfer + double-layer at
      the MXene/electrolyte interface.  CPE exponent n < 1 captures
      geometric roughness of delaminated flakes.
    - **Wfinite** (low frequency): bounded (finite-length) diffusion of
      protons / cations into the 2D interlayer spacing.  The finite
      boundary condition (tanh model) is correct because the interlayer
      is a closed medium of thickness d.

    This circuit is appropriate for:
    - Ti₃C₂Tₓ and Nb₂CTₓ in H₂SO₄ (proton intercalation)
    - Ti₃C₂Tₓ in Na₂SO₄ / NaCl (sodium intercalation)
    - Post-cycling characterisation to track RSEI evolution

    Parameters
    ----------
    Rs      — electrolyte resistance (Ω)
    Rsei    — surface termination layer resistance (Ω)
    Qsei    — surface layer CPE prefactor (F·s^(n-1))
    n_sei   — surface layer CPE exponent (0.6–1.0)
    Rct     — charge-transfer resistance (Ω)
    Qdl     — double-layer CPE prefactor (F·s^(n-1))
    n_dl    — double-layer CPE exponent (0.6–1.0)
    AW      — finite Warburg coefficient (Ω·s^(-1/2))
    tau_d   — diffusion time constant τ = d²/D (s)
    """
    param_names = ["Rs", "Rsei", "Qsei", "n_sei", "Rct", "Qdl", "n_dl", "AW", "tau_d"]
    bounds = (
        [1e-3, 1e-3, 1e-14, 0.50, 1e-3, 1e-14, 0.50, 1e-4, 1e-4],
        [500.0, 1e5, 1e-5, 1.00, 1e6, 1e-3, 1.00, 1e4, 1e4],
    )

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rsei, Qsei, n_sei, Rct, Qdl, n_dl, AW, tau_d = p
        # Surface termination layer: RSEI ‖ CPSEI
        Zsei = 1.0 / (1.0 / Rsei + 1.0 / _cpe(omega, Qsei, n_sei))
        # Charge-transfer: Rct ‖ CPEdl
        Zdl = 1.0 / (1.0 / Rct + 1.0 / _cpe(omega, Qdl, n_dl))
        # Finite-length Warburg (tanh): Z = AW/sqrt(jω) * tanh(sqrt(jω·τ_d))
        # For large |x|: tanh(x)→1 so Z→AW/sqrt(jω); at ω→0 use Taylor: Z→AW*sqrt(τ_d)
        jw_eps = 1j * omega + 1e-30
        sqrt_jomega = np.sqrt(jw_eps)
        gamma = np.sqrt(jw_eps * tau_d)
        with np.errstate(over="ignore", invalid="ignore"):
            tanh_gamma = np.where(
                np.abs(gamma.real) > 20,
                np.ones_like(gamma, dtype=complex),
                np.tanh(gamma),
            )
        Zw = AW / sqrt_jomega * tanh_gamma
        return Rs + Zsei + Zdl + Zw

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        span = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array(
            [rs, span * 0.05, 1e-9, 0.85, span * 0.5, 1e-10, 0.80, 10.0, 0.1]
        )

    return CircuitTemplate(
        name="MXene-Intercalation",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − (Rsei ‖ CPEsei) − (Rct ‖ CPEdl) − Wfinite",
        description=(
            "Equivalent circuit for MXene (Ti₃C₂Tₓ, Nb₂CTₓ) intercalation "
            "pseudocapacitance in aqueous electrolytes.  Three frequency regions: "
            "(1) surface termination layer response at high ω, (2) charge-transfer "
            "at mid ω, (3) finite diffusion of intercalating ions (H⁺, Na⁺) into "
            "2D interlayer spacing at low ω (tanh/Warburg model with closed boundary)."
        ),
        physical_meaning={
            "Rs": "Electrolyte / uncompensated resistance (Ω)",
            "Rsei": "Surface termination passivation resistance (Ω); "
            "increases after cycling in acid → tracks degradation",
            "Qsei": "Surface layer CPE prefactor (F·s^(n-1))",
            "n_sei": "Surface layer CPE exponent; n < 0.8 → disordered terminations",
            "Rct": "Charge-transfer resistance at MXene/electrolyte interface (Ω)",
            "Qdl": "Double-layer CPE prefactor (F·s^(n-1)); proportional to ECSA",
            "n_dl": "Double-layer CPE exponent; n < 0.85 → rough delaminated surface",
            "AW": "Finite Warburg coefficient Aw (Ω·s^(-1/2)); related to diffusivity",
            "tau_d": "Diffusion time constant τ=d²/D (s); d=interlayer spacing",
        },
        typical_systems=[
            "Ti₃C₂Tₓ in 1 M H₂SO₄ (proton intercalation pseudocapacitance)",
            "Nb₂CTₓ in 1 M H₂SO₄ (Nb⁵⁺/Nb⁴⁺ redox + proton insertion)",
            "Ti₃C₂Tₓ in 1 M Na₂SO₄ (sodium intercalation)",
            "Delaminated MXene films post-cycling — track RSEI evolution",
            "MXene/CNT composites with electrolyte accessible pores",
        ],
    )


# ---------- 14. De-Levie-TLM ------------------------------------------
def _make_de_levie_tlm() -> CircuitTemplate:
    """Full De Levie transmission-line for a porous electrode.

    Circuit topology
    ~~~~~~~~~~~~~~~~
    Rs + Z_TLM  where  Z_TLM = sqrt(Ri/Ydl) · coth(L·sqrt(Ri·Ydl))

    Physical description
    ~~~~~~~~~~~~~~~~~~~~
    The De Levie model (1963) treats a porous electrode as a cylindrical
    pore of length L filled with electrolyte.  Along the pore:
    - **Ri** (Ω/cm) is the ionic resistance per unit length of the pore
      electrolyte
    - **Ydl** = CPE admittance of the pore wall per unit length

    At **high frequencies** the TLM impedance approaches Rs + sqrt(Ri/Ydl)
    (semi-infinite diffusion into the pore → 45° line in Nyquist).
    At **low frequencies** the coth approaches 1/x and the response is
    dominated by the total double-layer capacitance of all pore walls.

    This is the correct model for:
    - MXene films used as supercapacitor electrodes (stacked 2D channels)
    - Activated carbon, CNTs, and other high-surface-area porous carbons
    - Any EDLC electrode where the geometric capacitance is distributed

    Parameters
    ----------
    Rs      — external electrolyte resistance (Ω)
    Ri      — ionic pore resistance per unit pore length  (Ω·cm⁻¹ or Ω·m⁻¹
               depending on normalisation; keep consistent with L)
    Qdl     — CPE prefactor per unit pore length for the pore-wall DL
    n_dl    — CPE exponent for the pore-wall DL (0.8–1.0 for clean carbon)
    L       — effective pore length (same units as 1/Ri)
    """
    param_names = ["Rs", "Ri", "Qdl", "n_dl", "L"]
    bounds = (
        [1e-4, 1e-3, 1e-14, 0.60, 1e-4],
        [500.0, 1e5, 1e-3, 1.00, 1e2],
    )

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Ri, Qdl, n_dl, L = p
        # Admittance of pore wall CPE per unit length
        Ydl_per_L = Qdl * (1j * omega) ** n_dl
        # Characteristic impedance of the transmission line
        Zc = np.sqrt(Ri / (Ydl_per_L + 1e-30))
        # Propagation argument
        gamma_L = L * np.sqrt(Ri * Ydl_per_L + 1e-30)
        # De Levie: Z_TLM = Zc * coth(γL)
        # coth(x) = cosh(x)/sinh(x); guard against overflow for large |x|
        with np.errstate(over="ignore", invalid="ignore"):
            coth_gL = np.where(
                np.abs(gamma_L.real) > 20,
                np.sign(gamma_L.real),  # limit coth(x)→1 for large x
                1.0 / np.tanh(gamma_L + 1e-30),
            )
        Z_tlm = Zc * coth_gL
        return Rs + Z_tlm

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        span = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, span * 0.1, 1e-8, 0.90, 1.0])

    return CircuitTemplate(
        name="De-Levie-TLM",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs + sqrt(Ri/Ydl)·coth(L·sqrt(Ri·Ydl))",
        description=(
            "Full De Levie transmission-line model for a porous electrode "
            "with distributed ionic resistance and double-layer CPE admittance "
            "along the pore walls.  Predicts a 45° high-frequency Nyquist line "
            "transitioning to a near-vertical capacitive response at low ω.  "
            "Preferred over simplified TLM variants when the porous structure "
            "and pore length are well-defined (e.g., MXene films, activated carbon, "
            "CNT electrodes).  "
            "Ref: De Levie (1963) Electrochim. Acta 9:1231; "
            "Tribollet & Orazem (2022) §15."
        ),
        physical_meaning={
            "Rs": "External electrolyte / contact resistance (Ω)",
            "Ri": "Ionic resistance per unit pore length (Ω/length); "
            "high Ri → narrow or long pores",
            "Qdl": "CPE prefactor of pore-wall double-layer per unit length",
            "n_dl": "CPE exponent (0.8–1.0); n=1 → ideal capacitive pore walls",
            "L": "Effective pore depth (same length unit as 1/Ri)",
        },
        typical_systems=[
            "Ti₃C₂Tₓ MXene film electrodes (stacked 2D channels)",
            "Activated carbon EDLC electrodes in KOH / H₂SO₄",
            "Vertically aligned CNT arrays",
            "Graphene aerogel supercapacitors",
            "Ni(OH)₂ / Co₃O₄ porous battery-type electrodes",
        ],
    )


# ---------- 15. Pseudo-Capacitance-CPE --------------------------------
def _make_pseudo_capacitance() -> CircuitTemplate:
    """EIS model for combined double-layer + adsorption pseudocapacitance.

    Circuit topology
    ~~~~~~~~~~~~~~~~
    Rs − (Rct ‖ CPEdl) − (Rads ‖ Cads)

    Physical description
    ~~~~~~~~~~~~~~~~~~~~
    Two serial time-constant processes:
    1. **Rct ‖ CPEdl** (high–mid ω): outer-Helmholtz charge-transfer and
       double-layer capacitance.
    2. **Rads ‖ Cads** (low ω): adsorption/desorption step.  The ideal
       capacitor Cads (not CPE) captures adsorption as a surface coverage
       change at quasi-equilibrium — the Langmuir isotherm gives an
       ideal capacitive element in the linear limit.

    This circuit is the minimal model for **adsorption pseudocapacitance**,
    distinguished from intercalation by:
    - The adsorption time constant (Rads·Cads) is shorter than intercalation
    - No diffusion element is needed (unlike MXene-Intercalation)
    - Two arcs in Nyquist without a 45° tail at low ω

    Typical for:
    - Nb₂O₅ (proton / cation intercalation with surface-limited kinetics)
    - RuO₂ in H₂SO₄ (proton adsorption pseudocapacitance)
    - MnO₂ in Na₂SO₄ (surface redox)
    - Underpotential deposition (UPD) systems

    Parameters
    ----------
    Rs      — electrolyte resistance (Ω)
    Rct     — charge-transfer resistance (Ω)
    Qdl     — double-layer CPE prefactor (F·s^(n-1))
    n_dl    — double-layer CPE exponent (0.6–1.0)
    Rads    — adsorption resistance — kinetics of surface coverage (Ω)
    Cads    — adsorption pseudo-capacitance (F); ideal (n=1)
    """
    param_names = ["Rs", "Rct", "Qdl", "n_dl", "Rads", "Cads"]
    bounds = (
        [1e-3, 1e-2, 1e-13, 0.50, 1e-2, 1e-10],
        [500.0, 1e7, 1e-3, 1.00, 1e7, 1e-1],
    )

    def model(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rs, Rct, Qdl, n_dl, Rads, Cads = p
        # Charge-transfer + DL
        Zct = 1.0 / (1.0 / Rct + 1.0 / _cpe(omega, Qdl, n_dl))
        # Adsorption branch: Rads ‖ Cads (ideal capacitor)
        Zads = 1.0 / (1.0 / Rads + 1j * omega * Cads)
        return Rs + Zct + Zads

    def init(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        rs = _rs_from_z(z)
        span = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, span * 0.4, 1e-9, 0.85, span * 0.6, 1e-6])

    return CircuitTemplate(
        name="Pseudo-Capacitance-CPE",
        param_names=param_names,
        bounds=bounds,
        model_fn=model,
        init_fn=init,
        diagram="Rs − (Rct ‖ CPEdl) − (Rads ‖ Cads)",
        description=(
            "Charge-transfer + double-layer (CPE) in series with an adsorption "
            "pseudo-capacitance element (ideal RC branch).  Models surface-limited "
            "faradaic processes (adsorption, underpotential deposition, surface redox) "
            "where no semi-infinite diffusion tail is observed.  "
            "Distinguishable from Two-Arc-CPE by the ideal (n=1) capacitor Cads "
            "vs. a CPE second arc."
        ),
        physical_meaning={
            "Rs": "Electrolyte / uncompensated resistance (Ω)",
            "Rct": "Outer charge-transfer resistance (Ω)",
            "Qdl": "Double-layer CPE prefactor (F·s^(n-1))",
            "n_dl": "DL CPE exponent; n=1 → smooth surface",
            "Rads": "Adsorption kinetic resistance (Ω); large → slow adsorption",
            "Cads": "Adsorption pseudo-capacitance (F); proportional to "
            "∂θ/∂E at equilibrium surface coverage θ",
        },
        typical_systems=[
            "RuO₂ in H₂SO₄ — prototypical adsorption pseudocapacitance",
            "MnO₂ in Na₂SO₄ — surface cation redox",
            "Nb₂O₅ nanocrystals — surface-limited proton insertion",
            "Underpotential deposition of Cu, Pb, Bi on Au/Pt",
            "MXene in dilute acid — low-frequency adsorption arc",
        ],
    )


# ── Auto-register all built-in circuits at import time ───────────────

# ── EXT circuits (22 composite topologies) ───────────────────────────


def _make_ext_zarc_tlm() -> CircuitTemplate:
    """Rs − ZARC − TLM  [EXT-01]"""
    pn = ["Rs", "R", "Q", "n", "Ri", "Ydl", "nt"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-4, 1e-14, 0.3]
    ub = [1e4, 1e7, 1.0, 1.0, 1e5, 1e2, 1.0]

    def model(p, o):
        Rs, R, Q, n, Ri, Ydl, nt = p
        return Rs + _zarc_h(o, R, Q, n) + _tlm_h(o, Ri, Ydl, nt)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.5, 1e-9, 0.85, sp * 0.1, 1e-7, 0.50])

    return CircuitTemplate(
        name="Rs-ZARC-TLM",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC − TLM",
        description="Single ZARC arc in series with a semi-infinite De Levie TLM. "
        "Models a faradaic reaction at a porous electrode with distributed "
        "double-layer capacitance.",
        physical_meaning={
            "Rs": "Series resistance (Ω)",
            "R": "Arc resistance (Ω)",
            "Q": "CPE prefactor",
            "n": "CPE exponent",
            "Ri": "Pore ionic resistance",
            "Ydl": "DL admittance per length",
            "nt": "TLM CPE exponent",
        },
        typical_systems=[
            "Functionalised CNTs",
            "MXene/RGO composites",
            "Activated carbon with surface redox groups",
        ],
    )


def _make_ext_zarc2_wfinite() -> CircuitTemplate:
    """Rs − ZARC₁ − ZARC₂ − Wfinite  [EXT-02]"""
    pn = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "Rd", "Td"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-3, 1e-6]
    ub = [1e4, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0, 1e7, 1e5]

    def model(p, o):
        Rs, R1, Q1, n1, R2, Q2, n2, Rd, Td = p
        return Rs + _zarc_h(o, R1, Q1, n1) + _zarc_h(o, R2, Q2, n2) + _wfin_h(o, Rd, Td)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array(
            [rs, sp * 0.3, 1e-9, 0.85, sp * 0.5, 1e-10, 0.80, sp * 0.3, 0.5]
        )

    return CircuitTemplate(
        name="Rs-ZARC-ZARC-Wfinite",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − ZARC₂ − Wfinite",
        description="Two ZARC arcs followed by finite-length Warburg diffusion. "
        "Typical for Li-ion anodes with SEI + CT + limited solid diffusion.",
        physical_meaning={
            "Rs": "Electrolyte resistance",
            "R1": "HF arc resistance",
            "Q1": "HF CPE",
            "n1": "HF CPE exp",
            "R2": "LF arc resistance",
            "Q2": "LF CPE",
            "n2": "LF CPE exp",
            "Rd": "Diffusion resistance",
            "Td": "Diffusion time constant",
        },
        typical_systems=[
            "Li-ion graphite anodes",
            "Thick oxide electrodes",
            "Multi-layer battery cathodes",
        ],
    )


def _make_ext_zarc2_wshort() -> CircuitTemplate:
    """Rs − ZARC₁ − ZARC₂ − Wshort  [EXT-03]"""
    pn = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "Rd", "Td"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-3, 1e-6]
    ub = [1e4, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0, 1e7, 1e5]

    def model(p, o):
        Rs, R1, Q1, n1, R2, Q2, n2, Rd, Td = p
        return Rs + _zarc_h(o, R1, Q1, n1) + _zarc_h(o, R2, Q2, n2) + _wsho_h(o, Rd, Td)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array(
            [rs, sp * 0.3, 1e-9, 0.85, sp * 0.5, 1e-10, 0.80, sp * 0.3, 0.5]
        )

    return CircuitTemplate(
        name="Rs-ZARC-ZARC-Wshort",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − ZARC₂ − Wshort",
        description="Two ZARC arcs followed by reflective-boundary Warburg. "
        "Use when diffusion layer terminates at a blocking interface.",
        physical_meaning={
            "Rs": "Electrolyte resistance",
            "R1": "HF arc",
            "Q1": "HF CPE",
            "n1": "HF CPE exp",
            "R2": "LF arc",
            "Q2": "LF CPE",
            "n2": "LF CPE exp",
            "Rd": "Diffusion resistance",
            "Td": "Diffusion time",
        },
        typical_systems=[
            "Ion-exchange membranes",
            "Closed-pore electrodes",
            "Polymer electrolytes with blocking interfaces",
        ],
    )


def _make_ext_zarc2_gerischer() -> CircuitTemplate:
    """Rs − ZARC₁ − ZARC₂ − Gerischer  [EXT-04]"""
    pn = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "Rg", "Tg"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-3, 1e-6]
    ub = [1e4, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0, 1e7, 1e4]

    def model(p, o):
        Rs, R1, Q1, n1, R2, Q2, n2, Rg, Tg = p
        return Rs + _zarc_h(o, R1, Q1, n1) + _zarc_h(o, R2, Q2, n2) + _ger_h(o, Rg, Tg)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array(
            [rs, sp * 0.2, 1e-9, 0.85, sp * 0.3, 1e-10, 0.80, sp * 0.3, 1e-3]
        )

    return CircuitTemplate(
        name="Rs-ZARC-ZARC-Gerischer",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − ZARC₂ − Gerischer",
        description="Two ZARC arcs plus Gerischer distributed reaction element. "
        "SOFC cathodes with two distinct interface processes.",
        physical_meaning={
            "Rs": "Electrolyte resistance",
            "R1": "HF arc",
            "Q1": "HF CPE",
            "n1": "HF CPE exp",
            "R2": "MF arc",
            "Q2": "MF CPE",
            "n2": "MF CPE exp",
            "Rg": "Gerischer resistance",
            "Tg": "Gerischer time constant",
        },
        typical_systems=[
            "SOFC cathodes with bilayer microstructure",
            "ORR catalysts with coupled chemical steps",
        ],
    )


def _make_ext_zarc2_tlm() -> CircuitTemplate:
    """Rs − ZARC₁ − ZARC₂ − TLM  [EXT-05]"""
    pn = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "Ri", "Ydl", "nt"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-4, 1e-14, 0.3]
    ub = [1e4, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0, 1e5, 1e2, 1.0]

    def model(p, o):
        Rs, R1, Q1, n1, R2, Q2, n2, Ri, Ydl, nt = p
        return (
            Rs
            + _zarc_h(o, R1, Q1, n1)
            + _zarc_h(o, R2, Q2, n2)
            + _tlm_h(o, Ri, Ydl, nt)
        )

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array(
            [rs, sp * 0.2, 1e-9, 0.85, sp * 0.3, 1e-10, 0.80, sp * 0.1, 1e-7, 0.5]
        )

    return CircuitTemplate(
        name="Rs-ZARC-ZARC-TLM",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − ZARC₂ − TLM",
        description="Two interface arcs plus semi-infinite TLM. Composite electrodes "
        "with two distinct interfaces plus distributed porous capacitance.",
        physical_meaning={
            "Rs": "Series resistance",
            "R1": "HF arc",
            "Q1": "HF CPE",
            "n1": "HF CPE exp",
            "R2": "LF arc",
            "Q2": "LF CPE",
            "n2": "LF CPE exp",
            "Ri": "Pore ionic resistance",
            "Ydl": "DL admittance",
            "nt": "TLM exp",
        },
        typical_systems=["MXene/carbon composites", "Hierarchically porous electrodes"],
    )


def _make_ext_rc_zarc_w() -> CircuitTemplate:
    """Rs − (R₁‖C₁) − ZARC₂ − W  [EXT-06]"""
    pn = ["Rs", "R1", "C1", "R2", "Q2", "n2", "Sigma"]
    lb = [1e-6, 1e-3, 1e-14, 1e-3, 1e-14, 0.3, 1e-10]
    ub = [1e4, 1e7, 1.0, 1e7, 1.0, 1.0, 1e5]

    def model(p, o):
        Rs, R1, C1, R2, Q2, n2, sigma = p
        return Rs + _rc_par_h(o, R1, C1) + _zarc_h(o, R2, Q2, n2) + _warburg(o, sigma)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.3, 1e-9, sp * 0.5, 1e-9, 0.85, 0.01])

    return CircuitTemplate(
        name="Rs-RC-ZARC-W",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − (R₁‖C₁) − ZARC₂ − W",
        description="Ideal dielectric coating + CPE charge-transfer arc + Warburg. "
        "Uniform films over corrosion-active metals.",
        physical_meaning={
            "Rs": "Electrolyte resistance",
            "R1": "Coating resistance",
            "C1": "Coating dielectric capacitance",
            "R2": "Charge-transfer resistance",
            "Q2": "DL CPE",
            "n2": "DL CPE exp",
            "Sigma": "Warburg coefficient",
        },
        typical_systems=[
            "Uniform oxide coatings on steel",
            "Anodised aluminium with barrier layer",
        ],
    )


def _make_ext_zarc_rc_wfinite() -> CircuitTemplate:
    """Rs − ZARC₁ − (R₂‖C₂) − Wfinite  [EXT-07]"""
    pn = ["Rs", "R1", "Q1", "n1", "R2", "C2", "Rd", "Td"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 1e-3, 1e-6]
    ub = [1e4, 1e7, 1.0, 1.0, 1e7, 1.0, 1e7, 1e5]

    def model(p, o):
        Rs, R1, Q1, n1, R2, C2, Rd, Td = p
        return Rs + _zarc_h(o, R1, Q1, n1) + _rc_par_h(o, R2, C2) + _wfin_h(o, Rd, Td)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.3, 1e-9, 0.85, sp * 0.4, 1e-7, sp * 0.3, 0.5])

    return CircuitTemplate(
        name="Rs-ZARC-RC-Wfinite",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − (R₂‖C₂) − Wfinite",
        description="CPE arc + ideal adsorption RC + finite diffusion. "
        "Surface kinetics with Langmuir adsorption and bounded diffusion.",
        physical_meaning={
            "Rs": "Series resistance",
            "R1": "CT resistance",
            "Q1": "DL CPE",
            "n1": "CPE exp",
            "R2": "Adsorption resistance",
            "C2": "Adsorption capacitance",
            "Rd": "Diffusion resistance",
            "Td": "Diffusion time",
        },
        typical_systems=[
            "Nb₂O₅ with adsorption + diffusion",
            "MXene at low frequencies",
        ],
    )


def _make_ext_l_zarc_w() -> CircuitTemplate:
    """Rs − L − ZARC − W  [EXT-08]"""
    pn = ["Rs", "L", "R", "Q", "n", "Sigma"]
    lb = [1e-6, 1e-12, 1e-3, 1e-14, 0.3, 1e-10]
    ub = [1e4, 1e-1, 1e7, 1.0, 1.0, 1e5]

    def model(p, o):
        Rs, L, R, Q, n, sigma = p
        return Rs + _inductor(o, L) + _zarc_h(o, R, Q, n) + _warburg(o, sigma)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, 1e-6, sp * 0.5, 1e-9, 0.85, 0.01])

    return CircuitTemplate(
        name="Rs-L-ZARC-W",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − L − ZARC − W",
        description="Inductive loop + Randles-CPE-W. Inductive cable artefact or "
        "adsorbed intermediate combined with semi-infinite diffusion.",
        physical_meaning={
            "Rs": "Series resistance",
            "L": "Inductance",
            "R": "Charge-transfer resistance",
            "Q": "DL CPE",
            "n": "CPE exp",
            "Sigma": "Warburg coefficient",
        },
        typical_systems=[
            "PEM fuel cells with cable inductance",
            "Corrosion with inductive loop + diffusion",
        ],
    )


def _make_ext_l_zarc_wfinite() -> CircuitTemplate:
    """Rs − L − ZARC − Wfinite  [EXT-09]"""
    pn = ["Rs", "L", "R", "Q", "n", "Rd", "Td"]
    lb = [1e-6, 1e-12, 1e-3, 1e-14, 0.3, 1e-3, 1e-6]
    ub = [1e4, 1e-1, 1e7, 1.0, 1.0, 1e7, 1e5]

    def model(p, o):
        Rs, L, R, Q, n, Rd, Td = p
        return Rs + _inductor(o, L) + _zarc_h(o, R, Q, n) + _wfin_h(o, Rd, Td)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, 1e-6, sp * 0.5, 1e-9, 0.85, sp * 0.3, 0.5])

    return CircuitTemplate(
        name="Rs-L-ZARC-Wfinite",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − L − ZARC − Wfinite",
        description="Inductive loop + ZARC + bounded diffusion. Thin-film systems "
        "where diffusion layer is limited and measurement cables add inductance.",
        physical_meaning={
            "Rs": "Series resistance",
            "L": "Inductance",
            "R": "Charge-transfer resistance",
            "Q": "DL CPE",
            "n": "CPE exp",
            "Rd": "Diffusion resistance",
            "Td": "Diffusion time",
        },
        typical_systems=[
            "Thin-film batteries with inductive measurement artefact",
            "PEM cells with finite diffusion",
        ],
    )


def _make_ext_l_zarc2() -> CircuitTemplate:
    """Rs − L − ZARC₁ − ZARC₂  [EXT-10]"""
    pn = ["Rs", "L", "R1", "Q1", "n1", "R2", "Q2", "n2"]
    lb = [1e-6, 1e-12, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3]
    ub = [1e4, 1e-1, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0]

    def model(p, o):
        Rs, L, R1, Q1, n1, R2, Q2, n2 = p
        return Rs + _inductor(o, L) + _zarc_h(o, R1, Q1, n1) + _zarc_h(o, R2, Q2, n2)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, 1e-6, sp * 0.3, 1e-9, 0.85, sp * 0.5, 1e-10, 0.80])

    return CircuitTemplate(
        name="Rs-L-ZARC-ZARC",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − L − ZARC₁ − ZARC₂",
        description="Inductive loop + two ZARC arcs. PEM fuel cells with anode "
        "and cathode arcs plus high-frequency inductance.",
        physical_meaning={
            "Rs": "Series resistance",
            "L": "Inductance",
            "R1": "HF arc resistance",
            "Q1": "HF CPE",
            "n1": "HF exp",
            "R2": "LF arc resistance",
            "Q2": "LF CPE",
            "n2": "LF exp",
        },
        typical_systems=["PEM fuel cells", "Flow batteries with two electrode arcs"],
    )


def _make_ext_zarc3_w() -> CircuitTemplate:
    """Rs − ZARC₁ − ZARC₂ − ZARC₃ − W  [EXT-11]"""
    pn = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "R3", "Q3", "n3", "Sigma"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-10]
    ub = [1e4, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0, 1e8, 1.0, 1.0, 1e5]

    def model(p, o):
        Rs, R1, Q1, n1, R2, Q2, n2, R3, Q3, n3, sigma = p
        return (
            Rs
            + _zarc_h(o, R1, Q1, n1)
            + _zarc_h(o, R2, Q2, n2)
            + _zarc_h(o, R3, Q3, n3)
            + _warburg(o, sigma)
        )

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array(
            [
                rs,
                sp * 0.1,
                1e-12,
                0.95,
                sp * 0.2,
                1e-10,
                0.85,
                sp * 0.3,
                1e-8,
                0.75,
                0.01,
            ]
        )

    return CircuitTemplate(
        name="Rs-ZARC-ZARC-ZARC-W",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − ZARC₂ − ZARC₃ − W",
        description="Three ZARC arcs plus semi-infinite Warburg. Solid electrolytes "
        "with bulk, grain-boundary, and electrode arcs plus ionic diffusion.",
        physical_meaning={
            "Rs": "Geometric resistance",
            "R1": "Bulk arc resistance",
            "Q1": "Bulk arc CPE prefactor",
            "n1": "Bulk arc CPE exponent",
            "R2": "Grain-boundary arc resistance",
            "Q2": "GB arc CPE prefactor",
            "n2": "GB arc CPE exponent",
            "R3": "Electrode arc resistance",
            "Q3": "Electrode arc CPE prefactor",
            "n3": "Electrode arc CPE exponent",
            "Sigma": "Warburg coefficient",
        },
        typical_systems=[
            "Polycrystalline LLZO/NASICON with porous electrode",
            "YSZ pellets in SOFC stack",
        ],
    )


def _make_ext_zarc3_wfinite() -> CircuitTemplate:
    """Rs − ZARC₁ − ZARC₂ − ZARC₃ − Wfinite  [EXT-12]"""
    pn = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "R3", "Q3", "n3", "Rd", "Td"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-3, 1e-6]
    ub = [1e4, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0, 1e8, 1.0, 1.0, 1e7, 1e5]

    def model(p, o):
        Rs, R1, Q1, n1, R2, Q2, n2, R3, Q3, n3, Rd, Td = p
        return (
            Rs
            + _zarc_h(o, R1, Q1, n1)
            + _zarc_h(o, R2, Q2, n2)
            + _zarc_h(o, R3, Q3, n3)
            + _wfin_h(o, Rd, Td)
        )

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array(
            [
                rs,
                sp * 0.1,
                1e-12,
                0.95,
                sp * 0.2,
                1e-10,
                0.85,
                sp * 0.3,
                1e-8,
                0.75,
                sp * 0.3,
                0.5,
            ]
        )

    return CircuitTemplate(
        name="Rs-ZARC-ZARC-ZARC-Wfinite",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − ZARC₂ − ZARC₃ − Wfinite",
        description="Three ZARC arcs plus finite-length Warburg. Full-cell solid "
        "electrolyte with three bulk/grain/electrode processes and bounded diffusion.",
        physical_meaning={
            "Rs": "Geometric resistance",
            "R1": "Bulk arc resistance",
            "Q1": "Bulk arc CPE prefactor",
            "n1": "Bulk arc CPE exponent",
            "R2": "Grain-boundary arc resistance",
            "Q2": "GB arc CPE prefactor",
            "n2": "GB arc CPE exponent",
            "R3": "Electrode arc resistance",
            "Q3": "Electrode arc CPE prefactor",
            "n3": "Electrode arc CPE exponent",
            "Rd": "Diffusion resistance",
            "Td": "Diffusion time constant",
        },
        typical_systems=[
            "All-solid-state battery cells (full-cell)",
            "Garnet electrolyte with thick porous electrode",
        ],
    )


def _make_ext_zarc_cpe() -> CircuitTemplate:
    """Rs − ZARC − CPE  [EXT-13]"""
    pn = ["Rs", "R", "Q1", "n1", "Q2", "n2"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-14, 0.3]
    ub = [1e4, 1e7, 1.0, 1.0, 1.0, 1.0]

    def model(p, o):
        Rs, R, Q1, n1, Q2, n2 = p
        return Rs + _zarc_h(o, R, Q1, n1) + _cpe(o, Q2, n2)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.5, 1e-9, 0.85, 1e-4, 0.95])

    return CircuitTemplate(
        name="Rs-ZARC-CPE",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC − CPE",
        description="ZARC arc followed by a blocking CPE. Models a charge-transfer "
        "process with a partially blocking passivation film at low frequencies.",
        physical_meaning={
            "Rs": "Series resistance",
            "R": "Arc resistance",
            "Q1": "Arc CPE",
            "n1": "Arc CPE exp",
            "Q2": "Blocking CPE",
            "n2": "Blocking CPE exp (close to 1)",
        },
        typical_systems=[
            "Metals with thick passive films",
            "Partially blocked electrodes",
            "SEI-covered electrodes at low SoC",
        ],
    )


def _make_ext_rc_w() -> CircuitTemplate:
    """Rs − (R‖C) − W  [EXT-14]"""
    pn = ["Rs", "R", "C", "Sigma"]
    lb = [1e-6, 1e-3, 1e-14, 1e-10]
    ub = [1e4, 1e7, 1.0, 1e5]

    def model(p, o):
        Rs, R, C, sigma = p
        return Rs + _rc_par_h(o, R, C) + _warburg(o, sigma)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.6, 1e-6, 0.01])

    return CircuitTemplate(
        name="Rs-RC-W",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − (R‖C) − W",
        description="Ideal RC arc plus semi-infinite Warburg. Parsimonious Randles "
        "variant for smooth electrodes where a CPE exponent is not justified.",
        physical_meaning={
            "Rs": "Electrolyte resistance",
            "R": "CT resistance",
            "C": "Double-layer capacitance",
            "Sigma": "Warburg coefficient",
        },
        typical_systems=[
            "Smooth Pt / Au in aqueous electrolyte",
            "Model systems for extracting diffusion coefficients",
        ],
    )


def _make_ext_rc_wfinite() -> CircuitTemplate:
    """Rs − (R‖C) − Wfinite  [EXT-15]"""
    pn = ["Rs", "R", "C", "Rd", "Td"]
    lb = [1e-6, 1e-3, 1e-14, 1e-3, 1e-6]
    ub = [1e4, 1e7, 1.0, 1e7, 1e5]

    def model(p, o):
        Rs, R, C, Rd, Td = p
        return Rs + _rc_par_h(o, R, C) + _wfin_h(o, Rd, Td)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.5, 1e-6, sp * 0.3, 0.5])

    return CircuitTemplate(
        name="Rs-RC-Wfinite",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − (R‖C) − Wfinite",
        description="Ideal RC arc plus finite Warburg. Parsimonious thin-film model "
        "for smooth electrodes with a well-defined diffusion layer.",
        physical_meaning={
            "Rs": "Electrolyte resistance",
            "R": "CT resistance",
            "C": "Double-layer capacitance",
            "Rd": "Diffusion resistance",
            "Td": "Diffusion time constant",
        },
        typical_systems=[
            "Smooth thin-film electrodes",
            "Well-characterized model systems",
        ],
    )


def _make_ext_zarc2_cpe() -> CircuitTemplate:
    """Rs − ZARC₁ − ZARC₂ − CPE  [EXT-16]"""
    pn = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "Q3", "n3"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-14, 0.3]
    ub = [1e4, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0, 1.0, 1.0]

    def model(p, o):
        Rs, R1, Q1, n1, R2, Q2, n2, Q3, n3 = p
        return Rs + _zarc_h(o, R1, Q1, n1) + _zarc_h(o, R2, Q2, n2) + _cpe(o, Q3, n3)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.3, 1e-9, 0.85, sp * 0.4, 1e-10, 0.80, 1e-4, 0.95])

    return CircuitTemplate(
        name="Rs-ZARC-ZARC-CPE",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − ZARC₂ − CPE",
        description="Two ZARC arcs plus blocking CPE. Two kinetic processes with "
        "a very resistive passivation layer blocking DC current.",
        physical_meaning={
            "Rs": "Series resistance",
            "R1": "HF arc resistance",
            "Q1": "HF arc CPE prefactor",
            "n1": "HF arc CPE exponent",
            "R2": "LF arc resistance",
            "Q2": "LF arc CPE prefactor",
            "n2": "LF arc CPE exponent",
            "Q3": "Blocking CPE prefactor",
            "n3": "Blocking CPE exponent",
        },
        typical_systems=["Thick SEI-covered anodes", "Heavily passivated metals"],
    )


def _make_ext_rc_zarc_wfinite() -> CircuitTemplate:
    """Rs − (R₁‖C₁) − ZARC₂ − Wfinite  [EXT-17]"""
    pn = ["Rs", "R1", "C1", "R2", "Q2", "n2", "Rd", "Td"]
    lb = [1e-6, 1e-3, 1e-14, 1e-3, 1e-14, 0.3, 1e-3, 1e-6]
    ub = [1e4, 1e7, 1.0, 1e7, 1.0, 1.0, 1e7, 1e5]

    def model(p, o):
        Rs, R1, C1, R2, Q2, n2, Rd, Td = p
        return Rs + _rc_par_h(o, R1, C1) + _zarc_h(o, R2, Q2, n2) + _wfin_h(o, Rd, Td)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.3, 1e-9, sp * 0.4, 1e-9, 0.85, sp * 0.3, 0.5])

    return CircuitTemplate(
        name="Rs-RC-ZARC-Wfinite",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − (R₁‖C₁) − ZARC₂ − Wfinite",
        description="Ideal coating + CPE charge-transfer arc + finite diffusion. "
        "Protective coatings on intercalation electrodes.",
        physical_meaning={
            "Rs": "Electrolyte resistance",
            "R1": "Coating resistance",
            "C1": "Coating capacitance",
            "R2": "CT resistance",
            "Q2": "DL CPE",
            "n2": "DL CPE exp",
            "Rd": "Diffusion resistance",
            "Td": "Diffusion time",
        },
        typical_systems=[
            "Coated Li-ion cathodes",
            "Conversion electrodes with protective layer",
        ],
    )


def _make_ext_zarc_rc_wshort() -> CircuitTemplate:
    """Rs − ZARC₁ − (R₂‖C₂) − Wshort  [EXT-18]"""
    pn = ["Rs", "R1", "Q1", "n1", "R2", "C2", "Rd", "Td"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 1e-3, 1e-6]
    ub = [1e4, 1e7, 1.0, 1.0, 1e7, 1.0, 1e7, 1e5]

    def model(p, o):
        Rs, R1, Q1, n1, R2, C2, Rd, Td = p
        return Rs + _zarc_h(o, R1, Q1, n1) + _rc_par_h(o, R2, C2) + _wsho_h(o, Rd, Td)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.3, 1e-9, 0.85, sp * 0.4, 1e-7, sp * 0.3, 0.5])

    return CircuitTemplate(
        name="Rs-ZARC-RC-Wshort",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − (R₂‖C₂) − Wshort",
        description="CPE arc + ideal adsorption RC + reflective Warburg. "
        "Systems with surface intermediates and a closed diffusion boundary.",
        physical_meaning={
            "Rs": "Series resistance",
            "R1": "CT resistance",
            "Q1": "DL CPE",
            "n1": "CPE exp",
            "R2": "Adsorption resistance",
            "C2": "Adsorption capacitance",
            "Rd": "Diffusion resistance",
            "Td": "Diffusion time",
        },
        typical_systems=[
            "Ion-selective membranes with adsorption",
            "Polymer electrolytes with surface reaction",
        ],
    )


def _make_ext_l_zarc2_w() -> CircuitTemplate:
    """Rs − L − ZARC₁ − ZARC₂ − W  [EXT-19]"""
    pn = ["Rs", "L", "R1", "Q1", "n1", "R2", "Q2", "n2", "Sigma"]
    lb = [1e-6, 1e-12, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-10]
    ub = [1e4, 1e-1, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0, 1e5]

    def model(p, o):
        Rs, L, R1, Q1, n1, R2, Q2, n2, sigma = p
        return (
            Rs
            + _inductor(o, L)
            + _zarc_h(o, R1, Q1, n1)
            + _zarc_h(o, R2, Q2, n2)
            + _warburg(o, sigma)
        )

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, 1e-6, sp * 0.2, 1e-9, 0.85, sp * 0.4, 1e-10, 0.80, 0.01])

    return CircuitTemplate(
        name="Rs-L-ZARC-ZARC-W",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − L − ZARC₁ − ZARC₂ − W",
        description="Inductive loop + two ZARC arcs + semi-infinite Warburg. "
        "PEM fuel cells with two electrode processes and measurement inductance.",
        physical_meaning={
            "Rs": "Series resistance",
            "L": "Inductance",
            "R1": "HF arc resistance",
            "Q1": "HF arc CPE prefactor",
            "n1": "HF arc CPE exponent",
            "R2": "LF arc resistance",
            "Q2": "LF arc CPE prefactor",
            "n2": "LF arc CPE exponent",
            "Sigma": "Warburg coefficient",
        },
        typical_systems=[
            "PEM fuel cells with diffusion artefact",
            "Flow batteries with two arcs and mass transport",
        ],
    )


def _make_ext_tlm() -> CircuitTemplate:
    """Rs − TLM (pure De Levie)  [EXT-20]"""
    pn = ["Rs", "Ri", "Ydl", "nt"]
    lb = [1e-6, 1e-4, 1e-14, 0.3]
    ub = [1e4, 1e5, 1e2, 1.0]

    def model(p, o):
        Rs, Ri, Ydl, nt = p
        return Rs + _tlm_h(o, Ri, Ydl, nt)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.1, 1e-7, 0.50])

    return CircuitTemplate(
        name="Rs-TLM",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − TLM",
        description="Pure De Levie TLM without a kinetic arc. Purely capacitive "
        "porous electrode with distributed ionic resistance.",
        physical_meaning={
            "Rs": "External resistance",
            "Ri": "Pore ionic resistance per unit length",
            "Ydl": "DL admittance per unit length",
            "nt": "Distributed CPE exponent (0.5 = ideal)",
        },
        typical_systems=[
            "Ideal EDLC porous carbon electrodes",
            "Pure double-layer supercapacitors without redox",
        ],
    )


def _make_ext_zarc3_gerischer() -> CircuitTemplate:
    """Rs − ZARC₁ − ZARC₂ − ZARC₃ − Gerischer  [EXT-21]"""
    pn = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "R3", "Q3", "n3", "Rg", "Tg"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-3, 1e-14, 0.3, 1e-3, 1e-6]
    ub = [1e4, 1e7, 1.0, 1.0, 1e7, 1.0, 1.0, 1e8, 1.0, 1.0, 1e7, 1e4]

    def model(p, o):
        Rs, R1, Q1, n1, R2, Q2, n2, R3, Q3, n3, Rg, Tg = p
        return (
            Rs
            + _zarc_h(o, R1, Q1, n1)
            + _zarc_h(o, R2, Q2, n2)
            + _zarc_h(o, R3, Q3, n3)
            + _ger_h(o, Rg, Tg)
        )

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array(
            [
                rs,
                sp * 0.1,
                1e-12,
                0.95,
                sp * 0.2,
                1e-10,
                0.85,
                sp * 0.3,
                1e-8,
                0.75,
                sp * 0.2,
                1e-3,
            ]
        )

    return CircuitTemplate(
        name="Rs-ZARC-ZARC-ZARC-Gerischer",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC₁ − ZARC₂ − ZARC₃ − Gerischer",
        description="Three ZARC arcs plus Gerischer element. Complex SOFC ceramics "
        "with bulk, grain-boundary, electrode and distributed reaction processes.",
        physical_meaning={
            "Rs": "Geometric resistance",
            "R1": "Bulk arc resistance",
            "Q1": "Bulk arc CPE prefactor",
            "n1": "Bulk arc CPE exponent",
            "R2": "Grain-boundary arc resistance",
            "Q2": "GB arc CPE prefactor",
            "n2": "GB arc CPE exponent",
            "R3": "Electrode arc resistance",
            "Q3": "Electrode arc CPE prefactor",
            "n3": "Electrode arc CPE exponent",
            "Rg": "Gerischer resistance",
            "Tg": "Gerischer time constant",
        },
        typical_systems=[
            "SOFC polycrystalline cathode ceramics",
            "MIEC with three distinct microstructural features",
        ],
    )


def _make_ext_zarc_tlm_w() -> CircuitTemplate:
    """Rs − ZARC − TLM − W  [EXT-22]"""
    pn = ["Rs", "R", "Q", "n", "Ri", "Ydl", "nt", "Sigma"]
    lb = [1e-6, 1e-3, 1e-14, 0.3, 1e-4, 1e-14, 0.3, 1e-10]
    ub = [1e4, 1e7, 1.0, 1.0, 1e5, 1e2, 1.0, 1e5]

    def model(p, o):
        Rs, R, Q, n, Ri, Ydl, nt, sigma = p
        return Rs + _zarc_h(o, R, Q, n) + _tlm_h(o, Ri, Ydl, nt) + _warburg(o, sigma)

    def init(o, z):
        rs = _rs_from_z(z)
        sp = float(max(z.real.max() - z.real.min(), 0.1))
        return np.array([rs, sp * 0.3, 1e-9, 0.85, sp * 0.1, 1e-7, 0.5, 0.01])

    return CircuitTemplate(
        name="Rs-ZARC-TLM-W",
        param_names=pn,
        bounds=(lb, ub),
        model_fn=model,
        init_fn=init,
        diagram="Rs − ZARC − TLM − W",
        description="ZARC arc + porous TLM + semi-infinite Warburg. Asymmetric "
        "supercapacitors with a faradaic electrode, porous carbon and bulk diffusion.",
        physical_meaning={
            "Rs": "Series resistance",
            "R": "Arc resistance",
            "Q": "DL CPE",
            "n": "CPE exp",
            "Ri": "Pore resistance",
            "Ydl": "DL admittance",
            "nt": "TLM exp",
            "Sigma": "Warburg coefficient",
        },
        typical_systems=[
            "Asymmetric supercapacitors",
            "Hybrid EDLC/battery electrodes with porous structure",
        ],
    )


# ── Auto-register all built-in circuits at import time ───────────────
_BUILTIN_MAKERS = [
    # Base circuits (11)
    _make_randles_cpe_w,
    _make_two_arc_cpe,
    _make_inductive_cpe,
    _make_coating_cpe,
    _make_warburg_finite,
    _make_zarc_zarc_w,
    _make_simple_rc,
    _make_cpe_simple,
    _make_warburg_short,
    _make_gerischer,
    _make_three_zarc,
    # Registry-specific circuits (4)
    _make_porous_coating_tlm,
    _make_mxene_intercalation,
    _make_de_levie_tlm,
    _make_pseudo_capacitance,
    # Extended circuits (22)
    _make_ext_zarc_tlm,
    _make_ext_zarc2_wfinite,
    _make_ext_zarc2_wshort,
    _make_ext_zarc2_gerischer,
    _make_ext_zarc2_tlm,
    _make_ext_rc_zarc_w,
    _make_ext_zarc_rc_wfinite,
    _make_ext_l_zarc_w,
    _make_ext_l_zarc_wfinite,
    _make_ext_l_zarc2,
    _make_ext_zarc3_w,
    _make_ext_zarc3_wfinite,
    _make_ext_zarc_cpe,
    _make_ext_rc_w,
    _make_ext_rc_wfinite,
    _make_ext_zarc2_cpe,
    _make_ext_rc_zarc_wfinite,
    _make_ext_zarc_rc_wshort,
    _make_ext_l_zarc2_w,
    _make_ext_tlm,
    _make_ext_zarc3_gerischer,
    _make_ext_zarc_tlm_w,
]


def _register_builtins() -> None:
    for maker in _BUILTIN_MAKERS:
        CircuitRegistry.register(maker())


_register_builtins()
