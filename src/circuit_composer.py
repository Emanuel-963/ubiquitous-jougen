"""Automatic circuit recombination — compose EIS circuits from building blocks.

This module generates equivalent-circuit candidates by combining fundamental
impedance blocks (R, C, CPE, W, W_finite, L, ZARC) in series, parallel, and
series-parallel topologies.  The ``auto_select`` method performs quick fitting
and returns the top candidates ranked by BIC.

Public API
----------
``CircuitBlock``
    Dataclass describing a single impedance element.
``CircuitComposer``
    Main composer: ``compose``, ``enumerate_candidates``, ``auto_select``.

Day 8 of the UPGRADE_PLAN_v0.2.0 schedule.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from src.circuit_fitting import (
    CircuitTemplate,
    _cpe,
    _warburg,
    _inductor,
    _bic_aic,
)

logger = logging.getLogger(__name__)

# ── Maximum candidates per enumeration run (timeout safety) ──────────
MAX_CANDIDATES = 50

# ── Fast-fit defaults ────────────────────────────────────────────────
_FAST_MAX_NFEV = 1000
_FAST_SEEDS = 1


# =====================================================================
# CircuitBlock — fundamental impedance element
# =====================================================================

@dataclass
class CircuitBlock:
    """A single impedance building block.

    Attributes
    ----------
    name : str
        Human-readable block name (e.g. ``"R"``, ``"CPE"``, ``"ZARC"``).
    impedance : Callable[[np.ndarray, np.ndarray], np.ndarray]
        ``impedance(params, omega) -> Z``.  *params* is a 1-D array of the
        block's own parameters.
    n_params : int
        Number of free parameters.
    param_names : list[str]
        Names for each parameter (length == ``n_params``).
    bounds : list[tuple[float, float]]
        ``[(lo, hi), ...]`` — one pair per parameter.
    """

    name: str
    impedance: Callable[[np.ndarray, np.ndarray], np.ndarray]
    n_params: int
    param_names: List[str]
    bounds: List[Tuple[float, float]]

    # Convenience: default initial value (geometric mean of bounds)
    def default_p0(self) -> np.ndarray:
        """Return a reasonable initial guess (geometric mean of bounds)."""
        p0 = []
        for lo, hi in self.bounds:
            if lo > 0 and hi > 0:
                p0.append(np.sqrt(lo * hi))
            else:
                p0.append((lo + hi) / 2.0)
        return np.array(p0)


# =====================================================================
# Built-in blocks
# =====================================================================

def _block_R() -> CircuitBlock:
    """Pure resistor R."""
    def z_func(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return np.full_like(omega, p[0], dtype=complex)
    return CircuitBlock(
        name="R",
        impedance=z_func,
        n_params=1,
        param_names=["R"],
        bounds=[(1e-6, 1e8)],
    )


def _block_C() -> CircuitBlock:
    """Ideal capacitor C."""
    def z_func(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return 1.0 / (1j * omega * p[0])
    return CircuitBlock(
        name="C",
        impedance=z_func,
        n_params=1,
        param_names=["C"],
        bounds=[(1e-15, 1e-1)],
    )


def _block_CPE() -> CircuitBlock:
    """Constant Phase Element (Q, n)."""
    def z_func(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return _cpe(omega, p[0], p[1])
    return CircuitBlock(
        name="CPE",
        impedance=z_func,
        n_params=2,
        param_names=["Q", "n"],
        bounds=[(1e-12, 1.0), (0.3, 1.0)],
    )


def _block_W() -> CircuitBlock:
    """Semi-infinite Warburg."""
    def z_func(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return _warburg(omega, p[0])
    return CircuitBlock(
        name="W",
        impedance=z_func,
        n_params=1,
        param_names=["Sigma"],
        bounds=[(1e-10, 1e5)],
    )


def _block_W_finite() -> CircuitBlock:
    """Finite-length Warburg (Rd, Td)."""
    def z_func(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        Rd, Td = p
        s = np.sqrt(1j * omega * Td)
        s_safe = np.where(np.abs(s) < 1e-30, 1e-30, s)
        return Rd * np.tanh(s_safe) / s_safe
    return CircuitBlock(
        name="W_finite",
        impedance=z_func,
        n_params=2,
        param_names=["Rd", "Td"],
        bounds=[(1e-6, 1e8), (1e-6, 1e4)],
    )


def _block_L() -> CircuitBlock:
    """Ideal inductor L."""
    def z_func(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return _inductor(omega, p[0])
    return CircuitBlock(
        name="L",
        impedance=z_func,
        n_params=1,
        param_names=["L"],
        bounds=[(1e-9, 1.0)],
    )


def _block_ZARC() -> CircuitBlock:
    """ZARC element: R || CPE (3 params: R, Q, n)."""
    def z_func(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        R, Q, n = p
        Zcpe = _cpe(omega, Q, n)
        return 1.0 / (1.0 / R + 1.0 / Zcpe)
    return CircuitBlock(
        name="ZARC",
        impedance=z_func,
        n_params=3,
        param_names=["R", "Q", "n"],
        bounds=[(1e-3, 1e8), (1e-12, 1.0), (0.3, 1.0)],
    )


# Canonical list of all built-in blocks
BUILTIN_BLOCKS: Dict[str, Callable[[], CircuitBlock]] = {
    "R": _block_R,
    "C": _block_C,
    "CPE": _block_CPE,
    "W": _block_W,
    "W_finite": _block_W_finite,
    "L": _block_L,
    "ZARC": _block_ZARC,
}


def get_builtin_blocks() -> List[CircuitBlock]:
    """Instantiate and return all built-in blocks."""
    return [maker() for maker in BUILTIN_BLOCKS.values()]


# =====================================================================
# Topology helpers — combine blocks into a composite impedance function
# =====================================================================

def _series_impedance(
    blocks: List[CircuitBlock],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a function ``Z(params, omega)`` for blocks in series."""
    slices = _param_slices(blocks)

    def z_func(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        z_total = np.zeros_like(omega, dtype=complex)
        for blk, sl in zip(blocks, slices):
            z_total += blk.impedance(p[sl], omega)
        return z_total

    return z_func


def _parallel_impedance(
    blocks: List[CircuitBlock],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a function ``Z(params, omega)`` for blocks in parallel."""
    slices = _param_slices(blocks)

    def z_func(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        y_total = np.zeros_like(omega, dtype=complex)
        for blk, sl in zip(blocks, slices):
            z_blk = blk.impedance(p[sl], omega)
            # Avoid division by zero
            z_safe = np.where(np.abs(z_blk) < 1e-30, 1e-30, z_blk)
            y_total += 1.0 / z_safe
        y_safe = np.where(np.abs(y_total) < 1e-30, 1e-30, y_total)
        return 1.0 / y_safe

    return z_func


def _series_parallel_impedance(
    series_block: CircuitBlock,
    parallel_blocks: List[CircuitBlock],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return ``Z = Z_series + (Z_par1 || Z_par2 || ...)``.

    The first block is in series; the remaining blocks are in parallel
    with each other.
    """
    all_blocks = [series_block] + parallel_blocks
    slices = _param_slices(all_blocks)

    def z_func(p: np.ndarray, omega: np.ndarray) -> np.ndarray:
        # Series element
        z_ser = series_block.impedance(p[slices[0]], omega)
        # Parallel group
        y_par = np.zeros_like(omega, dtype=complex)
        for blk, sl in zip(parallel_blocks, slices[1:]):
            z_blk = blk.impedance(p[sl], omega)
            z_safe = np.where(np.abs(z_blk) < 1e-30, 1e-30, z_blk)
            y_par += 1.0 / z_safe
        y_safe = np.where(np.abs(y_par) < 1e-30, 1e-30, y_par)
        return z_ser + 1.0 / y_safe

    return z_func


def _param_slices(blocks: List[CircuitBlock]) -> List[slice]:
    """Compute contiguous parameter slices for each block."""
    slices: List[slice] = []
    offset = 0
    for blk in blocks:
        slices.append(slice(offset, offset + blk.n_params))
        offset += blk.n_params
    return slices


# =====================================================================
# CircuitComposer
# =====================================================================

class CircuitComposer:
    """Compose EIS equivalent circuits from fundamental blocks.

    Parameters
    ----------
    blocks : list[CircuitBlock] | None
        Custom block library.  Defaults to all built-in blocks.

    Examples
    --------
    >>> composer = CircuitComposer()
    >>> tpl = composer.compose(["R", "ZARC", "W"], topology="series")
    >>> candidates = composer.enumerate_candidates(max_elements=3)
    >>> top5 = composer.auto_select(freq, z, max_elements=3, top_n=5)
    """

    def __init__(self, blocks: Optional[List[CircuitBlock]] = None) -> None:
        if blocks is None:
            blocks = get_builtin_blocks()
        self._blocks_by_name: Dict[str, CircuitBlock] = {b.name: b for b in blocks}
        self._blocks = list(blocks)

    # ── Public properties ────────────────────────────────────────
    @property
    def block_names(self) -> List[str]:
        """Names of available blocks."""
        return list(self._blocks_by_name.keys())

    @property
    def blocks(self) -> List[CircuitBlock]:
        """All available blocks."""
        return list(self._blocks)

    # ── compose ──────────────────────────────────────────────────

    def compose(
        self,
        block_names: List[str],
        topology: str = "series",
    ) -> CircuitTemplate:
        """Create a :class:`CircuitTemplate` from named blocks.

        Parameters
        ----------
        block_names : list[str]
            Ordered list of block names (must exist in the library).
        topology : str
            ``"series"`` — all blocks in series.
            ``"parallel"`` — all blocks in parallel.
            ``"series-parallel"`` — first block in series, rest in parallel.

        Returns
        -------
        CircuitTemplate
            Ready to use with :func:`fit_template`.

        Raises
        ------
        ValueError
            If a block name is unknown or topology is invalid.
        """
        resolved = self._resolve_blocks(block_names)

        if topology == "series":
            model_fn = _series_impedance(resolved)
            diagram = " − ".join(b.name for b in resolved)
        elif topology == "parallel":
            model_fn = _parallel_impedance(resolved)
            diagram = " ‖ ".join(b.name for b in resolved)
        elif topology == "series-parallel":
            if len(resolved) < 2:
                raise ValueError(
                    "series-parallel topology requires at least 2 blocks"
                )
            model_fn = _series_parallel_impedance(resolved[0], resolved[1:])
            par_part = " ‖ ".join(b.name for b in resolved[1:])
            diagram = f"{resolved[0].name} − ({par_part})"
        else:
            raise ValueError(
                f"Unknown topology '{topology}'. "
                "Choose 'series', 'parallel', or 'series-parallel'."
            )

        # Aggregate parameter metadata
        param_names: List[str] = []
        lower: List[float] = []
        upper: List[float] = []
        seen: Dict[str, int] = {}
        for blk in resolved:
            for pname, (lo, hi) in zip(blk.param_names, blk.bounds):
                # Disambiguate repeated param names with a suffix
                key = f"{blk.name}_{pname}"
                count = seen.get(key, 0)
                seen[key] = count + 1
                suffix = f"_{count + 1}" if count > 0 else ""
                param_names.append(f"{blk.name}_{pname}{suffix}")
                lower.append(lo)
                upper.append(hi)

        bounds = (lower, upper)

        # Generic init: geometric mean of bounds
        def init_fn(omega: np.ndarray, z: np.ndarray) -> np.ndarray:
            p0 = []
            for lo, hi in zip(lower, upper):
                if lo > 0 and hi > 0:
                    p0.append(np.sqrt(lo * hi))
                else:
                    p0.append((lo + hi) / 2.0)
            return np.array(p0)

        name = f"Composed_{'_'.join(block_names)}_{topology}"

        return CircuitTemplate(
            name=name,
            param_names=param_names,
            bounds=bounds,
            model_fn=model_fn,
            init_fn=init_fn,
            diagram=diagram,
        )

    # ── enumerate_candidates ─────────────────────────────────────

    def enumerate_candidates(
        self,
        max_elements: int = 4,
        *,
        topologies: Optional[Sequence[str]] = None,
        must_include: Optional[List[str]] = None,
    ) -> List[CircuitTemplate]:
        """Generate circuit candidates up to *max_elements* blocks.

        Parameters
        ----------
        max_elements : int
            Maximum number of blocks per candidate (1 … max_elements).
        topologies : list[str] | None
            Topologies to try.  Defaults to
            ``["series", "parallel", "series-parallel"]``.
        must_include : list[str] | None
            Block names that *must* appear in every candidate (e.g.
            ``["R"]`` to ensure an ohmic resistor is always present).

        Returns
        -------
        list[CircuitTemplate]
            At most :data:`MAX_CANDIDATES` unique templates.
        """
        if topologies is None:
            topologies = ["series", "parallel", "series-parallel"]

        block_names = self.block_names
        candidates: List[CircuitTemplate] = []
        seen_names: set = set()

        for n_elem in range(1, max_elements + 1):
            # Combinations with replacement (order doesn't matter for
            # series/parallel — we deduplicate via canonical name).
            for combo in itertools.combinations_with_replacement(block_names, n_elem):
                # must_include filter
                if must_include:
                    if not all(m in combo for m in must_include):
                        continue

                for topo in topologies:
                    # Skip parallel / series-parallel with 1 element
                    if n_elem == 1 and topo != "series":
                        continue
                    # series-parallel needs ≥ 2
                    if topo == "series-parallel" and n_elem < 2:
                        continue

                    try:
                        tpl = self.compose(list(combo), topology=topo)
                    except ValueError:
                        continue

                    if tpl.name not in seen_names:
                        seen_names.add(tpl.name)
                        candidates.append(tpl)

                    if len(candidates) >= MAX_CANDIDATES:
                        logger.info(
                            "enumerate_candidates: reached MAX_CANDIDATES=%d",
                            MAX_CANDIDATES,
                        )
                        return candidates

        logger.info(
            "enumerate_candidates: generated %d candidates (max_elements=%d)",
            len(candidates),
            max_elements,
        )
        return candidates

    # ── auto_select ──────────────────────────────────────────────

    def auto_select(
        self,
        freq: np.ndarray,
        z: np.ndarray,
        *,
        max_elements: int = 3,
        top_n: int = 5,
        max_nfev: int = _FAST_MAX_NFEV,
        must_include: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Generate candidates, fit each quickly, return top-N by BIC.

        Parameters
        ----------
        freq : np.ndarray
            Frequency vector (Hz).
        z : np.ndarray
            Complex impedance vector.
        max_elements : int
            Max blocks per candidate.
        top_n : int
            Number of best candidates to return.
        max_nfev : int
            Maximum function evaluations per fit (fast mode).
        must_include : list[str] | None
            Blocks that must appear in every candidate.

        Returns
        -------
        list[dict]
            Top-N results, each with keys: ``template``, ``diagram``,
            ``params``, ``rss``, ``bic``, ``aic``, ``n_params``,
            ``n_points``, ``success``.
        """
        candidates = self.enumerate_candidates(
            max_elements=max_elements,
            must_include=must_include,
        )

        if not candidates:
            logger.warning("auto_select: no candidates generated.")
            return []

        omega = 2.0 * np.pi * freq
        results: List[Dict] = []

        for tpl in candidates:
            try:
                result = self._fast_fit(tpl, freq, z, omega, max_nfev=max_nfev)
                results.append(result)
            except Exception as exc:
                logger.debug(
                    "auto_select: fit failed for %s — %s", tpl.name, exc
                )
                results.append({
                    "template": tpl.name,
                    "diagram": tpl.diagram,
                    "params": {},
                    "rss": np.inf,
                    "bic": np.inf,
                    "aic": np.inf,
                    "n_params": len(tpl.param_names),
                    "n_points": len(freq),
                    "success": False,
                })

        # Sort by BIC (lower = better)
        results.sort(key=lambda r: (r.get("bic", np.inf), r.get("rss", np.inf)))
        top = results[:top_n]

        logger.info(
            "auto_select: evaluated %d candidates, top-%d returned. "
            "Best: %s (BIC=%.2f)",
            len(results),
            min(top_n, len(results)),
            top[0]["template"] if top else "—",
            top[0].get("bic", np.inf) if top else np.inf,
        )
        return top

    # ── Private helpers ──────────────────────────────────────────

    def _resolve_blocks(self, names: List[str]) -> List[CircuitBlock]:
        """Look up block names → CircuitBlock list.  Raises on unknown."""
        resolved: List[CircuitBlock] = []
        for n in names:
            if n not in self._blocks_by_name:
                raise ValueError(
                    f"Unknown block '{n}'. "
                    f"Available: {list(self._blocks_by_name.keys())}"
                )
            resolved.append(self._blocks_by_name[n])
        return resolved

    @staticmethod
    def _fast_fit(
        tpl: CircuitTemplate,
        freq: np.ndarray,
        z: np.ndarray,
        omega: np.ndarray,
        *,
        max_nfev: int = _FAST_MAX_NFEV,
    ) -> Dict:
        """Quick single-seed fit (for screening, not final results)."""
        p0 = tpl.init_fn(omega, z)
        lb = np.array(tpl.bounds[0], dtype=float)
        ub = np.array(tpl.bounds[1], dtype=float)

        # Clip p0 to bounds
        p0 = np.clip(p0, lb, ub)

        def residuals(p: np.ndarray) -> np.ndarray:
            z_model = tpl.model_fn(p, omega)
            return np.concatenate([(z_model.real - z.real),
                                   (z_model.imag - z.imag)])

        res = least_squares(
            residuals,
            p0,
            bounds=tpl.bounds,
            max_nfev=max_nfev,
        )

        rss = float(np.sum(res.fun ** 2))
        n_pts = len(res.fun)
        k = len(p0)
        bic, aic = _bic_aic(rss, n_pts, k)

        return {
            "template": tpl.name,
            "diagram": tpl.diagram,
            "params": {n: float(v) for n, v in zip(tpl.param_names, res.x)},
            "rss": rss,
            "bic": bic,
            "aic": aic,
            "n_params": k,
            "n_points": len(freq),
            "success": bool(res.success),
        }
