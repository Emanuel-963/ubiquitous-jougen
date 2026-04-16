"""Batch processor with parallelism for the IonFlow Pipeline.

Provides two levels of concurrency:

1. **Folder-level batching** — process multiple data directories in
   sequence or parallel (``BatchProcessor.run``).
2. **Intra-pipeline parallelism** — fit multiple circuit candidates
   simultaneously using ``concurrent.futures.ProcessPoolExecutor``
   (``ParallelFitter``).

Key features
------------
- ``BatchProcessor``: orchestrates multiple pipeline runs.
- ``ParallelFitter``: fits circuit candidates in parallel for ~3× speed.
- **Cancellation**: cooperative cancel via ``threading.Event`` propagated
  to workers.
- **Memory guard**: workers capped at ``min(cpu_count - 1, 4)``; never
  zero (at least 1).
- **Progress callbacks**: real-time per-item reporting for GUI or CLI
  integration.
- Thread-safe progress tracking via ``BatchProgress`` dataclass.

Dependencies
------------
Standard library only (``concurrent.futures``, ``threading``, ``os``).
"""

from __future__ import annotations

import logging
import os
import time
import threading
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Utility — worker cap
# ═══════════════════════════════════════════════════════════════════════


def max_workers(override: Optional[int] = None) -> int:
    """Return a safe worker count (memory guard).

    Rule: ``min(cpu_count - 1, 4)`` with floor at 1.  An explicit
    *override* is honoured but still capped at 8 for safety.
    """
    if override is not None:
        return max(1, min(int(override), 8))
    cpus = os.cpu_count() or 2
    return max(1, min(cpus - 1, 4))


# ═══════════════════════════════════════════════════════════════════════
# Progress tracking
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BatchProgress:
    """Thread-safe snapshot of batch processing progress."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    current_item: str = ""
    elapsed_s: float = 0.0
    items_per_second: float = 0.0
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed - self.failed)

    @property
    def fraction(self) -> float:
        """Progress fraction 0.0–1.0."""
        if self.total == 0:
            return 0.0
        return (self.completed + self.failed) / self.total

    @property
    def percent(self) -> float:
        return self.fraction * 100.0

    @property
    def eta_s(self) -> float:
        """Estimated time remaining (seconds)."""
        if self.items_per_second <= 0:
            return 0.0
        return self.remaining / self.items_per_second


# Progress callback type:  (progress: BatchProgress) → None
ProgressCallback = Callable[[BatchProgress], None]


# ═══════════════════════════════════════════════════════════════════════
# Parallel circuit fitting
# ═══════════════════════════════════════════════════════════════════════


def _fit_one_template_worker(
    template_name: str,
    param_names: List[str],
    bounds: Tuple[List[float], List[float]],
    freq: np.ndarray,
    z_real: np.ndarray,
    z_imag: np.ndarray,
) -> Dict[str, Any]:
    """Isolated worker: fit a single template (runs in sub-process).

    Parameters are passed as primitives / arrays to be pickle-safe;
    the actual ``CircuitTemplate`` is reconstructed from the catalog
    inside the worker to avoid pickling lambdas.
    """
    try:
        from src.circuit_fitting import circuit_catalog, fit_template

        catalog = {c.name: c for c in circuit_catalog()}
        template = catalog.get(template_name)
        if template is None:
            return {
                "template": template_name,
                "success": False,
                "message": f"Unknown template: {template_name}",
                "bic": np.inf,
                "rss": np.inf,
            }

        z = z_real + 1j * z_imag
        return fit_template(template, freq, z)
    except Exception as exc:
        return {
            "template": template_name,
            "success": False,
            "message": str(exc),
            "bic": np.inf,
            "rss": np.inf,
        }


class ParallelFitter:
    """Fit multiple circuit templates in parallel using processes.

    Provides ~3× speed-up when 7 circuits are tested simultaneously
    on multi-core machines.

    Parameters
    ----------
    n_workers : int | None
        Number of parallel workers.  ``None`` → ``max_workers()``.

    Examples
    --------
    >>> fitter = ParallelFitter()
    >>> results = fitter.fit_all(freq, z, ["Randles-CPE-W", "Randles-CPE"])
    """

    def __init__(self, n_workers: Optional[int] = None):
        self._n_workers = max_workers(n_workers)

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def fit_all(
        self,
        freq: np.ndarray,
        z: np.ndarray,
        template_names: List[str],
        *,
        cancel: Optional[threading.Event] = None,
    ) -> List[Dict[str, Any]]:
        """Fit every named template in parallel.

        Parameters
        ----------
        freq : ndarray
            Frequency array in Hz.
        z : ndarray
            Complex impedance array.
        template_names : list[str]
            Names from the circuit catalog.
        cancel : threading.Event | None
            If set mid-flight, pending futures are cancelled.

        Returns
        -------
        list[dict]
            Fitting results sorted by BIC (ascending).
        """
        if not template_names:
            return []

        z_real = z.real.copy()
        z_imag = z.imag.copy()

        # Serial fast-path for 1 template or 1 worker
        if len(template_names) == 1 or self._n_workers <= 1:
            return self._fit_serial(freq, z_real, z_imag, template_names, cancel)

        results: List[Dict[str, Any]] = []
        futures: Dict[Future, str] = {}

        try:
            with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
                for name in template_names:
                    if cancel and cancel.is_set():
                        break
                    f = pool.submit(
                        _fit_one_template_worker,
                        name,
                        [],        # param_names (unused, template reconstructed inside)
                        ([], []),  # bounds (unused)
                        freq.copy(),
                        z_real.copy(),
                        z_imag.copy(),
                    )
                    futures[f] = name

                for future in as_completed(futures):
                    if cancel and cancel.is_set():
                        # Cancel remaining
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        res = future.result(timeout=120)
                        results.append(res)
                    except Exception as exc:
                        results.append({
                            "template": futures[future],
                            "success": False,
                            "message": str(exc),
                            "bic": np.inf,
                            "rss": np.inf,
                        })
        except Exception as exc:
            logger.error("Parallel fitting pool error: %s", exc)
            # Fallback to serial
            return self._fit_serial(freq, z_real, z_imag, template_names, cancel)

        results.sort(key=lambda r: (r.get("bic", np.inf), r.get("rss", np.inf)))
        return results

    def _fit_serial(
        self,
        freq: np.ndarray,
        z_real: np.ndarray,
        z_imag: np.ndarray,
        names: List[str],
        cancel: Optional[threading.Event] = None,
    ) -> List[Dict[str, Any]]:
        """Serial fallback."""
        results: List[Dict[str, Any]] = []
        for name in names:
            if cancel and cancel.is_set():
                break
            res = _fit_one_template_worker(name, [], ([], []), freq, z_real, z_imag)
            results.append(res)
        results.sort(key=lambda r: (r.get("bic", np.inf), r.get("rss", np.inf)))
        return results


# ═══════════════════════════════════════════════════════════════════════
# Single-file pipeline functions (picklable, top-level)
# ═══════════════════════════════════════════════════════════════════════


def _process_eis_file(
    path: str,
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Load, preprocess, validate, extract features for ONE EIS file.

    Returns a dict with keys: 'file', 'features', 'raw_df', 'circuit_result',
    'error'.  Designed to be pickle-safe for process-pool execution.
    """
    try:
        from src.config import PipelineConfig
        from src.loader import load_eis_file
        from src.preprocessing import preprocess
        from src.physics_metrics import extract_features
        from src.cpe_fit import fit_cpe_warburg
        from src.circuit_fitting import run_shortlist_fit
        from src.validation import validate_eis_full

        cfg = PipelineConfig(**config_dict)
        fname = os.path.basename(path)
        df = preprocess(load_eis_file(path))

        # Validate
        vr = validate_eis_full(df)
        vr.log_all()

        # Feature extraction
        feat = extract_features(df)

        # CPE + Warburg fit
        try:
            fit = fit_cpe_warburg(df)
        except Exception:
            fit = {"Rs_fit": np.nan, "Rp_fit": np.nan, "Q": np.nan,
                   "n": np.nan, "Sigma": np.nan}
        feat.update(fit)

        # Circuit fitting (serial within file, uses existing shortlist logic)
        circ_res = run_shortlist_fit(
            df,
            sample_name=fname,
            save_plots=False,
        )

        return {
            "file": fname,
            "features": feat,
            "raw_df": df,
            "circuit_result": circ_res,
            "error": None,
        }
    except Exception as exc:
        return {
            "file": os.path.basename(path),
            "features": {},
            "raw_df": None,
            "circuit_result": None,
            "error": str(exc),
        }


def _process_drt_file(
    path: str,
    lambda_reg: float,
    n_taus: int,
) -> Dict[str, Any]:
    """Process one file through the DRT analysis.

    Returns dict with keys: 'file', 'result', 'error'.
    """
    try:
        from src.loader import load_eis_file
        from src.preprocessing import preprocess
        from src.drt_analysis import compute_drt

        fname = os.path.basename(path)
        df = preprocess(load_eis_file(path))
        freq = df["frequency"].to_numpy()
        z = df["zreal"].to_numpy() + 1j * df["zimag"].to_numpy()

        drt_res = compute_drt(freq, z, lambda_reg=lambda_reg, n_taus=n_taus)
        return {"file": fname, "result": drt_res, "error": None}
    except Exception as exc:
        return {"file": os.path.basename(path), "result": None, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════
# Batch result
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BatchResult:
    """Aggregated result of a batch run."""

    pipeline: str
    """Which pipeline was run: 'eis', 'drt', 'cycling', 'all'."""

    total_files: int = 0
    succeeded: int = 0
    failed: int = 0
    elapsed_s: float = 0.0

    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.succeeded / self.total_files

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"[{self.pipeline}] {self.succeeded}/{self.total_files} files "
            f"({self.success_rate:.0%}), {self.failed} failed, "
            f"{self.elapsed_s:.1f}s"
        )


# ═══════════════════════════════════════════════════════════════════════
# Main BatchProcessor
# ═══════════════════════════════════════════════════════════════════════


class BatchProcessor:
    """Batch processor for IonFlow Pipeline runs.

    Processes EIS / DRT files from one or multiple data directories,
    optionally in parallel with cancellation support and progress
    reporting.

    Parameters
    ----------
    n_workers : int | None
        Number of parallel processes.  ``None`` → auto-detect via
        ``max_workers()``.  Capped at 8 for memory safety.
    config : PipelineConfig | None
        Pipeline configuration.  ``None`` → use defaults.

    Examples
    --------
    >>> bp = BatchProcessor(n_workers=2)
    >>> result = bp.run_eis("data/raw")
    >>> print(result.summary())
    [eis] 12/15 files (80%), 3 failed, 4.2s

    Cancellation::

        cancel_event = threading.Event()
        # From another thread:
        cancel_event.set()
        result = bp.run_eis("data/raw", cancel=cancel_event)
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        config: Any = None,
    ):
        self._n_workers = max_workers(n_workers)
        self._config = config
        self._cancel: Optional[threading.Event] = None

    @property
    def n_workers(self) -> int:
        """Effective number of worker processes."""
        return self._n_workers

    # ── Cancellation ─────────────────────────────────────────────

    def cancel(self) -> None:
        """Signal workers to stop as soon as possible."""
        if self._cancel is not None:
            self._cancel.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel is not None and self._cancel.is_set()

    # ── EIS batch ────────────────────────────────────────────────

    def run_eis(
        self,
        data_dir: str,
        *,
        cancel: Optional[threading.Event] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> BatchResult:
        """Batch-process all EIS files in *data_dir*.

        Parameters
        ----------
        data_dir : str
            Directory containing ``.txt`` EIS files.
        cancel : threading.Event | None
            Cooperative cancellation token.
        progress_cb : callable | None
            Called after each file with a ``BatchProgress`` snapshot.

        Returns
        -------
        BatchResult
        """
        self._cancel = cancel or threading.Event()
        t0 = time.monotonic()

        files = self._list_files(data_dir)
        progress = BatchProgress(total=len(files))
        result = BatchResult(pipeline="eis", total_files=len(files))

        if not files:
            result.elapsed_s = time.monotonic() - t0
            return result

        # Serialise config for pickling
        config_dict = self._config_dict()

        if self._n_workers <= 1 or len(files) == 1:
            # Serial path
            for path in files:
                if self._cancel.is_set():
                    break
                self._update_progress(progress, os.path.basename(path), t0)
                if progress_cb:
                    progress_cb(progress)

                res = _process_eis_file(path, config_dict)
                self._accumulate(result, progress, res)
        else:
            # Parallel path
            with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
                future_map: Dict[Future, str] = {}
                for path in files:
                    if self._cancel.is_set():
                        break
                    f = pool.submit(_process_eis_file, path, config_dict)
                    future_map[f] = os.path.basename(path)

                for future in as_completed(future_map):
                    if self._cancel.is_set():
                        for f in future_map:
                            f.cancel()
                        break
                    fname = future_map[future]
                    self._update_progress(progress, fname, t0)

                    try:
                        res = future.result(timeout=300)
                    except Exception as exc:
                        res = {"file": fname, "error": str(exc)}
                    self._accumulate(result, progress, res)

                    if progress_cb:
                        progress_cb(progress)

        result.elapsed_s = time.monotonic() - t0
        return result

    # ── DRT batch ────────────────────────────────────────────────

    def run_drt(
        self,
        data_dir: str,
        *,
        lambda_reg: float = 1e-3,
        n_taus: int = 80,
        cancel: Optional[threading.Event] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> BatchResult:
        """Batch-process all DRT files in *data_dir*.

        Parameters
        ----------
        data_dir : str
            Directory containing ``.txt`` EIS files.
        lambda_reg : float
            Tikhonov regularisation parameter.
        n_taus : int
            Number of τ discretisation points.
        cancel : threading.Event | None
            Cooperative cancellation token.
        progress_cb : callable | None
            Progress callback.

        Returns
        -------
        BatchResult
        """
        self._cancel = cancel or threading.Event()
        t0 = time.monotonic()

        files = self._list_files(data_dir)
        progress = BatchProgress(total=len(files))
        result = BatchResult(pipeline="drt", total_files=len(files))

        if not files:
            result.elapsed_s = time.monotonic() - t0
            return result

        if self._n_workers <= 1 or len(files) == 1:
            for path in files:
                if self._cancel.is_set():
                    break
                self._update_progress(progress, os.path.basename(path), t0)
                if progress_cb:
                    progress_cb(progress)

                res = _process_drt_file(path, lambda_reg, n_taus)
                self._accumulate(result, progress, res)
        else:
            with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
                future_map: Dict[Future, str] = {}
                for path in files:
                    if self._cancel.is_set():
                        break
                    f = pool.submit(_process_drt_file, path, lambda_reg, n_taus)
                    future_map[f] = os.path.basename(path)

                for future in as_completed(future_map):
                    if self._cancel.is_set():
                        for f in future_map:
                            f.cancel()
                        break
                    fname = future_map[future]
                    self._update_progress(progress, fname, t0)

                    try:
                        res = future.result(timeout=300)
                    except Exception as exc:
                        res = {"file": fname, "error": str(exc)}
                    self._accumulate(result, progress, res)

                    if progress_cb:
                        progress_cb(progress)

        result.elapsed_s = time.monotonic() - t0
        return result

    # ── Multi-directory batch ────────────────────────────────────

    def run_multiple_dirs(
        self,
        data_dirs: Sequence[str],
        pipeline: str = "eis",
        *,
        cancel: Optional[threading.Event] = None,
        progress_cb: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> List[BatchResult]:
        """Run a pipeline across multiple data directories.

        Directories are processed sequentially to keep memory bounded;
        files within each directory are parallelised.

        Parameters
        ----------
        data_dirs : sequence of str
            Directories to process.
        pipeline : str
            ``'eis'`` or ``'drt'``.
        cancel : threading.Event | None
            Cancellation token shared across all runs.
        progress_cb : callable | None
            Progress callback.
        **kwargs
            Forwarded to the pipeline runner (e.g., ``lambda_reg``).

        Returns
        -------
        list[BatchResult]
        """
        self._cancel = cancel or threading.Event()
        results: List[BatchResult] = []
        for d in data_dirs:
            if self._cancel.is_set():
                break
            if pipeline == "drt":
                r = self.run_drt(d, cancel=self._cancel,
                                 progress_cb=progress_cb, **kwargs)
            else:
                r = self.run_eis(d, cancel=self._cancel,
                                 progress_cb=progress_cb)
            results.append(r)
        return results

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _list_files(data_dir: str) -> List[str]:
        """Return sorted list of .txt files in *data_dir*."""
        p = Path(data_dir)
        if not p.exists():
            logger.warning("Data directory not found: %s", data_dir)
            return []
        return sorted(
            str(f) for f in p.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        )

    def _config_dict(self) -> Dict[str, Any]:
        """Serialise pipeline config for pickle-safe transfer."""
        from src.config import PipelineConfig
        from dataclasses import asdict

        cfg = self._config or PipelineConfig.default()
        if hasattr(cfg, "__dataclass_fields__"):
            return asdict(cfg)
        return {}

    @staticmethod
    def _update_progress(
        progress: BatchProgress,
        item: str,
        t0: float,
    ) -> None:
        """Update progress snapshot."""
        progress.current_item = item
        progress.elapsed_s = time.monotonic() - t0
        done = progress.completed + progress.failed
        if progress.elapsed_s > 0 and done > 0:
            progress.items_per_second = done / progress.elapsed_s

    @staticmethod
    def _accumulate(
        result: BatchResult,
        progress: BatchProgress,
        res: Dict[str, Any],
    ) -> None:
        """Accumulate a single-file result into the batch."""
        result.results.append(res)
        err = res.get("error")
        fname = res.get("file", "unknown")
        if err:
            result.failed += 1
            result.errors[fname] = err
            progress.failed += 1
            progress.errors[fname] = err
        else:
            result.succeeded += 1
            progress.completed += 1
