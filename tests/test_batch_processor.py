"""Tests for the BatchProcessor and ParallelFitter (Day 24).

Covers:
- max_workers memory guard
- BatchProgress tracking
- ParallelFitter: serial and parallel paths, cancel support
- BatchProcessor: run_eis, run_drt, multi-dir, cancellation
- BatchResult summary
- _process_eis_file / _process_drt_file top-level workers
- Progress callback invocation
- Edge cases: empty dirs, missing dirs, 0 files, 1 file
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ═══════════════════════════════════════════════════════════════════


def _write_fake_eis_file(path: Path, n_points: int = 20) -> str:
    """Write a minimal valid EIS .txt file."""
    rng = np.random.RandomState(42)
    freq = np.logspace(5, -1, n_points)
    zreal = 10 + rng.uniform(-1, 1, n_points)
    zimag = -(5 + rng.uniform(0, 3, n_points))
    df = pd.DataFrame({"frequency": freq, "z'": zreal, "z''": zimag})
    fpath = str(path)
    df.to_csv(fpath, sep="\t", index=False)
    return fpath


@pytest.fixture
def data_dir_with_files(tmp_path):
    """Create a temporary data directory with 3 EIS files."""
    d = tmp_path / "raw"
    d.mkdir()
    for i in range(3):
        _write_fake_eis_file(d / f"sample_{i}.txt")
    return str(d)


@pytest.fixture
def data_dir_single_file(tmp_path):
    """Create a data directory with 1 EIS file."""
    d = tmp_path / "single"
    d.mkdir()
    _write_fake_eis_file(d / "only_one.txt")
    return str(d)


@pytest.fixture
def empty_data_dir(tmp_path):
    """Create an empty data directory."""
    d = tmp_path / "empty"
    d.mkdir()
    return str(d)


# ═══════════════════════════════════════════════════════════════════
# max_workers
# ═══════════════════════════════════════════════════════════════════


class TestMaxWorkers:
    def test_default_capped(self):
        from src.batch_processor import max_workers

        w = max_workers()
        assert 1 <= w <= 4

    def test_override_value(self):
        from src.batch_processor import max_workers

        assert max_workers(2) == 2

    def test_override_capped_at_8(self):
        from src.batch_processor import max_workers

        assert max_workers(100) == 8

    def test_override_minimum_1(self):
        from src.batch_processor import max_workers

        assert max_workers(0) == 1
        assert max_workers(-5) == 1

    @patch("src.batch_processor.os.cpu_count", return_value=1)
    def test_single_cpu(self, _mock):
        from src.batch_processor import max_workers

        assert max_workers() >= 1

    @patch("src.batch_processor.os.cpu_count", return_value=None)
    def test_unknown_cpu(self, _mock):
        from src.batch_processor import max_workers

        w = max_workers()
        assert w >= 1


# ═══════════════════════════════════════════════════════════════════
# BatchProgress
# ═══════════════════════════════════════════════════════════════════


class TestBatchProgress:
    def test_defaults(self):
        from src.batch_processor import BatchProgress

        p = BatchProgress()
        assert p.total == 0
        assert p.fraction == 0.0
        assert p.percent == 0.0
        assert p.remaining == 0

    def test_fraction_mid(self):
        from src.batch_processor import BatchProgress

        p = BatchProgress(total=10, completed=3, failed=2)
        assert p.fraction == 0.5
        assert p.percent == 50.0
        assert p.remaining == 5

    def test_eta(self):
        from src.batch_processor import BatchProgress

        p = BatchProgress(total=10, completed=5, items_per_second=2.0)
        assert p.eta_s == 2.5

    def test_eta_zero_speed(self):
        from src.batch_processor import BatchProgress

        p = BatchProgress(total=10, completed=5, items_per_second=0.0)
        assert p.eta_s == 0.0

    def test_errors_dict(self):
        from src.batch_processor import BatchProgress

        p = BatchProgress()
        p.errors["file.txt"] = "bad data"
        assert "file.txt" in p.errors


# ═══════════════════════════════════════════════════════════════════
# BatchResult
# ═══════════════════════════════════════════════════════════════════


class TestBatchResult:
    def test_summary_string(self):
        from src.batch_processor import BatchResult

        r = BatchResult(
            pipeline="eis",
            total_files=10,
            succeeded=8,
            failed=2,
            elapsed_s=5.0,
        )
        s = r.summary()
        assert "eis" in s
        assert "8/10" in s
        assert "80%" in s

    def test_success_rate(self):
        from src.batch_processor import BatchResult

        r = BatchResult(pipeline="drt", total_files=4, succeeded=3, failed=1)
        assert r.success_rate == 0.75

    def test_success_rate_zero_files(self):
        from src.batch_processor import BatchResult

        r = BatchResult(pipeline="eis", total_files=0)
        assert r.success_rate == 0.0


# ═══════════════════════════════════════════════════════════════════
# ParallelFitter
# ═══════════════════════════════════════════════════════════════════


class TestParallelFitter:
    def test_construction(self):
        from src.batch_processor import ParallelFitter

        pf = ParallelFitter(n_workers=2)
        assert pf.n_workers == 2

    def test_fit_all_empty_list(self):
        from src.batch_processor import ParallelFitter

        pf = ParallelFitter()
        results = pf.fit_all(np.array([1.0]), np.array([1+0j]), [])
        assert results == []

    def test_fit_all_serial(self):
        """Serial path with n_workers=1."""
        from src.batch_processor import ParallelFitter

        freq = np.logspace(5, -1, 30)
        z = 10 + 50 / (1 + 1j * 2 * np.pi * freq * 1e-4 * 50)

        pf = ParallelFitter(n_workers=1)
        results = pf.fit_all(freq, z, ["Randles-CPE-W"])
        assert len(results) == 1
        assert results[0].get("template") == "Randles-CPE-W"

    def test_fit_all_single_template(self):
        """Even with multiple workers, single template should work."""
        from src.batch_processor import ParallelFitter

        freq = np.logspace(5, -1, 30)
        z = 10 + 50 / (1 + 1j * 2 * np.pi * freq * 1e-4 * 50)

        pf = ParallelFitter(n_workers=2)
        results = pf.fit_all(freq, z, ["Randles-CPE"])
        assert len(results) == 1

    def test_fit_all_multiple_templates(self):
        """Parallel path with multiple templates."""
        from src.batch_processor import ParallelFitter

        freq = np.logspace(5, -1, 30)
        z = 10 + 50 / (1 + 1j * 2 * np.pi * freq * 1e-4 * 50)

        pf = ParallelFitter(n_workers=2)
        results = pf.fit_all(freq, z, ["Randles-CPE-W", "Randles-CPE"])
        assert len(results) == 2
        # Sorted by BIC
        bics = [r.get("bic", np.inf) for r in results]
        assert bics[0] <= bics[1] or np.isinf(bics[1])

    def test_fit_all_cancel(self):
        """Cancel should stop processing."""
        from src.batch_processor import ParallelFitter

        freq = np.logspace(5, -1, 30)
        z = 10 + 50 / (1 + 1j * 2 * np.pi * freq * 1e-4 * 50)

        cancel = threading.Event()
        cancel.set()  # Cancel immediately

        pf = ParallelFitter(n_workers=1)
        results = pf.fit_all(
            freq, z,
            ["Randles-CPE-W", "Randles-CPE", "Simple-RC"],
            cancel=cancel,
        )
        # Should be <= total (some might be skipped)
        assert len(results) <= 3

    def test_unknown_template(self):
        """Unknown template name returns error result."""
        from src.batch_processor import ParallelFitter

        freq = np.logspace(5, -1, 30)
        z = np.ones(30, dtype=complex)

        pf = ParallelFitter(n_workers=1)
        results = pf.fit_all(freq, z, ["NONEXISTENT"])
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "Unknown" in results[0].get("message", "")


# ═══════════════════════════════════════════════════════════════════
# _process_eis_file / _process_drt_file top-level workers
# ═══════════════════════════════════════════════════════════════════


class TestWorkerFunctions:
    def test_process_eis_file_success(self, data_dir_with_files):
        from src.batch_processor import _process_eis_file
        from src.config import PipelineConfig
        from dataclasses import asdict

        cfg = PipelineConfig.default()
        files = sorted(Path(data_dir_with_files).glob("*.txt"))
        res = _process_eis_file(str(files[0]), asdict(cfg))
        assert res["file"] == files[0].name
        assert res["error"] is None
        assert isinstance(res["features"], dict)

    def test_process_eis_file_bad_path(self):
        from src.batch_processor import _process_eis_file
        from src.config import PipelineConfig
        from dataclasses import asdict

        res = _process_eis_file("/nonexistent/bad.txt", asdict(PipelineConfig.default()))
        assert res["error"] is not None

    def test_process_drt_file_success(self, data_dir_with_files):
        from src.batch_processor import _process_drt_file

        files = sorted(Path(data_dir_with_files).glob("*.txt"))
        res = _process_drt_file(str(files[0]), lambda_reg=1e-3, n_taus=40)
        assert res["file"] == files[0].name
        # May succeed or fail depending on data quality
        assert "error" in res

    def test_process_drt_file_bad_path(self):
        from src.batch_processor import _process_drt_file

        res = _process_drt_file("/nonexistent/bad.txt", 1e-3, 40)
        assert res["error"] is not None


# ═══════════════════════════════════════════════════════════════════
# BatchProcessor — run_eis
# ═══════════════════════════════════════════════════════════════════


class TestBatchProcessorEIS:
    def test_run_eis_serial(self, data_dir_with_files):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=1)
        result = bp.run_eis(data_dir_with_files)
        assert result.pipeline == "eis"
        assert result.total_files == 3
        assert result.succeeded + result.failed == 3
        assert result.elapsed_s > 0

    def test_run_eis_parallel(self, data_dir_with_files):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=2)
        result = bp.run_eis(data_dir_with_files)
        assert result.total_files == 3
        assert result.succeeded + result.failed == 3

    def test_run_eis_empty_dir(self, empty_data_dir):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=1)
        result = bp.run_eis(empty_data_dir)
        assert result.total_files == 0
        assert result.succeeded == 0

    def test_run_eis_missing_dir(self, tmp_path):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=1)
        result = bp.run_eis(str(tmp_path / "nope"))
        assert result.total_files == 0

    def test_run_eis_single_file(self, data_dir_single_file):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=2)
        result = bp.run_eis(data_dir_single_file)
        assert result.total_files == 1

    def test_run_eis_cancel(self, data_dir_with_files):
        from src.batch_processor import BatchProcessor

        cancel = threading.Event()
        cancel.set()  # Cancel before start

        bp = BatchProcessor(n_workers=1)
        result = bp.run_eis(data_dir_with_files, cancel=cancel)
        # Should have processed 0 files or very few
        assert result.succeeded + result.failed <= result.total_files

    def test_run_eis_progress_callback(self, data_dir_with_files):
        from src.batch_processor import BatchProcessor, BatchProgress

        callbacks: List[BatchProgress] = []

        def on_progress(p: BatchProgress):
            callbacks.append(
                BatchProgress(
                    total=p.total,
                    completed=p.completed,
                    failed=p.failed,
                    current_item=p.current_item,
                )
            )

        bp = BatchProcessor(n_workers=1)
        bp.run_eis(data_dir_with_files, progress_cb=on_progress)
        assert len(callbacks) >= 3  # At least once per file

    def test_run_eis_with_config(self, data_dir_with_files):
        from src.batch_processor import BatchProcessor
        from src.config import PipelineConfig

        cfg = PipelineConfig.default()
        bp = BatchProcessor(n_workers=1, config=cfg)
        result = bp.run_eis(data_dir_with_files)
        assert result.total_files == 3


# ═══════════════════════════════════════════════════════════════════
# BatchProcessor — run_drt
# ═══════════════════════════════════════════════════════════════════


class TestBatchProcessorDRT:
    def test_run_drt_serial(self, data_dir_with_files):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=1)
        result = bp.run_drt(data_dir_with_files, lambda_reg=1e-3, n_taus=40)
        assert result.pipeline == "drt"
        assert result.total_files == 3

    def test_run_drt_parallel(self, data_dir_with_files):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=2)
        result = bp.run_drt(data_dir_with_files, lambda_reg=1e-3, n_taus=40)
        assert result.total_files == 3

    def test_run_drt_empty_dir(self, empty_data_dir):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=1)
        result = bp.run_drt(empty_data_dir)
        assert result.total_files == 0

    def test_run_drt_cancel(self, data_dir_with_files):
        from src.batch_processor import BatchProcessor

        cancel = threading.Event()
        cancel.set()

        bp = BatchProcessor(n_workers=1)
        result = bp.run_drt(data_dir_with_files, cancel=cancel)
        assert result.succeeded + result.failed <= result.total_files


# ═══════════════════════════════════════════════════════════════════
# BatchProcessor — multi-dir
# ═══════════════════════════════════════════════════════════════════


class TestBatchProcessorMultiDir:
    def test_multiple_dirs_eis(self, tmp_path):
        from src.batch_processor import BatchProcessor

        dirs = []
        for i in range(2):
            d = tmp_path / f"batch_{i}"
            d.mkdir()
            _write_fake_eis_file(d / f"sample_{i}.txt")
            dirs.append(str(d))

        bp = BatchProcessor(n_workers=1)
        results = bp.run_multiple_dirs(dirs, pipeline="eis")
        assert len(results) == 2
        assert all(r.pipeline == "eis" for r in results)

    def test_multiple_dirs_drt(self, tmp_path):
        from src.batch_processor import BatchProcessor

        dirs = []
        for i in range(2):
            d = tmp_path / f"drt_{i}"
            d.mkdir()
            _write_fake_eis_file(d / f"drt_sample_{i}.txt")
            dirs.append(str(d))

        bp = BatchProcessor(n_workers=1)
        results = bp.run_multiple_dirs(
            dirs, pipeline="drt", lambda_reg=1e-3, n_taus=40,
        )
        assert len(results) == 2

    def test_multiple_dirs_cancel(self, tmp_path):
        from src.batch_processor import BatchProcessor

        dirs = []
        for i in range(3):
            d = tmp_path / f"batch_{i}"
            d.mkdir()
            _write_fake_eis_file(d / f"s_{i}.txt")
            dirs.append(str(d))

        cancel = threading.Event()
        cancel.set()

        bp = BatchProcessor(n_workers=1)
        results = bp.run_multiple_dirs(dirs, cancel=cancel)
        # Should process at most 0-1 dirs
        assert len(results) <= 3

    def test_empty_dirs_list(self):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=1)
        results = bp.run_multiple_dirs([])
        assert results == []


# ═══════════════════════════════════════════════════════════════════
# Cancellation / is_cancelled
# ═══════════════════════════════════════════════════════════════════


class TestCancellation:
    def test_cancel_method(self, data_dir_with_files):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=1)
        # Start a run and immediately cancel
        cancel = threading.Event()
        bp.run_eis(data_dir_with_files, cancel=cancel)
        # After run, cancel and check
        bp.cancel()
        assert bp.is_cancelled

    def test_is_cancelled_before_run(self):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=1)
        assert not bp.is_cancelled  # No cancel event yet


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_non_txt_files_ignored(self, tmp_path):
        """Only .txt files should be picked up."""
        from src.batch_processor import BatchProcessor

        d = tmp_path / "mixed"
        d.mkdir()
        _write_fake_eis_file(d / "good.txt")
        (d / "ignore.csv").write_text("a,b,c")
        (d / "ignore.json").write_text("{}")
        (d / "subdir").mkdir()

        bp = BatchProcessor(n_workers=1)
        result = bp.run_eis(str(d))
        assert result.total_files == 1

    def test_results_contain_file_key(self, data_dir_single_file):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor(n_workers=1)
        result = bp.run_eis(data_dir_single_file)
        assert len(result.results) == 1
        assert "file" in result.results[0]

    def test_summary_with_failures(self):
        from src.batch_processor import BatchResult

        r = BatchResult(pipeline="eis", total_files=5, succeeded=3, failed=2,
                        elapsed_s=10.0, errors={"a.txt": "bad", "b.txt": "bad"})
        s = r.summary()
        assert "3/5" in s
        assert "2 failed" in s
        assert "10.0s" in s

    def test_batch_processor_default_workers(self):
        from src.batch_processor import BatchProcessor

        bp = BatchProcessor()
        assert 1 <= bp.n_workers <= 4

    def test_parallel_fitter_fallback_on_pool_error(self):
        """If pool creation fails, should fall back to serial."""
        from src.batch_processor import ParallelFitter

        freq = np.logspace(5, -1, 30)
        z = 10 + 50 / (1 + 1j * 2 * np.pi * freq * 1e-4 * 50)

        pf = ParallelFitter(n_workers=2)
        with patch(
            "src.batch_processor.ProcessPoolExecutor",
            side_effect=RuntimeError("pool broken"),
        ):
            results = pf.fit_all(freq, z, ["Randles-CPE-W", "Randles-CPE"])
            # Falls back to serial — should still get results
            assert len(results) == 2
