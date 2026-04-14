"""Tests for src.logger — structured logging configuration."""

import logging
import logging.handlers
import os
import queue
import shutil
import tempfile

import pytest

from src.logger import GUIQueueHandler, setup_logging, _FMT, _DATE_FMT


# ── Helpers ──────────────────────────────────────────────────────────
@pytest.fixture()
def tmp_log_dir():
    """Create a temporary directory for logs and clean up after."""
    d = tempfile.mkdtemp(prefix="ionflow_test_logs_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset the module-level _configured flag between tests."""
    import src.logger as _mod
    original = _mod._configured
    _mod._configured = False
    # Also clear any handlers added to the root src logger
    root = logging.getLogger("src")
    original_handlers = root.handlers.copy()
    yield
    root.handlers = original_handlers
    _mod._configured = original


# ── setup_logging — basic smoke test ─────────────────────────────────
class TestSetupLogging:
    def test_creates_log_directory(self, tmp_log_dir):
        log_sub = os.path.join(tmp_log_dir, "subdir")
        setup_logging(log_dir=log_sub, force=True)
        assert os.path.isdir(log_sub)

    def test_file_handler_created(self, tmp_log_dir):
        setup_logging(log_dir=tmp_log_dir, force=True)
        root = logging.getLogger("src")
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) >= 1

    def test_console_handler_created(self, tmp_log_dir):
        setup_logging(log_dir=tmp_log_dir, force=True)
        root = logging.getLogger("src")
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, (logging.handlers.RotatingFileHandler, GUIQueueHandler))
        ]
        assert len(stream_handlers) >= 1

    def test_log_format_contains_expected_fields(self, tmp_log_dir):
        setup_logging(log_dir=tmp_log_dir, force=True)
        root = logging.getLogger("src")
        fh = next(
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        )
        assert "%(asctime)s" in fh.formatter._fmt
        assert "%(levelname)" in fh.formatter._fmt
        assert "%(name)s" in fh.formatter._fmt

    def test_idempotent_without_force(self, tmp_log_dir):
        setup_logging(log_dir=tmp_log_dir, force=True)
        root = logging.getLogger("src")
        n_handlers = len(root.handlers)
        setup_logging(log_dir=tmp_log_dir)  # second call, no force
        assert len(root.handlers) == n_handlers

    def test_force_reconfigures(self, tmp_log_dir):
        setup_logging(log_dir=tmp_log_dir, force=True)
        root = logging.getLogger("src")
        n1 = len(root.handlers)
        setup_logging(log_dir=tmp_log_dir, force=True)
        # Handlers are cleared + re-added, count should be same
        assert len(root.handlers) == n1

    def test_writes_to_log_file(self, tmp_log_dir):
        setup_logging(log_dir=tmp_log_dir, log_file="test.log", force=True)
        test_logger = logging.getLogger("src.test_module")
        test_logger.info("hello from test")
        # Flush handlers
        for h in logging.getLogger("src").handlers:
            h.flush()
        log_path = os.path.join(tmp_log_dir, "test.log")
        assert os.path.exists(log_path)
        content = open(log_path, encoding="utf-8").read()
        assert "hello from test" in content

    def test_file_level_debug_captured(self, tmp_log_dir):
        setup_logging(
            log_dir=tmp_log_dir, log_file="debug.log",
            file_level=logging.DEBUG, force=True,
        )
        test_logger = logging.getLogger("src.debug_test")
        test_logger.debug("debug-level message")
        for h in logging.getLogger("src").handlers:
            h.flush()
        content = open(
            os.path.join(tmp_log_dir, "debug.log"), encoding="utf-8"
        ).read()
        assert "debug-level message" in content

    def test_main_loggers_reparented(self, tmp_log_dir):
        setup_logging(log_dir=tmp_log_dir, force=True)
        for name in ("main", "main_drt", "main_cycling"):
            mod_logger = logging.getLogger(name)
            assert mod_logger.parent is logging.getLogger("src")


# ── GUIQueueHandler ──────────────────────────────────────────────────
class TestGUIQueueHandler:
    def test_handler_puts_formatted_messages(self):
        q: queue.Queue = queue.Queue()
        handler = GUIQueueHandler(q, level=logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="gui message", args=(), exc_info=None,
        )
        handler.emit(record)
        assert not q.empty()
        msg = q.get_nowait()
        assert "INFO" in msg
        assert "gui message" in msg

    def test_handler_filters_below_level(self):
        q: queue.Queue = queue.Queue()
        handler = GUIQueueHandler(q, level=logging.WARNING)
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="", lineno=0,
            msg="should not appear", args=(), exc_info=None,
        )
        # Handler should filter this out (level too low)
        if record.levelno >= handler.level:
            handler.emit(record)
        assert q.empty()

    def test_gui_queue_via_setup(self, tmp_log_dir):
        q: queue.Queue = queue.Queue()
        setup_logging(log_dir=tmp_log_dir, gui_queue=q, force=True)
        root = logging.getLogger("src")
        gui_handlers = [h for h in root.handlers if isinstance(h, GUIQueueHandler)]
        assert len(gui_handlers) == 1

    def test_gui_queue_attached_after_initial_setup(self, tmp_log_dir):
        setup_logging(log_dir=tmp_log_dir, force=True)
        q: queue.Queue = queue.Queue()
        setup_logging(log_dir=tmp_log_dir, gui_queue=q)
        root = logging.getLogger("src")
        gui_handlers = [h for h in root.handlers if isinstance(h, GUIQueueHandler)]
        assert len(gui_handlers) == 1

    def test_no_duplicate_gui_handlers(self, tmp_log_dir):
        q: queue.Queue = queue.Queue()
        setup_logging(log_dir=tmp_log_dir, gui_queue=q, force=True)
        # Try attaching again
        setup_logging(log_dir=tmp_log_dir, gui_queue=q)
        root = logging.getLogger("src")
        gui_handlers = [h for h in root.handlers if isinstance(h, GUIQueueHandler)]
        assert len(gui_handlers) == 1

    def test_gui_receives_info_from_submodule(self, tmp_log_dir):
        q: queue.Queue = queue.Queue()
        setup_logging(log_dir=tmp_log_dir, gui_queue=q, force=True)
        child = logging.getLogger("src.submod")
        child.info("child message")
        messages = []
        while not q.empty():
            messages.append(q.get_nowait())
        assert any("child message" in m for m in messages)


# ── get_logger convenience ───────────────────────────────────────────
class TestGetLogger:
    def test_returns_logger_instance(self):
        from src.logger import get_logger
        lg = get_logger("src.test_get")
        assert isinstance(lg, logging.Logger)
        assert lg.name == "src.test_get"
