"""Structured logging configuration for the IonFlow Pipeline.

Provides a single ``setup_logging()`` call that wires:

* **RotatingFileHandler** → ``logs/ionflow.log``
  (DEBUG level, 2 MB max, 5 backups)
* **StreamHandler** (stderr) → INFO level for console / IDE output
* **QueueHandler → GUI** (optional) — the GUI passes a
  ``queue.Queue`` and polls it with ``after()`` to append messages
  to the CTkTextbox without threading issues.

All modules should use ``logging.getLogger(__name__)`` — they never
need to import anything from here.  ``setup_logging()`` is called
**once** at application start (by ``gui_app.py`` or the CLI
entry-points).

Usage — CLI
-----------
::

    from src.logger import setup_logging
    setup_logging()                       # file + stderr handlers

Usage — GUI
-----------
::

    import queue
    from src.logger import setup_logging, GUIQueueHandler

    log_queue: queue.Queue[str] = queue.Queue()
    setup_logging(gui_queue=log_queue)

    # In the Tk main-loop poll:
    def _poll_log_queue(self):
        while not log_queue.empty():
            self._append_log(log_queue.get_nowait())
        self.after(100, self._poll_log_queue)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import queue
from pathlib import Path
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────
_LOG_DIR = "logs"
_LOG_FILE = "ionflow.log"
_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
_BACKUP_COUNT = 5

_FMT = "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_ROOT_LOGGER_NAME = "src"

# Sentinel so we can call ``setup_logging()`` multiple times safely
_configured = False


# ── Custom handler: write formatted records into a queue ─────────────
class GUIQueueHandler(logging.Handler):
    """Logging handler that pushes **formatted** log strings into a
    :class:`queue.Queue`.

    The GUI thread polls this queue and appends messages to the TextBox.
    Only records ≥ INFO are forwarded so the GUI stays readable.
    """

    def __init__(self, log_queue: queue.Queue, level: int = logging.INFO):
        super().__init__(level)
        self.log_queue: queue.Queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        """Format and enqueue a log record for the GUI.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be formatted and placed onto the
            queue for consumption by the GUI thread.
        """
        try:
            msg = self.format(record)
            self.log_queue.put_nowait(msg)
        except Exception:  # pragma: no cover — queue full / shutdown
            self.handleError(record)


# ── Public API ───────────────────────────────────────────────────────
def setup_logging(
    *,
    log_dir: str = _LOG_DIR,
    log_file: str = _LOG_FILE,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    gui_queue: Optional[queue.Queue] = None,
    gui_level: int = logging.INFO,
    force: bool = False,
) -> None:
    """Configure the root ``src`` logger with structured handlers.

    Parameters
    ----------
    log_dir : str
        Directory for the rotating log file.
    log_file : str
        Name of the log file inside *log_dir*.
    file_level : int
        Minimum level written to the file (default ``DEBUG``).
    console_level : int
        Minimum level written to stderr (default ``INFO``).
    gui_queue : queue.Queue, optional
        If provided, a :class:`GUIQueueHandler` is attached so the GUI
        can poll messages without blocking.
    gui_level : int
        Minimum level forwarded to the GUI queue (default ``INFO``).
    force : bool
        Re-configure even if already called once.
    """
    global _configured  # noqa: PLW0603
    if _configured and not force:
        # If a GUI queue is supplied after initial setup, still attach it.
        if gui_queue is not None:
            _attach_gui_handler(gui_queue, gui_level)
        return
    _configured = True

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # ── Root logger for the whole ``src`` package ──────────────────
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)  # allow everything; handlers filter
    root.handlers.clear()

    # Also configure the top-level loggers used by main*.py
    for name in ("__main__", "main", "main_drt", "main_cycling"):
        mod_logger = logging.getLogger(name)
        mod_logger.setLevel(logging.DEBUG)
        mod_logger.handlers.clear()
        # Propagate to root so the same handlers apply
        mod_logger.parent = root

    # ── File handler (rotating) ───────────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # ── Console handler ───────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # ── GUI handler (optional) ────────────────────────────────────
    if gui_queue is not None:
        _attach_gui_handler(gui_queue, gui_level)


def _attach_gui_handler(
    log_queue: queue.Queue, level: int = logging.INFO
) -> None:
    """Add a :class:`GUIQueueHandler` to the root logger (idempotent)."""
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    # Avoid duplicate GUI handlers
    for h in root.handlers:
        if isinstance(h, GUIQueueHandler):
            return
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)
    gh = GUIQueueHandler(log_queue, level=level)
    gh.setFormatter(formatter)
    root.addHandler(gh)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — identical to ``logging.getLogger(name)``
    but ensures ``setup_logging()`` has been called at least once with
    sensible defaults (useful in tests / one-off scripts).
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
