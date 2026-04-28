"""Schema migration runner for the IonFlow SQLite backend.

Each migration is a ``(version, description, callable)`` tuple.  The
callable receives an open :class:`sqlite3.Connection` and must commit any
changes itself.

The baseline schema (v1) is created by :func:`src.db.schema.init_db`
before any migrations run, so migrations start from v2 onwards.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------
# Tuple: (schema_version: int, description: str, migrate_fn: Callable)
_MIGRATIONS: List[Tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    # v1 baseline is handled by schema.init_db — nothing to add here yet.
    # Future migrations: (2, "add column X to Y", _migrate_v2), ...
]


def run_migrations(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations in version order.

    Idempotent: already-applied migrations are skipped.

    Parameters
    ----------
    conn:
        An open SQLite connection (``PRAGMA foreign_keys`` already set).

    Returns
    -------
    int
        Number of new migrations applied.
    """
    # Ensure the tracking table exists (may be called before init_db in tests)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS _migrations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            version     INTEGER NOT NULL UNIQUE,
            applied_at  TEXT    NOT NULL,
            description TEXT
        )"""
    )
    conn.commit()

    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM _migrations").fetchone()
    current_version: int = row[0] if row else 0

    applied = 0
    for version, description, migrate_fn in sorted(_MIGRATIONS, key=lambda t: t[0]):
        if version <= current_version:
            continue
        logger.info("Applying DB migration v%d: %s", version, description)
        migrate_fn(conn)
        conn.execute(
            "INSERT INTO _migrations (version, applied_at, description) VALUES (?,?,?)",
            (
                version,
                datetime.now(timezone.utc).isoformat(),
                description,
            ),
        )
        conn.commit()
        applied += 1
        logger.info("Migration v%d applied successfully", version)

    return applied
