"""SQLite schema for IonFlow Pipeline — v1 (Phase 7).

All DDL lives here so migrations and tests can reference it from one place.
Calling :func:`init_db` is safe to repeat: every statement uses
``CREATE TABLE IF NOT EXISTS``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_DDL_STATEMENTS = [
    # ── Migration tracker ────────────────────────────────────────────
    """CREATE TABLE IF NOT EXISTS _migrations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        version     INTEGER NOT NULL UNIQUE,
        applied_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        description TEXT
    )""",
    # ── Samples (one row per pipeline run or uploaded file) ──────────
    """CREATE TABLE IF NOT EXISTS samples (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        type        TEXT    NOT NULL
                            CHECK(type IN ('eis','cycling','drt','mixed')),
        created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        file_path   TEXT    DEFAULT '',
        meta_json   TEXT    DEFAULT '{}'
    )""",
    # ── EIS results (one row per file in a run) ──────────────────────
    """CREATE TABLE IF NOT EXISTS eis_results (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id    INTEGER NOT NULL
                             REFERENCES samples(id) ON DELETE CASCADE,
        file_label   TEXT    DEFAULT '',
        rs_fit       REAL,
        rp_fit       REAL,
        circuit_name TEXT    DEFAULT '',
        bic          REAL,
        confidence   REAL,
        c_mean       REAL,
        energy_mean  REAL,
        score        REAL,
        rank         INTEGER,
        category     TEXT    DEFAULT '',
        data_json    TEXT    DEFAULT '{}'
    )""",
    # ── Cycling results ──────────────────────────────────────────────
    """CREATE TABLE IF NOT EXISTS cycling_results (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id     INTEGER NOT NULL
                              REFERENCES samples(id) ON DELETE CASCADE,
        cycle_number  INTEGER,
        energy_wh_kg  REAL,
        power_w_kg    REAL,
        retention_pct REAL,
        data_json     TEXT    DEFAULT '{}'
    )""",
    # ── DRT results ──────────────────────────────────────────────────
    """CREATE TABLE IF NOT EXISTS drt_results (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id   INTEGER NOT NULL
                            REFERENCES samples(id) ON DELETE CASCADE,
        file_label  TEXT    DEFAULT '',
        tau_peak1   REAL,
        gamma_peak1 REAL,
        tau_peak2   REAL,
        gamma_peak2 REAL,
        tau_peak3   REAL,
        gamma_peak3 REAL,
        data_json   TEXT    DEFAULT '{}'
    )""",
    # ── Generic parameter store ──────────────────────────────────────
    """CREATE TABLE IF NOT EXISTS parameters (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id   INTEGER NOT NULL
                            REFERENCES samples(id) ON DELETE CASCADE,
        param_name  TEXT    NOT NULL,
        param_value REAL,
        param_unit  TEXT    DEFAULT '',
        created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    )""",
    # ── Fitting history (ML feature store) ──────────────────────────
    """CREATE TABLE IF NOT EXISTS fitting_history (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id               TEXT    NOT NULL,
        circuit_name            TEXT    NOT NULL,
        bic                     REAL,
        confidence              REAL,
        spectral_features_json  TEXT    DEFAULT '{}',
        circuit_params_json     TEXT    DEFAULT '{}',
        created_at              TEXT    NOT NULL DEFAULT (datetime('now'))
    )""",
    # ── Indexes ──────────────────────────────────────────────────────
    "CREATE INDEX IF NOT EXISTS idx_eis_sample     ON eis_results(sample_id)",
    "CREATE INDEX IF NOT EXISTS idx_cycling_sample ON cycling_results(sample_id)",
    "CREATE INDEX IF NOT EXISTS idx_drt_sample     ON drt_results(sample_id)",
    "CREATE INDEX IF NOT EXISTS idx_params_sample  ON parameters(sample_id)",
    "CREATE INDEX IF NOT EXISTS idx_hist_circuit   ON fitting_history(circuit_name)",
    "CREATE INDEX IF NOT EXISTS idx_hist_sample    ON fitting_history(sample_id)",
]


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Open (or create) *db_path* and apply the full schema.

    Safe to call multiple times — all statements use ``IF NOT EXISTS``.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Parent directories are created
        automatically.

    Returns
    -------
    sqlite3.Connection
        An open connection with ``row_factory = sqlite3.Row`` and WAL
        journal mode enabled.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    for stmt in _DDL_STATEMENTS:
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)

    conn.commit()
    return conn
