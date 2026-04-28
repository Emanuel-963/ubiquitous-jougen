"""IonFlowRepository — CRUD layer on top of the IonFlow SQLite backend.

Typical usage
-------------
::

    from src.db.repository import IonFlowRepository
    import pandas as pd

    repo = IonFlowRepository("data/ionflow.db")

    # Store a run
    run_id = repo.add_sample("Run 2026-05-01", "eis")
    repo.save_eis_results(run_id, ranked_df)

    # Browse
    df = repo.get_eis_results()
    print(df[["sample_name", "circuit_name", "score"]].head())
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.db.migrations import run_migrations
from src.db.schema import init_db

logger = logging.getLogger(__name__)


def _sf(v: Any) -> Optional[float]:
    """Safe float cast; returns None on failure."""
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _si(v: Any) -> Optional[int]:
    """Safe int cast; returns None on failure."""
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IonFlowRepository:
    """Repository for IonFlow pipeline results backed by SQLite.

    Parameters
    ----------
    db_path:
        Path to the ``.db`` file.  Defaults to ``data/ionflow.db``.
    """

    DEFAULT_DB_PATH = "data/ionflow.db"

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection = init_db(self._db_path)
        run_migrations(self._conn)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    # ── Samples ───────────────────────────────────────────────────────

    def add_sample(
        self,
        name: str,
        type_: str,
        file_path: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a new sample/run record.

        Parameters
        ----------
        name:
            Human-readable label (e.g. file stem or run timestamp).
        type_:
            One of ``'eis'``, ``'cycling'``, ``'drt'``, ``'mixed'``.
        file_path:
            Source file path (informational only).
        meta:
            Optional dict of extra metadata (stored as JSON).

        Returns
        -------
        int
            The new ``samples.id``.
        """
        meta_json = json.dumps(meta or {}, default=str)
        cur = self._conn.execute(
            "INSERT INTO samples (name, type, created_at, file_path, meta_json)"
            " VALUES (?,?,?,?,?)",
            (name, type_, _now(), file_path, meta_json),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_all_samples(self) -> pd.DataFrame:
        """Return all samples ordered by most-recent first."""
        rows = self._conn.execute(
            "SELECT id, name, type, created_at, file_path"
            " FROM samples ORDER BY created_at DESC"
        ).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["id", "name", "type", "created_at", "file_path"]
            )
        return pd.DataFrame([dict(r) for r in rows])

    def get_sample_by_id(self, sample_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM samples WHERE id=?", (sample_id,)
        ).fetchone()
        return dict(row) if row else None

    def delete_sample(self, sample_id: int) -> None:
        """Delete a sample and all its child rows (cascade)."""
        self._conn.execute("DELETE FROM samples WHERE id=?", (sample_id,))
        self._conn.commit()

    # ── EIS Results ──────────────────────────────────────────────────

    def save_eis_results(self, sample_id: int, ranked_df: pd.DataFrame) -> None:
        """Persist all rows from *ranked_df* linked to *sample_id*.

        Column names are looked up with multiple fallbacks so the method
        works with both the GUI's internal column names and the exported
        table column names.
        """

        def _get(row: pd.Series, *keys: str) -> Any:
            for k in keys:
                if k in row.index and row[k] is not None:
                    return row[k]
            return None

        rows: List[tuple] = []
        for label, row in ranked_df.iterrows():
            rows.append(
                (
                    sample_id,
                    str(label),
                    _sf(_get(row, "Rs_fit", "rs_fit", "Rs")),
                    _sf(_get(row, "Rp_fit", "rp_fit", "Rp")),
                    str(_get(row, "circuit_name", "Circuit", "circuit") or ""),
                    _sf(_get(row, "BIC", "bic")),
                    _sf(_get(row, "confidence", "Confidence")),
                    _sf(_get(row, "C_mean", "c_mean", "Capacitance_mean")),
                    _sf(_get(row, "Energy_mean", "energy_mean")),
                    _sf(_get(row, "Score", "score")),
                    _si(_get(row, "Rank", "rank")),
                    str(_get(row, "category", "Category") or ""),
                    row.to_json(default_handler=str),
                )
            )

        self._conn.executemany(
            "INSERT INTO eis_results"
            " (sample_id, file_label, rs_fit, rp_fit, circuit_name, bic,"
            "  confidence, c_mean, energy_mean, score, rank, category, data_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()

    def get_eis_results(self, sample_id: Optional[int] = None) -> pd.DataFrame:
        """Return EIS results, optionally filtered by *sample_id*."""
        if sample_id is not None:
            rows = self._conn.execute(
                "SELECT e.*, s.name AS sample_name"
                " FROM eis_results e JOIN samples s ON s.id=e.sample_id"
                " WHERE e.sample_id=? ORDER BY e.rank",
                (sample_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT e.*, s.name AS sample_name"
                " FROM eis_results e JOIN samples s ON s.id=e.sample_id"
                " ORDER BY e.sample_id, e.rank"
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # ── Cycling Results ───────────────────────────────────────────────

    def save_cycling_results(self, sample_id: int, cycles_df: pd.DataFrame) -> None:
        """Persist per-cycle metrics from *cycles_df*."""

        def _col(row: pd.Series, *keys: str) -> Any:
            for k in keys:
                if k in row.index:
                    return row[k]
            return None

        rows: List[tuple] = []
        for _, row in cycles_df.iterrows():
            rows.append(
                (
                    sample_id,
                    _si(_col(row, "cycle", "Cycle", "cycle_number", "Cycle_Number")),
                    _sf(_col(row, "Energy_Wh_kg", "energy_wh_kg", "Energy")),
                    _sf(_col(row, "Power_W_kg", "power_w_kg", "Power")),
                    _sf(_col(row, "retention", "retention_pct", "Retention")),
                    row.to_json(default_handler=str),
                )
            )
        self._conn.executemany(
            "INSERT INTO cycling_results"
            " (sample_id, cycle_number, energy_wh_kg, power_w_kg,"
            "  retention_pct, data_json)"
            " VALUES (?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()

    def get_cycling_results(self, sample_id: Optional[int] = None) -> pd.DataFrame:
        """Return cycling results, optionally filtered by *sample_id*."""
        if sample_id is not None:
            rows = self._conn.execute(
                "SELECT c.*, s.name AS sample_name"
                " FROM cycling_results c JOIN samples s ON s.id=c.sample_id"
                " WHERE c.sample_id=? ORDER BY c.cycle_number",
                (sample_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT c.*, s.name AS sample_name"
                " FROM cycling_results c JOIN samples s ON s.id=c.sample_id"
                " ORDER BY c.sample_id, c.cycle_number"
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # ── DRT Results ───────────────────────────────────────────────────

    def save_drt_results(self, sample_id: int, drt_df: pd.DataFrame) -> None:
        """Persist DRT peak data from *drt_df*."""

        def _col(row: pd.Series, *keys: str) -> Any:
            for k in keys:
                if k in row.index:
                    return row[k]
            return None

        rows: List[tuple] = []
        for label, row in drt_df.iterrows():
            rows.append(
                (
                    sample_id,
                    str(label),
                    _sf(_col(row, "tau_peak_1", "Tau_1", "tau1")),
                    _sf(_col(row, "gamma_peak_1", "Gamma_1", "gamma1")),
                    _sf(_col(row, "tau_peak_2", "Tau_2", "tau2")),
                    _sf(_col(row, "gamma_peak_2", "Gamma_2", "gamma2")),
                    _sf(_col(row, "tau_peak_3", "Tau_3", "tau3")),
                    _sf(_col(row, "gamma_peak_3", "Gamma_3", "gamma3")),
                    row.to_json(default_handler=str),
                )
            )
        self._conn.executemany(
            "INSERT INTO drt_results"
            " (sample_id, file_label, tau_peak1, gamma_peak1, tau_peak2,"
            "  gamma_peak2, tau_peak3, gamma_peak3, data_json)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()

    def get_drt_results(self, sample_id: Optional[int] = None) -> pd.DataFrame:
        """Return DRT results, optionally filtered by *sample_id*."""
        if sample_id is not None:
            rows = self._conn.execute(
                "SELECT d.*, s.name AS sample_name"
                " FROM drt_results d JOIN samples s ON s.id=d.sample_id"
                " WHERE d.sample_id=?",
                (sample_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT d.*, s.name AS sample_name"
                " FROM drt_results d JOIN samples s ON s.id=d.sample_id"
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # ── Parameters ────────────────────────────────────────────────────

    def save_parameter(
        self, sample_id: int, name: str, value: float, unit: str = ""
    ) -> None:
        """Insert a single named scalar parameter."""
        self._conn.execute(
            "INSERT INTO parameters"
            " (sample_id, param_name, param_value, param_unit, created_at)"
            " VALUES (?,?,?,?,?)",
            (sample_id, name, value, unit, _now()),
        )
        self._conn.commit()

    def get_parameters(self, sample_id: Optional[int] = None) -> pd.DataFrame:
        """Return parameters, optionally filtered by *sample_id*."""
        if sample_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM parameters WHERE sample_id=? ORDER BY created_at",
                (sample_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT p.*, s.name AS sample_name"
                " FROM parameters p JOIN samples s ON s.id=p.sample_id"
                " ORDER BY p.created_at"
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Return row counts for every main table."""
        tables = [
            "samples",
            "eis_results",
            "cycling_results",
            "drt_results",
            "parameters",
            "fitting_history",
        ]
        result: Dict[str, int] = {}
        for tbl in tables:
            row = self._conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
            result[tbl] = row[0] if row else 0
        return result

    def close(self) -> None:
        self._conn.close()
