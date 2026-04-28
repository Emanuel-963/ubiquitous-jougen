"""SQLite-backed FeatureStore — drop-in replacement for :class:`src.feature_store.FeatureStore`.

Stores fitting history in the ``fitting_history`` table of the IonFlow
SQLite database.  The public API is intentionally identical to
:class:`FeatureStore` / :class:`FittingHistory` so callers can swap
implementations without changing call-sites.

Typical usage
-------------
::

    from src.db.feature_store_v2 import FeatureStoreV2

    store = FeatureStoreV2("data/ionflow.db")
    store.add_record({
        "sample_id": "sample_001.txt",
        "circuit_name": "Randles-CPE-W",
        "bic": -120.5,
        "confidence": 0.78,
        "logf_slope_low": -0.42,
        ...
    })

    best = store.best_circuit_for_features(my_spectral_features)
    print(store.summary_text())
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.db.migrations import run_migrations
from src.db.schema import init_db

logger = logging.getLogger(__name__)

# Spectral feature keys used for distance-based neighbour search.
_SPECTRAL_KEYS: List[str] = [
    "logf_slope_low",
    "logf_slope_high",
    "phase_min",
    "phase_max",
    "phase_range",
    "freq_at_phase_min",
    "mag_range",
    "zreal_min",
    "zreal_max",
]


def _sf(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _row_to_record(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a DB row to the dict format expected by callers."""
    d = dict(row)

    # Expand JSON blobs back to dicts
    for json_key in ("spectral_features_json", "circuit_params_json"):
        raw = d.pop(json_key, None)
        target = json_key.replace("_json", "")
        try:
            d[target] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            d[target] = {}

    # Flatten spectral features into top-level keys for backward compat
    sf = d.get("spectral_features") or {}
    for k in _SPECTRAL_KEYS:
        if k in sf:
            d[k] = sf[k]

    return d


class FeatureStoreV2:
    """SQLite-backed fitting-history store.

    Implements the same interface as :class:`src.feature_store.FeatureStore`
    and the query helpers from :class:`src.feature_store.FittingHistory`.

    Parameters
    ----------
    db_path:
        Path to the SQLite ``.db`` file.
    """

    def __init__(self, db_path: str | Path = "data/ionflow.db") -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection = init_db(self._db_path)
        run_migrations(self._conn)

    # ── Write ─────────────────────────────────────────────────────────

    def add_record(self, record: Dict[str, Any]) -> None:
        """Append a single fitting record.

        Required keys: ``sample_id``, ``circuit_name``.
        All :data:`_SPECTRAL_KEYS` present in *record* are stored in the
        ``spectral_features_json`` column and are also searchable by
        :meth:`similar_samples`.
        """
        if not record.get("sample_id") or not record.get("circuit_name"):
            logger.debug("FeatureStoreV2: skipping record missing required keys")
            return

        spectral = {k: record.get(k) for k in _SPECTRAL_KEYS if k in record}
        # Also support nested "spectral_features" dict
        if isinstance(record.get("spectral_features"), dict):
            spectral.update(record["spectral_features"])

        params = record.get("circuit_params", record.get("params", {})) or {}

        self._conn.execute(
            "INSERT INTO fitting_history"
            " (sample_id, circuit_name, bic, confidence,"
            "  spectral_features_json, circuit_params_json, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                str(record["sample_id"]),
                str(record["circuit_name"]),
                _sf(record.get("bic")),
                _sf(record.get("confidence")),
                json.dumps(spectral, default=str),
                json.dumps(params, default=str),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def add_records(self, records: List[Dict[str, Any]]) -> None:
        """Append a batch of records (calls :meth:`add_record` for each)."""
        for r in records:
            self.add_record(r)

    # ── Read ──────────────────────────────────────────────────────────

    @property
    def records(self) -> List[Dict[str, Any]]:
        """All records, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM fitting_history ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM fitting_history").fetchone()
        return row[0] if row else 0

    def __bool__(self) -> bool:
        return len(self) > 0

    def query(
        self,
        circuit_name: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return records matching the given filters."""
        conditions: List[str] = []
        params: List[Any] = []
        if circuit_name is not None:
            conditions.append("circuit_name = ?")
            params.append(circuit_name)
        if sample_id is not None:
            conditions.append("sample_id = ?")
            params.append(sample_id)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = self._conn.execute(
            f"SELECT * FROM fitting_history {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def unique_circuits(self) -> List[str]:
        """All distinct circuit names in the store."""
        rows = self._conn.execute(
            "SELECT DISTINCT circuit_name FROM fitting_history ORDER BY circuit_name"
        ).fetchall()
        return [r[0] for r in rows]

    def unique_samples(self) -> List[str]:
        """All distinct sample IDs in the store."""
        rows = self._conn.execute(
            "SELECT DISTINCT sample_id FROM fitting_history ORDER BY sample_id"
        ).fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        """Delete all fitting-history records."""
        self._conn.execute("DELETE FROM fitting_history")
        self._conn.commit()

    def reload(self) -> None:
        """No-op — data is always fresh from the DB."""

    # ── FittingHistory API ────────────────────────────────────────────

    def similar_samples(
        self,
        features: Dict[str, float],
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the *n* records whose spectral features are closest to *features*.

        Distance is Euclidean over the shared key subset.  Records with no
        overlap with *features* are excluded.

        Parameters
        ----------
        features:
            Dict mapping spectral feature names to float values.
        n:
            Maximum number of neighbours to return.

        Returns
        -------
        list of record dicts, sorted by ascending distance.
        """
        rows = self._conn.execute(
            "SELECT * FROM fitting_history"
            " WHERE spectral_features_json IS NOT NULL"
            "   AND spectral_features_json != '{}'"
        ).fetchall()
        if not rows:
            return []

        candidates: List[tuple] = []
        for row in rows:
            rec = _row_to_record(row)
            sf = rec.get("spectral_features") or {}
            common = {k for k in features if k in sf and sf[k] is not None}
            if not common:
                continue
            dist = float(
                np.sqrt(sum((features[k] - float(sf[k])) ** 2 for k in common))
            )
            candidates.append((dist, rec))

        candidates.sort(key=lambda x: x[0])
        return [r for _, r in candidates[:n]]

    def circuit_stats(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate statistics per circuit name.

        Returns
        -------
        dict
            ``{circuit_name: {count, mean_bic, mean_confidence, pct}}``
        """
        rows = self._conn.execute(
            "SELECT circuit_name,"
            "       COUNT(*)        AS cnt,"
            "       AVG(bic)        AS mean_bic,"
            "       AVG(confidence) AS mean_conf"
            " FROM fitting_history GROUP BY circuit_name"
        ).fetchall()

        total = sum(r["cnt"] for r in rows)
        result: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            result[r["circuit_name"]] = {
                "count": r["cnt"],
                "mean_bic": r["mean_bic"],
                "mean_confidence": r["mean_conf"],
                "pct": round(100.0 * r["cnt"] / total, 1) if total else 0.0,
            }
        return result

    def best_circuit_for_features(
        self,
        features: Dict[str, float],
        n: int = 10,
    ) -> Optional[str]:
        """Return the most common circuit among the *n* nearest neighbours.

        Returns ``None`` when the store is empty.
        """
        similar = self.similar_samples(features, n=n)
        if not similar:
            return None
        counts: Dict[str, int] = {}
        for rec in similar:
            name = rec.get("circuit_name", "")
            counts[name] = counts.get(name, 0) + 1
        return max(counts, key=lambda k: counts[k])

    def summary_text(self) -> str:
        """Human-readable summary in Portuguese."""
        total = len(self)
        if total == 0:
            return "Histórico SQLite vazio."

        stats = self.circuit_stats()
        lines = [f"Histórico SQLite: {total} registos"]
        for name, s in sorted(stats.items(), key=lambda x: -x[1]["count"]):
            bic_str = (
                f"BIC médio={s['mean_bic']:.1f}  " if s["mean_bic"] is not None else ""
            )
            conf_str = (
                f"conf={s['mean_confidence']:.2f}  "
                if s["mean_confidence"] is not None
                else ""
            )
            lines.append(
                f"  • {name}: {s['count']} usos  {bic_str}{conf_str}({s['pct']}%)"
            )
        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()
