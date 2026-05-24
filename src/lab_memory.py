"""Experimental memory store for lab historical similarity search.

This module persists condensed experiment signatures and retrieves
nearest historical matches for new samples.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_DB = "data/knowledge/lab_memory.db"


@dataclass
class SimilarityHit:
    """One historical match."""

    sample_name: str
    timestamp: str
    similarity: float
    notes: str


FEATURE_COLUMNS = [
    "rs",
    "rp",
    "c_mean",
    "chi2_over_nu",
    "confidence",
    "kk_valid",
]


def _safe_float(v: object, default: float = np.nan) -> float:
    try:
        out = float(v)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _normalize_vector(values: Sequence[float]) -> np.ndarray:
    vec = np.asarray(values, dtype=float)
    # Replace NaN with column-neutral values before norm
    vec = np.where(np.isfinite(vec), vec, 0.0)
    n = np.linalg.norm(vec)
    if n <= 1e-12:
        return vec
    return vec / n


class ExperimentalMemory:
    """SQLite-backed memory for experiment signatures."""

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_name TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    rs REAL,
                    rp REAL,
                    c_mean REAL,
                    chi2_over_nu REAL,
                    confidence REAL,
                    kk_valid REAL,
                    notes TEXT DEFAULT ''
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_sample ON memory(sample_name)"
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_memory_ts ON memory(ts)")
            con.commit()

    def add_signature(
        self,
        sample_name: str,
        signature: Dict[str, float],
        *,
        notes: str = "",
        timestamp: Optional[str] = None,
    ) -> int:
        """Insert one experiment signature into memory."""
        ts = timestamp or datetime.now().isoformat(timespec="seconds")
        payload = {k: _safe_float(signature.get(k, np.nan)) for k in FEATURE_COLUMNS}

        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO memory (
                    sample_name, ts, rs, rp, c_mean, chi2_over_nu,
                    confidence, kk_valid, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_name,
                    ts,
                    payload["rs"],
                    payload["rp"],
                    payload["c_mean"],
                    payload["chi2_over_nu"],
                    payload["confidence"],
                    payload["kk_valid"],
                    notes,
                ),
            )
            con.commit()
            return int(cur.lastrowid)

    def add_from_benchmark_table(
        self,
        bench_df: pd.DataFrame,
        *,
        notes: str = "",
    ) -> int:
        """Bulk insert signatures from canonical benchmark table."""
        if bench_df is None or bench_df.empty:
            return 0

        inserted = 0
        for _, row in bench_df.iterrows():
            sig = {k: _safe_float(row.get(k, np.nan)) for k in FEATURE_COLUMNS}
            self.add_signature(
                str(row.get("sample", f"sample_{inserted}")), sig, notes=notes
            )
            inserted += 1
        return inserted

    def all_signatures(self) -> pd.DataFrame:
        """Return all memory rows as DataFrame."""
        with self._connect() as con:
            df = pd.read_sql_query(
                "SELECT sample_name, ts, rs, rp, c_mean, chi2_over_nu, confidence, kk_valid, notes FROM memory",
                con,
            )
        return df

    def find_similar(
        self,
        query_signature: Dict[str, float],
        *,
        top_k: int = 5,
    ) -> List[SimilarityHit]:
        """Find most similar historical experiments via cosine similarity."""
        df = self.all_signatures()
        if df.empty:
            return []

        q = np.array(
            [_safe_float(query_signature.get(k, 0.0), 0.0) for k in FEATURE_COLUMNS],
            dtype=float,
        )
        qn = _normalize_vector(q)

        hits: List[SimilarityHit] = []
        for _, row in df.iterrows():
            x = np.array(
                [_safe_float(row.get(k, 0.0), 0.0) for k in FEATURE_COLUMNS],
                dtype=float,
            )
            xn = _normalize_vector(x)
            sim = float(np.dot(qn, xn))
            sim = max(-1.0, min(1.0, sim))
            # Map [-1, 1] -> [0, 1]
            sim01 = 0.5 * (sim + 1.0)
            hits.append(
                SimilarityHit(
                    sample_name=str(row.get("sample_name", "unknown")),
                    timestamp=str(row.get("ts", "")),
                    similarity=sim01,
                    notes=str(row.get("notes", "")),
                )
            )

        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits[: max(1, top_k)]


def similarity_report(
    memory: ExperimentalMemory,
    query_name: str,
    query_signature: Dict[str, float],
    *,
    top_k: int = 5,
) -> str:
    """Return a text report for historical similarity lookup."""
    hits = memory.find_similar(query_signature, top_k=top_k)

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  MEMORIA EXPERIMENTAL DO LABORATORIO")
    lines.append("=" * 72)
    lines.append(f"Consulta: {query_name}")
    lines.append("")

    if not hits:
        lines.append("Sem historico salvo para comparar.")
    else:
        lines.append("Ensaios historicos mais similares:")
        for i, hit in enumerate(hits, start=1):
            pct = 100.0 * hit.similarity
            lines.append(
                f"  {i}. {hit.sample_name}  | similaridade={pct:.1f}%  | {hit.timestamp}"
            )
            if hit.notes:
                lines.append(f"     notas: {hit.notes}")

        best = hits[0]
        lines.append("")
        lines.append(
            f"Conclusao: esta amostra se parece {best.similarity * 100:.1f}% com {best.sample_name}."
        )

    lines.append("=" * 72)
    return "\n".join(lines)
