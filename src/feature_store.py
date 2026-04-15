"""Feature Store & Fitting History for ML-driven circuit selection.

Persists every circuit-fitting record so that the ML selector (Day 6) can
learn from previous runs.  Two public classes:

* :class:`FeatureStore` — low-level append-only store backed by a JSON file.
* :class:`FittingHistory` — higher-level queries on top of the store
  (similar samples, circuit statistics, best-circuit lookups).

Typical flow
------------
::

    from src.feature_store import FeatureStore, FittingHistory

    store = FeatureStore("data/ml/fitting_history.json")
    store.add_record({
        "sample_id": "sample_001.txt",
        "timestamp": "2026-04-14T10:00:00",
        "spectral_features": {...},
        "circuit_name": "Randles-CPE-W",
        "params": {"Rs": 1.2, "Rp": 50.0, ...},
        "bic": -120.5,
        "confidence": 0.78,
        "user_label": None,
    })

    history = FittingHistory(store)
    similar = history.similar_samples(spectral_features, n=5)
    stats  = history.circuit_stats()
    best   = history.best_circuit_for_features(spectral_features, n=10)
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Keys expected in every record
_REQUIRED_KEYS = ("sample_id", "circuit_name")

# Keys used for the spectral feature vector (order matters for distance)
_SPECTRAL_KEYS: Sequence[str] = (
    "logf_slope_low",
    "logf_slope_high",
    "phase_min",
    "phase_max",
    "phase_range",
    "freq_at_phase_min",
    "mag_range",
    "zreal_min",
    "zreal_max",
)


# ══════════════════════════════════════════════════════════════════════
# FeatureStore — append-only JSON persistence
# ══════════════════════════════════════════════════════════════════════

class FeatureStore:
    """Append-only store of circuit-fitting records backed by JSON.

    Each record is a flat dict with at least ``sample_id`` and
    ``circuit_name``.  The full schema:

    +-----------------------+--------+-------------------------------------+
    | Key                   | Type   | Description                         |
    +=======================+========+=====================================+
    | sample_id             | str    | Filename / unique sample key        |
    | timestamp             | str    | ISO-8601 when the fitting ran       |
    | spectral_features     | dict   | 9 ML features from ``extract_eis…`` |
    | circuit_name          | str    | Best circuit name (BIC winner)      |
    | params                | dict   | Fitted parameter values             |
    | bic                   | float  | Bayesian Information Criterion      |
    | confidence            | float  | Softmax-based confidence 0–1        |
    | user_label            | str?   | Optional human-assigned label       |
    +-----------------------+--------+-------------------------------------+
    """

    def __init__(self, path: str | Path = "data/ml/fitting_history.json"):
        self.path = Path(path)
        self._records: List[Dict[str, Any]] = []
        if self.path.exists():
            self._load()

    # ── Core CRUD ────────────────────────────────────────────────────

    def add_record(self, record: Dict[str, Any]) -> None:
        """Append a record and persist to disk."""
        missing = [k for k in _REQUIRED_KEYS if k not in record]
        if missing:
            raise ValueError(f"Record missing required keys: {missing}")

        # Auto-fill timestamp if absent
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat(timespec="seconds")

        self._records.append(record)
        self._save()
        logger.debug("FeatureStore: added record for '%s'", record["sample_id"])

    def add_records(self, records: Sequence[Dict[str, Any]]) -> None:
        """Bulk-add records (single save at the end)."""
        for rec in records:
            missing = [k for k in _REQUIRED_KEYS if k not in rec]
            if missing:
                raise ValueError(f"Record missing required keys: {missing}")
            if "timestamp" not in rec:
                rec["timestamp"] = datetime.now().isoformat(timespec="seconds")
            self._records.append(rec)
        self._save()

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of all records."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return len(self._records) > 0

    # ── Query helpers ────────────────────────────────────────────────

    def query(
        self,
        *,
        circuit_name: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Simple filter — returns records matching all supplied criteria."""
        results = self._records
        if circuit_name is not None:
            results = [r for r in results if r.get("circuit_name") == circuit_name]
        if sample_id is not None:
            results = [r for r in results if r.get("sample_id") == sample_id]
        return results

    def unique_circuits(self) -> List[str]:
        """Return sorted list of distinct circuit names."""
        return sorted({r.get("circuit_name", "") for r in self._records})

    def unique_samples(self) -> List[str]:
        """Return sorted list of distinct sample IDs."""
        return sorted({r.get("sample_id", "") for r in self._records})

    # ── Persistence ──────────────────────────────────────────────────

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._records, fh, default=_json_default,
                      indent=2, ensure_ascii=False)

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self._records = data
            else:
                logger.warning("FeatureStore: unexpected JSON root type, resetting.")
                self._records = []
        except Exception as exc:
            logger.warning("FeatureStore: failed to load %s: %s", self.path, exc)
            self._records = []

    def clear(self) -> None:
        """Remove all records and delete the JSON file."""
        self._records.clear()
        if self.path.exists():
            self.path.unlink()

    def reload(self) -> None:
        """Re-read from disk (useful if another process wrote)."""
        self._records.clear()
        if self.path.exists():
            self._load()


# ══════════════════════════════════════════════════════════════════════
# FittingHistory — higher-level analytics on the store
# ══════════════════════════════════════════════════════════════════════

class FittingHistory:
    """Query layer over a :class:`FeatureStore` for ML-driven analysis.

    All methods are read-only — they never mutate the store.
    """

    def __init__(self, store: FeatureStore):
        self.store = store

    # ── Similar samples via normalised Euclidean distance ────────────

    def similar_samples(
        self,
        features: Dict[str, float],
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the *n* most spectrally-similar records.

        Distance is Euclidean in a z-score–normalised 9-D feature space.
        Records without ``spectral_features`` are silently skipped.
        """
        records = self.store.records
        if not records:
            return []

        # Build matrix of stored feature vectors
        valid: List[tuple] = []  # (index, vector)
        for i, rec in enumerate(records):
            sf = rec.get("spectral_features")
            if not isinstance(sf, dict):
                continue
            vec = np.array([float(sf.get(k, np.nan)) for k in _SPECTRAL_KEYS])
            if np.all(np.isfinite(vec)):
                valid.append((i, vec))

        if not valid:
            return []

        mat = np.array([v for _, v in valid])  # (N, 9)

        # Z-score normalisation
        mu = mat.mean(axis=0)
        sigma = mat.std(axis=0)
        sigma[sigma == 0] = 1.0  # avoid div-by-zero
        mat_norm = (mat - mu) / sigma

        # Normalise the query vector the same way
        q = np.array([float(features.get(k, np.nan)) for k in _SPECTRAL_KEYS])
        if not np.all(np.isfinite(q)):
            return []
        q_norm = (q - mu) / sigma

        # Euclidean distances
        dists = np.linalg.norm(mat_norm - q_norm, axis=1)
        top_idx = np.argsort(dists)[:n]

        result = []
        for idx in top_idx:
            orig_i, _ = valid[idx]
            rec = dict(records[orig_i])
            rec["_distance"] = float(dists[idx])
            result.append(rec)
        return result

    # ── Circuit statistics ───────────────────────────────────────────

    def circuit_stats(self) -> Dict[str, Dict[str, Any]]:
        """Per-circuit aggregate statistics.

        Returns
        -------
        dict
            ``{circuit_name: {count, mean_bic, mean_confidence, pct}}``
        """
        records = self.store.records
        if not records:
            return {}

        counter: Dict[str, List[Dict[str, Any]]] = {}
        for rec in records:
            name = rec.get("circuit_name", "unknown")
            counter.setdefault(name, []).append(rec)

        total = len(records)
        stats: Dict[str, Dict[str, Any]] = {}
        for name, recs in counter.items():
            bics = [r.get("bic") for r in recs
                    if isinstance(r.get("bic"), (int, float)) and np.isfinite(r["bic"])]
            confs = [r.get("confidence") for r in recs
                     if isinstance(r.get("confidence"), (int, float)) and np.isfinite(r["confidence"])]
            stats[name] = {
                "count": len(recs),
                "mean_bic": float(np.mean(bics)) if bics else None,
                "mean_confidence": float(np.mean(confs)) if confs else None,
                "pct": len(recs) / total * 100 if total else 0,
            }
        return stats

    # ── Best circuit for given features ──────────────────────────────

    def best_circuit_for_features(
        self,
        features: Dict[str, float],
        n: int = 10,
    ) -> Optional[str]:
        """Return the circuit name most commonly chosen by similar samples.

        Finds the *n* most similar historical records and returns the
        most-frequent ``circuit_name`` among them.  Returns ``None``
        when the store is empty or has no valid feature vectors.
        """
        similar = self.similar_samples(features, n=n)
        if not similar:
            return None
        names = [r.get("circuit_name") for r in similar if r.get("circuit_name")]
        if not names:
            return None
        counter = Counter(names)
        return counter.most_common(1)[0][0]

    # ── Summary text (for GUI / reports) ─────────────────────────────

    def summary_text(
        self,
        features: Dict[str, float],
        n: int = 10,
    ) -> str:
        """Generate a human-readable summary of what the history suggests.

        Intended for display in the GUI log or circuit-fitting report.
        """
        total = len(self.store)
        if total == 0:
            return "Sem histórico de fittings — recomendação baseada em heurística."

        similar = self.similar_samples(features, n=n)
        if not similar:
            return (
                f"Histórico tem {total} registros, mas nenhum com features "
                f"espectrais válidas para comparação."
            )

        names = [r.get("circuit_name") for r in similar if r.get("circuit_name")]
        if not names:
            return "Nenhum circuito encontrado nos registros similares."

        counter = Counter(names)
        best_name, best_count = counter.most_common(1)[0]
        pct = best_count / len(names) * 100

        slope_low = features.get("logf_slope_low")
        phase_min = features.get("phase_min")

        parts = [
            f"Com base em {total} amostras anteriores",
        ]
        if len(similar) < total:
            parts[0] += f" ({len(similar)} mais similares)"

        parts.append(
            f"o modelo {best_name} tem {pct:.0f}% de probabilidade de ser o melhor."
        )

        if slope_low is not None and phase_min is not None:
            parts.append(
                f"Amostras com slope_low={slope_low:.2f} e phase_min={phase_min:.1f}° "
                f"tipicamente convergem para {best_name}."
            )

        return ", ".join(parts)


# ══════════════════════════════════════════════════════════════════════
# Pipeline integration helper
# ══════════════════════════════════════════════════════════════════════

def record_from_shortlist_result(
    sample_id: str,
    circ_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a :class:`FeatureStore` record from ``run_shortlist_fit()`` output.

    Returns ``None`` if the fitting failed or has no best result.
    """
    best = circ_result.get("best")
    if not best or not best.get("success"):
        return None

    features = circ_result.get("features") or {}

    return {
        "sample_id": sample_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "spectral_features": {k: _safe_float(features.get(k)) for k in _SPECTRAL_KEYS},
        "circuit_name": best.get("template", "unknown"),
        "params": _safe_params(best.get("params")),
        "bic": _safe_float(best.get("bic")),
        "confidence": _safe_float(best.get("confidence")),
        "user_label": None,
    }


# ── JSON helpers ─────────────────────────────────────────────────────

def _safe_float(val: Any) -> Optional[float]:
    """Convert to float, replacing NaN/Inf with None for JSON safety."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_params(params: Any) -> Dict[str, Optional[float]]:
    """Sanitise a parameter dict for JSON serialisation."""
    if not isinstance(params, dict):
        return {}
    return {str(k): _safe_float(v) for k, v in params.items()}


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for ``json.dump``."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
