"""ML-driven circuit shortlist selector.

Replaces the deterministic heuristic in ``circuit_fitting.shortlist_circuits``
with a **trainable** classifier.  When enough historical records exist in the
:class:`~src.feature_store.FeatureStore` (≥ 30 by default) a
:class:`sklearn.ensemble.RandomForestClassifier` is trained on the 9 spectral
features extracted by :func:`~src.circuit_fitting.extract_eis_features_for_ml`.

Public API
----------
::

    selector = CircuitMLSelector()
    selector.train(feature_store)           # fit the forest
    ranked  = selector.predict(features)    # ['Randles-CPE-W', 'Two-Arc-CPE', …]
    probs   = selector.confidence(features) # {'Randles-CPE-W': 0.78, …}
    text    = selector.explain(features)    # human-readable recommendation

When the store has fewer than ``min_samples`` records the selector
transparently falls back to the existing rule-based heuristic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# The 9 spectral keys must match feature_store._SPECTRAL_KEYS ordering.
_FEATURE_KEYS = (
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

# Minimum number of labelled samples before ML kicks in.
_DEFAULT_MIN_SAMPLES = 30


# ══════════════════════════════════════════════════════════════════════
# CircuitMLSelector
# ══════════════════════════════════════════════════════════════════════

class CircuitMLSelector:
    """Trainable circuit shortlist selector backed by RandomForest.

    Parameters
    ----------
    min_samples : int
        Minimum store records required to train.  Below this threshold
        every call transparently falls back to the rule-based heuristic.
    n_estimators : int
        Number of trees in the Random Forest.
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
        n_estimators: int = 100,
        random_state: int | None = 42,
    ):
        self.min_samples = min_samples
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._model: Any = None          # RandomForestClassifier or None
        self._classes: List[str] = []    # class labels (circuit names)
        self._trained: bool = False
        self._n_train: int = 0           # how many samples were used

    # ── Properties ───────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        """``True`` when the forest has been fitted on enough data."""
        return self._trained

    @property
    def n_training_samples(self) -> int:
        """Number of samples the last ``train()`` used."""
        return self._n_train

    @property
    def classes(self) -> List[str]:
        """Circuit names the model can predict."""
        return list(self._classes)

    # ── Train ────────────────────────────────────────────────────────

    def train(self, feature_store: Any) -> bool:
        """Train the classifier from a :class:`FeatureStore`.

        Returns ``True`` if the model was successfully fitted, ``False``
        if there were not enough valid records.
        """
        X, y = self._build_dataset(feature_store)

        if len(X) < self.min_samples:
            logger.info(
                "CircuitMLSelector: only %d valid records (need %d) — "
                "falling back to heuristic.",
                len(X), self.min_samples,
            )
            self._trained = False
            self._model = None
            self._classes = []
            self._n_train = 0
            return False

        # Ensure at least 2 classes
        unique_classes = set(y)
        if len(unique_classes) < 2:
            logger.info(
                "CircuitMLSelector: only %d class(es) — need ≥ 2 for training.",
                len(unique_classes),
            )
            self._trained = False
            self._model = None
            self._classes = []
            self._n_train = 0
            return False

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=2,
        )
        clf.fit(X, y)

        self._model = clf
        self._classes = list(clf.classes_)
        self._trained = True
        self._n_train = len(X)

        logger.info(
            "CircuitMLSelector: trained on %d samples, %d classes (%s).",
            len(X), len(self._classes), ", ".join(self._classes),
        )
        return True

    # ── Predict ──────────────────────────────────────────────────────

    def predict(self, features: Dict[str, float], top_n: int = 3) -> List[str]:
        """Return circuit names ranked by predicted probability.

        Falls back to an empty list when the model is not trained
        (caller should use the heuristic shortlist instead).
        """
        if not self._trained or self._model is None:
            return []

        x = self._features_to_array(features)
        if x is None:
            return []

        probs = self._model.predict_proba(x)[0]
        ranked_idx = np.argsort(probs)[::-1]
        return [self._classes[i] for i in ranked_idx[:top_n]]

    # ── Confidence ───────────────────────────────────────────────────

    def confidence(self, features: Dict[str, float]) -> Dict[str, float]:
        """Return ``{circuit_name: probability}`` for all known circuits.

        Returns an empty dict when the model is not trained.
        """
        if not self._trained or self._model is None:
            return {}

        x = self._features_to_array(features)
        if x is None:
            return {}

        probs = self._model.predict_proba(x)[0]
        return {name: float(p) for name, p in zip(self._classes, probs)}

    # ── Explain ──────────────────────────────────────────────────────

    def explain(self, features: Dict[str, float]) -> str:
        """Generate a human-readable explanation of the prediction.

        Includes top circuit, probability, feature importances, and a
        note about key spectral indicators.
        """
        if not self._trained or self._model is None:
            return (
                "Modelo ML não treinado — usando heurística de shortlist. "
                f"Necessário ≥ {self.min_samples} amostras no histórico."
            )

        probs = self.confidence(features)
        if not probs:
            return "Não foi possível calcular probabilidades (features inválidas)."

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_prob = ranked[0]

        # Feature importances
        importances = dict(
            zip(_FEATURE_KEYS, self._model.feature_importances_)
        )
        top_feats = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]

        parts = [
            f"Com base em {self._n_train} amostras anteriores com perfil espectral similar, "
            f"o modelo {best_name} tem {best_prob * 100:.0f}% de probabilidade de ser o melhor.",
        ]

        # Add context about key spectral features
        slope_low = features.get("logf_slope_low")
        phase_min = features.get("phase_min")
        if slope_low is not None and phase_min is not None:
            try:
                parts.append(
                    f"Amostras com slope_low={float(slope_low):.2f} e "
                    f"phase_min={float(phase_min):.1f}° tipicamente "
                    f"convergem para {best_name}."
                )
            except (TypeError, ValueError):
                pass

        # Top feature importances
        feat_strs = [f"{name} ({imp:.0%})" for name, imp in top_feats]
        parts.append(
            f"Features mais influentes: {', '.join(feat_strs)}."
        )

        if len(ranked) > 1:
            second_name, second_prob = ranked[1]
            parts.append(
                f"Alternativa: {second_name} ({second_prob * 100:.0f}%)."
            )

        return " ".join(parts)

    # ── Feature importance ───────────────────────────────────────────

    def feature_importances(self) -> Dict[str, float]:
        """Return ``{feature_name: importance}`` from the trained forest.

        Returns an empty dict when the model is not trained.
        """
        if not self._trained or self._model is None:
            return {}
        return dict(zip(_FEATURE_KEYS, self._model.feature_importances_))

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _build_dataset(feature_store: Any):
        """Extract (X, y) arrays from a FeatureStore.

        Only records with a complete set of finite spectral features and
        a non-empty ``circuit_name`` are included.
        """
        records = feature_store.records
        X_list: List[List[float]] = []
        y_list: List[str] = []

        for rec in records:
            name = rec.get("circuit_name")
            sf = rec.get("spectral_features")
            if not name or not isinstance(sf, dict):
                continue

            vec = [float(sf.get(k, float("nan"))) for k in _FEATURE_KEYS]
            if not all(np.isfinite(v) for v in vec):
                continue

            X_list.append(vec)
            y_list.append(name)

        X = np.array(X_list) if X_list else np.empty((0, len(_FEATURE_KEYS)))
        y = np.array(y_list) if y_list else np.empty((0,))
        return X, y

    @staticmethod
    def _features_to_array(features: Dict[str, float]):
        """Convert a spectral-features dict to a (1, 9) array.

        Returns ``None`` if any value is non-finite.
        """
        vec = [float(features.get(k, float("nan"))) for k in _FEATURE_KEYS]
        if not all(np.isfinite(v) for v in vec):
            return None
        return np.array(vec).reshape(1, -1)
