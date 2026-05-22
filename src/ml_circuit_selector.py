"""ML-driven circuit shortlist selector.

Replaces the deterministic heuristic in ``circuit_fitting.shortlist_circuits``
with a **trainable** classifier.  When enough historical records exist in the
:class:`~src.feature_store.FeatureStore` (≥ 30 by default) a
:class:`sklearn.ensemble.RandomForestClassifier` is trained on the 12 spectral
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

# The 12 spectral keys must match feature_store._SPECTRAL_KEYS ordering.
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
    # KK validation quality features (nan-safe)
    "kk_residual_real",
    "kk_residual_imag",
    "kk_valid",
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

        self._model: Any = None  # RandomForestClassifier or None
        self._classes: List[str] = []  # class labels (circuit names)
        self._trained: bool = False
        self._n_train: int = 0  # how many samples were used

    # ── Properties ───────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        """Check whether the Random Forest has been successfully trained.

        Returns
        -------
        bool
            ``True`` if :meth:`train` has been called and the store
            contained at least ``min_samples`` records with ≥ 2 distinct
            circuit classes; ``False`` otherwise.
        """
        return self._trained

    @property
    def n_training_samples(self) -> int:
        """Number of valid samples used in the most recent training run.

        Returns
        -------
        int
            Count of records with complete, finite spectral features that
            were passed to ``RandomForestClassifier.fit``.  Returns 0 if
            the model has never been trained.
        """
        return self._n_train

    @property
    def classes(self) -> List[str]:
        """Circuit names the trained model can predict.

        Returns
        -------
        list[str]
            Sorted class labels learnt by the Random Forest.  Returns an
            empty list when the model is not trained.
        """
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
                len(X),
                self.min_samples,
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
            len(X),
            len(self._classes),
            ", ".join(self._classes),
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
        importances = dict(zip(_FEATURE_KEYS, self._model.feature_importances_))
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
        parts.append(f"Features mais influentes: {', '.join(feat_strs)}.")

        if len(ranked) > 1:
            second_name, second_prob = ranked[1]
            parts.append(f"Alternativa: {second_name} ({second_prob * 100:.0f}%).")

        return " ".join(parts)

    # ── Feature importance ───────────────────────────────────────────

    def feature_importances(self) -> Dict[str, float]:
        """Return ``{feature_name: importance}`` from the trained forest.

        Returns an empty dict when the model is not trained.
        Supports both raw RandomForestClassifier and CalibratedClassifierCV wrappers.
        """
        if not self._trained or self._model is None:
            return {}
        # CalibratedClassifierCV stores per-fold estimators in calibrated_classifiers_
        if hasattr(self._model, "calibrated_classifiers_"):
            try:
                fi = np.mean(
                    [
                        c.estimator.feature_importances_
                        for c in self._model.calibrated_classifiers_
                    ],
                    axis=0,
                )
                return dict(zip(_FEATURE_KEYS, fi))
            except Exception:
                pass
        if hasattr(self._model, "feature_importances_"):
            return dict(zip(_FEATURE_KEYS, self._model.feature_importances_))
        return {}

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
        """Convert a spectral-features dict to a (1, 12) array.

        Returns ``None`` if any value is non-finite.
        """
        vec = [float(features.get(k, float("nan"))) for k in _FEATURE_KEYS]
        if not all(np.isfinite(v) for v in vec):
            return None
        return np.array(vec).reshape(1, -1)

    # ── Persistence ──────────────────────────────────────────────────

    def save_model(self, path) -> None:
        """Serialize the trained model to a joblib file.

        Parameters
        ----------
        path : str or Path
            Destination file (conventionally ``*.joblib``).

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._trained or self._model is None:
            raise RuntimeError("No trained model to save — call train() first.")

        from pathlib import Path as _Path

        import joblib

        _path = _Path(path)
        _path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "classes": self._classes,
            "n_train": self._n_train,
            "min_samples": self.min_samples,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
        joblib.dump(payload, _path)
        logger.info("CircuitMLSelector: model saved to %s", _path)

    @classmethod
    def load_model(cls, path) -> "CircuitMLSelector":
        """Load a serialized model and return a ready-to-use selector.

        Parameters
        ----------
        path : str or Path
            File previously created by :meth:`save_model`.

        Returns
        -------
        CircuitMLSelector
            Fully initialized instance with the loaded model.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        from pathlib import Path as _Path

        import joblib

        _path = _Path(path)
        if not _path.exists():
            raise FileNotFoundError(f"Model file not found: {_path}")
        payload = joblib.load(_path)
        instance = cls(
            min_samples=payload.get("min_samples", _DEFAULT_MIN_SAMPLES),
            n_estimators=payload.get("n_estimators", 100),
            random_state=payload.get("random_state", 42),
        )
        instance._model = payload["model"]
        instance._classes = payload["classes"]
        instance._n_train = payload["n_train"]
        instance._trained = True
        logger.info(
            "CircuitMLSelector: loaded model from %s (%d samples, %d classes)",
            _path,
            instance._n_train,
            len(instance._classes),
        )
        return instance

    # ── Train from synthetic files ────────────────────────────────────

    @classmethod
    def train_from_synthetic(
        cls,
        data_dir,
        model_path=None,
        min_samples_per_class: int = 5,
        cv_folds: int = 5,
    ) -> dict:
        """Train the classifier from synthetic EIS files (``SYN_*.txt``).

        Scans *data_dir* for files matching ``SYN_*.txt``, extracts the
        circuit label from the filename, loads each file with
        :func:`~src.loader.load_eis_file`, computes spectral features, and
        fits a :class:`~sklearn.ensemble.RandomForestClassifier`.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing ``SYN_*.txt`` files (typically ``data/raw``).
        model_path : str or Path, optional
            Where to save the trained model.  Defaults to
            ``data/knowledge/ml_classifier.joblib``.
        min_samples_per_class : int
            Classes with fewer samples than this are excluded from training.
        cv_folds : int
            Number of stratified cross-validation folds for the accuracy estimate.

        Returns
        -------
        dict
            Keys: ``n_samples``, ``n_classes``, ``classes``, ``cv_accuracy``,
            ``model_path``, ``skipped``.
        """
        from collections import Counter
        from pathlib import Path as _Path

        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        from src.circuit_fitting import extract_eis_features_for_ml
        from src.loader import load_eis_file

        data_dir = _Path(data_dir)
        if model_path is None:
            model_path = _Path("data/knowledge/ml_classifier.joblib")
        else:
            model_path = _Path(model_path)

        syn_files = sorted(data_dir.glob("SYN_*.txt"))
        if not syn_files:
            raise FileNotFoundError(
                f"No SYN_*.txt files found in '{data_dir}'. "
                "Generate synthetic data first using 'Gerar Dados Sintéticos'."
            )

        X_list: List[List[float]] = []
        y_list: List[str] = []
        skipped = 0

        for fpath in syn_files:
            # Extract label from filename: SYN_{safe_name}_{NNN}.txt
            # safe_name = circuit_name.replace(" ", "_")
            stem = fpath.stem  # e.g. "SYN_Randles-CPE-W_001"
            without_prefix = stem[4:]  # strip "SYN_"
            parts = without_prefix.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                logger.debug("Skipping unexpected SYN filename: %s", fpath.name)
                skipped += 1
                continue
            circuit_name = parts[0].replace("_", " ")

            try:
                df = load_eis_file(str(fpath))
                feats = extract_eis_features_for_ml(df)
                vec = [float(feats.get(k, float("nan"))) for k in _FEATURE_KEYS]
                if not all(np.isfinite(v) for v in vec):
                    skipped += 1
                    continue
                X_list.append(vec)
                y_list.append(circuit_name)
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath.name, exc)
                skipped += 1

        if not X_list:
            raise ValueError(
                "No valid spectral features could be extracted from synthetic files."
            )

        X = np.array(X_list)
        y = np.array(y_list)

        # Drop classes with too few samples
        counts = Counter(y_list)
        valid_classes = {c for c, n in counts.items() if n >= min_samples_per_class}
        mask = np.array([yi in valid_classes for yi in y])
        X = X[mask]
        y = y[mask]

        n_unique = len(np.unique(y))
        if n_unique < 2:
            raise ValueError(
                f"Need ≥ 2 circuit classes with ≥ {min_samples_per_class} samples each. "
                f"Found: {dict(counts)}"
            )

        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=2,
        )

        # Cross-validation accuracy estimate (on uncalibrated RF)
        cv_accuracy = 0.0
        if len(X) >= cv_folds * n_unique:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
            cv_accuracy = float(scores.mean())

        # Fit base RF, then wrap with probability calibration (Platt scaling)
        clf.fit(X, y)
        n_cal_folds = min(3, max(2, len(X) // (n_unique * 5)))
        try:
            calibrated = CalibratedClassifierCV(clf, cv=n_cal_folds, method="sigmoid")
            calibrated.fit(X, y)
            final_clf = calibrated
        except Exception as exc:
            logger.warning(
                "Probability calibration failed (%s); using uncalibrated RF.", exc
            )
            final_clf = clf

        instance = cls()
        instance._model = final_clf
        instance._classes = list(clf.classes_)
        instance._trained = True
        instance._n_train = len(X)
        instance.save_model(model_path)

        logger.info(
            "CircuitMLSelector: trained on %d samples, %d classes, CV acc=%.1f%%",
            len(X),
            n_unique,
            cv_accuracy * 100,
        )

        return {
            "n_samples": len(X),
            "n_classes": n_unique,
            "classes": list(clf.classes_),
            "cv_accuracy": cv_accuracy,
            "model_path": str(model_path),
            "skipped": skipped,
        }


# ── Circuit family taxonomy ───────────────────────────────────────────────

_CIRCUIT_FAMILIES: Dict[str, List[str]] = {
    "simple_capacitive": [
        "Simple-RC",
        "CPE-Simple",
        "Rs-RC-W",
        "Rs-RC-Wfinite",
        "Rs-TLM",
    ],
    "randles_single": [
        "Randles-CPE-W",
        "Warburg-Finite",
        "Warburg-Short",
        "Gerischer",
        "Rs-ZARC-CPE",
        "Pseudo-Capacitance-CPE",
    ],
    "multi_arc": [
        "Two-Arc-CPE",
        "ZARC-ZARC-W",
        "Three-ZARC",
        "Rs-ZARC-ZARC-Wfinite",
        "Rs-ZARC-ZARC-Wshort",
        "Rs-ZARC-ZARC-Gerischer",
        "Rs-ZARC-ZARC-CPE",
    ],
    "inductive": [
        "Inductive-CPE",
        "Rs-L-ZARC-W",
        "Rs-L-ZARC-Wfinite",
        "Rs-L-ZARC-ZARC",
        "Rs-L-ZARC-ZARC-W",
    ],
    "coating": [
        "Coating-CPE",
        "Porous-Coating-TLM",
        "Rs-RC-ZARC-W",
        "Rs-RC-ZARC-Wfinite",
        "Rs-ZARC-RC-Wfinite",
        "Rs-ZARC-RC-Wshort",
    ],
    "porous_tlm": [
        "De-Levie-TLM",
        "MXene-Intercalation",
        "Rs-ZARC-TLM",
        "Rs-ZARC-ZARC-TLM",
        "Rs-ZARC-TLM-W",
    ],
    "solid_state": [
        "Rs-ZARC-ZARC-ZARC-W",
        "Rs-ZARC-ZARC-ZARC-Wfinite",
        "Rs-ZARC-ZARC-ZARC-Gerischer",
    ],
}

#: Reverse mapping: circuit name → family name
_CIRCUIT_TO_FAMILY: Dict[str, str] = {
    circuit: family
    for family, circuits in _CIRCUIT_FAMILIES.items()
    for circuit in circuits
}


class HierarchicalCircuitSelector:
    """Two-stage circuit classifier: (1) family, then (2) specific circuit.

    Reduces inter-class confusion by first routing the spectrum to one of
    the 7 morphological families, then running a family-specific classifier
    to identify the exact circuit topology.

    When a family-specific model cannot be trained (too few samples) the
    class falls back to the flat *CircuitMLSelector* prediction.

    Parameters
    ----------
    None — call :meth:`train` or :meth:`train_from_synthetic` to fit.
    """

    def __init__(self) -> None:
        self._family_clf = None  # flat RF: features → family
        self._circuit_clfs: Dict[str, object] = {}  # family → RF: features → circuit
        self._family_classes: List[str] = []
        self._circuit_classes: Dict[str, List[str]] = {}
        self._fallback: Optional[CircuitMLSelector] = None
        self._trained: bool = False
        self._n_train: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, feature_store) -> None:
        """Train both classifiers from a :class:`~src.feature_store.FeatureStore`.

        Parameters
        ----------
        feature_store :
            Object with an ``all_records()`` method that returns a list of
            dicts, each with keys matching ``_FEATURE_KEYS`` plus
            ``"circuit_name"``.
        """
        from collections import Counter

        from sklearn.ensemble import RandomForestClassifier

        records = feature_store.all_records()
        if not records:
            raise ValueError("FeatureStore is empty — nothing to train on.")

        X_list: List[List[float]] = []
        y_circuit: List[str] = []
        y_family: List[str] = []

        for rec in records:
            vec = [float(rec.get(k, float("nan"))) for k in _FEATURE_KEYS]
            if not all(np.isfinite(v) for v in vec):
                continue
            circuit = rec.get("circuit_name", "")
            family = _CIRCUIT_TO_FAMILY.get(circuit, "unknown")
            if family == "unknown":
                continue
            X_list.append(vec)
            y_circuit.append(circuit)
            y_family.append(family)

        if len(X_list) < 10:
            raise ValueError(
                f"Only {len(X_list)} valid records after filtering. "
                "Need ≥ 10 to train."
            )

        X = np.array(X_list)
        y_fam = np.array(y_family)
        y_cir = np.array(y_circuit)

        # Stage 1: family classifier
        self._family_clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            max_depth=6,
            min_samples_leaf=2,
        )
        self._family_clf.fit(X, y_fam)
        self._family_classes = list(self._family_clf.classes_)

        # Stage 2: per-family circuit classifiers
        self._circuit_clfs = {}
        self._circuit_classes = {}
        for fam in np.unique(y_fam):
            mask = y_fam == fam
            X_fam = X[mask]
            y_fam_cir = y_cir[mask]
            unique_circuits = np.unique(y_fam_cir)
            if len(unique_circuits) < 2:
                continue  # no point classifying within a single circuit
            # Require at least 3 samples per circuit
            counts = Counter(y_fam_cir)
            keep = {c for c, n in counts.items() if n >= 3}
            sub_mask = np.array([c in keep for c in y_fam_cir])
            if sub_mask.sum() < 6 or len(keep) < 2:
                continue
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                max_depth=8,
                min_samples_leaf=2,
            )
            clf.fit(X_fam[sub_mask], y_fam_cir[sub_mask])
            self._circuit_clfs[fam] = clf
            self._circuit_classes[fam] = list(clf.classes_)

        # Keep a flat fallback for when the per-family clf is unavailable
        self._fallback = CircuitMLSelector()
        self._fallback.train(feature_store)

        self._trained = True
        self._n_train = len(X)
        logger.info(
            "HierarchicalCircuitSelector: trained on %d samples, "
            "%d families, %d per-family classifiers",
            self._n_train,
            len(self._family_classes),
            len(self._circuit_clfs),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, float], top_n: int = 3) -> List[str]:
        """Return up to *top_n* circuit names ranked by probability.

        Parameters
        ----------
        features : dict
            Spectral features keyed by ``_FEATURE_KEYS``.
        top_n : int
            Maximum number of circuits to return.

        Returns
        -------
        list[str]
            Circuit names in descending probability order.
        """
        if not self._trained:
            raise RuntimeError("HierarchicalCircuitSelector is not trained yet.")

        vec = np.array([[float(features.get(k, float("nan"))) for k in _FEATURE_KEYS]])
        if not np.all(np.isfinite(vec)):
            if self._fallback is not None:
                return self._fallback.predict(features, top_n=top_n)
            return []

        # Stage 1: predict family
        fam_probs = self._family_clf.predict_proba(vec)[0]
        ranked_families = sorted(
            zip(self._family_clf.classes_, fam_probs),
            key=lambda x: -x[1],
        )

        results: List[tuple] = []  # (circuit, score)
        for fam, fam_prob in ranked_families:
            if fam_prob < 0.01:
                break
            if fam in self._circuit_clfs:
                # Stage 2: predict within family
                clf = self._circuit_clfs[fam]
                cir_probs = clf.predict_proba(vec)[0]
                for circuit, cir_prob in zip(clf.classes_, cir_probs):
                    results.append((circuit, fam_prob * cir_prob))
            else:
                # Family has only one registered circuit — return it directly
                fam_circuits = _CIRCUIT_FAMILIES.get(fam, [])
                for c in fam_circuits:
                    results.append((c, fam_prob / max(len(fam_circuits), 1)))

        if not results:
            if self._fallback is not None:
                return self._fallback.predict(features, top_n=top_n)
            return []

        results.sort(key=lambda x: -x[1])
        return [c for c, _ in results[:top_n]]

    def confidence(self, features: Dict[str, float]) -> Dict[str, float]:
        """Return probability scores for all predicted circuits.

        Parameters
        ----------
        features : dict
            Spectral features keyed by ``_FEATURE_KEYS``.

        Returns
        -------
        dict[str, float]
            Mapping of circuit name → combined probability score.
        """
        if not self._trained:
            raise RuntimeError("HierarchicalCircuitSelector is not trained yet.")

        vec = np.array([[float(features.get(k, float("nan"))) for k in _FEATURE_KEYS]])
        if not np.all(np.isfinite(vec)):
            return {}

        fam_probs = self._family_clf.predict_proba(vec)[0]
        scores: Dict[str, float] = {}
        for fam, fam_prob in zip(self._family_clf.classes_, fam_probs):
            if fam in self._circuit_clfs:
                clf = self._circuit_clfs[fam]
                for circuit, cir_prob in zip(clf.classes_, clf.predict_proba(vec)[0]):
                    scores[circuit] = round(fam_prob * cir_prob, 4)
            else:
                fam_circuits = _CIRCUIT_FAMILIES.get(fam, [])
                for c in fam_circuits:
                    scores[c] = round(fam_prob / max(len(fam_circuits), 1), 4)

        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    def explain(self, features: Dict[str, float]) -> str:
        """Return a human-readable explanation of the top prediction.

        Parameters
        ----------
        features : dict
            Spectral features keyed by ``_FEATURE_KEYS``.

        Returns
        -------
        str
            Multi-line explanation with top family and top circuits.
        """
        if not self._trained:
            return "Hierarchical classifier not trained yet."

        vec = np.array([[float(features.get(k, float("nan"))) for k in _FEATURE_KEYS]])
        if not np.all(np.isfinite(vec)):
            return "Cannot explain: invalid feature vector (NaN/Inf)."

        fam_probs = self._family_clf.predict_proba(vec)[0]
        top_fam_idx = int(np.argmax(fam_probs))
        top_fam = self._family_clf.classes_[top_fam_idx]
        top_fam_prob = float(fam_probs[top_fam_idx])

        lines = [
            f"Top family : {top_fam} ({top_fam_prob:.1%})",
            "Top circuits:",
        ]
        scores = self.confidence(features)
        for i, (circuit, prob) in enumerate(list(scores.items())[:5]):
            lines.append(f"  {i + 1}. {circuit} ({prob:.1%})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path) -> None:
        """Serialise both classifiers to *path* using joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``data/knowledge/hier_classifier.joblib``).
        """
        from pathlib import Path as _Path

        import joblib

        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "family_clf": self._family_clf,
            "circuit_clfs": self._circuit_clfs,
            "family_classes": self._family_classes,
            "circuit_classes": self._circuit_classes,
            "n_train": self._n_train,
        }
        joblib.dump(payload, path)
        logger.info("HierarchicalCircuitSelector saved to '%s'", path)

    @classmethod
    def load_model(cls, path) -> "HierarchicalCircuitSelector":
        """Deserialise a previously saved model.

        Parameters
        ----------
        path : str or Path
            File written by :meth:`save_model`.

        Returns
        -------
        HierarchicalCircuitSelector
            Loaded and ready instance.
        """
        from pathlib import Path as _Path

        import joblib

        path = _Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)
        instance = cls()
        instance._family_clf = payload["family_clf"]
        instance._circuit_clfs = payload["circuit_clfs"]
        instance._family_classes = payload["family_classes"]
        instance._circuit_classes = payload["circuit_classes"]
        instance._n_train = payload.get("n_train", 0)
        instance._trained = True
        logger.info("HierarchicalCircuitSelector loaded from '%s'", path)
        return instance

    @classmethod
    def train_from_synthetic(
        cls,
        data_dir,
        model_path=None,
        min_samples_per_class: int = 5,
        cv_folds: int = 5,
    ) -> dict:
        """Train both stages from synthetic EIS files (``SYN_*.txt``).

        Mirrors the interface of
        :meth:`CircuitMLSelector.train_from_synthetic` for drop-in use.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing ``SYN_*.txt`` files.
        model_path : str or Path, optional
            Destination for the serialised model.  Defaults to
            ``data/knowledge/hier_classifier.joblib``.
        min_samples_per_class : int
            Minimum per-circuit sample count to include in training.
        cv_folds : int
            Number of cross-validation folds for the accuracy estimate.

        Returns
        -------
        dict
            Keys: ``n_samples``, ``n_classes``, ``classes``,
            ``cv_accuracy``, ``model_path``, ``skipped``.
        """
        from collections import Counter
        from pathlib import Path as _Path

        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        from src.circuit_fitting import extract_eis_features_for_ml
        from src.loader import load_eis_file

        data_dir = _Path(data_dir)
        if model_path is None:
            model_path = _Path("data/knowledge/hier_classifier.joblib")
        else:
            model_path = _Path(model_path)

        syn_files = sorted(data_dir.glob("SYN_*.txt"))
        if not syn_files:
            raise FileNotFoundError(
                f"No SYN_*.txt files found in '{data_dir}'. "
                "Generate synthetic data first."
            )

        X_list: List[List[float]] = []
        y_circuit: List[str] = []
        y_family: List[str] = []
        skipped = 0

        for fpath in syn_files:
            stem = fpath.stem
            without_prefix = stem[4:]
            parts = without_prefix.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                skipped += 1
                continue
            circuit_name = parts[0].replace("_", " ")
            family = _CIRCUIT_TO_FAMILY.get(circuit_name, "unknown")
            if family == "unknown":
                skipped += 1
                continue
            try:
                df = load_eis_file(str(fpath))
                feats = extract_eis_features_for_ml(df)
                vec = [float(feats.get(k, float("nan"))) for k in _FEATURE_KEYS]
                if not all(np.isfinite(v) for v in vec):
                    skipped += 1
                    continue
                X_list.append(vec)
                y_circuit.append(circuit_name)
                y_family.append(family)
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath.name, exc)
                skipped += 1

        if not X_list:
            raise ValueError("No valid spectral features could be extracted.")

        X = np.array(X_list)
        y_cir = np.array(y_circuit)
        y_fam = np.array(y_family)

        # Filter by minimum sample count per circuit
        counts = Counter(y_circuit)
        valid = {c for c, n in counts.items() if n >= min_samples_per_class}
        mask = np.array([c in valid for c in y_cir])
        X, y_cir, y_fam = X[mask], y_cir[mask], y_fam[mask]

        n_classes = len(np.unique(y_cir))
        if n_classes < 2:
            raise ValueError(
                f"Need ≥ 2 classes with ≥ {min_samples_per_class} samples. "
                f"Found: {dict(Counter(y_circuit))}"
            )

        # Stage 1: family classifier with CV, then calibrated
        fam_clf_base = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            max_depth=6,
            min_samples_leaf=2,
        )
        cv_accuracy = 0.0
        n_fam = len(np.unique(y_fam))
        if len(X) >= cv_folds * n_fam:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(fam_clf_base, X, y_fam, cv=skf, scoring="accuracy")
            cv_accuracy = float(scores.mean())
        fam_clf_base.fit(X, y_fam)
        n_fam_cal = min(3, max(2, len(X) // (n_fam * 5)))
        try:
            fam_clf: object = CalibratedClassifierCV(
                fam_clf_base, cv=n_fam_cal, method="sigmoid"
            )
            fam_clf.fit(X, y_fam)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                "Family clf calibration failed (%s); using uncalibrated.", exc
            )
            fam_clf = fam_clf_base

        # Stage 2: per-family classifiers (calibrated when sufficient samples)
        circuit_clfs: Dict[str, object] = {}
        circuit_classes: Dict[str, List[str]] = {}
        for fam in np.unique(y_fam):
            fam_mask = y_fam == fam
            X_fam = X[fam_mask]
            y_fam_cir = y_cir[fam_mask]
            unique_c = np.unique(y_fam_cir)
            if len(unique_c) < 2:
                continue
            fam_counts = Counter(y_fam_cir)
            keep = {c for c, n in fam_counts.items() if n >= 3}
            sub_mask = np.array([c in keep for c in y_fam_cir])
            if sub_mask.sum() < 6 or len(keep) < 2:
                continue
            clf_base = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                max_depth=8,
                min_samples_leaf=2,
            )
            clf_base.fit(X_fam[sub_mask], y_fam_cir[sub_mask])
            n_cir_cal = min(3, max(2, int(sub_mask.sum()) // (len(keep) * 4)))
            try:
                clf: object = CalibratedClassifierCV(
                    clf_base, cv=n_cir_cal, method="sigmoid"
                )
                clf.fit(X_fam[sub_mask], y_fam_cir[sub_mask])  # type: ignore[union-attr]
            except Exception:
                clf = clf_base
            circuit_clfs[fam] = clf
            circuit_classes[fam] = list(clf_base.classes_)

        instance = cls()
        instance._family_clf = fam_clf
        instance._circuit_clfs = circuit_clfs
        instance._family_classes = list(fam_clf_base.classes_)
        instance._circuit_classes = circuit_classes
        instance._trained = True
        instance._n_train = len(X)
        instance.save_model(model_path)

        logger.info(
            "HierarchicalCircuitSelector: trained on %d samples, "
            "%d circuits, %d families, CV family acc=%.1f%%",
            len(X),
            n_classes,
            len(np.unique(y_fam)),
            cv_accuracy * 100,
        )

        return {
            "n_samples": len(X),
            "n_classes": n_classes,
            "classes": list(np.unique(y_cir)),
            "cv_accuracy": cv_accuracy,
            "model_path": str(model_path),
            "skipped": skipped,
        }
