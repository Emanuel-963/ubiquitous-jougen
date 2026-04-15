"""Performance predictor for IonFlow AI (Day 17).

Provides three core capabilities:

1. **Predict cycling from EIS** — estimate energy, power and retention
   from impedance parameters using regression models trained on historical
   data from the :class:`~src.feature_store.FeatureStore`.
2. **Predict degradation** — compare EIS before/after cycling and classify
   the degradation mechanism.
3. **Recommend improvements** — suggest concrete experimental changes to
   overcome identified bottlenecks.

The predictor works in two regimes:

* **Data-driven** (≥ 30 historical records with ``eis_params`` and
  ``cycling_targets``): trains Ridge / RandomForest regressors.
* **Heuristic** (< 30 records or no history): uses physics-informed
  rules derived from the knowledge base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)

# Minimum records needed to switch from heuristic to ML prediction
_MIN_RECORDS_FOR_ML = 30

# EIS parameter keys used as regression inputs
_EIS_FEATURE_KEYS: Tuple[str, ...] = (
    "Rs_fit",
    "Rp_fit",
    "Q",
    "n",
    "Sigma",
    "C_mean",
    "Tau",
    "Dispersion",
    "Energy_mean",
)

# Cycling target keys the predictor can estimate
_CYCLING_TARGET_KEYS: Tuple[str, ...] = (
    "energy",
    "power",
    "retention",
)


# ═══════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════

class DegradationMechanism(str, Enum):
    """Classification of degradation mechanism."""

    FILM_GROWTH = "film_growth"
    ACTIVE_MATERIAL_LOSS = "active_material_loss"
    CONTACT_DEGRADATION = "contact_degradation"
    ELECTROLYTE_DEGRADATION = "electrolyte_degradation"
    MIXED = "mixed"
    NONE = "none"

    def __str__(self) -> str:  # noqa: D105
        return self.value


class ImprovementArea(str, Enum):
    """Area targeted by an improvement recommendation."""

    OHMIC_RESISTANCE = "ohmic_resistance"
    CHARGE_TRANSFER = "charge_transfer"
    SURFACE_MORPHOLOGY = "surface_morphology"
    DIFFUSION = "diffusion"
    CYCLING_STABILITY = "cycling_stability"
    GENERAL = "general"

    def __str__(self) -> str:  # noqa: D105
        return self.value


# ═══════════════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CyclingPrediction:
    """Predicted cycling performance from EIS parameters.

    Attributes
    ----------
    energy : float | None
        Estimated energy (µJ).
    power : float | None
        Estimated power (µW).
    retention : float | None
        Estimated retention (%).
    confidence : float
        Overall confidence 0–1 (higher is better).
    method : str
        ``"ml"`` or ``"heuristic"``.
    explanation : str
        Human-readable paragraph explaining the prediction.
    feature_importances : dict[str, float]
        Which EIS features contributed most (only for ML method).
    """

    energy: Optional[float] = None
    power: Optional[float] = None
    retention: Optional[float] = None
    confidence: float = 0.0
    method: str = "heuristic"
    explanation: str = ""
    feature_importances: Dict[str, float] = field(default_factory=dict)


@dataclass
class DegradationPrediction:
    """Result of comparing EIS before/after cycling.

    Attributes
    ----------
    mechanism : DegradationMechanism
        Primary degradation mechanism identified.
    delta : dict[str, float]
        Absolute change per parameter (after − before).
    delta_pct : dict[str, float]
        Percentage change per parameter.
    severity : float
        0–1 severity score (0 = no degradation, 1 = severe).
    explanation : str
        Textual description of the degradation.
    secondary_mechanisms : list[DegradationMechanism]
        Other mechanisms that may be contributing.
    """

    mechanism: DegradationMechanism = DegradationMechanism.NONE
    delta: Dict[str, float] = field(default_factory=dict)
    delta_pct: Dict[str, float] = field(default_factory=dict)
    severity: float = 0.0
    explanation: str = ""
    secondary_mechanisms: List[DegradationMechanism] = field(default_factory=list)


@dataclass
class Improvement:
    """A concrete improvement suggestion.

    Attributes
    ----------
    area : ImprovementArea
        Category of the improvement.
    action : str
        Concrete experimental action.
    expected_impact : str
        What improvement is expected.
    priority : int
        1 = highest, 3 = lowest.
    rationale : str
        Why this is recommended.
    """

    area: ImprovementArea = ImprovementArea.GENERAL
    action: str = ""
    expected_impact: str = ""
    priority: int = 2
    rationale: str = ""

    def __str__(self) -> str:
        return f"[P{self.priority}] {self.action} — {self.expected_impact}"


# ═══════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════

def _extract_eis_vector(
    params: Dict[str, float],
    keys: Sequence[str] = _EIS_FEATURE_KEYS,
) -> Optional[np.ndarray]:
    """Build a numeric feature vector from an EIS parameter dict.

    Returns ``None`` if any key is missing or non-finite.
    """
    vals = []
    for k in keys:
        v = params.get(k)
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(f):
            return None
        vals.append(f)
    return np.array(vals)


def _safe_pct_change(before: float, after: float) -> float:
    """Percentage change, safe against division by zero."""
    if before == 0:
        return 0.0 if after == 0 else 100.0
    return (after - before) / abs(before) * 100.0


def _build_training_data(
    records: List[Dict[str, Any]],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """Build X (EIS features) and y (cycling targets) from store records.

    Returns ``(None, None)`` if insufficient data.
    """
    X_rows: List[np.ndarray] = []
    y_dict: Dict[str, List[float]] = {k: [] for k in _CYCLING_TARGET_KEYS}

    for rec in records:
        eis = rec.get("eis_params")
        cyc = rec.get("cycling_targets")
        if not isinstance(eis, dict) or not isinstance(cyc, dict):
            continue

        vec = _extract_eis_vector(eis)
        if vec is None:
            continue

        # Need at least one cycling target
        has_target = False
        row_targets: Dict[str, float] = {}
        for tk in _CYCLING_TARGET_KEYS:
            val = cyc.get(tk)
            if val is not None:
                try:
                    fv = float(val)
                    if np.isfinite(fv):
                        row_targets[tk] = fv
                        has_target = True
                except (TypeError, ValueError):
                    pass

        if not has_target:
            continue

        X_rows.append(vec)
        for tk in _CYCLING_TARGET_KEYS:
            y_dict[tk].append(row_targets.get(tk, np.nan))

    if len(X_rows) < _MIN_RECORDS_FOR_ML:
        return None, None

    X = np.array(X_rows)
    Y = {k: np.array(v) for k, v in y_dict.items()}
    return X, Y


# ═══════════════════════════════════════════════════════════════════════
#  Heuristic predictor
# ═══════════════════════════════════════════════════════════════════════

def _heuristic_cycling_prediction(
    params: Dict[str, float],
) -> CyclingPrediction:
    """Estimate cycling performance from EIS params using physics rules.

    Simple scaling relationships based on electrochemical theory:
    - Lower Rs → higher power delivery
    - Lower Rp → faster charge transfer → better retention
    - Higher C_mean → higher energy storage
    """
    rs = params.get("Rs_fit")
    rp = params.get("Rp_fit")
    c_mean = params.get("C_mean")
    n = params.get("n")
    sigma = params.get("Sigma")
    energy_eis = params.get("Energy_mean")

    pred = CyclingPrediction(method="heuristic")
    parts: List[str] = []

    # Energy estimate: proportional to C_mean
    if c_mean is not None and c_mean > 0:
        # Rough: E ∝ C × V² / 2, assume V ~ 1V → energy ≈ C_mean in µJ scale
        pred.energy = round(c_mean * 0.5, 2)
        parts.append(f"Energia estimada ≈ {pred.energy:.2f} µJ (proporcional a C_mean)")
    elif energy_eis is not None and energy_eis > 0:
        pred.energy = round(energy_eis, 2)
        parts.append(f"Energia estimada ≈ {pred.energy:.2f} µJ (do EIS)")

    # Power estimate: inversely proportional to Rs
    if rs is not None and rs > 0:
        # P ∝ V² / (4 × Rs)  for V = 1V → P = 0.25 / Rs (W), convert to µW
        pred.power = round(250_000 / rs, 2)
        parts.append(f"Potência estimada ≈ {pred.power:.2f} µW (inversamente proporcional a Rs)")

    # Retention estimate: based on n and Rp
    if rp is not None and n is not None:
        # Higher n (closer to 1) → more ideal → better retention
        # Lower Rp → less charge transfer resistance → better cycling
        base_ret = 70.0
        n_bonus = max(0, (n - 0.5)) * 40.0  # n=0.5 → 0, n=1.0 → 20
        rp_penalty = min(20.0, rp * 0.1)  # high Rp reduces retention
        sigma_penalty = 0.0
        if sigma is not None and sigma > 30:
            sigma_penalty = min(10.0, (sigma - 30) * 0.2)
        pred.retention = round(
            max(30.0, min(99.0, base_ret + n_bonus - rp_penalty - sigma_penalty)), 1,
        )
        parts.append(f"Retenção estimada ≈ {pred.retention:.1f}%")

    # Confidence: low for heuristic
    pred.confidence = 0.3
    parts.insert(0, "Predição baseada em regras heurísticas (sem histórico ML suficiente).")
    pred.explanation = " ".join(parts)

    return pred


# ═══════════════════════════════════════════════════════════════════════
#  Degradation classifier
# ═══════════════════════════════════════════════════════════════════════

def _classify_degradation(
    before: Dict[str, float],
    after: Dict[str, float],
) -> DegradationPrediction:
    """Compare EIS parameters before/after and classify degradation."""
    pred = DegradationPrediction()

    # Compute deltas
    common_keys = set(before.keys()) & set(after.keys())
    for k in sorted(common_keys):
        b, a = before[k], after[k]
        pred.delta[k] = round(a - b, 6)
        pred.delta_pct[k] = round(_safe_pct_change(b, a), 2)

    # Key parameter changes
    d_rs = pred.delta_pct.get("Rs_fit", 0.0)
    d_rp = pred.delta_pct.get("Rp_fit", 0.0)
    d_n = pred.delta_pct.get("n", 0.0)
    d_c = pred.delta_pct.get("C_mean", 0.0)
    d_sigma = pred.delta_pct.get("Sigma", 0.0)

    # Severity: weighted combination of absolute percentage changes
    abs_changes = [abs(v) for v in pred.delta_pct.values()]
    pred.severity = round(
        min(1.0, np.mean(abs_changes) / 100.0) if abs_changes else 0.0, 3,
    )

    # Classification logic
    mechanisms: List[Tuple[DegradationMechanism, float]] = []
    parts: List[str] = []

    # Film growth: Rs increases significantly, Rp may increase
    if d_rs > 20:
        score = min(1.0, d_rs / 100.0)
        mechanisms.append((DegradationMechanism.FILM_GROWTH, score))
        parts.append(
            f"Rs aumentou {d_rs:.1f}% — sugere crescimento de filme passivo ou "
            f"aumento de resistência de interface."
        )

    # Active material loss: C decreases, Rp increases
    if d_c < -15:
        score = min(1.0, abs(d_c) / 100.0)
        mechanisms.append((DegradationMechanism.ACTIVE_MATERIAL_LOSS, score))
        parts.append(
            f"C_mean diminuiu {abs(d_c):.1f}% — sugere perda de material ativo "
            f"ou redução de área eletroativa."
        )

    # Contact degradation: Rs increases, n decreases
    if d_rs > 10 and d_n < -5:
        score = min(1.0, (d_rs + abs(d_n)) / 100.0)
        mechanisms.append((DegradationMechanism.CONTACT_DEGRADATION, score))
        parts.append(
            f"Rs +{d_rs:.1f}% e n {d_n:.1f}% — possível degradação de contato "
            f"elétrico ou desprendimento do material."
        )

    # Electrolyte degradation: Sigma increases significantly
    if d_sigma > 30:
        score = min(1.0, d_sigma / 100.0)
        mechanisms.append((DegradationMechanism.ELECTROLYTE_DEGRADATION, score))
        parts.append(
            f"Sigma aumentou {d_sigma:.1f}% — possível degradação do eletrólito "
            f"ou aumento da resistência de difusão."
        )

    # Pick primary mechanism
    if not mechanisms:
        # Check if there's any meaningful change at all
        if pred.severity < 0.05:
            pred.mechanism = DegradationMechanism.NONE
            parts.append("Sem degradação significativa detectada.")
        else:
            pred.mechanism = DegradationMechanism.MIXED
            parts.append("Mudanças sutis detectadas sem mecanismo dominante claro.")
    elif len(mechanisms) == 1:
        pred.mechanism = mechanisms[0][0]
    else:
        # Sort by score, pick highest
        mechanisms.sort(key=lambda x: x[1], reverse=True)
        pred.mechanism = mechanisms[0][0]
        pred.secondary_mechanisms = [m for m, _ in mechanisms[1:]]
        if len(mechanisms) >= 2:
            parts.append("Mecanismo primário identificado com contribuições secundárias.")

    pred.explanation = " ".join(parts) if parts else "Análise de degradação inconclusiva."
    return pred


# ═══════════════════════════════════════════════════════════════════════
#  Improvement recommender
# ═══════════════════════════════════════════════════════════════════════

_IMPROVEMENT_RULES: List[Dict[str, Any]] = [
    # Ohmic resistance
    {
        "condition": lambda p: p.get("Rs_fit", 0) > 10,
        "area": ImprovementArea.OHMIC_RESISTANCE,
        "action": "Reduzir resistência ôhmica: polir eletrodo, usar cola de prata ou melhorar contato.",
        "impact": "Esperada redução de Rs em 30-60% e aumento proporcional de potência.",
        "priority": 1,
        "rationale": "Rs > 10 Ω limita severamente a potência entregue pelo dispositivo.",
    },
    {
        "condition": lambda p: 5 < p.get("Rs_fit", 0) <= 10,
        "area": ImprovementArea.OHMIC_RESISTANCE,
        "action": "Considerar eletrólito mais condutivo (ex: H₂SO₄ 1M vs Na₂SO₄ 0.5M).",
        "impact": "Redução de Rs tipicamente 2-4× ao trocar eletrólito.",
        "priority": 2,
        "rationale": "Rs moderado — margem para melhoria via eletrólito.",
    },
    # Charge transfer
    {
        "condition": lambda p: p.get("Rp_fit", 0) > 200,
        "area": ImprovementArea.CHARGE_TRANSFER,
        "action": "Aumentar área eletroativa: usar nanoestruturas, depositar mais material ou ativar superfície.",
        "impact": "Redução de Rp melhora cinética de transferência de carga.",
        "priority": 1,
        "rationale": "Rp > 200 Ω indica cinética lenta de transferência de carga.",
    },
    {
        "condition": lambda p: 50 < p.get("Rp_fit", 0) <= 200,
        "area": ImprovementArea.CHARGE_TRANSFER,
        "action": "Otimizar tratamento térmico ou tempo de deposição para melhorar interface.",
        "impact": "Possível redução de Rp em 20-40%.",
        "priority": 2,
        "rationale": "Rp moderado — otimização de processo pode trazer melhorias.",
    },
    # Surface morphology
    {
        "condition": lambda p: p.get("n", 1) < 0.6,
        "area": ImprovementArea.SURFACE_MORPHOLOGY,
        "action": "Otimizar morfologia: controlar taxa de deposição, tratar superfície.",
        "impact": "n mais próximo de 1.0 indica interface mais homogénea.",
        "priority": 1,
        "rationale": "n < 0.6 sugere superfície muito heterogénea ou porosa.",
    },
    {
        "condition": lambda p: 0.6 <= p.get("n", 1) < 0.8,
        "area": ImprovementArea.SURFACE_MORPHOLOGY,
        "action": "Aumentar homogeneidade: recozimento térmico ou polimento mecânico.",
        "impact": "Melhoria incremental na uniformidade da interface.",
        "priority": 3,
        "rationale": "n moderado — interface com rugosidade aceitável mas otimizável.",
    },
    # Diffusion
    {
        "condition": lambda p: p.get("Sigma", 0) > 100,
        "area": ImprovementArea.DIFFUSION,
        "action": "Reduzir limitação difusional: aumentar concentração do eletrólito ou reduzir espessura do filme.",
        "impact": "Sigma menor melhora resposta em baixas frequências.",
        "priority": 1,
        "rationale": "Sigma > 100 indica difusão como gargalo principal.",
    },
    {
        "condition": lambda p: 30 < p.get("Sigma", 0) <= 100,
        "area": ImprovementArea.DIFFUSION,
        "action": "Medir EIS em frequências mais baixas (< 10 mHz) para melhor resolução difusional.",
        "impact": "Mais informação sobre processos lentos de transporte.",
        "priority": 2,
        "rationale": "Sigma moderado — contribuição difusional presente.",
    },
    # Cycling stability
    {
        "condition": lambda p: p.get("retention", 100) < 70,
        "area": ImprovementArea.CYCLING_STABILITY,
        "action": "Reduzir janela de potencial ou adicionar agente estabilizante para melhorar retenção.",
        "impact": "Retenção > 80% com janela de potencial otimizada.",
        "priority": 1,
        "rationale": "Retenção < 70% indica problemas sérios de estabilidade cíclica.",
    },
    {
        "condition": lambda p: 70 <= p.get("retention", 100) < 90,
        "area": ImprovementArea.CYCLING_STABILITY,
        "action": "Repetir ciclagem com 5000+ ciclos para avaliar estabilidade de longo prazo.",
        "impact": "Verificar se a tendência de degradação se mantém ou estabiliza.",
        "priority": 2,
        "rationale": "Retenção moderada — importante confirmar tendência.",
    },
]


def _recommend_improvements(
    params: Dict[str, float],
) -> List[Improvement]:
    """Generate improvement suggestions based on current EIS parameters."""
    improvements: List[Improvement] = []
    for rule in _IMPROVEMENT_RULES:
        try:
            if rule["condition"](params):
                improvements.append(Improvement(
                    area=rule["area"],
                    action=rule["action"],
                    expected_impact=rule["impact"],
                    priority=rule["priority"],
                    rationale=rule["rationale"],
                ))
        except (KeyError, TypeError):
            continue

    # Sort by priority
    improvements.sort(key=lambda imp: imp.priority)
    return improvements


# ═══════════════════════════════════════════════════════════════════════
#  ML predictor (Ridge + RandomForest)
# ═══════════════════════════════════════════════════════════════════════

class _MLPredictor:
    """Internal ML model wrapper.

    Uses sklearn Ridge for linear baseline and RandomForest for
    non-linear prediction.  Falls back gracefully if sklearn is
    not available.
    """

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._feature_names: Tuple[str, ...] = _EIS_FEATURE_KEYS
        self._is_trained = False
        self._n_samples = 0
        self._feature_importances: Dict[str, float] = {}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        X: np.ndarray,
        Y: Dict[str, np.ndarray],
    ) -> None:
        """Train regressors for each cycling target.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        Y : dict of target_name → ndarray (n_samples,)
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("sklearn not available — ML prediction disabled.")
            return

        self._n_samples = X.shape[0]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scaler = scaler

        importances_sum = np.zeros(X.shape[1])

        for target_name, y in Y.items():
            # Skip if all NaN
            mask = np.isfinite(y)
            if mask.sum() < _MIN_RECORDS_FOR_ML:
                continue

            X_t = X_scaled[mask]
            y_t = y[mask]

            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=1,
            )
            model.fit(X_t, y_t)
            self._models[target_name] = model
            importances_sum += model.feature_importances_

        if self._models:
            self._is_trained = True
            # Average feature importances across targets
            avg_imp = importances_sum / len(self._models)
            self._feature_importances = {
                name: round(float(imp), 4)
                for name, imp in zip(self._feature_names, avg_imp)
            }

    def predict(
        self,
        params: Dict[str, float],
    ) -> CyclingPrediction:
        """Predict cycling performance from EIS parameters."""
        if not self._is_trained:
            return _heuristic_cycling_prediction(params)

        vec = _extract_eis_vector(params)
        if vec is None:
            return _heuristic_cycling_prediction(params)

        try:
            X = self._scaler.transform(vec.reshape(1, -1))
        except Exception:
            return _heuristic_cycling_prediction(params)

        pred = CyclingPrediction(method="ml", feature_importances=self._feature_importances)
        parts: List[str] = [
            f"Predição baseada em modelo ML treinado com {self._n_samples} amostras.",
        ]

        for target_name, model in self._models.items():
            try:
                val = float(model.predict(X)[0])
                if target_name == "energy":
                    pred.energy = round(val, 2)
                    parts.append(f"Energia estimada: {val:.2f} µJ")
                elif target_name == "power":
                    pred.power = round(val, 2)
                    parts.append(f"Potência estimada: {val:.2f} µW")
                elif target_name == "retention":
                    pred.retention = round(max(0, min(100, val)), 1)
                    parts.append(f"Retenção estimada: {pred.retention:.1f}%")
            except Exception:
                continue

        # Confidence based on number of training samples and successful targets
        n_targets = len(self._models)
        sample_factor = min(1.0, self._n_samples / 100)
        pred.confidence = round(0.5 + 0.5 * sample_factor * (n_targets / 3), 2)

        # Top-3 important features
        sorted_imp = sorted(
            self._feature_importances.items(), key=lambda x: x[1], reverse=True,
        )[:3]
        if sorted_imp:
            top_str = ", ".join(f"{k} ({v:.0%})" for k, v in sorted_imp)
            parts.append(f"Features mais importantes: {top_str}.")

        pred.explanation = " ".join(parts)
        return pred


# ═══════════════════════════════════════════════════════════════════════
#  EIS parameter extractor from result objects
# ═══════════════════════════════════════════════════════════════════════

def _extract_median_params(ranked_df: pd.DataFrame) -> Dict[str, float]:
    """Extract median EIS parameter values from a ranked DataFrame."""
    params: Dict[str, float] = {}
    for col in _EIS_FEATURE_KEYS:
        if col in ranked_df.columns:
            vals = pd.to_numeric(ranked_df[col], errors="coerce").dropna()
            if not vals.empty:
                params[col] = float(vals.median())
    # Also grab retention if present
    if "Retenção (%)" in ranked_df.columns:
        vals = pd.to_numeric(ranked_df["Retenção (%)"], errors="coerce").dropna()
        if not vals.empty:
            params["retention"] = float(vals.median())
    return params


def _extract_cycling_targets(cycling_result: Any) -> Dict[str, float]:
    """Extract cycling target values from a CyclingResult."""
    targets: Dict[str, float] = {}
    merged = getattr(cycling_result, "merged_table", None)
    if merged is None or not isinstance(merged, pd.DataFrame) or merged.empty:
        return targets

    col_map = {
        "Retenção (%)": "retention",
        "Energia (µJ)": "energy",
        "Potência (µW)": "power",
    }
    for col, key in col_map.items():
        if col in merged.columns:
            vals = pd.to_numeric(merged[col], errors="coerce").dropna()
            if not vals.empty:
                targets[key] = float(vals.median())
    return targets


# ═══════════════════════════════════════════════════════════════════════
#  PerformancePredictor
# ═══════════════════════════════════════════════════════════════════════

class PerformancePredictor:
    """Predicts cycling performance, degradation and improvements.

    Parameters
    ----------
    feature_store : FeatureStore | None
        Historical fitting records for ML training.  If ``None``,
        heuristic mode is used exclusively.
    config : PipelineConfig | None
        Pipeline configuration.
    """

    def __init__(
        self,
        feature_store: Any = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self._store = feature_store
        self._config = config if config is not None else PipelineConfig.default()
        self._ml = _MLPredictor()
        self._trained = False

        # Auto-train if store has enough data
        if feature_store is not None:
            self._try_train()

    @property
    def is_ml_trained(self) -> bool:
        """Whether the ML model has been trained."""
        return self._ml.is_trained

    # ── Training ──────────────────────────────────────────────────────

    def _try_train(self) -> None:
        """Attempt to train ML models from the feature store."""
        if self._store is None:
            return
        records = getattr(self._store, "records", [])
        if not records:
            return

        X, Y = _build_training_data(records)
        if X is not None and Y is not None:
            self._ml.train(X, Y)
            if self._ml.is_trained:
                logger.info(
                    "PerformancePredictor: ML trained on %d records.", X.shape[0],
                )
                self._trained = True

    def train(self, records: List[Dict[str, Any]]) -> None:
        """Explicitly train from a list of records.

        Each record should have ``eis_params`` (dict) and ``cycling_targets`` (dict).
        """
        X, Y = _build_training_data(records)
        if X is not None and Y is not None:
            self._ml.train(X, Y)

    # ── Predict cycling from EIS ──────────────────────────────────────

    def predict_cycling_from_eis(
        self,
        eis_params: Dict[str, float],
    ) -> CyclingPrediction:
        """Predict cycling performance from EIS parameters.

        Parameters
        ----------
        eis_params : dict
            EIS parameters (at minimum: Rs_fit, Rp_fit, n).

        Returns
        -------
        CyclingPrediction
        """
        if self._ml.is_trained:
            return self._ml.predict(eis_params)
        return _heuristic_cycling_prediction(eis_params)

    def predict_cycling_from_result(
        self,
        eis_result: Any,
    ) -> CyclingPrediction:
        """Predict cycling performance from an EISResult object."""
        ranked_df = getattr(eis_result, "ranked_df", None)
        if ranked_df is None or not isinstance(ranked_df, pd.DataFrame) or ranked_df.empty:
            return CyclingPrediction(
                explanation="Sem dados EIS disponíveis para predição.",
            )
        params = _extract_median_params(ranked_df)
        return self.predict_cycling_from_eis(params)

    # ── Predict degradation ───────────────────────────────────────────

    def predict_degradation(
        self,
        eis_before: Dict[str, float],
        eis_after: Dict[str, float],
    ) -> DegradationPrediction:
        """Compare EIS parameters before/after cycling.

        Parameters
        ----------
        eis_before : dict
            EIS parameters before cycling.
        eis_after : dict
            EIS parameters after cycling.

        Returns
        -------
        DegradationPrediction
        """
        return _classify_degradation(eis_before, eis_after)

    def predict_degradation_from_results(
        self,
        eis_before: Any,
        eis_after: Any,
    ) -> DegradationPrediction:
        """Compare two EISResult objects."""
        df_before = getattr(eis_before, "ranked_df", None)
        df_after = getattr(eis_after, "ranked_df", None)
        if df_before is None or df_after is None:
            return DegradationPrediction(
                explanation="Dados EIS insuficientes para análise de degradação.",
            )
        if isinstance(df_before, pd.DataFrame) and isinstance(df_after, pd.DataFrame):
            p_before = _extract_median_params(df_before)
            p_after = _extract_median_params(df_after)
            if p_before and p_after:
                return _classify_degradation(p_before, p_after)
        return DegradationPrediction(
            explanation="Dados EIS insuficientes para análise de degradação.",
        )

    # ── Recommend improvements ────────────────────────────────────────

    def recommend_improvements(
        self,
        eis_params: Dict[str, float],
    ) -> List[Improvement]:
        """Suggest improvements based on current EIS parameters.

        Parameters
        ----------
        eis_params : dict
            Current EIS parameters.

        Returns
        -------
        list[Improvement]
            Sorted by priority (1 = highest).
        """
        return _recommend_improvements(eis_params)

    def recommend_improvements_from_result(
        self,
        eis_result: Any,
    ) -> List[Improvement]:
        """Suggest improvements from an EISResult object."""
        ranked_df = getattr(eis_result, "ranked_df", None)
        if ranked_df is None or not isinstance(ranked_df, pd.DataFrame) or ranked_df.empty:
            return []
        params = _extract_median_params(ranked_df)
        return _recommend_improvements(params)
