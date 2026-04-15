"""Electrochemistry knowledge base for the IonFlow AI agent.

This module provides:

* :class:`ElectrochemicalRule` — a structured rule that maps a measurable
  condition (e.g. *Rs > 10 Ω*) to an interpretation, possible causes,
  recommendations and severity level.
* :class:`KnowledgeBase` — a registry of rules that can be loaded from /
  saved to a JSON file and queried at runtime.
* :class:`RuleMatch` — the result of evaluating a single rule against a
  concrete set of measurements.

Rules are designed to be **parameterisable**: thresholds that appear inside
condition expressions can be overridden at evaluation time via a
``thresholds`` dict, which is typically sourced from
:class:`src.config.PipelineConfig`.

Workflow
--------
1. ``kb = KnowledgeBase.default()`` — load the bundled 50+ rules.
2. ``kb.evaluate(measurements)`` — return a list of :class:`RuleMatch`
   for every rule whose condition is satisfied.
3. Results feed the inference engine (Day 16) and the GUI AI panel (Day 19).
"""

from __future__ import annotations

import json
import logging
import operator
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Severity enum
# ═══════════════════════════════════════════════════════════════════════

class Severity(str, Enum):
    """Severity / importance level of a rule match."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    def __str__(self) -> str:  # noqa: D105
        return self.value


# ═══════════════════════════════════════════════════════════════════════
#  ElectrochemicalRule
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ElectrochemicalRule:
    """A single interpretive rule for an electrochemical measurement.

    Parameters
    ----------
    rule_id : str
        Unique identifier (e.g. ``"RS_HIGH"``).
    category : str
        Grouping tag: ``"impedance"``, ``"cycling"``, ``"drt"``,
        ``"correlation"``, ``"general"``.
    condition : str
        A human-readable expression **and** machine-evaluable micro-DSL::

            ``"Rs > {Rs_high_threshold}"``
            ``"n < 0.5"``
            ``"retention < {retention_low_threshold}"``

        Curly-brace tokens are replaced at evaluation time with values
        from the ``thresholds`` dict.
    parameter : str
        The primary measurement parameter this rule targets (e.g. ``"Rs"``).
    interpretation : str
        Plain-language interpretation when the condition is met.
    possible_causes : list[str]
        What might cause this observation.
    recommendations : list[str]
        Suggested corrective / investigative actions.
    severity : Severity
        ``info``, ``warning`` or ``critical``.
    references : list[str]
        Optional literature / tutorial references.
    """

    rule_id: str = ""
    category: str = "general"
    condition: str = ""
    parameter: str = ""
    interpretation: str = ""
    possible_causes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    severity: Severity = Severity.INFO
    references: List[str] = field(default_factory=list)

    # ── serialisation helpers ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-safe dict."""
        d = asdict(self)
        d["severity"] = str(self.severity)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElectrochemicalRule":
        """Restore from a dict (as read from JSON)."""
        data = dict(data)  # shallow copy
        sev = data.pop("severity", "info")
        try:
            severity = Severity(sev)
        except ValueError:
            severity = Severity.INFO
        return cls(severity=severity, **data)


# ═══════════════════════════════════════════════════════════════════════
#  RuleMatch
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RuleMatch:
    """Result produced when a rule's condition evaluates to *True*.

    Attributes
    ----------
    rule : ElectrochemicalRule
        The rule that was triggered.
    actual_value : float | None
        The measured value that triggered the rule.
    resolved_condition : str
        The condition string after threshold substitution.
    """

    rule: ElectrochemicalRule
    actual_value: Optional[float] = None
    resolved_condition: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  Condition evaluator (safe micro-DSL)
# ═══════════════════════════════════════════════════════════════════════

# Pattern:  "param op value"   e.g.  "Rs > 10"  or  "n < 0.5"
_COND_RE = re.compile(
    r"^\s*(?P<param>[A-Za-z_][\w.]*)"   # parameter name
    r"\s*(?P<op>[><=!]+)\s*"             # operator
    r"(?P<value>-?[\d.]+(?:e[+-]?\d+)?)" # numeric value
    r"\s*$",
    re.IGNORECASE,
)

_OPS: Dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _resolve_thresholds(condition: str, thresholds: Dict[str, float]) -> str:
    """Replace ``{token}`` placeholders in *condition* with threshold values."""
    def _sub(m: re.Match) -> str:
        key = m.group(1)
        return str(thresholds.get(key, m.group(0)))
    return re.sub(r"\{(\w+)\}", _sub, condition)


def _eval_condition(
    condition: str,
    measurements: Dict[str, float],
) -> Tuple[bool, Optional[float]]:
    """Evaluate a single micro-DSL condition.

    Returns ``(matched, actual_value)``.
    """
    m = _COND_RE.match(condition)
    if m is None:
        return False, None

    param = m.group("param")
    op_str = m.group("op")
    threshold = float(m.group("value"))

    op_fn = _OPS.get(op_str)
    if op_fn is None:
        return False, None

    actual = measurements.get(param)
    if actual is None:
        return False, None

    try:
        return op_fn(float(actual), threshold), float(actual)
    except (TypeError, ValueError):
        return False, None


# ═══════════════════════════════════════════════════════════════════════
#  KnowledgeBase
# ═══════════════════════════════════════════════════════════════════════

# Default path for the bundled rule set (relative to project root)
_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "electrochemistry_rules.json"


class KnowledgeBase:
    """Registry of :class:`ElectrochemicalRule` instances.

    A ``KnowledgeBase`` can be populated programmatically via
    :meth:`add_rule` / :meth:`add_rules`, loaded from a JSON file via
    :meth:`from_json`, or instantiated with the bundled rule set via
    :meth:`default`.

    Parameters
    ----------
    rules : list[ElectrochemicalRule] | None
        Initial rule list.  ``None`` creates an empty base.
    """

    def __init__(self, rules: Optional[List[ElectrochemicalRule]] = None) -> None:
        self._rules: Dict[str, ElectrochemicalRule] = {}
        if rules:
            self.add_rules(rules)

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def rules(self) -> List[ElectrochemicalRule]:
        """Return all rules (order: insertion)."""
        return list(self._rules.values())

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, rule_id: str) -> bool:
        return rule_id in self._rules

    def get(self, rule_id: str) -> Optional[ElectrochemicalRule]:
        """Return a rule by ID, or ``None``."""
        return self._rules.get(rule_id)

    def by_category(self, category: str) -> List[ElectrochemicalRule]:
        """Return rules whose category matches *category* (case-insensitive)."""
        cat = category.lower()
        return [r for r in self._rules.values() if r.category.lower() == cat]

    def by_severity(self, severity: Severity) -> List[ElectrochemicalRule]:
        """Return rules matching *severity*."""
        return [r for r in self._rules.values() if r.severity == severity]

    def by_parameter(self, parameter: str) -> List[ElectrochemicalRule]:
        """Return rules targeting *parameter* (case-insensitive)."""
        p = parameter.lower()
        return [r for r in self._rules.values() if r.parameter.lower() == p]

    @property
    def categories(self) -> List[str]:
        """Unique categories across all rules, sorted."""
        return sorted({r.category for r in self._rules.values()})

    # ── Mutation ──────────────────────────────────────────────────────

    def add_rule(self, rule: ElectrochemicalRule) -> None:
        """Register or update a rule (keyed by ``rule_id``)."""
        self._rules[rule.rule_id] = rule

    def add_rules(self, rules: Sequence[ElectrochemicalRule]) -> None:
        """Batch-add rules."""
        for r in rules:
            self.add_rule(r)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule.  Returns ``True`` if it existed."""
        return self._rules.pop(rule_id, None) is not None

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(
        self,
        measurements: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
        *,
        categories: Optional[Sequence[str]] = None,
    ) -> List[RuleMatch]:
        """Evaluate all rules against concrete *measurements*.

        Parameters
        ----------
        measurements : dict
            Parameter name → measured value, e.g.
            ``{"Rs": 2.5, "Rp": 120.0, "n": 0.85, "retention": 92}``.
        thresholds : dict | None
            Override default thresholds inside condition expressions.
            E.g. ``{"Rs_high_threshold": 10}``.
        categories : list[str] | None
            If given, only evaluate rules from these categories.

        Returns
        -------
        list[RuleMatch]
            Matches sorted by severity (critical → warning → info).
        """
        thresholds = thresholds or {}
        severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        matches: List[RuleMatch] = []

        for rule in self._rules.values():
            if categories and rule.category.lower() not in {c.lower() for c in categories}:
                continue
            resolved = _resolve_thresholds(rule.condition, thresholds)
            hit, actual = _eval_condition(resolved, measurements)
            if hit:
                matches.append(RuleMatch(
                    rule=rule,
                    actual_value=actual,
                    resolved_condition=resolved,
                ))

        matches.sort(key=lambda m: severity_order.get(m.rule.severity, 9))
        return matches

    # ── JSON persistence ──────────────────────────────────────────────

    def to_json(self, path: str | Path) -> None:
        """Write rules to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._rules.values()]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("KnowledgeBase saved → %s  (%d rules)", path, len(data))

    @classmethod
    def from_json(cls, path: str | Path) -> "KnowledgeBase":
        """Load rules from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            raw: list = json.load(fh)
        rules = [ElectrochemicalRule.from_dict(d) for d in raw]
        kb = cls(rules)
        logger.info("KnowledgeBase loaded ← %s  (%d rules)", path, len(rules))
        return kb

    @classmethod
    def from_json_safe(cls, path: str | Path) -> "KnowledgeBase":
        """Load rules from JSON, returning an empty base on failure."""
        try:
            return cls.from_json(path)
        except Exception as exc:
            logger.warning("Failed to load knowledge base from %s: %s", path, exc)
            return cls()

    # ── Bundled rule set ──────────────────────────────────────────────

    @classmethod
    def default(cls) -> "KnowledgeBase":
        """Return the built-in knowledge base shipped with IonFlow.

        If the JSON file is missing, falls back to the hard-coded rules
        defined in :func:`_builtin_rules`.
        """
        if _DEFAULT_RULES_PATH.exists():
            return cls.from_json(_DEFAULT_RULES_PATH)
        logger.info("Bundled JSON not found — using hard-coded rules")
        return cls(_builtin_rules())

    # ── Threshold helpers ─────────────────────────────────────────────

    @staticmethod
    def thresholds_from_config(config: Any) -> Dict[str, float]:
        """Extract evaluation thresholds from a ``PipelineConfig``.

        This lets the rule conditions use ``{voltage}``,
        ``{capacitance_filter_min}`` etc. and have them resolved
        from the user's configuration object.
        """
        try:
            d = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        except TypeError:
            return {}
        return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and v is not None}


# ═══════════════════════════════════════════════════════════════════════
#  Built-in rules (hard-coded fallback)
# ═══════════════════════════════════════════════════════════════════════

def _builtin_rules() -> List[ElectrochemicalRule]:  # noqa: C901
    """Return the hard-coded set of 50+ electrochemistry rules.

    Each rule follows the micro-DSL ``"param op value"`` pattern.
    Curly-brace tokens (e.g. ``{Rs_high}``) are resolved at evaluation
    time; their numeric defaults here represent sensible starting points.
    """
    R = ElectrochemicalRule  # alias for brevity
    S = Severity
    rules: List[ElectrochemicalRule] = []

    # ------------------------------------------------------------------
    # Rs — Ohmic / solution resistance
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="RS_VERY_LOW",
        category="impedance",
        condition="Rs < 0.5",
        parameter="Rs",
        interpretation="Resistência ôhmica muito baixa — contato elétrico excelente ou eletrólito altamente condutivo.",
        possible_causes=[
            "Eletrólito de alta condutividade (H₂SO₄ concentrado, KOH).",
            "Excelente contato entre eletrodo e coletor de corrente.",
        ],
        recommendations=[
            "Verificar se o valor é fisicamente razoável (curto-circuito?).",
        ],
        severity=S.INFO,
        references=["Barsoukov & Macdonald, Impedance Spectroscopy, 3rd ed."],
    ))
    rules.append(R(
        rule_id="RS_LOW",
        category="impedance",
        condition="Rs < 2.0",
        parameter="Rs",
        interpretation="Resistência ôhmica baixa — condição favorável para o desempenho.",
        possible_causes=[
            "Eletrólito condutivo (ex.: H₂SO₄ 1 M).",
            "Bom contato eletrodo–coletor.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="RS_MODERATE",
        category="impedance",
        condition="Rs > 5.0",
        parameter="Rs",
        interpretation="Resistência ôhmica moderadamente elevada — possível limitação na potência.",
        possible_causes=[
            "Eletrólito de condutividade média (ex.: Na₂SO₄).",
            "Resistência de contato entre eletrodo e substrato.",
            "Distância inter-eletrodos elevada na célula.",
        ],
        recommendations=[
            "Verificar o aperto das conexões da célula.",
            "Considerar eletrólito de maior condutividade.",
        ],
        severity=S.WARNING,
        references=["Tutorial IonFlow 03 — Métricas de impedância"],
    ))
    rules.append(R(
        rule_id="RS_HIGH",
        category="impedance",
        condition="Rs > 10.0",
        parameter="Rs",
        interpretation="Resistência ôhmica elevada — limitante significativo para densidade de potência.",
        possible_causes=[
            "Eletrólito de baixa condutividade.",
            "Contato elétrico deficiente (oxidação, sujidade).",
            "Membrana separadora com resistência excessiva.",
        ],
        recommendations=[
            "Polir eletrodo ou usar cola de prata para melhorar contato.",
            "Substituir eletrólito por solução de maior condutividade.",
            "Reduzir a espessura do separador, se aplicável.",
        ],
        severity=S.CRITICAL,
    ))
    rules.append(R(
        rule_id="RS_VERY_HIGH",
        category="impedance",
        condition="Rs > 50.0",
        parameter="Rs",
        interpretation="Resistência ôhmica extremamente alta — possível problema de montagem da célula.",
        possible_causes=[
            "Fio desconectado ou contato intermitente.",
            "Secagem do eletrólito.",
            "Célula montada incorretamente.",
        ],
        recommendations=[
            "Remontar a célula com atenção aos contatos.",
            "Verificar nível de eletrólito.",
            "Repetir a medição após inspeção visual.",
        ],
        severity=S.CRITICAL,
    ))

    # ------------------------------------------------------------------
    # Rp — Charge-transfer resistance (polarization)
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="RP_VERY_LOW",
        category="impedance",
        condition="Rp < 1.0",
        parameter="Rp",
        interpretation="Resistência de transferência de carga muito baixa — cinética eletroquímica extremamente rápida.",
        possible_causes=[
            "Material altamente eletrocatalítico.",
            "Grande área superficial ativa (nanoestruturas).",
        ],
        recommendations=[
            "Confirmar que o valor não é artefato (semicírculo mal resolvido).",
        ],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="RP_LOW",
        category="impedance",
        condition="Rp < 10.0",
        parameter="Rp",
        interpretation="Resistência de transferência de carga baixa — boa cinética de reação.",
        possible_causes=[
            "Material com boa atividade eletroquímica.",
            "Interface eletrodo–eletrólito bem formada.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="RP_MODERATE",
        category="impedance",
        condition="Rp > 50.0",
        parameter="Rp",
        interpretation="Resistência de transferência de carga moderada — cinética pode ser limitante.",
        possible_causes=[
            "Material com atividade catalítica intermediária.",
            "Área ativa limitada.",
            "Barreira energética na interface eletrodo–eletrólito.",
        ],
        recommendations=[
            "Considerar tratamentos de superfície para aumentar área ativa.",
            "Avaliar uso de aditivos catalíticos.",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="RP_HIGH",
        category="impedance",
        condition="Rp > 200.0",
        parameter="Rp",
        interpretation="Resistência de transferência de carga alta — reação eletroquímica muito lenta.",
        possible_causes=[
            "Material pouco eletroativo.",
            "Passivação da superfície do eletrodo.",
            "Contaminação do eletrólito.",
        ],
        recommendations=[
            "Ativar eletroquimicamente (ciclagem de potencial).",
            "Purificar eletrólito ou preparar solução fresca.",
            "Investigar tratamento térmico ou químico do eletrodo.",
        ],
        severity=S.CRITICAL,
    ))
    rules.append(R(
        rule_id="RP_VERY_HIGH",
        category="impedance",
        condition="Rp > 1000.0",
        parameter="Rp",
        interpretation="Resistência de transferência extrema — possível bloqueio da reação.",
        possible_causes=[
            "Filme passivo espesso na superfície.",
            "Incompatibilidade eletrodo–eletrólito.",
            "Eletrodo isolante ou semi-condutor de gap largo.",
        ],
        recommendations=[
            "Verificar integridade do eletrodo.",
            "Testar com outro par eletrodo/eletrólito.",
        ],
        severity=S.CRITICAL,
    ))

    # ------------------------------------------------------------------
    # n — CPE exponent (ideality factor)
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="N_IDEAL",
        category="impedance",
        condition="n > 0.95",
        parameter="n",
        interpretation="CPE exponent próximo de 1 — interface se comporta como capacitor ideal (superfície lisa).",
        possible_causes=[
            "Eletrodo com superfície polida e homogénea.",
            "Monocristal ou filme fino de alta qualidade.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="N_GOOD",
        category="impedance",
        condition="n > 0.85",
        parameter="n",
        interpretation="CPE exponent alto — rugosidade ou porosidade moderada.",
        possible_causes=[
            "Superfície levemente rugosa.",
            "Distribuição estreita de constantes de tempo.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="N_MODERATE",
        category="impedance",
        condition="n < 0.85",
        parameter="n",
        interpretation="CPE exponent moderado — dispersão significativa das constantes de tempo.",
        possible_causes=[
            "Rugosidade superficial elevada.",
            "Porosidade ou heterogeneidade da interface.",
            "Distribuição variada de tamanhos de poro.",
        ],
        recommendations=[
            "Caracterizar morfologia com MEV.",
            "Considerar modelos com múltiplos ZARCs.",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="N_LOW",
        category="impedance",
        condition="n < 0.7",
        parameter="n",
        interpretation="CPE exponent baixo — interface muito heterogénea ou porosa.",
        possible_causes=[
            "Superfície altamente porosa ou fractal.",
            "Múltiplos processos sobrepostos em frequência.",
            "Distribuição larga de constantes de tempo.",
        ],
        recommendations=[
            "Verificar se modelo de circuito é adequado (ZARC duplo?).",
            "Analisar DRT para identificar processos sobrepostos.",
        ],
        severity=S.WARNING,
        references=["Lasia, Electrochemical Impedance Spectroscopy, Ch. 5"],
    ))
    rules.append(R(
        rule_id="N_WARBURG_LIKE",
        category="impedance",
        condition="n < 0.55",
        parameter="n",
        interpretation="CPE exponent próximo de 0.5 — comportamento dominado por difusão (Warburg).",
        possible_causes=[
            "Difusão semi-infinita domina a impedância.",
            "Eletrodo poroso com difusão lenta de íons.",
        ],
        recommendations=[
            "Incluir elemento de Warburg no modelo.",
            "Verificar se a faixa de frequência é suficientemente baixa.",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="N_VERY_LOW",
        category="impedance",
        condition="n < 0.3",
        parameter="n",
        interpretation="CPE exponent muito baixo — possível artefato ou sistema altamente não-ideal.",
        possible_causes=[
            "Fitting deficiente (mínimo local).",
            "Sistema não descritível por circuito equivalente simples.",
        ],
        recommendations=[
            "Rever os dados brutos e refazer o fitting.",
            "Testar com circuitos de maior complexidade.",
        ],
        severity=S.CRITICAL,
    ))

    # ------------------------------------------------------------------
    # Sigma — Warburg coefficient
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="SIGMA_LOW",
        category="impedance",
        condition="Sigma < 5.0",
        parameter="Sigma",
        interpretation="Coeficiente de Warburg baixo — difusão muito rápida ou inexistente.",
        possible_causes=[
            "Material com alta difusividade iônica.",
            "Eletrólito de baixa viscosidade.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="SIGMA_MODERATE",
        category="impedance",
        condition="Sigma > 20.0",
        parameter="Sigma",
        interpretation="Coeficiente de Warburg moderado — difusão contribui significativamente.",
        possible_causes=[
            "Difusão semi-infinita em material poroso.",
            "Concentração limitante de espécie ativa.",
        ],
        recommendations=[
            "Considerar medição a frequências mais baixas (< 10 mHz).",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="SIGMA_HIGH",
        category="impedance",
        condition="Sigma > 100.0",
        parameter="Sigma",
        interpretation="Coeficiente de Warburg alto — difusão é o principal gargalo.",
        possible_causes=[
            "Material com baixa difusividade iônica.",
            "Eletrodo espesso limitando transporte de massa.",
            "Depleção de espécie eletroativa.",
        ],
        recommendations=[
            "Reduzir espessura do eletrodo.",
            "Aumentar concentração do eletrólito.",
            "Considerar nanoestruturas para encurtar caminhos de difusão.",
        ],
        severity=S.CRITICAL,
    ))

    # ------------------------------------------------------------------
    # Capacitance (C_mean)
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="C_VERY_LOW",
        category="impedance",
        condition="C_mean < 1e-06",
        parameter="C_mean",
        interpretation="Capacitância muito baixa (< 1 μF) — pouca carga armazenável.",
        possible_causes=[
            "Área superficial ativa insuficiente.",
            "Material com baixa constante dielétrica.",
        ],
        recommendations=[
            "Aumentar a área ativa (nanopartículas, nanofibras).",
            "Trocar material de eletrodo por pseudocapacitivo.",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="C_LOW",
        category="impedance",
        condition="C_mean < 1e-04",
        parameter="C_mean",
        interpretation="Capacitância baixa — armazenamento de carga limitado.",
        possible_causes=[
            "Área superficial moderada.",
            "Material predominantemente EDLC com pouca pseudocapacitância.",
        ],
        recommendations=[
            "Considerar deposição de óxidos de metal de transição.",
        ],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="C_HIGH",
        category="impedance",
        condition="C_mean > 0.01",
        parameter="C_mean",
        interpretation="Capacitância elevada — excelente armazenamento de carga.",
        possible_causes=[
            "Alta área superficial (carvão ativado, grafeno).",
            "Forte contribuição pseudocapacitiva.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="C_UNREASONABLE",
        category="impedance",
        condition="C_mean > 10.0",
        parameter="C_mean",
        interpretation="Capacitância irrealisticamente alta — possível artefato.",
        possible_causes=[
            "Fitting convergiu para mínimo local.",
            "Dados ruidosos em baixa frequência.",
            "Corrente faradaica domina (bateria, não supercapacitor).",
        ],
        recommendations=[
            "Rever dados brutos em baixa frequência.",
            "Refazer fitting com bounds mais restritivos.",
        ],
        severity=S.CRITICAL,
    ))

    # ------------------------------------------------------------------
    # Q — CPE pseudo-capacitance
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="Q_VERY_LOW",
        category="impedance",
        condition="Q < 1e-08",
        parameter="Q",
        interpretation="Q do CPE muito baixo — interface com carácter capacitivo fraco.",
        possible_causes=[
            "Área ativa muito pequena.",
            "Material isolante parcial.",
        ],
        recommendations=[
            "Verificar se o eletrodo está em bom estado.",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="Q_HIGH",
        category="impedance",
        condition="Q > 0.1",
        parameter="Q",
        interpretation="Q do CPE elevado — forte comportamento pseudocapacitivo.",
        possible_causes=[
            "Reações faradaicas de superfície.",
            "Área ativa muito grande.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))

    # ------------------------------------------------------------------
    # Tau — time constant
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="TAU_FAST",
        category="impedance",
        condition="Tau < 0.001",
        parameter="Tau",
        interpretation="Constante de tempo muito rápida (< 1 ms) — resposta eletroquímica rápida.",
        possible_causes=[
            "Processo de transferência de carga rápido.",
            "Sistema de alta potência.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="TAU_SLOW",
        category="impedance",
        condition="Tau > 1.0",
        parameter="Tau",
        interpretation="Constante de tempo lenta (> 1 s) — resposta eletroquímica lenta.",
        possible_causes=[
            "Processo difusivo lento.",
            "Interface com cinética limitada.",
        ],
        recommendations=[
            "Verificar se a medição cobriu frequências baixas o suficiente.",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="TAU_VERY_SLOW",
        category="impedance",
        condition="Tau > 10.0",
        parameter="Tau",
        interpretation="Constante de tempo muito lenta — possível artefato ou processo de degradação.",
        possible_causes=[
            "Difusão em estado sólido.",
            "Crescimento de filme passivo durante a medição.",
        ],
        recommendations=[
            "Repetir medição com menor tempo de aquisição.",
            "Verificar estabilidade do OCP antes da medição.",
        ],
        severity=S.CRITICAL,
    ))

    # ------------------------------------------------------------------
    # Dispersion
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="DISP_LOW",
        category="impedance",
        condition="Dispersion < 0.1",
        parameter="Dispersion",
        interpretation="Dispersão baixa — dados de impedância muito consistentes.",
        possible_causes=["Medição estável e reprodutível."],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="DISP_HIGH",
        category="impedance",
        condition="Dispersion > 0.5",
        parameter="Dispersion",
        interpretation="Dispersão alta — ruído significativo nos dados.",
        possible_causes=[
            "Instabilidade do sistema durante a medição.",
            "Interferência eletromagnética.",
            "OCP não estabilizado.",
        ],
        recommendations=[
            "Verificar blindagem dos cabos.",
            "Estabilizar OCP antes da medição (min. 30 min).",
        ],
        severity=S.WARNING,
    ))

    # ------------------------------------------------------------------
    # Cycling — Retention (%)
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="RET_EXCELLENT",
        category="cycling",
        condition="retention > 95.0",
        parameter="retention",
        interpretation="Retenção excelente (> 95 %) — material muito estável ciclicamente.",
        possible_causes=[
            "Estrutura robusta sem degradação mecânica.",
            "Boa adesão do material ativo ao substrato.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="RET_GOOD",
        category="cycling",
        condition="retention > 80.0",
        parameter="retention",
        interpretation="Retenção boa (80–95 %) — estabilidade aceitável para a maioria das aplicações.",
        possible_causes=[
            "Perda gradual de material ativo.",
            "Degradação leve da interface.",
        ],
        recommendations=[
            "Considerar aumentar ciclos para confirmar tendência.",
        ],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="RET_MODERATE",
        category="cycling",
        condition="retention < 80.0",
        parameter="retention",
        interpretation="Retenção moderada (< 80 %) — degradação significativa.",
        possible_causes=[
            "Dissolução parcial do material ativo.",
            "Descolamento do eletrodo do substrato.",
            "Mudanças estruturais irreversíveis.",
        ],
        recommendations=[
            "Adicionar binder ou agente de ligação.",
            "Reduzir janela de potencial.",
            "Investigar mecanismo de degradação com SEM/XRD.",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="RET_POOR",
        category="cycling",
        condition="retention < 60.0",
        parameter="retention",
        interpretation="Retenção fraca (< 60 %) — degradação severa do material.",
        possible_causes=[
            "Material ativo se dissolve no eletrólito.",
            "Colapso estrutural do eletrodo.",
            "Evolução de gás destruindo a interface.",
        ],
        recommendations=[
            "Revisar composição do eletrodo e processo de fabricação.",
            "Testar com janela de potencial mais estreita.",
            "Considerar encapsulamento ou coating protetor.",
        ],
        severity=S.CRITICAL,
    ))
    rules.append(R(
        rule_id="RET_VERY_POOR",
        category="cycling",
        condition="retention < 40.0",
        parameter="retention",
        interpretation="Retenção muito fraca (< 40 %) — o material não é adequado para esta aplicação.",
        possible_causes=[
            "Incompatibilidade completa material–eletrólito.",
            "Falha mecânica catastrófica.",
        ],
        recommendations=[
            "Mudar a estratégia de material/eletrólito.",
            "Verificar se a célula não está danificada.",
        ],
        severity=S.CRITICAL,
    ))

    # ------------------------------------------------------------------
    # Cycling — Energy
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="ENERGY_HIGH",
        category="cycling",
        condition="Energy_mean > 50.0",
        parameter="Energy_mean",
        interpretation="Energia específica elevada — bom armazenamento de energia.",
        possible_causes=["Alta capacitância e/ou larga janela de potencial."],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="ENERGY_LOW",
        category="cycling",
        condition="Energy_mean < 1.0",
        parameter="Energy_mean",
        interpretation="Energia específica muito baixa.",
        possible_causes=[
            "Capacitância reduzida.",
            "Janela de potencial estreita.",
        ],
        recommendations=[
            "Otimizar eletrólito ou janela de potencial.",
        ],
        severity=S.WARNING,
    ))

    # ------------------------------------------------------------------
    # DRT — peaks
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="DRT_SINGLE_PEAK",
        category="drt",
        condition="n_peaks == 1",
        parameter="n_peaks",
        interpretation="Apenas um pico DRT — processo eletroquímico único e bem definido.",
        possible_causes=[
            "Interface simples com uma constante de tempo dominante.",
            "Transferência de carga como processo limitante.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="DRT_TWO_PEAKS",
        category="drt",
        condition="n_peaks == 2",
        parameter="n_peaks",
        interpretation="Dois picos DRT — dois processos eletroquímicos distintos.",
        possible_causes=[
            "Transferência de carga + difusão.",
            "Duas interfaces (ex.: coating + dupla camada).",
        ],
        recommendations=[
            "Considerar circuito de dois arcos (Two-Arc-CPE ou ZARC-ZARC-W).",
        ],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="DRT_MANY_PEAKS",
        category="drt",
        condition="n_peaks > 3",
        parameter="n_peaks",
        interpretation="Muitos picos DRT (> 3) — sistema complexo com múltiplos processos.",
        possible_causes=[
            "Múltiplas interfaces (eletrodo poroso).",
            "Processos sobrepostos de cinética diferente.",
            "Artefatos do DRT (ruído amplificado).",
        ],
        recommendations=[
            "Verificar qualidade dos dados (KK validation).",
            "Ajustar λ do DRT para suavizar artefatos.",
            "Usar circuito com ≥ 2 ZARCs + Warburg.",
        ],
        severity=S.WARNING,
    ))

    # ------------------------------------------------------------------
    # DRT — peak positions (tau_peak_main)
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="DRT_PEAK_FAST",
        category="drt",
        condition="tau_peak_main < 1e-04",
        parameter="tau_peak_main",
        interpretation="Pico DRT principal em alta frequência (τ < 0.1 ms) — processo rápido.",
        possible_causes=[
            "Transferência de carga rápida.",
            "Migração iônica no bulk do eletrólito.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="DRT_PEAK_MEDIUM",
        category="drt",
        condition="tau_peak_main > 0.01",
        parameter="tau_peak_main",
        interpretation="Pico DRT principal em média frequência (τ > 10 ms) — transferência de carga moderada.",
        possible_causes=[
            "Cinética de transferência de carga na interface.",
            "Processo de adsorção/dessorção.",
        ],
        recommendations=[],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="DRT_PEAK_SLOW",
        category="drt",
        condition="tau_peak_main > 1.0",
        parameter="tau_peak_main",
        interpretation="Pico DRT principal em baixa frequência (τ > 1 s) — processo lento (difusão).",
        possible_causes=[
            "Difusão de espécies em poros ou bulk.",
            "Transporte lento de massa.",
        ],
        recommendations=[
            "Avaliar elemento de Warburg no circuito.",
            "Medir a frequências mais baixas para resolver completamente.",
        ],
        severity=S.WARNING,
    ))

    # ------------------------------------------------------------------
    # DRT — gamma_peak_main (amplitude)
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="DRT_GAMMA_HIGH",
        category="drt",
        condition="gamma_peak_main > 100.0",
        parameter="gamma_peak_main",
        interpretation="Amplitude DRT alta — contribuição resistiva dominante neste processo.",
        possible_causes=[
            "Resistência de transferência de carga significativa.",
            "Interface com área ativa limitada.",
        ],
        recommendations=[
            "Correlacionar com Rp do fitting de circuito.",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="DRT_GAMMA_LOW",
        category="drt",
        condition="gamma_peak_main < 1.0",
        parameter="gamma_peak_main",
        interpretation="Amplitude DRT baixa — contribuição resistiva pequena.",
        possible_causes=[
            "Cinética muito rápida.",
            "Pico mal resolvido (pode ser artefato).",
        ],
        recommendations=[
            "Verificar resolução dos dados de impedância.",
        ],
        severity=S.INFO,
    ))

    # ------------------------------------------------------------------
    # Cross-pipeline / Correlations
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="CORR_RS_HIGH_POWER_LOW",
        category="correlation",
        condition="Rs > 10.0",
        parameter="Rs",
        interpretation="Rs alto correlaciona com baixa potência — resistência ôhmica é o gargalo.",
        possible_causes=[
            "Perda ôhmica domina, especialmente em altas taxas.",
        ],
        recommendations=[
            "Priorizar redução de Rs para aumentar potência.",
            "Verificar contatos e trocar eletrólito.",
        ],
        severity=S.CRITICAL,
    ))
    rules.append(R(
        rule_id="CORR_RP_HIGH_ENERGY_LOW",
        category="correlation",
        condition="Rp > 200.0",
        parameter="Rp",
        interpretation="Rp alto com energia baixa — transferência de carga é o limitante.",
        possible_causes=[
            "Cinética lenta impede acesso à capacitância total.",
        ],
        recommendations=[
            "Ativar eletrodo (ciclagem de potencial) ou adicionar catalisador.",
        ],
        severity=S.CRITICAL,
    ))
    rules.append(R(
        rule_id="CORR_N_LOW_PEAKS_MANY",
        category="correlation",
        condition="n < 0.7",
        parameter="n",
        interpretation="n baixo + múltiplos picos DRT — interface heterogénea com processos sobrepostos.",
        possible_causes=[
            "Rugosidade extrema gerando distribuição de constantes de tempo.",
        ],
        recommendations=[
            "Usar modelo de circuito com múltiplos ZARCs.",
            "Caracterizar morfologia com BET e MEV.",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="CORR_RET_LOW_RP_STABLE",
        category="correlation",
        condition="retention < 70.0",
        parameter="retention",
        interpretation="Retenção baixa com Rp estável sugere degradação mecânica, não eletroquímica.",
        possible_causes=[
            "Material ativo se descolando do substrato.",
            "Expansão/contração volumétrica durante ciclagem.",
        ],
        recommendations=[
            "Melhorar adesão ao substrato (binder, pressing).",
            "Reduzir janela de potencial.",
        ],
        severity=S.WARNING,
    ))

    # ------------------------------------------------------------------
    # Score / Rank
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="SCORE_TOP",
        category="general",
        condition="Score > 0.8",
        parameter="Score",
        interpretation="Score composto elevado — esta amostra se destaca nas métricas combinadas.",
        possible_causes=[
            "Bom equilíbrio entre resistência, capacitância e energia.",
        ],
        recommendations=[
            "Priorizar esta amostra para estudos aprofundados.",
        ],
        severity=S.INFO,
    ))
    rules.append(R(
        rule_id="SCORE_LOW",
        category="general",
        condition="Score < 0.2",
        parameter="Score",
        interpretation="Score composto baixo — esta amostra tem desempenho inferior na maioria das métricas.",
        possible_causes=[
            "Resistências altas, capacitância baixa, ou energia insuficiente.",
        ],
        recommendations=[
            "Identificar o parâmetro mais limitante e focar na sua melhoria.",
        ],
        severity=S.WARNING,
    ))

    # ------------------------------------------------------------------
    # KK validation
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="KK_FAILED",
        category="general",
        condition="kk_residual > 5.0",
        parameter="kk_residual",
        interpretation="Teste de Kramers-Kronig falhou (resíduos > 5 %) — dados podem não ser lineares/estacionários.",
        possible_causes=[
            "Sistema não estava em estado estacionário.",
            "Não-linearidade nas medições.",
            "Ruído excessivo ou artefatos instrumentais.",
        ],
        recommendations=[
            "Repetir a medição com OCP estabilizado.",
            "Reduzir amplitude de excitação (≤ 10 mV).",
            "Verificar blindagem da célula eletroquímica.",
        ],
        severity=S.CRITICAL,
        references=["Boukamp, Solid State Ionics, 1986"],
    ))
    rules.append(R(
        rule_id="KK_MARGINAL",
        category="general",
        condition="kk_residual > 1.0",
        parameter="kk_residual",
        interpretation="Kramers-Kronig marginal (1–5 %) — dados aceitáveis, mas com algum desvio.",
        possible_causes=[
            "Pequena instabilidade durante a medição.",
            "Ruído em alta ou baixa frequência.",
        ],
        recommendations=[
            "Verificar estabilidade do OCP.",
            "Considerar descartar pontos extremos de frequência.",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="KK_GOOD",
        category="general",
        condition="kk_residual < 1.0",
        parameter="kk_residual",
        interpretation="Kramers-Kronig excelente (< 1 %) — dados confiáveis.",
        possible_causes=["Medição estável e linear."],
        recommendations=[],
        severity=S.INFO,
    ))

    # ------------------------------------------------------------------
    # Fitting quality
    # ------------------------------------------------------------------
    rules.append(R(
        rule_id="FIT_RSS_HIGH",
        category="general",
        condition="fit_rss > 0.1",
        parameter="fit_rss",
        interpretation="Resíduo do fitting alto — o modelo de circuito pode não ser adequado.",
        possible_causes=[
            "Modelo muito simples para os dados.",
            "Dados ruidosos perturbando a convergência.",
        ],
        recommendations=[
            "Testar circuitos com mais elementos.",
            "Verificar qualidade dos dados (KK).",
        ],
        severity=S.WARNING,
    ))
    rules.append(R(
        rule_id="FIT_BOUND_HITS",
        category="general",
        condition="fit_bound_hits > 2",
        parameter="fit_bound_hits",
        interpretation="Múltiplos parâmetros atingiram os limites — o fitting pode não ter convergido bem.",
        possible_causes=[
            "Bounds demasiado restritivos.",
            "Modelo não adequado aos dados.",
        ],
        recommendations=[
            "Ampliar os limites dos parâmetros.",
            "Experimentar outro circuito equivalente.",
        ],
        severity=S.WARNING,
    ))

    return rules
