"""Intelligent textual fitting report — interprets circuit parameters.

Generates human-readable reports that explain *why* a circuit was chosen,
what each parameter means physically, how the fit quality compares with
previous samples, and what improvements can be made.

Public API
----------
``FittingReport``
    Dataclass holding the report sections.
``FittingReportGenerator``
    Main generator: ``generate(fit_result, …) → FittingReport``.

Day 10 of the UPGRADE_PLAN_v0.2.0 schedule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.fitting_diagnostics import QualityIndicator, assess_quality

logger = logging.getLogger(__name__)


# =====================================================================
# FittingReport dataclass
# =====================================================================

@dataclass
class FittingReport:
    """Structured textual report for a single circuit fitting result.

    Attributes
    ----------
    summary : str
        One-paragraph overview of the fitting outcome.
    circuit_justification : str
        Why this particular circuit was selected.
    parameter_interpretation : dict[str, str]
        Per-parameter physical interpretation.
    quality_assessment : str
        Traffic-light text on fit quality.
    recommendations : list[str]
        Actionable suggestions for improving the fit.
    comparison_with_similar : str
        How this sample compares with historical data.
    """

    summary: str = ""
    circuit_justification: str = ""
    parameter_interpretation: Dict[str, str] = field(default_factory=dict)
    quality_assessment: str = ""
    recommendations: List[str] = field(default_factory=list)
    comparison_with_similar: str = ""

    def to_text(self) -> str:
        """Render the full report as plain text."""
        sections: List[str] = []

        if self.summary:
            sections.append(f"## Resumo\n{self.summary}")

        if self.circuit_justification:
            sections.append(
                f"## Justificação do Circuito\n{self.circuit_justification}"
            )

        if self.parameter_interpretation:
            lines = ["## Interpretação dos Parâmetros"]
            for pname, interp in self.parameter_interpretation.items():
                lines.append(f"  • **{pname}**: {interp}")
            sections.append("\n".join(lines))

        if self.quality_assessment:
            sections.append(f"## Qualidade do Ajuste\n{self.quality_assessment}")

        if self.recommendations:
            lines = ["## Recomendações"]
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            sections.append("\n".join(lines))

        if self.comparison_with_similar:
            sections.append(
                f"## Comparação com Amostras Anteriores\n"
                f"{self.comparison_with_similar}"
            )

        return "\n\n".join(sections)


# =====================================================================
# Parameter interpretation templates
# =====================================================================

# Physical meaning templates keyed by canonical parameter name.
# Each entry is a tuple (unit, low_text, mid_text, high_text, threshold_low, threshold_high)

_PARAM_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Rs": {
        "unit": "Ω",
        "desc": "resistência ôhmica do eletrólito",
        "low": "valor baixo, típico de eletrólito concentrado ou bem condutor",
        "mid": "valor moderado, compatível com eletrólito aquoso convencional",
        "high": "valor elevado, pode indicar eletrólito resistivo ou distância inter-eléctrodos grande",
        "th_low": 5.0,
        "th_high": 50.0,
    },
    "Rp": {
        "unit": "Ω",
        "desc": "resistência de transferência de carga (polarização)",
        "low": "cinética rápida — bom desempenho electroquímico",
        "mid": "cinética moderada, típica de eléctrodos standard",
        "high": "cinética lenta, possível passivação ou degradação da superfície",
        "th_low": 20.0,
        "th_high": 500.0,
    },
    "Q": {
        "unit": "F·s^(n−1)",
        "desc": "pseudo-capacitância CPE",
        "low": "baixa capacitância, área electroactiva reduzida",
        "mid": "capacitância moderada",
        "high": "alta capacitância, superfície rugosa ou porosa com grande área",
        "th_low": 1e-6,
        "th_high": 1e-3,
    },
    "n": {
        "unit": "(adimensional)",
        "desc": "expoente CPE — dispersão da constante de tempo",
        "low": "forte dispersão, superfície muito heterogénea (n≈0.5 → comportamento Warburg)",
        "mid": "dispersão moderada, rugosidade de superfície comum",
        "high": "próximo do ideal capacitivo (n≈1.0)",
        "th_low": 0.6,
        "th_high": 0.9,
    },
    "Sigma": {
        "unit": "Ω·s^(−½)",
        "desc": "coeficiente de Warburg (difusão semi-infinita)",
        "low": "difusão rápida — bom transporte de massa",
        "mid": "difusão moderada",
        "high": "difusão lenta — possível limitação por transporte de massa",
        "th_low": 1.0,
        "th_high": 100.0,
    },
    "C": {
        "unit": "F",
        "desc": "capacitância ideal da dupla camada",
        "low": "área electroactiva pequena ou dupla camada pouco desenvolvida",
        "mid": "capacitância típica de dupla camada em meio aquoso",
        "high": "alta capacitância, típica de eléctrodos de elevada área específica",
        "th_low": 1e-7,
        "th_high": 1e-4,
    },
    "L": {
        "unit": "H",
        "desc": "indutância em série",
        "low": "indutância residual (artefacto de cabos curtos)",
        "mid": "indutância moderada, verificar ligações experimentais",
        "high": "indutância significativa — possível adsorção ou artefacto de medição",
        "th_low": 1e-7,
        "th_high": 1e-3,
    },
    "Rd": {
        "unit": "Ω",
        "desc": "resistência de difusão (Warburg finito)",
        "low": "camada de difusão fina com baixa resistência",
        "mid": "resistência de difusão moderada",
        "high": "camada de difusão espessa ou difusão muito restrita",
        "th_low": 1.0,
        "th_high": 200.0,
    },
    "Td": {
        "unit": "s",
        "desc": "constante de tempo de difusão (L²/D)",
        "low": "difusão rápida — filme fino ou alto coeficiente D",
        "mid": "constante de tempo moderada",
        "high": "difusão lenta — filme espesso ou baixo coeficiente D",
        "th_low": 0.01,
        "th_high": 10.0,
    },
}

# Map common suffixed parameter names to their canonical root
_CANONICAL_MAP: Dict[str, str] = {
    "Rp1": "Rp", "Rp2": "Rp",
    "Q1": "Q", "Q2": "Q",
    "n1": "n", "n2": "n",
    "R1": "Rp", "R2": "Rp",
    "Rcoat": "Rp", "Rct": "Rp",
    "Qcoat": "Q", "Qdl": "Q",
    "ncoat": "n", "ndl": "n",
}


def _canonical(param_name: str) -> str:
    """Map a parameter name to its canonical form for template lookup."""
    if param_name in _PARAM_TEMPLATES:
        return param_name
    if param_name in _CANONICAL_MAP:
        return _CANONICAL_MAP[param_name]
    # Try stripping prefix like "R_R", "ZARC_R" etc.
    parts = param_name.split("_")
    for part in reversed(parts):
        if part in _PARAM_TEMPLATES:
            return part
        if part in _CANONICAL_MAP:
            return _CANONICAL_MAP[part]
    return param_name


# =====================================================================
# FittingReportGenerator
# =====================================================================

class FittingReportGenerator:
    """Generate intelligent textual reports for circuit fitting results.

    Parameters
    ----------
    thresholds : dict | None
        Override quality-assessment thresholds (passed to ``assess_quality``).

    Examples
    --------
    >>> gen = FittingReportGenerator()
    >>> report = gen.generate(fit_result=best, history=history, config=cfg)
    >>> print(report.to_text())
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.thresholds = thresholds

    def generate(
        self,
        fit_result: Dict[str, Any],
        *,
        history: Optional[Any] = None,
        config: Optional[Any] = None,
        all_results: Optional[List[Dict[str, Any]]] = None,
        spectral_features: Optional[Dict[str, float]] = None,
        registry_template: Optional[Any] = None,
    ) -> FittingReport:
        """Generate a complete :class:`FittingReport`.

        Parameters
        ----------
        fit_result : dict
            As returned by ``fit_template()`` or ``run_shortlist_fit()["best"]``.
        history : FittingHistory | None
            For comparison with previous samples.
        config : PipelineConfig | None
            Pipeline configuration (optional context).
        all_results : list[dict] | None
            All candidate results (for ranking context).
        spectral_features : dict | None
            Spectral features for history comparison.
        registry_template : CircuitTemplate | None
            Registry template with ``physical_meaning`` and ``description``.
        """
        quality = assess_quality(fit_result, thresholds=self.thresholds)

        report = FittingReport(
            summary=self._build_summary(fit_result, quality),
            circuit_justification=self._build_justification(
                fit_result, all_results, registry_template,
            ),
            parameter_interpretation=self._build_param_interpretation(
                fit_result, registry_template,
            ),
            quality_assessment=self._build_quality_text(fit_result, quality),
            recommendations=self._build_recommendations(fit_result, quality),
            comparison_with_similar=self._build_comparison(
                fit_result, history, spectral_features,
            ),
        )

        logger.info(
            "FittingReport generated for %s — %s",
            fit_result.get("template", "?"),
            quality.label,
        )
        return report

    # ── Section builders ─────────────────────────────────────────

    def _build_summary(
        self,
        fit_result: Dict[str, Any],
        quality: QualityIndicator,
    ) -> str:
        """One-paragraph summary of the fit."""
        name = fit_result.get("template", "desconhecido")
        diagram = fit_result.get("diagram", "")
        rss = fit_result.get("rss", np.nan)
        bic = fit_result.get("bic", np.nan)
        n_params = fit_result.get("n_params", 0)
        n_points = fit_result.get("n_points", 0)
        confidence = fit_result.get("confidence", np.nan)

        parts = [
            f"O circuito **{name}** ({diagram}) foi ajustado com "
            f"{n_params} parâmetros a {n_points} pontos experimentais.",
        ]

        if np.isfinite(rss):
            parts.append(f"RSS = {rss:.4f}.")
        if np.isfinite(bic):
            parts.append(f"BIC = {bic:.2f}.")
        if np.isfinite(confidence):
            parts.append(f"Confiança relativa = {confidence:.1%}.")

        parts.append(f"{quality.emoji} Avaliação: {quality.label}.")

        return " ".join(parts)

    def _build_justification(
        self,
        fit_result: Dict[str, Any],
        all_results: Optional[List[Dict[str, Any]]],
        registry_template: Optional[Any],
    ) -> str:
        """Explain why this circuit was selected."""
        name = fit_result.get("template", "desconhecido")
        parts: List[str] = []

        # Description from registry
        if registry_template and hasattr(registry_template, "description"):
            desc = getattr(registry_template, "description", "")
            if desc:
                parts.append(desc)

        # Typical systems
        if registry_template and hasattr(registry_template, "typical_systems"):
            systems = getattr(registry_template, "typical_systems", [])
            if systems:
                parts.append(
                    f"Sistemas típicos: {', '.join(systems[:4])}."
                )

        # Ranking context
        if all_results and len(all_results) > 1:
            rank = None
            for i, r in enumerate(all_results):
                if r.get("template") == name:
                    rank = i + 1
                    break
            if rank is not None:
                parts.append(
                    f"Classificado em {rank}º lugar entre "
                    f"{len(all_results)} candidatos por BIC."
                )
            # Delta BIC to runner-up
            bics_valid = [
                r["bic"] for r in all_results
                if np.isfinite(r.get("bic", np.inf))
            ]
            if len(bics_valid) >= 2:
                bics_sorted = sorted(bics_valid)
                delta = bics_sorted[1] - bics_sorted[0]
                if delta > 10:
                    parts.append(
                        f"ΔBIC = {delta:.1f} em relação ao segundo candidato "
                        f"(forte evidência a favor)."
                    )
                elif delta > 2:
                    parts.append(
                        f"ΔBIC = {delta:.1f} (evidência moderada a favor)."
                    )
                else:
                    parts.append(
                        f"ΔBIC = {delta:.1f} (pouca diferença entre candidatos)."
                    )

        if not parts:
            parts.append(
                f"O circuito {name} foi selecionado com base nos critérios BIC/AIC."
            )

        return " ".join(parts)

    def _build_param_interpretation(
        self,
        fit_result: Dict[str, Any],
        registry_template: Optional[Any],
    ) -> Dict[str, str]:
        """Interpret each fitted parameter using templates."""
        params = fit_result.get("params", {})
        params_std = fit_result.get("params_std", {})
        if not params:
            return {}

        # Get physical_meaning from registry if available
        phys_meaning: Dict[str, str] = {}
        if registry_template and hasattr(registry_template, "physical_meaning"):
            phys_meaning = getattr(registry_template, "physical_meaning", {}) or {}

        interp: Dict[str, str] = {}
        for pname, value in params.items():
            canon = _canonical(pname)
            std_val = params_std.get(pname, 0.0)

            # Value + uncertainty string
            if std_val and np.isfinite(std_val) and std_val > 0:
                val_str = f"{value:.4g} ± {std_val:.2g}"
            else:
                val_str = f"{value:.4g}"

            parts: List[str] = []

            # Registry meaning (highest priority)
            if pname in phys_meaning:
                parts.append(phys_meaning[pname])

            # Template-based interpretation
            tpl = _PARAM_TEMPLATES.get(canon)
            if tpl:
                unit = tpl["unit"]
                parts.insert(0, f"{val_str} {unit}")

                if value <= tpl["th_low"]:
                    parts.append(tpl["low"])
                elif value >= tpl["th_high"]:
                    parts.append(tpl["high"])
                else:
                    parts.append(tpl["mid"])
            else:
                parts.insert(0, f"{val_str}")

            interp[pname] = " — ".join(parts)

        return interp

    def _build_quality_text(
        self,
        fit_result: Dict[str, Any],
        quality: QualityIndicator,
    ) -> str:
        """Textual quality assessment."""
        parts = [f"{quality.emoji} **{quality.label}**"]
        for reason in quality.reasons:
            parts.append(f"  • {reason}")
        return "\n".join(parts)

    def _build_recommendations(
        self,
        fit_result: Dict[str, Any],
        quality: QualityIndicator,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs: List[str] = []

        rss = fit_result.get("rss", np.nan)
        n_pts = fit_result.get("n_points", 1)
        structured = fit_result.get("res_structured", False)
        bound_hits = fit_result.get("bound_hits", 0)
        success = fit_result.get("success", True)
        autocorr = fit_result.get("res_autocorr", 0.0)

        if not success:
            recs.append(
                "O optimizador não convergiu. Tente aumentar max_nfev ou "
                "ajustar os valores iniciais (p0)."
            )

        if structured:
            recs.append(
                "Os resíduos são estruturados (autocorrelação alta). "
                "Considere um circuito mais complexo que capture a "
                "resposta em frequência não modelada."
            )

        if bound_hits >= 2:
            recs.append(
                f"{bound_hits} parâmetros estão nos limites dos bounds. "
                "Verifique se os bounds são adequados ao sistema em estudo "
                "ou alargue-os cautelosamente."
            )

        norm_rss = rss / max(n_pts, 1)
        if np.isfinite(norm_rss) and norm_rss > 0.1:
            recs.append(
                "O RSS normalizado é elevado. Considere verificar a "
                "qualidade dos dados experimentais (outliers, ruído "
                "excessivo, artefactos de medição)."
            )

        if np.isfinite(autocorr) and abs(autocorr) > 0.3 and not structured:
            recs.append(
                "A autocorrelação residual está alta. Um elemento de "
                "difusão (W ou W_finite) ou um segundo ZARC podem melhorar."
            )

        if quality.level == "green" and not recs:
            recs.append(
                "O ajuste está excelente. Nenhuma acção adicional necessária."
            )

        return recs

    def _build_comparison(
        self,
        fit_result: Dict[str, Any],
        history: Optional[Any],
        spectral_features: Optional[Dict[str, float]],
    ) -> str:
        """Compare with historical fittings via FittingHistory."""
        if history is None or spectral_features is None:
            return "Sem dados históricos para comparação."

        # Use FittingHistory.summary_text() if available
        if hasattr(history, "summary_text"):
            base_text = history.summary_text(spectral_features)
        else:
            return "Histórico indisponível."

        # Parameter-level comparison
        params = fit_result.get("params", {})
        if not params:
            return base_text

        if hasattr(history, "similar_samples"):
            similar = history.similar_samples(spectral_features, n=10)
        else:
            return base_text

        if not similar:
            return base_text

        # Compute average parameters from similar samples
        param_comparison_lines: List[str] = []
        for pname, value in params.items():
            hist_values = []
            for rec in similar:
                rec_params = rec.get("params", {})
                if isinstance(rec_params, dict) and pname in rec_params:
                    v = rec_params[pname]
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        hist_values.append(v)
            if hist_values and np.isfinite(value) and value != 0:
                mean_hist = np.mean(hist_values)
                if mean_hist != 0:
                    pct_diff = ((value - mean_hist) / abs(mean_hist)) * 100
                    direction = "maior" if pct_diff > 0 else "menor"
                    param_comparison_lines.append(
                        f"{pname} = {value:.4g} ({abs(pct_diff):.0f}% "
                        f"{direction} que a média histórica de {mean_hist:.4g})"
                    )

        if param_comparison_lines:
            comparisons = "\n  • ".join(param_comparison_lines)
            return f"{base_text}\n\nComparação paramétrica:\n  • {comparisons}"

        return base_text
