"""Modo Orientador — critical experiment evaluation for EIS data.

Aggregates all quality signals (KK, fitting diagnostics, zombie params,
signal-level checks) into a single "professor-style" verdict.

Public API
----------
``CriterionResult``
    Score (0–10), status, and message for one diagnostic criterion.
``ExperimentEvaluation``
    Full evaluation result with overall score and article fragment.
``ExperimentEvaluator``
    Main evaluator: ``evaluate(freq, z, …) → ExperimentEvaluation``.
``format_report(evaluation) → str``
    Render the evaluation as a formatted console report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────


@dataclass
class CriterionResult:
    """Result for one diagnostic criterion.

    Attributes
    ----------
    name : str
        Human-readable criterion name.
    score : float
        Score in [0, 10].
    emoji : str
        Traffic-light emoji (🟢 / 🟡 / 🔴).
    message : str
        One-line summary.
    details : list[str]
        Additional findings or recommendations.
    """

    name: str
    score: float
    emoji: str
    message: str
    details: List[str] = field(default_factory=list)


@dataclass
class ExperimentEvaluation:
    """Complete experiment evaluation.

    Attributes
    ----------
    sample_name : str
        Identifier for the evaluated sample.
    criteria : list[CriterionResult]
        Per-criterion scores.
    overall_score : float
        Weighted average, [0, 10].
    verdict : str
        One of ``"Publicável ✅"``, ``"Revisão necessária ⚠️"``,
        ``"Refazer ensaio ❌"``.
    human_errors : list[str]
        Possible operator / setup errors detected.
    recommendations : list[str]
        Actionable next steps.
    article_fragment : str
        Ready-to-use sentence fragment for Methods / Results sections.
    """

    sample_name: str
    criteria: List[CriterionResult]
    overall_score: float
    verdict: str
    human_errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    article_fragment: str = ""


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _traffic_light(score: float) -> str:
    """Map a 0–10 score to a traffic-light emoji."""
    if score >= 7.5:
        return "🟢"
    if score >= 4.0:
        return "🟡"
    return "🔴"


def _detect_powerline(
    freq: np.ndarray,
    z: np.ndarray,
    *,
    targets_hz: tuple = (50.0, 60.0, 100.0, 120.0),
    window_hz: float = 3.0,
    spike_factor: float = 3.0,
) -> List[float]:
    """Return list of powerline frequencies where a spike is detected.

    Uses the magnitude of the impedance derivative w.r.t. log-frequency as
    a proxy for artifacts.  A spike is flagged when the local |dZ/d(log f)|
    exceeds *spike_factor* × median in a ±window_hz band.
    """
    detected: List[float] = []
    if len(freq) < 8:
        return detected

    log_f = np.log10(freq + 1e-30)
    mag = np.abs(z)

    # Smooth derivative via finite differences on log-scale
    d_mag = np.abs(np.gradient(mag, log_f))
    median_d = np.median(d_mag)
    if median_d < 1e-30:
        return detected

    for target in targets_hz:
        mask = np.abs(freq - target) <= window_hz
        if mask.sum() < 1:
            continue
        local_d = d_mag[mask].max()
        if local_d > spike_factor * median_d:
            detected.append(target)

    return detected


def _count_zombie_params(fit_result: Dict[str, Any]) -> List[str]:
    """Return list of parameter names whose 95% CI includes zero."""
    zombies: List[str] = []
    params = fit_result.get("params") or {}
    params_std = fit_result.get("params_std") or {}
    for pname, val in params.items():
        std = params_std.get(pname, None)
        if std is None or not np.isfinite(std):
            continue
        ci95 = 1.96 * std
        if abs(val) < ci95:
            zombies.append(pname)
    return zombies


# ─────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────


class ExperimentEvaluator:
    """Synthesise all EIS quality signals into a single expert verdict.

    Parameters
    ----------
    freq_min_ok : float
        Minimum low-frequency limit considered adequate (Hz).
        Default 0.1 Hz; extend to 0.01 Hz for diffusion studies.
    freq_max_ok : float
        Minimum high-frequency limit considered adequate (Hz).
        Default 10 000 Hz.
    pts_per_decade_ok : float
        Minimum point density considered adequate.
        Default 7 pts / decade.
    """

    def __init__(
        self,
        *,
        freq_min_ok: float = 0.1,
        freq_max_ok: float = 10_000.0,
        pts_per_decade_ok: float = 7.0,
    ) -> None:
        self.freq_min_ok = freq_min_ok
        self.freq_max_ok = freq_max_ok
        self.pts_per_decade_ok = pts_per_decade_ok

    # ── public ────────────────────────────────────────────────────

    def evaluate(
        self,
        sample_name: str,
        freq: np.ndarray,
        z: np.ndarray,
        *,
        fit_result: Optional[Dict[str, Any]] = None,
        kk_result: Optional[Any] = None,
    ) -> ExperimentEvaluation:
        """Run the full evaluation.

        Parameters
        ----------
        sample_name : str
            Label for the sample (used in reports and article fragment).
        freq : np.ndarray
            Frequency vector in Hz.
        z : np.ndarray
            Complex impedance array.
        fit_result : dict | None
            Output from ``fit_template()`` or ``run_shortlist_fit()``.
        kk_result : KKResult | None
            Output from ``KramersKronigValidator.validate()``.
            If ``None``, the KK test is run internally.

        Returns
        -------
        ExperimentEvaluation
        """
        freq = np.asarray(freq, dtype=float)
        z = np.asarray(z, dtype=complex)

        criteria: List[CriterionResult] = []

        # 1. Kramers-Kronig validity
        criteria.append(self._check_kk(freq, z, kk_result=kk_result))

        # 2. Frequency range
        criteria.append(self._check_freq_range(freq))

        # 3. Point density
        criteria.append(self._check_point_density(freq))

        # 4. Zreal positivity (electrode inversion check)
        criteria.append(self._check_zreal_positive(freq, z))

        # 5. Powerline noise
        criteria.append(self._check_powerline(freq, z))

        # 6. Fitting convergence
        criteria.append(self._check_fitting_converged(fit_result))

        # 7. Fitting quality (RSS / structured residuals)
        criteria.append(self._check_fitting_quality(fit_result))

        # 8. Chi²/ν
        criteria.append(self._check_chi2_nu(fit_result))

        # 9. Zombie parameters
        criteria.append(self._check_zombie_params(fit_result))

        # 10. Physical plausibility (CPE n, negative resistance)
        criteria.append(self._check_physical_plausibility(fit_result))

        # ── Compute overall score (weighted average) ──
        weights = [1.5, 0.8, 0.6, 1.0, 0.8, 0.7, 1.2, 1.2, 1.0, 0.7]
        total_w = sum(weights)
        overall = sum(c.score * w for c, w in zip(criteria, weights)) / total_w

        # ── Verdict ──
        if overall >= 7.5:
            verdict = "Publicável ✅"
        elif overall >= 5.0:
            verdict = "Revisão necessária ⚠️"
        else:
            verdict = "Refazer ensaio ❌"

        # ── Human errors ──
        human_errors = self._collect_human_errors(criteria)

        # ── Global recommendations ──
        recommendations = self._collect_recommendations(criteria, overall)

        # ── Article fragment ──
        article_fragment = self._build_article_fragment(
            sample_name, freq, fit_result, kk_result, criteria
        )

        return ExperimentEvaluation(
            sample_name=sample_name,
            criteria=criteria,
            overall_score=round(overall, 2),
            verdict=verdict,
            human_errors=human_errors,
            recommendations=recommendations,
            article_fragment=article_fragment,
        )

    # ── criteria ──────────────────────────────────────────────────

    def _check_kk(
        self,
        freq: np.ndarray,
        z: np.ndarray,
        *,
        kk_result: Optional[Any] = None,
    ) -> CriterionResult:
        try:
            if kk_result is None:
                from src.kramers_kronig import KramersKronigValidator

                kk_result = KramersKronigValidator().validate(freq, z)

            cls = kk_result.classification
            mean_re = kk_result.mean_residual_real * 100
            mean_im = kk_result.mean_residual_imag * 100

            if cls == "excelente":
                score = 10.0
                msg = f"excelente — ΔRe={mean_re:.2f}%, ΔIm={mean_im:.2f}%"
                details = [
                    "Dados KK-compatíveis; sem evidência de não-estacionaridade."
                ]
            elif cls == "aceitável":
                score = 6.0
                msg = f"aceitável — ΔRe={mean_re:.2f}%, ΔIm={mean_im:.2f}%"
                details = [
                    "Desvios menores detectados.",
                    "Verificar estabilização do OCP antes da medição.",
                    "Amplitude de excitação ≤ 10 mV recomendada.",
                ]
            else:
                score = 1.0
                msg = f"FALHOU — ΔRe={mean_re:.2f}%, ΔIm={mean_im:.2f}%"
                details = [
                    "Sistema pode não estar em estado estacionário.",
                    "Possível não-linearidade ou ruído excessivo.",
                    "Repetir medição após estabilização do OCP (≥ 30 min).",
                ]
        except Exception as exc:
            logger.debug("KK check failed: %s", exc)
            score = 5.0
            msg = "KK não executado"
            details = [f"Erro interno: {exc}"]

        return CriterionResult(
            name="Validação Kramers-Kronig",
            score=score,
            emoji=_traffic_light(score),
            message=msg,
            details=details,
        )

    def _check_freq_range(self, freq: np.ndarray) -> CriterionResult:
        fmin = float(freq.min())
        fmax = float(freq.max())
        decades = np.log10(fmax / max(fmin, 1e-30))
        details: List[str] = [
            f"Faixa: {fmin:.3g} – {fmax:.3g} Hz ({decades:.1f} décadas)"
        ]

        score = 10.0
        msg_parts: List[str] = []

        if fmin > self.freq_min_ok:
            penalty = min(4.0, (np.log10(fmin / self.freq_min_ok)) * 2.0)
            score -= penalty
            msg_parts.append(f"f_min={fmin:.3g} Hz alto")
            details.append(
                f"f_min={fmin:.3g} Hz > {self.freq_min_ok} Hz recomendado. "
                "Processos lentos (difusão, adsorção) podem estar incompletos."
            )

        if fmax < self.freq_max_ok:
            penalty = min(3.0, (np.log10(self.freq_max_ok / fmax)) * 1.5)
            score -= penalty
            msg_parts.append(f"f_max={fmax:.3g} Hz baixo")
            details.append(
                f"f_max={fmax:.3g} Hz < {self.freq_max_ok:.0f} Hz recomendado. "
                "Rs pode estar sub-estimado."
            )

        score = max(0.0, score)
        msg = (
            ", ".join(msg_parts) if msg_parts else f"adequada: {fmin:.3g}–{fmax:.3g} Hz"
        )
        return CriterionResult(
            name="Cobertura de frequência",
            score=score,
            emoji=_traffic_light(score),
            message=msg,
            details=details,
        )

    def _check_point_density(self, freq: np.ndarray) -> CriterionResult:
        fmin = float(freq.min())
        fmax = float(freq.max())
        decades = np.log10(fmax / max(fmin, 1e-30))
        n_pts = len(freq)
        density = n_pts / max(decades, 0.1)
        details = [
            f"{n_pts} pontos em {decades:.1f} décadas → {density:.1f} pts/década"
        ]

        if density >= self.pts_per_decade_ok:
            score = 10.0
            msg = f"{density:.1f} pts/décade — adequado"
        elif density >= self.pts_per_decade_ok * 0.5:
            score = 6.0
            msg = f"{density:.1f} pts/décade — baixo"
            details.append(
                f"Recomendado ≥ {self.pts_per_decade_ok:.0f} pts/décade para "
                "melhor definição de picos DRT."
            )
        else:
            score = 2.0
            msg = f"{density:.1f} pts/décade — insuficiente"
            details.append(
                "Espectro esparso; fitting pode ter múltiplos mínimos locais."
            )

        return CriterionResult(
            name="Densidade de pontos",
            score=score,
            emoji=_traffic_light(score),
            message=msg,
            details=details,
        )

    def _check_zreal_positive(self, freq: np.ndarray, z: np.ndarray) -> CriterionResult:
        """Check for negative Z' at high frequencies — indicates electrode inversion."""
        high_freq_mask = freq >= 0.1 * float(freq.max())
        zr_high = z.real[high_freq_mask]
        n_negative = int((zr_high < 0).sum())
        details: List[str] = []

        if n_negative == 0:
            score = 10.0
            msg = "Z' > 0 — sem inversão de eletrodo"
        elif n_negative <= 2:
            score = 6.5
            msg = f"{n_negative} ponto(s) com Z' < 0 em alta frequência"
            details.append(
                "Pequenos valores negativos de Z' podem ser artefatos de "
                "indutância de cabo. Verificar conexões e blindagem."
            )
        else:
            score = 1.0
            msg = f"{n_negative} pontos com Z' < 0 — possível inversão de eletrodo!"
            details.append("⚠️ Múltiplos pontos com Z' < 0. Possíveis causas:")
            details.append("  • Eletrodos trocados (WE ↔ CE ou WE ↔ RE)")
            details.append("  • Comprimento excessivo de cabo sem blindagem")
            details.append("  • Eletrodo danificado ou com mau contato")

        return CriterionResult(
            name="Positividade de Z' (inversão de eletrodo)",
            score=score,
            emoji=_traffic_light(score),
            message=msg,
            details=details,
        )

    def _check_powerline(self, freq: np.ndarray, z: np.ndarray) -> CriterionResult:
        detected = _detect_powerline(freq, z)
        if not detected:
            return CriterionResult(
                name="Ruído de rede (50/60/100 Hz)",
                score=10.0,
                emoji="🟢",
                message="não detectado",
                details=["Nenhum artefato de rede elétrica identificado."],
            )
        labels = ", ".join(f"{f:.0f} Hz" for f in detected)
        score = max(0.0, 10.0 - 3.0 * len(detected))
        details = [
            f"Interferência detectada em: {labels}",
            "Ações corretivas:",
            "  • Adicionar blindagem (gaiola de Faraday) na célula",
            "  • Usar cabo coaxial blindado para todos os eletrodos",
            "  • Afastar a célula de fontes de CA (transformadores, inversores)",
            "  • Ligar a blindagem ao terra do instrumento",
        ]
        return CriterionResult(
            name="Ruído de rede (50/60/100 Hz)",
            score=score,
            emoji=_traffic_light(score),
            message=f"detectado em {labels}",
            details=details,
        )

    def _check_fitting_converged(
        self, fit_result: Optional[Dict[str, Any]]
    ) -> CriterionResult:
        if fit_result is None:
            return CriterionResult(
                name="Convergência do fitting",
                score=5.0,
                emoji="⚪",
                message="fitting não fornecido",
                details=["Execute o pipeline EIS para obter resultado de fitting."],
            )
        success = fit_result.get("success", False)
        bound_hits = fit_result.get("bound_hits", 0)
        details: List[str] = []

        if success and bound_hits == 0:
            score, msg = 10.0, "convergiu sem bound hits"
        elif success and bound_hits <= 2:
            score = 7.0
            msg = f"convergiu ({bound_hits} parâmetro(s) no limite)"
            details.append(
                "Parâmetro(s) atingiram limite da busca. "
                "Revisar bounds ou tentar circuito mais simples."
            )
        elif success:
            score = 4.0
            msg = f"convergência parcial ({bound_hits} bound hits)"
            details.append("Vários parâmetros nos limites — fitting pode ser instável.")
            details.append("Tentar circuito com menos elementos ou ajustar bounds.")
        else:
            score = 1.0
            msg = "NÃO convergiu"
            details.append("Otimizador não encontrou solução satisfatória.")
            details.append(
                "Aumentar max_nfev, revisar p0 inicial, ou escolher circuito diferente."
            )

        return CriterionResult(
            name="Convergência do fitting",
            score=score,
            emoji=_traffic_light(score),
            message=msg,
            details=details,
        )

    def _check_fitting_quality(
        self, fit_result: Optional[Dict[str, Any]]
    ) -> CriterionResult:
        if fit_result is None:
            return CriterionResult(
                name="Qualidade do fitting (RSS)",
                score=5.0,
                emoji="⚪",
                message="fitting não fornecido",
                details=[],
            )
        try:
            from src.fitting_diagnostics import assess_quality

            qi = assess_quality(fit_result)
            if qi.level == "green":
                score, msg = 10.0, f"excelente — {qi.label}"
            elif qi.level == "yellow":
                score, msg = 6.0, f"aceitável — {qi.label}"
            else:
                score, msg = 2.0, f"problemático — {qi.label}"
            details = qi.reasons
        except Exception as exc:
            score, msg = 5.0, f"avaliação não disponível ({exc})"
            details = []

        return CriterionResult(
            name="Qualidade do fitting (RSS)",
            score=score,
            emoji=_traffic_light(score),
            message=msg,
            details=details,
        )

    def _check_chi2_nu(self, fit_result: Optional[Dict[str, Any]]) -> CriterionResult:
        if fit_result is None:
            return CriterionResult(
                name="χ²/ν (qui-quadrado reduzido)",
                score=5.0,
                emoji="⚪",
                message="fitting não fornecido",
                details=[],
            )
        chi2 = fit_result.get("chi2_over_nu", None)
        if chi2 is None:
            # Estimate from RSS if possible
            rss = fit_result.get("rss", None)
            n_pts = fit_result.get("n_points", 1)
            n_par = fit_result.get("n_params", 1)
            nu = max(n_pts - n_par, 1)
            chi2 = rss / nu if rss is not None else None

        if chi2 is None or not np.isfinite(float(chi2)):
            return CriterionResult(
                name="χ²/ν (qui-quadrado reduzido)",
                score=5.0,
                emoji="⚪",
                message="não disponível",
                details=[
                    "χ²/ν não calculado — fornecimento de chi2_over_nu recomendado."
                ],
            )

        chi2 = float(chi2)
        if chi2 <= 2.0:
            score = 10.0
            msg = f"χ²/ν = {chi2:.2f} — excelente (≤ 2)"
            details = ["Fitting excelente pelo critério metrológico Orazem 2026."]
        elif chi2 <= 5.0:
            score = 7.0
            msg = f"χ²/ν = {chi2:.2f} — aceitável (2–5)"
            details = [
                "Fitting aceitável. Considere circuito mais rico se houver "
                "estrutura sistemática nos resíduos."
            ]
        elif chi2 <= 10.0:
            score = 4.0
            msg = f"χ²/ν = {chi2:.2f} — alto (5–10)"
            details = [
                "Fitting com χ²/ν alto. Possível modelo inadequado.",
                "Testar circuitos com mais elementos (ex.: Porous-TLM).",
            ]
        else:
            score = 1.0
            msg = f"χ²/ν = {chi2:.2f} — inaceitável (> 10)"
            details = [
                "χ²/ν > 10: modelo claramente inadequado para os dados.",
                "Rever circuito, verificar qualidade dos dados (KK), ou "
                "refazer a medição.",
            ]

        return CriterionResult(
            name="χ²/ν (qui-quadrado reduzido)",
            score=score,
            emoji=_traffic_light(score),
            message=msg,
            details=details,
        )

    def _check_zombie_params(
        self, fit_result: Optional[Dict[str, Any]]
    ) -> CriterionResult:
        if fit_result is None:
            return CriterionResult(
                name="Parâmetros zumbi",
                score=5.0,
                emoji="⚪",
                message="fitting não fornecido",
                details=[],
            )
        zombies = _count_zombie_params(fit_result)
        if not zombies:
            return CriterionResult(
                name="Parâmetros zumbi",
                score=10.0,
                emoji="🟢",
                message="nenhum parâmetro zumbi",
                details=["Todos os parâmetros são estatisticamente significativos."],
            )
        score = max(0.0, 10.0 - 3.0 * len(zombies))
        details = [
            f"Parâmetro(s) com IC 95% contendo zero: {', '.join(zombies)}",
            "Parâmetros zumbi não contribuem significativamente para o modelo.",
            "Considere remover esses elementos e refazer o fitting.",
        ]
        return CriterionResult(
            name="Parâmetros zumbi",
            score=score,
            emoji=_traffic_light(score),
            message=f"{len(zombies)} parâmetro(s) zumbi: {', '.join(zombies)}",
            details=details,
        )

    def _check_physical_plausibility(
        self, fit_result: Optional[Dict[str, Any]]
    ) -> CriterionResult:
        if fit_result is None:
            return CriterionResult(
                name="Plausibilidade física",
                score=5.0,
                emoji="⚪",
                message="fitting não fornecido",
                details=[],
            )
        params = fit_result.get("params") or {}
        issues: List[str] = []

        # CPE exponent check
        n_val = params.get("n", params.get("n1", params.get("n2", None)))
        if n_val is not None:
            if float(n_val) < 0.3:
                issues.append(
                    f"n = {float(n_val):.3f} < 0.3 — fisicamente improvável para CPE; "
                    "sugere modelo errado ou dados ruidosos"
                )
            elif float(n_val) < 0.5:
                issues.append(
                    f"n = {float(n_val):.3f} — valor baixo para CPE; "
                    "verificar se o elemento é realmente CPE"
                )

        # Negative resistance checks
        for pname, val in params.items():
            if pname.startswith("R") and float(val) < 0:
                issues.append(
                    f"{pname} = {float(val):.3g} Ω < 0 — resistência negativa "
                    "é fisicamente inválida"
                )

        if not issues:
            return CriterionResult(
                name="Plausibilidade física",
                score=10.0,
                emoji="🟢",
                message="todos os parâmetros dentro dos limites físicos",
                details=["Nenhuma irregularidade física detectada nos parâmetros."],
            )

        score = max(1.0, 10.0 - 4.0 * len(issues))
        return CriterionResult(
            name="Plausibilidade física",
            score=score,
            emoji=_traffic_light(score),
            message=f"{len(issues)} irregularidade(s) detectada(s)",
            details=issues,
        )

    # ── synthesis helpers ─────────────────────────────────────────

    def _collect_human_errors(self, criteria: List[CriterionResult]) -> List[str]:
        errors: List[str] = []
        for c in criteria:
            if c.score < 4.0:
                for d in c.details:
                    if any(
                        kw in d
                        for kw in (
                            "inversão",
                            "Eletrodos",
                            "Z' < 0",
                            "FALHOU",
                            "negativos",
                            "resistência negativa",
                        )
                    ):
                        errors.append(f"[{c.name}] {d}")
        return errors

    def _collect_recommendations(
        self, criteria: List[CriterionResult], overall: float
    ) -> List[str]:
        recs: List[str] = []

        # Collect from low-scoring criteria
        for c in sorted(criteria, key=lambda x: x.score):
            if c.score < 7.0 and c.details:
                # Add top detail as recommendation
                recs.append(f"[{c.name}] {c.details[0]}")

        if overall >= 8.0:
            recs.append("Espectro de alta qualidade — pronto para publicação.")
        elif overall >= 6.5:
            recs.append(
                "Espectro utilizável — corrigir os pontos indicados antes de publicar."
            )
        else:
            recs.append(
                "Qualidade insuficiente para publicação — reavaliar condições "
                "experimentais."
            )

        return recs

    def _build_article_fragment(
        self,
        sample_name: str,
        freq: np.ndarray,
        fit_result: Optional[Dict[str, Any]],
        kk_result: Optional[Any],
        criteria: List[CriterionResult],
    ) -> str:
        kk_part = ""
        kk_crit = next((c for c in criteria if "Kramers" in c.name), None)
        if kk_crit and kk_crit.score >= 7.5:
            kk_part = (
                " The impedance data passed the Kramers-Kronig consistency "
                "test (linear KK, Boukamp 1995), confirming data linearity "
                "and stationarity."
            )
        elif kk_crit and kk_crit.score >= 4.0:
            kk_part = (
                " The data showed minor deviations from Kramers-Kronig "
                "compliance; data were used with caution."
            )

        freq_part = (
            f"EIS measurements were performed over a frequency range of "
            f"{freq.min():.3g}–{freq.max():.3g} Hz ({len(freq)} points)."
        )

        fit_part = ""
        if fit_result:
            circuit = fit_result.get("template", "equivalent circuit")
            chi2_nu = fit_result.get("chi2_over_nu", None)
            chi2_str = f" (χ²/ν = {chi2_nu:.2f})" if chi2_nu is not None else ""
            fit_part = (
                f" Data were fitted to the {circuit} model using a "
                f"weighted nonlinear least-squares algorithm{chi2_str}."
            )

        return f"{freq_part}{kk_part}{fit_part}"


# ─────────────────────────────────────────────────────────────────────
# Report formatter
# ─────────────────────────────────────────────────────────────────────

_SEP = "═" * 68
_LINE = "─" * 68


def format_report(evaluation: ExperimentEvaluation) -> str:
    """Render an ExperimentEvaluation as a formatted text report.

    Parameters
    ----------
    evaluation : ExperimentEvaluation
        Output from ``ExperimentEvaluator.evaluate()``.

    Returns
    -------
    str
        Multi-line report suitable for console or GUI textbox display.
    """
    lines: List[str] = [
        _SEP,
        "  🎓  AVALIAÇÃO CRÍTICA DO EXPERIMENTO",
        "  Prof. IonFlow  |  Rigor Orazem & Tribollet 2026",
        _SEP,
        f"  Amostra: {evaluation.sample_name}",
        _LINE,
        "",
        f"  {'CRITÉRIO':<42} {'NOTA':>5}  STATUS",
        f"  {'-'*42} {'-----':>5}  ------",
    ]

    for c in evaluation.criteria:
        note_str = f"{c.score:.1f}/10"
        lines.append(f"  {c.name:<42} {note_str:>6}  {c.emoji} {c.message}")

    lines += [
        "",
        _LINE,
        f"  NOTA GERAL: {evaluation.overall_score:.1f}/10  —  {evaluation.verdict}",
        _LINE,
    ]

    # Human errors
    if evaluation.human_errors:
        lines += ["", "  ⚠️  ERROS DE OPERAÇÃO DETECTADOS"]
        for e in evaluation.human_errors:
            lines.append(f"  • {e}")

    # Recommendations
    if evaluation.recommendations:
        lines += ["", "  📋  RECOMENDAÇÕES"]
        for r in evaluation.recommendations:
            lines.append(f"  → {r}")

    # Article fragment
    if evaluation.article_fragment:
        lines += [
            "",
            "  📄  FRASE PRONTA PARA ARTIGO  (seção Métodos/Resultados)",
            _LINE,
            "",
        ]
        # Word-wrap at ~66 chars
        words = evaluation.article_fragment.split()
        current = ""
        for w in words:
            if len(current) + len(w) + 1 > 66:
                lines.append("  " + current.strip())
                current = w + " "
            else:
                current += w + " "
        if current.strip():
            lines.append("  " + current.strip())

    # Details for low-scoring criteria
    low = [c for c in evaluation.criteria if c.score < 7.5 and c.details]
    if low:
        lines += ["", "  🔍  DETALHES DOS CRITÉRIOS COM FALHA / ATENÇÃO"]
        for c in low:
            lines.append(f"\n  [{c.emoji} {c.name}]")
            for d in c.details:
                lines.append(f"    • {d}")

    lines.append("")
    lines.append(_SEP)
    return "\n".join(lines)
