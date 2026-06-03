"""Automatic executive summary and next-step suggestions.

AI-01: Generates a short executive summary (3–4 lines) immediately after
pipeline execution, without requiring the user to click "Análise IA".

AI-02: Provides automatic next-step suggestions based on the results
(e.g. "Your data shows a depressed arc with n=0.72 — consider polishing
the electrode or re-running with a freshly prepared surface").

These functions are lightweight rule-based heuristics that run instantly
(no LLM call required) and integrate with the GUI log panel.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# AI-01: Auto Executive Summary
# ═══════════════════════════════════════════════════════════════════════


def generate_auto_summary(
    eis_result: Optional[Dict[str, Any]] = None,
    cycling_result: Optional[Dict[str, Any]] = None,
    drt_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a short executive summary from pipeline results.

    This is called automatically after each pipeline run to provide
    immediate feedback in the log panel. It does NOT require an LLM.

    Parameters
    ----------
    eis_result : dict or None
        EIS pipeline results (contains 'results_df', 'best_circuit', etc.).
    cycling_result : dict or None
        Cycling pipeline results.
    drt_result : dict or None
        DRT pipeline results.

    Returns
    -------
    str
        A 3–5 line summary in Portuguese for the researcher.
    """
    lines: List[str] = []
    lines.append("📊 Resumo automático:")

    if eis_result:
        lines.extend(_summarise_eis(eis_result))

    if cycling_result:
        lines.extend(_summarise_cycling(cycling_result))

    if drt_result:
        lines.extend(_summarise_drt(drt_result))

    if len(lines) == 1:
        lines.append("  • Nenhum resultado disponível para resumir.")

    return "\n".join(lines)


def _summarise_eis(result: Dict[str, Any]) -> List[str]:
    """Extract key EIS findings for the summary."""
    lines: List[str] = []
    try:
        df = result.get("results_df")
        if df is not None and len(df) > 0:
            n_samples = len(df)
            best = result.get("best_circuit", "N/A")

            # Key metrics
            rs_mean = df["Rs_fit"].mean() if "Rs_fit" in df.columns else None
            rp_mean = df["Rp_fit"].mean() if "Rp_fit" in df.columns else None
            chi2 = df["chi2_over_nu"].mean() if "chi2_over_nu" in df.columns else None

            summary_parts = [f"  • EIS: {n_samples} amostra(s) analisada(s)"]
            if best != "N/A":
                summary_parts[0] += f", melhor circuito: {best}"

            lines.append(summary_parts[0])

            metrics_line = "  •"
            if rs_mean is not None:
                metrics_line += f" Rs={rs_mean:.3g}Ω"
            if rp_mean is not None:
                metrics_line += f", Rp={rp_mean:.3g}Ω"
            if chi2 is not None:
                quality = "excelente" if chi2 < 0.01 else "bom" if chi2 < 0.05 else "aceitável" if chi2 < 0.1 else "revisar"
                metrics_line += f", χ²/ν={chi2:.4f} ({quality})"
            if metrics_line != "  •":
                lines.append(metrics_line)
    except Exception:
        pass
    return lines


def _summarise_cycling(result: Dict[str, Any]) -> List[str]:
    """Extract key cycling findings for the summary."""
    lines: List[str] = []
    try:
        df = result.get("results_df")
        if df is not None and len(df) > 0:
            n_cycles = df["cycle"].max() if "cycle" in df.columns else len(df)
            energy_col = next((c for c in df.columns if "energy" in c.lower()), None)
            retention_col = next((c for c in df.columns if "retention" in c.lower()), None)

            line = f"  • Ciclagem: {n_cycles} ciclos"
            if energy_col and len(df) > 0:
                e_first = df[energy_col].iloc[0]
                e_last = df[energy_col].iloc[-1]
                line += f", energia: {e_first:.2g} → {e_last:.2g}"
            if retention_col and len(df) > 0:
                ret = df[retention_col].iloc[-1]
                line += f", retenção: {ret:.1f}%"
            lines.append(line)
    except Exception:
        pass
    return lines


def _summarise_drt(result: Dict[str, Any]) -> List[str]:
    """Extract key DRT findings for the summary."""
    lines: List[str] = []
    try:
        peaks = result.get("peaks", [])
        if peaks:
            n_peaks = len(peaks)
            tau_values = [p.get("tau_peak", 0) for p in peaks if "tau_peak" in p]
            line = f"  • DRT: {n_peaks} pico(s) detectado(s)"
            if tau_values:
                line += f", τ dominante = {tau_values[0]:.2e} s"
            lines.append(line)
    except Exception:
        pass
    return lines


# ═══════════════════════════════════════════════════════════════════════
# AI-02: Next-Step Suggestions
# ═══════════════════════════════════════════════════════════════════════

def generate_next_steps(
    eis_result: Optional[Dict[str, Any]] = None,
    cycling_result: Optional[Dict[str, Any]] = None,
    drt_result: Optional[Dict[str, Any]] = None,
    material_preset: str = "generic",
) -> List[str]:
    """Generate automatic next-step suggestions based on results.

    Parameters
    ----------
    eis_result : dict or None
        EIS pipeline results.
    cycling_result : dict or None
        Cycling pipeline results.
    drt_result : dict or None
        DRT pipeline results.
    material_preset : str
        Active material preset for context-aware suggestions.

    Returns
    -------
    List[str]
        List of actionable suggestions for the researcher.
    """
    suggestions: List[str] = []

    if eis_result:
        suggestions.extend(_eis_suggestions(eis_result, material_preset))

    if cycling_result:
        suggestions.extend(_cycling_suggestions(cycling_result))

    if drt_result:
        suggestions.extend(_drt_suggestions(drt_result))

    # General suggestions if nothing specific found
    if not suggestions:
        suggestions.append(
            "💡 Execute a Análise IA completa para recomendações detalhadas."
        )

    return suggestions


def _eis_suggestions(result: Dict[str, Any], preset: str) -> List[str]:
    """Generate EIS-specific suggestions."""
    suggestions: List[str] = []
    try:
        df = result.get("results_df")
        if df is None or len(df) == 0:
            return suggestions

        # Check for high Rs
        if "Rs_fit" in df.columns:
            rs_mean = df["Rs_fit"].mean()
            if rs_mean > 10:
                suggestions.append(
                    f"⚠️ Rs médio alto ({rs_mean:.1f}Ω). "
                    "Sugestão: verificar contato elétrico, polir eletrodo ou usar cola de prata."
                )

        # Check for low n (depressed arc)
        if "n" in df.columns:
            n_mean = df["n"].mean()
            if n_mean < 0.75:
                suggestions.append(
                    f"⚠️ Expoente CPE baixo (n={n_mean:.2f} → arco deprimido). "
                    "Sugestão: rugosidade superficial elevada — polir eletrodo ou considerar "
                    "modelo de eletrodo poroso (TLM)."
                )
            elif n_mean > 0.95:
                suggestions.append(
                    f"✅ n={n_mean:.2f} próximo de 1 — comportamento capacitivo ideal."
                )

        # Check chi2
        if "chi2_over_nu" in df.columns:
            chi2_mean = df["chi2_over_nu"].mean()
            if chi2_mean > 0.1:
                suggestions.append(
                    f"⚠️ Ajuste com χ²/ν alto ({chi2_mean:.3f}). "
                    "Sugestão: tentar circuito com mais elementos ou verificar "
                    "se há ruído/artefactos nos dados."
                )

        # Check if Warburg is significant
        if "Sigma" in df.columns:
            sigma_mean = df["Sigma"].mean()
            if sigma_mean > 0 and not np.isnan(sigma_mean):
                suggestions.append(
                    "💡 Difusão detectada (σ Warburg significativo). "
                    "Considerar expandir a faixa de baixa frequência (< 10 mHz) "
                    "para melhor resolução do processo difusivo."
                )
    except Exception:
        pass
    return suggestions


def _cycling_suggestions(result: Dict[str, Any]) -> List[str]:
    """Generate cycling-specific suggestions."""
    suggestions: List[str] = []
    try:
        df = result.get("results_df")
        if df is None or len(df) == 0:
            return suggestions

        retention_col = next((c for c in df.columns if "retention" in c.lower()), None)
        if retention_col and len(df) > 1:
            ret_last = df[retention_col].iloc[-1]
            if ret_last < 80:
                suggestions.append(
                    f"⚠️ Retenção capacitiva baixa ({ret_last:.1f}%). "
                    "Sugestão: investigar mecanismo de degradação — "
                    "executar EIS antes e depois da ciclagem para comparar."
                )
            elif ret_last > 95:
                suggestions.append(
                    f"✅ Excelente retenção ({ret_last:.1f}%) — material estável."
                )
    except Exception:
        pass
    return suggestions


def _drt_suggestions(result: Dict[str, Any]) -> List[str]:
    """Generate DRT-specific suggestions."""
    suggestions: List[str] = []
    try:
        peaks = result.get("peaks", [])
        if len(peaks) > 3:
            suggestions.append(
                f"💡 {len(peaks)} picos DRT detectados — sistema com múltiplos "
                "processos. Considerar circuito com ≥3 elementos RC/ZARC."
            )
        elif len(peaks) == 1:
            tau = peaks[0].get("tau_peak", 0)
            if tau > 1:
                suggestions.append(
                    "💡 Único pico DRT com τ > 1 s — processo lento dominante. "
                    "Possivelmente difusão ou transferência de carga lenta."
                )
    except Exception:
        pass
    return suggestions
