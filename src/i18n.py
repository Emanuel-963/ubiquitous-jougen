"""Internationalisation (i18n) support for IonFlow Pipeline.

Usage
-----
    from src.i18n import tr, set_language, get_language, LANGUAGES

    set_language("en")          # switch to English
    label = tr("Pipelines")    # returns "Pipelines"

    set_language("pt")          # switch to Portuguese
    label = tr("Pipelines")    # returns "Pipelines" (same, it's intl.)

Design decisions:
    * Keys are the *Portuguese* strings used in the original codebase, so
      existing code only needs ``tr(original_string)`` wrappers.
    * When the active language is ``"pt"`` the key is returned as-is (zero-cost).
    * A missing key always returns the key itself — the app never crashes
      because of a missing translation.
"""
from __future__ import annotations

import threading
from typing import Dict

# ── Public API ───────────────────────────────────────────────────────
LANGUAGES = ("pt", "en")

_lock = threading.Lock()
_current: str = "pt"


def set_language(lang: str) -> None:
    """Set the active language.  Must be one of ``LANGUAGES``."""
    global _current
    code = lang.lower().strip()[:2]
    if code not in LANGUAGES:
        code = "pt"
    with _lock:
        _current = code


def get_language() -> str:
    """Return the current language code."""
    return _current


def tr(key: str) -> str:
    """Translate *key* (Portuguese original) to the current language."""
    if _current == "pt":
        return key
    table = _TABLES.get(_current)
    if table is None:
        return key
    return table.get(key, key)


# ── English translations ─────────────────────────────────────────────
# Grouped by UI region for maintainability.

_EN: Dict[str, str] = {
    # -- Sidebar buttons --------------------------------------------------
    "Pipelines": "Pipelines",
    "Scan rate (A/g)": "Scan rate (A/g)",
    "Importar EIS para raw": "Import EIS to raw",
    "Importar Ciclagem para processed": "Import Cycling to processed",
    "Gráficos Interativos": "Interactive Charts",
    "Rodar Pipeline EIS": "Run EIS Pipeline",
    "Rodar Pipeline Ciclagem": "Run Cycling Pipeline",
    "Rodar Ambos": "Run Both",
    "Rodar Pipeline DRT": "Run DRT Pipeline",
    "Aplicar preset": "Apply preset",
    "Reset DRT": "Reset DRT",
    "Preset DRT": "DRT Preset",
    "λ DRT": "λ DRT",
    "n_taus": "n_taus",
    "Rápido:30 | Balanceado:50 | Alta:80": "Fast:30 | Balanced:50 | High:80",

    # -- Appearance / Theme -----------------------------------------------
    "Tema": "Theme",
    "Claro": "Light",
    "Escuro": "Dark",
    "Sistema": "System",
    "Idioma": "Language",

    # -- DRT presets -------------------------------------------------------
    "Rápido": "Fast",
    "Balanceado": "Balanced",
    "Alta resolução": "High resolution",

    # -- DRT modes ---------------------------------------------------------
    "Espectro": "Spectrum",

    # -- Main tabs ---------------------------------------------------------
    "Gráficos": "Charts",
    "Tabelas": "Tables",
    "Logs": "Logs",

    # -- Table sub-tabs ----------------------------------------------------
    "EIS": "EIS",
    "Ciclagem": "Cycling",
    "Circuitos": "Circuits",
    "DRT": "DRT",
    "DRT Peaks": "DRT Peaks",
    "DRT + EIS": "DRT + EIS",

    # -- Interactive window tabs -------------------------------------------
    "Rank vs Retenção": "Rank vs Retention",
    "Nyquist": "Nyquist",
    "Bode": "Bode",
    "PCA 2D": "PCA 2D",
    "PCA Retenção": "PCA Retention",
    "Correlação": "Correlation",
    "Séries": "Series",
    "Energia × Potência": "Energy × Power",
    "Ragone": "Ragone",
    "Energia vs Ciclo": "Energy vs Cycle",
    "Retenção vs Ciclo": "Retention vs Cycle",
    "Heatmap |Z|": "Heatmap |Z|",
    "Box-plot": "Box-plot",
    "Radar": "Radar",
    "DRT × EIS": "DRT × EIS",

    # -- Common labels -----------------------------------------------------
    "Filtro:": "Filter:",
    "Amostra:": "Sample:",
    "Arquivo:": "File:",
    "Destaque:": "Highlight:",
    "Métrica:": "Metric:",
    "Coluna:": "Column:",
    "Série:": "Series:",
    "Modo:": "Mode:",
    "Overlay (vírgula):": "Overlay (comma):",
    "Amostras (separe por ';'):": "Samples (separate with ';'):",
    "Linhas: 0/0": "Rows: 0/0",
    "— nenhum —": "— none —",

    # -- Buttons -----------------------------------------------------------
    "Salvar imagem": "Save image",
    "Salvar filtrado": "Save filtered",
    "Salvar tudo": "Save all",
    "Salvar gráfico": "Save chart",
    "Aplicar overlay": "Apply overlay",
    "Atualizar": "Update",
    "Salvar visual": "Save visual",
    "Ver interativo": "View interactive",
    "Top 5": "Top 5",
    "Todos": "All",

    # -- Placeholders ------------------------------------------------------
    "Buscar em todas as colunas": "Search all columns",

    # -- Status messages ---------------------------------------------------
    "Status: pronto": "Status: ready",
    "Pronto": "Ready",
    "rodando EIS": "running EIS",
    "rodando Ciclagem": "running Cycling",
    "rodando ambos": "running both",
    "rodando DRT": "running DRT",
    "erro no EIS": "EIS error",
    "erro na Ciclagem": "Cycling error",
    "erro ao rodar ambos": "error running both",
    "erro no DRT": "DRT error",
    "EIS concluído": "EIS completed",
    "Ciclagem concluída": "Cycling completed",
    "Ambos concluídos": "Both completed",
    "DRT concluído": "DRT completed",

    # -- Progress messages -------------------------------------------------
    "Identificando amostras EIS...": "Identifying EIS samples…",
    "Calculando valores EIS...": "Calculating EIS values…",
    "Gerando tabelas e gráficos...": "Generating tables and charts…",
    "Identificando ciclos...": "Identifying cycles…",
    "Calculando valores de energia...": "Calculating energy values…",
    "Gerando gráficos e tabelas...": "Generating charts and tables…",
    "Identificando dados EIS e ciclos...": "Identifying EIS data and cycles…",
    "Calculando EIS...": "Calculating EIS…",
    "Calculando ciclagem...": "Calculating cycling…",
    "Calculando DRT...": "Calculating DRT…",
    "Executando inversão DRT...": "Running DRT inversion…",
    "Organizando resultados DRT...": "Organising DRT results…",
    "Erro": "Error",

    # -- Empty-state labels ------------------------------------------------
    "Nenhum dado disponível.": "No data available.",
    "Sem dados para Rank vs Retenção": "No data for Rank vs Retention",
    "Sem dados EIS disponíveis": "No EIS data available",
    "Sem dados de PCA": "No PCA data",
    "Sem dados de PCA/Retenção": "No PCA/Retention data",
    "Sem dados suficientes para correlação": "Insufficient data for correlation",
    "Sem dados combinados DRT+EIS": "No combined DRT+EIS data",
    "Sem dados de ciclagem disponíveis": "No cycling data available",
    "Sem dados de ciclos para esta amostra": "No cycle data for this sample",
    "Sem dados para Ragone": "No data for Ragone",
    "Sem dados de ciclagem para Ragone": "No cycling data for Ragone",
    "Sem métricas de ciclagem": "No cycling metrics",
    "Sem dados EIS para heatmap": "No EIS data for heatmap",
    "Não foi possível gerar heatmap": "Could not generate heatmap",
    "Sem métricas numéricas para box-plot": "No numeric metrics for box-plot",
    "Sem dados EIS para box-plot": "No EIS data for box-plot",
    "Precisa de ≥2 amostras para radar": "Need ≥2 samples for radar",
    "Sem dados EIS para radar": "No EIS data for radar",
    "Selecione ao menos 2 amostras com ≥3 métricas numéricas":
        "Select at least 2 samples with ≥3 numeric metrics",
    "Sem dados suficientes para retenção": "Insufficient data for retention",
    "Sem colunas de série disponíveis": "No series columns available",
    "Sem séries identificadas": "No series identified",
    "Sem dados de séries disponíveis": "No series data available",
    "Sem resultados DRT disponíveis": "No DRT results available",
    "Sem dados DRT disponíveis": "No DRT data available",
    "Sem dados disponíveis": "No data available",
    "Dados insuficientes\npara radar": "Insufficient data\nfor radar",
    "Todas": "All",

    # -- Log messages (selection of most common) ---------------------------
    "Interativo não disponível para este gráfico.":
        "Interactive not available for this chart.",
    "DRT resetado para preset padrão (Balanceado).":
        "DRT reset to default preset (Balanced).",
    "Tabela não encontrada.":
        "Table not found.",
    "Nenhum dado para salvar.":
        "No data to save.",
    "Sem dados de rank vs retenção para exibir.":
        "No rank vs retention data to display.",
    "Dados de Rank/Retenção ausentes.":
        "Rank/Retention data missing.",
    "mplcursors não encontrado; hover desabilitado.":
        "mplcursors not found; hover disabled.",
    "Sem dados de PCA para exibir.":
        "No PCA data to display.",
    "Sem dados para correlação.":
        "No data for correlation.",
    "Dados insuficientes para correlação.":
        "Insufficient data for correlation.",
    "Sem dados de séries para exibir.":
        "No series data to display.",
    "Formato de série não reconhecido.":
        "Series format not recognised.",
    "Sem pontos para esta série.":
        "No data points for this series.",
    "Sem visual DRT para salvar.":
        "No DRT visual to save.",
    "Nenhum gráfico EP para salvar.":
        "No EP chart to save.",
    "Subclasse": "Subclass",
    "Amostras": "Samples",
}

# ── Table of all languages → dict ────────────────────────────────────
_TABLES: Dict[str, Dict[str, str]] = {
    "en": _EN,
}
