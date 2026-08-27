"""Reproducible report and output writers for the scientific protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Criterion, criteria_as_dict

DEFAULT_REPORT_DPI = 300


RANKING_PRESENTATION_COLUMNS = (
    ("electrolyte", "Eletrólito"),
    ("specific_capacitance", "Capacitância\nespecífica [F/g]"),
    ("capacitance_retention", "Retenção de\ncapacitância [%]"),
    ("coulombic_efficiency", "Eficiência\ncoulômbica [%]"),
    ("energy_density", "Densidade de\nenergia [Wh/kg]"),
    ("power_density", "Densidade de\npotência [W/kg]"),
    ("capacitive_contribution", "Contribuição\ncapacitiva [%]"),
    ("rs", "Rs [Ω]"),
    ("rct_rp", "Rct/Rp [Ω]"),
    ("area_drt_lenta", "Área DRT de\nprocessos lentos [Ω]"),
    ("score_available", "Score disponível"),
    ("coverage_pct", "Cobertura"),
    ("protocol_status", "Status de\nProtocolo"),
    ("rank", "Classificação"),
)


def _ranking_presentation_frame(ranking: pd.DataFrame) -> pd.DataFrame:
    """Build the user-facing table in the same order and vocabulary as Excel."""
    result = pd.DataFrame(index=ranking.index)
    for key, label in RANKING_PRESENTATION_COLUMNS:
        if key == "electrolyte" and key not in ranking:
            result[label] = ranking.index.astype(str)
        elif key in ranking:
            result[label] = ranking[key]
        else:
            result[label] = np.nan
    if "Classificação" in result:
        result["Classificação"] = (
            pd.to_numeric(result["Classificação"], errors="coerce")
            .round()
            .astype("Int64")
        )
    return result


def _format_ranking_value(value, key: str) -> str:
    if pd.isna(value):
        return "N/D"
    if key == "electrolyte" or key == "protocol_status":
        return str(value)
    if key in {"score_available", "coverage_pct"}:
        return f"{float(value):.1f}%"
    if key == "rank":
        return str(int(float(value)))
    if key in {"specific_capacitance", "rs", "rct_rp", "area_drt_lenta"}:
        return f"{float(value):.3g}"
    if key in {"energy_density", "power_density"}:
        return f"{float(value):.4g}"
    return f"{float(value):.2f}"


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_classification_figure(
    ranking: pd.DataFrame,
    output_path: str | Path,
    criteria: Iterable[Criterion],
    dpi: int = DEFAULT_REPORT_DPI,
) -> str:
    """Create a generic table figure equivalent to the supplied workbook view."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visible = _ranking_presentation_frame(ranking)
    table_values = []
    for index, row in ranking.iterrows():
        table_values.append(
            [
                _format_ranking_value(row.get(key, np.nan), key)
                for key, _label in RANKING_PRESENTATION_COLUMNS
            ]
        )

    fig_height = max(6.5, 4.7 + 0.38 * len(visible))
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.08)
    fig.suptitle(
        "Classificação automática dos eletrólitos",
        fontsize=18,
        fontweight="bold",
        color="#17233d",
    )

    if visible.empty:
        ax.text(
            0.5, 0.60, "Nenhum eletrólito com dados válidos", ha="center", fontsize=13
        )
    else:
        headers = [label for _key, label in RANKING_PRESENTATION_COLUMNS]
        table = ax.table(
            cellText=table_values,
            colLabels=headers,
            loc="upper center",
            cellLoc="center",
            colLoc="center",
            bbox=[0.01, 0.49, 0.98, 0.38],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        for (row_index, col_index), cell in table.get_celld().items():
            cell.set_edgecolor("#aab7c4")
            if row_index == 0:
                cell.set_facecolor("#1e4773")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#edf3f8")

        best = ranking.iloc[0]
        best_name = best.get("electrolyte", "N/D")
        best_score = _format_ranking_value(
            best.get("score_available", np.nan), "score_available"
        )
        best_coverage = _format_ranking_value(
            best.get("coverage_pct", np.nan), "coverage_pct"
        )
        ax.text(
            0.02,
            0.92,
            "Melhor eletrólito (score disponível)",
            fontsize=9,
            color="#315f3d",
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            0.885,
            str(best_name),
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.34,
            0.92,
            "Maior score disponível",
            fontsize=9,
            color="#315f3d",
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.34,
            0.885,
            best_score,
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.56,
            0.92,
            "Cobertura do vencedor",
            fontsize=9,
            color="#315f3d",
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.56,
            0.885,
            best_coverage,
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )

    criteria_rows = [
        [
            c.label,
            c.unit,
            "Maior é melhor" if c.direction == "max" else "Menor é melhor",
            f"{c.weight:.2f}",
            f"{c.weight:.2f}%",
        ]
        for c in criteria
    ]
    criteria_table = ax.table(
        cellText=criteria_rows,
        colLabels=["Critério", "Unidade", "Direção", "Peso", "Peso normalizado"],
        loc="lower center",
        cellLoc="left",
        colLoc="center",
        bbox=[0.01, 0.08, 0.98, 0.29],
    )
    criteria_table.auto_set_font_size(False)
    criteria_table.set_fontsize(8)
    for (row_index, _col_index), cell in criteria_table.get_celld().items():
        cell.set_edgecolor("#aab7c4")
        if row_index == 0:
            cell.set_facecolor("#1e4773")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f2f6fa")

    ax.text(
        0.01,
        0.045,
        "Score disponível usa apenas métricas presentes; N/D não é convertido em zero. Cobertura = peso disponível / 100.",
        fontsize=8,
        color="#34495e",
        transform=ax.transAxes,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def write_outputs(
    output_dir: str | Path,
    *,
    ranking: pd.DataFrame,
    metrics: dict,
    figures: list[str],
    tables: dict,
    summary: str,
    criteria: Iterable[Criterion],
    dpi: int = DEFAULT_REPORT_DPI,
) -> dict:
    """Write CSV, JSON, summary text, and classification figure outputs."""
    root = Path(output_dir)
    ranking_dir = root / "ranking"
    report_dir = root / "report"
    ranking_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    ranking_csv = ranking_dir / "classificacao_multicriterio.csv"
    presentation = _ranking_presentation_frame(ranking)
    presentation.to_csv(ranking_csv, index=False, sep=";", encoding="utf-8-sig")
    figure = write_classification_figure(
        ranking, ranking_dir / "classificacao.png", criteria, dpi=dpi
    )
    criteria_csv = ranking_dir / "criterios_classificacao.csv"
    pd.DataFrame(
        [
            {
                "Critério": c.label,
                "Unidade": c.unit,
                "Direção": "Maior é melhor"
                if c.direction == "max"
                else "Menor é melhor",
                "Peso": c.weight,
                "Peso normalizado (%)": c.weight,
            }
            for c in criteria
        ]
    ).to_csv(criteria_csv, index=False, sep=";", encoding="utf-8-sig")
    json_path = report_dir / "metricas_completas.json"
    payload = {
        "metrics": metrics,
        "ranking": ranking.to_dict(orient="records"),
        "figures": figures + [figure],
        "tables": {key: value for key, value in tables.items()},
        "criteria": criteria_as_dict(criteria),
        "summary": summary,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    summary_path = report_dir / "resumo_cientifico.txt"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    return {
        "ranking_csv": str(ranking_csv),
        "criteria_csv": str(criteria_csv),
        "classification_figure": figure,
        "json": str(json_path),
        "summary": str(summary_path),
    }
