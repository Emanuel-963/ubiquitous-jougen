"""Public orchestration API for the independent scientific protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import criteria_from_config
from .cv_dunn import analyze_cv
from .eis_drt import analyze_eis_drt
from .gcd import analyze_gcd
from .ranking import rank_multicriteria
from .report import write_outputs

DEFAULT_SCIENTIFIC_PROTOCOL_DPI = 300


def _load_config(config):
    if isinstance(config, (str, Path)) and str(config) != "default":
        return json.loads(Path(config).read_text(encoding="utf-8"))
    return config


def _discover_electrolytes(data_dir: Path) -> dict[str, dict]:
    manifest = data_dir / "protocol.json"
    if not manifest.is_file():
        project_manifest = data_dir / "project.json"
        if project_manifest.is_file():
            manifest = project_manifest
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        electrolytes = data.get("electrolytes", data)
        if data.get("schema_version") == "2.0":
            flattened = {}
            for electrolyte, group in electrolytes.items():
                for cell_id, cell in (
                    group.get("cells", {}) if isinstance(group, dict) else {}
                ).items():
                    item = dict(cell)
                    item["electrolyte"] = electrolyte
                    item["cell_id"] = cell_id
                    item["replicate"] = item.get("replicate", 1)
                    flattened[f"{electrolyte} [{cell_id}]"] = item
            electrolytes = flattened
        for spec in electrolytes.values():
            if not isinstance(spec, dict):
                continue
            for key in ("gcd",):
                if isinstance(spec.get(key), str) and not Path(spec[key]).is_absolute():
                    spec[key] = str(data_dir / spec[key])
            for key in ("cv", "eis"):
                entries = spec.get(key, {})
                if isinstance(entries, dict):
                    converted = {}
                    for name, value in entries.items():
                        if key == "cv":
                            try:
                                name = float(name)
                            except (TypeError, ValueError):
                                continue
                        converted[name] = (
                            str(data_dir / value)
                            if isinstance(value, str) and not Path(value).is_absolute()
                            else value
                        )
                    spec[key] = converted
        return electrolytes
    discovered = {}
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        cv_dir, eis_dir = folder / "cv", folder / "eis"
        gcd_files = list(folder.glob("*gcd*.txt")) + list(folder.glob("*GCD*.txt"))
        discovered[folder.name] = {
            "cv": {float(p.stem.split("_")[-1]): str(p) for p in cv_dir.glob("*.txt")}
            if cv_dir.is_dir()
            else {},
            "gcd": str(gcd_files[0]) if gcd_files else None,
            "eis": {
                state: str(eis_dir / f"{state}.txt")
                for state in ("initial", "post_cv", "final")
                if (eis_dir / f"{state}.txt").is_file()
            }
            if eis_dir.is_dir()
            else {},
        }
    return discovered


def _to_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def run_scientific_protocol(
    data_dir: str | Path,
    output_dir: str | Path = "results",
    config: object = "default",
) -> dict:
    """Run CV, GCD, EIS, DRT and workbook-compatible multicriteria ranking.

    Input can be described by ``data_dir/protocol.json`` or by one directory per
    electrolyte containing ``cv/*.txt``, ``eis/{initial,post_cv,final}.txt`` and
    one GCD text file. A manifest is recommended when filenames are irregular.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cv", "gcd", "eis", "drt", "ranking", "report"):
        (output_dir / name).mkdir(exist_ok=True)
    settings = _load_config(config)
    criteria = criteria_from_config(settings)
    dpi = (
        int(settings.get("dpi", DEFAULT_SCIENTIFIC_PROTOCOL_DPI))
        if isinstance(settings, dict)
        else DEFAULT_SCIENTIFIC_PROTOCOL_DPI
    )
    if dpi not in {150, 200, 300, 600}:
        raise ValueError("dpi deve ser 150, 200, 300 ou 600")
    electrolytes = _discover_electrolytes(data_dir)
    rows, all_figures, all_tables, raw_metrics = [], [], {}, {}

    for electrolyte, spec in electrolytes.items():
        spec = spec or {}
        safe_name = (
            "_".join(part for part in electrolyte.replace("/", "_").split() if part)
            or "electrolyte"
        )
        cv = analyze_cv(
            spec.get("cv", {}),
            output_dir / "cv" / safe_name,
            electrodes=spec.get("electrodes"),
            cell_area_cm2=spec.get("cell_area_cm2", 1.0),
            dpi=dpi,
        )
        gcd = {}
        if spec.get("gcd"):
            gcd = analyze_gcd(
                spec["gcd"],
                output_dir / "gcd" / safe_name,
                mass_g=spec.get("mass_g", 1.0),
                current_sequence_a_g=spec.get("current_sequence_a_g", []),
                dpi=dpi,
            )
        eis_drt = analyze_eis_drt(
            spec.get("eis", {}),
            output_dir / "eis" / safe_name,
            exclude_last_n=spec.get("exclude_last_n"),
            slow_tau_threshold_s=spec.get("slow_tau_threshold_s", 0.1),
            dpi=dpi,
        )
        merged = {}
        merged.update(cv.get("metrics", {}))
        merged.update(gcd.get("metrics", {}))
        merged.update(eis_drt.get("metrics", {}))
        merged["electrolyte"] = electrolyte
        rows.append(merged)
        raw_metrics[electrolyte] = merged
        all_figures.extend(
            cv.get("figures", []) + gcd.get("figures", []) + eis_drt.get("figures", [])
        )
        all_tables[f"{electrolyte}_gcd"] = gcd.get("table", [])
        all_tables[f"{electrolyte}_eis"] = eis_drt.get("tables", {})

    metrics_df = pd.DataFrame(rows)
    ranking = (
        rank_multicriteria(metrics_df, criteria) if not metrics_df.empty else metrics_df
    )
    summary = _build_summary(ranking)
    output_files = write_outputs(
        output_dir,
        ranking=ranking,
        metrics=raw_metrics,
        figures=all_figures,
        tables=all_tables,
        summary=summary,
        criteria=criteria,
        dpi=dpi,
    )
    return {
        "metrics": raw_metrics,
        "ranking": ranking,
        "figures": all_figures + [output_files["classification_figure"]],
        "tables": all_tables,
        "summary": summary,
        "output_files": output_files,
    }


def _build_summary(ranking: pd.DataFrame) -> str:
    if ranking.empty:
        return "Nenhum eletrólito com dados válidos foi encontrado."
    best = ranking.iloc[0]
    lines = [
        "Resumo científico reproduzível — protocolo multicritério IonFlow",
        "",
        f"Melhor classificação: {best['electrolyte']}",
        f"Score disponível: {best['score_available']:.2f}%",
        f"Cobertura: {best['coverage_pct']:.1f}%",
        f"Status: {best['protocol_status']}",
    ]
    partial = ranking[ranking["protocol_status"] == "Parcial"]
    if not partial.empty:
        lines.append(
            f"Observação: {len(partial)} eletrólito(s) possuem métricas ausentes e foram classificados como Parciais."
        )
    return "\n".join(lines)
