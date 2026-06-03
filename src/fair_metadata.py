"""Geração de metadados FAIR em JSON-LD para resultados eletroquímicos.

Este módulo produz descrições interoperáveis baseadas em JSON-LD, combinando
termos do Schema.org com um pequeno vocabulário orientado à eletroquímica para
facilitar rastreabilidade, compartilhamento e reuso de dados.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = ["@context", "@type", "name", "author", "dateCreated", "measurementTechnique"]


def generate_jsonld(
    results: Any,
    sample_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gera um documento JSON-LD para um conjunto de resultados.

    Parameters
    ----------
    results : Any
        Resultado bruto ou resumido do experimento, preferencialmente em forma
        de ``dict``.
    sample_info : Mapping[str, Any], optional
        Metadados complementares da amostra, como autor, instrumento,
        identificador interno ou parâmetros experimentais.
    """
    result_map = _to_mapping(results)
    info = dict(sample_info or {})
    method = _infer_method(result_map, info)
    parameters = _normalise_parameters(result_map, info)

    metadata = {
        "@context": {
            "schema": "https://schema.org/",
            "dcterms": "http://purl.org/dc/terms/",
            "electrochem": "https://w3id.org/emmo/domain/electrochemistry#",
            "name": "schema:name",
            "description": "schema:description",
            "author": "schema:author",
            "dateCreated": "schema:dateCreated",
            "instrument": "schema:instrument",
            "measurementTechnique": "schema:measurementTechnique",
            "variableMeasured": "schema:variableMeasured",
            "isBasedOn": "schema:isBasedOn",
            "parameters": "electrochem:hasExperimentalParameter",
        },
        "@type": "schema:Dataset",
        "name": info.get("dataset_name") or info.get("name") or result_map.get("name") or "IonFlow Dataset",
        "description": info.get("description") or result_map.get("summary") or "Resultados analíticos exportados pelo IonFlow.",
        "author": {
            "@type": "schema:Person",
            "name": info.get("author") or result_map.get("author") or "Desconhecido",
        },
        "dateCreated": info.get("date") or result_map.get("date") or datetime.now(timezone.utc).isoformat(),
        "measurementTechnique": {
            "@type": "schema:DefinedTerm",
            "name": method,
            "inDefinedTermSet": "https://w3id.org/emmo/domain/electrochemistry",
        },
        "instrument": {
            "@type": "schema:Thing",
            "name": info.get("instrument") or result_map.get("instrument") or "Não informado",
        },
        "parameters": parameters,
        "variableMeasured": info.get("variables") or result_map.get("variables") or ["impedance"],
        "isBasedOn": info.get("source") or result_map.get("source_file"),
    }

    logger.debug("JSON-LD FAIR gerado para método %s", method)
    return metadata


def validate_jsonld(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Valida a presença dos campos mínimos esperados no JSON-LD."""
    missing = [field for field in _REQUIRED_FIELDS if not metadata.get(field)]
    is_valid = not missing
    message = "Metadados JSON-LD válidos." if is_valid else f"Campos obrigatórios ausentes: {', '.join(missing)}"
    if is_valid:
        logger.info("Validação JSON-LD concluída com sucesso.")
    else:
        logger.warning(message)
    return {"valid": is_valid, "missing_fields": missing, "message": message}


def save_jsonld(metadata: Mapping[str, Any], path: str | Path) -> Path:
    """Salva os metadados JSON-LD em disco com codificação UTF-8."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Metadados JSON-LD salvos em %s", destination)
    return destination


def _to_mapping(results: Any) -> dict[str, Any]:
    """Converte diferentes tipos de entrada em um dicionário simples."""
    if isinstance(results, Mapping):
        return dict(results)
    if hasattr(results, "__dict__"):
        return dict(vars(results))
    return {"value": results}


def _infer_method(result_map: Mapping[str, Any], sample_info: Mapping[str, Any]) -> str:
    """Infere o método principal entre EIS, DRT e cycling."""
    candidates = [
        sample_info.get("method"),
        result_map.get("method"),
        result_map.get("analysis_type"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        value = str(candidate).strip().lower()
        if "drt" in value:
            return "DRT"
        if "cycl" in value:
            return "cycling"
        if "eis" in value or "impedance" in value:
            return "EIS"
    if "gamma" in result_map or "tau" in result_map:
        return "DRT"
    if "cycle" in result_map or "capacity" in result_map:
        return "cycling"
    return "EIS"


def _normalise_parameters(
    result_map: Mapping[str, Any],
    sample_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Agrupa parâmetros experimentais relevantes em um único objeto."""
    parameters = {}
    for source in (
        sample_info.get("parameters"),
        result_map.get("parameters"),
        result_map.get("params"),
    ):
        if isinstance(source, Mapping):
            parameters.update(source)

    for key in ("frequency_range", "temperature", "electrolyte", "amplitude", "scan_rate"):
        value = sample_info.get(key, result_map.get(key))
        if value is not None:
            parameters[key] = value

    return parameters
