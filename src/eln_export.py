"""Integrações de exportação para Electronic Lab Notebooks (ELNs).

Este módulo reúne funções simples para publicar resultados analíticos em
plataformas ELN populares. As dependências de rede são opcionais em tempo de
importação para que o pacote continue utilizável mesmo em ambientes mínimos.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import requests
    from requests import Response
    from requests.exceptions import RequestException
except ImportError:  # pragma: no cover - dependência opcional
    requests = None  # type: ignore[assignment]
    Response = Any  # type: ignore[assignment,misc]

    class RequestException(Exception):
        """Exceção substituta usada quando requests não está instalado."""


_SUPPORTED_ELNS = ["rspace", "labarchives", "benchling"]


def get_supported_elns() -> list[str]:
    """Retorna a lista de ELNs suportados pelo módulo."""
    return list(_SUPPORTED_ELNS)


def export_to_rspace(results: Any, api_key: str, server_url: str) -> dict[str, Any]:
    """Exporta resultados para uma instância do RSpace via API HTTP.

    Parameters
    ----------
    results : Any
        Estrutura serializável contendo os resultados a serem enviados.
    api_key : str
        Chave de API do RSpace.
    server_url : str
        URL base do servidor RSpace ou URL completa do endpoint.
    """
    endpoint = server_url.rstrip("/")
    if "/api/" not in endpoint:
        endpoint = f"{endpoint}/api/v1/documents"
    payload = {
        "name": "IonFlow Export",
        "fields": {"content": _serialise_results(results)},
    }
    headers = {"Authorization": f"ApiKey {api_key}"}
    return _post_export("RSpace", endpoint, payload, headers=headers)


def export_to_labarchives(
    results: Any,
    api_key: str,
    notebook_id: str,
) -> dict[str, Any]:
    """Exporta resultados para o LabArchives usando o identificador do caderno."""
    endpoint = f"https://api.labarchives.com/v2/notebooks/{notebook_id}/entries"
    payload = {
        "title": "IonFlow Export",
        "content": _serialise_results(results),
        "format": "json",
    }
    headers = {"X-Api-Key": api_key}
    return _post_export("LabArchives", endpoint, payload, headers=headers)


def export_to_benchling(results: Any, api_key: str, folder_id: str) -> dict[str, Any]:
    """Exporta resultados para o Benchling como uma entrada textual em pasta."""
    endpoint = "https://benchling.com/api/v2/entries"
    payload = {
        "name": "IonFlow Export",
        "folderId": folder_id,
        "text": _serialise_results(results),
    }
    headers = {"Authorization": f"******"}
    return _post_export("Benchling", endpoint, payload, headers=headers)


def _post_export(
    service_name: str,
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Executa uma requisição POST padronizada para serviços ELN."""
    if requests is None:
        message = f"Biblioteca requests não está disponível para exportar para {service_name}."
        logger.error(message)
        return {"success": False, "message": message, "service": service_name}

    request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    try:
        response = requests.post(
            url,
            headers=request_headers,
            json=payload,
            timeout=timeout,
        )
        return _build_response(service_name, response)
    except RequestException as exc:
        message = f"Falha de comunicação com {service_name}: {exc}"
        logger.error(message)
        return {"success": False, "message": message, "service": service_name}
    except Exception as exc:  # pragma: no cover - proteção extra
        message = f"Erro inesperado ao exportar para {service_name}: {exc}"
        logger.exception(message)
        return {"success": False, "message": message, "service": service_name}


def _build_response(service_name: str, response: Response) -> dict[str, Any]:
    """Normaliza a resposta HTTP em um dicionário amigável ao chamador."""
    try:
        body = response.json()
    except ValueError:
        body = response.text

    if response.ok:
        message = f"Exportação para {service_name} concluída com sucesso."
        logger.info("%s Status HTTP: %s", message, response.status_code)
        return {
            "success": True,
            "message": message,
            "service": service_name,
            "status_code": response.status_code,
            "response": body,
        }

    detail = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    message = (
        f"Exportação para {service_name} falhou com HTTP {response.status_code}: {detail}"
    )
    logger.warning(message)
    return {
        "success": False,
        "message": message,
        "service": service_name,
        "status_code": response.status_code,
        "response": body,
    }


def _serialise_results(results: Any) -> str:
    """Converte resultados em texto JSON legível para envio aos ELNs."""
    try:
        return json.dumps(results, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        logger.debug("Resultados não eram serializáveis diretamente; usando representação textual.")
        return str(results)
