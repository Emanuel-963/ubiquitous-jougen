"""Gestão de modo offline para ambientes isolados da rede.

Este módulo fornece um gestor de operação air-gapped para instalações que
precisam bloquear tráfego de rede e trabalhar apenas com recursos previamente
copiados. O comportamento é persistido em ficheiro JSON para que a aplicação
arranque já no estado configurado, inclusive após reinícios.

Quando o modo offline está ativo, chamadas conhecidas de rede são
monkey-patched para levantar uma exceção explícita. Isso evita acessos
acidentais a serviços externos e facilita a evidência de conformidade em
ambientes regulados.
"""

from __future__ import annotations

import http.client
import json
import logging
import shutil
import socket
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependência opcional
    import requests
except ImportError:  # pragma: no cover - dependência opcional
    requests = None


class OfflineModeError(RuntimeError):
    """Erro levantado quando uma operação de rede é bloqueada pelo modo offline."""


class OfflineManager:
    """Controla o estado offline e o cache local de recursos críticos.

    Parameters
    ----------
    settings_path:
        Caminho do ficheiro JSON com o estado persistido.
    cache_dir:
        Diretório onde os recursos pré-carregados são armazenados.
    """

    def __init__(
        self,
        settings_path: str | Path = "offline_settings.json",
        cache_dir: str | Path = "offline_cache",
    ) -> None:
        self.settings_path = Path(settings_path)
        self.cache_dir = Path(cache_dir)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._originals: dict[str, Any] = {}
        self._settings = self._load_settings()
        if self._settings.get("offline_mode"):
            self._apply_network_patches()

    def enable_offline_mode(self) -> None:
        """Ativa o modo offline e bloqueia chamadas de rede suportadas."""
        self._apply_network_patches()
        self._settings["offline_mode"] = True
        self._save_settings()
        logger.warning("Modo offline ativado")

    def disable_offline_mode(self) -> None:
        """Desativa o modo offline e restaura o comportamento normal de rede."""
        self._restore_network_patches()
        self._settings["offline_mode"] = False
        self._save_settings()
        logger.info("Modo offline desativado")

    def is_offline(self) -> bool:
        """Indica se o sistema está atualmente a operar sem rede."""
        return bool(self._settings.get("offline_mode", False))

    def cache_resources(self, resources_dir: str | Path) -> list[str]:
        """Copia recursos necessários para um cache local verificável.

        Todos os ficheiros encontrados em ``resources_dir`` são copiados para o
        ``cache_dir`` preservando a estrutura relativa.
        """
        source_dir = Path(resources_dir)
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"Diretório de recursos inexistente: {source_dir}")

        copied: list[str] = []
        for file_path in source_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(source_dir)
            destination = self.cache_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, destination)
            copied.append(relative.as_posix())

        self._settings["cached_resources"] = copied
        self._settings["source_resources_dir"] = str(source_dir)
        self._save_settings()
        logger.info("%s recurso(s) copiado(s) para cache offline", len(copied))
        return copied

    def verify_cached_resources(self) -> bool:
        """Valida se todos os recursos declarados no manifesto estão presentes."""
        cached_resources = self._settings.get("cached_resources", [])
        if not cached_resources:
            logger.warning("Nenhum recurso offline foi registado no manifesto")
            return False

        missing = [
            resource
            for resource in cached_resources
            if not (self.cache_dir / resource).exists()
        ]
        if missing:
            logger.error("Recursos offline em falta: %s", ", ".join(missing))
            return False
        return True

    def _load_settings(self) -> dict[str, Any]:
        if self.settings_path.exists():
            try:
                return json.loads(self.settings_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("Configuração offline inválida, a recriar: %s", exc)
        return {"offline_mode": False, "cached_resources": []}

    def _save_settings(self) -> None:
        self.settings_path.write_text(
            json.dumps(self._settings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _apply_network_patches(self) -> None:
        if self._originals:
            return

        def _blocked(*args: Any, **kwargs: Any) -> Any:
            raise OfflineModeError(
                "Chamadas de rede estão desativadas porque o modo offline está ativo."
            )

        self._patch(socket, "create_connection", _blocked)
        self._patch(socket.socket, "connect", _blocked)
        self._patch(socket.socket, "connect_ex", _blocked)
        self._patch(urllib.request, "urlopen", _blocked)
        self._patch(http.client.HTTPConnection, "connect", _blocked)
        self._patch(http.client.HTTPSConnection, "connect", _blocked)

        if requests is not None:
            self._patch(requests.sessions.Session, "request", _blocked)
            self._patch(requests.api, "request", _blocked)

    def _restore_network_patches(self) -> None:
        for patch_key, original in list(self._originals.items()):
            target_name, attribute = patch_key.split(":", maxsplit=1)
            target = self._resolve_patch_target(target_name)
            setattr(target, attribute, original)
        self._originals.clear()

    def _patch(self, target: Any, attribute: str, replacement: Any) -> None:
        patch_key = f"{self._target_name(target)}:{attribute}"
        if patch_key in self._originals:
            return
        self._originals[patch_key] = getattr(target, attribute)
        setattr(target, attribute, replacement)

    @staticmethod
    def _target_name(target: Any) -> str:
        module = getattr(target, "__module__", "")
        qualname = getattr(target, "__qualname__", getattr(target, "__name__", repr(target)))
        return f"{module}.{qualname}".strip(".")

    @staticmethod
    def _resolve_patch_target(target_name: str) -> Any:
        mapping = {
            OfflineManager._target_name(socket): socket,
            OfflineManager._target_name(socket.socket): socket.socket,
            OfflineManager._target_name(urllib.request): urllib.request,
            OfflineManager._target_name(http.client.HTTPConnection): http.client.HTTPConnection,
            OfflineManager._target_name(http.client.HTTPSConnection): http.client.HTTPSConnection,
        }
        if requests is not None:
            mapping[OfflineManager._target_name(requests.sessions.Session)] = requests.sessions.Session
            mapping[OfflineManager._target_name(requests.api)] = requests.api
        return mapping[target_name]
