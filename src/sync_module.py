"""Módulo de sincronização para resultados e diretórios laboratoriais.

A classe :class:`SyncManager` centraliza sincronizações por ``rsync`` e S3,
registrando metadados locais em JSON para auditoria simples. Importações
opcionais são protegidas para manter o módulo importável em instalações
mínimas.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import boto3
except ImportError:  # pragma: no cover - dependência opcional
    boto3 = None  # type: ignore[assignment]


class SyncManager:
    """Gerencia sincronizações locais/remotas e o histórico associado.

    Parameters
    ----------
    metadata_path : str | Path, optional
        Caminho do arquivo JSON usado para persistir o estado da última
        sincronização.
    """

    def __init__(self, metadata_path: str | Path = ".sync_metadata.json") -> None:
        self._metadata_path = Path(metadata_path)
        self._lock = threading.RLock()
        self._metadata = self._load_metadata()

    def sync_rsync(self, local_dir: str | Path, remote_path: str) -> dict[str, Any]:
        """Sincroniza um diretório local com um destino remoto usando rsync."""
        source = Path(local_dir).expanduser().resolve()
        if not source.is_dir():
            message = f"Diretório local inválido para rsync: {source}"
            logger.error(message)
            return {"success": False, "message": message}

        if shutil.which("rsync") is None:
            message = "Comando rsync não está disponível no ambiente atual."
            logger.error(message)
            return {"success": False, "message": message}

        files_synced = self._count_files(source)
        command = ["rsync", "-a", f"{source}/", remote_path]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as exc:
            message = f"Falha ao executar rsync: {exc}"
            logger.error(message)
            return {"success": False, "message": message}

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            message = f"rsync retornou código {completed.returncode}: {stderr}"
            logger.warning(message)
            return {"success": False, "message": message, "files_synced": 0}

        return self._record_sync(
            backend="rsync",
            target=remote_path,
            files_synced=files_synced,
            message="Sincronização via rsync concluída com sucesso.",
        )

    def sync_s3(self, local_dir: str | Path, bucket: str, prefix: str = "") -> dict[str, Any]:
        """Sincroniza arquivos para um bucket S3 usando boto3, quando disponível."""
        if boto3 is None:
            message = "Biblioteca boto3 não está disponível para sincronização S3."
            logger.error(message)
            return {"success": False, "message": message}

        source = Path(local_dir).expanduser().resolve()
        if not source.is_dir():
            message = f"Diretório local inválido para S3: {source}"
            logger.error(message)
            return {"success": False, "message": message}

        client = boto3.client("s3")
        uploaded = 0
        cleaned_prefix = prefix.strip("/")

        try:
            for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
                relative_path = file_path.relative_to(source).as_posix()
                key = "/".join(part for part in (cleaned_prefix, relative_path) if part)
                client.upload_file(str(file_path), bucket, key)
                uploaded += 1
                logger.debug("Arquivo enviado ao S3: %s -> s3://%s/%s", file_path, bucket, key)
        except Exception as exc:  # pragma: no cover - depende de serviço externo
            message = f"Falha durante sincronização S3: {exc}"
            logger.exception(message)
            return {"success": False, "message": message, "files_synced": uploaded}

        return self._record_sync(
            backend="s3",
            target=f"s3://{bucket}/{cleaned_prefix}".rstrip("/"),
            files_synced=uploaded,
            message="Sincronização para S3 concluída com sucesso.",
        )

    def get_sync_status(self) -> dict[str, Any]:
        """Retorna o estado persistido da última sincronização conhecida."""
        with self._lock:
            return dict(self._metadata)

    def _record_sync(
        self,
        *,
        backend: str,
        target: str,
        files_synced: int,
        message: str,
    ) -> dict[str, Any]:
        """Atualiza o arquivo de metadados após uma sincronização bem-sucedida."""
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "last_sync_time": timestamp,
            "files_synced": files_synced,
            "backend": backend,
            "target": target,
        }
        with self._lock:
            self._metadata = payload
            self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
            self._metadata_path.write_text(
                json.dumps(self._metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        logger.info("%s Destino: %s", message, target)
        return {"success": True, "message": message, **payload}

    def _load_metadata(self) -> dict[str, Any]:
        """Carrega o histórico anterior de sincronização, se existir."""
        if not self._metadata_path.exists():
            return {"last_sync_time": None, "files_synced": 0}
        try:
            return json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Não foi possível ler metadados de sincronização: %s", exc)
            return {"last_sync_time": None, "files_synced": 0}

    @staticmethod
    def _count_files(directory: Path) -> int:
        """Conta arquivos regulares em um diretório recursivamente."""
        return sum(1 for path in directory.rglob("*") if path.is_file())
