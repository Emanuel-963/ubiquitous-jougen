"""Configuração multi-tenant com isolamento lógico de dados.

Este módulo centraliza a criação e gestão de tenants para instalações que
precisam separar dados, bases SQLite e capacidades por cliente ou unidade
organizacional. As configurações são persistidas em JSON para facilitar backup,
revisão e auditoria operacional.

O gestor valida colisões de caminhos entre tenants para reduzir riscos de fuga
acidental de dados entre contextos distintos.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    """Representa a configuração isolada de um tenant."""

    tenant_id: str
    tenant_name: str
    data_dir: str
    db_path: str
    max_users: int
    features: list[str] = field(default_factory=list)
    storage_quota_mb: int = 0


class TenantManager:
    """Cria, persiste e alterna contextos de tenants de forma segura."""

    def __init__(self, config_path: str | Path = "tenants.json") -> None:
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def create_tenant(
        self,
        name: str,
        config: TenantConfig | dict[str, Any],
    ) -> TenantConfig:
        """Cria um novo tenant e persiste a sua configuração isolada."""
        tenant = self._coerce_config(name, config)
        self._validate_isolation(tenant)

        tenant_data_dir = Path(tenant.data_dir)
        tenant_db_path = Path(tenant.db_path)
        tenant_data_dir.mkdir(parents=True, exist_ok=True)
        tenant_db_path.parent.mkdir(parents=True, exist_ok=True)

        tenants = self._state.setdefault("tenants", {})
        if tenant.tenant_id in tenants:
            raise ValueError(f"Tenant já existente: {tenant.tenant_id}")

        tenants[tenant.tenant_id] = asdict(tenant)
        if self._state.get("current_tenant") is None:
            self._state["current_tenant"] = tenant.tenant_id
        self._save_state()
        logger.info("Tenant criado: %s", tenant.tenant_id)
        return tenant

    def get_tenant(self, tenant_id: str) -> TenantConfig | None:
        """Obtém a configuração de um tenant pelo identificador."""
        data = self._state.get("tenants", {}).get(tenant_id)
        return TenantConfig(**data) if data else None

    def list_tenants(self) -> list[TenantConfig]:
        """Lista todos os tenants configurados."""
        return [TenantConfig(**data) for data in self._state.get("tenants", {}).values()]

    def delete_tenant(self, tenant_id: str) -> bool:
        """Remove o tenant da configuração e apaga os seus artefactos locais."""
        tenants = self._state.get("tenants", {})
        tenant_data = tenants.pop(tenant_id, None)
        if tenant_data is None:
            return False

        tenant = TenantConfig(**tenant_data)
        self._delete_tenant_artifacts(tenant)
        if self._state.get("current_tenant") == tenant_id:
            self._state["current_tenant"] = None
        self._save_state()
        logger.info("Tenant removido: %s", tenant_id)
        return True

    def switch_tenant(self, tenant_id: str) -> TenantConfig:
        """Ativa o contexto de um tenant existente."""
        tenant = self.get_tenant(tenant_id)
        if tenant is None:
            raise KeyError(f"Tenant inexistente: {tenant_id}")
        self._state["current_tenant"] = tenant_id
        self._save_state()
        logger.info("Contexto ativo alterado para o tenant %s", tenant_id)
        return tenant

    def get_current_tenant(self) -> TenantConfig | None:
        """Devolve o tenant atualmente ativo, se existir."""
        current_id = self._state.get("current_tenant")
        return self.get_tenant(current_id) if current_id else None

    def _load_state(self) -> dict[str, Any]:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("Configuração multi-tenant inválida, a recriar: %s", exc)
        return {"current_tenant": None, "tenants": {}}

    def _save_state(self) -> None:
        self.config_path.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _coerce_config(
        self,
        name: str,
        config: TenantConfig | dict[str, Any],
    ) -> TenantConfig:
        if isinstance(config, TenantConfig):
            tenant = config
        else:
            config_data = dict(config)
            config_data.setdefault("tenant_id", self._slugify(name))
            config_data.setdefault("tenant_name", name)
            tenant = TenantConfig(**config_data)
        if not tenant.tenant_name:
            tenant.tenant_name = name
        return tenant

    def _validate_isolation(self, candidate: TenantConfig) -> None:
        candidate_data_dir = Path(candidate.data_dir).resolve()
        candidate_db_path = Path(candidate.db_path).resolve()
        if candidate_data_dir == candidate_db_path:
            raise ValueError("data_dir e db_path devem ser distintos para manter isolamento")

        for tenant in self.list_tenants():
            existing_data_dir = Path(tenant.data_dir).resolve()
            existing_db_path = Path(tenant.db_path).resolve()
            if candidate_data_dir == existing_data_dir:
                raise ValueError(
                    f"Diretório de dados já usado pelo tenant {tenant.tenant_id}: {candidate_data_dir}"
                )
            if candidate_db_path == existing_db_path:
                raise ValueError(
                    f"Base de dados já usada pelo tenant {tenant.tenant_id}: {candidate_db_path}"
                )
            if candidate_data_dir.is_relative_to(existing_data_dir):
                raise ValueError(
                    "data_dir do novo tenant não pode ficar dentro do data_dir de outro tenant"
                )
            if existing_data_dir.is_relative_to(candidate_data_dir):
                raise ValueError(
                    "data_dir do novo tenant não pode englobar o diretório de dados de outro tenant"
                )
            if candidate_db_path.is_relative_to(existing_data_dir):
                raise ValueError(
                    "db_path do novo tenant não pode ficar dentro do data_dir de outro tenant"
                )
            if existing_db_path.is_relative_to(candidate_data_dir):
                raise ValueError(
                    "data_dir do novo tenant não pode conter a base de dados de outro tenant"
                )

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "tenant"

    @staticmethod
    def _delete_tenant_artifacts(tenant: TenantConfig) -> None:
        data_dir = Path(tenant.data_dir)
        db_path = Path(tenant.db_path)
        if db_path.exists() and db_path.is_file():
            db_path.unlink()
        if data_dir.exists() and data_dir.is_dir():
            shutil.rmtree(data_dir)
