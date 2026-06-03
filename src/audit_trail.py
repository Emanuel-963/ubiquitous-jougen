"""Trilha de auditoria com evidência contra adulteração.

Este módulo implementa uma trilha de auditoria orientada a conformidade para
cenários regulados, incluindo ISO 17025 e 21 CFR Part 11. Cada evento é
persistido em SQLite com carimbo de data/hora em UTC e com encadeamento de hash
SHA-256, tornando alterações retroativas detectáveis.

A implementação foi desenhada para uso local, simples de auditar e resiliente a
falhas comuns de I/O. Dependências opcionais são tratadas com degradação
controlada; por exemplo, a exportação em PDF utiliza ``fpdf2`` quando
instalado.
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependência opcional
    from fpdf import FPDF
except ImportError:  # pragma: no cover - dependência opcional
    FPDF = None


@dataclass(frozen=True)
class AuditEntry:
    """Representa uma entrada individual da trilha de auditoria.

    Attributes
    ----------
    timestamp:
        Instante UTC em formato ISO-8601.
    user:
        Identificador do utilizador ou serviço que originou a ação.
    action:
        Nome curto da ação auditada.
    details:
        Dados adicionais serializáveis em JSON associados ao evento.
    hash_chain:
        Hash SHA-256 do hash anterior concatenado com os dados correntes.
    """

    timestamp: str
    user: str
    action: str
    details: Any = None
    hash_chain: str = ""


class AuditTrail:
    """Gere uma trilha de auditoria persistente, consultável e verificável.

    Parameters
    ----------
    db_path:
        Caminho do ficheiro SQLite usado para armazenar os eventos. Por omissão,
        utiliza ``audit_trail.db`` no diretório de trabalho da aplicação.
    """

    def __init__(self, db_path: str | Path = "audit_trail.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise_database()

    def log_action(
        self, user: str, action: str, details: Any = None
    ) -> AuditEntry:
        """Regista uma nova ação relevante na trilha de auditoria.

        O hash da nova entrada é calculado a partir do hash anterior e do
        conteúdo atual, formando uma cadeia encadeada e tamper-evident.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        details_json = self._serialise_details(details)
        previous_hash = self._get_last_hash()
        current_hash = self._calculate_hash(
            previous_hash=previous_hash,
            timestamp=timestamp,
            user=user,
            action=action,
            details_json=details_json,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_entries (timestamp, user, action, details, hash_chain)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, user, action, details_json, current_hash),
            )
            conn.commit()

        entry = AuditEntry(
            timestamp=timestamp,
            user=user,
            action=action,
            details=json.loads(details_json),
            hash_chain=current_hash,
        )
        logger.info("Ação auditada registada: %s por %s", action, user)
        return entry

    def get_entries(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> list[AuditEntry]:
        """Obtém entradas por intervalo temporal.

        Parameters
        ----------
        start, end:
            Limites opcionalmente fornecidos como ``datetime`` ou texto
            ISO-8601. Quando omitidos, todo o histórico é devolvido.
        """
        query = (
            "SELECT timestamp, user, action, details, hash_chain "
            "FROM audit_entries WHERE 1=1"
        )
        params: list[str] = []

        if start is not None:
            query += " AND timestamp >= ?"
            params.append(self._normalise_timestamp(start))
        if end is not None:
            query += " AND timestamp <= ?"
            params.append(self._normalise_timestamp(end))
        query += " ORDER BY id ASC"

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def verify_integrity(self) -> bool:
        """Verifica se a cadeia de hashes permanece íntegra.

        Returns
        -------
        bool
            ``True`` quando todas as entradas recalculam para o mesmo hash
            armazenado; ``False`` em qualquer divergência.
        """
        previous_hash = "GENESIS"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT timestamp, user, action, details, hash_chain FROM audit_entries ORDER BY id ASC"
            ).fetchall()

        for timestamp, user, action, details_json, stored_hash in rows:
            expected_hash = self._calculate_hash(
                previous_hash=previous_hash,
                timestamp=timestamp,
                user=user,
                action=action,
                details_json=details_json,
            )
            if stored_hash != expected_hash:
                logger.error(
                    "Falha de integridade na trilha de auditoria em %s (%s)",
                    timestamp,
                    action,
                )
                return False
            previous_hash = stored_hash

        return True

    def export_csv(self, path: str | Path) -> Path:
        """Exporta a trilha em CSV para inspeção de auditoria."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        entries = self.get_entries()

        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["timestamp", "user", "action", "details", "hash_chain"],
            )
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        "timestamp": entry.timestamp,
                        "user": entry.user,
                        "action": entry.action,
                        "details": json.dumps(entry.details, ensure_ascii=False),
                        "hash_chain": entry.hash_chain,
                    }
                )

        logger.info("Trilha de auditoria exportada em CSV: %s", destination)
        return destination

    def export_pdf(self, path: str | Path) -> Path:
        """Exporta a trilha de auditoria em PDF formatado.

        A exportação depende de ``fpdf2``. Quando a biblioteca não está
        instalada, é levantado um erro com mensagem objetiva para facilitar a
        correção do ambiente.
        """
        if FPDF is None:
            raise RuntimeError(
                "Exportação PDF indisponível: instale a dependência opcional 'fpdf2'."
            )

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Audit Trail", ln=True)
        pdf.set_font("Helvetica", size=9)
        pdf.multi_cell(
            0,
            5,
            f"Base de dados: {self.db_path} | Integridade: {'OK' if self.verify_integrity() else 'FALHA'}",
        )
        pdf.ln(2)

        for entry in self.get_entries():
            details_text = json.dumps(entry.details, ensure_ascii=False)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"{entry.timestamp} | {entry.user} | {entry.action}", ln=True)
            pdf.set_font("Helvetica", size=9)
            pdf.multi_cell(0, 5, f"Detalhes: {details_text}")
            pdf.multi_cell(0, 5, f"Hash chain: {entry.hash_chain}")
            pdf.ln(2)

        pdf.output(str(destination))
        logger.info("Trilha de auditoria exportada em PDF: %s", destination)
        return destination

    def _initialise_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL,
                    hash_chain TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _get_last_hash(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT hash_chain FROM audit_entries ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else "GENESIS"

    @staticmethod
    def _serialise_details(details: Any) -> str:
        return json.dumps(details, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _calculate_hash(
        *,
        previous_hash: str,
        timestamp: str,
        user: str,
        action: str,
        details_json: str,
    ) -> str:
        payload = "|".join([previous_hash, timestamp, user, action, details_json])
        return sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalise_timestamp(value: datetime | str) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        return value

    @staticmethod
    def _row_to_entry(row: tuple[str, str, str, str, str]) -> AuditEntry:
        timestamp, user, action, details_json, hash_chain = row
        return AuditEntry(
            timestamp=timestamp,
            user=user,
            action=action,
            details=json.loads(details_json),
            hash_chain=hash_chain,
        )
