"""Assinatura digital de PDFs com degradação controlada.

Este módulo oferece uma interface simples para gerar e verificar evidências de
assinatura de ficheiros PDF. Quando a biblioteca ``cryptography`` está
instalada, é criado um artefacto PKCS#7 destacado (detached signature) em ficheiro
sidecar ``.p7s``. Em ambientes mínimos, o módulo degrada de forma segura para
um modo baseado em hash SHA-256 com carimbo temporal, preservando rastreabilidade.

Importante: validação completa de assinatura PKCS#7 embutida no próprio PDF pode
exigir bibliotecas e ferramentas adicionais. Aqui a verificação local confirma a
integridade do conteúdo e a presença dos metadados de assinatura gerados por este
módulo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependência opcional
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.serialization import pkcs7

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:  # pragma: no cover - dependência opcional
    x509 = None
    hashes = None
    serialization = None
    pkcs7 = None
    CRYPTOGRAPHY_AVAILABLE = False


@dataclass(frozen=True)
class SignatureInfo:
    """Resumo do estado de uma assinatura digital conhecida pela aplicação."""

    signer: str
    timestamp: str
    algorithm: str
    valid: bool


def generate_signature_hash(pdf_bytes: bytes) -> dict[str, str]:
    """Gera um resumo SHA-256 acompanhado de carimbo temporal UTC."""
    return {
        "sha256": sha256(pdf_bytes).hexdigest(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def sign_pdf(
    pdf_path: str | Path,
    private_key_path: str | Path,
    cert_path: str | Path,
) -> Path:
    """Assina um PDF com PKCS#7 quando possível, ou usa modo hash-only.

    A assinatura é persistida em ficheiros sidecar ao lado do PDF para manter o
    documento original intacto. O ficheiro ``.sig.json`` contém metadados da
    operação e é usado pela rotina de verificação.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF inexistente: {pdf_file}")

    pdf_bytes = pdf_file.read_bytes()
    signature_hash = generate_signature_hash(pdf_bytes)
    signature_metadata_path = _signature_metadata_path(pdf_file)
    signature_blob_path = _signature_blob_path(pdf_file)

    metadata: dict[str, Any] = {
        "pdf_path": str(pdf_file),
        "timestamp": signature_hash["timestamp"],
        "sha256": signature_hash["sha256"],
        "algorithm": "SHA256-only",
        "signer": "hash-only",
        "signature_file": None,
        "certificate_file": str(cert_path),
        "private_key_file": str(private_key_path),
    }

    if CRYPTOGRAPHY_AVAILABLE:
        try:
            private_key = serialization.load_pem_private_key(
                Path(private_key_path).read_bytes(),
                None,
            )
            certificate = _load_certificate(Path(cert_path))
            signature_bytes = (
                pkcs7.PKCS7SignatureBuilder()
                .set_data(pdf_bytes)
                .add_signer(certificate, private_key, hashes.SHA256())
                .sign(
                    serialization.Encoding.DER,
                    [pkcs7.PKCS7Options.DetachedSignature, pkcs7.PKCS7Options.Binary],
                )
            )
            signature_blob_path.write_bytes(signature_bytes)
            metadata.update(
                {
                    "algorithm": "PKCS7-SHA256",
                    "signer": certificate.subject.rfc4514_string(),
                    "signature_file": str(signature_blob_path),
                }
            )
            logger.info("Assinatura PKCS#7 destacada criada para %s", pdf_file)
        except Exception as exc:  # pragma: no cover - depende do ambiente
            logger.warning(
                "Falha ao gerar PKCS#7; a usar modo hash-only para %s: %s",
                pdf_file,
                exc,
            )
    else:
        logger.info(
            "Biblioteca cryptography indisponível; a usar assinatura baseada em hash"
        )

    signature_metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return signature_metadata_path


def verify_signature(pdf_path: str | Path) -> SignatureInfo:
    """Verifica a integridade de uma assinatura previamente criada.

    A verificação confirma se o hash atual do PDF coincide com o hash registado
    e se os artefactos sidecar necessários continuam presentes.
    """
    pdf_file = Path(pdf_path)
    metadata_path = _signature_metadata_path(pdf_file)
    if not pdf_file.exists() or not metadata_path.exists():
        return SignatureInfo(
            signer="desconhecido",
            timestamp="",
            algorithm="unknown",
            valid=False,
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    current_hash = generate_signature_hash(pdf_file.read_bytes())["sha256"]
    valid = current_hash == metadata.get("sha256")

    signature_file = metadata.get("signature_file")
    if metadata.get("algorithm", "").startswith("PKCS7"):
        valid = valid and bool(signature_file) and Path(signature_file).exists()

    return SignatureInfo(
        signer=metadata.get("signer", "desconhecido"),
        timestamp=metadata.get("timestamp", ""),
        algorithm=metadata.get("algorithm", "unknown"),
        valid=valid,
    )


def _signature_metadata_path(pdf_file: Path) -> Path:
    return pdf_file.with_suffix(f"{pdf_file.suffix}.sig.json")


def _signature_blob_path(pdf_file: Path) -> Path:
    return pdf_file.with_suffix(f"{pdf_file.suffix}.p7s")


def _load_certificate(cert_path: Path) -> Any:
    cert_bytes = cert_path.read_bytes()
    try:
        return x509.load_pem_x509_certificate(cert_bytes)
    except ValueError:
        return x509.load_der_x509_certificate(cert_bytes)
