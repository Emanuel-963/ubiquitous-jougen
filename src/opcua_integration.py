"""Integração OPC-UA em formato de stub para expansão futura.

Este módulo define uma camada inicial para comunicação com servidores OPC-UA em
contextos industriais. O foco é fornecer uma API estável dentro da aplicação,
permitindo que uma implementação completa seja adicionada posteriormente sem
alterar o código cliente.

Quando a biblioteca opcional ``opcua`` não está instalada, os métodos levantam
``NotImplementedError`` com mensagens orientativas. Isso permite distribuir o
módulo em ambientes onde a integração ainda não foi provisionada.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependência opcional
    from opcua import Client as SyncOPCUAClient

    OPCUA_LIBRARY = "opcua"
    OPCUA_IMPORT_ERROR: Exception | None = None
except ImportError as opcua_error:  # pragma: no cover - dependência opcional
    SyncOPCUAClient = None
    OPCUA_LIBRARY = ""
    OPCUA_IMPORT_ERROR = opcua_error
    try:  # pragma: no cover - dependência opcional
        import asyncua  # noqa: F401

        OPCUA_LIBRARY = "asyncua"
    except ImportError as asyncua_error:  # pragma: no cover - dependência opcional
        OPCUA_IMPORT_ERROR = asyncua_error


EXAMPLE_EIS_NODE_IDS = {
    "frequency_hz": "ns=2;s=IonFlow.EIS.FrequencyHz",
    "impedance_real_ohm": "ns=2;s=IonFlow.EIS.ImpedanceRealOhm",
    "impedance_imag_ohm": "ns=2;s=IonFlow.EIS.ImpedanceImagOhm",
    "phase_deg": "ns=2;s=IonFlow.EIS.PhaseDeg",
}


class _SubscriptionHandler:
    """Adaptador simples para callbacks de alteração de dados."""

    def __init__(self, callback: Callable[[str, Any, Any], None]) -> None:
        self.callback = callback

    def datachange_notification(self, node: Any, value: Any, data: Any) -> None:
        self.callback(str(node), value, data)


class OPCUAClient:
    """Cliente OPC-UA inicial com operações básicas e mensagens orientativas."""

    def __init__(self) -> None:
        self.endpoint_url: str | None = None
        self._client: Any = None
        self._subscription: Any = None

    def connect(self, endpoint_url: str) -> None:
        """Liga-se a um servidor OPC-UA suportado."""
        self.endpoint_url = endpoint_url
        if SyncOPCUAClient is None:
            self._raise_missing_dependency(
                "Instale 'opcua' para ativar a ligação síncrona ao servidor OPC-UA."
            )
        self._client = SyncOPCUAClient(endpoint_url)
        self._client.connect()
        logger.info("Ligação OPC-UA estabelecida com %s", endpoint_url)

    def disconnect(self) -> None:
        """Encerra a ligação OPC-UA atual."""
        if self._client is None:
            return
        self._client.disconnect()
        self._client = None
        self._subscription = None
        logger.info("Ligação OPC-UA encerrada")

    def read_node(self, node_id: str) -> Any:
        """Lê o valor corrente de um nó OPC-UA."""
        self._ensure_connected()
        return self._client.get_node(node_id).get_value()

    def write_node(self, node_id: str, value: Any) -> None:
        """Escreve um valor num nó OPC-UA."""
        self._ensure_connected()
        self._client.get_node(node_id).set_value(value)
        logger.info("Valor publicado em %s", node_id)

    def subscribe_changes(
        self,
        node_ids: list[str],
        callback: Callable[[str, Any, Any], None],
    ) -> Any:
        """Subscreve alterações de dados para uma lista de nós.

        O callback recebe ``(node_id, value, data)``.
        """
        self._ensure_connected()
        handler = _SubscriptionHandler(callback)
        self._subscription = self._client.create_subscription(500, handler)
        nodes = [self._client.get_node(node_id) for node_id in node_ids]
        self._subscription.subscribe_data_change(nodes)
        logger.info("Subscrição OPC-UA criada para %s nó(s)", len(node_ids))
        return self._subscription

    def publish_result(self, result: Any, node_id: str) -> None:
        """Publica um resultado analítico num nó de destino."""
        self.write_node(node_id, result)

    def _ensure_connected(self) -> None:
        if SyncOPCUAClient is None and OPCUA_LIBRARY == "asyncua":
            raise NotImplementedError(
                "A biblioteca 'asyncua' foi detetada, mas este stub ainda não oferece um adaptador síncrono para ela."
            )
        if self._client is None:
            raise RuntimeError("Cliente OPC-UA não está ligado. Chame connect() primeiro.")

    @staticmethod
    def _raise_missing_dependency(message: str) -> None:
        if OPCUA_IMPORT_ERROR is not None:
            raise NotImplementedError(f"{message} Erro original: {OPCUA_IMPORT_ERROR}")
        raise NotImplementedError(message)
