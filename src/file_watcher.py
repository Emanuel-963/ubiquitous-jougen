"""Plugin de monitoramento de arquivos para ingestão automática de EIS.

Este módulo fornece a classe :class:`FileWatcher`, responsável por observar
um diretório e disparar um *callback* quando novos arquivos suportados forem
criados ou movidos para a pasta monitorada.

A implementação prioriza o uso da biblioteca ``watchdog`` quando disponível,
mas permanece importável e funcional sem dependências opcionais, utilizando um
modo de *polling* em segundo plano como alternativa.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # pragma: no cover - dependência opcional
    FileSystemEvent = object  # type: ignore[assignment]
    FileSystemEventHandler = object  # type: ignore[assignment]
    Observer = None

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".dta", ".mpr", ".idf", ".isc"}


class _WatchdogEventHandler(FileSystemEventHandler):
    """Encaminha eventos do watchdog para o observador principal."""

    def __init__(self, watcher: "FileWatcher") -> None:
        self._watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        self._watcher._handle_event_path(getattr(event, "src_path", ""))

    def on_moved(self, event: FileSystemEvent) -> None:
        self._watcher._handle_event_path(getattr(event, "dest_path", ""))


class FileWatcher:
    """Monitora um diretório em busca de novos arquivos de EIS.

    Parameters
    ----------
    directory : str | Path
        Diretório a ser observado.
    callback : Callable[[Path], None]
        Função chamada sempre que um novo arquivo suportado é detectado.
        O caminho absoluto do arquivo é passado como argumento.
    polling_interval : float, optional
        Intervalo, em segundos, entre verificações no modo de *polling*.
    recursive : bool, optional
        Define se subdiretórios também devem ser monitorados.
    """

    def __init__(
        self,
        directory: str | Path,
        callback: Callable[[Path], None],
        *,
        polling_interval: float = 1.0,
        recursive: bool = False,
    ) -> None:
        self._directory = Path(directory).expanduser().resolve()
        self._callback = callback
        self._polling_interval = max(float(polling_interval), 0.1)
        self._recursive = recursive
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer: Observer | None = None
        self._seen_files: set[Path] = set()
        self._running = False

    @property
    def is_running(self) -> bool:
        """Indica, de forma *thread-safe*, se o monitor está ativo."""
        with self._lock:
            return self._running

    def start(self) -> None:
        """Inicia o monitoramento do diretório configurado."""
        with self._lock:
            if self._running:
                logger.debug("FileWatcher já está em execução para %s", self._directory)
                return

            self._directory.mkdir(parents=True, exist_ok=True)
            self._seen_files = self._scan_supported_files()
            self._stop_event.clear()

            if Observer is not None:
                handler = _WatchdogEventHandler(self)
                observer = Observer()
                observer.schedule(handler, str(self._directory), recursive=self._recursive)
                observer.start()
                self._observer = observer
                self._thread = None
                logger.info("FileWatcher iniciado com watchdog em %s", self._directory)
            else:
                self._observer = None
                self._thread = threading.Thread(
                    target=self._polling_loop,
                    name=f"FileWatcher[{self._directory.name}]",
                    daemon=True,
                )
                self._thread.start()
                logger.info(
                    "Watchdog indisponível; FileWatcher iniciou em modo polling para %s",
                    self._directory,
                )

            self._running = True

    def stop(self) -> None:
        """Interrompe o monitoramento e aguarda o encerramento das threads."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
            observer = self._observer
            thread = self._thread
            self._observer = None
            self._thread = None

        if observer is not None:
            observer.stop()
            observer.join(timeout=max(self._polling_interval * 2, 2.0))
            logger.info("FileWatcher com watchdog finalizado para %s", self._directory)

        if thread is not None:
            thread.join(timeout=max(self._polling_interval * 2, 2.0))
            logger.info("FileWatcher em modo polling finalizado para %s", self._directory)

    def _polling_loop(self) -> None:
        """Executa varreduras periódicas quando watchdog não está disponível."""
        while not self._stop_event.wait(self._polling_interval):
            try:
                current_files = self._scan_supported_files()
                new_files = sorted(current_files - self._seen_files)
                self._seen_files = current_files
                for path in new_files:
                    self._dispatch_callback(path)
            except Exception as exc:  # pragma: no cover - logging defensivo
                logger.exception("Falha no polling de arquivos em %s: %s", self._directory, exc)

    def _handle_event_path(self, raw_path: str) -> None:
        """Processa um caminho recebido do watchdog."""
        if not raw_path:
            return

        path = Path(raw_path).expanduser().resolve()
        if not self._is_supported_file(path):
            return

        with self._lock:
            if path in self._seen_files:
                return
            self._seen_files.add(path)

        logger.debug("Novo arquivo detectado via watchdog: %s", path)
        self._dispatch_callback(path)

    def _scan_supported_files(self) -> set[Path]:
        """Lista arquivos suportados existentes no diretório monitorado."""
        pattern = "**/*" if self._recursive else "*"
        return {
            path.resolve()
            for path in self._directory.glob(pattern)
            if self._is_supported_file(path)
        }

    def _is_supported_file(self, path: Path) -> bool:
        """Verifica se o caminho aponta para um arquivo suportado."""
        return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS

    def _dispatch_callback(self, path: Path) -> None:
        """Executa o *callback* associado ao arquivo detectado."""
        try:
            logger.info("Arquivo EIS detectado: %s", path)
            self._callback(path)
        except Exception as exc:  # pragma: no cover - callback externo
            logger.exception("Erro ao executar callback para %s: %s", path, exc)
