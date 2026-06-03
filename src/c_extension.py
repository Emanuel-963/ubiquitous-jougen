"""Wrapper ctypes para aceleração nativa opcional do IonFlow.

Este módulo encapsula o carregamento de uma biblioteca compartilhada escrita em
C/C++ para acelerar rotinas de *fitting* de circuitos equivalentes e cálculo de
DRT. A implementação foi desenhada para ser segura em ambientes onde a
biblioteca nativa ainda não existe: nessas situações, o código cai
automaticamente para a implementação Python já presente no projeto.

Interface C esperada
--------------------
A biblioteca compartilhada deve exportar, idealmente, as funções abaixo::

    int fit_impedance(
        const double* freq,
        const double* zreal,
        const double* zimag,
        size_t n_points,
        const char* circuit_name,
        double* out_params,
        size_t max_params,
        double* out_chi2,
        char* error_buffer,
        size_t error_buffer_size
    );

    int compute_drt(
        const double* freq,
        const double* zreal,
        const double* zimag,
        size_t n_points,
        double lambda_reg,
        size_t n_taus,
        double* out_tau,
        double* out_gamma,
        double* out_r_inf,
        char* error_buffer,
        size_t error_buffer_size
    );

Convenções esperadas
--------------------
* Código de retorno ``0`` indica sucesso.
* Qualquer valor diferente de ``0`` indica erro.
* ``error_buffer`` recebe uma mensagem textual UTF-8 quando houver falha.
* ``out_params`` deve conter os parâmetros do circuito na mesma ordem definida
  pelo catálogo Python.
* ``out_tau`` e ``out_gamma`` devem ser preenchidos com ``n_taus`` pontos.

Caso a biblioteca adote outra ABI, basta ajustar os ``argtypes`` e o contrato
neste módulo.
"""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:  # pragma: no cover - dependência obrigatória do projeto
    np = None  # type: ignore[assignment]

try:
    from src.circuit_fitting import circuit_catalog, fit_template
except Exception as exc:  # pragma: no cover - guarda de dependência opcional
    circuit_catalog = None  # type: ignore[assignment]
    fit_template = None  # type: ignore[assignment]
    _FITTING_IMPORT_ERROR: Exception | None = exc
else:
    _FITTING_IMPORT_ERROR = None

try:
    from src.drt_analysis import compute_drt
except Exception as exc:  # pragma: no cover - guarda de dependência opcional
    compute_drt = None  # type: ignore[assignment]
    _DRT_IMPORT_ERROR: Exception | None = exc
else:
    _DRT_IMPORT_ERROR = None


@dataclass(frozen=True)
class NativeLibrarySpec:
    """Metadados simples sobre a biblioteca nativa procurada."""

    path: Path
    loaded: bool = False


class NativeAccelerator:
    """Wrapper de alto nível para funções nativas acessadas via ``ctypes``.

    A classe procura automaticamente uma biblioteca compartilhada compatível
    (``.so``, ``.dll`` ou ``.dylib``). Se nada for encontrado, os métodos
    públicos continuam funcionando ao delegar para as implementações Python.
    """

    _DEFAULT_LIBRARY_NAMES = (
        "libionflow_accel.so",
        "ionflow_accel.so",
        "ionflow_accel.dll",
        "libionflow_accel.dylib",
    )

    def __init__(
        self,
        library_path: str | Path | None = None,
        *,
        n_taus_default: int = 64,
    ) -> None:
        self._library_path = Path(library_path).expanduser() if library_path else None
        self._n_taus_default = n_taus_default
        self._library: Any | None = None
        self._load_error: Exception | None = None
        self._resolved_spec = self._discover_library()

    def is_available(self) -> bool:
        """Retorna ``True`` quando uma biblioteca nativa válida está acessível."""
        if self._resolved_spec is None:
            return False
        if self._library is not None:
            return True
        return self._load_library() is not None

    def fit_impedance_native(
        self,
        freq: Any,
        zreal: Any,
        zimag: Any,
        circuit: str,
    ) -> dict[str, Any]:
        """Executa *fitting* por backend nativo, com fallback em Python.

        Returns
        -------
        dict[str, Any]
            Estrutura compatível com o resultado de ``src.circuit_fitting``.
        """
        freq_arr, zreal_arr, zimag_arr = self._coerce_inputs(freq, zreal, zimag)
        library = self._load_library()
        if library is None:
            logger.info("Biblioteca nativa indisponível; usando fitting Python")
            result = self._fit_impedance_python(freq_arr, zreal_arr, zimag_arr, circuit)
            result.setdefault("backend", "python")
            return result

        try:
            return self._fit_impedance_ctypes(
                library,
                freq_arr,
                zreal_arr,
                zimag_arr,
                circuit,
            )
        except Exception as exc:
            logger.warning("Fitting nativo falhou (%s); usando fallback Python", exc)
            result = self._fit_impedance_python(freq_arr, zreal_arr, zimag_arr, circuit)
            result.setdefault("backend", "python")
            result.setdefault("native_error", str(exc))
            return result

    def compute_drt_native(
        self,
        freq: Any,
        zreal: Any,
        zimag: Any,
        lambda_reg: float,
    ) -> dict[str, Any]:
        """Calcula DRT via backend nativo, com fallback em Python."""
        freq_arr, zreal_arr, zimag_arr = self._coerce_inputs(freq, zreal, zimag)
        library = self._load_library()
        if library is None:
            logger.info("Biblioteca nativa indisponível; usando DRT Python")
            result = self._compute_drt_python(freq_arr, zreal_arr, zimag_arr, lambda_reg)
            result.setdefault("backend", "python")
            return result

        try:
            return self._compute_drt_ctypes(
                library,
                freq_arr,
                zreal_arr,
                zimag_arr,
                lambda_reg,
            )
        except Exception as exc:
            logger.warning("DRT nativo falhou (%s); usando fallback Python", exc)
            result = self._compute_drt_python(freq_arr, zreal_arr, zimag_arr, lambda_reg)
            result.setdefault("backend", "python")
            result.setdefault("native_error", str(exc))
            return result

    def _discover_library(self) -> NativeLibrarySpec | None:
        """Localiza a biblioteca nativa usando variável de ambiente e caminhos padrão."""
        candidates: list[Path] = []
        env_path = os.environ.get("IONFLOW_NATIVE_LIB")
        if env_path:
            candidates.append(Path(env_path).expanduser())
        if self._library_path is not None:
            candidates.append(self._library_path)

        base_dirs = [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent / "native"]
        for base_dir in base_dirs:
            for name in self._DEFAULT_LIBRARY_NAMES:
                candidates.append(base_dir / name)

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.exists() and candidate.is_file():
                logger.info("Biblioteca nativa encontrada em %s", candidate)
                return NativeLibrarySpec(path=candidate, loaded=False)

        logger.info("Nenhuma biblioteca nativa encontrada para NativeAccelerator")
        return None

    def _load_library(self) -> Any | None:
        """Carrega a biblioteca compartilhada de forma preguiçosa."""
        if self._library is not None:
            return self._library
        if self._resolved_spec is None:
            return None

        try:
            self._library = ctypes.CDLL(str(self._resolved_spec.path))
            self._configure_function_signatures(self._library)
            logger.info("Biblioteca nativa carregada: %s", self._resolved_spec.path)
        except Exception as exc:
            self._load_error = exc
            self._library = None
            logger.warning("Não foi possível carregar biblioteca nativa: %s", exc)
        return self._library

    def _configure_function_signatures(self, library: Any) -> None:
        """Define ``argtypes`` e ``restype`` do contrato esperado da ABI C."""
        double_ptr = ctypes.POINTER(ctypes.c_double)
        size_t = ctypes.c_size_t
        char_ptr = ctypes.c_char_p

        fit_func = getattr(library, "fit_impedance", None)
        if fit_func is not None:
            fit_func.argtypes = [
                double_ptr,
                double_ptr,
                double_ptr,
                size_t,
                char_ptr,
                double_ptr,
                size_t,
                double_ptr,
                ctypes.POINTER(ctypes.c_char),
                size_t,
            ]
            fit_func.restype = ctypes.c_int

        drt_func = getattr(library, "compute_drt", None)
        if drt_func is not None:
            drt_func.argtypes = [
                double_ptr,
                double_ptr,
                double_ptr,
                size_t,
                ctypes.c_double,
                size_t,
                double_ptr,
                double_ptr,
                double_ptr,
                ctypes.POINTER(ctypes.c_char),
                size_t,
            ]
            drt_func.restype = ctypes.c_int

    def _fit_impedance_ctypes(
        self,
        library: Any,
        freq: "np.ndarray",
        zreal: "np.ndarray",
        zimag: "np.ndarray",
        circuit: str,
    ) -> dict[str, Any]:
        """Executa o fitting usando a ABI C documentada no módulo."""
        fit_func = getattr(library, "fit_impedance", None)
        if fit_func is None:
            raise AttributeError("A biblioteca nativa não exporta fit_impedance")

        template = self._get_template(circuit)
        n_params = len(template.param_names)
        out_params = np.zeros(n_params, dtype=np.float64)
        out_chi2 = ctypes.c_double(0.0)
        error_buffer = ctypes.create_string_buffer(512)

        status = fit_func(
            self._as_double_ptr(freq),
            self._as_double_ptr(zreal),
            self._as_double_ptr(zimag),
            freq.size,
            circuit.encode("utf-8"),
            self._as_double_ptr(out_params),
            n_params,
            ctypes.byref(out_chi2),
            error_buffer,
            ctypes.sizeof(error_buffer),
        )
        if status != 0:
            message = error_buffer.value.decode("utf-8", errors="replace") or "erro nativo"
            raise RuntimeError(message)

        params = {name: float(value) for name, value in zip(template.param_names, out_params)}
        return {
            "template": template.name,
            "diagram": template.diagram,
            "params": params,
            "chi2_over_nu": float(out_chi2.value),
            "success": True,
            "message": "native-ok",
            "backend": "native",
            "native_library": str(self._resolved_spec.path) if self._resolved_spec else "",
        }

    def _compute_drt_ctypes(
        self,
        library: Any,
        freq: "np.ndarray",
        zreal: "np.ndarray",
        zimag: "np.ndarray",
        lambda_reg: float,
    ) -> dict[str, Any]:
        """Executa DRT usando a ABI C documentada no módulo."""
        drt_func = getattr(library, "compute_drt", None)
        if drt_func is None:
            raise AttributeError("A biblioteca nativa não exporta compute_drt")

        n_taus = self._n_taus_default
        out_tau = np.zeros(n_taus, dtype=np.float64)
        out_gamma = np.zeros(n_taus, dtype=np.float64)
        out_r_inf = ctypes.c_double(0.0)
        error_buffer = ctypes.create_string_buffer(512)

        status = drt_func(
            self._as_double_ptr(freq),
            self._as_double_ptr(zreal),
            self._as_double_ptr(zimag),
            freq.size,
            float(lambda_reg),
            n_taus,
            self._as_double_ptr(out_tau),
            self._as_double_ptr(out_gamma),
            ctypes.byref(out_r_inf),
            error_buffer,
            ctypes.sizeof(error_buffer),
        )
        if status != 0:
            message = error_buffer.value.decode("utf-8", errors="replace") or "erro nativo"
            raise RuntimeError(message)

        return {
            "tau": out_tau,
            "gamma": out_gamma,
            "r_inf": float(out_r_inf.value),
            "peaks": [],
            "residuals": np.zeros(freq.size, dtype=np.float64),
            "lambda_reg": float(lambda_reg),
            "n_taus": n_taus,
            "backend": "native",
            "native_library": str(self._resolved_spec.path) if self._resolved_spec else "",
        }

    def _fit_impedance_python(
        self,
        freq: "np.ndarray",
        zreal: "np.ndarray",
        zimag: "np.ndarray",
        circuit: str,
    ) -> dict[str, Any]:
        """Fallback Python para fitting quando a DLL/SO não está disponível."""
        if fit_template is None or circuit_catalog is None:
            raise RuntimeError(
                "Fallback Python de fitting indisponível"
            ) from _FITTING_IMPORT_ERROR

        template = self._get_template(circuit)
        z = zreal + 1j * zimag
        result = fit_template(template, freq, z)
        result["backend"] = "python"
        return result

    def _compute_drt_python(
        self,
        freq: "np.ndarray",
        zreal: "np.ndarray",
        zimag: "np.ndarray",
        lambda_reg: float,
    ) -> dict[str, Any]:
        """Fallback Python para DRT quando a DLL/SO não está disponível."""
        if compute_drt is None:
            raise RuntimeError("Fallback Python de DRT indisponível") from _DRT_IMPORT_ERROR
        result = compute_drt(
            freq,
            zreal,
            zimag,
            n_taus=self._n_taus_default,
            lambda_reg=lambda_reg,
        )
        result["backend"] = "python"
        return result

    def _get_template(self, circuit: str) -> Any:
        """Obtém o template de circuito pelo nome usado no catálogo Python."""
        if circuit_catalog is None:
            raise RuntimeError("Catálogo de circuitos indisponível") from _FITTING_IMPORT_ERROR
        for template in circuit_catalog():
            if template.name == circuit:
                return template
        available = ", ".join(template.name for template in circuit_catalog())
        raise ValueError(f"Circuito desconhecido: {circuit}. Disponíveis: {available}")

    @staticmethod
    def _coerce_inputs(freq: Any, zreal: Any, zimag: Any) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Converte entradas para ``numpy.ndarray`` e valida dimensões básicas."""
        if np is None:
            raise RuntimeError("numpy é necessário para NativeAccelerator")
        freq_arr = np.ascontiguousarray(freq, dtype=np.float64)
        zreal_arr = np.ascontiguousarray(zreal, dtype=np.float64)
        zimag_arr = np.ascontiguousarray(zimag, dtype=np.float64)
        if not (freq_arr.shape == zreal_arr.shape == zimag_arr.shape):
            raise ValueError("freq, zreal e zimag devem possuir o mesmo shape")
        if freq_arr.ndim != 1:
            raise ValueError("freq, zreal e zimag devem ser vetores 1D")
        return freq_arr, zreal_arr, zimag_arr

    @staticmethod
    def _as_double_ptr(array: "np.ndarray") -> Any:
        """Retorna um ponteiro C para um vetor ``float64`` contíguo."""
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


C_API_STUB = """
/* Stub documental da ABI esperada pela classe NativeAccelerator. */
int fit_impedance(
    const double* freq,
    const double* zreal,
    const double* zimag,
    size_t n_points,
    const char* circuit_name,
    double* out_params,
    size_t max_params,
    double* out_chi2,
    char* error_buffer,
    size_t error_buffer_size
);

int compute_drt(
    const double* freq,
    const double* zreal,
    const double* zimag,
    size_t n_points,
    double lambda_reg,
    size_t n_taus,
    double* out_tau,
    double* out_gamma,
    double* out_r_inf,
    char* error_buffer,
    size_t error_buffer_size
);
"""
