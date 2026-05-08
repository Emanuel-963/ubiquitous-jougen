import logging
import os
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# OPT-02: in-process cache to avoid re-reading unchanged files.
# Key = absolute path; value = (mtime_ns, DataFrame copy).
# The cache lives for the duration of the process — safe because EIS files
# in a lab session are written once and never modified mid-run.
_LOAD_CACHE: Dict[str, Tuple[int, pd.DataFrame]] = {}


def _cache_get(path: str) -> Optional[pd.DataFrame]:
    """Return a cached DataFrame if *path* has not changed since last read."""
    try:
        mtime_ns = os.stat(path).st_mtime_ns
        entry = _LOAD_CACHE.get(path)
        if entry is not None and entry[0] == mtime_ns:
            return entry[1].copy()
    except OSError:
        pass
    return None


def _cache_put(path: str, df: pd.DataFrame) -> None:
    """Store *df* in the cache, keyed by *path* + current mtime."""
    try:
        mtime_ns = os.stat(path).st_mtime_ns
        _LOAD_CACHE[path] = (mtime_ns, df.copy())
    except OSError:
        pass


def clear_load_cache() -> None:
    """Evict all cached entries (useful in tests or after batch imports)."""
    _LOAD_CACHE.clear()


def load_eis_file(path: str) -> pd.DataFrame:
    """Lê um arquivo EIS e normaliza colunas esperadas.

    Tenta diversos separadores, converte vírgulas decimais e valida
    que temos ao menos 3 colunas com dados.

    Results are cached in-process by file mtime (OPT-02).  Re-reading the
    same unchanged file is a no-op after the first call.
    """
    # OPT-02: return cached copy when file has not changed
    cached = _cache_get(path)
    if cached is not None:
        logger.debug("load_eis_file: cache hit for %s", path)
        return cached

    # Tentar com diferentes separadores
    separators = [";", "\t", ",", None]
    df: Optional[pd.DataFrame] = None

    for sep in separators:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                comment="#",
                dtype=str,  # Ler tudo como string primeiro
                skipinitialspace=True,
            )
            if df.shape[1] >= 3:
                break
        except Exception as e:
            logger.debug("Falha ao ler %s com sep=%s: %s", path, sep, e)
            continue

    if df is None or df.shape[1] < 3:
        cols = None if df is None else df.shape[1]
        raise ValueError(
            f"Arquivo {path} não pode ser parseado; numero de colunas: {cols}"
        )

    # Normalização dos headers
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Encontrar colunas por padrão
    freq_col = None
    zreal_col = None
    zimag_col = None

    for c in df.columns:
        if freq_col is None and "freq" in c:
            freq_col = c
        elif zreal_col is None and ("z'" in c and "z''" not in c and "-z" not in c):
            zreal_col = c
        elif zimag_col is None and ("z''" in c or ("-z" in c and "imag" not in c)):
            zimag_col = c

    # Fallback por posição se necessário
    if freq_col is None or zreal_col is None or zimag_col is None:
        if df.shape[1] < 3:
            raise ValueError(f"Arquivo {path} não possui colunas suficientes")

        # Se temos mais de 3 colunas, tenta usar as 3 primeiras não-nulas
        valid_cols = [c for c in df.columns if df[c].notna().sum() > 0]
        if len(valid_cols) >= 3:
            freq_col = valid_cols[0]
            zreal_col = valid_cols[1]
            zimag_col = valid_cols[2]
        else:
            freq_col = df.columns[0]
            zreal_col = df.columns[1]
            zimag_col = df.columns[2]

    # Selecionar apenas as colunas necessárias
    df = df[[freq_col, zreal_col, zimag_col]].copy()
    df.columns = ["frequency", "zreal", "zimag"]

    # Conversão numérica robusta com suporte a vírgula como decimal
    for col in ["frequency", "zreal", "zimag"]:
        # Remover espaços e converter vírgula em ponto
        df[col] = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convenção Nyquist: -Z'' (imaginária negativa)
    df["zimag"] = -df["zimag"].abs()

    # Remover linhas com NaN
    df = df.dropna()

    # Garantir que temos dados válidos
    if len(df) == 0:
        raise ValueError(f"Arquivo {path} resultou em 0 linhas de dados válidos")

    # OPT-02: store in cache before returning
    _cache_put(path, df)
    return df
