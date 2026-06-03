"""EIS file loader with auto-detection of encoding and delimiter.

Handles CSV/TXT files from various potentiostats with automatic
separator sniffing, encoding detection (UTF-8, Latin-1, CP1252),
and column name normalisation.
"""

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ── VAL-03: Encoding & separator auto-detection ─────────────────────────

_ENCODINGS_TO_TRY: List[str] = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
"""Ordered list of encodings to attempt when reading EIS text files."""


# ── VAL-02: Researcher-friendly error class ─────────────────────────────

class EISLoadError(ValueError):
    """User-friendly error raised when an EIS file cannot be loaded.

    Attributes
    ----------
    path : str or None
        Path to the file that failed to load.
    detected_encoding : str or None
        Encoding that was detected for the file.
    detected_separator : str or None
        Separator that was detected (or attempted).
    columns_found : int or None
        Number of columns found in the file.
    suggestion : str
        A human-readable suggestion for how to fix the issue.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Optional[str] = None,
        detected_encoding: Optional[str] = None,
        detected_separator: Optional[str] = None,
        columns_found: Optional[int] = None,
    ):
        super().__init__(message)
        self.path = path
        self.detected_encoding = detected_encoding
        self.detected_separator = detected_separator
        self.columns_found = columns_found


def _detect_encoding(path: str) -> str:
    """Detect the most likely text encoding for *path*.

    Tries common scientific-data encodings in order and returns the
    first one that can decode the file without errors.

    Parameters
    ----------
    path : str
        Path to the text file to inspect.

    Returns
    -------
    str
        Encoding name (e.g. ``'utf-8'``, ``'latin-1'``).
    """
    for enc in _ENCODINGS_TO_TRY:
        try:
            with open(path, encoding=enc, errors="strict") as fh:
                fh.read(8192)  # read first 8 KB to validate
            return enc
        except (UnicodeDecodeError, ValueError):
            continue
    # Ultimate fallback — will replace bad chars
    return "latin-1"


def _sniff_delimiter(path: str, encoding: str) -> Optional[str]:
    """Use csv.Sniffer to detect the column separator.

    Falls back to None (which tells pandas to use its own detection).

    Parameters
    ----------
    path : str
        Path to the text file.
    encoding : str
        Encoding to use when opening the file.

    Returns
    -------
    str or None
        Detected delimiter character, or None if sniffing fails.
    """
    try:
        with open(path, encoding=encoding, errors="replace") as fh:
            # Skip comment lines (common in EIS files)
            lines = []
            for line in fh:
                if not line.startswith("#") and line.strip():
                    lines.append(line)
                if len(lines) >= 10:
                    break
            if not lines:
                return None
            sample = "\n".join(lines)
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t |")
            return dialect.delimiter
    except (csv.Error, OSError):
        return None

# Supported EIS file extensions — used by every pipeline and batch processor
# to skip non-EIS files (images, spreadsheets, docs…) in mixed-content folders.
EIS_EXTENSIONS: frozenset = frozenset({
    ".csv", ".txt", ".dat", ".asc",
    ".mpt", ".mpr",            # BioLogic
    ".dta",                    # Gamry
    ".idf", ".z", ".dfr",     # Solartron / generic
    ".ism", ".isc",            # Zahner
})

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

    For potentiostat-specific extensions (.dta, .mpr, .mpt, .ism, .isc,
    .idf, .dfr) the specialised parser from ``src.parsers`` is tried first
    so that binary and vendor-specific text formats are handled correctly.
    """
    # OPT-02: return cached copy when file has not changed
    cached = _cache_get(path)
    if cached is not None:
        logger.debug("load_eis_file: cache hit for %s", path)
        return cached

    # ── Delegate to specialised parsers for vendor formats ────────────
    import pathlib as _pl
    _SPECIALIZED_EXTS = frozenset({".dta", ".mpr", ".mpt", ".ism", ".isc", ".idf", ".dfr"})
    if _pl.Path(path).suffix.lower() in _SPECIALIZED_EXTS:
        try:
            from src.parsers import detect_parser, GenericCSVParser
            parser_cls = detect_parser(path)
            if parser_cls is not None and parser_cls is not GenericCSVParser:
                result = parser_cls().parse(path)
                required = {"frequency", "zreal", "zimag"}
                if required.issubset(result.data.columns):
                    df_out = result.data[list(required)].copy().dropna()
                    if len(df_out) > 0:
                        _cache_put(path, df_out)
                        return df_out
        except Exception as exc:
            logger.warning(
                "load_eis_file: specialised parser failed for %s: %s — falling back to CSV",
                path, exc,
            )

    # ── VAL-03: Auto-detect encoding and separator ─────────────────────
    encoding = _detect_encoding(path)
    sniffed_sep = _sniff_delimiter(path, encoding)

    # Build prioritised separator list: sniffed delimiter first, then fallbacks
    separators: List[Optional[str]] = []
    if sniffed_sep:
        separators.append(sniffed_sep)
    for fallback in [";", "\t", ",", None]:
        if fallback not in separators:
            separators.append(fallback)

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
                encoding=encoding,
            )
            if df.shape[1] >= 3:
                break
        except Exception as e:
            logger.debug("Falha ao ler %s com sep=%s: %s", path, sep, e)
            continue

    # ── VAL-02: Researcher-friendly error messages ───────────────────
    if df is None or df.shape[1] < 3:
        cols = None if df is None else df.shape[1]
        fname = os.path.basename(path)
        raise EISLoadError(
            f"O arquivo '{fname}' não pôde ser lido corretamente.\n"
            f"  • Colunas encontradas: {cols} (mínimo necessário: 3)\n"
            f"  • Encoding detectado: {encoding}\n"
            f"  • Separador testado: {sniffed_sep or 'auto'}\n\n"
            f"Dica: verifique se o arquivo contém pelo menos 3 colunas "
            f"(frequência, Z' e Z'') separadas por ; ou , ou TAB.",
            path=path,
            detected_encoding=encoding,
            detected_separator=sniffed_sep,
            columns_found=cols,
        )

    # Normalização dos headers
    df.columns = [str(c).lower().strip() for c in df.columns]

    # ── VAL-02: Smart column matching with helpful messages ──────────
    # Known column name aliases from common potentiostat software
    _FREQ_ALIASES = {"freq", "frequency", "f", "freq.", "frequency (hz)", "freq (hz)", "f (hz)", "freq/hz"}
    _ZREAL_ALIASES = {"z'", "zreal", "z_re", "z_real", "z' (ohm)", "z'/ohm", "zre", "re(z)", "re_z"}
    _ZIMAG_ALIASES = {"z''", "zimag", "z_im", "z_imag", "-z''", "-z'' (ohm)", "z''/ohm", "zim", "im(z)", "im_z", "-z\"", "-z'' (ohm)"}

    freq_col = None
    zreal_col = None
    zimag_col = None

    for c in df.columns:
        if freq_col is None and (c in _FREQ_ALIASES or "freq" in c):
            freq_col = c
        elif zreal_col is None and (c in _ZREAL_ALIASES or ("z'" in c and "z''" not in c and "-z" not in c)):
            zreal_col = c
        elif zimag_col is None and (c in _ZIMAG_ALIASES or "z''" in c or ("-z" in c and "imag" not in c)):
            zimag_col = c

    # Fallback por posição se necessário
    if freq_col is None or zreal_col is None or zimag_col is None:
        if df.shape[1] < 3:
            fname = os.path.basename(path)
            raise EISLoadError(
                f"O arquivo '{fname}' não possui colunas suficientes.\n"
                f"  • Colunas encontradas: {list(df.columns)}\n"
                f"  • Esperado: frequency, zreal (Z'), zimag (Z'')\n\n"
                f"Dica: verifique se as colunas estão nomeadas corretamente "
                f"ou se o separador está correto.",
                path=path,
                columns_found=df.shape[1],
            )

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
        logger.info(
            "Colunas não reconhecidas em '%s' — usando posicionais: %s, %s, %s",
            os.path.basename(path), freq_col, zreal_col, zimag_col,
        )

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
        fname = os.path.basename(path)
        raise EISLoadError(
            f"O arquivo '{fname}' foi lido, mas resultou em 0 linhas válidas.\n"
            f"  • Isso pode ocorrer se todos os valores são texto ou NaN.\n\n"
            f"Dica: verifique se o arquivo usa '.' ou ',' como separador "
            f"decimal e se não há linhas de cabeçalho extras.",
            path=path,
        )

    # OPT-02: store in cache before returning
    _cache_put(path, df)
    return df
