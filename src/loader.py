import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_eis_file(path: str) -> pd.DataFrame:
    """Lê um arquivo EIS e normaliza colunas esperadas.

    Tenta diversos separadores, converte vírgulas decimais e valida
    que temos ao menos 3 colunas com dados.
    """
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

    return df
