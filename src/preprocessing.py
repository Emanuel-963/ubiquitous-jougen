from typing import cast

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Valida dados EIS básicos e computa omega.

    - Garante que colunas essenciais existam
    - Remove linhas com NaN apenas nas colunas necessárias
    - Ordena por frequência (desc) e calcula omega
    """
    required = {"frequency", "zreal", "zimag"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Colunas faltando: {missing}")

    # Remover linhas que tenham NaN nas colunas essenciais
    df = df.dropna(subset=list(required)).copy()

    # Forçar tipos numéricos, caso venham como string
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")

    # Filtrar frequências inválidas
    df = df[df["frequency"] > 0].copy()
    df = df.sort_values("frequency", ascending=False)

    df["omega"] = 2 * np.pi * df["frequency"]
    return cast(pd.DataFrame, df)
