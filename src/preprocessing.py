from typing import cast

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Validate basic EIS data and compute angular frequency.

    Ensures that required columns exist, removes rows with NaN in
    essential columns, filters non-positive frequencies, sorts by
    frequency (descending) and adds an ``omega`` column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw EIS DataFrame. Must contain columns ``frequency``,
        ``zreal`` and ``zimag``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame sorted by frequency with an additional
        ``omega`` (2πf) column.

    Raises
    ------
    ValueError
        If any of the required columns are missing.
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
