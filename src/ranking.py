import numpy as np
import pandas as pd


def apply_classification(df: pd.DataFrame, safe: bool = True) -> pd.DataFrame:
    """
    Classificação física robusta baseada em Rs e Rp (quando disponíveis).

    safe=True:
        - Não quebra se colunas não existirem
        - Classifica como 'Indefinida (dados insuficientes)'
    """

    df = df.copy()

    if not {"Rs_fit", "Rp_fit"}.issubset(df.columns):
        df["Subclass"] = "Indefinida (sem ajuste físico)"
        return df

    if safe:
        rs = df["Rs_fit"]
        rp = df["Rp_fit"]

        if rs.dropna().empty or rp.dropna().empty:
            df["Subclass"] = "Indefinida (dados insuficientes)"
            return df

        rs_q75 = rs.quantile(0.75)
        rp_q25 = rp.quantile(0.25)

    else:
        rs_q75 = df["Rs_fit"].quantile(0.75)
        rp_q25 = df["Rp_fit"].quantile(0.25)

    def classify(row):
        if np.isnan(row["Rs_fit"]) or np.isnan(row["Rp_fit"]):
            return "Indefinida (fit falhou)"

        if row["Rs_fit"] < rs_q75 and row["Rp_fit"] > rp_q25:
            return "Interface eficiente"

        return "Genérica estável"

    df["Subclass"] = df.apply(classify, axis=1)
    return df


def rank_within_subclass(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranking interno por desempenho eletroquímico.
    """

    df = df.copy()

    if "Rp_fit" not in df.columns:
        df["Rank"] = np.nan
        return df

    df["Rank"] = df.groupby("Subclass")["Rp_fit"].rank(ascending=False, method="dense")

    return df
