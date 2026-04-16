import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def apply_classification(df: pd.DataFrame, safe: bool = True) -> pd.DataFrame:
    """Classify samples based on Rs and Rp fitted values.

    Uses K-Means (2 clusters) on standardised ``(Rs_fit, Rp_fit)``
    when enough data is available.  Falls back to a robust
    quartile-based heuristic otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``Rs_fit`` and ``Rp_fit``
        columns from circuit fitting.
    safe : bool, default True
        Reserved for future use (safe-mode toggle).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``Subclass`` column.
    """

    df = df.copy()

    if not {"Rs_fit", "Rp_fit"}.issubset(df.columns):
        df["Subclass"] = "Indefinida (sem ajuste físico)"
        return df

    subset = df[["Rs_fit", "Rp_fit"]].dropna()
    if subset.shape[0] >= 3 and subset.var().sum() > 1e-12:
        # Clustering em dados padronizados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(subset)

        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        # Determina qual cluster representa melhor desempenho (menor Rs, maior Rp)
        cluster_stats = (
            subset.assign(cluster=labels)
            .groupby("cluster")
            .agg({"Rs_fit": "median", "Rp_fit": "median"})
        )
        best_cluster = cluster_stats.assign(score=lambda d: -d["Rs_fit"] + d["Rp_fit"]).idxmax()["score"]

        label_map = {
            best_cluster: "Interface eficiente",
            1 - best_cluster: "Genérica estável",
        }

        df["Subclass"] = "Indefinida (dados insuficientes)"
        df.loc[subset.index, "Subclass"] = [label_map[lbl] for lbl in labels]
        return df

    # Fallback baseado em quartis (robusto a outliers)
    rs = df["Rs_fit"]
    rp = df["Rp_fit"]

    if rs.dropna().empty or rp.dropna().empty:
        df["Subclass"] = "Indefinida (dados insuficientes)"
        return df

    rs_q75 = rs.quantile(0.75)
    rp_q25 = rp.quantile(0.25)

    def classify(row):
        """Classify a single sample by quartile thresholds.

        Parameters
        ----------
        row : pd.Series
            A row containing ``Rs_fit`` and ``Rp_fit`` values.

        Returns
        -------
        str
            Classification label: ``'Interface eficiente'``,
            ``'Genérica estável'`` or ``'Indefinida (fit falhou)'``.
        """
        if np.isnan(row["Rs_fit"]) or np.isnan(row["Rp_fit"]):
            return "Indefinida (fit falhou)"
        if row["Rs_fit"] < rs_q75 and row["Rp_fit"] > rp_q25:
            return "Interface eficiente"
        return "Genérica estável"

    df["Subclass"] = df.apply(classify, axis=1)
    return df


def _compute_composite_score(df: pd.DataFrame) -> pd.Series:
    """Compute a weighted composite score from key metrics.

    Default weights: ``Rp_fit`` (+0.35), ``Rs_fit`` (−0.25),
    ``C_mean`` (+0.25), ``Energy_mean`` (+0.15).  Each metric is
    z-score normalised before weighting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with optional columns ``Rp_fit``, ``Rs_fit``,
        ``C_mean`` and ``Energy_mean``.

    Returns
    -------
    pd.Series
        Composite score aligned with *df*'s index.  ``NaN`` when
        no metric columns are available.
    """

    weights = {
        "Rp_fit": 0.35,
        "Rs_fit": -0.25,  # sinal negativo já aplicado aqui
        "C_mean": 0.25,
        "Energy_mean": 0.15,
    }

    available = [col for col in weights if col in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)

    zcols = {}
    for col in available:
        series = df[col]
        if series.dropna().std() == 0 or series.dropna().empty:
            zcols[col] = pd.Series(0.0, index=df.index)
        else:
            zcols[col] = (series - series.mean()) / series.std()

    score = pd.Series(0.0, index=df.index)
    for col in available:
        score = score + weights[col] * zcols[col]
    return score


def rank_within_subclass(df: pd.DataFrame) -> pd.DataFrame:
    """Rank samples within each subclass by electrochemical performance.

    Uses the composite score (high Rp, low Rs, high C_mean,
    high Energy_mean).  Falls back to ``Rp_fit`` ranking when the
    composite score is entirely ``NaN``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``Subclass`` column (from
        :func:`apply_classification`) and metric columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with added ``Score`` and ``Rank`` columns.
    """

    df = df.copy()

    score = _compute_composite_score(df)
    df["Score"] = score

    # Se score inteiro for NaN, tentar fallback por Rp
    if score.isna().all():
        if "Rp_fit" not in df.columns:
            df["Rank"] = np.nan
            return df
        df["Rank"] = df.groupby("Subclass")["Rp_fit"].rank(ascending=False, method="dense")
        return df

    df["Rank"] = df.groupby("Subclass")["Score"].rank(ascending=False, method="dense")
    return df
