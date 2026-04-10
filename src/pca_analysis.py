from typing import Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(
    df_features: pd.DataFrame, n_components: int = 3, var_threshold: float = 1e-12
) -> Tuple[PCA, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Executa PCA retornando modelo, scores, loadings e variância explicada.

    - Remove colunas com variância negligível
    - Imputa NaNs com quantil 5% (conservador)
    - Retorna também loadings para interpretação
    """

    X = df_features.copy()

    # Remove colunas com variância ~ zero
    variances = X.var()
    X = X.loc[:, variances > var_threshold]

    if X.shape[1] < 2:
        raise ValueError(
            "Variância insuficiente para PCA: "
            "todas as métricas são quase constantes."
        )

    # Imputação mínima
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].quantile(0.05))

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, X.shape[1]))
    Xp = pca.fit_transform(X_scaled)

    cols = [f"PC{i+1}" for i in range(Xp.shape[1])]
    scores = pd.DataFrame(Xp, columns=cols, index=df_features.index)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=cols,
    )

    evr = pd.Series(pca.explained_variance_ratio_, index=cols)

    return pca, scores, loadings, evr
