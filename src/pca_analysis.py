from typing import Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(
    df_features: pd.DataFrame, n_components: int = 3, var_threshold: float = 1e-12
) -> Tuple[PCA, pd.DataFrame]:
    """Executa PCA em `df_features` retornando o objeto PCA e o DataFrame de scores.

    - Remove colunas com variância negligível
    - Faz imputação mínima nas colunas restantes
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

    # Imputação física mínima
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].quantile(0.05))

    X_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    pca = PCA(n_components=min(n_components, X.shape[1]))
    Xp = pca.fit_transform(X_scaled)

    cols = [f"PC{i+1}" for i in range(Xp.shape[1])]
    return pca, pd.DataFrame(Xp, columns=cols, index=df_features.index)
