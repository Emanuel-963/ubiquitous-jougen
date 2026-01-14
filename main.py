import logging
import os

import numpy as np
import pandas as pd

from src.cpe_fit import fit_cpe_warburg
from src.loader import load_eis_file
from src.metadata import extract_metadata
from src.pca_analysis import run_pca
from src.physics_metrics import extract_features
from src.preprocessing import preprocess
from src.ranking import apply_classification, rank_within_subclass
from src.stability import extract_sample_id, stability_metrics
from src.visualization import pca_2d, pca_3d

# Configurar logging no início da execução (mas manter imports no topo)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

# ===============================
# CONFIG
# ===============================

DATA_DIR = "data/raw"
OUT_DIR = "outputs/tables"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# EXTRAÇÃO DE FEATURES
# ===============================

records = {}

for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)

    try:
        df = preprocess(load_eis_file(path))

        feat = extract_features(df)

        # ---- Fit CPE + Warburg (tolerante a falha)
        try:
            fit = fit_cpe_warburg(df)
        except Exception as e:
            print(f"Fit falhou em {file}: {e}")
            fit = {
                "Rs_fit": np.nan,
                "Rp_fit": np.nan,
                "Q": np.nan,
                "n": np.nan,
                "Sigma": np.nan,
            }

        feat.update(fit)
        records[file] = feat

    except Exception as e:
        print(f"Erro ao processar {file}: {e}")

# ===============================
# DATAFRAME GLOBAL
# ===============================

df = pd.DataFrame(records).T

# ===============================
# METADADOS
# ===============================

df["Electrolyte"], df["Current"], df["Treatment"] = zip(
    *[extract_metadata(name) for name in df.index]
)

df["Sample"] = [extract_sample_id(name) for name in df.index]

# ===============================
# CLASSIFICAÇÃO (ROBUSTA)
# ===============================

df = apply_classification(df, safe=True)

# ===============================
# ESTABILIDADE (CAMINHO B)
# ===============================

for col in ["Rs_fit", "Rp_fit", "Q", "n"]:
    if col in df.columns:
        stab = stability_metrics(df, col)
        stab.to_csv(f"{OUT_DIR}/stability_{col}.csv")

# ===============================
# PCA (CONDICIONAL)
# ===============================

pca_cols = ["Rs_fit", "Rp_fit", "Q", "n", "Sigma"]
valid_cols = [c for c in pca_cols if c in df.columns]

df_pca = None

X = df[valid_cols].dropna()

if X.shape[0] >= 3 and X.var().sum() > 1e-12:
    try:
        pca, df_pca = run_pca(X)
        print("Variância explicada:", pca.explained_variance_ratio_)
        df_pca.to_csv(f"{OUT_DIR}/pca_scores.csv")

        # Gerar gráficos de PCA 2D e 3D
        fig_2d = pca_2d(
            df_pca,
            df.loc[df_pca.index, "Subclass"],
            title="Análise PCA 2D - Amostras por Tipo",
        )
        print(f"Gráfico PCA 2D salvo em: {fig_2d}")

        fig_3d = pca_3d(
            df_pca,
            df.loc[df_pca.index, "Subclass"],
            title="Análise PCA 3D - Amostras por Tipo",
        )
        if fig_3d:
            print(f"Gráfico PCA 3D salvo em: {fig_3d}")

    except Exception as e:
        print("PCA ignorado:", e)
else:
    print("PCA não informativo: variância insuficiente ou poucos dados válidos.")

# ===============================
# RANKING
# ===============================

df_ranked = rank_within_subclass(df)

# ===============================
# OUTPUTS
# ===============================

df.to_csv(f"{OUT_DIR}/summary_features.csv")
df_ranked.to_csv(f"{OUT_DIR}/ranked_results.csv")

print("Análise concluída.")
print("\n" + "=" * 80)
print("RESUMO DAS AMOSTRAS")
print("=" * 80)
print(df[["Subclass", "Electrolyte", "Current", "Treatment"]])

# ===============================
# TABELA DE CAPACITÂNCIA E ENERGIA
# ===============================

print("\n" + "=" * 80)
print("PROPRIEDADES ELÉTRICAS - CAPACITÂNCIA E ENERGIA")
print("=" * 80)

cap_energy_cols = ["C_mean", "C_max", "Energy_mean"]
cap_energy_data = df[cap_energy_cols].copy()

# Renomear colunas para melhor legibilidade
cap_energy_data.columns = ["C média (F)", "C máxima (F)", "Energia média (J)"]

# Formatação para exibição
pd.options.display.float_format = "{:.3e}".format
print(cap_energy_data)
pd.reset_option("display.float_format")

# Salvar tabela
cap_energy_data.to_csv(f"{OUT_DIR}/capacitance_energy.csv")
print(f"\nTabela salva em: {OUT_DIR}/capacitance_energy.csv")
