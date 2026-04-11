import json
import logging
import os

import numpy as np
import pandas as pd

from src.cpe_fit import fit_cpe_warburg
from src.circuit_fitting import run_shortlist_fit
from src.loader import load_eis_file
from src.metadata import extract_metadata
from src.pca_analysis import run_pca
from src.physics_metrics import extract_features
from src.preprocessing import preprocess
from src.ranking import apply_classification, rank_within_subclass
from src.stability import extract_sample_id, stability_metrics
from src.visualization import (
    pca_2d,
    pca_3d,
    pca_biplot_2d,
    pca_scree_plot,
    pca_2d_metric,
    scatter_rank_retention,
    correlation_heatmap,
    series_by_prefix,
)

# Configurar logging no início da execução (mas manter imports no topo)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


def run_eis_pipeline() -> dict:
    # ===============================
    # CONFIG
    # ===============================

    data_dir = "data/raw"
    out_dir = "outputs/tables"
    os.makedirs(out_dir, exist_ok=True)

    # ===============================
    # EXTRAÇÃO DE FEATURES
    # ===============================

    records = {}
    raw_eis: dict[str, pd.DataFrame] = {}
    circuit_rows = []
    reports_dir = "outputs/circuit_reports"
    os.makedirs(reports_dir, exist_ok=True)

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)

        try:
            df = preprocess(load_eis_file(path))
            raw_eis[file] = df.copy()

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

            # Circuit shortlist + fitting selection (best by BIC)
            try:
                circ_res = run_shortlist_fit(
                    df,
                    sample_name=file,
                    save_plots=True,
                    plots_dir="outputs/figures/circuits",
                )

                # Salvar relatório JSON da amostra
                try:
                    with open(os.path.join(reports_dir, f"{file}.json"), "w", encoding="utf-8") as f:
                        json.dump(circ_res, f, default=str, indent=2, ensure_ascii=False)
                except Exception as exc:
                    print(f"Falha ao salvar relatório de circuito para {file}: {exc}")

                best = circ_res.get("best") or {}
                results_list = circ_res.get("results") or []
                second = results_list[1] if len(results_list) > 1 else {}
                circuit_rows.append(
                    {
                        "Arquivo": file,
                        "Circuito": best.get("template"),
                        "Representacao": best.get("diagram"),
                        "BIC": best.get("bic"),
                        "BIC_penalizado": best.get("bic_penalized"),
                        "AIC": best.get("aic"),
                        "RSS": best.get("rss"),
                        "Confianca": best.get("confidence"),
                        "Res_autocorr": best.get("res_autocorr"),
                        "Res_estruturado": best.get("res_structured"),
                        "Bound_hits": best.get("bound_hits"),
                        "Params_std": best.get("params_std"),
                        "Params": best.get("params"),
                        "Shortlist": ", ".join(circ_res.get("shortlist", [])),
                        "Sucesso": best.get("success"),
                        "Diag_plot": best.get("diagnostic_plot"),
                        "Circuito2": second.get("template"),
                        "Confianca2": second.get("confidence"),
                        "BIC2": second.get("bic"),
                        "BIC2_penalizado": second.get("bic_penalized"),
                    }
                )
            except Exception as exc:
                circuit_rows.append(
                    {
                        "Arquivo": file,
                        "Circuito": None,
                        "Representacao": None,
                        "BIC": None,
                        "BIC_penalizado": None,
                        "AIC": None,
                        "RSS": None,
                        "Confianca": None,
                        "Res_autocorr": None,
                        "Res_estruturado": None,
                        "Bound_hits": None,
                        "Params_std": None,
                        "Params": str(exc),
                        "Shortlist": "",
                        "Sucesso": False,
                        "Diag_plot": None,
                        "Circuito2": None,
                        "Confianca2": None,
                        "BIC2": None,
                        "BIC2_penalizado": None,
                    }
                )

        except Exception as e:
            print(f"Erro ao processar {file}: {e}")

    # ===============================
    # DATAFRAME GLOBAL
    # ===============================

    df = pd.DataFrame(records).T

    if df.empty:
        print("Nenhum arquivo EIS encontrado em data/raw/.")
        print("Use o botão 'Importar EIS para raw' para adicionar arquivos.")
        return {
            "df": df,
            "df_ranked": df,
            "cap_energy": pd.DataFrame(),
            "df_pca": None,
            "pca_loadings": None,
            "pca_evr": None,
            "pca_paths": [],
            "out_dir": out_dir,
            "circuit_table": None,
            "raw_eis": raw_eis,
        }

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
            stab.to_csv(f"{out_dir}/stability_{col}.csv")

    # ===============================
    # PCA (CONDICIONAL)
    # ===============================

    pca_cols = ["Rs_fit", "Rp_fit", "Q", "n", "Sigma"]
    valid_cols = [c for c in pca_cols if c in df.columns]

    df_pca = None
    loadings = None
    evr = None
    fig_2d = None
    fig_3d = None
    fig_scree = None
    fig_biplot = None

    X = df[valid_cols].dropna()

    if X.shape[0] >= 3 and X.var().sum() > 1e-12:
        try:
            pca, df_pca, loadings, evr = run_pca(X)
            print("Variância explicada:", evr.to_dict())
            df_pca.to_csv(f"{out_dir}/pca_scores.csv")
            loadings.to_csv(f"{out_dir}/pca_loadings.csv")

            fig_2d = pca_2d(
                df_pca,
                df.loc[df_pca.index, "Subclass"],
                title="Análise PCA 2D - Amostras por Tipo",
                evr=evr,
            )
            print(f"Gráfico PCA 2D salvo em: {fig_2d}")

            fig_3d = pca_3d(
                df_pca,
                df.loc[df_pca.index, "Subclass"],
                title="Análise PCA 3D - Amostras por Tipo",
                evr=evr,
            )
            if fig_3d:
                print(f"Gráfico PCA 3D salvo em: {fig_3d}")

            fig_scree = pca_scree_plot(evr)
            fig_biplot = pca_biplot_2d(
                df_pca,
                loadings,
                df.loc[df_pca.index, "Subclass"],
                title="Biplot PCA (PC1 x PC2)",
                evr=evr,
            )
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

    df.to_csv(f"{out_dir}/summary_features.csv")
    df_ranked.to_csv(f"{out_dir}/ranked_results.csv")

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

    cap_energy_cols = ["C_mean", "C_max", "C_lowfreq", "Energy_mean"]
    cap_energy_data = df[cap_energy_cols].copy()

    # Capacitância específica (F/g) com base na menor frequência
    mass_col = next((c for c in ["mass_g", "Mass_g", "mass"] if c in df.columns), None)
    if mass_col is not None:
        cap_energy_data["C_espec (F/g)"] = cap_energy_data["C_lowfreq"] / df[mass_col]
    else:
        cap_energy_data["C_espec (F/g)"] = cap_energy_data["C_lowfreq"]

    # Retenção baseada nos nomes dos arquivos (ordem por prefixo numérico):
    # usa a energia (J) da tabela para calcular retenção.
    def _split_name(name: str):
        parts = name.replace(".txt", "").split()
        if not parts:
            return (0.0, name)
        try:
            lead = float(parts[0])
            base = " ".join(parts[1:]) if len(parts) > 1 else name
            return (lead, base)
        except ValueError:
            return (0.0, name)

    cap_energy_data["Retenção (%)"] = np.nan
    name_info = cap_energy_data.index.to_series().apply(_split_name)
    cap_energy_data["_base"] = name_info.apply(lambda x: x[1])
    cap_energy_data["_lead"] = name_info.apply(lambda x: x[0])

    for base, grp in cap_energy_data.groupby("_base"):
        grp_sorted = grp.sort_values("_lead")
        vals = grp_sorted["Energy_mean"].dropna()
        if vals.empty:
            continue
        initial = vals.iloc[0]
        final = vals.iloc[-1]
        if np.isfinite(initial) and initial != 0:
            retention = (final / initial) * 100.0
            cap_energy_data.loc[grp.index, "Retenção (%)"] = retention

    cap_energy_data = cap_energy_data.drop(columns=["_base", "_lead"])

    # Renomear colunas para melhor legibilidade
    cap_energy_data = cap_energy_data.rename(
        columns={
            "C_mean": "C média (F)",
            "C_max": "C máxima (F)",
            "C_lowfreq": "C menor f (F)",
            "Energy_mean": "Energia média (J)",
        }
    )

    # Formatação para exibição
    pd.options.display.float_format = "{:.3e}".format
    print(cap_energy_data)
    pd.reset_option("display.float_format")

    # Salvar tabela
    cap_energy_data.to_csv(f"{out_dir}/capacitance_energy.csv")
    print(f"\nTabela salva em: {out_dir}/capacitance_energy.csv")

    extra_plots = []

    # Correlation heatmap
    corr_cols = [
        "Rs_fit",
        "Rp_fit",
        "Q",
        "n",
        "Sigma",
        "C_mean",
        "C_lowfreq",
        "C_espec (F/g)",
        "Energy_mean",
        "Retenção (%)",
        "Tau",
        "Dispersion",
        "Score",
        "Rank",
    ]
    corr_path = correlation_heatmap(df_ranked, corr_cols, out_dir="outputs/figures/analytics")
    if corr_path:
        extra_plots.append(corr_path)

    # Rank vs Retenção
    rr_path = scatter_rank_retention(df_ranked, out_dir="outputs/figures/analytics")
    if rr_path:
        extra_plots.append(rr_path)

    # PCA colorido por retenção
    if df_pca is not None:
        retention_series = cap_energy_data.get("Retenção (%)")
        if retention_series is not None:
            pca_metric_path = pca_2d_metric(df_pca, retention_series.reindex(df_pca.index), title="PCA 2D - colorido por Retenção", out_dir="outputs/figures/analytics")
            if pca_metric_path:
                extra_plots.append(pca_metric_path)

    # Séries por prefixo numérico (energia e C_espec)
    series_energy = series_by_prefix(cap_energy_data, "Energia média (J)", out_dir="outputs/figures/analytics")
    if series_energy:
        extra_plots.extend(series_energy)
    series_cspec = series_by_prefix(cap_energy_data, "C_espec (F/g)", out_dir="outputs/figures/analytics")
    if series_cspec:
        extra_plots.extend(series_cspec)

    pca_paths = [p for p in [fig_2d, fig_3d, fig_scree, fig_biplot] if p]
    pca_paths.extend(extra_plots)

    circuit_table = None
    if circuit_rows:
        circuit_table = pd.DataFrame(circuit_rows)
        circuit_table.to_csv(f"{out_dir}/circuit_fits.csv", index=False)

        # Resumo agregando por circuito escolhido (melhor)
        summary_cols = [
            "Circuito",
            "Representacao",
            "Sucesso",
            "Confianca",
            "BIC_penalizado",
            "RSS",
            "Res_estruturado",
            "Bound_hits",
        ]
        intersect = [c for c in summary_cols if c in circuit_table.columns]
        if intersect:
            summary = (
                circuit_table[intersect]
                .groupby("Circuito")
                .agg({
                    "Representacao": "first",
                    "Sucesso": "mean",
                    "Confianca": "mean",
                    "BIC_penalizado": "mean",
                    "RSS": "mean",
                    "Res_estruturado": "mean",
                    "Bound_hits": "mean",
                })
                .rename(columns={
                    "Sucesso": "Sucesso_medio",
                    "Confianca": "Confianca_media",
                    "BIC_penalizado": "BIC_penalizado_medio",
                    "RSS": "RSS_medio",
                    "Res_estruturado": "Res_estruturado_pct",
                    "Bound_hits": "Bound_hits_medio",
                })
            )
            summary["Contagem"] = circuit_table.groupby("Circuito").size()
            summary = summary.reset_index()
            summary.to_csv(f"{out_dir}/circuit_summary.csv", index=False)

    return {
        "df": df,
        "df_ranked": df_ranked,
        "cap_energy": cap_energy_data,
        "df_pca": df_pca,
        "pca_loadings": loadings,
        "pca_evr": evr,
        "pca_paths": pca_paths,
        "out_dir": out_dir,
        "circuit_table": circuit_table,
        "raw_eis": raw_eis,
    }


if __name__ == "__main__":
    run_eis_pipeline()
