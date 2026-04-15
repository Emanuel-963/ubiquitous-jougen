"""EIS pipeline — orchestrated from typed stage functions.

Public API
----------
``run_eis_pipeline(config=None) -> EISResult``

Each stage is a pure-ish function that takes a DataFrame (or the config)
and returns a well-scoped result.  The orchestrator at the bottom wires
them together.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.cpe_fit import fit_cpe_warburg
from src.circuit_fitting import run_shortlist_fit
from src.feature_store import FeatureStore, FittingHistory, record_from_shortlist_result
from src.loader import load_eis_file
from src.ml_circuit_selector import CircuitMLSelector
from src.logger import setup_logging
from src.metadata import extract_metadata
from src.models import EISResult, PCAResult
from src.pca_analysis import run_pca
from src.physics_metrics import extract_features
from src.preprocessing import preprocess
from src.ranking import apply_classification, rank_within_subclass
from src.stability import extract_sample_id, stability_metrics
from src.validation import validate_eis_full
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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Load & extract features
# ═══════════════════════════════════════════════════════════════════════════

def load_and_extract(
    cfg: PipelineConfig,
) -> Tuple[Dict[str, dict], Dict[str, pd.DataFrame], List[dict]]:
    """Load every EIS file, extract features, fit CPE+Warburg and circuits.

    Returns
    -------
    records : dict[str, dict]
        ``{filename: feature_dict}``
    raw_eis : dict[str, DataFrame]
        ``{filename: preprocessed_df}``
    circuit_rows : list[dict]
        One row per file with circuit fitting results.
    """
    data_dir = cfg.data_dir
    reports_dir = cfg.reports_dir
    os.makedirs(reports_dir, exist_ok=True)

    # Feature store for ML history
    store = FeatureStore(cfg.feature_store_path)
    history = FittingHistory(store)

    # ML circuit selector — train on existing history
    ml_selector = CircuitMLSelector()
    ml_selector.train(store)

    records: Dict[str, dict] = {}
    raw_eis: Dict[str, pd.DataFrame] = {}
    circuit_rows: List[dict] = []

    for file in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, file)
        try:
            df = preprocess(load_eis_file(path))
            raw_eis[file] = df.copy()

            # Validate loaded EIS data
            vr = validate_eis_full(df)
            vr.log_all()

            feat = extract_features(df)

            # CPE + Warburg fit (fault-tolerant)
            try:
                fit = fit_cpe_warburg(df)
            except Exception as exc:
                logger.warning("CPE fit failed for %s: %s", file, exc)
                fit = {
                    "Rs_fit": np.nan, "Rp_fit": np.nan,
                    "Q": np.nan, "n": np.nan, "Sigma": np.nan,
                }
            feat.update(fit)
            records[file] = feat

            # Circuit shortlist + BIC-best selection
            _fit_circuit_for_file(df, file, cfg, reports_dir,
                                  circuit_rows, store, history,
                                  ml_selector)

        except Exception as exc:
            logger.error("Failed to process %s: %s", file, exc)

    return records, raw_eis, circuit_rows


def _fit_circuit_for_file(
    df: pd.DataFrame,
    file: str,
    cfg: PipelineConfig,
    reports_dir: str,
    circuit_rows: List[dict],
    store: FeatureStore,
    history: FittingHistory,
    ml_selector: Optional["CircuitMLSelector"] = None,
) -> None:
    """Fit circuit models to a single file and append to *circuit_rows*.

    After a successful fit the result is also persisted in the
    :class:`FeatureStore` so that future runs can leverage the history
    for ML-driven circuit selection.
    """
    try:
        # ML-ranked shortlist (empty if not trained → heuristic fallback)
        from src.circuit_fitting import extract_eis_features_for_ml
        ml_ranked: Optional[List[str]] = None
        if ml_selector is not None and ml_selector.is_trained:
            try:
                spec_feats = extract_eis_features_for_ml(df)
                ml_ranked = ml_selector.predict(spec_feats, top_n=3)
                if ml_ranked:
                    logger.info(
                        "ML selector for %s: %s (%s)",
                        file, ml_ranked,
                        ml_selector.explain(spec_feats),
                    )
            except Exception as exc:
                logger.debug("ML predict failed for %s: %s", file, exc)

        circ_res = run_shortlist_fit(
            df,
            sample_name=file,
            save_plots=True,
            plots_dir=cfg.circuits_fig_dir,
            ml_ranked=ml_ranked,
        )

        # Persist fitting record in the feature store
        try:
            rec = record_from_shortlist_result(file, circ_res)
            if rec is not None:
                summary = history.summary_text(
                    rec.get("spectral_features", {}), n=10,
                )
                logger.info("Feature history for %s: %s", file, summary)
                store.add_record(rec)
        except Exception as exc:
            logger.warning("Feature store record failed for %s: %s", file, exc)
        # Save JSON report
        try:
            rpt_path = os.path.join(reports_dir, f"{file}.json")
            with open(rpt_path, "w", encoding="utf-8") as fh:
                json.dump(circ_res, fh, default=str, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Report save failed for %s: %s", file, exc)

        best = circ_res.get("best") or {}
        results_list = circ_res.get("results") or []
        second = results_list[1] if len(results_list) > 1 else {}
        circuit_rows.append({
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
        })
    except Exception as exc:
        circuit_rows.append({
            "Arquivo": file,
            "Circuito": None, "Representacao": None,
            "BIC": None, "BIC_penalizado": None, "AIC": None,
            "RSS": None, "Confianca": None, "Res_autocorr": None,
            "Res_estruturado": None, "Bound_hits": None,
            "Params_std": None, "Params": str(exc),
            "Shortlist": "", "Sucesso": False, "Diag_plot": None,
            "Circuito2": None, "Confianca2": None,
            "BIC2": None, "BIC2_penalizado": None,
        })


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Build features DataFrame + metadata + classification
# ═══════════════════════════════════════════════════════════════════════════

def build_features_df(records: Dict[str, dict]) -> pd.DataFrame:
    """Convert raw per-file feature dicts into a single DataFrame."""
    df = pd.DataFrame(records).T
    if df.empty:
        return df

    # Metadata extraction
    df["Electrolyte"], df["Current"], df["Treatment"] = zip(
        *[extract_metadata(name) for name in df.index]
    )
    df["Sample"] = [extract_sample_id(name) for name in df.index]
    return df


def classify_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clustering classification then composite-score ranking."""
    df = apply_classification(df, safe=True)
    df = rank_within_subclass(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — Stability
# ═══════════════════════════════════════════════════════════════════════════

def compute_stability(
    df: pd.DataFrame, cfg: PipelineConfig, out_dir: str,
) -> Dict[str, pd.DataFrame]:
    """Compute and save CV stability for each configured column."""
    stab_dict: Dict[str, pd.DataFrame] = {}
    for col in cfg.stability_columns:
        if col in df.columns:
            stab = stability_metrics(df, col)
            stab.to_csv(f"{out_dir}/stability_{col}.csv")
            stab_dict[col] = stab
    return stab_dict


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 — PCA
# ═══════════════════════════════════════════════════════════════════════════

def compute_pca_stage(
    df: pd.DataFrame, cfg: PipelineConfig, out_dir: str,
) -> PCAResult:
    """Run PCA if data is sufficient, return PCAResult."""
    valid_cols = [c for c in cfg.pca_columns if c in df.columns]
    X = df[valid_cols].dropna()

    if X.shape[0] < cfg.pca_min_rows or X.var().sum() <= cfg.kmeans_variance_threshold:
        logger.info("PCA skipped: insufficient variance or too few rows.")
        return PCAResult()

    try:
        pca_obj, df_pca, loadings, evr = run_pca(X)
        df_pca.to_csv(f"{out_dir}/pca_scores.csv")
        loadings.to_csv(f"{out_dir}/pca_loadings.csv")

        paths: List[str] = []

        fig_2d = pca_2d(
            df_pca, df.loc[df_pca.index, "Subclass"],
            title="Análise PCA 2D - Amostras por Tipo", evr=evr,
        )
        if fig_2d:
            paths.append(fig_2d)

        fig_3d = pca_3d(
            df_pca, df.loc[df_pca.index, "Subclass"],
            title="Análise PCA 3D - Amostras por Tipo", evr=evr,
        )
        if fig_3d:
            paths.append(fig_3d)

        fig_scree = pca_scree_plot(evr)
        if fig_scree:
            paths.append(fig_scree)

        fig_biplot = pca_biplot_2d(
            df_pca, loadings, df.loc[df_pca.index, "Subclass"],
            title="Biplot PCA (PC1 x PC2)", evr=evr,
        )
        if fig_biplot:
            paths.append(fig_biplot)

        return PCAResult(df_pca=df_pca, loadings=loadings, evr=evr, figure_paths=paths)
    except Exception as exc:
        logger.warning("PCA failed: %s", exc)
        return PCAResult()


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5 — Capacitance / energy / retention table
# ═══════════════════════════════════════════════════════════════════════════

def build_cap_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Build the capacitance-energy-retention table from features_df."""
    cap_cols = ["C_mean", "C_max", "C_lowfreq", "Energy_mean"]
    cap = df[cap_cols].copy()

    mass_col = next((c for c in ["mass_g", "Mass_g", "mass"] if c in df.columns), None)
    if mass_col is not None:
        cap["C_espec (F/g)"] = cap["C_lowfreq"] / df[mass_col]
    else:
        cap["C_espec (F/g)"] = cap["C_lowfreq"]

    # Retention based on numeric prefix grouping
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

    cap["Retenção (%)"] = np.nan
    name_info = cap.index.to_series().apply(_split_name)
    cap["_base"] = name_info.apply(lambda x: x[1])
    cap["_lead"] = name_info.apply(lambda x: x[0])

    for _base, grp in cap.groupby("_base"):
        grp_sorted = grp.sort_values("_lead")
        vals = grp_sorted["Energy_mean"].dropna()
        if vals.empty:
            continue
        initial, final = vals.iloc[0], vals.iloc[-1]
        if np.isfinite(initial) and initial != 0:
            cap.loc[grp.index, "Retenção (%)"] = (final / initial) * 100.0

    cap = cap.drop(columns=["_base", "_lead"])
    cap = cap.rename(columns={
        "C_mean": "C média (F)",
        "C_max": "C máxima (F)",
        "C_lowfreq": "C menor f (F)",
        "Energy_mean": "Energia média (J)",
    })
    return cap


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6 — Analytics plots (correlation, retention scatter, series)
# ═══════════════════════════════════════════════════════════════════════════

def generate_analytics_plots(
    df_ranked: pd.DataFrame,
    cap_energy: pd.DataFrame,
    pca_result: PCAResult,
    cfg: PipelineConfig,
) -> List[str]:
    """Generate correlation heatmap, scatter, series and PCA-metric plots."""
    extra: List[str] = []
    analytics_dir = cfg.analytics_fig_dir

    corr_cols = [
        "Rs_fit", "Rp_fit", "Q", "n", "Sigma",
        "C_mean", "C_lowfreq", "C_espec (F/g)",
        "Energy_mean", "Retenção (%)", "Tau", "Dispersion", "Score", "Rank",
    ]
    path = correlation_heatmap(df_ranked, corr_cols, out_dir=analytics_dir)
    if path:
        extra.append(path)

    rr = scatter_rank_retention(df_ranked, out_dir=analytics_dir)
    if rr:
        extra.append(rr)

    if pca_result.df_pca is not None:
        retention = cap_energy.get("Retenção (%)")
        if retention is not None:
            pm = pca_2d_metric(
                pca_result.df_pca,
                retention.reindex(pca_result.df_pca.index),
                title="PCA 2D - colorido por Retenção",
                out_dir=analytics_dir,
            )
            if pm:
                extra.append(pm)

    for col_name in ("Energia média (J)", "C_espec (F/g)"):
        paths = series_by_prefix(cap_energy, col_name, out_dir=analytics_dir)
        if paths:
            extra.extend(paths)

    return extra


# ═══════════════════════════════════════════════════════════════════════════
# Stage 7 — Circuit summary table
# ═══════════════════════════════════════════════════════════════════════════

def build_circuit_tables(
    circuit_rows: List[dict], out_dir: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build circuit_table and circuit_summary from raw rows."""
    if not circuit_rows:
        return None, None

    circuit_table = pd.DataFrame(circuit_rows)
    circuit_table.to_csv(f"{out_dir}/circuit_fits.csv", index=False)

    summary_cols = [
        "Circuito", "Representacao", "Sucesso", "Confianca",
        "BIC_penalizado", "RSS", "Res_estruturado", "Bound_hits",
    ]
    intersect = [c for c in summary_cols if c in circuit_table.columns]

    circuit_summary = None
    if intersect and "Circuito" in intersect:
        try:
            agg_cols = {c: "mean" for c in intersect if c not in ("Circuito", "Representacao")}
            if "Representacao" in intersect:
                agg_cols["Representacao"] = "first"
            summary = circuit_table[intersect].groupby("Circuito").agg(agg_cols)
            summary = summary.rename(columns={
                "Sucesso": "Sucesso_medio",
                "Confianca": "Confianca_media",
                "BIC_penalizado": "BIC_penalizado_medio",
                "RSS": "RSS_medio",
                "Res_estruturado": "Res_estruturado_pct",
                "Bound_hits": "Bound_hits_medio",
            })
            summary["Contagem"] = circuit_table.groupby("Circuito").size()
            circuit_summary = summary.reset_index()
            circuit_summary.to_csv(f"{out_dir}/circuit_summary.csv", index=False)
        except Exception as exc:
            logger.warning("Circuit summary failed: %s", exc)

    return circuit_table, circuit_summary


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_eis_pipeline(config: Optional[PipelineConfig] = None) -> EISResult:
    """Run the full EIS analysis pipeline.

    Parameters
    ----------
    config : PipelineConfig, optional
        Centralised configuration.  Defaults are used when ``None``.

    Returns
    -------
    EISResult
        Typed result with dict-compatible bracket access for migration.
    """
    cfg = config or PipelineConfig.default()
    out_dir = cfg.tables_dir
    os.makedirs(out_dir, exist_ok=True)

    # Stage 1 — load & extract
    records, raw_eis, circuit_rows = load_and_extract(cfg)

    # Stage 2 — build DataFrame, metadata, classify + rank
    features_df = build_features_df(records)

    if features_df.empty:
        logger.warning("No EIS files found in %s", cfg.data_dir)
        return EISResult(out_dir=out_dir, raw_eis=raw_eis, config_used=cfg)

    ranked_df = classify_and_rank(features_df)

    # Stage 3 — stability
    stab = compute_stability(ranked_df, cfg, out_dir)

    # Stage 4 — PCA
    pca_result = compute_pca_stage(ranked_df, cfg, out_dir)

    # Stage 5 — capacitance / energy / retention
    cap_energy = build_cap_energy(ranked_df)
    cap_energy.to_csv(f"{out_dir}/capacitance_energy.csv")

    # Stage 6 — analytics plots
    extra_plots = generate_analytics_plots(ranked_df, cap_energy, pca_result, cfg)
    pca_result.figure_paths.extend(extra_plots)

    # Stage 7 — circuit tables
    circuit_table, circuit_summary = build_circuit_tables(circuit_rows, out_dir)

    # Save feature tables
    features_df.to_csv(f"{out_dir}/summary_features.csv")
    ranked_df.to_csv(f"{out_dir}/ranked_results.csv")

    logger.info("EIS pipeline complete — %d files processed.", len(raw_eis))

    return EISResult(
        features_df=features_df,
        ranked_df=ranked_df,
        cap_energy_df=cap_energy,
        circuit_table=circuit_table,
        circuit_summary=circuit_summary,
        pca=pca_result,
        stability=stab,
        raw_eis=raw_eis,
        out_dir=out_dir,
        config_used=cfg,
    )


if __name__ == "__main__":
    setup_logging()
    run_eis_pipeline()
