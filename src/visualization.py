import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# Paleta base para subclasses
COLOR_MAP = {
    "Interface eficiente": "#1f77b4",
    "Genérica estável": "#ff7f0e",
    "Indefinida (dados insuficientes)": "#d62728",
    "Indefinida (sem ajuste físico)": "#9467bd",
}
WATERMARK_TEXT = "IonFlow Pipeline"
WATERMARK_COLOR = "#0b84ff"


def _add_watermark(fig):
    """Adiciona marca discreta no canto inferior direito."""
    fig.text(0.995, 0.01, WATERMARK_TEXT, ha="right", va="bottom", fontsize=8, color=WATERMARK_COLOR, alpha=0.55)


def _save_with_watermark(fig, filepath, dpi=300, bbox_inches="tight"):
    _add_watermark(fig)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)


def nyquist(df, label=None):
    plt.plot(df["zreal"], -df["zimag"], "o-", label=label)
    plt.xlabel("Z' (Ω)")
    plt.ylabel("-Z'' (Ω)")
    plt.axis("equal")
    if label:
        plt.legend()


def pca_2d(df_pca, labels, title="PCA 2D", out_dir="outputs/figures", evr=None):
    """Gráfico 2D de PCA colorido por subclasse (PC1 x PC2)."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))

    for lbl in labels.unique():
        idx = labels == lbl
        color = COLOR_MAP.get(lbl, "#999999")
        plt.scatter(
            df_pca.loc[idx, "PC1"],
            df_pca.loc[idx, "PC2"],
            label=lbl,
            s=100,
            alpha=0.7,
            color=color,
            edgecolors="black",
            linewidth=0.5,
        )

    pc1_label = "PC1"
    pc2_label = "PC2"
    if evr is not None and len(evr) >= 2:
        pc1_label = f"PC1 ({evr.iloc[0]*100:.1f}%)"
        pc2_label = f"PC2 ({evr.iloc[1]*100:.1f}%)"

    plt.xlabel(pc1_label)
    plt.ylabel(pc2_label)
    plt.title(title)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(out_dir, "pca_2d.png")
    fig = plt.gcf()
    _save_with_watermark(fig, filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def pca_3d(df_pca, labels, title="PCA 3D", out_dir="outputs/figures", evr=None):
    """Gráfico 3D de PCA colorido por subclasse."""
    os.makedirs(out_dir, exist_ok=True)

    if "PC3" not in df_pca.columns:
        return None

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for lbl in labels.unique():
        idx = labels == lbl
        color = COLOR_MAP.get(lbl, "#999999")
        ax.scatter(
            df_pca.loc[idx, "PC1"],
            df_pca.loc[idx, "PC2"],
            df_pca.loc[idx, "PC3"],
            label=lbl,
            s=100,
            alpha=0.7,
            color=color,
            edgecolors="black",
            linewidth=0.5,
        )

    pc1_label = "PC1"
    pc2_label = "PC2"
    pc3_label = "PC3"
    if evr is not None and len(evr) >= 3:
        pc1_label = f"PC1 ({evr.iloc[0]*100:.1f}%)"
        pc2_label = f"PC2 ({evr.iloc[1]*100:.1f}%)"
        pc3_label = f"PC3 ({evr.iloc[2]*100:.1f}%)"

    ax.set_xlabel(pc1_label)
    ax.set_ylabel(pc2_label)
    ax.set_zlabel(pc3_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()

    filepath = os.path.join(out_dir, "pca_3d.png")
    _save_with_watermark(fig, filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def pca_scree_plot(evr, out_dir="outputs/figures") -> str:
    """Salva scree plot da variância explicada."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(evr) + 1), evr * 100, marker="o")
    plt.xlabel("Componente")
    plt.ylabel("Variância explicada (%)")
    plt.title("Scree plot")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(out_dir, "pca_scree.png")
    fig = plt.gcf()
    _save_with_watermark(fig, filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def pca_biplot_2d(df_pca, loadings, labels, title="PCA biplot", out_dir="outputs/figures", evr=None):
    """Biplot 2D com scores e vetores de loadings."""
    os.makedirs(out_dir, exist_ok=True)

    if not {"PC1", "PC2"}.issubset(df_pca.columns):
        return None

    plt.figure(figsize=(10, 8))

    # Pontos
    for lbl in labels.unique():
        idx = labels == lbl
        color = COLOR_MAP.get(lbl, "#999999")
        plt.scatter(
            df_pca.loc[idx, "PC1"],
            df_pca.loc[idx, "PC2"],
            label=lbl,
            s=80,
            alpha=0.7,
            color=color,
            edgecolors="black",
            linewidth=0.4,
        )

    # Vetores de loadings
    arrow_scale = 2.0
    for feature, row in loadings.iterrows():
        plt.arrow(
            0,
            0,
            row["PC1"] * arrow_scale,
            row["PC2"] * arrow_scale,
            color="#444444",
            alpha=0.7,
            width=0.005,
            head_width=0.06,
            length_includes_head=True,
        )
        plt.text(
            row["PC1"] * arrow_scale * 1.08,
            row["PC2"] * arrow_scale * 1.08,
            feature,
            fontsize=9,
            color="#222222",
        )

    pc1_label = "PC1"
    pc2_label = "PC2"
    if evr is not None and len(evr) >= 2:
        pc1_label = f"PC1 ({evr.iloc[0]*100:.1f}%)"
        pc2_label = f"PC2 ({evr.iloc[1]*100:.1f}%)"

    plt.xlabel(pc1_label)
    plt.ylabel(pc2_label)
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.25)
    plt.axhline(0, color="#888888", lw=0.8, alpha=0.6)
    plt.axvline(0, color="#888888", lw=0.8, alpha=0.6)
    plt.tight_layout()

    filepath = os.path.join(out_dir, "pca_biplot.png")
    fig = plt.gcf()
    _save_with_watermark(fig, filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def pca_2d_metric(df_pca, metric: pd.Series, title="PCA 2D (métrica contínua)", out_dir="outputs/figures"):
    """PCA 2D colorido por uma métrica contínua (ex.: retenção)."""
    os.makedirs(out_dir, exist_ok=True)

    if not {"PC1", "PC2"}.issubset(df_pca.columns):
        return None

    aligned = metric.reindex(df_pca.index)
    if aligned.dropna().empty:
        return None

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df_pca["PC1"], df_pca["PC2"], c=aligned, cmap="viridis", s=90, edgecolors="black", linewidths=0.4)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(sc, label=metric.name or "métrica")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(out_dir, "pca_2d_metric.png")
    fig = plt.gcf()
    _save_with_watermark(fig, filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def scatter_rank_retention(df, out_dir="outputs/figures"):
    """Rank vs Retenção (%)."""
    os.makedirs(out_dir, exist_ok=True)
    if "Rank" not in df.columns or "Retenção (%)" not in df.columns:
        return None
    if df["Retenção (%)"].dropna().empty:
        return None

    plt.figure(figsize=(8, 6))
    plt.scatter(df["Rank"], df["Retenção (%)"], c="#1f77b4", alpha=0.8, edgecolors="black", linewidths=0.4)
    plt.xlabel("Rank")
    plt.ylabel("Retenção (%)")
    plt.title("Rank vs Retenção")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(out_dir, "rank_vs_retencao.png")
    fig = plt.gcf()
    _save_with_watermark(fig, filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath


def correlation_heatmap(df, cols, out_dir="outputs/figures"):
    """Heatmap de correlação (Spearman) com anotação de coeficientes."""
    os.makedirs(out_dir, exist_ok=True)
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return None

    data = df[cols].apply(pd.to_numeric, errors="coerce")
    data = data.where(np.isfinite(data))
    data = data.dropna(how="all")
    data = data.infer_objects(copy=False)
    if data.empty:
        return None

    corr = data.corr(method="spearman")
    pvals = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i <= j:
                rho, p = spearmanr(data[c1], data[c2], nan_policy="omit")
                corr.loc[c1, c2] = rho
                corr.loc[c2, c1] = rho
                pvals.loc[c1, c2] = p
                pvals.loc[c2, c1] = p

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            if np.isnan(val):
                text = ""
            else:
                text = f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    plt.title("Correlação (Spearman)")
    plt.tight_layout()

    filepath = os.path.join(out_dir, "correlation_heatmap.png")
    _save_with_watermark(fig, filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    corr.to_csv(os.path.join(out_dir, "correlation_matrix.csv"))
    pvals.to_csv(os.path.join(out_dir, "correlation_pvalues.csv"))

    return filepath


def _safe_filename(text: str) -> str:
    """Remove separadores e caracteres inválidos para nomes de arquivo."""
    text = text.strip()
    text = re.sub(r"[\\/:*?\"<>|]", "_", text)
    text = text.replace(" ", "_")
    return text


def series_by_prefix(df, value_col: str, out_dir="outputs/figures"):
    """Plota série (linha) ordenada por prefixo numérico do nome do arquivo."""
    os.makedirs(out_dir, exist_ok=True)
    if value_col not in df.columns:
        return None

    def _split(name: str):
        parts = name.replace(".txt", "").split()
        if not parts:
            return (0.0, name)
        try:
            return (float(parts[0]), " ".join(parts[1:]) if len(parts) > 1 else name)
        except ValueError:
            return (0.0, name)

    info = df.index.to_series().apply(_split)
    lead = info.apply(lambda x: x[0])
    base = info.apply(lambda x: x[1])
    df_plot = df.copy()
    df_plot["_lead"] = lead
    df_plot["_base"] = base

    paths = []
    for base_name, grp in df_plot.groupby("_base"):
        grp = grp.sort_values("_lead")
        if grp[value_col].dropna().empty:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(grp["_lead"], grp[value_col], "o-", color="#1f77b4")
        plt.xlabel("Prefixo numérico")
        plt.ylabel(value_col)
        plt.title(f"{value_col} - {base_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_value = _safe_filename(value_col)
        safe_base = _safe_filename(base_name)
        fname = f"series_{safe_value}_{safe_base}.png"
        path = os.path.join(out_dir, fname)
        fig = plt.gcf()
        _save_with_watermark(fig, path, dpi=300, bbox_inches="tight")
        plt.close()
        paths.append(path)

    return paths or None


def boxplot_param(df, param, by, out_dir="outputs/figures"):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    df.boxplot(column=param, by=by, ax=ax)
    ax.set_title(f"{param} por {by}")
    ax.set_ylabel(param)
    plt.suptitle("")
    plt.tight_layout()

    filepath = os.path.join(out_dir, f"boxplot_{param}_by_{by}.png")
    _save_with_watermark(fig, filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filepath
