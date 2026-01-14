import os

import matplotlib.pyplot as plt


def nyquist(df, label=None):
    plt.plot(df["zreal"], -df["zimag"], "o-", label=label)
    plt.xlabel("Z' (Ω)")
    plt.ylabel("-Z'' (Ω)")
    plt.axis("equal")
    if label:
        plt.legend()


def pca_2d(df_pca, labels, title="PCA 2D", out_dir="outputs/figures"):
    """Gráfico 2D de PCA colorido por subclasse"""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))

    # Cores para cada subclasse
    color_map = {
        "Interface eficiente": "#1f77b4",
        "Genérica estável": "#ff7f0e",
        "Indefinida (dados insuficientes)": "#d62728",
    }

    for lbl in labels.unique():
        idx = labels == lbl
        color = color_map.get(lbl, "#999999")
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

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(out_dir, "pca_2d.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def pca_3d(df_pca, labels, title="PCA 3D", out_dir="outputs/figures"):
    """Gráfico 3D de PCA colorido por subclasse"""
    os.makedirs(out_dir, exist_ok=True)

    if "PC3" not in df_pca.columns:
        # Se não há PC3, retornar None
        return None

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    color_map = {
        "Interface eficiente": "#1f77b4",
        "Genérica estável": "#ff7f0e",
        "Indefinida (dados insuficientes)": "#d62728",
    }

    for lbl in labels.unique():
        idx = labels == lbl
        color = color_map.get(lbl, "#999999")
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

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()

    filepath = os.path.join(out_dir, "pca_3d.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


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
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filepath
