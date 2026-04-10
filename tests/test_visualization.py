import os

import matplotlib.pyplot as plt
import pandas as pd

from src import visualization as viz


def test_nyquist_plots_points():
    df = pd.DataFrame({"zreal": [1.0, 2.0, 3.0], "zimag": [0.1, 0.2, 0.3]})
    # Ensure no exception and points are plotted
    plt.close("all")
    viz.nyquist(df, label="test")
    ax = plt.gca()
    lines = ax.get_lines()
    assert len(lines) >= 1
    xdata = lines[0].get_xdata()
    ydata = lines[0].get_ydata()
    assert list(xdata) == [1.0, 2.0, 3.0]
    assert list(ydata) == [-0.1, -0.2, -0.3]
    plt.close()


def test_pca_2d_and_3d_and_boxplot_files_created(tmp_path):
    outdir = str(tmp_path)
    df_pca = pd.DataFrame({"PC1": [0.1, -0.1], "PC2": [1.0, -1.0]})
    labels = pd.Series(["Interface eficiente", "Genérica estável"], index=df_pca.index)

    p2 = viz.pca_2d(df_pca, labels, out_dir=outdir)
    assert os.path.exists(p2)
    assert p2.endswith("pca_2d.png")

    # pca_3d should return None when PC3 missing
    p3_none = viz.pca_3d(df_pca, labels, out_dir=outdir)
    assert p3_none is None

    # Add PC3 and test 3d plotting
    df_pca["PC3"] = [0.5, -0.5]
    p3 = viz.pca_3d(df_pca, labels, out_dir=outdir)
    assert os.path.exists(p3)
    assert p3.endswith("pca_3d.png")

    # boxplot
    df = pd.DataFrame({"param": [1.0, 2.0, 3.0, 4.0], "group": ["a", "a", "b", "b"]})
    box = viz.boxplot_param(df, "param", "group", out_dir=outdir)
    assert os.path.exists(box)
    assert "boxplot_param_by_group" in os.path.basename(box)
