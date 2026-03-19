"""ROC grid plots and 2D AUC heatmaps."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from evaluation.roc import roc_from_scores


def plot_roc_grid(
    roc_results_by_bin,
    var1_name,
    var1_ranges,
    var2_name,
    var2_ranges,
    collection_label="PNet",
    working_point=0.1,
    figsize=None,
):
    """Grid of ROC curves, each subplot is a var1 bin with different var2 lines.

    Parameters
    ----------
    roc_results_by_bin : dict
        Mapping (i, j) -> (fpr, tpr, auc, thresholds) for each (var1, var2) bin.
    var1_name, var2_name : str
        Axis label names (e.g. "pT", "|eta|").
    var1_ranges, var2_ranges : list of (low, high)
        Bin edges for each variable.
    collection_label : str
        Label for the tagger/collection.
    working_point : float
        FPR value to mark on each curve.
    figsize : tuple, optional
        Figure size.
    """
    n_rows = int(np.ceil(len(var1_ranges) / 3))
    n_cols = min(3, len(var1_ranges))
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (v1_lo, v1_hi) in enumerate(var1_ranges):
        if idx >= len(axes):
            break
        ax = axes[idx]
        for j, (v2_lo, v2_hi) in enumerate(var2_ranges):
            key = (idx, j)
            if key not in roc_results_by_bin:
                continue
            fpr, tpr, auc_val, _ = roc_results_by_bin[key]
            label = f"{var2_name} [{v2_lo}, {v2_hi}) AUC={auc_val:.3f}"
            ax.plot(fpr, tpr, label=label)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        if working_point is not None:
            ax.axvline(working_point, color="grey", linestyle=":", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{collection_label}: {var1_name} [{v1_lo}, {v1_hi})")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for idx in range(len(var1_ranges), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


def plot_2d_performance_heatmap(
    auc_matrix,
    var1_ranges,
    var2_ranges,
    var1_name="pT",
    var2_name="|eta|",
    count_matrix=None,
    unc_matrix=None,
    title=None,
    vmin=0.5,
    vmax=1.0,
    cmap="viridis",
    figsize=(12, 9),
):
    """Heatmap of AUC (or other metric) across 2D kinematic bins.

    Parameters
    ----------
    auc_matrix : ndarray (n_var1, n_var2)
    count_matrix : ndarray, optional
        If provided, annotations include count information.
    unc_matrix : ndarray, optional
        If provided, annotations include uncertainty.
    """
    pt_labels = [f"[{lo},{hi})" for lo, hi in var1_ranges]
    eta_labels = [f"[{lo},{hi})" for lo, hi in var2_ranges]

    annot = np.empty_like(auc_matrix, dtype=object)
    for i in range(auc_matrix.shape[0]):
        for j in range(auc_matrix.shape[1]):
            if np.isnan(auc_matrix[i, j]):
                n_str = f"N={int(count_matrix[i, j])}" if count_matrix is not None else ""
                annot[i, j] = f"{n_str}\n(N/A)" if n_str else "N/A"
            elif unc_matrix is not None and not np.isnan(unc_matrix[i, j]):
                n_str = f"(N={int(count_matrix[i, j])})" if count_matrix is not None else ""
                annot[i, j] = f"{auc_matrix[i, j]:.3f}+/-{unc_matrix[i, j]:.3f}\n{n_str}"
            elif count_matrix is not None:
                annot[i, j] = f"{auc_matrix[i, j]:.3f}\n(N={int(count_matrix[i, j])})"
            else:
                annot[i, j] = f"{auc_matrix[i, j]:.3f}"

    fig, ax = plt.subplots(figsize=figsize)
    df = pd.DataFrame(auc_matrix, index=pt_labels, columns=eta_labels)
    sns.heatmap(
        df,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "AUC"},
        annot_kws={"fontsize": 9},
    )
    ax.set_xlabel(f"${var2_name}$")
    ax.set_ylabel(f"{var1_name} [GeV]")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig
