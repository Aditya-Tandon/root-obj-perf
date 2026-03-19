"""Feature importance bar plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(
    importance,
    feature_names,
    title="Feature Importance",
    figsize=(10, 5),
    color=None,
):
    """Horizontal or vertical bar plot of per-feature importance scores.

    Parameters
    ----------
    importance : ndarray (F,)
        Normalised importance per feature.
    feature_names : list of str
        Feature labels (length >= len(importance)).
    title : str
    figsize : tuple
    color : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n = len(importance)
    names = list(feature_names[:n])
    if len(names) < n:
        names += [f"f{i}" for i in range(len(names), n)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(n), importance, color=color)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    return fig
