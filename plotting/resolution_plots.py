"""Resolution and scale plots for jet pT response."""

import numpy as np
import matplotlib.pyplot as plt


def plot_resolution_vs_var(
    bin_centers,
    sigmas,
    sigma_errs,
    mus=None,
    mu_errs=None,
    var_label="Gen $p_T$ [GeV]",
    labels=None,
    colors=None,
    title=None,
    figsize=(14, 5),
):
    """Plot resolution (sigma) and optionally scale (mu) vs a kinematic variable.

    Parameters
    ----------
    bin_centers : ndarray or list of ndarrays
        Bin centres for each curve.
    sigmas : ndarray or list of ndarrays
        Resolution values per bin.
    sigma_errs : ndarray or list of ndarrays
        Uncertainties on resolution.
    mus, mu_errs : optional, same structure
        Scale values and uncertainties per bin.
    labels : list of str, optional
        Legend labels per curve.
    colors : list, optional
        Colors per curve.
    """
    # Normalise inputs to lists
    if not isinstance(sigmas, list):
        bin_centers = [bin_centers]
        sigmas = [sigmas]
        sigma_errs = [sigma_errs]
        if mus is not None:
            mus = [mus]
            mu_errs = [mu_errs]
        if labels is None:
            labels = [None]
        if colors is None:
            colors = [None]

    if labels is None:
        labels = [None] * len(sigmas)
    if colors is None:
        colors = [None] * len(sigmas)

    n_plots = 2 if mus is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Resolution
    ax = axes[0]
    for bc, sig, sig_e, lbl, col in zip(bin_centers, sigmas, sigma_errs, labels, colors):
        ax.errorbar(bc, sig, yerr=sig_e, fmt="o-", label=lbl, color=col, markersize=4)
    ax.set_xlabel(var_label)
    ax.set_ylabel(r"Resolution ($\sigma$)")
    ax.set_title("Resolution" if title is None else f"{title} — Resolution")
    if any(l is not None for l in labels):
        ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    # Scale
    if mus is not None:
        ax = axes[1]
        for bc, mu, mu_e, lbl, col in zip(bin_centers, mus, mu_errs, labels, colors):
            ax.errorbar(bc, mu, yerr=mu_e, fmt="o-", label=lbl, color=col, markersize=4)
        ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel(var_label)
        ax.set_ylabel(r"Scale ($\mu$)")
        ax.set_title("Scale" if title is None else f"{title} — Scale")
        if any(l is not None for l in labels):
            ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig
