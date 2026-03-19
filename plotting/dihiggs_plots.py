"""Di-Higgs mass distribution and 2D mass-plane plots."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as _Ellipse


def plot_ellipse(ax, center_x, center_y, width, height, angle=0, **kwargs):
    """Add an ellipse patch to an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    center_x, center_y : float
        Centre coordinates.
    width, height : float
        Full width/height of the ellipse.
    angle : float
        Rotation angle in degrees.
    **kwargs
        Passed to matplotlib.patches.Ellipse (e.g. edgecolor, facecolor, linestyle).
    """
    ellipse = _Ellipse(
        xy=(center_x, center_y), width=width, height=height, angle=angle, **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse


def plot_mass_1d(
    sig_lead_mass, sig_sub_mass, sig_hh_mass,
    qcd_lead_mass=None, qcd_sub_mass=None, qcd_hh_mass=None,
    qcd_weights=None,
    label="",
    sig_window_h=(90, 160),
    sig_window_hh=(250, 550),
    figsize=(24, 7),
):
    """1x3 overlay of leading mH, subleading mH, and mHH distributions.

    Signal is shown filled; QCD background as dashed step histogram.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    bins_h = np.linspace(0, 300, 61)
    bins_hh = np.linspace(200, 800, 61)
    color = "purple"

    for ax, sig_m, qcd_m, bins, xlabel, title_suffix in [
        (axes[0], sig_lead_mass, qcd_lead_mass, bins_h, r"Leading $m_H$ [GeV]", "Leading Higgs"),
        (axes[1], sig_sub_mass, qcd_sub_mass, bins_h, r"Subleading $m_H$ [GeV]", "Subleading Higgs"),
        (axes[2], sig_hh_mass, qcd_hh_mass, bins_hh, r"$m_{HH}$ [GeV]", r"$m_{HH}$"),
    ]:
        if sig_m is not None and len(sig_m) > 0:
            ax.hist(sig_m, bins=bins, histtype="stepfilled", alpha=0.3, color=color,
                    label=f"Signal ({len(sig_m)})", density=True)
            ax.hist(sig_m, bins=bins, histtype="step", linewidth=2, color=color, density=True)
        if qcd_m is not None and len(qcd_m) > 0:
            ax.hist(qcd_m, bins=bins, histtype="step", linewidth=2, color="grey",
                    linestyle="--", label=f"QCD ({len(qcd_m)})", density=True)
        if bins is bins_h:
            ax.axvline(125, color="green", linestyle=":", linewidth=1.5)
            ax.axvspan(*sig_window_h, alpha=0.05, color="green")
        else:
            ax.axvspan(*sig_window_hh, alpha=0.05, color="green")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{label} — {title_suffix}")
        ax.legend(fontsize=10)

    fig.suptitle(f"Di-Higgs Reconstruction — {label}", fontsize=16, y=1.01)
    plt.tight_layout()
    return fig


def plot_mass_2d(
    sig_lead_mass, sig_sub_mass,
    qcd_lead_mass=None, qcd_sub_mass=None,
    qcd_weights=None,
    label="",
    r_hh_cut=55.0,
    mh_centers=(125.0, 120.0),
    figsize=(20, 8),
):
    """2D m_H1 vs m_H2 scatter/hist2d with R_HH ellipse overlay.

    Parameters
    ----------
    r_hh_cut : float
        Radius of the R_HH circle to draw.
    mh_centers : tuple
        (m_H1_center, m_H2_center) for the ellipse.
    """
    bins_2d = np.linspace(0, 300, 61)
    n_panels = 1 + (1 if qcd_lead_mass is not None and len(qcd_lead_mass) > 0 else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panels = [("Signal", sig_lead_mass, sig_sub_mass, None)]
    if qcd_lead_mass is not None and len(qcd_lead_mass) > 0:
        panels.append(("QCD", qcd_lead_mass, qcd_sub_mass, qcd_weights))

    for ax, (cat, lm, sm, w) in zip(axes, panels):
        if len(lm) > 0:
            h = ax.hist2d(lm, sm, bins=[bins_2d, bins_2d], cmap="viridis")
            fig.colorbar(h[3], ax=ax, label="Events")
        c1, c2 = mh_centers
        ax.axvline(c1, color="red", linestyle="--", linewidth=1.5, label=f"$m_H$ = {c1} GeV")
        ax.axhline(c2, color="red", linestyle="--", linewidth=1.5)
        plot_ellipse(
            ax, c1, c2,
            width=2 * r_hh_cut,
            height=2 * r_hh_cut,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
            linewidth=2,
            label=f"$R_{{HH}}$ = {r_hh_cut} GeV",
        )
        ax.set_xlabel("Leading Higgs Mass [GeV]")
        ax.set_ylabel("Subleading Higgs Mass [GeV]")
        ax.set_title(f"{label} — {cat}")
        ax.legend(loc="upper right", fontsize=10)

    fig.suptitle(f"2D $m_{{H1}}$ vs $m_{{H2}}$ — {label}", fontsize=16, y=1.02)
    plt.tight_layout()
    return fig
