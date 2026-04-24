"""B-tagging efficiency and mistag-rate computation in kinematic bins."""

import numpy as np
import pandas as pd
import awkward as ak


def efficiency_table(pt_bins, eta_bins, labels, scores, jet_pt, jet_eta, wp, sample_weights=None):
    """Per-bin signal efficiency / background rejection at a given working point.

    Parameters
    ----------
    wp : float
        Score threshold (working point).
    sample_weights : array-like, optional
        Per-jet weights (e.g. QCD xsec). Applied to background efficiency only.

    Returns
    -------
    pandas.DataFrame
    """
    rows = []
    for i in range(len(pt_bins) - 1):
        for j in range(len(eta_bins) - 1):
            pt_lo, pt_hi = pt_bins[i], pt_bins[i + 1]
            eta_lo, eta_hi = eta_bins[j], eta_bins[j + 1]
            bin_mask = (
                (jet_pt >= pt_lo)
                & (jet_pt < pt_hi)
                & (np.abs(jet_eta) >= eta_lo)
                & (np.abs(jet_eta) < eta_hi)
            )
            bin_labels = labels[bin_mask]
            bin_scores = scores[bin_mask]
            bin_w = sample_weights[bin_mask] if sample_weights is not None else None
            n = len(bin_labels)
            if n == 0:
                rows.append(
                    {
                        "pt_bin": f"[{pt_lo},{pt_hi})",
                        "eta_bin": f"[{eta_lo},{eta_hi})",
                        "N": 0,
                        "sig_eff": np.nan,
                        "bkg_eff": np.nan,
                        "bkg_rej": np.nan,
                    }
                )
                continue
            sig_mask = bin_labels == 1
            bkg_mask = bin_labels == 0
            sig_eff = np.mean(bin_scores[sig_mask] >= wp) if sig_mask.any() else np.nan
            if bkg_mask.any():
                if bin_w is not None:
                    bkg_w = bin_w[bkg_mask]
                    bkg_eff = np.sum(bkg_w * (bin_scores[bkg_mask] >= wp)) / np.sum(bkg_w)
                else:
                    bkg_eff = np.mean(bin_scores[bkg_mask] >= wp)
            else:
                bkg_eff = np.nan
            bkg_rej = 1.0 / bkg_eff if bkg_eff and bkg_eff > 0 else np.inf
            rows.append(
                {
                    "pt_bin": f"[{pt_lo},{pt_hi})",
                    "eta_bin": f"[{eta_lo},{eta_hi})",
                    "N": n,
                    "sig_eff": sig_eff,
                    "bkg_eff": bkg_eff,
                    "bkg_rej": bkg_rej,
                }
            )
    return pd.DataFrame(rows)


def btag_efficiency_vs_var(sig_jets, pure_mask, tagger_name, b_tag_cut, var, bins):
    """Compute b-tag efficiency in bins of a variable for pure signal jets.

    Parameters
    ----------
    sig_jets : awkward array
        Jet collection (all signal events).
    pure_mask : awkward boolean array
        Event-level mask of gen-matched (pure) jets.
    tagger_name : str
        Attribute name for the b-tag score.
    b_tag_cut : float
        Working-point threshold.
    var : str
        Jet attribute to bin against (e.g. "pt", "eta").
    bins : array-like
        Bin edges.

    Returns
    -------
    bin_centers, efficiencies, errors : tuple of ndarrays
    """
    pure_jets = sig_jets[pure_mask]
    var_vals = ak.to_numpy(ak.flatten(getattr(pure_jets, var)))
    scores = ak.to_numpy(ak.flatten(getattr(pure_jets, tagger_name)))

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    effs = np.zeros(len(bin_centers))
    errs = np.zeros(len(bin_centers))

    for i in range(len(bin_centers)):
        mask = (var_vals >= bins[i]) & (var_vals < bins[i + 1])
        n_total = np.sum(mask)
        if n_total == 0:
            effs[i] = 0
            errs[i] = 0
            continue
        n_pass = np.sum(scores[mask] > b_tag_cut)
        effs[i] = n_pass / n_total
        errs[i] = np.sqrt(effs[i] * (1 - effs[i]) / n_total)

    return bin_centers, effs, errs


def btag_efficiency_vs_var_flat(sig_scores, sig_var_vals, b_tag_cut, bins):
    """Compute b-tag efficiency in bins of a variable from flat numpy arrays.

    Drop-in alternative to btag_efficiency_vs_var for pre-flattened data
    (e.g. loaded from a cache .npz where gen-matching is already applied).

    Parameters
    ----------
    sig_scores : ndarray
        Tagger scores for pure (gen-matched) signal jets.
    sig_var_vals : ndarray
        Variable values (pT or eta) for the same jets.
    b_tag_cut : float
        Working-point threshold.
    bins : array-like
        Bin edges.

    Returns
    -------
    bin_centers, efficiencies, errors : tuple of ndarrays
    """
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    effs = np.zeros(len(bin_centers))
    errs = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        mask = (sig_var_vals >= bins[i]) & (sig_var_vals < bins[i + 1])
        n_total = np.sum(mask)
        if n_total == 0:
            continue
        n_pass = np.sum(sig_scores[mask] > b_tag_cut)
        effs[i] = n_pass / n_total
        errs[i] = np.sqrt(effs[i] * (1 - effs[i]) / n_total)
    return bin_centers, effs, errs


def purity_vs_var_flat(all_var_vals, matched_var_vals, bins):
    """Compute purity in bins of a variable from flat numpy arrays.

    Purity = fraction of all jets in each bin that are gen-matched.
    Uses Ullrich-Xu binomial error estimate.

    Parameters
    ----------
    all_var_vals : ndarray
        Variable values (pT or eta) for ALL jets (signal + QCD combined).
    matched_var_vals : ndarray
        Variable values for gen-matched jets only.
    bins : array-like
        Bin edges.

    Returns
    -------
    bin_centers, purities, errors : tuple of ndarrays
    """
    h_all, _ = np.histogram(all_var_vals, bins=bins)
    h_matched, _ = np.histogram(matched_var_vals, bins=bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    purity = np.divide(h_matched, h_all, out=np.zeros_like(h_all, dtype=float), where=h_all > 0)
    err = np.sqrt(
        ((h_matched + 1) * (h_all - h_matched + 1)) / ((h_all + 2) ** 2 * (h_all + 3))
    )
    return bin_centers, purity, err


def mistag_rate_vs_var(bkg_scores, bkg_var_vals, b_tag_cut, bins, bkg_weights=None):
    """Compute QCD mistag rate in bins of a variable.

    Supports QCD cross-section weights with effective-N error estimation.

    Returns
    -------
    bin_centers, rates, errors : tuple of ndarrays
    """
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    rates = np.zeros(len(bin_centers))
    errs = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        mask = (bkg_var_vals >= bins[i]) & (bkg_var_vals < bins[i + 1])
        if bkg_weights is not None:
            w_total = np.sum(bkg_weights[mask])
            if w_total == 0:
                continue
            w_pass = np.sum(bkg_weights[mask] * (bkg_scores[mask] > b_tag_cut))
            rates[i] = w_pass / w_total
            n_eff = w_total**2 / np.sum(bkg_weights[mask] ** 2)  # effective N
            errs[i] = np.sqrt(rates[i] * (1 - rates[i]) / max(n_eff, 1))
        else:
            n_total = np.sum(mask)
            if n_total == 0:
                continue
            n_pass = np.sum(bkg_scores[mask] > b_tag_cut)
            rates[i] = n_pass / n_total
            errs[i] = np.sqrt(rates[i] * (1 - rates[i]) / n_total)
    return bin_centers, rates, errs
