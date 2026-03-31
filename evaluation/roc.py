"""ROC curve computation and working-point utilities.

All ROC functions support QCD cross-section weights for background samples.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score


def roc_from_scores(sig_scores, bkg_scores, sig_weights=None, bkg_weights=None):
    """Compute (fpr, tpr, auc, thresholds) from separate signal and background score arrays.

    Parameters
    ----------
    sig_scores : array-like
        Signal-class scores.
    bkg_scores : array-like
        Background-class scores.
    sig_weights : array-like, optional
        Per-sample weights for the signal class.
    bkg_weights : array-like, optional
        Per-sample weights for the background class.
    """
    if len(sig_scores) == 0 or len(bkg_scores) == 0:
        return np.array([0, 1]), np.array([0, 0]), 0.5, np.array([1.0])
    y_true = np.concatenate([np.ones(len(sig_scores)), np.zeros(len(bkg_scores))])
    y_scores = np.concatenate([sig_scores, bkg_scores])

    if sig_weights is not None and len(sig_weights) != len(sig_scores):
        raise ValueError("sig_weights must have same length as sig_scores")
    if bkg_weights is not None and len(bkg_weights) != len(bkg_scores):
        raise ValueError("bkg_weights must have same length as bkg_scores")

    if sig_weights is not None or bkg_weights is not None:
        sig_w = sig_weights if sig_weights is not None else np.ones(len(sig_scores))
        bkg_w = bkg_weights if bkg_weights is not None else np.ones(len(bkg_scores))
        sample_weight = np.concatenate([sig_w, bkg_w])
    else:
        sample_weight = None

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, sample_weight=sample_weight)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score, thresholds


def get_roc_point_at_mistag(mistag, eff, thresh, target_mistag):
    """Find the first (FPR, TPR, threshold) tuple where FPR >= target_mistag."""
    return [(m, e, th) for m, e, th in zip(mistag, eff, thresh) if m >= target_mistag][0]


def get_roc_point_at_efficiency(mistag, eff, thresh, target_eff):
    """Find the first (FPR, TPR, threshold) tuple where TPR >= target_eff."""
    return [(m, e, th) for m, e, th in zip(mistag, eff, thresh) if e >= target_eff][0]


def get_mistag_at_fixed_efficiency(fpr, tpr, target_eff=0.7):
    """Interpolate mistag rate (FPR) at a target signal efficiency (TPR)."""
    if not np.all(np.diff(tpr) >= 0):
        sorted_indices = np.argsort(tpr)
        tpr = tpr[sorted_indices]
        fpr = fpr[sorted_indices]
    return np.interp(target_eff, tpr, fpr)


def get_working_points(name, roc_data, targets=None):
    """Print and return threshold values at Tight/Medium/Loose working points.

    Parameters
    ----------
    name : str
        Tagger label for printing.
    roc_data : tuple
        (fpr, tpr, auc, thresholds) as returned by roc_from_scores.
    targets : list of (str, float), optional
        Working point definitions. Defaults to Tight=0.001, Medium=0.01, Loose=0.1.

    Returns
    -------
    list of float
        Threshold values at each working point.
    """
    if targets is None:
        targets = [("Tight", 0.001), ("Medium", 0.01), ("Loose", 0.1)]
    wps = []
    mistag, eff, auc_val, thresh = roc_data
    print(f"\n{name} (AUC: {auc_val:.5f})")
    for wp_name, target in targets:
        fpr_wp, tpr_wp, thresh_wp = get_roc_point_at_mistag(mistag, eff, thresh, target)
        print(
            f"  {wp_name}: TPR={tpr_wp*100:.2f}%, 1/FPR={1/fpr_wp:.1f}, thresh={thresh_wp:.4f}"
        )
        wps.append(thresh_wp)
    return wps


def calculate_trained_roc_2d_bins(
    pt_ranges, eta_ranges, jet_pts, jet_etas, labels, outputs, weights=None
):
    """Compute AUC in 2D (pT x |eta|) bins.

    Returns
    -------
    auc_matrix : ndarray (n_pt, n_eta)
    count_matrix : ndarray (n_pt, n_eta)
    """
    auc_matrix = np.zeros((len(pt_ranges), len(eta_ranges)))
    count_matrix = np.zeros((len(pt_ranges), len(eta_ranges)))
    for j, (pt_low, pt_high) in enumerate(pt_ranges):
        for i, (eta_low, eta_high) in enumerate(eta_ranges):
            bin_mask = (
                (jet_pts >= pt_low)
                & (jet_pts < pt_high)
                & (np.abs(jet_etas) >= eta_low)
                & (np.abs(jet_etas) < eta_high)
            )
            bin_labels, bin_outputs = labels[bin_mask], outputs[bin_mask]
            bin_weights = weights[bin_mask] if weights is not None else None
            count_matrix[j, i] = len(bin_labels)
            if len(np.unique(bin_labels)) < 2 or len(bin_labels) < 10:
                auc_matrix[j, i] = np.nan
            else:
                try:
                    auc_matrix[j, i] = roc_auc_score(
                        bin_labels, bin_outputs, sample_weight=bin_weights
                    )
                except Exception:
                    auc_matrix[j, i] = np.nan
    return auc_matrix, count_matrix


def calculate_auc_uncertainty_2d_bins(
    pt_ranges, eta_ranges, jet_pts, jet_etas, labels, outputs, weights=None, n_boot=50
):
    """Compute AUC with bootstrap uncertainty for each pT-eta bin.

    Returns
    -------
    auc_mat : ndarray (n_pt, n_eta)
    unc_mat : ndarray (n_pt, n_eta)
    cnt_mat : ndarray (n_pt, n_eta)
    """
    auc_mat = np.zeros((len(pt_ranges), len(eta_ranges)))
    unc_mat = np.zeros((len(pt_ranges), len(eta_ranges)))
    cnt_mat = np.zeros((len(pt_ranges), len(eta_ranges)))

    for j, (pt_low, pt_high) in enumerate(pt_ranges):
        for i, (eta_low, eta_high) in enumerate(eta_ranges):
            bin_mask = (
                (jet_pts >= pt_low)
                & (jet_pts < pt_high)
                & (np.abs(jet_etas) >= eta_low)
                & (np.abs(jet_etas) < eta_high)
            )
            bin_labels = labels[bin_mask]
            bin_outputs = outputs[bin_mask]
            bin_weights = weights[bin_mask] if weights is not None else None
            cnt_mat[j, i] = len(bin_labels)

            if len(np.unique(bin_labels)) < 2 or len(bin_labels) < 10:
                auc_mat[j, i] = np.nan
                unc_mat[j, i] = np.nan
            else:
                boot_aucs = []
                for _ in range(n_boot):
                    idx = np.random.choice(len(bin_labels), len(bin_labels), replace=True)
                    if len(np.unique(bin_labels[idx])) < 2:
                        continue
                    try:
                        boot_aucs.append(
                            roc_auc_score(
                                bin_labels[idx],
                                bin_outputs[idx],
                                sample_weight=(
                                    bin_weights[idx] if bin_weights is not None else None
                                ),
                            )
                        )
                    except Exception:
                        continue
                if len(boot_aucs) >= 10:
                    auc_mat[j, i] = np.mean(boot_aucs)
                    unc_mat[j, i] = np.std(boot_aucs)
                else:
                    try:
                        auc_mat[j, i] = roc_auc_score(
                            bin_labels, bin_outputs, sample_weight=bin_weights
                        )
                        unc_mat[j, i] = np.nan
                    except Exception:
                        auc_mat[j, i] = np.nan
                        unc_mat[j, i] = np.nan
    return auc_mat, unc_mat, cnt_mat


def pr_auc_and_opt_s_over_root_b(
    pt_bins, eta_bins, labels, scores, jet_pt, jet_eta, thr_grid, sample_weights=None
):
    """Compute PR-AUC and optimal S/sqrt(B) per kinematic (pT x |eta|) bin.

    Parameters
    ----------
    thr_grid : array-like
        Threshold grid to scan for S/sqrt(B) optimisation.
    sample_weights : array-like, optional
        Per-jet weights (e.g. QCD xsec weights).

    Returns
    -------
    pandas.DataFrame with columns: pt_bin, eta_bin, N, pr_auc, s_over_root_b_max,
        thr_opt, s_at_opt, b_at_opt.
    """
    import pandas as pd

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
            n = len(bin_labels)
            if n < 10 or len(np.unique(bin_labels)) < 2:
                rows.append(
                    {
                        "pt_bin": f"[{pt_lo},{pt_hi})",
                        "eta_bin": f"[{eta_lo},{eta_hi})",
                        "N": n,
                        "pr_auc": np.nan,
                        "s_over_root_b_max": np.nan,
                        "thr_opt": np.nan,
                        "s_at_opt": np.nan,
                        "b_at_opt": np.nan,
                    }
                )
                continue
            bin_w = sample_weights[bin_mask] if sample_weights is not None else None
            pr_auc_bin = average_precision_score(bin_labels, bin_scores, sample_weight=bin_w)
            srb_vals = []
            s_vals = []
            b_vals = []
            for t in thr_grid:
                preds = bin_scores >= t
                if bin_w is not None:
                    s = bin_w[(preds == 1) & (bin_labels == 1)].sum()
                    b = bin_w[(preds == 1) & (bin_labels == 0)].sum()
                else:
                    s = ((preds == 1) & (bin_labels == 1)).sum()
                    b = ((preds == 1) & (bin_labels == 0)).sum()
                srb_vals.append(s / np.sqrt(b + 1e-9))
                s_vals.append(s)
                b_vals.append(b)
            srb_vals = np.array(srb_vals)
            opt_idx = int(np.nanargmax(srb_vals))
            rows.append(
                {
                    "pt_bin": f"[{pt_lo},{pt_hi})",
                    "eta_bin": f"[{eta_lo},{eta_hi})",
                    "N": n,
                    "pr_auc": pr_auc_bin,
                    "s_over_root_b_max": srb_vals[opt_idx],
                    "thr_opt": thr_grid[opt_idx],
                    "s_at_opt": s_vals[opt_idx],
                    "b_at_opt": b_vals[opt_idx],
                }
            )
    return pd.DataFrame(rows)
