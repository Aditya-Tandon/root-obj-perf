import os
import json
import glob as glob_module
import argparse

import numpy as np
import awkward as ak
import vector
import fastjet

from tqdm import tqdm
from data_loading_helpers import (
    load_and_prepare_data,
    select_gen_b_quarks_from_higgs,
    select_gen_b_quarks_by_status,
    apply_custom_cuts,
    one_hot_encode_l1_puppi,
    select_gen_higgs,
)
from analysis_helpers import get_purity_mask_cross_matched

# Register the vector library with awkward array
ak.behavior.update(vector.backends.awkward.behavior)


def compute_kinematic_weights(
    pt, eta, y, n_bins_pt=100, n_bins_eta=50, max_pt=500, clip_max=50.0,
    on_signal=True, sample_weights=None, flatten_both=False, iterative=False,
    flatten_n_bins_pt=40, flatten_n_bins_eta=20, flatten_min_count=10, percentile=99.9
):
    """
    Calculates per-jet kinematic weights.

    Default mode (flatten_both=False):
        Makes Signal (y=1) kinematics match Background (y=0).
        Weight = P(k|B) / P(k|S) applied to Signal events.
        Background events get weight = 1.0.

    Flat spectrum mode (flatten_both=True):
        Reweights **both** signal and background to a flat pT-eta distribution.
        Weight = 1 / P(k|class) for each class independently.
        sample_weights are ignored in this mode.
        Iterative option alternately flattens pT and eta until convergence.
        Non-iterative option applies a single-pass 2D flattening with log-spaced pT bins.

    Parameters
    ----------
    sample_weights : np.ndarray, optional
        Per-jet sample weights (e.g. QCD cross-section weights for background).
        Used when building the pT-eta histograms so that the background
        distribution reflects the true (weighted) QCD spectrum.
        Ignored when flatten_both=True.
    flatten_both : bool
        If True, reweight both classes to flat pT-eta distributions
        independently.  No cross-section weights are used.
    iterative : bool
        If True, use a form of the sinkhorn algorithm to flatten the 1D pt and eta distributions
        If False, use a 2d Histogram based reweighting scheme to flatten the distributions
    flatten_n_bins_pt : int
        Number of pT bins for flatten mode (coarser than default to avoid
        low-statistics bins).  Uses log-spaced bins.
    flatten_n_bins_eta : int
        Number of eta bins for flatten mode.
    flatten_min_count : int
        Minimum number of jets in a bin for it to receive a non-zero weight.
        Bins below this threshold get weight = 0.
    """
    # 1. Define Bins
    pt_bins = np.linspace(0, max_pt, n_bins_pt + 1)
    eta_bins = np.linspace(-2.5, 2.5, n_bins_eta + 1)

    # 2. Split Data
    sig_mask = y == 1
    bkg_mask = y == 0

    pt_sig, eta_sig = pt[sig_mask], eta[sig_mask]
    pt_bkg, eta_bkg = pt[bkg_mask], eta[bkg_mask]

    # 3. Index arrays for all events
    pt_idx = np.clip(np.searchsorted(pt_bins, pt) - 1, 0, n_bins_pt - 1)
    eta_idx = np.clip(np.searchsorted(eta_bins, eta) - 1, 0, n_bins_eta - 1)

    if flatten_both:
        # ----- Flat spectrum mode (iterative 1D reweighting) -----
        # Alternately flatten pT (log-spaced bins) and eta (linear bins)
        # until both 1D projections converge to flat.  This handles
        # correlations between pT and eta (e.g., high-pT jets being
        # more central) that break simple factorized approaches.
        #
        # If iterative, then
            # Each iteration:
            #   1. Compute 1D weighted pT histogram → correction = target / count
            #   2. Multiply per-jet weight by pT correction
            #   3. Compute 1D weighted eta histogram → correction = target / count
            #   4. Multiply per-jet weight by eta correction
            # Converges in ~5-10 iterations.  Bins below min_count get weight 0.
        # else:
        #  Single-pass 2D flattening → sparse bins get weight 0, outliers get last-bin weight.
        # pT range is capped at the 99th percentile to avoid sparse extreme
        # tail bins that can never reach the target under clip_max.  Jets
        # above the cap get the weight of the last populated pT bin so they
        # participate in training with a sensible weight.  Sparse bins
        # (below min_count) get interpolated weights from neighbours.
        n_iters = 10
        print(f"  [flatten_both] Iterative 1D reweighting ({n_iters} iterations)")

        # Log-spaced pT bins: from min jet pT to 99th percentile
        pt_lo = max(pt[pt > 0].min(), 1.0)
        pt_99 = np.percentile(pt, percentile)
        pt_hi = pt_99 * 1.01
        flat_pt_bins = np.geomspace(pt[pt > 0].min(), np.percentile(pt, percentile) * 1.01, flatten_n_bins_pt + 1)
        flat_eta_bins = np.linspace(-2.5, 2.5, flatten_n_bins_eta + 1)
        print(f"  pT bins:  {flatten_n_bins_pt} log-spaced [{pt_lo:.1f}, {pt_hi:.1f}] GeV  (99th pctl: {pt_99:.1f})")
        print(f"  eta bins: {flatten_n_bins_eta} linear [-2.5, 2.5]")

        def _bin_idx(vals, bins, n_bins):
            return np.clip(np.searchsorted(bins, vals) - 1, 0, n_bins - 1)
        if iterative:
            def _flatten_class(pt_cls, eta_cls, min_count):
                """Iteratively flatten pT and eta for one class.

                After convergence:
                - Sparse pT/eta bins (< min_count jets) get weights
                interpolated from neighbouring populated bins.
                - Outlier jets (pT > 99th pctl) get the weight of the
                last populated pT bin (combined with their eta weight).
                """
                n_jets = len(pt_cls)
                w = np.ones(n_jets, dtype=np.float64)
                pt_idx = _bin_idx(pt_cls, flat_pt_bins, flatten_n_bins_pt)
                eta_idx = _bin_idx(eta_cls, flat_eta_bins, flatten_n_bins_eta)

                alive = np.ones(n_jets, dtype=bool)
                outlier = pt_cls > pt_99
                alive &= ~outlier

                for it in range(n_iters):
                    # --- pT correction ---
                    H_pt = np.zeros(flatten_n_bins_pt, dtype=np.float64)
                    np.add.at(H_pt, pt_idx, w * alive)
                    ok_pt = H_pt >= min_count
                    if ok_pt.any():
                        target_pt = np.median(H_pt[ok_pt])
                        corr_pt = np.ones(flatten_n_bins_pt, dtype=np.float64)
                        corr_pt[ok_pt] = target_pt / H_pt[ok_pt]
                        corr_pt[~ok_pt] = 0.0
                        w *= corr_pt[pt_idx]
                        alive &= (corr_pt[pt_idx] > 0)

                    # --- eta correction ---
                    H_eta = np.zeros(flatten_n_bins_eta, dtype=np.float64)
                    np.add.at(H_eta, eta_idx, w * alive)
                    ok_eta = H_eta >= min_count
                    if ok_eta.any():
                        target_eta = np.median(H_eta[ok_eta])
                        corr_eta = np.ones(flatten_n_bins_eta, dtype=np.float64)
                        corr_eta[ok_eta] = target_eta / H_eta[ok_eta]
                        corr_eta[~ok_eta] = 0.0
                        w *= corr_eta[eta_idx]
                        alive &= (corr_eta[eta_idx] > 0)

                # ---- Post-processing: interpolate sparse bins, fix outliers ----
                # Build per-bin mean weight maps from converged alive jets.
                def _bin_mean(idx_arr, n_bins):
                    wsum = np.zeros(n_bins, dtype=np.float64)
                    cnt = np.zeros(n_bins, dtype=np.float64)
                    np.add.at(wsum, idx_arr[alive], w[alive])
                    np.add.at(cnt, idx_arr[alive], 1)
                    mean_w = np.zeros(n_bins, dtype=np.float64)
                    pop = cnt > 0
                    mean_w[pop] = wsum[pop] / cnt[pop]
                    return mean_w, pop

                def _interp_gaps(mean_w, pop):
                    """Fill gaps by linear interpolation; edges use nearest."""
                    if pop.all() or not pop.any():
                        return mean_w
                    filled = mean_w.copy()
                    pi, ui = np.where(pop)[0], np.where(~pop)[0]
                    filled[ui] = np.interp(ui, pi, mean_w[pi])
                    return filled

                pt_mean, pt_pop = _bin_mean(pt_idx, flatten_n_bins_pt)
                eta_mean, eta_pop = _bin_mean(eta_idx, flatten_n_bins_eta)
                pt_filled = _interp_gaps(pt_mean, pt_pop)
                eta_filled = _interp_gaps(eta_mean, eta_pop)

                # Global alive-jet scale factors for factored reconstruction
                global_mean = w[alive].mean() if alive.any() else 1.0
                pt_pop_mean = pt_mean[pt_pop].mean() if pt_pop.any() else 1.0
                eta_pop_mean = eta_mean[eta_pop].mean() if eta_pop.any() else 1.0

                def _reconstruct(pt_w, eta_w):
                    """Reconstruct weight from 1D maps, scaled to alive level."""
                    return (pt_w / max(pt_pop_mean, 1e-12)
                            * eta_w / max(eta_pop_mean, 1e-12)
                            * global_mean)

                # Sparse-bin jets: interpolated weight from neighbours
                sparse = ~alive & ~outlier
                if sparse.any():
                    w[sparse] = _reconstruct(
                        pt_filled[pt_idx[sparse]],
                        eta_filled[eta_idx[sparse]],
                    )

                # Outlier jets: use last pT bin weight × their eta correction
                if outlier.any():
                    w[outlier] = _reconstruct(
                        pt_filled[-1],               # last (highest) pT bin
                        eta_filled[eta_idx[outlier]],
                    )

                return w.astype(np.float32), int(sparse.sum()), int(outlier.sum())
            
            w_sig, n_sig_sparse, n_sig_outlier = _flatten_class(pt_sig, eta_sig, flatten_min_count)
            w_bkg, n_bkg_sparse, n_bkg_outlier = _flatten_class(pt_bkg, eta_bkg, flatten_min_count)
            
            print(f"  min_count threshold: {flatten_min_count}")
            print(f"  Sig — {n_sig_outlier} outliers (last-bin wt), "
                f"{n_sig_sparse} sparse-bin jets (interpolated)")
            print(f"  Bkg — {n_bkg_outlier} outliers (last-bin wt), "
                f"{n_bkg_sparse} sparse-bin jets (interpolated)")
        else: 
            def _flatten_class(pt, eta):
                """Single-pass 2D flattening."""
                # 2D histogram counts for this class
                h_ij, _, _ = np.histogram2d(pt, eta, bins=[flat_pt_bins, flat_eta_bins], density=False)
                h_ij = np.where(h_ij == 0, np.inf, h_ij)  # Avoid division by zero; these bins will get zero weight
                w_ij = np.where(h_ij > 0, 1.0 / h_ij, 0)
                pt_idx = np.clip(np.digitize(pt, flat_pt_bins) - 1, 0, flatten_n_bins_pt - 2)
                eta_idx = np.clip(np.digitize(eta, flat_eta_bins) - 1, 0, flatten_n_bins_eta - 2)
                w = w_ij[pt_idx, eta_idx]
                max_w = np.percentile(w[w > 0], percentile)
                w = np.clip(w, 0, max_w)

                # w = w / np.mean(w[jet_pt <= np.percentile(jet_pt, percentile)])  # Normalize to mean weight in the reweighting range
                w = w / np.mean(w)

                return w

            w_sig = _flatten_class(pt_sig, eta_sig)
            w_bkg = _flatten_class(pt_bkg, eta_bkg)

        final_weights = np.ones_like(y, dtype=np.float32)
        final_weights[sig_mask] = w_sig
        final_weights[bkg_mask] = w_bkg
        final_weights = np.clip(final_weights, 0.0, clip_max)

        # Diagnostics
        n_sig_total, n_bkg_total = len(pt_sig), len(pt_bkg)
        print(f"  Signal  — mean: {final_weights[sig_mask].mean():.3f}, "
              f"max: {final_weights[sig_mask].max():.3f}")
        print(f"  Bkg     — mean: {final_weights[bkg_mask].mean():.3f}, "
              f"max: {final_weights[bkg_mask].max():.3f}")
        return final_weights

    # ----- Default mode: match one class to the other -----
    # Per-class sample weights for histogram construction
    w_sig = None if sample_weights is None else sample_weights[sig_mask]
    w_bkg = None if sample_weights is None else sample_weights[bkg_mask]

    H_sig, _, _ = np.histogram2d(
        pt_sig, eta_sig, bins=[pt_bins, eta_bins], density=True, weights=w_sig
    )
    H_bkg, _, _ = np.histogram2d(
        pt_bkg, eta_bkg, bins=[pt_bins, eta_bins], density=True, weights=w_bkg
    )

    H_sig = np.maximum(H_sig, 1e-10)
    H_bkg = np.maximum(H_bkg, 1e-10)

    # 4. Calculate Ratio Map: P(B) / P(S)
    # This is the weight we need to apply to Signal to make it look like Bkg
    if on_signal:
        ratio_map = np.divide(H_bkg, H_sig)
    else:
        ratio_map = np.divide(H_sig, H_bkg)

    all_weights = ratio_map[pt_idx, eta_idx]

    final_weights = np.ones_like(y, dtype=np.float32)
    if on_signal:
        final_weights[sig_mask] = all_weights[sig_mask]
    else:
        final_weights[bkg_mask] = all_weights[bkg_mask]

    final_weights = np.clip(final_weights, 0.1, clip_max)

    if not on_signal:
        print(f"Weight Statistics (Background only):")
        print(f"  Mean: {np.mean(final_weights[bkg_mask]):.3f}")
        print(f"  Max:  {np.max(final_weights[bkg_mask]):.3f}")
    else:
        print(f"Weight Statistics (Signal only):")
        print(f"  Mean: {np.mean(final_weights[sig_mask]):.3f}")
        print(f"  Max:  {np.max(final_weights[sig_mask]):.3f}")

    return final_weights


def cluster_candidates(events, config, key, dist_param=0.4):
    """
    Clusters candidates using Anti-kt and recovers features via integer indices.
    """
    # 1. Get Candidates
    collection_name = config[key]["collection_name"]
    candidates = events[collection_name]

    # Apply cuts
    candidates = apply_custom_cuts(candidates, config, key, kinematic_only=True)

    # 2. Prepare Inputs with 'user_index'
    # FastJet ONLY preserves 'px', 'py', 'pz', 'E', and 'user_index'.
    # We map the local index (0..N_particles) to 'user_index'.
    fastjet_inputs = ak.zip(
        {
            "px": candidates.vector.px,
            "py": candidates.vector.py,
            "pz": candidates.vector.pz,
            "E": candidates.vector.e,
            "user_index": ak.local_index(
                candidates, axis=1
            ),  # CRITICAL: Must be named 'user_index'
        },
        with_name="Momentum4D",
    )

    # 3. Cluster
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, dist_param)
    cluster = fastjet.ClusterSequence(fastjet_inputs, jet_def)

    # 4. Extract Jets with jet-level pT cut (separate from candidate-level cut)
    min_jet_pt = config[key].get("jet_pt_cut", config[key].get("pt_cut", 25.0))
    out_jets = cluster.inclusive_jets(min_pt=min_jet_pt)

    # 5. Extract Constituents with Indices
    # The output constituents will now contain the 'user_index' field we set earlier
    out_constituents = cluster.constituents(min_pt=min_jet_pt)

    # 6. Recover Full Objects using per-event indexing
    # constituent_indices shape: (Events, Jets, Constituents)
    # candidates shape: (Events, Particles)
    # We need to index per-event, so we use ak.unflatten to handle the jagged structure
    constituent_indices = out_constituents.user_index

    # Approach: Process each event to recover constituents with full properties
    # Using a cartesian-like approach: for each (event, jet, constituent), look up
    # the original candidate from that event using the user_index

    # Get the structure we need to restore
    n_constituents_per_jet = ak.num(constituent_indices, axis=2)  # (events, jets)
    n_jets_per_event = ak.num(constituent_indices, axis=1)  # (events,)

    # Flatten indices to 2D: (events, all_constituents_in_event)
    flat_indices = ak.flatten(
        constituent_indices, axis=2
    )  # (events, total_constituents)

    # Index into candidates - this works because both have the same event axis
    flat_recovered = candidates[
        flat_indices
    ]  # (events, total_constituents) with record fields

    # Now we need to unflatten the inner axis back to (jets, constituents) structure
    # Flatten n_constituents_per_jet to get counts per jet across all events
    flat_counts = ak.flatten(n_constituents_per_jet)  # 1D: counts for each jet

    # Flatten flat_recovered to (total_jets_across_all_events, total_constituents)
    # But we need to group by jets first, so we flatten the event axis
    all_constituents = ak.flatten(flat_recovered, axis=1)  # 1D array of records

    # Unflatten to (total_jets, constituents_per_jet)
    jets_constituents = ak.unflatten(all_constituents, flat_counts, axis=0)

    # Unflatten to (events, jets, constituents)
    recovered_constituents = ak.unflatten(jets_constituents, n_jets_per_event, axis=0)

    # 7. Structure Output
    # Create jet-level records with constituents stored as a nested list
    # First, create the vector field at the jet level
    jet_vectors = ak.zip(
        {
            "pt": out_jets.pt,
            "eta": out_jets.eta,
            "phi": out_jets.phi,
            "mass": out_jets.mass,
        },
        with_name="Momentum4D",
    )

    # Create the basic jet structure without constituents first
    jets_base = ak.zip(
        {
            "pt": out_jets.pt,
            "eta": out_jets.eta,
            "phi": out_jets.phi,
            "mass": out_jets.mass,
            "vector": jet_vectors,
        },
    )

    # Add constituents as a field using ak.with_field
    # This preserves the structure without broadcasting
    jets_with_vector = ak.with_field(jets_base, recovered_constituents, "constituents")

    return jets_with_vector


def extract_features(l1_jets, matched_cands, n_constituents, min_constituents=1):
    """
    Extract constituent-level features from clustered jets and build
    the flat numpy arrays (X, particle_mask) needed for training.

    Parameters
    ----------
    l1_jets : ak.Array
        Jagged array of jets with fields pt, eta, phi, mass (events, jets).
    matched_cands : ak.Array
        Jagged array of constituents (events, jets, constituents) with fields
        vector.{pt,eta,phi,mass}, dxy, z0, charge, puppiWeight, id.
    n_constituents : int
        Number of constituents to pad/clip to.
    min_constituents : int
        Minimum real constituents per jet; jets below this are flagged.

    Returns
    -------
    X : np.ndarray, shape (n_jets, n_constituents, n_features)
    particle_mask : np.ndarray, shape (n_jets, n_constituents)
    valid_jets : np.ndarray[bool], shape (n_jets,)
        Mask of jets passing the min_constituents cut (not yet applied).
    """
    j_pt = l1_jets.pt[:, :, None]
    j_eta = l1_jets.eta[:, :, None]
    j_phi = l1_jets.phi[:, :, None]

    # Constituent 4-vector
    m_pt = matched_cands.vector.pt
    m_eta = matched_cands.vector.eta
    m_phi = matched_cands.vector.phi
    m_mass = matched_cands.vector.mass

    # Impact parameters
    m_dxy = matched_cands.dxy
    m_z0 = matched_cands.z0

    # Charge
    m_charge = matched_cands.charge

    # log pT relative to jet
    log_pt_rel = np.log(np.maximum(m_pt, 1e-3) / np.maximum(j_pt, 1e-3))

    # Delta Eta / Phi
    deta = m_eta - j_eta
    dphi = m_phi - j_phi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]

    # Puppi weight
    m_w = matched_cands.puppiWeight

    # Log Delta R
    log_dr = np.log(np.maximum(np.sqrt(deta**2 + dphi**2), 1e-3))

    # Particle id (will be one-hot encoded)
    m_id = matched_cands.id

    def pad_and_fill(arr, target=n_constituents):
        return ak.fill_none(ak.pad_none(arr, target, axis=2, clip=True), 0.0)

    feature_list = [
        pad_and_fill(m_mass),
        pad_and_fill(m_pt),
        pad_and_fill(m_eta),
        pad_and_fill(m_phi),
        pad_and_fill(m_dxy),
        pad_and_fill(m_z0),
        pad_and_fill(m_charge),
        pad_and_fill(log_pt_rel),
        pad_and_fill(deta),
        pad_and_fill(dphi),
        pad_and_fill(m_w),
        pad_and_fill(log_dr),
        pad_and_fill(m_id),
    ]

    x_ini = np.stack(
        [ak.to_numpy(ak.flatten(f, axis=1)) for f in feature_list], axis=-1
    )
    flat_ids = x_ini[..., -1]
    one_hot_ids = one_hot_encode_l1_puppi(flat_ids, n_classes=5)
    X = np.concatenate([x_ini[..., :-1], one_hot_ids], axis=-1)

    # Particle mask: True for real constituents, False for padding
    n_actual_constituents = ak.num(matched_cands, axis=2)  # (events, jets)
    n_actual_flat = ak.to_numpy(ak.flatten(n_actual_constituents, axis=1))

    particle_mask = np.zeros((X.shape[0], n_constituents), dtype=bool)
    for i in range(X.shape[0]):
        n_real = min(n_actual_flat[i], n_constituents)
        particle_mask[i, :n_real] = True

    # Valid-jet mask (not yet applied — caller decides)
    valid_jets = n_actual_flat >= min_constituents
    n_removed = (~valid_jets).sum()
    if n_removed > 0:
        print(f"  Removing {n_removed} jets with < {min_constituents} constituents")

    return X, particle_mask, valid_jets


def process_batch(
    config,
    collections_to_load=None,
    n_constituents=16,
    min_constituents=1,
    collection_key="l1barrelextpuppi",
    cluster_using_fastjet=True,
    cluster_dist_param=0.4,
):

    file_pattern = config["file_pattern"]
    tree_name = config["tree_name"]
    max_events = config["max_events"]

    if collections_to_load is None:
        collections_to_load = [
            config[collection_key]["collection_name"],
            "GenPart",
        ]

    events = load_and_prepare_data(
        file_pattern,
        tree_name,
        collections_to_load,
        max_events=max_events,
        correct_pt=False,
        CONFIG=config,
    )

    # --- Labels ---
    gen_b = select_gen_b_quarks_from_higgs(events)
    n_gen_b_before_cuts = ak.sum(ak.num(gen_b, axis=1))
    gen_b = apply_custom_cuts(gen_b, config, "gen", kinematic_only=True)
    n_gen_b_after_cuts = ak.sum(ak.num(gen_b, axis=1))
    print(
        f"  Gen b-quarks: {n_gen_b_before_cuts} -> {n_gen_b_after_cuts} after pT/eta cuts"
    )

    if cluster_using_fastjet:
        print(f"Clustering {collection_key}...")
        clustered_jets = cluster_candidates(
            events, config, collection_key, dist_param=cluster_dist_param
        )
        n_clustered_jets = ak.sum(ak.num(clustered_jets, axis=1))
        print(f"  Clustered jets (pT > 25 GeV, |eta| < 2.4): {n_clustered_jets}")

        # Sort jets by pT (descending)
        sorted_indices = ak.argsort(clustered_jets.pt, axis=1, ascending=False)
        l1_jets = clustered_jets[sorted_indices]

        # Get constituents (already attached and recovered by helper)
        matched_cands = l1_jets.constituents

        # Sort constituents by pT (standard for ParT inputs)
        const_pt_sort = ak.argsort(matched_cands.pt, axis=2, ascending=False)
        matched_cands = matched_cands[const_pt_sort]
    else:
        l1_col = config["l1ng"]["collection_name"]
        l1_puppi_col = "L1BarrelExtPuppi"

        l1_jets = events[l1_col]
        l1_jets = apply_custom_cuts(l1_jets, config, "l1ng", kinematic_only=True)
        l1_jets = l1_jets[ak.argsort(l1_jets.pt, axis=1, ascending=False, stable=True)]
        l1_puppi_cands = events[l1_puppi_col]
        l1_puppi_cands = apply_custom_cuts(
            l1_puppi_cands, config, "l1barrelextpuppi", kinematic_only=True
        )
        l1_puppi_cands = l1_puppi_cands[
            ak.argsort(l1_puppi_cands.pt, axis=1, ascending=False, stable=True)
        ]

        l1_jet_vecs = l1_jets.vector[:, :, None]
        l1_cand_vec = l1_puppi_cands.vector[:, None, :]

        dR_matrix_l1_puppi = l1_jet_vecs.deltaR(l1_cand_vec)
        in_cone = dR_matrix_l1_puppi < 0.4

        cands_broadcast, mask_broadcast = ak.broadcast_arrays(
            l1_puppi_cands[:, None, :], in_cone
        )
        matched_cands = cands_broadcast[mask_broadcast]
        matched_pt_sorted_idxs = ak.argsort(
            matched_cands.pt, axis=2, ascending=False, stable=True
        )
        matched_cands = matched_cands[matched_pt_sorted_idxs]

    # --- Extract features ---
    X, particle_mask, valid_jets = extract_features(
        l1_jets, matched_cands, n_constituents, min_constituents
    )

    is_pure_label, idx_closest_gen = get_purity_mask_cross_matched(
        gen_b, l1_jets, return_gen_idx=True
    )
    labels = ak.values_astype(is_pure_label, np.float32)
    Y = ak.to_numpy(ak.flatten(labels, axis=None))

    X = X[valid_jets]
    Y = Y[valid_jets]
    particle_mask = particle_mask[valid_jets]

    # Compute gen pT per reco jet: for matched (signal) jets use the
    # matched gen b-quark pT; for unmatched (background) jets use 0.
    gen_pt_lookup = gen_b.pt[idx_closest_gen]
    gen_pt_per_jet = ak.fill_none(
        ak.where(is_pure_label, gen_pt_lookup, 0.0), 0.0
    )
    gen_pt_flat = ak.to_numpy(ak.flatten(gen_pt_per_jet, axis=1))
    gen_pt_flat = gen_pt_flat[valid_jets]

    return X, Y, particle_mask, gen_pt_flat


def process_qcd_batch(
    config,
    qcd_weight,
    collections_to_load=None,
    n_constituents=16,
    min_constituents=1,
    collection_key="l1extpuppi",
    cluster_using_fastjet=True,
    cluster_dist_param=0.4,
    chunk_size=50_000,
):
    """
    Process a single QCD ROOT file in event chunks to limit memory usage.
    Clusters L1 candidates with anti-kT, cross-matches to gen b-quarks
    selected via statusFlags (isLastCopy & fromHardProcess).

    Jets matched to a gen b-quark are labelled y=1 (signal).
    Unmatched jets are labelled y=0 (background).
    All jets carry the QCD pT-bin cross-section weight.

    Parameters
    ----------
    config : dict
        Configuration dict (file_pattern, tree_name, max_events set for this batch).
    qcd_weight : float
        Cross-section weight for this QCD pT bin.
    collection_key : str
        Key for the L1 candidate collection (e.g. 'l1extpuppi').
    chunk_size : int
        Maximum number of events to process at once.

    Returns
    -------
    X : np.ndarray, shape (n_jets, n_constituents, n_features)
    Y : np.ndarray, shape (n_jets,)  — 1 for b-matched, 0 otherwise
    particle_mask : np.ndarray, shape (n_jets, n_constituents)
    qcd_weights_per_jet : np.ndarray, shape (n_jets,)
    """
    import uproot

    file_pattern = config["file_pattern"]
    tree_name = config["tree_name"]
    max_events = config["max_events"]

    if collections_to_load is None:
        collections_to_load = [
            config[collection_key]["collection_name"],
            "GenPart",
        ]

    # Determine total number of events in the file
    try:
        with uproot.open(f"{file_pattern}:{tree_name}") as tree:
            total_events = tree.num_entries
    except Exception:
        total_events = None

    if max_events is not None and total_events is not None:
        total_events = min(int(max_events), total_events)
    elif max_events is not None:
        total_events = int(max_events)

    # Fall back to loading everything if total_events is unknown
    if total_events is None:
        chunk_boundaries = [(None, None)]
    else:
        chunk_boundaries = [
            (start, min(start + chunk_size, total_events))
            for start in range(0, total_events, chunk_size)
        ]

    chunk_X, chunk_Y, chunk_mask, chunk_w = [], [], [], []

    for start, stop in chunk_boundaries:
        chunk_cfg = dict(config)
        chunk_cfg["max_events"] = stop  # entry_stop in uproot

        events = load_and_prepare_data(
            file_pattern,
            tree_name,
            collections_to_load,
            max_events=stop,
            correct_pt=False,
            CONFIG=chunk_cfg,
        )
        # Slice to the chunk window
        if start is not None and start > 0:
            events = events[start:]

        if len(events) == 0:
            continue

        print(
            f"  QCD chunk events {start}-{stop}  "
            f"({len(events)} events loaded)"
        )

        # --- Gen-level b-quarks via statusFlags for signal labelling ---
        gen_b_quarks = select_gen_b_quarks_by_status(events, config)

        # --- Cluster candidates ---
        if cluster_using_fastjet:
            clustered_jets = cluster_candidates(
                events, config, collection_key, dist_param=cluster_dist_param
            )
            sorted_indices = ak.argsort(clustered_jets.pt, axis=1, ascending=False)
            l1_jets = clustered_jets[sorted_indices]

            matched_cands = l1_jets.constituents
            const_pt_sort = ak.argsort(matched_cands.pt, axis=2, ascending=False)
            matched_cands = matched_cands[const_pt_sort]
        else:
            raise NotImplementedError(
                "Non-fastjet clustering not implemented for QCD batch"
            )

        # --- Extract features ---
        X, particle_mask, valid_jets = extract_features(
            l1_jets, matched_cands, n_constituents, min_constituents
        )

        # --- Cross-match to gen b-quarks ---
        # Jets matched to a gen b-quark → signal (y=1)
        # Unmatched jets → background (y=0)
        is_b_matched = get_purity_mask_cross_matched(gen_b_quarks, l1_jets)
        labels_flat = ak.to_numpy(
            ak.flatten(ak.values_astype(is_b_matched, np.float32), axis=None)
        )

        X = X[valid_jets]
        labels_flat = labels_flat[valid_jets]
        particle_mask = particle_mask[valid_jets]

        n_sig = int((labels_flat == 1.0).sum())
        n_bkg = int((labels_flat == 0.0).sum())
        n_total = len(labels_flat)
        print(
            f"    {n_total} valid jets: {n_sig} b-matched (signal), "
            f"{n_bkg} unmatched (background)"
        )

        if n_total > 0:
            chunk_X.append(X)
            chunk_Y.append(labels_flat)
            chunk_mask.append(particle_mask)
            chunk_w.append(np.full(n_total, qcd_weight, dtype=np.float32))

        # Free memory before next chunk
        del events, l1_jets, matched_cands, gen_b_quarks

    if len(chunk_X) == 0:
        empty_X = np.empty((0, n_constituents, 0), dtype=np.float32)
        return (
            empty_X,
            np.empty(0, dtype=np.float32),
            np.empty((0, n_constituents), dtype=bool),
            np.empty(0, dtype=np.float32),
        )

    return (
        np.concatenate(chunk_X, axis=0),
        np.concatenate(chunk_Y, axis=0),
        np.concatenate(chunk_mask, axis=0),
        np.concatenate(chunk_w, axis=0),
    )


def process_batch_for_higgs(
    config,
    collections_to_load=None,
    n_constituents=128,
    min_constituents=1,
    collection_key="l1barrelextpuppi",
    cluster_using_fastjet=True,
    cluster_dist_param=0.8,
):

    file_pattern = config["file_pattern"]
    tree_name = config["tree_name"]
    max_events = config["max_events"]

    if collections_to_load is None:
        collections_to_load = [
            config[collection_key]["collection_name"],
            "GenPart",
        ]

    events = load_and_prepare_data(
        file_pattern,
        tree_name,
        collections_to_load,
        max_events=max_events,
        correct_pt=False,
        CONFIG=config,
    )

    # --- Labels ---
    gen_higgs = select_gen_higgs(events)
    n_gen_higgs_before_cuts = ak.sum(ak.num(gen_higgs, axis=1))
    gen_higgs = apply_custom_cuts(gen_higgs, config, "gen", kinematic_only=True)
    n_gen_higgs_after_cuts = ak.sum(ak.num(gen_higgs, axis=1))
    print(
        f"  Gen Higgs: {n_gen_higgs_before_cuts} -> {n_gen_higgs_after_cuts} after pT/eta cuts"
    )

    if cluster_using_fastjet:
        print(f"Clustering {collection_key}...")
        clustered_jets = cluster_candidates(
            events, config, collection_key, dist_param=cluster_dist_param
        )
        n_clustered_jets = ak.sum(ak.num(clustered_jets, axis=1))
        print(f"  Clustered jets (pT > 25 GeV, |eta| < 2.4): {n_clustered_jets}")

        # Sort jets by pT (descending)
        sorted_indices = ak.argsort(clustered_jets.pt, axis=1, ascending=False)
        l1_jets = clustered_jets[sorted_indices]

        # Get constituents (already attached and recovered by helper)
        matched_cands = l1_jets.constituents

        # Sort constituents by pT (standard for ParT inputs)
        const_pt_sort = ak.argsort(matched_cands.pt, axis=2, ascending=False)
        matched_cands = matched_cands[const_pt_sort]
    else:
        l1_col = config["l1ng"]["collection_name"]
        l1_puppi_col = "L1BarrelExtPuppi"

        l1_jets = events[l1_col]
        l1_jets = apply_custom_cuts(l1_jets, config, "l1ng", kinematic_only=True)
        l1_jets = l1_jets[ak.argsort(l1_jets.pt, axis=1, ascending=False, stable=True)]
        l1_puppi_cands = events[l1_puppi_col]
        l1_puppi_cands = apply_custom_cuts(
            l1_puppi_cands, config, "l1barrelextpuppi", kinematic_only=True
        )
        l1_puppi_cands = l1_puppi_cands[
            ak.argsort(l1_puppi_cands.pt, axis=1, ascending=False, stable=True)
        ]

        l1_jet_vecs = l1_jets.vector[:, :, None]
        l1_cand_vec = l1_puppi_cands.vector[:, None, :]

        dR_matrix_l1_puppi = l1_jet_vecs.deltaR(l1_cand_vec)
        in_cone = dR_matrix_l1_puppi < 0.4

        cands_broadcast, mask_broadcast = ak.broadcast_arrays(
            l1_puppi_cands[:, None, :], in_cone
        )
        matched_cands = cands_broadcast[mask_broadcast]
        matched_pt_sorted_idxs = ak.argsort(
            matched_cands.pt, axis=2, ascending=False, stable=True
        )
        matched_cands = matched_cands[matched_pt_sorted_idxs]

    # --- Extract features ---
    X, particle_mask, valid_jets = extract_features(
        l1_jets, matched_cands, n_constituents, min_constituents
    )

    is_pure_label, idx_closest_gen = get_purity_mask_cross_matched(
        gen_higgs, l1_jets, return_gen_idx=True
    )
    labels = ak.values_astype(is_pure_label, np.float32)
    Y = ak.to_numpy(ak.flatten(labels, axis=None))

    X = X[valid_jets]
    Y = Y[valid_jets]
    particle_mask = particle_mask[valid_jets]

    # Compute gen pT per reco jet: matched (signal) jets get the
    # matched gen Higgs pT; unmatched (background) jets get 0.
    gen_pt_lookup = gen_higgs.pt[idx_closest_gen]
    gen_pt_per_jet = ak.fill_none(
        ak.where(is_pure_label, gen_pt_lookup, 0.0), 0.0
    )
    gen_pt_flat = ak.to_numpy(ak.flatten(gen_pt_per_jet, axis=1))
    gen_pt_flat = gen_pt_flat[valid_jets]

    return X, Y, particle_mask, gen_pt_flat


def generate_dataset_with_qcd_background(
    config_path,
    output_file="l1_qcd_bkg_training_data.npz",
    signal_data_dir="data/hh4b_puppi_pf/hh4b",
    collections_to_load=None,
    num_constituents=16,
    collection_key="l1extpuppi",
    min_constituents=1,
    on_signal=True,
    cluster_dist_param=0.4,
    flatten_spectrum=False,
):
    """
    Build a training dataset where:
      - Signal = clustered AK4 jets from HH→4b matched to gen b-quarks from Higgs
      - Background = clustered AK4 jets from QCD samples matched to gen quarks/gluons

    Each QCD pT-bin carries a cross-section weight so that the combined
    background reflects the true inclusive QCD spectrum.  Kinematic
    reweighting is applied to make signal kinematics match the
    QCD-weighted background distribution.

    Parameters
    ----------
    config_path : str
        Path to the JSON config (must contain 'QCD_background' section).
    output_file : str
        Output NPZ file.
    signal_data_dir : str
        Directory with HH→4b ROOT files.
    collections_to_load : list, optional
        Collections to read.  Defaults to [candidate_collection, 'GenPart'].
    num_constituents : int
        Constituents per jet.
    collection_key : str
        Config key for the L1 candidate collection (e.g. 'l1extpuppi').
    min_constituents : int
        Minimum real constituents a jet must have.
    on_signal : bool
        If True, reweight signal to match background kinematics.
        Ignored when flatten_spectrum=True.
    cluster_dist_param : float
        Anti-kT distance parameter.
    flatten_spectrum : bool
        If True, reweight both signal and background to flat pT-eta
        distributions.  QCD cross-section weights are ignored; the final
        stored weights are purely kinematic flattening weights.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    coll_list = collections_to_load or [
        config[collection_key]["collection_name"],
        "GenPart",
    ]

    # ================================================================
    # SIGNAL  –  from HH→4b
    # ================================================================
    print("=" * 60)
    print("Processing SIGNAL from HH→4b...")
    print("=" * 60)

    signal_root_files = sorted(
        [
            os.path.join(signal_data_dir, f)
            for f in os.listdir(signal_data_dir)
            if f.endswith(".root")
        ]
    )

    all_X_sig, all_y_sig, all_masks_sig, all_gen_pt_sig = [], [], [], []

    for root_file in tqdm(signal_root_files, desc="Signal files"):
        config["file_pattern"] = root_file
        X_chunk, y_chunk, mask_chunk, gen_pt_chunk = process_batch(
            config=config,
            collections_to_load=coll_list,
            n_constituents=num_constituents,
            min_constituents=min_constituents,
            collection_key=collection_key,
            cluster_dist_param=cluster_dist_param,
        )
        # Keep only signal jets (matched to gen b-quarks from Higgs)
        sig_sel = y_chunk == 1
        all_X_sig.append(X_chunk[sig_sel])
        all_y_sig.append(y_chunk[sig_sel])
        all_masks_sig.append(mask_chunk[sig_sel])
        all_gen_pt_sig.append(gen_pt_chunk[sig_sel])
        print(f"  Signal jets in batch: {int(sig_sel.sum())}")

    X_sig = np.concatenate(all_X_sig, axis=0)
    Y_sig = np.concatenate(all_y_sig, axis=0)
    mask_sig = np.concatenate(all_masks_sig, axis=0)
    gen_pt_sig = np.concatenate(all_gen_pt_sig, axis=0)
    print(f"\nTotal signal jets: {len(Y_sig)}")
    del all_X_sig, all_y_sig, all_masks_sig, all_gen_pt_sig  # free memory
    print(f"Deleted intermediate signal data to free memory.")

    # ================================================================
    # BACKGROUND  –  from QCD pT-binned samples
    # ================================================================
    print("\n" + "=" * 60)
    print("Processing BACKGROUND from QCD...")
    print("=" * 60)

    qcd_config = config["QCD_background"]
    all_X_qcd, all_y_qcd, all_masks_qcd, all_qcd_weights = [], [], [], []

    for bin_name, bin_cfg in qcd_config.items():
        print(f"\n--- QCD bin: {bin_name}  (weight={bin_cfg['weight']}) ---")

        qcd_files = sorted(glob_module.glob(bin_cfg["file_pattern"]))
        if not qcd_files:
            print(f"  Warning: No files found for {bin_name}, skipping.")
            continue

        for qcd_file in tqdm(qcd_files, desc=f"QCD {bin_name}"):
            qcd_cfg = dict(config)  # shallow copy
            qcd_cfg["file_pattern"] = qcd_file
            qcd_cfg["tree_name"] = bin_cfg["tree_name"]
            qcd_cfg["max_events"] = bin_cfg["max_events"]

            X_chunk, y_chunk, mask_chunk, w_chunk = process_qcd_batch(
                config=qcd_cfg,
                qcd_weight=bin_cfg["weight"],
                collections_to_load=coll_list,
                n_constituents=num_constituents,
                min_constituents=min_constituents,
                collection_key=collection_key,
                cluster_dist_param=cluster_dist_param,
            )

            if len(X_chunk) > 0:
                all_X_qcd.append(X_chunk)
                all_y_qcd.append(y_chunk)
                all_masks_qcd.append(mask_chunk)
                all_qcd_weights.append(w_chunk)
                n_sig_chunk = int((y_chunk == 1).sum())
                n_bkg_chunk = int((y_chunk == 0).sum())
                print(f"  QCD jets in batch: {len(X_chunk)} "
                      f"({n_sig_chunk} b-signal, {n_bkg_chunk} background)")
            del X_chunk, y_chunk, mask_chunk, w_chunk  # free memory before next file
    
    print(f"\nFinished processing QCD bins. Combining data...")
    X_qcd = np.concatenate(all_X_qcd, axis=0)
    del all_X_qcd  # free memory
    print(f"Deleted intermediate QCD data to free memory.")
    Y_qcd = np.concatenate(all_y_qcd, axis=0)
    del all_y_qcd
    print(f"Deleted intermediate QCD labels to free memory.")
    mask_qcd = np.concatenate(all_masks_qcd, axis=0)
    del all_masks_qcd
    print(f"Deleted intermediate QCD masks to free memory.")
    qcd_weights = np.concatenate(all_qcd_weights, axis=0)
    del all_qcd_weights
    print(f"Deleted intermediate QCD weights to free memory.")

    n_qcd_sig = int((Y_qcd == 1).sum())
    n_qcd_bkg = int((Y_qcd == 0).sum())
    print(f"\nTotal QCD jets: {len(Y_qcd)}")
    print(f"  b-matched (signal): {n_qcd_sig}")
    print(f"  unmatched (background): {n_qcd_bkg}")

    # ================================================================
    # COMBINE
    # ================================================================
    print("\n" + "=" * 60)
    print("Combining signal and background...")
    print("=" * 60)

    # Merge HH signal + QCD (signal & background)
    final_X = np.concatenate([X_sig, X_qcd], axis=0)
    final_y = np.concatenate([Y_sig, Y_qcd], axis=0)
    final_mask = np.concatenate([mask_sig, mask_qcd], axis=0)
    final_gen_pt = np.concatenate(
        [gen_pt_sig, np.zeros(len(Y_qcd), dtype=np.float32)], axis=0
    )

    # Per-jet sample weights:
    #   HH signal jets  → 1.0  (no cross-section reweighting needed)
    #   ALL QCD jets    → QCD pT-bin cross-section weight
    #     (both b-matched signal and unmatched background from QCD
    #      carry the same bin weight)
    sample_weights = np.concatenate(
        [np.ones(len(Y_sig), dtype=np.float32), qcd_weights], axis=0
    )

    print(f"\nCombined dataset:")
    print(f"  HH signal jets:       {len(Y_sig)}")
    print(f"  QCD b-signal jets:    {n_qcd_sig}")
    print(f"  QCD background jets:  {n_qcd_bkg}")
    print(f"  Total signal (y=1):   {int((final_y == 1).sum())}")
    print(f"  Total background (y=0): {int((final_y == 0).sum())}")

    # Jet kinematics from constituent 4-vectors
    final_4_vecs = vector.array(
        {
            "mass": final_X[:, :, 0],
            "pt": final_X[:, :, 1],
            "eta": final_X[:, :, 2],
            "phi": final_X[:, :, 3],
        }
    )
    final_jet_4_vecs = final_4_vecs.sum(axis=1)
    final_pt = final_jet_4_vecs.pt
    final_eta = final_jet_4_vecs.eta

    # ================================================================
    # KINEMATIC WEIGHTS
    # ================================================================
    print("Computing kinematic weights...")

    if flatten_spectrum:
        # Flat spectrum mode: reweight both signal and background to flat
        # pT-eta distributions.  QCD cross-section weights are NOT used.
        print("  Mode: flatten_spectrum (ignoring QCD xsec weights)")
        kinematic_weights = compute_kinematic_weights(
            final_pt,
            final_eta,
            final_y,
            flatten_both=True,
        )
        # Final weights are purely kinematic — no xsec multiplication
        raw_weights = kinematic_weights.copy()
    else:
        # Default mode: match signal kinematics to QCD-weighted background
        print("  Mode: match signal to background (using QCD xsec weights)")
        kinematic_weights = compute_kinematic_weights(
            final_pt,
            final_eta,
            final_y,
            on_signal=on_signal,
            sample_weights=sample_weights,
        )
        # Final training weights:
        #   signal  → kinematic_weight × sample_weight
        #     HH jets: kinematic_weight × 1.0
        #     QCD b-jets: kinematic_weight × qcd_xsec_weight
        #   background → 1.0 × qcd_xsec_weight  (kinematic=1.0 when on_signal=True)
        raw_weights = (kinematic_weights * sample_weights).astype(np.float32)

    # Normalize weights per class so each class has mean weight ≈ 1.
    sig_mask_final = final_y == 1
    bkg_mask_final = final_y == 0

    sig_mean = raw_weights[sig_mask_final].mean()
    bkg_mean = raw_weights[bkg_mask_final].mean()

    final_weights = raw_weights.copy()
    final_weights[sig_mask_final] /= sig_mean
    final_weights[bkg_mask_final] /= bkg_mean

    print(f"\nWeight normalization:")
    print(f"  Signal  — raw mean: {sig_mean:.4g}, normalized mean: {final_weights[sig_mask_final].mean():.4f}")
    print(f"  Bkg     — raw mean: {bkg_mean:.4g}, normalized mean: {final_weights[bkg_mask_final].mean():.4f}")
    print(f"  Bkg     — normalized min: {final_weights[bkg_mask_final].min():.4g}, max: {final_weights[bkg_mask_final].max():.4g}")

    print(f"\nSaving {final_X.shape} dataset to {output_file}...")
    print(f"  - Features shape:       {final_X.shape}")
    print(f"  - Labels shape:         {final_y.shape}")
    print(f"  - Particle mask shape:  {final_mask.shape}")
    print(f"  - Sample weights shape: {final_weights.shape}")
    print(f"  - Signal jets:          {int((final_y == 1).sum())}")
    print(f"  - Background jets:      {int((final_y == 0).sum())}")
    np.savez_compressed(
        output_file,
        x=final_X,
        y=final_y,
        mask=final_mask,
        jet_pt=final_pt,
        jet_eta=final_eta,
        weights=final_weights,
        weights_raw=raw_weights,
        gen_pt=final_gen_pt,
        qcd_weights=sample_weights,
    )


def generate_dataset(
    config_path,
    output_file="l1_training_data.npz",
    data_dir="~/data/hh4b_puppi_pf/hh4b",
    collections_to_load=None,
    num_constituents=16,
    collection_key="l1barrelextpuppi",
    min_constituents=1,
    on_signal=True,
    higgs_mode=False,
):
    """
    Main loop that processes the file in chunks and saves to disk incrementally.

    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file.
    output_file : str
        Output NPZ file name.
    data_dir : str
        Directory containing ROOT files.
    collections_to_load : list, optional
        List of collections to load from ROOT files.
    num_constituents : int
        Number of constituents per jet.
    collection_key : str
        Key for the candidate collection to use.
    min_constituents : int
        Minimum number of constituents required per jet. Jets with fewer
        constituents are removed to avoid duplicate padding-only samples.
    on_signal : bool
        Whether to apply kinematic reweighting to signal (True) or background (False).
    higgs_mode : bool
        Whether to process the dataset for Higgs tagging.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    root_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".root")
    ]
    root_files.sort()

    # Define branches to read (Optimization: Only read what we need)
    if collections_to_load is None:
        collections_to_load = [
            config["l1ng"]["collection_name"],  # Jets
            config[collection_key]["collection_name"],  # Candidates
            "GenPart",  # Labels
        ]

    all_X = []
    all_y = []
    all_masks = []
    all_gen_pt = []

    # Iterate over the file
    for root_file in tqdm(root_files):

        config["file_pattern"] = root_file
        if higgs_mode:
            X_chunk, y_chunk, mask_chunk, gen_pt_chunk = process_batch_for_higgs(
                config=config,
                collections_to_load=collections_to_load,
                n_constituents=num_constituents,
                min_constituents=min_constituents,
                collection_key=collection_key,
            )
        else:
            X_chunk, y_chunk, mask_chunk, gen_pt_chunk = process_batch(
                config=config,
                collections_to_load=collections_to_load,
                n_constituents=num_constituents,
                min_constituents=min_constituents,
                collection_key=collection_key,
            )

        all_X.append(X_chunk)
        all_y.append(y_chunk)
        all_masks.append(mask_chunk)
        all_gen_pt.append(gen_pt_chunk)
        print(f"  Processed batch: {len(X_chunk)} jets")

    # Final Concatenation
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    final_mask = np.concatenate(all_masks, axis=0)
    final_gen_pt = np.concatenate(all_gen_pt, axis=0)

    final_4_vecs = vector.array(
        {
            "mass": final_X[:, :, 0],
            "pt": final_X[:, :, 1],
            "eta": final_X[:, :, 2],
            "phi": final_X[:, :, 3],
        }
    )
    final_jet_4_vecs = final_4_vecs.sum(axis=1)
    final_pt = final_jet_4_vecs.pt
    final_eta = final_jet_4_vecs.eta

    print("Computing kinematic weights...")
    sample_weights = compute_kinematic_weights(
        final_pt, final_eta, final_y, on_signal=on_signal
    )

    print(f"Saving {final_X.shape} dataset to {output_file}...")
    print(f"  - Features shape: {final_X.shape}")
    print(f"  - Labels shape: {final_y.shape}")
    print(f"  - Particle mask shape: {final_mask.shape}")
    print(f"  - Sample weights shape: {sample_weights.shape}")
    np.savez_compressed(
        output_file,
        x=final_X,
        y=final_y,
        mask=final_mask,
        jet_pt=final_pt,
        jet_eta=final_eta,
        weights=sample_weights,
        gen_pt=final_gen_pt,
    )
    print("Done.")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(
        description="Generate L1 Jet Dataset for ParT Training"
    )
    argparse.add_argument(
        "--config",
        type=str,
        default="hh-bbbb-obj-config.json",
        help="Path to the configuration JSON file.",
    )
    argparse.add_argument(
        "--output",
        type=str,
        default="data/l1_ak4_training_data.npz",
        help="Output NPZ file name.",
    )
    argparse.add_argument(
        "--data_dir",
        type=str,
        default="data/hh4b_puppi_pf/hh4b",
        help="Directory containing ROOT files.",
    )
    argparse.add_argument(
        "--num_constituents",
        type=int,
        default=16,
        help="Number of constituents per jet.",
    )
    argparse.add_argument(
        "--collection_key",
        type=str,
        default="l1barrelextpuppi",
        help="Key for the L1 candidate collection to use.",
    )
    argparse.add_argument(
        "--on_signal",
        type=bool,
        default=True,
        help="Whether to apply kinematic reweighting to signal (True) or background (False).",
    )
    argparse.add_argument(
        "--higgs_mode",
        type=bool,
        default=False,
        help="Whether to process the dataset for Higgs tagging.",
    )
    argparse.add_argument(
        "--use_qcd_background",
        type=bool,
        default=False,
        help="Use QCD pT-binned samples as background instead of unmatched signal-file jets.",
    )
    argparse.add_argument(
        "--flatten_spectrum",
        type=bool,
        default=False,
        help="Reweight both signal and background to flat pT-eta distributions. "
             "QCD cross-section weights are ignored; final weights are purely kinematic.",
    )
    args = argparse.parse_args()

    print(f"Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    if args.use_qcd_background:
        generate_dataset_with_qcd_background(
            config_path=args.config,
            output_file=args.output,
            signal_data_dir=args.data_dir,
            num_constituents=args.num_constituents,
            collection_key=args.collection_key,
            min_constituents=1,
            on_signal=args.on_signal,
            flatten_spectrum=args.flatten_spectrum,
        )
    else:
        generate_dataset(
            config_path=args.config,
            output_file=args.output,
            data_dir=args.data_dir,
            num_constituents=args.num_constituents,
            collection_key=args.collection_key,
            min_constituents=1,
            on_signal=args.on_signal,
            higgs_mode=args.higgs_mode,
        )
