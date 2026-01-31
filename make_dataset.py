import os
import json
import argparse

import numpy as np
import awkward as ak
import vector
import fastjet

from tqdm import tqdm
from data_loading_helpers import (
    load_and_prepare_data,
    select_gen_b_quarks_from_higgs,
    apply_custom_cuts,
    one_hot_encode_l1_puppi,
)
from analysis_helpers import get_purity_mask_cross_matched

# Register the vector library with awkward array
ak.behavior.update(vector.backends.awkward.behavior)


def compute_kinematic_weights(
    pt, eta, y, n_bins_pt=50, n_bins_eta=20, max_pt=500, clip_max=10.0, on_signal=True
):
    """
    Calculates weights to make Signal (y=1) kinematics match Background (y=0).
    Weight = P(k|B) / P(k|S) applied to Signal events.
    Background events get weight = 1.0.
    """
    # 1. Define Bins
    # Use fixed ranges to ensure stability.
    pt_bins = np.linspace(0, max_pt, n_bins_pt + 1)
    eta_bins = np.linspace(-2.0, 2.0, n_bins_eta + 1)

    # 2. Split Data
    sig_mask = y == 1
    bkg_mask = y == 0

    pt_sig, eta_sig = pt[sig_mask], eta[sig_mask]
    pt_bkg, eta_bkg = pt[bkg_mask], eta[bkg_mask]

    # 3. Compute 2D Histograms (Probability Densities)
    # density=True normalizes sum of integrals to 1
    # Adding a tiny epsilon (1e-10) prevents division by zero
    H_sig, _, _ = np.histogram2d(
        pt_sig, eta_sig, bins=[pt_bins, eta_bins], density=True
    )
    H_bkg, _, _ = np.histogram2d(
        pt_bkg, eta_bkg, bins=[pt_bins, eta_bins], density=True
    )

    H_sig = np.maximum(H_sig, 1e-10)
    H_bkg = np.maximum(H_bkg, 1e-10)

    # 4. Calculate Ratio Map: P(B) / P(S)
    # This is the weight we need to apply to Signal to make it look like Bkg
    if on_signal:
        ratio_map = np.divide(H_bkg, H_sig)
    else:
        ratio_map = np.divide(H_sig, H_bkg)

    # 5. Map Weights back to individual events
    # Find which bin each event falls into
    # We clip indices to ensure they stay within the histogram bounds
    pt_idx = np.searchsorted(pt_bins, pt) - 1
    eta_idx = np.searchsorted(eta_bins, eta) - 1

    pt_idx = np.clip(pt_idx, 0, n_bins_pt - 1)
    eta_idx = np.clip(eta_idx, 0, n_bins_eta - 1)

    # Lookup weights for all events
    all_weights = ratio_map[pt_idx, eta_idx]

    # 6. Apply Logic: Only reweight Signal or Background based on on_signal flag
    # If on_signal=True
    # If y=0 (Bkg), weight = 1.0
    # If y=1 (Sig), weight = Ratio
    # If on_signal=False, reverse
    final_weights = np.ones_like(y, dtype=np.float32)
    if on_signal:
        final_weights[sig_mask] = all_weights[sig_mask]
    else:
        final_weights[bkg_mask] = all_weights[bkg_mask]
    # 7. Clip Weights
    # If P(S) is very small in some bin, the weight becomes huge.
    # This destabilises training. Clipping (e.g., at 10) is standard practice.
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

    j_pt = l1_jets.pt[:, :, None]
    j_eta = l1_jets.eta[:, :, None]
    j_phi = l1_jets.phi[:, :, None]

    m_pt = matched_cands.pt
    m_eta = matched_cands.eta
    m_phi = matched_cands.phi

    # 1. 4-vec
    m_pt = matched_cands.vector.pt
    m_eta = matched_cands.vector.eta
    m_phi = matched_cands.vector.phi
    m_mass = matched_cands.vector.mass

    # 5. impact parameter dxy and z0
    m_dxy = matched_cands.dxy
    m_z0 = matched_cands.z0

    # 7. charge
    m_charge = matched_cands.charge

    # 8. log pT Rel
    log_pt_rel = np.log(np.maximum(m_pt, 1e-3) / np.maximum(j_pt, 1e-3))

    # 9. Delta Eta/Phi
    deta = m_eta - j_eta
    dphi = m_phi - j_phi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]

    # 11. puppi_weight
    m_w = matched_cands.puppiWeight

    # 12. Log Delta R
    log_dr = np.log(np.maximum(np.sqrt(deta**2 + dphi**2), 1e-3))

    # 13. Id
    m_id = matched_cands.id

    def pad_and_fill(arr, target=n_constituents):
        # Pad axis 2 (constituents) and fill None with 0
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

    # [n_jets, n_constituents, n_features] exactly because clip=True in pad_none
    x_ini = np.stack(
        [ak.to_numpy(ak.flatten(f, axis=1)) for f in feature_list], axis=-1
    )
    flat_ids = x_ini[..., -1]

    one_hot_ids = one_hot_encode_l1_puppi(flat_ids, n_classes=5)
    X = np.concatenate([x_ini[..., :-1], one_hot_ids], axis=-1)

    is_pure_label = get_purity_mask_cross_matched(gen_b, l1_jets)
    labels = ak.values_astype(is_pure_label, np.float32)
    Y = ak.to_numpy(ak.flatten(labels, axis=None))

    # Generate particle mask: True for real particles, False for padding
    # Count actual constituents per jet before padding
    n_actual_constituents = ak.num(matched_cands, axis=2)  # (events, jets)
    n_actual_flat = ak.to_numpy(ak.flatten(n_actual_constituents, axis=1))

    # Create mask: shape (n_jets, n_constituents)
    particle_mask = np.zeros((X.shape[0], n_constituents), dtype=bool)
    for i in range(X.shape[0]):
        n_real = min(n_actual_flat[i], n_constituents)
        particle_mask[i, :n_real] = True

    # Filter out jets with fewer than min_constituents
    valid_jets = n_actual_flat >= min_constituents
    n_removed = (~valid_jets).sum()
    if n_removed > 0:
        print(f"  Removing {n_removed} jets with < {min_constituents} constituents")

    X = X[valid_jets]
    Y = Y[valid_jets]
    particle_mask = particle_mask[valid_jets]

    return X, Y, particle_mask


def generate_dataset(
    config_path,
    output_file="l1_training_data.npz",
    data_dir="~/data/hh4b_puppi_pf/hh4b",
    collections_to_load=None,
    num_constituents=16,
    collection_key="l1barrelextpuppi",
    min_constituents=1,
    on_signal=True,
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
    min_constituents : int
        Minimum number of constituents required per jet. Jets with fewer
        constituents are removed to avoid duplicate padding-only samples.
    on_signal : bool
        Whether to apply kinematic reweighting to signal (True) or background (False).
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

    # Iterate over the file
    for root_file in tqdm(root_files):

        config["file_pattern"] = root_file
        X_chunk, y_chunk, mask_chunk = process_batch(
            config=config,
            collections_to_load=collections_to_load,
            n_constituents=num_constituents,
            min_constituents=min_constituents,
            collection_key=collection_key,
        )

        all_X.append(X_chunk)
        all_y.append(y_chunk)
        all_masks.append(mask_chunk)
        print(f"  Processed batch: {len(X_chunk)} jets")

    # Final Concatenation
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    final_mask = np.concatenate(all_masks, axis=0)

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
    args = argparse.parse_args()
    generate_dataset(
        config_path=args.config,
        output_file=args.output,
        data_dir=args.data_dir,
        num_constituents=args.num_constituents,
        collection_key=args.collection_key,
        min_constituents=1,
        on_signal=args.on_signal,
    )
