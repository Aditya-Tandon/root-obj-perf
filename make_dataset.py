import os
import json
import argparse

import numpy as np
import awkward as ak
import vector

from tqdm import tqdm
from data_loading_helpers import load_and_prepare_data, select_gen_b_quarks_from_higgs, apply_custom_cuts

# Register the vector library with awkward array
ak.behavior.update(vector.backends.awkward.behavior)

def process_batch(config, collections_to_load=None, n_constituents=16):

    file_pattern = config["file_pattern"]
    tree_name = config["tree_name"]
    max_events = config["max_events"]
    if collections_to_load is None:
        collections_to_load = [
            config["l1ng"]["collection_name"],      # Jets
            "L1BarrelExtPuppi",                     # Candidates
            "GenPart"                               # Labels
        ]

    events = load_and_prepare_data(file_pattern, tree_name, collections_to_load, max_events=max_events, correct_pt=False)

    l1_col = config["l1ng"]["collection_name"]
    l1_puppi_col = "L1BarrelExtPuppi"

    gen_b = select_gen_b_quarks_from_higgs(events)
    gen_b = apply_custom_cuts(gen_b, config, "gen", kinematic_only=True)

    l1_jets = events[l1_col]
    l1_jets = apply_custom_cuts(l1_jets, config, "l1ng", kinematic_only=True)
    l1_jets = l1_jets[ak.argsort(l1_jets.pt, axis=1, ascending=False, stable=True)]
    l1_puppi_cands = events[l1_puppi_col]
    l1_puppi_cands = apply_custom_cuts(l1_puppi_cands, config, "l1barrelextpuppi", kinematic_only=True)
    l1_puppi_cands = l1_puppi_cands[ak.argsort(l1_puppi_cands.pt, axis=1, ascending=False, stable=True)]

    l1_jet_vecs = l1_jets.vector[:, :, None]
    l1_cand_vec = l1_puppi_cands.vector[:, None, :]

    dR_matrix = l1_jet_vecs.deltaR(l1_cand_vec)
    in_cone = dR_matrix < 0.4

    cands_indices = ak.local_index(l1_puppi_cands, axis=1)
    indices_broadcast, mask_broadcast = ak.broadcast_arrays(cands_indices[:, None, :], in_cone)
    cand_idxs_matched = indices_broadcast[mask_broadcast] 

    cands_broadcast, mask_broadcast = ak.broadcast_arrays(l1_puppi_cands[:, None, :], in_cone)
    matched_cands = cands_broadcast[mask_broadcast] 
    matched_pt_sorted_idxs = ak.argsort(matched_cands.pt, axis=2, ascending=False, stable=True)
    matched_cands = matched_cands[matched_pt_sorted_idxs]

    j_pt = l1_jets.pt[:, :, None]
    j_eta = l1_jets.eta[:, :, None]
    j_phi = l1_jets.phi[:, :, None]

    m_pt = matched_cands.pt
    m_eta = matched_cands.eta
    m_phi = matched_cands.phi

    # 1. Log pT Rel
    log_pt_rel = np.log(np.maximum(m_pt, 1e-3) / np.maximum(j_pt, 1e-3))

    # 2. Delta Eta/Phi
    deta = m_eta - j_eta
    dphi = m_phi - j_phi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi # Wrap to [-pi, pi]

    # 3. Log Delta R
    log_dr = np.log(np.maximum(np.sqrt(deta**2 + dphi**2), 1e-3))

    # 4. impact parameter dxy and z0
    m_dxy = matched_cands.dxy
    m_z0 = matched_cands.z0

    # 5. puppi weight
    m_w = matched_cands.puppiWeight

    # 6. pdgId
    m_id = matched_cands.id

    # 7. charge
    m_charge = matched_cands.charge

    # 8. energy
    m_e = matched_cands.e


    def pad_and_fill(arr, target=n_constituents):
        # Pad axis 2 (constituents) and fill None with 0
        return ak.fill_none(ak.pad_none(arr, target, axis=2, clip=True), 0.0)

    feature_list = [
        pad_and_fill(log_pt_rel),
        pad_and_fill(log_dr),
        pad_and_fill(deta),
        pad_and_fill(dphi),
        pad_and_fill(m_dxy),
        pad_and_fill(m_z0),
        pad_and_fill(m_w),
        pad_and_fill(m_id),
        pad_and_fill(m_charge),
        pad_and_fill(m_e),
    ]

    # [n_jets, n_constituents, n_features] exactly because clip=True in pad_none
    X = np.stack([ak.to_numpy(ak.flatten(f, axis=1)) for f in feature_list], axis=-1)

    from analysis_helpers import get_purity_mask_cross_matched
    is_pure_label = get_purity_mask_cross_matched(gen_b, l1_jets)
    labels = ak.values_astype(is_pure_label, np.float32)
    Y = ak.to_numpy(ak.flatten(labels, axis=None))

    return X, Y

def generate_dataset(config_path, output_file="l1_training_data.npz", data_dir="/vols/cms/at3722/root-obj-perf/data/hh4b_puppi_pf/hh4b", collections_to_load=None):
    """
    Main loop that processes the file in chunks and saves to disk incrementally.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    root_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".root")]
    root_files.sort()
    
    # Define branches to read (Optimization: Only read what we need)
    if collections_to_load is None:
        collections_to_load = [
            config["l1ng"]["collection_name"],      # Jets
            "L1BarrelExtPuppi",                     # Candidates
            "GenPart"                               # Labels
        ]
    
    all_X = []
    all_y = []
    
    # Iterate over the file
    for root_file in tqdm(root_files):
        
        config["file_pattern"] = root_file
        X_chunk, y_chunk = process_batch(config=config, collections_to_load=collections_to_load, n_constituents=16)
        
        all_X.append(X_chunk)
        all_y.append(y_chunk)
        print(f"  Processed batch: {len(X_chunk)} jets")

    # Final Concatenation
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    
    print(f"Saving {final_X.shape} dataset to {output_file}...")
    np.savez_compressed(output_file, x=final_X, y=final_y)
    print("Done.")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Generate L1 Jet Dataset for ParT Training")
    argparse.add_argument("--config", type=str, default="hh-bbbb-obj-config.json", help="Path to the configuration JSON file.")
    argparse.add_argument("--output", type=str, default="l1_training_data.npz", help="Output NPZ file name.")
    argparse.add_argument("--data_dir", type=str, default="/vols/cms/at3722/root-obj-perf/data/hh4b_puppi_pf/hh4b", help="Directory containing ROOT files.")
    args = argparse.parse_args()
    generate_dataset(config_path=args.config, output_file=args.output, data_dir=args.data_dir)