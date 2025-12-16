import json
import numpy as np
import awkward as ak
from sklearn.metrics import auc, roc_curve
from scipy.optimize import linear_sum_assignment


def get_efficiency_mask(gen_particles, reco_objects, CONFIG=None):
    """Returns a boolean mask for gen_particles, True if matched."""

    if CONFIG is None:
        with open("hh-bbbb-obj-config.json", "r") as config_file:
            CONFIG = json.load(config_file)
    
    gen_expanded = gen_particles.vector[:, :, None]
    reco_expanded = reco_objects.vector[:, None, :]
    delta_r_matrix = gen_expanded.deltaR(reco_expanded)
    min_delta_r = ak.min(delta_r_matrix, axis=2)
    is_matched = min_delta_r < CONFIG["matching_cone_size"]
    return ak.fill_none(is_matched, False)

def get_purity_mask(gen_particles, reco_objects, CONFIG=None):
    """Returns a boolean mask for reco_objects, True if matched."""
    if CONFIG is None:
        with open("hh-bbbb-obj-config.json", "r") as config_file:
            CONFIG = json.load(config_file)
    gen_expanded = gen_particles.vector[:, None, :]
    reco_expanded = reco_objects.vector[:, :, None]
    delta_r_matrix = reco_expanded.deltaR(gen_expanded)
    min_delta_r = ak.min(delta_r_matrix, axis=2)
    is_matched = min_delta_r < CONFIG["matching_cone_size"]
    return ak.fill_none(is_matched, False)

def get_purity_mask_one_one(gen_particles, reco_objects, CONFIG=None):
    """
    Mathes using 1-to-1 uniqueness without Hungarian Algorithm. 
    Loops explicitly implemented in the function.
    """

    if CONFIG is None:
        with open("hh-bbbb-obj-config.json", "r") as config_file:
            CONFIG = json.load(config_file)

    is_matched_list_gen = []
    is_matched_list_reco = []
    for i in range(len(gen_particles)):
        gen_vec = gen_particles[i].vector
        reco_vec = reco_objects[i].vector

        if len(gen_vec) == 0 or len(reco_vec) == 0:
            is_matched_list_gen.append(np.zeros(len(reco_vec), dtype=bool))
            continue

        matrix = reco_vec[:, None].deltaR(gen_vec[None, :])
        used_gen_indices = set()
        event_mask = np.zeros(len(reco_vec), dtype=bool)
        for reco_idx in range(len(reco_vec)):
            min_gen_idx = np.argmin(matrix[reco_idx, :])
            min_delta_r = matrix[reco_idx, min_gen_idx]
            if min_delta_r < CONFIG["matching_cone_size"] and min_gen_idx not in used_gen_indices:
                event_mask[reco_idx] = True
                used_gen_indices.add(min_gen_idx)
        is_matched_list_reco.append(event_mask)
        is_matched_list_gen.append(used_gen_indices)

    return ak.Array(is_matched_list_gen), ak.Array(is_matched_list_reco)

def get_purity_mask_cross_matched(gen_particles, reco_objects, CONFIG=None):

    if CONFIG is None:
        with open("hh-bbbb-obj-config.json", "r") as config_file:
            CONFIG = json.load(config_file)
    
    dr_matrix_reco_major = reco_objects[:, :, None].vector.deltaR(gen_particles[:, None, :].vector)
    # Find closest GEN quark for each RECO jet. Values are Gen Indices (or None if N_gen=0)
    idx_closest_gen_for_reco = ak.argmin(dr_matrix_reco_major, axis=2) # events, n_reco
    min_dr_reco_major = ak.fill_none(ak.min(dr_matrix_reco_major, axis=2), np.inf) # events, n_reco

    dr_matrix_gen_major = gen_particles[:, :, None].vector.deltaR(reco_objects[:, None, :].vector)
    # Find closest RECO jet for each GEN quark. Values are Reco Indices
    idx_closest_reco_for_gen = ak.argmin(dr_matrix_gen_major, axis=2) # events, n_gen

    # We are indexing (Events, N_gen) using indices of shape (Events, N_reco)
    # If idx_closest_gen_for_reco is None (no Gen quarks), this returns None safely.
    back_check_reco_idx = idx_closest_reco_for_gen[idx_closest_gen_for_reco]

    reco_idxs = ak.local_index(reco_objects, axis=1)
    # fill_none(-1) handles the cases where N_gen=0 (where back_check is None)
    # so they don't crash the comparison against integer Reco IDs.
    pure_cross_matched = (
        (ak.fill_none(back_check_reco_idx, -1) == reco_idxs) & 
        (min_dr_reco_major < CONFIG["matching_cone_size"])
    )
    
    return pure_cross_matched

def get_efficiency_mask_hungarian(gen_particles, reco_objects, CONFIG=None):
    """
    Matches using Hungarian Algorithm (1-to-1 uniqueness).
    Returns a boolean mask for gen_particles.
    """
    if CONFIG is None:
        with open("hh-bbbb-obj-config.json", "r") as config_file:
            CONFIG = json.load(config_file)

    is_matched_list = []
    for i in range(len(gen_particles)):
        gen_vec = gen_particles[i].vector
        reco_vec = reco_objects[i].vector
        
        if len(gen_vec) == 0 or len(reco_vec) == 0:
            is_matched_list.append(np.zeros(len(gen_vec), dtype=bool))
            continue
            
        # Shape: (N_gen, N_reco)
        # Note: vector library deltaR expects (N, 1) vs (1, M) broadcasting
        matrix = reco_vec[:, None].deltaR(gen_vec[None, :])
        
        # row_ind are indices in Reco, col_ind are indices in Gen
        row_ind, col_ind = linear_sum_assignment(matrix)
        
        valid_matches = matrix[row_ind, col_ind] < CONFIG["matching_cone_size"]
        event_mask = np.zeros(len(gen_vec), dtype=bool)
        
        # Set True only for gen indices that were assigned AND within cone
        event_mask[col_ind[valid_matches]] = True
        
        is_matched_list.append(event_mask)
        
    return ak.Array(is_matched_list)

def get_purity_mask_hungarian(gen_particles, reco_objects, CONFIG=None):
    """
    Matches using Hungarian Algorithm (1-to-1 uniqueness).
    Returns a boolean mask for reco_objects.
    """
    if CONFIG is None:
        with open("hh-bbbb-obj-config.json", "r") as config_file:
            CONFIG = json.load(config_file)

    is_matched_list = []
    for i in range(len(gen_particles)):
        gen_vec = gen_particles[i].vector
        reco_vec = reco_objects[i].vector
        
        if len(gen_vec) == 0 or len(reco_vec) == 0:
            is_matched_list.append(np.zeros(len(reco_vec), dtype=bool))
            continue
            
        # Shape: (N_reco, N_gen)
        # Note: vector library deltaR expects (N, 1) vs (1, M) broadcasting
        matrix = reco_vec[:, None].deltaR(gen_vec[None, :])
        
        # row_ind are indices in Reco, col_ind are indices in Gen
        row_ind, col_ind = linear_sum_assignment(matrix)
        
        valid_matches = matrix[row_ind, col_ind] < CONFIG["matching_cone_size"]
        event_mask = np.zeros(len(reco_vec), dtype=bool)
        
        # Set True only for reco indices that were assigned AND within cone
        event_mask[row_ind[valid_matches]] = True
        
        is_matched_list.append(event_mask)
        
    return ak.Array(is_matched_list)

def calculate_pur_eff_vs_variable(gen_particles, reco_objects, mask, variable, bins, is_purity_plot=False):
    """
    Calculates purity or efficiency vs. a kinematic variable for given reconstructed objects.
    Purity is defined as the fraction of reconstructed objects that are matched to a generated particle.
    Efficiency is defined as the fraction of generated particles that are matched to a reconstructed object.
    Returns the fraction and the error for each bin. The error is calculated using the Ullrich and Xu method.
    """

    if is_purity_plot:
        all_var = ak.to_numpy(ak.flatten(getattr(reco_objects, variable)))
        matched_var = ak.to_numpy(ak.flatten(getattr(reco_objects[mask], variable)))
    else:
        all_var = ak.to_numpy(ak.flatten(getattr(gen_particles, variable)))
        matched_var = ak.to_numpy(ak.flatten(getattr(gen_particles[mask], variable)))
    
    h_total, _ = np.histogram(all_var, bins=bins)
    h_matched, _ = np.histogram(matched_var, bins=bins)
    
    frac_offline = np.divide(h_matched, h_total, out=np.zeros_like(h_total, dtype=float), where=h_total!=0)
    err_offline = np.sqrt(((h_matched + 1) * (h_total - h_matched + 1)) / ((h_total + 2)**2 * (h_total + 3)))  # Ullrich and Xu

    return frac_offline, err_offline

def calculate_roc_points(reco_jets, is_pure_mask, tagger_name):
    """
    Calculates efficiency and mistag points for a ROC curve using the roc_curve function from sklearn.
    Returns (mistag_points, efficiency_points, auc_score, thresholds).
    """
    sig_scores = ak.to_numpy(ak.flatten(getattr(reco_jets[is_pure_mask], tagger_name)))
    sig_labels = np.ones_like(sig_scores)
    
    # Get Background scores (Truth = 0)
    bkg_scores = ak.to_numpy(ak.flatten(getattr(reco_jets[~is_pure_mask], tagger_name)))
    bkg_labels = np.zeros_like(bkg_scores)
    
    y_true = np.concatenate([sig_labels, bkg_labels])
    y_scores = np.concatenate([sig_scores, bkg_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, auc_score, thresholds

def get_pure_jet_idxs_cross_matched(gen_particles, reco_objects, CONFIG=None):
    """
    Returns the indices of the matched reco_objects for each gen_particle using cross-matching.
    
    gen_particles: generated b-jets
    reco_objects: reconstructed b-jets
    CONFIG: Configuration dictionary
    """
    if CONFIG is None:
        with open("hh-bbbb-obj-config.json", "r") as config_file:
            CONFIG = json.load(config_file)
    
    gen_vec = gen_particles.vector[:, :, None]
    reco_vec = reco_objects.vector[:, None, :]
    
    delta_r_matrix = gen_vec.deltaR(reco_vec)
    # shape: (events, n_gen)
    idx_closest_reco_to_gen = ak.argmin(delta_r_matrix, axis=2)
    # shape: (events, n_reco)
    idx_closest_gen_to_reco = ak.argmin(delta_r_matrix, axis=1)
    # Get the index of the Gen particle that the closest Reco jet points to
    back_check_idx = idx_closest_gen_to_reco[idx_closest_reco_to_gen]
    # Create indices for comparison (0, 1, 2...)
    gen_indices = ak.local_index(gen_particles, axis=1)
    min_dr_values = ak.min(delta_r_matrix, axis=2)
    is_mutual_match = (
        (back_check_idx == gen_indices) & 
        (min_dr_values < CONFIG["matching_cone_size"])
    )
    
    matched_reco_indices = idx_closest_reco_to_gen[is_mutual_match]
    
    return matched_reco_indices
