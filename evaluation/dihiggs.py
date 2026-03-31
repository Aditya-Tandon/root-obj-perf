"""Di-Higgs (HH -> 4b) reconstruction: D_HH pairing, gen-matching, and significance.

Supports both rectangular (m_HH window) and circular (R_HH radius) signal regions.
"""

import numpy as np
import awkward as ak

try:
    import vector
    ak.behavior.update(vector.backends.awkward.behavior)
except Exception:
    pass


def pair_from_4jets(js4):
    """D_HH minimisation on 4-jet events.

    Tries all three pair permutations and picks the one minimising
    D_HH = |m1 - (125/120)*m2| / sqrt(1 + (125/120)^2).

    Parameters
    ----------
    js4 : awkward array (Events, 4)
        Must have a ``vector`` field supporting addition and ``.mass`` / ``.pt``.

    Returns
    -------
    lead, sub, hh : awkward 4-vectors
        Leading/subleading Higgs candidates (ordered by pT) and the di-Higgs system.
    """
    j = [js4[:, i] for i in range(4)]
    perm_pairs = [([0, 1], [2, 3]), ([0, 2], [1, 3]), ([0, 3], [1, 2])]
    h_vecs = [
        (j[a].vector + j[b].vector, j[c].vector + j[d].vector)
        for (a, b), (c, d) in perm_pairs
    ]
    m1 = ak.concatenate([v[0].mass[:, None] for v in h_vecs], axis=1)
    m2 = ak.concatenate([v[1].mass[:, None] for v in h_vecs], axis=1)
    d_hh = abs(m1 - (125.0 / 120.0) * m2) / np.sqrt(1 + (125.0 / 120.0) ** 2)
    best = ak.argmin(d_hh, axis=1)

    c0, c1 = (best == 0), (best == 1)
    raw_h1 = ak.where(c0, h_vecs[0][0], ak.where(c1, h_vecs[1][0], h_vecs[2][0]))
    raw_h2 = ak.where(c0, h_vecs[0][1], ak.where(c1, h_vecs[1][1], h_vecs[2][1]))

    is_lead = raw_h1.pt >= raw_h2.pt
    lead = ak.where(is_lead, raw_h1, raw_h2)
    sub = ak.where(is_lead, raw_h2, raw_h1)
    hh = lead + sub
    return lead, sub, hh


def find_gen_b_pairs_with_indices(gmpr, gen_b_quarks, gen_particles):
    """Group matched gen b-quarks into pairs by Higgs parent.

    Sorts pairs so h1 is the leading-pT Higgs and h2 is subleading.

    Parameters
    ----------
    gmpr : awkward array (Events, 4)
        Indices into ``gen_b_quarks`` for matched reco jets.
    gen_b_quarks : awkward array
        Gen b-quark collection (filtered to >= 4 per event).
    gen_particles : awkward array
        Full GenPart collection (filtered same as gen_b_quarks).

    Returns
    -------
    h1_bs, h2_bs : awkward arrays (Events, 2) of GenParticle objects
    h1_local_idxs, h2_local_idxs : awkward arrays (Events, 2) of indices
    """
    matched_bs = gen_b_quarks[gmpr]  # Shape: (Events, 4)
    parent_indices = matched_bs.genPartIdxMother  # Shape: (Events, 4)

    sorter = ak.argsort(parent_indices, axis=1)
    sorted_bs = matched_bs[sorter]
    sorted_parents = parent_indices[sorter]

    pair_A_bs = sorted_bs[:, 0:2]
    pair_B_bs = sorted_bs[:, 2:4]

    parent_A_idx = sorted_parents[:, 0:1]
    parent_B_idx = sorted_parents[:, 2:3]
    parent_A = gen_particles[parent_A_idx][:, 0]
    parent_B = gen_particles[parent_B_idx][:, 0]

    is_A_leading = parent_A.pt > parent_B.pt
    h1_bs = ak.where(is_A_leading, pair_A_bs, pair_B_bs)
    h2_bs = ak.where(is_A_leading, pair_B_bs, pair_A_bs)

    sorted_gmpr = gmpr[sorter]
    pair_A_gmpr = sorted_gmpr[:, 0:2]
    pair_B_gmpr = sorted_gmpr[:, 2:4]
    h1_local_idxs = ak.where(is_A_leading, pair_A_gmpr, pair_B_gmpr)
    h2_local_idxs = ak.where(is_A_leading, pair_B_gmpr, pair_A_gmpr)

    return h1_bs, h2_bs, h1_local_idxs, h2_local_idxs


def R_hh_func(mh1, mh2, center1=125.0, center2=120.0):
    """Circular distance in the (m_H1, m_H2) plane from the expected Higgs masses."""
    return np.sqrt((mh1 - center1) ** 2 + (mh2 - center2) ** 2)


def compute_significance(
    sig_mh1, sig_mh2,
    bkg_mh1, bkg_mh2,
    sig_weights=None,
    bkg_weights=None,
    region="rectangular",
    rect_window=(250, 500),
    r_hh_cut=55.0,
    mh_centers=(125.0, 120.0),
):
    """Compute S/sqrt(S+B) for a given signal region.

    Takes per-Higgs masses so either signal-region variant can be computed
    without re-running the matching/pairing step.

    Parameters
    ----------
    sig_mh1, sig_mh2 : array-like
        Leading/subleading Higgs masses for signal events.
    bkg_mh1, bkg_mh2 : array-like
        Leading/subleading Higgs masses for background events.
    sig_weights, bkg_weights : array-like, optional
        Per-event weights. Defaults to unit weights.
    region : {"rectangular", "circular"}
        Signal region type.
    rect_window : tuple (low, high)
        For rectangular: window on m_HH = m_H1 + m_H2.
    r_hh_cut : float
        For circular: R_HH < r_hh_cut.
    mh_centers : tuple (c1, c2)
        For circular: centre of the ellipse in (m_H1, m_H2) plane.

    Returns
    -------
    dict with keys: S, B, significance, n_sig_in_region, n_bkg_in_region, region_desc
    """
    sig_mh1 = np.asarray(sig_mh1, dtype=np.float64)
    sig_mh2 = np.asarray(sig_mh2, dtype=np.float64)
    bkg_mh1 = np.asarray(bkg_mh1, dtype=np.float64)
    bkg_mh2 = np.asarray(bkg_mh2, dtype=np.float64)

    if sig_weights is None:
        sig_weights = np.ones(len(sig_mh1))
    else:
        sig_weights = np.asarray(sig_weights, dtype=np.float64)
    if bkg_weights is None:
        bkg_weights = np.ones(len(bkg_mh1))
    else:
        bkg_weights = np.asarray(bkg_weights, dtype=np.float64)

    if region == "rectangular":
        low, high = rect_window
        sig_mhh = sig_mh1 + sig_mh2
        bkg_mhh = bkg_mh1 + bkg_mh2
        sig_mask = (sig_mhh >= low) & (sig_mhh <= high)
        bkg_mask = (bkg_mhh >= low) & (bkg_mhh <= high)
        region_desc = f"rectangular m_HH in [{low}, {high}] GeV"
    elif region == "circular":
        c1, c2 = mh_centers
        sig_r = R_hh_func(sig_mh1, sig_mh2, center1=c1, center2=c2)
        bkg_r = R_hh_func(bkg_mh1, bkg_mh2, center1=c1, center2=c2)
        sig_mask = sig_r < r_hh_cut
        bkg_mask = bkg_r < r_hh_cut
        region_desc = f"circular R_HH < {r_hh_cut} GeV (centers {c1}, {c2})"
    else:
        raise ValueError(f"Unknown region type: {region!r}. Use 'rectangular' or 'circular'.")

    S = float(np.sum(sig_weights[sig_mask]))
    B = float(np.sum(bkg_weights[bkg_mask]))
    significance = S / np.sqrt(S + B) if (S + B) > 0 else 0.0

    return {
        "S": S,
        "B": B,
        "significance": significance,
        "n_sig_in_region": int(np.sum(sig_mask)),
        "n_bkg_in_region": int(np.sum(bkg_mask)),
        "region_desc": region_desc,
    }


def compute_significance_at_luminosity(
    sig_mh1, sig_mh2,
    bkg_mh1, bkg_mh2,
    bkg_raw_weights,
    sigma_to_ngen,
    n_gen_signal=None,
    luminosity_fb=1000.0,
    signal_xsec_pb=0.0113,
    convention="c",
    **kwargs,
):
    """Compute significance with proper luminosity scaling.

    Convenience wrapper around ``compute_significance`` that handles
    converting raw cross-section weights to expected event counts.

    Parameters
    ----------
    sig_mh1, sig_mh2 : array-like
        Leading/subleading Higgs masses for signal events.
    bkg_mh1, bkg_mh2 : array-like
        Leading/subleading Higgs masses for background events.
    bkg_raw_weights : array-like
        Raw QCD weights (Convention C: each event = sigma_bin).
    sigma_to_ngen : dict
        Maps sigma_bin (float) -> n_gen (int) per QCD pT bin.
    n_gen_signal : int or None
        Total generated signal events. If None, uses len(sig_mh1).
    luminosity_fb : float
        Target integrated luminosity in fb^-1.
    signal_xsec_pb : float
        Signal cross-section in pb.
    convention: str
        Weighting convention for bkg_raw_weights. 
        "a" = per-bin weights (not implemented yet).
        "b" = per-event weights stored without luminosity scaling. Use with event-level classifiers. 
        "c" = only cross-section are stored. Need to convert to per-event weights using sigma_to_ngen and luminosity.
            Use with legacy b-tagging datasets currently used for all models in this repo.
    **kwargs
        Passed to ``compute_significance`` (region, rect_window, r_hh_cut, mh_centers).

    Returns
    -------
    dict — same as ``compute_significance``
    """
    from evaluation.luminosity import signal_weight, scale_qcd_weights_raw, scale_qcd_weights_per_event

    if convention == "c":
        sig_w = signal_weight(
            len(sig_mh1), luminosity_fb, signal_xsec_pb, n_gen_signal,
        )
        bkg_w = scale_qcd_weights_raw(
            np.asarray(bkg_raw_weights, dtype=np.float64), sigma_to_ngen, luminosity_fb,
        )
    elif convention == "a":
        raise NotImplementedError("Convention A (per-bin weights) not implemented yet.")
    elif convention == "b":
        sig_w = signal_weight(
            len(sig_mh1), luminosity_fb, signal_xsec_pb, n_gen_signal,
        )
        bkg_w = scale_qcd_weights_per_event(np.asarray(bkg_raw_weights, dtype=np.float64), luminosity_fb)
    return compute_significance(
        sig_mh1, sig_mh2, bkg_mh1, bkg_mh2,
        sig_weights=sig_w, bkg_weights=bkg_w, **kwargs,
    )


def cluster_event_constituents(
    constituent_features,
    constituent_mask=None,
    jet_R=0.4,
    min_jet_pt=25.0,
    min_constituent_pt=1e-3,
):
    """Cluster one event into anti-kt jets from constituent-level 4-vector features.

    Parameters
    ----------
    constituent_features : array-like, shape (n_constituents, n_features)
        Per-constituent features. Columns 0..3 must be [mass, pt, eta, phi].
    constituent_mask : array-like[bool], optional
        Valid-constituent mask. If None, constituents with pt > min_constituent_pt are used.
    jet_R : float
        Anti-kt distance parameter.
    min_jet_pt : float
        Minimum jet pT for ``inclusive_jets``.
    min_constituent_pt : float
        Minimum constituent pT used when building pseudojets.

    Returns
    -------
    dict
        Keys: pt, eta, phi, mass (numpy arrays), sorted by descending jet pT.
    """
    try:
        import fastjet
    except Exception as exc:
        raise ImportError(
            "fastjet is required for constituent-level jet clustering"
        ) from exc

    feats = np.asarray(constituent_features)
    if feats.ndim != 2 or feats.shape[1] < 4:
        raise ValueError(
            "constituent_features must have shape (n_constituents, n_features>=4)"
        )

    if constituent_mask is None:
        valid = feats[:, 1] > min_constituent_pt
    else:
        mask = np.asarray(constituent_mask).astype(bool)
        if mask.ndim != 1 or mask.shape[0] != feats.shape[0]:
            raise ValueError(
                "constituent_mask must be 1D with same length as constituent_features"
            )
        valid = mask & (feats[:, 1] > min_constituent_pt)

    valid_feats = feats[valid]
    if valid_feats.shape[0] == 0:
        return {
            "pt": np.array([], dtype=np.float64),
            "eta": np.array([], dtype=np.float64),
            "phi": np.array([], dtype=np.float64),
            "mass": np.array([], dtype=np.float64),
        }

    masses = valid_feats[:, 0].astype(np.float64)
    pts = valid_feats[:, 1].astype(np.float64)
    etas = valid_feats[:, 2].astype(np.float64)
    phis = valid_feats[:, 3].astype(np.float64)

    px = pts * np.cos(phis)
    py = pts * np.sin(phis)
    pz = pts * np.sinh(etas)
    energies = np.sqrt((pts * np.cosh(etas)) ** 2 + np.maximum(masses, 0.0) ** 2)

    pseudojets = []
    for idx, (px_i, py_i, pz_i, e_i) in enumerate(zip(px, py, pz, energies)):
        pj = fastjet.PseudoJet(float(px_i), float(py_i), float(pz_i), float(e_i))
        pj.set_user_index(int(idx))
        pseudojets.append(pj)

    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, float(jet_R))
    cluster = fastjet.ClusterSequence(pseudojets, jet_def)
    jets = fastjet.sorted_by_pt(cluster.inclusive_jets(float(min_jet_pt)))

    return {
        "pt": np.asarray([j.pt() for j in jets], dtype=np.float64),
        "eta": np.asarray([j.eta() for j in jets], dtype=np.float64),
        "phi": np.asarray([j.phi() for j in jets], dtype=np.float64),
        "mass": np.asarray([j.m() for j in jets], dtype=np.float64),
    }


def reconstruct_dihiggs_from_constituents(
    constituents,
    masks=None,
    top_k=4,
    jet_R=0.4,
    min_jet_pt=25.0,
    min_constituent_pt=1e-3,
):
    """Reconstruct di-Higgs masses from event-level constituent tensors.

    Each event is clustered to jets with anti-kt, jets are sorted by pT,
    the top ``top_k`` jets are retained, and HH candidates are built with
    ``pair_from_4jets``.

    Parameters
    ----------
    constituents : array-like, shape (n_events, n_constituents, n_features)
        Event-level constituent feature tensor, first 4 columns [mass, pt, eta, phi].
    masks : array-like[bool], optional, shape (n_events, n_constituents)
        Valid-constituent mask.
    top_k : int
        Number of leading jets to use for HH reconstruction (default: 4).
    jet_R : float
        Anti-kt distance parameter.
    min_jet_pt : float
        Minimum jet pT for inclusive jet selection.
    min_constituent_pt : float
        Minimum constituent pT entering clustering.

    Returns
    -------
    dict
        Keys: event_indices, selected_mask, n_jets_per_event, lead_m, sub_m, hh_m.
    """
    arr = np.asarray(constituents)
    if arr.ndim != 3 or arr.shape[2] < 4:
        raise ValueError(
            "constituents must have shape (n_events, n_constituents, n_features>=4)"
        )

    if masks is not None:
        mask_arr = np.asarray(masks).astype(bool)
        if mask_arr.shape[:2] != arr.shape[:2]:
            raise ValueError(
                "masks must have shape (n_events, n_constituents) matching constituents"
            )
    else:
        mask_arr = None

    top_pts, top_etas, top_phis, top_masses = [], [], [], []
    selected_indices = []
    n_jets_per_event = np.zeros(arr.shape[0], dtype=np.int32)

    for i_evt in range(arr.shape[0]):
        evt_mask = mask_arr[i_evt] if mask_arr is not None else None
        jets = cluster_event_constituents(
            arr[i_evt],
            constituent_mask=evt_mask,
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            min_constituent_pt=min_constituent_pt,
        )
        n_jets = int(len(jets["pt"]))
        n_jets_per_event[i_evt] = n_jets
        if n_jets < top_k:
            continue

        selected_indices.append(i_evt)
        top_pts.append(jets["pt"][:top_k])
        top_etas.append(jets["eta"][:top_k])
        top_phis.append(jets["phi"][:top_k])
        top_masses.append(jets["mass"][:top_k])

    selected_mask = np.zeros(arr.shape[0], dtype=bool)
    if len(selected_indices) > 0:
        selected_mask[np.asarray(selected_indices, dtype=np.int32)] = True

    if len(selected_indices) == 0:
        return {
            "event_indices": np.array([], dtype=np.int32),
            "selected_mask": selected_mask,
            "n_jets_per_event": n_jets_per_event,
            "lead_m": np.array([], dtype=np.float64),
            "sub_m": np.array([], dtype=np.float64),
            "hh_m": np.array([], dtype=np.float64),
        }

    js4 = ak.zip(
        {
            "pt": ak.Array(top_pts),
            "eta": ak.Array(top_etas),
            "phi": ak.Array(top_phis),
            "mass": ak.Array(top_masses),
        }
    )
    js4 = ak.with_field(
        js4,
        ak.zip(
            {
                "pt": js4.pt,
                "eta": js4.eta,
                "phi": js4.phi,
                "mass": js4.mass,
            },
            with_name="Momentum4D",
        ),
        "vector",
    )

    lead, sub, hh = pair_from_4jets(js4)
    return {
        "event_indices": np.asarray(selected_indices, dtype=np.int32),
        "selected_mask": selected_mask,
        "n_jets_per_event": n_jets_per_event,
        "lead_m": ak.to_numpy(lead.mass),
        "sub_m": ak.to_numpy(sub.mass),
        "hh_m": ak.to_numpy(hh.mass),
    }


def reconstruct_dihiggs(
    reco_jets_collection,
    gen_b_quarks,
    events,
    config,
    config_key,
    attr_to_sort_jets=None,
    top_k=4,
):
    """Full di-Higgs reconstruction pipeline for a given jet collection (signal only).

    Parameters
    ----------
    reco_jets_collection : awkward array
        Reco jet collection with vector + tagger score fields.
    gen_b_quarks : awkward array
        Gen b-quarks from Higgs decay (pt/eta cuts already applied).
    events : awkward array
        Full event record (needs .GenPart for parent-matching).
    config : dict
        Analysis config (needs config_key sub-dict and matching_cone_size).
    config_key : str
        Key into config for tagger/collection info.
    attr_to_sort_jets : str, optional
        Attribute to sort jets by (defaults to tagger_name).
    top_k : int
        Number of leading jets to consider (default 4).

    Returns
    -------
    dict with keys: sig_lead, sig_sub, sig_hh, n_signal, n_total, pair_eff, signal_mask
    """
    subcfg = config[config_key]
    tagger_name = subcfg["tagger_name"]
    if attr_to_sort_jets is None:
        attr_to_sort_jets = tagger_name

    jets_btag = reco_jets_collection[
        ak.argsort(reco_jets_collection[attr_to_sort_jets], ascending=False)
    ]
    sig_jets_all = jets_btag[:, :top_k]

    # Cross-match signal jets for purity
    dr_reco = sig_jets_all[:, :, None].vector.deltaR(gen_b_quarks[:, None, :].vector)
    idx_gen_for_reco = ak.argmin(dr_reco, axis=2)
    min_dr_reco = ak.fill_none(ak.min(dr_reco, axis=2), np.inf)

    dr_gen = gen_b_quarks[:, :, None].vector.deltaR(sig_jets_all[:, None, :].vector)
    idx_reco_for_gen = ak.argmin(dr_gen, axis=2)

    back_check = idx_reco_for_gen[idx_gen_for_reco]
    reco_idx = ak.local_index(sig_jets_all, axis=1)

    pure_mask = (ak.fill_none(back_check, -1) == reco_idx) & (
        min_dr_reco < config["matching_cone_size"]
    )

    has_k = ak.num(sig_jets_all) >= top_k
    signal_mask = has_k & (ak.sum(pure_mask, axis=1) == top_k)

    n_total = int(ak.sum(has_k))
    n_signal = int(ak.sum(signal_mask))

    sig_jets_4 = sig_jets_all[signal_mask][:, :4]
    if n_signal > 0:
        sig_lead, sig_sub, sig_hh = pair_from_4jets(sig_jets_4)
    else:
        sig_lead = sig_sub = sig_hh = ak.Array([])

    # Pairing efficiency
    gen_sig = gen_b_quarks[signal_mask]
    genpart_sig = events.GenPart[signal_mask]
    idx_gen_sig = idx_gen_for_reco[signal_mask]
    pure_sig = pure_mask[signal_mask]

    gmpr_sig = idx_gen_sig[pure_sig]
    gmpr_sig = ak.drop_none(gmpr_sig)
    mask4_sig = ak.num(gmpr_sig) >= 4

    pair_eff = np.nan
    if ak.sum(mask4_sig) > 0:
        gmpr_4 = gmpr_sig[mask4_sig]
        gen_4 = gen_sig[mask4_sig]
        genpart_4 = genpart_sig[mask4_sig]

        _, _, h1_idx, h2_idx = find_gen_b_pairs_with_indices(gmpr_4, gen_4, genpart_4)
        h1_s = ak.sort(h1_idx, axis=1)
        h2_s = ak.sort(h2_idx, axis=1)

        js4_eff = sig_jets_4[mask4_sig]
        j_eff = [js4_eff[:, i] for i in range(4)]
        perm_pairs_eff = [([0, 1], [2, 3]), ([0, 2], [1, 3]), ([0, 3], [1, 2])]
        gen_pairs_eff = [
            (gmpr_4[:, [a, b]], gmpr_4[:, [c, d]]) for (a, b), (c, d) in perm_pairs_eff
        ]

        h_vecs_eff = [
            (j_eff[a].vector + j_eff[b].vector, j_eff[c].vector + j_eff[d].vector)
            for (a, b), (c, d) in perm_pairs_eff
        ]
        m1_eff = ak.concatenate([v[0].mass[:, None] for v in h_vecs_eff], axis=1)
        m2_eff = ak.concatenate([v[1].mass[:, None] for v in h_vecs_eff], axis=1)
        d_eff = abs(m1_eff - (125.0 / 120.0) * m2_eff) / np.sqrt(1 + (125.0 / 120.0) ** 2)
        best_eff = ak.argmin(d_eff, axis=1)
        c0e, c1e = (best_eff == 0), (best_eff == 1)

        algo_A = ak.where(
            c0e[:, None],
            gen_pairs_eff[0][0],
            ak.where(c1e[:, None], gen_pairs_eff[1][0], gen_pairs_eff[2][0]),
        )
        algo_B = ak.where(
            c0e[:, None],
            gen_pairs_eff[0][1],
            ak.where(c1e[:, None], gen_pairs_eff[1][1], gen_pairs_eff[2][1]),
        )

        algo_A_s = ak.sort(algo_A, axis=1)
        algo_B_s = ak.sort(algo_B, axis=1)

        correct_A = ak.all(algo_A_s == h1_s, axis=1) | ak.all(algo_A_s == h2_s, axis=1)
        correct_B = ak.all(algo_B_s == h1_s, axis=1) | ak.all(algo_B_s == h2_s, axis=1)
        correct = correct_A & correct_B
        pair_eff = float(ak.mean(correct))

    return {
        "sig_lead": sig_lead,
        "sig_sub": sig_sub,
        "sig_hh": sig_hh,
        "n_signal": n_signal,
        "n_total": n_total,
        "pair_eff": pair_eff,
        "signal_mask": signal_mask,
    }
