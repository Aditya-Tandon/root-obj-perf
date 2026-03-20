"""
Event-level dataset pipeline: ROOT → .npz

Converts L1ExtPuppi particle-level ROOT ntuples into event-level arrays
suitable for training a ParticleTransformer binary classifier (HH→4b vs QCD).

Feature layout (17 features, first 4 = [mass, pt, eta, phi] for PairwiseEmbedding):
  Col 0:  mass (GeV)       — estimated from particle type id
  Col 1:  pt (GeV)
  Col 2:  eta
  Col 3:  phi
  Col 4:  dxy (d0)
  Col 5:  z0 (dz)
  Col 6:  charge
  Col 7:  ln(pT / sum_pT)  — pT fraction relative to event sum
  Col 8:  Δη from pT-weighted centroid
  Col 9:  Δφ from pT-weighted centroid (wrapped to [−π, π])
  Col 10: ΔR from pT-weighted centroid
  Col 11: puppiWeight
  Col 12–16: one-hot particle type (charged hadron, neutral hadron, photon, electron, muon)

Output (.npz keys):
  x           — (N, num_constituents, 18) float32 feature array
  mask        — (N, num_constituents) bool validity mask
  y           — (N,) float32 labels (1 = signal, 0 = background)
  weights     — (N,) float32 kinematic reweighting weights
  jet_pt      — (N,) float32 scalar HT (stored as jet_pt for compatibility)
  jet_eta     — (N,) float32 zeros (placeholder)
  gen_pt      — (N,) float32 zeros (placeholder)
  qcd_weights — (N,) float32 per-event cross-section weights

Usage:
  python make_event_dataset.py --output data/event_level/event_hh4b_qcd.npz
  python make_event_dataset.py --output data/event_level/event_hh4b_qcd.npz --skip_trigger
"""

import os
import json
import glob as glob_module
import argparse

import numpy as np
import awkward as ak
from tqdm import tqdm

from data_pipeline.root_loading import load_event_level_data

# L1ExtPuppi particle type id → estimated mass (GeV)
# id: 0=charged hadron, 1=neutral hadron, 2=photon, 3=electron, 4=muon
PARTICLE_MASSES = {
    0: 0.13957,  # charged pion mass
    1: 0.13957,  # neutral pion mass (approximate)
    2: 0.0,  # photon
    3: 0.000511,  # electron
    4: 0.10566,  # muon
}

N_FEATURES = 17
N_PARTICLE_TYPES = 5


def passes_trigger_emulation(
    jet_events,
    pt_thresholds=(75, 60, 45, 40),
    ht_threshold=330.0,
    n_btag_required=3,
    btag_wp=0.37053,
    btag_field="btagScore",
):
    """
    Emulates CMS resolved-channel HH→4b HLT trigger using L1Ext jets.

    Requirements:
      - ≥4 jets with pT above descending thresholds [75, 60, 45, 40] GeV
      - Scalar HT (sum of all jet pT) > 330 GeV
      - ≥3 jets with <btag_field> > btag_wp

    Parameters
    ----------
    jet_events : ak.Array
        Per-event jet arrays with fields pt and <btag_field>.
    pt_thresholds : tuple
        Descending pT thresholds for the leading 4 jets.
    ht_threshold : float
        Minimum scalar HT.
    n_btag_required : int
        Minimum number of b-tagged jets.
    btag_wp : float
        B-tag working point threshold (default: 0.37053 for l1ext btagScore).
    btag_field : str
        Name of the b-tagger score field in jet_events (default: "btagScore").

    Returns
    -------
    mask : np.ndarray of bool, shape (N_events,)
    """
    # Sort jets by pT descending per event
    sorted_idx = ak.argsort(jet_events.pt, ascending=False)
    jets_sorted = jet_events[sorted_idx]

    n_jets = ak.num(jets_sorted)

    # Require at least 4 jets
    has_4_jets = n_jets >= 4

    # Leading jet pT thresholds
    pt_cuts = np.ones(len(jet_events), dtype=bool)
    for i, thr in enumerate(pt_thresholds):
        # For events with fewer jets, pad with False
        jet_pt_i = ak.fill_none(
            ak.pad_none(jets_sorted.pt, i + 1, clip=True)[:, i], 0.0
        )
        pt_cuts &= ak.to_numpy(jet_pt_i) > thr

    # Scalar HT
    ht = ak.to_numpy(ak.sum(jet_events.pt, axis=1))
    ht_cut = ht > ht_threshold

    # B-tag count
    n_btag = ak.to_numpy(ak.sum(jet_events[btag_field] > btag_wp, axis=1))
    btag_cut = n_btag >= n_btag_required

    mask = ak.to_numpy(has_4_jets) & pt_cuts & ht_cut & btag_cut
    return mask


def extract_event_features(puppi_cands, num_constituents=128):
    """
    Extract 18 features from L1ExtPuppi candidates for a single event.

    Parameters
    ----------
    puppi_cands : ak.Array
        All L1ExtPuppi candidates for one event, with fields:
        pt, eta, phi, charge, dxy, z0, id, puppiWeight.
    num_constituents : int
        Max number of particles to keep (sorted by pT descending).

    Returns
    -------
    x : np.ndarray, shape (num_constituents, 18), float32
    mask : np.ndarray, shape (num_constituents,), bool
    """
    # Sort by pT descending
    sort_idx = ak.argsort(puppi_cands.pt, ascending=False)
    cands = puppi_cands[sort_idx]

    # Convert to numpy
    pt = ak.to_numpy(cands.pt).astype(np.float32)
    eta = ak.to_numpy(cands.eta).astype(np.float32)
    phi = ak.to_numpy(cands.phi).astype(np.float32)
    charge = ak.to_numpy(cands.charge).astype(np.float32)
    dxy = ak.to_numpy(cands.dxy).astype(np.float32)
    z0 = ak.to_numpy(cands.z0).astype(np.float32)
    pid = ak.to_numpy(cands.id).astype(np.int32)
    puppi_weight = ak.to_numpy(cands.puppiWeight).astype(np.float32)

    n_real = min(len(pt), num_constituents)

    # Truncate
    pt = pt[:n_real]
    eta = eta[:n_real]
    phi = phi[:n_real]
    charge = charge[:n_real]
    dxy = dxy[:n_real]
    z0 = z0[:n_real]
    pid = pid[:n_real]

    # Mass from particle type
    mass = np.array(
        [PARTICLE_MASSES.get(int(p), 0.13957) for p in pid], dtype=np.float32
    )

    # Energy: E = sqrt((pt * cosh(eta))^2 + m^2)
    energy = np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2)

    # pT-weighted centroid
    sum_pt = pt.sum()
    if sum_pt > 0:
        eta_c = np.average(eta, weights=pt)
        # Circular mean for phi
        phi_c = np.arctan2(
            np.average(np.sin(phi), weights=pt),
            np.average(np.cos(phi), weights=pt),
        )
    else:
        eta_c, phi_c = 0.0, 0.0

    # Relative coordinates
    d_eta = eta - eta_c
    d_phi = np.arctan2(np.sin(phi - phi_c), np.cos(phi - phi_c))  # wrap to [-pi, pi]
    d_r = np.sqrt(d_eta**2 + d_phi**2)

    # Log features (clip to avoid log(0))
    ln_pt = np.log(np.clip(pt, 1e-6, None))
    ln_e = np.log(np.clip(energy, 1e-6, None))
    ln_pt_rel = np.log(np.clip(pt / max(sum_pt, 1e-6), 1e-12, None))

    # One-hot particle type
    one_hot = np.zeros((n_real, N_PARTICLE_TYPES), dtype=np.float32)
    for i in range(N_PARTICLE_TYPES):
        one_hot[:, i] = (pid == i).astype(np.float32)

    # Assemble feature array: (n_real, 18)
    features = np.column_stack(
        [
            mass,  # col 0
            pt,  # col 1
            eta,  # col 2
            phi,  # col 3
            dxy,  # col 4
            z0,  # col 5
            charge,  # col 6
            ln_pt_rel,  # col 7
            d_eta,  # col 8
            d_phi,  # col 9
            d_r,  # col 10
            puppi_weight,  # col 11
            one_hot,  # cols 12-16
        ]
    )

    # Pad to num_constituents
    x = np.zeros((num_constituents, N_FEATURES), dtype=np.float32)
    x[:n_real] = features
    mask = np.zeros(num_constituents, dtype=bool)
    mask[:n_real] = True

    return x, mask


def generate_event_dataset(
    config_path="hh-bbbb-obj-config.json",
    output_npz="data/event_level/event_hh4b_qcd.npz",
    num_constituents=128,
    skip_trigger=True,
    max_signal_events=None,
    max_qcd_events_per_bin=None,
):
    """
    Full pipeline: load ROOT files → feature extraction
    → kinematic reweighting → compressed .npz output.

    Parameters
    ----------
    config_path : str
        Path to hh-bbbb-obj-config.json.
    output_npz : str
        Output .npz file path (saved with np.savez_compressed).
    num_constituents : int
        Max particles per event.
    skip_trigger : bool
        If True, skip trigger emulation (use all events).
    max_signal_events : int or None
        Cap on signal events to load.
    max_qcd_events_per_bin : int or None
        Cap on QCD events per pT bin.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # ── L1Ext jet collection config (used for trigger emulation) ─────
    l1ext_cfg = config["l1ext"]
    l1ext_collection = l1ext_cfg["collection_name"]  # "L1puppiExtJetSC4"
    l1ext_tagger = l1ext_cfg["tagger_name"]  # "btagScore"
    l1ext_btag_wp = l1ext_cfg["b_tag_cut"]  # 0.37053

    all_x = []
    all_mask = []
    all_y = []
    all_ht = []
    all_qcd_weights = []

    # ── Signal (HH→4b) ──────────────────────────────────────────────
    print("\n=== Loading HH→4b signal ===")
    sig_puppi, sig_jets, n_sig = load_event_level_data(
        file_pattern=config["file_pattern"],
        tree_name=config["tree_name"],
        jet_collection=l1ext_collection,
        jet_tagger_field=l1ext_tagger,
        max_events=max_signal_events,
    )

    if skip_trigger:
        sig_mask = np.ones(n_sig, dtype=bool)
        print(f"Skipping trigger: keeping all {n_sig} signal events")
    else:
        sig_mask = passes_trigger_emulation(
            sig_jets,
            btag_wp=l1ext_btag_wp,
            btag_field=l1ext_tagger,
        )
        print(
            f"Trigger emulation: {sig_mask.sum()}/{n_sig} signal events pass "
            f"({100 * sig_mask.sum() / n_sig:.1f}%)"
        )

    sig_puppi_pass = sig_puppi[sig_mask]
    sig_jets_pass = sig_jets[sig_mask]
    n_sig_pass = int(sig_mask.sum())

    print(f"Extracting features for {n_sig_pass} signal events...")
    for i in tqdm(range(n_sig_pass), desc="Signal features"):
        x, m = extract_event_features(sig_puppi_pass[i], num_constituents)
        all_x.append(x)
        all_mask.append(m)

    all_y.extend([1.0] * n_sig_pass)
    sig_ht = ak.to_numpy(ak.sum(sig_jets_pass.pt, axis=1)).astype(np.float32)
    all_ht.extend(sig_ht.tolist())
    all_qcd_weights.extend([1.0] * n_sig_pass)  # signal gets weight 1.0

    # ── QCD background ───────────────────────────────────────────────
    print("\n=== Loading QCD background ===")
    qcd_config = config["QCD_background"]

    for bin_name, bin_cfg in qcd_config.items():
        print(f"\n--- {bin_name} ---")
        xsec_weight = bin_cfg["weight"]
        max_evts = max_qcd_events_per_bin or bin_cfg.get("max_events", None)

        try:
            qcd_puppi, qcd_jets, n_qcd = load_event_level_data(
                file_pattern=bin_cfg["file_pattern"],
                tree_name=bin_cfg["tree_name"],
                jet_collection=l1ext_collection,
                jet_tagger_field=l1ext_tagger,
                max_events=max_evts,
            )
        except (FileNotFoundError, Exception) as e:
            print(f"  Skipping {bin_name}: {e}")
            continue

        if skip_trigger:
            qcd_mask = np.ones(n_qcd, dtype=bool)
            print(f"  Skipping trigger: keeping all {n_qcd} events")
        else:
            qcd_mask = passes_trigger_emulation(
                qcd_jets,
                btag_wp=l1ext_btag_wp,
                btag_field=l1ext_tagger,
            )
            print(
                f"  Trigger: {qcd_mask.sum()}/{n_qcd} events pass "
                f"({100 * qcd_mask.sum() / max(n_qcd, 1):.1f}%)"
            )

        if qcd_mask.sum() == 0:
            print(f"  No events pass trigger, skipping {bin_name}")
            continue

        qcd_puppi_pass = qcd_puppi[qcd_mask]
        qcd_jets_pass = qcd_jets[qcd_mask]
        n_qcd_pass = int(qcd_mask.sum())

        # Per-event cross-section weight: xsec / n_gen
        # Use n_gen from config if available, else fall back to n_loaded
        n_gen = bin_cfg.get("n_gen", n_qcd)
        per_event_xsec = xsec_weight / n_gen

        print(f"  Extracting features for {n_qcd_pass} QCD events...")
        for i in tqdm(range(n_qcd_pass), desc=f"  {bin_name}"):
            x, m = extract_event_features(qcd_puppi_pass[i], num_constituents)
            all_x.append(x)
            all_mask.append(m)

        all_y.extend([0.0] * n_qcd_pass)
        qcd_ht = ak.to_numpy(ak.sum(qcd_jets_pass.pt, axis=1)).astype(np.float32)
        all_ht.extend(qcd_ht.tolist())
        all_qcd_weights.extend([per_event_xsec] * n_qcd_pass)

    # ── Assemble arrays ──────────────────────────────────────────────
    X = np.stack(all_x, axis=0)  # (N, num_constituents, 18)
    mask_arr = np.stack(all_mask, axis=0)  # (N, num_constituents)
    y = np.array(all_y, dtype=np.float32)
    ht = np.array(all_ht, dtype=np.float32)
    qcd_w = np.array(all_qcd_weights, dtype=np.float32)

    N = len(y)
    n_sig_total = int(y.sum())
    n_bkg_total = N - n_sig_total
    print(f"\n=== Dataset summary ===")
    print(f"Total events: {N} (signal: {n_sig_total}, background: {n_bkg_total})")

    # ── Kinematic reweighting ────────────────────────────────────────
    print("\nComputing kinematic weights (HT-based, 1D)...")
    from data_pipeline.make_particle_dataset import compute_kinematic_weights

    weights = compute_kinematic_weights(
        pt=ht,
        eta=np.zeros_like(ht),  # dummy — 1D reweighting only
        y=y,
        n_bins_pt=50,
        n_bins_eta=1,
        max_pt=2000.0,
        on_signal=True,
        sample_weights=qcd_w,
    )

    # ── Write .npz ───────────────────────────────────────────────────
    out_dir = os.path.dirname(output_npz)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"\nWriting compressed .npz to {output_npz}...")

    np.savez_compressed(
        output_npz,
        x=X,
        mask=mask_arr,
        y=y,
        weights=weights,
        jet_pt=ht,  # HT stored as jet_pt for compatibility
        jet_eta=np.zeros(N, dtype=np.float32),
        gen_pt=np.zeros(N, dtype=np.float32),
        qcd_weights=qcd_w,
    )

    print(f"Done. Dataset shape: x={X.shape}, y={y.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate event-level HH4b vs QCD dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hh-bbbb-obj-config.json",
        help="Path to config JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/event_level/event_hh4b_qcd.npz",
        help="Output .npz path",
    )
    parser.add_argument(
        "--num_constituents", type=int, default=128, help="Max particles per event"
    )
    parser.add_argument(
        "--skip_trigger",
        action="store_true",
        help="Skip trigger emulation (use all events)",
    )
    parser.add_argument(
        "--max_signal_events", type=int, default=None, help="Cap on signal events"
    )
    parser.add_argument(
        "--max_qcd_events_per_bin",
        type=int,
        default=None,
        help="Cap on QCD events per pT bin",
    )
    args = parser.parse_args()

    generate_event_dataset(
        config_path=args.config,
        output_npz=args.output,
        num_constituents=args.num_constituents,
        skip_trigger=args.skip_trigger,
        max_signal_events=args.max_signal_events,
        max_qcd_events_per_bin=args.max_qcd_events_per_bin,
    )
