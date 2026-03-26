"""Pre-compute and cache reference-tagger scores with gen-matched labels.

Caches signal b-jet scores (from HH ntuples) and QCD jet scores + labels
(from QCD pT-binned ntuples) for offline and L1 taggers.  Running this once
avoids expensive ROOT I/O and gen matching on every notebook evaluation.

Usage
-----
conda run -n hep-root-ml python -m data_pipeline.cache_reference_taggers \
    --config hh-bbbb-obj-config.json \
    --output data/reference_tagger_cache.npz \
    --collections offline l1ng l1ext \
    --eta-cut 2.4
"""

import argparse
import gc
import json
import sys
import time

import awkward as ak
import numpy as np

from data_pipeline.root_loading import (
    load_and_prepare_data,
    select_gen_b_quarks_from_higgs,
    select_gen_b_quarks_by_status,
)
from evaluation.jet_matching import get_purity_mask_cross_matched


# ── Tagger definitions ──────────────────────────────────────────────────────

# Each entry: (label, collection_name, tagger_field, extra_fields_to_cache)
TAGGER_REGISTRY = {
    "offline": ("Offline_PNet", "Jet", "btagPNetB", []),
    "l1ng": ("L1NG", "L1puppiJetSC4NG", "b_v_udscg_score", []),
    "l1ext": ("L1Ext", "L1puppiExtJetSC4", "btagScore", []),
}


def _eta_mask(jets, eta_cut):
    """Boolean mask for |eta| < eta_cut."""
    return abs(jets.eta) < eta_cut


def _flatten(arr):
    """Flatten awkward array to 1-D numpy."""
    return ak.to_numpy(ak.flatten(arr, axis=None)).astype(np.float32)


# ── Signal processing ────────────────────────────────────────────────────────

def process_signal(config, tagger_defs, eta_cut):
    """Load signal ntuples, gen-match, and extract reference tagger scores.

    Returns dict keyed by tagger label with arrays for matched (signal) jets.
    """
    collections_needed = list({td[1] for td in tagger_defs}) + ["GenPart"]
    print(f"\n[Signal] Loading collections: {collections_needed}")

    events = load_and_prepare_data(
        config["file_pattern"],
        config["tree_name"],
        collections_needed,
        config["max_events"],
        correct_pt=True,
        CONFIG=config,
    )
    n_events = len(events)
    print(f"[Signal] Loaded {n_events} events")

    # Gen b-quarks from Higgs
    gen_b = select_gen_b_quarks_from_higgs(events)
    gen_b = gen_b[
        (gen_b.pt > config["gen"]["pt_cut"])
        & (abs(gen_b.eta) < config["gen"]["eta_cut"])
    ]
    print(f"[Signal] Gen b-quarks from Higgs: {ak.sum(ak.num(gen_b))} total")

    result = {}
    for label, collection, tagger_field, _ in tagger_defs:
        jets = getattr(events, collection)
        # Eta cut only — no pT cut at cache time
        eta_ok = _eta_mask(jets, eta_cut)
        jets_cut = jets[eta_ok]

        matched_mask = get_purity_mask_cross_matched(
            gen_b, jets_cut, CONFIG=config
        )

        scores = _flatten(getattr(jets_cut, tagger_field)[matched_mask])
        pt = _flatten(jets_cut.pt[matched_mask])
        eta = _flatten(jets_cut.eta[matched_mask])

        result[label] = {
            "sig_scores": scores,
            "sig_pt": pt,
            "sig_eta": eta,
        }
        print(f"  [{label}] {len(scores)} signal jets cached")

    del events, gen_b
    gc.collect()
    return result


# ── QCD processing ───────────────────────────────────────────────────────────

def process_qcd(config, tagger_defs, eta_cut):
    """Load QCD pT-binned ntuples, gen-match, and extract tagger scores.

    Returns dict keyed by tagger label, each holding concatenated arrays
    for ALL QCD jets (with gen-match labels).
    """
    qcd_config = config["QCD_background"]
    collections_needed = list({td[1] for td in tagger_defs}) + ["GenPart"]

    # Accumulators per tagger
    accum = {
        td[0]: {"scores": [], "labels": [], "pt": [], "eta": [], "weights": []}
        for td in tagger_defs
    }

    for bin_name, bin_cfg in qcd_config.items():
        sigma_bin = bin_cfg["weight"]
        pt_range = bin_cfg["pt_range"]
        print(f"\n[QCD] Processing {bin_name} (pT {pt_range}, sigma={sigma_bin:.3e})")

        try:
            events = load_and_prepare_data(
                bin_cfg["file_pattern"],
                bin_cfg["tree_name"],
                collections_needed,
                bin_cfg["max_events"],
                correct_pt=True,
                CONFIG=config,
            )
        except Exception as e:
            print(f"  WARNING: Failed to load {bin_name}: {e}")
            continue

        n_events = len(events)
        if n_events == 0:
            print(f"  Skipping {bin_name} — 0 events")
            continue
        print(f"  Loaded {n_events} events")

        # Gen b-quarks by status flags (same as make_particle_dataset)
        gen_b = select_gen_b_quarks_by_status(events, config)

        for label, collection, tagger_field, _ in tagger_defs:
            jets = getattr(events, collection)
            eta_ok = _eta_mask(jets, eta_cut)
            jets_cut = jets[eta_ok]

            matched_mask = get_purity_mask_cross_matched(
                gen_b, jets_cut, CONFIG=config
            )

            flat_scores = _flatten(getattr(jets_cut, tagger_field))
            flat_labels = _flatten(ak.values_astype(matched_mask, np.float32))
            flat_pt = _flatten(jets_cut.pt)
            flat_eta = _flatten(jets_cut.eta)
            flat_weights = np.full(len(flat_scores), sigma_bin, dtype=np.float64)

            accum[label]["scores"].append(flat_scores)
            accum[label]["labels"].append(flat_labels)
            accum[label]["pt"].append(flat_pt)
            accum[label]["eta"].append(flat_eta)
            accum[label]["weights"].append(flat_weights)

            n_b = int(flat_labels.sum())
            print(f"    [{label}] {len(flat_scores)} jets ({n_b} b-matched)")

        del events, gen_b
        gc.collect()

    # Concatenate
    result = {}
    for label in accum:
        if not accum[label]["scores"]:
            print(f"  WARNING: No QCD data for {label}")
            result[label] = {
                "qcd_scores": np.array([], dtype=np.float32),
                "qcd_labels": np.array([], dtype=np.float32),
                "qcd_pt": np.array([], dtype=np.float32),
                "qcd_eta": np.array([], dtype=np.float32),
                "qcd_weights": np.array([], dtype=np.float64),
            }
        else:
            result[label] = {
                "qcd_scores": np.concatenate(accum[label]["scores"]),
                "qcd_labels": np.concatenate(accum[label]["labels"]),
                "qcd_pt": np.concatenate(accum[label]["pt"]),
                "qcd_eta": np.concatenate(accum[label]["eta"]),
                "qcd_weights": np.concatenate(accum[label]["weights"]),
            }
        n_total = len(result[label]["qcd_scores"])
        n_b = int(result[label]["qcd_labels"].sum())
        print(f"  [{label}] QCD total: {n_total} jets ({n_b} b-matched)")

    return result


# ── Save ─────────────────────────────────────────────────────────────────────

def save_cache(sig_data, qcd_data, config, output_path):
    """Save cache to .npz (or .h5 if >5 GB)."""
    output = {}

    # Metadata: sigma_to_ngen as parallel arrays
    qcd_bg = config["QCD_background"]
    sigmas = np.array([b["weight"] for b in qcd_bg.values()], dtype=np.float64)
    ngens = np.array([b["n_gen"] for b in qcd_bg.values()], dtype=np.int64)
    output["meta_sigmas"] = sigmas
    output["meta_ngens"] = ngens

    # Per-tagger arrays
    for label in sig_data:
        for key, arr in sig_data[label].items():
            output[f"{label}_{key}"] = arr
    for label in qcd_data:
        for key, arr in qcd_data[label].items():
            output[f"{label}_{key}"] = arr

    # Estimate size
    total_bytes = sum(arr.nbytes for arr in output.values())
    total_mb = total_bytes / (1024 * 1024)
    print(f"\nEstimated cache size: {total_mb:.1f} MB")

    if total_mb > 5000:
        # Fall back to HDF5
        h5_path = output_path.replace(".npz", ".h5")
        print(f"Cache >5 GB — saving as HDF5: {h5_path}")
        import h5py

        with h5py.File(h5_path, "w") as f:
            for key, arr in output.items():
                f.create_dataset(key, data=arr, compression="gzip")
        print(f"Saved HDF5 cache: {h5_path}")
    else:
        np.savez_compressed(output_path, **output)
        print(f"Saved cache: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cache reference tagger scores with gen-matched labels."
    )
    parser.add_argument(
        "--config", default="hh-bbbb-obj-config.json",
        help="Path to physics/object config JSON",
    )
    parser.add_argument(
        "--output", default="data/reference_tagger_cache.npz",
        help="Output cache file path (.npz)",
    )
    parser.add_argument(
        "--collections", nargs="+", default=["offline", "l1ng", "l1ext"],
        choices=list(TAGGER_REGISTRY.keys()),
        help="Which reference tagger collections to cache",
    )
    parser.add_argument(
        "--eta-cut", type=float, default=2.4,
        help="Eta cut (no pT cut applied — vary pT at evaluation time)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    tagger_defs = [TAGGER_REGISTRY[c] for c in args.collections]
    print(f"Caching taggers: {[td[0] for td in tagger_defs]}")
    print(f"Eta cut: |eta| < {args.eta_cut}")

    t0 = time.time()
    sig_data = process_signal(config, tagger_defs, args.eta_cut)
    qcd_data = process_qcd(config, tagger_defs, args.eta_cut)
    save_cache(sig_data, qcd_data, config, args.output)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
