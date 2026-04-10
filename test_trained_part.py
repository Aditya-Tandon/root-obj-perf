#!/usr/bin/env python
"""
test_trained_part.py

Standalone script that reproduces all outputs and plots from
test_trained_part.ipynb.  Usage:

    python test_trained_part.py --config config_part.json
"""

import argparse
import os
import json
import warnings
import gc
from typing import Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

# ── Plotting defaults ──────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.6,
        "grid.linestyle": "--",
        "font.size": 14,
        "figure.dpi": 200,
    }
)

# ── Cell 1: Imports & configuration ───────────────────────────────
import uproot
import awkward as ak
import vector

warnings.filterwarnings("ignore", message="Passing an awkward array to a ufunc")
ak.behavior.update(vector.backends.awkward.behavior)


PROFILE_LEVELS = {
    "core": 1,
    "standard": 2,
    "full": 3,
}


def should_run(profile: str, minimum: str) -> bool:
    return PROFILE_LEVELS[profile] >= PROFILE_LEVELS[minimum]


def flush_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ParT model and produce analysis plots."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_part.json",
        help="Path to the config_part JSON file (default: config_part.json)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=("core", "standard", "full"),
        default="standard",
        help=(
            "Execution profile: core (memory-safe baseline), standard (full AK4/b-tag suite), "
            "full (standard + AK8 H-tag parity sections from notebook)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Execution profile: {args.profile}")

    # Load configs
    with open("hh-bbbb-obj-config.json", "r") as f:
        config = json.load(f)

    from evaluation.luminosity import load_physics_config

    _physics = load_physics_config()
    LUMINOSITY_FB = _physics["luminosity_fb"]
    SIGNAL_XSEC_PB = _physics["signal_xsec_pb"]
    N_GEN_SIGNAL = _physics["n_gen_signal"]

    with open(args.config, "r") as f:
        CONFIG_PART = json.load(f)

    config_part = CONFIG_PART  # will be overwritten by checkpoint later

    print(f"Loaded config from {args.config}")

    # ── Cell 2: Load model from W&B checkpoint ────────────────────────
    import wandb
    from model.parT import ParticleTransformer
    from data_pipeline.datasets import L1JetDataset, StratifiedJetDataset
    from data_pipeline.splitting import stratified_split
    from data_pipeline.combined_loader import CombinedJetDataLoader
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="part-btag-analysis")
    artifact_path = (
        f"{config_part['wandb']['entity']}/"
        f"{config_part['wandb']['project']}/"
        f"{config_part['wandb']['artifact_name']}:{config_part['wandb']['ckpt_type']}"
    )
    artifact = wandb.use_artifact(artifact_path, type="model")
    artifact_dir = artifact.download()
    wandb.finish()
    print(f"Model artifact downloaded from W&B: {artifact_dir}")

    assert os.path.exists(artifact_dir), f"Artifact dir not found: {artifact_dir}"
    print("Loading checkpoint")
    checkpoint_dir = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
    checkpoint = torch.load(checkpoint_dir, map_location=device, weights_only=False)
    config_part = checkpoint["config"]

    print(f"Exp name: {config_part['exp_name']}")
    try:
        print(f"Data path: {config_part['data_path']}")
        torch.manual_seed(42)

        dataset = L1JetDataset(config_part["data_path"])
        num_classes = config_part["model"]["num_classes"]

        train_dataset, val_dataset, train_indices, val_indices, stratify_labels = (
            stratified_split(
                dataset=dataset,
                val_split=config_part["training"]["val_split"],
                num_classes=num_classes,
                random_state=42,
                verbose=True,
            )
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config_part["training"]["batch_size"],
            shuffle=True,
            num_workers=config_part["training"]["num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config_part["training"]["batch_size"],
            shuffle=False,
            num_workers=config_part["training"]["num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )
        print("Data loaders prepared.")

    except KeyError:
        print("\nData path not found in checkpoint config")
        print("Using the new config structure.")
        dataset_used = config_part["training"]["data"]["use_dataset"]
        config_part["data_path"] = config_part["training"]["data"][
            f"{dataset_used}_data_path"
        ]
        config_part["training"]["val_split"] = config_part["training"]["data"][
            "val_split"
        ]
        config_part["training"]["num_workers"] = config_part["training"]["data"][
            "num_workers"
        ]
        print(f"Data path: {config_part['data_path']}")
        pt_regression = config_part.get("model", {}).get("pt_regression", False)
        combined_loader = CombinedJetDataLoader(
            pf_data_path=config_part["training"]["data"]["pf_data_path"],
            puppi_data_path=config_part["training"]["data"]["puppi_data_path"],
            val_split=config_part["training"]["data"]["val_split"],
            batch_size=config_part["training"]["batch_size"],
            match_mode=config_part["training"]["data"]["match_mode"],
            num_workers=config_part["training"]["data"]["num_workers"],
            random_state=42,
            pt_regression=pt_regression,
        )
        if config_part["training"]["data"]["use_dataset"] == "pf":
            print("\nUsing PF dataset for training.")
            (
                train_loader, train_indices,
                val_loader, val_indices,
                train_labels, val_labels,
            ) = combined_loader.get_pf_loaders(shuffle=True)
        elif config_part["training"]["data"]["use_dataset"] == "puppi":
            print("\nUsing PUPPI dataset for training.")
            (
                train_loader, train_indices,
                val_loader, val_indices,
                train_labels, val_labels,
            ) = combined_loader.get_puppi_loaders(shuffle=True)
        print(
            f"Data loaders prepared with {len(train_loader.dataset)} training samples "
            f"and {len(val_loader.dataset)} validation samples."
        )

    print(f"\nEpoch: {checkpoint['epoch']}")
    print(f"Val loss: {checkpoint['val_loss']}")
    print(f"Val_auc: {checkpoint['val_auc']}")
    pt_regression = config_part.get("model", {}).get("pt_regression", False)
    quantile_regression = config_part.get("model", {}).get("quantile_regression", False)
    model = ParticleTransformer(
        input_dim=config_part["input_dim"],
        embed_dim=config_part["model"]["embed_dim"],
        num_heads=config_part["model"]["num_heads"],
        num_layers=config_part["model"]["num_layers"],
        num_cls_layers=config_part["model"]["num_cls_layers"],
        dropout=config_part["model"]["dropout"],
        num_classes=config_part["model"]["num_classes"],
        pt_regression=pt_regression,
        quantile_regression=quantile_regression,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Checkpoint loaded (pt_regression={pt_regression}, "
        f"quantile_regression={quantile_regression})"
    )

    # ── Cell 3: Run inference on validation set ───────────────────────
    model.eval()
    all_outputs = []
    all_labels = []
    
    val_jet_pt_list = []
    val_jet_eta_list = []
    val_jet_phi_list = []
    val_jet_mass_list = []
    val_n_const_list = []

    collect_all_constituents = should_run(args.profile, "standard")
    all_constituents = [] if collect_all_constituents else None
    n_constituents_model = None
    all_reg_pt = []
    all_quantiles = []
    all_jet_pt_val = []
    all_gen_pt_val = []
    all_weights_val = []  # Kinematic reweighting weights from dataset (for training)
    all_qcd_weights_val = []  # QCD cross-section weights (for testing metrics)


    with torch.no_grad():
        for X_batch, y_batch, mask_batch, weights_batch, jet_pt_batch, _, gen_pt_batch, qcd_weights_batch in tqdm(
            val_loader, desc="Validation inference"
        ):
            if n_constituents_model is None:
                n_constituents_model = int(X_batch.shape[1])

            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)
            weights_batch = weights_batch.to(device)

            out = model(X_batch, particle_mask=mask_batch)
            cls_logits = out["classification"]
            outputs = torch.nn.functional.sigmoid(cls_logits).squeeze()

            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_jet_pt_val.append(jet_pt_batch.squeeze().cpu().numpy())
            all_gen_pt_val.append(gen_pt_batch.squeeze().cpu().numpy())
            all_weights_val.append(weights_batch.squeeze().cpu().numpy())
            all_qcd_weights_val.append(qcd_weights_batch.squeeze().cpu().numpy())
            
            # Reconstruct jet 4-vectors locally to save memory
            X_cpu = X_batch.cpu().numpy()
            const_mass_b = X_cpu[:, :, 0]
            const_pt_b = X_cpu[:, :, 1]
            const_eta_b = X_cpu[:, :, 2]
            const_phi_b = X_cpu[:, :, 3]
            const_vectors_b = vector.array(
                {"pt": const_pt_b, "eta": const_eta_b, "phi": const_phi_b, "mass": const_mass_b}
            )
            jet_vectors_b = const_vectors_b.sum(axis=1)
            
            val_jet_pt_list.append(jet_vectors_b.pt)
            val_jet_eta_list.append(jet_vectors_b.eta)
            val_jet_phi_list.append(jet_vectors_b.phi)
            val_jet_mass_list.append(jet_vectors_b.mass)
            
            val_n_const_list.append((X_cpu[:, :, 1] > 0).sum(axis=1))
            
            if collect_all_constituents:
                all_constituents.append(X_batch.half().cpu())

            if "pt" in out:
                all_reg_pt.append(out["pt"].squeeze().cpu().numpy())
            if "quantiles" in out:
                all_quantiles.append(out["quantiles"].cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    all_jet_pt_val = np.concatenate(all_jet_pt_val)
    all_gen_pt_val = np.concatenate(all_gen_pt_val)
    all_weights_val = np.concatenate(all_weights_val)
    all_qcd_weights_val = np.concatenate(all_qcd_weights_val)

    if collect_all_constituents:
        all_constituents = torch.cat(all_constituents)
    else:
        all_constituents = None
    
    val_jet_pt = np.concatenate(val_jet_pt_list)
    val_jet_eta = np.concatenate(val_jet_eta_list)
    val_jet_phi = np.concatenate(val_jet_phi_list)
    val_jet_mass = np.concatenate(val_jet_mass_list)
    n_const_val = np.concatenate(val_n_const_list)

    has_regression = len(all_reg_pt) > 0
    if has_regression:
        all_reg_pt = np.concatenate(all_reg_pt)
        print(f"Regression outputs collected: {all_reg_pt.shape}")

    has_quantile = len(all_quantiles) > 0
    if has_quantile:
        all_quantiles = np.concatenate(all_quantiles)
        print(f"Quantile regression outputs collected: {all_quantiles.shape}")
        print(f"  q16 range: {all_quantiles[:, 0].min():.4f} – {all_quantiles[:, 0].max():.4f}")
        print(f"  q84 range: {all_quantiles[:, 1].min():.4f} – {all_quantiles[:, 1].max():.4f}")
    else:
        print("No quantile regression head in this model.")

    # Reconstruct jet 4-vectors already done during batch processing
    print(f"\nJet kinematics reconstructed using vector library:")
    print(f"  pT range: {val_jet_pt.min():.2f} - {val_jet_pt.max():.2f} GeV")
    print(f"  eta range: {val_jet_eta.min():.2f} - {val_jet_eta.max():.2f}")

    if has_regression:
        sig_mask = all_labels.squeeze() == 1
        print(f"\nRegression summary (signal jets only):")
        print(
            f"  Predicted correction range: "
            f"{all_reg_pt[sig_mask].min():.3f} – {all_reg_pt[sig_mask].max():.3f}"
        )
        print(
            f"  True correction (gen_pt/jet_pt) range: "
            f"{(all_gen_pt_val[sig_mask] / (all_jet_pt_val[sig_mask] + 1e-9)).min():.3f} – "
            f"{(all_gen_pt_val[sig_mask] / (all_jet_pt_val[sig_mask] + 1e-9)).max():.3f}"
        )

    # ── Cell 4: Validation statistics ─────────────────────────────────
    print(f"\nTotal validation samples: {len(all_outputs)}")
    print(f"Num positive samples: {np.sum(all_labels)}")
    print(f"Num negative samples: {len(all_outputs) - np.sum(all_labels)}")
    print(f"% of positive samples: {100 * np.sum(all_labels) / len(all_outputs):.2f}%")

    # ── Cell 5: Regression head analysis plots ────────────────────────
    # Helper: set up plot output directory

    run_id = config_part.get("exp_name", "unknown_run").replace("/", "_")
    artifact_name = CONFIG_PART.get("wandb", {}).get("artifact_name", "unknown_artifact")
    artifact_type = CONFIG_PART.get("wandb", {}).get("ckpt_type", "unknown_type")
    plot_dir = f"../Updates/plots_{run_id}/{artifact_name}:{artifact_type}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nSaving plots to: {plot_dir}")

    def save_fig(fig, name):
        """Save figure to plot directory."""
        filepath = os.path.join(plot_dir, f"{name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {name}.png")

    signal_mask = all_labels.squeeze() == 1
    jet_pt = all_jet_pt_val
    gen_pt = all_gen_pt_val

    print(f"Total validation jets: {len(all_labels)}")
    print(f"  Signal jets: {signal_mask.sum()}")
    print(f"  Background jets: {(~signal_mask).sum()}")
    if has_regression:
        reg_pt = all_reg_pt
        print(f"\nRegression output (signal only):")
        print(f"  reg_pt  range: {reg_pt[signal_mask].min():.2f} – {reg_pt[signal_mask].max():.2f}")
        print(f"  jet_pt  range: {jet_pt[signal_mask].min():.2f} – {jet_pt[signal_mask].max():.2f}")
        print(f"  gen_pt  range: {gen_pt[signal_mask].min():.2f} – {gen_pt[signal_mask].max():.2f}")
        correction = gen_pt[signal_mask] / (jet_pt[signal_mask] + 1e-9)
        print(f"  Correction factor (gen_pt / jet_pt) range: {correction.min():.3f} – {correction.max():.3f}")
    else:
        print("\nNo regression head in this model.")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. jet_pt distribution (signal vs background)
    ax = axes[0, 0]
    ax.hist(jet_pt[signal_mask], bins=60, range=(0, 500), histtype="step", density=True, label="Signal")
    ax.hist(jet_pt[~signal_mask], bins=60, range=(0, 500), histtype="step", density=True, label="Background")
    ax.set_xlabel("Reco jet $p_T$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Reco jet $p_T$ distribution")
    ax.legend()

    # 2. gen_pt distribution (signal only)
    ax = axes[0, 1]
    ax.hist(gen_pt[signal_mask], bins=60, range=(0, 500), histtype="step", density=True, color="C0")
    ax.set_xlabel("Gen $p_T$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Gen $p_T$ (signal jets only)")

    # 3. Correction factor (signal only)
    ax = axes[0, 2]
    correction = gen_pt[signal_mask] / (jet_pt[signal_mask] + 1e-9)
    ax.hist(correction, bins=60, range=(0, 3), histtype="step", density=True, color="C2")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="No correction")
    ax.set_xlabel("gen $p_T$ / reco $p_T$")
    ax.set_ylabel("Density")
    ax.set_title("Target correction factor (signal)")
    ax.legend()

    if has_regression:
        # 4. Regression output distribution
        ax = axes[1, 0]
        ax.hist(reg_pt[signal_mask], bins=60, histtype="step", density=True, label="Signal")
        ax.hist(reg_pt[~signal_mask], bins=60, histtype="step", density=True, label="Background")
        ax.set_xlabel("Regression output (predicted correction)")
        ax.set_ylabel("Density")
        ax.set_title("Regression head output")
        ax.legend()

        # 5. Predicted vs true correction (signal only)
        ax = axes[1, 1]
        true_corr = gen_pt[signal_mask] / (jet_pt[signal_mask] + 1e-9)
        pred_corr = reg_pt[signal_mask]
        ax.scatter(true_corr, pred_corr, s=1, alpha=0.3)
        lim = max(true_corr.max(), pred_corr.max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="Ideal")
        ax.set_xlim(0, lim)
        ax.set_xlabel("True correction (gen $p_T$ / reco $p_T$)")
        ax.set_ylabel("Predicted correction")
        ax.set_title("Predicted vs true correction (signal)")
        ax.legend()

        # 6. Corrected jet pT vs gen pT (signal)
        ax = axes[1, 2]
        corrected_pt = jet_pt[signal_mask] * pred_corr
        ax.scatter(gen_pt[signal_mask], corrected_pt, s=1, alpha=0.3, label="Corrected")
        ax.scatter(gen_pt[signal_mask], jet_pt[signal_mask], s=1, alpha=0.1, color="gray", label="Uncorrected")
        lim = 500
        ax.plot([0, lim], [0, lim], "r--", alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Gen $p_T$ [GeV]")
        ax.set_ylabel("Jet $p_T$ [GeV]")
        ax.set_title("Corrected vs uncorrected $p_T$ (signal)")
        ax.legend()
    else:
        for i in range(3):
            axes[1, i].text(0.5, 0.5, "No regression head", ha="center", va="center", transform=axes[1, i].transAxes)
            axes[1, i].set_axis_off()

    plt.tight_layout()
    save_fig(fig, "regression_head_analysis_initial")
    plt.close(fig)

    # ── Cell 6: Training data loading & jet kinematics ────────────────
    all_labels_train = []
    all_weights_train = []
    
    train_jet_pt_list = []
    train_jet_eta_list = []
    train_jet_phi_list = []
    train_jet_mass_list = []
    n_const_train_list = []

    for X_batch, y_batch, mask_batch, weights_batch, _, _, _, _ in tqdm(
        train_loader, desc="Loading training data"
    ):
        all_labels_train.append(y_batch.squeeze(-1).cpu().numpy())
        all_weights_train.append(weights_batch.cpu().numpy())
        
        X_cpu = X_batch.cpu().numpy()
        const_mass_train = X_cpu[:, :, 0]
        const_pt_train = X_cpu[:, :, 1]
        const_eta_train = X_cpu[:, :, 2]
        const_phi_train = X_cpu[:, :, 3]
        
        const_vectors_train = vector.array(
            {"pt": const_pt_train, "eta": const_eta_train, "phi": const_phi_train, "mass": const_mass_train}
        )
        jet_vectors_train = const_vectors_train.sum(axis=1)
        
        train_jet_pt_list.append(jet_vectors_train.pt)
        train_jet_eta_list.append(jet_vectors_train.eta)
        train_jet_phi_list.append(jet_vectors_train.phi)
        train_jet_mass_list.append(jet_vectors_train.mass)
        n_const_train_list.append(mask_batch.sum(dim=1).cpu().numpy())

    all_labels_train = np.concatenate(all_labels_train)
    all_weights_train = np.concatenate(all_weights_train)

    train_jet_pt = np.concatenate(train_jet_pt_list)
    train_jet_eta = np.concatenate(train_jet_eta_list)
    train_jet_phi = np.concatenate(train_jet_phi_list)
    train_jet_mass = np.concatenate(train_jet_mass_list)
    n_const_train = np.concatenate(n_const_train_list)

    # ── Cell 7: Train vs Val comparison plots ─────────────────────────
    # 1. Train vs Val jet pT
    fig, ax = plt.subplots()
    ax.hist(train_jet_pt[all_labels_train == 1], bins=50, range=(0, 500), density=True, alpha=0.5, label="Train Signal")
    ax.hist(train_jet_pt[all_labels_train == 0], bins=50, range=(0, 500), density=True, alpha=0.5, label="Train Background")
    ax.hist(val_jet_pt[all_labels.reshape(all_labels.shape[0]) == 1], bins=50, range=(0, 500), density=True, histtype="step", label="Val Signal")
    ax.hist(val_jet_pt[all_labels.reshape(all_labels.shape[0]) == 0], bins=50, range=(0, 500), density=True, histtype="step", label="Val Background")
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Normalized Entries")
    ax.set_title("Jet $p_T$ Distribution: Train vs Val")
    ax.legend()
    save_fig(fig, "train_vs_val_jet_pt")
    plt.close(fig)

    # 2. Reweighted training pT
    fig, ax = plt.subplots()
    ax.hist(train_jet_pt[all_labels_train == 1], bins=50, range=(0, 500), weights=all_weights_train[all_labels_train == 1], density=True, alpha=0.5, label="Reweighted Train Signal")
    ax.hist(train_jet_pt[all_labels_train == 0], bins=50, range=(0, 500), weights=all_weights_train[all_labels_train == 0], density=True, alpha=0.5, label="Reweighted Train Background")
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Normalized Entries")
    ax.set_title("Reweighted Jet $p_T$ Distribution: Train Set")
    ax.legend()
    save_fig(fig, "reweighted_train_jet_pt")
    plt.close(fig)

    # 3. Reweighted training eta
    fig, ax = plt.subplots()
    ax.hist(train_jet_eta[all_labels_train == 1], bins=50, range=(-3, 3), weights=all_weights_train[all_labels_train == 1], density=True, alpha=0.5, label="Reweighted Train Signal")
    ax.hist(train_jet_eta[all_labels_train == 0], bins=50, range=(-3, 3), weights=all_weights_train[all_labels_train == 0], density=True, alpha=0.5, label="Reweighted Train Background")
    ax.set_xlabel("Jet $\\eta$")
    ax.set_ylabel("Normalized Entries")
    ax.set_title("Reweighted Jet $\\eta$ Distribution: Train Set")
    ax.legend()
    save_fig(fig, "reweighted_train_jet_eta")
    plt.close(fig)

    # 4. N_constituents vs eta (2D)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    h0 = ax[0].hist2d(train_jet_eta, n_const_train, bins=50, cmap="viridis", cmin=1)
    ax[0].set_xlabel(r"Jet $\eta$")
    ax[0].set_ylabel("Number of Constituents")
    ax[0].set_title("Train Set: $N_{const}$ vs $\\eta$")
    plt.colorbar(h0[3], ax=ax[0], label="Counts")

    h1 = ax[1].hist2d(val_jet_eta, n_const_val, bins=50, cmap="viridis", cmin=1)
    ax[1].set_xlabel(r"Jet $\eta$")
    ax[1].set_ylabel("Number of Constituents")
    ax[1].set_title("Validation Set: $N_{const}$ vs $\\eta$")
    plt.colorbar(h1[3], ax=ax[1], label="Counts")

    plt.tight_layout()
    save_fig(fig, "n_const_vs_eta_train_val")
    plt.close(fig)

    # ── Cell 8: ROC analysis & comparison with other taggers ──────────
    import copy, gc
    from data_pipeline.root_loading import apply_custom_cuts, load_and_prepare_data, select_gen_b_quarks_from_higgs
    from evaluation.jet_matching import get_purity_mask_cross_matched
    from evaluation.roc import roc_from_scores, get_roc_point_at_mistag, get_working_points
    from evaluation.efficiency import efficiency_table
    from evaluation.roc import pr_auc_and_opt_s_over_root_b
    from plotting.base import plot_roc_comparison
    from sklearn.metrics import roc_curve, auc

    # Re-load config (may have been mutated)
    with open("hh-bbbb-obj-config.json", "r") as f:
        config = json.load(f)

    # Create vector array from reconstructed jet kinematics for cuts
    val_jet_vectors = vector.array(
        {"pt": val_jet_pt, "eta": val_jet_eta, "phi": val_jet_phi, "mass": val_jet_mass}
    )
    val_cuts_mask = apply_custom_cuts(
        val_jet_vectors, config, "l1barrelextpf", kinematic_only=True, return_jets=False
    )
    print(f"Jets after kinematic cuts: {np.sum(val_cuts_mask)}/{len(val_cuts_mask)}")

    all_labels = all_labels.reshape(-1)
    all_labels_after_cuts = all_labels[val_cuts_mask]
    all_outputs_after_cuts = all_outputs[val_cuts_mask]
    # Use QCD cross-section weights for metrics (not kinematic training weights)
    all_weights_after_cuts = all_qcd_weights_val[val_cuts_mask]
    # Keep kinematic training weights separately for reweighting visualisation plots
    all_kinematic_weights_after_cuts = all_weights_val[val_cuts_mask]


    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels_after_cuts, all_outputs_after_cuts, sample_weight=all_weights_after_cuts)
    roc_auc = auc(fpr, tpr)

    # Load reference collections for comparison
    events = load_and_prepare_data(
        config["file_pattern"],
        config["tree_name"],
        [
            "GenPart",
            config["offline"]["collection_name"],
            config["l1ng"]["collection_name"],
            config["l1ext"]["collection_name"],
        ],
        config["max_events"],
        correct_pt=True,
    )

    # UParT config
    upart_CONFIG = copy.deepcopy(config)
    upart_CONFIG["offline"]["tagger_name"] = "btagUParTAK4B"
    upart_CONFIG["offline"]["b_tag_cut"] = 0.00496

    upart_events = load_and_prepare_data(
        upart_CONFIG["file_pattern"],
        upart_CONFIG["tree_name"],
        [upart_CONFIG["offline"]["collection_name"]],
        upart_CONFIG["max_events"],
        CONFIG=upart_CONFIG,
    )

    # Gen-level b-quarks for matching
    gen_b_quarks = select_gen_b_quarks_from_higgs(events)
    gen_b_quarks = gen_b_quarks[
        (gen_b_quarks.pt > config["gen"]["pt_cut"])
        & (abs(gen_b_quarks.eta) < config["gen"]["eta_cut"])
    ]

    # Build signal jets (gen-matched pure) for each tagger
    tagger_defs = [
        ("Offline PNet", "offline", config, config["offline"]["tagger_name"]),
        ("Offline UParT", "offline", upart_CONFIG, upart_CONFIG["offline"]["tagger_name"]),
        ("L1NG", "l1ng", config, config["l1ng"]["tagger_name"]),
        ("L1Ext", "l1ext", config, config["l1ext"]["tagger_name"]),
    ]

    signal_jets_kinematic = {
        "Offline PNet": apply_custom_cuts(
            events[config["offline"]["collection_name"]],
            config,
            "offline",
            kinematic_only=True,
        ),
        "Offline UParT": apply_custom_cuts(
            upart_events[upart_CONFIG["offline"]["collection_name"]],
            upart_CONFIG,
            "offline",
            kinematic_only=True,
        ),
        "L1NG": apply_custom_cuts(
            events[config["l1ng"]["collection_name"]], config, "l1ng", kinematic_only=True
        ),
        "L1Ext": apply_custom_cuts(
            events[config["l1ext"]["collection_name"]], config, "l1ext", kinematic_only=True
        ),
    }

    signal_purity_masks = {}
    for lbl in signal_jets_kinematic:
        signal_purity_masks[lbl] = get_purity_mask_cross_matched(
            gen_b_quarks, signal_jets_kinematic[lbl]
        )

    # Load QCD jets in chunks and build per-jet raw cross-section weights.
    QCD_CHUNK_SIZE = 5000
    all_collections = list({cfg[key]["collection_name"] for _, key, cfg, _ in tagger_defs})
    qcd_config = config.get("QCD_background", {})

    qcd_scores_inclusive = {lbl: [] for lbl, *_ in tagger_defs}
    qcd_weights_inclusive = {lbl: [] for lbl, *_ in tagger_defs}

    print("\nLoading QCD jets for reference tagger ROCs (chunked, all events)...")
    for bin_name, bin_cfg in sorted(qcd_config.items(), key=lambda x: x[1]["weight"]):
        w_qcd_cross_section = bin_cfg["weight"]  # raw sigma_bin
        n_bin_total = 0
        offset = 0
        chunk_idx = 0

        while True:
            try:
                qcd_events = load_and_prepare_data(
                    bin_cfg["file_pattern"],
                    bin_cfg.get("tree_name", "Events"),
                    all_collections,
                    QCD_CHUNK_SIZE,
                    correct_pt=True,
                    CONFIG=config,
                    entry_start=offset,
                )
            except Exception as e:
                print(f"  {bin_name} chunk {chunk_idx}: error - {e}")
                break

            n_loaded = len(qcd_events)
            if n_loaded == 0:
                del qcd_events
                gc.collect()
                break

            for lbl, key, cfg, tagger_name in tagger_defs:
                collection_name = cfg[key]["collection_name"]
                qcd_jets = apply_custom_cuts(
                    qcd_events[collection_name], cfg, key, kinematic_only=True
                )
                scores = ak.to_numpy(ak.flatten(getattr(qcd_jets, tagger_name)))
                qcd_scores_inclusive[lbl].append(scores)
                qcd_weights_inclusive[lbl].append(
                    np.full(len(scores), w_qcd_cross_section, dtype=np.float64)
                )

            n_bin_total += n_loaded
            offset += n_loaded
            chunk_idx += 1
            del qcd_events
            gc.collect()

        print(
            f"  {bin_name}: {chunk_idx} chunks, {n_bin_total} events, raw_sigma={w_qcd_cross_section:.3e}"
        )

    # Concatenate all chunks
    for lbl in qcd_scores_inclusive:
        if qcd_scores_inclusive[lbl]:
            qcd_scores_inclusive[lbl] = np.concatenate(qcd_scores_inclusive[lbl])
            qcd_weights_inclusive[lbl] = np.concatenate(qcd_weights_inclusive[lbl])
        else:
            qcd_scores_inclusive[lbl] = np.array([])
            qcd_weights_inclusive[lbl] = np.array([], dtype=np.float64)

    print("\nQCD jets loaded.")
    for lbl in qcd_scores_inclusive:
        print(f"  {lbl}: {len(qcd_scores_inclusive[lbl]):,} QCD jets")

    # ROC: signal (gen-matched) vs QCD background
    ref_roc_results = {}
    for lbl, key, cfg, tagger_name in tagger_defs:
        sig_jets = signal_jets_kinematic[lbl]
        pure_mask = signal_purity_masks[lbl]
        sig_scores = ak.to_numpy(ak.flatten(getattr(sig_jets[pure_mask], tagger_name)))
        bkg_scores = qcd_scores_inclusive[lbl]
        bkg_wts = qcd_weights_inclusive[lbl]
        roc_data = roc_from_scores(sig_scores, bkg_scores, bkg_weights=bkg_wts)
        ref_roc_results[lbl] = roc_data

    offline_roc = ref_roc_results["Offline PNet"]
    offline_roc_upart = ref_roc_results["Offline UParT"]
    l1ng_roc = ref_roc_results["L1NG"]
    l1ext_roc = ref_roc_results["L1Ext"]


    pnet_wps = get_working_points("Offline PNet", offline_roc)
    upart_wps = get_working_points("Offline UParT", offline_roc_upart)
    l1ng_wps = get_working_points("L1NG", l1ng_roc)
    l1ext_wps = get_working_points("L1ExtJet", l1ext_roc)
    part_wps = get_working_points("Trained ParT", (fpr, tpr, roc_auc, thresholds))

    # ── Cell 9: PR curve, S/√B vs threshold, per-bin efficiency ──────
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, average_precision_score

    labels = all_labels_after_cuts
    scores = all_outputs_after_cuts
    jet_pt_cuts = val_jet_pt[val_cuts_mask]
    jet_eta_cuts = val_jet_eta[val_cuts_mask]

    # 1) Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores, sample_weight=all_weights_after_cuts)
    pr_auc = average_precision_score(labels, scores, sample_weight=all_weights_after_cuts)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="purple")
    ax.set_xlabel("Recall (Signal Efficiency)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (post-cuts) | PR-AUC={pr_auc:.4f}")
    ax.grid(True, alpha=0.3)
    save_fig(fig, "precision_recall_curve")
    plt.close(fig)

    # 2) S/√B vs threshold
    thr = np.linspace(0, 1, 200)
    s_over_root_b = []
    for t in thr:
        preds = scores >= t
        s = all_weights_after_cuts[(preds == 1) & (labels == 1)].sum()
        b = all_weights_after_cuts[(preds == 1) & (labels == 0)].sum()
        s_over_root_b.append(s / np.sqrt(b + 1e-9))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thr, s_over_root_b, color="darkgreen")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(r"$S/\sqrt{B}$")
    ax.set_title(r"$S/\sqrt{B}$ vs Threshold (post-cuts)")
    ax.grid(True, alpha=0.3)
    save_fig(fig, "s_over_root_b_vs_threshold")
    plt.close(fig)

    # 3) Per-bin efficiency tables
    pt_bins_eff = [25, 100, 200, 300, 400, 500, np.inf]
    eta_bins_eff = [0.0, 0.5, 1.0, 1.5, 2.4]
    working_points = pnet_wps


    for wp in working_points:
        df = efficiency_table(
            pt_bins_eff,
            eta_bins_eff,
            labels,
            scores,
            jet_pt_cuts,
            jet_eta_cuts,
            wp,
            sample_weights=all_weights_after_cuts,
        )
        print(f"\nPer-bin efficiencies @ threshold = {wp}")
        print(df.to_string())

    # 4) PR-AUC and optimal S/√B per kinematic bin

    df_pr_srb = pr_auc_and_opt_s_over_root_b(pt_bins_eff, eta_bins_eff, labels, scores, jet_pt_cuts, jet_eta_cuts, thr, sample_weights=all_weights_after_cuts)
    print("\nPer-bin PR-AUC and optimal S/sqrt(B)")
    print(df_pr_srb.to_string())

    if not should_run(args.profile, "standard"):
        print(
            "\nCore profile complete. Skipping full-dataset, di-Higgs, attention, "
            "activation, feature-importance, model-behavior, and AK8 parity sections."
        )
        print("\nAll analysis complete!")
        return

    # ── Cell 10: Load full dataset for constituent analysis ───────────
    data_path = config_part.get("data_path")
    if data_path is None or not os.path.exists(data_path):
        dataset_used = config_part.get("training", {}).get("data", {}).get("use_dataset", "pf")
        # In case "both" was used, default to "pf" for constituent analysis
        if dataset_used == "both": dataset_used = "pf"
        data_path = config_part.get("training", {}).get("data", {}).get(f"{dataset_used}_data_path")

    if os.path.isdir(data_path):
        pt_regression = config_part.get("model", {}).get("pt_regression", False)
        full_dataset = StratifiedJetDataset(filepath=data_path, pt_regression=pt_regression)
    else:
        full_dataset = L1JetDataset(filepath=data_path)
        
    n_total = len(full_dataset)
    BATCH_SIZE = 10000

    # We only take a maximum 100,000 jets to fit feature distributions to avoid OOM
    MAX_JETS_FOR_HIST = 100000
    n_to_load = min(n_total, MAX_JETS_FOR_HIST)

    # Accumulate batched data
    x_chunks, y_chunks, mask_chunks = [], [], []
    for start in range(0, n_to_load, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_to_load)
        if isinstance(full_dataset, StratifiedJetDataset):
            batch = full_dataset.__getitems__(list(range(start, end)))[0]
            x_b, y_b, m_b = batch[0], batch[1], batch[2]
        else:
            x_b, y_b, m_b, _ = full_dataset[start:end]
            
        x_chunks.append(x_b.numpy())
        y_chunks.append(y_b.numpy())
        mask_chunks.append(m_b.numpy())
        print(f"  Loaded batch {start}-{end} / {n_to_load}", end="\r")

    x_full = np.concatenate(x_chunks, axis=0)
    del x_chunks
    y_full = np.concatenate(y_chunks, axis=0)
    del y_chunks
    mask_full = np.concatenate(mask_chunks, axis=0)
    del mask_chunks
    print(f"\nDataset loaded: {len(x_full)} jets")

    const_mass_full = x_full[:, :, 0]
    const_pt_full = x_full[:, :, 1]
    const_eta_full = x_full[:, :, 2]
    const_phi_full = x_full[:, :, 3]

    const_vectors_full = vector.array(
        {"pt": const_pt_full, "eta": const_eta_full, "phi": const_phi_full, "mass": const_mass_full}
    )
    jet_vectors_full = const_vectors_full.sum(axis=1)

    feature_names = [
        "Mass", "Pt", "Eta", "Phi", "Dxy", "Z0", "Charge",
        "Log Pt Rel", "Delta Eta", "Delta Phi", "PUPPI weight", "Log Delta R",
    ]
    x_features = x_full[:, :, : len(feature_names)]

    # Notebook parity: decode one-hot particle IDs (features 12:17) for type-wise studies.
    PARTICLE_TYPE_NAMES = ["Charged Hadron", "Electron", "Neutral Hadron", "Photon", "Muon"]
    PARTICLE_TYPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    N_PARTICLE_TYPES = 5
    particle_type_full = None
    particle_counts_full = None
    dominant_type_full = None

    n_features_constituents = x_full.shape[2]
    if n_features_constituents > 16:
        particle_type_full = np.argmax(x_full[:, :, 12:17], axis=-1).astype(np.int8)
        mask_np_full = mask_full.astype(bool)
        particle_type_full[~mask_np_full] = -1

        particle_counts_full = np.zeros((len(x_full), N_PARTICLE_TYPES), dtype=np.uint16)
        for pid in range(N_PARTICLE_TYPES):
            particle_counts_full[:, pid] = (particle_type_full == pid).sum(axis=1)

        dominant_type_full = np.argmax(particle_counts_full, axis=1).astype(np.int8)
        total_valid = int(mask_np_full.sum())
        print("Particle type distribution across all valid constituents:")
        for pid in range(N_PARTICLE_TYPES):
            count_pid = int((particle_type_full == pid).sum())
            frac_pid = 100.0 * count_pid / max(total_valid, 1)
            print(f"  {PARTICLE_TYPE_NAMES[pid]}: {count_pid} ({frac_pid:.1f}%)")
    else:
        print("WARNING: constituent tensor has <=16 features, particle-type plots are disabled.")

    del x_full

    # Load reference jet collections for comparison
    with open("hh-bbbb-obj-config.json", "r") as f:
        config = json.load(f)

    events = load_and_prepare_data(
        config["file_pattern"],
        config["tree_name"],
        [config["offline"]["collection_name"], config["l1ng"]["collection_name"]],
        config["max_events"],
        correct_pt=True,
    )

    offline_jets = apply_custom_cuts(
        events[config["offline"]["collection_name"]], config, "offline", kinematic_only=True
    )
    l1ng_jets = apply_custom_cuts(
        events[config["l1ng"]["collection_name"]], config, "l1ng", kinematic_only=True
    )

    offline_pt = ak.to_numpy(ak.flatten(offline_jets.pt))
    offline_eta = ak.to_numpy(ak.flatten(offline_jets.eta))
    l1ng_pt = ak.to_numpy(ak.flatten(l1ng_jets.pt))
    l1ng_eta = ak.to_numpy(ak.flatten(l1ng_jets.eta))

    print(f"Reconstructed jet pT range: {jet_vectors_full.pt.min():.2f} - {jet_vectors_full.pt.max():.2f} GeV")
    print(f"Reconstructed jet eta range: {jet_vectors_full.eta.min():.2f} - {jet_vectors_full.eta.max():.2f}")

    # ── Cell 11: Generate and save all main plots ─────────────────────
    import seaborn as sns
    from sklearn.metrics import roc_auc_score
    from evaluation.roc import calculate_trained_roc_2d_bins, calculate_auc_uncertainty_2d_bins

    print(f"Saving plots to: {plot_dir}")

    # 1. Model output distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_outputs_after_cuts[all_labels_after_cuts == 0], bins=50, range=(0, 1), label="Background", histtype="step", color="red", density=True)
    ax.hist(all_outputs_after_cuts[all_labels_after_cuts == 1], bins=50, range=(0, 1), label="Signal", histtype="step", color="blue", density=True)
    ax.set_xlabel("Model Output Score")
    ax.set_ylabel("Density")
    ax.set_title("Model Output Distribution on Validation Set")
    ax.legend()
    save_fig(fig, "model_output_distribution")
    plt.close(fig)

    # 2. ROC curve comparison
    fig_roc = plot_roc_comparison(
        [
            ("Offline PNet", offline_roc),
            ("Offline UParT", offline_roc_upart),
            ("L1NG", l1ng_roc),
            ("L1ExtJet", l1ext_roc),
            ("Trained ParT", (fpr, tpr, roc_auc, thresholds)),
        ],
        working_point=0.1,
        return_fig=True,
    )
    if fig_roc:
        save_fig(fig_roc, "roc_comparison")
        plt.close(fig_roc)

    # 3. 2D AUC heatmap
    pt_ranges = [(25, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, np.inf), (25, np.inf)]
    eta_ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.4), (0, 2.4)]


    val_jet_pt_cuts = val_jet_pt[val_cuts_mask]
    val_jet_eta_cuts = val_jet_eta[val_cuts_mask]

    auc_matrix, count_matrix = calculate_trained_roc_2d_bins(
        pt_ranges, eta_ranges, val_jet_pt_cuts, val_jet_eta_cuts,
        all_labels_after_cuts, all_outputs_after_cuts, weights=all_weights_after_cuts,
    )

    pt_labels_hm = [f"[{low},{high})" for low, high in pt_ranges]
    eta_labels_hm = [f"[{low},{high})" for low, high in eta_ranges]
    annot = np.empty_like(auc_matrix, dtype=object)
    for i in range(auc_matrix.shape[0]):
        for j in range(auc_matrix.shape[1]):
            if np.isnan(auc_matrix[i, j]):
                annot[i, j] = f"N={int(count_matrix[i, j])}\n(N/A)"
            else:
                annot[i, j] = f"{auc_matrix[i, j]:.3f}\n(N={int(count_matrix[i, j])})"

    fig, ax = plt.subplots(figsize=(12, 9))
    df_hm = pd.DataFrame(auc_matrix, index=pt_labels_hm, columns=eta_labels_hm)
    sns.heatmap(df_hm, annot=annot, fmt="", cmap="viridis", vmin=0.5, vmax=1.0, ax=ax, cbar_kws={"label": "AUC"})
    ax.set_xlabel(r"$|\eta|$")
    ax.set_ylabel("Jet pT [GeV]")
    ax.set_title(r"Trained ParT: AUC in pT vs $|\eta|$ Bins")
    plt.tight_layout()
    save_fig(fig, "auc_heatmap_pt_eta")
    plt.close(fig)

    # AUC heatmap with bootstrap uncertainties
    print("Computing AUC with bootstrap uncertainties...")
    auc_matrix_u, unc_matrix_u, count_matrix_u = calculate_auc_uncertainty_2d_bins(
        pt_ranges, eta_ranges, val_jet_pt_cuts, val_jet_eta_cuts,
        all_labels_after_cuts, all_outputs_after_cuts, weights=all_weights_after_cuts, n_boot=50,
    )

    annot_u = np.empty_like(auc_matrix_u, dtype=object)
    for i in range(auc_matrix_u.shape[0]):
        for j in range(auc_matrix_u.shape[1]):
            if np.isnan(auc_matrix_u[i, j]):
                annot_u[i, j] = f"N={int(count_matrix_u[i, j])}\n(N/A)"
            elif np.isnan(unc_matrix_u[i, j]):
                annot_u[i, j] = f"{auc_matrix_u[i, j]:.3f}\n(N={int(count_matrix_u[i, j])})"
            else:
                annot_u[i, j] = f"{auc_matrix_u[i, j]:.3f}±{unc_matrix_u[i, j]:.3f}\n(N={int(count_matrix_u[i, j])})"

    fig_unc, ax_unc = plt.subplots(figsize=(14, 10))
    df_unc = pd.DataFrame(auc_matrix_u, index=pt_labels_hm, columns=eta_labels_hm)
    sns.heatmap(df_unc, annot=annot_u, fmt="", cmap="viridis", vmin=0.5, vmax=1.0, ax=ax_unc, cbar_kws={"label": "AUC"}, annot_kws={"fontsize": 9})
    ax_unc.set_xlabel(r"$|\eta|$")
    ax_unc.set_ylabel("Jet pT [GeV]")
    ax_unc.set_title(r"Trained ParT: AUC with Bootstrap Uncertainty in pT vs $|\eta|$ Bins")
    plt.tight_layout()
    save_fig(fig_unc, "auc_heatmap_pt_eta_uncertainty")
    plt.close(fig_unc)
    print("AUC uncertainty heatmap saved!")

    # 4. Constituent feature distributions
    signal_mask_full = (y_full == 1).flatten()
    background_mask_full = (y_full == 0).flatten()
    mask_np_full = mask_full

    for i, feat_name in enumerate(feature_names):
        fig, ax = plt.subplots(figsize=(10, 6))
        sig_mask_2d = (y_full == 1).squeeze()[:, np.newaxis] & mask_np_full if mask_np_full.ndim == 2 else (y_full == 1) & mask_np_full
        
        # Handle the case where squeeze might eliminate everything or broadcast is simple
        # Let's be safer: y_full has shape (N,) or (N,1). mask has shape (N, P).
        y_expanded = y_full.reshape(-1, 1)
        sig_mask_2d = (y_expanded == 1) & mask_np_full
        bkg_mask_2d = (y_expanded == 0) & mask_np_full

        sig_vals = x_features[:, :, i][sig_mask_2d]
        bkg_vals = x_features[:, :, i][bkg_mask_2d]
        ax.hist(sig_vals.flatten(), bins=50, range=(np.percentile(sig_vals, 1), np.percentile(sig_vals, 99)), histtype="step", label="Signal (b-jets)", color="blue", density=True)
        ax.hist(bkg_vals.flatten(), bins=50, range=(np.percentile(bkg_vals, 1), np.percentile(bkg_vals, 99)), histtype="step", label="Background", color="red", density=True)
        ax.legend()
        ax.set_xlabel(f"Constituent {feat_name}")
        ax.set_ylabel("Density")
        ax.set_title(f"Constituent {feat_name} Distribution")
        save_fig(fig, f"constituent_{feat_name.lower().replace(' ', '_')}")
        plt.close(fig)
    print(f"  Saved {len(feature_names)} constituent feature plots")

    # Notebook parity: constituent feature distributions stratified by particle type.
    if particle_type_full is not None:
        y_full_flat = y_full.reshape(-1)
        sig_mask_jets = y_full_flat == 1
        mask_bool = mask_np_full.astype(bool)

        for i, feat_name in enumerate(feature_names):
            fig, (ax_all, ax_sig) = plt.subplots(1, 2, figsize=(18, 6))

            for pid in range(N_PARTICLE_TYPES):
                type_mask = (particle_type_full == pid) & mask_bool
                vals_all = x_features[:, :, i][type_mask]
                vals_all = vals_all[np.isfinite(vals_all)]

                if len(vals_all) > 10:
                    lo_all = np.percentile(vals_all, 1)
                    hi_all = np.percentile(vals_all, 99)
                    if hi_all > lo_all:
                        ax_all.hist(
                            vals_all,
                            bins=50,
                            range=(lo_all, hi_all),
                            histtype="step",
                            label=PARTICLE_TYPE_NAMES[pid],
                            color=PARTICLE_TYPE_COLORS[pid],
                            density=True,
                        )

                type_sig_mask = type_mask & sig_mask_jets[:, None]
                vals_sig = x_features[:, :, i][type_sig_mask]
                vals_sig = vals_sig[np.isfinite(vals_sig)]

                if len(vals_sig) > 10:
                    lo_sig = np.percentile(vals_sig, 1)
                    hi_sig = np.percentile(vals_sig, 99)
                    if hi_sig > lo_sig:
                        ax_sig.hist(
                            vals_sig,
                            bins=50,
                            range=(lo_sig, hi_sig),
                            histtype="step",
                            label=PARTICLE_TYPE_NAMES[pid],
                            color=PARTICLE_TYPE_COLORS[pid],
                            density=True,
                        )

            ax_all.set_title(f"{feat_name} - All Jets, by Particle Type")
            ax_all.set_xlabel(f"Constituent {feat_name}")
            ax_all.set_ylabel("Density")
            ax_all.legend(fontsize="small")

            ax_sig.set_title(f"{feat_name} - Signal (b-jets), by Particle Type")
            ax_sig.set_xlabel(f"Constituent {feat_name}")
            ax_sig.set_ylabel("Density")
            ax_sig.legend(fontsize="small")

            plt.tight_layout()
            save_fig(fig, f"constituent_{feat_name.lower().replace(' ', '_')}_by_particle_type")
            plt.close(fig)

        print(f"  Saved {len(feature_names)} constituent feature plots (by particle type)")

        sig_jets = y_full_flat == 1
        bkg_jets = y_full_flat == 0

        if sig_jets.any() and bkg_jets.any():
            sig_counts_mean = particle_counts_full[sig_jets].mean(axis=0)
            bkg_counts_mean = particle_counts_full[bkg_jets].mean(axis=0)
            sig_counts_std = particle_counts_full[sig_jets].std(axis=0)
            bkg_counts_std = particle_counts_full[bkg_jets].std(axis=0)

            x_pid = np.arange(N_PARTICLE_TYPES)
            width = 0.38
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            ax1.bar(
                x_pid - width / 2,
                sig_counts_mean,
                width,
                yerr=sig_counts_std,
                label="Signal",
                color="royalblue",
                alpha=0.85,
            )
            ax1.bar(
                x_pid + width / 2,
                bkg_counts_mean,
                width,
                yerr=bkg_counts_std,
                label="Background",
                color="tomato",
                alpha=0.85,
            )
            ax1.set_xticks(x_pid)
            ax1.set_xticklabels(PARTICLE_TYPE_NAMES, rotation=30, ha="right")
            ax1.set_ylabel("Mean constituents per jet")
            ax1.set_title("Particle Type Counts: Signal vs Background")
            ax1.legend()

            sig_frac = sig_counts_mean / np.maximum(sig_counts_mean.sum(), 1e-12)
            bkg_frac = bkg_counts_mean / np.maximum(bkg_counts_mean.sum(), 1e-12)
            ax2.bar(
                x_pid - width / 2,
                sig_frac,
                width,
                label="Signal",
                color="royalblue",
                alpha=0.85,
            )
            ax2.bar(
                x_pid + width / 2,
                bkg_frac,
                width,
                label="Background",
                color="tomato",
                alpha=0.85,
            )
            ax2.set_xticks(x_pid)
            ax2.set_xticklabels(PARTICLE_TYPE_NAMES, rotation=30, ha="right")
            ax2.set_ylabel("Fraction of constituents")
            ax2.set_title("Particle Type Fractions: Signal vs Background")
            ax2.legend()

            plt.tight_layout()
            save_fig(fig, "particle_type_composition_signal_vs_background")
            plt.close(fig)

            fig, axes = plt.subplots(1, N_PARTICLE_TYPES, figsize=(24, 5), sharey=True)
            for pid in range(N_PARTICLE_TYPES):
                ax = axes[pid]
                sig_counts = particle_counts_full[sig_jets, pid]
                bkg_counts = particle_counts_full[bkg_jets, pid]
                max_sig = int(sig_counts.max()) if sig_counts.size else 0
                max_bkg = int(bkg_counts.max()) if bkg_counts.size else 0
                max_count = max(max_sig, max_bkg)
                bins = np.arange(0, max_count + 2) - 0.5

                ax.hist(
                    sig_counts,
                    bins=bins,
                    histtype="step",
                    linewidth=1.8,
                    color="royalblue",
                    density=True,
                    label="Signal",
                )
                ax.hist(
                    bkg_counts,
                    bins=bins,
                    histtype="step",
                    linewidth=1.8,
                    color="tomato",
                    density=True,
                    label="Background",
                )
                ax.set_xlabel(f"# {PARTICLE_TYPE_NAMES[pid]}s / jet")
                ax.set_title(PARTICLE_TYPE_NAMES[pid])
                if pid == 0:
                    ax.set_ylabel("Density")
                    ax.legend(fontsize="small")

            plt.suptitle("Constituent Multiplicity by Particle Type", fontsize=14)
            plt.tight_layout()
            save_fig(fig, "multiplicity_by_particle_type")
            plt.close(fig)

    # 5. Jet kinematics comparison
    trained_pt = jet_vectors_full.pt
    trained_eta = jet_vectors_full.eta

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(trained_pt, bins=100, range=(0, 500), histtype="step", label=f"Trained AK4 Jets (N={len(trained_pt)})", color="green", density=True)
    ax.hist(l1ng_pt, bins=100, range=(0, 500), histtype="step", label=f"L1NG Jets (N={len(l1ng_pt)})", color="blue", density=True)
    ax.hist(offline_pt, bins=100, range=(0, 500), histtype="step", label=f"Offline Jets (N={len(offline_pt)})", color="red", density=True)
    ax.axvline(25, color="black", linestyle="--", alpha=0.7, label="25 GeV cut")
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Jet $p_T$ Distribution Comparison")
    ax.legend()
    save_fig(fig, "jet_pt_comparison")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(trained_eta, bins=100, range=(-3, 3), histtype="step", label="Trained AK4 Jets", color="green", density=True)
    ax.hist(l1ng_eta, bins=100, range=(-3, 3), histtype="step", label="L1NG Jets", color="blue", density=True)
    ax.hist(offline_eta, bins=100, range=(-3, 3), histtype="step", label="Offline Jets", color="red", density=True)
    ax.set_xlabel(r"Jet $\eta$")
    ax.set_ylabel("Density")
    ax.set_title(r"Jet $\eta$ Distribution Comparison")
    ax.legend()
    save_fig(fig, "jet_eta_comparison")
    plt.close(fig)

    # 6. Signal vs background jet kinematics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.hist(trained_pt[signal_mask_full], bins=50, range=(0, 500), histtype="step", label=f"Signal (N={signal_mask_full.sum()})", color="blue", density=True)
    ax.hist(trained_pt[background_mask_full], bins=50, range=(0, 500), histtype="step", label=f"Background (N={background_mask_full.sum()})", color="red", density=True)
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Signal vs Background Jet $p_T$")
    ax.legend()

    ax = axes[1]
    ax.hist(trained_eta[signal_mask_full], bins=50, range=(-3, 3), histtype="step", label="Signal", color="blue", density=True)
    ax.hist(trained_eta[background_mask_full], bins=50, range=(-3, 3), histtype="step", label="Background", color="red", density=True)
    ax.set_xlabel(r"Jet $\eta$")
    ax.set_ylabel("Density")
    ax.set_title(r"Signal vs Background Jet $\eta$")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "signal_vs_background_kinematics")
    plt.close(fig)

    # Signal vs background reweighting (pT) — uses kinematic training weights
    # (not QCD weights) to visualize the effect of kinematic reweighting
    # Use val_jet_pt_cuts / val_jet_eta_cuts (validation set after cuts) to match
    # the shape of all_kinematic_weights_after_cuts
    sig_mask_cuts = all_labels_after_cuts == 1
    bkg_mask_cuts = all_labels_after_cuts == 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bins = np.linspace(0, 1000, 50)
    ax = axes[0]
    ax.hist(
        val_jet_pt_cuts[sig_mask_cuts],
        bins=bins,
        histtype="step",
        label="Signal",
        color="blue",
        density=True,
        weights=all_kinematic_weights_after_cuts[sig_mask_cuts],
    )
    ax.hist(
        val_jet_pt_cuts[bkg_mask_cuts],
        bins=bins,
        histtype="step",
        label="Background",
        color="red",
        density=True,
        weights=all_kinematic_weights_after_cuts[bkg_mask_cuts],
    )
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Signal vs Background Jet $p_T$ (Kinematic Reweighted)")

    bins_eta = np.linspace(-3, 3, 50)
    ax = axes[1]
    ax.hist(
        val_jet_eta_cuts[sig_mask_cuts],
        bins=bins_eta,
        histtype="step",
        label="Signal",
        color="blue",
        density=True,
        weights=all_kinematic_weights_after_cuts[sig_mask_cuts],
    )
    ax.hist(
        val_jet_eta_cuts[bkg_mask_cuts],
        bins=bins_eta,
        histtype="step",
        label="Background",
        color="red",
        density=True,
        weights=all_kinematic_weights_after_cuts[bkg_mask_cuts],
    )
    ax.set_xlabel(r"Jet $\eta$")
    ax.set_ylabel("Density")
    ax.set_title(r"Signal vs Background Jet $\eta$ (Kinematic Reweighted)")

    ax.legend()
    save_fig(fig, "signal_vs_background_kinematics_reweighted")
    plt.close(fig)

    print("Signal vs background reweighted kinematic plots saved!")

    # 7. 2D pT-eta distributions
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, (data_pt, data_eta, title) in zip(
        axes,
        [
            (trained_pt, trained_eta, "Trained AK4 Jets"),
            (l1ng_pt, l1ng_eta, "L1NG Jets"),
            (offline_pt, offline_eta, "Offline Jets"),
        ],
    ):
        h = ax.hist2d(data_eta, data_pt, bins=[50, 50], range=[[-3, 3], [0, 300]], cmap="viridis", norm=plt.matplotlib.colors.LogNorm())
        ax.set_xlabel(r"Jet $\eta$")
        ax.set_ylabel("Jet $p_T$ [GeV]")
        ax.set_title(f"{title}: $p_T$ vs $\\eta$")
        plt.colorbar(h[3], ax=ax, label="Counts")
    plt.tight_layout()
    save_fig(fig, "jet_pt_eta_2d_comparison")
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {plot_dir}")
    print(f"{'='*60}")

    # ── Cell 12: Regression head analysis (saved version) ─────────────
    print(f"\n--- Saving regression head analysis plot ---")
    signal_mask_reg = all_labels.squeeze() == 1
    jet_pt_reg = all_jet_pt_val
    gen_pt_reg = all_gen_pt_val

    fig_reg, axes_reg = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes_reg[0, 0]
    ax.hist(jet_pt_reg[signal_mask_reg], bins=60, range=(0, 500), histtype="step", density=True, label="Signal")
    ax.hist(jet_pt_reg[~signal_mask_reg], bins=60, range=(0, 500), histtype="step", density=True, label="Background")
    ax.set_xlabel("Reco jet $p_T$ [GeV]"); ax.set_ylabel("Density"); ax.set_title("Reco jet $p_T$ distribution"); ax.legend()

    ax = axes_reg[0, 1]
    ax.hist(gen_pt_reg[signal_mask_reg], bins=60, range=(0, 500), histtype="step", density=True, color="C0")
    ax.set_xlabel("Gen $p_T$ [GeV]"); ax.set_ylabel("Density"); ax.set_title("Gen $p_T$ (signal jets only)")

    ax = axes_reg[0, 2]
    correction_reg = gen_pt_reg[signal_mask_reg] / (jet_pt_reg[signal_mask_reg] + 1e-9)
    ax.hist(correction_reg, bins=60, range=(0, 3), histtype="step", density=True, color="C2")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="No correction")
    ax.set_xlabel("gen $p_T$ / reco $p_T$"); ax.set_ylabel("Density"); ax.set_title("Target correction factor (signal)"); ax.legend()

    if has_regression:
        reg_pt_reg = all_reg_pt
        ax = axes_reg[1, 0]
        ax.hist(reg_pt_reg[signal_mask_reg], bins=60, histtype="step", density=True, label="Signal")
        ax.hist(reg_pt_reg[~signal_mask_reg], bins=60, histtype="step", density=True, label="Background")
        ax.set_xlabel("Regression output"); ax.set_ylabel("Density"); ax.set_title("Regression head output"); ax.legend()

        ax = axes_reg[1, 1]
        true_corr_reg = gen_pt_reg[signal_mask_reg] / (jet_pt_reg[signal_mask_reg] + 1e-9)
        pred_corr_reg = reg_pt_reg[signal_mask_reg]
        ax.scatter(true_corr_reg, pred_corr_reg, s=1, alpha=0.3)
        lim = max(true_corr_reg.max(), pred_corr_reg.max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="Ideal")
        ax.set_xlabel("True correction"); ax.set_ylabel("Predicted correction"); ax.set_title("Predicted vs true correction"); ax.legend()

        ax = axes_reg[1, 2]
        corrected_pt_reg = jet_pt_reg[signal_mask_reg] * pred_corr_reg
        ax.scatter(gen_pt_reg[signal_mask_reg], corrected_pt_reg, s=1, alpha=0.3, label="Corrected")
        ax.scatter(gen_pt_reg[signal_mask_reg], jet_pt_reg[signal_mask_reg], s=1, alpha=0.1, color="gray", label="Uncorrected")
        ax.plot([0, 500], [0, 500], "r--", alpha=0.5)
        ax.set_xlabel("Gen $p_T$ [GeV]"); ax.set_ylabel("Jet $p_T$ [GeV]"); ax.set_xlim(0, 500); ax.set_ylim(0, 500)
        ax.set_title("Corrected vs uncorrected $p_T$"); ax.legend()
    else:
        for i in range(3):
            axes_reg[1, i].text(0.5, 0.5, "No regression head", ha="center", va="center", transform=axes_reg[1, i].transAxes)
            axes_reg[1, i].set_axis_off()

    plt.tight_layout()
    save_fig(fig_reg, "regression_head_analysis")
    plt.close(fig_reg)
    print(f"\nAll regression plots saved to: {plot_dir}")

    # ── Cell 13: Uncorrected / corrected pT scatter plots ─────────────
    lim = 500
    fig, ax = plt.subplots()
    ax.scatter(gen_pt_reg[signal_mask_reg], jet_pt_reg[signal_mask_reg], s=1, alpha=0.1, color="black", label="Uncorrected")
    ax.plot([0, lim], [0, lim], "r--", alpha=0.5)
    ax.set_xlabel("Gen $p_T$ [GeV]"); ax.set_ylabel("Jet $p_T$ [GeV]"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title("Uncorrected $p_T$"); ax.legend()
    save_fig(fig, "uncorrected_pt_scatter")
    plt.close(fig)

    if has_regression:
        corrected_pt_reg = jet_pt_reg[signal_mask_reg] * pred_corr_reg
        fig, ax = plt.subplots()
        ax.scatter(gen_pt_reg[signal_mask_reg], corrected_pt_reg, s=1, alpha=0.3, label="Corrected")
        ax.plot([0, lim], [0, lim], "r--", alpha=0.5)
        ax.set_xlabel("Gen $p_T$ [GeV]"); ax.set_ylabel("Jet $p_T$ [GeV]"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title("Corrected $p_T$"); ax.legend()
        save_fig(fig, "corrected_pt_scatter")
        plt.close(fig)

    # ── Cell 14: Jet pT resolution analysis ──────────────────────────
    from scipy.optimize import curve_fit
    from evaluation.resolution import gaussian, fit_response_in_bin, get_resolution_vs_var

    # Select signal jets only
    signal_mask_res = all_labels.squeeze() == 1
    jet_pt_sig = all_jet_pt_val[signal_mask_res]
    gen_pt_sig = all_gen_pt_val[signal_mask_res]
    gen_eta_sig = val_jet_eta[signal_mask_res]

    # Uncorrected pT response
    raw_response = jet_pt_sig / (gen_pt_sig + 1e-9)

    # Corrected pT response
    if has_regression:
        pred_scale = all_reg_pt[signal_mask_res]
        corrected_pt_res = jet_pt_sig * pred_scale
        corr_response = corrected_pt_res / (gen_pt_sig + 1e-9)
    else:
        corrected_pt_res = jet_pt_sig
        corr_response = raw_response
        pred_scale = np.ones_like(jet_pt_sig)

    # Predicted resolution from quantile head
    if has_quantile:
        q16 = all_quantiles[signal_mask_res, 0]
        q84 = all_quantiles[signal_mask_res, 1]
        predicted_resolution_abs = 0.5 * (q84 - q16) * jet_pt_sig * pred_scale
        predicted_resolution_rel = 0.5 * (q84 - q16)
        print(f"Predicted resolution (absolute) range: {predicted_resolution_abs.min():.2f} – {predicted_resolution_abs.max():.2f} GeV")
        print(f"Predicted resolution (relative)  range: {predicted_resolution_rel.min():.4f} – {predicted_resolution_rel.max():.4f}")

    # Binning
    pt_bins_res = np.linspace(25, 500, 30)
    eta_bins_res = np.linspace(-2.4, 2.4, 20)

    bc_pt, mu_raw_pt, sig_raw_pt, mu_raw_pt_err, sig_raw_pt_err = get_resolution_vs_var(gen_pt_sig, raw_response, pt_bins_res)
    bc_pt, mu_corr_pt, sig_corr_pt, mu_corr_pt_err, sig_corr_pt_err = get_resolution_vs_var(gen_pt_sig, corr_response, pt_bins_res)
    bc_eta, mu_raw_eta, sig_raw_eta, mu_raw_eta_err, sig_raw_eta_err = get_resolution_vs_var(gen_eta_sig, raw_response, eta_bins_res)
    bc_eta, mu_corr_eta, sig_corr_eta, mu_corr_eta_err, sig_corr_eta_err = get_resolution_vs_var(gen_eta_sig, corr_response, eta_bins_res)

    if has_quantile:
        def avg_in_bins(gen_var, values, var_bins):
            bc = 0.5 * (var_bins[1:] + var_bins[:-1])
            means, errs = [], []
            for i in range(len(var_bins) - 1):
                m = (gen_var > var_bins[i]) & (gen_var <= var_bins[i + 1])
                v = values[m]
                if len(v) > 5:
                    means.append(np.mean(v)); errs.append(np.std(v) / np.sqrt(len(v)))
                else:
                    means.append(np.nan); errs.append(0.0)
            return bc, np.array(means), np.array(errs)

        _, pred_res_vs_pt, pred_res_vs_pt_err = avg_in_bins(gen_pt_sig, predicted_resolution_rel, pt_bins_res)
        _, pred_res_vs_eta, pred_res_vs_eta_err = avg_in_bins(gen_eta_sig, predicted_resolution_rel, eta_bins_res)

    print("Resolution computation complete.")

    # ── Cell 15: pT response distributions ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1a. Overall response distribution
    ax = axes[0]
    ax.hist(
        raw_response,
        bins=np.linspace(0, 2, 81),
        histtype="step",
        density=True,
        label="Uncorrected",
        color="C0",
        linewidth=1.5,
    )
    if has_regression:
        ax.hist(
            corr_response,
            bins=np.linspace(0, 2, 81),
            histtype="step",
            density=True,
            label="Corrected (regression)",
            color="C1",
            linewidth=1.5,
        )
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal (1.0)")
    ax.set_xlabel("$p_T$ Response (reco $p_T$ / gen $p_T$)")
    ax.set_ylabel("Density")
    ax.set_title("Jet $p_T$ Response Distribution (Signal Jets)")
    ax.legend()
    ax.set_xlim(0, 2)

    # 1b. Response in selected gen pT bins
    ax = axes[1]
    pt_slices = [(50, 100), (100, 200), (200, 300), (300, 500)]
    colors_sl = plt.cm.viridis(np.linspace(0.15, 0.85, len(pt_slices)))
    for (pt_lo, pt_hi), color_sl in zip(pt_slices, colors_sl):
        mask = (gen_pt_sig >= pt_lo) & (gen_pt_sig < pt_hi)
        if has_regression:
            ax.hist(
                corr_response[mask],
                bins=np.linspace(0, 2, 61),
                histtype="step",
                density=True,
                color=color_sl,
                linewidth=1.5,
                label=f"Corrected [{pt_lo},{pt_hi}) GeV",
            )
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("$p_T$ Response")
    ax.set_ylabel("Density")
    ax.set_title("$p_T$ Response in gen $p_T$ Bins")
    ax.legend(fontsize="x-small", ncol=1)
    ax.set_xlim(0, 2)

    plt.tight_layout()
    save_fig(fig, "pt_response_distributions")
    plt.close(fig)

    # ── Cell 16: Uncorrected response in gen pT bins ──────────────────
    fig, ax = plt.subplots()
    for (pt_lo, pt_hi), color_sl in zip(pt_slices, colors_sl):
        mask = (gen_pt_sig >= pt_lo) & (gen_pt_sig < pt_hi)
        if has_regression:
            ax.hist(
                raw_response[mask],
                bins=np.linspace(0, 2, 61),
                histtype="step",
                density=True,
                color=color_sl,
                linewidth=1.0,
                linestyle="--",
                label=f"Uncorrected [{pt_lo},{pt_hi}) GeV",
            )
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("$p_T$ Response")
    ax.set_ylabel("Density")
    ax.set_title("$p_T$ Response in gen $p_T$ Bins")
    ax.legend(fontsize="x-small", ncol=1)
    ax.set_xlim(0, 2)
    plt.tight_layout()
    plt.close(fig)

    # ── Cell 17: Scale & resolution vs gen pT ────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top panel: Scale (mean of response) vs gen pT
    ax = axes[0]
    ax.errorbar(
        bc_pt, mu_raw_pt, yerr=mu_raw_pt_err,
        marker="o", linestyle="-", capsize=4, label="Uncorrected", color="C0",
    )
    if has_regression:
        ax.errorbar(
            bc_pt, mu_corr_pt, yerr=mu_corr_pt_err,
            marker="s", linestyle="-", capsize=4, label="Corrected (regression)", color="C1",
        )
    ax.axhline(1.0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("Jet $p_T$ Scale (Mean of Response)")
    ax.set_title("Jet $p_T$ Scale & Resolution vs. Generated $p_T$ (Signal Jets)")
    ax.legend()
    ax.set_ylim(0.5, 1.5)

    # Bottom panel: Resolution (sigma) vs gen pT
    ax = axes[1]
    ax.errorbar(
        bc_pt, sig_raw_pt, yerr=sig_raw_pt_err,
        marker="o", linestyle="-", capsize=4, label="Calculated (uncorrected)", color="C0",
    )
    if has_regression:
        ax.errorbar(
            bc_pt, sig_corr_pt, yerr=sig_corr_pt_err,
            marker="s", linestyle="-", capsize=4, label="Calculated (corrected)", color="C1",
        )
    if has_quantile:
        ax.errorbar(
            bc_pt, pred_res_vs_pt, yerr=pred_res_vs_pt_err,
            marker="^", linestyle="--", capsize=4, label="Predicted (quantile head)", color="C2",
        )
    ax.set_xlabel("Generated Jet $p_T$ [GeV]")
    ax.set_ylabel("Resolution ($\\sigma$ of Response)")
    ax.legend()
    ax.set_ylim(0)

    plt.tight_layout()
    save_fig(fig, "resolution_vs_gen_pt")
    plt.close(fig)

    # ── Cell 18: Scale & resolution vs eta ───────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top: Scale vs eta
    ax = axes[0]
    ax.errorbar(
        bc_eta, mu_raw_eta, yerr=mu_raw_eta_err,
        marker="o", linestyle="-", capsize=4, label="Uncorrected", color="C0",
    )
    if has_regression:
        ax.errorbar(
            bc_eta, mu_corr_eta, yerr=mu_corr_eta_err,
            marker="s", linestyle="-", capsize=4, label="Corrected (regression)", color="C1",
        )
    ax.axhline(1.0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel("Jet $p_T$ Scale (Mean of Response)")
    ax.set_title("Jet $p_T$ Scale & Resolution vs. Jet $\\eta$ (Signal Jets)")
    ax.legend()
    ax.set_ylim(0.5, 1.5)

    # Bottom: Resolution vs eta
    ax = axes[1]
    ax.errorbar(
        bc_eta, sig_raw_eta, yerr=sig_raw_eta_err,
        marker="o", linestyle="-", capsize=4, label="Calculated (uncorrected)", color="C0",
    )
    if has_regression:
        ax.errorbar(
            bc_eta, sig_corr_eta, yerr=sig_corr_eta_err,
            marker="s", linestyle="-", capsize=4, label="Calculated (corrected)", color="C1",
        )
    if has_quantile:
        ax.errorbar(
            bc_eta, pred_res_vs_eta, yerr=pred_res_vs_eta_err,
            marker="^", linestyle="--", capsize=4, label="Predicted (quantile head)", color="C2",
        )
    ax.set_xlabel("Jet $\\eta$")
    ax.set_ylabel("Resolution ($\\sigma$ of Response)")
    ax.legend()
    ax.set_ylim(0)

    plt.tight_layout()
    save_fig(fig, "resolution_vs_eta")
    plt.close(fig)

    # ── Cell 19: 2D response maps ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # 4a. Uncorrected pT response vs gen pT
    ax = axes[0]
    h0 = ax.hist2d(
        gen_pt_sig, raw_response,
        bins=[np.linspace(25, 500, 51), np.linspace(0, 2, 51)],
        cmap="viridis", cmin=1,
    )
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Generated $p_T$ [GeV]")
    ax.set_ylabel("$p_T$ Response (reco / gen)")
    ax.set_title("Uncorrected $p_T$ Response")
    plt.colorbar(h0[3], ax=ax, label="Counts")

    # 4b. Corrected pT response vs gen pT
    ax = axes[1]
    h1 = ax.hist2d(
        gen_pt_sig, corr_response,
        bins=[np.linspace(25, 500, 51), np.linspace(0, 2, 51)],
        cmap="viridis", cmin=1,
    )
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Generated $p_T$ [GeV]")
    ax.set_ylabel("$p_T$ Response (corrected / gen)")
    ax.set_title("Corrected $p_T$ Response")
    plt.colorbar(h1[3], ax=ax, label="Counts")

    # 4c. Predicted resolution vs gen pT (scatter)
    ax = axes[2]
    if has_quantile:
        sc = ax.scatter(
            gen_pt_sig, predicted_resolution_rel,
            c=jet_pt_sig, s=1, alpha=0.3, cmap="plasma",
        )
        plt.colorbar(sc, ax=ax, label="Reco Jet $p_T$ [GeV]")
        ax.set_xlabel("Generated $p_T$ [GeV]")
        ax.set_ylabel("Predicted Relative Resolution")
        ax.set_title("Quantile-Predicted Resolution vs. Gen $p_T$")
        ax.set_ylim(0, 0.5)
    else:
        ax.text(0.5, 0.5, "No quantile head", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()

    plt.tight_layout()
    save_fig(fig, "response_2d_maps")
    plt.close(fig)

    # ── Cell 20: Gaussian fits in gen pT slices ──────────────────────
    pt_slices_fit = [(25, 100), (100, 150), (200, 250), (300, 400)]
    n_slices = len(pt_slices_fit)
    fig, axes = plt.subplots(2, n_slices, figsize=(5 * n_slices, 10))

    response_bins = np.linspace(0, 2, 61)
    x_fit = np.linspace(0, 2, 200)

    for j, (pt_lo, pt_hi) in enumerate(pt_slices_fit):
        mask = (gen_pt_sig >= pt_lo) & (gen_pt_sig < pt_hi)

        # Top row: Uncorrected
        ax = axes[0, j]
        vals_raw = raw_response[mask]
        counts_raw, _ = np.histogram(vals_raw, bins=response_bins)
        (mu_r, sig_r, A_r), _ = fit_response_in_bin(vals_raw, response_bins)
        ax.stairs(counts_raw, response_bins, color="C0", linewidth=1.5, label="Uncorrected")
        if not np.isnan(mu_r):
            ax.plot(
                x_fit, gaussian(x_fit, mu_r, sig_r, A_r), "C0--",
                label=f"$\\mu$={mu_r:.3f}, $\\sigma$={abs(sig_r):.3f}",
            )
        ax.set_title(f"gen $p_T$ [{pt_lo},{pt_hi}) GeV\n(Uncorrected)")
        ax.set_xlabel("$p_T$ Response")
        ax.set_ylabel("Counts")
        ax.legend(fontsize="x-small")

        # Bottom row: Corrected
        ax = axes[1, j]
        if has_regression:
            vals_corr = corr_response[mask]
            counts_corr, _ = np.histogram(vals_corr, bins=response_bins)
            (mu_c, sig_c, A_c), _ = fit_response_in_bin(vals_corr, response_bins)
            ax.stairs(counts_corr, response_bins, color="C1", linewidth=1.5, label="Corrected")
            if not np.isnan(mu_c):
                ax.plot(
                    x_fit, gaussian(x_fit, mu_c, sig_c, A_c), "C1--",
                    label=f"$\\mu$={mu_c:.3f}, $\\sigma$={abs(sig_c):.3f}",
                )
            ax.set_title(f"gen $p_T$ [{pt_lo},{pt_hi}) GeV\n(Corrected)")
        else:
            ax.text(0.5, 0.5, "No regression head", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel("$p_T$ Response")
        ax.set_ylabel("Counts")
        ax.legend(fontsize="x-small")

    plt.tight_layout()
    save_fig(fig, "response_gaussian_fits_pt_slices")
    plt.close(fig)

    # ── Cell 21: Absolute pT resolution & improvement ────────────────
    if has_quantile:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 6a. Absolute resolution: predicted vs calculated, before & after
        ax = axes[0]
        abs_raw_res = sig_raw_pt * bc_pt
        abs_corr_res = sig_corr_pt * bc_pt
        ax.errorbar(
            bc_pt, abs_raw_res,
            marker="o", linestyle="-", capsize=4, label="Calculated (uncorrected)", color="C0",
        )
        if has_regression:
            ax.errorbar(
                bc_pt, abs_corr_res,
                marker="s", linestyle="-", capsize=4, label="Calculated (corrected)", color="C1",
            )
        _, pred_abs_vs_pt, pred_abs_vs_pt_err = avg_in_bins(
            gen_pt_sig, predicted_resolution_abs, pt_bins_res
        )
        ax.errorbar(
            bc_pt, pred_abs_vs_pt, yerr=pred_abs_vs_pt_err,
            marker="^", linestyle="--", capsize=4, label="Predicted (quantile head)", color="C2",
        )
        ax.set_xlabel("Generated Jet $p_T$ [GeV]")
        ax.set_ylabel("Absolute $p_T$ Resolution [GeV]")
        ax.set_title("Absolute Jet $p_T$ Resolution vs. Gen $p_T$")
        ax.legend()
        ax.set_ylim(0)

        # 6b. Fractional improvement
        ax = axes[1]
        if has_regression:
            valid = (~np.isnan(sig_raw_pt)) & (~np.isnan(sig_corr_pt)) & (sig_raw_pt > 0)
            improvement = (sig_raw_pt[valid] - sig_corr_pt[valid]) / sig_raw_pt[valid] * 100
            ax.bar(
                bc_pt[valid], improvement,
                width=np.diff(pt_bins_res).mean() * 0.7, color="C3", alpha=0.7,
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Generated Jet $p_T$ [GeV]")
            ax.set_ylabel("Resolution Improvement [%]")
            ax.set_title("Resolution Improvement from Regression Correction")
            mean_improvement = np.nanmean(improvement)
            ax.text(
                0.05, 0.95, f"Mean improvement: {mean_improvement:.1f}%",
                transform=ax.transAxes, va="top", fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            ax.text(0.5, 0.5, "No regression head", ha="center", va="center",
                    transform=ax.transAxes)

        plt.tight_layout()
        save_fig(fig, "absolute_resolution_and_improvement")
        plt.close(fig)
    else:
        print("Skipping absolute resolution plots (no quantile regression head).")

    print(f"\nAll resolution plots saved to: {plot_dir}")

    # ── Cell 22: Di-Higgs mass reconstruction with trained ParT ──────
    import fastjet
    from itertools import combinations
    from data_pipeline.make_particle_dataset import cluster_candidates
    from data_pipeline.root_loading import load_and_prepare_data, select_gen_b_quarks_from_higgs, apply_custom_cuts, one_hot_encode_l1_puppi
    from evaluation.jet_matching import get_purity_mask_cross_matched
    from evaluation.dihiggs import pair_from_4jets, find_gen_b_pairs_with_indices, R_hh_func

    # Configuration
    apply_pt_correction = True
    WP_SELECTION = "tight"
    wp_index = {"tight": 0, "medium": 1, "loose": 2}[WP_SELECTION]
    PART_BTAG_THRESHOLD = part_wps[wp_index]
    print(f"\nWorking point: {WP_SELECTION} → ParT b-tag threshold = {PART_BTAG_THRESHOLD:.4f}")

    dataset_used = config_part.get("training", {}).get("data", {}).get("use_dataset", "pf")
    if dataset_used == "pf":
        collection_key = "l1extpf"
    elif dataset_used == "puppi":
        collection_key = "l1extpuppi"
    else:
        collection_key = "l1barrelextpf"
    print(f"Model was trained on: {dataset_used} → clustering {collection_key}")

    root_data_pattern = config["file_pattern"]
    collection_name = config[collection_key]["collection_name"]
    print(f"ROOT data: {root_data_pattern}")
    print(f"Collection: {collection_name}")

    n_constituents = n_constituents_model if n_constituents_model is not None else all_constituents.shape[1]

    def cluster_and_score(
        events,
        cfg,
        collection_key,
        model,
        device,
        config_part,
        n_constituents,
        apply_pt_correction=True,
    ):
        clustered_jets = cluster_candidates(events, cfg, collection_key, dist_param=0.4)
        sorted_indices = ak.argsort(clustered_jets.pt, axis=1, ascending=False)
        l1_clustered = clustered_jets[sorted_indices]
        matched_cands = l1_clustered.constituents
        const_pt_sort = ak.argsort(matched_cands.pt, axis=2, ascending=False)
        matched_cands = matched_cands[const_pt_sort]

        j_pt = l1_clustered.pt[:, :, None]
        j_eta = l1_clustered.eta[:, :, None]
        j_phi = l1_clustered.phi[:, :, None]

        m_pt = matched_cands.vector.pt
        m_eta = matched_cands.vector.eta
        m_phi = matched_cands.vector.phi
        m_mass = matched_cands.vector.mass
        m_dxy = matched_cands.dxy
        m_z0 = matched_cands.z0
        m_charge = matched_cands.charge
        m_w = matched_cands.puppiWeight
        m_id = matched_cands.id

        log_pt_rel = np.log(np.maximum(m_pt, 1e-3) / np.maximum(j_pt, 1e-3))
        deta = m_eta - j_eta
        dphi = m_phi - j_phi
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        log_dr = np.log(np.maximum(np.sqrt(deta**2 + dphi**2), 1e-3))

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

        n_jets_per_event = ak.num(l1_clustered, axis=1)
        n_actual_constituents = ak.num(matched_cands, axis=2)
        n_actual_flat = ak.to_numpy(ak.flatten(n_actual_constituents, axis=1))

        x_ini = np.stack([ak.to_numpy(ak.flatten(f, axis=1)) for f in feature_list], axis=-1)
        flat_ids = x_ini[..., -1]
        one_hot_ids = one_hot_encode_l1_puppi(flat_ids, n_classes=5)
        X_feat = np.concatenate([x_ini[..., :-1], one_hot_ids], axis=-1)

        particle_mask = np.zeros((X_feat.shape[0], n_constituents), dtype=bool)
        for i in range(X_feat.shape[0]):
            n_real = min(n_actual_flat[i], n_constituents)
            particle_mask[i, :n_real] = True

        const_vecs = vector.array(
            {
                "pt": x_ini[:, :, 1],
                "eta": x_ini[:, :, 2],
                "phi": x_ini[:, :, 3],
                "mass": x_ini[:, :, 0],
            }
        )
        jet_4v = const_vecs.sum(axis=1)
        flat_jet_pt = jet_4v.pt
        flat_jet_eta = jet_4v.eta
        flat_jet_phi = jet_4v.phi
        flat_jet_mass = jet_4v.mass

        batch_size = config_part.get("training", {}).get("batch_size", 512)
        all_scores, all_reg = [], []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(X_feat), batch_size):
                end = min(start + batch_size, len(X_feat))
                xb = torch.tensor(X_feat[start:end], dtype=torch.float32).to(device)
                mb = torch.tensor(particle_mask[start:end], dtype=torch.bool).to(device)
                out = model(xb, particle_mask=mb)
                scores = (
                    torch.nn.functional.sigmoid(out["classification"])
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                all_scores.append(scores)
                if "pt" in out:
                    all_reg.append(out["pt"].squeeze().cpu().numpy())

        all_scores = np.concatenate(all_scores)
        has_reg = len(all_reg) > 0
        if has_reg:
            all_reg = np.concatenate(all_reg)

        if has_reg and apply_pt_correction:
            corrected_pt = flat_jet_pt * all_reg
        else:
            corrected_pt = flat_jet_pt

        corr_vecs = vector.array(
            {
                "pt": corrected_pt,
                "eta": flat_jet_eta,
                "phi": flat_jet_phi,
                "mass": flat_jet_mass * (corrected_pt / (flat_jet_pt + 1e-9)),
            }
        )

        n_jets_np = ak.to_numpy(n_jets_per_event)
        cumulative = np.concatenate([[0], np.cumsum(n_jets_np)])
        evt_pts, evt_etas, evt_phis, evt_masses, evt_scores = [], [], [], [], []
        for i in range(len(n_jets_np)):
            s, e = cumulative[i], cumulative[i + 1]
            evt_pts.append(corr_vecs.pt[s:e])
            evt_etas.append(corr_vecs.eta[s:e])
            evt_phis.append(corr_vecs.phi[s:e])
            evt_masses.append(corr_vecs.mass[s:e])
            evt_scores.append(all_scores[s:e])

        scored_jets = ak.zip(
            {
                "pt": ak.Array(evt_pts),
                "eta": ak.Array(evt_etas),
                "phi": ak.Array(evt_phis),
                "mass": ak.Array(evt_masses),
                "btag_score": ak.Array(evt_scores),
            }
        )
        scored_jets["vector"] = ak.zip(
            {
                "pt": scored_jets.pt,
                "eta": scored_jets.eta,
                "phi": scored_jets.phi,
                "mass": scored_jets.mass,
            },
            with_name="Momentum4D",
        )

        return scored_jets, has_reg

    print("\n--- Loading ROOT data ---")
    dihiggs_events = load_and_prepare_data(
        root_data_pattern,
        config["tree_name"],
        [collection_name, "GenPart"],
        max_events=config["max_events"],
        correct_pt=False,
        CONFIG=config,
    )

    print("\n--- Clustering & scoring signal jets ---")
    scored_jets, has_reg = cluster_and_score(
        dihiggs_events,
        config,
        collection_key,
        model,
        device,
        config_part,
        n_constituents,
        apply_pt_correction,
    )

    n_events_total = len(scored_jets)
    n_jets_total = int(ak.sum(ak.num(scored_jets, axis=1)))
    print(f"Clustered & scored {n_jets_total} jets across {n_events_total} events")

    # Step 4: Di-Higgs reconstruction
    print("\n--- Running di-Higgs reconstruction ---")

    dihiggs_gen_b = select_gen_b_quarks_from_higgs(dihiggs_events)
    dihiggs_gen_b = dihiggs_gen_b[
        (dihiggs_gen_b.pt > config["gen"]["pt_cut"])
        & (abs(dihiggs_gen_b.eta) < config["gen"]["eta_cut"])
    ]

    jets_btag_sorted = scored_jets[ak.argsort(scored_jets.btag_score, ascending=False)]
    jets_tagged = jets_btag_sorted[jets_btag_sorted.btag_score > PART_BTAG_THRESHOLD]
    print(f"Jets per event after b-tag cut: mean={ak.mean(ak.num(jets_tagged)):.1f}")

    has_4_tagged = ak.num(jets_tagged) >= 4
    sig_jets_all = jets_tagged[has_4_tagged][:, :4]
    print(f"Events with >=4 tagged jets: {ak.sum(has_4_tagged)}/{len(jets_tagged)}")

    gen_b_for_match = dihiggs_gen_b[has_4_tagged]
    dr_reco = sig_jets_all[:, :, None].vector.deltaR(gen_b_for_match[:, None, :].vector)
    idx_gen_for_reco = ak.argmin(dr_reco, axis=2)
    min_dr_reco = ak.fill_none(ak.min(dr_reco, axis=2), np.inf)

    dr_gen = gen_b_for_match[:, :, None].vector.deltaR(sig_jets_all[:, None, :].vector)
    idx_reco_for_gen = ak.argmin(dr_gen, axis=2)

    back_check = idx_reco_for_gen[idx_gen_for_reco]
    reco_idx = ak.local_index(sig_jets_all, axis=1)

    pure_mask = (ak.fill_none(back_check, -1) == reco_idx) & (min_dr_reco < config["matching_cone_size"])

    signal_mask_evt = ak.sum(pure_mask, axis=1) == 4

    n_signal = int(ak.sum(signal_mask_evt))
    n_total = int(ak.sum(has_4_tagged))
    print(f"Signal events (all 4 pure): {n_signal}")
    print(f"Total events with >=4 tagged jets: {n_total}")


    sig_jets_4 = sig_jets_all[signal_mask_evt][:, :4]
    if n_signal > 0:
        sig_lead, sig_sub, sig_hh = pair_from_4jets(sig_jets_4)
        print(f"\nSignal mHH: mean={ak.mean(sig_hh.mass):.1f}, median={np.median(ak.to_numpy(sig_hh.mass)):.1f} GeV")
    else:
        sig_lead = sig_sub = sig_hh = ak.Array([])
        print("\nNo signal events found!")

    # Pairing efficiency
    pair_eff = 0.0
    eff_swapped = 0.0
    if n_signal > 0:
        print("\n--- Computing Pairing Efficiency ---")
        sig_gen_b = gen_b_for_match[signal_mask_evt]
        sig_gen_particles = dihiggs_events.GenPart[has_4_tagged][signal_mask_evt]

        dr_sig = sig_jets_4[:, :, None].vector.deltaR(sig_gen_b[:, None, :].vector)
        gmpr = ak.argmin(dr_sig, axis=2)


        _, _, true_h1_idxs, true_h2_idxs = find_gen_b_pairs_with_indices(gmpr, sig_gen_b, sig_gen_particles)
        true_h1_sorted = ak.sort(true_h1_idxs, axis=1)
        true_h2_sorted = ak.sort(true_h2_idxs, axis=1)

        j = [sig_jets_4[:, i] for i in range(4)]
        perm_pairs = [([0, 1], [2, 3]), ([0, 2], [1, 3]), ([0, 3], [1, 2])]
        h_vecs = [
            (j[a].vector + j[b].vector, j[c].vector + j[d].vector)
            for (a, b), (c, d) in perm_pairs
        ]
        m1 = ak.concatenate([v[0].mass[:, None] for v in h_vecs], axis=1)
        m2 = ak.concatenate([v[1].mass[:, None] for v in h_vecs], axis=1)
        d_hh = abs(m1 - (125.0 / 120.0) * m2) / np.sqrt(1 + (125.0 / 120.0) ** 2)
        best = ak.argmin(d_hh, axis=1)

        p1_c1, p1_c2 = gmpr[:, [0, 1]], gmpr[:, [2, 3]]
        p2_c1, p2_c2 = gmpr[:, [0, 2]], gmpr[:, [1, 3]]
        p3_c1, p3_c2 = gmpr[:, [0, 3]], gmpr[:, [1, 2]]

        c0, c1_flag = (best == 0), (best == 1)
        algo_pair_A = ak.where(c0[:, None], p1_c1, ak.where(c1_flag[:, None], p2_c1, p3_c1))
        algo_pair_B = ak.where(c0[:, None], p1_c2, ak.where(c1_flag[:, None], p2_c2, p3_c2))

        raw_h1_v = ak.where(c0, h_vecs[0][0], ak.where(c1_flag, h_vecs[1][0], h_vecs[2][0]))
        raw_h2_v = ak.where(c0, h_vecs[0][1], ak.where(c1_flag, h_vecs[1][1], h_vecs[2][1]))
        is_lead_v = raw_h1_v.pt >= raw_h2_v.pt
        algo_pair_leading = ak.where(is_lead_v[:, None], algo_pair_A, algo_pair_B)
        algo_pair_subleading = ak.where(is_lead_v[:, None], algo_pair_B, algo_pair_A)
        algo_A_sorted = ak.sort(algo_pair_leading, axis=1)
        algo_B_sorted = ak.sort(algo_pair_subleading, axis=1)

        match_direct = ak.all(algo_A_sorted == true_h1_sorted, axis=1) & ak.all(algo_B_sorted == true_h2_sorted, axis=1)
        pair_eff = ak.mean(match_direct)
        print(f"Pairing Efficiency: {pair_eff:.2%}")

        match_swapped = ak.all(algo_A_sorted == true_h2_sorted, axis=1) & ak.all(algo_B_sorted == true_h1_sorted, axis=1)
        eff_swapped = ak.mean(match_swapped)
        print(f"Pairing Efficiency with pairs swapped: {eff_swapped:.2%}")
        print(f"Total Pairing Efficiency (including swapped): {pair_eff + eff_swapped:.2%}")
    else:
        print("\nSkipping pairing efficiency (no signal events)")

    # QCD background — from QCD pT-binned ROOT files
    print("\n" + "=" * 60)
    print("Processing QCD BACKGROUND...")
    print("=" * 60)
    qcd_config = config["QCD_background"]
    sigma_to_ngen = {bin_cfg["weight"]: bin_cfg["n_gen"] for bin_cfg in qcd_config.values()}
    all_qcd_lead, all_qcd_sub, all_qcd_hh = [], [], []
    all_qcd_weights_list = []
    n_qcd_total = 0
    n_qcd_events_processed = 0

    for bin_name, bin_cfg in qcd_config.items():
        print(f"\n--- QCD bin: {bin_name}  (weight={bin_cfg['weight']:.3e}) ---")
        qcd_file_pattern = bin_cfg["file_pattern"]
        max_events_bin = bin_cfg.get("max_events", 1000)

        qcd_cfg = dict(config)
        qcd_cfg["file_pattern"] = qcd_file_pattern
        qcd_cfg["tree_name"] = bin_cfg["tree_name"]
        qcd_cfg["max_events"] = max_events_bin

        try:
            qcd_events = load_and_prepare_data(
                qcd_file_pattern,
                bin_cfg["tree_name"],
                [collection_name, "GenPart"],
                max_events=max_events_bin,
                correct_pt=False,
                CONFIG=qcd_cfg,
            )
        except Exception as e:
            print(f"  Error loading {bin_name}: {e}")
            continue

        if len(qcd_events) == 0:
            print(f"  No events loaded for {bin_name}, skipping.")
            continue

        n_loaded = len(qcd_events)
        n_qcd_events_processed += n_loaded
        print(f"  Loaded {n_loaded} events, clustering & scoring...")

        qcd_scored, _ = cluster_and_score(
            qcd_events,
            qcd_cfg,
            collection_key,
            model,
            device,
            config_part,
            n_constituents,
            apply_pt_correction,
        )

        qcd_btag_sorted = qcd_scored[ak.argsort(qcd_scored.btag_score, ascending=False)]
        qcd_tagged = qcd_btag_sorted[qcd_btag_sorted.btag_score > PART_BTAG_THRESHOLD]
        has_4_tagged_qcd = ak.num(qcd_tagged) >= 4
        qcd_4jets = qcd_tagged[has_4_tagged_qcd][:, :4]

        n_events_bin = int(ak.sum(has_4_tagged_qcd))
        n_events_total_bin = len(qcd_scored)
        print(f"  Events with >=4 tagged jets: {n_events_bin}/{n_events_total_bin}")

        if n_events_bin > 0:
            q_lead, q_sub, q_hh = pair_from_4jets(qcd_4jets)
            all_qcd_lead.append(q_lead)
            all_qcd_sub.append(q_sub)
            all_qcd_hh.append(q_hh)
            all_qcd_weights_list.append(
                np.full(n_events_bin, bin_cfg["weight"], dtype=np.float64)
            )
            n_qcd_total += n_events_bin
            print(f"  → {n_events_bin} QCD events with >=4 b-tagged jets reconstructed")
        else:
            print(f"  → No events with >=4 b-tagged jets in {bin_name}")

    if n_qcd_total > 0:
        qcd_lead = ak.concatenate(all_qcd_lead)
        qcd_sub = ak.concatenate(all_qcd_sub)
        qcd_hh = ak.concatenate(all_qcd_hh)
        qcd_weights = np.concatenate(all_qcd_weights_list)
        print(
            f"\nTotal QCD: {n_qcd_total} reconstructed events from {n_qcd_events_processed} processed"
        )
        print(
            f"QCD weights summary: min={qcd_weights.min():.1e}, max={qcd_weights.max():.1e}, "
            f"sum={qcd_weights.sum():.3e}"
        )
        print(
            f"QCD mHH: weighted mean={np.average(ak.to_numpy(qcd_hh.mass), weights=qcd_weights):.1f}, "
            f"unweighted median={np.median(ak.to_numpy(qcd_hh.mass)):.1f} GeV"
        )
    else:
        qcd_lead = qcd_sub = qcd_hh = ak.Array([])
        qcd_weights = np.array([], dtype=np.float64)
        print("\nNo QCD background events found!")

    n_qcd = n_qcd_total

    part_dihiggs_result = {
        "label": f"Trained ParT ({WP_SELECTION})",
        "n_total": n_total, "n_signal": n_signal, "n_qcd": n_qcd,
        "sig_lead": sig_lead, "sig_sub": sig_sub, "sig_hh": sig_hh,
        "qcd_lead": qcd_lead, "qcd_sub": qcd_sub, "qcd_hh": qcd_hh,
        "qcd_weights": qcd_weights,
        "sigma_to_ngen": sigma_to_ngen,
        "collection_key": collection_key, "wp": WP_SELECTION,
        "threshold": PART_BTAG_THRESHOLD, "has_regression": has_reg,
        "pair_eff": float(pair_eff), "eff_swapped": float(eff_swapped),
    }

    print(f"\n{'='*60}")
    print(f"Di-Higgs reconstruction complete")
    print(f"  Collection: {collection_key} ({dataset_used})")
    print(f"  Working point: {WP_SELECTION} (threshold={PART_BTAG_THRESHOLD:.4f})")
    print(
        f"  Selected: {n_total} signal events with >=4 tagged jets = {n_total/n_events_total*100:.2f}% of {n_events_total} total HH events"
    )
    print(f"  Signal (all 4 pure): {n_signal} events ({n_signal/max(n_total,1)*100:.1f}%)")
    print(
        f"  QCD background: {n_qcd} events (from {n_qcd_events_processed} QCD events processed)"
    )
    print(f"  pT regression: {'applied' if has_reg else 'not available'}")
    print(f"  Pairing eff: {pair_eff:.2%} | Swapped: {eff_swapped:.2%} | Total: {pair_eff + eff_swapped:.2%}")
    print(f"{'='*60}")

    # ── Cell 23: Top-N jet purity efficiency ─────────────────────────
    from evaluation.jet_matching import get_pure_jet_idxs_cross_matched

    def get_eff_first_jet_pure(gen_b_quarks, reco_jets, tagger_name, n, k):
        pt_ordered = reco_jets[ak.argsort(reco_jets.vector.pt, ascending=False)]
        tag_ordered = reco_jets[ak.argsort(reco_jets[tagger_name], ascending=False)]

        pt_ordered_purity_idxs = get_pure_jet_idxs_cross_matched(gen_b_quarks, pt_ordered)
        tag_ordered_purity_idxs = get_pure_jet_idxs_cross_matched(gen_b_quarks, tag_ordered)

        num_highest_pt_pure = ak.any(pt_ordered_purity_idxs == 0, axis=1)
        eff_highest_pt_pure = ak.sum(num_highest_pt_pure) / len(gen_b_quarks)
        print(f"Eff highest pt jet is pure: {eff_highest_pt_pure:.4f}")

        num_highest_tag_pure = ak.any(tag_ordered_purity_idxs == 0, axis=1)
        eff_highest_tag_pure = ak.sum(num_highest_tag_pure) / len(gen_b_quarks)
        print(f"Eff highest {tagger_name} jet is pure: {eff_highest_tag_pure:.4f}")

        n = 4
        range_k = range(k + 1)
        more_than_n_eff_pt = []
        more_than_n_eff_tag = []
        for k in range_k:
            num_k_highest_pt_pure = (ak.num(pt_ordered_purity_idxs) == n) & (ak.all(pt_ordered_purity_idxs < k, axis=1))
            eff_k_highest_pt_pure = ak.sum(num_k_highest_pt_pure) / len(gen_b_quarks)
            more_than_n_eff_pt.append(eff_k_highest_pt_pure)

            num_k_highest_tag_pure = (ak.num(tag_ordered_purity_idxs) == n) & (ak.all(tag_ordered_purity_idxs < k, axis=1))
            eff_k_highest_tag_pure = ak.sum(num_k_highest_tag_pure) / len(gen_b_quarks)
            more_than_n_eff_tag.append(eff_k_highest_tag_pure)

        return eff_highest_pt_pure, eff_highest_tag_pure, more_than_n_eff_pt, more_than_n_eff_tag

    n_eff = 4
    k_eff = 25
    range_k = range(k_eff + 1)

    (eff_highest_pt_pure_part, eff_highest_btag_pure_part,
     more_than_n_eff_pt_part, more_than_n_eff_btag_part,
    ) = get_eff_first_jet_pure(dihiggs_gen_b, scored_jets, "btag_score", n_eff, k_eff)

    print(f"\nPlotting Top-N Jet Efficiencies for N = {n_eff}...")
    fig_eff_pt, ax_pt = plt.subplots(figsize=(10, 6))
    ax_pt.step(range_k, more_than_n_eff_pt_part, where="mid", label="ParT pT", color="purple")
    ax_pt.set_xlabel("k (Number of Top Jets Considered)")
    ax_pt.set_ylabel(f"Probability of Finding at least {n_eff} b-jets")
    ax_pt.set_title(f"Probability of Finding at least {n_eff} b-jets vs. Top k Jets (pT ordered)")
    ax_pt.grid(True, linestyle="--", alpha=0.6)
    ax_pt.legend()
    save_fig(fig_eff_pt, "top_n_eff_pt_ordering")
    plt.close(fig_eff_pt)

    fig_eff_btag, ax_btag = plt.subplots(figsize=(10, 6))
    ax_btag.step(range_k, more_than_n_eff_btag_part, where="mid", label="ParT BTag", color="purple")
    ax_btag.set_xlabel("k (Number of Top Jets Considered)")
    ax_btag.set_ylabel(f"Probability of Finding at least {n_eff} b-jets")
    ax_btag.set_title(f"Probability of Finding at least {n_eff} b-jets vs. Top k Jets (BTag ordered)")
    ax_btag.grid(True, linestyle="--", alpha=0.6)
    ax_btag.legend()
    save_fig(fig_eff_btag, "top_n_eff_btag_ordering")
    plt.close(fig_eff_btag)

    # ── Cell 24: Di-Higgs mass distribution plots ────────────────────
    from matplotlib.patches import Ellipse

    res = part_dihiggs_result
    label = res["label"]
    color_dh = "purple"
    qcd_weights = res.get("qcd_weights", np.ones(res["n_qcd"]))

    sig_window_h = (90, 160)
    sig_window_hh = (250, 550)

    # 1. Signal vs QCD: 1x3 overlay
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    bins_h = np.linspace(0, 300, 61)
    bins_hh = np.linspace(200, 800, 61)

    ax = axes[0]
    if res["n_signal"] > 0:
        ax.hist(ak.to_numpy(res["sig_lead"].mass), bins=bins_h, histtype="stepfilled", alpha=0.3, color=color_dh, label=f'Signal ({res["n_signal"]})', density=True)
        ax.hist(ak.to_numpy(res["sig_lead"].mass), bins=bins_h, histtype="step", linewidth=2, color=color_dh, density=True)
    if res["n_qcd"] > 0:
        ax.hist(ak.to_numpy(res["qcd_lead"].mass), bins=bins_h, histtype="step", linewidth=2, color="grey", linestyle="--", label=f'QCD bkg ({res["n_qcd"]})', density=True)
    ax.axvline(125, color="green", linestyle=":", linewidth=1.5)
    ax.axvspan(*sig_window_h, alpha=0.05, color="green")
    ax.set_xlabel("Leading $m_H$ [GeV]"); ax.set_ylabel("Events / 5 GeV")
    ax.set_title(f"{label} — Leading Higgs"); ax.legend(fontsize=10)

    ax = axes[1]
    if res["n_signal"] > 0:
        ax.hist(ak.to_numpy(res["sig_sub"].mass), bins=bins_h, histtype="stepfilled", alpha=0.3, color=color_dh, label="Signal", density=True)
        ax.hist(ak.to_numpy(res["sig_sub"].mass), bins=bins_h, histtype="step", linewidth=2, color=color_dh, density=True)
    if res["n_qcd"] > 0:
        ax.hist(ak.to_numpy(res["qcd_sub"].mass), bins=bins_h, histtype="step", linewidth=2, color="grey", linestyle="--", label="QCD bkg", density=True)
    ax.axvline(125, color="green", linestyle=":", linewidth=1.5)
    ax.axvspan(*sig_window_h, alpha=0.05, color="green")
    ax.set_xlabel("Subleading $m_H$ [GeV]"); ax.set_ylabel("Events / 5 GeV")
    ax.set_title(f"{label} — Subleading Higgs"); ax.legend(fontsize=10)

    ax = axes[2]
    if res["n_signal"] > 0:
        ax.hist(ak.to_numpy(res["sig_hh"].mass), bins=bins_hh, histtype="stepfilled", alpha=0.3, color=color_dh, label="Signal", density=True)
        ax.hist(ak.to_numpy(res["sig_hh"].mass), bins=bins_hh, histtype="step", linewidth=2, color=color_dh, density=True)
    if res["n_qcd"] > 0:
        ax.hist(ak.to_numpy(res["qcd_hh"].mass), bins=bins_hh, histtype="step", linewidth=2, color="grey", linestyle="--", label="QCD bkg", density=True)
    ax.axvspan(*sig_window_hh, alpha=0.05, color="green")
    ax.set_xlabel("$m_{HH}$ [GeV]"); ax.set_ylabel("Events / 10 GeV")
    ax.set_title(f"{label} — $m_{{HH}}$"); ax.legend(fontsize=10)

    reg_tag = "pT-corrected" if res["has_regression"] and apply_pt_correction else "uncorrected pT"
    fig.suptitle(f"Di-Higgs Reconstruction — {label} ({reg_tag})", fontsize=16, y=1.01)
    plt.tight_layout()
    save_fig(fig, f"dihiggs_mass_1d_{WP_SELECTION}")
    plt.close(fig)

    # 2. 2D mH1 vs mH2

    bins_2d = np.linspace(0, 300, 61)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax_idx, (category, lead_key, sub_key, n_events) in enumerate([
        ("Signal", "sig_lead", "sig_sub", res["n_signal"]),
        ("QCD Background", "qcd_lead", "qcd_sub", res["n_qcd"]),
    ]):
        ax = axes[ax_idx]
        if n_events > 0:
            lead_mass = ak.to_numpy(res[lead_key].mass)
            sub_mass = ak.to_numpy(res[sub_key].mass)
            r_hh_vals = R_hh_func(lead_mass, sub_mass)
            sel = r_hh_vals < 550.0
            h = ax.hist2d(lead_mass[sel], sub_mass[sel], bins=[bins_2d, bins_2d], cmap="viridis")
            ax.axvline(125, color="red", linestyle="--", linewidth=1.5, label="$m_H$ = 125 GeV")
            ax.axhline(120, color="red", linestyle="--", linewidth=1.5)
            ellipse = Ellipse(xy=(125, 120), width=110, height=96, angle=0,
                              edgecolor="yellow", facecolor="none", linestyle="--", linewidth=2, label="$R_{HH}$ = 55 GeV")
            ax.add_patch(ellipse)
            n_sel = int(np.sum(sel))
            ax.set_title(f"{label} — {category}\n({n_sel} events inside $R_{{HH}}$ / {n_events} total)")
            fig.colorbar(h[3], ax=ax, label="Events")
        else:
            ax.text(0.5, 0.5, f"No {category.lower()} events", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{label} — {category}")
        ax.set_xlabel("Leading Higgs Mass [GeV]"); ax.set_ylabel("Subleading Higgs Mass [GeV]")
        ax.legend(loc="upper right", fontsize=10)

    fig.suptitle(f"2D $m_{{H1}}$ vs $m_{{H2}}$ — {label} ({reg_tag})", fontsize=16, y=1.02)
    plt.tight_layout()
    save_fig(fig, f"dihiggs_mass_2d_{WP_SELECTION}")
    plt.close(fig)

    # 3. Significance summary
    print(f"\n{'='*70}")
    print(
        f"{'Tagger':<30} {'S':>6} {'B (wtd)':>10} {'S/√(S+B)':>10}  |  {'S (window)':>10} {'B (window)':>12} {'S/√(S+B)':>10}"
    )
    print(
        f"{'':30} {'(all)':>6} {'(all)':>10} {'(all)':>10}  |  {str(sig_window_hh):>10} {str(sig_window_hh):>12} {'(window)':>10}"
    )
    print("-" * 90)
    if res["n_signal"] > 0 and res["n_qcd"] > 0:
        from evaluation.dihiggs import compute_significance_at_luminosity
        from evaluation.luminosity import signal_weight, scale_qcd_weights_raw

        sigma_to_ngen = res.get("sigma_to_ngen", {})
        sig_mh1 = ak.to_numpy(res["sig_lead"].mass)
        sig_mh2 = ak.to_numpy(res["sig_sub"].mass)
        bkg_mh1 = ak.to_numpy(res["qcd_lead"].mass)
        bkg_mh2 = ak.to_numpy(res["qcd_sub"].mass)

        # mHH window significance via luminosity-scaled wrapper
        result_mhh = compute_significance_at_luminosity(
            sig_mh1,
            sig_mh2,
            bkg_mh1,
            bkg_mh2,
            bkg_raw_weights=qcd_weights,
            sigma_to_ngen=sigma_to_ngen,
            n_gen_signal=N_GEN_SIGNAL,
            luminosity_fb=LUMINOSITY_FB,
            signal_xsec_pb=SIGNAL_XSEC_PB,
            region="circular",
            rect_window=sig_window_hh,
        )
        S_win = result_mhh["S"]
        B_win = result_mhh["B"]
        signif_win = result_mhh["significance"]

        _sw = signal_weight(len(sig_mh1), LUMINOSITY_FB, SIGNAL_XSEC_PB, N_GEN_SIGNAL)
        _bw = scale_qcd_weights_raw(qcd_weights, sigma_to_ngen, LUMINOSITY_FB)
        S_all = float(np.sum(_sw))
        B_all = float(np.sum(_bw))
        signif_all = S_all / np.sqrt(S_all + B_all) if (S_all + B_all) > 0 else 0
        print(
            f"{label:<30} {S_all:>6.0f} {B_all:>10.1e} {signif_all:>10.4f}  |  {S_win:>10.0f} {B_win:>12.1e} {signif_win:>10.4f}"
        )
        print(
            f"\n  Note: S and B are luminosity-scaled expected event counts at {LUMINOSITY_FB:.0f} fb^-1."
        )
        print(f"  Unweighted QCD events: {res['n_qcd']}")
    else:
        print(f"{label:<30}  Insufficient events for significance")
    print("=" * 90)
    print(f"\nWorking point: {WP_SELECTION} (threshold={PART_BTAG_THRESHOLD:.4f})")
    print(f"pT regression: {'applied' if res['has_regression'] and apply_pt_correction else 'not applied'}")
    print(f"Collection: {res['collection_key']}")
    print(f"\nAll di-Higgs plots saved to: {plot_dir}")

    # ── Cell 25: Attention map visualization & pairwise feature analysis ──
    import torch.nn as nn
    from evaluation.attention import (
        compute_pairwise_features, AttentionHook, forward_with_attention,
        forward_with_activations, compute_separability,
    )


    print("=" * 60)
    print("ATTENTION & PAIRWISE FEATURE ANALYSIS")
    print("=" * 60)

    n_samples_attn = 5
    signal_indices_attn = np.where(all_labels == 1)[0][:n_samples_attn]
    background_indices_attn = np.where(all_labels == 0)[0][:n_samples_attn]
    sample_indices_attn = np.concatenate([signal_indices_attn, background_indices_attn])
    sample_x = all_constituents[sample_indices_attn].float().to(device)
    sample_mask = sample_x[:, :, 1] > 0
    pairwise_feats, pw_mask = compute_pairwise_features(sample_x.cpu(), sample_mask.cpu())
    model.eval()
    with torch.no_grad():
        attention_maps, u_ij_attn = forward_with_attention(model, sample_x, sample_mask)
    print(f"Analyzed {len(sample_indices_attn)} jets ({n_samples_attn} signal, {n_samples_attn} background)")
    print(f"Particle attention layers: {len(attention_maps['particle_attn'])}")
    print(f"Class attention layers: {len(attention_maps['class_attn'])}")

    # Pairwise feature distributions
    print("\nComputing pairwise features on full validation set...")
    n_jets_for_dist = min(1000, len(all_constituents))
    dist_x = all_constituents[:n_jets_for_dist]
    dist_mask = dist_x[:, :, 1] > 0
    dist_labels = all_labels[:n_jets_for_dist]
    pairwise_dist, _ = compute_pairwise_features(dist_x, dist_mask)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    feature_names_pairwise = [
        ("log_delta_R", r"$\log(\Delta R)$"), ("log_k_t", r"$\log(k_T)$"),
        ("log_z", r"$\log(z)$"), ("log_m_2", r"$\log(m^2)$"),
        ("d_dxy", r"$\Delta d_{xy}$"), ("d_z0", r"$\Delta z_0$"),
        ("q_ij", r"$q_i \cdot q_j$"),
    ]
    for idx_pw, (feat_key, feat_label) in enumerate(feature_names_pairwise):
        ax = axes[idx_pw]
        feat = pairwise_dist[feat_key].numpy()
        mask_np = dist_mask.numpy()
        sig_vals_pw, bkg_vals_pw = [], []
        for i in range(len(feat)):
            m = mask_np[i]
            n_valid = m.sum()
            if n_valid > 1:
                f_val = feat[i, :n_valid, :n_valid]
                upper_tri = f_val[np.triu_indices(n_valid, k=1)]
                if dist_labels[i] == 1:
                    sig_vals_pw.extend(upper_tri)
                else:
                    bkg_vals_pw.extend(upper_tri)
        sig_vals_pw = np.array(sig_vals_pw)
        bkg_vals_pw = np.array(bkg_vals_pw)
        sig_vals_pw = sig_vals_pw[np.isfinite(sig_vals_pw)]
        bkg_vals_pw = bkg_vals_pw[np.isfinite(bkg_vals_pw)]
        if len(sig_vals_pw) > 0 and len(bkg_vals_pw) > 0:
            range_min = min(np.percentile(sig_vals_pw, 1), np.percentile(bkg_vals_pw, 1))
            range_max = max(np.percentile(sig_vals_pw, 99), np.percentile(bkg_vals_pw, 99))
            ax.hist(sig_vals_pw, bins=50, range=(range_min, range_max), histtype="step", label="Signal", color="blue", density=True)
            ax.hist(bkg_vals_pw, bins=50, range=(range_min, range_max), histtype="step", label="Background", color="red", density=True)
        ax.set_xlabel(feat_label)
        ax.set_ylabel("Density")
        ax.set_title(f"Pairwise Feature: {feat_label}")
        ax.legend()
    axes[-1].axis("off")
    plt.tight_layout()
    save_fig(fig, "pairwise_feature_distributions")
    plt.close(fig)

    # Attention map visualization
    def plot_attention_maps_fn(attn_maps, sample_idx, sample_mask_cpu, is_signal, layer_idx=0):
        n_heads = attn_maps["particle_attn"][layer_idx].shape[1]
        n_valid = sample_mask_cpu[sample_idx].sum().item()
        fig_am, axes_am = plt.subplots(2, 4, figsize=(16, 8))
        axes_am = axes_am.flatten()
        attn = attn_maps["particle_attn"][layer_idx][sample_idx, :, :n_valid, :n_valid]
        for h in range(min(n_heads, 8)):
            ax = axes_am[h]
            attn_head = attn[h].numpy()
            im = ax.imshow(attn_head, cmap="viridis", aspect="auto", vmin=0, vmax=attn_head.max())
            ax.set_xlabel("Key Particle")
            ax.set_ylabel("Query Particle")
            ax.set_title(f"Head {h+1}")
            plt.colorbar(im, ax=ax, fraction=0.046)
        label_am = "Signal (b-jet)" if is_signal else "Background"
        fig_am.suptitle(f"Particle Attention Maps - Layer {layer_idx+1} - {label_am}", fontsize=14)
        plt.tight_layout()
        return fig_am

    print("\nPlotting attention maps...")
    fig_sig = plot_attention_maps_fn(attention_maps, 0, sample_mask.cpu(), is_signal=True, layer_idx=0)
    save_fig(fig_sig, "attention_map_signal_layer1")
    plt.close(fig_sig)
    fig_bkg = plot_attention_maps_fn(attention_maps, n_samples_attn, sample_mask.cpu(), is_signal=False, layer_idx=0)
    save_fig(fig_bkg, "attention_map_background_layer1")
    plt.close(fig_bkg)

    # Class attention visualization
    fig, axes = plt.subplots(2, n_samples_attn, figsize=(4 * n_samples_attn, 8))
    for i in range(n_samples_attn):
        cls_attn = attention_maps["class_attn"][-1][i, :, 0, 1:].mean(dim=0).numpy()
        n_valid = sample_mask[i].sum().item()
        ax = axes[0, i]
        ax.bar(range(n_valid), cls_attn[:n_valid])
        ax.set_xlabel("Constituent Index")
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"Signal Jet {i+1}")
        cls_attn = attention_maps["class_attn"][-1][n_samples_attn + i, :, 0, 1:].mean(dim=0).numpy()
        n_valid = sample_mask[n_samples_attn + i].sum().item()
        ax = axes[1, i]
        ax.bar(range(n_valid), cls_attn[:n_valid], color="red")
        ax.set_xlabel("Constituent Index")
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"Background Jet {i+1}")
    fig.suptitle("Class Token Attention Weights (Which particles the classifier focuses on)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "class_attention_weights")
    plt.close(fig)

    # Class attention colored by particle type (notebook parity)
    if all_constituents.shape[2] > 12:
        from matplotlib.patches import Patch

        fig, axes = plt.subplots(2, n_samples_attn, figsize=(4 * n_samples_attn, 8))
        sample_particle_types = np.argmax(
            all_constituents[sample_indices_attn, :, 12:17].numpy(), axis=-1
        )

        for i in range(n_samples_attn):
            cls_attn = attention_maps["class_attn"][-1][i, :, 0, 1:].mean(dim=0).numpy()
            n_valid = sample_mask[i].sum().item()
            ptypes = sample_particle_types[i, :n_valid]
            colors = [
                PARTICLE_TYPE_COLORS[int(p)] if 0 <= int(p) < N_PARTICLE_TYPES else "#999999"
                for p in ptypes
            ]

            ax = axes[0, i]
            ax.bar(range(n_valid), cls_attn[:n_valid], color=colors)
            ax.set_xlabel("Constituent Index")
            ax.set_ylabel("Attention Weight")
            ax.set_title(f"Signal Jet {i+1}")

            cls_attn = attention_maps["class_attn"][-1][n_samples_attn + i, :, 0, 1:].mean(dim=0).numpy()
            n_valid = sample_mask[n_samples_attn + i].sum().item()
            ptypes = sample_particle_types[n_samples_attn + i, :n_valid]
            colors = [
                PARTICLE_TYPE_COLORS[int(p)] if 0 <= int(p) < N_PARTICLE_TYPES else "#999999"
                for p in ptypes
            ]

            ax = axes[1, i]
            ax.bar(range(n_valid), cls_attn[:n_valid], color=colors)
            ax.set_xlabel("Constituent Index")
            ax.set_ylabel("Attention Weight")
            ax.set_title(f"Background Jet {i+1}")

        legend_elements = [
            Patch(facecolor=PARTICLE_TYPE_COLORS[p], label=PARTICLE_TYPE_NAMES[p])
            for p in range(N_PARTICLE_TYPES)
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=N_PARTICLE_TYPES,
            fontsize=10,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.suptitle("Class Token Attention - Colored by Particle Type", fontsize=14, y=1.05)
        plt.tight_layout()
        save_fig(fig, "class_attention_by_particle_type")
        plt.close(fig)

    # Average attention patterns
    n_avg = min(100, len(all_constituents))
    avg_x = all_constituents[:n_avg].float().to(device)
    avg_mask = avg_x[:, :, 1] > 0
    avg_labels_attn = all_labels[:n_avg]
    with torch.no_grad():
        avg_attention_maps, _ = forward_with_attention(model, avg_x, avg_mask)
    print("\nComputing attention vs Delta R correlation...")
    pairwise_avg, _ = compute_pairwise_features(avg_x.cpu(), avg_mask.cpu())
    delta_r_vals_attn, attn_vals_attn, labels_vals_attn = [], [], []
    for i in range(n_avg):
        n_valid = avg_mask[i].sum().item()
        if n_valid > 1:
            dr = pairwise_avg["delta_R"][i, :n_valid, :n_valid].numpy()
            attn = avg_attention_maps["particle_attn"][0][i, :, :n_valid, :n_valid].mean(dim=0).numpy()
            for j in range(n_valid):
                for kk in range(j + 1, n_valid):
                    delta_r_vals_attn.append(dr[j, kk])
                    attn_vals_attn.append(attn[j, kk])
                    labels_vals_attn.append(avg_labels_attn[i])
    delta_r_vals_attn = np.array(delta_r_vals_attn)
    attn_vals_attn = np.array(attn_vals_attn)
    labels_vals_attn = np.array(labels_vals_attn)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    hb = ax.hexbin(delta_r_vals_attn, attn_vals_attn, gridsize=50, cmap="viridis",
                   extent=[0, 2, 0, np.percentile(attn_vals_attn, 99)], mincnt=1)
    ax.set_xlabel(r"$\Delta R$")
    ax.set_ylabel("Attention Weight")
    ax.set_title("Attention Weight vs Angular Distance")
    plt.colorbar(hb, ax=ax, label="Count")
    ax = axes[1]
    sig_mask_v = labels_vals_attn == 1
    dr_bins = np.linspace(0, 2, 20)
    sig_attn_means, bkg_attn_means = [], []
    for j in range(len(dr_bins) - 1):
        bin_mask = (delta_r_vals_attn >= dr_bins[j]) & (delta_r_vals_attn < dr_bins[j + 1])
        sig_attn_means.append(np.mean(attn_vals_attn[bin_mask & sig_mask_v]) if (bin_mask & sig_mask_v).sum() > 0 else np.nan)
        bkg_attn_means.append(np.mean(attn_vals_attn[bin_mask & ~sig_mask_v]) if (bin_mask & ~sig_mask_v).sum() > 0 else np.nan)
    dr_centers = (dr_bins[:-1] + dr_bins[1:]) / 2
    ax.plot(dr_centers, sig_attn_means, "b-o", label="Signal", markersize=5)
    ax.plot(dr_centers, bkg_attn_means, "r-o", label="Background", markersize=5)
    ax.set_xlabel(r"$\Delta R$")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Attention vs Distance by Jet Type")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "attention_vs_delta_r")
    plt.close(fig)

    if all_constituents.shape[2] > 12:
        # Mean class attention by particle type (aggregated)
        avg_cls_attn = avg_attention_maps["class_attn"][-1][:, :, 0, 1:].mean(dim=1).numpy()
        avg_particle_types = np.argmax(avg_x[:, :, 12:17].cpu().numpy(), axis=-1)
        avg_mask_np_typed = avg_mask.cpu().numpy()
        avg_particle_types[~avg_mask_np_typed] = -1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for label_val, label_name, ax in [(1, "Signal", ax1), (0, "Background", ax2)]:
            jet_mask = avg_labels_attn[:n_avg] == label_val
            mean_attn_per_type = []
            std_attn_per_type = []

            for pid in range(N_PARTICLE_TYPES):
                attn_vals = []
                for j in np.where(jet_mask)[0]:
                    n_valid = int(avg_mask_np_typed[j].sum())
                    cls_a = avg_cls_attn[j, :n_valid]
                    ptypes_j = avg_particle_types[j, :n_valid]
                    type_attn = cls_a[ptypes_j == pid]
                    attn_vals.extend(type_attn)

                attn_vals = np.array(attn_vals) if attn_vals else np.array([0.0])
                mean_attn_per_type.append(attn_vals.mean())
                std_attn_per_type.append(attn_vals.std() / np.sqrt(len(attn_vals)))

            ax.bar(
                range(N_PARTICLE_TYPES),
                mean_attn_per_type,
                yerr=std_attn_per_type,
                color=PARTICLE_TYPE_COLORS,
                alpha=0.8,
                capsize=4,
            )
            ax.set_xticks(range(N_PARTICLE_TYPES))
            ax.set_xticklabels(PARTICLE_TYPE_NAMES, rotation=30, ha="right")
            ax.set_ylabel("Mean Class Attention Weight")
            ax.set_title(f"{label_name} Jets")

        plt.suptitle("Mean Class Token Attention by Particle Type", fontsize=14)
        plt.tight_layout()
        save_fig(fig, "mean_class_attention_by_particle_type")
        plt.close(fig)

        # Particle-type attention affinity matrix
        layer0_attn = avg_attention_maps["particle_attn"][0]
        head_avg_attn = layer0_attn.mean(dim=1).numpy()
        affinity_sig = np.zeros((N_PARTICLE_TYPES, N_PARTICLE_TYPES))
        affinity_bkg = np.zeros((N_PARTICLE_TYPES, N_PARTICLE_TYPES))
        count_sig = np.zeros((N_PARTICLE_TYPES, N_PARTICLE_TYPES))
        count_bkg = np.zeros((N_PARTICLE_TYPES, N_PARTICLE_TYPES))

        for j in range(n_avg):
            n_valid = int(avg_mask_np_typed[j].sum())
            if n_valid < 2:
                continue
            attn_j = head_avg_attn[j, :n_valid, :n_valid]
            ptypes_j = avg_particle_types[j, :n_valid]

            for qi in range(n_valid):
                for ki in range(n_valid):
                    if qi == ki:
                        continue
                    q_type = int(ptypes_j[qi])
                    k_type = int(ptypes_j[ki])
                    if q_type < 0 or k_type < 0:
                        continue
                    if avg_labels_attn[j] == 1:
                        affinity_sig[q_type, k_type] += attn_j[qi, ki]
                        count_sig[q_type, k_type] += 1
                    else:
                        affinity_bkg[q_type, k_type] += attn_j[qi, ki]
                        count_bkg[q_type, k_type] += 1

        affinity_sig_norm = np.divide(
            affinity_sig,
            count_sig,
            out=np.zeros_like(affinity_sig),
            where=count_sig > 0,
        )
        affinity_bkg_norm = np.divide(
            affinity_bkg,
            count_bkg,
            out=np.zeros_like(affinity_bkg),
            where=count_bkg > 0,
        )
        affinity_diff = affinity_sig_norm - affinity_bkg_norm

        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        short_names = ["ChHad", "Elec", "NeuHad", "Photon", "Muon"][:N_PARTICLE_TYPES]

        for ax, data, title, cmap in [
            (axes[0], affinity_sig_norm, "Signal (b-jets)", "viridis"),
            (axes[1], affinity_bkg_norm, "Background", "viridis"),
            (axes[2], affinity_diff, "Signal - Background", "RdBu_r"),
        ]:
            vmax = np.abs(data).max() if "-" in title else data.max()
            vmin = -vmax if "-" in title else 0
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(N_PARTICLE_TYPES))
            ax.set_yticks(range(N_PARTICLE_TYPES))
            ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(short_names, fontsize=9)
            ax.set_xlabel("Key (attended to)")
            ax.set_ylabel("Query (attending)")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)

            threshold = vmax * 0.5 if vmax > 0 else 0.0
            for ii in range(N_PARTICLE_TYPES):
                for jj in range(N_PARTICLE_TYPES):
                    ax.text(
                        jj,
                        ii,
                        f"{data[ii, jj]:.4f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(data[ii, jj]) > threshold else "black",
                    )

        plt.suptitle("Particle-Type Attention Affinity Matrix (Layer 0, Head-Averaged)", fontsize=14)
        plt.tight_layout()
        save_fig(fig, "particle_type_attention_affinity")
        plt.close(fig)

        # Attention vs Delta R stratified by particle type
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        type_dr_attn = {pid: ([], []) for pid in range(N_PARTICLE_TYPES)}

        for i in range(n_avg):
            n_valid = int(avg_mask_np_typed[i].sum())
            if n_valid < 2:
                continue
            dr = pairwise_avg["delta_R"][i, :n_valid, :n_valid].numpy()
            attn = avg_attention_maps["particle_attn"][0][i, :, :n_valid, :n_valid].mean(dim=0).numpy()
            ptypes_i = avg_particle_types[i, :n_valid]

            for q in range(n_valid):
                q_type = int(ptypes_i[q])
                if q_type < 0:
                    continue
                for k in range(n_valid):
                    if q == k:
                        continue
                    type_dr_attn[q_type][0].append(dr[q, k])
                    type_dr_attn[q_type][1].append(attn[q, k])

        dr_bins = np.linspace(0, 2, 20)
        dr_centers = (dr_bins[:-1] + dr_bins[1:]) / 2

        for pid in range(N_PARTICLE_TYPES):
            dr_arr = np.array(type_dr_attn[pid][0])
            attn_arr = np.array(type_dr_attn[pid][1])
            if len(dr_arr) == 0:
                continue

            means = []
            for b in range(len(dr_bins) - 1):
                bin_mask = (dr_arr >= dr_bins[b]) & (dr_arr < dr_bins[b + 1])
                means.append(attn_arr[bin_mask].mean() if bin_mask.sum() > 0 else np.nan)

            axes[0].plot(
                dr_centers,
                means,
                "-o",
                color=PARTICLE_TYPE_COLORS[pid],
                label=PARTICLE_TYPE_NAMES[pid],
                markersize=4,
                linewidth=1.5,
            )

        axes[0].set_xlabel(r"$\Delta R$")
        axes[0].set_ylabel("Mean Attention Weight")
        axes[0].set_title("Attention vs Angular Distance by Query Particle Type")
        axes[0].legend(fontsize=9)

        type_dr_attn_sig = {pid: ([], []) for pid in range(N_PARTICLE_TYPES)}
        for i in range(n_avg):
            if avg_labels_attn[i] != 1:
                continue
            n_valid = int(avg_mask_np_typed[i].sum())
            if n_valid < 2:
                continue
            dr = pairwise_avg["delta_R"][i, :n_valid, :n_valid].numpy()
            attn = avg_attention_maps["particle_attn"][0][i, :, :n_valid, :n_valid].mean(dim=0).numpy()
            ptypes_i = avg_particle_types[i, :n_valid]

            for q in range(n_valid):
                q_type = int(ptypes_i[q])
                if q_type < 0:
                    continue
                for k in range(n_valid):
                    if q == k:
                        continue
                    type_dr_attn_sig[q_type][0].append(dr[q, k])
                    type_dr_attn_sig[q_type][1].append(attn[q, k])

        for pid in range(N_PARTICLE_TYPES):
            dr_arr = np.array(type_dr_attn_sig[pid][0])
            attn_arr = np.array(type_dr_attn_sig[pid][1])
            if len(dr_arr) == 0:
                continue

            means = []
            for b in range(len(dr_bins) - 1):
                bin_mask = (dr_arr >= dr_bins[b]) & (dr_arr < dr_bins[b + 1])
                means.append(attn_arr[bin_mask].mean() if bin_mask.sum() > 0 else np.nan)

            axes[1].plot(
                dr_centers,
                means,
                "-o",
                color=PARTICLE_TYPE_COLORS[pid],
                label=PARTICLE_TYPE_NAMES[pid],
                markersize=4,
                linewidth=1.5,
            )

        axes[1].set_xlabel(r"$\Delta R$")
        axes[1].set_ylabel("Mean Attention Weight")
        axes[1].set_title("Attention vs $\\Delta R$ - Signal Jets Only, by Query Type")
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        save_fig(fig, "attention_vs_delta_r_by_particle_type")
        plt.close(fig)

    print(f"\n{'='*60}")
    print("Attention and pairwise feature analysis complete!")
    print(f"{'='*60}")

    # ── Cell 26: Layer activation visualization ───────────────────────
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import torch.nn.functional as F

    print("=" * 60)
    print("LAYER ACTIVATION ANALYSIS")
    print("=" * 60)

    n_vis = min(500, len(all_constituents))
    vis_x = all_constituents[:n_vis].float().to(device)
    vis_mask = vis_x[:, :, 1] > 0
    vis_labels = all_labels[:n_vis]
    print(f"Analyzing activations for {n_vis} jets...")
    print(f"  Signal jets: {(vis_labels == 1).sum()}")
    print(f"  Background jets: {(vis_labels == 0).sum()}")

    with torch.no_grad():
        activations = forward_with_activations(model, vis_x, vis_mask)

    # Activation magnitude distributions
    print("\nPlotting activation magnitude distributions...")
    layer_names_act = (
        ["embedding"]
        + [f"particle_attn_{i+1}" for i in range(len(activations["particle_attn_layers"]))]
        + ["pre_cls"]
        + [f"cls_attn_{i+1}" for i in range(len(activations["cls_attn_layers"]))]
        + ["final_cls_token"]
    )
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    layer_activations_list = [activations["embedding"]]
    layer_activations_list.extend(activations["particle_attn_layers"])
    layer_activations_list.append(activations["pre_cls"])
    layer_activations_list.extend(activations["cls_attn_layers"])
    layer_activations_list.append(activations["final_cls_token"].unsqueeze(1))

    for idx_la, (name_la, act) in enumerate(zip(layer_names_act, layer_activations_list)):
        if idx_la >= len(axes):
            break
        ax = axes[idx_la]
        if len(act.shape) == 3:
            act_flat = act.mean(dim=1).numpy()
        else:
            act_flat = act.numpy()
        sig_act = act_flat[vis_labels == 1].flatten()
        bkg_act = act_flat[vis_labels == 0].flatten()
        range_min = min(np.percentile(sig_act, 1), np.percentile(bkg_act, 1))
        range_max = max(np.percentile(sig_act, 99), np.percentile(bkg_act, 99))
        ax.hist(sig_act, bins=50, range=(range_min, range_max), histtype="step", label="Signal", color="blue", density=True, alpha=0.8)
        ax.hist(bkg_act, bins=50, range=(range_min, range_max), histtype="step", label="Background", color="red", density=True, alpha=0.8)
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Density")
        ax.set_title(f"{name_la}\n(dim={act_flat.shape[-1]})")
        ax.legend(fontsize=8)
    for idx_la in range(len(layer_names_act), len(axes)):
        axes[idx_la].axis("off")
    plt.suptitle("Activation Magnitude Distributions Across Layers", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "activation_magnitude_distributions")
    plt.close(fig)

    # t-SNE visualization
    print("\nComputing t-SNE projections...")
    tsne_layers = {
        "Input (processed)": activations["input_processed"].mean(dim=1).numpy(),
        "After Embedding": activations["embedding"].mean(dim=1).numpy(),
        f"After ParticleAttn-{len(activations['particle_attn_layers'])}": activations["particle_attn_layers"][-1].mean(dim=1).numpy(),
        "Final CLS Token": activations["final_cls_token"].numpy(),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx_ts, (layer_name_ts, layer_act) in enumerate(tsne_layers.items()):
        ax = axes[idx_ts]
        if layer_act.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            layer_act_reduced = pca.fit_transform(layer_act)
        else:
            layer_act_reduced = layer_act
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_vis // 4))
        tsne_result = tsne.fit_transform(layer_act_reduced)
        ax.scatter(tsne_result[vis_labels == 0, 0], tsne_result[vis_labels == 0, 1], c="red", alpha=0.5, s=10, label="Background")
        ax.scatter(tsne_result[vis_labels == 1, 0], tsne_result[vis_labels == 1, 1], c="blue", alpha=0.5, s=10, label="Signal")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(layer_name_ts)
        ax.legend()
    plt.suptitle("t-SNE Visualization: How Representations Evolve Through Layers", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "tsne_layer_evolution")
    plt.close(fig)

    # Neuron activation patterns
    print("\nAnalyzing neuron activation patterns...")
    final_act = activations["final_cls_token"].numpy()
    embed_dim = final_act.shape[1]
    sig_neuron_mean = final_act[vis_labels == 1].mean(axis=0)
    bkg_neuron_mean = final_act[vis_labels == 0].mean(axis=0)
    neuron_diff = sig_neuron_mean - bkg_neuron_mean
    sorted_idx_nd = np.argsort(neuron_diff)[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    top_n_nd = 20
    ax = axes[0, 0]
    ax.barh(range(top_n_nd), neuron_diff[sorted_idx_nd[:top_n_nd]], color="blue", alpha=0.7)
    ax.set_yticks(range(top_n_nd))
    ax.set_yticklabels([f"Neuron {i}" for i in sorted_idx_nd[:top_n_nd]])
    ax.set_xlabel("Mean Activation Difference (Signal - Background)")
    ax.set_title(f"Top {top_n_nd} Neurons Favoring Signal")
    ax.invert_yaxis()

    ax = axes[0, 1]
    ax.barh(range(top_n_nd), neuron_diff[sorted_idx_nd[-top_n_nd:]], color="red", alpha=0.7)
    ax.set_yticks(range(top_n_nd))
    ax.set_yticklabels([f"Neuron {i}" for i in sorted_idx_nd[-top_n_nd:]])
    ax.set_xlabel("Mean Activation Difference (Signal - Background)")
    ax.set_title(f"Top {top_n_nd} Neurons Favoring Background")
    ax.invert_yaxis()

    ax = axes[1, 0]
    n_show = min(50, embed_dim)
    top_neurons = sorted_idx_nd[:n_show // 2].tolist() + sorted_idx_nd[-n_show // 2:].tolist()
    n_jets_hm = 50
    sig_jets_hm = np.where(vis_labels == 1)[0][:n_jets_hm // 2]
    bkg_jets_hm = np.where(vis_labels == 0)[0][:n_jets_hm // 2]
    jet_order = np.concatenate([sig_jets_hm, bkg_jets_hm])
    heatmap_data = final_act[jet_order][:, top_neurons]
    im = ax.imshow(heatmap_data.T, aspect="auto", cmap="RdBu_r",
                   vmin=-np.percentile(np.abs(heatmap_data), 95),
                   vmax=np.percentile(np.abs(heatmap_data), 95))
    ax.axvline(len(sig_jets_hm) - 0.5, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Jet Index (Signal | Background)")
    ax.set_ylabel("Neuron Index")
    ax.set_title("Activation Heatmap (Most Discriminative Neurons)")
    plt.colorbar(im, ax=ax, label="Activation")

    ax = axes[1, 1]
    top_act_corr = final_act[:, sorted_idx_nd[:30]]
    corr = np.corrcoef(top_act_corr.T)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Neuron Index")
    ax.set_title("Correlation Between Top 30 Discriminative Neurons")
    plt.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    save_fig(fig, "neuron_activation_patterns")
    plt.close(fig)

    # Layer-wise separability
    print("\nComputing layer-wise class separability...")

    separability_stats = []
    layer_order_sep = (
        ["input_processed", "embedding"]
        + [f"particle_attn_{i}" for i in range(len(activations["particle_attn_layers"]))]
        + ["pre_cls"]
        + [f"cls_attn_{i}" for i in range(len(activations["cls_attn_layers"]))]
        + ["final_cls_token"]
    )
    layer_acts_sep = [
        activations["input_processed"].mean(dim=1).numpy(),
        activations["embedding"].mean(dim=1).numpy(),
    ]
    for act in activations["particle_attn_layers"]:
        layer_acts_sep.append(act.mean(dim=1).numpy())
    layer_acts_sep.append(activations["pre_cls"].mean(dim=1).numpy())
    for act in activations["cls_attn_layers"]:
        layer_acts_sep.append(act.mean(dim=1).numpy())
    layer_acts_sep.append(activations["final_cls_token"].numpy())

    for name_sep, act_sep in zip(layer_order_sep, layer_acts_sep):
        mean_fisher, max_fisher, _ = compute_separability(act_sep, vis_labels)
        separability_stats.append({"layer": name_sep, "mean_fisher": mean_fisher, "max_fisher": max_fisher, "dim": act_sep.shape[1]})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x_pos_sep = range(len(separability_stats))
    ax.bar(x_pos_sep, [s["mean_fisher"] for s in separability_stats], color="steelblue", alpha=0.7)
    ax.set_xticks(x_pos_sep)
    ax.set_xticklabels([s["layer"].replace("_", "\n") for s in separability_stats], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Fisher Discriminant Ratio")
    ax.set_title("Average Class Separability Across Layers")
    ax = axes[1]
    ax.bar(x_pos_sep, [s["max_fisher"] for s in separability_stats], color="darkorange", alpha=0.7)
    ax.set_xticks(x_pos_sep)
    ax.set_xticklabels([s["layer"].replace("_", "\n") for s in separability_stats], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Max Fisher Discriminant Ratio")
    ax.set_title("Best Single-Neuron Separability Across Layers")
    plt.tight_layout()
    save_fig(fig, "layer_separability_evolution")
    plt.close(fig)

    # Per-constituent activation analysis
    print("\nAnalyzing per-constituent activations...")
    final_part_attn = activations["particle_attn_layers"][-1]
    const_act_magnitude = torch.norm(final_part_attn, dim=-1).numpy()
    const_pt_vis = vis_x[:, :, 1].cpu().numpy()
    mask_vis_np = vis_mask.cpu().numpy()
    pt_flat_ca, act_flat_ca, label_flat_ca = [], [], []
    for i in range(n_vis):
        n_valid = int(mask_vis_np[i].sum())
        pt_flat_ca.extend(const_pt_vis[i, :n_valid])
        act_flat_ca.extend(const_act_magnitude[i, :n_valid])
        label_flat_ca.extend([vis_labels[i]] * n_valid)
    pt_flat_ca = np.array(pt_flat_ca)
    act_flat_ca = np.array(act_flat_ca)
    label_flat_ca = np.array(label_flat_ca)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    sig_mask_const = label_flat_ca == 1
    ax.scatter(pt_flat_ca[~sig_mask_const], act_flat_ca[~sig_mask_const], alpha=0.1, s=2, c="red", label="Background")
    ax.scatter(pt_flat_ca[sig_mask_const], act_flat_ca[sig_mask_const], alpha=0.1, s=2, c="blue", label="Signal")
    ax.set_xlabel("Constituent $p_T$ [GeV]")
    ax.set_ylabel("Activation Magnitude (L2 norm)")
    ax.set_title("Constituent Activation vs $p_T$")
    ax.legend()
    ax.set_xlim(0, np.percentile(pt_flat_ca, 99))
    ax = axes[1]
    ax.hist(act_flat_ca[sig_mask_const], bins=50, histtype="step", label="Signal", color="blue", density=True)
    ax.hist(act_flat_ca[~sig_mask_const], bins=50, histtype="step", label="Background", color="red", density=True)
    ax.set_xlabel("Activation Magnitude")
    ax.set_ylabel("Density")
    ax.set_title("Per-Constituent Activation Distribution")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "constituent_activation_analysis")
    plt.close(fig)

    if all_constituents.shape[2] > 12:
        # Per-constituent activation by particle type
        vis_particle_types = np.argmax(vis_x[:, :, 12:17].cpu().numpy(), axis=-1)
        vis_particle_types[~mask_vis_np] = -1

        ptype_flat = []
        for i in range(n_vis):
            n_valid = int(mask_vis_np[i].sum())
            ptype_flat.extend(vis_particle_types[i, :n_valid])
        ptype_flat = np.array(ptype_flat)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        for pid in range(N_PARTICLE_TYPES):
            pmask = ptype_flat == pid
            if pmask.sum() == 0:
                continue
            ax.scatter(
                pt_flat_ca[pmask],
                act_flat_ca[pmask],
                alpha=0.05,
                s=2,
                c=PARTICLE_TYPE_COLORS[pid],
                label=PARTICLE_TYPE_NAMES[pid],
            )
        ax.set_xlabel("Constituent $p_T$ [GeV]")
        ax.set_ylabel("Activation Magnitude (L2 norm)")
        ax.set_title("Activation vs $p_T$ by Particle Type")
        ax.legend(markerscale=10, fontsize=9)
        ax.set_xlim(0, np.percentile(pt_flat_ca, 99))

        ax = axes[1]
        for pid in range(N_PARTICLE_TYPES):
            pmask = ptype_flat == pid
            if pmask.sum() == 0:
                continue
            ax.hist(
                act_flat_ca[pmask],
                bins=50,
                histtype="step",
                label=PARTICLE_TYPE_NAMES[pid],
                color=PARTICLE_TYPE_COLORS[pid],
                density=True,
                linewidth=1.5,
            )
        ax.set_xlabel("Activation Magnitude")
        ax.set_ylabel("Density")
        ax.set_title("Per-Constituent Activation by Particle Type")
        ax.legend(fontsize=9)

        plt.tight_layout()
        save_fig(fig, "constituent_activation_by_particle_type")
        plt.close(fig)

        # Activation box plots: particle type x jet class
        fig, ax = plt.subplots(figsize=(14, 6))
        box_data = []
        box_labels_list = []
        box_positions = []
        box_colors = []
        pos = 0

        for pid in range(N_PARTICLE_TYPES):
            for label_val, label_name in [(1, "Sig"), (0, "Bkg")]:
                pmask = (ptype_flat == pid) & (label_flat_ca == label_val)
                if pmask.sum() > 0:
                    vals = act_flat_ca[pmask]
                    if len(vals) > 5000:
                        vals = np.random.choice(vals, 5000, replace=False)
                    box_data.append(vals)
                    box_labels_list.append(f"{PARTICLE_TYPE_NAMES[pid]}\n({label_name})")
                    box_positions.append(pos)
                    box_colors.append("blue" if label_val == 1 else "red")
                pos += 1
            pos += 0.5

        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_xticks(box_positions)
        ax.set_xticklabels(box_labels_list, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Activation Magnitude (L2 norm)")
        ax.set_title("Activation Magnitude by Particle Type and Jet Class")

        plt.tight_layout()
        save_fig(fig, "activation_boxplot_by_type_and_class")
        plt.close(fig)

        # Per-constituent t-SNE by particle type
        final_part_attn_np = final_part_attn.numpy()
        embed_list = []
        ptype_list = []
        jet_label_list = []

        for i in range(n_vis):
            n_valid = int(mask_vis_np[i].sum())
            embed_list.append(final_part_attn_np[i, :n_valid, :])
            ptype_list.extend(vis_particle_types[i, :n_valid])
            jet_label_list.extend([int(vis_labels[i])] * n_valid)

        embed_all = np.concatenate(embed_list, axis=0)
        ptype_all = np.array(ptype_list)
        jet_label_all = np.array(jet_label_list)

        n_tsne = min(10000, len(embed_all))
        tsne_idx = np.random.choice(len(embed_all), n_tsne, replace=False)
        embed_sub = embed_all[tsne_idx]
        ptype_sub = ptype_all[tsne_idx]
        label_sub = jet_label_all[tsne_idx]

        if embed_sub.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            embed_sub = pca.fit_transform(embed_sub)

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_tsne // 4))
        tsne_result = tsne.fit_transform(embed_sub)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        for pid in range(N_PARTICLE_TYPES):
            pmask = ptype_sub == pid
            if pmask.sum() == 0:
                continue
            ax1.scatter(
                tsne_result[pmask, 0],
                tsne_result[pmask, 1],
                c=PARTICLE_TYPE_COLORS[pid],
                alpha=0.3,
                s=5,
                label=PARTICLE_TYPE_NAMES[pid],
            )
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        ax1.set_title("Per-Constituent Embedding - by Particle Type")
        ax1.legend(markerscale=5, fontsize=9)

        ax2.scatter(
            tsne_result[label_sub == 0, 0],
            tsne_result[label_sub == 0, 1],
            c="red",
            alpha=0.3,
            s=5,
            label="Background",
        )
        ax2.scatter(
            tsne_result[label_sub == 1, 0],
            tsne_result[label_sub == 1, 1],
            c="blue",
            alpha=0.3,
            s=5,
            label="Signal",
        )
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.set_title("Per-Constituent Embedding - by Jet Class")
        ax2.legend(markerscale=5, fontsize=9)

        plt.suptitle("t-SNE of Per-Constituent Representations (Final Particle Attention Layer)", fontsize=13)
        plt.tight_layout()
        save_fig(fig, "constituent_tsne_by_particle_type")
        plt.close(fig)

        del embed_all, embed_list, embed_sub, tsne_result
        gc.collect()
        flush_memory()

    print(f"\n{'='*60}")
    print("Layer activation analysis complete!")
    print(f"{'='*60}")

    # ── Cell 27: Feature importance analysis ──────────────────────────
    from sklearn.metrics import roc_auc_score as roc_auc_score_fi
    from scipy.stats import ks_2samp

    print("=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    input_feature_names = [
        "Mass", "pT", "η", "φ", "d_xy", "z_0", "Charge",
        "log(pT_rel)", "Δη", "Δφ", "PUPPI weight", "log(ΔR)",
    ]
    n_features_fi = all_constituents.shape[2]
    if n_features_fi > 12:
        for i_fi in range(12, n_features_fi):
            input_feature_names.append(f"ParticleID_{i_fi-11}")
    print(f"Analyzing {len(input_feature_names)} input features")

    # 1. Gradient-based feature importance
    print("\n1. Computing gradient-based feature importance...")
    n_grad_samples = min(500, len(all_constituents))
    grad_x = all_constituents[:n_grad_samples].clone().float().to(device)
    grad_x.requires_grad_(True)
    grad_mask = grad_x[:, :, 1] > 0
    grad_labels = all_labels[:n_grad_samples]
    model.eval()
    tmep_outputs = model(grad_x, particle_mask=grad_mask)
    outputs_fi = tmep_outputs["classification"]
    grad_x.grad = None
    loss_signal = outputs_fi.sum()
    loss_signal.backward(retain_graph=True)
    grads_signal = grad_x.grad.clone().detach().cpu()

    grad_x.grad = None
    grad_x.requires_grad_(True)
    tmep_outputs = model(grad_x, particle_mask=grad_mask)
    outputs_fi = tmep_outputs["classification"]
    probs_fi = torch.sigmoid(outputs_fi)
    probs_fi.sum().backward()
    grads_prob = grad_x.grad.clone().detach().cpu()

    mask_np_fi = grad_mask.cpu().numpy()
    grad_importance = np.zeros(len(input_feature_names))
    grad_importance_signal = np.zeros(len(input_feature_names))
    grad_importance_background = np.zeros(len(input_feature_names))

    for feat_idx in range(min(len(input_feature_names), grads_prob.shape[2])):
        valid_grads, sig_grads, bkg_grads = [], [], []
        for i in range(n_grad_samples):
            n_valid = int(mask_np_fi[i].sum())
            feat_grads = np.abs(grads_prob[i, :n_valid, feat_idx].numpy())
            valid_grads.extend(feat_grads)
            if grad_labels[i] == 1:
                sig_grads.extend(feat_grads)
            else:
                bkg_grads.extend(feat_grads)
        grad_importance[feat_idx] = np.mean(valid_grads)
        grad_importance_signal[feat_idx] = np.mean(sig_grads) if sig_grads else 0
        grad_importance_background[feat_idx] = np.mean(bkg_grads) if bkg_grads else 0
    grad_importance = grad_importance / grad_importance.sum()
    grad_importance_signal = grad_importance_signal / grad_importance_signal.sum()
    grad_importance_background = grad_importance_background / grad_importance_background.sum()

    # 2. Permutation importance
    print("2. Computing permutation importance (this may take a minute)...")
    n_perm_samples = min(1000, len(all_constituents))
    perm_x = all_constituents[:n_perm_samples].float().to(device)
    perm_mask = perm_x[:, :, 1] > 0
    perm_labels = all_labels[:n_perm_samples]
    model.eval()
    with torch.no_grad():
        tmep_outputs = model(perm_x, particle_mask=perm_mask)
        baseline_outputs_fi = torch.sigmoid(tmep_outputs["classification"]).squeeze().cpu().numpy()
    baseline_auc = roc_auc_score_fi(perm_labels, baseline_outputs_fi)
    print(f"  Baseline AUC: {baseline_auc:.4f}")

    perm_importance = np.zeros(len(input_feature_names))
    n_permutations = 5
    for feat_idx in range(min(len(input_feature_names), perm_x.shape[2])):
        auc_drops = []
        for _ in range(n_permutations):
            perm_x_copy = perm_x.clone()
            perm_indices = torch.randperm(n_perm_samples)
            perm_x_copy[:, :, feat_idx] = perm_x[perm_indices, :, feat_idx]
            with torch.no_grad():
                tmep_outputs = model(perm_x_copy, particle_mask=perm_mask)
                perm_outputs_fi = torch.sigmoid(tmep_outputs["classification"]).squeeze().cpu().numpy()
            try:
                perm_auc = roc_auc_score_fi(perm_labels, perm_outputs_fi)
                auc_drops.append(baseline_auc - perm_auc)
            except:
                auc_drops.append(0)
        perm_importance[feat_idx] = np.mean(auc_drops)
        if (feat_idx + 1) % 4 == 0:
            print(f"    Processed {feat_idx + 1}/{min(len(input_feature_names), perm_x.shape[2])} features")

    # 3. Feature ablation
    print("3. Computing feature ablation importance...")
    ablation_importance = np.zeros(len(input_feature_names))
    for feat_idx in range(min(len(input_feature_names), perm_x.shape[2])):
        ablated_x = perm_x.clone()
        ablated_x[:, :, feat_idx] = 0
        with torch.no_grad():
            tmep_outputs = model(ablated_x, particle_mask=perm_mask)
            ablated_outputs = torch.sigmoid(tmep_outputs["classification"]).squeeze().cpu().numpy()
        try:
            ablated_auc = roc_auc_score_fi(perm_labels, ablated_outputs)
            ablation_importance[feat_idx] = baseline_auc - ablated_auc
        except:
            ablation_importance[feat_idx] = 0

    # 4. Statistical feature separability
    print("4. Computing statistical feature separability...")
    stat_x = all_constituents[:, :, :len(input_feature_names)].float().numpy()
    stat_mask_fi = (all_constituents[:, :, 1] > 0).numpy()
    stat_labels = all_labels
    fisher_scores = np.zeros(len(input_feature_names))
    ks_scores = np.zeros(len(input_feature_names))
    for feat_idx in range(min(len(input_feature_names), stat_x.shape[2])):
        sig_feat_vals = stat_x[stat_labels == 1, :, feat_idx]
        sig_mask_vals = stat_mask_fi[stat_labels == 1]
        sig_vals_fi = sig_feat_vals[sig_mask_vals].flatten()
        bkg_feat_vals = stat_x[stat_labels == 0, :, feat_idx]
        bkg_mask_vals = stat_mask_fi[stat_labels == 0]
        bkg_vals_fi = bkg_feat_vals[bkg_mask_vals].flatten()
        sig_mean_fi, bkg_mean_fi = sig_vals_fi.mean(), bkg_vals_fi.mean()
        sig_var_fi, bkg_var_fi = sig_vals_fi.var() + 1e-8, bkg_vals_fi.var() + 1e-8
        fisher_scores[feat_idx] = (sig_mean_fi - bkg_mean_fi) ** 2 / (sig_var_fi + bkg_var_fi)
        n_subsample = min(10000, len(sig_vals_fi), len(bkg_vals_fi))
        sig_sample = np.random.choice(sig_vals_fi, n_subsample, replace=False) if len(sig_vals_fi) > n_subsample else sig_vals_fi
        bkg_sample = np.random.choice(bkg_vals_fi, n_subsample, replace=False) if len(bkg_vals_fi) > n_subsample else bkg_vals_fi
        ks_stat, _ = ks_2samp(sig_sample, bkg_sample)
        ks_scores[feat_idx] = ks_stat

    # 5. Visualize results
    print("\nGenerating visualizations...")
    n_plot_features = min(len(input_feature_names), 12)
    plot_feature_names = input_feature_names[:n_plot_features]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    y_pos_fi = np.arange(n_plot_features)

    ax = axes[0, 0]
    sorted_idx_fi = np.argsort(grad_importance[:n_plot_features])[::-1]
    ax.barh(y_pos_fi, grad_importance[:n_plot_features][sorted_idx_fi], color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos_fi)
    ax.set_yticklabels([plot_feature_names[i] for i in sorted_idx_fi])
    ax.set_xlabel("Normalized Mean |Gradient|")
    ax.set_title("Gradient-Based Importance")
    ax.invert_yaxis()

    ax = axes[0, 1]
    sorted_idx_fi = np.argsort(perm_importance[:n_plot_features])[::-1]
    colors_fi = ["green" if v > 0 else "red" for v in perm_importance[:n_plot_features][sorted_idx_fi]]
    ax.barh(y_pos_fi, perm_importance[:n_plot_features][sorted_idx_fi], color=colors_fi, alpha=0.8)
    ax.set_yticks(y_pos_fi)
    ax.set_yticklabels([plot_feature_names[i] for i in sorted_idx_fi])
    ax.set_xlabel("AUC Drop When Permuted")
    ax.set_title("Permutation Importance")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.invert_yaxis()

    ax = axes[0, 2]
    sorted_idx_fi = np.argsort(ablation_importance[:n_plot_features])[::-1]
    colors_fi = ["green" if v > 0 else "red" for v in ablation_importance[:n_plot_features][sorted_idx_fi]]
    ax.barh(y_pos_fi, ablation_importance[:n_plot_features][sorted_idx_fi], color=colors_fi, alpha=0.8)
    ax.set_yticks(y_pos_fi)
    ax.set_yticklabels([plot_feature_names[i] for i in sorted_idx_fi])
    ax.set_xlabel("AUC Drop When Zeroed")
    ax.set_title("Feature Ablation Importance")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.invert_yaxis()

    ax = axes[1, 0]
    sorted_idx_fi = np.argsort(fisher_scores[:n_plot_features])[::-1]
    ax.barh(y_pos_fi, fisher_scores[:n_plot_features][sorted_idx_fi], color="darkorange", alpha=0.8)
    ax.set_yticks(y_pos_fi)
    ax.set_yticklabels([plot_feature_names[i] for i in sorted_idx_fi])
    ax.set_xlabel("Fisher Discriminant Ratio")
    ax.set_title("Statistical Separability (Fisher)")
    ax.invert_yaxis()

    ax = axes[1, 1]
    sorted_idx_fi = np.argsort(ks_scores[:n_plot_features])[::-1]
    ax.barh(y_pos_fi, ks_scores[:n_plot_features][sorted_idx_fi], color="purple", alpha=0.8)
    ax.set_yticks(y_pos_fi)
    ax.set_yticklabels([plot_feature_names[i] for i in sorted_idx_fi])
    ax.set_xlabel("KS Statistic")
    ax.set_title("Statistical Separability (KS Test)")
    ax.invert_yaxis()

    ax = axes[1, 2]
    def normalize_scores(scores):
        if scores.max() - scores.min() > 0:
            return (scores - scores.min()) / (scores.max() - scores.min())
        return scores
    combined_score = (
        normalize_scores(grad_importance[:n_plot_features])
        + normalize_scores(np.maximum(perm_importance[:n_plot_features], 0))
        + normalize_scores(np.maximum(ablation_importance[:n_plot_features], 0))
        + normalize_scores(fisher_scores[:n_plot_features])
        + normalize_scores(ks_scores[:n_plot_features])
    ) / 5
    sorted_idx_fi = np.argsort(combined_score)[::-1]
    ax.barh(y_pos_fi, combined_score[sorted_idx_fi], color="darkgreen", alpha=0.8)
    ax.set_yticks(y_pos_fi)
    ax.set_yticklabels([plot_feature_names[i] for i in sorted_idx_fi])
    ax.set_xlabel("Combined Importance Score")
    ax.set_title("Overall Feature Ranking")
    ax.invert_yaxis()
    plt.suptitle("Feature Importance Analysis for Signal vs Background Discrimination", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "feature_importance_analysis")
    plt.close(fig)

    # Signal vs background gradient comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x_fi = np.arange(n_plot_features)
    width_fi = 0.35
    ax.bar(x_fi - width_fi / 2, grad_importance_signal[:n_plot_features], width_fi, label="Signal (b-jets)", color="blue", alpha=0.7)
    ax.bar(x_fi + width_fi / 2, grad_importance_background[:n_plot_features], width_fi, label="Background", color="red", alpha=0.7)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Normalized Mean |Gradient|")
    ax.set_title("Gradient Importance: Signal vs Background")
    ax.set_xticks(x_fi)
    ax.set_xticklabels(plot_feature_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "gradient_importance_by_class")
    plt.close(fig)

    if n_features_fi > 12:
        print("Computing gradient importance per particle type...")
        grad_particle_types = np.argmax(
            all_constituents[:n_grad_samples, :, 12:17].numpy(), axis=-1
        )
        grad_particle_types[~mask_np_fi] = -1

        grad_by_type = np.zeros((N_PARTICLE_TYPES, n_plot_features), dtype=np.float64)
        for pid in range(N_PARTICLE_TYPES):
            for feat_idx in range(n_plot_features):
                grads_list = []
                for i in range(n_grad_samples):
                    n_valid = int(mask_np_fi[i].sum())
                    ptypes_i = grad_particle_types[i, :n_valid]
                    feat_grads = np.abs(grads_prob[i, :n_valid, feat_idx].numpy())
                    type_grads = feat_grads[ptypes_i == pid]
                    grads_list.extend(type_grads)
                grad_by_type[pid, feat_idx] = np.mean(grads_list) if grads_list else 0.0

        for pid in range(N_PARTICLE_TYPES):
            row_sum = grad_by_type[pid].sum()
            if row_sum > 0:
                grad_by_type[pid] /= row_sum

        fig, ax = plt.subplots(figsize=(16, 7))
        x = np.arange(n_plot_features)
        total_width = 0.8
        bar_width = total_width / N_PARTICLE_TYPES

        for pid in range(N_PARTICLE_TYPES):
            offset = (pid - N_PARTICLE_TYPES / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                grad_by_type[pid],
                bar_width,
                label=PARTICLE_TYPE_NAMES[pid],
                color=PARTICLE_TYPE_COLORS[pid],
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(plot_feature_names, rotation=45, ha="right")
        ax.set_ylabel("Normalized Mean |Gradient|")
        ax.set_title("Feature Importance by Particle Type (Gradient-Based)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_fig(fig, "gradient_importance_by_particle_type")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(grad_by_type, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(N_PARTICLE_TYPES))
        ax.set_yticklabels(PARTICLE_TYPE_NAMES)
        ax.set_xticks(range(n_plot_features))
        ax.set_xticklabels(plot_feature_names, rotation=45, ha="right")
        ax.set_title("Feature Importance Heatmap by Particle Type")
        plt.colorbar(im, ax=ax, label="Normalized |Gradient|")

        threshold = grad_by_type.max() * 0.5 if grad_by_type.size else 0.0
        for ii in range(N_PARTICLE_TYPES):
            for jj in range(n_plot_features):
                ax.text(
                    jj,
                    ii,
                    f"{grad_by_type[ii, jj]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if grad_by_type[ii, jj] > threshold else "black",
                )

        plt.tight_layout()
        save_fig(fig, "gradient_importance_heatmap_by_particle_type")
        plt.close(fig)
        print("Done with particle-type-specific gradient importance.")

        print("\nComputing permutation + ablation importance by particle type...")
        n_particle_types = min(5, perm_x.shape[2] - 12)
        ptype_names = PARTICLE_TYPE_NAMES[:n_particle_types]
        ptype_colors = PARTICLE_TYPE_COLORS[:n_particle_types]
        n_type_features = min(12, len(input_feature_names), perm_x.shape[2])
        type_feature_names = input_feature_names[:n_type_features]

        ptype_onehot = perm_x[:, :, 12: 12 + n_particle_types]
        ptype_idx = torch.argmax(ptype_onehot, dim=-1)
        has_ptype = ptype_onehot.sum(dim=-1) > 0
        valid_const = perm_mask & has_ptype

        sig_row_mask = torch.as_tensor(perm_labels == 1, device=perm_x.device)
        bkg_row_mask = torch.as_tensor(perm_labels == 0, device=perm_x.device)

        perm_by_type = np.zeros((n_particle_types, n_type_features), dtype=np.float64)
        abla_by_type = np.zeros((n_particle_types, n_type_features), dtype=np.float64)

        n_perm_local = n_permutations
        perm_weights = np.ones(n_perm_samples, dtype=np.float64)
        if "all_qcd_weights_val" in locals() and len(all_qcd_weights_val) >= n_perm_samples:
            perm_weights = np.asarray(all_qcd_weights_val[:n_perm_samples], dtype=np.float64)

        try:
            baseline_auc_type = roc_auc_score_fi(
                perm_labels,
                baseline_outputs_fi,
                sample_weight=perm_weights,
            )
        except Exception:
            baseline_auc_type = baseline_auc

        for pid in range(n_particle_types):
            type_mask = valid_const & (ptype_idx == pid)
            if int(type_mask.sum().item()) == 0:
                continue

            for f_idx in range(n_type_features):
                auc_drops = []
                for _ in range(n_perm_local):
                    x_perm_t = perm_x.clone()

                    sel_sig = type_mask & sig_row_mask[:, None]
                    vals_sig = x_perm_t[:, :, f_idx][sel_sig]
                    if vals_sig.numel() > 1:
                        vals_sig = vals_sig[torch.randperm(vals_sig.numel(), device=perm_x.device)]
                        x_perm_t[:, :, f_idx][sel_sig] = vals_sig

                    sel_bkg = type_mask & bkg_row_mask[:, None]
                    vals_bkg = x_perm_t[:, :, f_idx][sel_bkg]
                    if vals_bkg.numel() > 1:
                        vals_bkg = vals_bkg[torch.randperm(vals_bkg.numel(), device=perm_x.device)]
                        x_perm_t[:, :, f_idx][sel_bkg] = vals_bkg

                    with torch.no_grad():
                        eval_mask = perm_mask if f_idx != 1 else (x_perm_t[:, :, 1] > 0)
                        s_perm_t = torch.sigmoid(
                            model(x_perm_t, particle_mask=eval_mask)["classification"]
                        ).squeeze().cpu().numpy()

                    auc_perm_t = roc_auc_score_fi(
                        perm_labels,
                        s_perm_t,
                        sample_weight=perm_weights,
                    )
                    auc_drops.append(baseline_auc_type - auc_perm_t)

                perm_by_type[pid, f_idx] = float(np.mean(auc_drops))

                x_zero_t = perm_x.clone()
                x_zero_t[:, :, f_idx][type_mask] = 0.0
                with torch.no_grad():
                    eval_mask = perm_mask if f_idx != 1 else (x_zero_t[:, :, 1] > 0)
                    s_zero_t = torch.sigmoid(
                        model(x_zero_t, particle_mask=eval_mask)["classification"]
                    ).squeeze().cpu().numpy()

                auc_zero_t = roc_auc_score_fi(
                    perm_labels,
                    s_zero_t,
                    sample_weight=perm_weights,
                )
                abla_by_type[pid, f_idx] = baseline_auc_type - auc_zero_t

        def plot_stacked_particle_contrib(
            contrib_by_type,
            feature_names,
            type_names,
            type_colors,
            title,
            xlabel,
        ):
            contrib_by_type = np.asarray(contrib_by_type, dtype=float)
            totals = contrib_by_type.sum(axis=0)
            order = np.argsort(totals)[::-1]
            totals = totals[order]
            feat_sorted = [feature_names[i] for i in order]
            contrib_sorted = contrib_by_type[:, order]

            y = np.arange(len(feat_sorted))
            pos_base = np.zeros(len(feat_sorted), dtype=float)
            neg_base = np.zeros(len(feat_sorted), dtype=float)

            fig, ax = plt.subplots(figsize=(11, max(6, 0.5 * len(feat_sorted))))
            for t in range(contrib_sorted.shape[0]):
                vals = contrib_sorted[t]
                left = np.where(vals >= 0, pos_base, neg_base)
                ax.barh(
                    y,
                    vals,
                    left=left,
                    color=type_colors[t],
                    alpha=0.88,
                    label=type_names[t],
                    edgecolor="white",
                    linewidth=0.5,
                )
                pos_base += np.where(vals >= 0, vals, 0.0)
                neg_base += np.where(vals < 0, vals, 0.0)

            ax.scatter(totals, y, marker="|", s=260, color="black", zorder=4, label="Total (sum)")
            ax.axvline(0.0, color="black", linestyle="--", alpha=0.65)
            ax.set_yticks(y)
            ax.set_yticklabels(feat_sorted)
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.25)
            ax.legend(loc="best", fontsize=9)
            plt.tight_layout()

            print(f"\n{title}")
            print("-" * len(title))
            for i, f_name in enumerate(feat_sorted):
                print(f"{f_name:<14} total={totals[i]:+0.5f}")
            return fig

        fig_stacked_perm = plot_stacked_particle_contrib(
            perm_by_type,
            type_feature_names,
            ptype_names,
            ptype_colors,
            "Permutation Importance by Feature (Stacked by Particle Type)",
            "AUC Drop When Permuted",
        )
        fig_stacked_ablate = plot_stacked_particle_contrib(
            abla_by_type,
            type_feature_names,
            ptype_names,
            ptype_colors,
            "Ablation Importance by Feature (Stacked by Particle Type)",
            "AUC Drop When Zeroed",
        )

        print("\nTop 3 features by particle type (Permutation / Ablation):")
        for pid in range(n_particle_types):
            p_top = np.argsort(perm_by_type[pid])[::-1][:3]
            a_top = np.argsort(abla_by_type[pid])[::-1][:3]
            p_txt = ", ".join(
                [f"{type_feature_names[i]} ({perm_by_type[pid, i]:+.4f})" for i in p_top]
            )
            a_txt = ", ".join(
                [f"{type_feature_names[i]} ({abla_by_type[pid, i]:+.4f})" for i in a_top]
            )
            print(f"  {ptype_names[pid]}:")
            print(f"    Permute: {p_txt}")
            print(f"    Ablate : {a_txt}")

        permutation_importance_by_particle_type = perm_by_type
        ablation_importance_by_particle_type = abla_by_type
        particle_type_feature_names = type_feature_names
        print("\nStored arrays:")
        print(
            "  permutation_importance_by_particle_type  shape =",
            permutation_importance_by_particle_type.shape,
        )
        print(
            "  ablation_importance_by_particle_type     shape =",
            ablation_importance_by_particle_type.shape,
        )

        save_fig(fig_stacked_perm, "stacked_permutation_importance_by_particle_type")
        save_fig(fig_stacked_ablate, "stacked_ablation_importance_by_particle_type")
        plt.close(fig_stacked_perm)
        plt.close(fig_stacked_ablate)
        print("Saved stacked bar charts for both permutation and ablation importance by particle type.")
        print("Done with particle-type-specific permutation and ablation importance.")

    # Feature correlation with model output
    print("\nComputing feature correlations with model output...")
    with torch.no_grad():
        tmep_outputs = model(perm_x, particle_mask=perm_mask)
        pred_outputs_fi = torch.sigmoid(tmep_outputs["classification"]).squeeze().cpu().numpy()
    mean_features = np.zeros((n_perm_samples, n_plot_features))
    for i in range(n_perm_samples):
        n_valid = int(perm_mask[i].sum().item())
        for f_idx in range(n_plot_features):
            mean_features[i, f_idx] = perm_x[i, :n_valid, f_idx].cpu().numpy().mean()
    feature_pred_corr = np.zeros(n_plot_features)
    for f_idx in range(n_plot_features):
        feature_pred_corr[f_idx] = np.corrcoef(mean_features[:, f_idx], pred_outputs_fi)[0, 1]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_corr = ["blue" if c > 0 else "red" for c in feature_pred_corr]
    ax.bar(range(n_plot_features), feature_pred_corr, color=colors_corr, alpha=0.7)
    ax.set_xticks(range(n_plot_features))
    ax.set_xticklabels(plot_feature_names, rotation=45, ha="right")
    ax.set_ylabel("Pearson Correlation")
    ax.set_xlabel("Feature")
    ax.set_title("Feature Correlation with Model Output (b-tag Score)")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "feature_correlation_with_output")
    plt.close(fig)

    # Summary table
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 80)
    print(f"{'Feature':<15} {'Gradient':>10} {'Permute':>10} {'Ablation':>10} {'Fisher':>10} {'KS':>10} {'Combined':>10}")
    print("-" * 80)
    for i in range(n_plot_features):
        print(f"{plot_feature_names[i]:<15} {grad_importance[i]:>10.4f} {perm_importance[i]:>10.4f} {ablation_importance[i]:>10.4f} {fisher_scores[i]:>10.4f} {ks_scores[i]:>10.4f} {combined_score[i]:>10.4f}")
    print("\n" + "=" * 80)
    print("TOP 5 MOST IMPORTANT FEATURES (by combined score):")
    print("=" * 80)
    top_5_idx = np.argsort(combined_score)[::-1][:5]
    for rank, idx_t5 in enumerate(top_5_idx, 1):
        print(f"  {rank}. {plot_feature_names[idx_t5]} (score: {combined_score[idx_t5]:.4f})")
    print(f"\n{'='*60}")
    print("Feature importance analysis complete!")
    print(f"{'='*60}")

    # ── Cell 28: Model behavior analysis & maximum discriminative power ──
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss, log_loss
    from scipy.stats import entropy as sp_entropy
    from scipy.stats import gaussian_kde

    print("=" * 70)
    print("MODEL BEHAVIOR ANALYSIS & MAXIMUM DISCRIMINATIVE POWER ESTIMATION")
    print("=" * 70)

    # 1. Error analysis
    print("\n1. ERROR ANALYSIS")
    print("-" * 50)
    threshold_mb = 0.5
    predictions_mb = (all_outputs > threshold_mb).astype(int)
    correct_mb = predictions_mb == all_labels
    all_labels = all_labels.squeeze()

    false_positives_mb = (predictions_mb == 1) & (all_labels == 0)
    false_negatives_mb = (predictions_mb == 0) & (all_labels == 1)
    true_positives_mb = (predictions_mb == 1) & (all_labels == 1)
    true_negatives_mb = (predictions_mb == 0) & (all_labels == 0)

    print(f"Threshold: {threshold_mb}")
    print(f"  True Positives:  {true_positives_mb.sum():>6} ({100*true_positives_mb.sum()/all_labels.sum():.1f}% of signal)")
    print(f"  True Negatives:  {true_negatives_mb.sum():>6} ({100*true_negatives_mb.sum()/(len(all_labels)-all_labels.sum()):.1f}% of background)")
    print(f"  False Positives: {false_positives_mb.sum():>6} (background misclassified as signal)")
    print(f"  False Negatives: {false_negatives_mb.sum():>6} (signal misclassified as background)")

    print("\n  Kinematic properties of misclassified jets:")
    fp_pt = val_jet_pt[false_positives_mb]
    fn_pt = val_jet_pt[false_negatives_mb]
    tp_pt = val_jet_pt[true_positives_mb]
    tn_pt = val_jet_pt[true_negatives_mb]
    print(f"    False Positives: mean pT = {fp_pt.mean():.1f} GeV, mean |η| = {np.abs(val_jet_eta[false_positives_mb]).mean():.2f}")
    print(f"    False Negatives: mean pT = {fn_pt.mean():.1f} GeV, mean |η| = {np.abs(val_jet_eta[false_negatives_mb]).mean():.2f}")
    print(f"    True Positives:  mean pT = {tp_pt.mean():.1f} GeV, mean |η| = {np.abs(val_jet_eta[true_positives_mb]).mean():.2f}")
    print(f"    True Negatives:  mean pT = {tn_pt.mean():.1f} GeV, mean |η| = {np.abs(val_jet_eta[true_negatives_mb]).mean():.2f}")

    # 2. Confidence calibration
    print("\n2. CONFIDENCE CALIBRATION")
    print("-" * 50)
    brier = brier_score_loss(all_labels, all_outputs)
    logloss = log_loss(all_labels, np.clip(all_outputs, 1e-7, 1 - 1e-7))
    print(f"  Brier Score: {brier:.4f} (lower is better, perfect = 0)")
    print(f"  Log Loss: {logloss:.4f} (lower is better)")
    prob_true, prob_pred = calibration_curve(all_labels, all_outputs, n_bins=10, strategy="uniform")
    print(f"\n  Calibration by probability bin:")
    print(f"  {'Predicted':>12} {'Actual':>12} {'Difference':>12}")
    for pt_cal, pp_cal in zip(prob_true, prob_pred):
        print(f"  {pp_cal:>12.3f} {pt_cal:>12.3f} {pt_cal-pp_cal:>+12.3f}")
    bin_edges_ece = np.linspace(0, 1, 11)
    ece = 0
    for i_ece in range(10):
        mask_ece = (all_outputs >= bin_edges_ece[i_ece]) & (all_outputs < bin_edges_ece[i_ece + 1])
        if mask_ece.sum() > 0:
            bin_acc = all_labels[mask_ece].mean()
            bin_conf = all_outputs[mask_ece].mean()
            ece += mask_ece.sum() * np.abs(bin_acc - bin_conf)
    ece /= len(all_labels)
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")

    # 3. Hard sample analysis
    print("\n3. HARD SAMPLE ANALYSIS")
    print("-" * 50)
    uncertainty_mb = np.abs(all_outputs - 0.5)
    hard_threshold_mb = 0.2
    hard_samples = uncertainty_mb < hard_threshold_mb
    easy_samples = uncertainty_mb >= 0.4
    print(f"  Hard samples (output in [0.3, 0.7]): {hard_samples.sum()} ({100*hard_samples.mean():.1f}%)")
    print(f"  Easy samples (output < 0.1 or > 0.9): {easy_samples.sum()} ({100*easy_samples.mean():.1f}%)")
    hard_acc = (predictions_mb[hard_samples] == all_labels[hard_samples]).mean()
    easy_acc = (predictions_mb[easy_samples] == all_labels[easy_samples]).mean()
    print(f"\n  Accuracy on hard samples: {100*hard_acc:.1f}%")
    print(f"  Accuracy on easy samples: {100*easy_acc:.1f}%")
    print("\n  Characteristics of hard samples:")
    hard_mask_mb = all_constituents[:, :, 1] > 0
    hard_n_const = hard_mask_mb[hard_samples].sum(dim=1).float().mean().item()
    easy_n_const = hard_mask_mb[easy_samples].sum(dim=1).float().mean().item()
    print(f"    Mean constituents (hard): {hard_n_const:.1f}")
    print(f"    Mean constituents (easy): {easy_n_const:.1f}")
    print(f"    Mean pT (hard): {val_jet_pt[hard_samples].mean():.1f} GeV")
    print(f"    Mean pT (easy): {val_jet_pt[easy_samples].mean():.1f} GeV")

    # 4. Maximum discriminative power estimation
    print("\n4. MAXIMUM DISCRIMINATIVE POWER ESTIMATION")
    print("-" * 50)
    print("\n  Method 1: k-NN Class Overlap Estimation")
    n_knn = min(50000, len(all_constituents))
    knn_x = all_constituents[:n_knn]
    knn_mask = knn_x[:, :, 1] > 0
    knn_features = []
    for i_knn in range(n_knn):
        n_valid = int(knn_mask[i_knn].sum().item())
        jet_feats = knn_x[i_knn, :n_valid, :12].mean(dim=0).numpy()
        knn_features.append(jet_feats)
    knn_features = np.array(knn_features)
    knn_labels = all_labels[:n_knn]
    knn_aucs = []
    for k_knn in [1, 3, 5, 11, 21]:
        knn = KNeighborsClassifier(n_neighbors=k_knn, weights="distance")
        train_idx_knn = np.random.choice(n_knn, int(0.7 * n_knn), replace=False)
        test_idx_knn = np.setdiff1d(np.arange(n_knn), train_idx_knn)
        knn.fit(knn_features[train_idx_knn], knn_labels[train_idx_knn])
        knn_proba = knn.predict_proba(knn_features[test_idx_knn])[:, 1]
        knn_auc = roc_auc_score_fi(knn_labels[test_idx_knn], knn_proba)
        knn_aucs.append(knn_auc)
        print(f"    k={k_knn:>2}: AUC = {knn_auc:.4f}")
    best_knn_auc = max(knn_aucs)
    print(f"    Best k-NN AUC (upper bound estimate): {best_knn_auc:.4f}")

    print("\n  Method 2: Feature Distribution Overlap Analysis")
    feature_overlaps = []
    for feat_idx_ol in range(12):
        sig_vals_ol = knn_features[knn_labels == 1, feat_idx_ol]
        bkg_vals_ol = knn_features[knn_labels == 0, feat_idx_ol]
        n_sub = min(1000, len(sig_vals_ol), len(bkg_vals_ol))
        sig_sub = np.random.choice(sig_vals_ol, n_sub, replace=False)
        bkg_sub = np.random.choice(bkg_vals_ol, n_sub, replace=False)
        try:
            x_range_ol = np.linspace(min(sig_sub.min(), bkg_sub.min()), max(sig_sub.max(), bkg_sub.max()), 100)
            sig_kde = gaussian_kde(sig_sub)
            bkg_kde = gaussian_kde(bkg_sub)
            sig_pdf = sig_kde(x_range_ol)
            bkg_pdf = bkg_kde(x_range_ol)
            overlap = np.sum(np.sqrt(sig_pdf * bkg_pdf)) * (x_range_ol[1] - x_range_ol[0])
            feature_overlaps.append(overlap)
        except:
            feature_overlaps.append(1.0)
    feature_overlaps = np.array(feature_overlaps)
    mean_overlap = feature_overlaps.mean()
    estimated_max_auc = 1 - mean_overlap / 2
    print(f"    Mean feature overlap: {mean_overlap:.4f}")
    print(f"    Estimated max AUC (from overlap): {estimated_max_auc:.4f}")

    print("\n  Method 3: Theoretical Bounds")
    n_signal_mb = all_labels.sum()
    n_background_mb = len(all_labels) - n_signal_mb
    class_ratio = n_signal_mb / len(all_labels)
    print(f"    Class balance: {100*class_ratio:.1f}% signal / {100*(1-class_ratio):.1f}% background")
    current_auc = roc_auc_score_fi(all_labels, all_outputs)
    print(f"    Random classifier AUC: 0.5000")
    print(f"    Current model AUC: {current_auc:.4f}")
    print(f"    Best k-NN AUC: {best_knn_auc:.4f}")
    gap_to_knn = best_knn_auc - current_auc
    gap_to_perfect = 1.0 - current_auc
    print(f"\n    Performance gaps:")
    print(f"      Gap to k-NN (achievable): {gap_to_knn:+.4f}")
    print(f"      Gap to perfect: {gap_to_perfect:+.4f}")

    # 5. Improvement recommendations
    print("\n5. IMPROVEMENT RECOMMENDATIONS")
    print("-" * 50)
    recommendations = []
    if ece > 0.05:
        recommendations.append("• Model is poorly calibrated (ECE > 0.05). Consider temperature scaling or Platt calibration.")
    if class_ratio < 0.3 or class_ratio > 0.7:
        recommendations.append("• Class imbalance detected. Consider focal loss or class weighting.")
    if best_knn_auc < current_auc:
        recommendations.append("• Model outperforms k-NN, suggesting good feature learning.")
    else:
        recommendations.append(f"• k-NN achieves higher AUC ({best_knn_auc:.4f} vs {current_auc:.4f}). Model may benefit from more capacity or training.")
    if hard_samples.mean() > 0.3:
        recommendations.append(f"• {100*hard_samples.mean():.0f}% of samples are hard. Consider curriculum learning or sample weighting.")
    pt_corr_mb = np.corrcoef(val_jet_pt, all_outputs)[0, 1]
    if abs(pt_corr_mb) > 0.3:
        recommendations.append(f"• Strong pT-score correlation ({pt_corr_mb:.2f}). Consider pT reweighting for robustness.")
    for rec in recommendations:
        print(f"  {rec}")
    if not recommendations:
        print("  • Model appears well-optimized! Consider ensemble methods or architectural changes for further improvement.")

    # 6. Diagnostic visualization
    print("\nGenerating diagnostic plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "bo-", label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    pt_bins_mb = np.linspace(0, 300, 20)
    fp_hist, _ = np.histogram(fp_pt, bins=pt_bins_mb, density=True)
    fn_hist, _ = np.histogram(fn_pt, bins=pt_bins_mb, density=True)
    pt_centers_mb = (pt_bins_mb[:-1] + pt_bins_mb[1:]) / 2
    ax.plot(pt_centers_mb, fp_hist, "r-", label="False Positives", alpha=0.8)
    ax.plot(pt_centers_mb, fn_hist, "b-", label="False Negatives", alpha=0.8)
    ax.set_xlabel("Jet pT [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("pT Distribution of Misclassified Jets")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.hist(all_outputs[all_labels == 1], bins=50, range=(0, 1), histtype="step", label="Signal", color="blue", density=True)
    ax.hist(all_outputs[all_labels == 0], bins=50, range=(0, 1), histtype="step", label="Background", color="red", density=True)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.axvspan(0.3, 0.7, alpha=0.1, color="gray", label="Hard region")
    ax.set_xlabel("Model Output")
    ax.set_ylabel("Density")
    ax.set_title("Output Distribution with Hard Region")
    ax.legend()

    ax = axes[1, 0]
    pt_bin_edges_mb = [25, 50, 75, 100, 150, 200, 300, 500]
    pt_aucs = []
    for i_pt in range(len(pt_bin_edges_mb) - 1):
        mask_pt = (val_jet_pt >= pt_bin_edges_mb[i_pt]) & (val_jet_pt < pt_bin_edges_mb[i_pt + 1])
        if mask_pt.sum() > 50 and len(np.unique(all_labels[mask_pt])) > 1:
            pt_aucs.append(roc_auc_score_fi(all_labels[mask_pt], all_outputs[mask_pt]))
        else:
            pt_aucs.append(np.nan)
    ax.bar(range(len(pt_aucs)), pt_aucs, color="steelblue", alpha=0.7)
    ax.set_xticks(range(len(pt_aucs)))
    ax.set_xticklabels([f"{pt_bin_edges_mb[i]}-{pt_bin_edges_mb[i+1]}" for i in range(len(pt_bin_edges_mb) - 1)], rotation=45)
    ax.set_xlabel("pT Range [GeV]")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs Jet pT")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    feature_names_short = ["Mass", "pT", "η", "φ", "d_xy", "z_0", "Q", "log(pT_rel)", "Δη", "Δφ", "PUPPI", "log(ΔR)"]
    sorted_idx_ol = np.argsort(feature_overlaps)
    ax.barh(range(len(feature_overlaps)), feature_overlaps[sorted_idx_ol], color="darkorange", alpha=0.7)
    ax.set_yticks(range(len(feature_overlaps)))
    ax.set_yticklabels([feature_names_short[i] for i in sorted_idx_ol])
    ax.set_xlabel("Distribution Overlap (lower = more separable)")
    ax.set_title("Feature Separability")
    ax.invert_yaxis()

    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        f"\nMODEL PERFORMANCE SUMMARY\n{'='*40}\n\n"
        f"Current Performance:\n"
        f"  AUC: {current_auc:.4f}\n"
        f"  Brier Score: {brier:.4f}\n"
        f"  Log Loss: {logloss:.4f}\n"
        f"  ECE: {ece:.4f}\n\n"
        f"Error Analysis:\n"
        f"  FP Rate: {100*false_positives_mb.sum()/(len(all_labels)-all_labels.sum()):.2f}%\n"
        f"  FN Rate: {100*false_negatives_mb.sum()/all_labels.sum():.2f}%\n"
        f"  Hard Samples: {100*hard_samples.mean():.1f}%\n\n"
        f"Max Performance Estimate:\n"
        f"  k-NN Upper Bound: {best_knn_auc:.4f}\n"
        f"  From Feature Overlap: {estimated_max_auc:.4f}\n"
        f"  Improvement Potential: {max(0, best_knn_auc - current_auc):.4f}\n\n"
        f"Data Characteristics:\n"
        f"  Samples: {len(all_labels):,}\n"
        f"  Signal: {int(n_signal_mb):,} ({100*class_ratio:.1f}%)\n"
        f"  Background: {int(n_background_mb):,} ({100*(1-class_ratio):.1f}%)\n"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Model Behavior Analysis & Maximum Discriminative Power", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "model_behavior_analysis")
    plt.close(fig)

    # Error analysis by particle composition (notebook parity)
    incorrect_mb = ~correct_mb
    if particle_type_full is not None and all_constituents.shape[2] > 12:
        val_particle_types = np.argmax(all_constituents[:, :, 12:17].numpy(), axis=-1)
        val_mask = (all_constituents[:, :, 1] > 0).numpy()
        val_particle_types[~val_mask] = -1

        val_particle_counts = np.zeros((len(all_constituents), N_PARTICLE_TYPES), dtype=int)
        for pid in range(N_PARTICLE_TYPES):
            val_particle_counts[:, pid] = (val_particle_types == pid).sum(axis=1)
        val_dominant_type = np.argmax(val_particle_counts, axis=1)

        print("\n  Misclassification Rate by Dominant Particle Type:")
        print(f"  {'Type':<20} {'N_jets':>8} {'Error Rate':>12} {'FP Rate':>10} {'FN Rate':>10}")
        print("  " + "-" * 65)

        for pid in range(N_PARTICLE_TYPES):
            type_mask = val_dominant_type == pid
            n_type = int(type_mask.sum())
            if n_type == 0:
                continue

            n_incorrect = int(incorrect_mb[type_mask].sum())
            n_fp = int(false_positives_mb[type_mask].sum())
            n_fn = int(false_negatives_mb[type_mask].sum())
            n_sig_type = int((all_labels[type_mask] == 1).sum())
            n_bkg_type = int((all_labels[type_mask] == 0).sum())

            err_rate = n_incorrect / n_type
            fp_rate = n_fp / n_bkg_type if n_bkg_type > 0 else 0.0
            fn_rate = n_fn / n_sig_type if n_sig_type > 0 else 0.0

            print(
                f"  {PARTICLE_TYPE_NAMES[pid]:<20} {n_type:>8} {err_rate:>12.4f} "
                f"{fp_rate:>10.4f} {fn_rate:>10.4f}"
            )

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        categories = [
            (true_positives_mb, "True Positives"),
            (false_negatives_mb, "False Negatives"),
            (true_negatives_mb, "True Negatives"),
            (false_positives_mb, "False Positives"),
        ]

        short_names = ["ChHad", "Elec", "NeuHad", "Photon", "Muon"][:N_PARTICLE_TYPES]
        for ax, (cat_mask, cat_name) in zip(axes, categories):
            if int(cat_mask.sum()) > 0:
                cat_counts = val_particle_counts[cat_mask].mean(axis=0)
                cat_fracs = cat_counts / cat_counts.sum() if cat_counts.sum() > 0 else cat_counts
            else:
                cat_fracs = np.zeros(N_PARTICLE_TYPES)

            ax.bar(range(N_PARTICLE_TYPES), cat_fracs, color=PARTICLE_TYPE_COLORS, alpha=0.8)
            ax.set_xticks(range(N_PARTICLE_TYPES))
            ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Mean Fraction")
            ax.set_title(f"{cat_name}\n(n={int(cat_mask.sum())})")
            ax.set_ylim(0, 1)

        plt.suptitle("Particle Composition of Classification Outcomes", fontsize=14)
        plt.tight_layout()
        save_fig(fig, "error_analysis_by_particle_type")
        plt.close(fig)

    print(f"\n{'='*70}")
    print("Model behavior analysis complete!")
    print(f"{'='*70}")

    if not should_run(args.profile, "full"):
        print("\nAll analysis complete!")
        return

    # ── Full profile parity: AK8 H-tagging + AK8 di-Higgs sections ───
    print("\n" + "=" * 70)
    print("FULL PROFILE NOTEBOOK PARITY: AK8 H-TAGGING")
    print("=" * 70)

    from sklearn.metrics import roc_curve, auc, roc_auc_score
    import matplotlib.colors as mcolors

    htag_labels = all_labels.reshape(-1)
    htag_scores = all_outputs.reshape(-1)
    htag_weights = all_qcd_weights_val.reshape(-1)

    htag_fpr, htag_tpr, htag_thresholds = roc_curve(
        htag_labels, htag_scores, sample_weight=htag_weights
    )
    htag_auc = auc(htag_fpr, htag_tpr)

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.plot(
        htag_fpr,
        htag_tpr,
        color="darkred",
        linewidth=2,
        label=f"ParT H-tag (AUC = {htag_auc:.4f})",
    )
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate (QCD mistag)")
    ax1.set_ylabel("True Positive Rate (H-tag efficiency)")
    ax1.set_title("H-Tagging ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    plt.tight_layout()
    save_fig(fig1, "htag_roc_linear")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    valid_fpr = htag_fpr > 0
    ax2.plot(
        htag_fpr[valid_fpr],
        htag_tpr[valid_fpr],
        color="darkred",
        linewidth=2,
        label=f"ParT H-tag (AUC = {htag_auc:.4f})",
    )
    ax2.set_xlabel("Mistag Rate")
    ax2.set_ylabel("H-Tagging Efficiency")
    ax2.set_xscale("log")
    ax2.set_xlim(1e-4, 1.0)
    ax2.set_ylim(1e-4, 1.05)
    ax2.set_title("H-Tagging ROC Curve")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    save_fig(fig2, "htag_roc_log")
    plt.close(fig2)

    def get_htag_wp(fpr_arr, tpr_arr, thresh_arr, target_mistag):
        idx = np.argmin(np.abs(fpr_arr - target_mistag))
        return fpr_arr[idx], tpr_arr[idx], thresh_arr[idx]

    htag_wps = []
    print(f"\nParT H-Tag Working Points (AUC = {htag_auc:.5f}):")
    for wp_name, target in [("Tight", 0.001), ("Medium", 0.01), ("Loose", 0.1)]:
        fpr_wp, tpr_wp, thresh_wp = get_htag_wp(htag_fpr, htag_tpr, htag_thresholds, target)
        print(
            f"  {wp_name}: TPR={tpr_wp*100:.2f}%, FPR={fpr_wp:.4f}, "
            f"1/FPR={1/max(fpr_wp,1e-9):.1f}, thresh={thresh_wp:.4f}"
        )
        htag_wps.append(thresh_wp)

    htag_pt = val_jet_pt
    htag_eta = val_jet_eta
    pt_ranges_hm = [
        (25, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, 500),
        (500, np.inf),
        (25, np.inf),
    ]
    eta_ranges_hm = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.4), (0, 1.5), (0, 2.4)]

    n_pt = len(pt_ranges_hm)
    n_eta = len(eta_ranges_hm)
    auc_grid = np.full((n_pt, n_eta), np.nan)
    count_grid = np.zeros((n_pt, n_eta), dtype=int)

    for i_pt_hm, (pt_low, pt_high) in enumerate(pt_ranges_hm):
        for j_eta_hm, (eta_low, eta_high) in enumerate(eta_ranges_hm):
            mask_bin = (
                (htag_pt >= pt_low)
                & (htag_pt < pt_high)
                & (np.abs(htag_eta) >= eta_low)
                & (np.abs(htag_eta) < eta_high)
            )
            bin_y = htag_labels[mask_bin]
            bin_s = htag_scores[mask_bin]
            bin_w = htag_weights[mask_bin]
            count_grid[i_pt_hm, j_eta_hm] = len(bin_y)

            if len(np.unique(bin_y)) < 2 or len(bin_y) < 20:
                continue
            try:
                auc_grid[i_pt_hm, j_eta_hm] = roc_auc_score(bin_y, bin_s, sample_weight=bin_w)
            except ValueError:
                continue

    pt_labels_hm = [f"[{low},{high})" for low, high in pt_ranges_hm]
    eta_labels_hm = [f"[{low},{high})" for low, high in eta_ranges_hm]

    fig3, ax3 = plt.subplots(figsize=(9, 7))
    im_auc = ax3.imshow(auc_grid, aspect="auto", origin="lower", vmin=0.5, vmax=1.0)
    for i_pt_hm in range(n_pt):
        for j_eta_hm in range(n_eta):
            v = auc_grid[i_pt_hm, j_eta_hm]
            txt = f"{v:.3f}" if not np.isnan(v) else "-"
            ax3.text(j_eta_hm, i_pt_hm, txt, ha="center", va="center", fontsize=10)
    ax3.set_xticks(range(n_eta))
    ax3.set_xticklabels(eta_labels_hm, rotation=30, fontsize=10)
    ax3.set_yticks(range(n_pt))
    ax3.set_yticklabels(pt_labels_hm, fontsize=10)
    ax3.set_xlabel(r"$|\eta|$ bin")
    ax3.set_ylabel(r"$p_T$ bin [GeV]")
    ax3.set_title("H-Tag AUC per (pT, |eta|) bin")
    plt.colorbar(im_auc, ax=ax3, label="AUC")
    plt.tight_layout()
    save_fig(fig3, "htag_auc_heatmap")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(9, 7))
    positive_counts = count_grid[count_grid > 0]
    vmin_counts = max(1, positive_counts.min()) if positive_counts.size > 0 else 1
    vmax_counts = max(1, count_grid.max())
    im_cnt = ax4.imshow(
        count_grid,
        aspect="auto",
        origin="lower",
        norm=mcolors.LogNorm(vmin=vmin_counts, vmax=vmax_counts),
    )
    for i_pt_hm in range(n_pt):
        for j_eta_hm in range(n_eta):
            ax4.text(j_eta_hm, i_pt_hm, f"{count_grid[i_pt_hm, j_eta_hm]:,}", ha="center", va="center", fontsize=10, color="white")
    ax4.set_xticks(range(n_eta))
    ax4.set_xticklabels(eta_labels_hm, rotation=30, fontsize=10)
    ax4.set_yticks(range(n_pt))
    ax4.set_yticklabels(pt_labels_hm, fontsize=10)
    ax4.set_xlabel(r"$|\eta|$ bin")
    ax4.set_ylabel(r"$p_T$ bin [GeV]")
    ax4.set_title("Jet count per bin (signal + background)")
    plt.colorbar(im_cnt, ax=ax4, label="Count")
    plt.tight_layout()
    save_fig(fig4, "htag_count_heatmap")
    plt.close(fig4)

    wp_names = ["Tight", "Medium", "Loose"]
    for wp_idx, wp_name in enumerate(wp_names):
        fig_eff, ax_eff = plt.subplots(figsize=(9, 7))
        thresh = htag_wps[wp_idx]
        eff_grid = np.full((n_pt, n_eta), np.nan)
        rej_grid = np.full((n_pt, n_eta), np.nan)

        for i_pt_hm, (pt_low, pt_high) in enumerate(pt_ranges_hm):
            for j_eta_hm, (eta_low, eta_high) in enumerate(eta_ranges_hm):
                mask_bin = (
                    (htag_pt >= pt_low)
                    & (htag_pt < pt_high)
                    & (np.abs(htag_eta) >= eta_low)
                    & (np.abs(htag_eta) < eta_high)
                )
                bin_y = htag_labels[mask_bin]
                bin_s = htag_scores[mask_bin]
                bin_w = htag_weights[mask_bin]

                sig_mask_bin = bin_y == 1
                bkg_mask_bin = bin_y == 0
                if sig_mask_bin.sum() > 0:
                    eff_grid[i_pt_hm, j_eta_hm] = np.mean(bin_s[sig_mask_bin] >= thresh)
                if bkg_mask_bin.sum() > 0:
                    bkg_pass = np.sum(bin_w[bkg_mask_bin] * (bin_s[bkg_mask_bin] >= thresh))
                    bkg_total = np.sum(bin_w[bkg_mask_bin])
                    bkg_eff = bkg_pass / bkg_total if bkg_total > 0 else 0.0
                    rej_grid[i_pt_hm, j_eta_hm] = 1.0 / bkg_eff if bkg_eff > 0 else np.inf

        im_eff = ax_eff.imshow(eff_grid, aspect="auto", origin="lower", vmin=0, vmax=1)
        for i_pt_hm in range(n_pt):
            for j_eta_hm in range(n_eta):
                vv = eff_grid[i_pt_hm, j_eta_hm]
                if np.isnan(vv):
                    txt = "-"
                else:
                    rej = rej_grid[i_pt_hm, j_eta_hm]
                    rej_txt = f"\n1/FPR={rej:.0f}" if np.isfinite(rej) and rej < 1e6 else ""
                    txt = f"{vv:.2f}{rej_txt}"
                ax_eff.text(j_eta_hm, i_pt_hm, txt, ha="center", va="center", fontsize=9, color="white")

        ax_eff.set_xticks(range(n_eta))
        ax_eff.set_xticklabels(eta_labels_hm, rotation=30, fontsize=10)
        ax_eff.set_yticks(range(n_pt))
        ax_eff.set_yticklabels(pt_labels_hm, fontsize=10)
        ax_eff.set_xlabel(r"$|\eta|$ bin")
        ax_eff.set_ylabel(r"$p_T$ bin [GeV]")
        ax_eff.set_title(f"H-Tag Signal Efficiency - {wp_name} (thresh={thresh:.4f})")
        plt.colorbar(im_eff, ax=ax_eff, label="Signal Efficiency")
        plt.tight_layout()
        save_fig(fig_eff, f"htag_efficiency_heatmap_{wp_name}")
        plt.close(fig_eff)

    # AK8 H-tag di-Higgs reconstruction
    import fastjet
    from data_pipeline.make_particle_dataset import cluster_candidates
    from data_pipeline.root_loading import (
        load_and_prepare_data,
        select_gen_higgs,
        one_hot_encode_l1_puppi,
    )
    from evaluation.dihiggs import compute_significance_at_luminosity
    from evaluation.luminosity import signal_weight, scale_qcd_weights_raw

    apply_pt_correction_htag = True
    HTAG_WP_SELECTION = "medium"
    htag_wp_index = {"tight": 0, "medium": 1, "loose": 2}[HTAG_WP_SELECTION]
    HTAG_THRESHOLD = htag_wps[htag_wp_index]
    AK8_DIST_PARAM = 0.8
    N_CONSTITUENTS_AK8 = n_constituents_model if n_constituents_model is not None else 128

    dataset_used_htag = config_part.get("training", {}).get("data", {}).get("use_dataset", "pf")
    if dataset_used_htag == "pf":
        collection_key_htag = "l1extpf"
    elif dataset_used_htag == "puppi":
        collection_key_htag = "l1extpuppi"
    else:
        collection_key_htag = "l1barrelextpf"
    collection_name_htag = config[collection_key_htag]["collection_name"]

    print("\n" + "=" * 60)
    print("Processing AK8 H-tag di-Higgs reconstruction...")
    print("=" * 60)

    def cluster_and_score_ak8(
        events,
        cfg,
        collection_key,
        model,
        device,
        config_part,
        n_constituents,
        apply_pt_correction=True,
    ):
        clustered_jets = cluster_candidates(events, cfg, collection_key, dist_param=AK8_DIST_PARAM)
        sorted_indices = ak.argsort(clustered_jets.pt, axis=1, ascending=False)
        l1_clustered = clustered_jets[sorted_indices]
        matched_cands = l1_clustered.constituents
        const_pt_sort = ak.argsort(matched_cands.pt, axis=2, ascending=False)
        matched_cands = matched_cands[const_pt_sort]

        j_pt = l1_clustered.pt[:, :, None]
        j_eta = l1_clustered.eta[:, :, None]
        j_phi = l1_clustered.phi[:, :, None]

        m_pt = matched_cands.vector.pt
        m_eta = matched_cands.vector.eta
        m_phi = matched_cands.vector.phi
        m_mass = matched_cands.vector.mass
        m_dxy = matched_cands.dxy
        m_z0 = matched_cands.z0
        m_charge = matched_cands.charge
        m_w = matched_cands.puppiWeight
        m_id = matched_cands.id

        log_pt_rel = np.log(np.maximum(m_pt, 1e-3) / np.maximum(j_pt, 1e-3))
        deta = m_eta - j_eta
        dphi = m_phi - j_phi
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        log_dr = np.log(np.maximum(np.sqrt(deta**2 + dphi**2), 1e-3))

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

        n_jets_per_event = ak.num(l1_clustered, axis=1)
        n_actual_constituents = ak.num(matched_cands, axis=2)
        n_actual_flat = ak.to_numpy(ak.flatten(n_actual_constituents, axis=1))

        x_ini = np.stack([ak.to_numpy(ak.flatten(f, axis=1)) for f in feature_list], axis=-1)
        flat_ids = x_ini[..., -1]
        one_hot_ids = one_hot_encode_l1_puppi(flat_ids, n_classes=5)
        X_feat = np.concatenate([x_ini[..., :-1], one_hot_ids], axis=-1)

        particle_mask = np.zeros((X_feat.shape[0], n_constituents), dtype=bool)
        for i_evt in range(X_feat.shape[0]):
            n_real = min(n_actual_flat[i_evt], n_constituents)
            particle_mask[i_evt, :n_real] = True

        const_vecs = vector.array(
            {
                "pt": x_ini[:, :, 1],
                "eta": x_ini[:, :, 2],
                "phi": x_ini[:, :, 3],
                "mass": x_ini[:, :, 0],
            }
        )
        jet_4v = const_vecs.sum(axis=1)
        flat_jet_pt = np.array(jet_4v.pt)
        flat_jet_eta = np.array(jet_4v.eta)
        flat_jet_phi = np.array(jet_4v.phi)
        flat_jet_mass = np.array(jet_4v.mass)

        batch_size = config_part.get("training", {}).get("batch_size", 512)
        all_scores, all_reg = [], []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(X_feat), batch_size):
                end = min(start + batch_size, len(X_feat))
                xb = torch.tensor(X_feat[start:end], dtype=torch.float32).to(device)
                mb = torch.tensor(particle_mask[start:end], dtype=torch.bool).to(device)
                out = model(xb, particle_mask=mb)
                scores = torch.nn.functional.sigmoid(out["classification"]).squeeze().cpu().numpy()
                all_scores.append(scores)
                if "pt" in out:
                    all_reg.append(out["pt"].squeeze().cpu().numpy())
                del xb, mb, out

        all_scores = np.concatenate(all_scores)
        has_reg = len(all_reg) > 0
        all_reg = np.concatenate(all_reg) if has_reg else None

        corrected_pt = flat_jet_pt * all_reg if (has_reg and apply_pt_correction) else flat_jet_pt
        corr_vecs = vector.array(
            {
                "pt": corrected_pt,
                "eta": flat_jet_eta,
                "phi": flat_jet_phi,
                "mass": flat_jet_mass * (corrected_pt / (flat_jet_pt + 1e-9)),
            }
        )

        n_jets_np = ak.to_numpy(n_jets_per_event)
        cumulative = np.concatenate([[0], np.cumsum(n_jets_np)])
        evt_pts, evt_etas, evt_phis, evt_masses, evt_scores = [], [], [], [], []
        for i_evt in range(len(n_jets_np)):
            s, e = cumulative[i_evt], cumulative[i_evt + 1]
            evt_pts.append(corr_vecs.pt[s:e])
            evt_etas.append(corr_vecs.eta[s:e])
            evt_phis.append(corr_vecs.phi[s:e])
            evt_masses.append(corr_vecs.mass[s:e])
            evt_scores.append(all_scores[s:e])

        scored_jets = ak.zip(
            {
                "pt": ak.Array(evt_pts),
                "eta": ak.Array(evt_etas),
                "phi": ak.Array(evt_phis),
                "mass": ak.Array(evt_masses),
                "htag_score": ak.Array(evt_scores),
            }
        )
        scored_jets["vector"] = ak.zip(
            {
                "pt": scored_jets.pt,
                "eta": scored_jets.eta,
                "phi": scored_jets.phi,
                "mass": scored_jets.mass,
            },
            with_name="Momentum4D",
        )
        flush_memory()
        return scored_jets, has_reg

    root_data_pattern = config["file_pattern"]
    SIGNAL_CHUNK_SIZE = 20000
    max_signal_events = min(config.get("max_events", 200000), 200000)

    scored_jets_chunks_htag = []
    gen_higgs_chunks = []
    offset = 0
    events_remaining = max_signal_events

    while events_remaining > 0:
        chunk_size = min(SIGNAL_CHUNK_SIZE, events_remaining)
        chunk_events = load_and_prepare_data(
            root_data_pattern,
            config["tree_name"],
            [collection_name_htag, "GenPart"],
            max_events=chunk_size,
            correct_pt=False,
            CONFIG=config,
            entry_start=offset,
        )
        n_loaded = len(chunk_events)
        if n_loaded == 0:
            break
        events_remaining -= n_loaded
        offset += n_loaded

        chunk_scored, has_reg_htag = cluster_and_score_ak8(
            chunk_events,
            config,
            collection_key_htag,
            model,
            device,
            config_part,
            N_CONSTITUENTS_AK8,
            apply_pt_correction_htag,
        )
        scored_jets_chunks_htag.append(chunk_scored)
        gen_higgs_chunks.append(select_gen_higgs(chunk_events))
        del chunk_events, chunk_scored
        flush_memory()

    if scored_jets_chunks_htag:
        scored_jets_htag = ak.concatenate(scored_jets_chunks_htag)
        gen_higgs_all = ak.concatenate(gen_higgs_chunks)
    else:
        scored_jets_htag = ak.Array([])
        gen_higgs_all = ak.Array([])

    jets_htag_sorted = scored_jets_htag[ak.argsort(scored_jets_htag.htag_score, ascending=False)]
    jets_htag_pass = jets_htag_sorted[jets_htag_sorted.htag_score > HTAG_THRESHOLD]
    has_2_tagged = ak.num(jets_htag_pass) >= 2
    sig_jets_2 = jets_htag_pass[has_2_tagged][:, :2]

    gen_higgs_for_match = gen_higgs_all[has_2_tagged]
    if len(sig_jets_2) > 0:
        dr_reco_h = sig_jets_2[:, :, None].vector.deltaR(gen_higgs_for_match[:, None, :].vector)
        idx_gen_for_reco_h = ak.argmin(dr_reco_h, axis=2)
        min_dr_reco_h = ak.fill_none(ak.min(dr_reco_h, axis=2), np.inf)

        dr_gen_h = gen_higgs_for_match[:, :, None].vector.deltaR(sig_jets_2[:, None, :].vector)
        idx_reco_for_gen_h = ak.argmin(dr_gen_h, axis=2)
        back_check_h = idx_reco_for_gen_h[idx_gen_for_reco_h]
        reco_idx_h = ak.local_index(sig_jets_2, axis=1)
        pure_mask_h = (ak.fill_none(back_check_h, -1) == reco_idx_h) & (min_dr_reco_h < AK8_DIST_PARAM)
        signal_mask_evt_htag = ak.sum(pure_mask_h, axis=1) == 2
    else:
        signal_mask_evt_htag = ak.Array([])

    n_signal_htag = int(ak.sum(signal_mask_evt_htag)) if len(signal_mask_evt_htag) > 0 else 0
    n_total_htag = int(ak.sum(has_2_tagged)) if len(has_2_tagged) > 0 else 0

    sig_jets_htag_2 = sig_jets_2[signal_mask_evt_htag][:, :2] if n_signal_htag > 0 else ak.Array([])
    if n_signal_htag > 0:
        h1_vec = sig_jets_htag_2[:, 0].vector
        h2_vec = sig_jets_htag_2[:, 1].vector
        is_lead = h1_vec.pt >= h2_vec.pt
        sig_lead_htag = ak.where(is_lead, h1_vec, h2_vec)
        sig_sub_htag = ak.where(is_lead, h2_vec, h1_vec)
        sig_hh_htag = h1_vec + h2_vec
    else:
        sig_lead_htag = sig_sub_htag = sig_hh_htag = ak.Array([])

    QCD_CHUNK_SIZE_HTAG = 5000
    qcd_config_htag = config["QCD_background"]
    all_qcd_lead_htag, all_qcd_sub_htag, all_qcd_hh_htag = [], [], []
    all_qcd_weights_htag_list = []
    n_qcd_total_htag = 0
    n_qcd_events_processed_htag = 0

    for bin_name, bin_cfg in qcd_config_htag.items():
        qcd_file_pattern = bin_cfg["file_pattern"]
        max_events_bin = min(bin_cfg.get("max_events", 20000), 20000)
        qcd_cfg_htag = dict(config)
        qcd_cfg_htag["file_pattern"] = qcd_file_pattern
        qcd_cfg_htag["tree_name"] = bin_cfg["tree_name"]

        offset = 0
        events_remaining = max_events_bin
        while events_remaining > 0:
            chunk_size = min(QCD_CHUNK_SIZE_HTAG, events_remaining)
            qcd_events = load_and_prepare_data(
                qcd_file_pattern,
                bin_cfg["tree_name"],
                [collection_name_htag],
                max_events=chunk_size,
                correct_pt=False,
                CONFIG=qcd_cfg_htag,
                entry_start=offset,
            )
            n_loaded = len(qcd_events)
            if n_loaded == 0:
                break
            n_qcd_events_processed_htag += n_loaded
            events_remaining -= n_loaded
            offset += n_loaded

            qcd_scored_htag, _ = cluster_and_score_ak8(
                qcd_events,
                qcd_cfg_htag,
                collection_key_htag,
                model,
                device,
                config_part,
                N_CONSTITUENTS_AK8,
                apply_pt_correction_htag,
            )
            qcd_htag_sorted = qcd_scored_htag[ak.argsort(qcd_scored_htag.htag_score, ascending=False)]
            qcd_htag_pass = qcd_htag_sorted[qcd_htag_sorted.htag_score > HTAG_THRESHOLD]
            has_2_tagged_qcd = ak.num(qcd_htag_pass) >= 2
            qcd_2jets = qcd_htag_pass[has_2_tagged_qcd][:, :2]

            n_events_chunk = int(ak.sum(has_2_tagged_qcd))
            if n_events_chunk > 0:
                q_h1 = qcd_2jets[:, 0].vector
                q_h2 = qcd_2jets[:, 1].vector
                q_is_lead = q_h1.pt >= q_h2.pt
                q_lead = ak.where(q_is_lead, q_h1, q_h2)
                q_sub = ak.where(q_is_lead, q_h2, q_h1)
                q_hh = q_h1 + q_h2
                all_qcd_lead_htag.append(q_lead)
                all_qcd_sub_htag.append(q_sub)
                all_qcd_hh_htag.append(q_hh)
                all_qcd_weights_htag_list.append(np.full(n_events_chunk, bin_cfg["weight"], dtype=np.float64))
                n_qcd_total_htag += n_events_chunk

            del qcd_events, qcd_scored_htag, qcd_htag_sorted, qcd_htag_pass, qcd_2jets
            flush_memory()

    if n_qcd_total_htag > 0:
        qcd_lead_htag = ak.concatenate(all_qcd_lead_htag)
        qcd_sub_htag = ak.concatenate(all_qcd_sub_htag)
        qcd_hh_htag = ak.concatenate(all_qcd_hh_htag)
        qcd_weights_htag = np.concatenate(all_qcd_weights_htag_list)
    else:
        qcd_lead_htag = qcd_sub_htag = qcd_hh_htag = ak.Array([])
        qcd_weights_htag = np.array([], dtype=np.float64)

    sigma_to_ngen_htag = {bin_cfg["weight"]: bin_cfg["n_gen"] for bin_cfg in qcd_config_htag.values()}
    htag_dihiggs_result = {
        "label": f"ParT H-Tag ({HTAG_WP_SELECTION})",
        "n_total": n_total_htag,
        "n_signal": n_signal_htag,
        "n_qcd": n_qcd_total_htag,
        "sig_lead": sig_lead_htag,
        "sig_sub": sig_sub_htag,
        "sig_hh": sig_hh_htag,
        "qcd_lead": qcd_lead_htag,
        "qcd_sub": qcd_sub_htag,
        "qcd_hh": qcd_hh_htag,
        "qcd_weights": qcd_weights_htag,
        "sigma_to_ngen": sigma_to_ngen_htag,
        "collection_key": collection_key_htag,
        "wp": HTAG_WP_SELECTION,
        "threshold": HTAG_THRESHOLD,
        "has_regression": has_reg_htag,
    }

    print(
        f"AK8 H-tag reconstruction complete: signal={n_signal_htag}, "
        f"qcd={n_qcd_total_htag}, qcd_processed={n_qcd_events_processed_htag}"
    )

    # AK8 mass plots and significance
    res_h = htag_dihiggs_result
    label_h = res_h["label"]
    color_h = "teal"
    qcd_w_h = res_h.get("qcd_weights", np.ones(res_h["n_qcd"]))
    sig_window_h = (90, 160)
    sig_window_hh = (250, 550)
    bins_mh = np.linspace(0, 300, 61)
    bins_mhh = np.linspace(200, 800, 61)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    if res_h["n_signal"] > 0:
        ax.hist(ak.to_numpy(res_h["sig_lead"].mass), bins=bins_mh, histtype="stepfilled", alpha=0.3, color=color_h, label=f'Signal ({res_h["n_signal"]})', density=True)
        ax.hist(ak.to_numpy(res_h["sig_lead"].mass), bins=bins_mh, histtype="step", linewidth=2, color=color_h, density=True)
    if res_h["n_qcd"] > 0:
        ax.hist(ak.to_numpy(res_h["qcd_lead"].mass), bins=bins_mh, histtype="step", linewidth=2, color="grey", linestyle="--", label=f'QCD ({res_h["n_qcd"]} events)', density=True)
    ax.axvline(125, color="green", linestyle=":", linewidth=1.5)
    ax.axvspan(*sig_window_h, alpha=0.05, color="green")
    ax.set_xlabel("Leading $m_H$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title(f"{label_h} - Leading Higgs (AK8)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, "htag_dihiggs_mass_leading_full")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    if res_h["n_signal"] > 0:
        ax.hist(ak.to_numpy(res_h["sig_sub"].mass), bins=bins_mh, histtype="stepfilled", alpha=0.3, color=color_h, label="Signal", density=True)
        ax.hist(ak.to_numpy(res_h["sig_sub"].mass), bins=bins_mh, histtype="step", linewidth=2, color=color_h, density=True)
    if res_h["n_qcd"] > 0:
        ax.hist(ak.to_numpy(res_h["qcd_sub"].mass), bins=bins_mh, histtype="step", linewidth=2, color="grey", linestyle="--", label="QCD", density=True)
    ax.axvline(125, color="green", linestyle=":", linewidth=1.5)
    ax.axvspan(*sig_window_h, alpha=0.05, color="green")
    ax.set_xlabel("Subleading $m_H$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title(f"{label_h} - Subleading Higgs (AK8)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, "htag_dihiggs_mass_subleading_full")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    if res_h["n_signal"] > 0:
        ax.hist(ak.to_numpy(res_h["sig_hh"].mass), bins=bins_mhh, histtype="stepfilled", alpha=0.3, color=color_h, label="Signal", density=True)
        ax.hist(ak.to_numpy(res_h["sig_hh"].mass), bins=bins_mhh, histtype="step", linewidth=2, color=color_h, density=True)
    if res_h["n_qcd"] > 0:
        ax.hist(ak.to_numpy(res_h["qcd_hh"].mass), bins=bins_mhh, histtype="step", linewidth=2, color="grey", linestyle="--", label="QCD", density=True)
    ax.axvspan(*sig_window_hh, alpha=0.05, color="green")
    ax.set_xlabel("$m_{HH}$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_title(f"{label_h} - Di-Higgs Mass (AK8)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, "htag_dihiggs_mass_mhh_full")
    plt.close(fig)

    if res_h["n_signal"] > 0 and res_h["n_qcd"] > 0:
        sig_mh1_h = ak.to_numpy(res_h["sig_lead"].mass)
        sig_mh2_h = ak.to_numpy(res_h["sig_sub"].mass)
        bkg_mh1_h = ak.to_numpy(res_h["qcd_lead"].mass)
        bkg_mh2_h = ak.to_numpy(res_h["qcd_sub"].mass)
        result_mhh_h = compute_significance_at_luminosity(
            sig_mh1_h,
            sig_mh2_h,
            bkg_mh1_h,
            bkg_mh2_h,
            bkg_raw_weights=qcd_w_h,
            sigma_to_ngen=sigma_to_ngen_htag,
            n_gen_signal=N_GEN_SIGNAL,
            luminosity_fb=LUMINOSITY_FB,
            signal_xsec_pb=SIGNAL_XSEC_PB,
            region="rectangular",
            rect_window=sig_window_hh,
        )
        print(
            f"AK8 mHH-window significance: S={result_mhh_h['S']:.1f}, "
            f"B={result_mhh_h['B']:.3e}, Z={result_mhh_h['significance']:.4f}"
        )

    # r_HH AK8 region diagnostics
    if res_h["n_signal"] > 0 and res_h["n_qcd"] > 0:
        MH_CENTER = 125.0
        R_HH_CUT_HTAG = 30.0

        sig_lead_m = ak.to_numpy(res_h["sig_lead"].mass)
        sig_sub_m = ak.to_numpy(res_h["sig_sub"].mass)
        sig_hh_m = ak.to_numpy(res_h["sig_hh"].mass)
        bkg_lead_m = ak.to_numpy(res_h["qcd_lead"].mass)
        bkg_sub_m = ak.to_numpy(res_h["qcd_sub"].mass)
        bkg_hh_m = ak.to_numpy(res_h["qcd_hh"].mass)

        sig_rhh = np.sqrt((sig_lead_m - MH_CENTER) ** 2 + (sig_sub_m - MH_CENTER) ** 2)
        bkg_rhh = np.sqrt((bkg_lead_m - MH_CENTER) ** 2 + (bkg_sub_m - MH_CENTER) ** 2)
        sig_mask_rhh = sig_rhh < R_HH_CUT_HTAG
        bkg_mask_rhh = bkg_rhh < R_HH_CUT_HTAG
        sig_mask_mhh = (sig_hh_m >= sig_window_hh[0]) & (sig_hh_m <= sig_window_hh[1])
        bkg_mask_mhh = (bkg_hh_m >= sig_window_hh[0]) & (bkg_hh_m <= sig_window_hh[1])

        result_rhh_h = compute_significance_at_luminosity(
            sig_lead_m,
            sig_sub_m,
            bkg_lead_m,
            bkg_sub_m,
            bkg_raw_weights=qcd_w_h,
            sigma_to_ngen=sigma_to_ngen_htag,
            n_gen_signal=N_GEN_SIGNAL,
            luminosity_fb=LUMINOSITY_FB,
            signal_xsec_pb=SIGNAL_XSEC_PB,
            region="circular",
            r_hh_cut=R_HH_CUT_HTAG,
        )

        _sw_h = signal_weight(len(sig_lead_m), LUMINOSITY_FB, SIGNAL_XSEC_PB, N_GEN_SIGNAL)
        _bw_h = scale_qcd_weights_raw(qcd_w_h, sigma_to_ngen_htag, LUMINOSITY_FB)
        S_comb = float(np.sum(_sw_h[sig_mask_rhh & sig_mask_mhh]))
        B_comb = float(np.sum(_bw_h[bkg_mask_rhh & bkg_mask_mhh]))
        Z_comb = S_comb / np.sqrt(S_comb + B_comb) if (S_comb + B_comb) > 0 else 0.0

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111)
        ax.scatter(bkg_lead_m, bkg_sub_m, s=2, alpha=0.20, color="grey", label="QCD")
        ax.scatter(sig_lead_m, sig_sub_m, s=3, alpha=0.35, color="teal", label="Signal")
        ax.axvline(MH_CENTER, color="green", linestyle=":", alpha=0.6)
        ax.axhline(MH_CENTER, color="green", linestyle=":", alpha=0.6)
        circle = plt.Circle((MH_CENTER, MH_CENTER), R_HH_CUT_HTAG, fill=False, color="crimson", lw=2, label=f"r_HH < {R_HH_CUT_HTAG:g}")
        ax.add_patch(circle)
        ax.set_xlabel("Leading $m_H$ [GeV]")
        ax.set_ylabel("Subleading $m_H$ [GeV]")
        ax.set_title(f"{label_h} - AK8 $m_{{H_1}}$ vs $m_{{H_2}}$ with $r_{{HH}}$ SR")
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        ax.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        save_fig(fig, "htag_dihiggs_rhh_signal_region_scatter")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        bins_rhh = np.linspace(0, 200, 81)
        ax.hist(sig_rhh, bins=bins_rhh, histtype="stepfilled", alpha=0.25, color="teal", density=True, label="Signal")
        ax.hist(sig_rhh, bins=bins_rhh, histtype="step", linewidth=2, color="teal", density=True)
        ax.hist(bkg_rhh, bins=bins_rhh, weights=qcd_w_h, histtype="step", linewidth=2, color="grey", linestyle="--", density=True, label="QCD (weighted)")
        ax.axvline(R_HH_CUT_HTAG, color="crimson", linestyle="-", linewidth=2, label=f"$r_{{HH}}$ cut = {R_HH_CUT_HTAG:g}")
        ax.set_xlabel(r"$r_{HH}$ [GeV]")
        ax.set_ylabel("Density")
        ax.set_title(f"{label_h} - $r_{{HH}}$ distribution (AK8)")
        ax.legend(fontsize=10)
        plt.tight_layout()
        save_fig(fig, "htag_dihiggs_rhh_distribution")
        plt.close(fig)

        print(
            f"AK8 r_HH significance: S={result_rhh_h['S']:.1f}, B={result_rhh_h['B']:.3e}, "
            f"Z={result_rhh_h['significance']:.4f}; combined(r_HH+mHH): Z={Z_comb:.4f}"
        )

    print("\nAll analysis complete!")


if __name__ == "__main__":
    main()
