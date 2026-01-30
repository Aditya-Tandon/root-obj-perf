import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import wandb
import argparse
from parT import ParticleTransformer  # Import the new model
from parT_helpers import extract_wandb_run_id, get_model_ckpt

torch.manual_seed(42)
np.random.seed(42)


# --- Dataset Class ---
class L1JetDataset(Dataset):
    def __init__(self, filepath):
        print(f"Loading data from {filepath}...")
        data = np.load(filepath)
        # X: (N, n_constituents, Features)
        self.X = torch.from_numpy(data["x"]).float()
        self.y = torch.from_numpy(data["y"]).float().unsqueeze(1)

        # Load particle mask if available, otherwise infer from non-zero energy
        if "particle_mask" in data.files:
            self.mask = torch.from_numpy(data["particle_mask"]).bool()
            print("Loaded particle mask from dataset")
        else:
            # Fallback: infer mask from non-zero particles (E != 0)
            self.mask = self.X[..., 0] != 0
            print("No particle_mask in dataset, inferring from non-zero energy")

        if "weights" in data.files:
            self.weights = torch.from_numpy(data["weights"]).float().unsqueeze(1)
        else:
            self.weights = torch.ones_like(self.y)

        print(f"Data loaded: {self.X.shape} samples")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx], self.weights[idx]


# --- Stratified Split Helper ---
def stratified_split(dataset, val_split, num_classes, random_state=42, verbose=True):
    """
    Perform a stratified train/validation split on a dataset.

    Args:
        dataset: Dataset object with a .y attribute containing labels
        val_split: Fraction of data to use for validation (0-1)
        num_classes: Number of classes (1 for binary classification)
        random_state: Random seed for reproducibility
        verbose: Whether to print class distribution statistics

    Returns:
        train_ds: Training subset
        val_ds: Validation subset
        train_indices: Indices of training samples
        val_indices: Indices of validation samples
        stratify_labels: Labels used for stratification
    """
    # Get labels for stratification
    # For binary classification (num_classes=1), use the binary labels directly
    # For multi-class, the labels should already be class indices
    stratify_labels = dataset.y.squeeze().numpy().astype(int)

    # Create indices and perform stratified split
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        stratify=stratify_labels,
        random_state=random_state,
    )

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    if verbose:
        # Print class distribution for verification
        train_labels = stratify_labels[train_indices]
        val_labels = stratify_labels[val_indices]
        print(f"Stratified split complete:")
        print(f"  Train set: {len(train_ds)} samples")
        print(f"  Val set: {len(val_ds)} samples")
        for c in range(max(num_classes, 2)):
            train_count = np.sum(train_labels == c)
            val_count = np.sum(val_labels == c)
            print(
                f"  Class {c}: Train={train_count} ({100*train_count/len(train_ds):.1f}%), "
                f"Val={val_count} ({100*val_count/len(val_ds):.1f}%)"
            )

    return train_ds, val_ds, train_indices, val_indices, stratify_labels


# --- Training Helper ---
def run_training(config_path):
    # 1. Load Configuration
    with open(config_path, "r") as f:
        cfg = json.load(f)

    restart = cfg["training"]["restart"]
    # 2. Initialize WandB
    if restart:
        run_id = run_id = extract_wandb_run_id(cfg["training"]["wandb_run_path"])
    else:
        run_id = None
    wandb.init(
        project="L1-BTagging-ParT",
        config=cfg,
        name=cfg["exp_name"],
        id=run_id,
        resume="allow",
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 3. Prepare Data
    dataset = L1JetDataset(cfg["data_path"])
    num_classes = cfg["model"]["num_classes"]

    # Perform stratified split
    train_ds, val_ds, train_indices, val_indices, stratify_labels = stratified_split(
        dataset=dataset,
        val_split=cfg["training"]["val_split"],
        num_classes=num_classes,
        random_state=42,
        verbose=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    print("Data loaders prepared.")

    # 4. Initialize Particle Transformer
    model = ParticleTransformer(
        input_dim=cfg["input_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        num_cls_layers=cfg["model"]["num_cls_layers"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)
    print("Model loaded.")

    # 5. Optimiser & Scheduler
    optimiser = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        eps=1e-10,
    )
    print("Optimiser initialised.")

    if restart:
        # if config["wandb"]["load_from_artifact"]:
        artifact_path = f"{cfg['wandb']['entity']}/{cfg['wandb']['project']}/{cfg['wandb']['artifact_name']}:{cfg['wandb']['ckpt_type']}"
        artifact = wandb.use_artifact(
            artifact_path,
            type="model",
        )
        artifact_dir = artifact.download()
        print(f"Model artifact downloaded from W&B: {artifact_dir}")
        if os.path.exists(artifact_dir):
            checkpoint_dir = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
            checkpoint = torch.load(
                checkpoint_dir, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
            print("Model, and optimiser state loaded from W&B artifact")
        else:
            print(
                f"Artifact directory {artifact_dir} does not exist. Skipping loading from artifact."
            )
        ckpt = get_model_ckpt(
            run_id=run_id,
            ckpt_name=cfg["training"]["ckpt_lo_load"],
            wandb_dir=cfg["training"]["wandb_dir"],
        )
        model.load_state_dict(ckpt)
        print("Model state dict loaded.")

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    wandb.watch(model, log="gradients")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["training"]["epochs"] * 1.5
    )
    # Calculate pos_weight from training subset labels
    train_labels_tensor = torch.from_numpy(stratify_labels[train_indices]).float()
    num_neg = torch.sum(train_labels_tensor == 0)
    num_pos = torch.sum(train_labels_tensor == 1)
    pos_weight = num_neg / num_pos
    pos_weight = torch.tensor(pos_weight).to(device)
    # Use reduction='none' to apply per-sample kinematic weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    print(f"Criterion initialised with pos_weight: {pos_weight.item():.4f}")

    # # AMP Scaler for faster training
    # scaler = torch.amp.GradScaler(enabled=cfg["training"]["use_amp"])

    # --- Training Loop ---
    best_auc = 0.0
    start_epoch = 0 if restart is False else cfg["training"]["last_epoch_in_prev_run"]
    num_epochs = start_epoch + cfg["training"]["epochs"]
    print(f"Starting training from epoch {start_epoch} for {num_epochs} epochs.")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0

        train_outputs = []
        train_labels = []
        for X_batch, y_batch, mask_batch, weights in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            weights = weights.to(device)

            optimiser.zero_grad()

            outputs = model(X_batch, particle_mask=mask_batch)
            # Apply per-sample kinematic weights to the loss
            per_sample_loss = criterion(outputs, y_batch)
            weighted_loss = per_sample_loss * weights
            loss = weighted_loss.mean()

            loss.backward()
            optimiser.step()

            total_loss += loss.item()

            train_outputs.append(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.append(y_batch.cpu().numpy())
        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        train_auc = roc_auc_score(
            np.concatenate(train_labels), np.concatenate(train_outputs)
        )

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch, mask_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                outputs = model(X_batch, particle_mask=mask_batch)
                # For validation, use unweighted loss (mean of per-sample losses)
                val_loss += criterion(outputs, y_batch).mean().item()

                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        auc_score = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds))

        scheduler.step()
        # Logging
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {auc_score:.4f}"
        )
        train_metrics = {
            "train_loss": avg_train_loss,
            "train_auc": train_auc,
        }
        val_metrics = {
            "val_loss": avg_val_loss,
            "val_auc": auc_score,
        }
        epoch_metrics = {
            "epoch": epoch + 1,
            **train_metrics,
            **val_metrics,
            "lr": optimiser.param_groups[0]["lr"],
        }
        wandb.log(epoch_metrics)

        if auc_score > best_auc:
            best_auc = auc_score
            torch.save(
                {
                    "config": cfg,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                    "val_auc": val_metrics["val_auc"],
                },
                "best_part_model.pth",
            )
            print(f"New best model saved with AUC: {best_auc:.4f}")
        torch.save(
            {
                "config": cfg,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["val_loss"],
                "val_auc": val_metrics["val_auc"],
            },
            f"part_model_epoch_{epoch+1}.pth",
        )

        periodic_artifact = wandb.Artifact(
            "periodic_model",
            type="model",
            description=f"Model checkpoint at epoch {epoch+1} with AUC: {auc_score:.4f}",
            metadata={"epoch": epoch + 1, "val_auc": auc_score, "config": cfg},
        )
        periodic_artifact.add_file(f"part_model_epoch_{epoch+1}.pth")
        wandb.log_artifact(periodic_artifact, aliases=[f"epoch_{epoch+1}"])

    print(f"Training Complete. Best AUC: {best_auc:.4f}")
    print("Saving final model")

    torch.save(
        {
            "config": cfg,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_metrics["val_loss"],
            "val_auc": val_metrics["val_auc"],
        },
        f"final_model.pth",
    )
    final_artifact = wandb.Artifact(
        "final_model",
        type="model",
        description=f"Final model after {num_epochs} epochs with AUC: {auc_score:.4f}",
        metadata={"epoch": num_epochs, "val_auc": auc_score, "config": cfg},
    )
    final_artifact.add_file("final_model.pth")
    wandb.log_artifact(final_artifact, aliases=["final"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_part.json")
    args = parser.parse_args()
    run_training(args.config)
