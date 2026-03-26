"""
Training script for event-level ParticleTransformer binary classifier (HH→4b vs QCD).

Uses the same ParticleTransformer model as train_part.py but with event-level
inputs: all L1ExtPuppi particles per event (up to 128) instead of 16 jet
constituents. No regression heads — classification only.

Usage:
    python train_event.py --config config_event.json
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import wandb
import argparse

from model.parT import ParticleTransformer
from wandb_utils import extract_wandb_run_id
from data_pipeline.datasets import StratifiedJetDataset, precollated_collate
from data_pipeline.splitting import stratified_split
from model.warmup_cosine_lr import WarmupCosineSchedulerWithRestarts

torch.manual_seed(42)
np.random.seed(42)

VAL_EVERY_N_EPOCHS = 10
torch.backends.cuda.matmul.allow_tf32 = True


def run_training(cfg):
    """Run the full event-level training loop.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (already loaded and overridden).
    """

    # W&B init
    restart = cfg["training"]["restart"]
    run_id = (
        extract_wandb_run_id(cfg["training"]["wandb_run_path"]) if restart else None
    )
    wandb.init(
        project=cfg["wandb"]["project"],
        config=cfg,
        name=cfg["exp_name"],
        id=run_id,
        resume="allow",
    )

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data
    dataset = StratifiedJetDataset(cfg["training"]["data"]["data_path"])
    train_ds, val_ds, train_indices, val_indices, train_labels = stratified_split(
        dataset, cfg["training"]["data"]["val_split"], num_classes=1
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["data"]["num_workers"],
        collate_fn=precollated_collate,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["training"]["data"]["num_workers"],
        collate_fn=precollated_collate,
        pin_memory=True,
        persistent_workers=True,
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Model
    model = ParticleTransformer(
        input_dim=cfg["input_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        num_pairwise_feat=cfg["model"].get("num_pairwise_feat", 7),
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        num_cls_layers=cfg["model"]["num_cls_layers"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        use_batch_norm=cfg["model"].get("use_batch_norm", True),
        pt_regression=False,
        quantile_regression=False,
    ).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimiser & Scheduler
    optimiser = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["optimiser"]["lr"],
        weight_decay=cfg["training"]["optimiser"]["weight_decay"],
        eps=1e-10,
    )

    if restart:
        artifact_path = (
            f"{cfg['wandb']['entity']}/{cfg['wandb']['project']}/"
            f"{cfg['wandb']['artifact_name']}:{cfg['wandb']['ckpt_type']}"
        )
        artifact = wandb.use_artifact(artifact_path, type="model")
        artifact_dir = artifact.download()
        checkpoint_dir = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
        checkpoint = torch.load(checkpoint_dir, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        print("Loaded model & optimiser from W&B artifact")

    wandb.watch(model, log="gradients")

    scheduler = WarmupCosineSchedulerWithRestarts(
        optimiser,
        warmup_epochs=cfg["training"]["scheduler"]["warmup_epochs"],
        total_epochs=cfg["training"]["scheduler"]["total_epochs"],
        min_lr=cfg["training"]["scheduler"]["min_lr"],
        red_fac=cfg["training"]["scheduler"]["red_fac"],
        last_epoch=cfg["training"]["last_epoch_in_prev_run"] if restart else -1,
    )

    # Loss
    pos_weight = torch.tensor(cfg["training"]["pos_weight"], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    print(f"pos_weight: {pos_weight.item():.4f}")

    # AMP
    use_amp = cfg["training"].get("use_amp", False)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Training loop
    best_auc = 0.0
    start_epoch = cfg["training"]["last_epoch_in_prev_run"] if restart else 0
    num_epochs = start_epoch + cfg["training"]["epochs"]

    print(f"Training from epoch {start_epoch} to {num_epochs}")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_cls_loss_sum = 0.0
        train_outputs, train_labels_list, train_qcd_w = [], [], []
        n_train_batches = 0

        for X_batch, y_batch, mask_batch, weights, _, _, _, qcd_weights in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            weights = weights.to(device)

            optimiser.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(X_batch, particle_mask=mask_batch)
                cls_output = (
                    outputs["classification"] if isinstance(outputs, dict) else outputs
                )
                per_sample_loss = criterion(cls_output, y_batch)
                cls_loss = (per_sample_loss * weights).mean()

            scaler.scale(cls_loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            train_cls_loss_sum += cls_loss.item()
            n_train_batches += 1
            train_outputs.append(torch.sigmoid(cls_output).detach().cpu().numpy())
            train_labels_list.append(y_batch.cpu().numpy())
            train_qcd_w.append(qcd_weights.numpy())

        scheduler.step()

        # Train metrics
        train_metrics = {
            "train_loss": train_cls_loss_sum / n_train_batches,
            "train_auc": roc_auc_score(
                np.concatenate(train_labels_list),
                np.concatenate(train_outputs),
                sample_weight=np.concatenate(train_qcd_w),
            ),
        }

        # Validation
        model.eval()
        all_preds, all_labels, all_qcd_w = [], [], []
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch, mask_batch, _, _, _, _, qcd_weights in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(X_batch, particle_mask=mask_batch)
                    cls_output = (
                        outputs["classification"]
                        if isinstance(outputs, dict)
                        else outputs
                    )
                    val_loss_sum += criterion(cls_output, y_batch).mean().item()

                n_val_batches += 1
                all_preds.append(torch.sigmoid(cls_output).cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
                all_qcd_w.append(qcd_weights.numpy())

        auc_score = roc_auc_score(
            np.concatenate(all_labels),
            np.concatenate(all_preds),
            sample_weight=np.concatenate(all_qcd_w),
        )
        val_metrics = {
            "val_loss": val_loss_sum / n_val_batches,
            "val_auc": auc_score,
        }

        print(
            f"Epoch {epoch+1} | Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val AUC: {auc_score:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
                "lr": optimiser.param_groups[0]["lr"],
            }
        )

        # Checkpointing
        ckpt_dict = {
            "config": cfg,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_metrics["val_loss"],
            "val_auc": auc_score,
        }

        if auc_score > best_auc:
            best_auc = auc_score
            torch.save(ckpt_dict, f"best_event_model_{wandb.run.id}.pth")
            print(f"  New best model (AUC: {best_auc:.4f})")
            art = wandb.Artifact(
                "best_model",
                type="model",
                description=f"Best model AUC={best_auc:.4f} epoch={epoch+1}",
                metadata={"epoch": epoch + 1, "val_auc": best_auc, "config": cfg},
            )
            art.add_file(f"best_event_model_{wandb.run.id}.pth")
            wandb.log_artifact(art, aliases=["best"])

        if (epoch + 1) % cfg["training"]["log_freq"] == 0:
            torch.save(ckpt_dict, f"event_model_{wandb.run.id}_epoch_{epoch+1}.pth")
            art = wandb.Artifact(
                "periodic_model",
                type="model",
                description=f"Checkpoint epoch={epoch+1} AUC={auc_score:.4f}",
                metadata={"epoch": epoch + 1, "val_auc": auc_score, "config": cfg},
            )
            art.add_file(f"event_model_{wandb.run.id}_epoch_{epoch+1}.pth")
            wandb.log_artifact(art, aliases=[f"epoch_{epoch+1}"])

    # Final model
    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    torch.save(ckpt_dict, f"final_event_model_{wandb.run.id}.pth")
    art = wandb.Artifact(
        "final_model",
        type="model",
        description=f"Final model after {num_epochs} epochs AUC={auc_score:.4f}",
        metadata={"epoch": num_epochs, "val_auc": auc_score, "config": cfg},
    )
    art.add_file(f"final_event_model_{wandb.run.id}.pth")
    wandb.log_artifact(art, aliases=["final"])
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_event.json")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values: --set training.lr=1e-4",
    )
    parser.add_argument(
        "--exp-name", type=str, default=None, help="Override experiment name"
    )
    args = parser.parse_args()

    # Read config eagerly — file can be edited immediately after this line
    with open(args.config) as f:
        cfg = json.load(f)

    # Apply CLI overrides
    if args.exp_name:
        cfg["exp_name"] = args.exp_name
    for override in args.set:
        key, value = override.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        else:
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
        d[keys[-1]] = value

    run_training(cfg)
