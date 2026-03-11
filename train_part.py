import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score
import wandb
import argparse
from parT import ParticleTransformer  # Import the new model
from parT_helpers import extract_wandb_run_id, get_model_ckpt

from train_part_data import CombinedJetDataLoader
from warmup_cosine_lr import WarmupCosineSchedulerWithRestarts

torch.manual_seed(42)
np.random.seed(42)

REG_SCALE_FAC = 0.5
QREG_SCALE_FAC = 0.5
VAL_EVERY_N_EPOCHS = 10
torch.backends.cuda.matmul.allow_tf32 = True
print("TF32 matmul enabled for faster training on compatible GPUs.")

class QuantileLoss(nn.Module):
    def __init__(self, quantiles, reduction="sum"):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        self.reduction = reduction

    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            delta = target[:, 0] - preds[:, i]
            loss = torch.max(q * delta, (q - 1) * delta)
            losses.append(loss)
        loss = torch.sum(torch.stack(losses, dim=1), dim=-1)
        if self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "mean":
            return torch.mean(loss)
        return loss

# --- Training Helper ---
def run_training(config_path):
    # 1. Load Configuration
    with open(config_path, "r") as f:
        cfg = json.load(f)

    restart = cfg["training"]["restart"]
    # 2. Initialize WandB
    if restart:
        run_id = extract_wandb_run_id(cfg["training"]["wandb_run_path"])
    else:
        run_id = None
    wandb.init(
        project=cfg["wandb"]["project"],
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
    combined_loader = CombinedJetDataLoader(
        pf_data_path=cfg["training"]["data"]["pf_data_path"],
        puppi_data_path=cfg["training"]["data"]["puppi_data_path"],
        val_split=cfg["training"]["data"]["val_split"],
        batch_size=cfg["training"]["batch_size"],
        match_mode=cfg["training"]["data"]["match_mode"],
        num_workers=cfg["training"]["data"]["num_workers"],
        random_state=42,
        use_dataset=cfg["training"]["data"].get("use_dataset", "both"),
    )
    if cfg["training"]["data"]["use_dataset"] == "pf":
        print("\nUsing PF dataset for training.")
        (
            train_loader,
            train_indices,
            val_loader,
            val_indices,
            train_labels,
            val_labels,
        ) = combined_loader.get_pf_loaders(shuffle=True)
    elif cfg["training"]["data"]["use_dataset"] == "puppi":
        print("\nUsing PUPPI dataset for training.")
        (
            train_loader,
            train_indices,
            val_loader,
            val_indices,
            train_labels,
            val_labels,
        ) = combined_loader.get_puppi_loaders(shuffle=True)
    print(
        f"Data loaders prepared with {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples."
    )

    # 4. Initialize Particle Transformer
    pt_regression = cfg["model"].get("pt_regression", False)
    quantile_regression = cfg["model"].get("quantile_regression", False)

    model = ParticleTransformer(
        input_dim=cfg["input_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        num_cls_layers=cfg["model"]["num_cls_layers"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        pt_regression=pt_regression,
        quantile_regression=quantile_regression,
    ).to(device)
    print("Model loaded.")

    if pt_regression:
        print("PT regression head enabled in the model.")
    if quantile_regression:
        print("Quantile regression enabled in the model.")

    # 5. Optimiser & Scheduler
    optimiser = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["optimiser"]["lr"],
        weight_decay=cfg["training"]["optimiser"]["weight_decay"],
        eps=1e-10,
    )
    print("Optimiser initialised.")

    if restart:
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
        print("Model state dict loaded.")

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    wandb.watch(model, log="gradients")

    scheduler = WarmupCosineSchedulerWithRestarts(
        optimiser,
        warmup_epochs=cfg["training"]["scheduler"]["warmup_epochs"],
        total_epochs=cfg["training"]["scheduler"]["total_epochs"],
        min_lr=cfg["training"]["scheduler"]["min_lr"],
        red_fac=cfg["training"]["scheduler"]["red_fac"],
        last_epoch=cfg["training"]["last_epoch_in_prev_run"] if restart else -1,
    )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimiser, T_max=cfg["training"]["epochs"] * 1.5
    # )

    # Calculate pos_weight from training subset labels
    train_labels_tensor = torch.from_numpy(train_labels).float()
    num_neg = torch.sum(train_labels_tensor == 0)
    num_pos = torch.sum(train_labels_tensor == 1)
    # pos_weight = (num_neg / num_pos).clone().detach().to(device)
    pos_weight = torch.tensor(cfg["training"]["pos_weight"], device=device)
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

        train_cls_loss_sum = 0.0
        train_reg_loss_sum = 0.0
        train_quant_loss_sum = 0.0
        train_outputs = []
        train_labels = []
        train_qcd_weights = []
        n_train_batches = 0

        for X_batch, y_batch, mask_batch, weights, jet_pt, _, gen_pt, qcd_weights in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            weights = weights.to(device)
            jet_pt = jet_pt.to(device)
            gen_pt = gen_pt.to(device)
            qcd_weights = qcd_weights.to(device)
            optimiser.zero_grad()

            outputs = model(X_batch, particle_mask=mask_batch)
            cls_output = (
                outputs["classification"] if isinstance(outputs, dict) else outputs
            )
            pt_output = outputs.get("pt", None) if isinstance(outputs, dict) else None
            quant_output = outputs["quantiles"] if isinstance(outputs, dict) else None

            # Classification loss (weighted)
            per_sample_loss = criterion(cls_output, y_batch)
            cls_loss = (per_sample_loss * weights).mean()

            # Regression loss (signal-only)
            reg_loss = torch.tensor(0.0, device=device)
            if pt_output is not None:
                signal_mask = y_batch.squeeze() == 1
                if signal_mask.any():
                    pt_target = (gen_pt[signal_mask] / jet_pt[signal_mask]).squeeze()
                    pt_pred = pt_output[signal_mask].squeeze()
                    reg_loss = nn.functional.mse_loss(pt_pred, pt_target)

            # Quantile regression loss (if enabled, signal-only)
            quant_loss = torch.tensor(0.0, device=device)
            if quant_output is not None:
                quant_loss_fn = QuantileLoss(
                    quantiles=cfg["model"]["quantiles"], reduction="mean"
                )
                signal_mask = y_batch.squeeze() == 1
                if signal_mask.any():
                    quant_preds = quant_output[signal_mask]
                    quant_target = (gen_pt[signal_mask] / jet_pt[signal_mask]).reshape(1, quant_preds.shape[0])
                    quant_loss = quant_loss_fn(quant_preds, quant_target)

            loss = cls_loss + REG_SCALE_FAC * reg_loss + QREG_SCALE_FAC * quant_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            train_cls_loss_sum += cls_loss.item()
            train_reg_loss_sum += reg_loss.item() if pt_output is not None else 0.0
            train_quant_loss_sum += (
                quant_loss.item() if quant_output is not None else 0.0
            )
            n_train_batches += 1

            train_outputs.append(torch.sigmoid(cls_output).detach().cpu().numpy())
            train_labels.append(y_batch.cpu().numpy())
            train_qcd_weights.append(qcd_weights.detach().cpu().numpy())
        scheduler.step()

        # --- Train metrics ---
        train_metrics = {
            "train_loss": (
                train_cls_loss_sum + REG_SCALE_FAC * train_reg_loss_sum + QREG_SCALE_FAC * train_quant_loss_sum
            )
            / n_train_batches,
            "train_cls_loss": train_cls_loss_sum / n_train_batches,
            "train_auc": roc_auc_score(
                np.concatenate(train_labels), np.concatenate(train_outputs), sample_weight=np.concatenate(train_qcd_weights)
            ),
        }
        if pt_regression:
            train_metrics["train_reg_loss"] = train_reg_loss_sum / n_train_batches
        if quantile_regression:
            train_metrics["train_quant_loss"] = train_quant_loss_sum / n_train_batches

        # Validation
        model.eval()
        val_metrics = {
                "val_loss": None,
                "val_cls_loss": None,
                "val_quant_loss": None,
                "val_auc": None,
            }
        auc_score=0.0
        all_preds, all_labels, all_qcd_weights = [], [], []
        val_cls_loss_sum = 0.0
        val_reg_loss_sum = 0.0
        val_quant_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch, mask_batch, _, jet_pt, _, gen_pt, qcd_weights in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                jet_pt = jet_pt.to(device)
                gen_pt = gen_pt.to(device)
                qcd_weights = qcd_weights.to(device)

                outputs = model(X_batch, particle_mask=mask_batch)
                cls_output = (
                    outputs["classification"] if isinstance(outputs, dict) else outputs
                )
                pt_output = (
                    outputs.get("pt", None) if isinstance(outputs, dict) else None
                )
                quant_output = (
                    outputs["quantiles"] if isinstance(outputs, dict) else None
                )
                val_cls_loss_sum += criterion(cls_output, y_batch).mean().item()

                if pt_output is not None:
                    signal_mask = y_batch.squeeze() == 1
                    if signal_mask.any():
                        pt_target = (
                            gen_pt[signal_mask] / jet_pt[signal_mask]
                        ).squeeze()
                        pt_pred = pt_output[signal_mask].squeeze()
                        val_reg_loss_sum += nn.functional.mse_loss(
                            pt_pred, pt_target
                        ).item()

                if quant_output is not None:
                    signal_mask = y_batch.squeeze() == 1
                    if signal_mask.any():
                        quant_preds = quant_output[signal_mask]
                        quant_target = (gen_pt[signal_mask] / jet_pt[signal_mask]).reshape(1, quant_preds.shape[0])
                        val_quant_loss_sum += quant_loss_fn(quant_preds, quant_target).item()

                n_val_batches += 1
                all_preds.append(torch.sigmoid(cls_output).cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
                all_qcd_weights.append(qcd_weights.detach().cpu().numpy())
        # --- Val metrics ---
        val_metrics = {
            "val_loss": (val_cls_loss_sum + REG_SCALE_FAC * val_reg_loss_sum + QREG_SCALE_FAC * val_quant_loss_sum)
            / n_val_batches,
            "val_cls_loss": val_cls_loss_sum / n_val_batches,
            "val_quant_loss": val_quant_loss_sum / n_val_batches,
            "val_auc": roc_auc_score(
                np.concatenate(all_labels), np.concatenate(all_preds), sample_weight=np.concatenate(all_qcd_weights)
            ),
        }
        if pt_regression:
            val_metrics["val_reg_loss"] = val_reg_loss_sum / n_val_batches
        if quantile_regression:
            val_metrics["val_quant_loss"] = val_quant_loss_sum / n_val_batches
        # Logging
        auc_score = val_metrics["val_auc"]
        print(
            f"Epoch {epoch+1} | Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val AUC: {auc_score:.4f}"
        )

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
                f"best_part_model_{wandb.run.id}.pth",
            )
            print(f"New best model saved with AUC: {best_auc:.4f}")
            best_artifact = wandb.Artifact(
                "best_model",
                type="model",
                description=f"Best model with AUC: {best_auc:.4f} at epoch {epoch+1}",
                metadata={"epoch": epoch + 1, "val_auc": best_auc, "config": cfg},
            )
            best_artifact.add_file(f"best_part_model_{wandb.run.id}.pth")
            wandb.log_artifact(best_artifact, aliases=["best"])

        if (epoch + 1) % cfg["training"]["log_freq"] == 0:
            print(f"Saving periodic model at epoch {epoch+1}")
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
                f"part_model_{wandb.run.id}_epoch_{epoch+1}.pth",
            )

            periodic_artifact = wandb.Artifact(
                "periodic_model",
                type="model",
                description=f"Model checkpoint at epoch {epoch+1} with AUC: {auc_score:.4f}",
                metadata={"epoch": epoch + 1, "val_auc": auc_score, "config": cfg},
            )
            periodic_artifact.add_file(f"part_model_{wandb.run.id}_epoch_{epoch+1}.pth")
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
        f"final_model_{wandb.run.id}.pth at epoch {num_epochs} with AUC: {auc_score:.4f}",
    )
    final_artifact = wandb.Artifact(
        "final_model",
        type="model",
        description=f"Final model after {num_epochs} epochs with AUC: {auc_score:.4f}",
        metadata={"epoch": num_epochs, "val_auc": auc_score, "config": cfg},
    )
    final_artifact.add_file(f"final_model_{wandb.run.id}.pth")
    wandb.log_artifact(final_artifact, aliases=["final"])
    print("Final model saved and logged to W&B.")
    wandb.finish()
    print("Run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_part.json")
    args = parser.parse_args()
    run_training(args.config)


