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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


# ── DDP helpers ──────────────────────────────────────────────────────

def is_distributed(cfg):
    """Check whether DDP should be enabled based on config and environment."""
    dist_cfg = cfg.get("training", {}).get("distributed", {})
    if not dist_cfg.get("enabled", False):
        return False
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def init_distributed(cfg):
    """Initialise the DDP process group and return runtime info.

    Returns (rank, world_size, local_rank, device).
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist_cfg = cfg.get("training", {}).get("distributed", {})
    backend = dist_cfg.get("backend", "nccl")
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"DDP rank {rank}/{world_size} on cuda:{local_rank} (backend={backend})")
    return rank, world_size, local_rank, device


def is_main(rank):
    """True on the rank responsible for logging, checkpointing, and W&B."""
    return rank == 0


def cleanup_distributed():
    """Destroy the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_training(cfg):
    """Run the full event-level training loop.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (already loaded and overridden).
    """

    # Distributed setup
    distributed = is_distributed(cfg)
    if distributed:
        rank, world_size, local_rank, device = init_distributed(cfg)
    else:
        rank, world_size, local_rank = 0, 1, 0
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Single-process mode on {device}")

    # W&B init (main rank only)
    restart = cfg["training"]["restart"]
    run_id = (
        extract_wandb_run_id(cfg["training"]["wandb_run_path"]) if restart else None
    )
    if is_main(rank):
        wandb.init(
            project=cfg["wandb"]["project"],
            config=cfg,
            name=cfg["exp_name"],
            id=run_id,
            resume="allow",
        )

    # Data
    dataset = StratifiedJetDataset(cfg["training"]["data"]["data_path"])
    train_ds, val_ds, train_indices, val_indices, train_labels = stratified_split(
        dataset, cfg["training"]["data"]["val_split"], num_classes=1
    )

    # Samplers (DDP splits data across ranks; single-process uses default shuffle)
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=True,  # avoid padded duplicates in AUC
        )

    num_workers = cfg["training"]["data"]["num_workers"]
    if distributed:
        num_workers = max(1, num_workers // world_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=precollated_collate,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=precollated_collate,
        pin_memory=True,
        persistent_workers=True,
    )
    if is_main(rank):
        print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Model — build, load checkpoint, then wrap with DDP
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
    )

    # SyncBatchNorm for DDP (synchronises BN stats across GPUs)
    if distributed and cfg["model"].get("use_batch_norm", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    if is_main(rank):
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimiser (created before restart load so we can load its state_dict)
    optimiser = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["optimiser"]["lr"],
        weight_decay=cfg["training"]["optimiser"]["weight_decay"],
        eps=1e-10,
    )

    # Restart: load checkpoint before DDP wrapping
    if restart:
        if is_main(rank):
            artifact_path = (
                f"{cfg['wandb']['entity']}/{cfg['wandb']['project']}/"
                f"{cfg['wandb']['artifact_name']}:{cfg['wandb']['ckpt_type']}"
            )
            artifact = wandb.use_artifact(artifact_path, type="model")
            artifact_dir = artifact.download()
            checkpoint_dir = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
            checkpoint = torch.load(
                checkpoint_dir, map_location=device, weights_only=False
            )
        else:
            checkpoint = None

        # In DDP, broadcast checkpoint from rank 0 to all ranks
        if distributed:
            ckpt_list = [checkpoint]
            dist.broadcast_object_list(ckpt_list, src=0)
            checkpoint = ckpt_list[0]

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        if is_main(rank):
            print("Loaded model & optimiser from W&B artifact")

    # DDP wrapping (after checkpoint load so state_dict keys match)
    if distributed:
        dist_cfg = cfg.get("training", {}).get("distributed", {})
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=dist_cfg.get("find_unused_parameters", False),
        )

    # raw_model for checkpoint I/O and parameter counting
    raw_model = model.module if isinstance(model, DDP) else model

    if is_main(rank):
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
    if is_main(rank):
        print(f"pos_weight: {pos_weight.item():.4f}")

    # AMP
    use_amp = cfg["training"].get("use_amp", False)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Training loop
    best_auc = 0.0
    start_epoch = cfg["training"]["last_epoch_in_prev_run"] if restart else 0
    num_epochs = start_epoch + cfg["training"]["epochs"]

    if is_main(rank):
        print(f"Training from epoch {start_epoch} to {num_epochs}")

    try:
        for epoch in range(start_epoch, num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
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
                        outputs["classification"]
                        if isinstance(outputs, dict)
                        else outputs
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
                if is_main(rank):
                    print(
                        f"Epoch {epoch+1} Batch {n_train_batches} "
                        f"Loss: {cls_loss.item():.4f}"
                    )
                train_outputs.append(
                    torch.sigmoid(cls_output).detach().cpu().numpy()
                )
                train_labels_list.append(y_batch.cpu().numpy())
                train_qcd_w.append(qcd_weights.numpy())
            scheduler.step()

            # ── Aggregate training loss across ranks ──
            if distributed:
                loss_counts = torch.tensor(
                    [train_cls_loss_sum, float(n_train_batches)], device=device
                )
                dist.all_reduce(loss_counts, op=dist.ReduceOp.SUM)
                train_cls_loss_sum = loss_counts[0].item()
                n_train_batches = int(loss_counts[1].item())

            # Train metrics (local AUC on rank 0 only — approximation is fine)
            if is_main(rank):
                train_metrics = {
                    "train_loss": train_cls_loss_sum / max(n_train_batches, 1),
                    "train_auc": roc_auc_score(
                        np.concatenate(train_labels_list),
                        np.concatenate(train_outputs),
                        sample_weight=np.concatenate(train_qcd_w),
                    ),
                }

            # ── Validation ──
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

            # ── Aggregate val loss across ranks ──
            if distributed:
                val_counts = torch.tensor(
                    [val_loss_sum, float(n_val_batches)], device=device
                )
                dist.all_reduce(val_counts, op=dist.ReduceOp.SUM)
                val_loss_sum = val_counts[0].item()
                n_val_batches = int(val_counts[1].item())

            # ── Aggregate val AUC across ranks ──
            if distributed:
                local_data = (
                    np.concatenate(all_preds),
                    np.concatenate(all_labels),
                    np.concatenate(all_qcd_w),
                )
                gathered = [None] * world_size
                dist.all_gather_object(gathered, local_data)
                if is_main(rank):
                    all_preds_g = np.concatenate([g[0] for g in gathered])
                    all_labels_g = np.concatenate([g[1] for g in gathered])
                    all_qcd_w_g = np.concatenate([g[2] for g in gathered])
                    auc_score = roc_auc_score(
                        all_labels_g, all_preds_g, sample_weight=all_qcd_w_g
                    )
                else:
                    auc_score = 0.0
                # Broadcast global AUC so all ranks agree on best-model state
                auc_tensor = torch.tensor([auc_score], device=device)
                dist.broadcast(auc_tensor, src=0)
                auc_score = auc_tensor.item()
            else:
                auc_score = roc_auc_score(
                    np.concatenate(all_labels),
                    np.concatenate(all_preds),
                    sample_weight=np.concatenate(all_qcd_w),
                )

            val_metrics = {
                "val_loss": val_loss_sum / max(n_val_batches, 1),
                "val_auc": auc_score,
            }

            if is_main(rank):
                print(
                    f"Epoch {epoch+1} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val AUC: {auc_score:.4f}"
                )

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        **train_metrics,
                        **val_metrics,
                        "lr": optimiser.param_groups[0]["lr"],
                    }
                )

            # ── Checkpointing (main rank only) ──
            if is_main(rank):
                ckpt_dict = {
                    "config": cfg,
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                    "val_auc": auc_score,
                }

                if auc_score > best_auc:
                    best_auc = auc_score
                    torch.save(
                        ckpt_dict,
                        f"best_event_model_{wandb.run.id}.pth",
                    )
                    print(f"  New best model (AUC: {best_auc:.4f})")
                    art = wandb.Artifact(
                        "best_model",
                        type="model",
                        description=(
                            f"Best model AUC={best_auc:.4f} epoch={epoch+1}"
                        ),
                        metadata={
                            "epoch": epoch + 1,
                            "val_auc": best_auc,
                            "config": cfg,
                        },
                    )
                    art.add_file(f"best_event_model_{wandb.run.id}.pth")
                    wandb.log_artifact(art, aliases=["best"])

                if (epoch + 1) % cfg["training"]["log_freq"] == 0:
                    torch.save(
                        ckpt_dict,
                        f"event_model_{wandb.run.id}_epoch_{epoch+1}.pth",
                    )
                    art = wandb.Artifact(
                        "periodic_model",
                        type="model",
                        description=(
                            f"Checkpoint epoch={epoch+1} AUC={auc_score:.4f}"
                        ),
                        metadata={
                            "epoch": epoch + 1,
                            "val_auc": auc_score,
                            "config": cfg,
                        },
                    )
                    art.add_file(
                        f"event_model_{wandb.run.id}_epoch_{epoch+1}.pth"
                    )
                    wandb.log_artifact(art, aliases=[f"epoch_{epoch+1}"])
            else:
                # Non-main ranks still track best AUC for consistency
                if auc_score > best_auc:
                    best_auc = auc_score

            # Barrier so non-main ranks wait for checkpoint writes
            if distributed:
                dist.barrier()

        # ── Final model (main rank only) ──
        if is_main(rank):
            print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
            ckpt_dict = {
                "config": cfg,
                "epoch": num_epochs - 1,
                "model_state_dict": raw_model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["val_loss"],
                "val_auc": auc_score,
            }
            torch.save(ckpt_dict, f"final_event_model_{wandb.run.id}.pth")
            art = wandb.Artifact(
                "final_model",
                type="model",
                description=(
                    f"Final model after {num_epochs} epochs AUC={auc_score:.4f}"
                ),
                metadata={
                    "epoch": num_epochs,
                    "val_auc": auc_score,
                    "config": cfg,
                },
            )
            art.add_file(f"final_event_model_{wandb.run.id}.pth")
            wandb.log_artifact(art, aliases=["final"])
            wandb.finish()

    finally:
        cleanup_distributed()


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
