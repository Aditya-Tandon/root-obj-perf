import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score
import wandb
import argparse
from model.parT import ParticleTransformer
from model.ddp_helpers import (
    is_distributed,
    init_distributed,
    is_main,
    cleanup_distributed,
)
from wandb_utils import extract_wandb_run_id, get_model_ckpt

from data_pipeline.combined_loader import CombinedJetDataLoader
from model.warmup_cosine_lr import WarmupCosineSchedulerWithRestarts
from evaluation.luminosity import build_eval_weights

torch.manual_seed(42)
np.random.seed(42)

REG_SCALE_FAC = 1.0
QREG_SCALE_FAC = 1.0
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

def run_training(cfg):
    """Run the full training loop.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (already loaded and overridden).
    """

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
        if is_main(rank):
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
        if is_main(rank):
            print("\nUsing PUPPI dataset for training.")
        (
            train_loader,
            train_indices,
            val_loader,
            val_indices,
            train_labels,
            val_labels,
        ) = combined_loader.get_puppi_loaders(shuffle=True)
    # Rebuild loaders with DistributedSampler when running DDP.
    train_sampler = None
    val_sampler = None
    if distributed:
        per_rank_workers = max(1, cfg["training"]["data"]["num_workers"] // world_size)
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            sampler=train_sampler,
            num_workers=per_rank_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=per_rank_workers > 0,
            prefetch_factor=5 if per_rank_workers > 0 else None,
            collate_fn=train_loader.collate_fn,
        )
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            sampler=val_sampler,
            num_workers=per_rank_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=per_rank_workers > 0,
            prefetch_factor=5 if per_rank_workers > 0 else None,
            collate_fn=val_loader.collate_fn,
        )

    if is_main(rank):
        print(
            f"Data loaders prepared with {len(train_loader.dataset)} training samples and "
            f"{len(val_loader.dataset)} validation samples."
        )

    # 3b. Load physics config for evaluation weights (sigma_to_ngen mapping)
    physics_cfg_path = cfg["training"].get("physics_config", "hh-bbbb-obj-config.json")
    with open(physics_cfg_path) as f:
        phys_cfg = json.load(f)
    sigma_to_ngen = {
        b["weight"]: b["n_gen"] for b in phys_cfg["QCD_background"].values()
    }
    eval_weight_mode = cfg["training"].get("eval_weight_mode", "qcd_only")
    phys = phys_cfg.get("physics", {})
    luminosity_fb = phys.get("luminosity_fb", 1000.0)
    signal_xsec_pb = phys.get("signal_xsec_pb", 0.0113)
    n_gen_signal = phys.get("n_gen_signal")
    if is_main(rank):
        print(
            f"Eval weight mode: {eval_weight_mode} "
            f"(physics config: {physics_cfg_path})"
        )

    # 4. Initialize Particle Transformer
    pt_regression = cfg["model"].get("pt_regression", False)
    quantile_regression = cfg["model"].get("quantile_regression", False)

    QREG_SCALE_FAC = cfg["training"].get("qreg_loss_scale", 1.0)
    REG_SCALE_FAC = cfg["training"].get("pt_reg_loss_scale", 1.0)
    if is_main(rank):
        print(
            "Using regression loss scale factors - "
            f"PT: {REG_SCALE_FAC}, Quantile: {QREG_SCALE_FAC}"
        )

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
        pt_regression=pt_regression,
        quantile_regression=quantile_regression,
    )

    if distributed and cfg["model"].get("use_batch_norm", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    if is_main(rank):
        print("Model loaded.")

    if pt_regression:
        if is_main(rank):
            print("PT regression head enabled in the model.")
    if quantile_regression:
        if is_main(rank):
            print("Quantile regression enabled in the model.")

    # 5. Optimiser & Scheduler
    optimiser = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["optimiser"]["lr"],
        weight_decay=cfg["training"]["optimiser"]["weight_decay"],
        eps=1e-10,
    )
    if is_main(rank):
        print("Optimiser initialised.")

    if restart:
        if is_main(rank):
            artifact_path = (
                f"{cfg['wandb']['entity']}/{cfg['wandb']['project']}/"
                f"{cfg['wandb']['artifact_name']}:{cfg['wandb']['ckpt_type']}"
            )
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
            else:
                checkpoint = None
        else:
            checkpoint = None

        if distributed:
            ckpt_list = [checkpoint]
            dist.broadcast_object_list(ckpt_list, src=0)
            checkpoint = ckpt_list[0]

        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
            if is_main(rank):
                print("Model and optimiser state loaded from W&B artifact")
        elif is_main(rank):
            print("No checkpoint found in artifact directory. Starting fresh.")

    if distributed:
        dist_cfg = cfg.get("training", {}).get("distributed", {})
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=dist_cfg.get("find_unused_parameters", False),
        )

    raw_model = model.module if isinstance(model, DDP) else model

    if is_main(rank):
        print(f"Model Parameters: {sum(p.numel() for p in raw_model.parameters())}")
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

    # Calculate pos_weight from config
    pos_weight = torch.tensor(cfg["training"]["pos_weight"], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    if is_main(rank):
        print(f"Criterion initialised with pos_weight: {pos_weight.item():.4f}")

    use_amp = cfg["training"].get("use_amp", False)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    quant_loss_fn = (
        QuantileLoss(quantiles=cfg["model"]["quantiles"], reduction="mean")
        if quantile_regression
        else None
    )

    best_auc = 0.0
    start_epoch = 0 if restart is False else cfg["training"]["last_epoch_in_prev_run"]
    num_epochs = start_epoch + cfg["training"]["epochs"]
    if is_main(rank):
        print(f"Starting training from epoch {start_epoch} for {num_epochs} epochs.")

    try:
        for epoch in range(start_epoch, num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()

            train_cls_loss_sum = 0.0
            train_reg_loss_sum = 0.0
            train_quant_loss_sum = 0.0
            train_outputs = []
            train_labels = []
            train_qcd_weights = []
            n_train_batches = 0

            for (
                X_batch,
                y_batch,
                mask_batch,
                weights,
                jet_pt,
                _,
                gen_pt,
                qcd_weights,
            ) in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                weights = weights.to(device)
                jet_pt = jet_pt.to(device)
                gen_pt = gen_pt.to(device)
                qcd_weights = qcd_weights.to(device)
                optimiser.zero_grad()

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(X_batch, particle_mask=mask_batch)
                    cls_output = (
                        outputs["classification"]
                        if isinstance(outputs, dict)
                        else outputs
                    )
                    pt_output = (
                        outputs.get("pt", None)
                        if isinstance(outputs, dict)
                        else None
                    )
                    quant_output = (
                        outputs.get("quantiles", None)
                        if isinstance(outputs, dict)
                        else None
                    )

                    per_sample_loss = criterion(cls_output, y_batch)
                    cls_loss = (per_sample_loss * weights).mean()

                    reg_loss = torch.tensor(0.0, device=device)
                    if pt_output is not None:
                        signal_mask = y_batch.squeeze() == 1
                        if signal_mask.any():
                            pt_target = (
                                gen_pt[signal_mask] / jet_pt[signal_mask]
                            ).squeeze()
                            pt_pred = pt_output[signal_mask].squeeze()
                            reg_loss = nn.functional.mse_loss(pt_pred, pt_target)

                    quant_loss = torch.tensor(0.0, device=device)
                    if quant_output is not None:
                        signal_mask = y_batch.squeeze() == 1
                        if signal_mask.any():
                            quant_preds = quant_output[signal_mask]
                            quant_target = (
                                gen_pt[signal_mask] / jet_pt[signal_mask]
                            ).reshape(1, quant_preds.shape[0])
                            quant_loss = quant_loss_fn(quant_preds, quant_target)

                    loss = (
                        cls_loss
                        + REG_SCALE_FAC * reg_loss
                        + QREG_SCALE_FAC * quant_loss
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()

                train_cls_loss_sum += cls_loss.item()
                train_reg_loss_sum += reg_loss.item() if pt_output is not None else 0.0
                train_quant_loss_sum += (
                    quant_loss.item() if quant_output is not None else 0.0
                )
                n_train_batches += 1

                train_outputs.append(torch.sigmoid(cls_output).detach().cpu().numpy())
                train_labels.append(y_batch.detach().cpu().numpy())
                train_qcd_weights.append(qcd_weights.detach().cpu().numpy())

            scheduler.step()

            if distributed:
                train_loss_counts = torch.tensor(
                    [
                        train_cls_loss_sum,
                        train_reg_loss_sum,
                        train_quant_loss_sum,
                        float(n_train_batches),
                    ],
                    device=device,
                )
                dist.all_reduce(train_loss_counts, op=dist.ReduceOp.SUM)
                train_cls_loss_sum = train_loss_counts[0].item()
                train_reg_loss_sum = train_loss_counts[1].item()
                train_quant_loss_sum = train_loss_counts[2].item()
                n_train_batches = int(train_loss_counts[3].item())

                local_train_data = (
                    np.concatenate(train_outputs),
                    np.concatenate(train_labels),
                    np.concatenate(train_qcd_weights),
                )
                gathered_train = [None] * world_size
                dist.all_gather_object(gathered_train, local_train_data)
                if is_main(rank):
                    all_train_outputs = np.concatenate([g[0] for g in gathered_train])
                    all_train_labels = np.concatenate([g[1] for g in gathered_train])
                    all_train_qcd_w = np.concatenate([g[2] for g in gathered_train])
            else:
                all_train_labels = np.concatenate(train_labels)
                all_train_outputs = np.concatenate(train_outputs)
                all_train_qcd_w = np.concatenate(train_qcd_weights)

            if is_main(rank):
                n_train_sig = int((all_train_labels.ravel() == 1).sum())
                train_eval_w = build_eval_weights(
                    all_train_qcd_w[all_train_labels.ravel() == 0],
                    sigma_to_ngen,
                    n_train_sig,
                    mode=eval_weight_mode,
                    luminosity_fb=luminosity_fb,
                    signal_xsec_pb=signal_xsec_pb,
                    n_gen_signal=n_gen_signal,
                )
                train_w_out = np.empty(len(all_train_labels), dtype=np.float64)
                sig_mask_tr = all_train_labels.ravel() == 1
                train_w_out[sig_mask_tr] = train_eval_w[:n_train_sig]
                train_w_out[~sig_mask_tr] = train_eval_w[n_train_sig:]

                train_metrics = {
                    "train_loss": (
                        train_cls_loss_sum
                        + REG_SCALE_FAC * train_reg_loss_sum
                        + QREG_SCALE_FAC * train_quant_loss_sum
                    )
                    / max(n_train_batches, 1),
                    "train_cls_loss": train_cls_loss_sum / max(n_train_batches, 1),
                    "train_auc": roc_auc_score(
                        all_train_labels,
                        all_train_outputs,
                        sample_weight=train_w_out,
                    ),
                }
                if pt_regression:
                    train_metrics["train_reg_loss"] = (
                        train_reg_loss_sum / max(n_train_batches, 1)
                    )
                if quantile_regression:
                    train_metrics["train_quant_loss"] = (
                        train_quant_loss_sum / max(n_train_batches, 1)
                    )

            model.eval()
            all_preds, all_labels, all_qcd_weights = [], [], []
            val_cls_loss_sum = 0.0
            val_reg_loss_sum = 0.0
            val_quant_loss_sum = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for (
                    X_batch,
                    y_batch,
                    mask_batch,
                    _,
                    jet_pt,
                    _,
                    gen_pt,
                    qcd_weights,
                ) in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    jet_pt = jet_pt.to(device)
                    gen_pt = gen_pt.to(device)
                    qcd_weights = qcd_weights.to(device)

                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(X_batch, particle_mask=mask_batch)
                        cls_output = (
                            outputs["classification"]
                            if isinstance(outputs, dict)
                            else outputs
                        )
                        pt_output = (
                            outputs.get("pt", None)
                            if isinstance(outputs, dict)
                            else None
                        )
                        quant_output = (
                            outputs.get("quantiles", None)
                            if isinstance(outputs, dict)
                            else None
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
                                quant_target = (
                                    gen_pt[signal_mask] / jet_pt[signal_mask]
                                ).reshape(1, quant_preds.shape[0])
                                val_quant_loss_sum += quant_loss_fn(
                                    quant_preds, quant_target
                                ).item()

                    n_val_batches += 1
                    all_preds.append(torch.sigmoid(cls_output).detach().cpu().numpy())
                    all_labels.append(y_batch.detach().cpu().numpy())
                    all_qcd_weights.append(qcd_weights.detach().cpu().numpy())

            if distributed:
                val_loss_counts = torch.tensor(
                    [
                        val_cls_loss_sum,
                        val_reg_loss_sum,
                        val_quant_loss_sum,
                        float(n_val_batches),
                    ],
                    device=device,
                )
                dist.all_reduce(val_loss_counts, op=dist.ReduceOp.SUM)
                val_cls_loss_sum = val_loss_counts[0].item()
                val_reg_loss_sum = val_loss_counts[1].item()
                val_quant_loss_sum = val_loss_counts[2].item()
                n_val_batches = int(val_loss_counts[3].item())

                local_val_data = (
                    np.concatenate(all_preds),
                    np.concatenate(all_labels),
                    np.concatenate(all_qcd_weights),
                )
                gathered_val = [None] * world_size
                dist.all_gather_object(gathered_val, local_val_data)
                if is_main(rank):
                    all_val_preds = np.concatenate([g[0] for g in gathered_val])
                    all_val_labels = np.concatenate([g[1] for g in gathered_val])
                    all_val_qcd_w = np.concatenate([g[2] for g in gathered_val])
            else:
                all_val_labels = np.concatenate(all_labels)
                all_val_preds = np.concatenate(all_preds)
                all_val_qcd_w = np.concatenate(all_qcd_weights)

            if is_main(rank):
                n_val_sig = int((all_val_labels.ravel() == 1).sum())
                val_eval_w = build_eval_weights(
                    all_val_qcd_w[all_val_labels.ravel() == 0],
                    sigma_to_ngen,
                    n_val_sig,
                    mode=eval_weight_mode,
                    luminosity_fb=luminosity_fb,
                    signal_xsec_pb=signal_xsec_pb,
                    n_gen_signal=n_gen_signal,
                )
                val_w_out = np.empty(len(all_val_labels), dtype=np.float64)
                sig_mask_val = all_val_labels.ravel() == 1
                val_w_out[sig_mask_val] = val_eval_w[:n_val_sig]
                val_w_out[~sig_mask_val] = val_eval_w[n_val_sig:]

                val_metrics = {
                    "val_loss": (
                        val_cls_loss_sum
                        + REG_SCALE_FAC * val_reg_loss_sum
                        + QREG_SCALE_FAC * val_quant_loss_sum
                    )
                    / max(n_val_batches, 1),
                    "val_cls_loss": val_cls_loss_sum / max(n_val_batches, 1),
                    "val_quant_loss": val_quant_loss_sum / max(n_val_batches, 1),
                    "val_auc": roc_auc_score(
                        all_val_labels,
                        all_val_preds,
                        sample_weight=val_w_out,
                    ),
                }
                if pt_regression:
                    val_metrics["val_reg_loss"] = (
                        val_reg_loss_sum / max(n_val_batches, 1)
                    )
                if quantile_regression:
                    val_metrics["val_quant_loss"] = (
                        val_quant_loss_sum / max(n_val_batches, 1)
                    )
                auc_score = val_metrics["val_auc"]
            else:
                val_metrics = {
                    "val_loss": val_cls_loss_sum / max(n_val_batches, 1),
                    "val_cls_loss": val_cls_loss_sum / max(n_val_batches, 1),
                    "val_quant_loss": val_quant_loss_sum / max(n_val_batches, 1),
                    "val_auc": 0.0,
                }
                auc_score = 0.0

            if distributed:
                auc_tensor = torch.tensor([auc_score], device=device)
                dist.broadcast(auc_tensor, src=0)
                auc_score = auc_tensor.item()

            if is_main(rank):
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

                ckpt_dict = {
                    "config": cfg,
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                    "val_auc": val_metrics["val_auc"],
                }

                if auc_score > best_auc:
                    best_auc = auc_score
                    torch.save(
                        ckpt_dict,
                        f"best_part_model_{wandb.run.id}.pth",
                    )
                    print(f"New best model saved with AUC: {best_auc:.4f}")
                    best_artifact = wandb.Artifact(
                        "best_model",
                        type="model",
                        description=(
                            f"Best model with AUC: {best_auc:.4f} at epoch {epoch+1}"
                        ),
                        metadata={
                            "epoch": epoch + 1,
                            "val_auc": best_auc,
                            "config": cfg,
                        },
                    )
                    best_artifact.add_file(f"best_part_model_{wandb.run.id}.pth")
                    wandb.log_artifact(best_artifact, aliases=["best"])

                if (epoch + 1) % cfg["training"]["log_freq"] == 0:
                    print(f"Saving periodic model at epoch {epoch+1}")
                    torch.save(
                        ckpt_dict,
                        f"part_model_{wandb.run.id}_epoch_{epoch+1}.pth",
                    )
                    periodic_artifact = wandb.Artifact(
                        "periodic_model",
                        type="model",
                        description=(
                            f"Model checkpoint at epoch {epoch+1} "
                            f"with AUC: {auc_score:.4f}"
                        ),
                        metadata={
                            "epoch": epoch + 1,
                            "val_auc": auc_score,
                            "config": cfg,
                        },
                    )
                    periodic_artifact.add_file(
                        f"part_model_{wandb.run.id}_epoch_{epoch+1}.pth"
                    )
                    wandb.log_artifact(
                        periodic_artifact,
                        aliases=[f"epoch_{epoch+1}"],
                    )
            else:
                if auc_score > best_auc:
                    best_auc = auc_score

            if distributed:
                dist.barrier()

        if is_main(rank):
            print(f"Training Complete. Best AUC: {best_auc:.4f}")
            print("Saving final model")

            final_ckpt = {
                "config": cfg,
                "epoch": num_epochs - 1,
                "model_state_dict": raw_model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["val_loss"],
                "val_auc": auc_score,
            }
            final_path = f"final_model_{wandb.run.id}.pth"
            torch.save(final_ckpt, final_path)

            final_artifact = wandb.Artifact(
                "final_model",
                type="model",
                description=(
                    f"Final model after {num_epochs} epochs with AUC: {auc_score:.4f}"
                ),
                metadata={"epoch": num_epochs, "val_auc": auc_score, "config": cfg},
            )
            final_artifact.add_file(final_path)
            wandb.log_artifact(final_artifact, aliases=["final"])
            print("Final model saved and logged to W&B.")
            wandb.finish()
            print("Run finished.")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_part.json")
    parser.add_argument(
        "--set", action="append", default=[],
        help="Override config values: --set training.lr=1e-4",
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Override experiment name")
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
        # Auto-cast: try int, then float, then bool, else string
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
