import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
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

        print(f"Data loaded: {self.X.shape} samples")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Prepare Data
    dataset = L1JetDataset(cfg["data_path"])
    val_size = int(len(dataset) * cfg["training"]["val_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

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

    if restart:
        ckpt = get_model_ckpt(
            run_id=run_id,
            ckpt_name=cfg["training"]["ckpt_lo_load"],
            wandb_dir=cfg["training"]["wandb_dir"],
        )
        model.load_state_dict(ckpt)
        print("Model state dict loaded.")

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    wandb.watch(model, log="gradients")

    # 5. Optimiser & Scheduler
    optimiser = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        eps=1e-10,
    )
    print("Optimiser initialised.")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["training"]["epochs"] * 1.5
    )
    pos_weight = (len(train_ds.dataset) - torch.sum(train_ds.dataset.y)) / torch.sum(
        train_ds.dataset.y
    )
    pos_weight = torch.tensor(pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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

        for X_batch, y_batch, mask_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimiser.zero_grad()

            outputs = model(X_batch, particle_mask=mask_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimiser.step()

            total_loss += loss.item()
        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch, mask_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                outputs = model(X_batch, particle_mask=mask_batch)
                val_loss += criterion(outputs, y_batch).item()

                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        auc_score = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds))

        # Logging
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {auc_score:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_auc": auc_score,
                "lr": optimiser.param_groups[0]["lr"],
            }
        )

        if auc_score > best_auc:
            best_auc = auc_score
            torch.save(model.state_dict(), "best_part_model.pth")
            torch.save(optimiser.state_dict(), "best_model_optim.pth")
            wandb.save("best_part_model.pth")
            wandb.save("best_model_optim.pth")

        scheduler.step()

    print(f"Training Complete. Best AUC: {best_auc:.4f}")
    print("Saving final model")
    torch.save(model.state_dict(), "final_model.pth")
    torch.save(optimiser.state_dict(), "final_model_optim.pth")
    wandb.save("final_model.pth")
    wandb.save("final_model_optim.pth")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_part.json")
    args = parser.parse_args()
    run_training(args.config)
