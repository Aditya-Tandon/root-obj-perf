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

    # 2. Initialize WandB
    wandb.init(project="L1-BTagging-ParT", config=cfg, name=cfg["exp_name"])

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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

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

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    wandb.watch(model, log="gradients")

    # 5. Optimiser & Scheduler
    optimiser = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Cosine Annealing is standard for Transformers
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["training"]["epochs"]
    )
    criterion = nn.BCEWithLogitsLoss()

    # AMP Scaler for faster training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["use_amp"])

    # --- Training Loop ---
    best_auc = 0.0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0

        for X_batch, y_batch, mask_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimiser.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg["training"]["use_amp"]):
                outputs = model(X_batch, particle_mask=mask_batch)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            total_loss += loss.item()

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
            wandb.save("best_part_model.pth")

        scheduler.step()

    print(f"Training Complete. Best AUC: {best_auc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_part.json")
    args = parser.parse_args()
    run_training(args.config)
