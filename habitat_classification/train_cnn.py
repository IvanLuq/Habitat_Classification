# train_cnn.py  (COMPETITION-COMPLIANT: uses ONLY the original 15 input channels)

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

from utils import load_training_data


# ============================================================
# RAW 15-channel mean/std (NO feature enrichment allowed)
# ============================================================

def compute_mean_std_raw(patches: np.ndarray):
    """
    Compute per-channel mean/std for RAW 15-channel input.

    patches: (N, 15, 35, 35)
    returns:
      mean: (15,)
      std:  (15,)
    """
    mean = patches.mean(axis=(0, 2, 3)).astype(np.float32)
    std = patches.std(axis=(0, 2, 3)).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


# =========================
# Model
# =========================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.c1 = ConvBNAct(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.c1(x)
        y = self.bn2(self.c2(y))
        y = self.drop(y)
        return self.act(x + y)


class HabitatCNN(nn.Module):
    def __init__(self, in_channels: int = 15, num_classes: int = 71, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 32, 3, 1, 1),
            ConvBNAct(32, 32, 3, 1, 1),
        )
        self.down1 = nn.Sequential(
            ConvBNAct(32, 64, 3, 2, 1),   # 35 -> 18
            ResidualBlock(64, dropout=dropout),
        )
        self.down2 = nn.Sequential(
            ConvBNAct(64, 128, 3, 2, 1),  # 18 -> 9
            ResidualBlock(128, dropout=dropout),
            ResidualBlock(128, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.pool(x)
        return self.head(x)


# =========================
# Dataset (RAW 15 channels)
# =========================

class PatchDataset(Dataset):
    def __init__(self, patches, labels, mean, std, augment=False):
        self.patches = patches
        self.labels = labels
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def _augment(self, x):
        # x: (15,35,35)
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1].copy()
        if np.random.rand() < 0.5:
            x = x[:, ::-1, :].copy()
        k = np.random.randint(0, 4)
        if k:
            x = np.rot90(x, k, axes=(1, 2)).copy()
        return x

    def __getitem__(self, idx):
        x = self.patches[idx].astype(np.float32)  # (15,35,35)
        y = int(self.labels[idx])

        if self.augment:
            x = self._augment(x)

        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# =========================
# Train
# =========================

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    patches, labels = load_training_data()
    N = len(labels)
    print("Loaded:", patches.shape, labels.shape)  # (N,15,35,35), (N,)

    # ---- robust split (force very rare classes into train) ----
    counts = np.bincount(labels, minlength=71)
    rare_classes = set(np.where(counts < 2)[0])

    all_idx = np.arange(N)
    rare_idx = np.array([i for i in all_idx if labels[i] in rare_classes], dtype=int)
    common_idx = np.array([i for i in all_idx if labels[i] not in rare_classes], dtype=int)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    train_c, val_c = next(splitter.split(np.zeros(len(common_idx)), labels[common_idx]))

    train_idx = np.concatenate([common_idx[train_c], rare_idx])
    val_idx = common_idx[val_c]

    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    print(f"Rare classes forced into train: {len(rare_classes)}")
    print("Min class count:", counts.min(), "| #classes count==1:", int((counts == 1).sum()))

    # ---- mean/std on RAW 15-channel representation (train only to avoid leakage) ----
    mean, std = compute_mean_std_raw(patches[train_idx])
    assert mean.shape == (15,) and std.shape == (15,), f"mean/std shapes wrong: {mean.shape}, {std.shape}"
    in_channels = 15
    print("Channels (raw):", in_channels)

    train_ds = PatchDataset(patches[train_idx], labels[train_idx], mean, std, augment=True)
    val_ds = PatchDataset(patches[val_idx], labels[val_idx], mean, std, augment=False)

    # ---- Balanced sampling (oversample rare classes) ----
    y_train = labels[train_idx]
    class_counts = np.bincount(y_train, minlength=71).astype(np.float32)

    class_w = 1.0 / (class_counts + 1e-6)         # per-class weight
    sample_w = class_w[y_train].astype(np.float64) # per-sample weight

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_w),
        num_samples=len(sample_w),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        sampler=sampler,     # balanced sampling
        shuffle=False,       # must be False when sampler is used
        num_workers=2,
        pin_memory=False
    )

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=False)

    # ---- Class weights for CE ----
    train_counts = np.bincount(labels[train_idx], minlength=71).astype(np.float32)
    class_weights = train_counts.sum() / (train_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # ---- Logit Adjustment (class priors) ----
    priors = train_counts / train_counts.sum()
    log_priors = np.log(priors + 1e-12).astype(np.float32)
    log_priors_t = torch.tensor(log_priors, dtype=torch.float32, device=device)

    tau = 0.5

    model = HabitatCNN(in_channels=in_channels, num_classes=71, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    # ---- CosineAnnealing scheduler ----
    epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    # Save normalization stats expected by model.py (shape (15,))
    np.save(weights_dir / "channel_mean.npy", mean)
    np.save(weights_dir / "channel_std.npy", std)

    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            logits = logits + tau * log_priors_t[None, :]  # logit adjustment
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        train_loss = running / len(train_ds)

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)

                logits = model(xb)
                logits = logits + tau * log_priors_t[None, :]  # logit adjustment

                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_true.extend(yb.tolist())

        all_preds = np.array(all_preds, dtype=int)
        all_true = np.array(all_true, dtype=int)
        assert len(all_true) == len(all_preds)

        f1w = f1_score(all_true, all_preds, average="weighted")
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | val_f1_weighted={f1w:.4f}")

        if f1w > best_f1:
            best_f1 = f1w
            torch.save(model.state_dict(), weights_dir / "habitat_cnn.pt")
            print("  saved new best model")

        scheduler.step()

    print("Best val weighted F1:", best_f1)
    print("Saved:")
    print(" - weights/habitat_cnn.pt")
    print(" - weights/channel_mean.npy  (15,)")
    print(" - weights/channel_std.npy   (15,)")


if __name__ == "__main__":
    main()
