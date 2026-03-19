import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
import torchvision.models as models

from utils import load_training_data


class PatchDataset(Dataset):
    def __init__(self, patches, labels, mean, std, augment=False):
        self.patches, self.labels = patches, labels
        self.mean, self.std = mean, std
        self.augment = augment

    def __len__(self): return len(self.labels)

    def _augment(self, x):
        if np.random.rand() < 0.5: x = x[:,:,::-1].copy()
        if np.random.rand() < 0.5: x = x[:,::-1,:].copy()
        k = np.random.randint(0,4)
        if k: x = np.rot90(x,k,(1,2)).copy()
        return x

    def __getitem__(self, idx):
        x = self.patches[idx].astype(np.float32)
        y = int(self.labels[idx])
        if self.augment: x = self._augment(x)
        x = (x - self.mean[:,None,None]) / self.std[:,None,None]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def build_model(num_classes=71):
    model = models.resnet18(weights="IMAGENET1K_V1")

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(15, old_conv.out_channels,
                            kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride,
                            padding=old_conv.padding,
                            bias=False)

    with torch.no_grad():
        model.conv1.weight[:, :3] = old_conv.weight
        model.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    patches, labels = load_training_data()

    mean = patches.mean(axis=(0,2,3)).astype(np.float32)
    std = patches.std(axis=(0,2,3)).astype(np.float32)
    std = np.where(std<1e-6,1.0,std)

    counts = np.bincount(labels, minlength=71)
    rare_classes = set(np.where(counts < 2)[0])

    all_idx = np.arange(len(labels))
    rare_idx = np.array([i for i in all_idx if labels[i] in rare_classes], dtype=int)
    common_idx = np.array([i for i in all_idx if labels[i] not in rare_classes], dtype=int)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    train_c, val_c = next(splitter.split(np.zeros(len(common_idx)), labels[common_idx]))

    train_idx = np.concatenate([common_idx[train_c], rare_idx])
    val_idx   = common_idx[val_c]


    train_ds = PatchDataset(patches[train_idx], labels[train_idx], mean, std, augment=True)
    val_ds   = PatchDataset(patches[val_idx],   labels[val_idx],   mean, std, augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    model = build_model().to(device)

    # Freeze backbone first
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    weights_dir = Path("weights"); weights_dir.mkdir(exist_ok=True)
    np.save(weights_dir/"channel_mean.npy", mean)
    np.save(weights_dir/"channel_std.npy", std)

    for epoch in range(1, 21):
        model.train()
            # 🔓 Unfreeze full model after 10 epochs (fine-tuning)
        if epoch == 11:
            print("🔓 Unfreezing full model for fine-tuning...")
            for p in model.parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb,yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                p = torch.argmax(out,1).cpu().numpy()
                preds.append(p)
                trues.append(yb.numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        f1w = f1_score(trues, preds, average="weighted")
        print(f"Epoch {epoch} | Val F1 weighted: {f1w:.4f}")

        if f1w > best_f1:
            best_f1 = f1w
            torch.save(model.state_dict(), weights_dir/"habitat_resnet18.pt")
            print("  Saved best model")

    print("Best F1:", best_f1)


if __name__ == "__main__":
    main()
