import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report

from utils import load_training_data
from model import predict


def make_split(labels: np.ndarray, num_classes: int = 71, test_size: float = 0.15, seed: int = 0):
    """
    Same split logic as training:
      - classes with <2 samples forced into train
      - remaining classes stratified into train/val
    """
    counts = np.bincount(labels, minlength=num_classes)
    rare_classes = set(np.where(counts < 2)[0])

    all_idx = np.arange(len(labels))
    rare_idx = np.array([i for i in all_idx if labels[i] in rare_classes], dtype=int)
    common_idx = np.array([i for i in all_idx if labels[i] not in rare_classes], dtype=int)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_c, val_c = next(splitter.split(np.zeros(len(common_idx)), labels[common_idx]))

    train_idx = np.concatenate([common_idx[train_c], rare_idx])
    val_idx = common_idx[val_c]

    return train_idx, val_idx, rare_classes, counts


def main():
    patches, labels = load_training_data()

    train_idx, val_idx, rare_classes, counts = make_split(labels)

    print(f"Total samples: {len(labels)}")
    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    print(f"Rare classes forced into train: {len(rare_classes)}")
    print(f"Min class count: {counts.min()} | #classes count==1: {(counts==1).sum()}")

    # Run predictions on validation set
    y_true = labels[val_idx]
    y_pred = np.array([predict(patches[i]) for i in val_idx], dtype=int)

    f1w = f1_score(y_true, y_pred, average="weighted")
    f1m = f1_score(y_true, y_pred, average="macro")

    print("\n=== Validation Metrics ===")
    print(f"Weighted F1: {f1w:.4f}")
    print(f"Macro F1:    {f1m:.4f}")

    # Optional: show per-class report (can be long)
    # This is useful to see which classes are failing.
    print("\n=== Per-class report (only classes present in val) ===")
    present = np.unique(y_true)
    print(classification_report(y_true, y_pred, labels=present, digits=3))


if __name__ == "__main__":
    main()
