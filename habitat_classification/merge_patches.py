import numpy as np
from pathlib import Path

train_dir = Path("data/train")

p1 = np.load(train_dir / "patches_part1.npy")
p2 = np.load(train_dir / "patches_part2.npy")

patches = np.concatenate([p1, p2], axis=0)

print("Merged shape:", patches.shape)

np.save(train_dir / "patches.npy", patches)
print("Saved data/train/patches.npy")
