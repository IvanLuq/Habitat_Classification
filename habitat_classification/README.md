# Icelandic Habitat Classification

Classify Icelandic landscapes from satellite imagery!

![Fjallahveravist - Geothermal Alpine Habitat](photo_fjallahveravist.jpg)

*Fjallahveravist (geothermal alpine habitat) - Photo: Náttúrufræðistofnun Íslands*

Iceland has one of the most detailed habitat mapping systems in the world, with **71 habitat types** (*vistgerðir*) grouped into **13 habitat categories** (*vistlendi*). Your task is to build a model that identifies these habitats from satellite images.

Each image is a 35×35 pixel patch captured by the Sentinel-2 satellite, combined with terrain data from the Icelandic national elevation model. The patches cover a 350×350 meter area of land.

## The Challenge

Given a satellite image patch, predict which of the 71 habitat types (*vistgerð*) it belongs to.

**Input:** A numpy array of shape `(15, 35, 35)` containing:

| Channels | Description |
|----------|-------------|
| 0-11 | Sentinel-2 spectral bands (coastal aerosol through SWIR) |
| 12 | Elevation (meters above sea level) |
| 13 | Slope (degrees) |
| 14 | Aspect (compass direction of slope) |

**Output:** An integer from 0 to 70 representing the predicted habitat type (*vistgerð*).

### Example: Satellite Patch

Here's what a satellite patch looks like for the Fjallahveravist habitat shown above:

![Satellite Example](example.png)

## Data

The data comes from summer (July-August) satellite imagery over Iceland, combined with high-resolution terrain data.

| Dataset | Samples | Purpose |
|---------|---------|---------|
| Training | 5,186 | Build your model |
| Validation | 799 | Test during competition |
| Test | 1,998 | Final evaluation |

The validation and test sets were split to maintain similar class distributions as training.

**Training data:** `data/train/patches.npy` and `data/train.csv`

```python
from utils import load_training_data

patches, labels = load_training_data()
print(f"Patches: {patches.shape}")  # (5186, 15, 35, 35)
print(f"Labels: {labels.shape}")    # (5186,)
```

### All 71 Habitat Types

Here's a sample satellite image from each of the 71 habitat types:

![All Classes RGB](all_classes_rgb.png)

## Scoring

Your model is scored using **Weighted F1 Score**:

```
F1_weighted = Σ (n_c / N) × F1_c
```

Where `n_c` is the number of samples in class `c` and `N` is the total samples.

The baseline (random stratified sampling) achieves ~4% weighted F1.

## Our CNN Solution

The current training pipeline is implemented in `train_cnn.py` and uses only the original 15 input channels.

### 1) Split strategy for rare classes

- We use an 85/15 train/validation split with `StratifiedShuffleSplit` for classes that have at least 2 samples.
- Classes with fewer than 2 samples are forced into the training set so validation stays stable.

### 2) Preprocessing and normalization

- Input shape is `(15, 35, 35)`.
- Per-channel mean and std are computed from training samples only to avoid leakage.
- Saved normalization file: `weights/channel_mean.npy`
- Saved normalization file: `weights/channel_std.npy`

### 3) Data augmentation

Applied only on training batches:

- Random horizontal flip
- Random vertical flip
- Random 90 degree rotation (`k in {0,1,2,3}`)

### 4) CNN architecture

`HabitatCNN` is a compact residual CNN:

- Stem: 2 x Conv-BN-SiLU blocks (15 -> 32)
- Down block 1: strided conv (32 -> 64) + residual block
- Down block 2: strided conv (64 -> 128) + 2 residual blocks
- Head: global average pooling + MLP (128 -> 128 -> 71)

### 5) Imbalance handling

To handle long-tail class imbalance, we combine three techniques:

- `WeightedRandomSampler` with inverse-frequency sample weights
- Class-weighted cross-entropy loss
- Logit adjustment with class priors (`tau = 0.5`)

### 6) Optimization

- Optimizer: `AdamW` (`lr=3e-4`, `weight_decay=1e-3`)
- Scheduler: cosine annealing (`eta_min=1e-6`)
- Epochs: 30
- Batch size: 64 (train), 256 (validation)

### 7) Model selection and artifacts

- Validation metric: weighted F1
- Best checkpoint is saved as `weights/habitat_cnn.pt`
- Normalization stats are saved with the checkpoint for inference consistency

### 8) Reproduce training and evaluation

```bash
# Train CNN and save best model + normalization stats
python train_cnn.py

# Evaluate saved model on the validation split
python eval_saved.py

# Quick sanity check on random training patches
python quick_test.py
```

## Evaluation

During the competition, you can validate against the validation set multiple times. **You can only submit to the test set once!**

Your endpoint must respond within **10 seconds per image**.

## Getting Started

### 1. Setup

Clone the repo and create a virtual environment:

```bash
git clone <repo-url>
cd habitat-classification

# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the CNN Model

Run the training script to produce the CNN weights and channel normalization files:

```bash
python train_cnn.py
```

### 3. Evaluate the Saved CNN

```bash
python eval_saved.py
```

### 4. Test Locally

Start your API locally to test:

```bash
python api.py
```

Navigate to http://localhost:4321 to verify it's running.

### 5. Submit

When ready to submit:

1. Create a virtual machine (Azure, AWS, GCP, or similar)
2. Clone your repo on the VM
3. Run `python api.py` on the VM
4. Go to the submission portal
5. Enter your VM's IP address and API key
6. Press submit and wait for your score

See the main competition README for detailed submission instructions.

## About the Data

The satellite imagery was extracted from Google Earth Engine:

- **Sentinel-2 Level 2A** surface reflectance (12 spectral bands)
- **Cloud Score Plus** filtering (threshold 0.6)
- **Summer median composite** (July-August 2023-2025)
- **IslandsDEM v1** for terrain features

The habitat labels come from the Icelandic Institute of Natural History (*Náttúrufræðistofnun Íslands*).

Good luck! 🇮🇸
