# model.py
# Inference-time model + preprocessing that MATCHES the enriched training pipeline:
# (15,35,35) input -> enrich to (20,35,35) -> normalize with saved mean/std -> CNN -> class index

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# =========================
# Paths
# =========================

_THIS_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _THIS_DIR / "weights"
_MODEL_PATH = _WEIGHTS_DIR / "habitat_cnn.pt"
_MEAN_PATH = _WEIGHTS_DIR / "channel_mean.npy"
_STD_PATH = _WEIGHTS_DIR / "channel_std.npy"


# =========================
# Feature enrichment (LOCAL - does not touch utils.py)
# =========================

# Assumed Sentinel-2 ordering for channels 0..11 (common):
# 0:B1, 1:B2(Blue), 2:B3(Green), 3:B4(Red), 4:B5, 5:B6, 6:B7,
# 7:B8(NIR), 8:B8A, 9:B9, 10:B11(SWIR1), 11:B12(SWIR2)
B_BLUE = 1
B_GREEN = 2
B_RED = 3
B_NIR = 7
B_SWIR1 = 10


def _safe_norm_diff(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (a - b) / (a + b + eps)


def enrich_patch(patch: np.ndarray) -> np.ndarray:
    """
    Input:  (15,35,35) = 12 spectral + elev + slope + aspect
    Output: (20,35,35) = 12 spectral + elev + slope + aspect_sin + aspect_cos + NDVI + NDWI + NDBI + NDMI
    """
    if patch.shape != (15, 35, 35):
        raise ValueError(f"Expected patch shape (15,35,35), got {patch.shape}")

    x = patch.astype(np.float32)
    spec = x[0:12]    # (12,35,35)
    elev = x[12:13]   # (1,35,35)
    slope = x[13:14]  # (1,35,35)
    aspect = x[14]    # (35,35)

    # degrees vs radians auto-detect
    if np.nanmax(aspect) > 6.5:  # > ~2π
        aspect_rad = aspect * (np.pi / 180.0)
    else:
        aspect_rad = aspect

    aspect_sin = np.sin(aspect_rad)[None, ...].astype(np.float32)
    aspect_cos = np.cos(aspect_rad)[None, ...].astype(np.float32)

    blue = spec[B_BLUE]
    green = spec[B_GREEN]
    red = spec[B_RED]
    nir = spec[B_NIR]
    swir1 = spec[B_SWIR1]

    ndvi = _safe_norm_diff(nir, red)[None, ...].astype(np.float32)
    ndwi = _safe_norm_diff(green, nir)[None, ...].astype(np.float32)
    ndbi = _safe_norm_diff(swir1, nir)[None, ...].astype(np.float32)
    ndmi = _safe_norm_diff(nir, swir1)[None, ...].astype(np.float32)

    out = np.concatenate(
        [spec, elev, slope, aspect_sin, aspect_cos, ndvi, ndwi, ndbi, ndmi],
        axis=0
    )
    return out  # (20,35,35)


# =========================
# Model (MUST match train_cnn.py)
# =========================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        self.c1 = ConvBNAct(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x)
        y = self.bn2(self.c2(y))
        y = self.drop(y)
        return self.act(x + y)


class HabitatCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 71, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 32, 3, 1, 1),
            ConvBNAct(32, 32, 3, 1, 1),
        )
        self.down1 = nn.Sequential(
            ConvBNAct(32, 64, 3, 2, 1),  # 35 -> 18
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
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.pool(x)
        return self.head(x)


# =========================
# Lazy-loaded globals
# =========================

_MODEL: Optional[HabitatCNN] = None
_CH_MEAN: Optional[np.ndarray] = None
_CH_STD: Optional[np.ndarray] = None
_DEVICE = torch.device("cpu")


def _lazy_init() -> None:
    global _MODEL, _CH_MEAN, _CH_STD

    if _MODEL is not None:
        return

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights: {_MODEL_PATH}")
    if not _MEAN_PATH.exists() or not _STD_PATH.exists():
        raise FileNotFoundError(f"Missing normalization stats: {_MEAN_PATH} / {_STD_PATH}")

    _CH_MEAN = np.load(_MEAN_PATH).astype(np.float32)
    _CH_STD = np.load(_STD_PATH).astype(np.float32)

    in_channels = int(_CH_MEAN.shape[0])
    if _CH_STD.shape[0] != in_channels:
        raise ValueError(f"Mean/std channel mismatch: mean={_CH_MEAN.shape}, std={_CH_STD.shape}")

    # Build model matching train_cnn.py
    model = HabitatCNN(in_channels=in_channels, num_classes=71, dropout=0.0)
    state = torch.load(_MODEL_PATH, map_location="cpu")

    # Strict load: will error if architecture doesn't match training (good!)
    model.load_state_dict(state, strict=True)

    model.eval()
    _MODEL = model


@torch.inference_mode()
def predict(patch: np.ndarray) -> int:
    """
    patch: np.ndarray float/uint, shape (15,35,35)
    returns: int class index in [0,70]
    """
    _lazy_init()
    assert _MODEL is not None and _CH_MEAN is not None and _CH_STD is not None

    raw = patch.astype(np.float32)
    x = enrich_patch(raw)  # (20,35,35)

    # normalize
    x = (x - _CH_MEAN[:, None, None]) / _CH_STD[:, None, None]

    # torch: (1,C,H,W)
    xt = torch.from_numpy(x).unsqueeze(0)
    logits = _MODEL(xt)
    pred = int(torch.argmax(logits, dim=1).item())
    return pred
