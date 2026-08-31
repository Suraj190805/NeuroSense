"""Huntington's Disease 2D MRI Slice Dataset & Heavy Augmentation Pipeline.

Simulates clinical imaging variability and prevents overfitting via:
1. Random Affine & Rotation: Rotation (±15°), Translation (±10%), Scaling (0.9–1.1).
2. Random Gaussian Blur & Noise: Simulates patient head motion and varying scanner resolutions.
3. Random Color / Contrast Jitter: Simulates slice acquisition exposure differences.
4. Random Erasing / Cutout (p=0.3): Occludes random rectangular patches to force
   the neural network to evaluate distributed brain structures (caudate, putamen,
   ventricles, and cortical mantle) rather than memorizing a single localized feature.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from neurosense.data.harmonizer import harmonize_2d_slice

logger = logging.getLogger(__name__)

CLASS_NAMES = ["normal", "huntington"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def get_2d_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Build heavy data augmentation pipeline for training 2D MRI slices.

    Simulates clinical imaging variability:
    - Random Affine: Rotation (±15°), Translation (±10%), Scaling (0.90–1.10)
    - Random Horizontal Flip: (p=0.5)
    - Random Gaussian Blur: Simulates patient motion artifacts and scanner point-spread functions
    - Random Color Jitter: Brightness (±20%), Contrast (±20%)
    - Random Erasing / Cutout: (p=0.3) Prevents single-feature reliance

    Args:
        image_size: Target square image dimension (default: 224).

    Returns:
        torchvision Compose transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
        ], p=0.4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.20),
            ratio=(0.5, 2.0),
            value=0,
        ),
    ])


def get_2d_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Build validation/test transforms (deterministic resize & normalization only).

    Args:
        image_size: Target square image dimension (default: 224).

    Returns:
        torchvision Compose transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class HDImageDataset(Dataset):
    """Dataset for Huntington's Disease 2D Brain MRI slices.

    Supports group-level scanning and automated intensity harmonization.

    Args:
        root_dir: Directory containing class subfolders ('normal', 'huntington').
        transform: Optional torchvision transform to apply.
        harmonize: Whether to apply CLAHE + Z-score harmonization on load.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: transforms.Compose | None = None,
        harmonize: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.harmonize = harmonize
        self.samples: list[tuple[Path, int]] = []

        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp"):
                for img_path in sorted(class_dir.glob(ext)):
                    self.samples.append((img_path, class_idx))

        logger.info(
            "HDImageDataset: %d samples loaded from %s (harmonize=%s)",
            len(self.samples),
            self.root_dir,
            harmonize,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, label = self.samples[idx]

        if self.harmonize:
            harmonized_2d, _ = harmonize_2d_slice(img_path, target_size=224)
            pil_img = Image.fromarray((harmonized_2d * 255.0).clip(0, 255).astype(np.uint8)).convert("RGB")
        else:
            pil_img = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(pil_img)
        else:
            image = transforms.ToTensor()(pil_img)

        return {
            "image": image,
            "label": label,
            "filename": img_path.name,
        }
