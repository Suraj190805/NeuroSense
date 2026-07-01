"""Parkinson's Disease 2D MRI slice dataset.

Loads 2D brain MRI slices (PNG) from a folder-based dataset
with classes: normal, parkinson.

Structure expected:
    data/parkinsons/parkinsons_dataset/
        normal/
            Mag_Images_001.png
            ...
        parkinson/
            DUAL_TSE_001.png
            ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Class mapping
CLASS_NAMES = ["normal", "parkinson"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class ParkinsonsDataset(Dataset):
    """Dataset for Parkinson's disease classification from 2D MRI slices.

    Each sample is a PNG image of a brain MRI slice, labelled as
    either 'normal' (0) or 'parkinson' (1).

    Args:
        root_dir: Path to the dataset root containing class folders.
        transform: Optional torchvision transform to apply.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        # Walk class folders
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning("Class directory not found: %s", class_dir)
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((img_path, class_idx))

        logger.info(
            "ParkinsonsDataset: %d samples from %s",
            len(self.samples),
            self.root_dir,
        )
        # Log class distribution
        labels = [s[1] for s in self.samples]
        for name, idx in CLASS_TO_IDX.items():
            count = labels.count(idx)
            pct = count / len(labels) * 100 if labels else 0
            logger.info("  %s: %d samples (%.1f%%)", name, count, pct)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, label = self.samples[idx]

        # Load image — convert to RGB (some MRI PNGs may be grayscale)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return {
            "image": image,
            "label": label,
            "filename": img_path.name,
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced loss.

        Returns:
            Tensor of shape [num_classes] with weights.
        """
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        # Inverse frequency: total / (num_classes * count_per_class)
        weights = len(labels) / (len(CLASS_NAMES) * counts.astype(float))
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def get_train_transforms(image_size: int = 224) -> transforms.Compose:
        """Training augmentations for 2D MRI slices.

        Includes random flips, rotations, colour jitter, and affine
        transforms to increase effective dataset size and reduce
        overfitting on the small dataset.

        Args:
            image_size: Target image dimension (square).

        Returns:
            Composed transform pipeline.
        """
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def get_val_transforms(image_size: int = 224) -> transforms.Compose:
        """Validation/test transforms — resize and normalise only.

        Args:
            image_size: Target image dimension (square).

        Returns:
            Composed transform pipeline.
        """
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @classmethod
    def split(
        cls,
        root_dir: str | Path,
        seed: int = 42,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,
        image_size: int = 224,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create stratified train/val/test DataLoaders.

        Uses stratified splitting to maintain class balance across
        all splits, critical for the imbalanced dataset.

        Args:
            root_dir: Dataset root directory.
            seed: Random seed for reproducibility.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of data loading workers.
            pin_memory: Pin memory for CUDA transfers.
            image_size: Target image size.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        # Create dataset with no transforms (we'll apply per-split)
        full_dataset = cls(root_dir=root_dir, transform=None)

        # Stratified split
        rng = np.random.RandomState(seed)
        labels = np.array([s[1] for s in full_dataset.samples])

        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for class_idx in range(len(CLASS_NAMES)):
            class_indices = np.where(labels == class_idx)[0]
            rng.shuffle(class_indices)

            n = len(class_indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_indices.extend(class_indices[:n_train].tolist())
            val_indices.extend(
                class_indices[n_train : n_train + n_val].tolist()
            )
            test_indices.extend(
                class_indices[n_train + n_val :].tolist()
            )

        # Log split info
        logger.info(
            "Dataset split (seed=%d): train=%d, val=%d, test=%d",
            seed,
            len(train_indices),
            len(val_indices),
            len(test_indices),
        )
        for name, idx in CLASS_TO_IDX.items():
            t = sum(1 for i in train_indices if labels[i] == idx)
            v = sum(1 for i in val_indices if labels[i] == idx)
            te = sum(1 for i in test_indices if labels[i] == idx)
            logger.info("  %s: train=%d, val=%d, test=%d", name, t, v, te)

        # Create transforms
        train_tf = cls.get_train_transforms(image_size)
        val_tf = cls.get_val_transforms(image_size)

        # Create subset datasets with appropriate transforms
        train_ds = _TransformSubset(full_dataset, train_indices, train_tf)
        val_ds = _TransformSubset(full_dataset, val_indices, val_tf)
        test_ds = _TransformSubset(full_dataset, test_indices, val_tf)

        # DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader


class _TransformSubset(Dataset):
    """Subset of a dataset with a specific transform applied.

    This allows different transforms for train/val/test splits
    while sharing the same underlying dataset.
    """

    def __init__(
        self,
        dataset: ParkinsonsDataset,
        indices: list[int],
        transform: transforms.Compose,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self.indices[idx]
        img_path, label = self.dataset.samples[real_idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "filename": img_path.name,
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for this subset."""
        labels = [self.dataset.samples[i][1] for i in self.indices]
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        weights = len(labels) / (len(CLASS_NAMES) * counts.astype(float))
        return torch.tensor(weights, dtype=torch.float32)
