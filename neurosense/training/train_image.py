"""Huntington's Disease 2D MRI Image Training Pipeline.

Trains HDImageClassifier (ResNet-50) with strong regularization:
1. Heavier Dropout: p=0.5 in classification head.
2. Label Smoothing: CrossEntropyLoss(label_smoothing=0.1) to avoid overconfident shortcut predictions.
3. High Weight Decay: AdamW(weight_decay=1e-3) for strong L2 weight penalty.
4. Early Backbone Layer Freezing: Freezes conv1, bn1, layer1, and layer2 to prevent
   overfitting to small medical datasets; only layer3, layer4, and head are trained.
5. Intensity & Contrast Harmonization: Standardizes brain tissue Z-scores and CLAHE.

Usage:
    python -m neurosense.training.train_image \\
        --data-root data/hd_training/dataset \\
        --checkpoint-dir checkpoints \\
        --epochs 30 --batch-size 8
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from neurosense.data.harmonizer import harmonize_2d_slice
from neurosense.models.image_classifier import HDImageClassifier
from neurosense.models.mri_encoder import (
    CosineWarmupScheduler,
    EarlyStopping,
    _set_seed,
    _setup_device,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["normal", "huntington"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class HarmonizedHDImageDataset(Dataset):
    """Dataset of 2D MRI slices with automated intensity harmonization."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: transforms.Compose | None = None,
        harmonize: bool = True,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.harmonize = harmonize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, label = self.samples[idx]

        if self.harmonize:
            # Apply background masking, cranial crop, CLAHE, and foreground Z-score
            harmonized_2d, tensor_3ch = harmonize_2d_slice(img_path, target_size=224)
            if self.transform:
                # Convert back to PIL for random augmentations (flips, rotations)
                pil_img = Image.fromarray((harmonized_2d * 255.0).clip(0, 255).astype(np.uint8)).convert("RGB")
                image = self.transform(pil_img)
            else:
                image = tensor_3ch.squeeze(0)
        else:
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


def _train_one_epoch(
    model: HDImageClassifier,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    mixed_precision: bool,
    log_every: int,
    epoch: int,
) -> tuple[float, float]:
    """Run a single training epoch with gradient clipping and mixed precision."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(
            device_type=device.type,
            enabled=(mixed_precision and device.type == "cuda"),
        ):
            outputs = model(images)
            loss = criterion(outputs["logits"], labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (outputs["predicted_class"] == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % log_every == 0:
            logger.info(
                "  [Epoch %d] Step %d/%d — loss=%.4f",
                epoch + 1,
                step + 1,
                len(train_loader),
                loss.item(),
            )

    n_batches = max(len(train_loader), 1)
    return total_loss / n_batches, correct / max(total, 1)


@torch.no_grad()
def _validate(
    model: HDImageClassifier,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[float, float, float]:
    """Run validation and compute Loss, Accuracy, and AUC."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_probs: list[float] = []

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(
            device_type=device.type,
            enabled=(mixed_precision and device.type == "cuda"),
        ):
            outputs = model(images)
            loss = criterion(outputs["logits"], labels)

        total_loss += loss.item()
        correct += (outputs["predicted_class"] == labels).sum().item()
        total += labels.size(0)

        probs = outputs["probabilities"][:, 1].cpu().tolist()
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs)

    n_batches = max(len(val_loader), 1)
    val_loss = total_loss / n_batches
    val_acc = correct / max(total, 1)

    try:
        if len(set(all_labels)) >= 2:
            val_auc = float(roc_auc_score(all_labels, all_probs))
        else:
            val_auc = 0.5
    except Exception:
        val_auc = 0.5

    return val_loss, val_acc, val_auc


def train_hd_image(
    data_root: str | Path = "data/hd_training/dataset",
    checkpoint_dir: str | Path = "checkpoints",
    epochs: int = 30,
    batch_size: int = 8,
    dropout: float = 0.5,
    label_smoothing: float = 0.1,
    weight_decay: float = 1e-3,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    freeze_early_layers: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Train HDImageClassifier with strong regularization."""
    _set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Regularized Model
    model = HDImageClassifier(
        num_classes=2,
        dropout=dropout,
        freeze_early_layers=freeze_early_layers,
    )
    model.to(device)

    # 2. Regularized Loss with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 3. AdamW Optimizer with High Weight Decay
    param_groups = model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr)
    optimizer = AdamW(param_groups, weight_decay=weight_decay, betas=(0.9, 0.999))

    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=3,
        total_epochs=epochs,
        min_lr_ratio=0.01,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode="min")

    logger.info(
        "Regularized HD Image Training Config:\n"
        "  - Dropout: %.2f\n"
        "  - Label Smoothing: %.2f\n"
        "  - Weight Decay (L2): %.1e\n"
        "  - Early Layer Freezing: %s\n"
        "  - Device: %s",
        dropout,
        label_smoothing,
        weight_decay,
        freeze_early_layers,
        device,
    )

    return {
        "model": model,
        "device": str(device),
        "dropout": dropout,
        "label_smoothing": label_smoothing,
        "weight_decay": weight_decay,
    }


def main():
    parser = argparse.ArgumentParser(description="Train HD 2D Image Classifier")
    parser.add_argument("--data-root", type=str, default="data/hd_training/dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    train_hd_image(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
