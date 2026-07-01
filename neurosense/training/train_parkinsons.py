"""Parkinson's disease 2D MRI training pipeline.

Trains a ResNet-50 classifier on 2D brain MRI slices for binary
classification: normal vs parkinson.

Usage:
    python -m neurosense.training.train_parkinsons \\
        --data-root data/parkinsons/parkinsons_dataset \\
        --checkpoint-dir checkpoints \\
        --epochs 30

API:
    from neurosense.training.train_parkinsons import train_parkinsons
    results = train_parkinsons(data_root="data/parkinsons/parkinsons_dataset")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neurosense.data.parkinsons_dataset import (
    CLASS_NAMES,
    ParkinsonsDataset,
)
from neurosense.models.image_classifier import ParkinsonsClassifier
from neurosense.models.mri_encoder import (
    CosineWarmupScheduler,
    EarlyStopping,
    _set_seed,
    _setup_device,
    _load_config,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Training Step Helpers
# ═════════════════════════════════════════════════════════════════


def _train_one_epoch(
    model: ParkinsonsClassifier,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    mixed_precision: bool,
    log_every: int,
    epoch: int,
) -> tuple[float, float]:
    """Run a single training epoch.

    Args:
        model: ParkinsonsClassifier instance.
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for mixed precision.
        device: Compute device.
        mixed_precision: Whether to use AMP.
        log_every: Log frequency in steps.
        epoch: Current epoch number (0-indexed).

    Returns:
        Tuple of (epoch_loss, epoch_accuracy).
    """
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
    model: ParkinsonsClassifier,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[float, float, dict[str, float]]:
    """Run validation and compute AUC-ROC.

    Args:
        model: ParkinsonsClassifier instance.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Compute device.
        mixed_precision: Whether to use AMP.

    Returns:
        Tuple of (val_loss, val_accuracy, val_aucs).
    """
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

        # Collect for AUC — probability of parkinson class
        probs = outputs["probabilities"][:, 1].cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().tolist())

    n_batches = max(len(val_loader), 1)
    avg_loss = total_loss / n_batches
    avg_acc = correct / max(total, 1)

    # Compute AUC-ROC
    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.5  # Only one class present

    # Sensitivity and specificity
    labels_arr = np.array(all_labels)
    preds_arr = (np.array(all_probs) >= 0.5).astype(int)
    tp = ((preds_arr == 1) & (labels_arr == 1)).sum()
    tn = ((preds_arr == 0) & (labels_arr == 0)).sum()
    fp = ((preds_arr == 1) & (labels_arr == 0)).sum()
    fn = ((preds_arr == 0) & (labels_arr == 1)).sum()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    aucs = {
        "auc": auc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }

    return avg_loss, avg_acc, aucs


# ═════════════════════════════════════════════════════════════════
#  Main Training Function
# ═════════════════════════════════════════════════════════════════


def train_parkinsons(
    data_root: str | Path = "data/parkinsons/parkinsons_dataset",
    checkpoint_dir: str | Path = "checkpoints",
    max_epochs: int = 30,
    batch_size: int = 16,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 3,
    patience: int = 8,
    image_size: int = 224,
    dropout: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the ParkinsonsClassifier on 2D MRI slices.

    Args:
        data_root: Root directory with normal/ and parkinson/ folders.
        checkpoint_dir: Directory for saving checkpoints.
        max_epochs: Maximum training epochs.
        batch_size: Training batch size.
        backbone_lr: Learning rate for pretrained backbone.
        head_lr: Learning rate for classification head.
        weight_decay: Weight decay for AdamW.
        warmup_epochs: LR warmup epochs.
        patience: Early stopping patience.
        image_size: Input image size (square).
        dropout: Dropout rate in classifier head.
        seed: Random seed.

    Returns:
        Dictionary with training results.
    """
    _set_seed(seed)

    # Device
    config = _load_config(None)
    device = _setup_device(config)

    # Data
    data_root = Path(data_root)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Parkinson's dataset from %s ...", data_root)
    train_loader, val_loader, test_loader = ParkinsonsDataset.split(
        root_dir=data_root,
        seed=seed,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        image_size=image_size,
    )
    logger.info(
        "Data loaded: %d train batches, %d val batches, %d test batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    # Class weights for imbalanced dataset
    train_dataset = train_loader.dataset
    class_weights = None
    if hasattr(train_dataset, "get_class_weights"):
        class_weights = train_dataset.get_class_weights().to(device)
        logger.info("Class weights: %s", class_weights.tolist())

    # Model
    model = ParkinsonsClassifier(
        num_classes=2,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        "Model: %dM total params (%dM trainable)",
        total_params // 1_000_000,
        trainable_params // 1_000_000,
    )

    # Optimizer with differential LR
    param_groups = model.get_param_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
    )
    optimizer = AdamW(
        param_groups,
        weight_decay=weight_decay,
    )
    logger.info(
        "Optimizer: AdamW — backbone lr=%.2e, head lr=%.2e",
        backbone_lr,
        head_lr,
    )

    # Scheduler
    min_lr_ratio = 1e-6 / backbone_lr
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=max_epochs,
        min_lr_ratio=min_lr_ratio,
    )

    # Loss — weighted cross-entropy for class imbalance
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )
    logger.info("Loss: WeightedCrossEntropy (smoothing=0.05)")

    # Mixed precision
    scaler = GradScaler(
        device.type,
        enabled=(device.type == "cuda"),
    )

    # Early stopping
    early_stopper = EarlyStopping(
        patience=patience,
        min_delta=0.001,
        mode="max",
        verbose=True,
    )

    # Training header
    logger.info(
        "═" * 60 + "\n"
        "  Parkinson's Disease Classifier Training\n"
        "  Classes: normal vs parkinson\n"
        "  Epochs: %d | Batch: %d | Image: %d×%d\n"
        "  LR: backbone=%.2e, head=%.2e | Device: %s\n"
        "  Patience: %d | Dropout: %.1f\n"
        + "═" * 60,
        max_epochs,
        batch_size,
        image_size,
        image_size,
        backbone_lr,
        head_lr,
        device,
        patience,
        dropout,
    )

    results: dict[str, Any] = {
        "best_val_auc": 0.0,
        "best_epoch": 0,
        "final_train_loss": float("inf"),
        "total_epochs": 0,
        "checkpoint_path": "",
        "val_metrics": {},
    }
    best_val_metrics: dict[str, float] = {}

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixed_precision=(device.type == "cuda"),
            log_every=10,
            epoch=epoch,
        )

        # Validate
        val_loss, val_acc, val_aucs = _validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            mixed_precision=(device.type == "cuda"),
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # Log
        logger.info(
            "Epoch %d/%d (%.1fs) — "
            "train: loss=%.4f acc=%.3f | "
            "val: loss=%.4f acc=%.3f | "
            "AUC=%.3f sens=%.3f spec=%.3f | "
            "lr=%.2e",
            epoch + 1,
            max_epochs,
            epoch_time,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_aucs["auc"],
            val_aucs["sensitivity"],
            val_aucs["specificity"],
            current_lr,
        )

        # Early stopping on AUC
        val_auc = val_aucs["auc"]
        early_stopper(val_auc, model, epoch)

        results["final_train_loss"] = train_loss
        results["total_epochs"] = epoch + 1

        if early_stopper.best_epoch == epoch:
            best_val_metrics = val_aucs.copy()
            best_val_metrics["accuracy"] = val_acc
            best_val_metrics["loss"] = val_loss

        if early_stopper.should_stop:
            logger.info(
                "Early stopping at epoch %d. "
                "Best AUC=%.4f at epoch %d.",
                epoch + 1,
                early_stopper.best_score,
                early_stopper.best_epoch + 1,
            )
            break

    # Save checkpoints
    best_ckpt_path = checkpoint_dir / "parkinsons_best.pth"
    last_ckpt_path = checkpoint_dir / "parkinsons_last.pth"

    # Best checkpoint
    if early_stopper.best_state_dict is not None:
        current_state = model.state_dict()
        model.load_state_dict(early_stopper.best_state_dict)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": early_stopper.best_epoch,
                "val_metrics": best_val_metrics,
                "num_classes": 2,
                "class_names": CLASS_NAMES,
                "image_size": image_size,
            },
            best_ckpt_path,
        )
        logger.info("Best checkpoint saved to %s", best_ckpt_path)
        model.load_state_dict(current_state)
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": results["total_epochs"] - 1,
                "val_metrics": val_aucs,
                "num_classes": 2,
                "class_names": CLASS_NAMES,
                "image_size": image_size,
            },
            best_ckpt_path,
        )
        logger.info("Checkpoint saved to %s", best_ckpt_path)

    # Last checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": results["total_epochs"] - 1,
            "val_metrics": val_aucs,
            "num_classes": 2,
            "class_names": CLASS_NAMES,
            "image_size": image_size,
        },
        last_ckpt_path,
    )
    logger.info("Last checkpoint saved to %s", last_ckpt_path)

    # Final results
    results["best_val_auc"] = early_stopper.best_score or 0.0
    results["best_epoch"] = early_stopper.best_epoch
    results["checkpoint_path"] = str(best_ckpt_path)
    results["val_metrics"] = best_val_metrics

    logger.info(
        "═" * 60 + "\n"
        "  Training Complete\n"
        "  Best AUC: %.4f (epoch %d)\n"
        "  Total epochs: %d\n"
        "  Checkpoint: %s\n"
        + "═" * 60,
        results["best_val_auc"],
        results["best_epoch"] + 1,
        results["total_epochs"],
        results["checkpoint_path"],
    )

    return results


# ═════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for Parkinson's classifier training."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Train Parkinson's disease classifier on 2D MRI slices. "
            "Uses pretrained ResNet-50 with custom head."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/parkinsons/parkinsons_dataset",
        help="Root directory with normal/ and parkinson/ folders",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-4,
        help="Backbone learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
        help="Head learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = train_parkinsons(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        image_size=args.image_size,
        patience=args.patience,
        seed=args.seed,
    )

    print(f"\n{'='*50}")
    print("Parkinson's Classifier — Training Results:")
    print(f"  Best AUC:     {results['best_val_auc']:.4f}")
    print(f"  Best Epoch:   {results['best_epoch'] + 1}")
    print(f"  Total Epochs: {results['total_epochs']}")
    print(f"  Checkpoint:   {results['checkpoint_path']}")
    if results.get("val_metrics"):
        print("  Val Metrics:")
        for key, val in results["val_metrics"].items():
            if isinstance(val, float):
                print(f"    {key}: {val:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
