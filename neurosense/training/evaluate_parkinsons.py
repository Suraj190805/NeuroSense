"""Parkinson's disease classifier evaluation on held-out test set.

Loads a trained ParkinsonsClassifier checkpoint and evaluates
with comprehensive metrics: AUC-ROC, F1, accuracy, sensitivity,
specificity, confusion matrix.

Usage:
    python -m neurosense.training.evaluate_parkinsons \\
        --checkpoint checkpoints/parkinsons_best.pth \\
        --data-root data/parkinsons/parkinsons_dataset
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from neurosense.data.parkinsons_dataset import (
    CLASS_NAMES,
    ParkinsonsDataset,
)
from neurosense.models.image_classifier import ParkinsonsClassifier
from neurosense.models.mri_encoder import (
    _set_seed,
    _setup_device,
    _load_config,
)

logger = logging.getLogger(__name__)


def _load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ParkinsonsClassifier, dict[str, Any]]:
    """Load ParkinsonsClassifier from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        device: Device to load model onto.

    Returns:
        Tuple of (model, checkpoint_metadata).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    num_classes = checkpoint.get("num_classes", 2)
    model = ParkinsonsClassifier(num_classes=num_classes)

    state_dict = checkpoint.get("model_state_dict", {})
    if state_dict:
        model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded model from %s (epoch %d)",
            checkpoint_path.name,
            checkpoint.get("epoch", -1),
        )

    model = model.to(device)
    model.eval()
    return model, checkpoint


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Parkinson's Classifier — Confusion Matrix",
) -> None:
    """Create a publication-quality confusion matrix heatmap.

    Args:
        cm: Confusion matrix [C, C].
        class_names: Class label names.
        output_path: Output PNG path.
        title: Plot title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm_norm / row_sums * 100

    try:
        import seaborn as sns
        sns.heatmap(
            cm,
            annot=False,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Count"},
            linewidths=0.5,
            linecolor="white",
        )
    except ImportError:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, label="Count")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j + 0.5 if "seaborn" not in dir() else j,
                i + 0.5 if "seaborn" not in dir() else i,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                color=colour,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


@torch.no_grad()
def evaluate_parkinsons(
    checkpoint_path: str | Path,
    data_root: str | Path = "data/parkinsons/parkinsons_dataset",
    output_dir: str | Path = "outputs/reports",
    batch_size: int = 16,
    image_size: int = 224,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate Parkinson's classifier on the held-out test set.

    Computes AUC-ROC, F1, accuracy, sensitivity, specificity,
    confusion matrix, and full classification report.

    Args:
        checkpoint_path: Path to trained checkpoint.
        data_root: Dataset root directory.
        output_dir: Output directory for reports.
        batch_size: Evaluation batch size.
        image_size: Input image size.
        seed: Random seed.

    Returns:
        Dictionary with all computed metrics.
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    _set_seed(seed)
    config = _load_config(None)
    device = _setup_device(config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, ckpt_meta = _load_model(checkpoint_path, device)

    # Load test data
    logger.info("Loading test dataset from %s ...", data_root)
    _, _, test_loader = ParkinsonsDataset.split(
        root_dir=data_root,
        seed=seed,
        batch_size=batch_size,
        num_workers=4,
        image_size=image_size,
    )
    logger.info("Test set: %d batches", len(test_loader))

    # Inference
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    for batch in test_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]

        with autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(images)

        probs = outputs["probabilities"][:, 1].cpu().tolist()
        preds = outputs["predicted_class"].cpu().tolist()

        all_labels.extend(labels.tolist())
        all_preds.extend(preds)
        all_probs.extend(probs)

    # Metrics
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    accuracy = float((preds_arr == labels_arr).mean())

    try:
        auc = float(roc_auc_score(labels_arr, probs_arr))
    except ValueError:
        auc = 0.5

    f1_macro = float(
        f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    )
    f1_per_class = f1_score(
        labels_arr, preds_arr, average=None, zero_division=0
    ).tolist()

    # Sensitivity & specificity
    tp = ((preds_arr == 1) & (labels_arr == 1)).sum()
    tn = ((preds_arr == 0) & (labels_arr == 0)).sum()
    fp = ((preds_arr == 1) & (labels_arr == 0)).sum()
    fn = ((preds_arr == 0) & (labels_arr == 1)).sum()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))

    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])

    cls_report = classification_report(
        labels_arr,
        preds_arr,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "auc": auc,
        "f1_macro": f1_macro,
        "f1_per_class": {
            name: f1 for name, f1 in zip(CLASS_NAMES, f1_per_class)
        },
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
        "n_test_samples": len(labels_arr),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt_meta.get("epoch", -1),
    }

    # Log results
    logger.info(
        "═" * 60 + "\n"
        "  Parkinson's Classifier — Test Results\n"
        "  ─────────────────────────────────────\n"
        "  Accuracy:     %.4f\n"
        "  AUC-ROC:      %.4f\n"
        "  F1 (macro):   %.4f\n"
        "  Sensitivity:  %.4f\n"
        "  Specificity:  %.4f\n"
        "  ─────────────────────────────────────\n"
        "  Samples:      %d\n"
        + "═" * 60,
        accuracy,
        auc,
        f1_macro,
        sensitivity,
        specificity,
        len(labels_arr),
    )

    # Save outputs
    metrics_path = output_dir / "parkinsons_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    cm_path = output_dir / "parkinsons_confusion_matrix.png"
    _plot_confusion_matrix(
        cm=cm,
        class_names=CLASS_NAMES,
        output_path=cm_path,
    )

    report_path = output_dir / "parkinsons_classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Parkinson's Classifier — Test Set Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {ckpt_meta.get('epoch', 'N/A')}\n")
        f.write(f"Test samples: {len(labels_arr)}\n\n")
        f.write(cls_report)
        f.write(f"\nAUC-ROC:     {auc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write("\n\nConfusion Matrix:\n")
        f.write(f"{'':>15}")
        for name in CLASS_NAMES:
            f.write(f"{name:>15}")
        f.write("\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{name:>15}")
            for j in range(len(CLASS_NAMES)):
                f.write(f"{cm[i, j]:>15d}")
            f.write("\n")
    logger.info("Report saved to %s", report_path)

    return metrics


def main() -> None:
    """CLI entry point for Parkinson's model evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Parkinson's classifier on test set.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pth)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/parkinsons/parkinsons_dataset",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    metrics = evaluate_parkinsons(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    print(f"\n{'='*50}")
    print("Parkinson's Classifier — Test Results:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Samples:     {metrics['n_test_samples']}")
    print(f"\n  Outputs: {args.output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
