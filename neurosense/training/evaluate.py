"""NeuroSense test-set evaluation with comprehensive metrics.

Loads a trained NeuroSenseModel checkpoint and evaluates on the
held-out test set, producing:
- Per-class AUC-ROC (pre-manifest, early, advanced) + macro mean
- Per-class F1 score + macro F1
- Overall accuracy
- Confusion matrix (saved as publication-quality heatmap)
- Full classification report

All outputs are saved to the ``outputs/`` directory.

Usage:
    from neurosense.training.evaluate import evaluate_model

    metrics = evaluate_model(
        checkpoint_path="checkpoints/neurosense_best.pth",
        data_root="data/processed",
    )

CLI:
    python -m neurosense.training.evaluate \\
        --checkpoint checkpoints/neurosense_best.pth \\
        --data-root data/processed
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from neurosense.data.dataset import STAGE_NAMES, HuntingtonDataset
from neurosense.models.classifier import NeuroSenseModel
from neurosense.models.mri_encoder import (
    _compute_per_class_auc,
    _load_config,
    _set_seed,
    _setup_device,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Model Loading
# ═════════════════════════════════════════════════════════════════


def _load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> tuple[NeuroSenseModel, dict[str, Any]]:
    """Reconstruct and load NeuroSenseModel from checkpoint.

    Args:
        checkpoint_path: Path to ``.pth`` checkpoint file.
        device: Device to load model onto.
        config: Optional config dict override. If None, uses
            config stored in the checkpoint.

    Returns:
        Tuple of (loaded_model, checkpoint_metadata).

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    # Extract model configuration from checkpoint
    model_cfg = checkpoint.get("model_config", {})
    embed_dim = model_cfg.get("embed_dim", 256)
    use_mri = model_cfg.get("use_mri", True)
    use_clinical = model_cfg.get("use_clinical", True)
    fusion_type = model_cfg.get("fusion_type", "cross_attention")

    # Build model kwargs from stored or provided config
    cfg = config or checkpoint.get("training_config", {})
    mri_cfg = cfg.get("mri_encoder", {})
    clin_cfg = cfg.get("clinical_encoder", {})
    fusion_cfg = cfg.get("fusion", {})

    model_kwargs: dict[str, Any] = {
        "embed_dim": embed_dim,
        "use_mri": use_mri,
        "use_clinical": use_clinical,
        "fusion_type": fusion_type,
    }

    if use_mri:
        model_kwargs["mri_encoder_kwargs"] = {
            "backbone_out_dim": mri_cfg.get("backbone_out_dim", 2048),
        }

    if use_clinical:
        model_kwargs["clinical_encoder_kwargs"] = {
            "input_features": clin_cfg.get("input_features", 5),
            "hidden_size": clin_cfg.get("hidden_size", 256),
            "num_layers": clin_cfg.get("num_layers", 2),
            "dropout": clin_cfg.get("dropout", 0.3),
        }

    if use_mri and use_clinical:
        model_kwargs["fusion_kwargs"] = {
            "num_heads": fusion_cfg.get("num_heads", 8),
            "ffn_hidden_dim": fusion_cfg.get("ffn_hidden_dim", 1024),
        }

    model = NeuroSenseModel(**model_kwargs)

    # Load state dict
    state_dict = checkpoint.get("model_state_dict", {})
    if state_dict:
        model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded model weights from %s (epoch %d)",
            checkpoint_path.name,
            checkpoint.get("epoch", -1),
        )
    else:
        logger.warning(
            "No model_state_dict found in checkpoint %s",
            checkpoint_path,
        )

    model = model.to(device)
    model.eval()

    return model, checkpoint


# ═════════════════════════════════════════════════════════════════
#  Confusion Matrix Visualisation
# ═════════════════════════════════════════════════════════════════


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "NeuroSense — Confusion Matrix",
) -> None:
    """Create a publication-quality confusion matrix heatmap.

    Uses matplotlib and optionally seaborn for styling. The matrix
    shows both raw counts and percentages per true class.

    Args:
        cm: Confusion matrix array of shape ``[C, C]``.
        class_names: List of class label names.
        output_path: Path to save the figure (PNG).
        title: Plot title.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed — skipping confusion matrix plot"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalise for percentages (per row = per true class)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    cm_pct = cm_norm / row_sums * 100

    # Try seaborn for better aesthetics
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
        # Fallback to plain matplotlib
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, label="Count")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    # Annotate cells with count and percentage
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
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xlabel("Predicted Stage", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Stage", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Confusion matrix saved to %s", output_path)


# ═════════════════════════════════════════════════════════════════
#  Main Evaluation Function
# ═════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_model(
    checkpoint_path: str | Path,
    data_root: str | Path = "data/processed",
    output_dir: str | Path = "outputs",
    config_path: str | Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate a trained NeuroSenseModel on the held-out test set.

    Computes comprehensive metrics and saves results:
    - ``eval_metrics.json`` — numeric metrics
    - ``confusion_matrix.png`` — visualisation
    - ``classification_report.txt`` — sklearn text report

    Args:
        checkpoint_path: Path to trained model checkpoint.
        data_root: Root directory of BIDS-format dataset.
        output_dir: Directory for saving evaluation outputs.
        config_path: Path to train_config.yaml override.
        batch_size: Override batch size (uses config default if None).

    Returns:
        Dictionary containing all computed metrics:
        - accuracy: Overall test accuracy
        - auc_premanifest, auc_early, auc_advanced, auc_mean: AUC-ROC
        - f1_macro: Macro-averaged F1 score
        - f1_per_class: Per-class F1 scores
        - confusion_matrix: Raw confusion matrix as nested list

    Example:
        >>> metrics = evaluate_model(
        ...     "checkpoints/neurosense_best.pth",
        ...     data_root="data/processed",
        ... )
        >>> print(f"Test AUC: {metrics['auc_mean']:.4f}")
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
    )

    # ─── Configuration ───
    config = _load_config(config_path)
    repro_cfg = config.get("reproducibility", {})
    seed = repro_cfg.get("seed", 42)
    _set_seed(seed)

    device = _setup_device(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bs = batch_size or config.get("training", {}).get("batch_size", 4)
    num_workers = config.get("data", {}).get("num_workers", 4)
    mixed_precision = config.get("training", {}).get(
        "mixed_precision", True
    )

    # ─── Load Model ───
    model, ckpt_meta = _load_model_from_checkpoint(
        checkpoint_path, device, config
    )

    # ─── Create Test DataLoader ───
    logger.info("Loading test dataset from %s ...", data_root)
    _, _, test_loader = HuntingtonDataset.split(
        root_dir=data_root,
        seed=seed,
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info("Test set: %d batches", len(test_loader))

    # ─── Inference ───
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    all_risks: list[str] = []

    for batch in test_loader:
        mri = batch["mri"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        labels = batch["label"]

        if clinical.ndim == 2:
            clinical = clinical.unsqueeze(1)

        with autocast(
            device_type=device.type,
            enabled=(mixed_precision and device.type == "cuda"),
        ):
            # Only pass inputs for enabled modalities
            forward_kwargs: dict[str, Any] = {}
            if model.use_mri:
                forward_kwargs["mri"] = mri
            if model.use_clinical:
                forward_kwargs["clinical"] = clinical

            outputs = model(**forward_kwargs)

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().tolist()

        all_labels.extend(labels.tolist())
        all_preds.extend(preds)
        all_probs.append(probs)
        all_risks.extend(outputs["risk"])

    # ─── Compute Metrics ───
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.concatenate(all_probs, axis=0)

    # Accuracy
    accuracy = float((preds_arr == labels_arr).mean())

    # Per-class AUC-ROC
    aucs = _compute_per_class_auc(labels_arr, probs_arr)

    # F1 scores
    f1_macro = float(
        f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    )
    f1_per_class = f1_score(
        labels_arr, preds_arr, average=None, zero_division=0
    ).tolist()

    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1, 2])

    # Classification report
    cls_report = classification_report(
        labels_arr,
        preds_arr,
        target_names=STAGE_NAMES,
        zero_division=0,
    )

    # ─── Compile Results ───
    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": {
            name: f1 for name, f1 in zip(STAGE_NAMES, f1_per_class)
        },
        "confusion_matrix": cm.tolist(),
        "n_test_samples": len(labels_arr),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt_meta.get("epoch", -1),
        **aucs,
    }

    # ─── Log Results ───
    logger.info(
        "═" * 60 + "\n"
        "  Test Evaluation Results\n"
        "  ─────────────────────────────────────\n"
        "  Accuracy:  %.4f\n"
        "  F1 macro:  %.4f\n"
        "  AUC mean:  %.4f\n"
        "    pre-manifest:  %.4f\n"
        "    early:         %.4f\n"
        "    advanced:      %.4f\n"
        "  ─────────────────────────────────────\n"
        "  Samples:   %d\n"
        + "═" * 60,
        accuracy,
        f1_macro,
        aucs.get("auc_mean", 0.0),
        aucs.get("auc_premanifest", 0.0),
        aucs.get("auc_early", 0.0),
        aucs.get("auc_advanced", 0.0),
        len(labels_arr),
    )

    # ─── Save Outputs ───
    # 1. Metrics JSON
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    # 2. Confusion matrix plot
    cm_path = output_dir / "confusion_matrix.png"
    _plot_confusion_matrix(
        cm=cm,
        class_names=STAGE_NAMES,
        output_path=cm_path,
    )

    # 3. Classification report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("NeuroSense — Test Set Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {ckpt_meta.get('epoch', 'N/A')}\n")
        f.write(f"Test samples: {len(labels_arr)}\n\n")
        f.write(cls_report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(f"{'':>15}")
        for name in STAGE_NAMES:
            f.write(f"{name:>15}")
        f.write("\n")
        for i, name in enumerate(STAGE_NAMES):
            f.write(f"{name:>15}")
            for j in range(len(STAGE_NAMES)):
                f.write(f"{cm[i, j]:>15d}")
            f.write("\n")
    logger.info("Classification report saved to %s", report_path)

    return metrics


# ═════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for model evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained NeuroSenseModel on the test set. "
            "Computes AUC-ROC, F1, accuracy, and confusion matrix."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/processed",
        help="Root directory of processed BIDS data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to train_config.yaml",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    metrics = evaluate_model(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        config_path=args.config,
        batch_size=args.batch_size,
    )

    print(f"\n{'='*50}")
    print("Test Evaluation Results:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  AUC (mean): {metrics['auc_mean']:.4f}")
    print(f"  Samples:    {metrics['n_test_samples']}")
    print(f"\n  Outputs saved to: {args.output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
