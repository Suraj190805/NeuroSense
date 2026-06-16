"""NeuroSense end-to-end training pipeline for NeuroSenseModel.

Trains the full multi-modal composite model (MRI encoder + Clinical
encoder + Cross-Modal Fusion + Classification head + Progression head)
as specified in PRD Phase 5.

This pipeline builds on the Phase 2 ``train_mri_encoder()`` loop in
``mri_encoder.py``, extending it to handle:
- Multi-modal input (MRI volumes + clinical feature sequences)
- Multi-task loss (classification + progression via ``CombinedLoss``)
- Optional pre-trained encoder loading from Phase 2/3 checkpoints
- Separate parameter groups with differential learning rates

Usage:
    from neurosense.training.train import train_neurosense

    results = train_neurosense(
        data_root="data/processed",
        checkpoint_dir="checkpoints",
    )

CLI:
    python -m neurosense.training.train \\
        --data-root data/processed \\
        --checkpoint-dir checkpoints
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neurosense.data.dataset import HuntingtonDataset
from neurosense.models.classifier import NeuroSenseModel
from neurosense.models.mri_encoder import (
    CosineWarmupScheduler,
    EarlyStopping,
    _compute_per_class_auc,
    _load_config,
    _set_seed,
    _setup_device,
)
from neurosense.training.losses import CombinedLoss

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Training Step Helpers
# ═════════════════════════════════════════════════════════════════


def _train_one_epoch(
    model: NeuroSenseModel,
    train_loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    mixed_precision: bool,
    grad_accum_steps: int,
    log_every: int,
    epoch: int,
) -> tuple[float, float, dict[str, float]]:
    """Run a single training epoch.

    Args:
        model: NeuroSenseModel instance.
        train_loader: Training DataLoader.
        criterion: CombinedLoss instance.
        optimizer: AdamW optimizer.
        scaler: GradScaler for mixed precision.
        device: Compute device.
        mixed_precision: Whether to use AMP.
        grad_accum_steps: Gradient accumulation steps.
        log_every: Log frequency in steps.
        epoch: Current epoch number (0-indexed).

    Returns:
        Tuple of (epoch_loss, epoch_accuracy, loss_breakdown).
        loss_breakdown contains 'cls_loss' and 'prog_loss' averages.
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_prog_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        mri = batch["mri"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Clinical input is [B, 5] from dataset — unsqueeze for encoder
        if clinical.ndim == 2:
            clinical = clinical.unsqueeze(1)

        with autocast(
            device_type=device.type,
            enabled=(mixed_precision and device.type == "cuda"),
        ):
            outputs = model(mri=mri, clinical=clinical)
            logits = outputs["logits"]
            deltas = outputs["deltas"]

            # CombinedLoss handles missing progression targets
            loss, loss_dict = criterion(
                logits=logits,
                labels=labels,
                deltas_pred=deltas,
                deltas_target=None,  # No progression ground-truth in dataset yet
            )
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_dict["total"]
        total_cls_loss += loss_dict["cls_loss"]
        total_prog_loss += loss_dict["prog_loss"]
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % log_every == 0:
            logger.info(
                "  [Epoch %d] Step %d/%d — loss=%.4f "
                "(cls=%.4f, prog=%.4f)",
                epoch + 1,
                step + 1,
                len(train_loader),
                loss_dict["total"],
                loss_dict["cls_loss"],
                loss_dict["prog_loss"],
            )

    n_batches = max(len(train_loader), 1)
    avg_loss = total_loss / n_batches
    avg_acc = correct / max(total, 1)
    avg_breakdown = {
        "cls_loss": total_cls_loss / n_batches,
        "prog_loss": total_prog_loss / n_batches,
    }

    return avg_loss, avg_acc, avg_breakdown


@torch.no_grad()
def _validate(
    model: NeuroSenseModel,
    val_loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """Run validation and compute per-class AUC-ROC.

    Args:
        model: NeuroSenseModel instance.
        val_loader: Validation DataLoader.
        criterion: CombinedLoss instance.
        device: Compute device.
        mixed_precision: Whether to use AMP.

    Returns:
        Tuple of (val_loss, val_accuracy, val_aucs, loss_breakdown).
        val_aucs contains per-class and mean AUC-ROC values.
    """
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_prog_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    for batch in val_loader:
        mri = batch["mri"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if clinical.ndim == 2:
            clinical = clinical.unsqueeze(1)

        with autocast(
            device_type=device.type,
            enabled=(mixed_precision and device.type == "cuda"),
        ):
            outputs = model(mri=mri, clinical=clinical)
            logits = outputs["logits"]
            deltas = outputs["deltas"]

            loss, loss_dict = criterion(
                logits=logits,
                labels=labels,
                deltas_pred=deltas,
                deltas_target=None,
            )

        total_loss += loss_dict["total"]
        total_cls_loss += loss_dict["cls_loss"]
        total_prog_loss += loss_dict["prog_loss"]
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.cpu().tolist())

    n_batches = max(len(val_loader), 1)
    avg_loss = total_loss / n_batches
    avg_acc = correct / max(total, 1)

    # Per-class AUC-ROC
    labels_arr = np.array(all_labels)
    probs_arr = (
        np.concatenate(all_probs, axis=0)
        if all_probs
        else np.zeros((0, 3))
    )
    val_aucs = _compute_per_class_auc(labels_arr, probs_arr)

    avg_breakdown = {
        "cls_loss": total_cls_loss / n_batches,
        "prog_loss": total_prog_loss / n_batches,
    }

    return avg_loss, avg_acc, val_aucs, avg_breakdown


def _save_checkpoint(
    model: NeuroSenseModel,
    optimizer: AdamW,
    epoch: int,
    val_aucs: dict[str, float],
    config: dict[str, Any],
    checkpoint_path: Path,
    is_best: bool = False,
) -> None:
    """Save model checkpoint with full metadata.

    Args:
        model: NeuroSenseModel to save.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        val_aucs: Validation AUC metrics.
        config: Training configuration dict.
        checkpoint_path: Output file path.
        is_best: Whether this is the best checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_aucs": val_aucs,
        "model_config": {
            "embed_dim": model.embed_dim,
            "use_mri": model.use_mri,
            "use_clinical": model.use_clinical,
            "fusion_type": model.fusion_type,
        },
        "training_config": config,
        "is_best": is_best,
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(
        "%s checkpoint saved to %s",
        "Best" if is_best else "Last",
        checkpoint_path,
    )


# ═════════════════════════════════════════════════════════════════
#  Main Training Function
# ═════════════════════════════════════════════════════════════════


def train_neurosense(
    data_root: str | Path = "data/processed",
    checkpoint_dir: str | Path = "checkpoints",
    config_path: str | Path | None = None,
    pretrained_mri_ckpt: str | Path | None = None,
    wandb_enabled: bool | None = None,
    use_mri: bool = True,
    use_clinical: bool = True,
    fusion_type: str = "cross_attention",
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Train the full NeuroSenseModel end-to-end.

    Implements the production training pipeline for the multi-modal
    composite model. Supports all ablation configurations via the
    ``use_mri``, ``use_clinical``, and ``fusion_type`` parameters.

    Args:
        data_root: Root directory of the BIDS-format dataset.
        checkpoint_dir: Directory for saving checkpoints.
        config_path: Path to train_config.yaml override.
        pretrained_mri_ckpt: Optional path to pre-trained MRI
            encoder checkpoint from Phase 2.
        wandb_enabled: Override W&B logging. None uses config.
        use_mri: Whether to use MRI modality (default: True).
        use_clinical: Whether to use clinical modality (default: True).
        fusion_type: Fusion strategy — "cross_attention" or
            "concatenation" (default: "cross_attention").
        max_epochs: Override max epochs from config.

    Returns:
        Dictionary with training results:
        - best_val_auc: Best mean validation AUC-ROC
        - best_epoch: Epoch of best AUC
        - final_train_loss: Training loss at final epoch
        - total_epochs: Number of epochs trained
        - checkpoint_path: Path to best checkpoint
        - val_aucs: Per-class AUC dict at best epoch

    Example:
        >>> results = train_neurosense(
        ...     data_root="data/processed",
        ...     checkpoint_dir="checkpoints",
        ... )
        >>> print(f"Best AUC: {results['best_val_auc']:.4f}")
    """
    # ─── Configuration ───
    config = _load_config(config_path)
    train_cfg = config.get("training", {})
    es_cfg = config.get("early_stopping", {})
    loss_cfg = config.get("loss", {})
    repro_cfg = config.get("reproducibility", {})
    log_cfg = config.get("logging", {})
    mri_cfg = config.get("mri_encoder", {})
    clin_cfg = config.get("clinical_encoder", {})
    fusion_cfg = config.get("fusion", {})

    # Reproducibility
    seed = repro_cfg.get("seed", 42)
    deterministic = repro_cfg.get("deterministic", True)
    _set_seed(seed, deterministic)

    # Hyperparameters
    batch_size = train_cfg.get("batch_size", 4)
    epochs = max_epochs or train_cfg.get("epochs", 50)
    base_lr = train_cfg.get("optimizer", {}).get("lr", 1e-4)
    weight_decay = train_cfg.get("optimizer", {}).get("weight_decay", 1e-5)
    betas = tuple(train_cfg.get("optimizer", {}).get("betas", [0.9, 0.999]))
    warmup_epochs = train_cfg.get("scheduler", {}).get("warmup_epochs", 5)
    min_lr = train_cfg.get("scheduler", {}).get("min_lr", 1e-6)
    mixed_precision = train_cfg.get("mixed_precision", True)
    grad_checkpointing = train_cfg.get("gradient_checkpointing", False)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    num_workers = config.get("data", {}).get("num_workers", 4)
    log_every = log_cfg.get("log_every_n_steps", 10)

    # Early stopping
    es_patience = es_cfg.get("patience", 10)
    es_min_delta = es_cfg.get("min_delta", 0.001)

    # Loss weights
    cls_weight = loss_cfg.get("combined", {}).get(
        "classification_weight", 1.0
    )
    prog_weight = loss_cfg.get("combined", {}).get(
        "progression_weight", 0.5
    )
    huber_delta = loss_cfg.get("progression", {}).get("delta", 1.0)

    # Architecture
    embed_dim = mri_cfg.get("embedding_dim", 256)

    device = _setup_device(config)

    # ─── W&B Setup ───
    use_wandb = wandb_enabled
    if use_wandb is None:
        wandb_cfg = log_cfg.get("wandb", {})
        use_wandb = wandb_cfg.get("enabled", False)

    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_cfg = log_cfg.get("wandb", {})
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "neurosense"),
                entity=wandb_cfg.get("entity"),
                tags=wandb_cfg.get(
                    "tags", ["huntington", "multimodal", "fusion"]
                ),
                config={
                    "phase": "phase5_full_model",
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                    "warmup_epochs": warmup_epochs,
                    "embed_dim": embed_dim,
                    "mixed_precision": mixed_precision,
                    "use_mri": use_mri,
                    "use_clinical": use_clinical,
                    "fusion_type": fusion_type,
                },
            )
            logger.info("W&B logging enabled: %s", wandb_run.url)
        except ImportError:
            logger.warning(
                "wandb not installed, disabling W&B logging"
            )
            use_wandb = False

    # ─── Data Loading ───
    data_root = Path(data_root)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s ...", data_root)
    train_loader, val_loader, test_loader = HuntingtonDataset.split(
        root_dir=data_root,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(
        "Data loaded: %d train batches, %d val batches, %d test batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    # ─── Class Weights ───
    class_weights = None
    if loss_cfg.get("classification", {}).get("auto_weight", True):
        base_dataset = train_loader.dataset
        while hasattr(base_dataset, "dataset"):
            base_dataset = base_dataset.dataset
        if hasattr(base_dataset, "get_class_weights"):
            class_weights = base_dataset.get_class_weights().to(device)
            logger.info("Class weights: %s", class_weights.tolist())

    # ─── Model ───
    model_kwargs: dict[str, Any] = {
        "embed_dim": embed_dim,
        "use_mri": use_mri,
        "use_clinical": use_clinical,
        "fusion_type": fusion_type,
    }

    # MRI encoder config
    if use_mri:
        model_kwargs["mri_encoder_kwargs"] = {
            "backbone_out_dim": mri_cfg.get("backbone_out_dim", 2048),
            "gradient_checkpointing": grad_checkpointing,
        }

    # Clinical encoder config
    if use_clinical:
        model_kwargs["clinical_encoder_kwargs"] = {
            "input_features": clin_cfg.get("input_features", 5),
            "hidden_size": clin_cfg.get("hidden_size", 256),
            "num_layers": clin_cfg.get("num_layers", 2),
            "dropout": clin_cfg.get("dropout", 0.3),
        }

    # Fusion config
    if use_mri and use_clinical:
        model_kwargs["fusion_kwargs"] = {
            "num_heads": fusion_cfg.get("num_heads", 8),
            "ffn_hidden_dim": fusion_cfg.get("ffn_hidden_dim", 1024),
        }

    model = NeuroSenseModel(**model_kwargs).to(device)

    # ─── Optional: Load pre-trained MRI encoder ───
    if pretrained_mri_ckpt and use_mri:
        from neurosense.models.mri_encoder import load_mri_encoder

        logger.info(
            "Loading pre-trained MRI encoder from %s",
            pretrained_mri_ckpt,
        )
        pretrained_enc, _, _ = load_mri_encoder(
            pretrained_mri_ckpt, device=device
        )
        # Copy weights into model's MRI encoder
        model.mri_encoder.load_state_dict(
            pretrained_enc.state_dict(), strict=False
        )
        logger.info("Pre-trained MRI encoder weights loaded")

    # Log model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        "NeuroSenseModel: %dM total params (%dM trainable)",
        total_params // 1_000_000,
        trainable_params // 1_000_000,
    )

    # ─── Optimizer with differential learning rates ───
    # Encoders get base_lr, heads get 10× base_lr for faster convergence
    encoder_params: list[torch.Tensor] = []
    head_params: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classification_head" in name or "progression_head" in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    param_groups = [
        {
            "params": encoder_params,
            "lr": base_lr,
            "name": "encoders_fusion",
        },
        {
            "params": head_params,
            "lr": base_lr * 10.0,
            "name": "heads",
        },
    ]

    optimizer = AdamW(
        param_groups,
        weight_decay=weight_decay,
        betas=betas,
    )

    logger.info(
        "Optimizer: AdamW — encoders lr=%.2e, heads lr=%.2e, "
        "weight_decay=%.2e",
        base_lr,
        base_lr * 10.0,
        weight_decay,
    )

    # ─── Scheduler ───
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.01
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        min_lr_ratio=min_lr_ratio,
    )

    # ─── Loss ───
    criterion = CombinedLoss(
        class_weights=class_weights,
        classification_weight=cls_weight,
        progression_weight=prog_weight,
        label_smoothing=0.05,
        huber_delta=huber_delta,
    ).to(device)

    # ─── Mixed Precision ───
    scaler = GradScaler(
        enabled=(mixed_precision and device.type == "cuda")
    )

    # ─── Early Stopping ───
    early_stopper = EarlyStopping(
        patience=es_patience,
        min_delta=es_min_delta,
        mode="max",
        verbose=True,
    )

    # ─── Training Loop ───
    modality_desc = []
    if use_mri:
        modality_desc.append("MRI")
    if use_clinical:
        modality_desc.append("Clinical")
    modality_str = " + ".join(modality_desc)

    logger.info(
        "═" * 60 + "\n"
        "  Phase 5: NeuroSenseModel Training\n"
        "  Modalities: %s | Fusion: %s\n"
        "  Epochs: %d | Batch: %d | LR: %.2e | Device: %s\n"
        "  Mixed precision: %s | Grad checkpointing: %s\n"
        "  Loss: cls_w=%.1f, prog_w=%.1f\n"
        + "═" * 60,
        modality_str,
        fusion_type if (use_mri and use_clinical) else "N/A",
        epochs,
        batch_size,
        base_lr,
        device,
        mixed_precision,
        grad_checkpointing,
        cls_weight,
        prog_weight,
    )

    results: dict[str, Any] = {
        "best_val_auc": 0.0,
        "best_epoch": 0,
        "final_train_loss": float("inf"),
        "total_epochs": 0,
        "checkpoint_path": "",
        "val_aucs": {},
    }

    best_val_aucs: dict[str, float] = {}

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Train ──
        train_loss, train_acc, train_breakdown = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixed_precision=mixed_precision,
            grad_accum_steps=grad_accum_steps,
            log_every=log_every,
            epoch=epoch,
        )

        # ── Validate ──
        val_loss, val_acc, val_aucs, val_breakdown = _validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            mixed_precision=mixed_precision,
        )

        # ── Scheduler step ──
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # ── Logging ──
        logger.info(
            "Epoch %d/%d (%.1fs) — "
            "train: loss=%.4f acc=%.3f (cls=%.4f prog=%.4f) | "
            "val: loss=%.4f acc=%.3f | "
            "AUC: pre=%.3f early=%.3f adv=%.3f mean=%.3f | "
            "lr=%.2e",
            epoch + 1,
            epochs,
            epoch_time,
            train_loss,
            train_acc,
            train_breakdown["cls_loss"],
            train_breakdown["prog_loss"],
            val_loss,
            val_acc,
            val_aucs.get("auc_premanifest", 0.0),
            val_aucs.get("auc_early", 0.0),
            val_aucs.get("auc_advanced", 0.0),
            val_aucs.get("auc_mean", 0.0),
            current_lr,
        )

        if use_wandb and wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_cls_loss": train_breakdown["cls_loss"],
                    "train_prog_loss": train_breakdown["prog_loss"],
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                    **val_aucs,
                },
                step=epoch,
            )

        # ── Early Stopping ──
        val_auc_mean = val_aucs.get("auc_mean", 0.0)
        early_stopper(val_auc_mean, model, epoch)

        results["final_train_loss"] = train_loss
        results["total_epochs"] = epoch + 1

        # Track best AUCs for results
        if early_stopper.best_epoch == epoch:
            best_val_aucs = val_aucs.copy()

        if early_stopper.should_stop:
            logger.info(
                "Early stopping at epoch %d. "
                "Best val_auc_mean=%.4f at epoch %d.",
                epoch + 1,
                early_stopper.best_score,
                early_stopper.best_epoch + 1,
            )
            break

    # ─── Save Checkpoints ───
    # Best checkpoint
    best_ckpt_path = checkpoint_dir / "neurosense_best.pth"
    if early_stopper.best_state_dict is not None:
        # Temporarily load best state dict for saving
        current_state = model.state_dict()
        model.load_state_dict(early_stopper.best_state_dict)
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=early_stopper.best_epoch,
            val_aucs=best_val_aucs,
            config=config,
            checkpoint_path=best_ckpt_path,
            is_best=True,
        )
        # Restore current state
        model.load_state_dict(current_state)
    else:
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=results["total_epochs"] - 1,
            val_aucs=val_aucs,
            config=config,
            checkpoint_path=best_ckpt_path,
            is_best=True,
        )

    # Last checkpoint
    last_ckpt_path = checkpoint_dir / "neurosense_last.pth"
    _save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=results["total_epochs"] - 1,
        val_aucs=val_aucs,
        config=config,
        checkpoint_path=last_ckpt_path,
        is_best=False,
    )

    # ─── Final Results ───
    results["best_val_auc"] = early_stopper.best_score or 0.0
    results["best_epoch"] = early_stopper.best_epoch
    results["checkpoint_path"] = str(best_ckpt_path)
    results["val_aucs"] = best_val_aucs

    logger.info(
        "═" * 60 + "\n"
        "  Training Complete\n"
        "  Best val_auc_mean: %.4f (epoch %d)\n"
        "  Total epochs trained: %d\n"
        "  Checkpoint: %s\n"
        + "═" * 60,
        results["best_val_auc"],
        results["best_epoch"] + 1,
        results["total_epochs"],
        results["checkpoint_path"],
    )

    if use_wandb and wandb_run is not None:
        import wandb

        wandb.finish()

    return results


# ═════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for NeuroSenseModel training."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Train NeuroSense multi-modal model (Phase 5). "
            "Combines 3D ResNet-50 MRI encoder, Bi-LSTM clinical "
            "encoder, cross-attention fusion, and task heads."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/processed",
        help="Root directory of processed BIDS data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to train_config.yaml",
    )
    parser.add_argument(
        "--pretrained-mri",
        type=str,
        default=None,
        help="Path to pre-trained MRI encoder checkpoint (Phase 2)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--no-mri",
        action="store_true",
        help="Disable MRI modality (clinical-only mode)",
    )
    parser.add_argument(
        "--no-clinical",
        action="store_true",
        help="Disable clinical modality (MRI-only mode)",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="cross_attention",
        choices=["cross_attention", "concatenation"],
        help="Fusion strategy (default: cross_attention)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs from config",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = train_neurosense(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config,
        pretrained_mri_ckpt=args.pretrained_mri,
        wandb_enabled=not args.no_wandb if args.no_wandb else None,
        use_mri=not args.no_mri,
        use_clinical=not args.no_clinical,
        fusion_type=args.fusion,
        max_epochs=args.epochs,
    )

    print(f"\n{'='*50}")
    print("Phase 5 Training Results:")
    print(f"  Best AUC:     {results['best_val_auc']:.4f}")
    print(f"  Best Epoch:   {results['best_epoch'] + 1}")
    print(f"  Total Epochs: {results['total_epochs']}")
    print(f"  Checkpoint:   {results['checkpoint_path']}")
    if results.get("val_aucs"):
        print("  Per-class AUC:")
        for key, val in results["val_aucs"].items():
            print(f"    {key}: {val:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
