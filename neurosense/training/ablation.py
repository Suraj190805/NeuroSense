"""NeuroSense 5-condition ablation study runner.

Implements the ablation study specified in PRD Section 11 and
``train_config.yaml``. Trains and evaluates the NeuroSenseModel
under five modality/fusion conditions across multiple seeds,
aggregating results into a CSV for statistical comparison.

Ablation Conditions:
    1. mri_only         — MRI encoder only (no clinical)
    2. clinical_only    — Clinical encoder only (no MRI)
    3. genetic_only     — CAG repeat feature only (clinical encoder
                          with non-CAG features zeroed out)
    4. mri_clinical_concat — MRI + Clinical with concatenation fusion
    5. full_neurosense  — MRI + Clinical with cross-attention fusion

Each condition is run with 3 seeds (42, 123, 456) for robustness.

Usage:
    from neurosense.training.ablation import run_ablation_study

    results_df = run_ablation_study(
        data_root="data/processed",
        output_dir="outputs",
    )

CLI:
    python -m neurosense.training.ablation \\
        --data-root data/processed \\
        --output-dir outputs
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neurosense.data.dataset import STAGE_NAMES, HuntingtonDataset
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

# Default ablation conditions matching train_config.yaml
DEFAULT_CONDITIONS: list[dict[str, Any]] = [
    {
        "name": "mri_only",
        "use_mri": True,
        "use_clinical": False,
        "fusion": None,
        "use_cag_only": False,
    },
    {
        "name": "clinical_only",
        "use_mri": False,
        "use_clinical": True,
        "fusion": None,
        "use_cag_only": False,
    },
    {
        "name": "genetic_only",
        "use_mri": False,
        "use_clinical": True,
        "fusion": None,
        "use_cag_only": True,
    },
    {
        "name": "mri_clinical_concat",
        "use_mri": True,
        "use_clinical": True,
        "fusion": "concatenation",
        "use_cag_only": False,
    },
    {
        "name": "full_neurosense",
        "use_mri": True,
        "use_clinical": True,
        "fusion": "cross_attention",
        "use_cag_only": False,
    },
]

DEFAULT_SEEDS: list[int] = [42, 123, 456]

# CSV output columns matching train_config.yaml specification
OUTPUT_COLUMNS: list[str] = [
    "run_id",
    "condition",
    "seed",
    "auc_premanifest",
    "auc_early",
    "auc_advanced",
    "auc_mean",
    "f1_macro",
    "val_loss",
]


# ═════════════════════════════════════════════════════════════════
#  Genetic-Only Data Transform
# ═════════════════════════════════════════════════════════════════


def _zero_non_cag_features(batch: dict[str, Any]) -> dict[str, Any]:
    """Zero out non-CAG clinical features for genetic_only ablation.

    The clinical feature vector is [cag_repeat, uhdrs_motor,
    uhdrs_cognitive, tfc, age]. For genetic_only, we keep only
    the CAG repeat (index 0) and zero the rest.

    Args:
        batch: Batch dict from DataLoader.

    Returns:
        Modified batch with non-CAG features zeroed.
    """
    clinical = batch["clinical"]
    # Zero out indices 1-4 (uhdrs_motor, uhdrs_cognitive, tfc, age)
    if clinical.ndim == 2:
        clinical[:, 1:] = 0.0
    elif clinical.ndim == 3:
        clinical[:, :, 1:] = 0.0
    batch["clinical"] = clinical
    return batch


# ═════════════════════════════════════════════════════════════════
#  Single Condition Training
# ═════════════════════════════════════════════════════════════════


def _train_ablation_condition(
    condition: dict[str, Any],
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate a single ablation condition.

    Args:
        condition: Ablation condition specification dict.
        seed: Random seed for this run.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        config: Full configuration dict.
        device: Compute device.
        checkpoint_dir: Directory for condition checkpoints.

    Returns:
        Dictionary with evaluation metrics for this run.
    """
    from sklearn.metrics import f1_score

    condition_name = condition["name"]
    use_mri = condition["use_mri"]
    use_clinical = condition["use_clinical"]
    fusion_type = condition.get("fusion", "cross_attention")
    use_cag_only = condition.get("use_cag_only", False)

    # Set seed for this run
    _set_seed(seed)

    run_id = f"{condition_name}_seed{seed}"
    logger.info(
        "─" * 50 + "\n"
        "  Ablation: %s (seed=%d)\n"
        "  MRI=%s | Clinical=%s | Fusion=%s | CAG-only=%s\n"
        + "─" * 50,
        condition_name,
        seed,
        use_mri,
        use_clinical,
        fusion_type or "N/A",
        use_cag_only,
    )

    # ─── Config extraction ───
    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    es_cfg = config.get("early_stopping", {})
    mri_cfg = config.get("mri_encoder", {})
    clin_cfg = config.get("clinical_encoder", {})
    fusion_cfg = config.get("fusion", {})

    epochs = train_cfg.get("epochs", 50)
    base_lr = train_cfg.get("optimizer", {}).get("lr", 1e-4)
    weight_decay = train_cfg.get("optimizer", {}).get("weight_decay", 1e-5)
    betas = tuple(
        train_cfg.get("optimizer", {}).get("betas", [0.9, 0.999])
    )
    warmup_epochs = train_cfg.get("scheduler", {}).get("warmup_epochs", 5)
    min_lr = train_cfg.get("scheduler", {}).get("min_lr", 1e-6)
    mixed_precision = train_cfg.get("mixed_precision", True)
    grad_checkpointing = train_cfg.get("gradient_checkpointing", False)

    cls_weight = loss_cfg.get("combined", {}).get(
        "classification_weight", 1.0
    )
    prog_weight = loss_cfg.get("combined", {}).get(
        "progression_weight", 0.5
    )
    huber_delta = loss_cfg.get("progression", {}).get("delta", 1.0)

    es_patience = es_cfg.get("patience", 10)
    es_min_delta = es_cfg.get("min_delta", 0.001)

    embed_dim = mri_cfg.get("embedding_dim", 256)

    # ─── Class Weights ───
    class_weights = None
    if loss_cfg.get("classification", {}).get("auto_weight", True):
        base_dataset = train_loader.dataset
        while hasattr(base_dataset, "dataset"):
            base_dataset = base_dataset.dataset
        if hasattr(base_dataset, "get_class_weights"):
            class_weights = base_dataset.get_class_weights().to(device)

    # ─── Build Model ───
    model_kwargs: dict[str, Any] = {
        "embed_dim": embed_dim,
        "use_mri": use_mri,
        "use_clinical": use_clinical,
    }

    if use_mri and use_clinical and fusion_type:
        model_kwargs["fusion_type"] = fusion_type

    if use_mri:
        model_kwargs["mri_encoder_kwargs"] = {
            "backbone_out_dim": mri_cfg.get("backbone_out_dim", 2048),
            "gradient_checkpointing": grad_checkpointing,
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

    model = NeuroSenseModel(**model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "  Model: %dM params", total_params // 1_000_000
    )

    # ─── Optimizer ───
    optimizer = AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=betas,
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
        verbose=False,  # Reduce noise in ablation logs
    )

    # ─── Training Loop ───
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            mri = batch["mri"].to(device, non_blocking=True)
            clinical = batch["clinical"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # Apply genetic-only masking
            if use_cag_only:
                if clinical.ndim == 2:
                    clinical[:, 1:] = 0.0
                elif clinical.ndim == 3:
                    clinical[:, :, 1:] = 0.0

            if clinical.ndim == 2:
                clinical = clinical.unsqueeze(1)

            with autocast(
                device_type=device.type,
                enabled=(mixed_precision and device.type == "cuda"),
            ):
                forward_kwargs: dict[str, Any] = {}
                if use_mri:
                    forward_kwargs["mri"] = mri
                if use_clinical:
                    forward_kwargs["clinical"] = clinical

                outputs = model(**forward_kwargs)
                loss, loss_dict = criterion(
                    logits=outputs["logits"],
                    labels=labels,
                    deltas_pred=outputs["deltas"],
                    deltas_target=None,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss_dict["total"]

        scheduler.step()

        # ─── Validation ───
        model.eval()
        val_loss_sum = 0.0
        all_val_labels: list[int] = []
        all_val_probs: list[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                mri = batch["mri"].to(device, non_blocking=True)
                clinical = batch["clinical"].to(
                    device, non_blocking=True
                )
                labels = batch["label"].to(device, non_blocking=True)

                if use_cag_only:
                    if clinical.ndim == 2:
                        clinical[:, 1:] = 0.0
                    elif clinical.ndim == 3:
                        clinical[:, :, 1:] = 0.0

                if clinical.ndim == 2:
                    clinical = clinical.unsqueeze(1)

                with autocast(
                    device_type=device.type,
                    enabled=(
                        mixed_precision and device.type == "cuda"
                    ),
                ):
                    forward_kwargs = {}
                    if use_mri:
                        forward_kwargs["mri"] = mri
                    if use_clinical:
                        forward_kwargs["clinical"] = clinical

                    outputs = model(**forward_kwargs)
                    loss, loss_dict = criterion(
                        logits=outputs["logits"],
                        labels=labels,
                        deltas_pred=outputs["deltas"],
                        deltas_target=None,
                    )

                val_loss_sum += loss_dict["total"]
                probs = (
                    torch.softmax(outputs["logits"], dim=-1)
                    .cpu()
                    .numpy()
                )
                all_val_probs.append(probs)
                all_val_labels.extend(labels.cpu().tolist())

        # Compute AUC
        val_labels_arr = np.array(all_val_labels)
        val_probs_arr = (
            np.concatenate(all_val_probs, axis=0)
            if all_val_probs
            else np.zeros((0, 3))
        )
        val_aucs = _compute_per_class_auc(val_labels_arr, val_probs_arr)
        val_auc_mean = val_aucs.get("auc_mean", 0.0)

        # Early stopping
        early_stopper(val_auc_mean, model, epoch)

        # Periodic logging (every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_loss = val_loss_sum / max(len(val_loader), 1)
            logger.info(
                "  [%s] Epoch %d/%d — val_loss=%.4f, "
                "AUC_mean=%.3f",
                condition_name,
                epoch + 1,
                epochs,
                val_loss,
                val_auc_mean,
            )

        if early_stopper.should_stop:
            logger.info(
                "  [%s] Early stopping at epoch %d "
                "(best AUC=%.4f @ epoch %d)",
                condition_name,
                epoch + 1,
                early_stopper.best_score,
                early_stopper.best_epoch + 1,
            )
            break

    # ─── Load best model and evaluate on test set ───
    if early_stopper.best_state_dict is not None:
        model.load_state_dict(early_stopper.best_state_dict)

    model.eval()
    all_test_labels: list[int] = []
    all_test_preds: list[int] = []
    all_test_probs: list[np.ndarray] = []
    test_loss_sum = 0.0

    with torch.no_grad():
        for batch in test_loader:
            mri = batch["mri"].to(device, non_blocking=True)
            clinical = batch["clinical"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if use_cag_only:
                if clinical.ndim == 2:
                    clinical[:, 1:] = 0.0
                elif clinical.ndim == 3:
                    clinical[:, :, 1:] = 0.0

            if clinical.ndim == 2:
                clinical = clinical.unsqueeze(1)

            with autocast(
                device_type=device.type,
                enabled=(mixed_precision and device.type == "cuda"),
            ):
                forward_kwargs = {}
                if use_mri:
                    forward_kwargs["mri"] = mri
                if use_clinical:
                    forward_kwargs["clinical"] = clinical

                outputs = model(**forward_kwargs)
                loss, loss_dict = criterion(
                    logits=outputs["logits"],
                    labels=labels,
                    deltas_pred=outputs["deltas"],
                    deltas_target=None,
                )

            test_loss_sum += loss_dict["total"]
            probs = (
                torch.softmax(outputs["logits"], dim=-1)
                .cpu()
                .numpy()
            )
            preds = outputs["logits"].argmax(dim=-1).cpu().tolist()

            all_test_labels.extend(labels.cpu().tolist())
            all_test_preds.extend(preds)
            all_test_probs.append(probs)

    # Compute test metrics
    test_labels_arr = np.array(all_test_labels)
    test_preds_arr = np.array(all_test_preds)
    test_probs_arr = (
        np.concatenate(all_test_probs, axis=0)
        if all_test_probs
        else np.zeros((0, 3))
    )

    test_aucs = _compute_per_class_auc(test_labels_arr, test_probs_arr)
    test_f1_macro = float(
        f1_score(
            test_labels_arr,
            test_preds_arr,
            average="macro",
            zero_division=0,
        )
    )
    test_val_loss = test_loss_sum / max(len(test_loader), 1)

    # Save condition checkpoint
    ckpt_path = checkpoint_dir / f"ablation_{run_id}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "condition": condition,
            "seed": seed,
            "best_epoch": early_stopper.best_epoch,
            "test_aucs": test_aucs,
            "test_f1_macro": test_f1_macro,
        },
        ckpt_path,
    )

    result = {
        "run_id": run_id,
        "condition": condition_name,
        "seed": seed,
        "auc_premanifest": test_aucs.get("auc_premanifest", 0.0),
        "auc_early": test_aucs.get("auc_early", 0.0),
        "auc_advanced": test_aucs.get("auc_advanced", 0.0),
        "auc_mean": test_aucs.get("auc_mean", 0.0),
        "f1_macro": test_f1_macro,
        "val_loss": test_val_loss,
    }

    logger.info(
        "  [%s] Test results: AUC_mean=%.4f, F1_macro=%.4f, "
        "loss=%.4f",
        run_id,
        result["auc_mean"],
        result["f1_macro"],
        result["val_loss"],
    )

    return result


# ═════════════════════════════════════════════════════════════════
#  Main Ablation Study
# ═════════════════════════════════════════════════════════════════


def run_ablation_study(
    data_root: str | Path = "data/processed",
    output_dir: str | Path = "outputs",
    checkpoint_dir: str | Path = "checkpoints/ablation",
    config_path: str | Path | None = None,
    conditions: list[dict[str, Any]] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run the full 5-condition ablation study.

    Trains and evaluates each condition with each seed, then
    aggregates results into a CSV file. Conditions and seeds
    are loaded from ``train_config.yaml`` or provided directly.

    Args:
        data_root: Root directory of BIDS-format dataset.
        output_dir: Directory for output CSV and summary.
        checkpoint_dir: Directory for per-condition checkpoints.
        config_path: Path to train_config.yaml override.
        conditions: Override ablation conditions list.
        seeds: Override seed list.

    Returns:
        List of result dicts, one per condition×seed run.

    Example:
        >>> results = run_ablation_study(
        ...     data_root="data/processed",
        ... )
        >>> print(f"{len(results)} runs completed")
    """
    # ─── Configuration ───
    config = _load_config(config_path)
    device = _setup_device(config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load conditions from config or use defaults
    if conditions is None:
        ablation_cfg = config.get("ablation", {})
        config_conditions = ablation_cfg.get("conditions")
        if config_conditions:
            conditions = config_conditions
        else:
            conditions = DEFAULT_CONDITIONS

    if seeds is None:
        ablation_cfg = config.get("ablation", {})
        seeds = ablation_cfg.get("seeds", DEFAULT_SEEDS)

    total_runs = len(conditions) * len(seeds)
    logger.info(
        "═" * 60 + "\n"
        "  NeuroSense Ablation Study (PRD Section 11)\n"
        "  ─────────────────────────────────────\n"
        "  Conditions: %d\n"
        "  Seeds: %s\n"
        "  Total runs: %d\n"
        "  Device: %s\n"
        + "═" * 60,
        len(conditions),
        seeds,
        total_runs,
        device,
    )

    # ─── Create DataLoaders (shared across conditions) ───
    # Use a fixed seed for data splitting so all conditions see
    # the same train/val/test split
    data_seed = config.get("reproducibility", {}).get("seed", 42)
    batch_size = config.get("training", {}).get("batch_size", 4)
    num_workers = config.get("data", {}).get("num_workers", 4)

    logger.info("Loading dataset from %s ...", data_root)
    train_loader, val_loader, test_loader = HuntingtonDataset.split(
        root_dir=data_root,
        seed=data_seed,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(
        "Data loaded: train=%d, val=%d, test=%d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    # ─── Run All Conditions × Seeds ───
    all_results: list[dict[str, Any]] = []
    study_start = time.time()

    for cond_idx, condition in enumerate(conditions):
        for seed_idx, seed in enumerate(seeds):
            run_num = cond_idx * len(seeds) + seed_idx + 1
            logger.info(
                "\n▶ Run %d/%d: %s (seed=%d)",
                run_num,
                total_runs,
                condition["name"],
                seed,
            )

            run_start = time.time()
            try:
                result = _train_ablation_condition(
                    condition=condition,
                    seed=seed,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    config=config,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(
                    "  Run %s (seed=%d) FAILED: %s",
                    condition["name"],
                    seed,
                    str(e),
                    exc_info=True,
                )
                # Record failed run with zero metrics
                all_results.append(
                    {
                        "run_id": f"{condition['name']}_seed{seed}",
                        "condition": condition["name"],
                        "seed": seed,
                        "auc_premanifest": 0.0,
                        "auc_early": 0.0,
                        "auc_advanced": 0.0,
                        "auc_mean": 0.0,
                        "f1_macro": 0.0,
                        "val_loss": float("inf"),
                    }
                )

            run_time = time.time() - run_start
            logger.info(
                "  Run completed in %.1f seconds", run_time
            )

    study_time = time.time() - study_start

    # ─── Save CSV ───
    csv_filename = config.get("ablation", {}).get(
        "output_csv", "ablation_results.csv"
    )
    csv_path = output_dir / csv_filename

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for result in all_results:
            row = {col: result.get(col, "") for col in OUTPUT_COLUMNS}
            writer.writerow(row)

    logger.info("Ablation results saved to %s", csv_path)

    # ─── Summary Table ───
    logger.info(
        "\n" + "═" * 70 + "\n"
        "  Ablation Study Summary\n"
        + "═" * 70
    )
    logger.info(
        "  %-25s  %8s  %8s  %8s",
        "Condition",
        "AUC↑",
        "F1↑",
        "Loss↓",
    )
    logger.info("  " + "─" * 55)

    # Aggregate by condition (mean ± std across seeds)
    condition_names_seen: list[str] = []
    for condition in conditions:
        cname = condition["name"]
        if cname in condition_names_seen:
            continue
        condition_names_seen.append(cname)

        cond_results = [
            r for r in all_results if r["condition"] == cname
        ]
        if not cond_results:
            continue

        aucs = [r["auc_mean"] for r in cond_results]
        f1s = [r["f1_macro"] for r in cond_results]
        losses = [
            r["val_loss"]
            for r in cond_results
            if r["val_loss"] < float("inf")
        ]

        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs) if len(aucs) > 1 else 0.0
        f1_mean = np.mean(f1s)
        f1_std = np.std(f1s) if len(f1s) > 1 else 0.0
        loss_mean = np.mean(losses) if losses else float("inf")

        logger.info(
            "  %-25s  %.3f±%.3f  %.3f±%.3f  %.4f",
            cname,
            auc_mean,
            auc_std,
            f1_mean,
            f1_std,
            loss_mean,
        )

    logger.info(
        "\n  Total time: %.1f minutes\n"
        "  Results: %s\n"
        + "═" * 70,
        study_time / 60.0,
        csv_path,
    )

    return all_results


# ═════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for ablation study."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run NeuroSense 5-condition ablation study (PRD §11). "
            "Compares MRI-only, Clinical-only, Genetic-only, "
            "Concat fusion, and Cross-attention fusion."
        ),
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
        help="Directory for ablation results CSV",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/ablation",
        help="Directory for per-condition checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to train_config.yaml",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override seed list (default: 42 123 456)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Run only specified conditions by name "
            "(e.g., mri_only full_neurosense)"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Filter conditions if specified
    conditions = None
    if args.conditions:
        conditions = [
            c
            for c in DEFAULT_CONDITIONS
            if c["name"] in args.conditions
        ]
        if not conditions:
            logger.error(
                "No matching conditions found for: %s\n"
                "Available: %s",
                args.conditions,
                [c["name"] for c in DEFAULT_CONDITIONS],
            )
            return

    results = run_ablation_study(
        data_root=args.data_root,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config,
        conditions=conditions,
        seeds=args.seeds,
    )

    print(f"\n{'='*60}")
    print(f"Ablation Study Complete: {len(results)} runs")
    print(f"Results saved to: {args.output_dir}/ablation_results.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
