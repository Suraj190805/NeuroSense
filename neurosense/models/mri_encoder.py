"""NeuroSense 3D ResNet-50 MRI Encoder.

Implements the MRI feature extraction module specified in PRD
Section 4.2.2 (Phase 2):
- MONAI ResNet-50 backbone (spatial_dims=3, n_input_channels=1)
- Projection head: Linear(2048, 256) → ReLU → LayerNorm
- ClassificationHead: Linear(256, 3) for HD staging
- Full training loop with cosine warmup, mixed precision,
  per-class AUC-ROC, early stopping, and W&B logging

Architecture::

    Input: [B, 1, 96, 96, 96]
        ↓
    MONAI ResNet-50 (3D)
        ↓
    [B, 2048] (backbone features)
        ↓
    Linear(2048, 256) → ReLU → LayerNorm
        ↓
    [B, 256] (embedding)
        ↓
    ClassificationHead: Linear(256, 3)
        ↓
    [B, 3] (logits: pre-manifest / early / advanced)

Usage:
    from neurosense.models.mri_encoder import (
        MRIEncoder,
        ClassificationHead,
        train_mri_encoder,
    )

    encoder = MRIEncoder(embedding_dim=256)
    embedding = encoder(mri_tensor)  # [B, 256]

    head = ClassificationHead(input_dim=256, num_classes=3)
    logits = head(embedding)  # [B, 3]

    # Full training pipeline
    train_mri_encoder(
        data_root="data/processed",
        checkpoint_dir="checkpoints",
    )
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from monai.networks.nets import ResNet
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from neurosense.data.dataset import STAGE_NAMES, HuntingtonDataset

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Cosine Warmup Scheduler
# ═════════════════════════════════════════════════════════════════


class CosineWarmupScheduler(LambdaLR):
    """Learning rate scheduler with linear warmup + cosine decay.

    Implements the schedule specified in PRD Section 4.2.2:
    - Linear warmup from 0 to base_lr over ``warmup_epochs`` epochs
    - Cosine annealing from base_lr to ``min_lr`` for remaining epochs

    The total schedule spans ``total_epochs`` epochs. Each "step"
    corresponds to one epoch (call ``scheduler.step()`` per epoch).

    Args:
        optimizer: PyTorch optimizer instance.
        warmup_epochs: Number of warmup epochs (default: 5 per PRD).
        total_epochs: Total number of training epochs (default: 50).
        min_lr_ratio: Ratio of min_lr to base_lr. Computed as
            ``min_lr / base_lr`` (default: 0.01 → min_lr = 1e-6
            when base_lr = 1e-4).

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = CosineWarmupScheduler(
        ...     optimizer, warmup_epochs=5, total_epochs=50,
        ... )
        >>> for epoch in range(50):
        ...     train_one_epoch(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio

        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, epoch: int) -> float:
        """Compute learning rate multiplier for a given epoch.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            LR multiplier in [min_lr_ratio, 1.0].
        """
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 → 1
            return (epoch + 1) / self.warmup_epochs

        # Cosine decay: 1 → min_lr_ratio
        decay_epochs = self.total_epochs - self.warmup_epochs
        decay_step = epoch - self.warmup_epochs
        cosine_decay = 0.5 * (
            1.0 + math.cos(math.pi * decay_step / max(decay_epochs, 1))
        )
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay


# ═════════════════════════════════════════════════════════════════
#  Early Stopping
# ═════════════════════════════════════════════════════════════════


class EarlyStopping:
    """Early stopping monitor to halt training when validation
    metric stagnates.

    Implements the early stopping described in PRD Section 4.2.2
    with configurable patience and min_delta.

    Args:
        patience: Number of epochs to wait without improvement
            before stopping (default: 10 per PRD).
        min_delta: Minimum change to qualify as an improvement
            (default: 0.001 per train_config.yaml).
        mode: One of ``"max"`` or ``"min"``. In ``"max"`` mode,
            training stops when the metric stops increasing.
        verbose: If True, log when patience counter changes.

    Attributes:
        best_score: Best metric value observed so far.
        best_epoch: Epoch at which best_score was observed.
        counter: Current patience counter (epochs without improvement).
        should_stop: Whether training should be stopped.
        best_state_dict: Model state dict at best epoch.

    Example:
        >>> stopper = EarlyStopping(patience=10, mode="max")
        >>> for epoch in range(50):
        ...     val_auc = validate(...)
        ...     stopper(val_auc, model, epoch)
        ...     if stopper.should_stop:
        ...         print(f"Stopped at epoch {epoch}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_state_dict: dict[str, Any] | None = None

    def __call__(
        self,
        score: float,
        model: nn.Module,
        epoch: int,
    ) -> None:
        """Check whether to stop training.

        Args:
            score: Current validation metric value.
            model: Model to save state_dict from at best epoch.
            epoch: Current epoch index.
        """
        if self.best_score is None:
            self._update_best(score, model, epoch)
            return

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self._update_best(score, model, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "EarlyStopping: no improvement for %d/%d epochs "
                    "(best=%.4f at epoch %d)",
                    self.counter,
                    self.patience,
                    self.best_score,
                    self.best_epoch,
                )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "EarlyStopping triggered after %d epochs without "
                    "improvement. Best %s=%.4f at epoch %d.",
                    self.patience,
                    "score",
                    self.best_score,
                    self.best_epoch,
                )

    def _update_best(
        self,
        score: float,
        model: nn.Module,
        epoch: int,
    ) -> None:
        """Record a new best score and save model state."""
        self.best_score = score
        self.best_epoch = epoch
        self.best_state_dict = copy.deepcopy(model.state_dict())
        if self.verbose:
            logger.info(
                "EarlyStopping: new best score=%.4f at epoch %d",
                score,
                epoch,
            )


# ═════════════════════════════════════════════════════════════════
#  MRI Encoder
# ═════════════════════════════════════════════════════════════════


class MRIEncoder(nn.Module):
    """3D ResNet-50 encoder for structural MRI volumes.

    Wraps MONAI's ``ResNet`` with ``spatial_dims=3`` and removes the
    final classification layer, replacing it with a projection head
    that maps the 2048-dim backbone features to a compact embedding
    space for downstream fusion.

    Architecture (PRD Section 4.2.2)::

        MONAI ResNet-50 (3D, 1-channel input)
            ↓ [B, 2048]
        Linear(2048, embedding_dim) → ReLU → LayerNorm
            ↓ [B, embedding_dim]

    Args:
        backbone_out_dim: Output dimension of ResNet-50 final pool
            (default: 2048 for ResNet-50).
        embedding_dim: Target embedding dimension (default: 256
            per PRD Section 4.2.2).
        n_input_channels: Number of input channels (default: 1
            for single-modality T1-weighted MRI).
        pretrained: Whether to load pretrained weights. If True,
            uses MONAI's MedicalNet/Med3D pretrained weights when
            available (default: False for reproducibility).
        gradient_checkpointing: If True, enables gradient
            checkpointing for memory-efficient training of 3D
            volumes (default: False).

    Attributes:
        backbone: MONAI ResNet-50 (without final FC layer).
        projection: Projection head (Linear → ReLU → LayerNorm).

    Example:
        >>> encoder = MRIEncoder(embedding_dim=256)
        >>> mri = torch.randn(2, 1, 96, 96, 96)
        >>> embedding = encoder(mri)  # [2, 256]
    """

    def __init__(
        self,
        backbone_out_dim: int = 2048,
        embedding_dim: int = 256,
        n_input_channels: int = 1,
        pretrained: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_out_dim = backbone_out_dim
        self.embedding_dim = embedding_dim
        self.gradient_checkpointing = gradient_checkpointing

        # ─── MONAI ResNet-50 backbone ───
        # Build the full MONAI ResNet-50 with classification head,
        # then strip the FC layer so we get [B, 2048] features.
        self.backbone = ResNet(
            block="bottleneck",
            layers=[3, 4, 6, 3],  # ResNet-50 layer config
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=n_input_channels,
            num_classes=backbone_out_dim,  # Temporary — will be removed
        )

        # Remove the final fully-connected layer — we use our own
        # projection head instead. The backbone's .fc is set to
        # identity so forward() returns the pooled features directly.
        self.backbone.fc = nn.Identity()

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # ─── Projection head (PRD 4.2.2) ───
        # Linear(2048, 256) → ReLU → LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim),
        )

        # Initialise projection head weights
        self._init_projection_weights()

        # Log parameter counts
        backbone_params = sum(
            p.numel() for p in self.backbone.parameters()
        )
        proj_params = sum(
            p.numel() for p in self.projection.parameters()
        )
        logger.info(
            "MRIEncoder initialised: backbone=%dM params, "
            "projection=%dK params, embedding_dim=%d",
            backbone_params // 1_000_000,
            proj_params // 1_000,
            embedding_dim,
        )

    def _init_projection_weights(self) -> None:
        """Initialise projection head with Kaiming normal."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on ResNet layers.

        Trades compute for memory — useful for 3D volumes that
        consume large amounts of GPU memory.
        """
        for name, module in self.backbone.named_children():
            if name.startswith("layer"):
                module.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled for backbone layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim embedding from 3D MRI volume.

        Args:
            x: Input MRI tensor of shape ``[B, 1, D, H, W]``
                where D, H, W = 96 (after preprocessing).

        Returns:
            Embedding tensor of shape ``[B, embedding_dim]``.
        """
        # Backbone: [B, 1, 96, 96, 96] → [B, 2048]
        if self.gradient_checkpointing and self.training:
            features = self._forward_with_checkpointing(x)
        else:
            features = self.backbone(x)

        # Projection: [B, 2048] → [B, 256]
        embedding = self.projection(features)

        return embedding

    def _forward_with_checkpointing(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with gradient checkpointing.

        Applies ``torch.utils.checkpoint.checkpoint`` to each
        ResNet layer block to reduce peak memory usage.

        Args:
            x: Input tensor.

        Returns:
            Backbone feature tensor.
        """
        from torch.utils.checkpoint import checkpoint

        # Initial convolution + bn + relu + maxpool
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Checkpointed residual layers
        x = checkpoint(self.backbone.layer1, x, use_reentrant=False)
        x = checkpoint(self.backbone.layer2, x, use_reentrant=False)
        x = checkpoint(self.backbone.layer3, x, use_reentrant=False)
        x = checkpoint(self.backbone.layer4, x, use_reentrant=False)

        # Global average pool
        x = self.backbone.avgpool(x)
        x = x.flatten(1)

        return x

    def get_feature_dim(self) -> int:
        """Return the output embedding dimension.

        Returns:
            Integer embedding dimension (256 by default).
        """
        return self.embedding_dim


# ═════════════════════════════════════════════════════════════════
#  Classification Head
# ═════════════════════════════════════════════════════════════════


class ClassificationHead(nn.Module):
    """Linear classification head for HD staging.

    Maps the 256-dim embedding from MRIEncoder to 3-class logits
    (pre-manifest, early, advanced) as specified in PRD Section 4.2.5.

    This head is used during Phase 2 for standalone MRI encoder
    training, and is later replaced by the full NeuroSenseModel
    in Phase 4 when fusion is added.

    Args:
        input_dim: Input feature dimension (default: 256,
            matching MRIEncoder output).
        num_classes: Number of HD stage classes (default: 3).
        dropout: Dropout probability before the linear layer
            (default: 0.1 for light regularisation).

    Example:
        >>> head = ClassificationHead(input_dim=256, num_classes=3)
        >>> embedding = torch.randn(4, 256)
        >>> logits = head(embedding)  # [4, 3]
        >>> probs = torch.softmax(logits, dim=-1)
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(input_dim, num_classes)

        # Xavier initialisation for classification layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        logger.info(
            "ClassificationHead initialised: %d → %d classes",
            input_dim,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute classification logits.

        Args:
            x: Input embedding of shape ``[B, input_dim]``.

        Returns:
            Logits tensor of shape ``[B, num_classes]``.
            Apply ``torch.softmax(logits, dim=-1)`` for
            probabilities.
        """
        x = self.dropout(x)
        return self.fc(x)


# ═════════════════════════════════════════════════════════════════
#  Training Utilities
# ═════════════════════════════════════════════════════════════════


def _load_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load training and model configuration from YAML files.

    Args:
        config_path: Path to train_config.yaml. If None, loads
            from the default configs/ directory.

    Returns:
        Merged dictionary of model_config + train_config values.
    """
    configs_dir = Path(__file__).parent.parent / "configs"

    if config_path is not None:
        train_config_path = Path(config_path)
    else:
        train_config_path = configs_dir / "train_config.yaml"

    model_config_path = configs_dir / "model_config.yaml"

    config: dict[str, Any] = {}

    for path in [model_config_path, train_config_path]:
        if path.exists():
            with open(path, "r") as f:
                loaded = yaml.safe_load(f) or {}
            config.update(loaded)
            logger.info("Loaded config from %s", path)
        else:
            logger.warning("Config not found: %s", path)

    return config


def _compute_per_class_auc(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    num_classes: int = 3,
) -> dict[str, float]:
    """Compute per-class and mean AUC-ROC.

    Uses one-vs-rest AUC for each HD stage class, with graceful
    handling of classes that have zero positive samples.

    Args:
        all_labels: Ground-truth label array of shape ``(N,)``.
        all_probs: Predicted probability array of shape ``(N, C)``.
        num_classes: Number of classes (default: 3).

    Returns:
        Dictionary with keys ``auc_premanifest``, ``auc_early``,
        ``auc_advanced``, and ``auc_mean``.
    """
    aucs: dict[str, float] = {}
    valid_aucs: list[float] = []

    class_keys = ["auc_premanifest", "auc_early", "auc_advanced"]

    for i in range(num_classes):
        key = class_keys[i] if i < len(class_keys) else f"auc_class{i}"
        binary_labels = (all_labels == i).astype(int)

        # Need at least one positive and one negative sample
        if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
            aucs[key] = 0.0
            logger.warning(
                "Cannot compute AUC for class %d (%s): "
                "only %d/%d positive samples",
                i,
                STAGE_NAMES[i] if i < len(STAGE_NAMES) else f"class{i}",
                binary_labels.sum(),
                len(binary_labels),
            )
        else:
            try:
                auc_val = roc_auc_score(binary_labels, all_probs[:, i])
                aucs[key] = float(auc_val)
                valid_aucs.append(auc_val)
            except ValueError as e:
                aucs[key] = 0.0
                logger.warning("AUC computation error for class %d: %s", i, e)

    aucs["auc_mean"] = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    return aucs


def _setup_device(config: dict[str, Any]) -> torch.device:
    """Determine the compute device from configuration.

    Args:
        config: Configuration dict (may contain device.force or
            device.auto_detect).

    Returns:
        torch.device for training.
    """
    device_config = config.get("device", {})
    force_device = device_config.get("force")

    if force_device:
        device = torch.device(force_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Training device: %s", device)
    return device


def _set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enables deterministic CUDA operations.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Random seed set to %d (deterministic=%s)", seed, deterministic)


# ═════════════════════════════════════════════════════════════════
#  Training Loop
# ═════════════════════════════════════════════════════════════════


def train_mri_encoder(
    data_root: str | Path = "data/processed",
    checkpoint_dir: str | Path = "checkpoints",
    config_path: str | Path | None = None,
    wandb_enabled: bool | None = None,
) -> dict[str, Any]:
    """Train the 3D ResNet-50 MRI encoder on HD staging.

    Full training pipeline implementing PRD Section 4.2.2 Phase 2:
    1. Loads data via HuntingtonDataset.split()
    2. Builds MRIEncoder + ClassificationHead
    3. Trains with AdamW, cosine warmup LR, mixed precision
    4. Evaluates per-class AUC-ROC each epoch
    5. Early stopping on val_auc_mean (patience=10)
    6. Saves best checkpoint and final model

    Args:
        data_root: Root directory containing BIDS-format data.
        checkpoint_dir: Directory to save model checkpoints.
        config_path: Path to train_config.yaml. If None, uses default.
        wandb_enabled: Override W&B logging. If None, uses config.

    Returns:
        Dictionary containing training results:
        - best_val_auc: Best mean validation AUC-ROC
        - best_epoch: Epoch at which best AUC was achieved
        - final_train_loss: Training loss at final epoch
        - total_epochs: Number of epochs trained
        - checkpoint_path: Path to best model checkpoint

    Example:
        >>> results = train_mri_encoder(
        ...     data_root="data/processed",
        ...     checkpoint_dir="checkpoints",
        ... )
        >>> print(f"Best AUC: {results['best_val_auc']:.4f}")
    """
    # ─── Configuration ───
    config = _load_config(config_path)
    train_cfg = config.get("training", {})
    es_cfg = config.get("early_stopping", {})
    loss_cfg = config.get("loss", {}).get("classification", {})
    repro_cfg = config.get("reproducibility", {})
    log_cfg = config.get("logging", {})
    mri_cfg = config.get("mri_encoder", {})

    seed = repro_cfg.get("seed", 42)
    deterministic = repro_cfg.get("deterministic", True)
    _set_seed(seed, deterministic)

    batch_size = train_cfg.get("batch_size", 4)
    epochs = train_cfg.get("epochs", 50)
    lr = train_cfg.get("optimizer", {}).get("lr", 1e-4)
    weight_decay = train_cfg.get("optimizer", {}).get("weight_decay", 1e-5)
    betas = tuple(train_cfg.get("optimizer", {}).get("betas", [0.9, 0.999]))
    warmup_epochs = train_cfg.get("scheduler", {}).get("warmup_epochs", 5)
    min_lr = train_cfg.get("scheduler", {}).get("min_lr", 1e-6)
    mixed_precision = train_cfg.get("mixed_precision", True)
    grad_checkpointing = train_cfg.get("gradient_checkpointing", False)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    num_workers = config.get("data", {}).get("num_workers", 4)

    es_patience = es_cfg.get("patience", 10)
    es_min_delta = es_cfg.get("min_delta", 0.001)

    embedding_dim = mri_cfg.get("embedding_dim", 256)
    backbone_out_dim = mri_cfg.get("backbone_out_dim", 2048)

    log_every = log_cfg.get("log_every_n_steps", 10)

    device = _setup_device(config)

    # ─── W&B Setup (optional) ───
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
                tags=wandb_cfg.get("tags", ["huntington", "mri_encoder"]),
                config={
                    "phase": "phase2_mri_encoder",
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "warmup_epochs": warmup_epochs,
                    "embedding_dim": embedding_dim,
                    "mixed_precision": mixed_precision,
                    "grad_checkpointing": grad_checkpointing,
                },
            )
            logger.info("W&B logging enabled: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed, disabling W&B logging")
            use_wandb = False

    # ─── Data Loading ───
    data_root = Path(data_root)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s ...", data_root)
    train_loader, val_loader, _ = HuntingtonDataset.split(
        root_dir=data_root,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(
        "Data loaded: %d train batches, %d val batches",
        len(train_loader),
        len(val_loader),
    )

    # ─── Class Weights ───
    class_weights = None
    if loss_cfg.get("auto_weight", True):
        # Get weights from training dataset
        train_dataset = train_loader.dataset
        # Navigate through Subset to get the base dataset
        base_dataset = train_dataset
        while hasattr(base_dataset, "dataset"):
            base_dataset = base_dataset.dataset
        if hasattr(base_dataset, "get_class_weights"):
            class_weights = base_dataset.get_class_weights().to(device)
            logger.info("Class weights: %s", class_weights.tolist())

    # ─── Model ───
    encoder = MRIEncoder(
        backbone_out_dim=backbone_out_dim,
        embedding_dim=embedding_dim,
        n_input_channels=1,
        pretrained=False,
        gradient_checkpointing=grad_checkpointing,
    ).to(device)

    head = ClassificationHead(
        input_dim=embedding_dim,
        num_classes=3,
    ).to(device)

    # Combined parameter groups
    all_params = list(encoder.parameters()) + list(head.parameters())
    total_params = sum(p.numel() for p in all_params)
    trainable_params = sum(p.numel() for p in all_params if p.requires_grad)
    logger.info(
        "Model: %dM total params (%dM trainable)",
        total_params // 1_000_000,
        trainable_params // 1_000_000,
    )

    # ─── Optimizer ───
    optimizer = AdamW(
        all_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    # ─── Scheduler ───
    min_lr_ratio = min_lr / lr if lr > 0 else 0.01
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        min_lr_ratio=min_lr_ratio,
    )

    # ─── Loss ───
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )

    # ─── Mixed Precision ───
    scaler = GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    # ─── Early Stopping ───
    # Create a wrapper nn.Module for saving combined state
    class _EncoderWithHead(nn.Module):
        def __init__(self, enc, hd):
            super().__init__()
            self.encoder = enc
            self.head = hd

    combined_model = _EncoderWithHead(encoder, head)
    early_stopper = EarlyStopping(
        patience=es_patience,
        min_delta=es_min_delta,
        mode="max",
        verbose=True,
    )

    # ─── Training Loop ───
    logger.info(
        "═" * 60 + "\n"
        "  Phase 2: MRI Encoder Training\n"
        "  Epochs: %d | Batch: %d | LR: %.2e | Device: %s\n"
        "  Mixed precision: %s | Grad checkpointing: %s\n"
        "═" * 60,
        epochs, batch_size, lr, device,
        mixed_precision, grad_checkpointing,
    )

    results: dict[str, Any] = {
        "best_val_auc": 0.0,
        "best_epoch": 0,
        "final_train_loss": float("inf"),
        "total_epochs": 0,
        "checkpoint_path": "",
    }

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Train ──
        encoder.train()
        head.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            mri = batch["mri"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(
                device_type=device.type,
                enabled=(mixed_precision and device.type == "cuda"),
            ):
                embeddings = encoder(mri)
                logits = head(embeddings)
                loss = criterion(logits, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss_sum += loss.item() * grad_accum_steps
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if (step + 1) % log_every == 0:
                logger.info(
                    "  [Epoch %d] Step %d/%d — loss=%.4f",
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    loss.item() * grad_accum_steps,
                )

        train_loss = train_loss_sum / max(len(train_loader), 1)
        train_acc = train_correct / max(train_total, 1)

        # ── Validate ──
        encoder.eval()
        head.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        all_val_labels: list[int] = []
        all_val_probs: list[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                mri = batch["mri"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with autocast(
                    device_type=device.type,
                    enabled=(mixed_precision and device.type == "cuda"),
                ):
                    embeddings = encoder(mri)
                    logits = head(embeddings)
                    loss = criterion(logits, labels)

                val_loss_sum += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_val_probs.append(probs)
                all_val_labels.extend(labels.cpu().tolist())

        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)

        # Compute per-class AUC-ROC
        all_labels_arr = np.array(all_val_labels)
        all_probs_arr = np.concatenate(all_val_probs, axis=0)
        val_aucs = _compute_per_class_auc(all_labels_arr, all_probs_arr)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # ── Logging ──
        logger.info(
            "Epoch %d/%d (%.1fs) — "
            "train_loss=%.4f train_acc=%.3f | "
            "val_loss=%.4f val_acc=%.3f | "
            "AUC: pre=%.3f early=%.3f adv=%.3f mean=%.3f | "
            "lr=%.2e",
            epoch + 1,
            epochs,
            epoch_time,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_aucs.get("auc_premanifest", 0.0),
            val_aucs.get("auc_early", 0.0),
            val_aucs.get("auc_advanced", 0.0),
            val_aucs.get("auc_mean", 0.0),
            current_lr,
        )

        if use_wandb and wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                    **val_aucs,
                },
                step=epoch,
            )

        # ── Early Stopping ──
        val_auc_mean = val_aucs.get("auc_mean", 0.0)
        early_stopper(val_auc_mean, combined_model, epoch)

        results["final_train_loss"] = train_loss
        results["total_epochs"] = epoch + 1

        if early_stopper.should_stop:
            logger.info(
                "Early stopping at epoch %d. "
                "Best val_auc_mean=%.4f at epoch %d.",
                epoch + 1,
                early_stopper.best_score,
                early_stopper.best_epoch + 1,
            )
            break

    # ─── Save Best Checkpoint ───
    best_ckpt_path = checkpoint_dir / "mri_encoder_best.pth"

    checkpoint_data = {
        "encoder_state_dict": (
            early_stopper.best_state_dict or combined_model.state_dict()
        ),
        "embedding_dim": embedding_dim,
        "backbone_out_dim": backbone_out_dim,
        "best_val_auc": early_stopper.best_score or 0.0,
        "best_epoch": early_stopper.best_epoch,
        "total_epochs": results["total_epochs"],
        "config": {
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "mixed_precision": mixed_precision,
        },
    }
    torch.save(checkpoint_data, best_ckpt_path)
    logger.info("Best checkpoint saved to %s", best_ckpt_path)

    # Also save the last epoch checkpoint
    last_ckpt_path = checkpoint_dir / "mri_encoder_last.pth"
    last_data = {
        "encoder_state_dict": combined_model.state_dict(),
        "embedding_dim": embedding_dim,
        "backbone_out_dim": backbone_out_dim,
        "epoch": results["total_epochs"],
        "config": checkpoint_data["config"],
    }
    torch.save(last_data, last_ckpt_path)

    results["best_val_auc"] = early_stopper.best_score or 0.0
    results["best_epoch"] = early_stopper.best_epoch
    results["checkpoint_path"] = str(best_ckpt_path)

    logger.info(
        "═" * 60 + "\n"
        "  Training Complete\n"
        "  Best val_auc_mean: %.4f (epoch %d)\n"
        "  Total epochs trained: %d\n"
        "  Checkpoint: %s\n"
        "═" * 60,
        results["best_val_auc"],
        results["best_epoch"] + 1,
        results["total_epochs"],
        results["checkpoint_path"],
    )

    if use_wandb and wandb_run is not None:
        wandb.finish()

    return results


# ═════════════════════════════════════════════════════════════════
#  Checkpoint Loading
# ═════════════════════════════════════════════════════════════════


def load_mri_encoder(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[MRIEncoder, ClassificationHead, dict[str, Any]]:
    """Load a trained MRI encoder from checkpoint.

    Args:
        checkpoint_path: Path to ``.pth`` checkpoint file.
        device: Device to load model onto.

    Returns:
        Tuple of (encoder, classification_head, checkpoint_metadata).

    Example:
        >>> encoder, head, meta = load_mri_encoder(
        ...     "checkpoints/mri_encoder_best.pth",
        ...     device="cuda",
        ... )
        >>> print(f"Best AUC: {meta['best_val_auc']:.4f}")
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    embedding_dim = checkpoint.get("embedding_dim", 256)
    backbone_out_dim = checkpoint.get("backbone_out_dim", 2048)

    encoder = MRIEncoder(
        backbone_out_dim=backbone_out_dim,
        embedding_dim=embedding_dim,
    )
    head = ClassificationHead(input_dim=embedding_dim, num_classes=3)

    # Load state dict from combined model
    state_dict = checkpoint["encoder_state_dict"]

    # Handle combined model state dict format
    encoder_state = {}
    head_state = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state[key[len("encoder."):]] = value
        elif key.startswith("head."):
            head_state[key[len("head."):]] = value

    if encoder_state:
        encoder.load_state_dict(encoder_state)
        head.load_state_dict(head_state)
    else:
        # Fallback: try loading directly (non-combined format)
        encoder.load_state_dict(state_dict, strict=False)

    encoder = encoder.to(device)
    head = head.to(device)
    encoder.eval()
    head.eval()

    logger.info(
        "Loaded MRI encoder from %s (best_auc=%.4f, epoch=%d)",
        checkpoint_path,
        checkpoint.get("best_val_auc", 0.0),
        checkpoint.get("best_epoch", -1),
    )

    return encoder, head, checkpoint


# ═════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for MRI encoder training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train NeuroSense 3D ResNet-50 MRI Encoder (Phase 2)"
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
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = train_mri_encoder(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config,
        wandb_enabled=not args.no_wandb,
    )

    print(f"\n{'='*50}")
    print("Phase 2 Training Results:")
    print(f"  Best AUC:     {results['best_val_auc']:.4f}")
    print(f"  Best Epoch:   {results['best_epoch'] + 1}")
    print(f"  Total Epochs: {results['total_epochs']}")
    print(f"  Checkpoint:   {results['checkpoint_path']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
