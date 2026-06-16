"""NeuroSense custom loss functions.

Implements the loss functions specified in PRD Sections 4.2.5–4.2.6:
- Weighted cross-entropy with inverse frequency weighting for
  classification (handles class imbalance in HD staging)
- Huber loss for progression regression (robust to outliers)
- Combined multi-task loss with configurable weighting

Usage:
    from neurosense.training.losses import (
        WeightedCrossEntropyLoss,
        ProgressionHuberLoss,
        CombinedLoss,
    )

    criterion = CombinedLoss(class_weights=weights)
    loss, loss_dict = criterion(
        logits=logits,
        labels=labels,
        deltas_pred=deltas,
        deltas_target=targets,
    )
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with inverse frequency class weighting.

    Addresses the class imbalance in HD staging datasets where
    pre-manifest subjects significantly outnumber advanced HD
    subjects (PRD Section 4.2.5).

    Includes optional label smoothing for regularisation and
    better calibration of predicted probabilities.

    Args:
        class_weights: Per-class weight tensor of shape
            ``[num_classes]``. If None, uniform weights are used.
        label_smoothing: Label smoothing factor (default: 0.05).
            Redistributes probability mass from the true class
            to all classes for better calibration.
        num_classes: Number of classes (default: 3).

    Example:
        >>> weights = torch.tensor([0.5, 1.2, 2.3])
        >>> criterion = WeightedCrossEntropyLoss(class_weights=weights)
        >>> logits = torch.randn(4, 3)
        >>> labels = torch.tensor([0, 1, 2, 0])
        >>> loss = criterion(logits, labels)
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.05,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer(
                "class_weights",
                torch.ones(num_classes),
            )

        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing,
        )

        logger.info(
            "WeightedCrossEntropyLoss: weights=%s, smoothing=%.3f",
            self.class_weights.tolist(),
            label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits: Model output logits ``[B, num_classes]``.
            labels: Ground-truth class indices ``[B]``.

        Returns:
            Scalar loss tensor.
        """
        return self.criterion(logits, labels)

    def update_weights(self, class_weights: torch.Tensor) -> None:
        """Update class weights dynamically.

        Useful when class distribution changes between training
        runs or cross-validation folds.

        Args:
            class_weights: New per-class weights ``[num_classes]``.
        """
        self.class_weights.copy_(class_weights.float())
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        logger.info(
            "Updated class weights: %s",
            self.class_weights.tolist(),
        )


class ProgressionHuberLoss(nn.Module):
    """Huber loss for UHDRS motor score progression prediction.

    Combines the advantages of L1 (robust to outliers) and L2
    (smooth gradients near zero) loss functions. Uses delta=1.0
    as default, matching PRD Section 4.2.6.

    The Huber loss is defined as:
    - ``0.5 * (y - ŷ)²``  when ``|y - ŷ| < delta``
    - ``delta * (|y - ŷ| - 0.5 * delta)``  otherwise

    Args:
        delta: Threshold at which to switch from L2 to L1
            (default: 1.0 per train_config.yaml).

    Example:
        >>> criterion = ProgressionHuberLoss(delta=1.0)
        >>> pred = torch.randn(4, 2)
        >>> target = torch.randn(4, 2)
        >>> loss = criterion(pred, target)
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()

        self.delta = delta
        self.criterion = nn.HuberLoss(delta=delta, reduction="mean")

        logger.info("ProgressionHuberLoss: delta=%.2f", delta)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Huber loss for progression deltas.

        Args:
            predictions: Predicted deltas ``[B, 2]``.
            targets: Ground-truth deltas ``[B, 2]``.
            mask: Optional boolean mask ``[B]`` indicating which
                samples have valid progression targets. Useful
                when not all subjects have follow-up data.

        Returns:
            Scalar loss tensor.
        """
        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(
                    0.0, device=predictions.device, requires_grad=True
                )
            predictions = predictions[mask]
            targets = targets[mask]

        return self.criterion(predictions, targets)


class FocalLoss(nn.Module):
    """Focal loss for hard-example mining in imbalanced classification.

    Alternative to weighted cross-entropy that down-weights
    well-classified examples and focuses on hard negatives.
    Provided for ablation studies.

    Focal loss: ``-α_t * (1 - p_t)^γ * log(p_t)``

    Args:
        alpha: Per-class weight tensor ``[num_classes]`` or None.
        gamma: Focusing parameter (default: 2.0). Higher values
            focus more on hard examples.
        reduction: Loss reduction method (default: "mean").

    Example:
        >>> criterion = FocalLoss(gamma=2.0)
        >>> loss = criterion(logits, labels)
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

        logger.info("FocalLoss: gamma=%.1f", gamma)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw logits ``[B, C]``.
            labels: Ground-truth labels ``[B]``.

        Returns:
            Scalar focal loss.
        """
        ce_loss = F.cross_entropy(
            logits, labels, reduction="none"
        )
        p_t = torch.exp(-ce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[labels]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """Multi-task loss combining classification and progression.

    Weighted sum of classification and progression losses as
    specified in PRD train_config.yaml:
    ``L = w_cls * L_cls + w_prog * L_prog``

    Default weights: classification=1.0, progression=0.5

    Args:
        class_weights: Per-class weights for cross-entropy.
        classification_weight: Weight for classification loss
            (default: 1.0).
        progression_weight: Weight for progression loss
            (default: 0.5).
        label_smoothing: Label smoothing factor (default: 0.05).
        huber_delta: Huber loss delta (default: 1.0).
        use_focal: If True, use FocalLoss instead of cross-entropy
            (default: False).
        focal_gamma: Focal loss gamma (default: 2.0).

    Example:
        >>> criterion = CombinedLoss(
        ...     class_weights=torch.tensor([0.5, 1.2, 2.3]),
        ... )
        >>> loss, loss_dict = criterion(
        ...     logits=logits,
        ...     labels=labels,
        ...     deltas_pred=deltas,
        ...     deltas_target=targets,
        ... )
        >>> # loss_dict = {"cls_loss": ..., "prog_loss": ..., "total": ...}
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        classification_weight: float = 1.0,
        progression_weight: float = 0.5,
        label_smoothing: float = 0.05,
        huber_delta: float = 1.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()

        self.classification_weight = classification_weight
        self.progression_weight = progression_weight

        # Classification loss
        if use_focal:
            self.cls_criterion = FocalLoss(
                alpha=class_weights,
                gamma=focal_gamma,
            )
        else:
            self.cls_criterion = WeightedCrossEntropyLoss(
                class_weights=class_weights,
                label_smoothing=label_smoothing,
            )

        # Progression loss
        self.prog_criterion = ProgressionHuberLoss(delta=huber_delta)

        logger.info(
            "CombinedLoss: cls_weight=%.2f, prog_weight=%.2f, "
            "focal=%s",
            classification_weight,
            progression_weight,
            use_focal,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        deltas_pred: torch.Tensor | None = None,
        deltas_target: torch.Tensor | None = None,
        progression_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined multi-task loss.

        Args:
            logits: Classification logits ``[B, num_classes]``.
            labels: Ground-truth labels ``[B]``.
            deltas_pred: Predicted progression deltas ``[B, 2]``.
                If None, only classification loss is computed.
            deltas_target: Ground-truth progression deltas ``[B, 2]``.
                If None, only classification loss is computed.
            progression_mask: Boolean mask ``[B]`` for samples
                with valid progression data.

        Returns:
            Tuple of (total_loss, loss_breakdown_dict).
            The dict contains 'cls_loss', 'prog_loss', and 'total'
            as Python floats (for logging).
        """
        # Classification loss
        cls_loss = self.cls_criterion(logits, labels)

        # Progression loss (optional — not all datasets have follow-up)
        if (
            deltas_pred is not None
            and deltas_target is not None
            and self.progression_weight > 0
        ):
            prog_loss = self.prog_criterion(
                deltas_pred, deltas_target, mask=progression_mask
            )
        else:
            prog_loss = torch.tensor(
                0.0, device=logits.device, requires_grad=False
            )

        # Combined loss
        total_loss = (
            self.classification_weight * cls_loss
            + self.progression_weight * prog_loss
        )

        loss_dict = {
            "cls_loss": cls_loss.item(),
            "prog_loss": prog_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict
