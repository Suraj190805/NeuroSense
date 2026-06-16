"""NeuroSense Unit Tests — Loss Functions.

Tests for all training loss functions:
- WeightedCrossEntropyLoss
- ProgressionHuberLoss
- FocalLoss
- CombinedLoss
"""

from __future__ import annotations

import pytest
import torch


BATCH_SIZE = 4
NUM_CLASSES = 3


class TestWeightedCrossEntropyLoss:
    """Test suite for class-weighted cross-entropy loss."""

    def test_output_is_scalar(self):
        """Loss returns a scalar tensor."""
        from neurosense.training.losses import WeightedCrossEntropyLoss

        loss_fn = WeightedCrossEntropyLoss(num_classes=NUM_CLASSES)
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        loss = loss_fn(logits, labels)

        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
        assert loss.item() > 0, "Loss should be positive"

    def test_with_class_weights(self):
        """Loss accepts explicit class weights."""
        from neurosense.training.losses import WeightedCrossEntropyLoss

        weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = WeightedCrossEntropyLoss(
            num_classes=NUM_CLASSES, class_weights=weights
        )
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        loss = loss_fn(logits, labels)
        assert loss.item() > 0


class TestProgressionHuberLoss:
    """Test suite for progression regression loss."""

    def test_output_is_scalar(self):
        """Loss returns a scalar tensor."""
        from neurosense.training.losses import ProgressionHuberLoss

        loss_fn = ProgressionHuberLoss()
        preds = torch.randn(BATCH_SIZE, 2)
        targets = torch.randn(BATCH_SIZE, 2)

        loss = loss_fn(preds, targets)

        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_zero_loss_for_identical(self):
        """Loss is zero when predictions match targets."""
        from neurosense.training.losses import ProgressionHuberLoss

        loss_fn = ProgressionHuberLoss()
        values = torch.randn(BATCH_SIZE, 2)

        loss = loss_fn(values, values.clone())
        assert loss.item() < 1e-6


class TestFocalLoss:
    """Test suite for focal loss."""

    def test_output_is_scalar(self):
        """Focal loss returns a scalar."""
        from neurosense.training.losses import FocalLoss

        loss_fn = FocalLoss(num_classes=NUM_CLASSES, gamma=2.0)
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        loss = loss_fn(logits, labels)

        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gamma_zero_matches_ce(self):
        """Focal loss with gamma=0 approximates cross-entropy."""
        from neurosense.training.losses import FocalLoss

        loss_fn = FocalLoss(num_classes=NUM_CLASSES, gamma=0.0)
        ce_fn = torch.nn.CrossEntropyLoss()

        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        focal_loss = loss_fn(logits, labels)
        ce_loss = ce_fn(logits, labels)

        # Should be approximately equal (within 10%)
        assert abs(focal_loss.item() - ce_loss.item()) < ce_loss.item() * 0.1


class TestCombinedLoss:
    """Test suite for combined multi-task loss."""

    def test_output_is_scalar(self):
        """Combined loss returns a scalar."""
        from neurosense.training.losses import CombinedLoss

        loss_fn = CombinedLoss(num_classes=NUM_CLASSES)

        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        pred_deltas = torch.randn(BATCH_SIZE, 2)
        true_deltas = torch.randn(BATCH_SIZE, 2)

        loss = loss_fn(logits, labels, pred_deltas, true_deltas)

        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gradient_flow(self):
        """Gradients flow through combined loss."""
        from neurosense.training.losses import CombinedLoss

        loss_fn = CombinedLoss(num_classes=NUM_CLASSES)

        logits = torch.randn(BATCH_SIZE, NUM_CLASSES, requires_grad=True)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        pred_deltas = torch.randn(BATCH_SIZE, 2, requires_grad=True)
        true_deltas = torch.randn(BATCH_SIZE, 2)

        loss = loss_fn(logits, labels, pred_deltas, true_deltas)
        loss.backward()

        assert logits.grad is not None
        assert pred_deltas.grad is not None
