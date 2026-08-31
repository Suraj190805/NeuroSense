"""NeuroSense Unit Tests — Model Regularization & Layer Freezing.

Tests the 4 pillars of architecture regularization:
1. Dropout: p=0.5 in classification head.
2. Layer Freezing: conv1, bn1, layer1, layer2 frozen, layer3/4 trainable.
3. Label Smoothing: Cross-entropy loss with smoothing=0.1.
4. Parameter Groups: Selective parameter dispatch with differential learning rates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neurosense.models.image_classifier import HDImageClassifier
from neurosense.training.losses import WeightedCrossEntropyLoss
from neurosense.training.train_image import train_hd_image


def test_dropout_rate():
    """Verify classification head has dropout p=0.5."""
    model = HDImageClassifier(num_classes=2, dropout=0.5)

    # Find dropout module in classifier
    dropout_modules = [m for m in model.classifier if isinstance(m, nn.Dropout)]
    assert len(dropout_modules) >= 1, "Classification head must contain nn.Dropout"
    assert dropout_modules[0].p == 0.5, f"Expected dropout p=0.5, got {dropout_modules[0].p}"


def test_early_backbone_layer_freezing():
    """Verify conv1, bn1, layer1, layer2 are frozen and layer3, layer4 are trainable."""
    model = HDImageClassifier(num_classes=2, freeze_early_layers=True)

    # Check frozen early layers
    for param in model.backbone.conv1.parameters():
        assert not param.requires_grad, "conv1 must be frozen"
    for param in model.backbone.bn1.parameters():
        assert not param.requires_grad, "bn1 must be frozen"
    for param in model.backbone.layer1.parameters():
        assert not param.requires_grad, "layer1 must be frozen"
    for param in model.backbone.layer2.parameters():
        assert not param.requires_grad, "layer2 must be frozen"

    # Check trainable late layers and head
    for param in model.backbone.layer3.parameters():
        assert param.requires_grad, "layer3 must be trainable"
    for param in model.backbone.layer4.parameters():
        assert param.requires_grad, "layer4 must be trainable"
    for param in model.classifier.parameters():
        assert param.requires_grad, "classification head must be trainable"


def test_unfreeze_all_backbone():
    """Verify unfreeze_all() restores full trainability."""
    model = HDImageClassifier(num_classes=2, freeze_early_layers=True)
    model.unfreeze_all()

    for param in model.backbone.parameters():
        assert param.requires_grad, "All backbone parameters must be trainable after unfreeze_all()"


def test_selective_param_groups():
    """Param groups must only include parameters that require gradients."""
    model = HDImageClassifier(num_classes=2, freeze_early_layers=True)
    groups = model.get_param_groups(backbone_lr=1e-4, head_lr=1e-3)

    assert len(groups) == 2, "Should have backbone and head groups"
    for group in groups:
        for p in group["params"]:
            assert p.requires_grad, "Frozen parameters must not be included in optimizer groups"


def test_label_smoothing_regularization():
    """Verify label smoothing=0.1 distributes probability targets away from 1.0."""
    loss_smooth = WeightedCrossEntropyLoss(label_smoothing=0.1, num_classes=2)
    loss_hard = nn.CrossEntropyLoss(label_smoothing=0.0)

    logits = torch.tensor([[10.0, -10.0]])  # Highly confident prediction
    target = torch.tensor([0])

    val_smooth = loss_smooth(logits, target).item()
    val_hard = loss_hard(logits, target).item()

    # Label smoothing penalizes extreme overconfidence
    assert val_smooth > val_hard, "Label smoothing must penalize extreme logit overconfidence"


def test_regularized_training_pipeline_setup():
    """Verify training setup initializes with regularized parameters."""
    config = train_hd_image(
        dropout=0.5,
        label_smoothing=0.1,
        weight_decay=1e-3,
        freeze_early_layers=True,
    )
    assert config["dropout"] == 0.5
    assert config["label_smoothing"] == 0.1
    assert config["weight_decay"] == 1e-3


if __name__ == "__main__":
    test_dropout_rate()
    print("✅ Test 1: Dropout (p=0.5) passed")
    test_early_backbone_layer_freezing()
    print("✅ Test 2: Early Layer Freezing (conv1/bn1/layer1/layer2 frozen, layer3/4 trainable) passed")
    test_unfreeze_all_backbone()
    print("✅ Test 3: Unfreeze All Backbone passed")
    test_selective_param_groups()
    print("✅ Test 4: Selective Parameter Groups passed")
    test_label_smoothing_regularization()
    print("✅ Test 5: Label Smoothing (0.1) passed")
    test_regularized_training_pipeline_setup()
    print("✅ Test 6: Regularized Training Setup passed")
    print("\n🎉 ALL 6 REGULARIZATION TESTS PASSED CLEANLY!")
