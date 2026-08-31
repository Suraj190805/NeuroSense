"""Huntington's Disease 2D MRI Image Classifier.

Uses a pretrained ResNet-50 backbone with a custom classification
head for 2D Brain MRI slice feature extraction and classification.

The backbone is ImageNet-pretrained, providing robust feature
extraction on medical neuroimaging scans.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class HDImageClassifier(nn.Module):
    """2D ResNet-50 classifier for Huntington's Disease Brain MRI slice detection.

    Architecture & Regularization:
        - Backbone: ResNet-50 (ImageNet pretrained)
        - Early Layer Freezing: conv1, bn1, layer1, and layer2 frozen to prevent
          overfitting on small datasets and preserve general visual features.
        - Trainable Layers: layer3, layer4, and classification head.
        - Global Average Pooling (from ResNet)
        - Heavy Dropout Regularization (p=0.5) in classification head.
        - Head: Linear(2048→512) → ReLU → Dropout(0.5) → Linear(512→num_classes)

    Args:
        num_classes: Number of output classes (default: 2 for binary normal vs HD,
            or 3 for multi-stage staging).
        dropout: Dropout rate in classification head (default: 0.5 for strong regularization).
        freeze_early_layers: If True, freezes conv1, bn1, layer1, and layer2.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        freeze_early_layers: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Load pretrained ResNet-50
        import ssl
        _orig_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        finally:
            ssl._create_default_https_context = _orig_ctx
        backbone_out_dim = self.backbone.fc.in_features  # 2048

        # Replace the final FC with identity — custom head attached below
        self.backbone.fc = nn.Identity()

        # Classification head with strong dropout regularization
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        # Freeze early backbone layers (conv1, bn1, layer1, layer2)
        if freeze_early_layers:
            self.freeze_early_layers(freeze=True)

        self._log_parameter_summary()

    def freeze_early_layers(self, freeze: bool = True) -> None:
        """Freeze or unfreeze early backbone layers (conv1, bn1, layer1, layer2).

        Freezing lower layers prevents destruction of general visual features and
        dramatically reduces overfitting risks on small medical imaging datasets.
        High-level layers (layer3, layer4) remain trainable for brain pathology.

        Args:
            freeze: If True, freezes early layers (requires_grad=False).
        """
        early_components = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
            self.backbone.layer2,
        ]
        for comp in early_components:
            for param in comp.parameters():
                param.requires_grad = not freeze

        status = "FROZEN" if freeze else "UNFROZEN"
        logger.info(
            "ResNet-50 early backbone layers (conv1, bn1, layer1, layer2) %s for strong regularization.",
            status,
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("All ResNet-50 backbone layers unfrozen for full fine-tuning.")

    def _log_parameter_summary(self) -> None:
        """Log trainable vs frozen parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        logger.info(
            "HDImageClassifier parameter summary: total=%dM, trainable=%dM, frozen=%dM (dropout=%.2f)",
            total_params // 1_000_000,
            trainable_params // 1_000_000,
            frozen_params // 1_000_000,
            self.dropout_rate,
        )

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, 3, H, W].

        Returns:
            Dictionary with:
            - logits: Raw logits [B, num_classes]
            - probabilities: Softmax probabilities [B, num_classes]
            - predicted_class: Predicted class indices [B]
        """
        features = self.backbone(x)  # [B, 2048]
        logits = self.classifier(features)  # [B, num_classes]
        probs = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": logits.argmax(dim=-1),
        }

    def get_param_groups(
        self, backbone_lr: float = 1e-4, head_lr: float = 1e-3
    ) -> list[dict]:
        """Get parameter groups for optimizer with differential learning rates.

        Only includes trainable parameters (skipping frozen early layers).

        Args:
            backbone_lr: Learning rate for trainable backbone layers (layer3, layer4).
            head_lr: Learning rate for classification head.

        Returns:
            List of param group dicts for the optimizer.
        """
        trainable_backbone = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        trainable_head = [
            p for p in self.classifier.parameters() if p.requires_grad
        ]

        groups = []
        if trainable_backbone:
            groups.append({
                "params": trainable_backbone,
                "lr": backbone_lr,
                "name": "backbone_trainable",
            })
        if trainable_head:
            groups.append({
                "params": trainable_head,
                "lr": head_lr,
                "name": "classifier_head",
            })

        return groups


# Alias for backward compatibility
ParkinsonsClassifier = HDImageClassifier
