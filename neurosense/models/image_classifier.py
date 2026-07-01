"""Parkinson's Disease 2D Image Classifier.

Uses a pretrained ResNet-50 backbone with a custom classification
head for binary classification (normal vs parkinson) from 2D MRI
brain slices.

The backbone is ImageNet-pretrained, providing strong feature
extraction even on small medical datasets. The classification
head is trained from scratch with a higher learning rate.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class ParkinsonsClassifier(nn.Module):
    """2D ResNet-50 classifier for Parkinson's disease detection.

    Architecture:
        - Backbone: ResNet-50 (ImageNet pretrained)
        - Global Average Pooling (from ResNet)
        - Head: Linear(2048→512) → ReLU → Dropout → Linear(512→2)

    Args:
        num_classes: Number of output classes (default: 2).
        dropout: Dropout rate in classification head.
        freeze_backbone_epochs: Not used at init, but stored
            for the training loop to optionally freeze early.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained ResNet-50
        # Workaround for macOS SSL cert issues when downloading weights
        import ssl
        _orig_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        finally:
            ssl._create_default_https_context = _orig_ctx
        backbone_out_dim = self.backbone.fc.in_features  # 2048

        # Replace the final FC with identity — we add our own head
        self.backbone.fc = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        # Count parameters
        backbone_params = sum(
            p.numel() for p in self.backbone.parameters()
        )
        head_params = sum(
            p.numel() for p in self.classifier.parameters()
        )
        total_params = backbone_params + head_params

        logger.info(
            "ParkinsonsClassifier initialised: "
            "backbone=%dM params, head=%dK params, total=%dM params",
            backbone_params // 1_000_000,
            head_params // 1_000,
            total_params // 1_000_000,
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
        """Get parameter groups with differential learning rates.

        The backbone uses a lower LR since it's pretrained, while
        the classification head trains with a higher LR.

        Args:
            backbone_lr: Learning rate for backbone parameters.
            head_lr: Learning rate for classification head.

        Returns:
            List of param group dicts for the optimizer.
        """
        return [
            {
                "params": self.backbone.parameters(),
                "lr": backbone_lr,
                "name": "backbone",
            },
            {
                "params": self.classifier.parameters(),
                "lr": head_lr,
                "name": "classifier_head",
            },
        ]
