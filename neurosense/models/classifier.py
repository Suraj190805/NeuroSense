"""NeuroSense Composite Model and Task Heads.

Implements the classification and progression heads (PRD Sections
4.2.5–4.2.6) and the full NeuroSenseModel that composes all
modules into a single forward pass (Phase 4).

Components:
    ProgressionHead: Linear(256→2) for 12/24-month UHDRS motor delta
    NeuroSenseModel: MRI + Clinical + Fusion + Classification + Progression

Architecture::

    MRI [B,1,96,96,96]          Clinical [B,T,5]  +  lengths [B]
         ↓                              ↓
    MRIEncoder                   ClinicalEncoder
         ↓                              ↓
    img_emb [B,256]              clin_emb [B,256]
         ↓                              ↓
         └────── CrossModalFusion ──────┘
                       ↓
                fused_emb [B,256]
                 ↓              ↓
        ClassificationHead   ProgressionHead
             ↓                    ↓
        logits [B,3]         deltas [B,2]
        (staging)         (12mo, 24mo Δ)

Usage:
    from neurosense.models.classifier import NeuroSenseModel

    model = NeuroSenseModel()
    outputs = model(mri, clinical, lengths)
    # outputs["logits"]     → [B, 3]
    # outputs["probs"]      → [B, 3]
    # outputs["deltas"]     → [B, 2]
    # outputs["risk"]       → ["low", "medium", "high", ...]
    # outputs["embeddings"] → dict of intermediate embeddings
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neurosense.models.clinical_encoder import ClinicalEncoder
from neurosense.models.fusion import (
    ConcatenationFusion,
    CrossModalFusion,
)
from neurosense.models.mri_encoder import (
    ClassificationHead,
    MRIEncoder,
)

logger = logging.getLogger(__name__)


# ─── Risk thresholds from model_config.yaml (PRD 4.2.6) ───
DEFAULT_RISK_THRESHOLDS: dict[str, float] = {
    "low_max": 3.0,     # Δ < 3.0 → low risk
    "medium_max": 8.0,  # 3.0 ≤ Δ < 8.0 → medium risk
    # Δ ≥ 8.0 → high risk
}


class ProgressionHead(nn.Module):
    """Regression head for HD progression forecasting.

    Predicts 12-month and 24-month changes in UHDRS motor score
    from the fused 256-dim embedding (PRD Section 4.2.6).

    The output values represent predicted absolute changes (deltas)
    in UHDRS Total Motor Score. These are converted to risk
    categories via configurable thresholds:
    - Low:    Δ < 3.0
    - Medium: 3.0 ≤ Δ < 8.0
    - High:   Δ ≥ 8.0

    Args:
        input_dim: Input feature dimension (default: 256).
        output_dim: Number of prediction targets (default: 2 —
            12-month and 24-month deltas).
        dropout: Dropout probability (default: 0.1).
        risk_thresholds: Dictionary with 'low_max' and 'medium_max'
            keys for risk categorisation. If None, uses defaults.

    Example:
        >>> head = ProgressionHead(input_dim=256)
        >>> embedding = torch.randn(4, 256)
        >>> deltas = head(embedding)  # [4, 2]
        >>> risks = head.predict_risk(deltas)
        >>> # ["low", "medium", "high", "low"]
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 2,
        dropout: float = 0.1,
        risk_thresholds: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        self.risk_thresholds = risk_thresholds or DEFAULT_RISK_THRESHOLDS.copy()

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, output_dim),
        )

        # Initialise weights
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        logger.info(
            "ProgressionHead initialised: %d → %d outputs",
            input_dim,
            output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict UHDRS motor score deltas.

        Args:
            x: Fused embedding ``[B, input_dim]``.

        Returns:
            Predicted deltas ``[B, output_dim]`` where
            column 0 = 12-month Δ, column 1 = 24-month Δ.
        """
        return self.head(x)

    def predict_risk(
        self,
        deltas: torch.Tensor,
    ) -> list[str]:
        """Convert predicted deltas to risk categories.

        Uses the 12-month delta (column 0) for risk assignment
        as the primary clinical outcome measure.

        Args:
            deltas: Predicted deltas ``[B, 2]`` from ``forward()``.

        Returns:
            List of risk category strings: "low", "medium", or "high".
        """
        # Use absolute value of 12-month delta for risk
        delta_12mo = deltas[:, 0].abs().detach().cpu().numpy()

        low_max = self.risk_thresholds["low_max"]
        medium_max = self.risk_thresholds["medium_max"]

        risks: list[str] = []
        for d in delta_12mo:
            if d < low_max:
                risks.append("low")
            elif d < medium_max:
                risks.append("medium")
            else:
                risks.append("high")

        return risks


class NeuroSenseModel(nn.Module):
    """Full NeuroSense multi-modal model for HD analysis.

    Composes all sub-modules into a single model:
    1. MRIEncoder → 256-dim imaging embedding
    2. ClinicalEncoder → 256-dim clinical embedding
    3. CrossModalFusion → 256-dim fused representation
    4. ClassificationHead → 3-class HD staging logits
    5. ProgressionHead → 2-value regression (12mo/24mo UHDRS Δ)

    The model supports several operational modes for ablation:
    - ``use_mri=True, use_clinical=True``: Full multi-modal (default)
    - ``use_mri=True, use_clinical=False``: MRI-only
    - ``use_mri=False, use_clinical=True``: Clinical-only
    - ``fusion_type="cross_attention"``: Cross-attention (default)
    - ``fusion_type="concatenation"``: Concatenation baseline

    Args:
        mri_encoder_kwargs: Keyword arguments for MRIEncoder.
        clinical_encoder_kwargs: Keyword arguments for ClinicalEncoder.
        fusion_kwargs: Keyword arguments for CrossModalFusion.
        embed_dim: Shared embedding dimension (default: 256).
        num_classes: Number of HD staging classes (default: 3).
        num_progression_outputs: Number of regression targets
            (default: 2 — 12mo and 24mo deltas).
        fusion_type: Fusion strategy — "cross_attention" or
            "concatenation" (default: "cross_attention").
        use_mri: Whether to use MRI modality (default: True).
        use_clinical: Whether to use clinical modality (default: True).
        risk_thresholds: Risk categorisation thresholds.

    Attributes:
        mri_encoder: 3D ResNet-50 MRI feature extractor.
        clinical_encoder: Bi-LSTM clinical sequence encoder.
        fusion: Cross-modal fusion module.
        classification_head: HD staging classifier.
        progression_head: UHDRS delta regressor.

    Example:
        >>> model = NeuroSenseModel()
        >>> mri = torch.randn(4, 1, 96, 96, 96)
        >>> clinical = torch.randn(4, 1, 5)
        >>> lengths = torch.ones(4, dtype=torch.long)
        >>> outputs = model(mri, clinical, lengths)
        >>> print(outputs["logits"].shape)  # [4, 3]
        >>> print(outputs["deltas"].shape)  # [4, 2]
    """

    def __init__(
        self,
        mri_encoder_kwargs: dict[str, Any] | None = None,
        clinical_encoder_kwargs: dict[str, Any] | None = None,
        fusion_kwargs: dict[str, Any] | None = None,
        embed_dim: int = 256,
        num_classes: int = 3,
        num_progression_outputs: int = 2,
        fusion_type: str = "cross_attention",
        use_mri: bool = True,
        use_clinical: bool = True,
        risk_thresholds: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        if not use_mri and not use_clinical:
            raise ValueError(
                "At least one modality must be enabled "
                "(use_mri or use_clinical)"
            )

        self.embed_dim = embed_dim
        self.use_mri = use_mri
        self.use_clinical = use_clinical
        self.fusion_type = fusion_type

        # ─── MRI Encoder (PRD 4.2.2) ───
        if use_mri:
            mri_kwargs = {
                "embedding_dim": embed_dim,
                "backbone_out_dim": 2048,
                **(mri_encoder_kwargs or {}),
            }
            self.mri_encoder = MRIEncoder(**mri_kwargs)
        else:
            self.mri_encoder = None

        # ─── Clinical Encoder (PRD 4.2.3) ───
        if use_clinical:
            clin_kwargs = {
                "embedding_dim": embed_dim,
                "input_features": 5,
                "hidden_size": 256,
                "num_layers": 2,
                "dropout": 0.3,
                **(clinical_encoder_kwargs or {}),
            }
            self.clinical_encoder = ClinicalEncoder(**clin_kwargs)
        else:
            self.clinical_encoder = None

        # ─── Fusion (PRD 4.2.4) ───
        if use_mri and use_clinical:
            fusion_kwargs_merged = {
                "embed_dim": embed_dim,
                "num_heads": 8,
                "ffn_hidden_dim": 1024,
                **(fusion_kwargs or {}),
            }
            if fusion_type == "cross_attention":
                self.fusion = CrossModalFusion(**fusion_kwargs_merged)
            elif fusion_type == "concatenation":
                self.fusion = ConcatenationFusion(embed_dim=embed_dim)
            else:
                raise ValueError(
                    f"Unknown fusion_type: {fusion_type}. "
                    "Use 'cross_attention' or 'concatenation'."
                )
        else:
            self.fusion = None

        # ─── Classification Head (PRD 4.2.5) ───
        self.classification_head = ClassificationHead(
            input_dim=embed_dim,
            num_classes=num_classes,
        )

        # ─── Progression Head (PRD 4.2.6) ───
        self.progression_head = ProgressionHead(
            input_dim=embed_dim,
            output_dim=num_progression_outputs,
            risk_thresholds=risk_thresholds,
        )

        # Log model summary
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            "NeuroSenseModel initialised:\n"
            "  MRI: %s | Clinical: %s | Fusion: %s\n"
            "  Classes: %d | Progression outputs: %d\n"
            "  Total params: %dM (%dM trainable)",
            "ON" if use_mri else "OFF",
            "ON" if use_clinical else "OFF",
            fusion_type if (use_mri and use_clinical) else "N/A",
            num_classes,
            num_progression_outputs,
            total // 1_000_000,
            trainable // 1_000_000,
        )

    def forward(
        self,
        mri: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
        clinical_lengths: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Full forward pass through all modules.

        Args:
            mri: MRI volume ``[B, 1, 96, 96, 96]``. Required if
                ``use_mri=True``.
            clinical: Clinical features ``[B, T, 5]`` or ``[B, 5]``.
                Required if ``use_clinical=True``.
            clinical_lengths: Visit sequence lengths ``[B]``.
                Required for multi-visit clinical input.

        Returns:
            Dictionary containing:
            - logits: Classification logits ``[B, num_classes]``
            - probs: Classification probabilities ``[B, num_classes]``
            - stage_pred: Predicted stage index ``[B]``
            - deltas: Progression deltas ``[B, 2]``
            - risk: Risk categories (list of strings, inference only)
            - embeddings: Dict of intermediate embeddings
              - img_emb: MRI embedding ``[B, 256]`` (if use_mri)
              - clin_emb: Clinical embedding ``[B, 256]`` (if use_clinical)
              - fused_emb: Fused embedding ``[B, 256]``

        Raises:
            ValueError: If required modality inputs are missing.
        """
        embeddings: dict[str, torch.Tensor] = {}

        # ─── MRI Encoding ───
        img_emb = None
        if self.use_mri:
            if mri is None:
                raise ValueError(
                    "MRI input required when use_mri=True"
                )
            img_emb = self.mri_encoder(mri)
            embeddings["img_emb"] = img_emb

        # ─── Clinical Encoding ───
        clin_emb = None
        if self.use_clinical:
            if clinical is None:
                raise ValueError(
                    "Clinical input required when use_clinical=True"
                )
            # Handle [B, 5] input (single visit)
            if clinical.ndim == 2:
                clinical = clinical.unsqueeze(1)

            if clinical_lengths is None:
                clinical_lengths = torch.ones(
                    clinical.size(0),
                    dtype=torch.long,
                    device=clinical.device,
                )

            clin_emb = self.clinical_encoder(clinical, clinical_lengths)
            embeddings["clin_emb"] = clin_emb

        # ─── Fusion ───
        if self.use_mri and self.use_clinical:
            fused_emb = self.fusion(img_emb, clin_emb)
        elif self.use_mri:
            fused_emb = img_emb
        else:
            fused_emb = clin_emb

        embeddings["fused_emb"] = fused_emb

        # ─── Task Heads ───
        logits = self.classification_head(fused_emb)
        deltas = self.progression_head(fused_emb)

        probs = torch.softmax(logits, dim=-1)
        stage_pred = logits.argmax(dim=-1)

        # Risk categories (detach for inference use)
        risk = self.progression_head.predict_risk(deltas)

        return {
            "logits": logits,
            "probs": probs,
            "stage_pred": stage_pred,
            "deltas": deltas,
            "risk": risk,
            "embeddings": embeddings,
        }

    def get_attention_weights(
        self,
        mri: torch.Tensor,
        clinical: torch.Tensor,
        clinical_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Extract cross-attention weights for explainability.

        Args:
            mri: MRI volume ``[B, 1, 96, 96, 96]``.
            clinical: Clinical features ``[B, T, 5]``.
            clinical_lengths: Visit lengths ``[B]``.

        Returns:
            Attention weights ``[B, num_heads, 1, 1]`` or None
            if fusion doesn't support attention extraction.
        """
        if not (self.use_mri and self.use_clinical):
            return None

        if not hasattr(self.fusion, "get_attention_weights"):
            return None

        img_emb = self.mri_encoder(mri)

        if clinical.ndim == 2:
            clinical = clinical.unsqueeze(1)
        if clinical_lengths is None:
            clinical_lengths = torch.ones(
                clinical.size(0),
                dtype=torch.long,
                device=clinical.device,
            )

        clin_emb = self.clinical_encoder(clinical, clinical_lengths)

        return self.fusion.get_attention_weights(img_emb, clin_emb)

    def freeze_encoders(self) -> None:
        """Freeze encoder weights (for fine-tuning heads only).

        Useful when pre-trained encoders are loaded and only the
        fusion and task heads need training.
        """
        if self.mri_encoder is not None:
            for param in self.mri_encoder.parameters():
                param.requires_grad = False
            logger.info("MRI encoder frozen")

        if self.clinical_encoder is not None:
            for param in self.clinical_encoder.parameters():
                param.requires_grad = False
            logger.info("Clinical encoder frozen")

    def unfreeze_encoders(self) -> None:
        """Unfreeze all encoder weights for end-to-end training."""
        if self.mri_encoder is not None:
            for param in self.mri_encoder.parameters():
                param.requires_grad = True
            logger.info("MRI encoder unfrozen")

        if self.clinical_encoder is not None:
            for param in self.clinical_encoder.parameters():
                param.requires_grad = True
            logger.info("Clinical encoder unfrozen")
