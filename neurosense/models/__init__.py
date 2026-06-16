"""NeuroSense models module.

Neural network architectures for multi-modal Huntington's Disease
analysis: 3D ResNet MRI encoder, Bi-LSTM clinical encoder,
cross-modal attention fusion, and classification/progression heads.

Public API:
    MRIEncoder: 3D ResNet-50 backbone with projection head
    ClinicalEncoder: Bi-LSTM encoder for clinical visit sequences
    CrossModalFusion: Multi-head cross-attention fusion block
    ConcatenationFusion: Concat baseline for ablation
    ClassificationHead: Linear head for HD staging (3 classes)
    ProgressionHead: Regression head for UHDRS delta prediction
    NeuroSenseModel: Full composite multi-modal model
    CosineWarmupScheduler: LR schedule with warmup + cosine decay
    EarlyStopping: Patience-based training termination
    train_mri_encoder: Full Phase 2 training pipeline
    load_mri_encoder: Load trained encoder from checkpoint
"""

from neurosense.models.classifier import (
    NeuroSenseModel,
    ProgressionHead,
)
from neurosense.models.clinical_encoder import (
    ClinicalEncoder,
    ClinicalEncoderWithAttention,
    TemporalAttentionPooling,
)
from neurosense.models.fusion import (
    ConcatenationFusion,
    CrossModalFusion,
)
from neurosense.models.mri_encoder import (
    ClassificationHead,
    CosineWarmupScheduler,
    EarlyStopping,
    MRIEncoder,
    load_mri_encoder,
    train_mri_encoder,
)

__all__ = [
    # Encoders
    "MRIEncoder",
    "ClinicalEncoder",
    "ClinicalEncoderWithAttention",
    "TemporalAttentionPooling",
    # Fusion
    "CrossModalFusion",
    "ConcatenationFusion",
    # Heads
    "ClassificationHead",
    "ProgressionHead",
    # Composite model
    "NeuroSenseModel",
    # Training utilities
    "CosineWarmupScheduler",
    "EarlyStopping",
    "train_mri_encoder",
    "load_mri_encoder",
]
