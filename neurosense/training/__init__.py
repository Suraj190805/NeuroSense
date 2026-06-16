"""NeuroSense training module.

Training loops, evaluation pipelines, ablation study runners,
and custom loss functions for multi-modal HD classification
and progression forecasting.

Public API:
    train_neurosense: End-to-end NeuroSenseModel training pipeline
    evaluate_model: Test-set evaluation with metrics + confusion matrix
    run_ablation_study: 5-condition ablation runner (PRD Section 11)
    WeightedCrossEntropyLoss: Class-imbalance-aware CE loss
    ProgressionHuberLoss: Robust regression loss for UHDRS deltas
    FocalLoss: Hard-example focal loss (ablation alternative)
    CombinedLoss: Multi-task classification + progression loss
"""

from neurosense.training.ablation import run_ablation_study
from neurosense.training.evaluate import evaluate_model
from neurosense.training.losses import (
    CombinedLoss,
    FocalLoss,
    ProgressionHuberLoss,
    WeightedCrossEntropyLoss,
)
from neurosense.training.train import train_neurosense

__all__ = [
    # Training pipeline
    "train_neurosense",
    # Evaluation
    "evaluate_model",
    # Ablation
    "run_ablation_study",
    # Loss functions
    "WeightedCrossEntropyLoss",
    "ProgressionHuberLoss",
    "FocalLoss",
    "CombinedLoss",
]
