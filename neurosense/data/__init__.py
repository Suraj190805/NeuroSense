"""NeuroSense data module.

Dataset classes, preprocessing pipelines, and BIDS-format loaders
for Huntington's Disease neuroimaging and clinical data.

Public API:
    HuntingtonDataset: PyTorch Dataset for HD BIDS data
    ClinicalNormalizer: Min-max normaliser for clinical features
    get_train_transforms: MONAI transforms with augmentation
    get_val_transforms: MONAI transforms without augmentation
    CLINICAL_FEATURE_NAMES: Ordered clinical feature list
    STAGE_NAMES: HD stage class names
"""

from neurosense.data.dataset import (
    STAGE_LABELS,
    STAGE_NAMES,
    HuntingtonDataset,
)
from neurosense.data.preprocessing import (
    CLINICAL_FEATURE_NAMES,
    ClinicalNormalizer,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    "HuntingtonDataset",
    "ClinicalNormalizer",
    "get_train_transforms",
    "get_val_transforms",
    "CLINICAL_FEATURE_NAMES",
    "STAGE_NAMES",
    "STAGE_LABELS",
]
