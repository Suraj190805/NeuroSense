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
from neurosense.data.harmonizer import (
    MRIHarmonizer,
    harmonize_2d_slice,
    harmonize_3d_volume,
)
from neurosense.data.hd_image_dataset import (
    HDImageDataset,
    get_2d_train_transforms,
    get_2d_val_transforms,
)

__all__ = [
    "HuntingtonDataset",
    "HDImageDataset",
    "ClinicalNormalizer",
    "get_train_transforms",
    "get_val_transforms",
    "get_2d_train_transforms",
    "get_2d_val_transforms",
    "CLINICAL_FEATURE_NAMES",
    "STAGE_NAMES",
    "STAGE_LABELS",
    "MRIHarmonizer",
    "harmonize_2d_slice",
    "harmonize_3d_volume",
]
