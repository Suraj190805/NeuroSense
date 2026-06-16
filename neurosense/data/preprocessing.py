"""NeuroSense MRI and tabular data preprocessing pipeline.

Implements the preprocessing module specified in PRD Section 4.2.1:
- MRI pipeline: z-score normalisation, resampling to 96x96x96,
  MONAI-based spatial transforms and augmentation.
- Tabular pipeline: min-max normalisation for continuous clinical
  features, median imputation for missing values.

Usage (CLI):
    python -m neurosense.data.preprocessing \\
        --input-dir data/raw \\
        --output-dir data/processed \\
        --spatial-size 96 96 96

Usage (Python):
    from neurosense.data.preprocessing import (
        get_train_transforms,
        get_val_transforms,
        ClinicalNormalizer,
    )
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from monai.transforms import (
    AddChanneld,
    Compose,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

logger = logging.getLogger(__name__)

# ─── Default clinical feature ranges (PRD Section 4.2.1) ───
DEFAULT_CLINICAL_RANGES: dict[str, dict[str, float]] = {
    "cag_repeat": {"min": 36.0, "max": 120.0},
    "uhdrs_motor": {"min": 0.0, "max": 124.0},
    "uhdrs_cognitive": {"min": 0.0, "max": 300.0},
    "tfc": {"min": 0.0, "max": 13.0},
    "age": {"min": 18.0, "max": 90.0},
}

# Ordered feature names matching the clinical vector layout
CLINICAL_FEATURE_NAMES: list[str] = [
    "cag_repeat",
    "uhdrs_motor",
    "uhdrs_cognitive",
    "tfc",
    "age",
]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load model configuration from YAML file.

    Args:
        config_path: Path to model_config.yaml. If None, uses the
            default config from the configs/ directory.

    Returns:
        Dictionary of configuration values.
    """
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent / "configs" / "model_config.yaml"
        )
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(
            "Config file not found at %s, using defaults", config_path
        )
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_train_transforms(
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Build MONAI transform composition for training data.

    Implements the MRI preprocessing pipeline from PRD Section 4.2.1
    with data augmentation from PRD Section 4.2.2:
    - Load NIfTI → add channel → resample to 1mm isotropic
    - Orient to RAS → resize to 96³ → z-score normalise
    - Augmentation: random flip, rotation (±15°), intensity shift (±0.1)

    Args:
        spatial_size: Target spatial dimensions (D, H, W).
            Default is (96, 96, 96) per PRD.
        pixdim: Target voxel spacing in mm.
            Default is (1.0, 1.0, 1.0) for 1mm isotropic.

    Returns:
        MONAI Compose transform pipeline for training data.
    """
    transforms = Compose(
        [
            # ─── Loading & basic preprocessing ───
            LoadImaged(
                keys=["mri"],
                ensure_channel_first=True,
                image_only=False,
            ),
            # Reorient to standard RAS orientation
            Orientationd(keys=["mri"], axcodes="RAS"),
            # Resample to 1mm isotropic voxel spacing
            Spacingd(
                keys=["mri"],
                pixdim=pixdim,
                mode="bilinear",
            ),
            # Resize to target spatial dimensions (96×96×96)
            Resized(
                keys=["mri"],
                spatial_size=spatial_size,
                mode="trilinear",
            ),
            # Ensure minimum spatial size via padding
            SpatialPadd(
                keys=["mri"],
                spatial_size=spatial_size,
            ),
            # Z-score normalisation per subject (PRD 4.2.1)
            NormalizeIntensityd(
                keys=["mri"],
                nonzero=True,
                channel_wise=True,
            ),
            # ─── Data augmentation (PRD 4.2.2) ───
            # Random flip on all axes with p=0.5
            RandFlipd(
                keys=["mri"],
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=["mri"],
                prob=0.5,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=["mri"],
                prob=0.5,
                spatial_axis=2,
            ),
            # Random affine rotation ±15 degrees (PRD 4.2.2)
            RandAffined(
                keys=["mri"],
                prob=0.5,
                rotate_range=(
                    np.radians(15),
                    np.radians(15),
                    np.radians(15),
                ),
                mode="bilinear",
                padding_mode="zeros",
            ),
            # Random intensity shift ±0.1 (PRD 4.2.2)
            RandShiftIntensityd(
                keys=["mri"],
                offsets=0.1,
                prob=0.5,
            ),
            # Random intensity scale for robustness
            RandScaleIntensityd(
                keys=["mri"],
                factors=0.1,
                prob=0.3,
            ),
            # Minor Gaussian noise for regularisation
            RandGaussianNoised(
                keys=["mri"],
                prob=0.2,
                mean=0.0,
                std=0.02,
            ),
            # ─── Convert to PyTorch tensors ───
            EnsureTyped(
                keys=["mri"],
                dtype=torch.float32,
            ),
        ]
    )
    logger.debug(
        "Built training transforms: spatial_size=%s, pixdim=%s",
        spatial_size,
        pixdim,
    )
    return transforms


def get_val_transforms(
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Build MONAI transform composition for validation/test data.

    Same preprocessing as training but WITHOUT augmentation, ensuring
    deterministic inference per PRD NFR-02.

    Args:
        spatial_size: Target spatial dimensions (D, H, W).
        pixdim: Target voxel spacing in mm.

    Returns:
        MONAI Compose transform pipeline for validation/test data.
    """
    transforms = Compose(
        [
            LoadImaged(
                keys=["mri"],
                ensure_channel_first=True,
                image_only=False,
            ),
            Orientationd(keys=["mri"], axcodes="RAS"),
            Spacingd(
                keys=["mri"],
                pixdim=pixdim,
                mode="bilinear",
            ),
            Resized(
                keys=["mri"],
                spatial_size=spatial_size,
                mode="trilinear",
            ),
            SpatialPadd(
                keys=["mri"],
                spatial_size=spatial_size,
            ),
            NormalizeIntensityd(
                keys=["mri"],
                nonzero=True,
                channel_wise=True,
            ),
            EnsureTyped(
                keys=["mri"],
                dtype=torch.float32,
            ),
        ]
    )
    logger.debug(
        "Built validation transforms: spatial_size=%s, pixdim=%s",
        spatial_size,
        pixdim,
    )
    return transforms


class ClinicalNormalizer:
    """Min-max normaliser for clinical feature vectors.

    Normalises each clinical feature to [0, 1] using the ranges
    defined in PRD Section 4.2.1. Handles missing values via
    median imputation (computed from training data).

    Attributes:
        ranges: Dictionary mapping feature names to min/max values.
        medians: Dictionary mapping feature names to median values
            for imputation (computed from fit data).
        feature_names: Ordered list of clinical feature names.

    Example:
        >>> normalizer = ClinicalNormalizer()
        >>> normalizer.fit(training_clinical_data)
        >>> normalized = normalizer.transform(raw_clinical_vector)
    """

    def __init__(
        self,
        ranges: dict[str, dict[str, float]] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialise the clinical normaliser.

        Args:
            ranges: Feature name to {min, max} mapping. Defaults to
                PRD-specified clinical ranges.
            feature_names: Ordered feature names. Defaults to
                [cag_repeat, uhdrs_motor, uhdrs_cognitive, tfc, age].
        """
        self.ranges = ranges or DEFAULT_CLINICAL_RANGES.copy()
        self.feature_names = feature_names or CLINICAL_FEATURE_NAMES.copy()
        self.medians: dict[str, float] = {}
        self._is_fitted: bool = False
        logger.debug(
            "ClinicalNormalizer initialised with %d features",
            len(self.feature_names),
        )

    def fit(self, data: np.ndarray | list[dict[str, float]]) -> None:
        """Compute median values from training data for imputation.

        Args:
            data: Training clinical data. Either a 2D numpy array of
                shape (n_subjects, n_features) matching feature_names
                order, or a list of dicts mapping feature names to
                values. NaN values are excluded from median computation.
        """
        if isinstance(data, list):
            # Convert list of dicts to array
            array = np.full(
                (len(data), len(self.feature_names)), np.nan
            )
            for i, record in enumerate(data):
                for j, name in enumerate(self.feature_names):
                    if name in record and record[name] is not None:
                        array[i, j] = float(record[name])
            data = array

        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got shape {data.shape}"
            )

        if data.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"got {data.shape[1]}"
            )

        for j, name in enumerate(self.feature_names):
            col = data[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                self.medians[name] = float(np.median(valid))
            else:
                # Fallback to midpoint of range
                r = self.ranges[name]
                self.medians[name] = (r["min"] + r["max"]) / 2.0
                logger.warning(
                    "No valid values for %s, using range midpoint %.1f",
                    name,
                    self.medians[name],
                )

        self._is_fitted = True
        logger.info(
            "ClinicalNormalizer fitted on %d samples, "
            "medians: %s",
            data.shape[0],
            {k: f"{v:.2f}" for k, v in self.medians.items()},
        )

    def transform(
        self,
        values: dict[str, float | None] | np.ndarray | list[float],
    ) -> np.ndarray:
        """Normalise a clinical feature vector to [0, 1].

        Missing values (None or NaN) are replaced with the median
        from the training set. Values are clipped to the valid range
        before normalisation.

        Args:
            values: Clinical feature values. Can be a dict mapping
                feature names to values, a numpy array, or a list
                in feature_names order.

        Returns:
            Normalised numpy array of shape (n_features,) with all
            values in [0, 1].

        Raises:
            RuntimeError: If transform is called before fit.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ClinicalNormalizer must be fitted before transform. "
                "Call .fit(training_data) first."
            )

        # Convert to ordered array
        if isinstance(values, dict):
            raw = np.array(
                [
                    float(values.get(name, np.nan))
                    if values.get(name) is not None
                    else np.nan
                    for name in self.feature_names
                ],
                dtype=np.float64,
            )
        elif isinstance(values, list):
            raw = np.array(values, dtype=np.float64)
        else:
            raw = np.asarray(values, dtype=np.float64).flatten()

        if len(raw) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"got {len(raw)}"
            )

        # Median imputation for missing values
        for i, name in enumerate(self.feature_names):
            if np.isnan(raw[i]):
                raw[i] = self.medians[name]
                logger.debug(
                    "Imputed %s with median %.2f",
                    name,
                    self.medians[name],
                )

        # Min-max normalisation with clipping
        normalised = np.zeros_like(raw)
        for i, name in enumerate(self.feature_names):
            r = self.ranges[name]
            clipped = np.clip(raw[i], r["min"], r["max"])
            denom = r["max"] - r["min"]
            if denom > 0:
                normalised[i] = (clipped - r["min"]) / denom
            else:
                normalised[i] = 0.0

        return normalised.astype(np.float32)

    def inverse_transform(
        self,
        normalised: np.ndarray,
    ) -> dict[str, float]:
        """Convert normalised values back to original scale.

        Args:
            normalised: Array of normalised values in [0, 1].

        Returns:
            Dictionary mapping feature names to original-scale values.
        """
        result = {}
        for i, name in enumerate(self.feature_names):
            r = self.ranges[name]
            result[name] = float(
                normalised[i] * (r["max"] - r["min"]) + r["min"]
            )
        return result

    def save(self, path: str | Path) -> None:
        """Save normaliser state to JSON file.

        Args:
            path: Output file path.
        """
        state = {
            "ranges": self.ranges,
            "feature_names": self.feature_names,
            "medians": self.medians,
            "is_fitted": self._is_fitted,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("ClinicalNormalizer saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ClinicalNormalizer":
        """Load normaliser state from JSON file.

        Args:
            path: Input file path.

        Returns:
            Restored ClinicalNormalizer instance.
        """
        with open(path, "r") as f:
            state = json.load(f)

        normalizer = cls(
            ranges=state["ranges"],
            feature_names=state["feature_names"],
        )
        normalizer.medians = state["medians"]
        normalizer._is_fitted = state["is_fitted"]
        logger.info("ClinicalNormalizer loaded from %s", path)
        return normalizer


def preprocess_single_mri(
    input_path: Path,
    output_path: Path,
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> bool:
    """Preprocess a single MRI NIfTI file.

    Applies the validation (no-augmentation) transform pipeline
    and saves the result as a new NIfTI file.

    Args:
        input_path: Path to input .nii or .nii.gz file.
        output_path: Path to save processed .nii.gz file.
        spatial_size: Target spatial dimensions.
        pixdim: Target voxel spacing in mm.

    Returns:
        True if preprocessing succeeded, False otherwise.
    """
    import nibabel as nib

    try:
        transforms = get_val_transforms(spatial_size, pixdim)
        data_dict = {"mri": str(input_path)}
        result = transforms(data_dict)

        # Extract tensor and save as NIfTI
        mri_tensor = result["mri"]
        if isinstance(mri_tensor, torch.Tensor):
            mri_array = mri_tensor.numpy()
        else:
            mri_array = np.asarray(mri_tensor)

        # Remove channel dimension for NIfTI: (1, D, H, W) → (D, H, W)
        if mri_array.ndim == 4 and mri_array.shape[0] == 1:
            mri_array = mri_array[0]

        # Create NIfTI with 1mm isotropic affine
        affine = np.diag([*pixdim, 1.0])
        img = nib.Nifti1Image(mri_array, affine=affine)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(output_path))
        logger.info("Preprocessed: %s → %s", input_path.name, output_path)
        return True

    except Exception as e:
        logger.error(
            "Failed to preprocess %s: %s", input_path.name, str(e)
        )
        return False


def preprocess_bids_dataset(
    input_dir: Path,
    output_dir: Path,
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_workers: int = 1,
) -> dict[str, int]:
    """Preprocess all MRI files in a BIDS-format dataset.

    Scans the input directory for subject folders containing
    anatomical T1-weighted NIfTI files and processes each one.

    Args:
        input_dir: Root directory of BIDS-format dataset.
        output_dir: Output directory for processed files.
        spatial_size: Target spatial dimensions.
        pixdim: Target voxel spacing in mm.
        num_workers: Number of parallel workers (currently sequential).

    Returns:
        Dictionary with counts: {processed, failed, skipped, total}.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all NIfTI files in BIDS structure
    # Pattern: sub-XXX/anat/sub-XXX_T1w.nii.gz
    nifti_files: list[Path] = []

    # Search for subject directories
    for sub_dir in sorted(input_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        anat_dir = sub_dir / "anat"
        if anat_dir.exists():
            for nifti in anat_dir.glob("*.nii*"):
                nifti_files.append(nifti)
        # Also check directly in subject dir
        for nifti in sub_dir.glob("*.nii*"):
            nifti_files.append(nifti)

    # Also search root for non-BIDS layouts
    if not nifti_files:
        nifti_files = list(input_dir.rglob("*.nii*"))

    logger.info(
        "Found %d NIfTI files in %s", len(nifti_files), input_dir
    )

    stats = {"processed": 0, "failed": 0, "skipped": 0, "total": len(nifti_files)}

    for nifti_path in nifti_files:
        # Determine output path preserving directory structure
        rel_path = nifti_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        # Ensure .nii.gz extension
        if out_path.suffix == ".nii":
            out_path = out_path.with_suffix(".nii.gz")

        # Skip if already processed
        if out_path.exists():
            logger.debug("Skipping (exists): %s", out_path)
            stats["skipped"] += 1
            continue

        success = preprocess_single_mri(
            nifti_path, out_path, spatial_size, pixdim
        )
        if success:
            stats["processed"] += 1
        else:
            stats["failed"] += 1

    logger.info(
        "Preprocessing complete: %d processed, %d failed, "
        "%d skipped, %d total",
        stats["processed"],
        stats["failed"],
        stats["skipped"],
        stats["total"],
    )
    return stats


def main() -> None:
    """CLI entry point for MRI preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="NeuroSense MRI Preprocessing Pipeline"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root directory of raw BIDS-format dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="Target spatial dimensions (D H W), default: 96 96 96",
    )
    parser.add_argument(
        "--voxel-spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Target voxel spacing in mm, default: 1.0 1.0 1.0",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="zscore",
        choices=["zscore", "minmax"],
        help="Normalisation method (default: zscore)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    spatial_size = tuple(args.spatial_size)
    pixdim = tuple(args.voxel_spacing)

    stats = preprocess_bids_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        spatial_size=spatial_size,
        pixdim=pixdim,
        num_workers=args.num_workers,
    )

    print(f"\n{'='*50}")
    print("Preprocessing Results:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed:    {stats['failed']}")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"  Total:     {stats['total']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
