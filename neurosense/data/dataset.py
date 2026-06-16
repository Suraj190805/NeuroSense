"""NeuroSense Huntington's Disease dataset class.

Implements the MONAI-based PyTorch Dataset specified in PRD Phase 1:
- Loads BIDS-format NIfTI MRI volumes via MONAI LoadImage
- Returns {mri: Tensor[1,96,96,96], clinical: Tensor[5],
  label: int, subject_id: str}
- Clinical vector: [cag_repeat, uhdrs_motor, uhdrs_cognitive, tfc, age]
  all normalised to [0,1]
- Subject-level stratified split (70/15/15) ensuring no data leakage

Usage:
    from neurosense.data.dataset import HuntingtonDataset

    dataset = HuntingtonDataset(root_dir="data/processed")
    sample = dataset[0]
    # sample["mri"].shape == torch.Size([1, 96, 96, 96])
    # sample["clinical"].shape == torch.Size([5])
    # sample["label"] == 0, 1, or 2

    train_loader, val_loader, test_loader = HuntingtonDataset.split(
        root_dir="data/processed", seed=42, batch_size=4,
    )
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.transforms import Compose
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset

from neurosense.data.preprocessing import (
    CLINICAL_FEATURE_NAMES,
    ClinicalNormalizer,
    get_train_transforms,
    get_val_transforms,
)

logger = logging.getLogger(__name__)

# HD staging class labels (PRD Section 4.2.5)
STAGE_LABELS: dict[str, int] = {
    "pre-manifest": 0,
    "pre_manifest": 0,
    "premanifest": 0,
    "control": 0,
    "early": 1,
    "early_hd": 1,
    "advanced": 2,
    "advanced_hd": 2,
    "manifest": 2,
}

STAGE_NAMES: list[str] = ["pre-manifest", "early", "advanced"]

# HD staging thresholds (clinical consensus, stated in impl plan)
# Pre-manifest: CAG ≥ 36 AND UHDRS Motor ≤ 5 AND TFC ≥ 11
# Early HD: UHDRS Motor 6–30 AND TFC 7–10
# Advanced HD: UHDRS Motor > 30 OR TFC < 7
STAGING_THRESHOLDS: dict[str, dict[str, float]] = {
    "pre_manifest": {
        "cag_min": 36,
        "uhdrs_motor_max": 5,
        "tfc_min": 11,
    },
    "early": {
        "uhdrs_motor_min": 6,
        "uhdrs_motor_max": 30,
        "tfc_min": 7,
        "tfc_max": 10,
    },
    "advanced": {
        "uhdrs_motor_min": 31,
        "tfc_max": 6,
    },
}


def _assign_stage_label(
    cag_repeat: float,
    uhdrs_motor: float,
    tfc: float,
) -> int:
    """Assign HD stage label based on clinical thresholds.

    Uses clinically-established UHDRS thresholds:
    - Pre-manifest (0): CAG ≥ 36 AND motor ≤ 5 AND TFC ≥ 11
    - Early HD (1): motor 6–30 AND TFC 7–10
    - Advanced HD (2): motor > 30 OR TFC < 7

    Args:
        cag_repeat: CAG trinucleotide repeat count.
        uhdrs_motor: UHDRS Total Motor Score (0–124).
        tfc: Total Functional Capacity score (0–13).

    Returns:
        Integer label: 0 (pre-manifest), 1 (early), 2 (advanced).
    """
    if uhdrs_motor > 30 or tfc < 7:
        return 2  # Advanced HD
    elif uhdrs_motor >= 6 and tfc <= 10:
        return 1  # Early HD
    else:
        return 0  # Pre-manifest


def _parse_participants_tsv(
    tsv_path: Path,
) -> dict[str, dict[str, Any]]:
    """Parse a BIDS participants.tsv file.

    Args:
        tsv_path: Path to participants.tsv.

    Returns:
        Dictionary mapping subject IDs to their clinical data.
    """
    participants: dict[str, dict[str, Any]] = {}

    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            subject_id = row.get(
                "participant_id", row.get("subject_id", "")
            )
            if not subject_id:
                continue

            # Clean subject ID
            subject_id = subject_id.strip()
            if not subject_id.startswith("sub-"):
                subject_id = f"sub-{subject_id}"

            clinical: dict[str, Any] = {}

            # Map common TSV column names to our feature names
            column_mapping = {
                "cag_repeat": [
                    "cag_repeat", "cag", "CAG",
                    "cag_count", "CAG_repeat",
                ],
                "uhdrs_motor": [
                    "uhdrs_motor", "motor_score", "TMS",
                    "total_motor_score", "UHDRS_motor",
                ],
                "uhdrs_cognitive": [
                    "uhdrs_cognitive", "cognitive_score",
                    "cognitive", "UHDRS_cognitive", "SDMT",
                ],
                "tfc": [
                    "tfc", "tfc_score", "TFC",
                    "total_functional_capacity",
                ],
                "age": [
                    "age", "Age", "age_at_visit",
                    "age_at_assessment",
                ],
            }

            for feature, aliases in column_mapping.items():
                for alias in aliases:
                    if alias in row and row[alias]:
                        try:
                            clinical[feature] = float(row[alias])
                            break
                        except (ValueError, TypeError):
                            continue

            # Try to get label from TSV
            label_cols = [
                "group", "diagnosis", "stage",
                "label", "condition", "dx",
            ]
            for col in label_cols:
                if col in row and row[col]:
                    label_str = row[col].strip().lower()
                    if label_str in STAGE_LABELS:
                        clinical["label"] = STAGE_LABELS[label_str]
                        break

            participants[subject_id] = clinical

    logger.info(
        "Parsed %d participants from %s",
        len(participants),
        tsv_path.name,
    )
    return participants


def _parse_clinical_json(
    json_path: Path,
) -> dict[str, float | None]:
    """Parse a subject-level clinical JSON file.

    Args:
        json_path: Path to clinical.json file.

    Returns:
        Dictionary mapping feature names to values.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    clinical: dict[str, float | None] = {}
    for feature in CLINICAL_FEATURE_NAMES:
        val = data.get(feature)
        if val is not None:
            try:
                clinical[feature] = float(val)
            except (ValueError, TypeError):
                clinical[feature] = None
        else:
            clinical[feature] = None

    return clinical


class HuntingtonDataset(Dataset):
    """PyTorch Dataset for Huntington's Disease BIDS-format data.

    Loads structural MRI volumes and clinical features from a
    BIDS-format dataset directory. Each sample contains:
    - mri: Tensor[1, 96, 96, 96] — preprocessed T1-weighted MRI
    - clinical: Tensor[5] — normalised clinical vector
    - label: int — HD stage (0=pre-manifest, 1=early, 2=advanced)
    - subject_id: str — BIDS subject identifier

    The clinical vector contains [cag_repeat, uhdrs_motor,
    uhdrs_cognitive, tfc, age], each normalised to [0, 1].

    Attributes:
        root_dir: Root directory of the BIDS dataset.
        samples: List of sample metadata dicts.
        transform: MONAI transform pipeline for MRI.
        normalizer: ClinicalNormalizer for feature normalisation.

    Example:
        >>> dataset = HuntingtonDataset(
        ...     root_dir="data/processed",
        ...     transform=get_train_transforms(),
        ... )
        >>> sample = dataset[0]
        >>> print(sample["mri"].shape)   # [1, 96, 96, 96]
        >>> print(sample["label"])        # 0, 1, or 2
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Compose | None = None,
        normalizer: ClinicalNormalizer | None = None,
        clinical_data: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialise the HuntingtonDataset.

        Args:
            root_dir: Root directory of the BIDS-format dataset
                containing sub-XXX directories with anatomical
                NIfTI files.
            transform: MONAI transform pipeline. If None, uses
                default validation transforms (no augmentation).
            normalizer: Pre-fitted ClinicalNormalizer. If None,
                a new one is created and fitted on the dataset.
            clinical_data: Pre-loaded clinical data dict mapping
                subject IDs to feature dicts. If None, will be
                loaded from participants.tsv and/or per-subject
                clinical files.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or get_val_transforms()
        self.samples: list[dict[str, Any]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root directory not found: {self.root_dir}"
            )

        # ─── Load clinical data ───
        if clinical_data is not None:
            self._clinical_data = clinical_data
        else:
            self._clinical_data = self._load_clinical_data()

        # ─── Discover MRI files and build sample list ───
        self._discover_samples()

        # ─── Fit clinical normalizer ───
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = ClinicalNormalizer()
            self._fit_normalizer()

        logger.info(
            "HuntingtonDataset initialised: %d samples from %s",
            len(self.samples),
            self.root_dir,
        )

        # Log class distribution
        label_counts = self._get_label_distribution()
        for stage_name, count in zip(STAGE_NAMES, label_counts):
            logger.info(
                "  %s: %d samples (%.1f%%)",
                stage_name,
                count,
                100.0 * count / max(len(self.samples), 1),
            )

    def _load_clinical_data(self) -> dict[str, dict[str, Any]]:
        """Load clinical data from BIDS dataset files.

        Searches for:
        1. participants.tsv in root directory
        2. Per-subject clinical.tsv or clinical.json files

        Returns:
            Dictionary mapping subject IDs to clinical data.
        """
        clinical_data: dict[str, dict[str, Any]] = {}

        # Try participants.tsv first
        participants_tsv = self.root_dir / "participants.tsv"
        if participants_tsv.exists():
            clinical_data = _parse_participants_tsv(participants_tsv)

        # Also check for per-subject clinical files
        for sub_dir in sorted(self.root_dir.glob("sub-*")):
            if not sub_dir.is_dir():
                continue
            subject_id = sub_dir.name

            # Check for phenotype/clinical files
            for clinical_file in [
                sub_dir / "phenotype" / "clinical.json",
                sub_dir / "clinical.json",
                sub_dir / "phenotype" / "clinical.tsv",
            ]:
                if clinical_file.exists():
                    if clinical_file.suffix == ".json":
                        subj_clinical = _parse_clinical_json(
                            clinical_file
                        )
                        if subject_id in clinical_data:
                            clinical_data[subject_id].update(
                                {
                                    k: v
                                    for k, v in subj_clinical.items()
                                    if v is not None
                                }
                            )
                        else:
                            clinical_data[subject_id] = subj_clinical
                    break

        if not clinical_data:
            logger.warning(
                "No clinical data found in %s. Generating "
                "synthetic clinical data for development.",
                self.root_dir,
            )
            clinical_data = self._generate_synthetic_clinical()

        return clinical_data

    def _generate_synthetic_clinical(
        self,
    ) -> dict[str, dict[str, Any]]:
        """Generate realistic synthetic clinical data for development.

        Used when real clinical data is not available (e.g., during
        development with OpenNeuro dataset). Generates clinically
        plausible values based on HD literature.

        Returns:
            Dictionary mapping subject IDs to synthetic clinical data.
        """
        rng = np.random.RandomState(42)
        clinical_data: dict[str, dict[str, Any]] = {}

        subject_dirs = sorted(self.root_dir.glob("sub-*"))
        n_subjects = len(subject_dirs)

        if n_subjects == 0:
            return clinical_data

        # Distribute subjects across stages: ~50% pre, ~30% early, ~20% advanced
        n_pre = max(1, int(n_subjects * 0.50))
        n_early = max(1, int(n_subjects * 0.30))
        n_advanced = max(1, n_subjects - n_pre - n_early)

        stage_assignments = (
            [0] * n_pre + [1] * n_early + [2] * n_advanced
        )
        rng.shuffle(stage_assignments)

        for sub_dir, stage in zip(subject_dirs, stage_assignments):
            subject_id = sub_dir.name

            if stage == 0:  # Pre-manifest
                clinical_data[subject_id] = {
                    "cag_repeat": float(rng.randint(38, 45)),
                    "uhdrs_motor": float(rng.uniform(0, 5)),
                    "uhdrs_cognitive": float(rng.uniform(180, 250)),
                    "tfc": float(rng.randint(11, 14)),
                    "age": float(rng.randint(25, 55)),
                    "label": 0,
                }
            elif stage == 1:  # Early HD
                clinical_data[subject_id] = {
                    "cag_repeat": float(rng.randint(40, 55)),
                    "uhdrs_motor": float(rng.uniform(6, 30)),
                    "uhdrs_cognitive": float(rng.uniform(120, 180)),
                    "tfc": float(rng.randint(7, 11)),
                    "age": float(rng.randint(35, 60)),
                    "label": 1,
                }
            else:  # Advanced HD
                clinical_data[subject_id] = {
                    "cag_repeat": float(rng.randint(42, 65)),
                    "uhdrs_motor": float(rng.uniform(31, 80)),
                    "uhdrs_cognitive": float(rng.uniform(50, 120)),
                    "tfc": float(rng.randint(0, 7)),
                    "age": float(rng.randint(40, 70)),
                    "label": 2,
                }

        logger.info(
            "Generated synthetic clinical data for %d subjects "
            "(pre:%d, early:%d, advanced:%d)",
            n_subjects,
            n_pre,
            n_early,
            n_advanced,
        )
        return clinical_data

    def _discover_samples(self) -> None:
        """Discover MRI files and pair with clinical data.

        Scans the BIDS directory structure for T1-weighted NIfTI
        files and matches them with clinical data by subject ID.
        """
        self.samples = []

        for sub_dir in sorted(self.root_dir.glob("sub-*")):
            if not sub_dir.is_dir():
                continue

            subject_id = sub_dir.name

            # Find T1-weighted MRI files
            mri_paths: list[Path] = []

            # Standard BIDS: sub-XXX/anat/sub-XXX_T1w.nii.gz
            anat_dir = sub_dir / "anat"
            if anat_dir.exists():
                mri_paths.extend(anat_dir.glob("*T1w*.nii*"))
                # Fallback: any NIfTI in anat/
                if not mri_paths:
                    mri_paths.extend(anat_dir.glob("*.nii*"))

            # Non-standard: NIfTI directly in subject dir
            if not mri_paths:
                mri_paths.extend(sub_dir.glob("*.nii*"))

            if not mri_paths:
                logger.debug(
                    "No MRI files found for %s, skipping",
                    subject_id,
                )
                continue

            # Get clinical data for this subject
            clinical = self._clinical_data.get(subject_id, {})

            # Determine label
            label: int | None = clinical.get("label")
            if label is None:
                # Try to assign from clinical scores
                cag = clinical.get("cag_repeat")
                motor = clinical.get("uhdrs_motor")
                tfc = clinical.get("tfc")
                if all(v is not None for v in [cag, motor, tfc]):
                    label = _assign_stage_label(cag, motor, tfc)
                else:
                    label = 0  # Default to pre-manifest if unknown
                    logger.debug(
                        "Insufficient clinical data for staging "
                        "%s, defaulting to pre-manifest",
                        subject_id,
                    )

            # Use the first (or best) MRI file
            mri_path = mri_paths[0]

            sample = {
                "mri_path": str(mri_path),
                "subject_id": subject_id,
                "label": int(label),
                "clinical_raw": {
                    name: clinical.get(name)
                    for name in CLINICAL_FEATURE_NAMES
                },
            }
            self.samples.append(sample)

        if not self.samples:
            logger.warning(
                "No valid samples found in %s. Ensure the "
                "directory contains sub-XXX/anat/*.nii.gz files.",
                self.root_dir,
            )

    def _fit_normalizer(self) -> None:
        """Fit the clinical normalizer on all available clinical data."""
        clinical_arrays = []
        for sample in self.samples:
            raw = sample["clinical_raw"]
            values = [
                raw.get(name) if raw.get(name) is not None else np.nan
                for name in CLINICAL_FEATURE_NAMES
            ]
            clinical_arrays.append(values)

        if clinical_arrays:
            data = np.array(clinical_arrays, dtype=np.float64)
            self.normalizer.fit(data)
        else:
            logger.warning(
                "No clinical data available for normalizer fitting"
            )
            # Fit with dummy data using range midpoints
            dummy = np.array(
                [
                    [
                        (r["min"] + r["max"]) / 2
                        for r in self.normalizer.ranges.values()
                    ]
                ]
            )
            self.normalizer.fit(dummy)

    def _get_label_distribution(self) -> list[int]:
        """Get count of samples per class label.

        Returns:
            List of counts [n_pre, n_early, n_advanced].
        """
        counts = [0, 0, 0]
        for sample in self.samples:
            label = sample["label"]
            if 0 <= label < 3:
                counts[label] += 1
        return counts

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single sample by index.

        Args:
            index: Sample index.

        Returns:
            Dictionary containing:
            - mri: Tensor of shape [1, 96, 96, 96]
            - clinical: Tensor of shape [5]
            - label: int (0, 1, or 2)
            - subject_id: str

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Index {index} out of range for dataset "
                f"of size {len(self.samples)}"
            )

        sample = self.samples[index]

        # ─── Load and transform MRI ───
        data_dict = {"mri": sample["mri_path"]}
        try:
            transformed = self.transform(data_dict)
            mri_tensor = transformed["mri"]
        except Exception as e:
            logger.error(
                "Failed to load MRI for %s: %s",
                sample["subject_id"],
                str(e),
            )
            # Return zero tensor as fallback
            mri_tensor = torch.zeros(
                1, 96, 96, 96, dtype=torch.float32
            )

        # Ensure correct shape [1, 96, 96, 96]
        if isinstance(mri_tensor, torch.Tensor):
            if mri_tensor.ndim == 3:
                mri_tensor = mri_tensor.unsqueeze(0)
        else:
            mri_tensor = torch.as_tensor(
                mri_tensor, dtype=torch.float32
            )
            if mri_tensor.ndim == 3:
                mri_tensor = mri_tensor.unsqueeze(0)

        # ─── Normalise clinical features ───
        clinical_normalised = self.normalizer.transform(
            sample["clinical_raw"]
        )
        clinical_tensor = torch.as_tensor(
            clinical_normalised, dtype=torch.float32
        )

        return {
            "mri": mri_tensor,
            "clinical": clinical_tensor,
            "label": sample["label"],
            "subject_id": sample["subject_id"],
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for loss function.

        Weights are inversely proportional to class frequency,
        as specified in PRD Section 4.2.5 for handling class
        imbalance (pre-manifest >> advanced).

        Returns:
            Tensor of shape [3] with per-class weights.
        """
        counts = self._get_label_distribution()
        total = sum(counts)

        if total == 0:
            return torch.ones(3, dtype=torch.float32)

        weights = []
        for count in counts:
            if count > 0:
                weights.append(total / (3.0 * count))
            else:
                weights.append(1.0)

        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        # Normalise so weights sum to num_classes
        weight_tensor = weight_tensor * 3.0 / weight_tensor.sum()

        logger.info(
            "Class weights: pre=%.3f, early=%.3f, advanced=%.3f",
            weight_tensor[0].item(),
            weight_tensor[1].item(),
            weight_tensor[2].item(),
        )
        return weight_tensor

    @staticmethod
    def split(
        root_dir: str | Path,
        seed: int = 42,
        batch_size: int = 4,
        num_workers: int = 4,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        pin_memory: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test DataLoaders with subject-level stratification.

        Splits the dataset ensuring no subject appears in multiple
        splits (PRD Section 3.3). Uses stratified splitting to preserve
        class distribution across splits.

        Args:
            root_dir: Root directory of the BIDS dataset.
            seed: Random seed for reproducibility.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of data loading workers.
            train_ratio: Fraction of data for training (default 0.70).
            val_ratio: Fraction of data for validation (default 0.15).
            pin_memory: Whether to pin memory for GPU transfer.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).

        Example:
            >>> train_loader, val_loader, test_loader = \\
            ...     HuntingtonDataset.split("data/processed", seed=42)
            >>> for batch in train_loader:
            ...     mri = batch["mri"]  # [B, 1, 96, 96, 96]
            ...     labels = batch["label"]  # [B]
        """
        root_dir = Path(root_dir)

        # Create full dataset with validation transforms initially
        full_dataset = HuntingtonDataset(
            root_dir=root_dir,
            transform=get_val_transforms(),
        )

        n_samples = len(full_dataset)
        if n_samples == 0:
            raise ValueError(
                f"No samples found in {root_dir}. "
                "Ensure the directory contains BIDS-format data."
            )

        # Extract subject IDs and labels for stratification
        subject_ids = [s["subject_id"] for s in full_dataset.samples]
        labels = [s["label"] for s in full_dataset.samples]

        # ─── Subject-level stratified split ───
        # Group by subject to prevent leakage
        unique_subjects = list(set(subject_ids))
        subject_labels = {}
        for sid, label in zip(subject_ids, labels):
            subject_labels[sid] = label

        # Create arrays for sklearn splitting
        subject_arr = np.array(unique_subjects)
        label_arr = np.array(
            [subject_labels[s] for s in unique_subjects]
        )

        rng = np.random.RandomState(seed)

        # First split: separate test set
        n_test = max(1, int(len(unique_subjects) * (1 - train_ratio - val_ratio)))
        n_val = max(1, int(len(unique_subjects) * val_ratio))

        # Stratified shuffle split
        indices = np.arange(len(unique_subjects))
        rng.shuffle(indices)

        # Sort indices by label to enable stratification
        sorted_by_label: dict[int, list[int]] = {}
        for idx in indices:
            lbl = label_arr[idx]
            if lbl not in sorted_by_label:
                sorted_by_label[lbl] = []
            sorted_by_label[lbl].append(idx)

        train_subject_indices: list[int] = []
        val_subject_indices: list[int] = []
        test_subject_indices: list[int] = []

        for lbl, lbl_indices in sorted_by_label.items():
            n_lbl = len(lbl_indices)
            n_lbl_test = max(1, round(n_lbl * (1 - train_ratio - val_ratio)))
            n_lbl_val = max(1, round(n_lbl * val_ratio))
            n_lbl_train = n_lbl - n_lbl_test - n_lbl_val

            if n_lbl_train < 1:
                n_lbl_train = 1
                n_lbl_val = max(0, n_lbl - n_lbl_train - n_lbl_test)

            test_subject_indices.extend(lbl_indices[:n_lbl_test])
            val_subject_indices.extend(
                lbl_indices[n_lbl_test:n_lbl_test + n_lbl_val]
            )
            train_subject_indices.extend(
                lbl_indices[n_lbl_test + n_lbl_val:]
            )

        # Map subject indices back to sample indices
        train_subjects = set(
            subject_arr[i] for i in train_subject_indices
        )
        val_subjects = set(
            subject_arr[i] for i in val_subject_indices
        )
        test_subjects = set(
            subject_arr[i] for i in test_subject_indices
        )

        train_indices = [
            i for i, sid in enumerate(subject_ids)
            if sid in train_subjects
        ]
        val_indices = [
            i for i, sid in enumerate(subject_ids)
            if sid in val_subjects
        ]
        test_indices = [
            i for i, sid in enumerate(subject_ids)
            if sid in test_subjects
        ]

        logger.info(
            "Dataset split (seed=%d): train=%d, val=%d, test=%d",
            seed,
            len(train_indices),
            len(val_indices),
            len(test_indices),
        )

        # Log per-split class distribution
        for split_name, split_indices in [
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        ]:
            split_labels = [labels[i] for i in split_indices]
            counts = [
                split_labels.count(c) for c in range(3)
            ]
            logger.info(
                "  %s: pre=%d, early=%d, advanced=%d",
                split_name,
                counts[0],
                counts[1],
                counts[2],
            )

        # ─── Create subset datasets with appropriate transforms ───
        # Training set gets augmentation transforms
        train_dataset = HuntingtonDataset(
            root_dir=root_dir,
            transform=get_train_transforms(),
            normalizer=full_dataset.normalizer,
            clinical_data=full_dataset._clinical_data,
        )
        # Val and test keep the validation (no-augmentation) transforms
        val_dataset = full_dataset
        test_dataset = HuntingtonDataset(
            root_dir=root_dir,
            transform=get_val_transforms(),
            normalizer=full_dataset.normalizer,
            clinical_data=full_dataset._clinical_data,
        )

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        test_subset = Subset(test_dataset, test_indices)

        # ─── Create DataLoaders ───
        def _collate_fn(
            batch: list[dict[str, Any]],
        ) -> dict[str, Any]:
            """Custom collate function for HD dataset batches.

            Args:
                batch: List of sample dicts from __getitem__.

            Returns:
                Batched dictionary with stacked tensors.
            """
            return {
                "mri": torch.stack([s["mri"] for s in batch]),
                "clinical": torch.stack(
                    [s["clinical"] for s in batch]
                ),
                "label": torch.tensor(
                    [s["label"] for s in batch],
                    dtype=torch.long,
                ),
                "subject_id": [s["subject_id"] for s in batch],
            }

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_fn,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_fn,
            drop_last=False,
        )

        return train_loader, val_loader, test_loader
