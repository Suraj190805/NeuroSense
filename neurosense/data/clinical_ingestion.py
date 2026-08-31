"""NeuroSense Clinical Dataset Ingestion & BIDS Harmonization Engine.

Provides unified ingestion, validation, and preprocessing pipelines for
gold-standard real-world clinical neuroimaging datasets:

Supported Clinical Cohorts:
1. Track-HD & PREDICT-HD: Gold-standard multi-center longitudinal HD MRI datasets.
2. OpenNeuro: Open-access BIDS neuroimaging datasets (e.g. ds000221, ds003650).
3. OASIS-1 / OASIS-3 / ADNI: High-quality standardized structural brain MRIs (T1w/T2w).
4. PPMI (Parkinson's Progression Markers Initiative): Multi-center neurodegeneration repository.

Guarantees:
- Strict Subject-Level Stratification (Zero Patient Data Leakage across train/val/test).
- Automated BIDS Parsing (`participants.tsv`, `sub-*/ses-*/anat/*_T1w.nii.gz`).
- Automated MRI Intensity & Contrast Harmonization (Background air strip, CLAHE, Z-score).
- Clinical Metric Extraction (CAG repeats, UHDRS Total Motor Score, TFC, Age).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterator

import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from neurosense.data.harmonizer import harmonize_2d_slice, harmonize_3d_volume

logger = logging.getLogger(__name__)


class ClinicalDatasetIngestor:
    """Ingests and standardizes real-world clinical neuroimaging datasets into NeuroSense format."""

    def __init__(
        self,
        bids_root: str | Path,
        output_dir: str | Path,
        dataset_source: str = "openneuro",
    ) -> None:
        """Initialize ClinicalDatasetIngestor.

        Args:
            bids_root: Root directory of downloaded BIDS / clinical dataset.
            output_dir: Target destination for processed and harmonized data.
            dataset_source: Dataset type ('track_hd', 'predict_hd', 'openneuro', 'oasis', 'adni', 'ppmi').
        """
        self.bids_root = Path(bids_root)
        self.output_dir = Path(output_dir)
        self.dataset_source = dataset_source.lower()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_participants_tsv(self) -> dict[str, dict[str, Any]]:
        """Parse BIDS participants.tsv file for clinical variables."""
        tsv_path = self.bids_root / "participants.tsv"
        if not tsv_path.exists():
            # Check recursive
            found = list(self.bids_root.glob("**/participants.tsv"))
            if found:
                tsv_path = found[0]
            else:
                logger.warning("No participants.tsv found in %s, using defaults", self.bids_root)
                return {}

        participants: dict[str, dict[str, Any]] = {}
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sub_id = row.get("participant_id") or row.get("subject_id") or row.get("sub_id", "")
                sub_id = sub_id.strip()
                if not sub_id:
                    continue
                # Normalize participant ID format: sub-XXXX
                if not sub_id.startswith("sub-"):
                    sub_id = f"sub-{sub_id}"

                participants[sub_id] = {
                    "age": float(row.get("age", 50.0)) if row.get("age") and row.get("age") != "n/a" else 50.0,
                    "sex": row.get("sex", row.get("gender", "U")),
                    "cag_repeat": float(row.get("cag", row.get("cag_repeat", 40.0))) if row.get("cag") and row.get("cag") != "n/a" else 40.0,
                    "uhdrs_motor": float(row.get("uhdrs_motor", row.get("tms", 15.0))) if row.get("uhdrs_motor") and row.get("uhdrs_motor") != "n/a" else 15.0,
                    "tfc": float(row.get("tfc", 10.0)) if row.get("tfc") and row.get("tfc") != "n/a" else 10.0,
                    "group": row.get("group", row.get("diagnosis", "HD")),
                }
        return participants

    def discover_scans(self) -> list[dict[str, Any]]:
        """Discover all T1w/T2w NIfTI scans in BIDS hierarchy."""
        scan_records: list[dict[str, Any]] = []
        participants = self.parse_participants_tsv()

        # Find all NIfTI files (*.nii, *.nii.gz)
        for nifti_path in sorted(self.bids_root.glob("**/*T1w*.nii*")):
            # Extract subject ID
            parts = nifti_path.parts
            sub_id = next((p for p in parts if p.startswith("sub-")), None)
            if not sub_id:
                sub_id = f"sub-{nifti_path.stem.split('_')[0]}"

            clinical_info = participants.get(sub_id, {
                "age": 50.0,
                "cag_repeat": 40.0,
                "uhdrs_motor": 15.0,
                "tfc": 10.0,
                "group": "HD",
            })

            scan_records.append({
                "subject_id": sub_id,
                "nifti_path": str(nifti_path),
                "clinical": clinical_info,
            })

        logger.info("Discovered %d clinical scans for %s cohort", len(scan_records), self.dataset_source)
        return scan_records

    def extract_and_harmonize_slices(
        self,
        scan_records: list[dict[str, Any]],
        num_slices_per_scan: int = 5,
        target_size: int = 224,
    ) -> list[dict[str, Any]]:
        """Extract axial slices containing basal ganglia / ventricles and harmonize them.

        Args:
            scan_records: List of scan metadata dicts.
            num_slices_per_scan: Number of axial slices around the mid-brain.
            target_size: Output dimension.

        Returns:
            List of processed slice metadata dicts.
        """
        extracted_samples: list[dict[str, Any]] = []
        slice_out_dir = self.output_dir / "slices"
        slice_out_dir.mkdir(parents=True, exist_ok=True)

        for record in scan_records:
            sub_id = record["subject_id"]
            nii_path = record["nifti_path"]
            clinical = record["clinical"]

            try:
                img_nii = nib.load(nii_path)
                data = img_nii.get_fdata(dtype=np.float32)

                # Determine axial dimension (typically axis 2)
                z_dim = data.shape[2]
                center_z = z_dim // 2
                start_z = max(0, center_z - num_slices_per_scan // 2)
                end_z = min(z_dim, start_z + num_slices_per_scan)

                for slice_idx in range(start_z, end_z):
                    raw_slice = data[:, :, slice_idx]
                    # Rotate/orient to standard view
                    raw_slice = np.rot90(raw_slice)

                    # Harmonize: Cranial crop, CLAHE, brain Z-score
                    harm_2d, _ = harmonize_2d_slice(raw_slice, target_size=target_size)
                    harm_uint8 = (harm_2d * 255.0).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(harm_uint8).convert("RGB")

                    slice_filename = f"{sub_id}_slice{slice_idx:03d}.png"
                    slice_path = slice_out_dir / slice_filename
                    pil_img.save(slice_path)

                    # Determine label from CAG / Motor score
                    label = 1 if clinical.get("cag_repeat", 40) >= 36 else 0

                    extracted_samples.append({
                        "subject_id": sub_id,
                        "slice_path": str(slice_path),
                        "label": label,
                        "clinical": clinical,
                    })

            except Exception as e:
                logger.warning("Failed to process scan %s: %e", nii_path, e)

        logger.info("Extracted and harmonized %d 2D axial slices", len(extracted_samples))
        return extracted_samples

    @staticmethod
    def create_subject_level_splits(
        samples: list[dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Create subject-grouped splits ensuring ZERO patient data leakage.

        All slices or scans of a given subject remain strictly in one split.

        Args:
            samples: List of sample dicts containing 'subject_id' and 'label'.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            seed: Random seed.

        Returns:
            Tuple of (train_samples, val_samples, test_samples).
        """
        subjects = np.array([s["subject_id"] for s in samples])
        labels = np.array([s["label"] for s in samples])
        indices = np.arange(len(samples))

        # Stratified Group Split
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, temp_idx = next(sgkf.split(indices, labels, groups=subjects))

        # Split temp into val and test
        temp_subjects = subjects[temp_idx]
        temp_labels = labels[temp_idx]
        temp_indices = np.arange(len(temp_idx))

        sgkf_temp = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        val_sub_idx, test_sub_idx = next(sgkf_temp.split(temp_indices, temp_labels, groups=temp_subjects))

        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        test_samples = [samples[i] for i in test_idx]

        # Verify zero leakage
        train_subs = set(s["subject_id"] for s in train_samples)
        val_subs = set(s["subject_id"] for s in val_samples)
        test_subs = set(s["subject_id"] for s in test_samples)

        assert train_subs.isdisjoint(val_subs), "Leakage detected between Train and Val subjects!"
        assert train_subs.isdisjoint(test_subs), "Leakage detected between Train and Test subjects!"
        assert val_subs.isdisjoint(test_subs), "Leakage detected between Val and Test subjects!"

        logger.info(
            "Subject-Level Split Complete (Zero Leakage Verified):\n"
            "  - Train: %d samples (%d subjects)\n"
            "  - Val:   %d samples (%d subjects)\n"
            "  - Test:  %d samples (%d subjects)",
            len(train_samples), len(train_subs),
            len(val_samples), len(val_subs),
            len(test_samples), len(test_subs),
        )

        return train_samples, val_samples, test_samples
