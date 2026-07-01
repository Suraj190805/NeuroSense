#!/usr/bin/env python3
"""Convert Huntington's Disease CSV dataset to BIDS format with synthetic MRI.

Reads the CSV, samples a balanced subset, maps columns to NeuroSense
feature names, generates synthetic NIfTI MRI volumes, and creates the
BIDS directory structure expected by HuntingtonDataset.

Usage:
    python scripts/csv_to_bids.py \
        --csv /path/to/Huntington_Disease_Dataset.csv \
        --output-dir data/raw \
        --per-class 50

This will create:
    data/raw/
    ├── participants.tsv
    ├── sub-001/anat/sub-001_T1w.nii.gz
    ├── sub-002/anat/sub-002_T1w.nii.gz
    └── ...
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Stage mapping: CSV 4-class → NeuroSense 3-class ───
STAGE_MAP: dict[str, int] = {
    "Pre-Symptomatic": 0,   # pre-manifest
    "Early": 1,             # early HD
    "Middle": 2,            # advanced HD
    "Late": 2,              # advanced HD
}

STAGE_NAMES = {0: "pre-manifest", 1: "early", 2: "advanced"}

# ─── Motor symptoms: categorical → numeric UHDRS motor score ───
MOTOR_MAP: dict[str, float] = {
    "None": 2.0,
    "Mild": 15.0,
    "Moderate": 40.0,
    "Severe": 80.0,
}

# ─── Cognitive decline: categorical → numeric cognitive score ───
# Higher = better cognitive function (inverted severity)
COGNITIVE_MAP: dict[str, float] = {
    "None": 250.0,
    "Mild": 180.0,
    "Moderate": 130.0,
    "Severe": 60.0,
}


def read_csv(csv_path: Path) -> list[dict[str, str]]:
    """Read the CSV and return list of row dicts."""
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    logger.info("Read %d rows from %s", len(rows), csv_path.name)
    return rows


def sample_balanced(
    rows: list[dict[str, str]],
    per_class: int,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Sample a balanced subset with per_class samples from each stage."""
    rng = np.random.RandomState(seed)

    # Group by disease stage
    by_stage: dict[int, list[dict[str, str]]] = {0: [], 1: [], 2: []}
    for row in rows:
        stage_str = row.get("Disease_Stage", "").strip()
        label = STAGE_MAP.get(stage_str)
        if label is not None:
            by_stage[label].append(row)

    for label, name in STAGE_NAMES.items():
        logger.info(
            "  Stage %d (%s): %d available",
            label, name, len(by_stage[label]),
        )

    # Sample
    selected: list[dict[str, str]] = []
    for label in sorted(by_stage.keys()):
        pool = by_stage[label]
        n = min(per_class, len(pool))
        indices = rng.choice(len(pool), size=n, replace=False)
        for idx in indices:
            selected.append(pool[idx])
        logger.info(
            "  Sampled %d from stage %d (%s)",
            n, label, STAGE_NAMES[label],
        )

    rng.shuffle(selected)
    logger.info("Total selected: %d subjects", len(selected))
    return selected


def map_row_to_clinical(row: dict[str, str]) -> dict[str, float | str | int]:
    """Map a CSV row to NeuroSense clinical features."""
    stage_str = row.get("Disease_Stage", "").strip()
    label = STAGE_MAP.get(stage_str, 0)

    # CAG repeat — direct
    cag = float(row.get("HTT_CAG_Repeat_Length", 40))

    # Motor symptoms — categorical to numeric
    motor_str = row.get("Motor_Symptoms", "None").strip()
    uhdrs_motor = MOTOR_MAP.get(motor_str, 15.0)

    # Add some variance based on chorea score
    chorea = float(row.get("Chorea_Score", 5.0))
    uhdrs_motor = uhdrs_motor + (chorea - 5.0) * 2.0
    uhdrs_motor = max(0.0, min(124.0, uhdrs_motor))

    # Cognitive decline — categorical to numeric
    cog_str = row.get("Cognitive_Decline", "None").strip()
    uhdrs_cognitive = COGNITIVE_MAP.get(cog_str, 180.0)

    # Functional capacity — scale 0-100 to 0-13
    fc_raw = float(row.get("Functional_Capacity", 50))
    tfc = round(fc_raw * 13.0 / 100.0, 1)
    tfc = max(0.0, min(13.0, tfc))

    # Age — direct
    age = float(row.get("Age", 50))

    return {
        "cag_repeat": cag,
        "uhdrs_motor": round(uhdrs_motor, 1),
        "uhdrs_cognitive": round(uhdrs_cognitive, 1),
        "tfc": tfc,
        "age": age,
        "label": label,
        "label_name": STAGE_NAMES[label],
        "sex": row.get("Sex", "Unknown"),
        "family_history": row.get("Family_History", "Unknown"),
        "brain_volume_loss": float(row.get("Brain_Volume_Loss", 5.0)),
        "chorea_score": chorea,
        "original_stage": stage_str,
    }


def generate_synthetic_mri(
    label: int,
    brain_volume_loss: float,
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    seed: int | None = None,
) -> np.ndarray:
    """Generate a synthetic T1-like MRI volume shaped by disease stage.

    Creates a 3D volume with:
    - A brain-like ellipsoidal mask
    - Gray/white matter intensity patterns
    - Stage-dependent atrophy (volume loss in caudate/putamen regions)
    - Gaussian noise for realism

    Args:
        label: Disease stage (0=pre-manifest, 1=early, 2=advanced).
        brain_volume_loss: Brain volume loss percentage from CSV.
        spatial_size: Output volume dimensions.
        seed: Random seed for reproducibility.

    Returns:
        3D numpy array of shape spatial_size with float32 values.
    """
    rng = np.random.RandomState(seed)
    D, H, W = spatial_size
    volume = np.zeros((D, H, W), dtype=np.float32)

    # ─── Create brain-like ellipsoidal structure ───
    z, y, x = np.ogrid[
        -D // 2:D // 2,
        -H // 2:H // 2,
        -W // 2:W // 2,
    ]
    # Brain mask: ellipsoid
    brain_mask = (
        (z / (D * 0.42)) ** 2
        + (y / (H * 0.38)) ** 2
        + (x / (W * 0.40)) ** 2
    ) < 1.0

    # ─── Base tissue intensities ───
    # White matter core
    wm_mask = (
        (z / (D * 0.25)) ** 2
        + (y / (H * 0.22)) ** 2
        + (x / (W * 0.24)) ** 2
    ) < 1.0

    # Gray matter: brain minus white matter
    gm_mask = brain_mask & ~wm_mask

    volume[wm_mask] = rng.normal(0.75, 0.08, size=wm_mask.sum())
    volume[gm_mask] = rng.normal(0.55, 0.10, size=gm_mask.sum())

    # ─── Caudate nucleus regions (HD-affected) ───
    # Bilateral caudate heads — primary HD atrophy targets
    for side in [-1, 1]:
        caudate_center = (D // 2 + 5, H // 2 - 5, W // 2 + side * 12)
        caudate_mask = (
            (z - (caudate_center[0] - D // 2)) ** 2 / 64
            + (y - (caudate_center[1] - H // 2)) ** 2 / 36
            + (x - (caudate_center[2] - W // 2)) ** 2 / 25
        ) < 1.0

        # Atrophy increases with disease stage
        if label == 0:  # Pre-manifest: intact caudate
            volume[caudate_mask & brain_mask] = rng.normal(
                0.65, 0.05, size=(caudate_mask & brain_mask).sum()
            )
        elif label == 1:  # Early: mild atrophy
            atrophy_factor = 0.7 - brain_volume_loss * 0.03
            volume[caudate_mask & brain_mask] = rng.normal(
                0.50 * atrophy_factor, 0.08,
                size=(caudate_mask & brain_mask).sum(),
            )
        else:  # Advanced: significant atrophy
            atrophy_factor = 0.4 - brain_volume_loss * 0.04
            volume[caudate_mask & brain_mask] = rng.normal(
                0.30 * max(atrophy_factor, 0.1), 0.12,
                size=(caudate_mask & brain_mask).sum(),
            )

    # ─── Ventricular enlargement (compensatory) ───
    vent_size_factor = 1.0 + label * 0.3 + brain_volume_loss * 0.05
    vent_mask = (
        (z / (D * 0.08 * vent_size_factor)) ** 2
        + (y / (H * 0.06 * vent_size_factor)) ** 2
        + (x / (W * 0.04 * vent_size_factor)) ** 2
    ) < 1.0
    volume[vent_mask & brain_mask] = rng.normal(
        0.10, 0.03, size=(vent_mask & brain_mask).sum()
    )

    # ─── Global cortical thinning for advanced stages ───
    if label >= 1:
        # Reduce gray matter intensity proportional to stage
        reduction = 0.05 * label + brain_volume_loss * 0.01
        volume[gm_mask] *= (1.0 - reduction)

    # ─── Add realistic noise ───
    noise = rng.normal(0, 0.03, size=volume.shape).astype(np.float32)
    volume += noise

    # Background stays ~0
    volume[~brain_mask] = rng.normal(0.02, 0.01, size=(~brain_mask).sum())

    # Clip to valid range
    volume = np.clip(volume, 0.0, 1.0)

    return volume


def create_bids_dataset(
    selected_rows: list[dict[str, str]],
    output_dir: Path,
    spatial_size: tuple[int, int, int] = (96, 96, 96),
) -> None:
    """Create full BIDS dataset from selected CSV rows.

    Args:
        selected_rows: List of CSV row dicts.
        output_dir: Output BIDS directory.
        spatial_size: MRI volume dimensions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Write participants.tsv ───
    tsv_path = output_dir / "participants.tsv"
    tsv_columns = [
        "participant_id",
        "age",
        "sex",
        "family_history",
        "cag_repeat",
        "uhdrs_motor",
        "uhdrs_cognitive",
        "tfc",
        "chorea_score",
        "brain_volume_loss",
        "group",
    ]

    all_clinical: list[dict] = []
    for i, row in enumerate(selected_rows):
        clinical = map_row_to_clinical(row)
        subject_id = f"sub-{i + 1:03d}"
        clinical["participant_id"] = subject_id
        clinical["subject_index"] = i
        all_clinical.append(clinical)

    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=tsv_columns, delimiter="\t"
        )
        writer.writeheader()
        for clin in all_clinical:
            writer.writerow({
                "participant_id": clin["participant_id"],
                "age": clin["age"],
                "sex": clin["sex"],
                "family_history": clin["family_history"],
                "cag_repeat": clin["cag_repeat"],
                "uhdrs_motor": clin["uhdrs_motor"],
                "uhdrs_cognitive": clin["uhdrs_cognitive"],
                "tfc": clin["tfc"],
                "chorea_score": clin["chorea_score"],
                "brain_volume_loss": clin["brain_volume_loss"],
                "group": clin["label_name"],
            })

    logger.info("Wrote participants.tsv with %d entries", len(all_clinical))

    # ─── Generate MRI volumes ───
    n_total = len(all_clinical)
    for idx, clin in enumerate(all_clinical):
        subject_id = clin["participant_id"]
        label = clin["label"]
        bvl = clin["brain_volume_loss"]

        # Create BIDS directory structure
        anat_dir = output_dir / subject_id / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        nifti_path = anat_dir / f"{subject_id}_T1w.nii.gz"

        if nifti_path.exists():
            logger.debug("Skipping existing: %s", nifti_path)
            continue

        # Generate synthetic volume
        volume = generate_synthetic_mri(
            label=label,
            brain_volume_loss=bvl,
            spatial_size=spatial_size,
            seed=42 + idx,
        )

        # Save as NIfTI with 1mm isotropic affine
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        img = nib.Nifti1Image(volume, affine=affine)
        nib.save(img, str(nifti_path))

        if (idx + 1) % 25 == 0 or idx == 0 or idx == n_total - 1:
            logger.info(
                "  Generated %d/%d: %s (stage=%s, bvl=%.1f)",
                idx + 1, n_total, subject_id,
                STAGE_NAMES[label], bvl,
            )

    # ─── Summary ───
    label_counts = {0: 0, 1: 0, 2: 0}
    for clin in all_clinical:
        label_counts[clin["label"]] += 1

    logger.info("=" * 55)
    logger.info("BIDS Dataset Created Successfully!")
    logger.info("  Output: %s", output_dir)
    logger.info("  Total subjects: %d", n_total)
    for label, name in STAGE_NAMES.items():
        logger.info(
            "    %s (class %d): %d subjects",
            name, label, label_counts[label],
        )
    logger.info(
        "  MRI shape: %s, format: NIfTI (.nii.gz)",
        "96x96x96",
    )
    logger.info("=" * 55)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert HD CSV dataset to BIDS format with synthetic MRI"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to Huntington_Disease_Dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output BIDS directory (default: data/raw)",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=50,
        help="Number of subjects per class to sample (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="MRI volume dimensions (default: 96 96 96)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Read CSV
    rows = read_csv(csv_path)

    # Sample balanced subset
    selected = sample_balanced(rows, per_class=args.per_class, seed=args.seed)

    # Create BIDS dataset
    create_bids_dataset(
        selected_rows=selected,
        output_dir=output_dir,
        spatial_size=tuple(args.spatial_size),
    )


if __name__ == "__main__":
    main()
