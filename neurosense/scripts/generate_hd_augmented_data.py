"""HD-Specific Data Augmentation Script.

Generates augmented training data for Huntington's Disease
classification by applying HD-specific morphological transforms
to existing brain MRI images.

HD neuroimaging hallmarks that this script simulates:
    - Caudate nucleus atrophy (bilateral)
    - Ventricular enlargement (hydrocephalus ex vacuo)
    - Cortical thinning (frontal/parietal)
    - Striatal volume loss

Usage:
    python -m neurosense.scripts.generate_hd_augmented_data \\
        --source-dir data/parkinsons/parkinsons_dataset/normal \\
        --output-dir data/hd_augmented/huntington \\
        --num-augmentations 5

    This takes normal brain MRI images and creates simulated
    HD-like variants with morphological changes characteristic
    of Huntington's Disease.

For future retraining, copy augmented images into your
dataset directory and retrain:
    python -m neurosense.training.train_parkinsons \\
        --data-root data/hd_augmented \\
        --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)


def simulate_ventricular_enlargement(
    img: Image.Image,
    intensity: float = 0.5,
    seed: Optional[int] = None,
) -> Image.Image:
    """Simulate ventricular enlargement typical of HD.

    Creates darkened central regions that mimic enlarged
    lateral ventricles — a hallmark of HD on structural MRI.

    Args:
        img: Input brain MRI image.
        intensity: Augmentation intensity (0.0–1.0).
        seed: Random seed for reproducibility.

    Returns:
        Augmented image with simulated ventricular enlargement.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Create ventricle-shaped masks (bilateral ellipses)
    mask = np.zeros((h, w), dtype=np.float32)

    # Left lateral ventricle
    cx_l = int(w * (0.35 + rng.uniform(-0.03, 0.03)))
    cy = int(h * (0.40 + rng.uniform(-0.05, 0.05)))
    rx = int(w * (0.06 + intensity * 0.08 + rng.uniform(-0.01, 0.01)))
    ry = int(h * (0.10 + intensity * 0.12 + rng.uniform(-0.02, 0.02)))

    y_grid, x_grid = np.ogrid[:h, :w]
    left_mask = ((x_grid - cx_l) / max(rx, 1)) ** 2 + \
                ((y_grid - cy) / max(ry, 1)) ** 2
    mask += np.clip(1.0 - left_mask, 0, 1)

    # Right lateral ventricle (mirror)
    cx_r = w - cx_l
    right_mask = ((x_grid - cx_r) / max(rx, 1)) ** 2 + \
                 ((y_grid - cy) / max(ry, 1)) ** 2
    mask += np.clip(1.0 - right_mask, 0, 1)

    # Third ventricle (smaller, midline)
    cx_m = int(w * 0.5)
    rx_m = int(w * (0.02 + intensity * 0.03))
    ry_m = int(h * (0.04 + intensity * 0.06))
    mid_mask = ((x_grid - cx_m) / max(rx_m, 1)) ** 2 + \
               ((y_grid - cy) / max(ry_m, 1)) ** 2
    mask += np.clip(1.0 - mid_mask, 0, 1) * 0.5

    # Smooth the mask
    mask = np.clip(mask, 0, 1)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=5))
    mask = np.array(mask_img).astype(np.float32) / 255.0

    # Apply: darken ventricle regions (CSF is dark on T1)
    darkening = 1.0 - mask * intensity * 0.7
    if len(arr.shape) == 3:
        darkening = darkening[:, :, np.newaxis]
    result = arr * darkening

    return Image.fromarray(result.astype(np.uint8))


def simulate_cortical_atrophy(
    img: Image.Image,
    intensity: float = 0.5,
    seed: Optional[int] = None,
) -> Image.Image:
    """Simulate cortical atrophy pattern seen in HD.

    Widens sulci and reduces overall brain volume appearance
    by applying subtle shrinkage toward center.

    Args:
        img: Input brain MRI image.
        intensity: Augmentation intensity (0.0–1.0).
        seed: Random seed for reproducibility.

    Returns:
        Augmented image with simulated cortical atrophy.
    """
    rng = np.random.RandomState(seed)

    # Slight zoom-out to simulate brain shrinkage
    scale = 1.0 - intensity * 0.08 - rng.uniform(0, 0.02)
    w, h = img.size

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize smaller
    shrunk = img.resize((new_w, new_h), Image.LANCZOS)

    # Paste onto dark background (simulating wider sulci/CSF)
    result = Image.new(img.mode, (w, h), 0)
    x_offset = (w - new_w) // 2
    y_offset = (h - new_h) // 2
    result.paste(shrunk, (x_offset, y_offset))

    return result


def simulate_caudate_atrophy(
    img: Image.Image,
    intensity: float = 0.5,
    seed: Optional[int] = None,
) -> Image.Image:
    """Simulate caudate nucleus atrophy — the hallmark of HD.

    Darkens the caudate nucleus region (adjacent to lateral
    ventricles) to simulate the characteristic atrophy pattern.

    Args:
        img: Input brain MRI image.
        intensity: Augmentation intensity (0.0–1.0).
        seed: Random seed for reproducibility.

    Returns:
        Augmented image with simulated caudate atrophy.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    mask = np.zeros((h, w), dtype=np.float32)
    y_grid, x_grid = np.ogrid[:h, :w]

    # Caudate heads — bilateral, adjacent to frontal horns
    for side in [-1, 1]:
        cx = int(w * (0.5 + side * (0.12 + rng.uniform(-0.02, 0.02))))
        cy = int(h * (0.38 + rng.uniform(-0.03, 0.03)))
        rx = int(w * (0.04 + intensity * 0.03))
        ry = int(h * (0.06 + intensity * 0.04))

        caudate_mask = ((x_grid - cx) / max(rx, 1)) ** 2 + \
                       ((y_grid - cy) / max(ry, 1)) ** 2
        mask += np.clip(1.0 - caudate_mask, 0, 1)

    mask = np.clip(mask, 0, 1)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
    mask = np.array(mask_img).astype(np.float32) / 255.0

    darkening = 1.0 - mask * intensity * 0.5
    if len(arr.shape) == 3:
        darkening = darkening[:, :, np.newaxis]
    result = arr * darkening

    return Image.fromarray(result.astype(np.uint8))


def add_contrast_variation(
    img: Image.Image,
    seed: Optional[int] = None,
) -> Image.Image:
    """Add subtle contrast and brightness variations.

    Simulates different MRI scanner settings and sequences.

    Args:
        img: Input image.
        seed: Random seed.

    Returns:
        Image with contrast variation.
    """
    rng = np.random.RandomState(seed)
    from PIL import ImageEnhance

    # Random contrast
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(rng.uniform(0.8, 1.3))

    # Random brightness
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(rng.uniform(0.85, 1.15))

    return img


def augment_image(
    img: Image.Image,
    intensity: float = 0.5,
    seed: int = 42,
) -> Image.Image:
    """Apply full HD augmentation pipeline to an image.

    Combines multiple HD-specific augmentations:
    1. Ventricular enlargement
    2. Cortical atrophy
    3. Caudate nucleus atrophy
    4. Contrast variation

    Args:
        img: Input brain MRI image.
        intensity: Overall augmentation intensity (0.0–1.0).
        seed: Random seed for reproducibility.

    Returns:
        Augmented image with HD-like morphological changes.
    """
    rng = np.random.RandomState(seed)

    # Apply augmentations with random intensity variation
    if rng.random() > 0.2:
        sub_intensity = intensity * rng.uniform(0.6, 1.0)
        img = simulate_ventricular_enlargement(
            img, intensity=sub_intensity, seed=seed + 1,
        )

    if rng.random() > 0.3:
        sub_intensity = intensity * rng.uniform(0.4, 1.0)
        img = simulate_cortical_atrophy(
            img, intensity=sub_intensity, seed=seed + 2,
        )

    if rng.random() > 0.2:
        sub_intensity = intensity * rng.uniform(0.5, 1.0)
        img = simulate_caudate_atrophy(
            img, intensity=sub_intensity, seed=seed + 3,
        )

    # Always add contrast variation
    img = add_contrast_variation(img, seed=seed + 4)

    return img


def generate_augmented_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    num_augmentations: int = 5,
    intensity_range: tuple[float, float] = (0.3, 0.9),
    seed: int = 42,
) -> int:
    """Generate augmented HD training data from source images.

    Takes brain MRI images from a source directory and creates
    multiple augmented variants with HD-specific morphological
    changes at varying intensities.

    Args:
        source_dir: Directory containing source PNG images.
        output_dir: Output directory for augmented images.
        num_augmentations: Number of augmented variants per image.
        intensity_range: Min/max augmentation intensity.
        seed: Base random seed.

    Returns:
        Number of images generated.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_images = sorted(
        list(source_dir.glob("*.png"))
        + list(source_dir.glob("*.jpg"))
        + list(source_dir.glob("*.jpeg"))
    )

    if not source_images:
        logger.warning("No images found in %s", source_dir)
        return 0

    logger.info(
        "Generating %d augmentations for %d source images → %s",
        num_augmentations, len(source_images), output_dir,
    )

    rng = np.random.RandomState(seed)
    count = 0

    for img_path in source_images:
        img = Image.open(img_path).convert("RGB")

        for aug_idx in range(num_augmentations):
            intensity = rng.uniform(*intensity_range)
            aug_seed = seed + count

            augmented = augment_image(
                img, intensity=intensity, seed=aug_seed,
            )

            stem = img_path.stem
            out_name = f"HD_aug_{stem}_{aug_idx:03d}.png"
            augmented.save(output_dir / out_name)
            count += 1

            if count % 50 == 0:
                logger.info("  Generated %d images...", count)

    logger.info("Generated %d augmented HD images", count)
    return count


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate HD-augmented brain MRI training data. "
            "Applies HD-specific morphological transforms "
            "(ventricular enlargement, caudate atrophy, cortical "
            "thinning) to create synthetic training images."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/parkinsons/parkinsons_dataset/normal",
        help="Source directory with normal brain MRI images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/hd_augmented/huntington",
        help="Output directory for augmented HD images",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=5,
        help="Number of augmented variants per source image",
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.3,
        help="Minimum augmentation intensity (0.0–1.0)",
    )
    parser.add_argument(
        "--max-intensity",
        type=float,
        default=0.9,
        help="Maximum augmentation intensity (0.0–1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Also copy source normals to output normal directory
    normal_output = Path(args.output_dir).parent / "normal"
    normal_output.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.source_dir)
    normal_count = 0
    for img_path in sorted(source_dir.glob("*.png")):
        img = Image.open(img_path).convert("RGB")
        img.save(normal_output / img_path.name)
        normal_count += 1

    logger.info("Copied %d normal images to %s", normal_count, normal_output)

    # Generate augmented HD images
    hd_count = generate_augmented_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations,
        intensity_range=(args.min_intensity, args.max_intensity),
        seed=args.seed,
    )

    print(f"\n{'='*50}")
    print("HD Data Augmentation Complete")
    print(f"  Normal images:     {normal_count}")
    print(f"  Augmented HD:      {hd_count}")
    print(f"  Output directory:  {Path(args.output_dir).parent}")
    print(f"\nTo retrain with augmented data:")
    print(f"  python -m neurosense.training.train_parkinsons \\")
    print(f"    --data-root {Path(args.output_dir).parent} \\")
    print(f"    --epochs 30")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
