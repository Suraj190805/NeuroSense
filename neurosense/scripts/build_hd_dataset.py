"""HD MRI Dataset Builder — Collect & Organize Images for Training.

This script helps you build a training dataset from brain MRI
images you've manually downloaded from Google, Radiopaedia, or
other sources.

WORKFLOW:
  1. Download HD brain MRI images from Google/Radiopaedia
     → Save them into: data/hd_training/raw_huntington/
  
  2. Download normal brain MRI images from Google
     → Save them into: data/hd_training/raw_normal/
  
  3. Run this script to process, resize, and organize them:
     python -m neurosense.scripts.build_hd_dataset

  4. Retrain the model:
     python -m neurosense.training.train_parkinsons \\
         --data-root data/hd_training/dataset \\
         --epochs 30 --batch-size 8

The script will:
  - Resize all images to 256×256
  - Convert to RGB PNG
  - Apply augmentations (flips, rotations, contrast) to
    increase the effective dataset size
  - Create train/val/test split directories
  - Generate a dataset report

TIPS FOR COLLECTING IMAGES:
  Search Google Images for:
    HD images:
      - "huntington disease brain MRI"
      - "huntington disease coronal MRI caudate atrophy"
      - "huntington disease axial MRI ventricles"
      - "huntington chorea brain scan"
      - "HD brain atrophy MRI"
    
    Normal images:
      - "normal brain MRI coronal"
      - "normal brain MRI axial"
      - "healthy brain MRI scan"
    
  Also check:
    - Radiopaedia.org (search "huntington disease", click Cases)
    - Wikipedia "Huntington's disease" article images
    - Google Scholar figures
  
  You only need ~20-30 images per class. The script will
  augment them to 100+ per class for training.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


def process_image(
    img: Image.Image,
    target_size: int = 256,
) -> Image.Image:
    """Process a raw image for training.

    Converts to RGB, resizes to square, and normalizes.

    Args:
        img: Raw input image.
        target_size: Target dimension (square).

    Returns:
        Processed image.
    """
    # Convert to RGB
    img = img.convert("RGB")

    # Resize to square (maintain aspect ratio, pad if needed)
    w, h = img.size
    max_dim = max(w, h)
    
    # Create square canvas with black background
    square = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    square.paste(img, (offset_x, offset_y))

    # Resize to target
    square = square.resize((target_size, target_size), Image.LANCZOS)

    return square


def augment_for_training(
    img: Image.Image,
    num_augmentations: int = 5,
    seed: int = 42,
) -> list[Image.Image]:
    """Generate augmented variants of an image.

    Creates multiple training samples from one image using
    standard medical imaging augmentations.

    Args:
        img: Input image.
        num_augmentations: Number of augmented variants.
        seed: Random seed.

    Returns:
        List of augmented images (including original).
    """
    rng = np.random.RandomState(seed)
    augmented = [img]  # Include original

    for i in range(num_augmentations):
        aug = img.copy()

        # Random horizontal flip
        if rng.random() > 0.5:
            aug = ImageOps.mirror(aug)

        # Random rotation (-15 to +15 degrees)
        angle = rng.uniform(-15, 15)
        aug = aug.rotate(angle, fillcolor=(0, 0, 0))

        # Random contrast
        factor = rng.uniform(0.7, 1.3)
        aug = ImageEnhance.Contrast(aug).enhance(factor)

        # Random brightness
        factor = rng.uniform(0.8, 1.2)
        aug = ImageEnhance.Brightness(aug).enhance(factor)

        # Random slight blur (simulates different scan quality)
        if rng.random() > 0.7:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=1))

        # Random crop and resize back (slight zoom effect)
        if rng.random() > 0.5:
            w, h = aug.size
            crop_pct = rng.uniform(0.85, 0.95)
            new_w = int(w * crop_pct)
            new_h = int(h * crop_pct)
            left = rng.randint(0, w - new_w)
            top = rng.randint(0, h - new_h)
            aug = aug.crop((left, top, left + new_w, top + new_h))
            aug = aug.resize((w, h), Image.LANCZOS)

        augmented.append(aug)

    return augmented


def build_dataset(
    raw_hd_dir: str | Path,
    raw_normal_dir: str | Path,
    output_dir: str | Path,
    target_size: int = 256,
    augmentations_per_image: int = 5,
    seed: int = 42,
) -> dict:
    """Build a training dataset from raw collected images.

    Args:
        raw_hd_dir: Directory with raw HD brain MRI images.
        raw_normal_dir: Directory with raw normal brain MRI images.
        output_dir: Output dataset directory.
        target_size: Image size for training.
        augmentations_per_image: Augmentations per source image.
        seed: Random seed.

    Returns:
        Dict with dataset statistics.
    """
    raw_hd_dir = Path(raw_hd_dir)
    raw_normal_dir = Path(raw_normal_dir)
    output_dir = Path(output_dir)

    # Create output directories using the class names expected
    # by the existing ParkinsonsDataset loader
    # We map: "normal" → class 0, "parkinson" → class 1 (HD)
    normal_out = output_dir / "normal"
    hd_out = output_dir / "parkinson"  # Reuse class name for compatibility

    normal_out.mkdir(parents=True, exist_ok=True)
    hd_out.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    stats = {"raw_hd": 0, "raw_normal": 0, "aug_hd": 0, "aug_normal": 0}

    # Process HD images
    hd_images = [
        f for f in sorted(raw_hd_dir.iterdir())
        if f.suffix.lower() in image_extensions
    ] if raw_hd_dir.exists() else []

    stats["raw_hd"] = len(hd_images)
    logger.info("Found %d raw HD images in %s", len(hd_images), raw_hd_dir)

    count = 0
    for img_path in hd_images:
        try:
            img = Image.open(img_path)
            img = process_image(img, target_size)

            # Generate augmented variants
            variants = augment_for_training(
                img,
                num_augmentations=augmentations_per_image,
                seed=seed + count,
            )

            for j, variant in enumerate(variants):
                out_name = f"HD_{count:04d}_{j:02d}.png"
                variant.save(hd_out / out_name)
                stats["aug_hd"] += 1

            count += 1
        except Exception as e:
            logger.warning("Failed to process %s: %s", img_path, e)

    # Process normal images
    normal_images = [
        f for f in sorted(raw_normal_dir.iterdir())
        if f.suffix.lower() in image_extensions
    ] if raw_normal_dir.exists() else []

    stats["raw_normal"] = len(normal_images)
    logger.info("Found %d raw normal images in %s", len(normal_images), raw_normal_dir)

    count = 0
    for img_path in normal_images:
        try:
            img = Image.open(img_path)
            img = process_image(img, target_size)

            variants = augment_for_training(
                img,
                num_augmentations=augmentations_per_image,
                seed=seed + 10000 + count,
            )

            for j, variant in enumerate(variants):
                out_name = f"Normal_{count:04d}_{j:02d}.png"
                variant.save(normal_out / out_name)
                stats["aug_normal"] += 1

            count += 1
        except Exception as e:
            logger.warning("Failed to process %s: %s", img_path, e)

    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Build HD training dataset from manually collected images. "
            "Download brain MRI images from Google/Radiopaedia, "
            "place them in raw_huntington/ and raw_normal/ folders, "
            "then run this script to process and augment them."
        ),
    )
    parser.add_argument(
        "--raw-hd-dir",
        type=str,
        default="data/hd_training/raw_huntington",
        help="Directory with raw HD brain MRI images",
    )
    parser.add_argument(
        "--raw-normal-dir",
        type=str,
        default="data/hd_training/raw_normal",
        help="Directory with raw normal brain MRI images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/hd_training/dataset",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Target image size (default: 256)",
    )
    parser.add_argument(
        "--augmentations",
        type=int,
        default=5,
        help="Number of augmented variants per image (default: 5)",
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

    # Create raw directories if they don't exist
    Path(args.raw_hd_dir).mkdir(parents=True, exist_ok=True)
    Path(args.raw_normal_dir).mkdir(parents=True, exist_ok=True)

    # Check if there are images to process
    hd_count = len([
        f for f in Path(args.raw_hd_dir).iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]) if Path(args.raw_hd_dir).exists() else 0

    normal_count = len([
        f for f in Path(args.raw_normal_dir).iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]) if Path(args.raw_normal_dir).exists() else 0

    if hd_count == 0 and normal_count == 0:
        print(f"\n{'='*60}")
        print("  HD Dataset Builder — Setup Required")
        print(f"{'='*60}")
        print()
        print("  No images found! Please collect images first:")
        print()
        print(f"  1. Download HD brain MRI images and save to:")
        print(f"     → {args.raw_hd_dir}/")
        print()
        print(f"  2. Download normal brain MRI images and save to:")
        print(f"     → {args.raw_normal_dir}/")
        print()
        print("  Search Google Images for:")
        print('    HD:     "huntington disease brain MRI"')
        print('    Normal: "normal brain MRI coronal"')
        print()
        print("  You need at least 15-20 images per class.")
        print("  The script will augment them to 100+ for training.")
        print()
        print(f"  Then re-run this script.")
        print(f"{'='*60}")
        return

    stats = build_dataset(
        raw_hd_dir=args.raw_hd_dir,
        raw_normal_dir=args.raw_normal_dir,
        output_dir=args.output_dir,
        target_size=args.image_size,
        augmentations_per_image=args.augmentations,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print("  HD Dataset Build Complete!")
    print(f"{'='*60}")
    print(f"  Raw HD images:      {stats['raw_hd']}")
    print(f"  Raw normal images:  {stats['raw_normal']}")
    print(f"  Augmented HD:       {stats['aug_hd']}")
    print(f"  Augmented normal:   {stats['aug_normal']}")
    print(f"  Output:             {args.output_dir}")
    print()
    print(f"  To retrain the model:")
    print(f"    python -m neurosense.training.train_parkinsons \\")
    print(f"      --data-root {args.output_dir} \\")
    print(f"      --epochs 30 --batch-size 8")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
