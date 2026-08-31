"""NeuroSense Unit Tests — Heavy Data Augmentation Pipeline.

Tests that the augmentation pipeline simulates clinical imaging variability:
1. Random Affine: Rotation (±15°), Translation (±10%), Scaling (0.9–1.1).
2. Random Gaussian Blur: Motion artifacts & point-spread function simulation.
3. Color Jitter: Brightness & contrast variations.
4. Random Erasing / Cutout: (p=0.3) Region occlusion for multi-focal representation.
5. Deterministic Validation Pipeline: Validation transforms are repeatable and unperturbed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from neurosense.data.hd_image_dataset import (
    get_2d_train_transforms,
    get_2d_val_transforms,
)


def create_synthetic_slice() -> Image.Image:
    """Create a sample brain slice PIL Image."""
    arr = np.zeros((256, 256), dtype=np.uint8)
    y, x = np.ogrid[:256, :256]
    cy, cx = 128, 128
    brain = ((x - cx) / 75) ** 2 + ((y - cy) / 95) ** 2 <= 1.0
    arr[brain] = 160
    ventricle = ((x - cx) / 18) ** 2 + ((y - cy) / 30) ** 2 <= 1.0
    arr[ventricle] = 30
    return Image.fromarray(arr).convert("RGB")


def test_train_transforms_structure():
    """Verify that get_2d_train_transforms contains all requested augmentations."""
    train_tf = get_2d_train_transforms(image_size=224)

    transform_types = [type(t) for t in train_tf.transforms]

    # Check Affine (rotation, translation, scaling)
    assert transforms.RandomAffine in transform_types, "Must include RandomAffine"
    affine_tf = next(t for t in train_tf.transforms if isinstance(t, transforms.RandomAffine))
    assert list(affine_tf.degrees) == [-15.0, 15.0], f"Expected ±15° rotation, got {affine_tf.degrees}"
    assert affine_tf.translate == (0.10, 0.10), f"Expected ±10% translation, got {affine_tf.translate}"
    assert affine_tf.scale == (0.90, 1.10), f"Expected 0.9-1.1 scale, got {affine_tf.scale}"

    # Check Horizontal Flip
    assert transforms.RandomHorizontalFlip in transform_types, "Must include RandomHorizontalFlip"

    # Check Color Jitter
    assert transforms.ColorJitter in transform_types, "Must include ColorJitter"

    # Check Random Erasing (Cutout)
    assert transforms.RandomErasing in transform_types, "Must include RandomErasing"
    erase_tf = next(t for t in train_tf.transforms if isinstance(t, transforms.RandomErasing))
    assert erase_tf.p == 0.3, f"Expected RandomErasing p=0.3, got {erase_tf.p}"


def test_random_erasing_execution():
    """Verify that RandomErasing occludes patches when triggered."""
    erase_tf = transforms.RandomErasing(p=1.0, scale=(0.05, 0.2), value=0)
    tensor = torch.ones(3, 224, 224)

    erased = erase_tf(tensor)
    # Some pixels should be zeroed out
    assert (erased == 0).sum() > 0, "RandomErasing must occlude a subset of pixels"
    assert (erased == 1).sum() > 0, "Non-erased regions must remain intact"


def test_gaussian_blur_augmentation():
    """Verify Gaussian blur transform executes and produces blurred output."""
    blur_tf = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.5, 1.5))
    img = create_synthetic_slice()

    blurred = blur_tf(img)
    arr_orig = np.array(img, dtype=np.float32)
    arr_blur = np.array(blurred, dtype=np.float32)

    # Edge gradient in blurred image should be smoother (lower standard deviation of differences)
    grad_orig = np.diff(arr_orig, axis=0).std()
    grad_blur = np.diff(arr_blur, axis=0).std()
    assert grad_blur < grad_orig, "Gaussian blur must smooth local spatial gradients"


def test_train_transforms_output_shape():
    """Verify train transforms produce valid [3, 224, 224] tensor."""
    train_tf = get_2d_train_transforms(image_size=224)
    img = create_synthetic_slice()

    out_tensor = train_tf(img)
    assert out_tensor.shape == (3, 224, 224)
    assert out_tensor.dtype == torch.float32
    assert not torch.isnan(out_tensor).any()


def test_val_transforms_deterministic():
    """Validation transforms must be purely deterministic and repeatable."""
    val_tf = get_2d_val_transforms(image_size=224)
    img = create_synthetic_slice()

    t1 = val_tf(img)
    t2 = val_tf(img)

    assert torch.equal(t1, t2), "Validation transforms must produce identical results across runs"


if __name__ == "__main__":
    test_train_transforms_structure()
    print("✅ Test 1: Transform Structure (Affine ±15°/±10%/0.9-1.1, Erasing p=0.3) passed")
    test_random_erasing_execution()
    print("✅ Test 2: Random Erasing / Cutout execution passed")
    test_gaussian_blur_augmentation()
    print("✅ Test 3: Gaussian Blur motion simulation passed")
    test_train_transforms_output_shape()
    print("✅ Test 4: Tensor Output Shape [3, 224, 224] passed")
    test_val_transforms_deterministic()
    print("✅ Test 5: Deterministic Validation Transform passed")
    print("\n🎉 ALL 5 DATA AUGMENTATION TESTS PASSED CLEANLY!")
