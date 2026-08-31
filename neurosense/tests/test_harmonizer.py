"""NeuroSense Unit Tests — MRI Intensity & Contrast Harmonization.

Tests that the MRIHarmonizer effectively eliminates shortcut learning:
- Background scanner air and peripheral artifacts are zeroed out.
- Cranial region is centered and aspect-ratio preserved.
- Local CLAHE contrast equalization enhances anatomical tissue structures.
- Foreground-only Z-score standardizes brightness across scanner protocols.
- Synthetic brightness/contrast shifts achieve protocol invariance.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from neurosense.data.harmonizer import (
    MRIHarmonizer,
    harmonize_2d_slice,
    harmonize_3d_volume,
)


@pytest.fixture
def synthetic_brain_slice():
    """Create a synthetic 2D brain slice with cranial tissue and noisy scanner background."""
    h, w = 256, 256
    arr = np.zeros((h, w), dtype=np.uint8)

    # Brain ellipse
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    brain_mask = ((x - cx) / 80) ** 2 + ((y - cy) / 100) ** 2 <= 1.0

    # Gray/White matter intensities
    arr[brain_mask] = np.random.randint(90, 180, size=brain_mask.sum(), dtype=np.uint8)

    # Ventricles (darker inside)
    vent_mask = ((x - cx) / 20) ** 2 + ((y - cy) / 35) ** 2 <= 1.0
    arr[vent_mask] = np.random.randint(20, 50, size=vent_mask.sum(), dtype=np.uint8)

    # Add scanner noise floor in the background air
    bg_noise = (~brain_mask) & (np.random.rand(h, w) > 0.90)
    arr[bg_noise] = np.random.randint(1, 15, size=bg_noise.sum(), dtype=np.uint8)

    return Image.fromarray(arr)


class TestMRIHarmonizer:
    """Test suite for the MRI Harmonization pipeline."""

    def test_cranial_mask_extraction(self, synthetic_brain_slice):
        """Background scanner air and noise are eliminated by the mask."""
        harmonizer = MRIHarmonizer(target_size=224)
        img_gray = np.array(synthetic_brain_slice.convert("L"), dtype=np.uint8)
        mask = harmonizer.extract_cranial_mask(img_gray)

        assert mask.shape == img_gray.shape
        assert mask.dtype == np.uint8
        assert mask.max() == 1
        assert mask.min() == 0

        # Background corner pixels should be strictly 0
        assert mask[5, 5] == 0
        assert mask[5, -5] == 0
        assert mask[-5, 5] == 0
        assert mask[-5, -5] == 0

        # Center brain pixel should be 1
        assert mask[128, 128] == 1

    def test_crop_and_pad_dimensions(self, synthetic_brain_slice):
        """Cropped and padded image matches the target resolution."""
        harmonizer = MRIHarmonizer(target_size=224)
        img_gray = np.array(synthetic_brain_slice.convert("L"), dtype=np.uint8)
        mask = harmonizer.extract_cranial_mask(img_gray)

        cropped_img, cropped_mask = harmonizer.crop_and_pad_cranial(img_gray, mask)

        assert cropped_img.shape == (224, 224)
        assert cropped_mask.shape == (224, 224)
        assert cropped_img.dtype == np.uint8

    def test_clahe_contrast_enhancement(self, synthetic_brain_slice):
        """CLAHE enhances tissue contrast without altering zero background."""
        harmonizer = MRIHarmonizer(target_size=224)
        img_gray = np.array(synthetic_brain_slice.convert("L"), dtype=np.uint8)
        mask = harmonizer.extract_cranial_mask(img_gray)
        cropped_img, cropped_mask = harmonizer.crop_and_pad_cranial(img_gray, mask)

        clahe_img = harmonizer.apply_clahe(cropped_img, cropped_mask)

        assert clahe_img.shape == (224, 224)
        assert (clahe_img[cropped_mask == 0] == 0).all(), "Background should be strictly 0"
        assert clahe_img[cropped_mask > 0].max() > 0

    def test_foreground_zscore_properties(self, synthetic_brain_slice):
        """Z-Score normalizes foreground brain pixels into standardized [0, 1] range."""
        harmonizer = MRIHarmonizer(target_size=224)
        img_gray = np.array(synthetic_brain_slice.convert("L"), dtype=np.uint8)
        mask = harmonizer.extract_cranial_mask(img_gray)
        cropped_img, cropped_mask = harmonizer.crop_and_pad_cranial(img_gray, mask)
        clahe_img = harmonizer.apply_clahe(cropped_img, cropped_mask)

        normalized = harmonizer.apply_foreground_zscore(clahe_img, cropped_mask)

        assert normalized.shape == (224, 224)
        assert normalized.dtype == np.float32
        assert (normalized[cropped_mask == 0] == 0.0).all(), "Background must be 0.0"
        assert 0.0 <= normalized.min() <= 1.0
        assert 0.0 <= normalized.max() <= 1.0

    def test_harmonize_2d_slice_pipeline(self, synthetic_brain_slice):
        """Full 2D pipeline produces valid numpy array and PyTorch tensor."""
        harmonized_2d, tensor = harmonize_2d_slice(synthetic_brain_slice, target_size=224)

        assert harmonized_2d.shape == (224, 224)
        assert harmonized_2d.dtype == np.float32

        # PyTorch Tensor: [1, 3, 224, 224]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()

    def test_protocol_intensity_invariance(self, synthetic_brain_slice):
        """Simulating two different scanner brightness/contrast protocols results in harmonized parity."""
        orig_arr = np.array(synthetic_brain_slice, dtype=np.float32)

        # Scanner A: Darker / low gain
        scan_a = Image.fromarray((orig_arr * 0.6).astype(np.uint8))

        # Scanner B: Brighter / high gain + contrast shift
        scan_b = Image.fromarray(np.clip(orig_arr * 1.4 + 20, 0, 255).astype(np.uint8))

        harm_a, tensor_a = harmonize_2d_slice(scan_a, target_size=224)
        harm_b, tensor_b = harmonize_2d_slice(scan_b, target_size=224)

        # Extract foreground tissue regions
        mask_a = harm_a > 0
        mask_b = harm_b > 0

        # Mean brain tissue intensity should be tightly harmonized within 10%
        mean_a = float(harm_a[mask_a].mean())
        mean_b = float(harm_b[mask_b].mean())

        assert abs(mean_a - mean_b) < 0.10, (
            f"Harmonization should normalize brightness protocol shifts: {mean_a:.3f} vs {mean_b:.3f}"
        )

    def test_3d_volume_harmonization(self):
        """3D volume harmonization normalizes voxel intensities and background."""
        vol = np.zeros((48, 48, 48), dtype=np.float32)
        vol[12:36, 12:36, 12:36] = np.random.normal(150, 25, size=(24, 24, 24))
        vol = np.clip(vol, 0, 255)

        harm_vol = harmonize_3d_volume(vol, spatial_size=(48, 48, 48))

        assert harm_vol.shape == (48, 48, 48)
        assert harm_vol.dtype == np.float32
        assert harm_vol[0, 0, 0] == 0.0
        assert 0.0 <= harm_vol.min() <= 1.0
        assert 0.0 <= harm_vol.max() <= 1.0
