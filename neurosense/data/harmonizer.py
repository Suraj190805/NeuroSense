"""NeuroSense MRI Intensity & Contrast Harmonization Pipeline.

Kills shortcut learning by standardizing Brain MRI scans across different
scanner vendors (Siemens, GE, Philips), field strengths (1.5T, 3T), and pulse
sequences (T1w, T2w, FLAIR, SWI, Diffusion).

Core Harmonization Stages:
1. Background Air Masking & Noise Stripping:
   - Uses Otsu thresholding + morphological closure + largest connected component
     extraction to eliminate scanner background noise, vendor watermarks, and frame
     borders, preventing the network from learning scanner noise floor fingerprints.
2. Cranial Aspect-Preserving Cropping:
   - Identifies the cranial bounding box with 5% anatomical margin, centers the
     brain, and applies aspect-ratio-preserving padding to prevent geometric distortion
     of the ventricles and cortex.
3. CLAHE (Contrast-Limited Adaptive Histogram Equalization):
   - Balances local tissue contrast (gray matter vs white matter vs CSF) across
     image tiles, ensuring the network focuses on caudate atrophy and ventricular
     enlargement rather than global lighting or sequence exposure variations.
4. Foreground-Only Brain Z-Score Normalization:
   - Standardizes brain tissue intensities exclusively on non-zero cranial pixels:
     Z = (I - mu_brain) / sigma_brain
   - Clips intensity outliers to [-3.0, 3.0] and maps to [0, 1] for neural network
     stability, guaranteeing brightness consistency across all input scans.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class MRIHarmonizer:
    """Comprehensive MRI Intensity & Contrast Harmonizer."""

    def __init__(
        self,
        target_size: int = 224,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
        zscore_clip: tuple[float, float] = (-3.0, 3.0),
        cranial_margin: float = 0.05,
    ) -> None:
        """Initialize MRI Harmonizer parameters.

        Args:
            target_size: Standardized output square dimension (e.g. 224).
            clahe_clip_limit: Contrast limit threshold for CLAHE.
            clahe_tile_grid_size: Grid dimensions for local histogram equalization.
            zscore_clip: Min/max standard deviations for intensity clipping.
            cranial_margin: Relative bounding margin around the cranial contour.
        """
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.zscore_clip = zscore_clip
        self.cranial_margin = cranial_margin

    def extract_cranial_mask(self, image_gray: np.ndarray) -> np.ndarray:
        """Extract a clean binary mask of the brain/cranial tissue.

        Eliminates background scanner air, frame lines, and peripheral noise.

        Args:
            image_gray: 2D uint8 numpy array [H, W] with intensities in [0, 255].

        Returns:
            2D uint8 binary mask [H, W] where 1 = cranial tissue, 0 = background.
        """
        h, w = image_gray.shape

        # 1. Otsu thresholding to separate foreground head from background air
        _, otsu_mask = cv2.threshold(
            image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Ensure minimum threshold to cut out dark scanner grain/noise
        min_thresh = max(int(np.percentile(image_gray, 20)), 15)
        thresh_mask = np.where(image_gray >= min_thresh, otsu_mask, 0).astype(np.uint8)

        # 2. Morphological cleanup: remove small noise specks and close internal gaps
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        opened = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # 3. Connected components: select the largest component (the brain/head)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )

        if num_labels <= 1:
            # Fallback if no component found: center circular mask
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 3
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2).astype(np.uint8)
            return mask

        # Find largest non-background component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cranial_mask = (labels == largest_label).astype(np.uint8)

        # Fill internal holes (e.g. ventricles inside the brain boundary)
        contours, _ = cv2.findContours(
            cranial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cranial_mask = np.zeros_like(cranial_mask)
            cv2.drawContours(cranial_mask, contours, -1, 1, thickness=cv2.FILLED)

        return cranial_mask

    def crop_and_pad_cranial(
        self,
        image_gray: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop tightly around cranial bounding box and pad to square.

        Preserves exact anatomical proportions without stretching.

        Args:
            image_gray: 2D uint8 image array.
            mask: 2D binary cranial mask.

        Returns:
            Tuple of (resized_image, resized_mask) of shape [target_size, target_size].
        """
        h, w = image_gray.shape
        masked_img = image_gray * mask

        # Find bounding box of the cranial mask
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            resized_img = cv2.resize(
                masked_img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA
            )
            return resized_img, (resized_img > 0).astype(np.uint8)

        ymin, ymax = y_indices.min(), y_indices.max()
        xmin, xmax = x_indices.min(), x_indices.max()

        # Add margin
        margin_y = int((ymax - ymin) * self.cranial_margin)
        margin_x = int((xmax - xmin) * self.cranial_margin)

        ymin = max(0, ymin - margin_y)
        ymax = min(h, ymax + margin_y)
        xmin = max(0, xmin - margin_x)
        xmax = min(w, xmax + margin_x)

        cropped_img = masked_img[ymin:ymax, xmin:xmax]
        cropped_mask = mask[ymin:ymax, xmin:xmax]

        ch, cw = cropped_img.shape
        max_dim = max(ch, cw)

        # Pad to square with zero black background
        padded_img = np.zeros((max_dim, max_dim), dtype=cropped_img.dtype)
        padded_mask = np.zeros((max_dim, max_dim), dtype=cropped_mask.dtype)

        offset_y = (max_dim - ch) // 2
        offset_x = (max_dim - cw) // 2

        padded_img[offset_y : offset_y + ch, offset_x : offset_x + cw] = cropped_img
        padded_mask[offset_y : offset_y + ch, offset_x : offset_x + cw] = cropped_mask

        # Resize to standardized target size
        resized_img = cv2.resize(
            padded_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )
        resized_mask = cv2.resize(
            padded_mask,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST,
        )

        return resized_img, resized_mask

    def apply_clahe(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Apply Contrast-Limited Adaptive Histogram Equalization to cranial tissue.

        Args:
            image: 2D uint8 image array [H, W].
            mask: 2D binary mask [H, W].

        Returns:
            Contrast-harmonized uint8 image [H, W] with zeroed background.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )
        equalized = clahe.apply(image)

        # Keep background strictly zero
        equalized = equalized * mask
        return equalized

    def apply_foreground_zscore(
        self,
        image_equalized: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Apply Z-score intensity normalization exclusively to foreground brain pixels.

        Formula:
            Normalized = (Image - mean_brain) / (std_brain + eps)
            Clipped to [zscore_clip[0], zscore_clip[1]] and mapped to [0, 1].

        Args:
            image_equalized: 2D uint8 array with intensities [0, 255].
            mask: 2D binary cranial mask.

        Returns:
            2D float32 normalized image [H, W] in [0.0, 1.0].
        """
        img_float = image_equalized.astype(np.float32)
        tissue_pixels = img_float[mask > 0]

        if len(tissue_pixels) == 0 or tissue_pixels.std() < 1e-5:
            # Fallback min-max
            max_val = max(img_float.max(), 1.0)
            return (img_float / max_val).astype(np.float32)

        mu_brain = float(tissue_pixels.mean())
        sigma_brain = float(tissue_pixels.std())

        # Standardize foreground pixels
        z_norm = np.zeros_like(img_float)
        z_norm[mask > 0] = (img_float[mask > 0] - mu_brain) / (sigma_brain + 1e-7)

        # Clip outlier standard deviations (e.g. -3.0 to +3.0)
        z_min, z_max = self.zscore_clip
        z_clipped = np.clip(z_norm, z_min, z_max)

        # Map clipped Z-score to [0.0, 1.0] for neural network feature extraction
        normalized = np.zeros_like(img_float)
        normalized[mask > 0] = (z_clipped[mask > 0] - z_min) / (z_max - z_min)

        return normalized.astype(np.float32)

    def harmonize(
        self,
        image_input: Image.Image | bytes | np.ndarray,
        device: torch.device | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Run the complete harmonization pipeline on a 2D Brain MRI slice.

        Pipeline Steps:
        1. Parse input to grayscale uint8.
        2. Extract clean cranial mask (strips scanner background & air noise).
        3. Aspect-preserving crop & pad around head.
        4. Apply CLAHE local contrast normalization.
        5. Apply foreground-only brain Z-score normalization.
        6. Format as float32 array [H, W] and PyTorch tensor [1, 3, H, W].

        Args:
            image_input: PIL Image, raw image bytes, or 2D/3D numpy array.
            device: Optional torch.device for the output tensor.

        Returns:
            Tuple of:
            - harmonized_2d: 2D numpy array [H, W] with float32 values in [0, 1].
            - tensor_3ch: PyTorch Tensor [1, 3, H, W] normalized for vision models.
        """
        # 1. Parse input to single-channel uint8 array
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("L")
            img_arr = np.array(pil_img, dtype=np.uint8)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("L")
            img_arr = np.array(pil_img, dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[-1] in (3, 4):
                pil_img = Image.fromarray(image_input).convert("L")
                img_arr = np.array(pil_img, dtype=np.uint8)
            else:
                img_f = image_input.astype(np.float32)
                min_v, max_v = img_f.min(), img_f.max()
                ptp = max(max_v - min_v, 1e-6)
                img_arr = ((img_f - min_v) / ptp * 255.0).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # 2. Extract cranial mask (Otsu + connected component + hole filling)
        mask = self.extract_cranial_mask(img_arr)

        # 3. Crop tightly around cranial contour & pad to square
        cropped_img, cropped_mask = self.crop_and_pad_cranial(img_arr, mask)

        # 4. Apply CLAHE local contrast harmonization
        clahe_img = self.apply_clahe(cropped_img, cropped_mask)

        # 5. Foreground brain Z-score normalization
        harmonized_2d = self.apply_foreground_zscore(clahe_img, cropped_mask)

        # 6. Build 3-channel PyTorch Tensor [1, 3, H, W]
        # Replicate grayscale to 3 channels (RGB) for ResNet / DenseNet
        img_3ch = np.stack([harmonized_2d] * 3, axis=0)  # [3, H, W]

        # Standard ImageNet normalization for pretrained convolutional backbones
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        norm_3ch = (img_3ch - mean) / std

        tensor_3ch = torch.from_numpy(norm_3ch).unsqueeze(0).float()
        if device is not None:
            tensor_3ch = tensor_3ch.to(device)

        return harmonized_2d, tensor_3ch


# ─── Top-Level Convenience Functions ───

_DEFAULT_HARMONIZER = MRIHarmonizer(target_size=224)


def harmonize_2d_slice(
    image_input: Image.Image | bytes | np.ndarray,
    target_size: int = 224,
    device: torch.device | None = None,
) -> tuple[np.ndarray, torch.Tensor]:
    """Harmonize a 2D Brain MRI slice to eliminate shortcut learning.

    Args:
        image_input: Input image (PIL, bytes, or numpy).
        target_size: Output resolution (default 224).
        device: Target compute device for PyTorch tensor.

    Returns:
        Tuple of (harmonized_2d_array, tensor_3ch).
    """
    if target_size == 224:
        return _DEFAULT_HARMONIZER.harmonize(image_input, device=device)
    harmonizer = MRIHarmonizer(target_size=target_size)
    return harmonizer.harmonize(image_input, device=device)


def harmonize_3d_volume(
    volume: np.ndarray,
    spatial_size: tuple[int, int, int] = (96, 96, 96),
) -> np.ndarray:
    """Harmonize a 3D NIfTI Brain MRI volume.

    Applies 3D background thresholding, foreground brain Z-score normalization,
    and outlier intensity clipping.

    Args:
        volume: 3D float numpy array [D, H, W].
        spatial_size: Output 3D dimensions.

    Returns:
        3D float32 normalized volume [D, H, W] in [0.0, 1.0].
    """
    vol_float = volume.astype(np.float32)

    # 1. 3D Otsu / foreground mask
    min_thresh = max(float(np.percentile(vol_float, 25)), 0.05 * vol_float.max())
    mask = vol_float > min_thresh

    tissue_voxels = vol_float[mask]
    if len(tissue_voxels) == 0 or tissue_voxels.std() < 1e-5:
        max_val = max(vol_float.max(), 1.0)
        return (vol_float / max_val).astype(np.float32)

    # 2. Foreground 3D Z-Score
    mu = float(tissue_voxels.mean())
    sigma = float(tissue_voxels.std())

    z_vol = np.zeros_like(vol_float)
    z_vol[mask] = (vol_float[mask] - mu) / (sigma + 1e-7)

    # 3. Clip standard deviations to [-3.0, 3.0] and map to [0, 1]
    z_clipped = np.clip(z_vol, -3.0, 3.0)
    norm_vol = np.zeros_like(vol_float)
    norm_vol[mask] = (z_clipped[mask] + 3.0) / 6.0

    return norm_vol.astype(np.float32)
