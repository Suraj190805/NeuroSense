"""NeuroSense — Anatomical Brain MRI Verification & Gatekeeper Validator.

Verifies that uploaded imaging data is a Cranial (Brain) MRI scan rather than
non-brain anatomy (e.g. leg/femur/tibia bone scan, chest radiograph, knee MRI,
abdominal scan, or non-medical photography).
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass

logger = logging.getLogger(__name__)


def validate_brain_mri(
    image_input: Image.Image | bytes | np.ndarray,
) -> dict[str, Any]:
    """Validate whether an image is an authentic Brain MRI scan.

    Evaluates:
    1. Dark Background Air Perimeter (cranial scans are surrounded by empty space).
    2. Central Mass Localization & Cranial Centering.
    3. Bilateral Hemispheric Symmetry (left/right hemisphere structural reflection).
    4. Cranial Convex Bounding Aspect Ratio (head/skull is roughly circular/elliptical, not tubular).
    5. Continuous Boundary Infiltration (limb/bone scans span from frame top-to-bottom).
    6. Medical Intensity Gradient (brain parenchyma vs background air vs fluid).

    Args:
        image_input: PIL Image, raw image bytes, or 2D numpy array.

    Returns:
        Dict with:
        - is_valid: bool (True if image is a valid brain MRI)
        - confidence: float (0.0 to 1.0)
        - anatomy: str ("Brain MRI" | "Non-Brain Scan / Extremity" | "Invalid Image")
        - reasons: list of strings detailing why validation passed or failed.
    """
    try:
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("L")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                # [H, W, C] or [D, H, W]
                if image_input.shape[-1] in (3, 4):
                    image = Image.fromarray(image_input).convert("L")
                else:
                    # Take middle slice of 3D volume
                    mid_idx = image_input.shape[0] // 2
                    slice_data = image_input[mid_idx]
                    slice_norm = ((slice_data - slice_data.min()) / (slice_data.ptp() + 1e-8) * 255).astype(np.uint8)
                    image = Image.fromarray(slice_norm)
            else:
                slice_norm = ((image_input - image_input.min()) / (image_input.ptp() + 1e-8) * 255).astype(np.uint8)
                image = Image.fromarray(slice_norm)
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("L")
        else:
            return {
                "is_valid": False,
                "is_brain_mri": False,
                "confidence": 0.0,
                "anatomy": "Invalid Format",
                "reasons": ["Unsupported image data type."],
            }

        # Resize to standardized 256x256 working resolution
        img_resized = image.resize((256, 256))
        arr = np.array(img_resized, dtype=np.float32)

        # 1. Background perimeter analysis (Outer 16px frame border)
        top_edge = arr[:16, :].mean()
        bottom_edge = arr[-16:, :].mean()
        left_edge = arr[:, :16].mean()
        right_edge = arr[:, -16:].mean()
        perimeter_avg = (top_edge + bottom_edge + left_edge + right_edge) / 4.0

        # 2. Binary segmentation of tissue region
        threshold = max(float(np.percentile(arr, 25)), 20.0)
        binary_mask = arr > threshold

        total_pixels = 256 * 256
        tissue_pixels = int(binary_mask.sum())
        extent_ratio = tissue_pixels / total_pixels

        reasons: list[str] = []
        is_brain = True

        # Check: Completely blank or black image
        if extent_ratio < 0.08:
            return {
                "is_valid": False,
                "is_brain_mri": False,
                "confidence": 0.0,
                "anatomy": "Empty / Black Image",
                "reasons": ["Image contains insufficient anatomical tissue (< 8% scan area)."],
            }

        # Check: Solid white / uniform noise
        if extent_ratio > 0.92 and perimeter_avg > 80:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "anatomy": "Non-Medical / Solid Frame",
                "reasons": ["Image lacks the empty background boundary typical of cranial MRI."],
            }

        # 3. Center of mass deviation
        cy, cx = center_of_mass(binary_mask)
        center_dev = np.sqrt(((cy / 256.0) - 0.5) ** 2 + ((cx / 256.0) - 0.5) ** 2)

        # 4. Bilateral symmetry along vertical sagittal midline
        left_half = arr[:, :128]
        right_half_flipped = np.fliplr(arr[:, 128:])
        denom = np.sqrt((left_half ** 2).sum() * (right_half_flipped ** 2).sum()) + 1e-8
        symmetry_corr = float((left_half * right_half_flipped).sum() / denom)

        # 5. Bounding box & Aspect Ratio
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        if rows.any() and cols.any():
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            width = xmax - xmin + 1
            height = ymax - ymin + 1
            aspect_ratio = float(width) / float(height)
        else:
            aspect_ratio = 1.0

        # 6. Continuous vertical boundary infiltration (characteristic of long bones / legs / spine)
        touches_top_and_bottom = (top_edge > 35.0 and bottom_edge > 35.0)

        # ─── Evaluation Rules ───
        if touches_top_and_bottom and aspect_ratio < 0.70:
            is_brain = False
            reasons.append(
                "Continuous vertical tissue detected extending across top and bottom borders (typical of limb/leg/bone scans)."
            )

        if aspect_ratio < 0.60 or aspect_ratio > 1.65:
            is_brain = False
            reasons.append(
                f"Elongated aspect ratio ({aspect_ratio:.2f}) is non-cranial. Brain MRI scans have a roughly convex, elliptical cranial contour."
            )

        if perimeter_avg > 65.0:
            is_brain = False
            reasons.append(
                "Image borders contain high-intensity tissue (lacks the dark background surrounding a cranial MRI)."
            )

        if symmetry_corr < 0.55:
            is_brain = False
            reasons.append(
                f"Low bilateral hemispheric symmetry ({symmetry_corr:.2f}). Cranial axial MRI exhibits strong left-right hemisphere symmetry."
            )

        if center_dev > 0.22:
            is_brain = False
            reasons.append(
                f"Anatomical mass is significantly off-center (deviation: {center_dev:.2f})."
            )

        # Compute confidence score
        if is_brain:
            score = 0.50
            if symmetry_corr >= 0.75:
                score += 0.25
            elif symmetry_corr >= 0.65:
                score += 0.15

            if perimeter_avg < 25.0:
                score += 0.15
            elif perimeter_avg < 45.0:
                score += 0.08

            if 0.75 <= aspect_ratio <= 1.30:
                score += 0.10

            confidence = min(0.99, max(0.65, score))
            anatomy = "Brain MRI"
        else:
            confidence = max(0.05, min(0.40, 1.0 - symmetry_corr))
            anatomy = "Non-Brain Scan / Extremity"

        return {
            "is_brain_mri": is_brain,
            "confidence": round(float(confidence), 2),
            "anatomy": anatomy,
            "symmetry_score": round(float(symmetry_corr), 2),
            "aspect_ratio": round(float(aspect_ratio), 2),
            "center_deviation": round(float(center_dev), 2),
            "perimeter_luminance": round(float(perimeter_avg), 1),
            "reasons": reasons,
        }

    except Exception as e:
        logger.error("Anatomical validation failed: %s", e)
        return {
            "is_brain_mri": True,  # Non-blocking fallback
            "confidence": 0.5,
            "anatomy": "Unverified",
            "reasons": [f"Verification bypass: {e}"],
        }
