"""NeuroSense API — Inference Pipeline Orchestration.

Manages the end-to-end inference workflow from raw input to
structured prediction response (PRD Section 5.2):

1. Load and cache the NeuroSenseModel checkpoint
2. Preprocess uploaded MRI (NIfTI → tensor)
3. Prepare clinical features as tensor
4. Run model forward pass (staging + progression)
5. Generate GradCAM++ heatmap (if MRI provided)
6. Compute SHAP attributions (clinical features)
7. Assemble PredictionResponse

The pipeline is designed for single-request inference with
full explainability outputs. Model loading is lazy and
cached for subsequent requests.

Usage:
    from neurosense.api.inference import InferencePipeline

    pipeline = InferencePipeline(checkpoint_path="checkpoints/best.pt")
    result = pipeline.predict(mri_path, clinical_data)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from neurosense.api.schemas import (
    ClinicalInput,
    PredictionResponse,
    SHAPFeature,
    StageProbabilities,
)

logger = logging.getLogger(__name__)

# Stage name mapping (index → string)
STAGE_NAMES: dict[int, str] = {
    0: "pre_manifest",
    1: "early",
    2: "advanced",
}


class InferencePipeline:
    """End-to-end inference orchestrator for NeuroSense predictions.

    Manages model loading, input preprocessing, prediction,
    and explainability generation. Designed for use by the
    FastAPI backend to serve prediction requests.

    The pipeline supports:
    - Multi-modal prediction (MRI + clinical)
    - MRI-only prediction
    - Clinical-only prediction
    - Optional GradCAM++ heatmap generation
    - Optional SHAP feature attribution

    Args:
        checkpoint_path: Path to the saved model checkpoint.
            If None, creates a model with random weights
            (for development/testing).
        device: Compute device string ('cuda', 'mps', 'cpu').
            If None, auto-detects best available device.
        heatmap_dir: Directory for saving GradCAM++ heatmap images.
        enable_gradcam: Whether to generate GradCAM++ heatmaps
            (default: True).
        enable_shap: Whether to compute SHAP attributions
            (default: True).

    Attributes:
        model: The loaded NeuroSenseModel.
        device: Active compute device.
        is_loaded: Whether the model has been loaded.

    Example:
        >>> pipeline = InferencePipeline(
        ...     checkpoint_path="checkpoints/best.pt"
        ... )
        >>> result = pipeline.predict(
        ...     mri_path="scans/patient001.nii.gz",
        ...     clinical=ClinicalInput(
        ...         cag_repeat=44, uhdrs_motor=18,
        ...         uhdrs_cognitive=142, tfc_score=9, age=42,
        ...     ),
        ... )
        >>> print(result.stage, result.confidence)
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        heatmap_dir: str | Path = "outputs/heatmaps",
        enable_gradcam: bool = True,
        enable_shap: bool = True,
    ) -> None:
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else None
        )
        self.heatmap_dir = Path(heatmap_dir)
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

        self.enable_gradcam = enable_gradcam
        self.enable_shap = enable_shap

        # Device setup
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model: nn.Module | None = None
        self.is_loaded: bool = False
        self._gradcam_explainer = None
        self._shap_explainer = None
        self._checkpoint_hash: str | None = None

        logger.info(
            "InferencePipeline created: device=%s, checkpoint=%s",
            self.device,
            self.checkpoint_path,
        )

    def load_model(self) -> None:
        """Load the NeuroSenseModel from checkpoint.

        Loads model weights, moves to device, sets eval mode,
        and initialises explainability modules.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If checkpoint loading fails.
        """
        from neurosense.models.classifier import NeuroSenseModel

        logger.info("Loading NeuroSenseModel...")

        if self.checkpoint_path and self.checkpoint_path.exists():
            # Load from checkpoint
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            # Extract model config from checkpoint
            model_config = checkpoint.get("model_config", {})
            self.model = NeuroSenseModel(**model_config)

            # Load state dict
            state_dict_key = "model_state_dict"
            if state_dict_key in checkpoint:
                self.model.load_state_dict(checkpoint[state_dict_key])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                # Assume checkpoint IS the state dict
                self.model.load_state_dict(checkpoint)

            # Compute checkpoint hash
            self._checkpoint_hash = self._compute_file_hash(
                self.checkpoint_path
            )

            logger.info(
                "Model loaded from checkpoint: %s (hash=%s)",
                self.checkpoint_path,
                self._checkpoint_hash[:12],
            )
        else:
            # Development mode: random weights
            logger.warning(
                "No checkpoint found. Creating model with random weights."
            )
            self.model = NeuroSenseModel()
            self._checkpoint_hash = "dev-random-weights"

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

        # Initialise explainability modules
        self._init_explainability()

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Model ready: %dM params on %s",
            total_params // 1_000_000,
            self.device,
        )

    def _init_explainability(self) -> None:
        """Initialise GradCAM++ and SHAP explainers."""
        if self.model is None:
            return

        # GradCAM++
        if self.enable_gradcam:
            try:
                from neurosense.explainability.gradcam import GradCAM3D

                self._gradcam_explainer = GradCAM3D(self.model)
                logger.info("GradCAM++ explainer initialised")
            except Exception as e:
                logger.warning("GradCAM++ init failed: %s", e)
                self._gradcam_explainer = None

        # SHAP (initialised lazily with background data)
        self._shap_explainer = None

    @torch.no_grad()
    def predict(
        self,
        mri_path: str | Path | None = None,
        clinical: ClinicalInput | None = None,
        generate_heatmap: bool = True,
        generate_shap: bool = True,
    ) -> PredictionResponse:
        """Run full prediction pipeline.

        Performs model inference with optional explainability
        outputs. At least one of ``mri_path`` or ``clinical``
        must be provided.

        Args:
            mri_path: Path to NIfTI MRI file (.nii or .nii.gz).
            clinical: Clinical feature data.
            generate_heatmap: Whether to produce GradCAM++ output.
            generate_shap: Whether to produce SHAP output.

        Returns:
            PredictionResponse with all prediction fields filled.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If neither MRI nor clinical data provided.
        """
        if not self.is_loaded:
            self.load_model()

        if mri_path is None and clinical is None:
            raise ValueError(
                "At least one of mri_path or clinical must be provided"
            )

        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]

        logger.info(
            "Prediction request %s: mri=%s, clinical=%s",
            request_id,
            mri_path is not None,
            clinical is not None,
        )

        # ─── Prepare inputs ───
        mri_tensor = None
        if mri_path is not None:
            mri_tensor = self._load_mri(mri_path)
            mri_tensor = mri_tensor.to(self.device)

        clinical_tensor = None
        if clinical is not None:
            clinical_tensor = self._prepare_clinical(clinical)
            clinical_tensor = clinical_tensor.to(self.device)

        # ─── Model inference ───
        outputs = self.model(
            mri=mri_tensor,
            clinical=clinical_tensor,
        )

        # ─── Extract predictions ───
        probs = outputs["probs"][0].cpu().numpy()
        stage_idx = int(outputs["stage_pred"][0].cpu().item())
        deltas = outputs["deltas"][0].cpu().numpy()
        risk = outputs["risk"][0]

        stage_name = STAGE_NAMES.get(stage_idx, "unknown")
        confidence = float(probs[stage_idx])

        # ─── Explainability ───
        gradcam_url = None
        shap_features: list[SHAPFeature] = []

        if (
            generate_heatmap
            and self.enable_gradcam
            and self._gradcam_explainer is not None
            and mri_tensor is not None
        ):
            gradcam_url = self._generate_heatmap(
                mri_tensor, clinical_tensor, request_id
            )

        if (
            generate_shap
            and self.enable_shap
            and clinical is not None
        ):
            shap_features = self._generate_shap(
                clinical, clinical_tensor, stage_idx
            )

        # ─── Build response ───
        processing_time = time.time() - start_time

        response = PredictionResponse(
            stage=stage_name,
            confidence=round(confidence, 4),
            stage_probabilities=StageProbabilities(
                pre_manifest=round(float(probs[0]), 4),
                early=round(float(probs[1]), 4),
                advanced=round(float(probs[2]), 4),
            ),
            progression_12mo=round(float(deltas[0]), 2),
            progression_24mo=round(float(deltas[1]), 2),
            risk_category=risk,
            gradcam_url=gradcam_url,
            shap_features=shap_features,
            processing_time_s=round(processing_time, 2),
            request_id=request_id,
        )

        logger.info(
            "Prediction %s completed in %.2fs: stage=%s (%.1f%%)",
            request_id,
            processing_time,
            stage_name,
            confidence * 100,
        )

        return response

    def _load_mri(self, mri_path: str | Path) -> torch.Tensor:
        """Load and preprocess NIfTI MRI file to tensor.

        Applies the preprocessing pipeline defined in
        ``neurosense.data.preprocessing``:
        - Load NIfTI with nibabel
        - Resample to 1mm isotropic
        - Crop/pad to 96×96×96
        - Z-score normalisation
        - Convert to torch tensor

        Args:
            mri_path: Path to .nii or .nii.gz file.

        Returns:
            Tensor ``[1, 1, 96, 96, 96]``.

        Raises:
            FileNotFoundError: If MRI file doesn't exist.
            ValueError: If file format is invalid.
        """
        import nibabel as nib

        mri_path = Path(mri_path)
        if not mri_path.exists():
            raise FileNotFoundError(f"MRI file not found: {mri_path}")

        logger.info("Loading MRI: %s", mri_path)

        # Load NIfTI
        img = nib.load(str(mri_path))
        data = img.get_fdata(dtype=np.float32)

        # Simple preprocessing for inference
        # Full pipeline in data/preprocessing.py
        data = self._preprocess_mri_volume(data)

        # Convert to tensor [1, 1, D, H, W]
        tensor = torch.from_numpy(data).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        logger.info("MRI loaded: shape=%s", tensor.shape)
        return tensor

    def _preprocess_mri_volume(
        self,
        volume: np.ndarray,
        target_shape: tuple[int, int, int] = (96, 96, 96),
    ) -> np.ndarray:
        """Preprocess a raw MRI volume for inference.

        Simplified preprocessing for API inference:
        1. Crop or pad to target shape
        2. Z-score normalisation (brain region)

        Args:
            volume: Raw 3D volume from NIfTI.
            target_shape: Target spatial dimensions.

        Returns:
            Preprocessed volume ``[D, H, W]``.
        """
        # Crop or pad to target shape
        result = np.zeros(target_shape, dtype=np.float32)

        # Compute valid ranges
        for axis in range(3):
            src_size = volume.shape[axis]
            tgt_size = target_shape[axis]

            if src_size >= tgt_size:
                # Crop from centre
                start = (src_size - tgt_size) // 2
                volume = np.take(
                    volume,
                    range(start, start + tgt_size),
                    axis=axis,
                )
            # If smaller, will be zero-padded

        # Copy into result (handles padding)
        slices = tuple(
            slice(0, min(volume.shape[i], target_shape[i]))
            for i in range(3)
        )
        result[slices] = volume[slices]

        # Z-score normalisation (non-zero voxels)
        mask = result > 0
        if mask.any():
            mean = result[mask].mean()
            std = result[mask].std()
            if std > 1e-8:
                result[mask] = (result[mask] - mean) / std

        return result

    def _prepare_clinical(
        self,
        clinical: ClinicalInput,
    ) -> torch.Tensor:
        """Convert ClinicalInput to tensor for model input.

        Args:
            clinical: Validated clinical data.

        Returns:
            Tensor ``[1, 1, 5]`` (batch=1, visits=1, features=5).
        """
        features = clinical.to_tensor_list()
        tensor = torch.tensor(
            [features], dtype=torch.float32
        ).unsqueeze(1)  # [1, 1, 5]

        return tensor

    def _generate_heatmap(
        self,
        mri_tensor: torch.Tensor,
        clinical_tensor: torch.Tensor | None,
        request_id: str,
    ) -> str | None:
        """Generate and save GradCAM++ heatmap.

        Args:
            mri_tensor: MRI input tensor ``[1, 1, D, H, W]``.
            clinical_tensor: Optional clinical tensor.
            request_id: Request ID for filename.

        Returns:
            URL path to the saved heatmap image, or None on failure.
        """
        try:
            from neurosense.explainability.visualise import (
                save_heatmap_image,
            )

            # Generate heatmap
            heatmap = self._gradcam_explainer.generate(
                mri=mri_tensor,
                clinical=clinical_tensor,
            )

            # Save to file
            filename = f"{request_id}.png"
            output_path = self.heatmap_dir / filename

            # Get MRI volume for overlay
            mri_np = mri_tensor[0, 0].cpu().numpy()

            save_heatmap_image(
                mri_volume=mri_np,
                heatmap=heatmap,
                output_path=output_path,
            )

            return f"/static/heatmaps/{filename}"

        except Exception as e:
            logger.error("GradCAM++ generation failed: %s", e)
            return None

    def _generate_shap(
        self,
        clinical: ClinicalInput,
        clinical_tensor: torch.Tensor,
        target_class: int,
    ) -> list[SHAPFeature]:
        """Generate SHAP attributions for clinical features.

        Args:
            clinical: Clinical input data.
            clinical_tensor: Clinical tensor.
            target_class: Predicted class index.

        Returns:
            List of SHAPFeature objects sorted by importance.
        """
        try:
            from neurosense.explainability.shap_analysis import (
                SHAPExplainer,
            )

            # Create SHAP explainer with current sample as background
            # In production, use a representative background dataset
            bg_data = clinical_tensor.squeeze(1)  # [1, 5]

            explainer = SHAPExplainer(
                model=self.model,
                background_data=bg_data,
                output_type="probs",
            )

            attributions = explainer.explain(
                clinical_tensor.squeeze(1),
                target_class=target_class,
            )

            return [
                SHAPFeature(
                    name=attr["name"],
                    value=attr["value"],
                    impact=attr["impact"],
                )
                for attr in attributions
            ]

        except Exception as e:
            logger.error("SHAP attribution failed: %s", e)
            return []

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file.

        Args:
            path: File to hash.

        Returns:
            Hex digest of SHA-256 hash.
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_checkpoint_hash(self) -> str | None:
        """Return the checkpoint file hash."""
        return self._checkpoint_hash

    def get_model_info(self) -> dict[str, Any]:
        """Return model information for health/version endpoints.

        Returns:
            Dict with model metadata.
        """
        info: dict[str, Any] = {
            "is_loaded": self.is_loaded,
            "device": str(self.device),
            "checkpoint_hash": self._checkpoint_hash,
        }

        if self.model is not None:
            info["total_params"] = sum(
                p.numel() for p in self.model.parameters()
            )
            info["gradcam_enabled"] = (
                self._gradcam_explainer is not None
            )
            info["shap_enabled"] = self.enable_shap

        return info
