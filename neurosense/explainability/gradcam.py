"""NeuroSense GradCAM++ for 3D MRI Volumes.

Implements GradCAM++ spatial activation maps for the 3D ResNet-50
MRI encoder (PRD Section 4.3.1). Generates heatmaps highlighting
brain regions that most influence the HD staging prediction.

The implementation wraps ``pytorch-grad-cam`` for 3D volumes and
provides additional utilities for:
- Target-layer selection (defaults to ResNet layer4)
- Batch-wise heatmap generation
- Per-class activation extraction
- Volume-level and slice-level outputs

Architecture::

    MRI Volume [B, 1, 96, 96, 96]
           ↓
    NeuroSenseModel (forward pass)
           ↓
    GradCAM++ hooks on layer4
           ↓
    Gradient-weighted activations
           ↓
    3D Heatmap [B, 96, 96, 96]

Usage:
    from neurosense.explainability.gradcam import GradCAM3D

    explainer = GradCAM3D(model)
    heatmap = explainer.generate(mri_volume)
    # heatmap: np.ndarray [96, 96, 96], values in [0, 1]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _ActivationAndGradient:
    """Hook-based extractor for layer activations and gradients.

    Registers forward and backward hooks on a target layer to
    capture the intermediate feature maps and their gradients
    during a forward + backward pass.

    Args:
        model: The neural network model.
        target_layer: The layer to extract activations from.

    Attributes:
        activations: List of captured forward activations.
        gradients: List of captured backward gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.activations: list[torch.Tensor] = []
        self.gradients: list[torch.Tensor] = []

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(
            self._save_activation
        )
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(
        self,
        module: nn.Module,
        input: Any,
        output: torch.Tensor,
    ) -> None:
        """Save forward activation from hook."""
        self.activations.append(output.detach())

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Any,
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Save backward gradient from hook."""
        self.gradients.append(grad_output[0].detach())

    def clear(self) -> None:
        """Clear stored activations and gradients."""
        self.activations.clear()
        self.gradients.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


class GradCAM3D:
    """GradCAM++ for 3D MRI volumes in the NeuroSense pipeline.

    Generates gradient-weighted class activation maps for 3D
    volumetric data processed by the MRI encoder. Uses the
    GradCAM++ formulation which provides better localisation
    than vanilla GradCAM for medical imaging applications.

    GradCAM++ Formulation:
        α_k = Σ(ReLU(∂²Y / ∂A_k²)) / (2·(∂²Y / ∂A_k²) + Σ(A_k · ∂³Y / ∂A_k³))
        L = ReLU(Σ_k(α_k · A_k))

    Where A_k are the feature maps and Y is the target class score.

    Args:
        model: The NeuroSenseModel or MRIEncoder to explain.
        target_layer: Specific layer for activation extraction.
            If None, automatically selects the last convolutional
            layer of the MRI encoder's ResNet-50 backbone (layer4).
        use_gradcam_pp: If True, uses GradCAM++ (default). If
            False, falls back to standard GradCAM.

    Attributes:
        model: Reference to the model being explained.
        target_layer: The convolutional layer being analysed.

    Example:
        >>> from neurosense.models.classifier import NeuroSenseModel
        >>> model = NeuroSenseModel()
        >>> explainer = GradCAM3D(model)
        >>> mri = torch.randn(1, 1, 96, 96, 96)
        >>> heatmap = explainer.generate(mri)
        >>> print(heatmap.shape)  # (96, 96, 96)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        use_gradcam_pp: bool = True,
    ) -> None:
        self.model = model
        self.use_gradcam_pp = use_gradcam_pp

        # Auto-detect target layer
        if target_layer is None:
            target_layer = self._find_target_layer(model)

        self.target_layer = target_layer
        self._hook_handler = _ActivationAndGradient(model, target_layer)

        logger.info(
            "GradCAM3D initialised: target_layer=%s, mode=%s",
            type(target_layer).__name__,
            "GradCAM++" if use_gradcam_pp else "GradCAM",
        )

    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        """Auto-detect the last convolutional layer in the MRI encoder.

        Traverses the model to find the ResNet backbone's layer4,
        which contains the deepest feature maps and provides the
        best spatial resolution for activation mapping.

        Args:
            model: The model to search.

        Returns:
            The target layer module.

        Raises:
            ValueError: If no suitable target layer is found.
        """
        # Try NeuroSenseModel → mri_encoder → backbone → layer4
        if hasattr(model, "mri_encoder") and model.mri_encoder is not None:
            backbone = model.mri_encoder.backbone
            if hasattr(backbone, "layer4"):
                logger.info("Auto-selected target: mri_encoder.backbone.layer4")
                return backbone.layer4

        # Try MRIEncoder → backbone → layer4
        if hasattr(model, "backbone"):
            if hasattr(model.backbone, "layer4"):
                logger.info("Auto-selected target: backbone.layer4")
                return model.backbone.layer4

        # Fallback: find last Conv3d layer
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv3d):
                last_conv = module

        if last_conv is not None:
            logger.info("Auto-selected target: last Conv3d layer")
            return last_conv

        raise ValueError(
            "Could not auto-detect target layer. "
            "Please provide target_layer explicitly."
        )

    @torch.enable_grad()
    def generate(
        self,
        mri: torch.Tensor,
        clinical: torch.Tensor | None = None,
        clinical_lengths: torch.Tensor | None = None,
        target_class: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate GradCAM++ heatmap for an MRI volume.

        Performs a forward + backward pass through the model,
        extracts activation maps and gradients from the target
        layer, and computes the weighted activation map.

        Args:
            mri: MRI volume tensor ``[B, 1, D, H, W]`` or
                ``[1, 1, D, H, W]``. Only the first sample
                in the batch is processed.
            clinical: Optional clinical features ``[B, T, 5]``
                for the full NeuroSenseModel. Not needed for
                MRI-only models.
            clinical_lengths: Optional visit lengths ``[B]``.
            target_class: Class index to generate heatmap for.
                If None, uses the predicted class (argmax).
            normalize: If True, normalise heatmap to [0, 1].

        Returns:
            3D heatmap as numpy array ``[D, H, W]`` with values
            in [0, 1] (when normalised). Higher values indicate
            regions of greater importance for the prediction.
        """
        self.model.eval()
        self._hook_handler.clear()

        # Ensure gradient computation
        mri = mri.detach().requires_grad_(True)

        # Forward pass
        if hasattr(self.model, "mri_encoder"):
            # Full NeuroSenseModel
            outputs = self.model(
                mri=mri,
                clinical=clinical,
                clinical_lengths=clinical_lengths,
            )
            logits = outputs["logits"]
        else:
            # Standalone MRIEncoder + ClassificationHead
            embedding = self.model(mri)
            if hasattr(self.model, "classification_head"):
                logits = self.model.classification_head(embedding)
            else:
                # Assume the output is already logits
                logits = embedding

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=-1)[0].item()

        # Create one-hot target for backward pass
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0

        # Backward pass
        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)

        # Extract activations and gradients
        activations = self._hook_handler.activations[-1]  # [B, C, d, h, w]
        gradients = self._hook_handler.gradients[-1]      # [B, C, d, h, w]

        # Compute heatmap
        if self.use_gradcam_pp:
            heatmap = self._compute_gradcam_pp(activations, gradients)
        else:
            heatmap = self._compute_gradcam(activations, gradients)

        # Take first sample
        heatmap = heatmap[0]  # [d, h, w]

        # Resize to original MRI dimensions
        spatial_dims = mri.shape[2:]  # (D, H, W)
        heatmap = self._resize_heatmap(heatmap, spatial_dims)

        # Convert to numpy
        heatmap = heatmap.cpu().numpy()

        if normalize:
            heatmap = self._normalize(heatmap)

        logger.info(
            "GradCAM++ heatmap generated: shape=%s, target_class=%d, "
            "value_range=[%.4f, %.4f]",
            heatmap.shape,
            target_class,
            heatmap.min(),
            heatmap.max(),
        )

        return heatmap

    def _compute_gradcam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        """Standard GradCAM computation.

        Weights = global average pooling of gradients over spatial dims.
        Heatmap = ReLU(Σ_k(weight_k × activation_k))

        Args:
            activations: Feature maps ``[B, C, d, h, w]``.
            gradients: Gradient maps ``[B, C, d, h, w]``.

        Returns:
            Heatmap tensor ``[B, d, h, w]``.
        """
        # Global average pooling of gradients → channel weights
        # [B, C, d, h, w] → [B, C, 1, 1, 1]
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)

        # Weighted sum of activations
        # [B, C, d, h, w] × [B, C, 1, 1, 1] → [B, d, h, w]
        heatmap = (weights * activations).sum(dim=1)

        # ReLU — only keep positive contributions
        heatmap = F.relu(heatmap)

        return heatmap

    def _compute_gradcam_pp(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        """GradCAM++ computation for better localisation.

        Uses second and third-order gradients for more precise
        spatial weighting, improving localisation of multiple
        occurrences of the target pattern.

        Args:
            activations: Feature maps ``[B, C, d, h, w]``.
            gradients: Gradient maps ``[B, C, d, h, w]``.

        Returns:
            Heatmap tensor ``[B, d, h, w]``.
        """
        # Second-order gradients
        grad_2 = gradients.pow(2)
        # Third-order gradients
        grad_3 = gradients.pow(3)

        # Spatial sum of (activation × third-order gradient)
        # Avoid division by zero with epsilon
        spatial_sum = activations.mul(grad_3).sum(
            dim=(2, 3, 4), keepdim=True
        )

        # Alpha coefficients (GradCAM++ weights)
        denominator = 2.0 * grad_2 + spatial_sum + 1e-8
        alpha = grad_2 / denominator

        # Zero out alpha where gradients are zero
        alpha = alpha * F.relu(gradients)

        # Channel weights: spatial sum of alpha
        weights = alpha.sum(dim=(2, 3, 4), keepdim=True)

        # Weighted combination
        heatmap = (weights * activations).sum(dim=1)

        # ReLU
        heatmap = F.relu(heatmap)

        return heatmap

    def _resize_heatmap(
        self,
        heatmap: torch.Tensor,
        target_size: tuple[int, ...],
    ) -> torch.Tensor:
        """Resize heatmap to match original MRI spatial dimensions.

        Uses trilinear interpolation for smooth upsampling from
        the feature map resolution to the input volume resolution.

        Args:
            heatmap: Heatmap tensor ``[d, h, w]``.
            target_size: Target spatial dimensions ``(D, H, W)``.

        Returns:
            Resized heatmap ``[D, H, W]``.
        """
        # Add batch and channel dims for interpolation
        # [d, h, w] → [1, 1, d, h, w]
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)

        heatmap = F.interpolate(
            heatmap,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )

        # Remove batch and channel dims
        return heatmap.squeeze(0).squeeze(0)

    @staticmethod
    def _normalize(heatmap: np.ndarray) -> np.ndarray:
        """Normalise heatmap to [0, 1] range.

        Args:
            heatmap: Raw heatmap values.

        Returns:
            Normalised heatmap in [0, 1].
        """
        vmin = heatmap.min()
        vmax = heatmap.max()

        if vmax - vmin < 1e-8:
            return np.zeros_like(heatmap)

        return (heatmap - vmin) / (vmax - vmin)

    def generate_per_class(
        self,
        mri: torch.Tensor,
        clinical: torch.Tensor | None = None,
        clinical_lengths: torch.Tensor | None = None,
        num_classes: int = 3,
    ) -> dict[int, np.ndarray]:
        """Generate heatmaps for all HD staging classes.

        Produces separate activation maps for pre-manifest, early,
        and advanced stages, enabling comparison of which regions
        drive each classification decision.

        Args:
            mri: MRI volume ``[1, 1, D, H, W]``.
            clinical: Optional clinical features.
            clinical_lengths: Optional visit lengths.
            num_classes: Number of classes (default: 3).

        Returns:
            Dictionary mapping class index → heatmap array.
        """
        heatmaps: dict[int, np.ndarray] = {}

        for cls_idx in range(num_classes):
            heatmaps[cls_idx] = self.generate(
                mri=mri,
                clinical=clinical,
                clinical_lengths=clinical_lengths,
                target_class=cls_idx,
                normalize=True,
            )

        logger.info(
            "Per-class heatmaps generated for %d classes", num_classes
        )
        return heatmaps

    def generate_batch(
        self,
        mri_batch: torch.Tensor,
        clinical_batch: torch.Tensor | None = None,
        clinical_lengths: torch.Tensor | None = None,
    ) -> list[np.ndarray]:
        """Generate heatmaps for a batch of MRI volumes.

        Processes each volume individually to ensure correct
        gradient computation for each sample.

        Args:
            mri_batch: Batch of MRI volumes ``[B, 1, D, H, W]``.
            clinical_batch: Optional clinical features ``[B, T, 5]``.
            clinical_lengths: Optional visit lengths ``[B]``.

        Returns:
            List of B heatmap arrays, each ``[D, H, W]``.
        """
        batch_size = mri_batch.shape[0]
        heatmaps: list[np.ndarray] = []

        for i in range(batch_size):
            single_mri = mri_batch[i : i + 1]
            single_clinical = (
                clinical_batch[i : i + 1] if clinical_batch is not None
                else None
            )
            single_lengths = (
                clinical_lengths[i : i + 1] if clinical_lengths is not None
                else None
            )

            heatmap = self.generate(
                mri=single_mri,
                clinical=single_clinical,
                clinical_lengths=single_lengths,
            )
            heatmaps.append(heatmap)

        logger.info("Batch heatmaps generated: %d volumes", batch_size)
        return heatmaps

    def cleanup(self) -> None:
        """Remove all hooks and free resources.

        Should be called when the explainer is no longer needed
        to prevent memory leaks from lingering hooks.
        """
        self._hook_handler.remove_hooks()
        self._hook_handler.clear()
        logger.info("GradCAM3D hooks removed")

    def __del__(self) -> None:
        """Cleanup hooks on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass
