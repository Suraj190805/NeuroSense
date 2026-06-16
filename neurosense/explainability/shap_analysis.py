"""NeuroSense SHAP Feature Attribution.

Implements SHAP (SHapley Additive exPlanations) analysis for the
clinical features in the NeuroSense model (PRD Section 4.3.2).
Uses DeepExplainer for neural network attribution to quantify
each clinical feature's contribution to HD predictions.

The analysis produces:
- Per-feature SHAP values for each prediction
- Feature importance rankings
- Directional impact (positive/negative contribution)
- Formatted attribution dictionaries for API response

Clinical Features Analysed:
    0: CAG repeat count (genetic biomarker)
    1: UHDRS motor score (motor function assessment)
    2: UHDRS cognitive score (cognitive function assessment)
    3: TFC score (Total Functional Capacity)
    4: Age (patient age)

Usage:
    from neurosense.explainability.shap_analysis import SHAPExplainer

    explainer = SHAPExplainer(model, background_data)
    attributions = explainer.explain(clinical_features)
    # attributions: list of dicts with name, value, impact
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Clinical feature names matching dataset column order
CLINICAL_FEATURE_NAMES: list[str] = [
    "cag_repeat",
    "uhdrs_motor",
    "uhdrs_cognitive",
    "tfc",
    "age",
]

# Clinical feature descriptions for reporting
CLINICAL_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "cag_repeat": "CAG Repeat Count",
    "uhdrs_motor": "UHDRS Motor Score",
    "uhdrs_cognitive": "UHDRS Cognitive Score",
    "tfc": "Total Functional Capacity (TFC)",
    "age": "Patient Age",
}


class _ClinicalBranchWrapper(nn.Module):
    """Wrapper to isolate the clinical pathway for SHAP analysis.

    SHAP needs a model that takes clinical features as input and
    returns predictions. This wrapper routes clinical features
    through the full model pipeline while keeping MRI features
    fixed at a reference value (or using clinical-only mode).

    Args:
        model: The full NeuroSenseModel.
        fixed_mri: Optional fixed MRI embedding to use. If None,
            uses clinical-only pathway.
        output_type: What to extract — "logits", "probs", or
            "deltas" (default: "probs").
    """

    def __init__(
        self,
        model: nn.Module,
        fixed_mri: torch.Tensor | None = None,
        output_type: str = "probs",
    ) -> None:
        super().__init__()
        self.model = model
        self.fixed_mri = fixed_mri
        self.output_type = output_type

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        """Forward pass with clinical features only.

        Args:
            clinical: Clinical features ``[B, 5]`` or ``[B, T, 5]``.

        Returns:
            Model output based on output_type.
        """
        # Ensure proper shape: [B, 5] → [B, 1, 5]
        if clinical.ndim == 2:
            clinical = clinical.unsqueeze(1)

        lengths = torch.ones(
            clinical.size(0),
            dtype=torch.long,
            device=clinical.device,
        )

        mri = self.fixed_mri
        if mri is not None:
            # Expand MRI to match batch size
            if mri.size(0) != clinical.size(0):
                mri = mri.expand(clinical.size(0), -1, -1, -1, -1)

        outputs = self.model(
            mri=mri,
            clinical=clinical,
            clinical_lengths=lengths,
        )

        return outputs[self.output_type]


class SHAPExplainer:
    """SHAP feature attribution for NeuroSense clinical predictions.

    Computes SHAP values for each clinical feature to explain
    individual predictions. Uses gradient-based attribution
    (integrated gradients approximation) when SHAP's
    DeepExplainer is available, with a gradient-based fallback.

    The explainer quantifies how much each clinical feature
    (CAG repeat, UHDRS motor, UHDRS cognitive, TFC, age)
    contributes to the model's HD staging and progression
    predictions.

    Args:
        model: The NeuroSenseModel to explain.
        background_data: Background dataset for SHAP reference.
            Clinical features tensor ``[N, 5]`` where N ≥ 10.
            Used to establish the baseline expectation.
        fixed_mri: Optional fixed MRI tensor ``[1, 1, 96, 96, 96]``
            to use during clinical SHAP analysis. If None, the
            model must support clinical-only mode.
        output_type: Model output to explain — "probs" for
            classification probabilities or "deltas" for
            progression predictions (default: "probs").
        feature_names: Override default feature names.

    Attributes:
        feature_names: List of clinical feature names.
        shap_values: Cached SHAP values from last explanation.

    Example:
        >>> model = NeuroSenseModel(use_mri=False, use_clinical=True)
        >>> bg = torch.randn(50, 5)  # Background samples
        >>> explainer = SHAPExplainer(model, bg)
        >>> sample = torch.randn(1, 5)
        >>> result = explainer.explain(sample)
        >>> for feat in result:
        ...     print(f"{feat['name']}: impact={feat['impact']:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        fixed_mri: torch.Tensor | None = None,
        output_type: str = "probs",
        feature_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.output_type = output_type
        self.feature_names = feature_names or CLINICAL_FEATURE_NAMES.copy()
        self.shap_values: np.ndarray | None = None

        # Wrap model for clinical-only SHAP
        self._wrapper = _ClinicalBranchWrapper(
            model=model,
            fixed_mri=fixed_mri,
            output_type=output_type,
        )

        # Prepare background data
        if background_data.ndim == 3:
            # [N, T, 5] → [N, 5] (take last visit)
            background_data = background_data[:, -1, :]
        self._background = background_data.detach()

        # Try to initialise SHAP DeepExplainer
        self._shap_explainer = None
        self._use_gradient_fallback = False

        try:
            import shap
            self._shap = shap

            # Create DeepExplainer
            model.eval()
            self._shap_explainer = shap.DeepExplainer(
                self._wrapper,
                self._background,
            )
            logger.info(
                "SHAP DeepExplainer initialised with %d background samples",
                len(background_data),
            )
        except Exception as e:
            logger.warning(
                "SHAP DeepExplainer failed, using gradient fallback: %s", e
            )
            self._use_gradient_fallback = True

    @torch.no_grad()
    def explain(
        self,
        clinical_features: torch.Tensor,
        target_class: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compute SHAP attributions for clinical features.

        Generates per-feature importance values showing how each
        clinical measurement contributes to the model's prediction.

        Args:
            clinical_features: Clinical input ``[1, 5]`` or
                ``[1, T, 5]``. Only first sample is explained.
            target_class: Class to explain (for classification).
                If None, uses the predicted class.

        Returns:
            List of attribution dicts, sorted by absolute impact
            (descending). Each dict contains:
            - name: Feature name (e.g., "cag_repeat")
            - description: Human-readable name
            - value: Input feature value
            - impact: SHAP value (signed importance)
            - abs_impact: Absolute importance value
        """
        self.model.eval()

        # Ensure [B, 5] shape
        if clinical_features.ndim == 3:
            clinical_features = clinical_features[:, -1, :]
        if clinical_features.ndim == 1:
            clinical_features = clinical_features.unsqueeze(0)

        # Get prediction for target class determination
        if target_class is None:
            with torch.no_grad():
                wrapper_out = self._wrapper(clinical_features)
                if wrapper_out.shape[-1] > 1:
                    target_class = wrapper_out.argmax(dim=-1)[0].item()
                else:
                    target_class = 0

        # Compute SHAP values
        if self._use_gradient_fallback:
            shap_values = self._gradient_attribution(
                clinical_features, target_class
            )
        else:
            shap_values = self._shap_attribution(
                clinical_features, target_class
            )

        self.shap_values = shap_values

        # Format results
        feature_values = clinical_features[0].cpu().numpy()
        attributions = self._format_attributions(
            shap_values, feature_values
        )

        logger.info(
            "SHAP explanation generated: %d features, target_class=%d",
            len(attributions),
            target_class,
        )

        return attributions

    def _shap_attribution(
        self,
        clinical_features: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """Compute SHAP values using DeepExplainer.

        Args:
            clinical_features: Input features ``[1, 5]``.
            target_class: Target class index.

        Returns:
            SHAP values array ``[5]``.
        """
        shap_values = self._shap_explainer.shap_values(
            clinical_features
        )

        # shap_values may be a list (one per class) or array
        if isinstance(shap_values, list):
            if target_class < len(shap_values):
                values = shap_values[target_class]
            else:
                values = shap_values[0]
        else:
            values = shap_values

        # Take first sample
        if hasattr(values, "numpy"):
            values = values.numpy()
        if isinstance(values, np.ndarray) and values.ndim > 1:
            values = values[0]

        return values

    def _gradient_attribution(
        self,
        clinical_features: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """Compute gradient-based attribution as SHAP fallback.

        Uses integrated gradients approximation: compute gradients
        of the target output with respect to input features,
        weighted by (input - baseline).

        Args:
            clinical_features: Input features ``[1, 5]``.
            target_class: Target class index.

        Returns:
            Attribution values array ``[5]``.
        """
        # Enable gradients for input
        features = clinical_features.clone().detach().requires_grad_(True)

        # Baseline: mean of background data
        baseline = self._background.mean(dim=0, keepdim=True)

        # Integrated gradients with N steps
        n_steps = 50
        attributions = torch.zeros_like(features)

        for step in range(n_steps):
            alpha = step / n_steps
            interpolated = baseline + alpha * (features - baseline)
            interpolated = interpolated.requires_grad_(True)

            output = self._wrapper(interpolated)

            # Select target class score
            if output.shape[-1] > 1:
                target_score = output[0, target_class]
            else:
                target_score = output[0, 0]

            # Backward
            self._wrapper.zero_grad()
            target_score.backward(retain_graph=True)

            if interpolated.grad is not None:
                attributions += interpolated.grad.detach()

        # Scale by (input - baseline) / n_steps
        attributions = attributions * (features - baseline) / n_steps

        return attributions[0].detach().cpu().numpy()

    def _format_attributions(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Format SHAP values into sorted attribution dicts.

        Args:
            shap_values: SHAP values ``[5]``.
            feature_values: Input feature values ``[5]``.

        Returns:
            List of attribution dicts sorted by absolute impact.
        """
        attributions: list[dict[str, Any]] = []

        for i, name in enumerate(self.feature_names):
            impact = float(shap_values[i]) if i < len(shap_values) else 0.0
            value = float(feature_values[i]) if i < len(feature_values) else 0.0

            attributions.append({
                "name": name,
                "description": CLINICAL_FEATURE_DESCRIPTIONS.get(
                    name, name
                ),
                "value": round(value, 2),
                "impact": round(impact, 4),
                "abs_impact": round(abs(impact), 4),
            })

        # Sort by absolute impact (descending)
        attributions.sort(key=lambda x: x["abs_impact"], reverse=True)

        return attributions

    def explain_batch(
        self,
        clinical_batch: torch.Tensor,
        target_classes: list[int] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Compute SHAP attributions for a batch of samples.

        Args:
            clinical_batch: Clinical features ``[B, 5]``.
            target_classes: Optional target classes per sample.

        Returns:
            List of B attribution lists.
        """
        batch_size = clinical_batch.shape[0]
        results: list[list[dict[str, Any]]] = []

        for i in range(batch_size):
            target = (
                target_classes[i]
                if target_classes is not None
                else None
            )
            attrs = self.explain(
                clinical_batch[i : i + 1],
                target_class=target,
            )
            results.append(attrs)

        logger.info(
            "Batch SHAP explanation: %d samples processed", batch_size
        )
        return results

    def get_feature_importance(
        self,
        clinical_batch: torch.Tensor,
    ) -> dict[str, float]:
        """Compute mean absolute SHAP importance across samples.

        Provides a global view of feature importance by averaging
        absolute SHAP values over multiple samples.

        Args:
            clinical_batch: Clinical features ``[B, 5]``.

        Returns:
            Dict mapping feature name → mean absolute importance.
        """
        batch_results = self.explain_batch(clinical_batch)

        # Aggregate absolute impacts
        importance: dict[str, list[float]] = {
            name: [] for name in self.feature_names
        }

        for sample_attrs in batch_results:
            for attr in sample_attrs:
                if attr["name"] in importance:
                    importance[attr["name"]].append(attr["abs_impact"])

        # Compute means
        mean_importance: dict[str, float] = {}
        for name, values in importance.items():
            mean_importance[name] = (
                round(float(np.mean(values)), 4)
                if values
                else 0.0
            )

        # Sort by importance
        mean_importance = dict(
            sorted(
                mean_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        return mean_importance
