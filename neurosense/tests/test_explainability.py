"""NeuroSense Unit Tests — Explainability Module.

Tests for GradCAM++ and SHAP explainability:
- GradCAM3D heatmap generation and shape
- SHAP attribution output format
- Visualization utilities
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestGradCAM3D:
    """Test suite for GradCAM++ 3D heatmap generator."""

    def test_heatmap_shape(self):
        """GradCAM++ produces heatmap matching MRI spatial dims."""
        from neurosense.explainability.gradcam import GradCAM3D
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(use_mri=True, use_clinical=False)
        explainer = GradCAM3D(model)

        mri = torch.randn(1, 1, 96, 96, 96)

        heatmap = explainer.generate(mri=mri)

        assert heatmap.shape == (96, 96, 96), (
            f"Expected (96, 96, 96), got {heatmap.shape}"
        )

        explainer.cleanup()

    def test_heatmap_normalized(self):
        """Heatmap values are normalised to [0, 1]."""
        from neurosense.explainability.gradcam import GradCAM3D
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(use_mri=True, use_clinical=False)
        explainer = GradCAM3D(model)

        mri = torch.randn(1, 1, 96, 96, 96)
        heatmap = explainer.generate(mri=mri, normalize=True)

        assert heatmap.min() >= 0.0 - 1e-6
        assert heatmap.max() <= 1.0 + 1e-6

        explainer.cleanup()

    def test_per_class_heatmaps(self):
        """generate_per_class() returns heatmaps for all 3 classes."""
        from neurosense.explainability.gradcam import GradCAM3D
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(use_mri=True, use_clinical=False)
        explainer = GradCAM3D(model)

        mri = torch.randn(1, 1, 96, 96, 96)
        heatmaps = explainer.generate_per_class(mri=mri, num_classes=3)

        assert len(heatmaps) == 3
        for cls_idx, hm in heatmaps.items():
            assert hm.shape == (96, 96, 96)

        explainer.cleanup()

    def test_target_layer_auto_detection(self):
        """Auto-detects layer4 as target layer."""
        from neurosense.explainability.gradcam import GradCAM3D
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(use_mri=True, use_clinical=False)
        explainer = GradCAM3D(model)

        # Should have auto-selected layer4
        assert explainer.target_layer is not None

        explainer.cleanup()


class TestSHAPExplainer:
    """Test suite for SHAP clinical feature attribution."""

    def test_attribution_output_format(self):
        """SHAP explains returns list of attribution dicts."""
        from neurosense.explainability.shap_analysis import SHAPExplainer
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(use_mri=False, use_clinical=True)
        background = torch.randn(10, 5)

        explainer = SHAPExplainer(
            model=model,
            background_data=background,
            output_type="probs",
        )

        sample = torch.randn(1, 5)
        attrs = explainer.explain(sample)

        assert isinstance(attrs, list)
        assert len(attrs) == 5  # 5 clinical features

        for attr in attrs:
            assert "name" in attr
            assert "impact" in attr
            assert "value" in attr
            assert "abs_impact" in attr

    def test_attributions_sorted_by_importance(self):
        """Attributions are sorted by absolute impact (descending)."""
        from neurosense.explainability.shap_analysis import SHAPExplainer
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(use_mri=False, use_clinical=True)
        background = torch.randn(10, 5)

        explainer = SHAPExplainer(
            model=model,
            background_data=background,
        )

        sample = torch.randn(1, 5)
        attrs = explainer.explain(sample)

        abs_impacts = [a["abs_impact"] for a in attrs]
        assert abs_impacts == sorted(abs_impacts, reverse=True)

    def test_feature_names(self):
        """Attribution feature names match expected clinical features."""
        from neurosense.explainability.shap_analysis import (
            CLINICAL_FEATURE_NAMES,
            SHAPExplainer,
        )
        from neurosense.models.classifier import NeuroSenseModel

        model = NeuroSenseModel(use_mri=False, use_clinical=True)
        background = torch.randn(10, 5)

        explainer = SHAPExplainer(
            model=model,
            background_data=background,
        )

        sample = torch.randn(1, 5)
        attrs = explainer.explain(sample)

        attr_names = {a["name"] for a in attrs}
        expected = set(CLINICAL_FEATURE_NAMES)
        assert attr_names == expected


class TestVisualisation:
    """Test suite for visualisation utilities."""

    def test_overlay_single_slice_shape(self):
        """overlay_single_slice returns RGB image [H, W, 3]."""
        from neurosense.explainability.visualise import (
            overlay_single_slice,
        )

        mri_slice = np.random.rand(96, 96).astype(np.float32)
        heat_slice = np.random.rand(96, 96).astype(np.float32)

        rgb = overlay_single_slice(mri_slice, heat_slice)

        assert rgb.shape == (96, 96, 3)
        assert rgb.dtype == np.float32

    def test_overlay_gradcam_on_slices_returns_figure(self):
        """overlay_gradcam_on_slices returns matplotlib Figure."""
        import matplotlib.pyplot as plt

        from neurosense.explainability.visualise import (
            overlay_gradcam_on_slices,
        )

        mri = np.random.rand(96, 96, 96).astype(np.float32)
        heatmap = np.random.rand(96, 96, 96).astype(np.float32)

        fig = overlay_gradcam_on_slices(
            mri, heatmap, num_slices=4
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_shap_waterfall_returns_figure(self):
        """plot_shap_waterfall returns matplotlib Figure."""
        import matplotlib.pyplot as plt

        from neurosense.explainability.visualise import (
            plot_shap_waterfall,
        )

        attrs = [
            {"name": "cag_repeat", "description": "CAG Repeat",
             "value": 44.0, "impact": 0.32},
            {"name": "uhdrs_motor", "description": "UHDRS Motor",
             "value": 18.0, "impact": -0.15},
        ]

        fig = plot_shap_waterfall(attrs)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
