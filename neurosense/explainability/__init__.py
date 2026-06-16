"""NeuroSense explainability module.

GradCAM++ for 3D MRI spatial heatmaps, SHAP DeepExplainer
for clinical feature attribution, and visualization utilities
for generating publication-quality overlay images.

Public API:
    GradCAM3D: 3D GradCAM++ heatmap generator for MRI volumes
    SHAPExplainer: SHAP feature attribution for clinical inputs
    overlay_gradcam_on_slices: Axial slice montage with heatmap
    plot_shap_waterfall: SHAP waterfall bar chart
    create_report_figure: Combined clinical report figure
    save_heatmap_image: Save heatmap overlay to file
    save_shap_image: Save SHAP chart to file
"""

from neurosense.explainability.gradcam import GradCAM3D
from neurosense.explainability.shap_analysis import (
    CLINICAL_FEATURE_NAMES,
    SHAPExplainer,
)
from neurosense.explainability.visualise import (
    create_report_figure,
    overlay_gradcam_on_slices,
    overlay_single_slice,
    plot_shap_waterfall,
    save_heatmap_image,
    save_shap_image,
)

__all__ = [
    # GradCAM++
    "GradCAM3D",
    # SHAP
    "SHAPExplainer",
    "CLINICAL_FEATURE_NAMES",
    # Visualization
    "overlay_gradcam_on_slices",
    "overlay_single_slice",
    "plot_shap_waterfall",
    "create_report_figure",
    "save_heatmap_image",
    "save_shap_image",
]
