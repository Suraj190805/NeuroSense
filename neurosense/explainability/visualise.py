"""NeuroSense Visualisation Utilities.

Publication-quality heatmap overlay generation for GradCAM++ 3D
activation maps on axial MRI slices (PRD Section 4.3.3).

Produces:
- Axial slice montages with GradCAM++ overlay
- SHAP waterfall/bar plots for feature attribution
- Combined report figures with both modalities
- Individual slice renderings for API serving

Colour Mapping:
    - MRI background: greyscale
    - GradCAM++ overlay: 'jet' colourmap (blue→red)
    - Overlay alpha: 0.4 (configurable)

Usage:
    from neurosense.explainability.visualise import (
        overlay_gradcam_on_slices,
        plot_shap_waterfall,
        create_report_figure,
    )

    # Single-slice overlay
    fig = overlay_gradcam_on_slices(mri_volume, heatmap)
    fig.savefig("gradcam_overlay.png", dpi=300)

    # SHAP waterfall
    fig = plot_shap_waterfall(attributions)
    fig.savefig("shap_waterfall.png", dpi=300)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for server environments
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ─── Default styling ───
FIGURE_DPI = 300
OVERLAY_ALPHA = 0.4
COLORMAP = "jet"
MRI_COLORMAP = "gray"
BACKGROUND_COLOR = "#0a0a0a"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR = "#00b4d8"

# Stage colour coding
STAGE_COLORS = {
    0: "#4CAF50",  # Pre-manifest — green
    1: "#FF9800",  # Early — orange
    2: "#F44336",  # Advanced — red
}
STAGE_NAMES_VIZ = {
    0: "Pre-manifest",
    1: "Early HD",
    2: "Advanced HD",
}


def overlay_gradcam_on_slices(
    mri_volume: np.ndarray,
    heatmap: np.ndarray,
    num_slices: int = 9,
    alpha: float = OVERLAY_ALPHA,
    colormap: str = COLORMAP,
    title: str = "GradCAM++ Activation Map",
    figsize: tuple[float, float] | None = None,
    slice_axis: int = 2,
) -> plt.Figure:
    """Create an axial slice montage with GradCAM++ overlay.

    Selects evenly-spaced axial (or other axis) slices through
    the MRI volume and overlays the corresponding GradCAM++
    heatmap with a configurable colour map and transparency.

    Args:
        mri_volume: 3D MRI volume ``[D, H, W]``. Values should
            be in a displayable range (0–1 or raw intensities).
        heatmap: GradCAM++ heatmap ``[D, H, W]``, normalised
            to [0, 1]. Higher values = more important regions.
        num_slices: Number of slices to display (default: 9).
            Arranged in a 3×3 grid.
        alpha: Heatmap overlay transparency (default: 0.4).
        colormap: Matplotlib colourmap for heatmap (default: 'jet').
        title: Figure title string.
        figsize: Figure size in inches. If None, auto-computed.
        slice_axis: Axis along which to slice (default: 2 for
            axial). 0=sagittal, 1=coronal, 2=axial.

    Returns:
        matplotlib Figure object with the montage.

    Example:
        >>> fig = overlay_gradcam_on_slices(mri, heatmap, num_slices=9)
        >>> fig.savefig("gradcam.png", dpi=300, bbox_inches="tight")
    """
    # Normalise MRI to [0, 1] if needed
    mri_display = _normalize_volume(mri_volume)

    # Select slice indices
    n_total = mri_volume.shape[slice_axis]
    margin = max(1, n_total // 10)  # Skip edge slices
    slice_indices = np.linspace(
        margin, n_total - margin - 1, num_slices, dtype=int
    )

    # Grid layout
    ncols = min(num_slices, 3)
    nrows = (num_slices + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows + 0.8)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        facecolor=BACKGROUND_COLOR,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    fig.suptitle(
        title,
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for idx, (ax, slice_idx) in enumerate(
        zip(axes.flat, slice_indices)
    ):
        # Extract slices
        mri_slice = _take_slice(mri_display, slice_axis, slice_idx)
        heat_slice = _take_slice(heatmap, slice_axis, slice_idx)

        # Display MRI
        ax.imshow(
            mri_slice.T if slice_axis == 0 else mri_slice,
            cmap=MRI_COLORMAP,
            aspect="equal",
        )

        # Overlay heatmap
        heat_masked = np.ma.masked_where(heat_slice < 0.05, heat_slice)
        ax.imshow(
            heat_masked.T if slice_axis == 0 else heat_masked,
            cmap=colormap,
            alpha=alpha,
            vmin=0,
            vmax=1,
            aspect="equal",
        )

        # Slice label
        axis_names = ["Sagittal", "Coronal", "Axial"]
        ax.set_title(
            f"{axis_names[slice_axis]} #{slice_idx}",
            color=TEXT_COLOR,
            fontsize=10,
        )
        ax.axis("off")
        ax.set_facecolor(BACKGROUND_COLOR)

    # Hide unused axes
    for ax in axes.flat[len(slice_indices):]:
        ax.set_visible(False)

    # Colourbar
    cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.60])
    sm = plt.cm.ScalarMappable(
        cmap=colormap, norm=plt.Normalize(0, 1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0.05, 0.25, 0.50, 0.75, 1.0])
    cbar.set_ticklabels(
        ["0% (Preserved)", "25%", "50% (Moderate)", "75%", "100% (Critical)"],
        color=TEXT_COLOR,
        fontsize=8,
    )
    cbar.set_label("Neural Activation & Pathological Weight", color="#00b4d8", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(colors=TEXT_COLOR)

    # Subtitle / footer explaining colors and brain parts
    fig.text(
        0.46, 0.012,
        "● Red (80-100%): Critical Atrophy Focus  |  ● Yellow (50-80%): Moderate  |  ● Blue (5-50%): Preserved Tissue Baseline\n"
        "Key Anatomical HD Targets: Striatum (Caudate Nucleus & Putamen) • Lateral Ventricles • Frontal Cortex",
        ha="center", va="center", color=TEXT_COLOR, fontsize=8.5, linespacing=1.35
    )

    fig.subplots_adjust(
        left=0.02, right=0.90, top=0.92, bottom=0.04,
        hspace=0.15, wspace=0.05,
    )

    return fig


def overlay_single_slice(
    mri_slice: np.ndarray,
    heat_slice: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
    colormap: str = COLORMAP,
) -> np.ndarray:
    """Create a single-slice overlay as an RGB image array.

    Useful for API responses and web display where a raw
    numpy array or PIL image is needed instead of a matplotlib
    figure.

    Args:
        mri_slice: 2D MRI slice ``[H, W]``.
        heat_slice: 2D heatmap slice ``[H, W]`` in [0, 1].
        alpha: Overlay transparency.
        colormap: Matplotlib colourmap name.

    Returns:
        RGB image array ``[H, W, 3]`` with values in [0, 1].
    """
    # Normalise MRI
    mri_norm = _normalize_2d(mri_slice)

    # Convert MRI to RGB (greyscale → 3-channel)
    mri_rgb = np.stack([mri_norm] * 3, axis=-1)

    # Apply colourmap to heatmap
    cmap = plt.get_cmap(colormap)
    heat_rgb = cmap(heat_slice)[:, :, :3]  # Drop alpha channel

    # Blend: where heatmap > threshold, blend with alpha
    mask = heat_slice > 0.05
    blended = mri_rgb.copy()
    blended[mask] = (
        (1 - alpha) * mri_rgb[mask] + alpha * heat_rgb[mask]
    )

    return blended.astype(np.float32)


def plot_shap_waterfall(
    attributions: list[dict[str, Any]],
    title: str = "Clinical Feature Attribution (SHAP)",
    figsize: tuple[float, float] = (8, 5),
    max_features: int = 10,
) -> plt.Figure:
    """Create a SHAP waterfall-style horizontal bar chart.

    Displays each clinical feature's contribution to the
    prediction as a signed horizontal bar, coloured by
    direction (red=positive, blue=negative contribution).

    Args:
        attributions: List of attribution dicts from
            ``SHAPExplainer.explain()``. Each dict must contain
            'name', 'impact', and 'value'.
        title: Figure title.
        figsize: Figure size in inches.
        max_features: Maximum number of features to display.

    Returns:
        matplotlib Figure with the waterfall chart.

    Example:
        >>> attrs = [
        ...     {"name": "cag_repeat", "value": 44.0, "impact": 0.32},
        ...     {"name": "uhdrs_motor", "value": 18.0, "impact": 0.28},
        ... ]
        >>> fig = plot_shap_waterfall(attrs)
    """
    # Sort by absolute impact and limit
    attrs = sorted(
        attributions[:max_features],
        key=lambda x: abs(x.get("impact", 0)),
    )

    fig, ax = plt.subplots(figsize=figsize, facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    names = []
    impacts = []
    values = []

    for attr in attrs:
        name = attr.get("description", attr.get("name", ""))
        impact = attr.get("impact", 0)
        value = attr.get("value", 0)

        # Create label with value
        label = f"{name}\n(= {value})"
        names.append(label)
        impacts.append(impact)
        values.append(value)

    # Colours: red for positive, blue for negative
    colors = [
        "#ef5350" if imp > 0 else "#42a5f5"
        for imp in impacts
    ]

    y_pos = np.arange(len(names))

    bars = ax.barh(
        y_pos, impacts,
        color=colors,
        edgecolor="none",
        height=0.6,
        alpha=0.85,
    )

    # Value labels on bars
    for bar, impact in zip(bars, impacts):
        width = bar.get_width()
        label_x = width + (0.005 if width >= 0 else -0.005)
        ha = "left" if width >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{impact:+.4f}",
            va="center",
            ha=ha,
            color=TEXT_COLOR,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color=TEXT_COLOR, fontsize=10)
    ax.set_xlabel("SHAP Value (Impact on Prediction)", color=TEXT_COLOR)

    ax.set_title(
        title,
        color=TEXT_COLOR,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Styling
    ax.axvline(x=0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.spines["left"].set_color(TEXT_COLOR)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ef5350", label="Increases risk"),
        Patch(facecolor="#42a5f5", label="Decreases risk"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        framealpha=0.3,
        facecolor=BACKGROUND_COLOR,
        edgecolor=TEXT_COLOR,
        labelcolor=TEXT_COLOR,
    )

    fig.tight_layout()
    return fig


def create_report_figure(
    mri_volume: np.ndarray,
    heatmap: np.ndarray,
    attributions: list[dict[str, Any]],
    prediction: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (18, 10),
) -> plt.Figure:
    """Create a combined clinical report figure.

    Generates a publication-quality figure combining:
    - Left panel: GradCAM++ heatmap overlay (3×3 axial slices)
    - Right top: SHAP feature attribution bar chart
    - Right bottom: Prediction summary with confidence

    Args:
        mri_volume: 3D MRI volume ``[D, H, W]``.
        heatmap: GradCAM++ heatmap ``[D, H, W]`` in [0, 1].
        attributions: SHAP attributions from ``SHAPExplainer``.
        prediction: Optional prediction dict with keys:
            - stage: Predicted stage string
            - confidence: Prediction confidence
            - stage_probabilities: Per-class probabilities
            - progression_12mo: 12-month prediction
            - progression_24mo: 24-month prediction
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure with the combined report.
    """
    fig = plt.figure(figsize=figsize, facecolor=BACKGROUND_COLOR)

    # Layout: GradCAM on left (3×3), SHAP + summary on right
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.3, 1],
        height_ratios=[1.5, 1],
        hspace=0.3,
        wspace=0.25,
    )

    # ─── Left panel: GradCAM++ slices (3×3 grid) ───
    gs_left = gs[:, 0].subgridspec(3, 3, hspace=0.1, wspace=0.05)

    mri_display = _normalize_volume(mri_volume)
    n_slices = 9
    n_total = mri_volume.shape[2]
    margin = max(1, n_total // 10)
    slice_indices = np.linspace(
        margin, n_total - margin - 1, n_slices, dtype=int
    )

    for idx, slice_idx in enumerate(slice_indices):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs_left[row, col])

        mri_slice = mri_display[:, :, slice_idx]
        heat_slice = heatmap[:, :, slice_idx]

        ax.imshow(mri_slice, cmap=MRI_COLORMAP, aspect="equal")
        heat_masked = np.ma.masked_where(heat_slice < 0.05, heat_slice)
        ax.imshow(
            heat_masked, cmap=COLORMAP, alpha=OVERLAY_ALPHA,
            vmin=0, vmax=1, aspect="equal",
        )
        ax.set_title(
            f"Axial #{slice_idx}", color=TEXT_COLOR, fontsize=8
        )
        ax.axis("off")
        ax.set_facecolor(BACKGROUND_COLOR)

    # GradCAM title
    fig.text(
        0.28, 0.96,
        "GradCAM++ Spatial Activation",
        color=ACCENT_COLOR,
        fontsize=14,
        fontweight="bold",
        ha="center",
    )

    # ─── Right top: SHAP bar chart ───
    ax_shap = fig.add_subplot(gs[0, 1])
    ax_shap.set_facecolor(BACKGROUND_COLOR)

    attrs_sorted = sorted(
        attributions[:5],
        key=lambda x: abs(x.get("impact", 0)),
    )

    names = [a.get("name", "") for a in attrs_sorted]
    impacts = [a.get("impact", 0) for a in attrs_sorted]
    colors = ["#ef5350" if i > 0 else "#42a5f5" for i in impacts]

    y_pos = np.arange(len(names))
    ax_shap.barh(y_pos, impacts, color=colors, height=0.5, alpha=0.85)

    ax_shap.set_yticks(y_pos)
    ax_shap.set_yticklabels(names, color=TEXT_COLOR, fontsize=9)
    ax_shap.set_title(
        "Feature Attribution (SHAP)",
        color=ACCENT_COLOR,
        fontsize=12,
        fontweight="bold",
    )
    ax_shap.axvline(x=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.5)
    ax_shap.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax_shap.spines["top"].set_visible(False)
    ax_shap.spines["right"].set_visible(False)
    ax_shap.spines["bottom"].set_color(TEXT_COLOR)
    ax_shap.spines["left"].set_color(TEXT_COLOR)

    # ─── Right bottom: Prediction summary ───
    ax_pred = fig.add_subplot(gs[1, 1])
    ax_pred.set_facecolor(BACKGROUND_COLOR)
    ax_pred.axis("off")

    if prediction is not None:
        stage = prediction.get("stage", "Unknown")
        confidence = prediction.get("confidence", 0)
        probs = prediction.get("stage_probabilities", {})
        p12 = prediction.get("progression_12mo", 0)
        p24 = prediction.get("progression_24mo", 0)
        risk = prediction.get("risk_category", "unknown")

        summary_lines = [
            ("Predicted Stage:", f"  {stage.upper()}", ACCENT_COLOR),
            ("Confidence:", f"  {confidence:.1%}", TEXT_COLOR),
            ("", "", TEXT_COLOR),
            ("Stage Probabilities:", "", TEXT_COLOR),
        ]

        for stage_name, prob in probs.items():
            summary_lines.append(
                (f"  {stage_name}:", f"  {prob:.1%}", TEXT_COLOR)
            )

        summary_lines.extend([
            ("", "", TEXT_COLOR),
            ("12-Month Progression Δ:", f"  {p12:+.1f}", TEXT_COLOR),
            ("24-Month Progression Δ:", f"  {p24:+.1f}", TEXT_COLOR),
            ("Risk Category:", f"  {risk.upper()}", _risk_color(risk)),
        ])

        y_start = 0.95
        for i, (label, value, color) in enumerate(summary_lines):
            y = y_start - i * 0.08
            if label:
                ax_pred.text(
                    0.05, y, label,
                    transform=ax_pred.transAxes,
                    color=TEXT_COLOR,
                    fontsize=10,
                    fontweight="bold",
                    va="top",
                )
            if value:
                ax_pred.text(
                    0.55, y, value,
                    transform=ax_pred.transAxes,
                    color=color,
                    fontsize=10,
                    va="top",
                )

        ax_pred.set_title(
            "Prediction Summary",
            color=ACCENT_COLOR,
            fontsize=12,
            fontweight="bold",
        )
    else:
        ax_pred.text(
            0.5, 0.5,
            "No prediction data",
            transform=ax_pred.transAxes,
            color=TEXT_COLOR,
            fontsize=12,
            ha="center",
            va="center",
        )

    # Report title
    fig.suptitle(
        "NeuroSense — HD Analysis Report",
        color=TEXT_COLOR,
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )

    return fig


def save_heatmap_image(
    mri_volume: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    slice_index: int | None = None,
    dpi: int = FIGURE_DPI,
    **kwargs: Any,
) -> Path:
    """Save a GradCAM++ overlay to an image file.

    Convenience function for saving individual overlays as
    PNG files for API serving.

    Args:
        mri_volume: 3D MRI volume ``[D, H, W]``.
        heatmap: 3D heatmap ``[D, H, W]``.
        output_path: File path for the saved image.
        slice_index: Specific axial slice. If None, creates
            a 3×3 montage.
        dpi: Resolution in dots per inch.
        **kwargs: Additional arguments passed to
            ``overlay_gradcam_on_slices``.

    Returns:
        Path to the saved image file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if slice_index is not None:
        # Single slice
        mri_slice = mri_volume[:, :, slice_index]
        heat_slice = heatmap[:, :, slice_index]
        rgb = overlay_single_slice(mri_slice, heat_slice)

        fig, ax = plt.subplots(
            figsize=(6, 6), facecolor=BACKGROUND_COLOR
        )
        ax.imshow(rgb)
        ax.set_title(
            f"Axial Slice #{slice_index}",
            color=TEXT_COLOR,
            fontsize=12,
        )
        ax.axis("off")
        fig.tight_layout()
    else:
        # Montage
        fig = overlay_gradcam_on_slices(
            mri_volume, heatmap, **kwargs
        )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=BACKGROUND_COLOR,
        edgecolor="none",
    )
    plt.close(fig)

    logger.info("Heatmap saved to %s", output_path)
    return output_path


def save_shap_image(
    attributions: list[dict[str, Any]],
    output_path: str | Path,
    dpi: int = FIGURE_DPI,
    **kwargs: Any,
) -> Path:
    """Save SHAP waterfall chart to an image file.

    Args:
        attributions: Attribution dicts from ``SHAPExplainer``.
        output_path: File path for the saved image.
        dpi: Resolution in dots per inch.
        **kwargs: Additional arguments for ``plot_shap_waterfall``.

    Returns:
        Path to the saved image file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_shap_waterfall(attributions, **kwargs)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=BACKGROUND_COLOR,
        edgecolor="none",
    )
    plt.close(fig)

    logger.info("SHAP chart saved to %s", output_path)
    return output_path


# ─── Private helpers ───


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalise a 3D volume to [0, 1] range."""
    vmin = volume.min()
    vmax = volume.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32)


def _normalize_2d(image: np.ndarray) -> np.ndarray:
    """Normalise a 2D image to [0, 1] range."""
    vmin = image.min()
    vmax = image.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - vmin) / (vmax - vmin)).astype(np.float32)


def _take_slice(
    volume: np.ndarray,
    axis: int,
    index: int,
) -> np.ndarray:
    """Extract a 2D slice from a 3D volume along given axis."""
    if axis == 0:
        return volume[index, :, :]
    elif axis == 1:
        return volume[:, index, :]
    else:
        return volume[:, :, index]


def _risk_color(risk: str) -> str:
    """Get display colour for risk category."""
    risk_colors = {
        "low": "#4CAF50",
        "medium": "#FF9800",
        "high": "#F44336",
    }
    return risk_colors.get(risk.lower(), TEXT_COLOR)


def analyze_brain_regions(heatmap: np.ndarray) -> dict[str, Any]:
    """Analyze GradCAM++ heatmap activation across anatomical brain regions.

    Quantifies neural attention across key cranial structures implicated in
    Huntington's Disease (striatum, caudate nucleus, putamen, lateral ventricles,
    frontal cortex, temporal/insular cortex, parieto-occipital region).

    Args:
        heatmap: 2D or 3D numpy array of activation intensities in [0, 1].

    Returns:
        Dictionary containing primary targeted region, regional scores,
        peak coordinates, and semantic color meanings.
    """
    if heatmap.ndim == 3:
        # Take max projection or mid slice along shortest axis
        if heatmap.shape[0] <= heatmap.shape[1] and heatmap.shape[0] <= heatmap.shape[2]:
            heatmap_2d = np.max(heatmap, axis=0)
        else:
            mid = heatmap.shape[2] // 2
            heatmap_2d = heatmap[:, :, mid]
    else:
        heatmap_2d = heatmap

    h, w = heatmap_2d.shape[:2]
    y_grid, x_grid = np.ogrid[:h, :w]
    ny = y_grid / float(max(h, 1))
    nx = x_grid / float(max(w, 1))

    regions_def = [
        {
            "id": "caudate_bilateral",
            "name": "Bilateral Caudate Nuclei",
            "short_name": "Caudate Nuclei",
            "hemisphere": "Bilateral",
            "mask": (((nx >= 0.32) & (nx <= 0.44)) | ((nx >= 0.56) & (nx <= 0.68))) & ((ny >= 0.30) & (ny <= 0.48)),
            "clinical": "Hallmark HD neurodegeneration. Loss of GABAergic medium spiny neurons directly correlates with CAG repeat expansion.",
        },
        {
            "id": "caudate_left",
            "name": "Left Caudate Nucleus",
            "short_name": "Left Caudate",
            "hemisphere": "Left",
            "mask": ((nx >= 0.32) & (nx <= 0.44)) & ((ny >= 0.30) & (ny <= 0.48)),
            "clinical": "Subcortical striatal volume loss; critical driver of cognitive and motor staging.",
        },
        {
            "id": "caudate_right",
            "name": "Right Caudate Nucleus",
            "short_name": "Right Caudate",
            "hemisphere": "Right",
            "mask": ((nx >= 0.56) & (nx <= 0.68)) & ((ny >= 0.30) & (ny <= 0.48)),
            "clinical": "Hallmark early HD atrophy; bilateral degeneration drives motor incoordination and chorea.",
        },
        {
            "id": "putamen_bilateral",
            "name": "Striatum / Putamen",
            "short_name": "Putamen",
            "hemisphere": "Bilateral",
            "mask": (((nx >= 0.20) & (nx <= 0.34)) | ((nx >= 0.66) & (nx <= 0.80))) & ((ny >= 0.40) & (ny <= 0.62)),
            "clinical": "Basal ganglia motor circuit component; progressive degeneration triggers choreic movements and dystonia.",
        },
        {
            "id": "lateral_ventricles",
            "name": "Lateral Ventricles (Frontal Horns)",
            "short_name": "Lateral Ventricles",
            "hemisphere": "Bilateral",
            "mask": ((nx >= 0.42) & (nx <= 0.58)) & ((ny >= 0.30) & (ny <= 0.60)),
            "clinical": "Ex-vacuo ventriculomegaly (enlargement) secondary to progressive shrinkage of surrounding striatal tissue.",
        },
        {
            "id": "frontal_cortex",
            "name": "Frontal Cortex & Prefrontal Lobe",
            "short_name": "Frontal Cortex",
            "hemisphere": "Bilateral",
            "mask": ((ny < 0.30) & (nx >= 0.20) & (nx <= 0.80)),
            "clinical": "Cortical thinning driving executive dysfunction, apathy, and cognitive slowing.",
        },
        {
            "id": "temporal_cortex",
            "name": "Temporal / Insular Cortex",
            "short_name": "Temporal Cortex",
            "hemisphere": "Bilateral",
            "mask": (((nx < 0.20) | (nx > 0.80)) & (ny >= 0.30) & (ny <= 0.72)),
            "clinical": "Cortical thinning associated with psychiatric and affective changes in HD.",
        },
        {
            "id": "parieto_occipital",
            "name": "Parieto-Occipital Cortex",
            "short_name": "Occipital Cortex",
            "hemisphere": "Bilateral",
            "mask": ((ny > 0.68) & (nx >= 0.25) & (nx <= 0.75)),
            "clinical": "Secondary posterior cortical changes affecting visuospatial integration in advancing stages.",
        },
    ]

    peak_val = float(np.max(heatmap_2d)) if heatmap_2d.size > 0 else 0.0
    peak_idx = np.unravel_index(np.argmax(heatmap_2d), heatmap_2d.shape) if heatmap_2d.size > 0 else (0, 0)
    peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])

    scored_regions = []
    for r in regions_def:
        m = r["mask"]
        if np.any(m):
            sub = heatmap_2d[m]
            max_act = float(np.max(sub))
            mean_act = float(np.mean(sub))
            score = 0.65 * max_act + 0.35 * mean_act
        else:
            max_act, mean_act, score = 0.0, 0.0, 0.0

        pct = int(round(score * 100))
        severity = "Critical" if pct >= 75 else ("Moderate" if pct >= 45 else ("Mild" if pct >= 20 else "Baseline"))
        scored_regions.append({
            "id": r["id"],
            "name": r["name"],
            "short_name": r["short_name"],
            "hemisphere": r["hemisphere"],
            "score": round(score, 4),
            "max_activation": round(max_act, 4),
            "percentage": pct,
            "severity": severity,
            "clinical": r["clinical"],
        })

    scored_regions.sort(key=lambda x: x["score"], reverse=True)
    primary = scored_regions[0] if scored_regions else {
        "name": "Striatum / Caudate Nucleus",
        "short_name": "Striatum",
        "hemisphere": "Bilateral",
        "score": 0.5,
        "percentage": 50,
        "severity": "Moderate",
        "clinical": "Primary site of medium spiny neuron loss in Huntington's Disease.",
    }

    secondary = [r["name"] for r in scored_regions[1:4] if r["percentage"] >= 30]

    color_meanings = [
        {
            "color": "Red",
            "tag": "RED (80-100%)",
            "hex": "#ff4444",
            "level": "Critical Attention",
            "meaning": "Severe localized atrophy driving stage prediction.",
        },
        {
            "color": "Yellow",
            "tag": "YELLOW (50-80%)",
            "hex": "#ffcc00",
            "level": "Moderate Activation",
            "meaning": "Intermediate neurodegeneration / tissue stress.",
        },
        {
            "color": "Blue",
            "tag": "BLUE (5-50%)",
            "hex": "#3388ff",
            "level": "Preserved Baseline",
            "meaning": "Low/normal involvement without major atrophy.",
        },
        {
            "color": "Grayscale",
            "tag": "GRAYSCALE",
            "hex": "#aaaaaa",
            "level": "Structural MRI",
            "meaning": "T1-weighted brain anatomy (Parenchyma, CSF ventricles & sulci).",
        },
    ]

    return {
        "primary_region": primary["name"],
        "short_name": primary.get("short_name", primary["name"]),
        "primary_clinical_role": primary["clinical"],
        "primary_score": primary["score"],
        "primary_percentage": primary["percentage"],
        "primary_severity": primary["severity"],
        "peak_coordinates": [peak_x, peak_y],
        "normalized_peak": [round(peak_x / float(max(w, 1)), 4), round(peak_y / float(max(h, 1)), 4)],
        "detected_regions": scored_regions[:5],
        "secondary_summary": ", ".join(secondary) if secondary else "None (Focal)",
        "color_meanings": color_meanings,
    }


def save_2d_heatmap_overlay(
    image: Any,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.45,
    colormap: str = "jet",
    title: str = "GradCAM++ Brain Heatmap",
    dpi: int = 150,
    region_analysis: dict[str, Any] | None = None,
) -> Path:
    """Save a 2D MRI slice with GradCAM++ heatmap overlay, color legend, and brain region labels.

    Args:
        image: PIL Image or 2D numpy array [H, W].
        heatmap: 2D heatmap numpy array [H, W] normalized to [0, 1].
        output_path: Destination image path.
        alpha: Overlay blending transparency.
        colormap: Matplotlib colormap name.
        title: Image title.
        dpi: Resolution.
        region_analysis: Optional precomputed brain region analysis dict.

    Returns:
        Path to saved PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("L"))
    elif isinstance(image, np.ndarray):
        img_arr = image if image.ndim == 2 else image[:, :, 0]
    else:
        img_arr = np.array(image)

    h_img, w_img = img_arr.shape[:2]

    # Resize heatmap if dimensions differ from MRI
    if heatmap.shape[:2] != (h_img, w_img):
        from scipy.ndimage import zoom
        heatmap_resized = zoom(
            heatmap,
            (h_img / float(max(heatmap.shape[0], 1)), w_img / float(max(heatmap.shape[1], 1))),
            order=1,
        )
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    else:
        heatmap_resized = heatmap

    # Perform or use region analysis
    analysis = region_analysis or analyze_brain_regions(heatmap_resized)
    primary_name = analysis.get("primary_region", "Striatum (Caudate / Putamen)")
    primary_role = analysis.get("primary_clinical_role", "Primary site of neurodegeneration in HD.")
    primary_score = analysis.get("primary_score", 0.8)
    peak_x, peak_y = analysis.get("peak_coordinates", [w_img // 2, h_img // 2])
    peak_pct = analysis.get("primary_percentage", int(primary_score * 100))

    # Clamp peak coordinates inside image
    peak_x = max(2, min(w_img - 3, peak_x))
    peak_y = max(2, min(h_img - 3, peak_y))

    # Create publication figure with colorbar and informational legend
    fig = plt.figure(figsize=(9, 10.4), facecolor=BACKGROUND_COLOR)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.045], height_ratios=[1, 0.24], wspace=0.04, hspace=0.08)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_footer = fig.add_subplot(gs[1, :])

    # Display MRI and overlay
    ax_main.imshow(img_arr, cmap=MRI_COLORMAP)
    heat_masked = np.ma.masked_where(heatmap_resized < 0.05, heatmap_resized)
    im = ax_main.imshow(heat_masked, cmap=colormap, alpha=alpha, vmin=0.0, vmax=1.0)
    ax_main.axis("off")
    ax_main.set_facecolor(BACKGROUND_COLOR)

    # Highlight hotspot reticle
    reticle_r = max(h_img, w_img) * 0.035
    circle = patches.Circle(
        (peak_x, peak_y),
        radius=reticle_r,
        linewidth=2,
        edgecolor="#ffffff",
        facecolor="none",
        linestyle="--",
    )
    ax_main.add_patch(circle)
    ax_main.plot(peak_x, peak_y, "+", color="#ffffff", markersize=10, markeredgewidth=2)

    # Callout annotation arrow
    callout_x = peak_x + (w_img * 0.16 if peak_x < w_img * 0.6 else -w_img * 0.28)
    callout_y = peak_y - h_img * 0.12 if peak_y > h_img * 0.25 else peak_y + h_img * 0.14
    rad_dir = -0.2 if peak_x < w_img * 0.6 else 0.2

    ax_main.annotate(
        f"{primary_name}\n(Peak: {peak_pct}%)",
        xy=(peak_x, peak_y),
        xytext=(callout_x, callout_y),
        color="#ffffff",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#12151f", edgecolor="#00b4d8", alpha=0.92, lw=1.2),
        arrowprops=dict(arrowstyle="->", connectionstyle=f"arc3,rad={rad_dir}", color="#00b4d8", lw=1.5),
    )

    ax_main.set_title(
        f"{title}\nTargeted Anatomy: {primary_name}",
        color=TEXT_COLOR,
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Colorbar
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_ticks([0.05, 0.25, 0.50, 0.75, 1.0])
    cbar.set_ticklabels(
        ["0% (Preserved)", "25% (Mild)", "50% (Moderate)", "75% (Elevated)", "100% (Critical)"],
        color="#dddddd",
        fontsize=8.5,
    )
    cbar.set_label("Neural Activation & Pathological Weight", color="#00b4d8", fontsize=9.5, fontweight="bold", labelpad=10)
    ax_cbar.tick_params(colors="#dddddd", length=3)

    # Footer Card explaining colors and brain anatomy
    ax_footer.axis("off")
    ax_footer.set_facecolor(BACKGROUND_COLOR)

    footer_box = patches.FancyBboxPatch(
        (0.005, 0.02),
        0.99,
        0.96,
        boxstyle="round,pad=0.02",
        facecolor="#12151f",
        edgecolor="#232a3b",
        linewidth=1.2,
    )
    ax_footer.add_patch(footer_box)

    # Section 1: Color Representation
    ax_footer.text(
        0.025,
        0.88,
        "WHAT THE COLORS REPRESENT IN THIS HEATMAP:",
        transform=ax_footer.transAxes,
        color="#00b4d8",
        fontsize=8.5,
        fontweight="bold",
    )

    color_lines = [
        ("● RED (80-100%):", "#ff4444", "Critical Neural Attention — Severe localized atrophy driving the stage prediction."),
        ("● YELLOW (50-80%):", "#ffcc00", "Moderate Activation — Intermediate neurodegeneration / tissue stress."),
        ("● BLUE (5-50%):", "#3388ff", "Baseline Preserved Tissue — Low/normal involvement without major atrophy."),
        ("● GRAYSCALE:", "#aaaaaa", "Structural Brain Anatomy — T1-weighted MRI (Parenchyma, CSF ventricles & sulci)."),
    ]

    y_start = 0.72
    for label, col, desc in color_lines:
        ax_footer.text(0.025, y_start, label, transform=ax_footer.transAxes, color=col, fontsize=7.8, fontweight="bold")
        ax_footer.text(0.210, y_start, desc, transform=ax_footer.transAxes, color="#dddddd", fontsize=7.8)
        y_start -= 0.12

    # Section 2: Part of the Brain Highlighted
    ax_footer.text(
        0.025,
        0.20,
        "PART OF THE BRAIN HIGHLIGHTED:",
        transform=ax_footer.transAxes,
        color="#00b4d8",
        fontsize=8.5,
        fontweight="bold",
    )
    ax_footer.text(
        0.025,
        0.07,
        f"● Primary Target: {primary_name} ({primary_score*100:.0f}% Activation) — {primary_role}",
        transform=ax_footer.transAxes,
        color="#ffffff",
        fontsize=7.8,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=BACKGROUND_COLOR,
        edgecolor="none",
    )
    plt.close(fig)
    logger.info("2D Heatmap overlay saved to %s (Target: %s)", output_path, primary_name)
    return output_path

