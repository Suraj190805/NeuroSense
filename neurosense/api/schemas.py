"""NeuroSense API — Pydantic Request/Response Schemas.

Defines the data models for the FastAPI prediction endpoint
(PRD Section 5.1). All schemas use Pydantic v2 with strict
validation including field constraints, custom validators,
and comprehensive documentation.

Schemas:
    ClinicalInput: Clinical feature data from the request
    PredictionResponse: Full prediction result with XAI outputs
    HealthResponse: Service health status
    VersionResponse: API version information
    ErrorResponse: Structured error messages
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ClinicalInput(BaseModel):
    """Clinical data for HD prediction.

    Represents the clinical measurements provided alongside
    MRI data for multi-modal prediction (PRD Section 5.1).

    All fields have clinical validation ranges based on
    established assessment scales.
    """

    cag_repeat: float = Field(
        ...,
        ge=36.0,
        le=120.0,
        description=(
            "CAG trinucleotide repeat count. "
            "Values ≥36 indicate HD gene mutation carrier status. "
            "Normal range: 10–35 (not accepted). "
            "HD range: 36–120."
        ),
        json_schema_extra={"example": 44.0},
    )

    uhdrs_motor: float = Field(
        ...,
        ge=0.0,
        le=124.0,
        description=(
            "UHDRS Total Motor Score (TMS). "
            "Assesses motor function across 15 items. "
            "0 = no motor abnormalities, 124 = maximum impairment."
        ),
        json_schema_extra={"example": 18.0},
    )

    uhdrs_cognitive: float = Field(
        ...,
        ge=0.0,
        description=(
            "UHDRS Cognitive Assessment composite score. "
            "Includes Symbol Digit Modalities, Stroop, "
            "and verbal fluency tests. Higher = better function."
        ),
        json_schema_extra={"example": 142.0},
    )

    tfc_score: float = Field(
        default=13.0,
        ge=0.0,
        le=13.0,
        description=(
            "Total Functional Capacity score. "
            "13 = fully functional, 0 = total disability. "
            "Used for HD staging (Shoulson-Fahn)."
        ),
        json_schema_extra={"example": 9.0},
    )

    age: float = Field(
        ...,
        ge=18.0,
        le=90.0,
        description=(
            "Patient age in years. "
            "Must be between 18 and 90."
        ),
        json_schema_extra={"example": 42.0},
    )

    @field_validator("cag_repeat")
    @classmethod
    def validate_cag_range(cls, v: float) -> float:
        """Ensure CAG repeat is in HD mutation range."""
        if v < 36:
            raise ValueError(
                f"CAG repeat {v} is below HD threshold (36). "
                "Only HD mutation carriers (CAG ≥ 36) are accepted."
            )
        return v

    def to_tensor_list(self) -> list[float]:
        """Convert to ordered feature list for model input.

        Returns feature values in the order expected by the
        clinical encoder: [CAG, UHDRS motor, UHDRS cognitive,
        TFC, age].

        Returns:
            List of 5 float values.
        """
        return [
            self.cag_repeat,
            self.uhdrs_motor,
            self.uhdrs_cognitive,
            self.tfc_score,
            self.age,
        ]


class SHAPFeature(BaseModel):
    """Single feature attribution from SHAP analysis."""

    name: str = Field(
        ...,
        description="Feature name (e.g., 'cag_repeat')",
    )
    value: float = Field(
        ...,
        description="Input feature value for this prediction",
    )
    impact: float = Field(
        ...,
        description=(
            "SHAP value indicating feature contribution. "
            "Positive = increases predicted risk, "
            "negative = decreases predicted risk."
        ),
    )


class StageProbabilities(BaseModel):
    """Per-class classification probabilities."""

    pre_manifest: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of pre-manifest HD stage",
    )
    early: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of early HD stage",
    )
    advanced: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of advanced HD stage",
    )


class PredictionResponse(BaseModel):
    """Full HD prediction response.

    Contains staging classification, progression forecasts,
    and explainability outputs (GradCAM++ heatmap URL and
    SHAP feature attributions).
    """

    # ─── Classification ───
    stage: str = Field(
        ...,
        description=(
            "Predicted HD stage: 'pre_manifest', 'early', or 'advanced'"
        ),
        json_schema_extra={"example": "early"},
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence (max probability)",
        json_schema_extra={"example": 0.87},
    )

    stage_probabilities: StageProbabilities = Field(
        ...,
        description="Per-class probabilities for all HD stages",
    )

    # ─── Progression ───
    progression_12mo: float = Field(
        ...,
        description=(
            "Predicted 12-month change in UHDRS Total Motor Score. "
            "Positive values indicate expected worsening."
        ),
        json_schema_extra={"example": 4.2},
    )

    progression_24mo: float = Field(
        ...,
        description="Predicted 24-month change in UHDRS TMS",
        json_schema_extra={"example": 9.1},
    )

    risk_category: str = Field(
        ...,
        description=(
            "Risk category based on 12-month progression: "
            "'low' (Δ < 3), 'medium' (3 ≤ Δ < 8), 'high' (Δ ≥ 8)"
        ),
        json_schema_extra={"example": "medium"},
    )

    # ─── Explainability ───
    gradcam_url: str | None = Field(
        default=None,
        description=(
            "URL to download the GradCAM++ heatmap overlay image. "
            "Available when MRI is provided."
        ),
        json_schema_extra={"example": "/static/heatmaps/abc123.png"},
    )

    shap_features: list[SHAPFeature] = Field(
        default_factory=list,
        description="SHAP feature attributions sorted by importance",
    )

    # ─── Metadata ───
    processing_time_s: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in seconds",
        json_schema_extra={"example": 12.4},
    )

    request_id: str | None = Field(
        default=None,
        description="Unique request identifier for tracking",
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the prediction",
    )


class HealthResponse(BaseModel):
    """Service health status response."""

    status: str = Field(
        ...,
        description="Service status: 'healthy' or 'degraded'",
        json_schema_extra={"example": "healthy"},
    )

    gpu_available: bool = Field(
        ...,
        description="Whether CUDA GPU is available",
    )

    gpu_name: str | None = Field(
        default=None,
        description="GPU device name if available",
    )

    model_loaded: bool = Field(
        ...,
        description="Whether the prediction model is loaded",
    )

    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="Service uptime in seconds",
    )

    version: str = Field(
        ...,
        description="API version string",
    )


class VersionResponse(BaseModel):
    """API version information."""

    api_version: str = Field(
        ...,
        description="Semantic version of the API",
        json_schema_extra={"example": "1.0.0"},
    )

    model_version: str = Field(
        ...,
        description="Model checkpoint identifier",
    )

    checkpoint_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of the model checkpoint",
    )

    python_version: str = Field(
        ...,
        description="Python runtime version",
    )

    torch_version: str = Field(
        ...,
        description="PyTorch version",
    )


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str = Field(
        ...,
        description="Error type identifier",
    )

    message: str = Field(
        ...,
        description="Human-readable error description",
    )

    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error context",
    )

    request_id: str | None = Field(
        default=None,
        description="Request ID for error tracking",
    )
