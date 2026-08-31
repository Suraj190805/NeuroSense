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

from pydantic import BaseModel, Field, field_validator, model_validator


class ClinicalInput(BaseModel):
    """Clinical & digital assessment data for HD prediction.

    Represents patient biomarkers and self-administered digital test
    scores provided alongside MRI data for multi-modal prediction.
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

    motor_score: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description=(
            "Digital Motor Assessment Score (0–100%). "
            "Evaluates reaction speed, finger tapping, and coordination tracking. "
            "100% = normal / excellent motor speed, 0% = severe motor impairment."
        ),
        json_schema_extra={"example": 82.0},
    )

    memory_score: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description=(
            "Digital Memory & Cognitive Assessment Score (0–100%). "
            "Evaluates word recall, sequence working memory, and visual change detection. "
            "100% = normal / excellent memory function, 0% = severe cognitive decline."
        ),
        json_schema_extra={"example": 88.0},
    )

    functional_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description=(
            "Daily Functional Independence Score (0–100%). "
            "100% = fully independent in daily activities, 0% = total dependency."
        ),
        json_schema_extra={"example": 90.0},
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

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_fields(cls, data: Any) -> Any:
        """Handle legacy UHDRS and TFC field names for backwards compatibility."""
        if isinstance(data, dict):
            # Map legacy uhdrs_motor (0-124, lower=better) -> motor_score (0-100%, higher=better)
            if "motor_score" not in data and "uhdrs_motor" in data:
                uhdrs_m = float(data["uhdrs_motor"])
                data["motor_score"] = max(0.0, min(100.0, 100.0 - (uhdrs_m / 124.0 * 100.0)))

            # Map legacy uhdrs_cognitive (composite >=0) -> memory_score (0-100%)
            if "memory_score" not in data and "uhdrs_cognitive" in data:
                uhdrs_c = float(data["uhdrs_cognitive"])
                data["memory_score"] = max(0.0, min(100.0, (uhdrs_c / 200.0) * 100.0))

            # Map legacy tfc_score (0-13) -> functional_score (0-100%)
            if "functional_score" not in data and "tfc_score" in data:
                tfc = float(data["tfc_score"])
                data["functional_score"] = max(0.0, min(100.0, (tfc / 13.0) * 100.0))

        return data

    @property
    def uhdrs_motor(self) -> float:
        """Legacy accessor for UHDRS Motor Score estimate."""
        return max(0.0, min(124.0, (100.0 - self.motor_score) / 100.0 * 124.0))

    @property
    def uhdrs_cognitive(self) -> float:
        """Legacy accessor for UHDRS Cognitive Score estimate."""
        return self.memory_score * 2.0

    @property
    def tfc_score(self) -> float:
        """Legacy accessor for TFC score estimate."""
        return max(0.0, min(13.0, (self.functional_score / 100.0) * 13.0))

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
        clinical encoder: [CAG, motor_score, memory_score,
        functional_score, age].

        Returns:
            List of 5 float values.
        """
        return [
            self.cag_repeat,
            self.motor_score,
            self.memory_score,
            self.functional_score,
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
