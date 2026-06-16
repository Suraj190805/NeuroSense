"""NeuroSense API module.

FastAPI backend serving HD prediction endpoints with Pydantic
validation, model inference orchestration, and static file
serving for GradCAM++ heatmap downloads.

Public API:
    app: FastAPI application instance
    InferencePipeline: End-to-end inference orchestrator
    ClinicalInput: Pydantic schema for clinical data
    PredictionResponse: Pydantic schema for prediction output
"""

from neurosense.api.inference import InferencePipeline
from neurosense.api.schemas import (
    ClinicalInput,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    VersionResponse,
)

__all__ = [
    "InferencePipeline",
    "ClinicalInput",
    "PredictionResponse",
    "HealthResponse",
    "VersionResponse",
    "ErrorResponse",
]
