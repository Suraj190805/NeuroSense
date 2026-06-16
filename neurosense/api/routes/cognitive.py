"""NeuroSense API — Cognitive Assessment Endpoints.

FastAPI router for the digital cognitive test battery.

Endpoints:
    POST /cognitive/session/start
        Start a new cognitive session with version rotation.

    POST /cognitive/session/submit
        Submit results for a single test within a session.

    GET  /cognitive/patient/{patient_id}/history
        Get full longitudinal history with z-scores and trends.

    POST /cognitive/session/complete
        Complete a session: validate, score, build feature vector.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from neurosense.cognitive.scoring import ClinicalCategory
from neurosense.cognitive.session import session_manager
from neurosense.cognitive.tests import TEST_ORDER, TestName

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cognitive", tags=["Cognitive Assessment"])


# ═════════════════════════════════════════════════════════════════
#  Request / Response Schemas
# ═════════════════════════════════════════════════════════════════


class StartSessionRequest(BaseModel):
    """Request to start a new cognitive session."""

    patient_id: str = Field(
        ...,
        description="Unique patient identifier",
        json_schema_extra={"example": "patient-001"},
    )
    patient_age: float = Field(
        ...,
        ge=18.0,
        le=90.0,
        description="Patient age in years",
        json_schema_extra={"example": 42.0},
    )


class StartSessionResponse(BaseModel):
    """Response after starting a cognitive session."""

    session_id: str = Field(..., description="Unique session identifier")
    test_sequence: list[str] = Field(
        ..., description="Ordered list of test names"
    )
    stimulus_versions: dict[str, int] = Field(
        ..., description="Parallel form version per test (0–5)"
    )
    session_number: int = Field(..., description="1-based session number")
    is_baseline: bool = Field(
        ..., description="Whether this is a baseline session"
    )
    days_since_baseline: float = Field(
        ..., description="Days since first session"
    )
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Pre-start validation warnings (e.g., too_soon)",
    )


class SubmitResultRequest(BaseModel):
    """Request to submit results for a single test."""

    session_id: str = Field(
        ..., description="Session identifier from /session/start"
    )
    test_name: str = Field(
        ...,
        description="Test name: sdmt, nback, verbal_fluency, trail_making, delayed_recall",
    )
    raw_results: dict[str, Any] = Field(
        ...,
        description=(
            "Raw test data including responses, timing, and stimuli. "
            "Structure varies by test type."
        ),
    )


class SubmitResultResponse(BaseModel):
    """Response after submitting a test result."""

    test_name: str
    raw_score: float
    raw_score_secondary: float | None = None
    z_score: float | None = None
    category: str | None = None
    practice_accuracy: float
    validity_status: str = Field(
        default="valid",
        description="'valid' or description of the issue",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompleteSessionRequest(BaseModel):
    """Request to complete and finalize a session."""

    session_id: str = Field(
        ..., description="Session identifier to complete"
    )


class CompleteSessionResponse(BaseModel):
    """Full session summary after completion."""

    session_id: str
    patient_id: str
    is_valid: bool
    is_baseline: bool
    session_number: int
    overall_category: str
    validation_flags: list[str]
    feature_vector: list[float]
    domains: dict[str, dict[str, Any]]
    versions_used: dict[str, int]


class PatientHistoryResponse(BaseModel):
    """Response for patient history query."""

    patient_id: str
    total_sessions: int
    sessions: list[dict[str, Any]]


# ═════════════════════════════════════════════════════════════════
#  Endpoints
# ═════════════════════════════════════════════════════════════════


@router.post(
    "/session/start",
    response_model=StartSessionResponse,
    summary="Start Cognitive Session",
    description=(
        "Start a new cognitive assessment session. Returns the test "
        "sequence, assigned parallel form versions, and session metadata."
    ),
)
async def start_session(
    request: StartSessionRequest,
) -> StartSessionResponse:
    """Start a new cognitive assessment session.

    Creates a session with rotated parallel form versions,
    determines if baseline, and checks for too-soon scheduling.
    """
    try:
        session = session_manager.start_session(
            patient_id=request.patient_id,
            patient_age=request.patient_age,
        )

        return StartSessionResponse(
            session_id=session.session_id,
            test_sequence=[tn.value for tn in TEST_ORDER],
            stimulus_versions=session.versions,
            session_number=session.session_number,
            is_baseline=session.is_baseline,
            days_since_baseline=session.days_since_baseline,
            validation_warnings=session.validation_flags,
        )

    except Exception as e:
        logger.error("Failed to start session: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/session/submit",
    response_model=SubmitResultResponse,
    summary="Submit Test Result",
    description=(
        "Submit results for a single cognitive test within a session. "
        "Returns the raw score, practice accuracy, and metadata."
    ),
)
async def submit_result(
    request: SubmitResultRequest,
) -> SubmitResultResponse:
    """Submit results for a single test.

    Scores the test, stores the result, and checks practice
    accuracy against the 60% threshold.
    """
    try:
        result = session_manager.submit_result(
            session_id=request.session_id,
            test_name=request.test_name,
            raw_results=request.raw_results,
        )

        # Compute z-score if we have the session's age
        z_score = None
        category = None
        session = session_manager.get_session(request.session_id)

        if session is not None:
            from neurosense.cognitive.scoring import (
                classify_domain,
                compute_z_score,
            )

            try:
                tn = TestName(request.test_name)
                z_score = compute_z_score(
                    tn,
                    result["raw_score"],
                    session.patient_age,
                )
                category = classify_domain(z_score).value
            except (ValueError, KeyError) as e:
                logger.warning("Z-score computation failed: %s", e)

        # Validity status
        validity = "valid"
        if result["practice_accuracy"] < 0.60:
            validity = "low_practice_accuracy"

        return SubmitResultResponse(
            test_name=result["test_name"],
            raw_score=result["raw_score"],
            raw_score_secondary=result.get("raw_score_secondary"),
            z_score=z_score,
            category=category,
            practice_accuracy=result["practice_accuracy"],
            validity_status=validity,
            metadata=result.get("metadata", {}),
        )

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to submit result: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/patient/{patient_id}/history",
    response_model=PatientHistoryResponse,
    summary="Patient History",
    description=(
        "Get full longitudinal cognitive assessment history for a patient. "
        "Includes z-scores, validity flags, and trend direction per domain."
    ),
)
async def get_patient_history(
    patient_id: str,
) -> PatientHistoryResponse:
    """Retrieve all cognitive sessions for a patient.

    Returns sessions ordered by date with per-domain z-scores,
    clinical categories, validity flags, and trend analysis.
    """
    try:
        sessions = session_manager.get_patient_history(patient_id)

        return PatientHistoryResponse(
            patient_id=patient_id,
            total_sessions=len(sessions),
            sessions=sessions,
        )

    except Exception as e:
        logger.error("Failed to get history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/session/complete",
    response_model=CompleteSessionResponse,
    summary="Complete Session",
    description=(
        "Finalize a cognitive session: run validation checks, "
        "compute all z-scores, and generate the 9-dimensional "
        "feature vector for the ML pipeline."
    ),
)
async def complete_session(
    request: CompleteSessionRequest,
) -> CompleteSessionResponse:
    """Complete and finalize a cognitive assessment session.

    Validates the session, computes all domain z-scores,
    builds the 9-dimensional feature vector, and stores
    everything for the ML pipeline.
    """
    try:
        summary = session_manager.complete_session(
            session_id=request.session_id,
        )

        # Convert domain scores to serializable dict
        domains: dict[str, dict[str, Any]] = {}
        for test_name, ds in summary.domain_scores.items():
            domains[test_name.value] = {
                "raw_score": ds.raw_score,
                "z_score": ds.z_score,
                "category": ds.category.value,
                "norm_mean": ds.norm_mean,
                "norm_sd": ds.norm_sd,
                "age_band": ds.age_band,
            }

        # Convert versions to string keys
        versions: dict[str, int] = {}
        for tn, v in summary.versions_used.items():
            key = tn.value if hasattr(tn, "value") else str(tn)
            versions[key] = v

        return CompleteSessionResponse(
            session_id=summary.session_id,
            patient_id=summary.patient_id,
            is_valid=summary.is_valid,
            is_baseline=summary.is_baseline,
            session_number=summary.session_number,
            overall_category=summary.overall_category.value,
            validation_flags=summary.validation_flags,
            feature_vector=summary.feature_vector,
            domains=domains,
            versions_used=versions,
        )

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to complete session: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
