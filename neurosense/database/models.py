"""NeuroSense Database — Pydantic Document Models.

Defines the schema for all MongoDB documents stored in the
NeuroSense database. These models serve as both validation
layer and documentation of the database schema.

Collections:
    users                — User profiles (auth + demographics)
    predictions          — HD prediction results
    cognitive_sessions   — Cognitive assessment sessions
    chat_conversations   — AI chatbot conversations
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════
#  User Document
# ═════════════════════════════════════════════════════════════════


class UserDocument(BaseModel):
    """User profile stored in the `users` collection.

    Attributes:
        name: User's full name.
        email: User's email address (unique).
        password: Hashed password.
        role: User role (e.g. 'patient', 'doctor', 'admin').
        age: User's age in years.
        gender: User's gender.
        createdAt: UTC timestamp of account creation.
    """

    name: str = Field(
        ..., description="User's full name"
    )
    email: str = Field(
        ..., description="User's email address (unique)"
    )
    password: str = Field(
        ..., description="Hashed password"
    )
    role: str = Field(
        default="patient",
        description="User role: 'patient', 'doctor', or 'admin'",
    )
    age: float = Field(
        ...,
        ge=0.0,
        le=120.0,
        description="User's age in years",
    )
    gender: str = Field(
        ..., description="User's gender"
    )
    createdAt: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of account creation",
    )
    predictions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Embedded prediction history for this user",
    )
    tests: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Embedded test results for this user",
    )


# ═════════════════════════════════════════════════════════════════
#  Prediction Document
# ═════════════════════════════════════════════════════════════════


class PredictionDocument(BaseModel):
    """HD prediction result.

    When the user is logged in, predictions are embedded inside
    the UserDocument.predictions array. For anonymous users,
    they are stored in the standalone `predictions` collection.

    Attributes:
        userId: Reference to the user who requested the prediction.
        prediction: Predicted HD stage label (e.g. "Early Huntington's").
        confidence: Classification confidence percentage (0–100).
        riskLevel: Risk level category (Low/Medium/High).
        modelVersion: Version of the model used.
        uploadedFileId: Reference to the uploaded MRI file (if any).
        clinicalInputs: The clinical biomarker values used for prediction.
        stageProbabilities: Per-stage classification probabilities.
        progression12mo: Predicted 12-month UHDRS TMS change.
        progression24mo: Predicted 24-month UHDRS TMS change.
        createdAt: UTC timestamp of the prediction.
    """

    userId: str = Field(
        default="anonymous",
        description="Reference to the user who requested the prediction",
    )
    prediction: str = Field(
        ..., description="Predicted HD stage (e.g. \"Early Huntington's\")"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Classification confidence percentage (0–100)",
    )
    riskLevel: str = Field(
        ..., description="Risk level: 'Low', 'Medium', or 'High'"
    )
    modelVersion: str = Field(
        default="v1.0",
        description="Model version used for prediction",
    )
    uploadedFileId: str | None = Field(
        default=None,
        description="Reference to the uploaded MRI file",
    )
    clinicalInputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Assessment biomarker values used for prediction (cag_repeat, motor_score, memory_score, functional_score, age)",
    )
    stageProbabilities: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-stage classification probabilities (pre_manifest, early, advanced)",
    )
    progression12mo: float | None = Field(
        default=0.0,
        description="Predicted 12-month progression score change",
    )
    progression24mo: float | None = Field(
        default=0.0,
        description="Predicted 24-month progression score change",
    )
    createdAt: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the prediction",
    )


# ═════════════════════════════════════════════════════════════════
#  Cognitive Session Document
# ═════════════════════════════════════════════════════════════════


class CognitiveSessionDocument(BaseModel):
    """Cognitive assessment session stored in `cognitive_sessions`.

    Captures the full session lifecycle including test results,
    scores, validation flags, and the generated feature vector.

    Attributes:
        session_id: Unique session identifier.
        patient_id: Patient who completed this session.
        patient_age: Patient age at session time.
        status: Session lifecycle status.
        session_number: 1-based session index for this patient.
        is_baseline: Whether this is a baseline session.
        is_valid: Whether the session passed validation.
        versions_used: Parallel form versions per test.
        test_results: Scored results per test.
        domain_scores: Z-scores and categories per domain.
        overall_category: Overall cognitive category.
        feature_vector: 9-dimensional ML feature vector.
        validation_flags: Any validity issues detected.
        days_since_baseline: Days since first session.
        created_at: UTC session creation timestamp.
        completed_at: UTC session completion timestamp.
    """

    session_id: str = Field(
        ..., description="Unique session identifier"
    )
    patient_id: str = Field(
        default="anonymous", description="Patient identifier"
    )
    patient_age: float = Field(
        default=0.0, description="Patient age at session time"
    )
    status: str = Field(
        default="created",
        description="Session status: created/in_progress/completed/invalid",
    )
    session_number: int = Field(
        default=1, description="1-based session index"
    )
    is_baseline: bool = Field(
        default=False,
        description="Whether this is a baseline session",
    )
    is_valid: bool = Field(
        default=True,
        description="Whether session passed validation",
    )

    versions_used: dict[str, Any] = Field(
        default_factory=dict,
        description="Parallel form versions per test",
    )
    test_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Scored results per test",
    )
    domain_scores: dict[str, Any] = Field(
        default_factory=dict,
        description="Z-scores and categories per domain",
    )
    overall_category: str = Field(
        default="",
        description="Overall cognitive category",
    )
    feature_vector: list[float] = Field(
        default_factory=list,
        description="9-dimensional ML feature vector",
    )
    validation_flags: list[str] = Field(
        default_factory=list,
        description="Validity issues detected",
    )
    days_since_baseline: float = Field(
        default=0.0,
        description="Days since first session",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC session creation timestamp",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="UTC session completion timestamp",
    )


# ═════════════════════════════════════════════════════════════════
#  Test Result Document (Memory & Motor Tests)
# ═════════════════════════════════════════════════════════════════


class TestResultDocument(BaseModel):
    """Individual test result stored in the `test_results` collection.

    Stores results from frontend memory tests (word_recall,
    sequence_memory, visual_change) and motor tests (reaction_time,
    finger_tapping, tracking).

    Attributes:
        testName: Identifier of the test performed.
        testCategory: Category: 'memory' or 'motor'.
        score: Numeric score achieved.
        total: Maximum possible score.
        percentage: Score as percentage (0–100).
        details: Test-specific extra data (e.g. avgMs, wordsRecalled).
        patientId: Optional patient identifier.
        createdAt: UTC timestamp of when the test was completed.
    """

    testName: str = Field(
        ..., description="Test identifier (e.g. 'word_recall', 'reaction_time')"
    )
    testCategory: str = Field(
        ..., description="Test category: 'memory' or 'motor'"
    )
    score: float = Field(
        ..., description="Score achieved"
    )
    total: float = Field(
        ..., description="Maximum possible score"
    )
    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score as percentage (0–100)",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Test-specific extra data (avgMs, wordsRecalled, etc.)",
    )
    patientId: str = Field(
        default="anonymous",
        description="Patient identifier",
    )
    createdAt: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of test completion",
    )


# ═════════════════════════════════════════════════════════════════


class ChatMessageDoc(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(
        ..., description="Message role: 'user' or 'assistant'"
    )
    content: str = Field(
        ..., description="Message content text"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of this message",
    )


class ChatConversationDocument(BaseModel):
    """Chat conversation stored in `chat_conversations`.

    Each document represents a single conversation session
    with the NeuroSense AI assistant, containing all messages.

    Attributes:
        session_id: Unique conversation session identifier.
        patient_id: Patient identifier (optional).
        messages: Ordered list of chat messages.
        message_count: Total number of messages.
        created_at: UTC timestamp of conversation start.
        updated_at: UTC timestamp of last message.
    """

    session_id: str = Field(
        ..., description="Unique conversation session identifier"
    )
    patient_id: str = Field(
        default="anonymous",
        description="Patient identifier (optional)",
    )
    messages: list[ChatMessageDoc] = Field(
        default_factory=list,
        description="Ordered list of chat messages",
    )
    message_count: int = Field(
        default=0,
        description="Total number of messages",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of conversation start",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of last message",
    )
