"""NeuroSense Cognitive Assessment — Session Management.

Manages the full lifecycle of a cognitive assessment session:
    1. Session creation with parallel form version rotation
    2. Per-test result submission and storage
    3. Session validation (rushed, abandoned, misunderstood, too-soon)
    4. Session completion and feature vector generation

Session validation rules:
    - RUSHED: total time < 20% of patient's median session time
    - ABANDONED: any gap > 10 minutes mid-session
    - MISUNDERSTOOD: practice trial accuracy < 60%
    - TOO_SOON: < 20 days since previous session

Baseline rules:
    - First 2 sessions are marked as baseline
    - Baseline sessions establish personal norms
    - Excluded from decline detection analysis

Parallel form rotation:
    - 6 versions (0–5) per test
    - Never reuse the version from the immediately previous session
    - Versions are assigned deterministically from session_id

Storage:
    Sessions are stored in an in-memory dict (production would
    use a database). All raw event data is preserved for audit.

Public API:
    SessionManager — Singleton session lifecycle manager
    CognitiveSession — Dataclass for a single session
    ValidationFlag — Enum of session validity issues
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from neurosense.cognitive.scoring import (
    SessionSummary,
    build_session_summary,
)
from neurosense.cognitive.tests import (
    NUM_VERSIONS,
    TEST_ORDER,
    TestName,
    TestResult,
    create_test,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Enums & Constants
# ═════════════════════════════════════════════════════════════════


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INVALID = "invalid"


class ValidationFlag(str, Enum):
    """Flags indicating why a session was invalidated."""

    RUSHED = "rushed"
    ABANDONED = "abandoned"
    MISUNDERSTOOD = "misunderstood"
    TOO_SOON = "too_soon"


# Validation thresholds
RUSHED_THRESHOLD: float = 0.20        # 20% of median session time
ABANDONED_GAP_SECONDS: float = 600.0  # 10 minutes
PRACTICE_ACCURACY_MIN: float = 0.60   # 60% accuracy on practice
MIN_DAYS_BETWEEN_SESSIONS: int = 20   # Minimum days gap
NUM_BASELINE_SESSIONS: int = 2        # First N sessions are baseline


# ═════════════════════════════════════════════════════════════════
#  Session Data Model
# ═════════════════════════════════════════════════════════════════


@dataclass
class CognitiveSession:
    """A single cognitive assessment session.

    Attributes:
        session_id: Unique identifier (UUID).
        patient_id: Patient identifier.
        patient_age: Patient age at time of session.
        status: Current lifecycle status.
        created_at: UTC timestamp of session creation.
        completed_at: UTC timestamp of completion (if completed).
        versions: Parallel form versions assigned per test.
        results: TestResult objects submitted per test.
        test_timestamps: Start/end timestamps per test.
        validation_flags: Any validity issues detected.
        is_baseline: Whether this is a baseline session.
        session_number: 1-based index for this patient.
        days_since_baseline: Days since patient's first session.
        feature_vector: 9-dim vector (set on completion).
        summary: Full scored summary (set on completion).
    """

    session_id: str
    patient_id: str
    patient_age: float
    status: SessionStatus = SessionStatus.CREATED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    versions: dict[str, int] = field(default_factory=dict)
    results: dict[str, TestResult] = field(default_factory=dict)
    test_timestamps: dict[str, dict[str, float]] = field(default_factory=dict)

    validation_flags: list[str] = field(default_factory=list)
    is_baseline: bool = False
    session_number: int = 1
    days_since_baseline: float = 0.0

    feature_vector: list[float] | None = None
    summary: SessionSummary | None = None


# ═════════════════════════════════════════════════════════════════
#  Patient History
# ═════════════════════════════════════════════════════════════════


@dataclass
class PatientHistory:
    """Longitudinal history for a single patient.

    Attributes:
        patient_id: Patient identifier.
        sessions: All sessions ordered by creation time.
        session_count: Total number of sessions.
        last_version_used: Most recent version per test.
        median_session_time_s: Median total session time.
    """

    patient_id: str
    sessions: list[CognitiveSession] = field(default_factory=list)
    session_count: int = 0
    last_version_used: dict[str, int] = field(default_factory=dict)
    median_session_time_s: float | None = None


# ═════════════════════════════════════════════════════════════════
#  Session Manager
# ═════════════════════════════════════════════════════════════════


class SessionManager:
    """Manages cognitive assessment session lifecycle.

    Handles session creation, version rotation, result submission,
    validation, and completion. In production, this would be backed
    by a database; here we use in-memory storage.

    Usage:
        >>> manager = SessionManager()
        >>> session = manager.start_session("patient-001", age=42.0)
        >>> manager.submit_result(session.session_id, "sdmt", result)
        >>> summary = manager.complete_session(session.session_id)
    """

    def __init__(self) -> None:
        self._sessions: dict[str, CognitiveSession] = {}
        self._patients: dict[str, PatientHistory] = {}

    # ─── Session Creation ───

    def start_session(
        self,
        patient_id: str,
        patient_age: float,
    ) -> CognitiveSession:
        """Start a new cognitive assessment session.

        Creates a session with:
        1. Unique session_id
        2. Rotated parallel form versions (no consecutive repeats)
        3. Correct session_number and baseline flagging
        4. TOO_SOON validation if applicable

        Args:
            patient_id: Patient identifier.
            patient_age: Patient's current age.

        Returns:
            The created CognitiveSession with versions assigned.
        """
        session_id = str(uuid.uuid4())
        history = self._get_or_create_history(patient_id)

        # Determine session number (1-based)
        session_number = history.session_count + 1

        # Check if this is a baseline session
        is_baseline = session_number <= NUM_BASELINE_SESSIONS

        # Compute days since baseline
        days_since_baseline = 0.0
        if history.sessions:
            first_session = history.sessions[0]
            delta = datetime.now(timezone.utc) - first_session.created_at
            days_since_baseline = delta.total_seconds() / 86400.0

        # Assign versions with rotation
        versions = self._assign_versions(history, session_id)

        # Check TOO_SOON
        validation_flags: list[str] = []
        if history.sessions:
            last_session = history.sessions[-1]
            days_since_last = (
                datetime.now(timezone.utc) - last_session.created_at
            ).total_seconds() / 86400.0

            if days_since_last < MIN_DAYS_BETWEEN_SESSIONS:
                validation_flags.append(ValidationFlag.TOO_SOON.value)
                logger.warning(
                    "Session %s for patient %s is too soon: %.1f days (min %d)",
                    session_id,
                    patient_id,
                    days_since_last,
                    MIN_DAYS_BETWEEN_SESSIONS,
                )

        session = CognitiveSession(
            session_id=session_id,
            patient_id=patient_id,
            patient_age=patient_age,
            status=SessionStatus.CREATED,
            versions=versions,
            is_baseline=is_baseline,
            session_number=session_number,
            days_since_baseline=round(days_since_baseline, 1),
            validation_flags=validation_flags,
        )

        self._sessions[session_id] = session

        logger.info(
            "Session started: %s patient=%s age=%.0f session_num=%d "
            "baseline=%s versions=%s",
            session_id,
            patient_id,
            patient_age,
            session_number,
            is_baseline,
            versions,
        )

        return session

    def _assign_versions(
        self,
        history: PatientHistory,
        session_id: str,
    ) -> dict[str, int]:
        """Assign parallel form versions with rotation.

        For each test, excludes the version used in the most
        recent session and picks deterministically from the
        remaining 5 versions using the session_id as seed.

        Args:
            history: Patient's session history.
            session_id: Current session ID (for deterministic RNG).

        Returns:
            Dict mapping test name → version (0–5).
        """
        import hashlib
        import random

        seed_int = int(hashlib.sha256(session_id.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed_int)

        versions: dict[str, int] = {}

        for test_name in TEST_ORDER:
            available = list(range(NUM_VERSIONS))

            # Exclude the last version used for this test
            last_ver = history.last_version_used.get(test_name.value)
            if last_ver is not None and last_ver in available:
                available.remove(last_ver)

            version = rng.choice(available)
            versions[test_name.value] = version

        return versions

    # ─── Result Submission ───

    def submit_result(
        self,
        session_id: str,
        test_name: str,
        raw_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit results for a single test within a session.

        Instantiates the test class, scores the raw responses,
        and stores the TestResult on the session. Also records
        submission timestamps for abandonment detection.

        Args:
            session_id: Session identifier.
            test_name: Canonical test name (e.g., "sdmt").
            raw_results: Raw result data from the frontend,
                         containing responses, timing, etc.

        Returns:
            Dict with raw_score, metadata, and practice_accuracy.

        Raises:
            KeyError: If session_id is unknown.
            ValueError: If test_name is invalid.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session: {session_id}")

        # Parse test name
        try:
            tn = TestName(test_name)
        except ValueError:
            raise ValueError(f"Unknown test: {test_name}")

        # Get version for this test
        version = session.versions.get(test_name, 0)

        # Record timestamp
        now = time.perf_counter()
        session.test_timestamps[test_name] = {
            "submitted_at": now,
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Update status
        session.status = SessionStatus.IN_PROGRESS

        # Create test instance and score
        test = create_test(tn, session.session_id, version)

        result = self._score_test(tn, test, raw_results)

        # Store result
        session.results[test_name] = result

        logger.info(
            "Result submitted: session=%s test=%s score=%.1f practice_acc=%.2f",
            session_id,
            test_name,
            result.raw_score,
            result.practice_accuracy,
        )

        return {
            "test_name": test_name,
            "raw_score": result.raw_score,
            "raw_score_secondary": result.raw_score_secondary,
            "practice_accuracy": result.practice_accuracy,
            "metadata": result.metadata,
        }

    def _score_test(
        self,
        test_name: TestName,
        test: Any,
        raw_results: dict[str, Any],
    ) -> TestResult:
        """Score a single test using its class.

        Dispatches to the appropriate scoring method based on
        test type.

        Args:
            test_name: Which test.
            test: Test instance.
            raw_results: Raw frontend data.

        Returns:
            Scored TestResult.
        """
        responses = raw_results.get("responses", [])
        stimuli_data = raw_results.get("stimuli", [])

        if test_name == TestName.SDMT:
            stimuli = test.generate_stimuli()
            return test.score_responses(responses, stimuli)

        elif test_name == TestName.NBACK:
            stimuli = test.generate_stimuli()
            return test.score_responses(responses, stimuli)

        elif test_name == TestName.VERBAL_FLUENCY:
            return test.score_responses(responses)

        elif test_name == TestName.TRAIL_MAKING:
            responses_a = raw_results.get("responses_a", [])
            responses_b = raw_results.get("responses_b", [])
            stimuli_a = test.generate_stimuli_part_a()
            stimuli_b = test.generate_stimuli_part_b()
            return test.score_responses(responses_a, responses_b, stimuli_a, stimuli_b)

        elif test_name == TestName.DELAYED_RECALL:
            recalled = raw_results.get("recalled_words", [])
            encoding = test.get_encoding_words()
            return test.score_responses(recalled, encoding)

        else:
            raise ValueError(f"No scoring logic for: {test_name}")

    # ─── Session Completion ───

    def complete_session(
        self,
        session_id: str,
    ) -> SessionSummary:
        """Complete a session: validate, score, and build feature vector.

        Runs all validation checks, computes z-scores for every
        completed test, builds the 9-dim feature vector, and
        updates the patient history.

        Args:
            session_id: Session identifier.

        Returns:
            Complete SessionSummary.

        Raises:
            KeyError: If session_id is unknown.
            ValueError: If no tests have been submitted.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session: {session_id}")

        if not session.results:
            raise ValueError(f"No tests submitted for session {session_id}")

        # Run validation
        flags = self._validate_session(session)
        session.validation_flags.extend(flags)
        is_valid = len(session.validation_flags) == 0

        # Convert string keys to TestName enum
        typed_results: dict[TestName, TestResult] = {}
        for name_str, result in session.results.items():
            try:
                typed_results[TestName(name_str)] = result
            except ValueError:
                logger.warning("Skipping unknown test in results: %s", name_str)

        # Build session summary
        summary = build_session_summary(
            session_id=session.session_id,
            patient_id=session.patient_id,
            patient_age=session.patient_age,
            results=typed_results,
            is_valid=is_valid,
            is_baseline=session.is_baseline,
            session_number=session.session_number,
            days_since_baseline=session.days_since_baseline,
            validation_flags=session.validation_flags,
        )

        # Update session
        session.status = SessionStatus.COMPLETED if is_valid else SessionStatus.INVALID
        session.completed_at = datetime.now(timezone.utc)
        session.feature_vector = summary.feature_vector
        session.summary = summary

        # Update patient history
        history = self._get_or_create_history(session.patient_id)
        history.sessions.append(session)
        history.session_count += 1

        # Update last versions used
        for test_name_str, version in session.versions.items():
            history.last_version_used[test_name_str] = version

        # Update median session time
        self._update_median_time(history)

        logger.info(
            "Session completed: %s valid=%s flags=%s baseline=%s",
            session_id,
            is_valid,
            session.validation_flags,
            session.is_baseline,
        )

        return summary

    # ─── Validation ───

    def _validate_session(
        self,
        session: CognitiveSession,
    ) -> list[str]:
        """Run all validation checks on a session.

        Checks:
        1. RUSHED: total time < 20% of median
        2. ABANDONED: gap > 10 min between test submissions
        3. MISUNDERSTOOD: practice accuracy < 60% on any test

        (TOO_SOON is already checked at session creation.)

        Args:
            session: The session to validate.

        Returns:
            List of ValidationFlag values for detected issues.
        """
        flags: list[str] = []

        # ─── RUSHED check ───
        history = self._patients.get(session.patient_id)
        if history and history.median_session_time_s is not None:
            session_time = self._compute_session_time(session)
            threshold = history.median_session_time_s * RUSHED_THRESHOLD

            if session_time > 0 and session_time < threshold:
                flags.append(ValidationFlag.RUSHED.value)
                logger.warning(
                    "Session %s RUSHED: %.0fs < %.0fs (20%% of median %.0fs)",
                    session.session_id,
                    session_time,
                    threshold,
                    history.median_session_time_s,
                )

        # ─── ABANDONED check ───
        timestamps = sorted(
            ts.get("submitted_at", 0.0)
            for ts in session.test_timestamps.values()
            if ts.get("submitted_at")
        )
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            if gap > ABANDONED_GAP_SECONDS:
                flags.append(ValidationFlag.ABANDONED.value)
                logger.warning(
                    "Session %s ABANDONED: %.0fs gap between tests",
                    session.session_id,
                    gap,
                )
                break

        # ─── MISUNDERSTOOD check ───
        for test_name, result in session.results.items():
            if result.practice_accuracy < PRACTICE_ACCURACY_MIN:
                flags.append(ValidationFlag.MISUNDERSTOOD.value)
                logger.warning(
                    "Session %s MISUNDERSTOOD: %s practice accuracy %.0f%% < %.0f%%",
                    session.session_id,
                    test_name,
                    result.practice_accuracy * 100,
                    PRACTICE_ACCURACY_MIN * 100,
                )
                break

        return flags

    def _compute_session_time(
        self,
        session: CognitiveSession,
    ) -> float:
        """Compute total session time from test timestamps.

        Args:
            session: The session to measure.

        Returns:
            Total time in seconds, or 0 if insufficient data.
        """
        all_times = [
            ts.get("submitted_at", 0.0)
            for ts in session.test_timestamps.values()
            if ts.get("submitted_at")
        ]

        if len(all_times) < 2:
            return 0.0

        return max(all_times) - min(all_times)

    def _update_median_time(
        self,
        history: PatientHistory,
    ) -> None:
        """Update the median session time for a patient.

        Uses only completed, valid sessions.

        Args:
            history: Patient history to update.
        """
        times = []
        for s in history.sessions:
            if s.status == SessionStatus.COMPLETED:
                t = self._compute_session_time(s)
                if t > 0:
                    times.append(t)

        if times:
            times.sort()
            n = len(times)
            if n % 2 == 0:
                history.median_session_time_s = (times[n // 2 - 1] + times[n // 2]) / 2.0
            else:
                history.median_session_time_s = times[n // 2]

    # ─── History Queries ───

    def get_patient_history(
        self,
        patient_id: str,
    ) -> list[dict[str, Any]]:
        """Get full session history for a patient.

        Returns all sessions with scores, z-scores, validity
        flags, and trend direction per domain.

        Args:
            patient_id: Patient identifier.

        Returns:
            List of session summary dicts ordered by date.
        """
        history = self._patients.get(patient_id)
        if history is None:
            return []

        results: list[dict[str, Any]] = []

        for session in history.sessions:
            if session.summary is None:
                continue

            summary = session.summary
            domain_data: dict[str, Any] = {}

            for test_name, ds in summary.domain_scores.items():
                domain_data[test_name.value] = {
                    "raw_score": ds.raw_score,
                    "z_score": ds.z_score,
                    "category": ds.category.value,
                    "norm_mean": ds.norm_mean,
                    "norm_sd": ds.norm_sd,
                    "age_band": ds.age_band,
                }

            results.append({
                "session_id": session.session_id,
                "session_number": session.session_number,
                "date": session.created_at.isoformat(),
                "is_valid": summary.is_valid,
                "is_baseline": summary.is_baseline,
                "validation_flags": summary.validation_flags,
                "overall_category": summary.overall_category.value,
                "feature_vector": summary.feature_vector,
                "domains": domain_data,
                "versions_used": summary.versions_used,
            })

        # Compute trend per domain
        if len(results) >= 3:
            results = self._add_trend_direction(results)

        return results

    def _add_trend_direction(
        self,
        sessions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add trend direction per domain to session list.

        Compares the most recent 3 valid, non-baseline z-scores
        to determine if each domain is declining, stable, or
        improving.

        Args:
            sessions: List of session dicts.

        Returns:
            Sessions with 'trend' field added to each domain.
        """
        # Filter to valid, non-baseline sessions
        valid_sessions = [
            s for s in sessions
            if s.get("is_valid") and not s.get("is_baseline")
        ]

        if len(valid_sessions) < 2:
            return sessions

        # For each domain, compute trend from last 3 points
        test_names = [tn.value for tn in TEST_ORDER]

        for tn in test_names:
            z_scores = []
            for vs in valid_sessions:
                domain = vs.get("domains", {}).get(tn)
                if domain:
                    z_scores.append(domain["z_score"])

            if len(z_scores) < 2:
                continue

            recent = z_scores[-3:] if len(z_scores) >= 3 else z_scores

            # Simple linear trend: positive slope = improving
            n = len(recent)
            x_mean = (n - 1) / 2.0
            y_mean = sum(recent) / n
            numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            slope = numerator / denominator if denominator > 0 else 0.0

            if slope > 0.1:
                trend = "improving"
            elif slope < -0.1:
                trend = "declining"
            else:
                trend = "stable"

            # Add trend to most recent session
            for s in reversed(sessions):
                domain = s.get("domains", {}).get(tn)
                if domain:
                    domain["trend"] = trend
                    break

        return sessions

    # ─── Helpers ───

    def _get_or_create_history(
        self,
        patient_id: str,
    ) -> PatientHistory:
        """Get or create patient history.

        Args:
            patient_id: Patient identifier.

        Returns:
            PatientHistory instance.
        """
        if patient_id not in self._patients:
            self._patients[patient_id] = PatientHistory(patient_id=patient_id)
        return self._patients[patient_id]

    def get_session(self, session_id: str) -> CognitiveSession | None:
        """Retrieve a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            CognitiveSession or None if not found.
        """
        return self._sessions.get(session_id)


# Module-level singleton instance
session_manager = SessionManager()
