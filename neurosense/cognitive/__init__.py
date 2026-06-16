"""NeuroSense Cognitive Assessment Module.

Digital in-app cognitive test battery for longitudinal
Huntington's Disease monitoring. Patients complete 5 tests
every 30 days; results are converted to z-scores against
age-matched PREDICT-HD norms and fed as a 9-dimensional
feature vector into the Bi-LSTM clinical encoder.

Tests:
    1. SDMT — Symbol Digit Modalities Test
    2. N-Back — 2-Back Working Memory Task
    3. Verbal Fluency — Phonemic Verbal Fluency
    4. Trail Making — Trail Making Test (Parts A & B)
    5. Delayed Recall — Delayed Word Recall

Public API:
    TestName: Enum of test names
    SDMTTest, NBackTest, VerbalFluencyTest,
    TrailMakingTest, DelayedRecallTest: Test classes
    compute_z_score, build_feature_vector: Scoring functions
    SessionManager, session_manager: Session lifecycle
"""

from neurosense.cognitive.scoring import (
    ClinicalCategory,
    DomainScore,
    SessionSummary,
    build_feature_vector,
    classify_domain,
    compute_all_z_scores,
    compute_z_score,
)
from neurosense.cognitive.session import (
    SessionManager,
    SessionStatus,
    ValidationFlag,
    session_manager,
)
from neurosense.cognitive.tests import (
    NUM_VERSIONS,
    TEST_ORDER,
    TestName,
    TestResult,
    create_test,
)

__all__ = [
    # Tests
    "TestName",
    "TEST_ORDER",
    "NUM_VERSIONS",
    "TestResult",
    "create_test",
    # Scoring
    "compute_z_score",
    "compute_all_z_scores",
    "build_feature_vector",
    "classify_domain",
    "ClinicalCategory",
    "DomainScore",
    "SessionSummary",
    # Session
    "SessionManager",
    "session_manager",
    "SessionStatus",
    "ValidationFlag",
]
