"""NeuroSense Cognitive Assessment — Scoring & Normative Data.

Stateless scoring module that converts raw test scores into
age-normed z-scores using published PREDICT-HD normative data.
Assembles the 9-dimensional cognitive feature vector consumed
by the Bi-LSTM clinical encoder in the ML pipeline.

Normative tables:
    Based on PREDICT-HD study normative data stratified by
    age decade (30–39, 40–49, 50–59, 60–69). Patients outside
    these ranges are clamped to the nearest band.

Output feature vector per session (9 dimensions):
    [sdmt_z, nback_z, fluency_z, tmt_b_z, recall_z,
     sdmt_rt_mean, nback_rt_mean,
     days_since_baseline, session_number]

Public API:
    compute_z_score: Single test z-score computation
    classify_domain: Map z-score to clinical category
    compute_all_z_scores: All 5 z-scores from session results
    build_feature_vector: Assemble the 9-dim vector
    get_normative_stats: Look up norms for a test + age
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from neurosense.cognitive.tests import TestName, TestResult

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Normative Data Tables (PREDICT-HD)
# ═════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NormativeStats:
    """Normative statistics for a single age band.

    Attributes:
        mean: Population mean for this age band.
        sd: Population standard deviation.
        age_min: Minimum age (inclusive) for this band.
        age_max: Maximum age (inclusive) for this band.
    """

    mean: float
    sd: float
    age_min: int
    age_max: int


class ClinicalCategory(str, Enum):
    """Clinical interpretation of a cognitive z-score.

    Thresholds based on standard neuropsychological practice:
        z > -1.0  : NORMAL  — Within normal limits
        -2.0 ≤ z ≤ -1.0 : MILD — Mild cognitive decline
        z < -2.0  : SIGNIFICANT — Significant decline
    """

    NORMAL = "normal"
    MILD = "mild_decline"
    SIGNIFICANT = "significant_decline"


# Normative tables keyed by TestName → list of NormativeStats
# ordered by age band. Based on PREDICT-HD published normative data.

NORMATIVE_DATA: dict[TestName, list[NormativeStats]] = {
    # SDMT: correct responses in 90 seconds (higher = better)
    TestName.SDMT: [
        NormativeStats(mean=54.2, sd=9.1, age_min=30, age_max=39),
        NormativeStats(mean=51.8, sd=9.4, age_min=40, age_max=49),
        NormativeStats(mean=47.3, sd=9.8, age_min=50, age_max=59),
        NormativeStats(mean=42.1, sd=10.2, age_min=60, age_max=69),
    ],
    # 2-Back: accuracy percentage (higher = better)
    TestName.NBACK: [
        NormativeStats(mean=87.4, sd=8.2, age_min=30, age_max=39),
        NormativeStats(mean=84.1, sd=9.0, age_min=40, age_max=49),
        NormativeStats(mean=79.6, sd=9.7, age_min=50, age_max=59),
        NormativeStats(mean=74.2, sd=10.5, age_min=60, age_max=69),
    ],
    # Verbal Fluency: valid word count (higher = better)
    TestName.VERBAL_FLUENCY: [
        NormativeStats(mean=16.8, sd=4.2, age_min=30, age_max=39),
        NormativeStats(mean=15.9, sd=4.4, age_min=40, age_max=49),
        NormativeStats(mean=14.7, sd=4.6, age_min=50, age_max=59),
        NormativeStats(mean=13.2, sd=4.9, age_min=60, age_max=69),
    ],
    # Trail Making B: time in seconds (LOWER is better → z is negated)
    TestName.TRAIL_MAKING: [
        NormativeStats(mean=68.4, sd=22.1, age_min=30, age_max=39),
        NormativeStats(mean=78.2, sd=25.4, age_min=40, age_max=49),
        NormativeStats(mean=94.6, sd=31.2, age_min=50, age_max=59),
        NormativeStats(mean=118.3, sd=38.7, age_min=60, age_max=69),
    ],
    # Delayed Recall: correct words out of 10 (higher = better)
    TestName.DELAYED_RECALL: [
        NormativeStats(mean=7.8, sd=1.6, age_min=30, age_max=39),
        NormativeStats(mean=7.2, sd=1.8, age_min=40, age_max=49),
        NormativeStats(mean=6.5, sd=1.9, age_min=50, age_max=59),
        NormativeStats(mean=5.8, sd=2.1, age_min=60, age_max=69),
    ],
}

# Tests where lower raw score = better performance (time-based)
# For these tests the z-score is negated so that negative z
# always means worse performance.
LOWER_IS_BETTER_TESTS: set[TestName] = {TestName.TRAIL_MAKING}


# ═════════════════════════════════════════════════════════════════
#  Normative Lookup
# ═════════════════════════════════════════════════════════════════


def get_normative_stats(
    test_name: TestName,
    age: float,
) -> NormativeStats:
    """Look up normative mean and SD for a test and patient age.

    Selects the age band containing the patient's age. Patients
    younger than 30 are clamped to the 30–39 band; patients
    older than 69 are clamped to the 60–69 band.

    Args:
        test_name: Which cognitive test.
        age: Patient age in years.

    Returns:
        NormativeStats for the matching age band.

    Raises:
        KeyError: If test_name is not in NORMATIVE_DATA.
    """
    bands = NORMATIVE_DATA.get(test_name)
    if bands is None:
        raise KeyError(f"No normative data for test: {test_name}")

    age_int = int(age)

    # Find the matching band
    for band in bands:
        if band.age_min <= age_int <= band.age_max:
            return band

    # Clamp to nearest edge band
    if age_int < bands[0].age_min:
        logger.debug(
            "Age %d below norm range for %s, using band %d–%d",
            age_int, test_name.value, bands[0].age_min, bands[0].age_max,
        )
        return bands[0]

    logger.debug(
        "Age %d above norm range for %s, using band %d–%d",
        age_int, test_name.value, bands[-1].age_min, bands[-1].age_max,
    )
    return bands[-1]


# ═════════════════════════════════════════════════════════════════
#  Z-Score Computation
# ═════════════════════════════════════════════════════════════════


def compute_z_score(
    test_name: TestName,
    raw_score: float,
    age: float,
) -> float:
    """Compute age-normed z-score for a single test.

    For time-based tests (Trail Making B), the z-score is
    negated so that negative z always indicates worse
    performance relative to age norms.

    Formula:
        z = (patient_score - norm_mean) / norm_sd
        For lower-is-better tests: z = -z

    Args:
        test_name: Which cognitive test.
        raw_score: The patient's raw score on the test.
        age: Patient age in years.

    Returns:
        Z-score (negative = worse than age norms).

    Example:
        >>> compute_z_score(TestName.SDMT, 42.0, age=45)
        -1.04  # 42 is below 40–49 mean of 51.8
    """
    norms = get_normative_stats(test_name, age)

    if norms.sd <= 0:
        logger.warning(
            "SD is zero for %s age band %d–%d, returning 0.0",
            test_name.value, norms.age_min, norms.age_max,
        )
        return 0.0

    z = (raw_score - norms.mean) / norms.sd

    # Negate for lower-is-better tests so negative z = worse
    if test_name in LOWER_IS_BETTER_TESTS:
        z = -z

    return round(z, 4)


def classify_domain(z_score: float) -> ClinicalCategory:
    """Classify a z-score into a clinical category.

    Thresholds:
        z > -1.0  : NORMAL
        -2.0 ≤ z ≤ -1.0 : MILD
        z < -2.0  : SIGNIFICANT

    Args:
        z_score: The computed z-score.

    Returns:
        ClinicalCategory enum value.
    """
    if z_score > -1.0:
        return ClinicalCategory.NORMAL
    elif z_score >= -2.0:
        return ClinicalCategory.MILD
    else:
        return ClinicalCategory.SIGNIFICANT


# ═════════════════════════════════════════════════════════════════
#  Batch Scoring
# ═════════════════════════════════════════════════════════════════


@dataclass
class DomainScore:
    """Scored result for a single cognitive domain.

    Attributes:
        test_name: Canonical test name.
        raw_score: Patient's raw score.
        z_score: Age-normed z-score.
        category: Clinical category (normal/mild/significant).
        norm_mean: Normative mean used for comparison.
        norm_sd: Normative SD used for comparison.
        age_band: Age band label (e.g., "40–49").
        metadata: Extra scoring metadata.
    """

    test_name: TestName
    raw_score: float
    z_score: float
    category: ClinicalCategory
    norm_mean: float
    norm_sd: float
    age_band: str
    metadata: dict[str, Any] | None = None


def compute_all_z_scores(
    results: dict[TestName, TestResult],
    age: float,
) -> dict[TestName, DomainScore]:
    """Compute z-scores for all completed tests in a session.

    Args:
        results: Mapping from TestName to TestResult for each
                 completed test in the session.
        age: Patient age in years.

    Returns:
        Mapping from TestName to DomainScore with z-scores
        and clinical categories.
    """
    scores: dict[TestName, DomainScore] = {}

    for test_name, result in results.items():
        norms = get_normative_stats(test_name, age)
        z = compute_z_score(test_name, result.raw_score, age)
        category = classify_domain(z)

        scores[test_name] = DomainScore(
            test_name=test_name,
            raw_score=result.raw_score,
            z_score=z,
            category=category,
            norm_mean=norms.mean,
            norm_sd=norms.sd,
            age_band=f"{norms.age_min}–{norms.age_max}",
            metadata={
                "version": result.version,
                "practice_accuracy": result.practice_accuracy,
                **(result.metadata or {}),
            },
        )

        logger.info(
            "Scored %s: raw=%.1f, z=%.3f (%s), norms=%.1f±%.1f [age %s]",
            test_name.value,
            result.raw_score,
            z,
            category.value,
            norms.mean,
            norms.sd,
            scores[test_name].age_band,
        )

    return scores


# ═════════════════════════════════════════════════════════════════
#  9-Dimensional Feature Vector
# ═════════════════════════════════════════════════════════════════


def build_feature_vector(
    domain_scores: dict[TestName, DomainScore],
    results: dict[TestName, TestResult],
    days_since_baseline: float,
    session_number: int,
) -> list[float]:
    """Assemble the 9-dimensional cognitive feature vector.

    This vector is appended to the clinical feature sequence
    consumed by the Bi-LSTM clinical encoder.

    Dimensions:
        [0] sdmt_z_score
        [1] nback_z_score
        [2] verbal_fluency_z_score
        [3] tmt_b_z_score
        [4] delayed_recall_z_score
        [5] sdmt_mean_reaction_time_ms
        [6] nback_mean_reaction_time_ms
        [7] days_since_baseline
        [8] session_number

    Missing tests are filled with 0.0 (z=0 means average).

    Args:
        domain_scores: Z-scores from compute_all_z_scores.
        results: Raw TestResult objects for timing data.
        days_since_baseline: Calendar days since the patient's
                             first session.
        session_number: 1-based session index.

    Returns:
        List of 9 float values.
    """
    # Extract z-scores (default 0.0 if test missing)
    sdmt_z = domain_scores.get(TestName.SDMT, _default_domain()).z_score
    nback_z = domain_scores.get(TestName.NBACK, _default_domain()).z_score
    fluency_z = domain_scores.get(TestName.VERBAL_FLUENCY, _default_domain()).z_score
    tmt_z = domain_scores.get(TestName.TRAIL_MAKING, _default_domain()).z_score
    recall_z = domain_scores.get(TestName.DELAYED_RECALL, _default_domain()).z_score

    # Extract mean reaction times from raw results
    sdmt_rt = 0.0
    nback_rt = 0.0

    sdmt_result = results.get(TestName.SDMT)
    if sdmt_result is not None:
        sdmt_rt = sdmt_result.timing.get("mean_rt_ms", 0.0)

    nback_result = results.get(TestName.NBACK)
    if nback_result is not None:
        nback_rt = nback_result.timing.get("mean_rt_ms", 0.0)

    vector = [
        round(sdmt_z, 4),
        round(nback_z, 4),
        round(fluency_z, 4),
        round(tmt_z, 4),
        round(recall_z, 4),
        round(sdmt_rt, 2),
        round(nback_rt, 2),
        round(days_since_baseline, 1),
        float(session_number),
    ]

    logger.info(
        "Feature vector built: z=[%.2f, %.2f, %.2f, %.2f, %.2f] "
        "rt=[%.0f, %.0f] day=%.0f sess=%d",
        *vector,
    )

    return vector


def _default_domain() -> DomainScore:
    """Create a default DomainScore for missing tests.

    Returns a score with z=0.0 (population average) so that
    missing tests don't bias the ML pipeline.

    Returns:
        DomainScore with neutral values.
    """
    return DomainScore(
        test_name=TestName.SDMT,  # Placeholder
        raw_score=0.0,
        z_score=0.0,
        category=ClinicalCategory.NORMAL,
        norm_mean=0.0,
        norm_sd=1.0,
        age_band="unknown",
    )


# ═════════════════════════════════════════════════════════════════
#  Session Summary
# ═════════════════════════════════════════════════════════════════


@dataclass
class SessionSummary:
    """Complete scored summary for a cognitive session.

    Attributes:
        session_id: Unique session identifier.
        patient_id: Patient identifier.
        patient_age: Age at time of session.
        domain_scores: Per-test z-scores and categories.
        feature_vector: 9-dim vector for ML pipeline.
        is_valid: Whether the session passed validation.
        is_baseline: Whether this is a baseline session (1st or 2nd).
        session_number: 1-based session index for this patient.
        days_since_baseline: Calendar days since first session.
        overall_category: Worst category across all domains.
        versions_used: Which parallel form version per test.
        validation_flags: Any validation issues found.
    """

    session_id: str
    patient_id: str
    patient_age: float
    domain_scores: dict[TestName, DomainScore]
    feature_vector: list[float]
    is_valid: bool
    is_baseline: bool
    session_number: int
    days_since_baseline: float
    overall_category: ClinicalCategory
    versions_used: dict[TestName, int]
    validation_flags: list[str]


def compute_overall_category(
    domain_scores: dict[TestName, DomainScore],
) -> ClinicalCategory:
    """Determine the worst clinical category across all domains.

    The overall assessment uses the most impaired domain
    (most conservative interpretation).

    Args:
        domain_scores: Per-test domain scores.

    Returns:
        The worst (most severe) ClinicalCategory found.
    """
    if not domain_scores:
        return ClinicalCategory.NORMAL

    categories = [ds.category for ds in domain_scores.values()]

    if ClinicalCategory.SIGNIFICANT in categories:
        return ClinicalCategory.SIGNIFICANT
    elif ClinicalCategory.MILD in categories:
        return ClinicalCategory.MILD
    else:
        return ClinicalCategory.NORMAL


def build_session_summary(
    session_id: str,
    patient_id: str,
    patient_age: float,
    results: dict[TestName, TestResult],
    is_valid: bool,
    is_baseline: bool,
    session_number: int,
    days_since_baseline: float,
    validation_flags: list[str],
) -> SessionSummary:
    """Build a complete session summary with all scores.

    Orchestrates z-score computation, feature vector assembly,
    and clinical category determination.

    Args:
        session_id: Unique session identifier.
        patient_id: Patient identifier.
        patient_age: Patient age in years.
        results: Raw TestResult objects for all completed tests.
        is_valid: Whether the session passed validation checks.
        is_baseline: Whether this is a baseline session.
        session_number: 1-based session index.
        days_since_baseline: Days since patient's first session.
        validation_flags: List of validation issue strings.

    Returns:
        Complete SessionSummary ready for storage and display.
    """
    # Compute all z-scores
    domain_scores = compute_all_z_scores(results, patient_age)

    # Build feature vector
    feature_vector = build_feature_vector(
        domain_scores=domain_scores,
        results=results,
        days_since_baseline=days_since_baseline,
        session_number=session_number,
    )

    # Determine overall category
    overall = compute_overall_category(domain_scores)

    # Collect versions used
    versions_used = {
        name: result.version
        for name, result in results.items()
    }

    summary = SessionSummary(
        session_id=session_id,
        patient_id=patient_id,
        patient_age=patient_age,
        domain_scores=domain_scores,
        feature_vector=feature_vector,
        is_valid=is_valid,
        is_baseline=is_baseline,
        session_number=session_number,
        days_since_baseline=days_since_baseline,
        overall_category=overall,
        versions_used=versions_used,
        validation_flags=validation_flags,
    )

    logger.info(
        "Session summary built: %s patient=%s valid=%s baseline=%s overall=%s",
        session_id,
        patient_id,
        is_valid,
        is_baseline,
        overall.value,
    )

    return summary
