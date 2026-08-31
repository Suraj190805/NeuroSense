"""Clinical & Digital HD Assessment Engine.

Computes Huntington's Disease staging, progression, and explainability
from digital assessment biomarkers (Motor Test score, Memory/Cognitive score,
Daily Functional capacity, CAG repeat, Age, and optional clinical symptoms).

Provides a robust clinical/digital assessment signal that is fused with
brain MRI neuroimaging model predictions.

Criteria used:
    - Digital Motor Assessment performance (reaction speed, finger tapping, coordination)
    - Digital Memory & Cognitive performance (episodic recall, working memory, pattern change)
    - Functional independence level (activities of daily living)
    - CAG repeat genetic penetrance (36–120)
    - Langbehn age-adjusted onset estimation
    - Clinical symptoms (movement, cognitive, psychiatric) — optional
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════
#  HD Symptom Categories (Mayo Clinic & NIH clinical criteria)
# ═════════════════════════════════════════════════════════════════

SYMPTOM_CATEGORIES: dict[str, dict] = {
    "movement": {
        "weight": 0.40,
        "symptoms": [
            "chorea",
            "dystonia",
            "bradykinesia",
            "impaired_gait",
            "difficulty_swallowing",
            "slurred_speech",
            "abnormal_eye_movements",
        ],
    },
    "cognitive": {
        "weight": 0.35,
        "symptoms": [
            "difficulty_organizing",
            "slow_processing",
            "difficulty_learning",
            "perseveration",
            "lack_of_awareness",
            "poor_impulse_control",
        ],
    },
    "psychiatric": {
        "weight": 0.25,
        "symptoms": [
            "depression",
            "irritability",
            "apathy",
            "anxiety",
            "social_withdrawal",
            "insomnia",
            "weight_loss_fatigue",
        ],
    },
}

# Flat set for fast lookup
_ALL_SYMPTOM_IDS: set[str] = {
    s for cat in SYMPTOM_CATEGORIES.values() for s in cat["symptoms"]
}


@dataclass
class ClinicalScore:
    """Result of clinical/digital HD scoring.

    Attributes:
        stage: Predicted HD stage (pre_manifest / early / advanced).
        confidence: Confidence in the staging (0.0–1.0).
        pre_manifest_prob: Probability of pre-manifest stage.
        early_prob: Probability of early HD stage.
        advanced_prob: Probability of advanced HD stage.
        risk_category: Risk level (low / medium / high).
        progression_12mo: Expected 12-month progression score change.
        progression_24mo: Expected 24-month progression score change.
        feature_impacts: Dict of feature name → impact score.
        clinical_certainty: How strongly biomarkers indicate
            a definitive staging (0.0–1.0). Used to weight
            digital/clinical vs image model in fusion.
    """

    stage: str
    confidence: float
    pre_manifest_prob: float
    early_prob: float
    advanced_prob: float
    risk_category: str
    progression_12mo: float
    progression_24mo: float
    feature_impacts: dict[str, float] = field(default_factory=dict)
    clinical_certainty: float = 0.5


def compute_clinical_score(
    cag_repeat: float,
    motor_score: float | None = None,
    memory_score: float | None = None,
    functional_score: float | None = None,
    age: float = 45.0,
    # Backward compatibility with legacy parameter names:
    uhdrs_motor: float | None = None,
    uhdrs_cognitive: float | None = None,
    tfc_score: float | None = None,
    # Optional clinical symptoms:
    symptoms: list[str] | None = None,
) -> ClinicalScore:
    """Compute HD staging from digital motor/memory tests & biomarkers.

    Args:
        cag_repeat: CAG trinucleotide repeat count (36–120).
        motor_score: Digital motor assessment score (0–100%).
        memory_score: Digital memory/cognitive assessment score (0–100%).
        functional_score: Daily functional independence (0–100%).
        age: Patient age in years (18–90).
        uhdrs_motor: (Legacy) UHDRS Total Motor Score (0–124).
        uhdrs_cognitive: (Legacy) UHDRS Cognitive score.
        tfc_score: (Legacy) Total Functional Capacity (0–13).
        symptoms: Optional list of symptom IDs the patient reports.

    Returns:
        ClinicalScore with staging, probabilities, and forecasts.
    """
    # ─── 0. Normalize inputs & legacy parameters ───
    if motor_score is None:
        if uhdrs_motor is not None:
            motor_score = max(0.0, min(100.0, 100.0 - (uhdrs_motor / 124.0 * 100.0)))
        else:
            motor_score = 85.0

    if memory_score is None:
        if uhdrs_cognitive is not None:
            memory_score = max(0.0, min(100.0, (uhdrs_cognitive / 200.0) * 100.0))
        else:
            memory_score = 85.0

    if functional_score is None:
        if tfc_score is not None:
            functional_score = max(0.0, min(100.0, (tfc_score / 13.0) * 100.0))
        else:
            functional_score = 100.0

    # ─── 1. Functional Independence Staging ───
    func_result = _functional_severity(functional_score)

    # ─── 2. CAG Repeat Analysis ───
    cag_score = _cag_severity(cag_repeat)

    # ─── 3. Motor Test Performance Severity ───
    motor_result = _motor_severity(motor_score)

    # ─── 4. Memory / Cognitive Assessment ───
    memory_result = _cognitive_severity(memory_score)

    # ─── 5. Age-CAG Onset Estimation (Langbehn formula) ───
    years_to_onset = _langbehn_onset(cag_repeat, age)

    # ─── 5b. Symptom Severity (optional) ───
    symptom_result = _symptom_severity(symptoms)
    has_symptoms = symptom_result["severity"] > 0.0

    # ─── 6. Composite Scoring ───
    # Weights for self-administered assessment model:
    # Motor test & CAG are major primary indicators of manifest status,
    # Memory test & Functional capacity provide strong cognitive/ADL validation.
    # When symptoms are provided, they contribute ~25% and other weights
    # are proportionally reduced to make room.
    if has_symptoms:
        weights = {
            "motor": 0.22,
            "cag": 0.20,
            "functional": 0.15,
            "memory": 0.10,
            "onset": 0.08,
            "symptoms": 0.25,
        }
    else:
        weights = {
            "motor": 0.30,
            "cag": 0.25,
            "functional": 0.20,
            "memory": 0.15,
            "onset": 0.10,
        }

    # Each component produces a severity score (0=normal, 1=severe)
    components = {
        "motor": motor_result["severity"],
        "cag": cag_score["severity"],
        "functional": func_result["severity"],
        "memory": memory_result["severity"],
        "onset": _onset_severity(years_to_onset),
    }
    if has_symptoms:
        components["symptoms"] = symptom_result["severity"]

    # Weighted composite
    composite = sum(
        weights[k] * components[k] for k in weights
    )

    # ─── 7. Stage Classification ───
    if composite < 0.25:
        stage = "pre_manifest"
        pre_prob = 0.70 + (0.25 - composite) * 0.8
        early_prob = composite * 1.0
        advanced_prob = composite * 0.2
    elif composite < 0.55:
        stage = "early"
        pre_prob = max(0.05, 0.35 - composite * 0.5)
        early_prob = 0.50 + (composite - 0.25) * 1.0
        advanced_prob = composite * 0.3
    else:
        stage = "advanced"
        pre_prob = max(0.02, 0.15 - composite * 0.15)
        early_prob = max(0.05, 0.40 - composite * 0.35)
        advanced_prob = 0.55 + (composite - 0.55) * 0.8

    # Normalize probabilities
    total = pre_prob + early_prob + advanced_prob
    pre_prob /= total
    early_prob /= total
    advanced_prob /= total

    confidence = max(pre_prob, early_prob, advanced_prob)

    # ─── 8. Assessment Certainty ───
    agreement = _compute_agreement(components)
    clinical_certainty = min(1.0, agreement * 0.8 + composite * 0.2)

    # Boost certainty for extreme values
    if cag_repeat is not None:
        if cag_repeat >= 50 and motor_score <= 40:
            clinical_certainty = max(clinical_certainty, 0.90)
        elif cag_repeat >= 45 and motor_score <= 60:
            clinical_certainty = max(clinical_certainty, 0.75)
        elif cag_repeat <= 39 and motor_score >= 85:
            clinical_certainty = max(clinical_certainty, 0.70)

    # ─── 9. Progression Forecast ───
    prog_12, prog_24 = _estimate_progression(
        cag_repeat, motor_score, functional_score, age, composite,
    )

    # ─── 10. Risk Category ───
    if prog_12 < 3.0:
        risk = "low"
    elif prog_12 < 8.0:
        risk = "medium"
    else:
        risk = "high"

    # ─── 11. Feature Impacts (SHAP-like) ───
    feature_impacts = _compute_feature_impacts(
        cag_repeat, motor_score, memory_score,
        functional_score, age, components, weights,
        symptoms=symptoms,
    )

    return ClinicalScore(
        stage=stage,
        confidence=round(confidence, 4),
        pre_manifest_prob=round(pre_prob, 4),
        early_prob=round(early_prob, 4),
        advanced_prob=round(advanced_prob, 4),
        risk_category=risk,
        progression_12mo=round(prog_12, 1),
        progression_24mo=round(prog_24, 1),
        feature_impacts=feature_impacts,
        clinical_certainty=round(clinical_certainty, 4),
    )


# ═════════════════════════════════════════════════════════════════
#  Component Scoring Functions
# ═════════════════════════════════════════════════════════════════


def _functional_severity(functional_score: float) -> dict:
    """Map Daily Functional Independence score (0–100%) to severity.

    100% = fully independent (Stage I equivalent)
    0% = severe dependency (Stage V equivalent)
    """
    if functional_score >= 85:
        return {"stage": "I", "severity": 0.10}
    elif functional_score >= 60:
        return {"stage": "II", "severity": 0.35}
    elif functional_score >= 35:
        return {"stage": "III", "severity": 0.65}
    elif functional_score >= 15:
        return {"stage": "IV", "severity": 0.85}
    else:
        return {"stage": "V", "severity": 1.0}


def _cag_severity(cag: float | None) -> dict:
    """Score CAG repeat severity."""
    if cag is None or cag < 36:
        return {"desc": "normal", "severity": 0.0}
    elif cag <= 39:
        return {"desc": "reduced_penetrance", "severity": 0.15}
    elif cag <= 44:
        return {"desc": "full_penetrance", "severity": 0.35}
    elif cag <= 49:
        return {"desc": "high_repeat", "severity": 0.55}
    elif cag <= 59:
        return {"desc": "very_high", "severity": 0.80}
    else:
        return {"desc": "juvenile_range", "severity": 0.95}


def _motor_severity(motor_score: float) -> dict:
    """Score Digital Motor Assessment severity.

    100% = normal / high-performance reaction, tapping, coordination.
    < 30% = severe motor slowing, choreic instability, tremors.
    """
    if motor_score >= 85:
        return {"desc": "normal", "severity": 0.05}
    elif motor_score >= 70:
        return {"desc": "soft_signs", "severity": 0.20}
    elif motor_score >= 50:
        return {"desc": "mild_impairment", "severity": 0.40}
    elif motor_score >= 35:
        return {"desc": "moderate_impairment", "severity": 0.60}
    elif motor_score >= 20:
        return {"desc": "severe_impairment", "severity": 0.80}
    else:
        return {"desc": "very_severe", "severity": 0.95}


def _cognitive_severity(memory_score: float) -> dict:
    """Score Digital Memory & Cognitive test severity.

    100% = normal / strong episodic & working memory.
    < 35% = severe memory decline.
    """
    if memory_score >= 85:
        return {"desc": "normal", "severity": 0.05}
    elif memory_score >= 70:
        return {"desc": "mild_decline", "severity": 0.25}
    elif memory_score >= 50:
        return {"desc": "moderate_decline", "severity": 0.50}
    elif memory_score >= 35:
        return {"desc": "significant_decline", "severity": 0.75}
    else:
        return {"desc": "severe_decline", "severity": 0.90}


def _langbehn_onset(cag: float | None, age: float) -> float:
    """Estimate years to HD motor onset using Langbehn formula."""
    if cag is None or cag < 36:
        return 50.0

    try:
        median_onset = 21.54 + math.exp(9.556 - 0.146 * cag)
    except OverflowError:
        median_onset = 100.0

    median_onset = max(15.0, min(90.0, median_onset))
    return median_onset - age


def _onset_severity(years_to_onset: float) -> float:
    """Convert years-to-onset to severity score."""
    if years_to_onset > 15:
        return 0.05
    elif years_to_onset > 5:
        return 0.20
    elif years_to_onset > 0:
        return 0.40
    elif years_to_onset > -5:
        return 0.65
    elif years_to_onset > -15:
        return 0.80
    else:
        return 0.95


def _symptom_severity(symptoms: list[str] | None) -> dict:
    """Score patient-reported clinical symptoms.

    Maps a list of symptom IDs to a severity score (0.0–1.0)
    based on per-category clinical weights.

    Sources:
        - Mayo Clinic: https://www.mayoclinic.org/diseases-conditions/
          huntingtons-disease/symptoms-causes/syc-20356117
        - NIH/NINDS: https://www.ninds.nih.gov/health-information/
          disorders/huntingtons-disease
    """
    if not symptoms:
        return {"desc": "none_reported", "severity": 0.0, "count": 0}

    # Filter to known symptom IDs
    valid = [s for s in symptoms if s in _ALL_SYMPTOM_IDS]
    if not valid:
        return {"desc": "none_recognised", "severity": 0.0, "count": 0}

    valid_set = set(valid)
    category_scores = []

    for cat_name, cat_info in SYMPTOM_CATEGORIES.items():
        cat_symptoms = cat_info["symptoms"]
        cat_weight = cat_info["weight"]
        matched = sum(1 for s in cat_symptoms if s in valid_set)
        if matched > 0:
            # Ratio of symptoms in this category (0–1)
            ratio = matched / len(cat_symptoms)
            category_scores.append(ratio * cat_weight)

    if not category_scores:
        return {"desc": "none_matched", "severity": 0.0, "count": 0}

    # Raw severity is the sum of weighted category ratios, capped at 1.0
    raw_severity = min(1.0, sum(category_scores) / sum(
        c["weight"] for c in SYMPTOM_CATEGORIES.values()
    ))

    # Apply a mild scaling boost for multi-category spread
    categories_hit = sum(
        1 for cat_info in SYMPTOM_CATEGORIES.values()
        if any(s in valid_set for s in cat_info["symptoms"])
    )
    spread_bonus = 0.05 * (categories_hit - 1) if categories_hit > 1 else 0.0
    severity = min(1.0, raw_severity + spread_bonus)

    total = len(valid)
    if severity >= 0.7:
        desc = "severe_symptom_burden"
    elif severity >= 0.4:
        desc = "moderate_symptom_burden"
    elif severity >= 0.15:
        desc = "mild_symptom_burden"
    else:
        desc = "minimal_symptoms"

    logger.info(
        "Symptom severity: %d symptoms across %d categories → %.2f (%s)",
        total, categories_hit, severity, desc,
    )

    return {"desc": desc, "severity": round(severity, 4), "count": total}


def _compute_agreement(components: dict[str, float]) -> float:
    """Compute agreement between component scores."""
    values = list(components.values())
    if not values:
        return 0.5

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    agreement = math.exp(-variance * 10)
    return agreement


def _estimate_progression(
    cag: float | None,
    motor_score: float,
    functional_score: float,
    age: float,
    composite: float,
) -> tuple[float, float]:
    """Estimate progression rate at 12 and 24 months."""
    if composite < 0.2:
        base_rate = 1.5
    elif composite < 0.4:
        base_rate = 4.0
    elif composite < 0.6:
        base_rate = 6.5
    elif composite < 0.8:
        base_rate = 8.0
    else:
        base_rate = 5.0

    cag_modifier = 1.0
    if cag is not None:
        if cag >= 50:
            cag_modifier = 1.4
        elif cag >= 45:
            cag_modifier = 1.2
        elif cag >= 42:
            cag_modifier = 1.1

    age_modifier = 1.0
    if age >= 60:
        age_modifier = 1.15
    elif age >= 50:
        age_modifier = 1.05

    # If motor score is already very low, plateau factor
    if motor_score <= 15:
        ceiling_factor = 0.5
    elif motor_score <= 30:
        ceiling_factor = 0.7
    else:
        ceiling_factor = 1.0

    prog_12 = base_rate * cag_modifier * age_modifier * ceiling_factor
    prog_24 = prog_12 * 1.85

    return prog_12, prog_24


def _compute_feature_impacts(
    cag: float | None,
    motor_score: float,
    memory_score: float,
    functional_score: float,
    age: float,
    components: dict[str, float],
    weights: dict[str, float],
    symptoms: list[str] | None = None,
) -> dict[str, float]:
    """Compute SHAP-like feature importance scores."""
    baseline = 0.25

    impacts = {}

    cag_contrib = components["cag"] * weights["cag"]
    impacts["cag_repeat"] = round(cag_contrib - baseline * weights["cag"], 4)

    motor_contrib = components["motor"] * weights["motor"]
    impacts["motor_score"] = round(
        motor_contrib - baseline * weights["motor"], 4,
    )

    mem_contrib = components["memory"] * weights["memory"]
    impacts["memory_score"] = round(
        mem_contrib - baseline * weights["memory"], 4,
    )

    onset_contrib = components["onset"] * weights["onset"]
    impacts["age"] = round(
        onset_contrib - baseline * weights["onset"], 4,
    )

    # Symptom impact (only when symptoms were provided)
    if "symptoms" in components and "symptoms" in weights:
        sym_contrib = components["symptoms"] * weights["symptoms"]
        impacts["symptoms"] = round(
            sym_contrib - baseline * weights["symptoms"], 4,
        )

    return impacts


def fuse_image_clinical(
    image_hd_prob: float,
    clinical: ClinicalScore,
    image_weight: float = 0.35,
) -> ClinicalScore:
    """Fuse image model prediction with digital assessment scoring."""
    effective_image_w = image_weight * (1.0 - clinical.clinical_certainty * 0.6)
    effective_clinical_w = 1.0 - effective_image_w

    logger.info(
        "Fusion weights: clinical=%.2f, image=%.2f (certainty=%.2f)",
        effective_clinical_w, effective_image_w,
        clinical.clinical_certainty,
    )

    image_pre_prob = 1.0 - image_hd_prob
    image_early_prob = image_hd_prob * 0.6
    image_advanced_prob = image_hd_prob * 0.4

    fused_pre = (
        effective_clinical_w * clinical.pre_manifest_prob
        + effective_image_w * image_pre_prob
    )
    fused_early = (
        effective_clinical_w * clinical.early_prob
        + effective_image_w * image_early_prob
    )
    fused_advanced = (
        effective_clinical_w * clinical.advanced_prob
        + effective_image_w * image_advanced_prob
    )

    total = fused_pre + fused_early + fused_advanced
    if total > 0:
        fused_pre /= total
        fused_early /= total
        fused_advanced /= total

    probs = {
        "pre_manifest": fused_pre,
        "early": fused_early,
        "advanced": fused_advanced,
    }
    fused_stage = max(probs, key=probs.get)
    fused_confidence = max(probs.values())

    return ClinicalScore(
        stage=fused_stage,
        confidence=round(fused_confidence, 4),
        pre_manifest_prob=round(fused_pre, 4),
        early_prob=round(fused_early, 4),
        advanced_prob=round(fused_advanced, 4),
        risk_category=clinical.risk_category,
        progression_12mo=clinical.progression_12mo,
        progression_24mo=clinical.progression_24mo,
        feature_impacts=clinical.feature_impacts,
        clinical_certainty=clinical.clinical_certainty,
    )
