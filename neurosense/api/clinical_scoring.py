"""Clinical HD Scoring Engine.

Computes Huntington's Disease staging from clinical biomarkers
using established neurological criteria. This provides a robust
clinical signal that can be fused with image model predictions.

Criteria used:
    - Shoulson-Fahn TFC staging (TFC 11-13 = Stage I, etc.)
    - CAG repeat penetrance levels
    - UHDRS Total Motor Score severity bands
    - Langbehn age-adjusted onset estimation
    - Cognitive assessment decline mapping

References:
    - Shoulson I, Fahn S (1979). Huntington disease: clinical care
      and evaluation. Neurology.
    - Langbehn DR et al. (2004). A new model for prediction of the
      age of onset and penetrance for HD. Clinical Genetics.
    - Ross CA et al. (2014). Huntington disease: natural history,
      biomarkers and prospects for therapeutics. Nature Reviews.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ClinicalScore:
    """Result of clinical HD scoring.

    Attributes:
        stage: Predicted HD stage (pre_manifest / early / advanced).
        confidence: Confidence in the staging (0.0–1.0).
        pre_manifest_prob: Probability of pre-manifest stage.
        early_prob: Probability of early HD stage.
        advanced_prob: Probability of advanced HD stage.
        risk_category: Risk level (low / medium / high).
        progression_12mo: Expected 12-month UHDRS-TMS change.
        progression_24mo: Expected 24-month UHDRS-TMS change.
        feature_impacts: Dict of feature name → impact score.
        clinical_certainty: How strongly clinical data indicates
            a definitive staging (0.0–1.0). Used to weight
            clinical vs image model in fusion.
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
    uhdrs_motor: float,
    uhdrs_cognitive: float,
    tfc_score: float,
    age: float,
) -> ClinicalScore:
    """Compute HD staging from clinical biomarkers.

    Uses a multi-factor scoring system based on established
    neurological criteria to determine HD stage, confidence,
    and progression forecast.

    Args:
        cag_repeat: CAG trinucleotide repeat count (36–120).
        uhdrs_motor: UHDRS Total Motor Score (0–124).
        uhdrs_cognitive: UHDRS Cognitive Assessment score.
        tfc_score: Total Functional Capacity (0–13).
        age: Patient age in years (18–90).

    Returns:
        ClinicalScore with staging, probabilities, and forecasts.
    """
    # ─── 1. Shoulson-Fahn TFC Staging ───
    # This is the gold standard for HD staging
    tfc_stage = _tfc_to_shoulson_fahn(tfc_score)

    # ─── 2. CAG Repeat Analysis ───
    cag_score = _cag_severity(cag_repeat)

    # ─── 3. UHDRS Motor Severity ───
    motor_score = _motor_severity(uhdrs_motor)

    # ─── 4. Cognitive Assessment ───
    cognitive_score = _cognitive_severity(uhdrs_cognitive)

    # ─── 5. Age-CAG Onset Estimation (Langbehn formula) ───
    years_to_onset = _langbehn_onset(cag_repeat, age)

    # ─── 6. Composite Scoring ───
    # Weight the different indicators
    # TFC is the most clinically validated staging tool
    # CAG and motor are strong secondary indicators
    weights = {
        "tfc": 0.30,
        "cag": 0.25,
        "motor": 0.25,
        "cognitive": 0.10,
        "onset": 0.10,
    }

    # Each component produces a severity score (0=normal, 1=severe)
    components = {
        "tfc": tfc_stage["severity"],
        "cag": cag_score["severity"],
        "motor": motor_score["severity"],
        "cognitive": cognitive_score["severity"],
        "onset": _onset_severity(years_to_onset),
    }

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

    # Confidence is the max probability
    confidence = max(pre_prob, early_prob, advanced_prob)

    # ─── 8. Clinical Certainty ───
    # How strongly do clinical indicators agree on the staging?
    # High certainty when multiple indicators point to the same stage
    agreement = _compute_agreement(components)
    clinical_certainty = min(1.0, agreement * 0.8 + composite * 0.2)

    # Boost certainty for extreme values
    if cag_repeat >= 50 and uhdrs_motor >= 60:
        clinical_certainty = max(clinical_certainty, 0.90)
    elif cag_repeat >= 45 and uhdrs_motor >= 40:
        clinical_certainty = max(clinical_certainty, 0.75)
    elif cag_repeat <= 39 and uhdrs_motor <= 10:
        clinical_certainty = max(clinical_certainty, 0.70)

    # ─── 9. Progression Forecast ───
    prog_12, prog_24 = _estimate_progression(
        cag_repeat, uhdrs_motor, tfc_score, age, composite,
    )

    # ─── 10. Risk Category ───
    if prog_12 < 3.0:
        risk = "low"
    elif prog_12 < 8.0:
        risk = "medium"
    else:
        risk = "high"

    # ─── 11. Feature Impacts ───
    # SHAP-like attribution: how much each feature
    # contributed to the disease severity score
    feature_impacts = _compute_feature_impacts(
        cag_repeat, uhdrs_motor, uhdrs_cognitive,
        tfc_score, age, components, weights,
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


def _tfc_to_shoulson_fahn(tfc: float) -> dict:
    """Map TFC score to Shoulson-Fahn HD staging.

    Standard clinical staging:
        Stage I:   TFC 11–13 (early, independent)
        Stage II:  TFC 7–10 (reduced capacity)
        Stage III: TFC 3–6  (dependent)
        Stage IV:  TFC 1–2  (severe dependency)
        Stage V:   TFC 0    (total care)

    Returns:
        Dict with stage name and severity (0–1).
    """
    if tfc >= 11:
        return {"stage": "I", "severity": 0.10}
    elif tfc >= 7:
        return {"stage": "II", "severity": 0.35}
    elif tfc >= 3:
        return {"stage": "III", "severity": 0.65}
    elif tfc >= 1:
        return {"stage": "IV", "severity": 0.85}
    else:
        return {"stage": "V", "severity": 1.0}


def _cag_severity(cag: float) -> dict:
    """Score CAG repeat severity.

    Clinical significance:
        36–39: Reduced penetrance (may or may not develop HD)
        40–44: Full penetrance, typical adult onset
        45–49: Full penetrance, earlier onset likely
        50–59: High repeat, earlier onset + faster progression
        60+:   Juvenile HD range

    Returns:
        Dict with description and severity (0–1).
    """
    if cag < 36:
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


def _motor_severity(motor: float) -> dict:
    """Score UHDRS Total Motor Score severity.

    Clinical bands:
        0–5:    Normal / minimal signs
        6–15:   Soft signs (possible pre-manifest)
        16–30:  Mild motor impairment (early HD)
        31–60:  Moderate motor impairment
        61–90:  Severe motor impairment
        91–124: Very severe / end-stage motor

    Returns:
        Dict with description and severity (0–1).
    """
    if motor <= 5:
        return {"desc": "normal", "severity": 0.05}
    elif motor <= 15:
        return {"desc": "soft_signs", "severity": 0.20}
    elif motor <= 30:
        return {"desc": "mild", "severity": 0.40}
    elif motor <= 60:
        return {"desc": "moderate", "severity": 0.60}
    elif motor <= 90:
        return {"desc": "severe", "severity": 0.80}
    else:
        return {"desc": "very_severe", "severity": 0.95}


def _cognitive_severity(cognitive: float) -> dict:
    """Score UHDRS cognitive assessment severity.

    Higher scores indicate better cognitive function.
    Normal composite typically 180–250+.
    HD patients show progressive decline.

    Returns:
        Dict with description and severity (0–1).
    """
    if cognitive >= 200:
        return {"desc": "normal", "severity": 0.05}
    elif cognitive >= 160:
        return {"desc": "mild_decline", "severity": 0.25}
    elif cognitive >= 120:
        return {"desc": "moderate_decline", "severity": 0.50}
    elif cognitive >= 80:
        return {"desc": "significant_decline", "severity": 0.75}
    else:
        return {"desc": "severe_decline", "severity": 0.90}


def _langbehn_onset(cag: float, age: float) -> float:
    """Estimate years to HD motor onset using Langbehn formula.

    Based on Langbehn et al. (2004) parametric survival model.
    Estimates the expected age of motor onset for a given CAG
    repeat length, then computes years remaining.

    A simplified version of the published model:
        median_onset_age ≈ 21.54 + exp(9.556 - 0.146 * CAG)

    Args:
        cag: CAG repeat count.
        age: Current age.

    Returns:
        Estimated years to onset. Negative = past expected onset.
    """
    if cag < 36:
        return 50.0  # Not a carrier

    # Simplified Langbehn onset prediction
    try:
        median_onset = 21.54 + math.exp(9.556 - 0.146 * cag)
    except OverflowError:
        median_onset = 100.0  # Very low CAG

    # Clamp to reasonable range
    median_onset = max(15.0, min(90.0, median_onset))

    return median_onset - age


def _onset_severity(years_to_onset: float) -> float:
    """Convert years-to-onset to severity score.

    Args:
        years_to_onset: Estimated years to motor onset.
            Negative = past expected onset.

    Returns:
        Severity score (0–1). Higher = more severe.
    """
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


def _compute_agreement(components: dict[str, float]) -> float:
    """Compute agreement between component scores.

    High agreement = all components suggest similar severity
    = high clinical certainty.

    Args:
        components: Dict of component name → severity (0–1).

    Returns:
        Agreement score (0–1). 1.0 = perfect agreement.
    """
    values = list(components.values())
    if not values:
        return 0.5

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)

    # Low variance = high agreement
    # Scale: variance of 0 → agreement of 1.0
    #         variance of 0.1 → agreement of ~0.5
    agreement = math.exp(-variance * 10)
    return agreement


def _estimate_progression(
    cag: float,
    motor: float,
    tfc: float,
    age: float,
    composite: float,
) -> tuple[float, float]:
    """Estimate UHDRS-TMS progression at 12 and 24 months.

    Based on published HD natural history data:
    - Pre-manifest: 1–3 points/year TMS increase
    - Early HD: 4–8 points/year
    - Advanced: 5–12 points/year (can plateau in late stages)

    CAG repeat influences progression rate (higher = faster).

    Args:
        cag: CAG repeat count.
        motor: Current UHDRS-TMS.
        tfc: Current TFC score.
        age: Patient age.
        composite: Composite severity score.

    Returns:
        Tuple of (12-month change, 24-month change).
    """
    # Base progression rate from composite severity
    if composite < 0.2:
        base_rate = 1.5   # Pre-manifest: slow
    elif composite < 0.4:
        base_rate = 4.0   # Early: moderate
    elif composite < 0.6:
        base_rate = 6.5   # Mid-stage: faster
    elif composite < 0.8:
        base_rate = 8.0   # Advanced: significant
    else:
        base_rate = 5.0   # Late stage: may plateau

    # CAG modifier: higher CAG = faster progression
    cag_modifier = 1.0
    if cag >= 50:
        cag_modifier = 1.4
    elif cag >= 45:
        cag_modifier = 1.2
    elif cag >= 42:
        cag_modifier = 1.1

    # Age modifier: older patients may progress faster
    age_modifier = 1.0
    if age >= 60:
        age_modifier = 1.15
    elif age >= 50:
        age_modifier = 1.05

    # Ceiling effect: if already high TMS, less room to worsen
    if motor >= 100:
        ceiling_factor = 0.5
    elif motor >= 80:
        ceiling_factor = 0.7
    else:
        ceiling_factor = 1.0

    prog_12 = base_rate * cag_modifier * age_modifier * ceiling_factor
    prog_24 = prog_12 * 1.85  # Slightly less than 2x (some non-linearity)

    return prog_12, prog_24


def _compute_feature_impacts(
    cag: float,
    motor: float,
    cognitive: float,
    tfc: float,
    age: float,
    components: dict[str, float],
    weights: dict[str, float],
) -> dict[str, float]:
    """Compute SHAP-like feature importance scores.

    Shows how much each clinical feature contributed to the
    overall disease severity assessment. Positive values
    indicate the feature increased the predicted severity.

    Args:
        cag: CAG repeat count.
        motor: UHDRS motor score.
        cognitive: UHDRS cognitive score.
        tfc: TFC score.
        age: Patient age.
        components: Component severity scores.
        weights: Component weights.

    Returns:
        Dict of feature_name → impact score.
    """
    # Baseline: average severity that would give a neutral prediction
    baseline = 0.25  # Below this = pre-manifest

    impacts = {}

    # CAG impact: how much CAG pushes severity above baseline
    cag_contrib = components["cag"] * weights["cag"]
    impacts["cag_repeat"] = round(cag_contrib - baseline * weights["cag"], 4)

    # Motor impact
    motor_contrib = components["motor"] * weights["motor"]
    impacts["uhdrs_motor"] = round(
        motor_contrib - baseline * weights["motor"], 4,
    )

    # Cognitive impact
    cog_contrib = components["cognitive"] * weights["cognitive"]
    impacts["uhdrs_cognitive"] = round(
        cog_contrib - baseline * weights["cognitive"], 4,
    )

    # TFC impact (inverted: low TFC = high severity)
    tfc_contrib = components["tfc"] * weights["tfc"]
    impacts["tfc_score"] = round(
        tfc_contrib - baseline * weights["tfc"], 4,
    )

    # Age/onset impact
    onset_contrib = components["onset"] * weights["onset"]
    impacts["age"] = round(
        onset_contrib - baseline * weights["onset"], 4,
    )

    return impacts


def fuse_image_clinical(
    image_hd_prob: float,
    clinical: ClinicalScore,
    image_weight: float = 0.35,
) -> ClinicalScore:
    """Fuse image model prediction with clinical scoring.

    When clinical certainty is high, clinical data dominates.
    When clinical data is ambiguous, image model has more weight.

    The fusion uses adaptive weighting based on clinical certainty:
        effective_image_weight = image_weight * (1 - clinical_certainty * 0.6)
        effective_clinical_weight = 1 - effective_image_weight

    Args:
        image_hd_prob: Image model probability of disease (0–1).
        clinical: Clinical scoring result.
        image_weight: Base weight for image model (0–1).

    Returns:
        Updated ClinicalScore with fused probabilities.
    """
    # Adaptive weighting: reduce image weight when clinical is certain
    effective_image_w = image_weight * (1.0 - clinical.clinical_certainty * 0.6)
    effective_clinical_w = 1.0 - effective_image_w

    logger.info(
        "Fusion weights: clinical=%.2f, image=%.2f (certainty=%.2f)",
        effective_clinical_w, effective_image_w,
        clinical.clinical_certainty,
    )

    # Map image_hd_prob to 3-stage probabilities
    # If image says HD detected (high prob), split between early/advanced
    image_pre_prob = 1.0 - image_hd_prob
    image_early_prob = image_hd_prob * 0.6
    image_advanced_prob = image_hd_prob * 0.4

    # Fuse
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

    # Normalize
    total = fused_pre + fused_early + fused_advanced
    if total > 0:
        fused_pre /= total
        fused_early /= total
        fused_advanced /= total

    # Determine stage from fused probabilities
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
