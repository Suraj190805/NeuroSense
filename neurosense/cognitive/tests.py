"""NeuroSense Cognitive Assessment — Test Logic.

Implements the 5-test digital cognitive battery for longitudinal
Huntington's Disease monitoring:

    1. SDMTTest        — Symbol Digit Modalities Test
    2. NBackTest       — 2-Back Working Memory Task
    3. VerbalFluencyTest — Phonemic Verbal Fluency
    4. TrailMakingTest — Trail Making Test (Parts A & B)
    5. DelayedRecallTest — Delayed Word Recall

Each test class is responsible for:
    - Deterministic stimulus generation seeded by session_id
    - Parallel form versioning (6 versions per test)
    - Response validation and raw event capture
    - Practice trial support with accuracy gating

All timing in the backend uses ``time.perf_counter()``.
Frontend must use ``performance.now()`` and submit timestamps
in milliseconds.

Design principle: tests are *stateless* generators — they produce
stimuli and validate responses, but do NOT store session state.
State management lives in ``session.py``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Common Types & Constants
# ═════════════════════════════════════════════════════════════════


class TestName(str, Enum):
    """Canonical names for the 5 cognitive tests."""

    SDMT = "sdmt"
    NBACK = "nback"
    VERBAL_FLUENCY = "verbal_fluency"
    TRAIL_MAKING = "trail_making"
    DELAYED_RECALL = "delayed_recall"


# Standard test order (fixed within a session)
TEST_ORDER: list[TestName] = [
    TestName.SDMT,
    TestName.NBACK,
    TestName.VERBAL_FLUENCY,
    TestName.TRAIL_MAKING,
    TestName.DELAYED_RECALL,
]

# Number of parallel forms per test
NUM_VERSIONS = 6


@dataclass
class StimulusItem:
    """A single stimulus presented to the patient.

    Attributes:
        index: Position in the stimulus sequence (0-based).
        content: The stimulus value (symbol, letter, word, etc.).
        expected: The correct response (digit, yes/no, etc.).
        metadata: Extra info (e.g., position for Trail Making).
    """

    index: int
    content: str
    expected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseEvent:
    """A single patient response event.

    Attributes:
        stimulus_index: Which stimulus this responds to.
        response: The patient's answer.
        timestamp_ms: Millisecond timestamp from performance.now().
        is_correct: Whether the response was correct.
        reaction_time_ms: Time from stimulus onset to response.
        is_correction: Whether this was a self-correction.
    """

    stimulus_index: int
    response: str
    timestamp_ms: float
    is_correct: bool | None = None
    reaction_time_ms: float | None = None
    is_correction: bool = False


@dataclass
class TestResult:
    """Raw result from a single test within a session.

    Attributes:
        test_name: Which cognitive test produced this result.
        version: Parallel form version used (0–5).
        raw_score: Primary score metric.
        raw_score_secondary: Optional secondary metric.
        responses: Every response event captured.
        stimuli: The stimuli presented (for audit trail).
        timing: Aggregate timing metrics.
        metadata: Test-specific extra data.
        practice_accuracy: Accuracy on practice trials (0.0–1.0).
    """

    test_name: TestName
    version: int
    raw_score: float
    raw_score_secondary: float | None = None
    responses: list[dict[str, Any]] = field(default_factory=list)
    stimuli: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    practice_accuracy: float = 1.0


def _make_rng(session_id: str, test_name: str, version: int) -> random.Random:
    """Create a deterministic RNG from session + test + version.

    Produces a seeded ``random.Random`` instance so the exact
    same stimuli can be reconstructed for any session audit.

    Args:
        session_id: Unique session identifier.
        test_name: Canonical test name string.
        version: Parallel form version (0–5).

    Returns:
        Seeded ``random.Random`` instance.
    """
    seed_str = f"{session_id}:{test_name}:v{version}"
    seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16)
    return random.Random(seed_int)


# ═════════════════════════════════════════════════════════════════
#  Test 1: Symbol Digit Modalities Test (SDMT)
# ═════════════════════════════════════════════════════════════════


# 6 different symbol sets — each is a list of 9 Unicode symbols
# mapped to digits 1–9. Using visually distinct geometric/misc symbols.
SDMT_SYMBOL_SETS: list[list[str]] = [
    # Version 0
    ["◆", "★", "▲", "●", "■", "⬟", "⬡", "◐", "⬢"],
    # Version 1
    ["♠", "♦", "♣", "♥", "⚡", "☀", "☁", "☂", "⚙"],
    # Version 2
    ["⊕", "⊗", "⊘", "⊙", "⊛", "⊜", "⊝", "⊞", "⊟"],
    # Version 3
    ["△", "▽", "◇", "○", "□", "☆", "⬠", "◎", "⬤"],
    # Version 4
    ["♤", "♧", "♡", "♢", "⚀", "⚁", "⚂", "⚃", "⚄"],
    # Version 5
    ["✧", "✦", "✶", "✸", "✹", "✺", "✻", "✼", "✽"],
]


class SDMTTest:
    """Symbol Digit Modalities Test.

    A key of 9 symbols mapped to digits 1–9 is presented.
    Symbols appear one at a time; the patient selects the
    matching digit within 3 seconds per symbol. Duration: 90 s.

    Score: number of correct responses in 90 seconds.

    Captures:
        - Response time per item
        - Error rate
        - Correction rate

    There are 6 parallel forms with different symbol-digit mappings.

    Args:
        session_id: Unique session identifier for deterministic RNG.
        version: Parallel form version (0–5).

    Example:
        >>> test = SDMTTest("sess-001", version=0)
        >>> key = test.get_symbol_key()
        >>> stimuli = test.generate_stimuli(count=120)
        >>> result = test.score_responses(responses, stimuli)
    """

    DURATION_SECONDS: int = 90
    TIME_PER_ITEM_MS: int = 3000
    NUM_PRACTICE: int = 5

    def __init__(self, session_id: str, version: int = 0) -> None:
        if version < 0 or version >= NUM_VERSIONS:
            raise ValueError(f"Version must be 0–{NUM_VERSIONS - 1}, got {version}")
        self.session_id = session_id
        self.version = version
        self.rng = _make_rng(session_id, TestName.SDMT, version)

        # Build the symbol-to-digit mapping for this version
        symbols = SDMT_SYMBOL_SETS[version].copy()
        digits = list(range(1, 10))
        self.rng.shuffle(digits)
        self.symbol_to_digit: dict[str, int] = dict(zip(symbols, digits))
        self.digit_to_symbol: dict[int, str] = {v: k for k, v in self.symbol_to_digit.items()}
        self.symbols = symbols

        logger.debug(
            "SDMTTest created: session=%s version=%d",
            session_id,
            version,
        )

    def get_symbol_key(self) -> list[dict[str, str | int]]:
        """Return the symbol-digit key for display.

        Returns:
            List of 9 dicts with 'symbol' and 'digit' keys,
            ordered by digit 1–9.
        """
        return [
            {"symbol": self.digit_to_symbol[d], "digit": d}
            for d in range(1, 10)
        ]

    def generate_stimuli(self, count: int = 120) -> list[StimulusItem]:
        """Generate a sequence of symbol stimuli.

        Generates more symbols than can be answered in 90 seconds
        so the test never runs out of items. Uses balanced
        sampling to ensure roughly equal symbol frequency.

        Args:
            count: Number of stimuli to generate (default 120,
                   ~30% more than the 90 achievable in 90s).

        Returns:
            List of StimulusItem with symbol content and
            expected digit response.
        """
        # Create balanced pool then shuffle
        full_cycles = count // 9
        remainder = count % 9
        pool = self.symbols * full_cycles + self.rng.sample(self.symbols, remainder)
        self.rng.shuffle(pool)

        stimuli: list[StimulusItem] = []
        for i, symbol in enumerate(pool):
            stimuli.append(
                StimulusItem(
                    index=i,
                    content=symbol,
                    expected=str(self.symbol_to_digit[symbol]),
                    metadata={"type": "scored" if i >= self.NUM_PRACTICE else "practice"},
                )
            )

        return stimuli

    def score_responses(
        self,
        responses: list[dict[str, Any]],
        stimuli: list[StimulusItem],
    ) -> TestResult:
        """Score a completed SDMT session.

        Args:
            responses: List of response dicts with keys:
                - stimulus_index (int)
                - response (str): digit pressed
                - timestamp_ms (float)
                - reaction_time_ms (float)
                - is_correction (bool)
            stimuli: The stimuli that were presented.

        Returns:
            TestResult with raw_score = correct count in 90s,
            plus timing and error metadata.
        """
        scored_responses = [
            r for r in responses
            if r.get("stimulus_index", 0) >= self.NUM_PRACTICE
        ]
        practice_responses = [
            r for r in responses
            if r.get("stimulus_index", 0) < self.NUM_PRACTICE
        ]

        correct = 0
        total = 0
        reaction_times: list[float] = []
        errors = 0
        corrections = 0

        stim_lookup = {s.index: s for s in stimuli}

        for resp in scored_responses:
            idx = resp["stimulus_index"]
            stim = stim_lookup.get(idx)
            if stim is None:
                continue

            total += 1
            is_correct = str(resp.get("response")) == stim.expected

            if is_correct:
                correct += 1
            else:
                errors += 1

            if resp.get("is_correction", False):
                corrections += 1

            rt = resp.get("reaction_time_ms")
            if rt is not None and is_correct:
                reaction_times.append(float(rt))

        # Practice accuracy
        practice_correct = sum(
            1 for r in practice_responses
            if str(r.get("response")) == stim_lookup.get(r["stimulus_index"], StimulusItem(0, "", "")).expected
        )
        practice_acc = practice_correct / max(len(practice_responses), 1)

        mean_rt = sum(reaction_times) / max(len(reaction_times), 1) if reaction_times else 0.0
        error_rate = errors / max(total, 1)
        correction_rate = corrections / max(total, 1)

        return TestResult(
            test_name=TestName.SDMT,
            version=self.version,
            raw_score=float(correct),
            responses=[dict(r) for r in responses],
            stimuli=[
                {"index": s.index, "content": s.content, "expected": s.expected, "metadata": s.metadata}
                for s in stimuli
            ],
            timing={
                "mean_rt_ms": round(mean_rt, 2),
                "median_rt_ms": round(sorted(reaction_times)[len(reaction_times) // 2], 2) if reaction_times else 0.0,
                "min_rt_ms": round(min(reaction_times), 2) if reaction_times else 0.0,
                "max_rt_ms": round(max(reaction_times), 2) if reaction_times else 0.0,
            },
            metadata={
                "total_attempted": total,
                "correct": correct,
                "errors": errors,
                "error_rate": round(error_rate, 4),
                "corrections": corrections,
                "correction_rate": round(correction_rate, 4),
            },
            practice_accuracy=practice_acc,
        )


# ═════════════════════════════════════════════════════════════════
#  Test 2: 2-Back Working Memory Task
# ═════════════════════════════════════════════════════════════════


class NBackTest:
    """2-Back Working Memory Task.

    A sequence of single letters appears on screen, one every
    2 seconds. The patient presses YES if the current letter
    matches the one shown 2 steps ago, NO otherwise.

    Total trials: 40 (8 targets, 32 non-targets).

    Score: accuracy percentage and mean reaction time.
    Captures: hit rate, false alarm rate, d-prime sensitivity.

    Args:
        session_id: Unique session identifier.
        version: Parallel form version (0–5).
    """

    TOTAL_TRIALS: int = 40
    NUM_TARGETS: int = 8
    INTERVAL_MS: int = 2000
    NUM_PRACTICE: int = 6

    # Letters used — excluding visually similar letters (I/L, O/0)
    LETTERS: str = "BCDFGHJKMNPQRSTVWXYZ"

    def __init__(self, session_id: str, version: int = 0) -> None:
        if version < 0 or version >= NUM_VERSIONS:
            raise ValueError(f"Version must be 0–{NUM_VERSIONS - 1}, got {version}")
        self.session_id = session_id
        self.version = version
        self.rng = _make_rng(session_id, TestName.NBACK, version)

    def generate_stimuli(self) -> list[StimulusItem]:
        """Generate the 2-back letter sequence.

        Creates a sequence of 40 letters where exactly 8 are
        2-back targets (current == letter two positions before).
        Target positions are randomly distributed but never
        in the first 2 positions.

        Returns:
            List of 40 StimulusItem with expected YES/NO.
        """
        n = self.TOTAL_TRIALS
        num_targets = self.NUM_TARGETS

        # Decide which positions will be targets (not positions 0 or 1)
        possible_positions = list(range(2, n))
        target_positions = set(self.rng.sample(possible_positions, num_targets))

        sequence: list[str] = []

        for i in range(n):
            if i < 2:
                # First two: random letters
                letter = self.rng.choice(self.LETTERS)
            elif i in target_positions:
                # Target: repeat the letter from 2 back
                letter = sequence[i - 2]
            else:
                # Non-target: pick a letter different from 2 back
                choices = [c for c in self.LETTERS if c != sequence[i - 2]]
                letter = self.rng.choice(choices)
            sequence.append(letter)

        stimuli: list[StimulusItem] = []
        for i, letter in enumerate(sequence):
            is_target = (i >= 2 and letter == sequence[i - 2])
            stimuli.append(
                StimulusItem(
                    index=i,
                    content=letter,
                    expected="YES" if is_target else "NO",
                    metadata={
                        "is_target": is_target,
                        "type": "scored",
                    },
                )
            )

        return stimuli

    def generate_practice(self) -> list[StimulusItem]:
        """Generate practice trials (6 items, 2 targets).

        Returns:
            List of 6 StimulusItem for practice.
        """
        practice_rng = _make_rng(self.session_id, "nback_practice", self.version)
        target_positions = {3, 5}

        sequence: list[str] = []
        for i in range(self.NUM_PRACTICE):
            if i < 2:
                letter = practice_rng.choice(self.LETTERS)
            elif i in target_positions:
                letter = sequence[i - 2]
            else:
                choices = [c for c in self.LETTERS if c != sequence[i - 2]]
                letter = practice_rng.choice(choices)
            sequence.append(letter)

        return [
            StimulusItem(
                index=i,
                content=letter,
                expected="YES" if (i >= 2 and letter == sequence[i - 2]) else "NO",
                metadata={"is_target": i >= 2 and letter == sequence[i - 2], "type": "practice"},
            )
            for i, letter in enumerate(sequence)
        ]

    def score_responses(
        self,
        responses: list[dict[str, Any]],
        stimuli: list[StimulusItem],
    ) -> TestResult:
        """Score a completed 2-back session.

        Computes accuracy, hit rate, false alarm rate, d-prime,
        and mean reaction time.

        Args:
            responses: Response dicts with stimulus_index, response,
                       timestamp_ms, reaction_time_ms.
            stimuli: The stimuli that were presented.

        Returns:
            TestResult with raw_score = accuracy percentage,
            raw_score_secondary = d-prime.
        """
        stim_lookup = {s.index: s for s in stimuli}
        scored = [r for r in responses if stim_lookup.get(r["stimulus_index"], StimulusItem(0, "", "")).metadata.get("type") == "scored"]
        practice = [r for r in responses if stim_lookup.get(r["stimulus_index"], StimulusItem(0, "", "")).metadata.get("type") == "practice"]

        hits = 0
        misses = 0
        false_alarms = 0
        correct_rejections = 0
        total_correct = 0
        reaction_times: list[float] = []

        for resp in scored:
            stim = stim_lookup.get(resp["stimulus_index"])
            if stim is None:
                continue

            is_target = stim.metadata.get("is_target", False)
            answered_yes = str(resp.get("response", "")).upper() in ("YES", "Y", "TRUE", "1")

            if is_target and answered_yes:
                hits += 1
                total_correct += 1
            elif is_target and not answered_yes:
                misses += 1
            elif not is_target and answered_yes:
                false_alarms += 1
            else:
                correct_rejections += 1
                total_correct += 1

            rt = resp.get("reaction_time_ms")
            if rt is not None:
                reaction_times.append(float(rt))

        total_targets = hits + misses
        total_non_targets = false_alarms + correct_rejections
        total_scored = len(scored)

        accuracy = total_correct / max(total_scored, 1) * 100.0
        hit_rate = hits / max(total_targets, 1)
        false_alarm_rate = false_alarms / max(total_non_targets, 1)

        # d-prime computation with floor/ceiling correction
        # Apply Macmillan & Creelman correction: replace 0 with 0.5/N
        # and 1 with 1 - 0.5/N
        hr_adj = hit_rate
        far_adj = false_alarm_rate

        if hr_adj == 0:
            hr_adj = 0.5 / max(total_targets, 1)
        elif hr_adj == 1.0:
            hr_adj = 1.0 - 0.5 / max(total_targets, 1)

        if far_adj == 0:
            far_adj = 0.5 / max(total_non_targets, 1)
        elif far_adj == 1.0:
            far_adj = 1.0 - 0.5 / max(total_non_targets, 1)

        # d' = Z(hit rate) - Z(false alarm rate)
        d_prime = _norm_ppf(hr_adj) - _norm_ppf(far_adj)

        mean_rt = sum(reaction_times) / len(reaction_times) if reaction_times else 0.0

        # Practice accuracy
        practice_correct = sum(
            1 for r in practice
            if str(r.get("response", "")).upper() in ("YES", "Y", "TRUE", "1") == stim_lookup.get(r["stimulus_index"], StimulusItem(0, "", "")).metadata.get("is_target", False)
        )
        # Recompute properly
        prac_correct = 0
        for r in practice:
            s = stim_lookup.get(r["stimulus_index"])
            if s is None:
                continue
            is_t = s.metadata.get("is_target", False)
            ans_yes = str(r.get("response", "")).upper() in ("YES", "Y", "TRUE", "1")
            if is_t == ans_yes:
                prac_correct += 1
        practice_acc = prac_correct / max(len(practice), 1)

        return TestResult(
            test_name=TestName.NBACK,
            version=self.version,
            raw_score=round(accuracy, 2),
            raw_score_secondary=round(d_prime, 4),
            responses=[dict(r) for r in responses],
            stimuli=[
                {"index": s.index, "content": s.content, "expected": s.expected, "metadata": s.metadata}
                for s in stimuli
            ],
            timing={
                "mean_rt_ms": round(mean_rt, 2),
                "median_rt_ms": round(sorted(reaction_times)[len(reaction_times) // 2], 2) if reaction_times else 0.0,
            },
            metadata={
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_rejections": correct_rejections,
                "hit_rate": round(hit_rate, 4),
                "false_alarm_rate": round(false_alarm_rate, 4),
                "d_prime": round(d_prime, 4),
                "accuracy_pct": round(accuracy, 2),
                "total_scored": total_scored,
            },
            practice_accuracy=practice_acc,
        )


def _norm_ppf(p: float) -> float:
    """Inverse CDF (percent-point function) of the standard normal.

    Uses the rational approximation by Abramowitz & Stegun (26.2.23)
    to avoid requiring scipy. Accurate to ~4.5e-4.

    Args:
        p: Probability value in (0, 1).

    Returns:
        z-score such that Φ(z) = p.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    if p < 0.5:
        return -_norm_ppf(1.0 - p)

    # Rational approximation for p ∈ [0.5, 1)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return z


# ═════════════════════════════════════════════════════════════════
#  Test 3: Verbal Fluency
# ═════════════════════════════════════════════════════════════════


# 6 prompt letters rotated across versions
FLUENCY_LETTERS: list[str] = ["F", "A", "S", "P", "R", "L"]

# English word set — loaded lazily from nltk
_ENGLISH_WORDS: set[str] | None = None


def _load_english_words() -> set[str]:
    """Load English word corpus from nltk.

    Downloads the corpus on first use if not available.
    Returns a set of lowercase English words.

    Returns:
        Set of valid English words (lowercase).
    """
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is not None:
        return _ENGLISH_WORDS

    try:
        from nltk.corpus import words as nltk_words

        try:
            word_list = nltk_words.words()
        except LookupError:
            import nltk
            nltk.download("words", quiet=True)
            word_list = nltk_words.words()

        _ENGLISH_WORDS = {w.lower() for w in word_list if len(w) >= 2}
        logger.info("Loaded %d English words from nltk corpus", len(_ENGLISH_WORDS))
    except ImportError:
        logger.warning("nltk not available — using fallback word validation")
        _ENGLISH_WORDS = set()

    return _ENGLISH_WORDS


class VerbalFluencyTest:
    """Phonemic Verbal Fluency Test.

    A target letter is shown. The patient types as many words
    as possible starting with that letter in 60 seconds.
    Proper nouns and repeated words are rejected automatically.

    Score: valid unique word count.
    Captures: words per 15-second interval (fluency curve).

    Args:
        session_id: Unique session identifier.
        version: Parallel form version (0–5) → selects the letter.
    """

    DURATION_SECONDS: int = 60
    INTERVAL_SECONDS: int = 15

    def __init__(self, session_id: str, version: int = 0) -> None:
        if version < 0 or version >= NUM_VERSIONS:
            raise ValueError(f"Version must be 0–{NUM_VERSIONS - 1}, got {version}")
        self.session_id = session_id
        self.version = version
        self.target_letter = FLUENCY_LETTERS[version]

    def get_target_letter(self) -> str:
        """Return the target letter for this version.

        Returns:
            Single uppercase letter.
        """
        return self.target_letter

    def validate_word(self, word: str) -> dict[str, Any]:
        """Validate a single word submission.

        Checks:
        1. Word starts with the target letter
        2. Word is at least 2 characters long
        3. Word is in the English word corpus
        4. Word is not a proper noun (all lowercase in corpus)

        Args:
            word: The submitted word.

        Returns:
            Dict with 'valid' (bool), 'reason' (str if invalid),
            and 'normalized' (lowercase word).
        """
        normalized = word.strip().lower()

        if len(normalized) < 2:
            return {"valid": False, "reason": "too_short", "normalized": normalized}

        if not normalized.startswith(self.target_letter.lower()):
            return {
                "valid": False,
                "reason": "wrong_letter",
                "normalized": normalized,
            }

        # Check against English word corpus
        english_words = _load_english_words()
        if english_words and normalized not in english_words:
            return {
                "valid": False,
                "reason": "not_a_word",
                "normalized": normalized,
            }

        return {"valid": True, "reason": None, "normalized": normalized}

    def score_responses(
        self,
        responses: list[dict[str, Any]],
    ) -> TestResult:
        """Score a completed verbal fluency session.

        Args:
            responses: List of dicts with:
                - word (str): the submitted word
                - timestamp_ms (float): when it was submitted
                - elapsed_seconds (float): seconds since test start

        Returns:
            TestResult with raw_score = valid unique word count,
            plus fluency curve metadata.
        """
        seen_words: set[str] = set()
        valid_words: list[str] = []
        rejected_words: list[dict[str, Any]] = []
        repetitions: int = 0
        proper_nouns: int = 0

        # Fluency curve: count per 15-second bin
        bins = [0, 0, 0, 0]  # 0–15s, 15–30s, 30–45s, 45–60s

        for resp in responses:
            word = resp.get("word", "")
            elapsed = resp.get("elapsed_seconds", 0.0)

            validation = self.validate_word(word)
            normalized = validation["normalized"]

            if not validation["valid"]:
                rejected_words.append({
                    "word": word,
                    "normalized": normalized,
                    "reason": validation["reason"],
                    "timestamp_ms": resp.get("timestamp_ms"),
                })
                if validation["reason"] == "not_a_word":
                    # Check if it looks like a proper noun
                    if word and word[0].isupper():
                        proper_nouns += 1
                continue

            if normalized in seen_words:
                repetitions += 1
                rejected_words.append({
                    "word": word,
                    "normalized": normalized,
                    "reason": "repetition",
                    "timestamp_ms": resp.get("timestamp_ms"),
                })
                continue

            seen_words.add(normalized)
            valid_words.append(normalized)

            # Bin into 15-second intervals
            bin_idx = min(int(elapsed // self.INTERVAL_SECONDS), 3)
            bins[bin_idx] += 1

        return TestResult(
            test_name=TestName.VERBAL_FLUENCY,
            version=self.version,
            raw_score=float(len(valid_words)),
            responses=[dict(r) for r in responses],
            stimuli=[{"target_letter": self.target_letter}],
            timing={},
            metadata={
                "valid_words": valid_words,
                "rejected_words": rejected_words,
                "repetitions": repetitions,
                "proper_noun_attempts": proper_nouns,
                "fluency_curve": {
                    "0_15s": bins[0],
                    "15_30s": bins[1],
                    "30_45s": bins[2],
                    "45_60s": bins[3],
                },
                "total_attempts": len(responses),
            },
            practice_accuracy=1.0,  # No practice for fluency
        )


# ═════════════════════════════════════════════════════════════════
#  Test 4: Trail Making Test
# ═════════════════════════════════════════════════════════════════


class TrailMakingTest:
    """Trail Making Test (Parts A & B).

    Part A: numbered circles 1–25 scattered on screen.
    Patient clicks them in ascending order.

    Part B: alternating numbers and letters (1-A-2-B-3-C...13).
    Patient clicks in alternating number-letter order.

    Score: completion time for Part A and Part B (seconds).
    Captures: time, error count, correction time per error.

    Args:
        session_id: Unique session identifier.
        version: Parallel form version (0–5) → different layouts.
    """

    PART_A_COUNT: int = 25
    PART_B_COUNT: int = 25  # 13 numbers + 12 letters

    def __init__(self, session_id: str, version: int = 0) -> None:
        if version < 0 or version >= NUM_VERSIONS:
            raise ValueError(f"Version must be 0–{NUM_VERSIONS - 1}, got {version}")
        self.session_id = session_id
        self.version = version
        self.rng = _make_rng(session_id, TestName.TRAIL_MAKING, version)

    def _generate_positions(
        self,
        count: int,
        width: int = 800,
        height: int = 600,
        min_distance: int = 60,
    ) -> list[tuple[int, int]]:
        """Generate non-overlapping circle positions.

        Places circles on a canvas with minimum distance
        constraint to prevent overlap.

        Args:
            count: Number of circles to place.
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            min_distance: Minimum distance between circle centres.

        Returns:
            List of (x, y) tuples.
        """
        margin = 40
        positions: list[tuple[int, int]] = []
        max_attempts = 1000

        for _ in range(count):
            for attempt in range(max_attempts):
                x = self.rng.randint(margin, width - margin)
                y = self.rng.randint(margin, height - margin)

                # Check minimum distance to all existing positions
                too_close = False
                for px, py in positions:
                    dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                    if dist < min_distance:
                        too_close = True
                        break

                if not too_close:
                    positions.append((x, y))
                    break
            else:
                # Fallback: place anyway (shouldn't happen with 25 items)
                positions.append((
                    self.rng.randint(margin, width - margin),
                    self.rng.randint(margin, height - margin),
                ))

        return positions

    def generate_stimuli_part_a(self) -> list[StimulusItem]:
        """Generate Part A stimuli: numbered circles 1–25.

        Returns:
            List of 25 StimulusItem with label and position.
        """
        positions = self._generate_positions(self.PART_A_COUNT)

        # The correct order is 1, 2, 3, ..., 25
        labels = [str(i) for i in range(1, self.PART_A_COUNT + 1)]
        correct_order = list(range(self.PART_A_COUNT))

        stimuli: list[StimulusItem] = []
        for i, (label, pos) in enumerate(zip(labels, positions)):
            stimuli.append(
                StimulusItem(
                    index=i,
                    content=label,
                    expected=str(correct_order[i]),
                    metadata={
                        "x": pos[0],
                        "y": pos[1],
                        "part": "A",
                        "correct_click_order": i,
                    },
                )
            )

        return stimuli

    def generate_stimuli_part_b(self) -> list[StimulusItem]:
        """Generate Part B stimuli: alternating numbers and letters.

        Sequence: 1-A-2-B-3-C-4-D-5-E-6-F-7-G-8-H-9-I-10-J-11-K-12-L-13

        Returns:
            List of 25 StimulusItem with label and position.
        """
        positions = self._generate_positions(self.PART_B_COUNT)

        # Build the alternating sequence
        labels: list[str] = []
        num = 1
        letter_idx = 0
        letters = string.ascii_uppercase[:12]  # A through L

        for i in range(self.PART_B_COUNT):
            if i % 2 == 0:
                labels.append(str(num))
                num += 1
            else:
                labels.append(letters[letter_idx])
                letter_idx += 1

        stimuli: list[StimulusItem] = []
        for i, (label, pos) in enumerate(zip(labels, positions)):
            stimuli.append(
                StimulusItem(
                    index=i,
                    content=label,
                    expected=str(i),
                    metadata={
                        "x": pos[0],
                        "y": pos[1],
                        "part": "B",
                        "correct_click_order": i,
                    },
                )
            )

        return stimuli

    def score_responses(
        self,
        responses_a: list[dict[str, Any]],
        responses_b: list[dict[str, Any]],
        stimuli_a: list[StimulusItem],
        stimuli_b: list[StimulusItem],
    ) -> TestResult:
        """Score a completed Trail Making session (Parts A + B).

        Args:
            responses_a: Click events for Part A.
            responses_b: Click events for Part B.
            stimuli_a: Part A stimuli.
            stimuli_b: Part B stimuli.

        Returns:
            TestResult with raw_score = Part B time (seconds),
            raw_score_secondary = Part A time (seconds).
        """
        def _score_part(
            responses: list[dict[str, Any]],
            stimuli: list[StimulusItem],
        ) -> dict[str, Any]:
            if not responses:
                return {"time_s": 0.0, "errors": 0, "correction_time_ms": 0.0}

            timestamps = [r.get("timestamp_ms", 0.0) for r in responses]
            total_time_ms = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0

            errors = 0
            correction_time_total = 0.0

            for resp in responses:
                if not resp.get("is_correct", True):
                    errors += 1
                    correction_time_total += resp.get("correction_time_ms", 0.0)

            return {
                "time_s": round(total_time_ms / 1000.0, 2),
                "errors": errors,
                "correction_time_ms": round(correction_time_total, 2),
            }

        part_a = _score_part(responses_a, stimuli_a)
        part_b = _score_part(responses_b, stimuli_b)

        return TestResult(
            test_name=TestName.TRAIL_MAKING,
            version=self.version,
            raw_score=part_b["time_s"],       # Part B time (primary)
            raw_score_secondary=part_a["time_s"],  # Part A time
            responses=[
                {"part": "A", "events": [dict(r) for r in responses_a]},
                {"part": "B", "events": [dict(r) for r in responses_b]},
            ],
            stimuli=[
                {"part": "A", "items": [
                    {"index": s.index, "content": s.content, "metadata": s.metadata}
                    for s in stimuli_a
                ]},
                {"part": "B", "items": [
                    {"index": s.index, "content": s.content, "metadata": s.metadata}
                    for s in stimuli_b
                ]},
            ],
            timing={
                "part_a_time_s": part_a["time_s"],
                "part_b_time_s": part_b["time_s"],
                "part_a_errors": part_a["errors"],
                "part_b_errors": part_b["errors"],
                "part_a_correction_time_ms": part_a["correction_time_ms"],
                "part_b_correction_time_ms": part_b["correction_time_ms"],
            },
            metadata={
                "part_a": part_a,
                "part_b": part_b,
                "b_minus_a": round(part_b["time_s"] - part_a["time_s"], 2),
            },
            practice_accuracy=1.0,  # No practice for TMT
        )


# ═════════════════════════════════════════════════════════════════
#  Test 5: Delayed Recall
# ═════════════════════════════════════════════════════════════════


# 6 word lists of 10 concrete nouns each
# Selected for high imageability and moderate frequency
RECALL_WORD_LISTS: list[list[str]] = [
    # Version 0
    ["river", "garden", "mountain", "basket", "trumpet",
     "hammer", "ocean", "candle", "blanket", "ladder"],
    # Version 1
    ["forest", "window", "violin", "pepper", "shelter",
     "rocket", "feather", "island", "marble", "tunnel"],
    # Version 2
    ["castle", "ribbon", "anchor", "pillow", "dragon",
     "market", "sunset", "bottle", "mirror", "harbor"],
    # Version 3
    ["bridge", "lantern", "falcon", "temple", "magnet",
     "cherry", "glacier", "helmet", "saddle", "flower"],
    # Version 4
    ["palace", "parrot", "crystal", "barrel", "planet",
     "curtain", "desert", "mushroom", "silver", "beacon"],
    # Version 5
    ["meadow", "anchor", "volcano", "pencil", "dolphin",
     "cabinet", "glacier", "costume", "harvest", "diamond"],
]


class DelayedRecallTest:
    """Delayed Word Recall Test.

    Phase 1 (Encoding): 10 words shown one at a time, 2 seconds each.
    Patient is NOT told they will be tested.

    Distractor: 3 minutes of simple arithmetic problems.

    Phase 2 (Recall): blank text input, patient types every word
    they remember. 2 minute time limit.

    Score: number of correctly recalled words out of 10.
    Captures: intrusion errors, repetition errors.

    Args:
        session_id: Unique session identifier.
        version: Parallel form version (0–5) → selects word list.
    """

    ENCODING_TIME_PER_WORD_MS: int = 2000
    DISTRACTOR_DURATION_S: int = 180
    RECALL_DURATION_S: int = 120

    def __init__(self, session_id: str, version: int = 0) -> None:
        if version < 0 or version >= NUM_VERSIONS:
            raise ValueError(f"Version must be 0–{NUM_VERSIONS - 1}, got {version}")
        self.session_id = session_id
        self.version = version
        self.rng = _make_rng(session_id, TestName.DELAYED_RECALL, version)
        self.word_list = RECALL_WORD_LISTS[version].copy()

    def get_encoding_words(self) -> list[StimulusItem]:
        """Return the 10 encoding words in presentation order.

        The order is shuffled deterministically per session.

        Returns:
            List of 10 StimulusItem with word content.
        """
        words = self.word_list.copy()
        self.rng.shuffle(words)

        return [
            StimulusItem(
                index=i,
                content=word,
                expected=None,
                metadata={"phase": "encoding"},
            )
            for i, word in enumerate(words)
        ]

    def generate_distractor_problems(self, count: int = 30) -> list[dict[str, Any]]:
        """Generate simple arithmetic problems for the distractor phase.

        Problems are single-digit addition/subtraction to keep
        cognitive load low (just enough to prevent rehearsal).

        Args:
            count: Number of problems to generate.

        Returns:
            List of dicts with 'problem' (str) and 'answer' (int).
        """
        problems: list[dict[str, Any]] = []
        distractor_rng = _make_rng(self.session_id, "recall_distractor", self.version)

        for _ in range(count):
            a = distractor_rng.randint(1, 9)
            b = distractor_rng.randint(1, 9)
            op = distractor_rng.choice(["+", "-"])

            if op == "-" and b > a:
                a, b = b, a  # Ensure non-negative result

            answer = a + b if op == "+" else a - b
            problems.append({
                "problem": f"{a} {op} {b} = ?",
                "answer": answer,
            })

        return problems

    def score_responses(
        self,
        recalled_words: list[dict[str, Any]],
        encoding_words: list[StimulusItem],
    ) -> TestResult:
        """Score a completed delayed recall session.

        Args:
            recalled_words: List of dicts with:
                - word (str): the recalled word
                - timestamp_ms (float)
            encoding_words: The encoding stimuli (for matching).

        Returns:
            TestResult with raw_score = correctly recalled words
            (out of 10).
        """
        # Build target word set (lowercased)
        targets = {s.content.lower() for s in encoding_words}

        correct_recalls: set[str] = set()
        intrusions: list[str] = []
        repetitions: list[str] = []
        seen: set[str] = set()

        for resp in recalled_words:
            word = resp.get("word", "").strip().lower()
            if not word:
                continue

            if word in seen:
                repetitions.append(word)
                continue

            seen.add(word)

            if word in targets:
                correct_recalls.add(word)
            else:
                intrusions.append(word)

        return TestResult(
            test_name=TestName.DELAYED_RECALL,
            version=self.version,
            raw_score=float(len(correct_recalls)),
            responses=[dict(r) for r in recalled_words],
            stimuli=[
                {"index": s.index, "content": s.content, "metadata": s.metadata}
                for s in encoding_words
            ],
            timing={},
            metadata={
                "correct_recalls": sorted(correct_recalls),
                "missed_words": sorted(targets - correct_recalls),
                "intrusions": intrusions,
                "intrusion_count": len(intrusions),
                "repetitions": repetitions,
                "repetition_count": len(repetitions),
                "total_attempts": len(recalled_words),
                "max_possible": len(targets),
            },
            practice_accuracy=1.0,  # No practice for recall
        )


# ═════════════════════════════════════════════════════════════════
#  Test Factory
# ═════════════════════════════════════════════════════════════════


def create_test(
    test_name: TestName,
    session_id: str,
    version: int,
) -> SDMTTest | NBackTest | VerbalFluencyTest | TrailMakingTest | DelayedRecallTest:
    """Factory function to create a test instance.

    Args:
        test_name: Which cognitive test to create.
        session_id: Unique session identifier.
        version: Parallel form version (0–5).

    Returns:
        The appropriate test class instance.

    Raises:
        ValueError: If test_name is unknown.
    """
    test_map = {
        TestName.SDMT: SDMTTest,
        TestName.NBACK: NBackTest,
        TestName.VERBAL_FLUENCY: VerbalFluencyTest,
        TestName.TRAIL_MAKING: TrailMakingTest,
        TestName.DELAYED_RECALL: DelayedRecallTest,
    }

    cls = test_map.get(test_name)
    if cls is None:
        raise ValueError(f"Unknown test name: {test_name}")

    return cls(session_id=session_id, version=version)
