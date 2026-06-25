"""Psycholinguistic analysis schemas for Project ALICE.

Defines the typed contract produced by the psycholinguistic analyzer
(``backend/ml-inference/app/pipelines/psycholinguistic``). The analyzer is one
of the two highest-weighted analysis vectors -- linguistic features are the
strongest single modality (~80% accuracy; Li & Abouelenien 2024) and the
linguistic vector carries 0.30 of the late-fusion ensemble weight, tied with
facial Action Units (CLAUDE.md "The Five Analysis Vectors").

Eight deception-relevant dimensions are scored independently on a 0-100 scale,
then combined into a single ``composite_score`` (CLAUDE.md "Psycholinguistic
Analysis Stack"):

    1. pronoun_shift          first-person-singular pattern shifts
    2. hedging                modal verbs + epistemic phrases
    3. cognitive_complexity   subordinate-clause depth
    4. emotional_distribution anxiety/anger vs. family/home word balance
    5. disfluency             filler-word frequency
    6. negation               negation arc frequency
    7. detail_specificity     named-entity density
    8. certainty              over-certain vs. tentative language

A score is a *deviation* signal, never a verdict (CLAUDE.md: "ALICE is not a
lie detector"). The objects are frozen so a score can be shared across the
late-fusion ensemble and SHAP explainer without defensive copies.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PsycholinguisticDimension(BaseModel):
    """A single scored linguistic dimension with supporting evidence.

    ``score`` is a 0-100 anomaly signal for this dimension only -- higher means
    further from the (eventual) per-contact baseline in the deception-indicative
    direction defined for that dimension. ``evidence`` carries human-readable
    diagnostics (ratios, counts, detected tokens) used by the SHAP explainer and
    developer CLIs; it is never shown raw to end users (CLAUDE.md invariant #5).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(
        ge=0.0,
        le=100.0,
        description="0-100 anomaly signal for this dimension (higher = more anomalous).",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Human-readable diagnostics (ratios, counts, detected tokens).",
    )


class PsycholinguisticScore(BaseModel):
    """Composite psycholinguistic score across all eight dimensions.

    The field names mirror CLAUDE.md's linguistic vector exactly; do not rename
    them without updating the ensemble feature map. ``composite_score`` is the
    weighted average computed by ``PsycholinguisticAnalyzer.analyze`` (equal
    sub-dimension weights for Day 1). ``confidence`` reflects how many sessions
    of per-contact history back the score, never the model's certainty about a
    "lie" -- ALICE reports deviation, not truth (CLAUDE.md).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pronoun_shift_score: PsycholinguisticDimension
    hedging_score: PsycholinguisticDimension
    cognitive_complexity_score: PsycholinguisticDimension
    emotional_distribution_score: PsycholinguisticDimension
    disfluency_score: PsycholinguisticDimension
    negation_score: PsycholinguisticDimension
    detail_specificity_score: PsycholinguisticDimension
    certainty_score: PsycholinguisticDimension

    composite_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Weighted average of the eight dimension scores (0-100).",
    )
    statement_count: int = Field(
        ge=0,
        description="Number of statements analyzed.",
    )
    baseline_available: bool = Field(
        default=False,
        description="True once a per-contact baseline has been established.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Session-history confidence: low (1 session), medium (2-3), "
            "high (4+). Not a probability of deception."
        ),
    )
