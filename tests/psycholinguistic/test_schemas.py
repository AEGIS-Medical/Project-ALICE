"""Tests for P2-S1: psycholinguistic Pydantic schemas.

The schemas are the typed contract every dimension scorer and the composite
``analyze`` method produce. They must enforce the 0-100 score range, expose
all eight CLAUDE.md linguistic dimensions, and stay immutable so a score
object can be shared across the late-fusion ensemble without defensive copies.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.shared.schemas.psycholinguistic import (
    PsycholinguisticDimension,
    PsycholinguisticScore,
)


# The eight dimension fields, named to match CLAUDE.md's linguistic vector.
_DIMENSION_FIELDS = (
    "pronoun_shift_score",
    "hedging_score",
    "cognitive_complexity_score",
    "emotional_distribution_score",
    "disfluency_score",
    "negation_score",
    "detail_specificity_score",
    "certainty_score",
)


def _dim(score: float = 50.0) -> PsycholinguisticDimension:
    return PsycholinguisticDimension(score=score, evidence=["ratio=0.05"])


def _score(**overrides) -> PsycholinguisticScore:
    kwargs = {field: _dim() for field in _DIMENSION_FIELDS}
    kwargs.update(
        composite_score=50.0,
        statement_count=3,
        baseline_available=False,
        confidence="low",
    )
    kwargs.update(overrides)
    return PsycholinguisticScore(**kwargs)


def test_valid_schema_construction():
    score = _score()
    # All eight dimensions present and typed as PsycholinguisticDimension.
    for field in _DIMENSION_FIELDS:
        assert isinstance(getattr(score, field), PsycholinguisticDimension)
    assert score.composite_score == 50.0
    assert score.statement_count == 3
    assert score.baseline_available is False
    assert score.confidence == "low"


def test_composite_score_range():
    assert _score(composite_score=0.0).composite_score == 0.0
    assert _score(composite_score=100.0).composite_score == 100.0
    with pytest.raises(ValidationError):
        _score(composite_score=-1.0)
    with pytest.raises(ValidationError):
        _score(composite_score=100.1)


def test_invalid_score_raises():
    with pytest.raises(ValidationError):
        PsycholinguisticDimension(score=-5.0, evidence=[])
    with pytest.raises(ValidationError):
        PsycholinguisticDimension(score=150.0, evidence=[])


def test_schema_is_frozen():
    score = _score()
    with pytest.raises(ValidationError):
        score.composite_score = 75.0
    dim = _dim()
    with pytest.raises(ValidationError):
        dim.score = 10.0


def test_confidence_literal_enforced():
    with pytest.raises(ValidationError):
        _score(confidence="extreme")
