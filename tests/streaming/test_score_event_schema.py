"""Contract tests for the ScoreEvent streaming schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.shared.schemas.score_event import (
    ScoreEvent,
    ScoreEventKind,
    StreamScorerConfig,
    validate_event_stream,
)

from .conftest import make_pscore


def _event(
    kind: ScoreEventKind = ScoreEventKind.INTERIM,
    t: float = 5.0,
    composite: float = 50.0,
    recent: bool = False,
) -> ScoreEvent:
    score = make_pscore(composite=composite)
    return ScoreEvent(
        kind=kind,
        stream_time_seconds=t,
        cumulative=score,
        recent=make_pscore(composite=61.0) if recent else None,
        vector_scores={"psycholinguistic": score.composite_score},
        statement_count_so_far=score.statement_count,
        baseline_available=score.baseline_available,
        confidence=score.confidence,
    )


def test_defaults_and_kinds():
    ev = _event()
    assert ev.schema_version == 1
    assert ev.kind is ScoreEventKind.INTERIM
    assert ScoreEventKind("final") is ScoreEventKind.FINAL
    assert ev.recent is None


def test_event_is_frozen_and_forbids_extras():
    ev = _event()
    with pytest.raises(ValidationError):
        ev.kind = ScoreEventKind.FINAL  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ScoreEvent(
            kind=ScoreEventKind.INTERIM,
            stream_time_seconds=1.0,
            cumulative=make_pscore(),
            vector_scores={"psycholinguistic": 50.0},
            statement_count_so_far=1,
            baseline_available=False,
            confidence="low",
            surprise_field=1,  # type: ignore[call-arg]
        )


def test_final_must_not_carry_recent():
    with pytest.raises(ValidationError, match="recent"):
        _event(kind=ScoreEventKind.FINAL, recent=True)


def test_interim_may_carry_recent():
    ev = _event(recent=True)
    assert ev.recent is not None
    assert ev.recent.composite_score == 61.0


def test_vector_scores_must_match_cumulative_composite():
    score = make_pscore(composite=42.0)
    with pytest.raises(ValidationError, match="psycholinguistic"):
        ScoreEvent(
            kind=ScoreEventKind.INTERIM,
            stream_time_seconds=5.0,
            cumulative=score,
            vector_scores={"psycholinguistic": 41.0},
            statement_count_so_far=3,
            baseline_available=False,
            confidence="low",
        )
    with pytest.raises(ValidationError, match="psycholinguistic"):
        ScoreEvent(
            kind=ScoreEventKind.INTERIM,
            stream_time_seconds=5.0,
            cumulative=score,
            vector_scores={},
            statement_count_so_far=3,
            baseline_available=False,
            confidence="low",
        )


def test_negative_stream_time_rejected():
    with pytest.raises(ValidationError):
        _event(t=-0.1)


def test_config_defaults_and_bounds():
    cfg = StreamScorerConfig()
    assert cfg.tick_seconds == 5.0
    assert cfg.recent_window_seconds == 30.0
    assert cfg.min_recent_statements == 2

    with pytest.raises(ValidationError):
        StreamScorerConfig(tick_seconds=0.0)
    with pytest.raises(ValidationError):
        StreamScorerConfig(min_recent_statements=0)
    with pytest.raises(ValidationError, match="recent_window_seconds"):
        StreamScorerConfig(tick_seconds=10.0, recent_window_seconds=5.0)
    # Window may equal the tick exactly.
    assert StreamScorerConfig(
        tick_seconds=5.0, recent_window_seconds=5.0
    ).recent_window_seconds == 5.0


def test_config_is_frozen():
    cfg = StreamScorerConfig()
    with pytest.raises(ValidationError):
        cfg.tick_seconds = 1.0  # type: ignore[misc]


def test_validate_event_stream_accepts_valid_and_empty():
    validate_event_stream([])
    validate_event_stream(
        [_event(t=5.0), _event(t=10.0), _event(kind=ScoreEventKind.FINAL, t=12.0)]
    )


def test_validate_event_stream_rejects_bad_streams():
    final = _event(kind=ScoreEventKind.FINAL, t=12.0)
    with pytest.raises(ValueError, match="increasing"):
        validate_event_stream([_event(t=5.0), _event(t=5.0), final])
    with pytest.raises(ValueError, match="FINAL"):
        validate_event_stream([_event(t=5.0)])  # no final
    with pytest.raises(ValueError, match="FINAL"):
        validate_event_stream([final, _event(t=15.0)])  # final not last
    with pytest.raises(ValueError, match="FINAL"):
        validate_event_stream(
            [_event(kind=ScoreEventKind.FINAL, t=5.0), final]  # two finals
        )
