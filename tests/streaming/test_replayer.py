"""Pacing and cancellation tests for ScoreReplayer (injected sleep -- the
suite never actually sleeps)."""
from __future__ import annotations

import inspect

import pytest

from backend.shared.schemas.score_event import ScoreEventKind

from .conftest import make_transcript

from app.pipelines.streaming import ScoreReplayer, stream_scores
from app.pipelines.streaming import replayer as replayer_module

# Events land at t=5.0 (interim) and t=8.0 (final).
SPEC = [
    ("I think I was at home that night.", 0.0, 2.5),
    ("I never went anywhere near there.", 2.5, 5.0),
    ("Honestly, you know, I'm not really sure.", 5.0, 8.0),
]


def _sleep_spy(record: list[float]):
    def _sleep(seconds: float) -> None:
        record.append(seconds)

    return _sleep


def test_pace_zero_yields_immediately_in_order():
    sleeps: list[float] = []
    events = list(
        ScoreReplayer(make_transcript(SPEC)).replay(
            pace=0, sleep=_sleep_spy(sleeps)
        )
    )
    assert sleeps == []
    assert [e.stream_time_seconds for e in events] == [5.0, 8.0]
    assert events[-1].kind is ScoreEventKind.FINAL


def test_pace_one_sleeps_inter_event_gaps():
    sleeps: list[float] = []
    events = list(
        ScoreReplayer(make_transcript(SPEC)).replay(
            pace=1.0, sleep=_sleep_spy(sleeps)
        )
    )
    assert len(events) == 2
    assert sleeps == pytest.approx([5.0, 3.0])


def test_pace_two_halves_the_sleeps():
    sleeps: list[float] = []
    list(
        ScoreReplayer(make_transcript(SPEC)).replay(
            pace=2.0, sleep=_sleep_spy(sleeps)
        )
    )
    assert sleeps == pytest.approx([2.5, 1.5])


def test_negative_pace_rejected():
    with pytest.raises(ValueError, match="pace"):
        next(ScoreReplayer(make_transcript(SPEC)).replay(pace=-1.0))


def test_replay_matches_stream_scores_events():
    transcript = make_transcript(SPEC)
    direct = [e.model_dump() for e in stream_scores(transcript)]
    replayed = [
        e.model_dump()
        for e in ScoreReplayer(transcript).replay(pace=0)
    ]
    assert replayed == direct


def test_early_break_closes_underlying_generator(monkeypatch):
    """Consumer cancellation propagates: abandoning the replay closes the
    scoring generator (the story a future socket session reuses)."""
    closed: list[bool] = []
    real_stream_scores = replayer_module.stream_scores

    def tracking_stream_scores(*args, **kwargs):
        gen = real_stream_scores(*args, **kwargs)
        try:
            yield from gen
        finally:
            closed.append(True)

    monkeypatch.setattr(
        replayer_module, "stream_scores", tracking_stream_scores
    )

    it = ScoreReplayer(make_transcript(SPEC)).replay(pace=0)
    first = next(it)
    assert first.stream_time_seconds == 5.0
    it.close()  # consumer break / connection drop

    assert closed == [True]
    assert inspect.getgeneratorstate(it) == inspect.GEN_CLOSED
