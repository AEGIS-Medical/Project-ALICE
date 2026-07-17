"""Causality, windowing, and edge-case tests for stream_scores()."""
from __future__ import annotations

import pytest

from backend.shared.schemas.psycholinguistic import PsycholinguisticScore
from backend.shared.schemas.score_event import (
    ScoreEventKind,
    StreamScorerConfig,
    validate_event_stream,
)
from backend.shared.schemas.transcription import Transcript

from .conftest import make_pscore, make_transcript

from app.pipelines.psycholinguistic.analyzer import UnsupportedLanguageError
from app.pipelines.streaming import stream_scores


class CountingStubAnalyzer:
    """Duck-typed analyzer double: counts calls, echoes statement counts."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def analyze(self, statements: list[str]) -> PsycholinguisticScore:
        if not statements:
            raise ValueError("No statements provided")
        self.calls.append(len(statements))
        return make_pscore(composite=50.0, statement_count=len(statements))


# Four statements: two early, two later; duration 18.0s.
FOUR_SEGMENTS = [
    ("I think I was at home that night.", 0.0, 2.0),
    ("I never went anywhere near there.", 2.0, 4.0),
    ("Honestly, you know, I'm not really sure.", 12.0, 14.0),
    ("We stayed in and watched a movie in Chicago.", 14.0, 18.0),
]


def test_tick_schedule_and_final():
    events = list(stream_scores(make_transcript(FOUR_SEGMENTS)))
    validate_event_stream(events)
    # Ticks at 5, 10, 15 (< 18.0), then FINAL at 18.0.
    assert [e.stream_time_seconds for e in events] == [5.0, 10.0, 15.0, 18.0]
    assert [e.kind for e in events] == [
        ScoreEventKind.INTERIM,
        ScoreEventKind.INTERIM,
        ScoreEventKind.INTERIM,
        ScoreEventKind.FINAL,
    ]
    # Causal cumulative counts: 2 stmts by t=5 and t=10, 3 by t=15, 4 total.
    assert [e.statement_count_so_far for e in events] == [2, 2, 3, 4]
    assert events[-1].recent is None
    for ev in events:
        assert ev.vector_scores["psycholinguistic"] == pytest.approx(
            ev.cumulative.composite_score
        )
        assert ev.baseline_available is False
        assert ev.confidence == "low"


def test_causality_future_mutation_does_not_change_past_events():
    """THE causality gate: events up to t are identical when everything
    after t changes."""
    base = make_transcript(FOUR_SEGMENTS)
    mutated_spec = FOUR_SEGMENTS[:2] + [
        ("Actually everything was completely different that day.", 12.0, 14.0),
        ("Um, well, maybe we never really left the office at all.", 14.0, 18.0),
    ]
    mutated = make_transcript(mutated_spec)

    events_base = list(stream_scores(base))
    events_mut = list(stream_scores(mutated))

    # Events at t=5 and t=10 depend only on segments ending <= 10.
    for i in (0, 1):
        assert events_base[i].model_dump() == events_mut[i].model_dump()
    # Later events differ (sanity that the mutation actually bit).
    assert (
        events_base[2].cumulative.model_dump()
        != events_mut[2].cumulative.model_dump()
    )


def test_leading_silence_skips_empty_ticks():
    events = list(stream_scores(make_transcript(FOUR_SEGMENTS[2:])))
    validate_event_stream(events)
    # Segments end at 14 and 18: ticks 5, 10 have no statements -> skipped.
    assert [e.stream_time_seconds for e in events] == [15.0, 18.0]
    assert events[0].kind is ScoreEventKind.INTERIM
    assert events[0].statement_count_so_far == 1


def test_recent_window_and_sparse_recent_none():
    # min_recent_statements=2; at t=15 only one segment ends in (i.e. within)
    # the trailing 10s window (12, 15] -> recent must be None, not fabricated.
    cfg = StreamScorerConfig(
        tick_seconds=5.0, recent_window_seconds=10.0, min_recent_statements=2
    )
    events = list(stream_scores(make_transcript(FOUR_SEGMENTS), cfg))
    by_time = {e.stream_time_seconds: e for e in events}
    # t=5: both early segments end in (-5, 5] -> recent present.
    assert by_time[5.0].recent is not None
    assert by_time[5.0].recent.statement_count == 2
    # t=10: window (0, 10] still holds both early segments -> present.
    assert by_time[10.0].recent is not None
    # t=15: window (5, 15] holds only the 12-14 segment -> sparse -> None.
    assert by_time[15.0].recent is None


def test_boundary_tick_superseded_by_final():
    # Recording ends exactly on a tick boundary (10.0): no interim at 10.0.
    spec = [
        ("I think we left the house early.", 0.0, 4.5),
        ("Nobody saw us come back that evening.", 4.5, 10.0),
    ]
    events = list(stream_scores(make_transcript(spec)))
    validate_event_stream(events)
    assert [e.stream_time_seconds for e in events] == [5.0, 10.0]
    assert [e.kind for e in events] == [
        ScoreEventKind.INTERIM,
        ScoreEventKind.FINAL,
    ]


def test_short_recording_final_only():
    spec = [("I was home alone.", 0.0, 3.0)]
    events = list(stream_scores(make_transcript(spec)))
    validate_event_stream(events)
    assert len(events) == 1
    assert events[0].kind is ScoreEventKind.FINAL
    assert events[0].stream_time_seconds == 3.0
    assert events[0].statement_count_so_far == 1


def test_zero_statement_transcript_yields_empty_stream():
    empty = Transcript(
        segments=[],
        language="en",
        audio_duration_seconds=42.0,
        model_name="fixture",
        backend="fake",
    )
    assert list(stream_scores(empty)) == []


def test_out_of_order_segments_are_sorted_once():
    shuffled = [FOUR_SEGMENTS[2], FOUR_SEGMENTS[0], FOUR_SEGMENTS[3], FOUR_SEGMENTS[1]]
    events_sorted = list(stream_scores(make_transcript(FOUR_SEGMENTS)))
    events_shuffled = list(stream_scores(make_transcript(shuffled)))
    assert [e.model_dump() for e in events_sorted] == [
        e.model_dump() for e in events_shuffled
    ]


def test_determinism_two_runs_identical():
    t = make_transcript(FOUR_SEGMENTS)
    run1 = [e.model_dump() for e in stream_scores(t)]
    run2 = [e.model_dump() for e in stream_scores(t)]
    assert run1 == run2


def test_analyzer_reused_and_called_lazily():
    stub = CountingStubAnalyzer()
    events = list(
        stream_scores(make_transcript(FOUR_SEGMENTS), analyzer=stub)
    )
    # 3 interim ticks: cumulative each (3 calls) + recent where the window
    # meets min_recent_statements (t=5 and t=10 with defaults; at t=15 the
    # trailing 30s window holds 3 statements -> also present) = 3 calls,
    # + FINAL cumulative = 1. Total 7.
    assert len(stub.calls) == 7
    assert events[-1].kind is ScoreEventKind.FINAL


def test_generator_is_lazy_pull_based():
    stub = CountingStubAnalyzer()
    gen = stream_scores(make_transcript(FOUR_SEGMENTS), analyzer=stub)
    assert stub.calls == []  # nothing computed before the first pull
    first = next(gen)
    assert first.stream_time_seconds == 5.0
    # Only the first tick's work happened: cumulative + recent.
    assert len(stub.calls) == 2
    gen.close()


# ---- Session 6 / gap #8: language gate at stream entry ---------------------


def test_non_english_transcript_raises_on_first_iteration():
    transcript = make_transcript(
        [
            ("Hola, estaba en casa.", 0.0, 2.0),
            ("Nunca fui alli.", 2.0, 4.0),
        ],
        language="es",
    )
    gen = stream_scores(transcript)
    with pytest.raises(UnsupportedLanguageError, match="es"):
        next(gen)


def test_non_english_transcript_emits_zero_events():
    transcript = make_transcript(
        [("Hola, estaba en casa.", 0.0, 2.0)], language="es"
    )
    events = []
    with pytest.raises(UnsupportedLanguageError):
        for ev in stream_scores(transcript):
            events.append(ev)
    assert events == []


def test_english_regional_variant_streams_normally():
    transcript = make_transcript(
        [
            ("I think I was at home.", 0.0, 2.0),
            ("I never went there.", 2.0, 4.0),
        ],
        language="en-US",
    )
    events = list(stream_scores(transcript))
    assert events, "en-US must stream"
    assert events[-1].kind.value == "final"


def test_batch_and_stream_gates_are_symmetric():
    """Batch analyze(language=...) and stream_scores must refuse the same
    non-English transcript -- a refactor breaking one gate must fail here."""
    from app.pipelines.psycholinguistic.analyzer import PsycholinguisticAnalyzer

    transcript = make_transcript(
        [("Hola, estaba en casa.", 0.0, 2.0)], language="es"
    )
    with pytest.raises(UnsupportedLanguageError):
        next(stream_scores(transcript))
    with pytest.raises(UnsupportedLanguageError):
        PsycholinguisticAnalyzer().analyze(
            transcript.statements(), language=transcript.language
        )
