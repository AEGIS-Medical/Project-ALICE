"""THE acceptance gate for Session 5 (spec decision #3): the stream's FINAL
event must equal the batch analyzer's output field-for-field -- live and
batch converge on one contract.
"""
from __future__ import annotations

from pathlib import Path

from backend.shared.schemas.score_event import ScoreEventKind, validate_event_stream

from .conftest import make_transcript

from app.pipelines.psycholinguistic.analyzer import PsycholinguisticAnalyzer
from app.pipelines.streaming import stream_scores
from app.pipelines.transcription.backends import FakeTranscriptionBackend

# Deterministic synthetic transcript: 40 statements over ~3 minutes with
# varied linguistic signal (pronouns, hedges, negations, entities,
# disfluencies) so all eight dimension scorers do real work.
_SENTENCE_BANK = [
    "I think we probably left the office around six that evening.",
    "I never touched the files in that cabinet.",
    "Um, well, you know, it was sort of a complicated situation.",
    "We met Sarah and Daniel at the restaurant on Michigan Avenue in Chicago.",
    "Honestly I am not sure anyone checked the badge logs on Tuesday.",
    "My team finished the quarterly report before the deadline.",
    "Maybe the server room door was already open when I arrived.",
    "They definitely confirmed the shipment reached Denver on Friday.",
]


def _synthetic_spec(n: int = 40) -> list[tuple[str, float, float]]:
    spec = []
    for i in range(n):
        text = _SENTENCE_BANK[i % len(_SENTENCE_BANK)]
        start = i * 4.5
        spec.append((text, start, start + 4.0))
    return spec  # last end = 39*4.5 + 4.0 = 179.5s (~3 minutes)


def _assert_final_converges(transcript) -> None:
    events = list(stream_scores(transcript))
    validate_event_stream(events)

    final = events[-1]
    assert final.kind is ScoreEventKind.FINAL
    assert final.recent is None

    batch = PsycholinguisticAnalyzer().analyze(transcript.statements())

    # Field-for-field equality: all eight dimensions (scores AND evidence),
    # composite, counts, baseline, confidence.
    assert final.cumulative.model_dump() == batch.model_dump()
    assert final.vector_scores == {
        "psycholinguistic": batch.composite_score
    }
    assert final.statement_count_so_far == batch.statement_count
    assert final.baseline_available == batch.baseline_available
    assert final.confidence == batch.confidence


def test_convergence_on_fake_backend_canned_transcript():
    transcript = FakeTranscriptionBackend().transcribe(Path("unused.flac"))
    events = list(stream_scores(transcript))
    # Canned segments span 0-6s: one interim at t=5, then FINAL at 6.0.
    assert [e.kind for e in events] == [
        ScoreEventKind.INTERIM,
        ScoreEventKind.FINAL,
    ]
    _assert_final_converges(transcript)


def test_convergence_on_synthetic_three_minute_transcript():
    transcript = make_transcript(_synthetic_spec())
    events = list(stream_scores(transcript))
    # Ticks at 5, 10, ..., 175 (< 179.5) = 35 interims, plus the FINAL.
    assert len(events) == 36
    assert events[-1].stream_time_seconds == 179.5
    _assert_final_converges(transcript)


def test_interim_cumulative_converges_to_final():
    """The last interim's cumulative differs from FINAL only by the tail
    statements -- and an interim over ALL statements equals batch exactly."""
    transcript = make_transcript(_synthetic_spec())
    events = list(stream_scores(transcript))
    interims = [e for e in events if e.kind is ScoreEventKind.INTERIM]
    final = events[-1]
    # Monotone growth of the causal slice, ending at the full set.
    counts = [e.statement_count_so_far for e in interims]
    assert counts == sorted(counts)
    assert final.statement_count_so_far == 40
    assert counts[-1] <= 40
