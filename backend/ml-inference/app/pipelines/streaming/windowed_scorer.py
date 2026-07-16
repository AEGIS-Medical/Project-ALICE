"""Causal windowed scorer: Transcript -> ScoreEvent stream.

The core is a plain pull-based (sync) generator BY DESIGN -- spec decision
#4: per-tick work is CPU-bound (spaCy + lexicons), which Python async cannot
make concurrent; a pull generator is inherently backpressure-safe. The
future live-service session wraps this generator in an async
FastAPI/WebSocket shell via ``asyncio.to_thread(next, gen)`` with
per-session cancellation -- async where it earns its keep (socket I/O),
sync where async cannot help (compute).

Strict causality: an event at stream time ``t`` sees only segments with
``end_seconds <= t``. No lookahead, ever -- this is what makes a replayed
stream identical to a genuinely live one.

CLAUDE.md invariant #3: this module logs tick/statement counts and
durations only -- never transcript text. Invariant #5: emitted events are
ensemble- and developer-facing; user surfaces add calibration and labels.
"""
from __future__ import annotations

import logging
from typing import Iterator, Optional

from backend.shared.schemas.psycholinguistic import PsycholinguisticScore
from backend.shared.schemas.score_event import (
    ScoreEvent,
    ScoreEventKind,
    StreamScorerConfig,
)
from backend.shared.schemas.transcription import Transcript, TranscriptSegment

from app.pipelines.psycholinguistic.analyzer import (
    SUPPORTED_LANGUAGES,
    PsycholinguisticAnalyzer,
    UnsupportedLanguageError,
    _primary_subtag,
)

logger = logging.getLogger(__name__)


def _analyze_slice(
    analyzer: PsycholinguisticAnalyzer, segments: list[TranscriptSegment]
) -> PsycholinguisticScore:
    """Run the batch analyzer over a non-empty slice of segments."""
    return analyzer.analyze([s.text for s in segments])


def _build_event(
    kind: ScoreEventKind,
    stream_time: float,
    cumulative: PsycholinguisticScore,
    recent: Optional[PsycholinguisticScore],
) -> ScoreEvent:
    return ScoreEvent(
        kind=kind,
        stream_time_seconds=stream_time,
        cumulative=cumulative,
        recent=recent,
        vector_scores={"psycholinguistic": cumulative.composite_score},
        statement_count_so_far=cumulative.statement_count,
        baseline_available=cumulative.baseline_available,
        confidence=cumulative.confidence,
    )


def stream_scores(
    transcript: Transcript,
    config: Optional[StreamScorerConfig] = None,
    analyzer: Optional[PsycholinguisticAnalyzer] = None,
) -> Iterator[ScoreEvent]:
    """Yield causal ScoreEvents for a transcript, ending with one FINAL.

    Interim events fire at ``t = k * tick_seconds`` strictly before the
    recording's end (a tick coinciding exactly with the end is superseded by
    the FINAL). A tick whose cumulative slice is empty (leading silence) is
    skipped. ``recent`` is None when the trailing window holds fewer than
    ``min_recent_statements`` statements. A zero-statement transcript yields
    an empty stream -- nothing is fabricated.

    Args:
        transcript: Completed transcript (segments need not be pre-sorted).
        config: Windowing parameters; defaults to ``StreamScorerConfig()``.
        analyzer: Injectable analyzer (one instance reused across all ticks;
            constructed lazily on first use so an empty stream loads nothing).

    Yields:
        ``ScoreEvent`` objects with strictly increasing stream times; the
        last event of a non-empty stream is the authoritative FINAL whose
        ``cumulative`` equals the batch analyzer's output field-for-field.

    Raises:
        UnsupportedLanguageError: ``transcript.language`` is not supported.
            Because this function is a generator, the raise does not happen
            at call time -- it surfaces on the caller's first iteration
            (first ``next()``/first loop step), before any event is yielded.
    """
    # Gap #8 gate: never let a non-English transcript flow silently through
    # the English-only analyzer. Raised before any event is emitted, so the
    # ScoreEvent contract is untouched: an empty stream still means exactly
    # one thing (zero statements). NOTE: generators defer execution -- this
    # raise surfaces on the caller's first next()/iteration.
    if _primary_subtag(transcript.language) not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(
            f"language {transcript.language!r} is not supported by the "
            f"psycholinguistic vector; supported: "
            f"{', '.join(sorted(SUPPORTED_LANGUAGES))}"
        )

    cfg = config or StreamScorerConfig()
    segments = sorted(transcript.segments, key=lambda s: s.end_seconds)
    if not segments:
        logger.info("stream_scores_empty segments=0")
        return

    if analyzer is None:
        analyzer = PsycholinguisticAnalyzer()

    duration = segments[-1].end_seconds
    interim_count = 0

    k = 1
    tick = cfg.tick_seconds
    while (t := k * tick) < duration:
        k += 1
        cumulative_slice = [s for s in segments if s.end_seconds <= t]
        if not cumulative_slice:
            continue  # leading silence: no event for this tick
        recent_slice = [
            s
            for s in cumulative_slice
            if s.end_seconds > t - cfg.recent_window_seconds
        ]
        cumulative = _analyze_slice(analyzer, cumulative_slice)
        recent = (
            _analyze_slice(analyzer, recent_slice)
            if len(recent_slice) >= cfg.min_recent_statements
            else None
        )
        interim_count += 1
        yield _build_event(ScoreEventKind.INTERIM, t, cumulative, recent)

    final_cumulative = _analyze_slice(analyzer, segments)
    logger.info(
        "stream_scores_done interim_events=%d statements=%d duration_s=%.2f",
        interim_count,
        len(segments),
        duration,
    )
    yield _build_event(ScoreEventKind.FINAL, duration, final_cumulative, None)
