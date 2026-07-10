"""ScoreEvent streaming schemas for Project ALICE (Session 5).

Defines the typed contract emitted by the streaming windowed scorer
(``backend/ml-inference/app/pipelines/streaming``). Live surfaces and batch
replays converge on this one schema (CLAUDE.md known-gap #2): interim events
carry BOTH a ``cumulative`` reading (0 -> now; converges to the batch score)
and a ``recent`` reading (last N seconds; the moment detector), and exactly
one authoritative ``FINAL`` event closes every non-empty stream, equal
field-for-field to the batch analyzer's output.

``vector_scores`` maps vector name -> 0-100 composite. Today it holds only
``{"psycholinguistic": ...}``; AU / tonality / contradiction / gaze join the
dict when those vectors exist -- the schema does not change (late fusion,
CLAUDE.md "Ensemble").

CLAUDE.md invariant #5: raw scores and events are ensemble- and
developer-facing. Any user surface must add calibration, confidence display,
and qualitative labels -- never show these numbers bare.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.shared.schemas.psycholinguistic import PsycholinguisticScore


class ScoreEventKind(str, Enum):
    """Event kinds: periodic ``INTERIM`` ticks and the one ``FINAL``."""

    INTERIM = "interim"
    FINAL = "final"


class StreamScorerConfig(BaseModel):
    """Tunable parameters for the windowed streaming scorer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tick_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Interim event cadence in stream time.",
    )
    recent_window_seconds: float = Field(
        default=30.0,
        description="Span of the 'recent' moment-detector window; >= tick_seconds.",
    )
    min_recent_statements: int = Field(
        default=2,
        ge=1,
        description="Below this many statements in the window, recent=None.",
    )

    @model_validator(mode="after")
    def _window_covers_tick(self) -> "StreamScorerConfig":
        if self.recent_window_seconds < self.tick_seconds:
            raise ValueError(
                f"recent_window_seconds ({self.recent_window_seconds}) must be "
                f">= tick_seconds ({self.tick_seconds})"
            )
        return self


class ScoreEvent(BaseModel):
    """One streamed scoring event -- an interim tick or the final read.

    ``cumulative`` scores everything from stream start to
    ``stream_time_seconds``; on FINAL it covers the whole recording and must
    equal the batch analyzer's output exactly. ``recent`` scores the trailing
    ``recent_window_seconds`` and is None when that window holds too little
    speech (never fabricated) -- and always None on FINAL, where the
    whole-recording read IS ``cumulative``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(
        default=1,
        description="Version-locked contract, like ALTM's header (Session 4).",
    )
    kind: ScoreEventKind
    stream_time_seconds: float = Field(
        ge=0.0, description="Call-time position this event describes."
    )
    cumulative: PsycholinguisticScore = Field(
        description="Score over statements from 0 -> stream_time_seconds."
    )
    recent: Optional[PsycholinguisticScore] = Field(
        default=None,
        description="Score over the trailing window; None if sparse or FINAL.",
    )
    vector_scores: dict[str, float] = Field(
        description="Vector name -> 0-100 composite; future vectors extend it."
    )
    statement_count_so_far: int = Field(
        ge=0, description="Statements included in ``cumulative``."
    )
    baseline_available: bool = Field(
        description="Carried from the analyzer (CLAUDE.md invariant #12)."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Session-history confidence, carried from the analyzer."
    )

    @model_validator(mode="after")
    def _contract(self) -> "ScoreEvent":
        if self.kind is ScoreEventKind.FINAL and self.recent is not None:
            raise ValueError(
                "FINAL events must have recent=None -- the whole-recording "
                "read is cumulative"
            )
        expected = self.cumulative.composite_score
        got = self.vector_scores.get("psycholinguistic")
        if got != expected:
            raise ValueError(
                f"vector_scores['psycholinguistic'] ({got}) must equal "
                f"cumulative.composite_score ({expected})"
            )
        return self


def validate_event_stream(events: Sequence[ScoreEvent]) -> None:
    """Validate stream-level contract rules over a complete event sequence.

    An empty stream is legal (zero-statement transcript). A non-empty stream
    must have strictly increasing ``stream_time_seconds`` and exactly one
    FINAL event, in last position.

    Raises:
        ValueError: the sequence violates the stream contract.
    """
    if not events:
        return
    times = [ev.stream_time_seconds for ev in events]
    for earlier, later in zip(times, times[1:]):
        if later <= earlier:
            raise ValueError(
                f"stream_time_seconds must be strictly increasing "
                f"(got {earlier} then {later})"
            )
    final_positions = [
        i for i, ev in enumerate(events) if ev.kind is ScoreEventKind.FINAL
    ]
    if final_positions != [len(events) - 1]:
        raise ValueError(
            f"a non-empty stream must contain exactly one FINAL event in last "
            f"position (FINAL at indexes {final_positions}, "
            f"{len(events)} events)"
        )
