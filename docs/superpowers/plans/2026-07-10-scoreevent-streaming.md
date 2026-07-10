# ScoreEvent Streaming Contract + Windowed Scorer + Replayer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define the `ScoreEvent` streaming contract and prove it with a strictly causal windowed scorer plus a wall-clock replay consumer, so live and batch analysis converge on one schema — closing CLAUDE.md known-gap #2 at the contract layer (scope A: no live capture, no WebSocket server).

**Architecture:** A frozen Pydantic v2 event schema (`backend/shared/schemas/score_event.py`) carries both a `cumulative` (0→now) and a `recent` (last-N-seconds) `PsycholinguisticScore`. A plain **sync pull-based generator** `stream_scores()` (`backend/ml-inference/app/pipelines/streaming/windowed_scorer.py`) emits interim events on a tick schedule and exactly one authoritative `FINAL` event that must equal the batch analyzer's output field-for-field — that convergence is the session's acceptance gate. `ScoreReplayer` wraps the generator with injectable wall-clock pacing; a CLI replays a transcript (or video, via the existing pipeline) as a timed event stream. The async FastAPI/WebSocket shell is **deliberately deferred** (spec decision #4: CPU-bound per-tick work gains nothing from an async core; the future live service drives this generator via `asyncio.to_thread(next, gen)`).

**Tech Stack:** Python 3.12 (laptop venv; code must not use 3.13-only features), Pydantic v2 (frozen models), pytest. **No new dependencies.** spaCy `en_core_web_sm` already installed; the suite stays torch-free (fake transcripts only).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-08-scoreevent-streaming-design.md`. If plan and spec conflict, the spec governs.
- Repo root: `C:\Users\rphos\Project-ALICE` (Git Bash path `/c/Users/rphos/Project-ALICE`). Run everything from the repo root with `.venv/Scripts/python` (Windows venv, Python 3.12.10).
- Branch: all commits land on `session-5-scoreevent-streaming` (already created from `dev-build`). Never commit to `dev-build` directly.
- Full-suite runs need ffmpeg on PATH. In Git Bash prepend once per shell: `export PATH="$PATH:/c/Users/rphos/AppData/Local/Microsoft/WinGet/Links"`. Streaming-only test runs do not need it.
- All new schemas: Pydantic v2 with `model_config = ConfigDict(frozen=True, extra="forbid")` — exactly like `media.py` / `psycholinguistic.py` / `transcription.py`.
- Exact contract values: `schema_version=1`; `StreamScorerConfig` defaults `tick_seconds=5.0` (`gt=0`), `recent_window_seconds=30.0` (must be `>= tick_seconds`), `min_recent_statements=2` (`ge=1`).
- Contract rules (validator- or scorer-enforced, tested either way): `FINAL` ⇒ `recent is None`; `vector_scores["psycholinguistic"] == cumulative.composite_score` (exact float equality — the scorer copies the value); stream-level: strictly increasing `stream_time_seconds`, exactly one `FINAL`, always last.
- Strict causality: the cumulative slice at tick `t` is every segment with `end_seconds <= t`; the recent slice additionally requires `end_seconds > t - recent_window_seconds`. No lookahead, ever.
- Interim ticks are `t = k * tick_seconds` (computed by multiplication, never accumulation) for `k = 1, 2, ...` while `t < duration` (strict `<`: a tick coinciding exactly with the recording end is superseded by `FINAL` — the boundary rule).
- `backend/ml-inference` is a hyphenated service root: streaming code lives there and imports the analyzer as `from app.pipelines.psycholinguistic.analyzer import PsycholinguisticAnalyzer`; tests and scripts bridge `sys.path` exactly like `tests/transcription/conftest.py` and `scripts/test_compress_and_analyze.py`.
- `PsycholinguisticAnalyzer.analyze(statements: list[str])` raises `ValueError` on an empty list — the scorer must never call it with an empty slice.
- CLAUDE.md invariants: **#3** never log transcript text (counts/durations only), **#5** raw events are developer/ensemble-facing (module docstring note + CLI disclaimer), **#6** never use the forbidden phrase "lie detector", **#12** `baseline_available` travels with every event.
- Every public function/class gets a docstring; full type annotations (`from __future__ import annotations` at the top of every new file).
- TDD per task: write failing tests → run → implement → run green → commit. Conventional commit prefixes `feat(streaming):` / `test(streaming):` / `docs:`.
- Suite baseline before this session: `95 passed, 1 deselected`. Compression tests need no changes.

---

## File Map

| File | Action | Task |
|---|---|---|
| `backend/shared/schemas/score_event.py` | Create | 1 |
| `tests/streaming/__init__.py` | Create (empty) | 1 |
| `tests/streaming/conftest.py` | Create — sys.path bridge + fixtures | 1 |
| `tests/streaming/test_score_event_schema.py` | Create | 1 |
| `backend/ml-inference/app/pipelines/streaming/__init__.py` | Create — re-exports | 2 |
| `backend/ml-inference/app/pipelines/streaming/windowed_scorer.py` | Create — `stream_scores()` | 2 |
| `tests/streaming/test_windowed_scorer.py` | Create | 2 |
| `backend/ml-inference/app/pipelines/streaming/replayer.py` | Create — `ScoreReplayer` | 3 |
| `tests/streaming/test_replayer.py` | Create | 3 |
| `tests/streaming/test_convergence.py` | Create — THE acceptance gate | 4 |
| `scripts/replay_scores.py` | Create — CLI | 5 |
| `tests/streaming/test_cli_smoke.py` | Create | 5 |
| `CLAUDE.md` | Modify — status section + gap #2 resolved | 6 |

---

## Task 1: ScoreEvent Schemas + Stream Validation Helper

**Files:**
- Create: `backend/shared/schemas/score_event.py`
- Create: `tests/streaming/__init__.py` (empty)
- Create: `tests/streaming/conftest.py`
- Test: `tests/streaming/test_score_event_schema.py`

**Interfaces:**
- Produces (all in `backend.shared.schemas.score_event`, dotted import via editable install):
  - `class ScoreEventKind(str, Enum)`: `INTERIM = "interim"`, `FINAL = "final"`.
  - `class StreamScorerConfig(BaseModel)`: `tick_seconds: float = 5.0` (gt=0), `recent_window_seconds: float = 30.0` (validated `>= tick_seconds`), `min_recent_statements: int = 2` (ge=1). Frozen, extra="forbid".
  - `class ScoreEvent(BaseModel)`: `schema_version: int = 1`, `kind: ScoreEventKind`, `stream_time_seconds: float` (ge=0), `cumulative: PsycholinguisticScore`, `recent: Optional[PsycholinguisticScore] = None`, `vector_scores: dict[str, float]`, `statement_count_so_far: int` (ge=0), `baseline_available: bool`, `confidence: Literal["low", "medium", "high"]`. Frozen, extra="forbid". Model validator enforces FINAL⇒recent None and `vector_scores["psycholinguistic"] == cumulative.composite_score`.
  - `def validate_event_stream(events: Sequence[ScoreEvent]) -> None` — raises `ValueError` unless: empty, OR strictly increasing `stream_time_seconds` with exactly one FINAL in last position.
- Produces (in `tests/streaming/conftest.py`, used by every later test task):
  - `make_pscore(composite: float = 50.0, statement_count: int = 3) -> PsycholinguisticScore` — builds a valid score with all eight dimensions at `composite`.
  - `make_transcript(spec: list[tuple[str, float, float]]) -> Transcript` — builds a Transcript from `(text, start_seconds, end_seconds)` tuples (`language="en"`, `audio_duration_seconds` = max end or 0.0, `model_name="fixture"`, `backend="fake"`).
  - The `sys.path` bridge to `backend/ml-inference` (module-level, before fixtures).

- [ ] **Step 1: Create the test package and conftest**

```bash
cd /c/Users/rphos/Project-ALICE
mkdir -p tests/streaming
printf '' > tests/streaming/__init__.py
```

Create `tests/streaming/conftest.py`:

```python
"""Fixtures for the streaming suite.

The streaming pipeline lives under ``backend/ml-inference/`` -- a service
root whose directory name contains a hyphen, so it cannot be imported via a
dotted ``backend.ml_inference`` path. We insert that root onto ``sys.path``
exactly as ``tests/transcription/conftest.py`` does, then import
``from app.pipelines.streaming...``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from backend.shared.schemas.psycholinguistic import (
    PsycholinguisticDimension,
    PsycholinguisticScore,
)
from backend.shared.schemas.transcription import Transcript, TranscriptSegment

_ML_INFERENCE_ROOT = (
    Path(__file__).resolve().parents[2] / "backend" / "ml-inference"
)
if str(_ML_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_INFERENCE_ROOT))


def make_pscore(
    composite: float = 50.0, statement_count: int = 3
) -> PsycholinguisticScore:
    """A valid frozen score with all eight dimensions at ``composite``."""
    dim = PsycholinguisticDimension(score=composite, evidence=[])
    return PsycholinguisticScore(
        pronoun_shift_score=dim,
        hedging_score=dim,
        cognitive_complexity_score=dim,
        emotional_distribution_score=dim,
        disfluency_score=dim,
        negation_score=dim,
        detail_specificity_score=dim,
        certainty_score=dim,
        composite_score=composite,
        statement_count=statement_count,
        baseline_available=False,
        confidence="low",
    )


def make_transcript(spec: list[tuple[str, float, float]]) -> Transcript:
    """Build an ordered Transcript from (text, start_seconds, end_seconds)."""
    segments = [
        TranscriptSegment(text=t, start_seconds=s, end_seconds=e)
        for t, s, e in spec
    ]
    duration = max((e for _, _, e in spec), default=0.0)
    return Transcript(
        segments=segments,
        language="en",
        audio_duration_seconds=duration,
        model_name="fixture",
        backend="fake",
    )
```

- [ ] **Step 2: Write the failing schema tests**

Create `tests/streaming/test_score_event_schema.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_score_event_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.shared.schemas.score_event'`

- [ ] **Step 4: Create the schema module**

Create `backend/shared/schemas/score_event.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_score_event_schema.py -q`
Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/shared/schemas/score_event.py tests/streaming/
git commit -m "feat(streaming): ScoreEvent contract schemas + stream validation helper"
```

---

## Task 2: Causal Windowed Scorer (`stream_scores`)

**Files:**
- Create: `backend/ml-inference/app/pipelines/streaming/__init__.py`
- Create: `backend/ml-inference/app/pipelines/streaming/windowed_scorer.py`
- Test: `tests/streaming/test_windowed_scorer.py`

**Interfaces:**
- Consumes: `ScoreEvent`, `ScoreEventKind`, `StreamScorerConfig`, `validate_event_stream` (Task 1); `Transcript` / `TranscriptSegment` (existing); `PsycholinguisticAnalyzer` (existing, via `app.` import); `make_transcript`, `make_pscore` (Task 1 conftest).
- Produces (later tasks rely on these exact names):
  - `stream_scores(transcript: Transcript, config: StreamScorerConfig | None = None, analyzer: PsycholinguisticAnalyzer | None = None) -> Iterator[ScoreEvent]` in `app.pipelines.streaming.windowed_scorer`.
  - `app/pipelines/streaming/__init__.py` re-exports `stream_scores`.
- The `analyzer` parameter accepts any object with `analyze(statements: list[str]) -> PsycholinguisticScore` (duck-typed for tests; annotate the parameter `PsycholinguisticAnalyzer | None` and inject test doubles via the untyped path — mirrors how `Transcriber` accepts `FakeTranscriptionBackend`).

- [ ] **Step 1: Write the failing scorer tests**

Create `tests/streaming/test_windowed_scorer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_windowed_scorer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipelines.streaming'`

- [ ] **Step 3: Create the streaming package**

Create `backend/ml-inference/app/pipelines/streaming/__init__.py`:

```python
"""Streaming scoring pipeline (Session 5): causal windowed ScoreEvents.

Spec: docs/superpowers/specs/2026-07-08-scoreevent-streaming-design.md
"""
from app.pipelines.streaming.windowed_scorer import stream_scores

__all__ = ["stream_scores"]
```

Create `backend/ml-inference/app/pipelines/streaming/windowed_scorer.py`:

```python
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

from app.pipelines.psycholinguistic.analyzer import PsycholinguisticAnalyzer

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
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/streaming/ -q`
Expected: 21 passed (10 schema + 11 scorer). The scorer tests load spaCy once — allow ~30–60 s.

- [ ] **Step 5: Commit**

```bash
git add backend/ml-inference/app/pipelines/streaming/ tests/streaming/test_windowed_scorer.py
git commit -m "feat(streaming): stream_scores causal windowed generator (sync core by design)"
```

---

## Task 3: ScoreReplayer (pacing wrapper)

**Files:**
- Create: `backend/ml-inference/app/pipelines/streaming/replayer.py`
- Modify: `backend/ml-inference/app/pipelines/streaming/__init__.py` (add re-export)
- Test: `tests/streaming/test_replayer.py`

**Interfaces:**
- Consumes: `stream_scores` (Task 2), `StreamScorerConfig`, `ScoreEvent` (Task 1).
- Produces (exact names Task 5's CLI relies on):
  - `class ScoreReplayer` in `app.pipelines.streaming.replayer`, re-exported from `app.pipelines.streaming`:
    - `__init__(self, transcript: Transcript, config: StreamScorerConfig | None = None) -> None`
    - `replay(self, pace: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> Iterator[ScoreEvent]` — raises `ValueError` if `pace < 0`; `pace=0` never calls `sleep`.

- [ ] **Step 1: Write the failing replayer tests**

Create `tests/streaming/test_replayer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_replayer.py -q`
Expected: FAIL — `ImportError: cannot import name 'ScoreReplayer'`

- [ ] **Step 3: Implement the replayer**

Create `backend/ml-inference/app/pipelines/streaming/replayer.py`:

```python
"""ScoreReplayer: wall-clock pacing over the causal score stream.

Pacing is this class's entire job -- no scoring logic lives here. It
re-emits an already-transcribed recording's analysis as a timed stream:
``pace=1.0`` means 1 s of call time per 1 s of replay (demo mode),
``pace=2.0`` is double speed, ``pace=0`` is instant (tests / batch use).
The ``sleep`` callable is injectable so pacing tests assert requested
sleep durations instead of actually sleeping.

Cancellation story (reused per-connection by the future socket session):
stopping iteration (consumer ``break`` / ``close()``) raises GeneratorExit
inside ``replay``, whose ``finally`` closes the underlying scoring
generator -- no resources leak (no files or sockets are held).
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterator, Optional

from backend.shared.schemas.score_event import ScoreEvent, StreamScorerConfig
from backend.shared.schemas.transcription import Transcript

from app.pipelines.streaming.windowed_scorer import stream_scores

logger = logging.getLogger(__name__)


class ScoreReplayer:
    """Replay a transcript's score stream with wall-clock pacing."""

    def __init__(
        self,
        transcript: Transcript,
        config: Optional[StreamScorerConfig] = None,
    ) -> None:
        self._transcript = transcript
        self._config = config

    def replay(
        self,
        pace: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Iterator[ScoreEvent]:
        """Yield ScoreEvents, sleeping the gap to each event's stream time.

        Args:
            pace: Stream-seconds per wall-second. ``0`` disables sleeping
                entirely; ``2.0`` replays at double speed.
            sleep: Sleep function (injectable for tests).

        Raises:
            ValueError: ``pace`` is negative.
        """
        if pace < 0:
            raise ValueError(f"pace must be >= 0 (got {pace})")

        gen = stream_scores(self._transcript, self._config)
        emitted = 0
        try:
            previous_time = 0.0
            for event in gen:
                if pace > 0:
                    gap = (event.stream_time_seconds - previous_time) / pace
                    if gap > 0:
                        sleep(gap)
                previous_time = event.stream_time_seconds
                emitted += 1
                yield event
        finally:
            gen.close()
            logger.info("replay_done events=%d pace=%.2f", emitted, pace)
```

Update `backend/ml-inference/app/pipelines/streaming/__init__.py` to:

```python
"""Streaming scoring pipeline (Session 5): causal windowed ScoreEvents.

Spec: docs/superpowers/specs/2026-07-08-scoreevent-streaming-design.md
"""
from app.pipelines.streaming.replayer import ScoreReplayer
from app.pipelines.streaming.windowed_scorer import stream_scores

__all__ = ["ScoreReplayer", "stream_scores"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_replayer.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/ml-inference/app/pipelines/streaming/replayer.py backend/ml-inference/app/pipelines/streaming/__init__.py tests/streaming/test_replayer.py
git commit -m "feat(streaming): ScoreReplayer pacing wrapper with injectable sleep + clean cancellation"
```

---

## Task 4: The Convergence Gate (final == batch)

**Files:**
- Test: `tests/streaming/test_convergence.py` (no source changes expected; if a test exposes a scorer bug, fix it in `windowed_scorer.py` within this task — do not weaken the tests)

**Interfaces:**
- Consumes: `stream_scores` (Task 2), `validate_event_stream` (Task 1), `PsycholinguisticAnalyzer` (existing), `FakeTranscriptionBackend` (existing), `make_transcript` (Task 1 conftest).

- [ ] **Step 1: Write the convergence tests**

Create `tests/streaming/test_convergence.py`:

```python
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
```

- [ ] **Step 2: Run the gate**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_convergence.py -q`
Expected: 3 passed (the synthetic case runs ~36 ticks × 2 analyses ≈ 70 analyzer calls; allow a couple of minutes). If equality fails, the scorer (Task 2) has a causality or slicing bug — fix it there and re-run. **Do not weaken the gate** (no approx-equality, no field subsetting).

- [ ] **Step 3: Commit**

```bash
git add tests/streaming/test_convergence.py
git commit -m "test(streaming): batch-convergence acceptance gate (final == batch, field-for-field)"
```

---

## Task 5: Replay CLI (`scripts/replay_scores.py`)

**Files:**
- Create: `scripts/replay_scores.py`
- Test: `tests/streaming/test_cli_smoke.py`

**Interfaces:**
- Consumes: `ScoreReplayer`, `stream_scores` re-exports (Task 3), `StreamScorerConfig` (Task 1), `Transcript` (existing), `CompressionPipeline` / `Transcriber` / backends (existing — mirrors `scripts/test_compress_and_analyze.py`).
- Produces: `python scripts/replay_scores.py` with flags `--transcript PATH.json` OR `--video PATH` (+ `--fake`), `--pace 1.0` (0 = instant), `--tick 5.0`, `--recent-window 30.0`, `--mode edge_full`. Output: one line per event + the standard anomaly disclaimer. Exit 0 on success, 1 on bad input.

- [ ] **Step 1: Write the failing CLI smoke test**

Create `tests/streaming/test_cli_smoke.py`:

```python
"""Smoke test: replay_scores.py streams events from a transcript JSON."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from backend.shared.schemas.transcription import Transcript, TranscriptSegment

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "replay_scores.py"


def _write_transcript_json(path: Path) -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                text="I think I was at home that night.",
                start_seconds=0.0,
                end_seconds=2.4,
            ),
            TranscriptSegment(
                text="I never went anywhere near there.",
                start_seconds=2.4,
                end_seconds=4.1,
            ),
            TranscriptSegment(
                text="Honestly, you know, I'm not really sure.",
                start_seconds=4.1,
                end_seconds=6.0,
            ),
        ],
        language="en",
        audio_duration_seconds=6.0,
        model_name="fixture",
        backend="fake",
    )
    path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


def test_replays_transcript_json_instantly(tmp_path):
    tj = tmp_path / "transcript.json"
    _write_transcript_json(tj)
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--transcript", str(tj), "--pace", "0"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    interim_lines = [l for l in out.splitlines() if "interim" in l]
    final_lines = [l for l in out.splitlines() if "final" in l]
    assert len(interim_lines) >= 1
    assert len(final_lines) == 1
    assert "cumulative=" in interim_lines[0]
    # The standard dev-tool anomaly disclaimer (invariants #5/#6).
    assert "behavioral anomaly signal, not ground truth" in out
    assert "lie detector" not in out.lower()


def test_missing_transcript_file_exits_1(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--transcript",
            str(tmp_path / "nope.json"),
            "--pace",
            "0",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "not found" in result.stderr.lower()


def test_requires_exactly_one_input(tmp_path):
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--pace", "0"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_cli_smoke.py -q`
Expected: FAIL — the script does not exist (subprocess returncode 2, assertions fail).

- [ ] **Step 3: Create `scripts/replay_scores.py`**

```python
#!/usr/bin/env python
"""Replay a recording's analysis as a timed ScoreEvent stream.

Feed it a serialized Transcript JSON (offline/dev) or a video (runs the
existing CompressionPipeline -> Transcriber path; --fake uses canned
segments so the full path runs offline). Events print one per line:

    [t=   5.0s] interim  cumulative= 48.2  recent= 61.0  (conf: low, stmts: 2)
    [t=   6.0s] final    cumulative= 47.9  recent=  --   (conf: low, stmts: 3)

Usage:
    python scripts/replay_scores.py --transcript path/to/transcript.json --pace 0
    python scripts/replay_scores.py --video path/to/clip.mp4 --fake --pace 2
    python scripts/replay_scores.py --transcript t.json --tick 5 --recent-window 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ML_INFERENCE_ROOT = _REPO_ROOT / "backend" / "ml-inference"
for _p in (_ML_INFERENCE_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _format_event(event) -> str:
    recent = (
        f"{event.recent.composite_score:5.1f}"
        if event.recent is not None
        else "  -- "
    )
    return (
        f"[t={event.stream_time_seconds:6.1f}s] {event.kind.value:<8}"
        f"cumulative={event.cumulative.composite_score:5.1f}  "
        f"recent={recent}  "
        f"(conf: {event.confidence}, stmts: {event.statement_count_so_far})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay analysis as a timed ScoreEvent stream"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--transcript", type=Path, help="Path to a serialized Transcript JSON"
    )
    source.add_argument("--video", type=Path, help="Path to a video file")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="With --video: use the fake transcription backend (offline)",
    )
    parser.add_argument(
        "--mode",
        choices=["raw", "roi", "edge_full", "edge_minimal"],
        default="edge_full",
        help="With --video: compression mode (default: edge_full)",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help="Stream-seconds per wall-second; 0 = instant (default: 1.0)",
    )
    parser.add_argument(
        "--tick", type=float, default=5.0, help="Interim cadence in seconds"
    )
    parser.add_argument(
        "--recent-window",
        type=float,
        default=30.0,
        help="Trailing moment-detector window in seconds",
    )
    args = parser.parse_args()

    from backend.shared.schemas.score_event import StreamScorerConfig
    from backend.shared.schemas.transcription import Transcript

    if args.transcript is not None:
        if not args.transcript.exists():
            print(
                f"ERROR: transcript not found: {args.transcript}",
                file=sys.stderr,
            )
            return 1
        transcript = Transcript.model_validate_json(
            args.transcript.read_text(encoding="utf-8")
        )
    else:
        if not args.video.exists():
            print(f"ERROR: video not found: {args.video}", file=sys.stderr)
            return 1
        from backend.shared.schemas.media import CompressionMode
        from backend.workers.app.compression.pipeline import CompressionPipeline
        from app.pipelines.transcription.backends import (
            FakeTranscriptionBackend,
        )
        from app.pipelines.transcription.transcriber import Transcriber

        mode_map = {
            "raw": CompressionMode.RAW,
            "roi": CompressionMode.ROI_ENCODED,
            "edge_full": CompressionMode.EDGE_FULL,
            "edge_minimal": CompressionMode.EDGE_MINIMAL,
        }
        output_dir = _REPO_ROOT / "processed_output" / "replay_scores" / args.video.stem
        print(f"Compressing {args.video.name} ({args.mode}) ...")
        result = CompressionPipeline().process(
            args.video, output_dir, mode_map[args.mode]
        )
        if args.fake:
            backend = FakeTranscriptionBackend()
            print("Transcribing (fake backend, canned segments) ...")
        else:
            from app.pipelines.transcription.backends import WhisperXBackend

            backend = WhisperXBackend()
            print("Transcribing (WhisperX) ...")
        try:
            transcript = Transcriber(backend).transcribe(
                result.flac_audio_path
            )
        except Exception as exc:
            print(f"ERROR: transcription failed: {exc}", file=sys.stderr)
            print("Re-run with --fake to exercise the path offline.")
            return 1

    from app.pipelines.streaming import ScoreReplayer

    config = StreamScorerConfig(
        tick_seconds=args.tick, recent_window_seconds=args.recent_window
    )
    print(
        f"\nReplaying {len(transcript.segments)} statements "
        f"({transcript.audio_duration_seconds:.1f}s of audio) at pace "
        f"{args.pace:g} (tick {args.tick:g}s, window {args.recent_window:g}s)"
    )
    print("-" * 72)
    count = 0
    for event in ScoreReplayer(transcript, config).replay(pace=args.pace):
        print(_format_event(event), flush=True)
        count += 1
    if count == 0:
        print("(no speech -- empty stream; nothing to score)")
    print("-" * 72)
    print(
        "NOTE: behavioral anomaly signal, not ground truth. ~75% F1 ceiling; "
        "scores are deviations from baseline, developer-facing only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_cli_smoke.py -q`
Expected: 3 passed.

Also verify by hand (paced output appears over ~3 s):

```bash
.venv/Scripts/python - <<'EOF'
import sys
sys.path.insert(0, ".")
from pathlib import Path
sys.path.insert(0, str(Path("backend/ml-inference")))
from app.pipelines.transcription.backends import FakeTranscriptionBackend
t = FakeTranscriptionBackend().transcribe(Path("x.flac"))
Path("tmp_transcript.json").write_text(t.model_dump_json(indent=2), encoding="utf-8")
EOF
.venv/Scripts/python scripts/replay_scores.py --transcript tmp_transcript.json --pace 2
rm tmp_transcript.json
```

Expected: one `interim` line (t=5.0), one `final` line (t=6.0), the disclaimer, exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/replay_scores.py tests/streaming/test_cli_smoke.py
git commit -m "feat(streaming): replay_scores CLI -- transcript/video in, timed event stream out"
```

---

## Task 6: CLAUDE.md Sync + Full-Suite Verification

**Files:**
- Modify: `CLAUDE.md` (Implementation Status section)

**Interfaces:**
- Consumes: everything shipped in Tasks 1–5 (documentation only — no code changes).

- [ ] **Step 1: Add the Session 5 status section to CLAUDE.md**

In `CLAUDE.md`, immediately after the "### Session 3 — WhisperX Transcription Vector (complete)" block (i.e., before "### Known gaps & next-session priorities"), insert:

```markdown
### Session 5 — ScoreEvent Streaming Contract + Replayer (complete)

Closes gap #2 at the contract layer (scope A: no live infra). Design spec:
`docs/superpowers/specs/2026-07-08-scoreevent-streaming-design.md`.

| Component | Path | Status |
|---|---|---|
| Schemas | `backend/shared/schemas/score_event.py` | shipped — `ScoreEventKind`, `StreamScorerConfig`, `ScoreEvent` (cumulative + recent per event), `validate_event_stream` |
| Windowed scorer | `backend/ml-inference/app/pipelines/streaming/windowed_scorer.py` | shipped — `stream_scores()` strictly causal sync generator; ticks every `tick_seconds`, skips silent ticks, `recent=None` when sparse |
| Replayer | `backend/ml-inference/app/pipelines/streaming/replayer.py` | shipped — `ScoreReplayer` wall-clock pacing (pace 0 = instant), injectable sleep, clean cancellation via generator close |
| CLI | `scripts/replay_scores.py` | shipped — `--transcript json` or `--video` (`--fake` offline), `--pace/--tick/--recent-window` |
| Tests | `tests/streaming/` | shipped — schema contract, causality gate (future-mutation invariance), **batch-convergence acceptance gate** (`test_convergence.py`: FINAL == batch field-for-field), pacing/cancellation, CLI smoke |

Key decisions (locked in the spec — do not re-litigate): sync generator core
with the async FastAPI/WebSocket shell as a documented future session
(decision #4: per-tick work is CPU-bound; the shell drives the generator via
`asyncio.to_thread(next, gen)`); each interim carries cumulative + recent;
exactly one FINAL, equal to batch; gap #4 (hedging/certainty double-count)
deliberately deferred until after convergence landed, so the gate's baseline
stayed stable.
```

- [ ] **Step 2: Mark gap #2 resolved**

In "### Known gaps & next-session priorities", replace item 2 (the "**No real-time path exists.**" paragraph) with:

```markdown
2. ~~No real-time path exists~~ **RESOLVED at the contract layer (Session 5):**
   `ScoreEvent` schema + causal windowed scorer + replayer shipped; batch and
   stream converge on one contract, enforced by
   tests/streaming/test_convergence.py (FINAL == batch, field-for-field).
   Remaining for a true live surface (its own session): async
   FastAPI/WebSocket shell around the sync generator
   (`asyncio.to_thread(next, gen)`, per-session cancellation), incremental
   transcription, and platform media ingest. The windowed events already
   power the report's scrubbable score timeline directly.
```

- [ ] **Step 3: Run the full suite**

```bash
export PATH="$PATH:/c/Users/rphos/AppData/Local/Microsoft/WinGet/Links"
.venv/Scripts/python -m pytest tests/ -p no:cacheprovider
```

Expected: **128 passed, 1 deselected** (95 baseline + 33 new streaming tests). No skips introduced, no warnings from the new modules.

Then coverage on the new code only:

```bash
.venv/Scripts/python -m pytest tests/streaming/ -q --cov=backend/shared/schemas/score_event.py --cov=backend/ml-inference/app/pipelines/streaming --cov-report=term-missing
```

Expected: every streaming file ≥ 90% (ML-pipeline standard). If below, add the missing tests — do not ship uncovered branches.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: close gap #2 at the contract layer -- ScoreEvent streaming shipped (Session 5)"
```

---

## Final Verification

- [ ] `export PATH="$PATH:/c/Users/rphos/AppData/Local/Microsoft/WinGet/Links" && .venv/Scripts/python -m pytest tests/ -p no:cacheprovider` → 128 passed, 1 deselected (slow).
- [ ] `.venv/Scripts/python -m pytest tests/streaming/test_convergence.py -q` → 3 passed (THE gate).
- [ ] CLI hand-check (see Task 5 Step 4) → paced interim + final lines, disclaimer, exit 0.
- [ ] `git status --short` → only untracked local scratch (`.mcp.json`, `.superpowers/`), no unstaged source changes.
- [ ] `git log --oneline dev-build..HEAD` → 7 commits (plan + 6 task commits).

Note: `--video` mode needs demo media; `demo_data/` does not exist on this laptop (traveled by USB/cloud only). The `--transcript` path plus the compression suite (already green) covers the wiring; run a `--video --fake` spot-check on the desktop when convenient.

---

## Self-Review Checklist

- [x] Spec coverage: schemas + contract rules (T1), tick schedule / causality / sparse windows / boundary rule / zero-statement / sorting / analyzer reuse (T2), replayer pacing + injectable sleep + cancellation (T3), convergence gate on fake-canned AND synthetic 3-minute transcripts (T4), CLI with both input modes + disclaimer (T5), CLAUDE.md gap #2 + status sync (T6). Out-of-scope items (WebSocket shell, incremental ASR, protobuf events, gap #4) stay out.
- [x] Type consistency: `stream_scores(transcript, config=None, analyzer=None)`, `ScoreReplayer(transcript, config).replay(pace, sleep)`, `validate_event_stream(events)`, conftest helpers `make_pscore(composite, statement_count)` / `make_transcript(spec)` — used with identical signatures in every task.
- [x] No placeholders: every code step contains complete code; every run step names the command and expected outcome.
- [x] Contract rules enforced twice: model validator (FINAL/vector_scores) + `validate_event_stream` (ordering/single-FINAL), and both are directly tested.
- [x] Causality is tested by mutation-invariance, not by inspecting implementation internals.
- [x] Analyzer empty-slice precondition respected: scorer skips empty cumulative slices and guards recent by `min_recent_statements >= 1`; `analyze()` is never called with `[]`.
- [x] Invariants: #3 (scorer/replayer log counts only), #5 (module docstrings + CLI disclaimer), #6 (CLI smoke asserts the forbidden phrase is absent), #12 (`baseline_available` on every event; asserted in T2/T4).
- [x] Laptop environment honored: no new deps, no torch, no demo_data required by any test; ffmpeg PATH note included for full-suite runs.
