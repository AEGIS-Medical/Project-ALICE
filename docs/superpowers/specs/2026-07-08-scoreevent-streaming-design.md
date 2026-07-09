# ALICE — ScoreEvent Streaming Contract + Windowed Scorer + Replayer (Session 5)
## Design Spec · 2026-07-08

---

## Overview

Closes CLAUDE.md known-gap #2 ("no real-time path exists") at the layer the gap
actually prescribes: **define the ScoreEvent streaming schema before building any
live surface, so live and batch converge on one contract** — then prove the
contract with a causal windowed scorer and a replay consumer.

```
Transcript (existing, timestamped segments)
      │  stream_scores(transcript, config)      ← causal windowed core (sync generator)
      ▼
ScoreEvent, ScoreEvent, ... , ScoreEvent(final)
      │  ScoreReplayer(pace=1.0)                ← first consumer: wall-clock pacing
      ▼
CLI / timeline UX / (future) WebSocket shell
```

**Decisions locked during brainstorming:**

1. **Scope A: contract + replay consumer.** No live capture, no WebSocket server,
   no incremental transcription this session. The replayer takes an
   already-transcribed recording and re-emits its analysis as a timed stream of
   events — *live-computed* (strictly causal: each event uses only data up to its
   stream time) over a *replayed* source. For the upload business this is not a
   simulation: the windowed events power the report's scrubbable score timeline
   (CLAUDE.md UX screens 6a/7a/7b) directly.
2. **Each interim event carries BOTH readings:** `cumulative` (0→now — the steady
   gauge; converges to the batch score) and `recent` (last N seconds — the moment
   detector; absent when the window has too little speech).
3. **Final event is authoritative and must equal batch.** After the last segment,
   exactly one `kind="final"` event is emitted whose `cumulative` equals
   `PsycholinguisticAnalyzer().analyze(transcript.statements())` field-for-field.
   This convergence is the session's automated acceptance gate.
4. **Sync generator core; async shell deferred by design, not accident.** The
   scorer's per-tick work is CPU-bound (spaCy + lexicons); Python async cannot make
   CPU-bound work concurrent — an async-native core would still offload to a thread
   pool internally while adding pytest-asyncio and dual calling conventions
   everywhere. The core is therefore a plain pull-based generator (inherently
   backpressure-safe: the producer computes only when asked). **The future live
   service session commits to an async FastAPI/WebSocket shell** that drives this
   generator via `asyncio.to_thread(next, gen)` with per-session cancellation —
   async exactly where it earns its keep (socket I/O), sync where async cannot help
   (compute). `ScoreEvent` is serializable (Pydantic → JSON/protobuf), so a
   non-Python fan-out gateway remains possible at the edge without touching scoring.
5. **Latency reality:** per-tick scoring is ~10–50 ms against a 5 s tick and a
   2–3 s live-mode latency target dominated by ASR. Python is not the bottleneck
   at this layer.

---

## File Structure

```
backend/shared/schemas/score_event.py        NEW — ScoreEventKind, StreamScorerConfig, ScoreEvent
backend/ml-inference/app/pipelines/streaming/
  __init__.py                                NEW — re-exports
  windowed_scorer.py                         NEW — stream_scores() causal generator
  replayer.py                                NEW — ScoreReplayer (pacing wrapper)
scripts/replay_scores.py                     NEW — CLI: transcript/video in, event stream out
tests/streaming/
  __init__.py                                NEW
  conftest.py                                NEW — sys.path bridge (mirrors tests/transcription)
  test_score_event_schema.py                 NEW
  test_windowed_scorer.py                    NEW — causality, windowing, edge cases
  test_convergence.py                        NEW — THE gate: final == batch
  test_replayer.py                           NEW — pacing, cancellation
  test_cli_smoke.py                          NEW
```

Follows the established service-root pattern: streaming logic under the hyphenated
`backend/ml-inference/` (sys.path import, `from app.pipelines.streaming...`),
schemas in `backend/shared/schemas/` (dotted import via editable install).

---

## Schemas (`backend/shared/schemas/score_event.py`)

All frozen Pydantic v2 (`ConfigDict(frozen=True, extra="forbid")`), matching
`media.py` / `psycholinguistic.py` / `transcription.py`.

```python
class ScoreEventKind(str, Enum):
    INTERIM = "interim"
    FINAL = "final"


class StreamScorerConfig(BaseModel):
    tick_seconds: float = 5.0            # gt=0 — interim event cadence (stream time)
    recent_window_seconds: float = 30.0  # ge=tick_seconds — the "moment detector" span
    min_recent_statements: int = 2       # ge=1 — below this, recent=None (no fabricated scores)


class ScoreEvent(BaseModel):
    schema_version: int = 1              # version-locked contract, like ALTM's header
    kind: ScoreEventKind
    stream_time_seconds: float           # ge=0 — call-time position this event describes
    cumulative: PsycholinguisticScore    # 0 -> stream_time; on FINAL: whole recording
    recent: Optional[PsycholinguisticScore] = None
                                         # last recent_window_seconds; None if sparse;
                                         # always None on FINAL (whole-recording read is cumulative)
    vector_scores: dict[str, float]      # vector name -> 0-100 composite;
                                         # today {"psycholinguistic": ...}; future vectors
                                         # extend the dict — the schema does not change
    statement_count_so_far: int          # ge=0
    baseline_available: bool             # carried from analyzer (False until baselines exist)
    confidence: Literal["low", "medium", "high"]
```

Contract rules (enforced by validator or scorer, tested either way):

- `FINAL` events have `recent is None` and represent the complete recording.
- `vector_scores["psycholinguistic"] == cumulative.composite_score` (single-vector
  era; the dict exists so AU/tonality/contradiction/gaze join without a schema bump).
- Events in a stream have strictly increasing `stream_time_seconds`; exactly one
  `FINAL`, always last.
- CLAUDE.md invariant #5 note in the module docstring: raw scores/events are
  ensemble- and developer-facing; any user surface must add calibration,
  confidence display, and qualitative labels.

---

## Windowed Scorer (`windowed_scorer.py`)

```python
def stream_scores(
    transcript: Transcript,
    config: StreamScorerConfig | None = None,
    analyzer: PsycholinguisticAnalyzer | None = None,
) -> Iterator[ScoreEvent]:
```

Semantics:

- **Tick schedule:** interim events at `t = tick, 2*tick, ...` up to the last
  segment's `end_seconds`. (A recording shorter than one tick emits no interim
  events — only the final.)
- **Strict causality:** the cumulative slice at tick `t` is every segment with
  `end_seconds <= t`; the recent slice additionally requires
  `end_seconds > t - recent_window_seconds`. No lookahead, ever. (This is what
  makes the replayed stream identical to a genuinely live one.)
- **Sparse windows:** if the cumulative slice is empty at tick `t`, the tick is
  skipped (no event — silence at the start of a call produces no scores). If the
  recent slice has fewer than `min_recent_statements`, `recent=None`.
- **Final event:** after the last tick, exactly one `FINAL` over all statements —
  emitted even if no interim event ever fired (e.g., all speech in the first
  seconds), provided the transcript has ≥1 statement. A zero-statement transcript
  yields an empty stream (no events; the caller already knows duration from the
  transcript — nothing is fabricated).
- **Boundary rule:** if the recording's end coincides exactly with a tick boundary,
  that tick is superseded by the `FINAL` (no interim is emitted at the same
  timestamp) — preserving strictly increasing `stream_time_seconds` with the
  `FINAL` last.
- **Analyzer reuse:** one `PsycholinguisticAnalyzer` instance across all ticks
  (spaCy loads once); injectable for tests. Two `analyze()` calls per tick
  (cumulative + recent) at ~10–50 ms each against a 5 s tick — negligible.
- **Logging (invariant #3):** tick count, statement counts, timing — never
  transcript text.

## Replayer (`replayer.py`)

```python
class ScoreReplayer:
    def __init__(self, transcript: Transcript,
                 config: StreamScorerConfig | None = None) -> None: ...
    def replay(self, pace: float = 1.0,
               sleep: Callable[[float], None] = time.sleep) -> Iterator[ScoreEvent]:
```

- `pace=1.0`: wall-clock pacing (1 s of call = 1 s of replay — the demo mode);
  `pace=2.0`: double speed; `pace=0`: instant (no sleeping — test/batch mode).
- Sleeps the *gap to the next event's stream time*, then yields it; the sleep
  function is injectable so pacing tests assert requested sleep durations instead
  of actually sleeping.
- Pacing is its entire job — no scoring logic. Stopping iteration (consumer
  `break`/`close()`) cleanly abandons the underlying generator (`GeneratorExit`),
  which is the cancellation story a future socket session reuses per-connection.

## CLI (`scripts/replay_scores.py`)

- Input: `--video path` (runs the existing CompressionPipeline → Transcriber with
  `--fake` supported, mirroring `test_compress_and_analyze.py`) or
  `--transcript path.json` (a serialized Transcript, for offline/dev use).
- Flags: `--pace 1.0` (0 = instant), `--tick 5`, `--recent-window 30`.
- Output: one line per event — `[t=  25s] interim  cumulative=48.2 recent=61.0 (conf: low, stmts: 12)` —
  and a `final` line, plus the standard dev-tool anomaly disclaimer (invariant #5/#6
  compliant: no "lie detector," no bare-score guidance for end users).
- sys.path bridge to `backend/ml-inference` (same as existing scripts).

---

## Error Handling

| Condition | Behavior |
|---|---|
| Zero-statement transcript | Empty stream: no interim, no final — nothing fabricated |
| Cumulative slice empty at a tick (leading silence) | Tick skipped, no event |
| Recent slice below `min_recent_statements` | `recent=None` on that event |
| `tick_seconds <= 0` / `recent_window_seconds < tick_seconds` / `min_recent_statements < 1` | `ValidationError` at config construction |
| Analyzer raises on a slice (unexpected — slices are non-empty by construction) | Propagates; the stream dies loudly rather than emitting wrong scores |
| Consumer abandons mid-stream | Generator closes cleanly (`GeneratorExit`); no resource leak (no files/sockets held) |
| Out-of-order transcript segments | Scorer sorts by `end_seconds` once at start — WhisperX emits ordered segments, but the contract does not rely on it |

Invariants touched: **#3** (no transcript text in logs), **#5** (raw events are not
user-facing; documented), **#6** (no forbidden phrase), **#12** (`baseline_available`
travels with every event).

---

## Testing

| File | Coverage |
|---|---|
| `test_score_event_schema.py` | frozen/extra-forbid; kind enum; FINAL-has-no-recent rule; `vector_scores` consistency; strictly-increasing stream times helper; config validation bounds |
| `test_windowed_scorer.py` | **Causality gate:** build transcript T1, score to tick t; mutate all segments after t (different text) → events up to t are field-for-field identical. Tick schedule; leading-silence skip; sparse-recent → None; single-statement stream (final only); zero-statement stream (empty); determinism (two runs identical) |
| `test_convergence.py` | **THE acceptance gate:** `stream_scores(...)` last event is FINAL and its `cumulative` equals `PsycholinguisticAnalyzer().analyze(transcript.statements())` exactly (all 8 dimensions + composite + counts). Run against (a) the fake-backend canned transcript and (b) a synthetic 40-statement, 3-minute transcript |
| `test_replayer.py` | pace=0 yields immediately in order; pace=1 requests sleeps equal to inter-event gaps (injected fake sleep); pace=2 halves them; early `break` closes the generator (assert via generator finalization) |
| `test_cli_smoke.py` | subprocess: `--transcript <tmp json> --pace 0` → exit 0, ≥1 interim line, exactly one final line, disclaimer present; bad path → exit 1 |

All tests use the existing fake/synthetic transcripts — no torch, no downloads, no
real-time sleeping (injected sleep). Coverage target: 90 % (ML-pipeline standard).

---

## Out of Scope (future sessions — recorded so scope creep has to be deliberate)

- **Live service shell:** async FastAPI/WebSocket server driving this generator via
  `asyncio.to_thread`, per-session cancellation, auth, consent-token checks. The
  design decision (async shell over sync core) is made HERE; the build is its own
  session.
- Incremental/streaming transcription (WhisperX on growing audio) — the replayer
  consumes completed transcripts.
- Multi-vector fusion events (AU/tonality/contradiction join `vector_scores` when
  those vectors exist; XGBoost ensemble replaces the single-vector composite at
  fusion time).
- ScoreEvent protobuf serialization for the mobile client (JSON via Pydantic is
  sufficient until the KMP client session; the ALTM version-locking pattern applies
  when needed).
- Hedging/certainty word-list disjoint (gap #4) — deliberately not ridden into this
  session; it changes composite values and would muddy the convergence gate's
  baseline. Do it immediately after, when batch and stream move together.
