# Live Service Async Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI/WebSocket service that runs scoring sessions server-side (detached from clients) and streams `{session_id, seq, event}` envelopes to any number of subscribers, turning Session 5's replayer into a connectable real-time surface.

**Architecture:** REST creates/inspects/cancels sessions; a WebSocket per subscriber views a session's event stream. Each session runs the sync `ScoreReplayer` in one worker thread (`asyncio.to_thread`), publishing via `loop.call_soon_threadsafe` into an `InProcessPublisher` (seq, ring buffer, fan-out) — the v2 Kafka seam. Cancellation rides Session 5's injectable sleep. A TTL reaper collects terminal sessions.

**Tech Stack:** Python 3.13, FastAPI + uvicorn[standard] (new deps), Pydantic v2, existing `ScoreReplayer`/`stream_scores`/`Transcript` stack, pytest with FastAPI's TestClient (REST + WS in-process).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-19-live-service-async-shell-design.md` — governs on conflict.
- Run everything from `C:\Users\ryanh\ALICE\Project-ALICE` with `.venv/Scripts/python`.
- Hyphenated service root: service code lives under `backend/ml-inference/app/service/`; tests/scripts bridge via `sys.path` insert of `backend/ml-inference` (mirror `tests/streaming/conftest.py`).
- Wire contract (exact): data frame `{"session_id": str, "seq": int, "event": {ScoreEvent JSON}}`; terminal frame `{"session_id": str, "state": "finished"|"cancelled"|"failed", "reason": str|null}` — exactly one per completed stream, always last. `seq` starts at 0, strictly increasing per session.
- WS close codes: **4404** unknown/reaped session; **4408** slow client (subscriber queue overflow). Slow-client drop never touches the session.
- Detached semantics: client disconnect NEVER cancels a session; only `DELETE` or terminal completion ends it. `DELETE` on a terminal session is 200 idempotent no-op.
- Catch-up: `?last_seq=N` (default −1) replays buffered events with `seq > N`; if `N` predates the ring window, silently start at the oldest buffered event.
- Failure semantics: `UnsupportedLanguageError` → `FAILED`, reason = the exception message (language code only — invariant #3); any other scorer exception → `FAILED`, reason = exception class name only, full traceback to server log; zero-statement transcript → immediate `FINISHED`, zero data frames, terminal frame only.
- `LiveServiceConfig` defaults (frozen): `host="127.0.0.1"`, `port=8710`, `ring_size=256` (ge=1), `subscriber_queue_size=64` (ge=1), `session_ttl_seconds=900.0` (gt=0), `reaper_interval_seconds=5.0` (gt=0).
- Auth: none in v1 (localhost bind default) — documented posture, do not add.
- Invariants: #3 logs/reasons carry codes/counts/states, never transcript text; #5 module docstrings note frames are dev/ensemble-facing; #6 never "lie detector".
- All tests: `pace=0`, fake/synthetic transcripts, sub-second reaper TTLs, no real sleeping, no network, no models beyond spaCy.

---

## File Map

| File | Action | Task |
|---|---|---|
| `pyproject.toml` | Modify — add fastapi, uvicorn[standard] | 1 |
| `backend/ml-inference/app/service/__init__.py` | Create | 1 |
| `backend/ml-inference/app/service/config.py` | Create | 1 |
| `tests/service/__init__.py`, `tests/service/conftest.py` | Create | 1 |
| `tests/service/test_config.py` | Create | 1 |
| `backend/ml-inference/app/service/sessions.py` | Create | 2 |
| `tests/service/test_sessions.py` | Create | 2 |
| `backend/ml-inference/app/service/publisher.py` | Create | 3 |
| `tests/service/test_publisher.py` | Create | 3 |
| `backend/ml-inference/app/service/runner.py` | Create | 4 |
| `tests/service/test_runner.py` | Create | 4 |
| `backend/ml-inference/app/service/app.py` | Create — REST | 5 |
| `tests/service/test_rest_api.py` | Create | 5 |
| `backend/ml-inference/app/service/app.py` | Extend — WS | 6 |
| `tests/service/test_websocket.py` | Create | 6 |
| `scripts/run_live_service.py`, `scripts/live_client.py` | Create | 7 |
| `Makefile` (`live` target), `CLAUDE.md` (status sync) | Modify | 7 |

---

## Task 1: Dependencies + LiveServiceConfig + Test Scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `backend/ml-inference/app/service/__init__.py`, `backend/ml-inference/app/service/config.py`
- Create: `tests/service/__init__.py`, `tests/service/conftest.py`
- Test: `tests/service/test_config.py`

**Interfaces:**
- Produces: `LiveServiceConfig` (frozen Pydantic) with the Global-Constraints defaults/bounds. `tests/service/conftest.py` provides the `sys.path` bridge plus fixtures `fast_config` (sub-second TTLs: `session_ttl_seconds=0.2`, `reaper_interval_seconds=0.05`) and `transcript_file(tmp_path)` factory — writes a 3-segment English `Transcript` JSON (reuse the shape from `tests/streaming/test_cli_smoke.py`'s builder: `TranscriptSegment(text=..., start_seconds=..., end_seconds=...)`, `language="en"`, `backend="fake"`) and returns the `Path`; accepts `language="en"` override and `texts` override.

- [ ] **Step 1: Add deps and install**

In `pyproject.toml` dependencies, after `"protobuf>=5.26",` add:

```toml
    # Live service async shell (backend/ml-inference/app/service). The api-gateway
    # session will front this later; localhost-bind dev posture for v1.
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
```

Run: `.venv/Scripts/python -m pip install -e ".[dev]" -q && .venv/Scripts/python -c "import fastapi, uvicorn; print('web ok')"`
Expected: `web ok`

- [ ] **Step 2: Write conftest + failing config test**

Create `tests/service/__init__.py` (empty) and `tests/service/conftest.py`:

```python
"""Fixtures for the live-service suite (sys.path bridge + fast configs)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ML_INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "backend" / "ml-inference"
if str(_ML_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_INFERENCE_ROOT))


@pytest.fixture
def fast_config():
    from app.service.config import LiveServiceConfig

    return LiveServiceConfig(session_ttl_seconds=0.2, reaper_interval_seconds=0.05)


@pytest.fixture
def transcript_file(tmp_path: Path):
    """Factory: write a small Transcript JSON, return its Path."""
    from backend.shared.schemas.transcription import Transcript, TranscriptSegment

    def _make(language: str = "en", texts: list[str] | None = None) -> Path:
        texts = texts or [
            "I think I was at home that night.",
            "I never went anywhere near there.",
            "Honestly, you know, I'm not really sure.",
        ]
        segments = [
            TranscriptSegment(
                text=t, start_seconds=2.0 * i, end_seconds=2.0 * i + 1.8
            )
            for i, t in enumerate(texts)
        ]
        transcript = Transcript(
            segments=segments,
            language=language,
            audio_duration_seconds=2.0 * len(texts),
            model_name="fake-distil",
            backend="fake",
        )
        p = tmp_path / f"transcript_{language}_{len(texts)}.json"
        p.write_text(transcript.model_dump_json(), encoding="utf-8")
        return p

    return _make
```

Create `tests/service/test_config.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.service.config import LiveServiceConfig


def test_defaults():
    c = LiveServiceConfig()
    assert c.host == "127.0.0.1"
    assert c.port == 8710
    assert c.ring_size == 256
    assert c.subscriber_queue_size == 64
    assert c.session_ttl_seconds == 900.0
    assert c.reaper_interval_seconds == 5.0


@pytest.mark.parametrize(
    "field,value",
    [("ring_size", 0), ("subscriber_queue_size", 0),
     ("session_ttl_seconds", 0.0), ("reaper_interval_seconds", 0.0)],
)
def test_bounds(field, value):
    with pytest.raises(ValidationError):
        LiveServiceConfig(**{field: value})


def test_frozen():
    c = LiveServiceConfig()
    with pytest.raises(ValidationError):
        c.port = 9999
```

Run: `.venv/Scripts/python -m pytest tests/service/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.service'`.

- [ ] **Step 3: Create the package + config**

Create `backend/ml-inference/app/service/__init__.py`:

```python
"""Live service async shell (Session 7).

FastAPI/WebSocket surface over the streaming scorer. CLAUDE.md invariant #5:
every frame this service emits is ensemble/developer-facing; user surfaces
must add calibration, confidence display, and qualitative labels.
"""
```

Create `backend/ml-inference/app/service/config.py`:

```python
"""Live-service configuration (frozen, launcher-flag overridable)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LiveServiceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(default="127.0.0.1", description="Bind host (localhost = v1 auth posture).")
    port: int = Field(default=8710, ge=1, le=65535)
    ring_size: int = Field(default=256, ge=1, description="Per-session event ring buffer length.")
    subscriber_queue_size: int = Field(default=64, ge=1, description="Per-connection queue; overflow drops that connection (4408).")
    session_ttl_seconds: float = Field(default=900.0, gt=0.0, description="Reap terminal (or stuck-CREATED) sessions after this long.")
    reaper_interval_seconds: float = Field(default=5.0, gt=0.0)
```

Run: `.venv/Scripts/python -m pytest tests/service/test_config.py -q`
Expected: all pass (7).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml backend/ml-inference/app/service/ tests/service/
git commit -m "feat(service): live-service scaffolding -- deps, LiveServiceConfig, test bridge"
```

---

## Task 2: Sessions + Manager + Reaper

**Files:**
- Create: `backend/ml-inference/app/service/sessions.py`
- Test: `tests/service/test_sessions.py`

**Interfaces:**
- Consumes: `LiveServiceConfig` (Task 1). Publisher is attached in Task 3 — here it is an opaque `object | None` slot.
- Produces (exact, later tasks depend on these):
  - `SessionState(str, Enum)`: `CREATED, RUNNING, FINISHED, CANCELLED, FAILED`; `TERMINAL_STATES: frozenset[SessionState]`.
  - `class Session`: fields `id: str` (uuid4 hex), `state: SessionState`, `created_at: float`, `terminal_at: float | None`, `reason: str | None`, `cancel_event: threading.Event`, `publisher: object | None`, `stream_time_seconds: float`, `statement_count: int`, `language: str | None`, `subscriber_count: int` (property delegating to publisher if set, else 0); methods `to_summary() -> dict`, `to_detail() -> dict`.
  - `class SessionManager(config: LiveServiceConfig)`: `create() -> Session` (state CREATED, registered); `get(session_id) -> Session | None`; `list_sessions() -> list[Session]`; `mark_running(session)`; `mark_terminal(session, state: SessionState, reason: str | None = None)` (sets terminal_at, no-op if already terminal); `cancel(session_id) -> Session | None` (sets cancel_event + marks CANCELLED if not terminal; returns session or None if unknown); `async reaper_loop()` (every `reaper_interval_seconds`: drop sessions terminal for > ttl; force-cancel sessions still CREATED after > ttl); `reap_once()` sync core of the loop (separately testable).

- [ ] **Step 1: Write the failing tests**

Create `tests/service/test_sessions.py`:

```python
from __future__ import annotations

import time

from app.service.config import LiveServiceConfig
from app.service.sessions import (
    TERMINAL_STATES,
    SessionManager,
    SessionState,
)


def _mgr(**kw) -> SessionManager:
    return SessionManager(LiveServiceConfig(**kw))


def test_create_registers_unique_created_sessions():
    mgr = _mgr()
    a, b = mgr.create(), mgr.create()
    assert a.id != b.id
    assert a.state is SessionState.CREATED
    assert mgr.get(a.id) is a
    assert {s.id for s in mgr.list_sessions()} == {a.id, b.id}


def test_lifecycle_transitions():
    mgr = _mgr()
    s = mgr.create()
    mgr.mark_running(s)
    assert s.state is SessionState.RUNNING
    mgr.mark_terminal(s, SessionState.FINISHED)
    assert s.state in TERMINAL_STATES
    assert s.terminal_at is not None


def test_mark_terminal_is_first_writer_wins():
    mgr = _mgr()
    s = mgr.create()
    mgr.mark_terminal(s, SessionState.FAILED, reason="X")
    mgr.mark_terminal(s, SessionState.CANCELLED)
    assert s.state is SessionState.FAILED
    assert s.reason == "X"


def test_cancel_sets_event_and_state():
    mgr = _mgr()
    s = mgr.create()
    mgr.mark_running(s)
    out = mgr.cancel(s.id)
    assert out is s
    assert s.cancel_event.is_set()
    assert s.state is SessionState.CANCELLED


def test_cancel_terminal_is_idempotent_and_unknown_is_none():
    mgr = _mgr()
    s = mgr.create()
    mgr.mark_terminal(s, SessionState.FINISHED)
    assert mgr.cancel(s.id) is s          # no state change
    assert s.state is SessionState.FINISHED
    assert mgr.cancel("nope") is None


def test_reap_once_collects_expired_terminal_and_stuck_created():
    mgr = _mgr(session_ttl_seconds=0.01, reaper_interval_seconds=0.01)
    done = mgr.create()
    mgr.mark_terminal(done, SessionState.FINISHED)
    stuck = mgr.create()                  # never started
    fresh = mgr.create()
    mgr.mark_running(fresh)
    time.sleep(0.02)
    mgr.reap_once()
    assert mgr.get(done.id) is None       # expired terminal removed
    assert stuck.state is SessionState.CANCELLED  # force-cancelled (removed on a later pass)
    assert mgr.get(fresh.id) is fresh     # running untouched


def test_summary_and_detail_shapes():
    mgr = _mgr()
    s = mgr.create()
    summary = s.to_summary()
    assert set(summary) >= {"session_id", "state", "created_at",
                            "stream_time_seconds", "last_seq", "subscriber_count"}
    detail = s.to_detail()
    assert set(detail) >= set(summary) | {"language", "statement_count", "reason"}
```

Run: `.venv/Scripts/python -m pytest tests/service/test_sessions.py -q`
Expected: FAIL — module missing.

- [ ] **Step 2: Implement `sessions.py`**

```python
"""Session state, registry, and TTL reaper for the live service.

Sessions are DETACHED (spec decision #3): they run server-side regardless of
watchers; a WebSocket is a view, never a lifeline. Only DELETE (or terminal
completion) ends a session. This module is the future session-service
extraction point. Invariant #3: reasons/logs carry codes and states only.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Optional

from app.service.config import LiveServiceConfig

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_STATES: frozenset[SessionState] = frozenset(
    {SessionState.FINISHED, SessionState.CANCELLED, SessionState.FAILED}
)


class Session:
    """One scoring session's mutable state (registry-owned)."""

    def __init__(self) -> None:
        self.id: str = uuid.uuid4().hex
        self.state: SessionState = SessionState.CREATED
        self.created_at: float = time.time()
        self.terminal_at: Optional[float] = None
        self.reason: Optional[str] = None
        self.cancel_event: threading.Event = threading.Event()
        self.publisher: Optional[object] = None  # InProcessPublisher (Task 3)
        self.stream_time_seconds: float = 0.0
        self.statement_count: int = 0
        self.language: Optional[str] = None

    @property
    def subscriber_count(self) -> int:
        pub = self.publisher
        return getattr(pub, "subscriber_count", 0) if pub is not None else 0

    @property
    def last_seq(self) -> int:
        pub = self.publisher
        return getattr(pub, "last_seq", -1) if pub is not None else -1

    def to_summary(self) -> dict:
        return {
            "session_id": self.id,
            "state": self.state.value,
            "created_at": self.created_at,
            "stream_time_seconds": self.stream_time_seconds,
            "last_seq": self.last_seq,
            "subscriber_count": self.subscriber_count,
        }

    def to_detail(self) -> dict:
        return {
            **self.to_summary(),
            "language": self.language,
            "statement_count": self.statement_count,
            "reason": self.reason,
        }


class SessionManager:
    """Registry + lifecycle transitions + reaper. Event-loop-thread only
    (mutations arrive via call_soon_threadsafe from workers)."""

    def __init__(self, config: LiveServiceConfig) -> None:
        self._config = config
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session()
        self._sessions[session.id] = session
        logger.info("session_created id=%s", session.id)
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[Session]:
        return list(self._sessions.values())

    def mark_running(self, session: Session) -> None:
        if session.state is SessionState.CREATED:
            session.state = SessionState.RUNNING

    def mark_terminal(
        self, session: Session, state: SessionState, reason: Optional[str] = None
    ) -> None:
        """First terminal writer wins; later calls are no-ops."""
        if session.state in TERMINAL_STATES:
            return
        if state not in TERMINAL_STATES:
            raise ValueError(f"{state} is not terminal")
        session.state = state
        session.reason = reason
        session.terminal_at = time.time()
        logger.info(
            "session_terminal id=%s state=%s", session.id, state.value
        )

    def cancel(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.cancel_event.set()
        self.mark_terminal(session, SessionState.CANCELLED)
        return session

    # ---- reaper -----------------------------------------------------------

    def reap_once(self) -> None:
        now = time.time()
        ttl = self._config.session_ttl_seconds
        for session in list(self._sessions.values()):
            if session.state in TERMINAL_STATES:
                if session.terminal_at is not None and now - session.terminal_at > ttl:
                    del self._sessions[session.id]
                    logger.info("session_reaped id=%s", session.id)
            elif session.state is SessionState.CREATED and now - session.created_at > ttl:
                session.cancel_event.set()
                self.mark_terminal(session, SessionState.CANCELLED)
                logger.info("session_reaped_stuck id=%s", session.id)

    async def reaper_loop(self) -> None:
        import asyncio

        while True:
            await asyncio.sleep(self._config.reaper_interval_seconds)
            self.reap_once()
```

Run: `.venv/Scripts/python -m pytest tests/service/test_sessions.py -q`
Expected: 7 pass.

- [ ] **Step 3: Commit**

```bash
git add backend/ml-inference/app/service/sessions.py tests/service/test_sessions.py
git commit -m "feat(service): Session/SessionManager with detached lifecycle + TTL reaper"
```

---

## Task 3: InProcessPublisher (seq, ring buffer, fan-out, slow-client drop)

**Files:**
- Create: `backend/ml-inference/app/service/publisher.py`
- Test: `tests/service/test_publisher.py`

**Interfaces:**
- Consumes: `ScoreEvent` (existing schema).
- Produces (exact):
  - Module sentinel `DROPPED: object` — the marker a WS handler receives when its connection was dropped for slowness (close 4408).
  - `class InProcessPublisher(session_id: str, ring_size: int, queue_size: int)`:
    - `publish(event: ScoreEvent) -> None` — assigns next seq (starting 0), builds envelope dict `{"session_id", "seq", "event": event.model_dump(mode="json")}`, appends to ring, fans out.
    - `publish_terminal(state: str, reason: str | None = None) -> None` — terminal frame `{"session_id", "state", "reason"}`, appended to ring and fanned out; sets `self.terminated = True`.
    - `subscribe(last_seq: int = -1) -> asyncio.Queue` — new bounded queue preloaded with buffered frames whose `seq > last_seq` (terminal frame always included if present, regardless of seq); registers and returns it. Predates-window rule is automatic (ring only holds recent frames).
    - `unsubscribe(queue) -> None` — removes; idempotent.
    - Properties: `last_seq: int` (−1 before first publish), `subscriber_count: int`, `buffered: list[dict]` (copy).
  - Fan-out uses `put_nowait`; on `asyncio.QueueFull`: remove the subscriber, then make room for exactly one `DROPPED` marker (`get_nowait()` once, `put_nowait(DROPPED)`) so the handler always learns it was dropped. Event-loop-thread only (called via `call_soon_threadsafe`).
  - Terminal frames have NO `seq` key; data frames always do.

- [ ] **Step 1: Write the failing tests**

Create `tests/service/test_publisher.py`:

```python
from __future__ import annotations

import asyncio

import pytest

from app.service.publisher import DROPPED, InProcessPublisher
from backend.shared.schemas.psycholinguistic import (
    PsycholinguisticDimension,
    PsycholinguisticScore,
)
from backend.shared.schemas.score_event import ScoreEvent, ScoreEventKind


def _dim(score: float = 10.0) -> PsycholinguisticDimension:
    return PsycholinguisticDimension(score=score, evidence=[])


def _event(t: float, kind=ScoreEventKind.INTERIM) -> ScoreEvent:
    score = PsycholinguisticScore(
        pronoun_shift_score=_dim(), hedging_score=_dim(),
        cognitive_complexity_score=_dim(), emotional_distribution_score=_dim(),
        disfluency_score=_dim(), negation_score=_dim(),
        detail_specificity_score=_dim(), certainty_score=_dim(),
        composite_score=10.0, statement_count=1,
        baseline_available=False, confidence="low",
    )
    return ScoreEvent(
        kind=kind, stream_time_seconds=t, cumulative=score, recent=None,
        vector_scores={"psycholinguistic": 10.0},
        statement_count_so_far=1, baseline_available=False, confidence="low",
    )


def test_seq_starts_at_zero_and_increments():
    pub = InProcessPublisher("s1", ring_size=10, queue_size=8)
    assert pub.last_seq == -1
    pub.publish(_event(5.0))
    pub.publish(_event(10.0))
    assert pub.last_seq == 1
    assert [f["seq"] for f in pub.buffered] == [0, 1]
    assert pub.buffered[0]["session_id"] == "s1"
    assert pub.buffered[0]["event"]["stream_time_seconds"] == 5.0


def test_ring_trims_to_ring_size():
    pub = InProcessPublisher("s1", ring_size=3, queue_size=8)
    for i in range(6):
        pub.publish(_event(float(i)))
    seqs = [f["seq"] for f in pub.buffered]
    assert seqs == [3, 4, 5]


def test_subscribe_catch_up_and_predates_window():
    pub = InProcessPublisher("s1", ring_size=3, queue_size=8)
    for i in range(6):
        pub.publish(_event(float(i)))
    q = pub.subscribe(last_seq=1)          # 1 predates window (oldest is 3)
    got = [q.get_nowait()["seq"] for _ in range(q.qsize())]
    assert got == [3, 4, 5]                # silent start at oldest buffered


def test_two_subscribers_receive_identically():
    pub = InProcessPublisher("s1", ring_size=10, queue_size=8)
    q1, q2 = pub.subscribe(), pub.subscribe()
    assert pub.subscriber_count == 2
    pub.publish(_event(5.0))
    assert q1.get_nowait() == q2.get_nowait()


def test_terminal_frame_shape_and_always_included():
    pub = InProcessPublisher("s1", ring_size=10, queue_size=8)
    pub.publish(_event(5.0))
    pub.publish_terminal("finished")
    frames = pub.buffered
    assert frames[-1] == {"session_id": "s1", "state": "finished", "reason": None}
    assert "seq" not in frames[-1]
    # A late subscriber that has seen everything still gets the terminal frame.
    q = pub.subscribe(last_seq=0)
    got = [q.get_nowait() for _ in range(q.qsize())]
    assert got[-1]["state"] == "finished"


def test_slow_subscriber_dropped_with_marker_session_unaffected():
    pub = InProcessPublisher("s1", ring_size=64, queue_size=2)
    slow = pub.subscribe()
    fine = pub.subscribe()
    for i in range(4):                     # overflows queue_size=2
        pub.publish(_event(float(i)))
    assert pub.subscriber_count == 1       # slow removed
    drained = [slow.get_nowait() for _ in range(slow.qsize())]
    assert drained[-1] is DROPPED
    assert fine.qsize() == 0 or fine.qsize() > 0  # fine still registered
    pub.publish(_event(99.0))
    assert pub.last_seq == 4               # publishing never stalled


def test_unsubscribe_idempotent():
    pub = InProcessPublisher("s1", ring_size=4, queue_size=4)
    q = pub.subscribe()
    pub.unsubscribe(q)
    pub.unsubscribe(q)
    assert pub.subscriber_count == 0
```

> Note: `test_slow_subscriber...` — the `fine` queue also has size 2 and receives 4 events, so BOTH would overflow; fix the test intent by draining `fine` between publishes OR use `queue_size` large enough for `fine`. Implementer: replace the `fine` logic with: subscribe `fine` with a fresh publisher-level queue and drain it after each publish:
> ```python
>     for i in range(4):
>         pub.publish(_event(float(i)))
>         while fine.qsize():
>             fine.get_nowait()
> ```
> and assert `pub.subscriber_count == 1` after the loop (only `slow` dropped). Use this corrected form verbatim.

Run: `.venv/Scripts/python -m pytest tests/service/test_publisher.py -q`
Expected: FAIL — module missing.

- [ ] **Step 2: Implement `publisher.py`**

```python
"""In-process event publisher: the v2 Kafka seam (spec decision #2).

Assigns seq, keeps the per-session ring buffer, fans out to bounded
subscriber queues. A v2 bus publisher replaces this class on topic
`score-events.{session_id}` without touching scoring code. Event-loop-thread
only: workers publish via loop.call_soon_threadsafe.

Wire contract: data frames {"session_id", "seq", "event"}; exactly one
terminal frame {"session_id", "state", "reason"} (no "seq") ends a stream.
Slow subscribers are dropped (their queue receives the DROPPED sentinel);
the session is never stalled.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Optional

from backend.shared.schemas.score_event import ScoreEvent

logger = logging.getLogger(__name__)

# Sentinel a dropped-for-slowness subscriber finds at the end of its queue.
DROPPED: object = object()


class InProcessPublisher:
    def __init__(self, session_id: str, ring_size: int, queue_size: int) -> None:
        self._session_id = session_id
        self._queue_size = queue_size
        self._ring: deque[dict] = deque(maxlen=ring_size)
        self._subscribers: list[asyncio.Queue] = []
        self._next_seq = 0
        self.terminated = False

    # ---- properties -------------------------------------------------------
    @property
    def last_seq(self) -> int:
        return self._next_seq - 1

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def buffered(self) -> list[dict]:
        return list(self._ring)

    # ---- publishing -------------------------------------------------------
    def publish(self, event: ScoreEvent) -> None:
        frame = {
            "session_id": self._session_id,
            "seq": self._next_seq,
            "event": event.model_dump(mode="json"),
        }
        self._next_seq += 1
        self._ring.append(frame)
        self._fan_out(frame)

    def publish_terminal(self, state: str, reason: Optional[str] = None) -> None:
        frame = {"session_id": self._session_id, "state": state, "reason": reason}
        self.terminated = True
        self._ring.append(frame)
        self._fan_out(frame)

    def _fan_out(self, frame: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                self._subscribers.remove(q)
                try:
                    q.get_nowait()          # guarantee room for the marker
                except asyncio.QueueEmpty:  # pragma: no cover - full implies items
                    pass
                q.put_nowait(DROPPED)
                logger.info(
                    "subscriber_dropped session=%s slow_client", self._session_id
                )

    # ---- subscriptions ----------------------------------------------------
    def subscribe(self, last_seq: int = -1) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        for frame in self._ring:
            if "seq" in frame and frame["seq"] <= last_seq:
                continue
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                break  # catch-up larger than queue: client re-syncs via last_seq
        self._subscribers.append(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)
```

Run: `.venv/Scripts/python -m pytest tests/service/test_publisher.py -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add backend/ml-inference/app/service/publisher.py tests/service/test_publisher.py
git commit -m "feat(service): InProcessPublisher -- seq, ring buffer, fan-out, slow-client drop (Kafka seam)"
```

---

## Task 4: Worker-Thread Session Runner

**Files:**
- Create: `backend/ml-inference/app/service/runner.py`
- Test: `tests/service/test_runner.py`

**Interfaces:**
- Consumes: `Session`, `SessionManager`, `SessionState` (Task 2); `InProcessPublisher` (Task 3); existing `ScoreReplayer`, `StreamScorerConfig`, `Transcript`, `UnsupportedLanguageError`.
- Produces (exact):
  - `class SourceSpec(BaseModel)` (frozen): `transcript_path: Path | None = None`, `video_path: Path | None = None`, `fake: bool = True`, `mode: str = "edge_full"`; model_validator: exactly one of transcript_path/video_path set.
  - `async def start_session(session, manager, source: SourceSpec, scorer_config: StreamScorerConfig, pace: float, loop=None) -> asyncio.Task` — attaches publisher (caller did), marks RUNNING, spawns `asyncio.to_thread(_run_sync, ...)` wrapped so completion publishes the terminal frame ON THE LOOP and calls `manager.mark_terminal`.
  - `_run_sync(session, source, scorer_config, pace, loop) -> tuple[SessionState, str | None]` — loads the transcript (JSON path, or video via CompressionPipeline+Transcriber with Fake/WhisperX backend per `fake`), sets `session.language`/`statement_count`, then iterates `ScoreReplayer(...).replay(pace, sleep=hook)` publishing each event via `loop.call_soon_threadsafe(publisher.publish, event)` and updating `session.stream_time_seconds`; checks `session.cancel_event` each iteration AND inside the sleep hook (raise internal `_Cancelled`). Returns `(FINISHED, None)`, `(CANCELLED, None)`, `(FAILED, reason)` per Global-Constraints failure semantics (UnsupportedLanguageError → its message; other exceptions → class name only + `logger.exception`).
  - Terminal publication rule: if the outcome is CANCELLED but `manager` already marked the session CANCELLED (DELETE path), still publish exactly one terminal frame (guard on `publisher.terminated`).

- [ ] **Step 1: Write the failing tests**

Create `tests/service/test_runner.py`:

```python
from __future__ import annotations

import asyncio

import pytest

from app.service.config import LiveServiceConfig
from app.service.publisher import InProcessPublisher
from app.service.runner import SourceSpec, start_session
from app.service.sessions import SessionManager, SessionState
from backend.shared.schemas.score_event import StreamScorerConfig


def _setup(transcript_path):
    cfg = LiveServiceConfig()
    mgr = SessionManager(cfg)
    session = mgr.create()
    session.publisher = InProcessPublisher(
        session.id, cfg.ring_size, cfg.subscriber_queue_size
    )
    source = SourceSpec(transcript_path=transcript_path)
    return mgr, session, source


def _scfg() -> StreamScorerConfig:
    return StreamScorerConfig(tick_seconds=2.0, recent_window_seconds=4.0)


def test_source_spec_requires_exactly_one_source(tmp_path):
    with pytest.raises(Exception):
        SourceSpec()
    with pytest.raises(Exception):
        SourceSpec(transcript_path=tmp_path / "a.json", video_path=tmp_path / "b.mp4")


def test_session_runs_to_finished_with_events(transcript_file):
    async def go():
        mgr, session, source = _setup(transcript_file())
        task = await start_session(session, mgr, source, _scfg(), pace=0.0)
        await task
        return session

    session = asyncio.run(go())
    assert session.state is SessionState.FINISHED
    frames = session.publisher.buffered
    assert frames[-1]["state"] == "finished"
    assert any("seq" in f for f in frames)          # at least one data frame
    assert session.statement_count == 3
    assert session.language == "en"


def test_non_english_fails_with_code_only_reason(transcript_file):
    async def go():
        mgr, session, source = _setup(transcript_file(language="es"))
        task = await start_session(session, mgr, source, _scfg(), pace=0.0)
        await task
        return session

    session = asyncio.run(go())
    assert session.state is SessionState.FAILED
    assert "es" in session.reason and "not supported" in session.reason
    assert "Hola" not in (session.reason or "")
    assert session.publisher.buffered[-1]["state"] == "failed"


def test_zero_statement_transcript_finishes_with_terminal_only(transcript_file, tmp_path):
    from backend.shared.schemas.transcription import Transcript

    p = tmp_path / "empty.json"
    p.write_text(
        Transcript(segments=[], language="en", audio_duration_seconds=0.0,
                   model_name="fake-distil", backend="fake").model_dump_json(),
        encoding="utf-8",
    )

    async def go():
        mgr, session, source = _setup(p)
        task = await start_session(session, mgr, source, _scfg(), pace=0.0)
        await task
        return session

    session = asyncio.run(go())
    assert session.state is SessionState.FINISHED
    frames = session.publisher.buffered
    assert len(frames) == 1 and frames[0]["state"] == "finished"


def test_cancel_mid_stream_yields_cancelled_terminal(transcript_file):
    async def go():
        mgr, session, source = _setup(transcript_file())
        # pace=1 with a sleep hook: cancellation must interrupt promptly.
        task = await start_session(session, mgr, source, _scfg(), pace=1.0)
        await asyncio.sleep(0.05)          # let it start
        mgr.cancel(session.id)
        await asyncio.wait_for(task, timeout=5.0)
        return session

    session = asyncio.run(go())
    assert session.state is SessionState.CANCELLED
    assert session.publisher.buffered[-1]["state"] == "cancelled"
    terminals = [f for f in session.publisher.buffered if "state" in f]
    assert len(terminals) == 1             # exactly one terminal frame


def test_missing_transcript_file_fails(tmp_path):
    async def go():
        mgr, session, source = (
            lambda m, s, src: (m, s, src)
        )(*_setup(tmp_path / "missing.json"))
        task = await start_session(session, mgr, source, _scfg(), pace=0.0)
        await task
        return session

    session = asyncio.run(go())
    assert session.state is SessionState.FAILED
```

Run: `.venv/Scripts/python -m pytest tests/service/test_runner.py -q`
Expected: FAIL — module missing.

- [ ] **Step 2: Implement `runner.py`**

```python
"""Worker-thread session runner: sync replay loop -> async publisher.

One thread per session for its lifetime (asyncio.to_thread), publishing via
loop.call_soon_threadsafe -- the Session 5 bridging design realized.
Cancellation rides the replayer's injectable sleep: the hook raises when the
session's cancel_event is set, closing the scoring generator cleanly even
mid-pace; pace=0 sessions check the flag between events.

Invariant #3: failure reasons carry language codes or exception class names
only -- never transcript text. Full tracebacks go to the server log.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator

from app.pipelines.psycholinguistic.analyzer import UnsupportedLanguageError
from app.pipelines.streaming import ScoreReplayer
from app.service.sessions import Session, SessionManager, SessionState
from backend.shared.schemas.score_event import StreamScorerConfig
from backend.shared.schemas.transcription import Transcript

logger = logging.getLogger(__name__)


class _Cancelled(Exception):
    """Internal: cancellation requested via session.cancel_event."""


class SourceSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    transcript_path: Optional[Path] = None
    video_path: Optional[Path] = None
    fake: bool = True
    mode: str = "edge_full"

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "SourceSpec":
        if (self.transcript_path is None) == (self.video_path is None):
            raise ValueError(
                "exactly one of transcript_path / video_path must be set"
            )
        return self


def _load_transcript(source: SourceSpec) -> Transcript:
    if source.transcript_path is not None:
        return Transcript.model_validate_json(
            source.transcript_path.read_text(encoding="utf-8")
        )
    # Video path: run compression -> transcription inside the worker thread.
    from backend.shared.schemas.media import CompressionMode
    from backend.workers.app.compression.pipeline import CompressionPipeline
    from app.pipelines.transcription.backends import FakeTranscriptionBackend
    from app.pipelines.transcription.transcriber import Transcriber

    mode = CompressionMode(source.mode if source.mode != "roi" else "roi_encoded")
    out_dir = source.video_path.parent / "live_service_output" / source.video_path.stem
    result = CompressionPipeline().process(source.video_path, out_dir, mode)
    if source.fake:
        backend = FakeTranscriptionBackend()
    else:  # pragma: no cover - requires whisperx install
        from app.pipelines.transcription.backends import WhisperXBackend

        backend = WhisperXBackend()
    return Transcriber(backend).transcribe(result.flac_audio_path)


def _run_sync(
    session: Session,
    source: SourceSpec,
    scorer_config: StreamScorerConfig,
    pace: float,
    loop: asyncio.AbstractEventLoop,
) -> tuple[SessionState, Optional[str]]:
    publisher = session.publisher

    def hook(seconds: float) -> None:
        # Sleep in small slices so DELETE interrupts promptly mid-pace.
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if session.cancel_event.is_set():
                raise _Cancelled()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    try:
        transcript = _load_transcript(source)
        session.language = transcript.language
        session.statement_count = len(transcript.segments)

        replay = ScoreReplayer(transcript, scorer_config).replay(pace, sleep=hook)
        for event in replay:
            if session.cancel_event.is_set():
                raise _Cancelled()
            session.stream_time_seconds = event.stream_time_seconds
            loop.call_soon_threadsafe(publisher.publish, event)
        return SessionState.FINISHED, None
    except _Cancelled:
        return SessionState.CANCELLED, None
    except UnsupportedLanguageError as exc:
        return SessionState.FAILED, str(exc)   # language code only (invariant #3)
    except Exception as exc:
        logger.exception("session_failed id=%s", session.id)
        return SessionState.FAILED, type(exc).__name__


async def start_session(
    session: Session,
    manager: SessionManager,
    source: SourceSpec,
    scorer_config: StreamScorerConfig,
    pace: float,
) -> "asyncio.Task":
    loop = asyncio.get_running_loop()
    manager.mark_running(session)

    async def _run_and_finalize() -> None:
        state, reason = await asyncio.to_thread(
            _run_sync, session, source, scorer_config, pace, loop
        )
        manager.mark_terminal(session, state, reason)
        publisher = session.publisher
        if publisher is not None and not publisher.terminated:
            publisher.publish_terminal(session.state.value, session.reason)

    return asyncio.get_running_loop().create_task(_run_and_finalize())
```

Run: `.venv/Scripts/python -m pytest tests/service/test_runner.py -q`
Expected: all pass. (The cancel test takes <2 s: sliced sleep hook interrupts the 2 s tick gap.)

- [ ] **Step 3: Commit**

```bash
git add backend/ml-inference/app/service/runner.py tests/service/test_runner.py
git commit -m "feat(service): worker-thread session runner with sliced-sleep cancellation"
```

---

## Task 5: FastAPI App Factory — REST

**Files:**
- Create: `backend/ml-inference/app/service/app.py`
- Test: `tests/service/test_rest_api.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces (exact):
  - `def create_app(config: LiveServiceConfig | None = None) -> FastAPI` — app with `app.state.manager` (SessionManager) and `app.state.config`; lifespan starts/cancels the reaper task.
  - `class CreateSessionRequest(BaseModel)`: `source: SourceSpec`, `pace: float = 1.0` (ge=0), `tick_seconds: float = 5.0` (gt=0), `recent_window_seconds: float = 30.0`.
  - Routes per the spec's REST table. `POST /sessions`: validates the source path exists (400 with the path named — dev surface), creates session, attaches `InProcessPublisher(session.id, config.ring_size, config.subscriber_queue_size)`, `await start_session(...)`, returns 201 `{"session_id", "state"}`. `DELETE`: `manager.cancel`; 404 unknown; 200 `{"session_id", "state"}` (idempotent on terminal — publisher terminal frame comes from the runner OR, if the runner already exited, publish it here guarded by `publisher.terminated`).

- [ ] **Step 1: Write the failing tests**

Create `tests/service/test_rest_api.py`:

```python
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.service.app import create_app
from app.service.config import LiveServiceConfig


@pytest.fixture
def client():
    app = create_app(LiveServiceConfig())
    with TestClient(app) as c:
        yield c


def _create(client, transcript_file, **overrides) -> dict:
    body = {
        "source": {"transcript_path": str(transcript_file())},
        "pace": 0.0,
        "tick_seconds": 2.0,
        "recent_window_seconds": 4.0,
    }
    body.update(overrides)
    resp = client.post("/sessions", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_list_get_delete_roundtrip(client, transcript_file):
    created = _create(client, transcript_file)
    sid = created["session_id"]

    listed = client.get("/sessions").json()
    assert any(s["session_id"] == sid for s in listed)

    detail = client.get(f"/sessions/{sid}").json()
    assert detail["session_id"] == sid
    assert set(detail) >= {"state", "language", "statement_count", "reason"}

    resp = client.delete(f"/sessions/{sid}")
    assert resp.status_code == 200
    resp2 = client.delete(f"/sessions/{sid}")   # idempotent on terminal
    assert resp2.status_code == 200


def test_unknown_session_404s(client):
    assert client.get("/sessions/nope").status_code == 404
    assert client.delete("/sessions/nope").status_code == 404


def test_bad_source_400s(client, tmp_path):
    resp = client.post("/sessions", json={
        "source": {"transcript_path": str(tmp_path / "missing.json")},
        "pace": 0.0,
    })
    assert resp.status_code == 400
    assert "missing.json" in resp.json()["detail"]


def test_both_sources_rejected(client, tmp_path):
    resp = client.post("/sessions", json={
        "source": {"transcript_path": str(tmp_path / "a.json"),
                   "video_path": str(tmp_path / "b.mp4")},
    })
    assert resp.status_code == 422


def test_session_reaches_finished(client, transcript_file):
    import time

    sid = _create(client, transcript_file)["session_id"]
    for _ in range(100):                     # pace=0: finishes in well under 5s
        state = client.get(f"/sessions/{sid}").json()["state"]
        if state == "finished":
            break
        time.sleep(0.05)
    assert state == "finished"
```

Run: `.venv/Scripts/python -m pytest tests/service/test_rest_api.py -q`
Expected: FAIL — module missing.

- [ ] **Step 2: Implement `app.py` (REST portion)**

```python
"""FastAPI app factory for the live service (REST + WS).

v1 posture: localhost bind, no auth (the api-gateway session adds JWT --
CLAUDE.md @security). Every frame is dev/ensemble-facing (invariant #5).
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.service.config import LiveServiceConfig
from app.service.publisher import InProcessPublisher
from app.service.runner import SourceSpec, start_session
from app.service.sessions import SessionManager
from backend.shared.schemas.score_event import StreamScorerConfig


class CreateSessionRequest(BaseModel):
    source: SourceSpec
    pace: float = Field(default=1.0, ge=0.0)
    tick_seconds: float = Field(default=5.0, gt=0.0)
    recent_window_seconds: float = Field(default=30.0)


def create_app(config: Optional[LiveServiceConfig] = None) -> FastAPI:
    cfg = config or LiveServiceConfig()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        reaper = asyncio.create_task(app.state.manager.reaper_loop())
        try:
            yield
        finally:
            reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reaper

    app = FastAPI(title="ALICE Live Service", lifespan=lifespan)
    app.state.config = cfg
    app.state.manager = SessionManager(cfg)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "sessions_active": len(app.state.manager.list_sessions()),
        }

    @app.post("/sessions", status_code=201)
    async def create_session(req: CreateSessionRequest) -> dict:
        src = req.source
        path = src.transcript_path if src.transcript_path else src.video_path
        if path is None or not path.exists():
            raise HTTPException(
                status_code=400, detail=f"source not found: {path}"
            )
        manager: SessionManager = app.state.manager
        session = manager.create()
        session.publisher = InProcessPublisher(
            session.id, cfg.ring_size, cfg.subscriber_queue_size
        )
        scorer_config = StreamScorerConfig(
            tick_seconds=req.tick_seconds,
            recent_window_seconds=req.recent_window_seconds,
        )
        await start_session(session, manager, src, scorer_config, req.pace)
        return {"session_id": session.id, "state": session.state.value}

    @app.get("/sessions")
    async def list_sessions() -> list[dict]:
        return [s.to_summary() for s in app.state.manager.list_sessions()]

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict:
        session = app.state.manager.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return session.to_detail()

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        session = app.state.manager.cancel(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        publisher = session.publisher
        if publisher is not None and not publisher.terminated:
            publisher.publish_terminal(session.state.value, session.reason)
        return {"session_id": session.id, "state": session.state.value}

    return app
```

Run: `.venv/Scripts/python -m pytest tests/service/test_rest_api.py -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add backend/ml-inference/app/service/app.py tests/service/test_rest_api.py
git commit -m "feat(service): FastAPI app factory with session REST API + reaper lifespan"
```

---

## Task 6: WebSocket Endpoint

**Files:**
- Modify: `backend/ml-inference/app/service/app.py` (add the WS route inside `create_app`)
- Test: `tests/service/test_websocket.py`

**Interfaces:**
- Consumes: `InProcessPublisher.subscribe/unsubscribe`, `DROPPED` (Task 3); manager/session (Task 2).
- Produces: `WS /sessions/{session_id}/events?last_seq=N` behaving per Global Constraints: 4404 unknown; catch-up then live; terminal frame then server closes normally (code 1000); `DROPPED` sentinel → close 4408; client disconnect → unsubscribe only (session untouched).

- [ ] **Step 1: Write the failing tests**

Create `tests/service/test_websocket.py`:

```python
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.service.app import create_app
from app.service.config import LiveServiceConfig


@pytest.fixture
def client():
    with TestClient(create_app(LiveServiceConfig())) as c:
        yield c


def _create(client, transcript_file, **overrides) -> str:
    body = {
        "source": {"transcript_path": str(transcript_file())},
        "pace": 0.0, "tick_seconds": 2.0, "recent_window_seconds": 4.0,
    }
    body.update(overrides)
    resp = client.post("/sessions", json=body)
    assert resp.status_code == 201
    return resp.json()["session_id"]


def _wait_terminal(client, sid, want="finished"):
    for _ in range(100):
        if client.get(f"/sessions/{sid}").json()["state"] == want:
            return
        time.sleep(0.05)
    raise AssertionError(f"session never reached {want}")


def test_stream_ends_with_exactly_one_terminal_frame(client, transcript_file):
    sid = _create(client, transcript_file)
    _wait_terminal(client, sid)
    frames = []
    with client.websocket_connect(f"/sessions/{sid}/events") as ws:
        while True:
            msg = ws.receive_json()
            frames.append(msg)
            if "state" in msg:
                break
    data = [f for f in frames if "seq" in f]
    terminals = [f for f in frames if "state" in f]
    assert len(terminals) == 1 and terminals[0]["state"] == "finished"
    assert frames[-1] == terminals[0]
    seqs = [f["seq"] for f in data]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert all(f["session_id"] == sid for f in frames)
    assert data and data[-1]["event"]["kind"] == "final"


def test_last_seq_catch_up_skips_seen_events(client, transcript_file):
    sid = _create(client, transcript_file)
    _wait_terminal(client, sid)
    with client.websocket_connect(f"/sessions/{sid}/events") as ws:
        first = ws.receive_json()
    assert first["seq"] == 0
    with client.websocket_connect(f"/sessions/{sid}/events?last_seq=0") as ws:
        nxt = ws.receive_json()
    assert nxt.get("seq", None) != 0        # seq 0 skipped


def test_unknown_session_closes_4404(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/sessions/nope/events") as ws:
            ws.receive_json()
    assert exc_info.value.code == 4404


def test_delete_mid_stream_sends_cancelled_terminal(client, transcript_file):
    sid = _create(client, transcript_file, pace=1.0)   # slow: 2s ticks
    with client.websocket_connect(f"/sessions/{sid}/events") as ws:
        client.delete(f"/sessions/{sid}")
        # Drain until terminal frame arrives.
        for _ in range(50):
            msg = ws.receive_json()
            if "state" in msg:
                break
    assert msg["state"] == "cancelled"


def test_failed_language_session_terminal_frame(client, transcript_file):
    body = {
        "source": {"transcript_path": str(transcript_file(language="es"))},
        "pace": 0.0,
    }
    sid = client.post("/sessions", json=body).json()["session_id"]
    _wait_terminal(client, sid, want="failed")
    with client.websocket_connect(f"/sessions/{sid}/events") as ws:
        msg = ws.receive_json()
    assert msg["state"] == "failed"
    assert "es" in msg["reason"] and "Hola" not in msg["reason"]


def test_two_subscribers_see_identical_streams(client, transcript_file):
    sid = _create(client, transcript_file)
    _wait_terminal(client, sid)

    def drain():
        out = []
        with client.websocket_connect(f"/sessions/{sid}/events") as ws:
            while True:
                msg = ws.receive_json()
                out.append(msg)
                if "state" in msg:
                    return out

    assert drain() == drain()
```

Run: `.venv/Scripts/python -m pytest tests/service/test_websocket.py -q`
Expected: FAIL — no WS route (403/404 on connect).

- [ ] **Step 2: Add the WS route inside `create_app` (before `return app`)**

Add imports at top of `app.py`:

```python
from fastapi import WebSocket, WebSocketDisconnect
from app.service.publisher import DROPPED
```

Route:

```python
    @app.websocket("/sessions/{session_id}/events")
    async def session_events(
        websocket: WebSocket, session_id: str, last_seq: int = -1
    ) -> None:
        manager: SessionManager = app.state.manager
        session = manager.get(session_id)
        if session is None or session.publisher is None:
            await websocket.accept()
            await websocket.close(code=4404)
            return
        publisher = session.publisher
        await websocket.accept()
        queue = publisher.subscribe(last_seq=last_seq)
        try:
            while True:
                frame = await queue.get()
                if frame is DROPPED:
                    await websocket.close(code=4408)
                    return
                await websocket.send_json(frame)
                if "state" in frame:        # terminal frame: normal close
                    await websocket.close()
                    return
        except WebSocketDisconnect:
            pass                             # detached: session untouched
        finally:
            publisher.unsubscribe(queue)
```

Run: `.venv/Scripts/python -m pytest tests/service/test_websocket.py tests/service/ -q`
Expected: all service tests pass.

- [ ] **Step 3: Full suite**

Run: `.venv/Scripts/python -m pytest tests/ -p no:cacheprovider 2>&1 | tail -3`
Expected: green (~180+ passed, 1 deselected slow).

- [ ] **Step 4: Commit**

```bash
git add backend/ml-inference/app/service/app.py tests/service/test_websocket.py
git commit -m "feat(service): WebSocket event stream -- catch-up, terminal frames, 4404/4408, detached disconnect"
```

---

## Task 7: Launcher, Demo Client, Makefile, CLAUDE.md Sync

**Files:**
- Create: `scripts/run_live_service.py`, `scripts/live_client.py`
- Modify: `Makefile`, `CLAUDE.md`

**Interfaces:** consumes `create_app`/`LiveServiceConfig`; no new APIs.

- [ ] **Step 1: Create `scripts/run_live_service.py`**

```python
#!/usr/bin/env python
"""Launch the ALICE live service (dev posture: localhost, no auth).

Usage:
    python scripts/run_live_service.py
    python scripts/run_live_service.py --port 9000 --host 0.0.0.0
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


def main() -> int:
    parser = argparse.ArgumentParser(description="ALICE live service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8710)
    args = parser.parse_args()

    import uvicorn

    from app.service.app import create_app
    from app.service.config import LiveServiceConfig

    config = LiveServiceConfig(host=args.host, port=args.port)
    print(f"ALICE live service on http://{config.host}:{config.port}")
    print("Create a session:  POST /sessions   | Watch: WS /sessions/{id}/events")
    print("NOTE: dev surface -- events are ensemble/developer-facing only.")
    uvicorn.run(create_app(config), host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create `scripts/live_client.py`**

```python
#!/usr/bin/env python
"""Minimal demo client: create a session and watch its event stream.

Usage:
    python scripts/live_client.py --transcript path/to/t.json --pace 1
    python scripts/live_client.py --video demo_data/honest/trial_truth_001.mp4 --fake
    python scripts/live_client.py --watch SESSION_ID          # attach only
"""
from __future__ import annotations

import argparse
import json
import sys
from urllib.request import Request, urlopen

from websockets.sync.client import connect  # ships with uvicorn[standard]? NO --
# websockets is a direct dependency of uvicorn[standard]; import verified in tests.


def main() -> int:
    parser = argparse.ArgumentParser(description="ALICE live demo client")
    parser.add_argument("--base", default="http://127.0.0.1:8710")
    parser.add_argument("--transcript")
    parser.add_argument("--video")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--watch", help="Attach to an existing session id")
    args = parser.parse_args()

    if args.watch:
        session_id = args.watch
    else:
        source = (
            {"transcript_path": args.transcript}
            if args.transcript
            else {"video_path": args.video, "fake": args.fake}
        )
        req = Request(
            f"{args.base}/sessions",
            data=json.dumps({"source": source, "pace": args.pace}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req) as resp:
            session_id = json.loads(resp.read())["session_id"]
        print(f"session: {session_id}")

    ws_url = args.base.replace("http", "ws") + f"/sessions/{session_id}/events"
    with connect(ws_url) as ws:
        for raw in ws:
            frame = json.loads(raw)
            if "state" in frame:
                print(f"[terminal] {frame['state']}"
                      + (f" ({frame['reason']})" if frame.get("reason") else ""))
                break
            ev = frame["event"]
            recent = ev.get("recent")
            recent_s = f"{recent['composite_score']:5.1f}" if recent else "  -- "
            print(
                f"[t={ev['stream_time_seconds']:6.1f}s] seq={frame['seq']:<3} "
                f"{ev['kind']:<8} cumulative={ev['cumulative']['composite_score']:5.1f} "
                f"recent={recent_s}"
            )
    print("NOTE: anomaly signals, not ground truth. ~75% F1 ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

(Implementer: verify `import websockets` resolves in the venv — it is a `uvicorn[standard]` dependency. If not, add `websockets>=12` to the dev extra and note it in your report. Remove the stray inline comment noise on the import line — keep a clean single import with a one-line comment.)

- [ ] **Step 3: Makefile target**

Append to `Makefile` (and add `live` to `.PHONY`):

```makefile
# Run the live service (dev posture: localhost, no auth).
live:
	$(PYTHON) scripts/run_live_service.py
```

- [ ] **Step 4: Manual two-terminal smoke (run and capture output)**

Terminal A (background it): `.venv/Scripts/python scripts/run_live_service.py --port 8711`
Terminal B: `.venv/Scripts/python scripts/live_client.py --base http://127.0.0.1:8711 --video demo_data/honest/trial_truth_001.mp4 --fake --pace 2`
Expected: session id printed, paced interim frames, a `final` data frame, `[terminal] finished`, exit 0. Kill the server afterwards.

- [ ] **Step 5: CLAUDE.md sync**

In the IMPLEMENTATION STATUS section, add after the Session 6 material (adapt to current wording — read the file):

```markdown
### Session 7 — Live Service Async Shell (complete)

The real-time surface pre-designed in Session 5 (decision #4). Design spec:
`docs/superpowers/specs/2026-07-19-live-service-async-shell-design.md`.

| Component | Path | Status |
|---|---|---|
| Config | `backend/ml-inference/app/service/config.py` | shipped — frozen `LiveServiceConfig` |
| Sessions + reaper | `app/service/sessions.py` | shipped — detached lifecycle (CREATED/RUNNING/FINISHED/CANCELLED/FAILED), TTL reaper |
| Publisher (Kafka seam) | `app/service/publisher.py` | shipped — seq, ring buffer, fan-out, slow-client drop (4408); v2 swaps in a bus publisher |
| Runner | `app/service/runner.py` | shipped — one worker thread/session, sliced-sleep cancellation |
| REST + WS | `app/service/app.py` | shipped — POST/GET/DELETE /sessions, /healthz, WS /sessions/{id}/events?last_seq= (4404/4408) |
| Launch + demo | `scripts/run_live_service.py`, `scripts/live_client.py`, `make live` | shipped |

Wire contract: data frames `{session_id, seq, event}`; exactly one terminal
frame `{session_id, state, reason}`. Auth: none in v1 (localhost bind; JWT is
the api-gateway session's job). In update to "Known gaps" item 2: the async
shell is now SHIPPED; remaining for full live: incremental transcription +
platform media ingest.
```

Also update Known-gaps item 2's "Remaining for a true live surface" sentence to reflect the shell shipping (adapt in place).

- [ ] **Step 6: Full suite + commit**

Run: `.venv/Scripts/python -m pytest tests/ -p no:cacheprovider 2>&1 | tail -3`
Expected: green, 1 deselected.

```bash
git add scripts/run_live_service.py scripts/live_client.py Makefile CLAUDE.md
git commit -m "feat(service): launcher + demo client + make live; CLAUDE.md live-shell status sync"
```

---

## Final Verification

- [ ] Full suite green: `.venv/Scripts/python -m pytest tests/ -p no:cacheprovider` (~180+ passed, 1 deselected).
- [ ] Two-terminal demo runs end-to-end (Task 7 Step 4 captured output).
- [ ] `git status --short` clean.

---

## Self-Review Checklist

- [x] Spec API table ↔ Task 5/6 routes match exactly (incl. 4404/4408, last_seq, idempotent DELETE, healthz).
- [x] Detached semantics: WS disconnect path only unsubscribes (Task 6); DELETE is the sole cancel (Tasks 2/5); reaper covers terminal + stuck-CREATED (Task 2).
- [x] Envelope + terminal frame shapes identical in publisher (T3), runner (T4), WS tests (T6), demo client (T7).
- [x] Failure semantics (language/code-only, opaque class name, zero-statement FINISHED) in runner (T4) with tests; invariant #3 asserted (no "Hola" in reasons).
- [x] Kafka seams: publisher interface boundary (T3), envelope with seq (T3), explicit session state (T2) — all present.
- [x] Cancellation: sliced-sleep hook (T4) interrupts mid-pace; exactly-one-terminal-frame guard (`publisher.terminated`) in both runner and DELETE paths.
- [x] Type consistency: `SessionState/TERMINAL_STATES`, `to_summary/to_detail`, `subscribe(last_seq)/unsubscribe`, `DROPPED`, `SourceSpec`, `start_session(session, manager, source, scorer_config, pace)` used identically across tasks.
- [x] Known wrinkle documented in T3 tests (the `fine`-queue correction) — implementer instructed to use the corrected form verbatim.
- [x] No placeholders; every step has complete code + exact commands.
