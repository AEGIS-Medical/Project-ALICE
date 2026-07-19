# ALICE — Live Service Async Shell (Session 7)
## Design Spec · 2026-07-19

---

## Overview

Builds the real-time surface the Session 5 spec deliberately deferred and
pre-designed: an **async FastAPI/WebSocket shell** around the sync
`stream_scores` generator, bridged via a worker thread — async where it earns
its keep (socket I/O), sync where async cannot help (CPU-bound scoring).
Clients create scoring sessions over REST and watch ScoreEvents stream over a
WebSocket; the replayer's paced output becomes a genuinely connectable live
feed.

```
client ── POST /sessions ──────────► SessionManager ── worker thread ──► ScoreReplayer
client ◄─ WS /sessions/{id}/events ◄─ EventPublisher ◄─ call_soon_threadsafe ──┘
                                        │ (ring buffer, seq, fan-out)
```

**Decisions locked during brainstorming:**

1. **Scope: the shell only.** No Kafka, no gateway tier, no auth, no
   connectors, no incremental transcription. Sources are the replayer's
   existing ones (transcript JSON / video via the compression+transcription
   path, `--fake` supported).
2. **REST-then-WebSocket shape**, refined to be **gateway-ready** (the
   Glean-style three-tier architecture — WS gateway + bus + workers — is
   CLAUDE.md's own target via Kafka/Celery; v1 is one process playing both
   roles with three seams that make the v2 split an insertion, not a rewrite):
   - **Publisher seam:** the session loop hands events to an `EventPublisher`
     interface; v1 is in-process (ring buffer + subscriber queues); v2 swaps in
     a Kafka publisher on topic `score-events.{session_id}` without touching
     scoring code.
   - **Envelope, not schema change:** `ScoreEvent` stays exactly as shipped
     (already versioned + serializable — it IS the future bus payload). The
     wire envelope adds `{session_id, seq, event}`; `seq` doubles as the
     reconnect/replay offset.
   - **Sessions as explicit state** with UUIDs; cancellation is a state
     transition via `DELETE`; connectors later create sessions through the
     same `POST /sessions` shape, not a side door.
3. **Detached sessions.** A session runs server-side regardless of watchers;
   the WebSocket is a *view*, not a lifeline. Cost accepted (~1 extra task:
   ring buffer + reaper + catch-up) to avoid the known-breaking migration when
   live calls arrive (a Zoom call must not die because a phone hit a tunnel).
   Disconnect never cancels; only `DELETE` (or terminal completion) does.
4. **Latency reality (from Session 5):** per-tick scoring ~10–50 ms against a
   5 s tick; Python is not the bottleneck; the pull-based generator plus
   bounded per-connection queues make producer-outruns-consumer structurally
   impossible at the compute layer and survivable at the socket layer.

---

## File Structure

```
backend/ml-inference/app/service/
  __init__.py                 NEW — package
  config.py                   NEW — LiveServiceConfig (frozen Pydantic)
  sessions.py                 NEW — SessionState enum, Session, SessionManager, reaper
  publisher.py                NEW — EventPublisher protocol + InProcessPublisher
                                    (seq, ring buffer, subscriber queues, envelopes)
  runner.py                   NEW — worker-thread session runner (replayer loop →
                                    call_soon_threadsafe publish; cancel via
                                    injectable sleep hook)
  app.py                      NEW — FastAPI app factory: REST routes + WS endpoint
scripts/
  run_live_service.py         NEW — uvicorn launcher (127.0.0.1:8710 default)
  live_client.py              NEW — ~40-line demo client: connect, print events
tests/service/
  __init__.py, conftest.py    NEW — sys.path bridge (mirrors tests/streaming)
  test_config.py              NEW
  test_sessions.py            NEW — lifecycle, reaper
  test_publisher.py           NEW — seq, ring buffer, fan-out, slow-client drop
  test_rest_api.py            NEW — CRUD + healthz
  test_websocket.py           NEW — stream, catch-up, terminal frames, cancel
pyproject.toml                MODIFIED — add fastapi + uvicorn[standard]
Makefile                      MODIFIED — `make live` target
CLAUDE.md                     MODIFIED — status sync (live surface shipped at shell layer)
```

Follows every established pattern: hyphenated `ml-inference` root with sys.path
imports, frozen Pydantic configs, lazy heavy imports, conftest bridges.

---

## API Contract

### REST

| Route | Behavior |
|---|---|
| `POST /sessions` | Body: `{source: {transcript_path?} \| {video_path, fake?, mode?}, pace: 1.0, tick_seconds: 5.0, recent_window_seconds: 30.0}`. Validates the source exists; creates the session (state `CREATED`), starts its runner task (state `RUNNING`), returns `201 {session_id, state}`. `pace=0` = instant (test/batch mode). |
| `GET /sessions` | `[{session_id, state, created_at, stream_time_seconds, last_seq, subscriber_count}]` |
| `GET /sessions/{id}` | Detail: the above + `language`, `statement_count`, `reason` (if FAILED). 404 unknown. |
| `DELETE /sessions/{id}` | Cancels a non-terminal session (state → `CANCELLED`, runner unwound via the sleep hook, terminal frame fanned out). Idempotent on terminal sessions. 404 unknown. |
| `GET /healthz` | `{status: "ok", sessions_active: N}` |

### WebSocket — `WS /sessions/{id}/events?last_seq=N`

1. Unknown/reaped session → close code **4404**.
2. On connect: replay ring-buffer events with `seq > last_seq` (default −1 = all
   buffered), then live events as they publish. If `last_seq` predates the
   buffer window (event already trimmed), catch-up silently starts at the
   oldest buffered event — the client can detect the gap from the first seq it
   receives; full-history replay is a v2 bus feature.
3. Every data frame is the envelope:
   `{"session_id": "...", "seq": 17, "event": {…ScoreEvent JSON…}}` — `seq`
   strictly increasing per session, no gaps within the buffer window.
4. Exactly one terminal frame ends every completed stream:
   `{"session_id": "...", "state": "finished" | "cancelled" | "failed", "reason": "..."?}`
   then a normal close. Subscribers connecting *after* terminal state get the
   buffered events + terminal frame (until the reaper collects the session).
5. **Slow-client policy:** per-connection bounded queue (default 64). Overflow
   drops that connection (close code **4408**); the session is untouched;
   clients reconnect with `last_seq` and catch up. Compute is never stalled by
   a slow phone.

---

## Session Lifecycle

```
CREATED ──start──► RUNNING ──stream end──► FINISHED
                     │  ▲                      (terminal)
        DELETE ──────┘  └── UnsupportedLanguageError / scorer exception
                     ▼                         ▼
                 CANCELLED (terminal)      FAILED(reason) (terminal)
```

- States live on the `Session` object (UUID id, timestamps, state, reason,
  ring buffer, subscriber registry) inside `SessionManager` — the future
  session-service extraction point.
- **Reaper:** one asyncio loop; removes sessions `ttl_seconds` (default 900)
  after reaching a terminal state, and force-cancels sessions stuck in
  `CREATED` beyond the TTL. Config-tunable to sub-second for tests.
- **Cancellation mechanics:** the runner passes the replayer a sleep function
  that checks the session's `threading.Event` cancel flag (Session 5's
  injectable-sleep design paying off); setting it raises inside the worker
  thread, closing the generator cleanly (`GeneratorExit`) even mid-pace.
  `pace=0` sessions check the flag between events.

---

## Compute Bridging

- One worker thread per session: `asyncio.to_thread(run_session, session)`.
  The thread iterates `ScoreReplayer(transcript, config).replay(pace, sleep=hook)`
  and publishes each event with `loop.call_soon_threadsafe(publisher.publish, ...)`
  — one thread per session for its lifetime, not a thread hop per event.
- Video sources run the existing CompressionPipeline → Transcriber path inside
  the same worker thread before streaming begins (session shows `RUNNING`
  throughout; `stream_time_seconds` distinguishes progress).
- The publisher assigns `seq`, appends the serialized envelope to the ring
  buffer (`deque(maxlen=ring_size)`, default 256), and fans out to subscriber
  queues (`put_nowait`; full → drop that subscriber per slow-client policy).

## Config (`LiveServiceConfig`, frozen)

`host="127.0.0.1"`, `port=8710`, `ring_size=256` (≥1), `subscriber_queue_size=64`
(≥1), `session_ttl_seconds=900.0` (>0), `reaper_interval_seconds=5.0` (>0).
Env-overridable via the launcher flags only (no env-var magic in v1).

---

## Error Handling

| Condition | Behavior |
|---|---|
| Source path missing / unreadable | `POST` → 400 with reason (path named — dev surface; paths are not PII) |
| Non-English transcript | Session → `FAILED`, reason names the language code only (invariant #3); terminal frame to subscribers |
| Scorer exception mid-stream | Session → `FAILED` with opaque reason; full traceback to server log only |
| Zero-statement transcript | Immediate `FINISHED`, zero data frames, terminal frame only ("silence is not an error") |
| Unknown session (REST / WS) | 404 / close 4404 |
| DELETE on terminal session | 200, no-op (idempotent) |
| Slow client | Connection dropped (4408); session unaffected; reconnect + `last_seq` catches up |
| Server restart | Sessions are in-memory and lost — documented v1 limitation (resume-across-restart arrives with the bus in v2) |

**Invariants:** #3 — logs/reasons carry counts, codes, states; never transcript
text. #5 — a module-docstring note marks every frame ensemble/dev-facing; user
surfaces must add calibration + labels. #6 — no forbidden phrase. **Auth: none
in v1** — binds localhost by default; JWT is the api-gateway's job when it
arrives (CLAUDE.md @security); this is an explicit, documented posture, not an
oversight.

---

## Testing

All tests use FastAPI's in-process test client (REST + WS), `pace=0`, and
fake/synthetic transcripts — no sleeping (sub-second reaper TTLs in reaper
tests), no models beyond spaCy, no network.

| Suite | Coverage |
|---|---|
| `test_config.py` | defaults, bounds, frozen |
| `test_sessions.py` | state transitions incl. idempotent cancel; reaper collects terminal + stuck-CREATED sessions; UUIDs unique |
| `test_publisher.py` | seq strictly increasing from 0; ring buffer trims to `ring_size`; catch-up slice correct; two subscribers receive identical streams; full subscriber queue → dropped, session unaffected |
| `test_rest_api.py` | full CRUD; 404s; 400 on bad source; healthz counts; pace/tick params reach the scorer config |
| `test_websocket.py` | stream to terminal frame (exactly one); envelope contract; `last_seq` catch-up after simulated drop; late subscriber after FINISHED gets buffer + terminal; DELETE mid-stream → prompt CANCELLED terminal frame; failed-language session → FAILED frame with code-only reason; unknown id → 4404 |
| Manual | `scripts/run_live_service.py` + `scripts/live_client.py` on a demo video (`--fake`): the two-terminal live demo |

Convergence is untouched (the shell consumes `stream_scores` as-is); the full
suite must stay green.

---

## Out of Scope (recorded for v2+)

- Kafka/bus + gateway tier split (seams: publisher interface, envelope+seq,
  explicit session state)
- Auth/TLS (api-gateway session), resume across server restart, horizontal scale
- Incremental transcription; connector-created sessions (same POST shape later)
- Mobile/web UI beyond the demo CLI client
