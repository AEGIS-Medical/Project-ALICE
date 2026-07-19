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
