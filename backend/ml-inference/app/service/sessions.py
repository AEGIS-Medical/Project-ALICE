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
        self.runner_task: Optional[object] = None  # asyncio.Task (held for GC lifetime)

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
    (mutations arrive via call_soon_threadsafe from workers) (exception: the
    runner updates the progress scalars language/statement_count/
    stream_time_seconds from its worker thread -- GIL-safe atomic writes;
    session STATE is only ever mutated on the loop thread)."""

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
        if state not in TERMINAL_STATES:
            raise ValueError(f"{state} is not terminal")
        if session.state in TERMINAL_STATES:
            return
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
