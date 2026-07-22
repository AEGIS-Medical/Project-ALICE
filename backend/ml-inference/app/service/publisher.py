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
        backlog = [
            frame
            for frame in self._ring
            if ("seq" in frame and frame["seq"] > last_seq) or "state" in frame
        ]
        # If the backlog exceeds queue capacity, deliver only the newest
        # frames -- the tail always contains the terminal frame when one
        # exists (it is last in the ring). The client detects the skip from
        # the first seq it receives, exactly like the predates-window rule.
        for frame in backlog[-self._queue_size:]:
            q.put_nowait(frame)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)
