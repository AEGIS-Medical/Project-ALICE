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
