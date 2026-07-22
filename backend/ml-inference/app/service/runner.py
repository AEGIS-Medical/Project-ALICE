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
