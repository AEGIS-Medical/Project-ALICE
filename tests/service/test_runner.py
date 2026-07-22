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
        mgr, session, source = _setup(tmp_path / "missing.json")
        task = await start_session(session, mgr, source, _scfg(), pace=0.0)
        await task
        return session

    session = asyncio.run(go())
    assert session.state is SessionState.FAILED
