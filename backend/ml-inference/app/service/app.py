"""FastAPI app factory for the live service (REST + WS).

v1 posture: localhost bind, no auth (the api-gateway session adds JWT --
CLAUDE.md @security). Every frame is dev/ensemble-facing (invariant #5).
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.service.config import LiveServiceConfig
from app.service.publisher import DROPPED, InProcessPublisher
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

    return app
