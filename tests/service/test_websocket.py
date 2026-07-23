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
