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
