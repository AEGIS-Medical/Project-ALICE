"""Tests for P1-S8: mid-session bandwidth tier switching.

``update_bandwidth`` re-evaluates the active compression mode when mobile
uplink changes mid-call, firing a caller callback on a real transition and
recording an audit trail that surfaces in ``CompressionResult``. The heavy
pipeline stages are faked so ``process`` runs without ffmpeg/MediaPipe.
"""

from __future__ import annotations

from pathlib import Path

from backend.shared.schemas.media import CompressionMode
from backend.workers.app.compression.pipeline import CompressionPipeline


def _pipeline(**kwargs) -> CompressionPipeline:
    return CompressionPipeline(**kwargs)


def test_bandwidth_downgrade_fires_callback():
    events: list[tuple[CompressionMode, CompressionMode]] = []
    p = _pipeline(on_mode_change=lambda old, new: events.append((old, new)))
    p.current_mode = CompressionMode.ROI_ENCODED

    new_mode = p.update_bandwidth(0.5)

    assert new_mode == CompressionMode.EDGE_MINIMAL
    assert events == [(CompressionMode.ROI_ENCODED, CompressionMode.EDGE_MINIMAL)]


def test_bandwidth_noop_when_same_mode():
    events: list = []
    p = _pipeline(on_mode_change=lambda old, new: events.append((old, new)))
    p.current_mode = CompressionMode.EDGE_MINIMAL

    new_mode = p.update_bandwidth(0.5)

    assert new_mode == CompressionMode.EDGE_MINIMAL
    assert events == []  # no transition -> callback not fired


def test_bandwidth_upgrade():
    events: list = []
    p = _pipeline(on_mode_change=lambda old, new: events.append((old, new)))
    p.current_mode = CompressionMode.EDGE_MINIMAL

    new_mode = p.update_bandwidth(12.0)

    assert new_mode == CompressionMode.RAW
    assert events == [(CompressionMode.EDGE_MINIMAL, CompressionMode.RAW)]


# ---- result-integration test (fakes the heavy stages) --------------------


class _FakeAudio:
    def extract(self, _input_path: Path, audio_dir: Path):
        flac = audio_dir / "a.flac"
        flac.write_bytes(b"x")
        opus = audio_dir / "a.opus"
        opus.write_bytes(b"x")
        return flac, opus


class _FakeFeat:
    last_frames_processed = 1
    last_frames_with_face = 1

    def extract_landmarks(self, _input_path, landmarks_dir, flush_interval=None):
        p = landmarks_dir / "a_landmarks.jsonl"
        p.write_text("{}\n", encoding="utf-8")
        return p

    def extract_audio_features(self, _flac, features_dir, **_kw):
        p = features_dir / "a.npz"
        p.write_bytes(b"x")
        return p


def test_mode_transitions_logged_in_result(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    p = _pipeline(audio_extractor=_FakeAudio(), feature_extractor=_FakeFeat())
    p.current_mode = CompressionMode.ROI_ENCODED
    p.update_bandwidth(0.5)  # ROI_ENCODED -> EDGE_MINIMAL

    result = p.process(video, tmp_path / "out", mode=CompressionMode.EDGE_MINIMAL)

    assert len(result.mode_transitions) == 1
    timestamp, mode = result.mode_transitions[0]
    assert isinstance(timestamp, float)
    assert mode == CompressionMode.EDGE_MINIMAL
