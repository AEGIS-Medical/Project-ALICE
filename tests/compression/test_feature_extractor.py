"""Tests for streaming protobuf (.pb) landmark emitter.

The landmark path is exercised with MediaPipe and OpenCV faked out (see
conftest) so these run fast and offline. The behaviours under test are the
streaming contract: .pb output, bounded memory, periodic chunk cadence, and a
readable partial file when extraction is interrupted mid-stream.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import pytest

from backend.shared.telemetry.landmark_codec import LandmarkDecoder
from backend.workers.app.compression import feature_extractor as fe_mod
from backend.workers.app.compression.feature_extractor import FeatureExtractor
from tests.compression.conftest import FakeCapture, FakeLandmarker


def _patch_mediapipe(monkeypatch, *, n_frames, with_face=True, raise_after=None, fps=30.0):
    """Wire the fakes into the feature_extractor module namespace."""
    monkeypatch.setattr(
        fe_mod, "face_landmarker_model", lambda: Path("dummy.task")
    )
    monkeypatch.setattr(
        fe_mod.cv2, "VideoCapture",
        lambda _path: FakeCapture(n_frames, fps=fps),
    )
    landmarker = FakeLandmarker(with_face=with_face, raise_after=raise_after)
    monkeypatch.setattr(
        fe_mod.vision.FaceLandmarker,
        "create_from_options",
        staticmethod(lambda _options: landmarker),
    )
    return landmarker


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """A real on-disk file with a video extension (content never decoded)."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00")
    return p


def test_output_is_pb_and_roundtrips(monkeypatch, dummy_video, tmp_path):
    _patch_mediapipe(monkeypatch, n_frames=5)
    out_dir = tmp_path / "landmarks"

    out = FeatureExtractor().extract_landmarks(dummy_video, out_dir)

    assert out.suffix == ".pb"
    decoded = list(LandmarkDecoder(out).frames())
    assert len(decoded) == 5
    for i, d in enumerate(decoded):
        assert d.frame_number == i
        assert isinstance(d.timestamp_seconds, float)
        assert d.landmarks is not None and len(d.landmarks) == 478


def test_keyframe_interval_controls_chunk_cadence(monkeypatch, dummy_video, tmp_path):
    """60 frames at interval 30 -> exactly 2 chunks on the wire."""
    _patch_mediapipe(monkeypatch, n_frames=60)
    out_dir = tmp_path / "landmarks"

    out = FeatureExtractor().extract_landmarks(
        dummy_video, out_dir, flush_interval=30
    )

    dec = LandmarkDecoder(out)
    assert len(list(dec.frames())) == 60
    assert dec.chunks_read == 2


def test_streaming_write_partial_file_on_interrupt(monkeypatch, dummy_video, tmp_path):
    # Interrupt after 45 frames with interval 10: the context-managed encoder
    # flushes the tail chunk on exception, so all 45 frames survive.
    _patch_mediapipe(monkeypatch, n_frames=200, raise_after=45)
    out_dir = tmp_path / "landmarks"

    with pytest.raises(RuntimeError):
        FeatureExtractor(flush_interval=10).extract_landmarks(dummy_video, out_dir)

    partial = out_dir / "clip_landmarks.pb"
    assert partial.exists()
    decoded = list(LandmarkDecoder(partial).frames())
    assert len(decoded) == 45


def test_peak_memory_bounded(monkeypatch, dummy_video, tmp_path):
    # Synthetic 10-min clip at 30fps = 18000 frames. Chunked encoding keeps
    # peak RAM O(keyframe_interval), so the delta must stay well under 200MB.
    _patch_mediapipe(monkeypatch, n_frames=18000)
    out_dir = tmp_path / "landmarks"

    tracemalloc.start()
    FeatureExtractor().extract_landmarks(dummy_video, out_dir)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 200 * 1024 * 1024
