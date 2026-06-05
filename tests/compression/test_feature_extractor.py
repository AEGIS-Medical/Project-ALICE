"""Tests for P1-S6: streaming JSONL landmark emitter.

The landmark path is exercised with MediaPipe and OpenCV faked out (see
conftest) so these run fast and offline. The behaviours under test are the
streaming contract: JSONL output, bounded memory, periodic flush, and a
readable partial file when extraction is interrupted mid-stream.
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import pytest

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


def test_output_is_valid_jsonl(monkeypatch, dummy_video, tmp_path):
    _patch_mediapipe(monkeypatch, n_frames=5)
    out_dir = tmp_path / "landmarks"

    out = FeatureExtractor().extract_landmarks(dummy_video, out_dir)

    assert out.suffix == ".jsonl"
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        rec = json.loads(line)  # each line independently parseable
        assert isinstance(rec["frame_number"], int)
        assert isinstance(rec["timestamp_seconds"], float)
        assert len(rec["landmarks"]) == 478


def test_flush_interval_controls_disk_writes(monkeypatch, dummy_video, tmp_path):
    _patch_mediapipe(monkeypatch, n_frames=60)
    out_dir = tmp_path / "landmarks"

    # Spy on every file handle's flush() while still writing for real, so the
    # post-run stat() in the implementation continues to work.
    real_open = open
    spies: list = []

    class _FlushSpy:
        def __init__(self, fh):
            self._fh = fh
            self.flush_count = 0

        def write(self, s):
            return self._fh.write(s)

        def flush(self):
            self.flush_count += 1
            return self._fh.flush()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._fh.__exit__(*a)

    def spy_open(*args, **kwargs):
        spy = _FlushSpy(real_open(*args, **kwargs))
        spies.append(spy)
        return spy

    monkeypatch.setattr("builtins.open", spy_open)

    FeatureExtractor().extract_landmarks(dummy_video, out_dir, flush_interval=30)

    assert len(spies) == 1
    assert spies[0].flush_count == 60 // 30


def test_streaming_write_partial_file_on_interrupt(monkeypatch, dummy_video, tmp_path):
    # Interrupt after 45 frames; not a flush-interval multiple, so this also
    # exercises the close-time flush of the tail buffer.
    _patch_mediapipe(monkeypatch, n_frames=200, raise_after=45)
    out_dir = tmp_path / "landmarks"

    with pytest.raises(RuntimeError):
        FeatureExtractor(flush_interval=10).extract_landmarks(dummy_video, out_dir)

    partial = out_dir / "clip_landmarks.jsonl"
    assert partial.exists()
    lines = partial.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 45
    for line in lines:
        json.loads(line)  # partial file is fully readable


def test_peak_memory_bounded(monkeypatch, dummy_video, tmp_path):
    # Synthetic 10-min clip at 30fps = 18000 frames. Streaming keeps peak RAM
    # O(flush_interval), so the delta must stay well under 200MB.
    _patch_mediapipe(monkeypatch, n_frames=18000)
    out_dir = tmp_path / "landmarks"

    tracemalloc.start()
    FeatureExtractor().extract_landmarks(dummy_video, out_dir)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 200 * 1024 * 1024
