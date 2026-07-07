"""Encode -> decode round-trip tests for the landmark codec."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from backend.shared.telemetry.landmark_codec import (
    XY_SCALE,
    Z_SCALE,
    DecodedFrame,
    LandmarkDecoder,
    LandmarkEncoder,
)

MAX_XY_ERR = 1.0 / (2 * XY_SCALE)
MAX_Z_ERR = 1.0 / (2 * Z_SCALE)


def _random_frame(rng: random.Random) -> list[list[float]]:
    return [[rng.random(), rng.random(), rng.uniform(-1, 1)] for _ in range(478)]


def _drift(frame: list[list[float]], rng: random.Random) -> list[list[float]]:
    """Small per-landmark motion, clamped into legal ranges."""
    out = []
    for x, y, z in frame:
        out.append([
            min(1.0, max(0.0, x + rng.uniform(-0.002, 0.002))),
            min(1.0, max(0.0, y + rng.uniform(-0.002, 0.002))),
            min(1.0, max(-1.0, z + rng.uniform(-0.002, 0.002))),
        ])
    return out


def _roundtrip(tmp_path: Path, frames: list, **enc_kwargs) -> list[DecodedFrame]:
    path = tmp_path / "t.pb"
    with LandmarkEncoder(path, source_fps=30.0, **enc_kwargs) as enc:
        for i, lm in enumerate(frames):
            enc.add_frame(i, lm)
    return list(LandmarkDecoder(path).frames())


def test_smooth_motion_roundtrips_within_bound(tmp_path):
    rng = random.Random(7)
    frames = [_random_frame(rng)]
    for _ in range(89):
        frames.append(_drift(frames[-1], rng))

    decoded = _roundtrip(tmp_path, frames)

    assert len(decoded) == 90
    for i, d in enumerate(decoded):
        assert d.frame_number == i
        assert abs(d.timestamp_seconds - i / 30.0) < 1e-9
        assert d.landmarks is not None and len(d.landmarks) == 478
        for (x0, y0, z0), (x1, y1, z1) in zip(frames[i], d.landmarks):
            assert abs(x0 - x1) <= MAX_XY_ERR
            assert abs(y0 - y1) <= MAX_XY_ERR
            assert abs(z0 - z1) <= MAX_Z_ERR


def test_no_face_gap_roundtrips_and_forces_keyframe(tmp_path):
    rng = random.Random(11)
    f = _random_frame(rng)
    # face, face, gap, gap, face (must be keyframe -- decode still exact)
    frames = [f, _drift(f, rng), None, None, _drift(f, rng)]

    decoded = _roundtrip(tmp_path, frames)

    assert [d.landmarks is None for d in decoded] == [False, False, True, True, False]
    assert decoded[2].frame_number == 2
    last = decoded[4].landmarks
    for (x0, y0, _z0), (x1, y1, _z1) in zip(frames[4], last):
        assert abs(x0 - x1) <= MAX_XY_ERR
        assert abs(y0 - y1) <= MAX_XY_ERR


def test_multi_chunk_stream(tmp_path):
    """65 frames @ interval 30 -> chunks of 30/30/5; all decode."""
    rng = random.Random(13)
    frames = [_random_frame(rng)]
    for _ in range(64):
        frames.append(_drift(frames[-1], rng))

    path = tmp_path / "t.pb"
    with LandmarkEncoder(path, source_fps=30.0, keyframe_interval=30) as enc:
        for i, lm in enumerate(frames):
            enc.add_frame(i, lm)
    telemetry_chunks = enc.chunks_written
    telemetry_frames = enc.frames_written

    dec = LandmarkDecoder(path)
    decoded = list(dec.frames())
    assert len(decoded) == 65
    assert telemetry_frames == 65
    assert telemetry_chunks == 3
    assert dec.chunks_read == 3


def test_empty_stream_roundtrips(tmp_path):
    path = tmp_path / "t.pb"
    with LandmarkEncoder(path, source_fps=30.0) as enc:
        pass
    dec = LandmarkDecoder(path)
    assert dec.header.landmark_count == 478
    assert list(dec.frames()) == []


def test_header_fields_faithful(tmp_path):
    path = tmp_path / "t.pb"
    with LandmarkEncoder(
        path, source_fps=24.0, frame_skip=2, keyframe_interval=10
    ) as enc:
        enc.add_frame(0, [[0.5, 0.5, 0.0]] * 478)
    h = LandmarkDecoder(path).header
    assert h.version == 1
    assert abs(h.source_fps - 24.0) < 1e-6
    assert h.frame_skip == 2
    assert h.keyframe_interval == 10


def test_wrong_landmark_count_raises(tmp_path):
    path = tmp_path / "t.pb"
    with LandmarkEncoder(path, source_fps=30.0) as enc:
        with pytest.raises(ValueError, match="landmark"):
            enc.add_frame(0, [[0.5, 0.5, 0.0]] * 10)


def test_close_is_idempotent(tmp_path):
    path = tmp_path / "t.pb"
    enc = LandmarkEncoder(path, source_fps=30.0)
    enc.add_frame(0, [[0.5, 0.5, 0.0]] * 478)
    enc.close()
    enc.close()  # no error
    assert len(list(LandmarkDecoder(path).frames())) == 1
