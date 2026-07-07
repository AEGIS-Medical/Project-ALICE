"""Crash-recovery and corruption behavior for the landmark codec."""
from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest

from backend.shared.telemetry.landmark_codec import (
    MAGIC,
    LandmarkDecodeError,
    LandmarkDecoder,
    LandmarkEncoder,
)


def _write_stream(path: Path, n_frames: int = 65, interval: int = 30) -> None:
    rng = random.Random(3)
    frame = [[rng.random(), rng.random(), 0.0] for _ in range(478)]
    with LandmarkEncoder(
        path, source_fps=30.0, keyframe_interval=interval
    ) as enc:
        for i in range(n_frames):
            frame = [
                [min(1.0, max(0.0, x + rng.uniform(-0.001, 0.001))),
                 min(1.0, max(0.0, y + rng.uniform(-0.001, 0.001))), z]
                for x, y, z in frame
            ]
            enc.add_frame(i, frame)


def test_truncation_yields_complete_chunks_only(tmp_path):
    """Truncating anywhere inside the final chunk loses only that chunk."""
    path = tmp_path / "t.pb"
    _write_stream(path, n_frames=65, interval=30)  # chunks: 30, 30, 5
    data = path.read_bytes()

    # Truncate 3 bytes from the end (inside the last chunk's payload).
    cut = tmp_path / "cut.pb"
    cut.write_bytes(data[:-3])
    decoded = list(LandmarkDecoder(cut).frames())
    assert len(decoded) == 60  # first two complete chunks survive

    # Truncate mid-way through the second chunk header region too.
    cut2 = tmp_path / "cut2.pb"
    cut2.write_bytes(data[: len(data) // 2])
    decoded2 = list(LandmarkDecoder(cut2).frames())
    assert len(decoded2) in (0, 30)  # only whole chunks, never partial frames
    assert all(d.landmarks is not None for d in decoded2)


def test_corrupt_zlib_payload_raises_with_chunk_index(tmp_path):
    path = tmp_path / "t.pb"
    _write_stream(path, n_frames=60, interval=30)  # exactly 2 chunks
    data = bytearray(path.read_bytes())

    # Find the first chunk: skip magic(4) + u32 hlen + header.
    (hlen,) = struct.unpack_from("<I", data, 4)
    first_chunk_payload_at = 4 + 4 + hlen + 4 + 1  # + u32 clen + u8 method
    # Flip bytes well inside the first chunk's zlib payload.
    for off in range(first_chunk_payload_at + 4, first_chunk_payload_at + 8):
        data[off] ^= 0xFF
    bad = tmp_path / "bad.pb"
    bad.write_bytes(bytes(data))

    with pytest.raises(LandmarkDecodeError, match="chunk 0"):
        list(LandmarkDecoder(bad).frames())


def test_bad_magic_raises_valueerror(tmp_path):
    p = tmp_path / "not_landmarks.pb"
    p.write_bytes(b"XXXX" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not an ALICE landmark telemetry file"):
        LandmarkDecoder(p)


def test_unsupported_version_raises(tmp_path):
    path = tmp_path / "t.pb"
    _write_stream(path, n_frames=5, interval=30)
    data = bytearray(path.read_bytes())
    # Rewrite the header with version=99: parse then re-serialize.
    from backend.shared.proto_gen import landmarks_pb2 as pb

    (hlen,) = struct.unpack_from("<I", data, 4)
    header = pb.LandmarkStreamHeader.FromString(bytes(data[8 : 8 + hlen]))
    header.version = 99
    hb = header.SerializeToString()
    rebuilt = MAGIC + struct.pack("<I", len(hb)) + hb + bytes(data[8 + hlen :])
    p2 = tmp_path / "v99.pb"
    p2.write_bytes(rebuilt)

    with pytest.raises(ValueError, match="version 99"):
        LandmarkDecoder(p2)


def test_empty_file_and_header_only_file(tmp_path):
    empty = tmp_path / "empty.pb"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        LandmarkDecoder(empty)

    header_only = tmp_path / "h.pb"
    with LandmarkEncoder(header_only, source_fps=30.0):
        pass
    assert list(LandmarkDecoder(header_only).frames()) == []
