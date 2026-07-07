"""Smoke tests for the committed protobuf codegen."""
from __future__ import annotations

from backend.shared.proto_gen import landmarks_pb2 as pb


def test_header_roundtrip():
    h = pb.LandmarkStreamHeader(
        version=1, landmark_count=478, source_fps=30.0,
        keyframe_interval=30, frame_skip=1,
    )
    data = h.SerializeToString()
    h2 = pb.LandmarkStreamHeader.FromString(data)
    assert h2.version == 1
    assert h2.landmark_count == 478
    assert abs(h2.source_fps - 30.0) < 1e-6


def test_chunk_with_all_frame_kinds_roundtrips():
    chunk = pb.LandmarkChunk()
    kf = chunk.frames.add()
    kf.key.frame_number = 0
    kf.key.xy.extend([1, 2, 3, 4])
    kf.key.z.extend([-5, 6])
    df = chunk.frames.add()
    df.delta.frame_number = 1
    df.delta.dxy.extend([-1, 1, 0, 2])
    df.delta.dz.extend([3, -3])
    nf = chunk.frames.add()
    nf.no_face.frame_number = 2

    data = chunk.SerializeToString()
    c2 = pb.LandmarkChunk.FromString(data)
    kinds = [f.WhichOneof("kind") for f in c2.frames]
    assert kinds == ["key", "delta", "no_face"]
    assert list(c2.frames[0].key.xy) == [1, 2, 3, 4]
    assert list(c2.frames[1].delta.dxy) == [-1, 1, 0, 2]
