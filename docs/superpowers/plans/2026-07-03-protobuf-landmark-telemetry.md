# Protobuf Landmark Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the streaming JSONL landmark artifact with a keyframe/delta, uint16-quantized, zlib-chunked protobuf format that encodes a 60 s @ 30 fps full-mesh stream in ≤ 500 KB, closing CLAUDE.md known-gap #1.

**Architecture:** A `proto/landmarks.proto` schema (codegen committed to git) defines KeyFrame/DeltaFrame/NoFaceFrame messages. A standalone codec (`LandmarkEncoder`/`LandmarkDecoder` in `backend/shared/telemetry/landmark_codec.py`) owns quantization, the delta rule, and `ALTM` chunk framing; `FeatureExtractor.extract_landmarks()` swaps its JSONL writer for the encoder with an unchanged public contract. Every chunk starts at a keyframe, so any prefix ending on a chunk boundary is decodable — preserving P1-S6's crash-recovery guarantee.

**Tech Stack:** Python 3.13, `protobuf` (runtime), `grpcio-tools` (dev-only, bundles protoc), zlib + struct (stdlib), pytest, numpy (already present) for the synthetic budget test.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-03-protobuf-landmark-telemetry-design.md`. If plan and spec conflict, the spec governs.
- Run everything from `C:\Users\ryanh\ALICE\Project-ALICE` with `.venv/Scripts/python` (Windows, Python 3.13).
- File magic is exactly `b"ALTM"`. All integer prefixes little-endian: `<I` for lengths, `<B` for the compression-method byte. Framing: `MAGIC | u32 header_len | header_bytes` then repeating `u32 payload_len | u8 method | payload`. `method`: 1 = zlib, 0 = uncompressed.
- Quantization: x,y clamped to [0,1] then `round(v * 65535)` (uint16 range). z clamped to [-1,1] then `round(v * 32767)`. Max x/y reconstruction error `1/131070`.
- Keyframe/delta rule: a face frame is a `DeltaFrame` iff the previous emitted frame had a face AND it is not the first frame of a chunk. After a `NoFaceFrame`, the next face frame is a `KeyFrame`. `chunk_size == keyframe_interval` (default 30), so every chunk starts at a keyframe (or no-face frames followed by a keyframe).
- Timestamps are never stored per frame: `timestamp_seconds = frame_number / header.source_fps`.
- `landmarks_pb2.py` is generated ONCE and COMMITTED. Builds/CI never require protoc. `make proto` regenerates via `python -m grpc_tools.protoc`.
- CLAUDE.md invariant #3: codec logs counts/bytes only — never coordinate values. Invariant #9: the `.proto` carries a comment forbidding image data in this schema. Invariant #6: never use the phrase "lie detector".
- `FeatureExtractor.extract_landmarks()` keeps its exact signature and `Path` return; output becomes `{stem}_landmarks.pb`. `flush_interval` is kept as a deprecated alias mapping onto `keyframe_interval`.
- The hard acceptance gate: synthetic 60 s @ 30 fps realistic-motion stream encodes to ≤ 500,000 bytes.

---

## File Map

| File | Action | Task |
|---|---|---|
| `proto/landmarks.proto` | Create | 1 |
| `backend/shared/proto_gen/__init__.py` | Create (empty) | 1 |
| `backend/shared/proto_gen/landmarks_pb2.py` | Generate + commit | 1 |
| `pyproject.toml` | Modify — `protobuf` runtime dep, `grpcio-tools` dev dep | 1 |
| `Makefile` | Modify — `make proto` target | 1 |
| `tests/telemetry/__init__.py` | Create (empty) | 1 |
| `tests/telemetry/test_proto_gen.py` | Create | 1 |
| `backend/shared/telemetry/__init__.py` | Create | 2 |
| `backend/shared/telemetry/landmark_codec.py` | Create — quantization helpers | 2 |
| `tests/telemetry/test_quantization.py` | Create | 2 |
| `backend/shared/telemetry/landmark_codec.py` | Extend — encoder + decoder | 3 |
| `tests/telemetry/test_codec_roundtrip.py` | Create | 3 |
| `tests/telemetry/test_recovery.py` | Create | 4 |
| `tests/telemetry/test_budget.py` | Create — the ≤500 KB gate | 5 |
| `backend/workers/app/compression/feature_extractor.py` | Modify — swap JSONL for encoder | 6 |
| `tests/compression/test_feature_extractor.py` | Modify — rewrite 4 JSONL tests | 6 |
| `backend/shared/schemas/media.py` | Modify — `landmarks_path` docstring | 6 |
| `scripts/test_compress_and_analyze.py` | Modify — one print label | 6 |
| `scripts/inspect_landmarks.py` | Create — decode CLI | 7 |
| `tests/telemetry/test_inspect_cli.py` | Create | 7 |
| `CLAUDE.md` | Modify — status + measured figure + gap #1 resolved | 7 |

---

## Task 1: Proto Schema, Codegen, and Build Wiring

**Files:**
- Create: `proto/landmarks.proto`
- Create: `backend/shared/proto_gen/__init__.py` (empty)
- Generate + commit: `backend/shared/proto_gen/landmarks_pb2.py`
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Create: `tests/telemetry/__init__.py` (empty)
- Test: `tests/telemetry/test_proto_gen.py`

**Interfaces:**
- Produces: importable module `backend.shared.proto_gen.landmarks_pb2` with messages `LandmarkStreamHeader(version, landmark_count, source_fps, keyframe_interval, frame_skip)`, `KeyFrame(frame_number, xy, z)`, `DeltaFrame(frame_number, dxy, dz)`, `NoFaceFrame(frame_number)`, `Frame(oneof kind: key|delta|no_face)`, `LandmarkChunk(frames)`. All later tasks import this module.

- [ ] **Step 1: Add dependencies to pyproject.toml**

In the `[project] dependencies` list, append after `"vaderSentiment>=3.3.2",`:

```toml
    # Landmark telemetry wire format (proto/landmarks.proto).
    "protobuf>=5.26",
```

In the `dev` extra, append after `"mypy",`:

```toml
    # Bundles a protoc matching the protobuf runtime; used only by `make proto`
    # to regenerate backend/shared/proto_gen/landmarks_pb2.py (which is committed).
    "grpcio-tools>=1.66",
```

Then install:

```bash
cd C:\Users\ryanh\ALICE\Project-ALICE
.venv/Scripts/python -m pip install -e ".[dev]" -q
.venv/Scripts/python -c "import google.protobuf, grpc_tools; print('deps ok')"
```
Expected: `deps ok`

- [ ] **Step 2: Write the failing codegen smoke test**

```bash
mkdir -p tests/telemetry
printf '' > tests/telemetry/__init__.py
```

Create `tests/telemetry/test_proto_gen.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/telemetry/test_proto_gen.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.shared.proto_gen'`

- [ ] **Step 4: Create the .proto schema**

```bash
mkdir -p proto backend/shared/proto_gen
printf '' > backend/shared/proto_gen/__init__.py
```

Create `proto/landmarks.proto`:

```protobuf
syntax = "proto3";

package alice.telemetry.v1;

// ALICE landmark telemetry wire schema (Session 4).
// Spec: docs/superpowers/specs/2026-07-03-protobuf-landmark-telemetry-design.md
//
// CLAUDE.md invariant #9: this schema carries landmark geometry ONLY.
// Raw image or video data must NEVER be added to any message in this file.

message LandmarkStreamHeader {
  uint32 version = 1;           // format version, starts at 1
  uint32 landmark_count = 2;    // 478 for MediaPipe Face Mesh
  float source_fps = 3;         // timestamps reconstruct as frame_number / source_fps
  uint32 keyframe_interval = 4; // emitted frames between keyframes (== chunk size)
  uint32 frame_skip = 5;        // extractor's frame_skip setting
}

message KeyFrame {
  uint32 frame_number = 1;
  // 2*landmark_count values: x0,y0,x1,y1,... quantized round(v*65535), v in [0,1].
  repeated uint32 xy = 2 [packed = true];
  // landmark_count values: quantized round(v*32767), v in [-1,1].
  repeated sint32 z = 3 [packed = true];
}

message DeltaFrame {
  uint32 frame_number = 1;
  // Same layouts as KeyFrame, but each value is (current - previous frame).
  repeated sint32 dxy = 2 [packed = true];
  repeated sint32 dz = 3 [packed = true];
}

// A frame where no face was detected (preserves JSONL's "landmarks: null").
message NoFaceFrame {
  uint32 frame_number = 1;
}

message Frame {
  oneof kind {
    KeyFrame key = 1;
    DeltaFrame delta = 2;
    NoFaceFrame no_face = 3;
  }
}

message LandmarkChunk {
  repeated Frame frames = 1; // up to keyframe_interval frames; starts at a keyframe
}
```

- [ ] **Step 5: Generate the code and commit it**

```bash
.venv/Scripts/python -m grpc_tools.protoc -Iproto --python_out=backend/shared/proto_gen proto/landmarks.proto
ls backend/shared/proto_gen
```
Expected: `landmarks_pb2.py` listed (plus `__init__.py`).

- [ ] **Step 6: Add the `make proto` target**

In `Makefile`, change `.PHONY: install test-compress` to `.PHONY: install test-compress proto` and append:

```makefile
# Regenerate protobuf codegen after editing proto/landmarks.proto.
# The generated file is COMMITTED; normal builds never need this.
proto:
	$(PYTHON) -m grpc_tools.protoc -Iproto --python_out=backend/shared/proto_gen proto/landmarks.proto
```

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/telemetry/test_proto_gen.py -q`
Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add proto/landmarks.proto backend/shared/proto_gen/ pyproject.toml Makefile tests/telemetry/__init__.py tests/telemetry/test_proto_gen.py
git commit -m "feat(telemetry): add landmarks.proto schema + committed codegen + make proto"
```

---

## Task 2: Quantization Helpers

**Files:**
- Create: `backend/shared/telemetry/__init__.py`
- Create: `backend/shared/telemetry/landmark_codec.py` (quantization section only)
- Test: `tests/telemetry/test_quantization.py`

**Interfaces:**
- Produces (module-level functions in `backend.shared.telemetry.landmark_codec`):
  - `quantize_frame(landmarks: list[list[float]]) -> tuple[list[int], list[int], int]` — returns `(xy, z, clamped_count)`; `xy` has `2*len(landmarks)` uint16-range ints interleaved x0,y0,x1,y1...; `z` has `len(landmarks)` int16-range ints; `clamped_count` counts inputs that fell outside [0,1] (x,y) or [-1,1] (z).
  - `dequantize_frame(xy: Sequence[int], z: Sequence[int]) -> list[list[float]]` — inverse, returns `[[x, y, z], ...]` floats.
  - Constants `XY_SCALE = 65535`, `Z_SCALE = 32767`.

- [ ] **Step 1: Write the failing quantization tests**

Create `tests/telemetry/test_quantization.py`:

```python
"""Quantization round-trip and clamping tests."""
from __future__ import annotations

import random

from backend.shared.telemetry.landmark_codec import (
    XY_SCALE,
    Z_SCALE,
    dequantize_frame,
    quantize_frame,
)

MAX_XY_ERR = 1.0 / (2 * XY_SCALE)   # 1/131070
MAX_Z_ERR = 1.0 / (2 * Z_SCALE)


def test_roundtrip_error_bounded():
    rng = random.Random(42)
    landmarks = [[rng.random(), rng.random(), rng.uniform(-1, 1)] for _ in range(478)]
    xy, z, clamped = quantize_frame(landmarks)
    assert clamped == 0
    assert len(xy) == 956 and len(z) == 478
    out = dequantize_frame(xy, z)
    for (x0, y0, z0), (x1, y1, z1) in zip(landmarks, out):
        assert abs(x0 - x1) <= MAX_XY_ERR
        assert abs(y0 - y1) <= MAX_XY_ERR
        assert abs(z0 - z1) <= MAX_Z_ERR


def test_out_of_range_values_clamped_and_counted():
    landmarks = [[-0.01, 1.02, -1.5]] + [[0.5, 0.5, 0.0]] * 2
    xy, z, clamped = quantize_frame(landmarks)
    assert clamped == 3  # x under, y over, z under
    out = dequantize_frame(xy, z)
    assert out[0][0] == 0.0
    assert abs(out[0][1] - 1.0) <= MAX_XY_ERR
    assert abs(out[0][2] - (-1.0)) <= MAX_Z_ERR


def test_boundary_values_exact():
    landmarks = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    xy, z, clamped = quantize_frame(landmarks)
    assert clamped == 0
    assert xy == [0, XY_SCALE, XY_SCALE, 0]
    assert z == [-Z_SCALE, Z_SCALE]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/telemetry/test_quantization.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.shared.telemetry'`

- [ ] **Step 3: Create the telemetry package with quantization helpers**

Create `backend/shared/telemetry/__init__.py`:

```python
"""Telemetry codecs for Project ALICE edge-first payloads."""
```

Create `backend/shared/telemetry/landmark_codec.py`:

```python
"""Landmark telemetry codec: quantization, keyframe/delta encoding, chunk framing.

Wire format spec: docs/superpowers/specs/2026-07-03-protobuf-landmark-telemetry-design.md
Schema: proto/landmarks.proto (codegen committed at backend/shared/proto_gen).

CLAUDE.md invariant #3: this module logs counts/bytes/ratios only -- never
coordinate values.
"""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# Quantization scales. x,y are MediaPipe-normalized [0,1] -> uint16;
# z is approximately [-1,1] -> int16. Max x/y reconstruction error is
# 1/(2*XY_SCALE) ~= 0.008 px at 1080p -- far below AU-detection sensitivity.
XY_SCALE: int = 65535
Z_SCALE: int = 32767


def quantize_frame(
    landmarks: list[list[float]],
) -> tuple[list[int], list[int], int]:
    """Quantize one frame of [x, y, z] landmarks.

    Returns:
        (xy, z, clamped_count): ``xy`` interleaves x0,y0,x1,y1,... as ints in
        [0, XY_SCALE]; ``z`` holds ints in [-Z_SCALE, Z_SCALE];
        ``clamped_count`` is how many input values fell outside their legal
        range and were clamped (MediaPipe emits slight overshoot at frame
        edges -- clamping is expected, but counted for telemetry).
    """
    xy: list[int] = []
    z: list[int] = []
    clamped = 0
    for point in landmarks:
        px, py, pz = point[0], point[1], point[2]
        for v in (px, py):
            if v < 0.0 or v > 1.0:
                clamped += 1
                v = 0.0 if v < 0.0 else 1.0
            xy.append(round(v * XY_SCALE))
        if pz < -1.0 or pz > 1.0:
            clamped += 1
            pz = -1.0 if pz < -1.0 else 1.0
        z.append(round(pz * Z_SCALE))
    return xy, z, clamped


def dequantize_frame(
    xy: Sequence[int], z: Sequence[int]
) -> list[list[float]]:
    """Inverse of :func:`quantize_frame` -- returns [[x, y, z], ...] floats."""
    return [
        [xy[2 * i] / XY_SCALE, xy[2 * i + 1] / XY_SCALE, z[i] / Z_SCALE]
        for i in range(len(z))
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/telemetry/test_quantization.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/shared/telemetry/ tests/telemetry/test_quantization.py
git commit -m "feat(telemetry): add landmark quantization helpers (uint16 xy, int16 z, clamp counting)"
```

---

## Task 3: LandmarkEncoder + LandmarkDecoder (core round-trip)

**Files:**
- Modify: `backend/shared/telemetry/landmark_codec.py` (append encoder/decoder)
- Test: `tests/telemetry/test_codec_roundtrip.py`

**Interfaces:**
- Consumes: `quantize_frame`/`dequantize_frame` (Task 2); `backend.shared.proto_gen.landmarks_pb2` (Task 1).
- Produces:
  - `MAGIC = b"ALTM"`, `COMPRESSION_ZLIB = 1`, `COMPRESSION_NONE = 0`
  - `class LandmarkDecodeError(RuntimeError)`
  - `class DecodedFrame(NamedTuple)`: `frame_number: int`, `timestamp_seconds: float`, `landmarks: list[list[float]] | None`
  - `class LandmarkEncoder`: `__init__(path: Path, *, landmark_count: int = 478, source_fps: float, frame_skip: int = 1, keyframe_interval: int = 30, zlib_level: int = 6)`; `add_frame(frame_number: int, landmarks: list[list[float]] | None) -> None`; `close() -> None` (idempotent); context manager; telemetry attrs `frames_written`, `chunks_written`, `bytes_written`, `clamped_values`.
  - `class LandmarkDecoder`: `__init__(path: Path)`; `.header` property; `frames() -> Iterator[DecodedFrame]`; telemetry attr `chunks_read`.

- [ ] **Step 1: Write the failing round-trip tests**

Create `tests/telemetry/test_codec_roundtrip.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/telemetry/test_codec_roundtrip.py -q`
Expected: FAIL — `ImportError` (`LandmarkEncoder` not defined).

- [ ] **Step 3: Append encoder + decoder to `landmark_codec.py`**

Append to `backend/shared/telemetry/landmark_codec.py` (after the quantization helpers), and extend the module's imports at the top to:

```python
import logging
import struct
import zlib
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Sequence

from backend.shared.proto_gen import landmarks_pb2 as pb
```

Appended code:

```python
# ---- File framing constants --------------------------------------------
MAGIC: bytes = b"ALTM"
COMPRESSION_NONE: int = 0
COMPRESSION_ZLIB: int = 1

_LEN = struct.Struct("<I")
_METHOD = struct.Struct("<B")


class LandmarkDecodeError(RuntimeError):
    """Raised for real damage inside a length-valid chunk (corruption or a
    spec-violating frame sequence). Truncation is NOT an error -- see
    LandmarkDecoder.frames()."""


class DecodedFrame(NamedTuple):
    frame_number: int
    timestamp_seconds: float
    landmarks: Optional[list[list[float]]]  # 478 x [x, y, z], or None (no face)


class LandmarkEncoder:
    """Streaming writer for the ALTM landmark telemetry format.

    The header is written at construction, so the file is decodable from
    frame 0. Frames buffer into a LandmarkChunk that is zlib-compressed and
    flushed every ``keyframe_interval`` frames; every chunk therefore starts
    at a keyframe (or no-face frames followed by one), which is what makes
    any complete-chunk prefix of the file independently decodable.
    """

    def __init__(
        self,
        path: Path,
        *,
        landmark_count: int = 478,
        source_fps: float,
        frame_skip: int = 1,
        keyframe_interval: int = 30,
        zlib_level: int = 6,
    ) -> None:
        if source_fps <= 0:
            raise ValueError(f"source_fps must be > 0 (got {source_fps})")
        if keyframe_interval < 1:
            raise ValueError(
                f"keyframe_interval must be >= 1 (got {keyframe_interval})"
            )
        self._landmark_count = landmark_count
        self._keyframe_interval = keyframe_interval
        self._zlib_level = zlib_level
        self._closed = False

        # Previous emitted face frame's quantized values (delta reference).
        self._prev_xy: Optional[list[int]] = None
        self._prev_z: Optional[list[int]] = None

        self._chunk = pb.LandmarkChunk()
        self._frames_in_chunk = 0

        # Telemetry (final values valid after close()).
        self.frames_written = 0
        self.chunks_written = 0
        self.bytes_written = 0
        self.clamped_values = 0

        self._fh = open(path, "wb")
        header = pb.LandmarkStreamHeader(
            version=1,
            landmark_count=landmark_count,
            source_fps=source_fps,
            keyframe_interval=keyframe_interval,
            frame_skip=frame_skip,
        )
        hb = header.SerializeToString()
        self._fh.write(MAGIC)
        self._fh.write(_LEN.pack(len(hb)))
        self._fh.write(hb)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "LandmarkEncoder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- public API ----------------------------------------------------------
    def add_frame(
        self, frame_number: int, landmarks: Optional[list[list[float]]]
    ) -> None:
        if self._closed:
            raise ValueError("LandmarkEncoder is closed")

        frame = self._chunk.frames.add()
        if landmarks is None:
            frame.no_face.frame_number = frame_number
            self._prev_xy = None
            self._prev_z = None
        else:
            if len(landmarks) != self._landmark_count:
                raise ValueError(
                    f"Expected {self._landmark_count} landmarks, got "
                    f"{len(landmarks)} (frame {frame_number})"
                )
            xy, z, clamped = quantize_frame(landmarks)
            self.clamped_values += clamped

            # Delta iff we have a reference AND this is not a chunk start.
            if self._prev_xy is not None and self._frames_in_chunk > 0:
                frame.delta.frame_number = frame_number
                frame.delta.dxy.extend(
                    c - p for c, p in zip(xy, self._prev_xy)
                )
                frame.delta.dz.extend(
                    c - p for c, p in zip(z, self._prev_z)
                )
            else:
                frame.key.frame_number = frame_number
                frame.key.xy.extend(xy)
                frame.key.z.extend(z)
            self._prev_xy = xy
            self._prev_z = z

        self.frames_written += 1
        self._frames_in_chunk += 1
        if self._frames_in_chunk >= self._keyframe_interval:
            self._flush_chunk()

    def close(self) -> None:
        if self._closed:
            return
        self._flush_chunk()
        self.bytes_written = self._fh.tell()
        self._fh.close()
        self._closed = True
        logger.info(
            "landmark_encode_done frames=%d chunks=%d bytes=%d clamped=%d",
            self.frames_written, self.chunks_written,
            self.bytes_written, self.clamped_values,
        )

    # -- internals ------------------------------------------------------------
    def _flush_chunk(self) -> None:
        if not self._chunk.frames:
            return
        payload = self._chunk.SerializeToString()
        compressed = zlib.compress(payload, self._zlib_level)
        self._fh.write(_LEN.pack(len(compressed)))
        self._fh.write(_METHOD.pack(COMPRESSION_ZLIB))
        self._fh.write(compressed)
        self._fh.flush()
        self.chunks_written += 1
        self._chunk = pb.LandmarkChunk()
        self._frames_in_chunk = 0


class LandmarkDecoder:
    """Streaming reader for the ALTM landmark telemetry format.

    A truncated final chunk (crash mid-write) is skipped with a warning --
    recovery is the feature. Corruption inside a length-valid chunk raises
    LandmarkDecodeError.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self.chunks_read = 0
        with open(self._path, "rb") as fh:
            magic = fh.read(4)
            if magic != MAGIC:
                raise ValueError(
                    f"not an ALICE landmark telemetry file: {self._path}"
                )
            raw_len = fh.read(_LEN.size)
            if len(raw_len) < _LEN.size:
                raise ValueError(f"truncated header in {self._path}")
            (hlen,) = _LEN.unpack(raw_len)
            hb = fh.read(hlen)
            if len(hb) < hlen:
                raise ValueError(f"truncated header in {self._path}")
            self._header = pb.LandmarkStreamHeader.FromString(hb)
            if self._header.version != 1:
                raise ValueError(
                    f"unsupported landmark telemetry version "
                    f"{self._header.version} in {self._path}"
                )
            self._body_offset = fh.tell()

    @property
    def header(self) -> "pb.LandmarkStreamHeader":
        return self._header

    def frames(self) -> Iterator[DecodedFrame]:
        fps = self._header.source_fps
        chunk_index = 0
        with open(self._path, "rb") as fh:
            fh.seek(self._body_offset)
            while True:
                raw_len = fh.read(_LEN.size)
                if not raw_len:
                    return  # clean EOF
                if len(raw_len) < _LEN.size:
                    logger.warning(
                        "landmark_decode_truncated path=%s chunk=%d",
                        self._path, chunk_index,
                    )
                    return
                (clen,) = _LEN.unpack(raw_len)
                raw_method = fh.read(_METHOD.size)
                payload = fh.read(clen) if raw_method else b""
                if len(raw_method) < _METHOD.size or len(payload) < clen:
                    logger.warning(
                        "landmark_decode_truncated path=%s chunk=%d",
                        self._path, chunk_index,
                    )
                    return
                (method,) = _METHOD.unpack(raw_method)
                try:
                    if method == COMPRESSION_ZLIB:
                        payload = zlib.decompress(payload)
                    elif method != COMPRESSION_NONE:
                        raise LandmarkDecodeError(
                            f"unknown compression method {method} "
                            f"(chunk {chunk_index}) in {self._path}"
                        )
                    chunk = pb.LandmarkChunk.FromString(payload)
                except LandmarkDecodeError:
                    raise
                except Exception as exc:
                    raise LandmarkDecodeError(
                        f"corrupt chunk {chunk_index} in {self._path}: {exc}"
                    ) from exc

                # Delta reference resets at every chunk boundary by spec.
                prev_xy: Optional[list[int]] = None
                prev_z: Optional[list[int]] = None
                for f in chunk.frames:
                    kind = f.WhichOneof("kind")
                    if kind == "no_face":
                        prev_xy = None
                        prev_z = None
                        yield DecodedFrame(
                            f.no_face.frame_number,
                            f.no_face.frame_number / fps,
                            None,
                        )
                    elif kind == "key":
                        prev_xy = list(f.key.xy)
                        prev_z = list(f.key.z)
                        yield DecodedFrame(
                            f.key.frame_number,
                            f.key.frame_number / fps,
                            dequantize_frame(prev_xy, prev_z),
                        )
                    elif kind == "delta":
                        if prev_xy is None:
                            raise LandmarkDecodeError(
                                f"delta frame {f.delta.frame_number} with no "
                                f"prior keyframe (chunk {chunk_index}) in "
                                f"{self._path}"
                            )
                        prev_xy = [p + d for p, d in zip(prev_xy, f.delta.dxy)]
                        prev_z = [p + d for p, d in zip(prev_z, f.delta.dz)]
                        yield DecodedFrame(
                            f.delta.frame_number,
                            f.delta.frame_number / fps,
                            dequantize_frame(prev_xy, prev_z),
                        )
                    else:  # pragma: no cover - proto3 empty oneof
                        raise LandmarkDecodeError(
                            f"frame with no kind (chunk {chunk_index}) in "
                            f"{self._path}"
                        )
                chunk_index += 1
                self.chunks_read = chunk_index
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/telemetry/test_codec_roundtrip.py tests/telemetry/test_quantization.py -q`
Expected: all pass (7 round-trip + 3 quantization).

- [ ] **Step 5: Commit**

```bash
git add backend/shared/telemetry/landmark_codec.py tests/telemetry/test_codec_roundtrip.py
git commit -m "feat(telemetry): LandmarkEncoder/LandmarkDecoder with keyframe/delta + ALTM chunk framing"
```

---

## Task 4: Recovery & Corruption Behavior

**Files:**
- Test: `tests/telemetry/test_recovery.py` (no source changes expected; if a test exposes a codec bug, fix it in `landmark_codec.py` within this task)

**Interfaces:**
- Consumes: `LandmarkEncoder`, `LandmarkDecoder`, `LandmarkDecodeError`, `MAGIC` (Task 3).

- [ ] **Step 1: Write the recovery tests**

Create `tests/telemetry/test_recovery.py`:

```python
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
```

- [ ] **Step 2: Run tests**

Run: `.venv/Scripts/python -m pytest tests/telemetry/test_recovery.py -q`
Expected: all 5 pass. If any fail, the codec (Task 3) has a bug — fix it in `landmark_codec.py` and re-run until green. Do not weaken the tests.

- [ ] **Step 3: Commit**

```bash
git add tests/telemetry/test_recovery.py backend/shared/telemetry/landmark_codec.py
git commit -m "test(telemetry): crash-recovery, corruption, bad-magic, and version-gate coverage"
```

---

## Task 5: The Budget Gate (≤ 500 KB/min)

**Files:**
- Test: `tests/telemetry/test_budget.py`
- Possibly modify: `backend/shared/telemetry/landmark_codec.py` (only the default `zlib_level`, if needed to pass)

**Interfaces:**
- Consumes: `LandmarkEncoder` (Task 3).

- [ ] **Step 1: Write the budget gate test**

Create `tests/telemetry/test_budget.py`:

```python
"""THE acceptance gate for Session 4: a synthetic 60s @ 30fps full-mesh
stream with realistic motion must encode to <= 500 KB.

Motion model (realistic, not gamed): faces move mostly rigidly, so each
frame applies one shared rigid offset to all landmarks (head motion) plus
tiny per-landmark jitter (expression/detector noise). Every ~5s a saccade
jump; one 15-frame no-face gap (subject looks away).
"""
from __future__ import annotations

import random

from backend.shared.telemetry.landmark_codec import LandmarkEncoder

BUDGET_BYTES = 500_000  # <= 500 KB for 60s @ 30fps
N_FRAMES = 1800


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def test_bytes_per_minute_budget(tmp_path):
    rng = random.Random(42)
    # Base face: landmarks spread over the central image region.
    base = [
        [rng.uniform(0.35, 0.65), rng.uniform(0.3, 0.7), rng.uniform(-0.05, 0.05)]
        for _ in range(478)
    ]
    frame = [list(p) for p in base]

    path = tmp_path / "budget.pb"
    with LandmarkEncoder(path, source_fps=30.0) as enc:
        for i in range(N_FRAMES):
            if 900 <= i < 915:  # 15-frame no-face gap at t=30s
                enc.add_frame(i, None)
                continue
            # Shared rigid head motion per frame.
            dx = rng.gauss(0.0, 0.0008)
            dy = rng.gauss(0.0, 0.0008)
            # Occasional saccade jump (~every 5s).
            if i % 150 == 0 and i > 0:
                dx += rng.uniform(-0.02, 0.02)
                dy += rng.uniform(-0.02, 0.02)
            frame = [
                [
                    _clamp01(x + dx + rng.gauss(0.0, 0.0002)),
                    _clamp01(y + dy + rng.gauss(0.0, 0.0002)),
                    min(1.0, max(-1.0, z + rng.gauss(0.0, 0.0002))),
                ]
                for x, y, z in frame
            ]
            enc.add_frame(i, frame)
        enc.close()
        size = enc.bytes_written

    kb_per_min = size / 1024.0
    print(f"\nbudget test: {size} bytes for 60s @ 30fps = {kb_per_min:.0f} KB/min")
    assert size <= BUDGET_BYTES, (
        f"encoded {size} bytes ({kb_per_min:.0f} KB/min) — over the "
        f"{BUDGET_BYTES} byte budget. Tune zlib_level (default 6 -> 9) or "
        f"revisit the delta encoding before weakening this gate."
    )
```

- [ ] **Step 2: Run the gate**

Run: `.venv/Scripts/python -m pytest tests/telemetry/test_budget.py -q -s`
Expected: PASS, printing the measured size. If it FAILS: first change the encoder's default `zlib_level` from 6 to 9 in `landmark_codec.py` and re-run. If still over budget, STOP and report the measured number — the spec's size model needs revisiting, which is a design decision, not a test-tuning decision.

- [ ] **Step 3: Commit**

```bash
git add tests/telemetry/test_budget.py backend/shared/telemetry/landmark_codec.py
git commit -m "test(telemetry): enforce <=500KB/min budget gate on synthetic realistic motion"
```

---

## Task 6: FeatureExtractor Integration (JSONL → .pb)

**Files:**
- Modify: `backend/workers/app/compression/feature_extractor.py`
- Modify: `tests/compression/test_feature_extractor.py` (rewrite the 4 P1-S6 JSONL tests)
- Modify: `backend/shared/schemas/media.py` (one docstring)
- Modify: `scripts/test_compress_and_analyze.py` (one print label)

**Interfaces:**
- Consumes: `LandmarkEncoder`, `LandmarkDecoder` (Task 3).
- Produces: `FeatureExtractor.extract_landmarks(video_path, output_dir, flush_interval=None) -> Path` — unchanged signature, now returns `{stem}_landmarks.pb`. `flush_interval` (init and method param) is a deprecated alias for the encoder's `keyframe_interval`. Telemetry attrs (`last_frames_processed`, `last_frames_with_face`) unchanged.

- [ ] **Step 1: Rewrite the four P1-S6 tests against the codec**

In `tests/compression/test_feature_extractor.py`: replace the module docstring's "streaming JSONL" wording with "streaming protobuf (.pb)", delete the `import json` line, add `from backend.shared.telemetry.landmark_codec import LandmarkDecoder`, and replace the four test functions with:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/compression/test_feature_extractor.py -q`
Expected: FAIL — output still `.jsonl`.

- [ ] **Step 3: Swap the JSONL writer for the encoder in `feature_extractor.py`**

Read the current `extract_landmarks` implementation first. Make these changes:

1. Add the import near the other project imports:

```python
from backend.shared.telemetry.landmark_codec import LandmarkEncoder
```

2. In `__init__` and in `extract_landmarks`, keep the `flush_interval` parameters exactly as they are, but update their docstrings to say: `Deprecated alias for the telemetry encoder's keyframe_interval (chunk flush cadence). Retained for P1-S6 call-site compatibility.`

3. Replace the writer section — from the `output_path = output_dir / f"{video_path.stem}_landmarks.jsonl"` line through the end of the write loop (the block that builds `write_buffer`, `json.dumps(record)`, and the tail flush) — with:

```python
        output_path = output_dir / f"{video_path.stem}_landmarks.pb"
        interval = self.flush_interval if flush_interval is None else flush_interval
        if interval < 1:
            raise ValueError(f"flush_interval must be >= 1 (got {interval}).")

        landmarker = vision.FaceLandmarker.create_from_options(options)
        try:
            with LandmarkEncoder(
                output_path,
                source_fps=fps,
                frame_skip=self.frame_skip,
                keyframe_interval=interval,
            ) as encoder:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % self.frame_skip == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        timestamp_ms = int((frame_idx / fps) * 1000)
                        result = landmarker.detect_for_video(mp_image, timestamp_ms)

                        landmarks: list[list[float]] | None = None
                        if result.face_landmarks:
                            mesh = result.face_landmarks[0]
                            landmarks = [[lm.x, lm.y, lm.z] for lm in mesh]
                            frames_with_face += 1

                        encoder.add_frame(frame_idx, landmarks)
                        frames_processed += 1
                    frame_idx += 1
        finally:
            landmarker.close()
            cap.release()
```

Keep the existing MediaPipe `options` construction, the telemetry attribute assignments, and the final `logger.info` + `return output_path` exactly as they are (the log's `output_size` still comes from `output_path.stat().st_size`). Remove the now-unused `import json` if nothing else in the file uses it.

- [ ] **Step 4: Update the two cosmetic references**

In `backend/shared/schemas/media.py`, `landmarks_path` field description: change `parquet/protobuf` (or the current wording) to `ALTM protobuf telemetry (.pb; see proto/landmarks.proto)`.

In `scripts/test_compress_and_analyze.py`: change the print label `"Landmarks (JSONL):"` to `"Landmarks (protobuf):"`.

- [ ] **Step 5: Run the compression suite, then the full suite**

Run: `.venv/Scripts/python -m pytest tests/compression/ -q`
Expected: all pass.
Run: `.venv/Scripts/python -m pytest tests/ -p no:cacheprovider`
Expected: all pass (previous 74 + new telemetry tests), 1 deselected (slow).

- [ ] **Step 6: Commit**

```bash
git add backend/workers/app/compression/feature_extractor.py tests/compression/test_feature_extractor.py backend/shared/schemas/media.py scripts/test_compress_and_analyze.py
git commit -m "feat(compression): emit landmarks as ALTM protobuf telemetry (replaces JSONL)"
```

---

## Task 7: Inspector CLI + Measured Figure + CLAUDE.md Sync

**Files:**
- Create: `scripts/inspect_landmarks.py`
- Test: `tests/telemetry/test_inspect_cli.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `LandmarkDecoder`, `LandmarkEncoder` (Task 3), `XY_SCALE` (Task 2).

- [ ] **Step 1: Write the failing CLI smoke test**

Create `tests/telemetry/test_inspect_cli.py`:

```python
"""Smoke test: inspect_landmarks.py decodes a .pb and prints stats."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from backend.shared.telemetry.landmark_codec import LandmarkEncoder

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_stream(path: Path) -> None:
    with LandmarkEncoder(path, source_fps=30.0) as enc:
        for i in range(40):
            if i == 5:
                enc.add_frame(i, None)
            else:
                enc.add_frame(i, [[0.5, 0.5, 0.0]] * 478)


def test_inspect_prints_stats_and_head(tmp_path):
    pb_path = tmp_path / "clip_landmarks.pb"
    _make_stream(pb_path)
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "inspect_landmarks.py"),
         str(pb_path), "--head", "2"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "version" in out
    assert "frames:" in out
    assert "face coverage:" in out
    assert "KB/min" in out
    assert '"frame_number": 0' in out  # --head dump


def test_inspect_rejects_non_telemetry_file(tmp_path):
    bogus = tmp_path / "x.pb"
    bogus.write_bytes(b"XXXX not a telemetry file")
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "inspect_landmarks.py"),
         str(bogus)],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "not an ALICE landmark telemetry file" in result.stderr
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/telemetry/test_inspect_cli.py -q`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Create `scripts/inspect_landmarks.py`**

```python
#!/usr/bin/env python
"""Inspect an ALICE landmark telemetry (.pb) file.

The debugging replacement for the old JSONL format: prints the stream
header, frame/chunk counts, face coverage, effective bytes/min at source
fps, the theoretical quantization error bound, and a JSONL-size-equivalent
estimate for contrast. --head N dumps the first N decoded frames as JSON.

Usage:
    python scripts/inspect_landmarks.py path/to/clip_landmarks.pb
    python scripts/inspect_landmarks.py path/to/clip_landmarks.pb --head 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect landmark telemetry")
    parser.add_argument("pb_file", type=Path, help="Path to a _landmarks.pb file")
    parser.add_argument("--head", type=int, default=0, metavar="N",
                        help="Dump the first N decoded frames as JSON")
    args = parser.parse_args()

    if not args.pb_file.exists():
        print(f"ERROR: file not found: {args.pb_file}", file=sys.stderr)
        return 1

    from backend.shared.telemetry.landmark_codec import (
        XY_SCALE,
        LandmarkDecodeError,
        LandmarkDecoder,
    )

    try:
        dec = LandmarkDecoder(args.pb_file)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    h = dec.header
    print(f"\n{args.pb_file}")
    print("-" * 64)
    print(f"  version: {h.version} | landmarks: {h.landmark_count} | "
          f"fps: {h.source_fps:g} | keyframe interval: {h.keyframe_interval} | "
          f"frame_skip: {h.frame_skip}")

    head_dump: list[dict] = []
    n_frames = 0
    n_face = 0
    try:
        for d in dec.frames():
            if args.head and n_frames < args.head:
                head_dump.append({
                    "frame_number": d.frame_number,
                    "timestamp_seconds": round(d.timestamp_seconds, 4),
                    "landmarks": (
                        [[round(c, 5) for c in p] for p in d.landmarks[:2]]
                        + ["... (%d total)" % len(d.landmarks)]
                    ) if d.landmarks is not None else None,
                })
            n_frames += 1
            if d.landmarks is not None:
                n_face += 1
    except LandmarkDecodeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    size = args.pb_file.stat().st_size
    duration_s = (n_frames / h.source_fps) if (h.source_fps and n_frames) else 0.0
    kb_min = (size / duration_s * 60 / 1024) if duration_s else 0.0
    face_pct = (n_face / n_frames * 100.0) if n_frames else 0.0
    # JSONL contrast: ~25 bytes per landmark row in float text + framing.
    jsonl_equiv_mb = n_face * h.landmark_count * 25 / 1_048_576

    print(f"  frames: {n_frames} ({dec.chunks_read} chunks) | "
          f"face coverage: {face_pct:.1f}%")
    print(f"  size: {size:,} bytes | duration: {duration_s:.1f}s | "
          f"rate: {kb_min:.0f} KB/min")
    print(f"  quantization bound: <= 1/{2 * XY_SCALE} normalized "
          f"(~0.008 px @1080p)")
    print(f"  JSONL-equivalent estimate: ~{jsonl_equiv_mb:.1f} MB")

    if head_dump:
        print("-" * 64)
        print(json.dumps(head_dump, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI tests, then measure a real extraction**

Run: `.venv/Scripts/python -m pytest tests/telemetry/test_inspect_cli.py -q`
Expected: 2 passed.

Then produce a real measurement:

```bash
.venv/Scripts/python scripts/test_compress_and_analyze.py demo_data/honest/trial_truth_001.mp4 --fake --mode edge_full
.venv/Scripts/python scripts/inspect_landmarks.py "processed_output/compress_analyze_test/trial_truth_001/landmarks/trial_truth_001_landmarks.pb"
```
Record the printed `KB/min` figure — it goes into CLAUDE.md in the next step.

- [ ] **Step 5: Update CLAUDE.md**

Three edits:

1. In the Day-1 table, Feature extraction row: change `478-pt landmarks **streaming JSONL** (`_landmarks.jsonl`, flushed every N frames)` to `478-pt landmarks **ALTM protobuf telemetry** (`_landmarks.pb`, keyframe/delta + zlib chunks; see proto/landmarks.proto)`.

2. In "COMPRESSION ARCHITECTURE" → Tier 3, change `Transmitted: protobuf telemetry (~70KB/min)` to `Transmitted: landmark telemetry ~<MEASURED> KB/min at 30 fps (ALTM protobuf; ~70 KB/min remains the target for the future on-device AU-activation payload)` — substituting the measured figure from Step 4.

3. In "Known gaps & next-session priorities", rewrite item 1 to:

```markdown
1. ~~Landmark telemetry is ~170× over budget~~ **RESOLVED (Session 4):** ALTM
   protobuf format (proto/landmarks.proto) — uint16 quantization, keyframe/delta,
   zlib chunks. Measured ~<MEASURED> KB/min at 30 fps vs ~12 MB/min JSONL; ≤500 KB/min
   enforced by tests/telemetry/test_budget.py. The ~70 KB/min figure now applies to
   the future AU-activation payload.
```

- [ ] **Step 6: Full suite + commit**

Run: `.venv/Scripts/python -m pytest tests/ -p no:cacheprovider`
Expected: all pass, 1 deselected.

```bash
git add scripts/inspect_landmarks.py tests/telemetry/test_inspect_cli.py CLAUDE.md
git commit -m "feat(telemetry): add inspect_landmarks CLI; record measured KB/min; close gap #1 in CLAUDE.md"
```

---

## Final Verification

- [ ] `.venv/Scripts/python -m pytest tests/ -p no:cacheprovider` → all pass, 1 deselected (slow).
- [ ] `.venv/Scripts/python -m pytest tests/telemetry/test_budget.py -q -s` → prints measured size, ≤ 500,000 bytes.
- [ ] `.venv/Scripts/python scripts/test_compress_and_analyze.py demo_data/honest/trial_truth_001.mp4 --fake --mode edge_full` → runs end-to-end; landmarks line shows a `.pb` path.
- [ ] `git status --short` → clean.

---

## Self-Review Checklist

- [x] Spec coverage: schema+codegen (T1), quantization (T2), codec+framing+delta rule (T3), recovery/corruption table (T4), budget gate (T5), extractor swap + flush_interval deprecation + docstring/label updates (T6), inspector CLI + measured figure + CLAUDE.md sync (T7). JSONL replaced outright — no dual-write anywhere.
- [x] Framing consistent everywhere: `MAGIC | u32 hlen | header` then `u32 len | u8 method | payload`; method 1=zlib, 0=none.
- [x] Delta rule identical in encoder (T3), decoder (T3), and tests (T3/T4): chunk-start face frame is always a keyframe; NoFaceFrame resets the reference; decoder resets at chunk boundaries.
- [x] Type consistency: `DecodedFrame(frame_number, timestamp_seconds, landmarks)`, `chunks_read`, `bytes_written`, `frames_written`, `clamped_values` used identically across tasks.
- [x] No placeholders; every code step is complete; every run step has expected output.
- [x] Invariants: #3 (codec logs counts only), #9 (.proto comment forbids image data), #6 (no forbidden phrase).
- [x] Budget-gate escalation path defined (zlib 6→9, then STOP and report) — the gate is never weakened silently.
