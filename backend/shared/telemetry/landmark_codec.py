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
