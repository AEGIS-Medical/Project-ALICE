"""Quantization round-trip and clamping tests."""
from __future__ import annotations

import random

from backend.shared.telemetry.landmark_codec import (
    XY_SCALE,
    Z_SCALE,
    dequantize_frame,
    quantize_frame,
)

MAX_XY_ERR = 1.0 / (2 * XY_SCALE)   # 1/8190 at the v1 12-bit scale
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
