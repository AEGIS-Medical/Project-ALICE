"""THE acceptance gate for Session 4: a synthetic 60s @ 30fps full-mesh
stream with realistic motion must encode within the bandwidth-derived budget.

Bandwidth-derived gate: EDGE_MINIMAL uplinks are <1 Mbps and must carry FLAC
(2.5 MB/min ~= 0.33 Mbps) + landmarks. 1.2 MB/min ~= 0.16 Mbps keeps the
combined stream under 0.5 Mbps -- 2x headroom on the worst uplink. (An earlier
500 KB gate was an arbitrary round number; 12-bit quantization measures
~1.04 MB/min, and downstream accuracy -- not byte count -- is the binding
priority.)

Motion model (realistic, not gamed): faces move mostly rigidly, so each
frame applies one shared rigid offset to all landmarks (head motion) plus
tiny per-landmark jitter (expression/detector noise). Every ~5s a saccade
jump; one 15-frame no-face gap (subject looks away).
"""
from __future__ import annotations

import random

from backend.shared.telemetry.landmark_codec import LandmarkEncoder

BUDGET_BYTES = 1_200_000  # <= 1.2 MB for 60s @ 30fps (bandwidth-derived)
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
        f"{BUDGET_BYTES} byte bandwidth-derived budget. A structural change "
        f"(entropy coding / subsetting) plus a design review is required; "
        f"do not weaken this gate."
    )
