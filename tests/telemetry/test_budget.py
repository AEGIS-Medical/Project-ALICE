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
