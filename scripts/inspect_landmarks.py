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
          f"(~0.13 px @1080p)")
    print(f"  JSONL-equivalent estimate: ~{jsonl_equiv_mb:.1f} MB")

    if head_dump:
        print("-" * 64)
        print(json.dumps(head_dump, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
