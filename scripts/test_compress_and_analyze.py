#!/usr/bin/env python
"""Integration stub: video -> compression pipeline -> (pending) psycholinguistic.

Runs the real compression pipeline on a video, then shows exactly where WhisperX
transcription will plug in to feed PsycholinguisticAnalyzer. Transcription is not
yet implemented, so the script stops at a labelled stub.

Usage:
    python scripts/test_compress_and_analyze.py path/to/video.mp4
    python scripts/test_compress_and_analyze.py path/to/video.mp4 --mode edge_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compression -> psycholinguistic stub")
    parser.add_argument("video", type=Path, help="Path to a video file")
    parser.add_argument(
        "--mode",
        choices=["raw", "roi", "edge_full", "edge_minimal"],
        default="edge_full",
        help="Compression mode (default: edge_full)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1

    from backend.shared.schemas.media import CompressionMode
    from backend.workers.app.compression.pipeline import CompressionPipeline

    mode_map = {
        "raw": CompressionMode.RAW,
        "roi": CompressionMode.ROI_ENCODED,
        "edge_full": CompressionMode.EDGE_FULL,
        "edge_minimal": CompressionMode.EDGE_MINIMAL,
    }
    mode = mode_map[args.mode]
    output_dir = _REPO_ROOT / "processed_output" / "compress_analyze_test" / args.video.stem

    print(f"\nStep 1 -- Compression Pipeline ({mode.value})")
    print("-" * 64)
    pipeline = CompressionPipeline()
    result = pipeline.process(args.video, output_dir, mode)

    mb = 1_048_576
    print(f"  Input:              {result.input_size_bytes / mb:6.1f} MB")
    print(f"  FLAC audio (ML):    {result.flac_size_bytes / mb:6.2f} MB  -> {result.flac_audio_path}")
    if result.roi_video_path:
        print(f"  ROI video:          {(result.roi_video_size_bytes or 0) / mb:6.1f} MB  -> {result.roi_video_path}")
    if result.landmarks_path:
        print(f"  Landmarks (JSONL):  {(result.landmarks_size_bytes or 0) / mb:6.2f} MB  -> {result.landmarks_path}")
    if result.features_path:
        print(f"  Audio features:     {(result.features_size_bytes or 0) / mb:6.2f} MB  -> {result.features_path}")
    print(f"  Face detected:      {result.face_detected_pct:5.1f}% of frames")
    print(f"  Total time:         {result.processing_times.get('total', 0.0):5.1f}s")

    print("\nStep 2 -- Transcription (PENDING)")
    print("-" * 64)
    print(f"  Would run: WhisperX on {result.flac_audio_path}")
    print("  Would produce: speaker-attributed statement strings")
    print("  Status: WhisperX not yet integrated (next plan)")

    print("\nStep 3 -- Psycholinguistic Analysis (READY, awaiting transcript)")
    print("-" * 64)
    print("  Would run: PsycholinguisticAnalyzer.analyze(statements)")
    print("  Would produce: PsycholinguisticScore (8 dimensions + composite)")
    print("  Status: analyzer is built and tested -- needs the WhisperX feed above.")

    print("\nEnd-to-end path verified through compression. Add WhisperX to unlock scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
