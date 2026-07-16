#!/usr/bin/env python
"""Integration: video -> compression -> transcription -> psycholinguistic score.

Runs the real pipeline end to end. With --fake, transcription uses canned
segments (no models) so the full path runs offline. Without --fake it uses the
real WhisperX backend; if WhisperX is not installed it prints a clear message
rather than crashing.

Usage:
    python scripts/test_compress_and_analyze.py path/to/video.mp4
    python scripts/test_compress_and_analyze.py path/to/video.mp4 --fake
    python scripts/test_compress_and_analyze.py path/to/video.mp4 --mode edge_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ML_INFERENCE_ROOT = _REPO_ROOT / "backend" / "ml-inference"
for _p in (_ML_INFERENCE_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compression -> transcription -> psycholinguistic")
    parser.add_argument("video", type=Path, help="Path to a video file")
    parser.add_argument(
        "--mode",
        choices=["raw", "roi", "edge_full", "edge_minimal"],
        default="edge_full",
        help="Compression mode (default: edge_full)",
    )
    parser.add_argument("--fake", action="store_true", help="Use the fake transcription backend")
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

    # ---- Step 1: compression ------------------------------------------------
    print(f"\nStep 1 -- Compression Pipeline ({mode.value})")
    print("-" * 64)
    result = CompressionPipeline().process(args.video, output_dir, mode)
    mb = 1_048_576
    print(f"  Input:           {result.input_size_bytes / mb:6.1f} MB")
    print(f"  FLAC audio (ML): {result.flac_size_bytes / mb:6.2f} MB  -> {result.flac_audio_path}")
    print(f"  Total time:      {result.processing_times.get('total', 0.0):5.1f}s")

    # ---- Step 2: transcription ---------------------------------------------
    print("\nStep 2 -- Transcription")
    print("-" * 64)
    from app.pipelines.transcription.backends import FakeTranscriptionBackend
    from app.pipelines.transcription.transcriber import Transcriber

    if args.fake:
        backend = FakeTranscriptionBackend()
        print("  backend: fake (canned segments, no models)")
    else:
        from app.pipelines.transcription.backends import WhisperXBackend

        try:
            backend = WhisperXBackend()
        except Exception as exc:  # pragma: no cover
            print(f"  WhisperX backend unavailable: {exc}")
            print("  Re-run with --fake to exercise the path offline.")
            return 1

    try:
        transcript = Transcriber(backend).transcribe(result.flac_audio_path)
    except Exception as exc:  # pragma: no cover - real-backend runtime errors
        print(f"  Transcription failed: {exc}")
        print("  Re-run with --fake to exercise the path offline.")
        return 1

    print(f"  backend={transcript.backend} segments={len(transcript.segments)} "
          f"duration={transcript.audio_duration_seconds:.2f}s")
    for seg in transcript.segments[:5]:
        print(f"    [{seg.start_seconds:6.2f} -> {seg.end_seconds:6.2f}] {seg.text}")

    # ---- Step 3: psycholinguistic analysis ---------------------------------
    print("\nStep 3 -- Psycholinguistic Analysis")
    print("-" * 64)
    statements = transcript.statements()
    if not statements:
        print("  No speech detected -- nothing to analyze.")
        return 0

    from app.pipelines.psycholinguistic.analyzer import (
        PsycholinguisticAnalyzer,
        UnsupportedLanguageError,
    )

    try:
        score = PsycholinguisticAnalyzer().analyze(
            statements, language=transcript.language
        )
    except UnsupportedLanguageError as exc:
        print(f"  Skipped: {exc}", file=sys.stderr)
        return 1
    print(f"  Statements analyzed: {score.statement_count}")
    print(f"  Composite score:     {score.composite_score:5.1f}/100  "
          f"(confidence: {score.confidence})")
    print()
    print("  NOTE: behavioral anomaly signal, not ground truth. ~75% F1 ceiling.")
    print("\nLive path complete: video -> FLAC -> transcript -> psycholinguistic score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
