#!/usr/bin/env python
"""Replay a recording's analysis as a timed ScoreEvent stream.

Feed it a serialized Transcript JSON (offline/dev) or a video (runs the
existing CompressionPipeline -> Transcriber path; --fake uses canned
segments so the full path runs offline). Events print one per line:

    [t=   5.0s] interim  cumulative= 48.2  recent= 61.0  (conf: low, stmts: 2)
    [t=   6.0s] final    cumulative= 47.9  recent=  --   (conf: low, stmts: 3)

Usage:
    python scripts/replay_scores.py --transcript path/to/transcript.json --pace 0
    python scripts/replay_scores.py --video path/to/clip.mp4 --fake --pace 2
    python scripts/replay_scores.py --transcript t.json --tick 5 --recent-window 30
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


def _format_event(event) -> str:
    recent = (
        f"{event.recent.composite_score:5.1f}"
        if event.recent is not None
        else "  -- "
    )
    return (
        f"[t={event.stream_time_seconds:6.1f}s] {event.kind.value:<8}"
        f"cumulative={event.cumulative.composite_score:5.1f}  "
        f"recent={recent}  "
        f"(conf: {event.confidence}, stmts: {event.statement_count_so_far})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay analysis as a timed ScoreEvent stream"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--transcript", type=Path, help="Path to a serialized Transcript JSON"
    )
    source.add_argument("--video", type=Path, help="Path to a video file")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="With --video: use the fake transcription backend (offline)",
    )
    parser.add_argument(
        "--mode",
        choices=["raw", "roi", "edge_full", "edge_minimal"],
        default="edge_full",
        help="With --video: compression mode (default: edge_full)",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help="Stream-seconds per wall-second; 0 = instant (default: 1.0)",
    )
    parser.add_argument(
        "--tick", type=float, default=5.0, help="Interim cadence in seconds"
    )
    parser.add_argument(
        "--recent-window",
        type=float,
        default=30.0,
        help="Trailing moment-detector window in seconds",
    )
    args = parser.parse_args()

    from backend.shared.schemas.score_event import StreamScorerConfig
    from backend.shared.schemas.transcription import Transcript

    if args.transcript is not None:
        if not args.transcript.exists():
            print(
                f"ERROR: transcript not found: {args.transcript}",
                file=sys.stderr,
            )
            return 1
        transcript = Transcript.model_validate_json(
            args.transcript.read_text(encoding="utf-8")
        )
    else:
        if not args.video.exists():
            print(f"ERROR: video not found: {args.video}", file=sys.stderr)
            return 1
        from backend.shared.schemas.media import CompressionMode
        from backend.workers.app.compression.pipeline import CompressionPipeline
        from app.pipelines.transcription.backends import (
            FakeTranscriptionBackend,
        )
        from app.pipelines.transcription.transcriber import Transcriber

        mode_map = {
            "raw": CompressionMode.RAW,
            "roi": CompressionMode.ROI_ENCODED,
            "edge_full": CompressionMode.EDGE_FULL,
            "edge_minimal": CompressionMode.EDGE_MINIMAL,
        }
        output_dir = _REPO_ROOT / "processed_output" / "replay_scores" / args.video.stem
        print(f"Compressing {args.video.name} ({args.mode}) ...")
        result = CompressionPipeline().process(
            args.video, output_dir, mode_map[args.mode]
        )
        if args.fake:
            backend = FakeTranscriptionBackend()
            print("Transcribing (fake backend, canned segments) ...")
        else:
            from app.pipelines.transcription.backends import WhisperXBackend

            backend = WhisperXBackend()
            print("Transcribing (WhisperX) ...")
        try:
            transcript = Transcriber(backend).transcribe(
                result.flac_audio_path
            )
        except Exception as exc:
            print(f"ERROR: transcription failed: {exc}", file=sys.stderr)
            print("Re-run with --fake to exercise the path offline.")
            return 1

    from app.pipelines.streaming import ScoreReplayer

    config = StreamScorerConfig(
        tick_seconds=args.tick, recent_window_seconds=args.recent_window
    )
    print(
        f"\nReplaying {len(transcript.segments)} statements "
        f"({transcript.audio_duration_seconds:.1f}s of audio) at pace "
        f"{args.pace:g} (tick {args.tick:g}s, window {args.recent_window:g}s)"
    )
    print("-" * 72)
    count = 0
    for event in ScoreReplayer(transcript, config).replay(pace=args.pace):
        print(_format_event(event), flush=True)
        count += 1
    if count == 0:
        print("(no speech -- empty stream; nothing to score)")
    print("-" * 72)
    print(
        "NOTE: behavioral anomaly signal, not ground truth. ~75% F1 ceiling; "
        "scores are deviations from baseline, developer-facing only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
