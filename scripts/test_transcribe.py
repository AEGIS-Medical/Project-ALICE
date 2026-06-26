#!/usr/bin/env python
"""CLI: transcribe a FLAC/WAV file and print timestamped segments.

The transcription pipeline lives under ``backend/ml-inference/`` (hyphenated
service root), so we insert that root onto ``sys.path`` like
``tests/transcription/conftest.py`` does, then import ``from app.pipelines...``.

Usage:
    python scripts/test_transcribe.py path/to/audio.flac           # real WhisperX
    python scripts/test_transcribe.py path/to/audio.flac --fake    # canned, no models
    python scripts/test_transcribe.py path/to/audio.flac --model large-v3
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
    parser = argparse.ArgumentParser(description="Transcribe a FLAC/WAV file")
    parser.add_argument("audio", type=Path, help="Path to a .flac or .wav file")
    parser.add_argument("--fake", action="store_true", help="Use the fake backend (no models)")
    parser.add_argument("--model", default="distil-large-v3", help="Whisper model name")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"ERROR: file not found: {args.audio}", file=sys.stderr)
        return 1

    from app.pipelines.transcription.backends import FakeTranscriptionBackend
    from app.pipelines.transcription.transcriber import Transcriber

    if args.fake:
        backend = FakeTranscriptionBackend()
    else:
        from app.pipelines.transcription.backends import WhisperXBackend
        from backend.shared.schemas.transcription import TranscriptionConfig

        try:
            backend = WhisperXBackend(TranscriptionConfig(model_name=args.model))
        except Exception as exc:  # pragma: no cover - real-backend path
            print(f"ERROR: could not init WhisperX backend: {exc}", file=sys.stderr)
            return 1

    try:
        transcript = Transcriber(backend).transcribe(args.audio)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nTranscript ({transcript.backend}, {transcript.model_name}, "
          f"lang={transcript.language})")
    print("-" * 64)
    if not transcript.segments:
        print("  (no speech detected)")
    for seg in transcript.segments:
        print(f"  [{seg.start_seconds:7.2f} -> {seg.end_seconds:7.2f}]  {seg.text}")
    print("-" * 64)
    print(f"  Billable duration: {transcript.audio_duration_seconds:.2f}s  "
          f"| segments: {len(transcript.segments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
