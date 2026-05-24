"""CLI smoke test for the compression pipeline.

Usage:
    python scripts/test_compression.py path/to/video.mp4
    python scripts/test_compression.py path/to/video.mp4 --mode roi
    python scripts/test_compression.py path/to/video.mp4 --mode edge_full -v

Outputs land in ``processed_output/compression_test/`` under the project
root. The script prints a summary table of artifact sizes, compression
ratios relative to the source, and per-stage wall-clock times, plus the
face-detection rate sourced from whichever stage produced it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Make the project root importable when the script is run directly
# (`python scripts/test_compression.py ...`) without an editable install.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.shared.schemas.media import CompressionMode, CompressionResult
from backend.workers.app.compression import CompressionPipeline


# CLI mode strings -> CompressionMode enum. Kept small and explicit so the
# user-facing string set is independent of the enum's internal values.
_MODE_CHOICES: dict[str, CompressionMode] = {
    "raw": CompressionMode.RAW,
    "roi": CompressionMode.ROI_ENCODED,
    "edge_full": CompressionMode.EDGE_FULL,
    "edge_minimal": CompressionMode.EDGE_MINIMAL,
}

_OUTPUT_SUBDIR = Path("processed_output") / "compression_test"


# ---- Formatting helpers ----------------------------------------------------


def _fmt_size(num_bytes: Optional[int]) -> str:
    """Human-readable byte size. Returns '—' for None."""
    if num_bytes is None:
        return "—"
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _fmt_ratio(ratio: Optional[float]) -> str:
    return f"{ratio:.2f}x" if ratio is not None else "—"


def _fmt_time(seconds: Optional[float]) -> str:
    return f"{seconds:.1f}s" if seconds is not None else "—"


def _print_table(rows: list[tuple[str, str, str, str]]) -> None:
    """Pretty-print a fixed-column table.

    Columns are sized to the widest cell in each. The header is the first
    row passed in.
    """
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    sep = "-+-".join("-" * w for w in widths)
    for i, row in enumerate(rows):
        line = " | ".join(cell.ljust(widths[j]) for j, cell in enumerate(row))
        print(line)
        if i == 0:
            print(sep)


# ---- Row builders ---------------------------------------------------------


def _stage_status(
    mode: CompressionMode,
    stage: str,
    path_present: bool,
) -> str:
    """Return '(skipped)' if the mode does not run this stage, '(failed)'
    if it should have run but produced no path, or '' if it produced output.

    Mirrors the dispatch rules in CompressionPipeline so the CLI can
    distinguish "didn't try" from "tried and failed" without needing the
    pipeline to surface an extra status field.
    """
    runs = {
        "roi_video": mode in (CompressionMode.RAW, CompressionMode.ROI_ENCODED),
        "landmarks": mode in (CompressionMode.EDGE_FULL, CompressionMode.EDGE_MINIMAL),
        "audio_features": mode != CompressionMode.EDGE_MINIMAL,
    }[stage]
    if not runs:
        return "(skipped)"
    if not path_present:
        return "(failed)"
    return ""


def _build_rows(
    input_path: Path,
    result: CompressionResult,
    pipeline: CompressionPipeline,
) -> list[tuple[str, str, str, str]]:
    """Build the table rows from the pipeline result + per-stage telemetry."""
    input_size = result.input_size_bytes
    times = result.processing_times
    ratios = result.compression_ratios

    # Opus path is created by AudioExtractor but not tracked in
    # CompressionResult; reconstruct it from the known convention so the
    # table can show it as a separate line.
    audio_dir = result.output_dir / "audio"
    opus_path = audio_dir / f"{input_path.stem}.opus"
    opus_size = opus_path.stat().st_size if opus_path.exists() else None
    opus_ratio = (opus_size / input_size) if (opus_size and input_size) else None

    # Per-format audio timings come from AudioExtractor's telemetry --
    # CompressionResult only carries the combined ``audio_extract`` time.
    flac_seconds = pipeline.audio_extractor.last_flac_seconds or None
    opus_seconds = pipeline.audio_extractor.last_opus_seconds or None

    def status_or_size(stage: str, size: Optional[int]) -> str:
        status = _stage_status(result.mode, stage, size is not None)
        return status if status else _fmt_size(size)

    def status_or_ratio(stage: str, ratio: Optional[float]) -> str:
        status = _stage_status(result.mode, stage, ratio is not None)
        return "" if status else _fmt_ratio(ratio)

    def status_or_time(stage: str, seconds: Optional[float]) -> str:
        status = _stage_status(result.mode, stage, seconds is not None)
        return "" if status else _fmt_time(seconds)

    return [
        ("Component",      "Size",                                "Ratio",                              "Time"),
        ("Original Video", _fmt_size(input_size),                  "1.00x",                              "—"),
        ("FLAC Audio",     _fmt_size(result.flac_size_bytes),      _fmt_ratio(ratios.get("audio")),      _fmt_time(flac_seconds)),
        ("Opus Playback",  _fmt_size(opus_size),                   _fmt_ratio(opus_ratio),               _fmt_time(opus_seconds)),
        ("ROI Video",      status_or_size("roi_video",      result.roi_video_size_bytes),
                           status_or_ratio("roi_video",     ratios.get("video")),
                           status_or_time("roi_video",      times.get("roi_encode"))),
        ("Landmarks JSON", status_or_size("landmarks",      result.landmarks_size_bytes),
                           "",
                           status_or_time("landmarks",      times.get("landmarks_extract"))),
        ("Audio Features", status_or_size("audio_features", result.features_size_bytes),
                           "",
                           status_or_time("audio_features", times.get("audio_features"))),
    ]


# ---- Entry point ----------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ALICE compression pipeline against a video file.",
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to a source video file.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(_MODE_CHOICES),
        default="raw",
        help="Compression tier to run (default: raw).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show INFO-level pipeline logs (otherwise WARNING and up).",
    )
    args = parser.parse_args(argv)

    # Force UTF-8 on stdout so the em-dash placeholders render on Windows
    # consoles that default to cp1252. Best-effort: older Python builds
    # without reconfigure() get the plain ASCII fallback already used
    # elsewhere in the table.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Default to WARNING so the table is the primary output. -v opens the
    # firehose for debugging slow / failed runs.
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.video_path.exists():
        print(f"error: input not found: {args.video_path}", file=sys.stderr)
        return 2

    mode = _MODE_CHOICES[args.mode]
    output_dir = _PROJECT_ROOT / _OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Compression Pipeline Test")
    print("=" * 60)
    print(f"Input:  {args.video_path}")
    print(f"Mode:   {args.mode}")
    print(f"Output: {output_dir}")
    print()
    print("Running pipeline...")
    print()

    pipeline = CompressionPipeline()
    try:
        result = pipeline.process(args.video_path, output_dir, mode=mode)
    except Exception as exc:
        print(f"error: pipeline failed: {exc}", file=sys.stderr)
        logging.exception("pipeline_failed")
        return 1

    print("=" * 60)
    print("Results")
    print("=" * 60)
    rows = _build_rows(args.video_path, result, pipeline)
    _print_table(rows)
    print()
    total_seconds = result.processing_times.get("total", 0.0)
    print(f"Total pipeline time: {total_seconds:.1f}s")
    print()
    print(f"Face detected in {result.face_detected_pct:.0f}% of frames")
    print()

    print("Output files:")
    if result.flac_audio_path:
        print(f"  FLAC:        {result.flac_audio_path}")
    opus_path = result.output_dir / "audio" / f"{args.video_path.stem}.opus"
    if opus_path.exists():
        print(f"  Opus:        {opus_path}")
    if result.roi_video_path:
        print(f"  ROI Video:   {result.roi_video_path}")
    if result.landmarks_path:
        print(f"  Landmarks:   {result.landmarks_path}")
    if result.features_path:
        print(f"  Features:    {result.features_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
