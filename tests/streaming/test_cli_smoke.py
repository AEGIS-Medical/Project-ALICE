"""Smoke test: replay_scores.py streams events from a transcript JSON."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from backend.shared.schemas.transcription import Transcript, TranscriptSegment

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "replay_scores.py"


def _write_transcript_json(path: Path) -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                text="I think I was at home that night.",
                start_seconds=0.0,
                end_seconds=2.4,
            ),
            TranscriptSegment(
                text="I never went anywhere near there.",
                start_seconds=2.4,
                end_seconds=4.1,
            ),
            TranscriptSegment(
                text="Honestly, you know, I'm not really sure.",
                start_seconds=4.1,
                end_seconds=6.0,
            ),
        ],
        language="en",
        audio_duration_seconds=6.0,
        model_name="fixture",
        backend="fake",
    )
    path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


def test_replays_transcript_json_instantly(tmp_path):
    tj = tmp_path / "transcript.json"
    _write_transcript_json(tj)
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--transcript", str(tj), "--pace", "0"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    interim_lines = [l for l in out.splitlines() if "interim" in l]
    final_lines = [l for l in out.splitlines() if "final" in l]
    assert len(interim_lines) >= 1
    assert len(final_lines) == 1
    assert "cumulative=" in interim_lines[0]
    # The standard dev-tool anomaly disclaimer (invariants #5/#6).
    assert "behavioral anomaly signal, not ground truth" in out
    assert "lie detector" not in out.lower()


def test_missing_transcript_file_exits_1(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--transcript",
            str(tmp_path / "nope.json"),
            "--pace",
            "0",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "not found" in result.stderr.lower()


def test_requires_exactly_one_input(tmp_path):
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--pace", "0"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_replay_cli_rejects_non_english_transcript(tmp_path):
    from backend.shared.schemas.transcription import Transcript, TranscriptSegment

    transcript = Transcript(
        segments=[
            TranscriptSegment(
                text="Hola, estaba en casa.", start_seconds=0.0, end_seconds=2.0
            )
        ],
        language="es",
        audio_duration_seconds=2.0,
        model_name="fake-distil",
        backend="fake",
    )
    p = tmp_path / "es_transcript.json"
    p.write_text(transcript.model_dump_json(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "replay_scores.py"),
            "--transcript", str(p), "--pace", "0",
        ],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "es" in result.stderr
    assert "not supported" in result.stderr
    # Invariant #3: no transcript text in the error surface.
    assert "Hola" not in result.stderr and "casa" not in result.stderr
