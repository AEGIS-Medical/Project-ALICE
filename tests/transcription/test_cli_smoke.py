"""Smoke test: the transcribe CLI runs in --fake mode and prints segments."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_fake_mode_prints_transcript(tmp_path):
    flac = tmp_path / "clip.flac"
    flac.write_bytes(b"")
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "test_transcribe.py"),
            str(flac),
            "--fake",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Transcript" in result.stdout
    assert "Billable duration" in result.stdout
    assert "[" in result.stdout  # at least one [start -> end] timestamp line
