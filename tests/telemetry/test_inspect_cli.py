"""Smoke test: inspect_landmarks.py decodes a .pb and prints stats."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from backend.shared.telemetry.landmark_codec import LandmarkEncoder

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_stream(path: Path) -> None:
    with LandmarkEncoder(path, source_fps=30.0) as enc:
        for i in range(40):
            if i == 5:
                enc.add_frame(i, None)
            else:
                enc.add_frame(i, [[0.5, 0.5, 0.0]] * 478)


def test_inspect_prints_stats_and_head(tmp_path):
    pb_path = tmp_path / "clip_landmarks.pb"
    _make_stream(pb_path)
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "inspect_landmarks.py"),
         str(pb_path), "--head", "2"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "version" in out
    assert "frames:" in out
    assert "face coverage:" in out
    assert "KB/min" in out
    assert '"frame_number": 0' in out  # --head dump


def test_inspect_rejects_non_telemetry_file(tmp_path):
    bogus = tmp_path / "x.pb"
    bogus.write_bytes(b"XXXX not a telemetry file")
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "inspect_landmarks.py"),
         str(bogus)],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "not an ALICE landmark telemetry file" in result.stderr
