"""Smoke test: the psycholinguistic CLI runs and prints a composite score."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_prints_composite():
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "test_psycholinguistic.py"),
            "--text",
            "I think maybe I never did that, um, I guess so.",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "COMPOSITE" in result.stdout
    # The eight dimension labels must all appear.
    for label in (
        "Pronoun Shift", "Hedging", "Cognitive Complexity",
        "Emotional Distribution", "Disfluency", "Negation",
        "Detail Specificity", "Certainty",
    ):
        assert label in result.stdout, f"missing dimension: {label}"
