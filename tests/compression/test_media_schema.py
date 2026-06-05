"""Schema tests touched by the Phase 1-Bridge stories.

P1-S6 changes the landmark artifact extension to ``.jsonl``; the
``CompressionResult`` validator must accept both ``.json`` (legacy) and
``.jsonl`` (new) during the transition, and reject anything else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.shared.schemas.media import CompressionMode, CompressionResult


def _result(**overrides):
    base = dict(
        mode=CompressionMode.EDGE_MINIMAL,
        input_path=Path("in.mp4"),
        output_dir=Path("out"),
        input_size_bytes=10,
        flac_audio_path=Path("out/audio/in.flac"),
        flac_size_bytes=5,
        face_detected_pct=0.0,
    )
    base.update(overrides)
    return CompressionResult(**base)


def test_landmarks_jsonl_accepted():
    r = _result(
        landmarks_path=Path("out/landmarks/in_landmarks.jsonl"),
        landmarks_size_bytes=3,
    )
    assert r.landmarks_path.suffix == ".jsonl"


def test_landmarks_json_still_accepted():
    r = _result(
        landmarks_path=Path("out/landmarks/in_landmarks.json"),
        landmarks_size_bytes=3,
    )
    assert r.landmarks_path.suffix == ".json"


def test_landmarks_bad_suffix_rejected():
    with pytest.raises(ValidationError):
        _result(
            landmarks_path=Path("out/landmarks/in_landmarks.txt"),
            landmarks_size_bytes=3,
        )
