"""Schema tests for the CompressionResult landmark artifact validator.

Session 4 retired JSONL landmark streaming and replaced it outright with ALTM
protobuf (``.pb``).  ``.json`` and ``.jsonl`` are no longer accepted; ``.pb``
is the only valid suffix.  These tests pin that migration so any accidental
rollback is caught immediately.
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


def test_landmarks_pb_accepted():
    r = _result(
        landmarks_path=Path("out/landmarks/in_landmarks.pb"),
        landmarks_size_bytes=3,
    )
    assert r.landmarks_path.suffix == ".pb"


def test_landmarks_jsonl_now_rejected():
    """JSONL was the old streaming format — fully retired in Session 4."""
    with pytest.raises(ValidationError):
        _result(
            landmarks_path=Path("out/landmarks/in_landmarks.jsonl"),
            landmarks_size_bytes=3,
        )


def test_landmarks_json_now_rejected():
    """Legacy .json format — fully retired alongside .jsonl in Session 4."""
    with pytest.raises(ValidationError):
        _result(
            landmarks_path=Path("out/landmarks/in_landmarks.json"),
            landmarks_size_bytes=3,
        )


def test_landmarks_bad_suffix_rejected():
    with pytest.raises(ValidationError):
        _result(
            landmarks_path=Path("out/landmarks/in_landmarks.txt"),
            landmarks_size_bytes=3,
        )
