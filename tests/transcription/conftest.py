"""Fixtures for the transcription suite.

The transcription pipeline lives under ``backend/ml-inference/`` -- a service
root whose directory name contains a hyphen, so it cannot be imported via a
dotted ``backend.ml_inference`` path. We insert that root onto ``sys.path``
exactly as ``tests/psycholinguistic/conftest.py`` does, then import
``from app.pipelines.transcription...``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ML_INFERENCE_ROOT = (
    Path(__file__).resolve().parents[2] / "backend" / "ml-inference"
)
if str(_ML_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_INFERENCE_ROOT))


@pytest.fixture
def fake_backend():
    """A FakeTranscriptionBackend with its default canned segments."""
    from app.pipelines.transcription.backends import FakeTranscriptionBackend

    return FakeTranscriptionBackend()


@pytest.fixture
def tmp_flac(tmp_path: Path) -> Path:
    """A real (empty) .flac file path. Content is irrelevant to the fake backend
    and to extension-validation tests."""
    p = tmp_path / "clip.flac"
    p.write_bytes(b"")
    return p
