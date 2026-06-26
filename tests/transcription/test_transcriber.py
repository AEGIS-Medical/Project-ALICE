"""Tests for the Transcriber facade (validation + delegation)."""
from __future__ import annotations

import pytest

from app.pipelines.transcription.backends import FakeTranscriptionBackend
from app.pipelines.transcription.transcriber import Transcriber


def test_transcribe_flac_delegates_to_backend(tmp_flac):
    t = Transcriber(FakeTranscriptionBackend()).transcribe(tmp_flac)
    assert t.backend == "fake"
    assert len(t.segments) == 3


def test_transcribe_wav_is_accepted(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"")
    t = Transcriber(FakeTranscriptionBackend()).transcribe(p)
    assert t.backend == "fake"


def test_rejects_mp3_naming_invariant_1(tmp_path):
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="INVARIANT #1"):
        Transcriber(FakeTranscriptionBackend()).transcribe(p)


def test_rejects_opus(tmp_path):
    p = tmp_path / "clip.opus"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="INVARIANT #1"):
        Transcriber(FakeTranscriptionBackend()).transcribe(p)


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        Transcriber(FakeTranscriptionBackend()).transcribe(tmp_path / "nope.flac")


def test_directory_input_rejected(tmp_path):
    with pytest.raises(ValueError):
        Transcriber(FakeTranscriptionBackend()).transcribe(tmp_path)


def test_empty_audio_returns_valid_empty_transcript(tmp_flac):
    backend = FakeTranscriptionBackend(segments=[], audio_duration_seconds=2.0)
    t = Transcriber(backend).transcribe(tmp_flac)
    assert t.statements() == []
    assert t.audio_duration_seconds == 2.0
