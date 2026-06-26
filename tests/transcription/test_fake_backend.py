"""Tests for FakeTranscriptionBackend (no torch, no downloads)."""
from __future__ import annotations

from app.pipelines.transcription.backends import FakeTranscriptionBackend
from backend.shared.schemas.transcription import Transcript, TranscriptSegment


def test_default_returns_three_segments(tmp_flac):
    t = FakeTranscriptionBackend().transcribe(tmp_flac)
    assert isinstance(t, Transcript)
    assert len(t.segments) == 3
    assert t.backend == "fake"
    assert t.language == "en"


def test_custom_segments_are_used(tmp_flac):
    segs = [TranscriptSegment(text="only one", start_seconds=0.0, end_seconds=1.0)]
    t = FakeTranscriptionBackend(segments=segs, audio_duration_seconds=1.0).transcribe(tmp_flac)
    assert t.statements() == ["only one"]
    assert t.audio_duration_seconds == 1.0


def test_empty_segments_allowed(tmp_flac):
    t = FakeTranscriptionBackend(segments=[], audio_duration_seconds=4.2).transcribe(tmp_flac)
    assert t.statements() == []
    assert t.audio_duration_seconds == 4.2


def test_is_deterministic(tmp_flac):
    a = FakeTranscriptionBackend().transcribe(tmp_flac)
    b = FakeTranscriptionBackend().transcribe(tmp_flac)
    assert a.statements() == b.statements()
