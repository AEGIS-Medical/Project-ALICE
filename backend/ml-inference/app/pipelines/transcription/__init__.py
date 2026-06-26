"""Transcription pipeline package."""
from app.pipelines.transcription.backends import (
    FakeTranscriptionBackend,
    TranscriptionBackend,
)

__all__ = ["TranscriptionBackend", "FakeTranscriptionBackend"]
