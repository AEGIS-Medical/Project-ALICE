"""Transcription backends for Project ALICE.

A ``TranscriptionBackend`` is the seam that isolates the heavy WhisperX/torch
dependency behind a single method. ``FakeTranscriptionBackend`` returns
deterministic canned output so the default test suite runs with no torch and no
model downloads. ``WhisperXBackend`` (added in Task 5) is the real engine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from backend.shared.schemas.transcription import Transcript, TranscriptSegment


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Anything that can turn a FLAC path into a Transcript."""

    def transcribe(self, flac_path: Path) -> Transcript:
        ...


# Default canned segments for the fake backend -- a short, deception-flavored
# snippet so downstream analyzer tests get non-trivial linguistic signal.
_DEFAULT_FAKE_SEGMENTS: tuple[TranscriptSegment, ...] = (
    TranscriptSegment(text="I think I was at home that night.",
                      start_seconds=0.0, end_seconds=2.4),
    TranscriptSegment(text="I never went anywhere near there.",
                      start_seconds=2.4, end_seconds=4.1),
    TranscriptSegment(text="Honestly, you know, I'm not really sure.",
                      start_seconds=4.1, end_seconds=6.0),
)


class FakeTranscriptionBackend:
    """Deterministic in-memory backend for tests and offline smoke runs.

    Args:
        segments: Segments to return. Defaults to a 3-segment canned snippet.
            Pass ``[]`` to simulate silent audio.
        language: Language code to report.
        audio_duration_seconds: Billable duration to report.
        model_name: Model name to record in the Transcript.
    """

    def __init__(
        self,
        segments: Optional[list[TranscriptSegment]] = None,
        language: str = "en",
        audio_duration_seconds: float = 6.0,
        model_name: str = "fake-distil",
    ) -> None:
        self._segments = (
            list(_DEFAULT_FAKE_SEGMENTS) if segments is None else list(segments)
        )
        self._language = language
        self._audio_duration_seconds = audio_duration_seconds
        self._model_name = model_name

    def transcribe(self, flac_path: Path) -> Transcript:
        return Transcript(
            segments=list(self._segments),
            language=self._language,
            audio_duration_seconds=self._audio_duration_seconds,
            model_name=self._model_name,
            backend="fake",
        )
