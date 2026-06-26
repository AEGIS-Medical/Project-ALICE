"""Transcription schemas for Project ALICE (Session 3).

Defines the typed contract produced by the transcription vector
(``backend/ml-inference/app/pipelines/transcription``). The transcriber turns
the compression pipeline's lossless FLAC into a ``Transcript`` -- an ordered
list of timestamped ``TranscriptSegment`` -- whose ``statements()`` feed
``PsycholinguisticAnalyzer.analyze`` directly.

Design notes (see docs/superpowers/specs/2026-06-25-whisperx-transcription-vector-design.md):
  - One WhisperX segment == one statement. The analyzer reassembles all
    statements into a single document before parsing, so segmentation never
    costs linguistic context.
  - ``TranscriptSegment.speaker`` is reserved for the deferred pyannote
    diarization step; it is always None in this session.
  - ``Transcript.audio_duration_seconds`` is the billable / meterable unit
    (the product charges by content length).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranscriptSegment(BaseModel):
    """A single timestamped transcript segment (one statement)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(description="Transcribed text for this segment.")
    start_seconds: float = Field(ge=0.0, description="Segment start (word-aligned).")
    end_seconds: float = Field(ge=0.0, description="Segment end (word-aligned).")
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker label; populated later by pyannote diarization.",
    )

    @model_validator(mode="after")
    def _end_after_start(self) -> "TranscriptSegment":
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                f"end_seconds ({self.end_seconds}) < start_seconds "
                f"({self.start_seconds})"
            )
        return self


class TranscriptionConfig(BaseModel):
    """Tunable parameters for the WhisperX backend.

    Defaults favor throughput (``distil-large-v3``); callers selling premium
    accuracy set ``model_name='large-v3'``. ``device='auto'`` picks cuda when
    available else cpu, and ``compute_type`` is chosen to match (int8 on CPU,
    float16 on GPU).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(
        default="distil-large-v3",
        description="Whisper model: 'distil-large-v3' (fast) or 'large-v3' (accurate).",
    )
    device: str = Field(
        default="auto",
        description="'auto' selects cuda if available else cpu; or pin 'cpu'/'cuda'.",
    )
    compute_type: str = Field(
        default="int8",
        description="ctranslate2 compute type. int8 on CPU, float16 on GPU.",
    )
    batch_size: int = Field(
        default=16, ge=1, description="WhisperX batched inference size."
    )
    language: Optional[str] = Field(
        default=None,
        description="None autodetects; pin e.g. 'en' to skip language detection.",
    )
    vad_chunk_seconds: float = Field(
        default=30.0, gt=0.0, description="VAD speech-chunk window for long audio."
    )


class Transcript(BaseModel):
    """Ordered, timestamped transcript plus provenance and billing metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segments: list[TranscriptSegment]
    language: str = Field(description="Detected (or pinned) language code, e.g. 'en'.")
    audio_duration_seconds: float = Field(
        ge=0.0, description="Source audio length -- the billable / meterable unit."
    )
    model_name: str = Field(description="Model that produced this transcript.")
    backend: str = Field(description="Provenance: 'whisperx' | 'fake'.")

    def statements(self) -> list[str]:
        """Segment texts in order -- the exact input ``analyze()`` expects."""
        return [s.text for s in self.segments]

    def full_text(self) -> str:
        """All segment texts joined by single spaces."""
        return " ".join(s.text for s in self.segments)
