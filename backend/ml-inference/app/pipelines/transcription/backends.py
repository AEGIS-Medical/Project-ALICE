"""Transcription backends for Project ALICE.

A ``TranscriptionBackend`` is the seam that isolates the heavy WhisperX/torch
dependency behind a single method. ``FakeTranscriptionBackend`` returns
deterministic canned output so the default test suite runs with no torch and no
model downloads. ``WhisperXBackend`` (added in Task 5) is the real engine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from backend.shared.schemas.transcription import (
    Transcript,
    TranscriptionConfig,
    TranscriptSegment,
)


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


class WhisperXBackend:
    """Real transcription backend: WhisperX with alignment ON, diarization OFF.

    WhisperX (torch + faster-whisper + wav2vec2) is lazy-imported on first
    ``transcribe`` call so importing this module never pulls in torch. If
    whisperx is not installed, ``transcribe`` raises a RuntimeError naming the
    install extra.

    Long audio is handled by WhisperX's built-in Silero VAD chunking; peak
    memory is bounded by ``config.batch_size`` x chunk, not the whole file.
    """

    def __init__(self, config: Optional[TranscriptionConfig] = None) -> None:
        self._config = config or TranscriptionConfig()

    def _resolve_device(self) -> tuple[str, str]:
        """Return (device, compute_type), resolving 'auto'."""
        device = self._config.device
        compute_type = self._config.compute_type
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda", "float16"
            except Exception:
                pass
            return "cpu", "int8"
        return device, compute_type

    def transcribe(self, flac_path: Path) -> Transcript:
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "whisperx is not installed. Install the transcription extra: "
                "pip install -e \".[transcription]\" (install torch from the "
                "appropriate index first on Windows; WSL/Linux is the fallback "
                "runner)."
            ) from exc

        device, compute_type = self._resolve_device()
        audio = whisperx.load_audio(str(flac_path))
        duration = float(len(audio)) / 16000.0  # whisperx resamples to 16 kHz

        model = whisperx.load_model(
            self._config.model_name,
            device,
            compute_type=compute_type,
            language=self._config.language,
        )
        result = model.transcribe(
            audio, batch_size=self._config.batch_size, language=self._config.language
        )
        language = result.get("language", self._config.language or "en")

        # Word-level alignment (ON). Diarization is intentionally NOT run.
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        aligned = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False,
        )

        segments = [
            TranscriptSegment(
                text=str(seg.get("text", "")).strip(),
                start_seconds=float(seg.get("start", 0.0)),
                end_seconds=float(seg.get("end", seg.get("start", 0.0))),
            )
            for seg in aligned.get("segments", [])
            if str(seg.get("text", "")).strip()
        ]

        return Transcript(
            segments=segments,
            language=language,
            audio_duration_seconds=duration,
            model_name=self._config.model_name,
            backend="whisperx",
        )
