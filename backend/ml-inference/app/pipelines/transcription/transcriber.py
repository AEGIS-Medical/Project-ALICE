"""Transcriber facade: validate lossless input, delegate to a backend.

Enforces CLAUDE.md CRITICAL INVARIANT #1 (lossy audio must never reach an ML
model) before any backend runs. The transcriber is backend-agnostic: inject a
``WhisperXBackend`` in production or a ``FakeTranscriptionBackend`` in tests.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.pipelines.transcription.backends import TranscriptionBackend
from backend.shared.schemas.transcription import Transcript

logger = logging.getLogger(__name__)

# CLAUDE.md invariant #1: only lossless formats may feed an ML model.
LOSSLESS_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".flac", ".wav"})


class Transcriber:
    """Validate a FLAC/WAV path and delegate transcription to a backend.

    Args:
        backend: Any object satisfying the ``TranscriptionBackend`` protocol.
    """

    def __init__(self, backend: TranscriptionBackend) -> None:
        self._backend = backend

    def transcribe(self, flac_path: Path) -> Transcript:
        """Transcribe a lossless audio file.

        Args:
            flac_path: Path to a ``.flac`` or ``.wav`` file.

        Returns:
            A ``Transcript`` from the injected backend.

        Raises:
            FileNotFoundError: the path does not exist.
            ValueError: not a regular file, or a non-lossless extension
                (CLAUDE.md invariant #1).
        """
        flac_path = Path(flac_path)
        self._validate(flac_path)
        transcript = self._backend.transcribe(flac_path)
        # Invariant #3: log opaque facts only -- never transcript text.
        logger.info(
            "transcribed path=%s backend=%s model=%s segments=%d duration=%.2f",
            flac_path, transcript.backend, transcript.model_name,
            len(transcript.segments), transcript.audio_duration_seconds,
        )
        return transcript

    def _validate(self, flac_path: Path) -> None:
        if not flac_path.exists():
            raise FileNotFoundError(f"Audio file not found: {flac_path}")
        if not flac_path.is_file():
            raise ValueError(f"Not a regular file: {flac_path}")
        ext = flac_path.suffix.lower()
        if ext not in LOSSLESS_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Transcriber refuses {ext!r} input ({flac_path}). "
                f"CLAUDE.md CRITICAL INVARIANT #1: lossy audio must NEVER be fed "
                f"to an ML model. Accepted: {sorted(LOSSLESS_AUDIO_EXTENSIONS)}. "
                f"Re-extract via AudioExtractor (which always produces FLAC)."
            )
