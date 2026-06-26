"""Real-model WhisperX test. Gated: skipped unless run with `-m slow` AND
whisperx is importable. Never runs in the default suite."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _first_demo_video() -> Path:
    candidates = sorted((_REPO_ROOT / "demo_data").rglob("*.mp4"))
    if not candidates:
        pytest.skip("no demo video found under demo_data/")
    return candidates[0]


def test_whisperx_transcribes_real_audio(tmp_path):
    pytest.importorskip("whisperx", reason="whisperx not installed")

    from app.pipelines.transcription.backends import WhisperXBackend
    from backend.shared.schemas.transcription import TranscriptionConfig
    from backend.workers.app.compression.audio_extractor import AudioExtractor

    # Produce a real FLAC from a demo video via the existing extractor.
    flac_path, _opus = AudioExtractor().extract(_first_demo_video(), tmp_path)

    backend = WhisperXBackend(TranscriptionConfig(language="en", device="cpu"))
    transcript = backend.transcribe(flac_path)

    assert transcript.backend == "whisperx"
    assert transcript.language == "en"
    assert len(transcript.segments) >= 1
    # Timestamps must be monotonic and non-overlapping.
    for a, b in zip(transcript.segments, transcript.segments[1:]):
        assert a.start_seconds <= a.end_seconds
        assert a.end_seconds <= b.start_seconds + 1e-6
