"""Fixtures for the live-service suite (sys.path bridge + fast configs)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ML_INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "backend" / "ml-inference"
if str(_ML_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_INFERENCE_ROOT))


@pytest.fixture
def fast_config():
    from app.service.config import LiveServiceConfig

    return LiveServiceConfig(session_ttl_seconds=0.2, reaper_interval_seconds=0.05)


@pytest.fixture
def transcript_file(tmp_path: Path):
    """Factory: write a small Transcript JSON, return its Path."""
    from backend.shared.schemas.transcription import Transcript, TranscriptSegment

    def _make(language: str = "en", texts: list[str] | None = None) -> Path:
        texts = texts or [
            "I think I was at home that night.",
            "I never went anywhere near there.",
            "Honestly, you know, I'm not really sure.",
        ]
        segments = [
            TranscriptSegment(
                text=t, start_seconds=2.0 * i, end_seconds=2.0 * i + 1.8
            )
            for i, t in enumerate(texts)
        ]
        transcript = Transcript(
            segments=segments,
            language=language,
            audio_duration_seconds=2.0 * len(texts),
            model_name="fake-distil",
            backend="fake",
        )
        p = tmp_path / f"transcript_{language}_{len(texts)}.json"
        p.write_text(transcript.model_dump_json(), encoding="utf-8")
        return p

    return _make
