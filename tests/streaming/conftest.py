"""Fixtures for the streaming suite.

The streaming pipeline lives under ``backend/ml-inference/`` -- a service
root whose directory name contains a hyphen, so it cannot be imported via a
dotted ``backend.ml_inference`` path. We insert that root onto ``sys.path``
exactly as ``tests/transcription/conftest.py`` does, then import
``from app.pipelines.streaming...``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from backend.shared.schemas.psycholinguistic import (
    PsycholinguisticDimension,
    PsycholinguisticScore,
)
from backend.shared.schemas.transcription import Transcript, TranscriptSegment

_ML_INFERENCE_ROOT = (
    Path(__file__).resolve().parents[2] / "backend" / "ml-inference"
)
if str(_ML_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_INFERENCE_ROOT))


def make_pscore(
    composite: float = 50.0, statement_count: int = 3
) -> PsycholinguisticScore:
    """A valid frozen score with all eight dimensions at ``composite``."""
    dim = PsycholinguisticDimension(score=composite, evidence=[])
    return PsycholinguisticScore(
        pronoun_shift_score=dim,
        hedging_score=dim,
        cognitive_complexity_score=dim,
        emotional_distribution_score=dim,
        disfluency_score=dim,
        negation_score=dim,
        detail_specificity_score=dim,
        certainty_score=dim,
        composite_score=composite,
        statement_count=statement_count,
        baseline_available=False,
        confidence="low",
    )


def make_transcript(spec: list[tuple[str, float, float]]) -> Transcript:
    """Build an ordered Transcript from (text, start_seconds, end_seconds)."""
    segments = [
        TranscriptSegment(text=t, start_seconds=s, end_seconds=e)
        for t, s, e in spec
    ]
    duration = max((e for _, _, e in spec), default=0.0)
    return Transcript(
        segments=segments,
        language="en",
        audio_duration_seconds=duration,
        model_name="fixture",
        backend="fake",
    )
