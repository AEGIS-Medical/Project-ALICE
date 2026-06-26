"""Bridge test: Transcript.statements() feeds PsycholinguisticAnalyzer.analyze()."""
from __future__ import annotations

import pytest

from app.pipelines.psycholinguistic.analyzer import PsycholinguisticAnalyzer
from app.pipelines.transcription.backends import FakeTranscriptionBackend
from app.pipelines.transcription.transcriber import Transcriber
from backend.shared.schemas.psycholinguistic import PsycholinguisticScore


def test_transcript_statements_score_through_analyzer(tmp_flac):
    transcript = Transcriber(FakeTranscriptionBackend()).transcribe(tmp_flac)
    result = PsycholinguisticAnalyzer().analyze(transcript.statements())
    assert isinstance(result, PsycholinguisticScore)
    assert result.statement_count == 3
    assert 0.0 <= result.composite_score <= 100.0


def test_empty_transcript_makes_analyzer_raise(tmp_flac):
    backend = FakeTranscriptionBackend(segments=[], audio_duration_seconds=1.0)
    transcript = Transcriber(backend).transcribe(tmp_flac)
    # Silence is a valid transcript; the analyzer is what rejects zero statements.
    with pytest.raises(ValueError, match="No statements provided"):
        PsycholinguisticAnalyzer().analyze(transcript.statements())
