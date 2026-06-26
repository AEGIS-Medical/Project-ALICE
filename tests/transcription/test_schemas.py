"""Schema tests for the transcription vector."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.shared.schemas.transcription import (
    Transcript,
    TranscriptionConfig,
    TranscriptSegment,
)


def _seg(text="hello", start=0.0, end=1.0, speaker=None) -> TranscriptSegment:
    return TranscriptSegment(
        text=text, start_seconds=start, end_seconds=end, speaker=speaker
    )


class TestTranscriptSegment:
    def test_valid_construction(self):
        s = _seg("I was there", 1.5, 3.0)
        assert s.text == "I was there"
        assert s.start_seconds == 1.5
        assert s.end_seconds == 3.0
        assert s.speaker is None

    def test_end_before_start_raises(self):
        with pytest.raises(ValidationError):
            _seg(start=5.0, end=2.0)

    def test_equal_start_end_ok(self):
        assert _seg(start=2.0, end=2.0).end_seconds == 2.0

    def test_negative_start_raises(self):
        with pytest.raises(ValidationError):
            _seg(start=-0.1, end=1.0)

    def test_is_frozen(self):
        s = _seg()
        with pytest.raises(ValidationError):
            s.text = "changed"

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            TranscriptSegment(
                text="x", start_seconds=0.0, end_seconds=1.0, bogus=1
            )

    def test_speaker_defaults_none_but_settable(self):
        assert _seg(speaker="SPEAKER_00").speaker == "SPEAKER_00"


class TestTranscriptionConfig:
    def test_defaults(self):
        c = TranscriptionConfig()
        assert c.model_name == "distil-large-v3"
        assert c.device == "auto"
        assert c.compute_type == "int8"
        assert c.batch_size == 16
        assert c.language is None
        assert c.vad_chunk_seconds == 30.0

    def test_override(self):
        c = TranscriptionConfig(model_name="large-v3", language="en")
        assert c.model_name == "large-v3"
        assert c.language == "en"

    def test_batch_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            TranscriptionConfig(batch_size=0)

    def test_is_frozen(self):
        c = TranscriptionConfig()
        with pytest.raises(ValidationError):
            c.model_name = "x"


class TestTranscript:
    def _transcript(self, segs=None) -> Transcript:
        segs = segs if segs is not None else [_seg("one", 0, 1), _seg("two", 1, 2)]
        return Transcript(
            segments=segs,
            language="en",
            audio_duration_seconds=2.0,
            model_name="distil-large-v3",
            backend="fake",
        )

    def test_statements_returns_texts_in_order(self):
        t = self._transcript([_seg("first", 0, 1), _seg("second", 1, 2)])
        assert t.statements() == ["first", "second"]

    def test_full_text_joins_with_spaces(self):
        t = self._transcript([_seg("hello", 0, 1), _seg("world", 1, 2)])
        assert t.full_text() == "hello world"

    def test_empty_segments_is_valid(self):
        t = self._transcript(segs=[])
        assert t.statements() == []
        assert t.full_text() == ""
        assert t.audio_duration_seconds == 2.0  # silence still billable

    def test_duration_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            Transcript(
                segments=[], language="en", audio_duration_seconds=-1.0,
                model_name="m", backend="fake",
            )

    def test_is_frozen(self):
        t = self._transcript()
        with pytest.raises(ValidationError):
            t.language = "fr"
