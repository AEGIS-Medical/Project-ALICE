# WhisperX Transcription Vector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a WhisperX-based transcription vector that turns the compression pipeline's FLAC output into timestamped statements feeding `PsycholinguisticAnalyzer.analyze()`, converting `scripts/test_compress_and_analyze.py` from a stub into a live video→FLAC→transcript→score path.

**Architecture:** A `Transcriber` facade validates lossless input and delegates to a swappable `TranscriptionBackend` protocol. `WhisperXBackend` is the real, lazy-loaded engine (alignment ON, diarization OFF); `FakeTranscriptionBackend` returns deterministic canned segments so the default test suite needs no torch and no downloads. Frozen Pydantic schemas (`TranscriptSegment`, `Transcript`, `TranscriptionConfig`) carry the contract; `Transcript.statements()` hands `list[str]` straight to the analyzer.

**Tech Stack:** Python 3.13, Pydantic v2 (frozen models), WhisperX (faster-whisper + wav2vec2 alignment), pytest. WhisperX is quarantined behind the backend protocol and lazy-imported.

## Global Constraints

- **Python `>=3.12`** (project floor; runtime venv is 3.13.9 at `.venv/`).
- **CLAUDE.md Invariant #1 — lossless audio only.** The transcriber MUST reject any input that is not `.flac`/`.wav` with a `ValueError` naming invariant #1. WhisperX never sees lossy audio.
- **CLAUDE.md Invariant #3 — never log PII.** Log opaque facts only (path, duration, segment count, model, timing). NEVER log transcript text.
- **CLAUDE.md Invariant #6 — never use the phrase "lie detector"** anywhere in code, comments, or output.
- **Hyphenated service root.** Code lives under `backend/ml-inference/` (hyphen). It cannot be imported via a dotted path; tests and scripts insert `backend/ml-inference` onto `sys.path` and import `from app.pipelines.transcription...` — exactly as `tests/psycholinguistic/conftest.py` does. `app/` and `app/pipelines/` are namespace packages (no `__init__.py`); do NOT add `__init__.py` to them.
- **`backend` is importable everywhere** because the project is installed editable (`pip install -e ".[dev]"`), so `from backend.shared.schemas... import ...` works without a path hack.
- **Frozen schemas.** All Pydantic models use `ConfigDict(frozen=True, extra="forbid")`, matching `media.py` and `psycholinguistic.py`.
- **Run all commands from** `C:\Users\ryanh\ALICE\Project-ALICE` **using** `.venv/Scripts/python`.

---

## File Map

| File | Action | Task |
|---|---|---|
| `backend/shared/schemas/transcription.py` | Create — schemas | Task 1 |
| `tests/transcription/__init__.py` | Create (empty) | Task 1 |
| `tests/transcription/test_schemas.py` | Create | Task 1 |
| `backend/ml-inference/app/pipelines/transcription/__init__.py` | Create — re-exports | Task 2 |
| `backend/ml-inference/app/pipelines/transcription/backends.py` | Create — protocol + Fake | Task 2 |
| `tests/transcription/conftest.py` | Create — sys.path bridge + fixtures | Task 2 |
| `tests/transcription/test_fake_backend.py` | Create | Task 2 |
| `backend/ml-inference/app/pipelines/transcription/transcriber.py` | Create — facade | Task 3 |
| `tests/transcription/test_transcriber.py` | Create | Task 3 |
| `tests/transcription/test_bridge.py` | Create — transcript→analyzer | Task 4 |
| `backend/ml-inference/app/pipelines/transcription/backends.py` | Modify — add WhisperXBackend | Task 5 |
| `pyproject.toml` | Modify — register `slow` marker, add optional `transcription` deps | Task 5 |
| `tests/transcription/test_whisperx_backend.py` | Create — gated slow test | Task 5 |
| `scripts/test_transcribe.py` | Create — CLI | Task 6 |
| `tests/transcription/test_cli_smoke.py` | Create | Task 6 |
| `scripts/test_compress_and_analyze.py` | Modify — replace stub | Task 7 |

---

## Task 1: Transcription Schemas

**Files:**
- Create: `backend/shared/schemas/transcription.py`
- Create: `tests/transcription/__init__.py` (empty)
- Test: `tests/transcription/test_schemas.py`

**Interfaces:**
- Produces:
  - `TranscriptSegment(text: str, start_seconds: float, end_seconds: float, speaker: Optional[str] = None)` — frozen; validates `end_seconds >= start_seconds`.
  - `TranscriptionConfig(model_name="distil-large-v3", device="auto", compute_type="int8", batch_size=16, language: Optional[str]=None, vad_chunk_seconds=30.0)` — frozen.
  - `Transcript(segments: list[TranscriptSegment], language: str, audio_duration_seconds: float, model_name: str, backend: str)` — frozen; methods `statements() -> list[str]`, `full_text() -> str`.

- [ ] **Step 1: Create the empty test package init**

```bash
cd C:\Users\ryanh\ALICE\Project-ALICE
mkdir -p tests/transcription
printf '' > tests/transcription/__init__.py
```

- [ ] **Step 2: Write the failing schema tests**

Create `tests/transcription/test_schemas.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
.venv/Scripts/python -m pytest tests/transcription/test_schemas.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.shared.schemas.transcription'`.

- [ ] **Step 4: Create the schema module**

Create `backend/shared/schemas/transcription.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/Scripts/python -m pytest tests/transcription/test_schemas.py -q
```
Expected: PASS (all schema tests green).

- [ ] **Step 6: Commit**

```bash
git add backend/shared/schemas/transcription.py tests/transcription/__init__.py tests/transcription/test_schemas.py
git commit -m "feat(transcription): add TranscriptSegment, Transcript, TranscriptionConfig schemas"
```

---

## Task 2: Backend Protocol + FakeTranscriptionBackend

**Files:**
- Create: `backend/ml-inference/app/pipelines/transcription/__init__.py`
- Create: `backend/ml-inference/app/pipelines/transcription/backends.py`
- Create: `tests/transcription/conftest.py`
- Test: `tests/transcription/test_fake_backend.py`

**Interfaces:**
- Consumes: `Transcript`, `TranscriptSegment` (Task 1).
- Produces:
  - `TranscriptionBackend` — `typing.Protocol` with `transcribe(self, flac_path: Path) -> Transcript`.
  - `FakeTranscriptionBackend(segments: Optional[list[TranscriptSegment]] = None, language: str = "en", audio_duration_seconds: float = 3.0, model_name: str = "fake-distil")` — `.transcribe(flac_path)` returns a `Transcript` with `backend="fake"`; default is 3 canned segments.
  - `__init__.py` re-exports `TranscriptionBackend`, `FakeTranscriptionBackend`.
- `conftest.py` provides: `sys.path` insert of `backend/ml-inference`; fixtures `fake_backend` and `tmp_flac` (a real empty `.flac` file path created under `tmp_path`).

- [ ] **Step 1: Create the conftest sys.path bridge and fixtures**

Create `tests/transcription/conftest.py`:

```python
"""Fixtures for the transcription suite.

The transcription pipeline lives under ``backend/ml-inference/`` -- a service
root whose directory name contains a hyphen, so it cannot be imported via a
dotted ``backend.ml_inference`` path. We insert that root onto ``sys.path``
exactly as ``tests/psycholinguistic/conftest.py`` does, then import
``from app.pipelines.transcription...``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ML_INFERENCE_ROOT = (
    Path(__file__).resolve().parents[2] / "backend" / "ml-inference"
)
if str(_ML_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_INFERENCE_ROOT))


@pytest.fixture
def fake_backend():
    """A FakeTranscriptionBackend with its default canned segments."""
    from app.pipelines.transcription.backends import FakeTranscriptionBackend

    return FakeTranscriptionBackend()


@pytest.fixture
def tmp_flac(tmp_path: Path) -> Path:
    """A real (empty) .flac file path. Content is irrelevant to the fake backend
    and to extension-validation tests."""
    p = tmp_path / "clip.flac"
    p.write_bytes(b"")
    return p
```

- [ ] **Step 2: Write the failing fake-backend tests**

Create `tests/transcription/test_fake_backend.py`:

```python
"""Tests for FakeTranscriptionBackend (no torch, no downloads)."""
from __future__ import annotations

from app.pipelines.transcription.backends import FakeTranscriptionBackend
from backend.shared.schemas.transcription import Transcript, TranscriptSegment


def test_default_returns_three_segments(tmp_flac):
    t = FakeTranscriptionBackend().transcribe(tmp_flac)
    assert isinstance(t, Transcript)
    assert len(t.segments) == 3
    assert t.backend == "fake"
    assert t.language == "en"


def test_custom_segments_are_used(tmp_flac):
    segs = [TranscriptSegment(text="only one", start_seconds=0.0, end_seconds=1.0)]
    t = FakeTranscriptionBackend(segments=segs, audio_duration_seconds=1.0).transcribe(tmp_flac)
    assert t.statements() == ["only one"]
    assert t.audio_duration_seconds == 1.0


def test_empty_segments_allowed(tmp_flac):
    t = FakeTranscriptionBackend(segments=[], audio_duration_seconds=4.2).transcribe(tmp_flac)
    assert t.statements() == []
    assert t.audio_duration_seconds == 4.2


def test_is_deterministic(tmp_flac):
    a = FakeTranscriptionBackend().transcribe(tmp_flac)
    b = FakeTranscriptionBackend().transcribe(tmp_flac)
    assert a.statements() == b.statements()
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
.venv/Scripts/python -m pytest tests/transcription/test_fake_backend.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipelines.transcription'`.

- [ ] **Step 4: Create the transcription package + backends module (protocol + fake)**

```bash
mkdir -p backend/ml-inference/app/pipelines/transcription
```

Create `backend/ml-inference/app/pipelines/transcription/backends.py`:

```python
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
```

Create `backend/ml-inference/app/pipelines/transcription/__init__.py`:

```python
"""Transcription pipeline package."""
from app.pipelines.transcription.backends import (
    FakeTranscriptionBackend,
    TranscriptionBackend,
)

__all__ = ["TranscriptionBackend", "FakeTranscriptionBackend"]
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/Scripts/python -m pytest tests/transcription/test_fake_backend.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/ml-inference/app/pipelines/transcription/__init__.py backend/ml-inference/app/pipelines/transcription/backends.py tests/transcription/conftest.py tests/transcription/test_fake_backend.py
git commit -m "feat(transcription): add TranscriptionBackend protocol + FakeTranscriptionBackend"
```

---

## Task 3: Transcriber Facade (FLAC validation + delegation)

**Files:**
- Create: `backend/ml-inference/app/pipelines/transcription/transcriber.py`
- Test: `tests/transcription/test_transcriber.py`

**Interfaces:**
- Consumes: `TranscriptionBackend`, `FakeTranscriptionBackend` (Task 2); `Transcript` (Task 1).
- Produces:
  - `Transcriber(backend: TranscriptionBackend)` — `.transcribe(flac_path: Path) -> Transcript`. Validates the input exists, is a regular file, and has a `.flac`/`.wav` extension (else `ValueError` naming invariant #1 / `FileNotFoundError`), then delegates to the backend.
  - Module constant `LOSSLESS_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".flac", ".wav"})`.
- Re-exported from `__init__.py` (update in Step 4).

- [ ] **Step 1: Write the failing transcriber tests**

Create `tests/transcription/test_transcriber.py`:

```python
"""Tests for the Transcriber facade (validation + delegation)."""
from __future__ import annotations

import pytest

from app.pipelines.transcription.backends import FakeTranscriptionBackend
from app.pipelines.transcription.transcriber import Transcriber


def test_transcribe_flac_delegates_to_backend(tmp_flac):
    t = Transcriber(FakeTranscriptionBackend()).transcribe(tmp_flac)
    assert t.backend == "fake"
    assert len(t.segments) == 3


def test_transcribe_wav_is_accepted(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"")
    t = Transcriber(FakeTranscriptionBackend()).transcribe(p)
    assert t.backend == "fake"


def test_rejects_mp3_naming_invariant_1(tmp_path):
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="INVARIANT #1"):
        Transcriber(FakeTranscriptionBackend()).transcribe(p)


def test_rejects_opus(tmp_path):
    p = tmp_path / "clip.opus"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="INVARIANT #1"):
        Transcriber(FakeTranscriptionBackend()).transcribe(p)


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        Transcriber(FakeTranscriptionBackend()).transcribe(tmp_path / "nope.flac")


def test_directory_input_rejected(tmp_path):
    with pytest.raises(ValueError):
        Transcriber(FakeTranscriptionBackend()).transcribe(tmp_path)


def test_empty_audio_returns_valid_empty_transcript(tmp_flac):
    backend = FakeTranscriptionBackend(segments=[], audio_duration_seconds=2.0)
    t = Transcriber(backend).transcribe(tmp_flac)
    assert t.statements() == []
    assert t.audio_duration_seconds == 2.0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/Scripts/python -m pytest tests/transcription/test_transcriber.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipelines.transcription.transcriber'`.

- [ ] **Step 3: Create the transcriber module**

Create `backend/ml-inference/app/pipelines/transcription/transcriber.py`:

```python
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
```

- [ ] **Step 4: Re-export `Transcriber` from the package init**

Replace the contents of `backend/ml-inference/app/pipelines/transcription/__init__.py` with:

```python
"""Transcription pipeline package."""
from app.pipelines.transcription.backends import (
    FakeTranscriptionBackend,
    TranscriptionBackend,
)
from app.pipelines.transcription.transcriber import Transcriber

__all__ = ["TranscriptionBackend", "FakeTranscriptionBackend", "Transcriber"]
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/Scripts/python -m pytest tests/transcription/test_transcriber.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/ml-inference/app/pipelines/transcription/transcriber.py backend/ml-inference/app/pipelines/transcription/__init__.py tests/transcription/test_transcriber.py
git commit -m "feat(transcription): add Transcriber facade with FLAC-only invariant enforcement"
```

---

## Task 4: Bridge to the Psycholinguistic Analyzer

**Files:**
- Test: `tests/transcription/test_bridge.py`

**Interfaces:**
- Consumes: `Transcriber`, `FakeTranscriptionBackend` (Tasks 2-3); `Transcript.statements()` (Task 1); `PsycholinguisticAnalyzer.analyze(list[str]) -> PsycholinguisticScore` (existing, `app.pipelines.psycholinguistic.analyzer`).
- Produces: no new source — this task proves the contract `transcript.statements()` → `analyze()` holds end to end. (Pure integration test; the analyzer and transcriber already exist, so this is the gate that locks their interface together.)

- [ ] **Step 1: Write the failing bridge tests**

Create `tests/transcription/test_bridge.py`:

```python
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
```

> Note: the existing analyzer raises `ValueError("No statements provided")` on empty input (see `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py`). This test asserts the caller-visible contract, not new behavior.

- [ ] **Step 2: Run the tests to verify they fail or pass appropriately**

```bash
.venv/Scripts/python -m pytest tests/transcription/test_bridge.py -q
```
Expected: PASS (both the transcriber and analyzer already exist; this locks their contract). If `test_transcript_statements_score_through_analyzer` errors on import of the analyzer, confirm `tests/transcription/conftest.py` inserted the `backend/ml-inference` root — it must, because the analyzer is imported via `app.pipelines.psycholinguistic`.

- [ ] **Step 3: Commit**

```bash
git add tests/transcription/test_bridge.py
git commit -m "test(transcription): lock transcript->analyzer bridge contract"
```

---

## Task 5: WhisperXBackend (real, lazy) + slow test + deps

**Files:**
- Modify: `backend/ml-inference/app/pipelines/transcription/backends.py` (add `WhisperXBackend`)
- Modify: `backend/ml-inference/app/pipelines/transcription/__init__.py` (export `WhisperXBackend`)
- Modify: `pyproject.toml` (register `slow` marker; add optional `transcription` extra)
- Test: `tests/transcription/test_whisperx_backend.py`

**Interfaces:**
- Consumes: `Transcript`, `TranscriptSegment`, `TranscriptionConfig` (Task 1).
- Produces:
  - `WhisperXBackend(config: Optional[TranscriptionConfig] = None)` — `.transcribe(flac_path) -> Transcript` with `backend="whisperx"`. Lazy-imports `whisperx`; raises `RuntimeError` with an install command if absent. Alignment ON, diarization OFF.

- [ ] **Step 1: Register the `slow` marker and add the optional extra in pyproject.toml**

In `pyproject.toml`, under `[project.optional-dependencies]`, add a new extra after the `dev` list:

```toml
# Transcription vector (backend/ml-inference). Heavy: torch + faster-whisper +
# wav2vec2 alignment. Quarantined behind TranscriptionBackend; install only when
# running the real WhisperXBackend. On Windows + Python 3.13 install torch from
# the appropriate index first; WSL/Linux is the documented fallback runner.
transcription = [
    "whisperx>=3.1",
]
```

In `pyproject.toml`, replace the `[tool.pytest.ini_options]` block:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q -m 'not slow'"
markers = [
    "slow: requires heavy models / downloads (WhisperX). Run with -m slow.",
]
```

> This means `pytest` skips slow tests by default; run them explicitly with
> `.venv/Scripts/python -m pytest -m slow`.

- [ ] **Step 2: Write the gated slow test**

Create `tests/transcription/test_whisperx_backend.py`:

```python
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
```

- [ ] **Step 3: Run the slow test to verify it is skipped by default and gated when invoked**

```bash
.venv/Scripts/python -m pytest tests/transcription/test_whisperx_backend.py -q
```
Expected: `1 deselected` (skipped by the default `-m 'not slow'`).

```bash
.venv/Scripts/python -m pytest tests/transcription/test_whisperx_backend.py -m slow -q
```
Expected: either `skipped` (whisperx not installed — the `importorskip` fires) or, once installed, FAIL with `ImportError`/`AttributeError` for `WhisperXBackend` (not yet defined). Both confirm the gate works.

- [ ] **Step 4: Add `WhisperXBackend` to `backends.py`**

Append to `backend/ml-inference/app/pipelines/transcription/backends.py`:

```python
class WhisperXBackend:
    """Real transcription backend: WhisperX with alignment ON, diarization OFF.

    WhisperX (torch + faster-whisper + wav2vec2) is lazy-imported on first
    ``transcribe`` call so importing this module never pulls in torch. If
    whisperx is not installed, ``transcribe`` raises a RuntimeError naming the
    install extra.

    Long audio is handled by WhisperX's built-in Silero VAD chunking; peak
    memory is bounded by ``config.batch_size`` x chunk, not the whole file.
    """

    def __init__(self, config: "Optional[TranscriptionConfig]" = None) -> None:
        # Imported here (not at module top) to keep the schema import cheap and
        # avoid a hard dependency on the transcription extra at import time.
        from backend.shared.schemas.transcription import TranscriptionConfig

        self._config = config or TranscriptionConfig()
        self._model = None  # lazy

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
```

Add `TranscriptionConfig` to the top-of-file import from the schema (so the type is available for the annotation):

Change the existing import line in `backends.py` from:

```python
from backend.shared.schemas.transcription import Transcript, TranscriptSegment
```

to:

```python
from backend.shared.schemas.transcription import (
    Transcript,
    TranscriptionConfig,
    TranscriptSegment,
)
```

And update the `Optional` import at the top of `backends.py` — it is already imported (`from typing import Optional, Protocol, runtime_checkable`), so no change needed.

- [ ] **Step 5: Export `WhisperXBackend` from the package init**

Replace `backend/ml-inference/app/pipelines/transcription/__init__.py` with:

```python
"""Transcription pipeline package."""
from app.pipelines.transcription.backends import (
    FakeTranscriptionBackend,
    TranscriptionBackend,
    WhisperXBackend,
)
from app.pipelines.transcription.transcriber import Transcriber

__all__ = [
    "TranscriptionBackend",
    "FakeTranscriptionBackend",
    "WhisperXBackend",
    "Transcriber",
]
```

- [ ] **Step 6: Verify import works without whisperx installed, and the default suite stays green**

```bash
.venv/Scripts/python -c "import sys; sys.path.insert(0, 'backend/ml-inference'); from app.pipelines.transcription import WhisperXBackend; print('import ok (no torch needed)')"
```
Expected: `import ok (no torch needed)` — importing the class must NOT import torch/whisperx.

```bash
.venv/Scripts/python -m pytest tests/transcription/ -q
```
Expected: all non-slow transcription tests PASS; the slow test is deselected.

- [ ] **Step 7: Commit**

```bash
git add backend/ml-inference/app/pipelines/transcription/backends.py backend/ml-inference/app/pipelines/transcription/__init__.py pyproject.toml tests/transcription/test_whisperx_backend.py
git commit -m "feat(transcription): add lazy WhisperXBackend, gated slow test, slow marker + extra"
```

---

## Task 6: `scripts/test_transcribe.py` CLI + smoke test

**Files:**
- Create: `scripts/test_transcribe.py`
- Test: `tests/transcription/test_cli_smoke.py`

**Interfaces:**
- Consumes: `Transcriber`, `FakeTranscriptionBackend`, `WhisperXBackend` (Tasks 2-5).
- Produces: a CLI that prints a transcript with per-segment timestamps and the billable duration. `--fake` flag uses `FakeTranscriptionBackend` (so the smoke test needs no models).

- [ ] **Step 1: Write the failing smoke test**

Create `tests/transcription/test_cli_smoke.py`:

```python
"""Smoke test: the transcribe CLI runs in --fake mode and prints segments."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_fake_mode_prints_transcript(tmp_path):
    flac = tmp_path / "clip.flac"
    flac.write_bytes(b"")
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "test_transcribe.py"),
            str(flac),
            "--fake",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Transcript" in result.stdout
    assert "Billable duration" in result.stdout
    assert "[" in result.stdout  # at least one [start -> end] timestamp line
```

- [ ] **Step 2: Run the smoke test to verify it fails**

```bash
.venv/Scripts/python -m pytest tests/transcription/test_cli_smoke.py -q
```
Expected: FAIL — the script does not exist (`can't open file ... test_transcribe.py`).

- [ ] **Step 3: Create the CLI**

Create `scripts/test_transcribe.py`:

```python
#!/usr/bin/env python
"""CLI: transcribe a FLAC/WAV file and print timestamped segments.

The transcription pipeline lives under ``backend/ml-inference/`` (hyphenated
service root), so we insert that root onto ``sys.path`` like
``tests/transcription/conftest.py`` does, then import ``from app.pipelines...``.

Usage:
    python scripts/test_transcribe.py path/to/audio.flac           # real WhisperX
    python scripts/test_transcribe.py path/to/audio.flac --fake    # canned, no models
    python scripts/test_transcribe.py path/to/audio.flac --model large-v3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ML_INFERENCE_ROOT = _REPO_ROOT / "backend" / "ml-inference"
for _p in (_ML_INFERENCE_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe a FLAC/WAV file")
    parser.add_argument("audio", type=Path, help="Path to a .flac or .wav file")
    parser.add_argument("--fake", action="store_true", help="Use the fake backend (no models)")
    parser.add_argument("--model", default="distil-large-v3", help="Whisper model name")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"ERROR: file not found: {args.audio}", file=sys.stderr)
        return 1

    from app.pipelines.transcription.backends import FakeTranscriptionBackend
    from app.pipelines.transcription.transcriber import Transcriber

    if args.fake:
        backend = FakeTranscriptionBackend()
    else:
        from app.pipelines.transcription.backends import WhisperXBackend
        from backend.shared.schemas.transcription import TranscriptionConfig

        try:
            backend = WhisperXBackend(TranscriptionConfig(model_name=args.model))
        except Exception as exc:  # pragma: no cover - real-backend path
            print(f"ERROR: could not init WhisperX backend: {exc}", file=sys.stderr)
            return 1

    try:
        transcript = Transcriber(backend).transcribe(args.audio)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nTranscript ({transcript.backend}, {transcript.model_name}, "
          f"lang={transcript.language})")
    print("-" * 64)
    if not transcript.segments:
        print("  (no speech detected)")
    for seg in transcript.segments:
        print(f"  [{seg.start_seconds:7.2f} -> {seg.end_seconds:7.2f}]  {seg.text}")
    print("-" * 64)
    print(f"  Billable duration: {transcript.audio_duration_seconds:.2f}s  "
          f"| segments: {len(transcript.segments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the smoke test to verify it passes**

```bash
.venv/Scripts/python -m pytest tests/transcription/test_cli_smoke.py -q
```
Expected: PASS.

- [ ] **Step 5: Eyeball the output**

```bash
.venv/Scripts/python scripts/test_transcribe.py demo_data/honest/trial_truth_001.mp4 --fake
```
Expected: prints 3 canned segments with `[start -> end]` lines and a billable-duration footer. (The `.mp4` extension is fine here because `--fake` skips the FLAC-only validation path? No — validation still runs. Use a `.flac`/`.wav` path instead, or run on an extracted FLAC. To get a real FLAC: `python scripts/test_compress_and_analyze.py <video>` writes one under `processed_output/...`.) Run:

```bash
.venv/Scripts/python scripts/test_transcribe.py processed_output/compress_analyze_test/trial_truth_001/audio/trial_truth_001.flac --fake
```
Expected: same canned output, exit 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/test_transcribe.py tests/transcription/test_cli_smoke.py
git commit -m "feat(transcription): add test_transcribe.py CLI + smoke test"
```

---

## Task 7: Replace the Integration Stub (live compress → transcribe → analyze)

**Files:**
- Modify: `scripts/test_compress_and_analyze.py`

**Interfaces:**
- Consumes: `CompressionPipeline` (existing), `Transcriber` + `WhisperXBackend` + `FakeTranscriptionBackend` (Tasks 2-5), `PsycholinguisticAnalyzer` (existing).
- Produces: the script now runs the real path; `--fake` lets it complete end-to-end with no models. Graceful message if the real WhisperX backend is unavailable.

- [ ] **Step 1: Rewrite the script to replace the pending stub**

Replace the entire contents of `scripts/test_compress_and_analyze.py` with:

```python
#!/usr/bin/env python
"""Integration: video -> compression -> transcription -> psycholinguistic score.

Runs the real pipeline end to end. With --fake, transcription uses canned
segments (no models) so the full path runs offline. Without --fake it uses the
real WhisperX backend; if WhisperX is not installed it prints a clear message
rather than crashing.

Usage:
    python scripts/test_compress_and_analyze.py path/to/video.mp4
    python scripts/test_compress_and_analyze.py path/to/video.mp4 --fake
    python scripts/test_compress_and_analyze.py path/to/video.mp4 --mode edge_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ML_INFERENCE_ROOT = _REPO_ROOT / "backend" / "ml-inference"
for _p in (_ML_INFERENCE_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compression -> transcription -> psycholinguistic")
    parser.add_argument("video", type=Path, help="Path to a video file")
    parser.add_argument(
        "--mode",
        choices=["raw", "roi", "edge_full", "edge_minimal"],
        default="edge_full",
        help="Compression mode (default: edge_full)",
    )
    parser.add_argument("--fake", action="store_true", help="Use the fake transcription backend")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1

    from backend.shared.schemas.media import CompressionMode
    from backend.workers.app.compression.pipeline import CompressionPipeline

    mode_map = {
        "raw": CompressionMode.RAW,
        "roi": CompressionMode.ROI_ENCODED,
        "edge_full": CompressionMode.EDGE_FULL,
        "edge_minimal": CompressionMode.EDGE_MINIMAL,
    }
    mode = mode_map[args.mode]
    output_dir = _REPO_ROOT / "processed_output" / "compress_analyze_test" / args.video.stem

    # ---- Step 1: compression ------------------------------------------------
    print(f"\nStep 1 -- Compression Pipeline ({mode.value})")
    print("-" * 64)
    result = CompressionPipeline().process(args.video, output_dir, mode)
    mb = 1_048_576
    print(f"  Input:           {result.input_size_bytes / mb:6.1f} MB")
    print(f"  FLAC audio (ML): {result.flac_size_bytes / mb:6.2f} MB  -> {result.flac_audio_path}")
    print(f"  Total time:      {result.processing_times.get('total', 0.0):5.1f}s")

    # ---- Step 2: transcription ---------------------------------------------
    print("\nStep 2 -- Transcription")
    print("-" * 64)
    from app.pipelines.transcription.backends import FakeTranscriptionBackend
    from app.pipelines.transcription.transcriber import Transcriber

    if args.fake:
        backend = FakeTranscriptionBackend()
        print("  backend: fake (canned segments, no models)")
    else:
        from app.pipelines.transcription.backends import WhisperXBackend

        try:
            backend = WhisperXBackend()
        except Exception as exc:  # pragma: no cover
            print(f"  WhisperX backend unavailable: {exc}")
            print("  Re-run with --fake to exercise the path offline.")
            return 1

    try:
        transcript = Transcriber(backend).transcribe(result.flac_audio_path)
    except Exception as exc:  # pragma: no cover - real-backend runtime errors
        print(f"  Transcription failed: {exc}")
        print("  Re-run with --fake to exercise the path offline.")
        return 1

    print(f"  backend={transcript.backend} segments={len(transcript.segments)} "
          f"duration={transcript.audio_duration_seconds:.2f}s")
    for seg in transcript.segments[:5]:
        print(f"    [{seg.start_seconds:6.2f} -> {seg.end_seconds:6.2f}] {seg.text}")

    # ---- Step 3: psycholinguistic analysis ---------------------------------
    print("\nStep 3 -- Psycholinguistic Analysis")
    print("-" * 64)
    statements = transcript.statements()
    if not statements:
        print("  No speech detected -- nothing to analyze.")
        return 0

    from app.pipelines.psycholinguistic.analyzer import PsycholinguisticAnalyzer

    score = PsycholinguisticAnalyzer().analyze(statements)
    print(f"  Statements analyzed: {score.statement_count}")
    print(f"  Composite score:     {score.composite_score:5.1f}/100  "
          f"(confidence: {score.confidence})")
    print()
    print("  NOTE: behavioral anomaly signal, not ground truth. ~75% F1 ceiling.")
    print("\nLive path complete: video -> FLAC -> transcript -> psycholinguistic score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the full live path in fake mode on a demo video**

```bash
.venv/Scripts/python scripts/test_compress_and_analyze.py demo_data/honest/trial_truth_001.mp4 --fake
```
Expected: Step 1 prints compression summary; Step 2 prints `backend=fake segments=3`; Step 3 prints a composite score and confidence; final line `Live path complete...`. Exit 0.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

```bash
.venv/Scripts/python -m pytest tests/ -p no:cacheprovider
```
Expected: all pass (previous 44 + the new transcription tests; slow test deselected).

- [ ] **Step 4: Commit**

```bash
git add scripts/test_compress_and_analyze.py
git commit -m "feat(transcription): replace stub with live compress->transcribe->analyze path"
```

---

## Final Verification

- [ ] **Full suite green, slow test excluded by default**

```bash
.venv/Scripts/python -m pytest tests/ -p no:cacheprovider
```
Expected: all pass; output notes the slow test deselected.

- [ ] **The headline deliverable runs end to end offline**

```bash
.venv/Scripts/python scripts/test_compress_and_analyze.py demo_data/honest/trial_truth_001.mp4 --fake
```
Expected: a composite psycholinguistic score printed from a real compression + (fake) transcript.

- [ ] **WhisperX import is lazy (no torch needed to import the class)**

```bash
.venv/Scripts/python -c "import sys; sys.path.insert(0, 'backend/ml-inference'); from app.pipelines.transcription import WhisperXBackend; print('ok')"
```
Expected: `ok`.

- [ ] **Working tree clean**

```bash
git status --short
```
Expected: empty.

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - Schemas (TranscriptSegment/Transcript/TranscriptionConfig, billable duration, reserved speaker) → Task 1
  - Backend protocol + Fake backend → Task 2
  - Transcriber facade + FLAC-only invariant #1 → Task 3
  - Bridge to analyzer (segments-as-statements) → Task 4
  - WhisperXBackend (alignment on, diarization off, lazy, device auto, VAD chunking) → Task 5
  - Install caveat + gated slow test + `slow` marker + optional extra → Task 5
  - CLIs (test_transcribe.py; replace test_compress_and_analyze stub) → Tasks 6, 7
  - Testing strategy (fast fake suite + gated real test, conftest sys.path bridge) → all tasks
- [x] **Invariant #1** enforced in Transcriber with a message matching the test's `match="INVARIANT #1"` (case: the message contains "INVARIANT #1").
- [x] **Invariant #3** — no transcript text in any log statement.
- [x] **Invariant #6** — no "lie detector" phrasing anywhere.
- [x] **Type consistency:** `Transcript`, `TranscriptSegment`, `TranscriptionConfig`, `Transcriber`, `FakeTranscriptionBackend`, `WhisperXBackend`, `TranscriptionBackend`, `.statements()`, `.transcribe()`, `backend="fake"/"whisperx"` used identically across all tasks.
- [x] **Import pattern:** every test/script inserts `backend/ml-inference` on `sys.path` before `from app.pipelines...`; `backend.*` works via the editable install.
- [x] **No placeholders:** every code step shows complete code; every run step shows the exact command and expected output.
- [x] **Namespace packages:** no `__init__.py` added to `app/` or `app/pipelines/`; only `transcription/` gets one (a regular package nested in a namespace package, which Python permits).
```
