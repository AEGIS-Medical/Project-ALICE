# ALICE — WhisperX Transcription Vector (Session 3)
## Design Spec · 2026-06-25

---

## Overview

Bridge the compression pipeline's lossless FLAC output to the psycholinguistic
analyzer by adding speaker-attributed (speaker deferred) transcription. This
turns the `scripts/test_compress_and_analyze.py` *pending stub* into a genuine
live scoring path:

```
video → CompressionPipeline → FLAC → Transcriber (WhisperX) → Transcript
      → Transcript.statements() → PsycholinguisticAnalyzer.analyze() → score
```

**Decisions locked during brainstorming:**

1. **Transcription-first, diarization deferred.** WhisperX runs with **word-level
   alignment ON** and **pyannote diarization OFF**. Diarization is a follow-up
   session; the schema reserves a `speaker` field so it lands as a field
   population, not a migration.
2. **Engine: WhisperX (minus the diarize step).** Wraps `faster-whisper` and adds
   wav2vec2 forced alignment — the exact component CLAUDE.md specifies. Word-level
   timestamps make deferred diarization a "add a step," not a re-transcribe.
3. **Statement unit = one WhisperX segment.** Natural pause/breath groups with
   exact timestamps. The analyzer reassembles all statements into one document
   before parsing (`text = " ".join(statements)`), so full linguistic context is
   preserved regardless of segmentation — segmentation only sets `statement_count`
   and the timestamped units stored for the future contradiction vector.
4. **Backend-protocol structure.** A swappable `TranscriptionBackend` isolates the
   heavy WhisperX/torch dependency behind a one-method seam. A
   `FakeTranscriptionBackend` keeps the default test suite fast and download-free.

---

## Business / Scaling Context

This vector must support the product's billing model: **charge by video size /
length**, viable for real workloads such as a **30-minute interview call**.

- Long audio is handled by WhisperX's built-in **Silero VAD** chunking — a 30-min
  recording is processed as a stream of short speech chunks. Peak memory is
  O(batch_size × chunk), not O(whole file). No custom chunking code.
- **Large model downloads are explicitly acceptable** (a stated product
  requirement). We do not optimize for tiny footprint at the cost of accuracy.
- `Transcript.audio_duration_seconds` is a **first-class, billable field**; every
  transcript self-reports the meterable content length. `transcribe()` also
  records wall-clock processing time (the cost side) in logs.

---

## Integration Boundary (what this session does NOT build)

- **Platform connectors (WhatsApp / Teams / Zoom / Meet / common APIs).** Fetching
  recordings from platforms is the separate `PlatformConnector` subsystem (CLAUDE.md
  "PLATFORM CONNECTORS"), on the pending list. It produces *a file on disk*.
- **Container → FLAC normalization is already solved.** The existing `AudioExtractor`
  accepts `.mp4/.mov/.mkv/.webm/.avi/.m4a/.aac/.ogg/...` and demuxes any of them to
  FLAC 48 kHz mono via ffmpeg. Whatever a platform export *is*, once it is a file the
  compression pipeline produces the FLAC this vector consumes. The transcriber takes
  a FLAC path and does not care how the audio arrived — so connectors, when built,
  feed the same `CompressionPipeline.process()` entry point with **zero rework here**.
- **Diarization (pyannote) and the contradiction vector** are subsequent sessions.

---

## Architecture & File Structure

Server-side ML inference, so it lives under the hyphenated `ml-inference` service
root using the established `sys.path` import pattern (same as the psycholinguistic
analyzer). Schemas live in `backend/shared/schemas/`.

```
backend/
  shared/schemas/
    transcription.py          NEW — TranscriptSegment, Transcript, TranscriptionConfig (frozen Pydantic)
  ml-inference/app/pipelines/
    transcription/
      __init__.py             NEW — exports Transcriber + backends
      backends.py             NEW — TranscriptionBackend protocol
                                     + WhisperXBackend (real, lazy-loaded)
                                     + FakeTranscriptionBackend (tests)
      transcriber.py          NEW — Transcriber facade: validates FLAC, calls backend,
                                     returns Transcript
tests/transcription/
  __init__.py                 NEW
  conftest.py                 NEW — sys.path bridge to backend/ml-inference + Fake backend fixture
  test_schemas.py             NEW
  test_transcriber.py         NEW — Transcriber via FakeTranscriptionBackend
  test_bridge.py              NEW — Fake → statements() → analyzer.analyze()
  test_whisperx_backend.py    NEW — real model, @pytest.mark.slow, skipped by default
scripts/
  test_transcribe.py          NEW — CLI: FLAC in, print transcript + timestamps + billable duration
  test_compress_and_analyze.py MODIFIED — replace pending stub with real transcribe → analyze
```

### Component responsibilities

- **`TranscriptionBackend` (Protocol)** — one method,
  `transcribe(flac_path: Path) -> Transcript`. The seam isolating WhisperX/torch.
- **`WhisperXBackend`** — the real engine. Lazy-loads the Whisper model + alignment
  model on first call (mirrors the spaCy/MediaPipe lazy-load pattern already used).
  Holds model/device/compute-type config. Alignment ON, diarization OFF.
- **`FakeTranscriptionBackend`** — returns deterministic canned segments. No torch,
  no downloads. Lets the whole pipeline and bridge run green in milliseconds.
- **`Transcriber`** — the facade callers use. Validates input is FLAC/WAV (invariant
  #1), delegates to the injected backend, returns a `Transcript`.

---

## Data Flow & Schemas

```
video.mp4
   │  CompressionPipeline.process(mode=EDGE_FULL)
   ▼
FLAC (48 kHz mono, lossless — the ML-safe artifact)
   │  Transcriber.transcribe(flac_path)
   ▼
Transcript (ordered TranscriptSegment list; each text + start/end)
   │  transcript.statements()  →  list[str]
   ▼
PsycholinguisticAnalyzer.analyze(statements)  →  PsycholinguisticScore
```

### `backend/shared/schemas/transcription.py`

```python
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(description="Transcribed text for this segment.")
    start_seconds: float = Field(ge=0.0, description="Segment start (word-aligned).")
    end_seconds: float = Field(ge=0.0, description="Segment end (word-aligned).")
    # Reserved for the deferred diarization step; always None this session.
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
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(
        default="distil-large-v3",
        description="Whisper model. 'distil-large-v3' (fast) or 'large-v3' (accurate).",
    )
    device: str = Field(
        default="auto",
        description="'auto' selects cuda if available else cpu; or pin 'cpu'/'cuda'.",
    )
    compute_type: str = Field(
        default="int8",
        description="ctranslate2 compute type. int8 on CPU, float16 on GPU (auto).",
    )
    batch_size: int = Field(default=16, ge=1, description="WhisperX batched inference size.")
    language: Optional[str] = Field(
        default=None,
        description="None autodetects; pin e.g. 'en' to skip language detection.",
    )
    vad_chunk_seconds: float = Field(
        default=30.0, gt=0.0, description="VAD speech-chunk window for long audio.",
    )


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    segments: list[TranscriptSegment]
    language: str = Field(description="Detected (or pinned) language code, e.g. 'en'.")
    audio_duration_seconds: float = Field(
        ge=0.0, description="Source audio length — the billable / meterable unit.",
    )
    model_name: str = Field(description="Model that produced this transcript.")
    backend: str = Field(description="Provenance: 'whisperx' | 'fake'.")

    def statements(self) -> list[str]:
        """Segment texts in order — the exact input analyzer.analyze() expects."""
        return [s.text for s in self.segments]

    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)
```

Two business-aligned choices: `audio_duration_seconds` is the billing unit on every
transcript; `speaker` exists but stays `None` so diarization is additive.

---

## Configuration & Long-Audio Scaling

- **Long audio handled by WhisperX VAD chunking**, not custom code. Silero VAD splits
  the recording into ~`vad_chunk_seconds` speech chunks, batched at `batch_size`. A
  30-minute interview → peak memory O(batch_size × chunk).
- **`distil-large-v3` default = throughput/cost lever** (~6× faster than `large-v3`
  at near-equal English WER). Premium-accuracy callers set `model_name="large-v3"`.
- **`device="auto"` + `compute_type`** runs `int8` on CPU now (this Windows box, no
  GPU) and `float16` on CUDA in prod — same code, no edits.
- **Metering hook:** `transcribe()` logs wall-clock processing time alongside
  `audio_duration_seconds` — the two numbers billing needs (content length sold vs.
  compute cost incurred).

### Install caveat (documented, not blocking)

WhisperX on **Windows + Python 3.13** may be a rough install (torch wheels,
`ctranslate2`/`pyannote` pins). The fake-backend design ensures the pipeline and
default test suite are **never blocked** by this. The real `WhisperXBackend`:

- is lazy-loaded — importing the module never imports torch;
- raises a `RuntimeError` with the documented install command if `whisperx` is absent;
- has its only real-model test gated behind `@pytest.mark.slow` (skipped by default).

Spec records the install path: install torch from the appropriate index URL, then
`whisperx`; pin `ctranslate2`/`pyannote.audio` versions known to coexist with the
project's mediapipe/numpy. **WSL/Linux is the documented fallback runner** if Windows
wheels conflict.

---

## Error Handling & Invariants

**Invariant enforcement (CLAUDE.md):**
- **#1 Lossless audio only.** `Transcriber.transcribe()` rejects non-FLAC/WAV with a
  `ValueError` naming invariant #1 — same pattern as
  `FeatureExtractor.extract_audio_features`. WhisperX never sees lossy audio.
- **#3 Never log PII.** Logs carry opaque facts only (path, duration, segment count,
  model, timing) — **never transcript text**, which is PII.

**Failure modes:**

| Failure | Handling |
|---|---|
| Input not FLAC/WAV | `ValueError` (invariant #1), before any model load |
| File missing / not a regular file | `FileNotFoundError` / `ValueError`, fail fast |
| WhisperX/torch not installed | `WhisperXBackend` raises `RuntimeError` on first use w/ install command |
| Model download fails (network) | `RuntimeError` wrapping cause; no partial cache left |
| Empty / silent audio (VAD finds no speech) | Valid **empty** `Transcript` (`segments=[]`, duration set) — NOT an error |
| Empty transcript reaches analyzer | `analyzer.analyze([])` raises `ValueError("No statements provided")`; caller reports "no speech detected" |

The empty-audio path is deliberate: silence is a legitimate outcome (hold music,
muted participant), reported as a well-formed empty transcript with duration still
populated (still billable). The *caller* decides what to do with zero statements —
keeping the transcriber honest about "heard nothing" vs. "broke."

**Determinism:** WhisperX with a fixed model + `temperature=0` (greedy) is effectively
deterministic per file, so the real-model test can assert stable output. The fake
backend is trivially deterministic.

---

## Testing Strategy

Target: CLAUDE.md's 90% for ML pipelines — achieved with **no model downloads** in the
default suite via the fake backend.

**Default suite (fast, no torch, no network):**

| File | Coverage |
|---|---|
| `test_schemas.py` | frozen; `end ≥ start` guard; `statements()` order; `full_text()` join; `duration ≥ 0`; empty-segments transcript valid |
| `test_transcriber.py` | Fake backend: FLAC/WAV accepted; `.mp3`/`.opus` → `ValueError` naming invariant #1; missing file → `FileNotFoundError`; delegation; empty-audio → valid empty transcript |
| `test_bridge.py` | Fake → `statements()` → `analyzer.analyze()` returns valid `PsycholinguisticScore`; empty transcript → analyzer raises expected `ValueError`, caller handles it |

**Opt-in real-model test:**

| `test_whisperx_backend.py` | `@pytest.mark.slow`; skips if `whisperx` import fails. Transcribes a short real `demo_data/` FLAC; asserts ≥1 segment, language `"en"`, monotonic non-overlapping timestamps, `backend == "whisperx"`. Run via `pytest -m slow`. |

**Fixtures (`conftest.py`):** `sys.path` bridge to `backend/ml-inference` (mirrors
`tests/psycholinguistic/conftest.py`); a `FakeTranscriptionBackend` returning 3 canned
segments with known timestamps/text for exact assertions.

**Manual CLIs (smoke, not unit):**
- `scripts/test_transcribe.py <flac>` → prints transcript + per-segment timestamps +
  billable duration.
- `scripts/test_compress_and_analyze.py <video>` → **stub replaced**: real compress →
  transcribe → analyze, printing the 8-dimension score. With WhisperX uninstalled it
  prints a clear "transcription backend unavailable — install per spec" message rather
  than crashing.

---

## Deliverable

The headline outcome: `scripts/test_compress_and_analyze.py` becomes a genuine
**video → FLAC → transcript → psycholinguistic score** path. The transcription seam is
structured so the deferred pyannote diarization and the next-session contradiction
vector attach to existing timestamped segments without rework.

---

## Out of Scope (future sessions)

- pyannote speaker diarization (populates `TranscriptSegment.speaker`)
- Statement contradiction vector (WhisperX → embeddings → DeBERTa NLI + pgvector)
- Platform connectors (WhatsApp / Teams / Zoom / Meet / Webex / Slack)
- Per-contact transcript history / baselining
- The remaining analysis vectors (AU, tonality) and late-fusion ensemble
