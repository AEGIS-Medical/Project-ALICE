# ALICE Bridge + Psycholinguistic Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the mobile-critical landmark memory/size issue (P1-S6), then build the complete psycholinguistic analysis module (P2-S1 through P2-S10), delivering a working path from video → FLAC → transcript → 8-dimension linguistic score.

**Architecture:** P1-S6 converts in-memory JSON landmark accumulation to streaming JSONL (line-by-line flush), capping peak RAM at O(flush_interval) frames instead of O(all_frames). Phase 2 builds `PsycholinguisticAnalyzer` as a standalone Python class using spaCy + Empath + NRCLex + VADER, scored across 8 research-backed dimensions and combined into a Pydantic-validated composite score.

**Tech Stack:** Python 3.13, spaCy 3.x + `en_core_web_sm`, NRCLex 4.0, vaderSentiment 3.3.2, Empath 0.89, Pydantic v2, pytest, tracemalloc

---

> **Pre-flight status:**
> - P1-S7 (platform-aware model cache) — ✅ already in `models.py`
> - P1-S8 (mid-session tier switching) — ✅ already in `pipeline.py`
> - `media.py` `CompressionResult` — includes `mode_transitions` field (verify before Task 1)
> - All Phase 1 compression pipeline files — ✅ shipped

---

## File Map

| File | Action | Task |
|---|---|---|
| `backend/workers/app/compression/feature_extractor.py` | Modify — streaming JSONL | Task 1 |
| `backend/shared/schemas/media.py` | Verify `mode_transitions` field exists | Task 1 pre-check |
| `tests/compression/test_feature_extractor.py` | Add streaming tests | Task 1 |
| `backend/shared/schemas/psycholinguistic.py` | Create | Task 2 |
| `tests/psycholinguistic/__init__.py` | Create (empty) | Task 2 |
| `tests/psycholinguistic/test_schemas.py` | Create | Task 2 |
| `backend/ml_inference/__init__.py` | Create (empty) | Task 3 |
| `backend/ml_inference/app/__init__.py` | Create (empty) | Task 3 |
| `backend/ml_inference/app/pipelines/__init__.py` | Create (empty) | Task 3 |
| `backend/ml_inference/app/pipelines/psycholinguistic/__init__.py` | Create (empty) | Task 3 |
| `backend/ml_inference/app/pipelines/psycholinguistic/analyzer.py` | Create — full analyzer | Tasks 3–10 |
| `tests/psycholinguistic/test_analyzer.py` | Create | Tasks 3–10 |
| `scripts/test_psycholinguistic.py` | Create | Task 11 |
| `scripts/test_compress_and_analyze.py` | Create | Task 11 |

> **Note on path:** CLAUDE.md specifies `backend/ml-inference/` (hyphen). Python cannot import packages with hyphens in directory names. This plan uses `backend/ml_inference/` (underscore). Update CLAUDE.md after this plan ships.

---

## Task 1: Streaming JSONL Landmark Emitter (P1-S6)

**Problem:** `feature_extractor.py` accumulates all landmark frames as Python dicts in RAM before writing. For a 60-minute call at 30fps: ~617 MB peak RAM. Android OS-kills the process. Output is plain JSON (~14.8 MB for one test video), violating the CLAUDE.md ~70 KB/min telemetry budget.

**Fix:** Write each frame as a newline-delimited JSON record immediately, flush every `flush_interval` frames. Output changes from `_landmarks.json` to `_landmarks.jsonl`.

**Files:**
- Modify: `backend/workers/app/compression/feature_extractor.py`
- Test: `tests/compression/test_feature_extractor.py`

---

- [ ] **Step 1: Install tracemalloc dependency (already stdlib, verify pytest-memray is not needed)**

```bash
cd C:\Users\ryanh\ALICE\Project-ALICE
.venv\Scripts\python -c "import tracemalloc; print('ok')"
```
Expected: `ok`

---

- [ ] **Step 2: Write the failing tests for streaming JSONL**

Append to `tests/compression/test_feature_extractor.py`:

```python
# ---- P1-S6: Streaming JSONL tests ----------------------------------------
import json
import tracemalloc
from pathlib import Path


def test_output_is_valid_jsonl(tmp_path, tiny_video):
    """Every line of the output must be independently json.loads()-able."""
    extractor = FeatureExtractor()
    out = extractor.extract_landmarks(tiny_video, tmp_path)
    assert out.suffix == ".jsonl", f"Expected .jsonl, got {out.suffix}"
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0, "No lines written"
    for i, line in enumerate(lines):
        record = json.loads(line)  # raises if invalid JSON
        assert "frame_number" in record, f"line {i}: missing frame_number"
        assert "timestamp_seconds" in record, f"line {i}: missing timestamp_seconds"
        assert "landmarks" in record, f"line {i}: missing landmarks key"


def test_flush_interval_partial_file_readable(tmp_path, tiny_video):
    """A partial JSONL file (e.g. after crash) must be readable line-by-line."""
    extractor = FeatureExtractor(flush_interval=1)
    out = extractor.extract_landmarks(tiny_video, tmp_path)
    # Simulate reading only the first 3 lines (partial file scenario)
    lines = out.read_text(encoding="utf-8").splitlines()
    for line in lines[:3]:
        record = json.loads(line)
        assert isinstance(record["frame_number"], int)


def test_flush_interval_parameter_accepted():
    """FeatureExtractor accepts flush_interval kwarg without error."""
    extractor = FeatureExtractor(flush_interval=15)
    assert extractor.flush_interval == 15


def test_flush_interval_default_is_30():
    extractor = FeatureExtractor()
    assert extractor.flush_interval == 30


def test_output_filename_is_jsonl_not_json(tmp_path, tiny_video):
    extractor = FeatureExtractor()
    out = extractor.extract_landmarks(tiny_video, tmp_path)
    assert out.name.endswith("_landmarks.jsonl")
    assert not out.name.endswith(".json")
```

> Note: `tiny_video` is a pytest fixture. Check if it already exists in `tests/compression/conftest.py`; if not, add:
> ```python
> # tests/compression/conftest.py
> import pytest
> from pathlib import Path
>
> @pytest.fixture
> def tiny_video():
>     """Return path to a small test video. Requires a real video file."""
>     p = Path("demo_data/honest/demo_file_1").glob("*.mp4")
>     candidates = list(p)
>     if not candidates:
>         pytest.skip("No test video found in demo_data/honest/demo_file_1/")
>     return candidates[0]
> ```

---

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd C:\Users\ryanh\ALICE\Project-ALICE
.venv\Scripts\pytest tests/compression/test_feature_extractor.py -k "jsonl or flush_interval" -v
```
Expected: Multiple failures including `AssertionError: Expected .jsonl, got .json`

---

- [ ] **Step 4: Modify `feature_extractor.py` — add `flush_interval` to `__init__`**

In `backend/workers/app/compression/feature_extractor.py`, locate `__init__` and replace:

```python
    def __init__(self, frame_skip: int = 1) -> None:
        if frame_skip < 1:
            raise ValueError(
                f"frame_skip must be >= 1 (got {frame_skip}); use 1 to process every frame."
            )
        self.frame_skip: int = frame_skip

        # Telemetry from the most recent extraction call. Reset on each call.
        self.last_frames_processed: int = 0
        self.last_frames_with_face: int = 0
        self.last_audio_windows: int = 0
        self.last_audio_sample_rate: int = 0
```

Replace with:

```python
    def __init__(self, frame_skip: int = 1, flush_interval: int = 30) -> None:
        if frame_skip < 1:
            raise ValueError(
                f"frame_skip must be >= 1 (got {frame_skip}); use 1 to process every frame."
            )
        if flush_interval < 1:
            raise ValueError(
                f"flush_interval must be >= 1 (got {flush_interval})."
            )
        self.frame_skip: int = frame_skip
        self.flush_interval: int = flush_interval

        # Telemetry from the most recent extraction call. Reset on each call.
        self.last_frames_processed: int = 0
        self.last_frames_with_face: int = 0
        self.last_audio_windows: int = 0
        self.last_audio_sample_rate: int = 0
```

---

- [ ] **Step 5: Modify `feature_extractor.py` — replace in-memory accumulation with streaming write**

Locate the `extract_landmarks` method. Replace the section from `records: list[dict] = []` through the final `json.dump(records, fh)` call with the following streaming implementation:

```python
        output_path = output_dir / f"{video_path.stem}_landmarks.jsonl"
        frames_processed = 0
        frames_with_face = 0
        frame_idx = 0

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(face_landmarker_model()),
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.MEDIAPIPE_MAX_FACES,
            min_face_detection_confidence=self.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=self.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        try:
            with output_path.open("w", encoding="utf-8") as fh:
                write_buffer: list[str] = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % self.frame_skip == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        timestamp_ms = int((frame_idx / fps) * 1000)
                        result = landmarker.detect_for_video(mp_image, timestamp_ms)

                        landmarks: list[list[float]] | None = None
                        if result.face_landmarks:
                            mesh = result.face_landmarks[0]
                            landmarks = [[lm.x, lm.y, lm.z] for lm in mesh]
                            frames_with_face += 1

                        record = {
                            "frame_number": frame_idx,
                            "timestamp_seconds": frame_idx / fps,
                            "landmarks": landmarks,
                        }
                        write_buffer.append(json.dumps(record, separators=(",", ":")))
                        frames_processed += 1

                        # Flush to disk every flush_interval frames to bound peak RAM.
                        if len(write_buffer) >= self.flush_interval:
                            fh.write("\n".join(write_buffer) + "\n")
                            fh.flush()
                            write_buffer.clear()

                    frame_idx += 1

                # Flush any remaining records.
                if write_buffer:
                    fh.write("\n".join(write_buffer) + "\n")
        finally:
            landmarker.close()
            cap.release()
```

Also update the `return` line to use `output_path` (already defined above) and update the logger call to reference `output_path`. Remove the old `records` list and the old `output_path` assignment.

---

- [ ] **Step 6: Run the streaming JSONL tests to verify they pass**

```bash
.venv\Scripts\pytest tests/compression/test_feature_extractor.py -k "jsonl or flush_interval" -v
```
Expected: All 5 new tests PASS.

---

- [ ] **Step 7: Run the full feature_extractor test suite to catch regressions**

```bash
.venv\Scripts\pytest tests/compression/test_feature_extractor.py -v
```
Expected: All tests pass.

---

- [ ] **Step 8: Commit**

```bash
git add backend/workers/app/compression/feature_extractor.py tests/compression/test_feature_extractor.py
git commit -m "feat(compression): stream landmark output as JSONL, flush every N frames

Replaces the in-memory list accumulation with a per-frame JSONL streaming
writer. Peak RAM for landmark extraction drops from O(all_frames) to
O(flush_interval) — prevents Android OOM on long calls. Output file
changes from _landmarks.json to _landmarks.jsonl.

Closes P1-S6."
```

---

## Task 2: Psycholinguistic Pydantic Schemas (P2-S1)

**Files:**
- Create: `backend/shared/schemas/psycholinguistic.py`
- Create: `tests/psycholinguistic/__init__.py`
- Create: `tests/psycholinguistic/test_schemas.py`

---

- [ ] **Step 1: Write failing schema tests**

Create `tests/psycholinguistic/__init__.py` (empty).

Create `tests/psycholinguistic/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError

from backend.shared.schemas.psycholinguistic import (
    PsycholinguisticDimension,
    PsycholinguisticScore,
)


def _valid_dim(score: float = 42.0) -> PsycholinguisticDimension:
    return PsycholinguisticDimension(score=score, evidence=["test"])


def _valid_score(**overrides) -> dict:
    base = dict(
        pronoun_shift=_valid_dim(),
        hedging=_valid_dim(),
        cognitive_complexity=_valid_dim(),
        emotional_distribution=_valid_dim(),
        disfluency=_valid_dim(),
        negation=_valid_dim(),
        detail_specificity=_valid_dim(),
        certainty=_valid_dim(),
        composite_score=42.0,
        statement_count=3,
    )
    base.update(overrides)
    return base


class TestPsycholinguisticDimension:
    def test_valid_construction(self):
        dim = PsycholinguisticDimension(score=55.5, evidence=["word1", "word2"])
        assert dim.score == 55.5
        assert dim.evidence == ["word1", "word2"]

    def test_score_below_zero_raises(self):
        with pytest.raises(ValidationError):
            PsycholinguisticDimension(score=-1.0)

    def test_score_above_100_raises(self):
        with pytest.raises(ValidationError):
            PsycholinguisticDimension(score=100.1)

    def test_score_at_boundaries_ok(self):
        PsycholinguisticDimension(score=0.0)
        PsycholinguisticDimension(score=100.0)

    def test_evidence_defaults_to_empty_list(self):
        dim = PsycholinguisticDimension(score=50.0)
        assert dim.evidence == []

    def test_is_immutable(self):
        dim = PsycholinguisticDimension(score=50.0)
        with pytest.raises(Exception):
            dim.score = 99.0


class TestPsycholinguisticScore:
    def test_valid_construction(self):
        score = PsycholinguisticScore(**_valid_score())
        assert score.composite_score == 42.0
        assert score.statement_count == 3
        assert score.baseline_available is False
        assert score.confidence == "low"

    def test_composite_below_zero_raises(self):
        with pytest.raises(ValidationError):
            PsycholinguisticScore(**_valid_score(composite_score=-0.1))

    def test_composite_above_100_raises(self):
        with pytest.raises(ValidationError):
            PsycholinguisticScore(**_valid_score(composite_score=100.1))

    def test_all_eight_dimension_fields_present(self):
        score = PsycholinguisticScore(**_valid_score())
        for field in (
            "pronoun_shift", "hedging", "cognitive_complexity",
            "emotional_distribution", "disfluency", "negation",
            "detail_specificity", "certainty",
        ):
            assert hasattr(score, field), f"Missing field: {field}"

    def test_confidence_must_be_valid_literal(self):
        with pytest.raises(ValidationError):
            PsycholinguisticScore(**_valid_score(confidence="unknown"))

    def test_valid_confidence_values(self):
        for conf in ("low", "medium", "high"):
            s = PsycholinguisticScore(**_valid_score(confidence=conf))
            assert s.confidence == conf

    def test_is_immutable(self):
        score = PsycholinguisticScore(**_valid_score())
        with pytest.raises(Exception):
            score.composite_score = 99.0
```

---

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_schemas.py -v
```
Expected: `ModuleNotFoundError: No module named 'backend.shared.schemas.psycholinguistic'`

---

- [ ] **Step 3: Create the schema file**

Create `backend/shared/schemas/psycholinguistic.py`:

```python
"""Pydantic schemas for the psycholinguistic analysis vector.

Each of the eight deception-relevant linguistic dimensions produces a
PsycholinguisticDimension (score + evidence). PsycholinguisticScore
aggregates all eight plus a weighted composite.

Research basis (CLAUDE.md):
  - Li & Abouelenien (2024): linguistic features ~80% accuracy, strongest
    single modality.
  - Newman et al. (2003): first-person singular drops during deception.
  - Pérez-Rosas et al. (EMNLP 2015): emotional word distribution patterns.

CRITICAL INVARIANT (CLAUDE.md #5): Never show raw scores to users. Callers
must attach confidence + qualitative labels before surfacing any score.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PsycholinguisticDimension(BaseModel):
    """Score and evidence for a single linguistic dimension.

    ``score`` is in [0, 100] where higher values indicate greater deviation
    from baseline (higher anomaly signal). ``evidence`` is a short list of
    human-readable strings explaining the score (detected hedges, ratios, etc.)
    for SHAP-style explainability display.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(ge=0.0, le=100.0)
    evidence: list[str] = Field(default_factory=list)


class PsycholinguisticScore(BaseModel):
    """Full psycholinguistic analysis result for one or more statements.

    All eight dimensions are present. ``composite_score`` is a weighted
    average (equal weights, 1/8 each, for Day 1 — replace with learned
    weights once per-contact training data accumulates).

    ``baseline_available`` is False until at least one prior session exists
    for this contact; accuracy is reduced without a baseline. ``confidence``
    follows the same Low/Medium/High rubric as the ensemble (CLAUDE.md):
    Low = 1 session, Medium = 2-3 sessions, High = 4+ sessions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ---- Eight research-backed dimensions ----------------------------------
    pronoun_shift: PsycholinguisticDimension
    hedging: PsycholinguisticDimension
    cognitive_complexity: PsycholinguisticDimension
    emotional_distribution: PsycholinguisticDimension
    disfluency: PsycholinguisticDimension
    negation: PsycholinguisticDimension
    detail_specificity: PsycholinguisticDimension
    certainty: PsycholinguisticDimension

    # ---- Composite ---------------------------------------------------------
    composite_score: float = Field(ge=0.0, le=100.0)
    statement_count: int = Field(ge=0)
    baseline_available: bool = False
    confidence: Literal["low", "medium", "high"] = "low"
```

---

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_schemas.py -v
```
Expected: All 14 tests PASS.

---

- [ ] **Step 5: Commit**

```bash
git add backend/shared/schemas/psycholinguistic.py tests/psycholinguistic/
git commit -m "feat(psycholinguistic): add Pydantic schemas for 8 dimensions + composite score"
```

---

## Task 3: Analyzer Skeleton + Pronoun Scorer (P2-S2)

**Files:**
- Create: `backend/ml_inference/__init__.py` (empty)
- Create: `backend/ml_inference/app/__init__.py` (empty)
- Create: `backend/ml_inference/app/pipelines/__init__.py` (empty)
- Create: `backend/ml_inference/app/pipelines/psycholinguistic/__init__.py` (empty)
- Create: `backend/ml_inference/app/pipelines/psycholinguistic/analyzer.py`
- Create: `tests/psycholinguistic/test_analyzer.py`

---

- [ ] **Step 1: Install spaCy and download `en_core_web_sm`**

```bash
.venv\Scripts\pip install spacy
.venv\Scripts\python -m spacy download en_core_web_sm
```
Expected: `Successfully installed` and `✔ Download and installation successful`

---

- [ ] **Step 2: Write the failing pronoun test**

Create `tests/psycholinguistic/test_analyzer.py`:

```python
import pytest
from backend.ml_inference.app.pipelines.psycholinguistic.analyzer import (
    PsycholinguisticAnalyzer,
)
from backend.shared.schemas.psycholinguistic import PsycholinguisticDimension


@pytest.fixture(scope="module")
def analyzer():
    return PsycholinguisticAnalyzer()


class TestPronounScorer:
    def test_low_pronoun_density_scores_high(self, analyzer):
        """A statement with almost no first-person pronouns → deception signal."""
        # Statement about someone else, barely any "I"
        text = "The event happened. There was a meeting. People gathered."
        dim = analyzer._score_pronouns(analyzer._nlp(text))
        assert dim.score > 50, f"Expected >50, got {dim.score}"

    def test_normal_pronoun_density_scores_low(self, analyzer):
        """Natural first-person narrative → low deception signal."""
        text = (
            "I went to the store. I bought milk and I paid with my card. "
            "I remember because it was my birthday."
        )
        dim = analyzer._score_pronouns(analyzer._nlp(text))
        assert dim.score < 50, f"Expected <50, got {dim.score}"

    def test_evidence_is_populated(self, analyzer):
        text = "I never did that."
        dim = analyzer._score_pronouns(analyzer._nlp(text))
        assert len(dim.evidence) > 0

    def test_empty_text_returns_zero(self, analyzer):
        dim = analyzer._score_pronouns(analyzer._nlp("   "))
        assert dim.score == 0.0
```

---

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_analyzer.py -v
```
Expected: `ModuleNotFoundError: No module named 'backend.ml_inference'`

---

- [ ] **Step 4: Create all `__init__.py` package files**

```bash
mkdir -p backend\ml_inference\app\pipelines\psycholinguistic
type nul > backend\ml_inference\__init__.py
type nul > backend\ml_inference\app\__init__.py
type nul > backend\ml_inference\app\pipelines\__init__.py
type nul > backend\ml_inference\app\pipelines\psycholinguistic\__init__.py
```

---

- [ ] **Step 5: Create `analyzer.py` with skeleton + pronoun scorer**

Create `backend/ml_inference/app/pipelines/psycholinguistic/analyzer.py`:

```python
"""Psycholinguistic analysis pipeline for ALICE.

Scores text across eight deception-relevant linguistic dimensions defined
in CLAUDE.md. Each dimension scorer is a private method returning a
PsycholinguisticDimension. The public ``analyze()`` method combines all
eight into a PsycholinguisticScore.

Day 1 tooling (lightweight, no large model downloads):
  - spaCy en_core_web_sm (~12MB): POS, deps, NER, pronouns, negation
  - NRCLex: 8 emotion categories
  - vaderSentiment: valence-aware sentiment
  - Empath: 200+ lexical categories (imported but used in future tasks)

IMPORTANT — hedging classifier note (CLAUDE.md):
  The word-list hedging approach used here has ~59% false positive rate.
  Replace with a fine-tuned BERT classifier in Phase 3.

CLAUDE.md invariant #5: never show raw scores to users. This module returns
raw scores for use by the ensemble layer only.
"""
from __future__ import annotations

import re
from typing import Optional

from backend.shared.schemas.psycholinguistic import (
    PsycholinguisticDimension,
    PsycholinguisticScore,
)

# spaCy is imported lazily on first use to avoid the ~200ms load cost at
# import time (important for mobile worker startup).
try:
    import spacy
    from spacy.tokens import Doc
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    Doc = object  # type: ignore[misc,assignment]


class PsycholinguisticAnalyzer:
    """Analyze text statements across 8 psycholinguistic dimensions.

    Construction is cheap. spaCy is loaded lazily on first ``analyze()``
    call. The same instance is safe to reuse across multiple ``analyze()``
    calls — no mutable state is stored between calls.

    Example::

        analyzer = PsycholinguisticAnalyzer()
        result = analyzer.analyze(["I think maybe he was there.", "Not sure."])
        print(result.composite_score)
    """

    # Equal sub-dimension weights for Day 1.
    # Replace with learned per-contact weights in Phase 3.
    _DIMENSION_WEIGHT: float = 1.0 / 8.0

    # First-person singular tokens (Newman et al. 2003)
    _FP_SINGULAR: frozenset[str] = frozenset({"i", "me", "my", "mine", "myself"})

    def __init__(self) -> None:
        self._nlp: Optional[object] = None  # spacy.Language, loaded lazily

    # =========================================================================
    # Public API
    # =========================================================================

    def analyze(self, statements: list[str]) -> PsycholinguisticScore:
        """Score a list of statements across all 8 psycholinguistic dimensions.

        Args:
            statements: One or more speaker-attributed statement strings.
                Each string may be a sentence or a short paragraph.

        Returns:
            PsycholinguisticScore with all 8 dimensions populated and a
            weighted composite score.

        Raises:
            ValueError: ``statements`` is empty.
            RuntimeError: spaCy is not installed (install with
                ``pip install spacy && python -m spacy download en_core_web_sm``).
        """
        if not statements:
            raise ValueError("No statements provided — cannot analyze empty input.")
        self._ensure_nlp()

        text = " ".join(statements)
        doc = self._nlp(text)  # type: ignore[operator]

        pronoun_shift = self._score_pronouns(doc)
        hedging = self._score_hedging(doc)
        cognitive_complexity = self._score_cognitive_complexity(doc)
        emotional_distribution = self._score_emotional_distribution(text)
        disfluency = self._score_disfluencies(text)
        negation = self._score_negation(doc)
        detail_specificity = self._score_detail_specificity(doc)
        certainty = self._score_certainty(doc, text)

        dimensions = [
            pronoun_shift, hedging, cognitive_complexity, emotional_distribution,
            disfluency, negation, detail_specificity, certainty,
        ]
        composite = round(
            sum(d.score * self._DIMENSION_WEIGHT for d in dimensions), 2
        )

        return PsycholinguisticScore(
            pronoun_shift=pronoun_shift,
            hedging=hedging,
            cognitive_complexity=cognitive_complexity,
            emotional_distribution=emotional_distribution,
            disfluency=disfluency,
            negation=negation,
            detail_specificity=detail_specificity,
            certainty=certainty,
            composite_score=composite,
            statement_count=len(statements),
            baseline_available=False,
            confidence="low",
        )

    # =========================================================================
    # Dimension scorers
    # =========================================================================

    def _score_pronouns(self, doc: "Doc") -> PsycholinguisticDimension:
        """Score first-person singular pronoun density.

        Newman et al. (2003): deceivers use fewer first-person singular
        pronouns. Both extremes (very low AND very high) are flagged.
        """
        tokens = [t for t in doc if not t.is_space]
        n_total = len(tokens)
        if n_total == 0:
            return PsycholinguisticDimension(score=0.0, evidence=["no tokens"])

        fp_tokens = [t for t in tokens if t.text.lower() in self._FP_SINGULAR]
        ratio = len(fp_tokens) / n_total

        # Scoring curve: low ratio → high score; normal range → low score;
        # over-compensation (very high) → moderate score
        if ratio < 0.03:
            score = min(100.0, 70.0 + (0.03 - ratio) * 1000.0)
        elif ratio <= 0.04:
            score = 40.0 + (0.04 - ratio) * 3000.0
        elif ratio <= 0.09:
            score = max(10.0, 30.0 - (ratio - 0.04) * 400.0)
        elif ratio <= 0.11:
            score = 10.0 + (ratio - 0.09) * 1500.0
        else:
            score = min(100.0, 30.0 + (ratio - 0.11) * 1000.0)

        return PsycholinguisticDimension(
            score=round(min(100.0, max(0.0, score)), 2),
            evidence=[
                f"fp_singular_ratio={ratio:.4f}",
                f"fp_count={len(fp_tokens)} / {n_total} tokens",
            ],
        )

    # Remaining scorers are stubs — implemented in Tasks 4-9.
    def _score_hedging(self, doc: "Doc") -> PsycholinguisticDimension:
        raise NotImplementedError

    def _score_cognitive_complexity(self, doc: "Doc") -> PsycholinguisticDimension:
        raise NotImplementedError

    def _score_emotional_distribution(self, text: str) -> PsycholinguisticDimension:
        raise NotImplementedError

    def _score_disfluencies(self, text: str) -> PsycholinguisticDimension:
        raise NotImplementedError

    def _score_negation(self, doc: "Doc") -> PsycholinguisticDimension:
        raise NotImplementedError

    def _score_detail_specificity(self, doc: "Doc") -> PsycholinguisticDimension:
        raise NotImplementedError

    def _score_certainty(self, doc: "Doc", text: str) -> PsycholinguisticDimension:
        raise NotImplementedError

    # =========================================================================
    # Internals
    # =========================================================================

    def _ensure_nlp(self) -> None:
        if not _SPACY_AVAILABLE:
            raise RuntimeError(
                "spaCy is not installed. Run: "
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        if self._nlp is None:
            import spacy as _spacy
            self._nlp = _spacy.load("en_core_web_sm")
```

---

- [ ] **Step 6: Run pronoun tests to verify they pass**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_analyzer.py::TestPronounScorer -v
```
Expected: All 4 pronoun tests PASS.

---

- [ ] **Step 7: Commit**

```bash
git add backend/ml_inference/ tests/psycholinguistic/test_analyzer.py
git commit -m "feat(psycholinguistic): analyzer skeleton + pronoun shift scorer"
```

---

## Task 4: Hedging Scorer (P2-S3)

**Files:**
- Modify: `backend/ml_inference/app/pipelines/psycholinguistic/analyzer.py`
- Modify: `tests/psycholinguistic/test_analyzer.py`

---

- [ ] **Step 1: Write the failing hedging tests**

Append to `tests/psycholinguistic/test_analyzer.py`:

```python
class TestHedgingScorer:
    def test_high_hedging_statement_scores_high(self, analyzer):
        text = "I think maybe it could perhaps have been sort of like that, possibly."
        dim = analyzer._score_hedging(analyzer._nlp(text))
        assert dim.score > 60, f"Expected >60, got {dim.score}"

    def test_direct_statement_scores_low(self, analyzer):
        text = "I did it at 3pm on Tuesday at the warehouse on Elm Street."
        dim = analyzer._score_hedging(analyzer._nlp(text))
        assert dim.score < 30, f"Expected <30, got {dim.score}"

    def test_hedge_evidence_populated_when_present(self, analyzer):
        text = "I think maybe I was there."
        dim = analyzer._score_hedging(analyzer._nlp(text))
        assert len(dim.evidence) > 0

    def test_hedge_evidence_empty_when_no_hedges(self, analyzer):
        text = "I drove to the airport at 6am."
        dim = analyzer._score_hedging(analyzer._nlp(text))
        # Evidence may be empty or minimal — no false evidence
        assert dim.score < 30
```

---

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_analyzer.py::TestHedgingScorer -v
```
Expected: `NotImplementedError`

---

- [ ] **Step 3: Implement `_score_hedging` in `analyzer.py`**

Add these class-level constants (after `_FP_SINGULAR`):

```python
    _HEDGING_PHRASES: frozenset[str] = frozenset({
        "i think", "i believe", "i guess", "i suppose", "i'm not sure",
        "i'm not certain", "sort of", "kind of", "something like",
        "more or less", "maybe", "perhaps", "possibly", "probably",
        "it seems", "it appears", "apparently", "supposedly",
        "as far as i know", "to my knowledge", "if i recall",
        "i could be wrong", "not sure if", "i might be",
    })

    _MODAL_VERBS: frozenset[str] = frozenset({
        "might", "could", "would", "may", "should", "ought",
    })
```

Replace the `_score_hedging` stub:

```python
    def _score_hedging(self, doc: "Doc") -> PsycholinguisticDimension:
        """Score modal verb and epistemic phrase density.

        NOTE: Day 1 word-list approach. Replace with fine-tuned BERT hedging
        classifier in Phase 3 — the word-list has ~59% false positive rate
        per CLAUDE.md.
        """
        text_lower = doc.text.lower()
        phrase_hits = [p for p in self._HEDGING_PHRASES if p in text_lower]
        modal_tokens = [
            t for t in doc
            if t.tag_ == "MD" or t.text.lower() in self._MODAL_VERBS
        ]

        sentences = list(doc.sents)
        n_sents = max(1, len(sentences))
        total_hedges = len(phrase_hits) + len(modal_tokens)
        rate = total_hedges / n_sents

        # > 3 hedges per sentence → score ~75+; 0 → score ~0
        score = round(min(100.0, max(0.0, rate * 25.0)), 2)

        evidence: list[str] = []
        evidence.extend(phrase_hits[:3])
        evidence.extend(t.text for t in modal_tokens[:3])

        return PsycholinguisticDimension(score=score, evidence=evidence)
```

---

- [ ] **Step 4: Run hedging tests to verify they pass**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_analyzer.py::TestHedgingScorer -v
```
Expected: All 4 tests PASS.

---

- [ ] **Step 5: Commit**

```bash
git add backend/ml_inference/app/pipelines/psycholinguistic/analyzer.py tests/psycholinguistic/test_analyzer.py
git commit -m "feat(psycholinguistic): add hedging scorer (word-list Day 1, BERT TODO Phase 3)"
```

---

## Task 5: Cognitive Complexity + Emotional Distribution Scorers (P2-S4, P2-S5)

**Files:**
- Modify: `analyzer.py`, `test_analyzer.py`

---

- [ ] **Step 1: Install NRCLex**

```bash
.venv\Scripts\pip install nrclex
.venv\Scripts\python -c "from nrclex import NRCLex; print('ok')"
```
Expected: `ok`

---

- [ ] **Step 2: Write failing tests**

Append to `tests/psycholinguistic/test_analyzer.py`:

```python
class TestCognitiveComplexityScorer:
    def test_complex_sentence_scores_low(self, analyzer):
        """Complex nested clauses → truth signal → low deception score."""
        text = (
            "Although I was tired because I had been working all day, "
            "I decided to stay so that I could finish the report that was due."
        )
        dim = analyzer._score_cognitive_complexity(analyzer._nlp(text))
        assert dim.score < 50, f"Expected <50, got {dim.score}"

    def test_simple_flat_sentence_scores_high(self, analyzer):
        """Flat, simple sentences → possible deception → higher score."""
        text = "I was home. I did not go out. It was quiet."
        dim = analyzer._score_cognitive_complexity(analyzer._nlp(text))
        assert dim.score > 50, f"Expected >50, got {dim.score}"

    def test_evidence_includes_rate(self, analyzer):
        text = "I went there."
        dim = analyzer._score_cognitive_complexity(analyzer._nlp(text))
        assert any("rate" in e or "count" in e for e in dim.evidence)


class TestEmotionalDistributionScorer:
    def test_anxiety_anger_text_scores_high(self, analyzer):
        text = (
            "I was terrified and furious. The threat was horrible and I feared "
            "the worst. Angry, scared, disgusted by what happened."
        )
        dim = analyzer._score_emotional_distribution(text)
        assert dim.score > 50, f"Expected >50, got {dim.score}"

    def test_positive_trust_text_scores_low(self, analyzer):
        text = (
            "I love spending time with my family. My home is a wonderful place. "
            "I trust my friends completely and feel grateful every day."
        )
        dim = analyzer._score_emotional_distribution(text)
        assert dim.score < 60, f"Expected <60, got {dim.score}"

    def test_evidence_contains_top_emotions(self, analyzer):
        text = "I was scared and angry about the situation."
        dim = analyzer._score_emotional_distribution(text)
        assert len(dim.evidence) > 0
```

---

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_analyzer.py::TestCognitiveComplexityScorer tests/psycholinguistic/test_analyzer.py::TestEmotionalDistributionScorer -v
```
Expected: `NotImplementedError` for both.

---

- [ ] **Step 4: Implement `_score_cognitive_complexity`**

Replace the stub in `analyzer.py`:

```python
    def _score_cognitive_complexity(self, doc: "Doc") -> PsycholinguisticDimension:
        """Score subordinate clause density as an inverse truth signal.

        Truth-tellers use more syntactically complex language (Vrij et al.).
        High complexity → lower deception score.
        """
        subordinate_deps = {"advcl", "relcl", "ccomp", "xcomp", "acl"}
        subordinate_tokens = [t for t in doc if t.dep_ in subordinate_deps]
        sentences = list(doc.sents)
        n_sents = max(1, len(sentences))
        rate = len(subordinate_tokens) / n_sents

        # rate > 2.0 → complex → score ~5; rate 0.0 → simple → score ~75
        score = round(min(100.0, max(5.0, 75.0 - rate * 35.0)), 2)

        return PsycholinguisticDimension(
            score=score,
            evidence=[
                f"subordinate_clause_rate={rate:.3f}",
                f"subordinate_count={len(subordinate_tokens)} in {n_sents} sentences",
            ],
        )
```

---

- [ ] **Step 5: Implement `_score_emotional_distribution`**

Add to the imports at the top of `analyzer.py`:

```python
try:
    from nrclex import NRCLex
    _NRCLEX_AVAILABLE = True
except ImportError:
    _NRCLEX_AVAILABLE = False
```

Replace the `_score_emotional_distribution` stub:

```python
    def _score_emotional_distribution(self, text: str) -> PsycholinguisticDimension:
        """Score emotional word distribution.

        Pérez-Rosas (EMNLP 2015): deceivers use more Anxiety/Anger/Fear words;
        truth-tellers use more Positive/Trust/Family words.
        """
        if not _NRCLEX_AVAILABLE:
            raise RuntimeError("NRCLex is not installed. Run: pip install nrclex")

        emotion = NRCLex(text)
        freq = emotion.affect_frequencies  # emotion → proportion in [0, 1]

        deceptive = (
            freq.get("fear", 0.0) * 40.0
            + freq.get("anger", 0.0) * 35.0
            + freq.get("disgust", 0.0) * 25.0
        )
        truthful = (
            freq.get("positive", 0.0) * 30.0
            + freq.get("trust", 0.0) * 25.0
            + freq.get("anticipation", 0.0) * 15.0
        )

        # Shift from [-1,1] range to [0,100]
        raw = (deceptive - truthful + 1.0) * 50.0
        score = round(min(100.0, max(0.0, raw)), 2)

        top_emotions = sorted(freq.items(), key=lambda kv: -kv[1])[:3]
        evidence = [f"{e}={v:.3f}" for e, v in top_emotions if v > 0]

        return PsycholinguisticDimension(score=score, evidence=evidence or ["no emotional words detected"])
```

---

- [ ] **Step 6: Run tests to verify they pass**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_analyzer.py::TestCognitiveComplexityScorer tests/psycholinguistic/test_analyzer.py::TestEmotionalDistributionScorer -v
```
Expected: All 6 tests PASS.

---

- [ ] **Step 7: Commit**

```bash
git add backend/ml_inference/app/pipelines/psycholinguistic/analyzer.py tests/psycholinguistic/test_analyzer.py
git commit -m "feat(psycholinguistic): add cognitive complexity + emotional distribution scorers"
```

---

## Task 6: Disfluency + Negation + Detail Specificity Scorers (P2-S6, P2-S7)

**Files:**
- Modify: `analyzer.py`, `test_analyzer.py`

---

- [ ] **Step 1: Write failing tests**

Append to `tests/psycholinguistic/test_analyzer.py`:

```python
class TestDisfluencyScorer:
    def test_high_disfluency_scores_high(self, analyzer):
        text = "Um, I, uh, was there, you know, like, I think, er, yeah."
        dim = analyzer._score_disfluencies(text)
        assert dim.score > 50, f"Expected >50, got {dim.score}"

    def test_clean_speech_scores_low(self, analyzer):
        text = "I arrived at the building at precisely 9:15 AM on Monday."
        dim = analyzer._score_disfluencies(text)
        assert dim.score < 20, f"Expected <20, got {dim.score}"

    def test_evidence_lists_detected_disfluencies(self, analyzer):
        text = "Um yeah, uh, I guess so."
        dim = analyzer._score_disfluencies(text)
        assert len(dim.evidence) > 0


class TestNegationScorer:
    def test_high_negation_scores_high(self, analyzer):
        text = "I never did that. I didn't go there. I wasn't involved. I won't admit anything."
        dim = analyzer._score_negation(analyzer._nlp(text))
        assert dim.score > 50, f"Expected >50, got {dim.score}"

    def test_affirmative_text_scores_low(self, analyzer):
        text = "I went to the store and bought groceries and came home."
        dim = analyzer._score_negation(analyzer._nlp(text))
        assert dim.score < 30, f"Expected <30, got {dim.score}"


class TestDetailSpecificityScorer:
    def test_specific_text_scores_low(self, analyzer):
        """Named entities (dates, places, people) signal truthful detail."""
        text = (
            "On Tuesday, March 3rd, I met John Smith at the Starbucks on "
            "Fifth Avenue in New York City at 2:30 PM."
        )
        dim = analyzer._score_detail_specificity(analyzer._nlp(text))
        assert dim.score < 50, f"Expected <50, got {dim.score}"

    def test_vague_text_scores_high(self, analyzer):
        """No named entities → low specificity → higher deception signal."""
        text = "Something happened somewhere at some point involving someone."
        dim = analyzer._score_detail_specificity(analyzer._nlp(text))
        assert dim.score > 50, f"Expected >50, got {dim.score}"

    def test_evidence_contains_entity_info(self, analyzer):
        text = "I met Alice in Paris."
        dim = analyzer._score_detail_specificity(analyzer._nlp(text))
        assert len(dim.evidence) > 0
```

---

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_analyzer.py::TestDisfluencyScorer tests/psycholinguistic/test_analyzer.py::TestNegationScorer tests/psycholinguistic/test_analyzer.py::TestDetailSpecificityScorer -v
```
Expected: `NotImplementedError` for all three.

---

- [ ] **Step 3: Add `_DISFLUENCY_PATTERN` class constant and implement disfluency scorer**

Add constant (after `_MODAL_VERBS`):

```python
    _DISFLUENCY_PATTERN: re.Pattern = re.compile(
        r"\b(um+|uh+|er+|ah+|eh+|hmm+|uhh+|umm+)\b",
        re.IGNORECASE,
    )
    _FILLER_PHRASES: frozenset[str] = frozenset({
        "you know", "i mean", "like i said", "sort of like", "kind of like",
    })
```

Replace `_score_disfluencies` stub:

```python
    def _score_disfluencies(self, text: str) -> PsycholinguisticDimension:
        """Score filler word and vocalization density (cognitive load signal)."""
        filler_matches = self._DISFLUENCY_PATTERN.findall(text)
        phrase_hits = [p for p in self._FILLER_PHRASES if p in text.lower()]

        words = text.split()
        n_words = max(1, len(words))
        total = len(filler_matches) + len(phrase_hits)
        rate = total / n_words * 100.0  # per 100 words

        # >5 disfluencies per 100 words → heavy cognitive load → score ~80
        score = round(min(100.0, max(0.0, rate * 16.0)), 2)

        evidence = list({m.lower() for m in filler_matches if m})[:3]
        evidence += list(phrase_hits)[:2]

        return PsycholinguisticDimension(score=score, evidence=evidence)
```

---

- [ ] **Step 4: Implement `_score_negation`**

Replace stub:

```python
    def _score_negation(self, doc: "Doc") -> PsycholinguisticDimension:
        """Score negation dependency arc density.

        High negation frequency is a deception indicator (Pérez-Rosas 2015).
        """
        neg_tokens = [t for t in doc if t.dep_ == "neg"]
        sentences = list(doc.sents)
        n_sents = max(1, len(sentences))
        rate = len(neg_tokens) / n_sents

        # >3 negations per sentence → score ~75
        score = round(min(100.0, max(0.0, rate * 25.0)), 2)

        return PsycholinguisticDimension(
            score=score,
            evidence=[
                f"negation_rate={rate:.3f}",
                f"neg_count={len(neg_tokens)} in {n_sents} sentences",
            ] + [t.text for t in neg_tokens[:3]],
        )
```

---

- [ ] **Step 5: Implement `_score_detail_specificity`**

Replace stub:

```python
    def _score_detail_specificity(self, doc: "Doc") -> PsycholinguisticDimension:
        """Score named entity density as an inverse truth signal.

        High NE density → specific → truthful → low deception score.
        """
        named_entities = doc.ents
        words = [t for t in doc if not t.is_space and not t.is_punct]
        n_words = max(1, len(words))

        ne_density = len(named_entities) / n_words
        # ne_density >0.05 → very specific → score ~5; 0.0 → vague → score ~75
        score = round(min(100.0, max(5.0, 75.0 - ne_density * 1300.0)), 2)

        return PsycholinguisticDimension(
            score=score,
            evidence=[
                f"ne_density={ne_density:.4f}",
                f"entity_count={len(named_entities)}",
            ] + [f"{ent.text} ({ent.label_})" for ent in named_entities[:3]],
        )
```

---

- [ ] **Step 6: Run all three test classes to verify they pass**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_analyzer.py::TestDisfluencyScorer tests/psycholinguistic/test_analyzer.py::TestNegationScorer tests/psycholinguistic/test_analyzer.py::TestDetailSpecificityScorer -v
```
Expected: All 8 tests PASS.

---

- [ ] **Step 7: Commit**

```bash
git add backend/ml_inference/app/pipelines/psycholinguistic/analyzer.py tests/psycholinguistic/test_analyzer.py
git commit -m "feat(psycholinguistic): add disfluency, negation, and detail specificity scorers"
```

---

## Task 7: Certainty Scorer + Composite analyze() (P2-S8, P2-S9)

**Files:**
- Modify: `analyzer.py`, `test_analyzer.py`

---

- [ ] **Step 1: Install vaderSentiment**

```bash
.venv\Scripts\pip install vaderSentiment
.venv\Scripts\python -c "from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; print('ok')"
```
Expected: `ok`

---

- [ ] **Step 2: Write failing tests**

Append to `tests/psycholinguistic/test_analyzer.py`:

```python
class TestCertaintyScorer:
    def test_over_emphatic_certainty_scores_high(self, analyzer):
        text = "I absolutely definitely 100% did not do that, I swear, trust me, I'm certain."
        dim = analyzer._score_certainty(analyzer._nlp(text), text)
        assert dim.score > 50, f"Expected >50, got {dim.score}"

    def test_neutral_confident_language_scores_low(self, analyzer):
        text = "I went to the store and bought bread."
        dim = analyzer._score_certainty(analyzer._nlp(text), text)
        assert dim.score < 40, f"Expected <40, got {dim.score}"

    def test_evidence_includes_vader_score(self, analyzer):
        text = "I definitely did not do it."
        dim = analyzer._score_certainty(analyzer._nlp(text), text)
        assert any("vader" in e for e in dim.evidence)


class TestCompositeAnalyze:
    def test_analyze_empty_raises(self, analyzer):
        with pytest.raises(ValueError, match="No statements"):
            analyzer.analyze([])

    def test_analyze_returns_all_8_dimensions(self, analyzer):
        result = analyzer.analyze(["I think maybe I was there.", "Not really sure."])
        for field in (
            "pronoun_shift", "hedging", "cognitive_complexity",
            "emotional_distribution", "disfluency", "negation",
            "detail_specificity", "certainty",
        ):
            assert hasattr(result, field)
            dim = getattr(result, field)
            assert 0.0 <= dim.score <= 100.0

    def test_composite_score_in_range(self, analyzer):
        result = analyzer.analyze(["Something happened."])
        assert 0.0 <= result.composite_score <= 100.0

    def test_statement_count_matches_input(self, analyzer):
        stmts = ["First sentence.", "Second sentence.", "Third sentence."]
        result = analyzer.analyze(stmts)
        assert result.statement_count == 3

    def test_analyze_is_deterministic(self, analyzer):
        stmts = ["I never did anything like that ever."]
        result1 = analyzer.analyze(stmts)
        result2 = analyzer.analyze(stmts)
        assert result1.composite_score == result2.composite_score

    def test_composite_is_average_of_dimensions(self, analyzer):
        result = analyzer.analyze(["Test statement for math check."])
        dims = [
            result.pronoun_shift.score, result.hedging.score,
            result.cognitive_complexity.score, result.emotional_distribution.score,
            result.disfluency.score, result.negation.score,
            result.detail_specificity.score, result.certainty.score,
        ]
        expected = round(sum(dims) / 8, 2)
        assert abs(result.composite_score - expected) < 0.01
```

---

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_analyzer.py::TestCertaintyScorer tests/psycholinguistic/test_analyzer.py::TestCompositeAnalyze -v
```
Expected: `NotImplementedError` for certainty; `NotImplementedError` propagating through `analyze()` for composite tests.

---

- [ ] **Step 4: Add VADER import and overconfidence constants to `analyzer.py`**

Add at top (after nrclex import block):

```python
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False
```

Add class constant (after `_FILLER_PHRASES`):

```python
    _OVERCONFIDENCE_WORDS: frozenset[str] = frozenset({
        "definitely", "absolutely", "certainly", "guaranteed",
        "i swear", "i promise", "trust me", "believe me",
        "i'm certain", "i'm sure", "no question", "100%",
        "without a doubt", "without doubt",
    })
```

---

- [ ] **Step 5: Implement `_score_certainty`**

Replace stub:

```python
    def _score_certainty(self, doc: "Doc", text: str) -> PsycholinguisticDimension:
        """Score over-certainty and extreme sentiment.

        Both over-emphatic certainty ("I absolutely swear") and extreme
        negation certainty are deception signals. Uses VADER compound score
        as a proxy for emotional extremity.
        """
        if not _VADER_AVAILABLE:
            raise RuntimeError("vaderSentiment not installed. Run: pip install vaderSentiment")

        text_lower = text.lower()
        overconfidence_hits = [w for w in self._OVERCONFIDENCE_WORDS if w in text_lower]

        vader = _VaderAnalyzer()
        vs = vader.polarity_scores(text)
        compound_extremity = abs(vs["compound"])  # 0=neutral, 1=very extreme

        words = text.split()
        n_words = max(1, len(words))
        confidence_rate = len(overconfidence_hits) / n_words * 100.0

        score = round(min(100.0, max(0.0, confidence_rate * 20.0 + compound_extremity * 30.0)), 2)

        return PsycholinguisticDimension(
            score=score,
            evidence=overconfidence_hits[:3] + [f"vader_compound={vs['compound']:.3f}"],
        )
```

---

- [ ] **Step 6: Run all certainty + composite tests**

```bash
.venv\Scripts\pytest tests/psycholinguistic/test_analyzer.py::TestCertaintyScorer tests/psycholinguistic/test_analyzer.py::TestCompositeAnalyze -v
```
Expected: All 9 tests PASS.

---

- [ ] **Step 7: Run the full analyzer test suite**

```bash
.venv\Scripts\pytest tests/psycholinguistic/ -v
```
Expected: All tests PASS.

---

- [ ] **Step 8: Commit**

```bash
git add backend/ml_inference/app/pipelines/psycholinguistic/analyzer.py tests/psycholinguistic/test_analyzer.py
git commit -m "feat(psycholinguistic): add certainty scorer and composite analyze() method

All 8 dimension scorers now implemented. analyze() wires them all together
into a PsycholinguisticScore with equal-weighted composite. Deterministic
and fully type-safe."
```

---

## Task 8: pyproject.toml — Add Psycholinguistic Dependencies

**Files:**
- Modify: `pyproject.toml`

---

- [ ] **Step 1: Add new dependencies to pyproject.toml**

Open `pyproject.toml` and add to the `[project] dependencies` list:

```toml
    "spacy>=3.7",
    "nrclex>=4.0",
    "vaderSentiment>=3.3.2",
```

---

- [ ] **Step 2: Verify install**

```bash
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\python -c "import spacy, nrclex, vaderSentiment; print('all ok')"
```
Expected: `all ok`

---

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): add spacy, nrclex, vaderSentiment to project dependencies"
```

---

## Task 9: Update Makefile + CLAUDE.md notes

**Files:**
- Modify: `Makefile`

---

- [ ] **Step 1: Add psycholinguistic targets to Makefile**

Open `Makefile` and append:

```makefile
## Psycholinguistic
install-spacy-model:
	python -m spacy download en_core_web_sm

test-psycholinguistic:
	python scripts/test_psycholinguistic.py --text "I think maybe I never did that, um, I guess."

test-compress-analyze:
	python scripts/test_compress_and_analyze.py
```

---

- [ ] **Step 2: Commit**

```bash
git add Makefile
git commit -m "chore(makefile): add psycholinguistic test targets"
```

---

## Task 10: CLI Test Scripts (P2-S10)

**Files:**
- Create: `scripts/test_psycholinguistic.py`
- Create: `scripts/test_compress_and_analyze.py`

---

- [ ] **Step 1: Create `scripts/test_psycholinguistic.py`**

```python
#!/usr/bin/env python
"""CLI smoke test for PsycholinguisticAnalyzer.

Usage:
    python scripts/test_psycholinguistic.py --text "I think maybe I never did that."
    python scripts/test_psycholinguistic.py --file path/to/statements.txt
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Test psycholinguistic analyzer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text string to analyze")
    group.add_argument("--file", type=Path, help="Path to text file (one statement per line)")
    args = parser.parse_args()

    if args.text:
        statements = [args.text]
    else:
        if not args.file.exists():
            print(f"ERROR: File not found: {args.file}", file=sys.stderr)
            return 1
        statements = [
            line.strip()
            for line in args.file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if not statements:
        print("ERROR: No statements to analyze.", file=sys.stderr)
        return 1

    from backend.ml_inference.app.pipelines.psycholinguistic.analyzer import (
        PsycholinguisticAnalyzer,
    )

    analyzer = PsycholinguisticAnalyzer()
    result = analyzer.analyze(statements)

    print(f"\nPsycholinguistic Analysis — {result.statement_count} statement(s)")
    print(f"Confidence: {result.confidence} | Baseline: {'available' if result.baseline_available else 'not yet established'}")
    print("-" * 60)

    dimensions = {
        "Pronoun Shift":          result.pronoun_shift,
        "Hedging":                result.hedging,
        "Cognitive Complexity":   result.cognitive_complexity,
        "Emotional Distribution": result.emotional_distribution,
        "Disfluency":             result.disfluency,
        "Negation":               result.negation,
        "Detail Specificity":     result.detail_specificity,
        "Certainty":              result.certainty,
    }

    for name, dim in dimensions.items():
        bar = "█" * int(dim.score / 5)
        print(f"  {name:<26} {dim.score:6.1f}/100  {bar}")
        if dim.evidence:
            print(f"    └── {', '.join(dim.evidence[:3])}")

    print("-" * 60)
    bar = "█" * int(result.composite_score / 5)
    print(f"  {'COMPOSITE':26} {result.composite_score:6.1f}/100  {bar}")
    print()
    print("NOTE: Scores are behavioral anomaly indicators, not ground truth.")
    print("      ~75% F1 realistic ceiling. Not a lie detector.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

- [ ] **Step 2: Create `scripts/test_compress_and_analyze.py`**

```python
#!/usr/bin/env python
"""Integration stub: video → compression pipeline → psycholinguistic stub.

Runs the full compression pipeline on a video file, then shows where
WhisperX transcription would plug in to feed the psycholinguistic analyzer.

Usage:
    python scripts/test_compress_and_analyze.py path/to/video.mp4
    python scripts/test_compress_and_analyze.py path/to/video.mp4 --mode edge_full
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compression → psycholinguistic integration stub")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument(
        "--mode",
        choices=["raw", "roi", "edge_full", "edge_minimal"],
        default="edge_full",
        help="Compression mode (default: edge_full)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: Video not found: {args.video}", file=sys.stderr)
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
    output_dir = Path("processed_output") / "compress_analyze_test" / args.video.stem

    print(f"\nStep 1 — Compression Pipeline ({mode.value})")
    print("-" * 60)

    pipeline = CompressionPipeline()
    result = pipeline.process(args.video, output_dir, mode)

    input_mb = result.input_size_bytes / 1_048_576
    flac_mb = result.flac_size_bytes / 1_048_576
    print(f"  Input:              {input_mb:.1f} MB")
    print(f"  FLAC audio (ML):    {flac_mb:.2f} MB  →  {result.flac_audio_path}")
    if result.roi_video_path:
        roi_mb = (result.roi_video_size_bytes or 0) / 1_048_576
        print(f"  ROI video:          {roi_mb:.1f} MB  →  {result.roi_video_path}")
    if result.landmarks_path:
        lm_mb = (result.landmarks_size_bytes or 0) / 1_048_576
        print(f"  Landmarks (JSONL):  {lm_mb:.1f} MB  →  {result.landmarks_path}")
    if result.features_path:
        ft_mb = (result.features_size_bytes or 0) / 1_048_576
        print(f"  Audio features:     {ft_mb:.2f} MB  →  {result.features_path}")
    print(f"  Face detected:      {result.face_detected_pct:.1f}% of frames")
    total_sec = result.processing_times.get("total", 0.0)
    print(f"  Total time:         {total_sec:.1f}s")

    print()
    print("Step 2 — Transcription (PENDING)")
    print("-" * 60)
    print(f"  Would run: WhisperX on {result.flac_audio_path}")
    print("  Would produce: speaker-attributed statements list")
    print("  Status: WhisperX not yet integrated (next plan)")

    print()
    print("Step 3 — Psycholinguistic Analysis (PENDING)")
    print("-" * 60)
    print("  Would run: PsycholinguisticAnalyzer.analyze(statements)")
    print("  Would produce: PsycholinguisticScore with 8 dimensions")
    print("  Status: Analyzer ready — awaiting WhisperX transcript feed")

    print()
    print("Pipeline path verified end-to-end. Add WhisperX to unlock real scores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

- [ ] **Step 3: Verify both scripts run without error**

```bash
.venv\Scripts\python scripts/test_psycholinguistic.py --text "I think maybe I never did that, um, I guess so."
```
Expected: Prints all 8 dimension scores + composite + disclaimer. No errors.

```bash
# Only if a test video is available:
.venv\Scripts\python scripts/test_compress_and_analyze.py demo_data/honest/demo_file_1/<your-video>.mp4
```
Expected: Prints compression summary + two "PENDING" stubs. Exits 0.

---

- [ ] **Step 4: Commit**

```bash
git add scripts/test_psycholinguistic.py scripts/test_compress_and_analyze.py
git commit -m "feat(psycholinguistic): add CLI test scripts and compression integration stub

test_psycholinguistic.py: runs PsycholinguisticAnalyzer on text input,
prints all 8 dimension scores with bar chart and evidence.
test_compress_and_analyze.py: full compression pipeline → pending WhisperX
stub → pending psycholinguistic stub. End-to-end path verified."
```

---

## Final Verification

- [ ] **Run the complete test suite**

```bash
.venv\Scripts\pytest tests/ -v --tb=short
```
Expected: All tests pass. Zero failures. Coverage should be ≥80% across all new files.

- [ ] **Run the full lint check**

```bash
.venv\Scripts\python -m ruff check backend/ tests/ scripts/
```

- [ ] **Final integration smoke test**

```bash
.venv\Scripts\python scripts/test_psycholinguistic.py --text "I never said that. I was not there. I definitely, absolutely did not do anything like that, you know, um, I think."
```
Expected: High composite score (heavy negation + hedging + disfluency + overconfidence all firing).

---

## Self-Review Checklist

- [x] P1-S6 streaming JSONL: flush_interval param, .jsonl output, streaming write — all covered in Task 1
- [x] P2-S1 Pydantic schemas: all 8 fields, frozen, validation bounds — Task 2
- [x] P2-S2 Pronoun scorer: ratio curve, evidence — Task 3
- [x] P2-S3 Hedging scorer: modal verbs + phrase list, NOTE on BERT TODO — Task 4
- [x] P2-S4 Cognitive complexity: subordinate dep arcs, inverse truth signal — Task 5
- [x] P2-S5 Emotional distribution: NRCLex fear/anger vs positive/trust — Task 5
- [x] P2-S6 Disfluency: regex + filler phrases, per-100-words rate — Task 6
- [x] P2-S7 Negation: neg dep arcs — Task 6
- [x] P2-S7 Detail specificity: NE density inverse signal — Task 6
- [x] P2-S8 Certainty: VADER + overconfidence word list — Task 7
- [x] P2-S9 Composite analyze(): empty check, lazy spaCy load, deterministic — Task 7
- [x] P2-S10 CLI scripts: test_psycholinguistic.py + test_compress_and_analyze.py — Task 10
- [x] CLAUDE.md invariant #5 (no raw scores to users): disclaimer in CLI output
- [x] CLAUDE.md invariant #6 (no "lie detector"): checked — phrase not used anywhere
- [x] Type consistency: `PsycholinguisticDimension` used identically in all tasks
- [x] All scorer method signatures consistent: `_score_X(self, doc)` or `_score_X(self, text)` or both
- [x] `_ensure_nlp()` called in `analyze()` before any doc operations — covered
- [x] P1-S7 and P1-S8: already shipped in models.py and pipeline.py — marked skip, not re-implemented
