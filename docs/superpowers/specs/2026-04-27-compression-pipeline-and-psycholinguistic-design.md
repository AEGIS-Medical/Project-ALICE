# ALICE — Compression Pipeline Completion + Psycholinguistic Analyzer
## Design Spec · 2026-04-27

---

## Overview

This spec covers two sequential phases that together deliver ALICE's first
working end-to-end vertical slice:

```
Video file → Compression Pipeline → FLAC audio → Psycholinguistic Analyzer → Linguistic score
```

**Phase 1** finishes the compression pipeline started in Session 1.  
**Phase 2** adds the psycholinguistic analysis module (Session 2 plan).

Both phases are in scope for a single implementation plan with independently
testable stories.

---

## What Already Exists (Do Not Re-implement)

| File | Status |
|------|--------|
| `backend/shared/schemas/media.py` | Complete — CompressionMode, CompressionConfig, CompressionResult |
| `backend/workers/app/compression/audio_extractor.py` | Complete — FLAC + Opus extraction |
| `backend/workers/app/compression/roi_encoder.py` | Complete — simplified v1 ROI encoder |

All three files follow CLAUDE.md invariants and must not be modified unless a
story explicitly targets them.

---

## Phase 1 — Complete the Compression Pipeline

### Architecture

The pipeline has four stages that run conditionally based on `CompressionMode`:

```
input video
    │
    ├─ [always]      AudioExtractor      → FLAC (ML) + Opus (playback)
    ├─ [RAW/ROI]     ROIEncoder          → ROI-encoded MP4
    ├─ [EDGE_FULL/MINIMAL] FeatureExtractor → landmarks JSON + audio .npz
    │
    └─ CompressionPipeline.process()     → CompressionResult
```

Critical invariant from CLAUDE.md: **FLAC audio is the only artifact that
may be passed to any ML model.** The Opus file is playback-only. The
`FeatureExtractor` must enforce this by rejecting non-FLAC/WAV input with
a `ValueError`.

### Stories

#### P1-S1: FeatureExtractor — Landmark Extraction
**File:** `backend/workers/app/compression/feature_extractor.py`  
**Class:** `FeatureExtractor`  
**Method:** `extract_landmarks(video_path: Path, output_dir: Path, skip_frames: int = 1) -> Path`

- Uses MediaPipe Face Mesh (478 landmarks, NOT FaceDetection — this is for
  full landmark geometry, not just bbox)
- Processes every Nth frame (configurable via `skip_frames`)
- Output: JSON file at `output_dir/{stem}_landmarks.json`
  ```json
  [{"frame": 0, "timestamp_seconds": 0.0, "landmarks": [[x, y, z], ...]}, ...]
  ```
- Returns path to the JSON file
- If no face is detected in a frame, that frame is omitted (not an error)

**Acceptance criteria:**
- Given a valid MP4 with a face, returns a `.json` file with at least one frame entry
- Each entry has `frame` (int), `timestamp_seconds` (float), `landmarks` (list of 478 [x,y,z] triples)
- Given a video with no face, returns an empty array JSON (not an exception)
- Given a non-existent file, raises `FileNotFoundError`
- Given an unsupported extension, raises `UnsupportedMediaError`

**Tests:** `tests/compression/test_feature_extractor.py`
- `test_extract_landmarks_valid_video` — asserts JSON schema, landmark count = 478
- `test_extract_landmarks_no_face` — asserts empty array returned, no exception
- `test_extract_landmarks_missing_file` — asserts `FileNotFoundError`
- `test_extract_landmarks_bad_extension` — asserts `UnsupportedMediaError`

---

#### P1-S2: FeatureExtractor — Audio Feature Extraction
**File:** `backend/workers/app/compression/feature_extractor.py` (same class)  
**Method:** `extract_audio_features(flac_path: Path, output_dir: Path, window_sec: float = 1.0, stride_sec: float = 0.5) -> Path`

- Accepts ONLY FLAC or WAV. Raises `ValueError` with message referencing
  CLAUDE.md Critical Invariant #1 if given any other format.
- Uses librosa to extract per-window features:
  - MFCC (40 coefficients)
  - Chroma (12 bins)
  - Mel spectrogram (128 bins)
  - Spectral contrast (7 bands)
  - Tonnetz (6 dims)
- Output: `.npz` file at `output_dir/{stem}_features.npz`
  - Arrays keyed by: `mfcc`, `chroma`, `mel`, `spectral_contrast`, `tonnetz`, `timestamps`
- Returns path to the `.npz` file

**Acceptance criteria:**
- Given a valid FLAC, returns `.npz` with all 6 expected array keys
- `timestamps` array length matches number of windows
- Given an `.mp3` input, raises `ValueError` containing "Critical Invariant #1"
- Given a `.wav` input, succeeds (WAV is lossless)
- Window/stride params correctly control number of output frames

**Tests:** `tests/compression/test_feature_extractor.py`
- `test_extract_audio_features_valid_flac` — asserts all 6 keys, correct dtypes
- `test_extract_audio_features_rejects_mp3` — asserts `ValueError` with invariant message
- `test_extract_audio_features_accepts_wav`
- `test_window_stride_controls_frame_count`

---

#### P1-S3: CompressionPipeline Orchestrator
**File:** `backend/workers/app/compression/pipeline.py`  
**Class:** `CompressionPipeline`  
**Method:** `process(input_path: Path, output_dir: Path, mode: CompressionMode = CompressionMode.RAW) -> CompressionResult`

Stage execution per mode:

| Stage | RAW | ROI_ENCODED | EDGE_FULL | EDGE_MINIMAL |
|-------|-----|-------------|-----------|--------------|
| AudioExtractor (FLAC + Opus) | ✅ | ✅ | ✅ | ✅ |
| ROIEncoder | ✅ | ✅ | ❌ | ❌ |
| FeatureExtractor.extract_landmarks | ❌ | ❌ | ✅ | ✅ |
| FeatureExtractor.extract_audio_features | ❌ | ❌ | ✅ | ❌ |

- Creates subdirs: `audio/`, `video/`, `landmarks/`, `features/`
- Partial failure tolerance: if landmark extraction fails, still populate
  the audio/video fields in `CompressionResult` and set `landmarks_path=None`
- All timing goes into `CompressionResult.processing_times`
- `face_detected_pct` is taken from `ROIEncoder.last_face_detected_pct`
  (0.0 for modes that don't run ROI)

**Acceptance criteria:**
- RAW mode: `CompressionResult` has `flac_audio_path`, `roi_video_path` set; landmarks/features are None
- EDGE_FULL mode: `flac_audio_path`, `landmarks_path`, `features_path` set; `roi_video_path` is None
- EDGE_MINIMAL mode: `flac_audio_path`, `landmarks_path` set; `features_path` and `roi_video_path` are None
- `processing_times` dict has keys for each stage that ran
- Partial failure in landmarks does not prevent audio result from being returned
- Input validation: missing file → `FileNotFoundError`, unsupported extension → `UnsupportedMediaError`

**Tests:** `tests/compression/test_pipeline.py`
- `test_pipeline_raw_mode` — asserts correct artifact presence/absence
- `test_pipeline_roi_mode`
- `test_pipeline_edge_full_mode`
- `test_pipeline_edge_minimal_mode`
- `test_pipeline_partial_failure_landmarks` — mocks `extract_landmarks` to raise, asserts audio still returned
- `test_pipeline_timing_keys_present`

---

#### P1-S4: Package Init + pyproject.toml + Makefile
**Files:**
- `backend/workers/app/compression/__init__.py` — exports `CompressionPipeline`, `CompressionMode`
- `pyproject.toml` (project root) — single source of dependency truth:
  ```
  python >=3.12
  ffmpeg-python, mediapipe, librosa, opencv-python-headless,
  soundfile, numpy, pydantic>=2.0, fastapi, uvicorn
  ```
  Dev extras: `pytest`, `pytest-cov`, `ruff`, `mypy`
- `Makefile` (project root):
  - `make install` → `pip install -e ".[dev]"`
  - `make test` → `pytest tests/ -v`
  - `make test-compress` → `python scripts/test_compression.py`
  - `make lint` → `ruff check . && mypy backend/`

**Acceptance criteria:**
- `from backend.workers.app.compression import CompressionPipeline, CompressionMode` succeeds
- `pip install -e ".[dev]"` succeeds without conflicts
- `make test` runs pytest

---

#### P1-S5: CLI Test Script
**File:** `scripts/test_compression.py`

- `argparse`: required positional `video_path`, optional `--mode` (default: `raw`)
- Outputs to `processed_output/compression_test/{stem}/`
- Prints summary table:
  ```
  Component       │ Size      │ Ratio  │ Time
  Original Video  │ 18.2 MB   │ 1.00×  │ —
  FLAC Audio      │ 4.1 MB    │ 0.23×  │ 2.1s
  Opus Playback   │ 0.3 MB    │ 0.02×  │ 0.8s
  ROI Video       │ 11.4 MB   │ 0.63×  │ 8.3s
  Landmarks JSON  │ 2.1 MB    │ 0.12×  │ 12.1s
  Audio Features  │ 0.8 MB    │ 0.04×  │ 3.2s
  ```
- Prints: `Face detected in 94.0% of frames`
- Exits 0 on success, 1 on error with human-readable message

**Acceptance criteria:**
- Running `python scripts/test_compression.py <valid_mp4>` exits 0 and prints table
- Running with `--mode edge_full` shows N/A for ROI Video row
- Running with a missing file exits 1 with a clear error message

---

## Phase 2 — Psycholinguistic Analyzer

### Architecture

The psycholinguistic analyzer is a standalone Python module that takes a
list of statement strings and produces scores across 8 dimensions defined
in CLAUDE.md. It does NOT depend on the compression pipeline directly —
it receives already-transcribed text.

```
[str, str, ...]  (speaker-attributed statements)
        │
        ▼
PsycholinguisticAnalyzer.analyze(statements)
        │
        ▼
PsycholinguisticScore
  ├── pronoun_shift_score      (0-100)
  ├── hedging_score            (0-100)
  ├── cognitive_complexity_score (0-100)
  ├── emotional_distribution_score (0-100)
  ├── disfluency_score         (0-100)
  ├── negation_score           (0-100)
  ├── detail_specificity_score (0-100)
  ├── certainty_score          (0-100)
  └── composite_score          (0-100, weighted average per CLAUDE.md)
```

Research basis (CLAUDE.md): Li & Abouelenien (2024) confirmed linguistic
features are the strongest single modality (~80% accuracy). Pronoun pattern
shifts (Newman et al. 2003), hedging, cognitive complexity, and emotional
word distribution are the highest-signal dimensions.

**Day 1 tooling (lightweight, no large model downloads):**
- spaCy `en_core_web_sm` (~12MB) — POS, dependency parse, NER, pronouns,
  negation detection, clause depth
- Empath — 200+ lexical categories
- NRCLex — 8 granular emotion categories
- VADER — valence-aware sentiment
- Hedging: spaCy modal verb detection + curated word list (Day 1)
  — NOTE: must be replaced with fine-tuned BERT classifier in Phase 3.
  The word-list approach has ~59% false positive rate (CLAUDE.md).

### Stories

#### P2-S1: Psycholinguistic Pydantic Schemas
**File:** `backend/shared/schemas/psycholinguistic.py`

Models:
- `PsycholinguisticDimension(BaseModel)` — `score: float` (0-100), `evidence: list[str]`
- `PsycholinguisticScore(BaseModel)`:
  - One field per dimension (type `PsycholinguisticDimension`)
  - `composite_score: float` — weighted average per CLAUDE.md dimension weights
  - `statement_count: int`
  - `baseline_available: bool` — False until per-contact baseline established
  - `confidence: Literal["low", "medium", "high"]`

**Acceptance criteria:**
- All 8 dimension fields present with correct names matching CLAUDE.md
- `composite_score` is between 0 and 100
- Schema is frozen (immutable after construction)
- Invalid score ranges (negative, >100) raise `ValidationError`

**Tests:** `tests/psycholinguistic/test_schemas.py`
- `test_valid_schema_construction`
- `test_composite_score_range`
- `test_invalid_score_raises`

---

#### P2-S2: Pronoun Pattern Scorer
**File:** `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py`  
**Class:** `PsycholinguisticAnalyzer` (partial — this story adds one dimension)  
**Method:** `_score_pronouns(doc: spacy.tokens.Doc) -> PsycholinguisticDimension`

- Counts first-person singular tokens (`I`, `me`, `my`, `mine`, `myself`)
  normalized by total token count
- Computes first-person singular ratio
- Score maps: ratio < 0.03 → high score (deceptive per Newman et al. 2003),
  ratio > 0.12 → high score (over-compensation), midrange → low score
- `evidence` list includes the raw ratio and token count

**Acceptance criteria:**
- Statement with very few first-person pronouns produces score > 50
- Statement with normal pronoun density produces score < 50
- `evidence` list is non-empty

**Tests:** `tests/psycholinguistic/test_analyzer.py`
- `test_low_pronoun_density_scores_high`
- `test_normal_pronoun_density_scores_low`
- `test_evidence_populated`

---

#### P2-S3: Hedging Scorer
**Method:** `_score_hedging(doc) -> PsycholinguisticDimension`

- Detects modal verbs via spaCy POS tags (`MD` tag: might, could, would, etc.)
- Supplemented by curated epistemic phrase list: "I think", "I believe",
  "I'm not sure", "sort of", "kind of", "maybe", "perhaps", "possibly", etc.
- Score = (modal_count + phrase_count) / sentence_count × normalization factor
- `evidence` includes top 3 detected hedges

**NOTE in code:** Day 1 uses word list. Replace with fine-tuned BERT
hedging classifier in Phase 3 (CLAUDE.md: word-list has 59% FP rate).

**Acceptance criteria:**
- High-hedging statement ("I think maybe it could perhaps have been...") scores > 60
- Direct statement ("I did it at 3pm") scores < 30
- `evidence` surfaces at least one detected hedge when present

**Tests:**
- `test_high_hedging_statement_scores_high`
- `test_direct_statement_scores_low`
- `test_hedge_evidence_populated`

---

#### P2-S4: Cognitive Complexity Scorer
**Method:** `_score_cognitive_complexity(doc) -> PsycholinguisticDimension`

- Measures subordinate clause depth via spaCy dependency parse:
  counts `advcl`, `relcl`, `ccomp`, `xcomp` dependency arcs
- Normalizes by sentence count
- Higher complexity = lower deception score (truth-tellers use more complex
  language; deceivers keep it simple to maintain consistency)

**Acceptance criteria:**
- Complex nested sentence (multiple subordinate clauses) scores < 40
- Simple flat sentences score > 60
- Works on multi-sentence input

---

#### P2-S5: Emotional Word Distribution Scorer
**Method:** `_score_emotional_distribution(text: str) -> PsycholinguisticDimension`

Uses NRCLex to detect emotional word distribution.
Per Pérez-Rosas (EMNLP 2015):
- Deceivers use more Anxiety/Anger words
- Truth-tellers use more Family/Home/Humans words

Score = weighted sum of deceptive-indicator emotions
(anxiety, anger, fear) minus truth-indicator emotions (positive emotion,
trust), normalized to 0-100.

**Acceptance criteria:**
- Anxiety/anger-heavy text scores > 60
- Positive/family-word-heavy text scores < 40

---

#### P2-S6: Disfluency Scorer
**Method:** `_score_disfluencies(text: str) -> PsycholinguisticDimension`

- Counts filler words: `um`, `uh`, `er`, `ah`, `like` (as filler), `you know`
- Normalizes by word count
- Higher disfluency = higher deception score (cognitive load signal)

**Acceptance criteria:**
- Text with multiple "um", "uh" scores > 50
- Clean text scores < 20
- `evidence` lists the detected disfluencies

---

#### P2-S7: Negation + Detail Specificity Scorers
**Methods:**
- `_score_negation(doc) -> PsycholinguisticDimension` — count `neg` dependency arcs, normalize by sentence count
- `_score_detail_specificity(doc) -> PsycholinguisticDimension` — count named entities (NER) normalized by word count; low NE density = less specific = higher deception score

**Acceptance criteria (negation):**
- High negation text ("I never did, I didn't, I wasn't there") scores > 60
- Affirmative text scores < 30

**Acceptance criteria (specificity):**
- Text with named entities (dates, places, people) scores < 40 (specific = truthful)
- Vague text with no entities scores > 60

---

#### P2-S8: Certainty/Tentative Language Scorer
**Method:** `_score_certainty(doc, text: str) -> PsycholinguisticDimension`

- Uses VADER to measure sentiment certainty
- Also detects absolute certainty phrases: "definitely", "absolutely",
  "I'm certain", "100%"
- Extremes in either direction (over-certain or very tentative) score higher
- Returns mid-range score for neutral confident language

**Acceptance criteria:**
- Over-emphatic certainty ("I absolutely definitely 100% did not") scores > 60
- Neutral confident language scores < 40

---

#### P2-S9: PsycholinguisticAnalyzer.analyze() — Composite
**Method:** `analyze(statements: list[str]) -> PsycholinguisticScore`

- Calls all 8 dimension scorers
- Computes composite score as weighted average using dimension weights
  from CLAUDE.md (linguistic weight = 0.30 total; internal sub-dimension
  weights are equal for Day 1)
- Initializes spaCy on first call (lazy load)
- Returns `PsycholinguisticScore`

**Acceptance criteria:**
- Given empty list, raises `ValueError("No statements provided")`
- Given 3 statements, returns all 9 fields populated
- Composite score is within [0, 100]
- Calling twice with the same input returns identical scores (deterministic)

**Tests:** `tests/psycholinguistic/test_analyzer.py`
- `test_analyze_empty_raises`
- `test_analyze_returns_all_dimensions`
- `test_analyze_deterministic`
- `test_composite_score_in_range`

---

#### P2-S10: CLI Test Scripts
**Files:**
- `scripts/test_psycholinguistic.py` — takes `--text` string or `--file` path,
  prints all 8 dimension scores + composite
- `scripts/test_compress_and_analyze.py` — takes video path, runs
  `CompressionPipeline.process()` to extract FLAC, then prints:
  `"Transcription would run here — psycholinguistic analysis requires transcript text."`
  along with the compression summary. This wires up the integration path
  for when WhisperX is added.

**Acceptance criteria:**
- `python scripts/test_psycholinguistic.py --text "I never did that I swear"` prints all 9 scores
- `python scripts/test_compress_and_analyze.py <video>` completes without error and prints both compression summary and integration stub message

---

## Critical Invariants (from CLAUDE.md)

These must be enforced in every story:

1. **Lossy audio never reaches ML** — `FeatureExtractor.extract_audio_features` must reject non-FLAC/WAV with a `ValueError` referencing Invariant #1.
2. **Never log PII** — opaque IDs only in all log statements.
3. **Never use "lie detector"** — no string in code, comments, or test output may use this phrase.
4. **Never show raw scores to users** — the CLI scripts are developer tools only; any user-facing surface must include confidence + qualitative label.
5. **Never hardcode secrets** — `pyproject.toml` contains no API keys or credentials.

---

## File Structure After Completion

```
backend/
  shared/schemas/
    media.py                          ✅ exists
    psycholinguistic.py               ← P2-S1
  workers/app/compression/
    audio_extractor.py                ✅ exists
    roi_encoder.py                    ✅ exists
    feature_extractor.py              ← P1-S1, P1-S2
    pipeline.py                       ← P1-S3
    __init__.py                       ← P1-S4
  ml-inference/app/pipelines/
    psycholinguistic/
      analyzer.py                     ← P2-S2 through P2-S9
scripts/
  test_compression.py                 ← P1-S5
  test_psycholinguistic.py            ← P2-S10
  test_compress_and_analyze.py        ← P2-S10
tests/
  compression/
    test_feature_extractor.py
    test_pipeline.py
  psycholinguistic/
    test_schemas.py
    test_analyzer.py
pyproject.toml                        ← P1-S4
Makefile                              ← P1-S4
```

---

## Out of Scope (Next Plan)

- WhisperX transcription (bridges compression → psycholinguistic for real)
- Facial AU analysis pipeline
- Vocal tonality / emotion2vec+
- Contradiction detection (DeBERTa NLI + pgvector)
- Subject identification (LR-ASD + EdgeFace-XS)
- Late fusion ensemble (XGBoost + Platt + SHAP)
- Platform connectors (Zoom/Teams/Meet)
- Mobile app
