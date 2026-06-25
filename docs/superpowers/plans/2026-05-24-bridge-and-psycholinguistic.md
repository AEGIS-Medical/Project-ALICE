# ALICE Bridge + Psycholinguistic Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the last mile of the bridge + psycholinguistic work — declare the Phase 2 dependencies, add the two developer CLI scripts, commit the analyzer that already exists on disk, and bring CLAUDE.md's status section back in sync with reality.

**Architecture:** The mobile-critical bridge fixes (streaming JSONL landmarks, platform-aware model cache, mid-session tier switching) and the full psycholinguistic analyzer are **already implemented and passing tests**. This revision removes the now-obsolete build tasks and scopes the plan to the genuinely-remaining glue: packaging, CLIs, the commit, and doc sync.

**Tech Stack:** Python 3.13, spaCy 3.x + `en_core_web_sm`, NRCLex, vaderSentiment, Pydantic v2, pytest

---

## ⚠️ Revision 2026-05-24 — Reality Reconciliation

This plan was originally written assuming Phase 1-Bridge and Phase 2 were unbuilt.
A re-pull and inspection of `dev-build` showed **most of it is already done**.
The table below is the source of truth; do not re-implement completed work.

| Original Task | Story | Actual State |
|---|---|---|
| Task 1 | P1-S6 streaming JSONL | ✅ **DONE & committed** (`1fef136`) — `feature_extractor.py` emits `_landmarks.jsonl`, `flush_interval=30`, O(flush_interval) RAM |
| — | P1-S7 model cache | ✅ **DONE & committed** (`1fef136`) — `models.py` platform chain |
| — | P1-S8 tier switching | ✅ **DONE & committed** (`1fef136`) — `pipeline.py` `update_bandwidth()` |
| Task 2 | P2-S1 schemas | ✅ **DONE, untracked** — `backend/shared/schemas/psycholinguistic.py` |
| Tasks 3–7 | P2-S2…S9 analyzer | ✅ **DONE, untracked** — `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py`, all 8 scorers + composite |
| — | Phase 2 tests | ✅ **DONE, untracked** — `tests/psycholinguistic/` (28 tests); full suite **43 passed** |
| Task 8 | deps | ❌ **NOT done** — spacy/nrclex/vaderSentiment installed in venv but absent from `pyproject.toml` → Task A |
| Task 10 | P2-S10 CLI scripts | ❌ **NOT done** — neither script exists in `scripts/` → Tasks B, C |
| — | commit | ❌ **NOT done** — Phase 2 files are untracked → Task E |
| — | doc sync | ❌ **NOT done** — CLAUDE.md status stale → Task D |

### Two corrections to the original plan's assumptions

1. **Directory is `backend/ml-inference/` (hyphen), and that is intentional — do NOT rename to `ml_inference`.**
   Each ml-inference service runs with its own source root on `sys.path`. The analyzer imports itself as
   `from app.pipelines.psycholinguistic.analyzer import ...` and imports schemas as
   `from backend.shared.schemas.psycholinguistic import ...`. Tests bridge this in
   `tests/psycholinguistic/conftest.py` by inserting `backend/ml-inference` onto `sys.path`.
   **Any new script that imports the analyzer must replicate that `sys.path` insert.**

2. **Schema field names carry a `_score` suffix.** The shipped `PsycholinguisticScore` fields are
   `pronoun_shift_score`, `hedging_score`, `cognitive_complexity_score`, `emotional_distribution_score`,
   `disfluency_score`, `negation_score`, `detail_specificity_score`, `certainty_score` — **not** the
   un-suffixed names the original draft used. CLI code below uses the real names.

---

## Remaining File Map

| File | Action | Task |
|---|---|---|
| `pyproject.toml` | Modify — add spacy, nrclex, vaderSentiment | Task A |
| `scripts/test_psycholinguistic.py` | Create | Task B |
| `scripts/test_compress_and_analyze.py` | Create | Task C |
| `CLAUDE.md` | Modify — IMPLEMENTATION STATUS section | Task D |
| (git) commit untracked Phase 2 + push | — | Task E |

---

## Task A: Declare Phase 2 Dependencies in pyproject.toml

**Why:** spaCy, NRCLex, and vaderSentiment are imported by the analyzer and are present in the local `.venv`, but they are absent from `pyproject.toml`. A fresh `pip install -e ".[dev]"` would not install them, and the spaCy model is not documented. This is a reproducibility hole.

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the three runtime dependencies**

In `pyproject.toml`, change the `dependencies` list from:

```toml
dependencies = [
    "ffmpeg-python",
    "mediapipe",
    "librosa",
    "opencv-python-headless",
    "soundfile",
    "numpy",
    "pydantic>=2.0",
]
```

to:

```toml
dependencies = [
    "ffmpeg-python",
    "mediapipe",
    "librosa",
    "opencv-python-headless",
    "soundfile",
    "numpy",
    "pydantic>=2.0",
    # Psycholinguistic analysis vector (backend/ml-inference). Empath is in the
    # CLAUDE.md stack but the Day-1 analyzer does not import it yet, so it is
    # intentionally omitted until used.
    "spacy>=3.7",
    "nrclex>=4.0",
    "vaderSentiment>=3.3.2",
]
```

- [ ] **Step 2: Verify the manifest installs cleanly and the spaCy model note works**

Run:
```bash
cd C:\Users\ryanh\ALICE\Project-ALICE
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -c "import spacy, nrclex, vaderSentiment; print('libs ok')"
.venv/Scripts/python -c "import spacy; spacy.load('en_core_web_sm'); print('model ok')"
```
Expected: `libs ok` then `model ok`. (The `en_core_web_sm` model is fetched via
`python -m spacy download en_core_web_sm`, which the Makefile target documents —
pip cannot pull it as a normal dependency.)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): declare spacy, nrclex, vaderSentiment for psycholinguistic vector"
```

---

## Task B: `scripts/test_psycholinguistic.py` CLI (P2-S10, part 1)

**Why:** Developer smoke test for the analyzer. Must use the real `_score`-suffixed field
names and the `sys.path` bridge to the hyphenated ml-inference root.

**Files:**
- Create: `scripts/test_psycholinguistic.py`
- Create: `tests/psycholinguistic/test_cli_smoke.py`

- [ ] **Step 1: Write a smoke test that asserts the script runs and prints a composite**

Create `tests/psycholinguistic/test_cli_smoke.py`:

```python
"""Smoke test: the psycholinguistic CLI runs and prints a composite score."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_prints_composite():
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "test_psycholinguistic.py"),
            "--text",
            "I think maybe I never did that, um, I guess so.",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "COMPOSITE" in result.stdout
    # The eight dimension labels must all appear.
    for label in (
        "Pronoun Shift", "Hedging", "Cognitive Complexity",
        "Emotional Distribution", "Disfluency", "Negation",
        "Detail Specificity", "Certainty",
    ):
        assert label in result.stdout, f"missing dimension: {label}"
```

- [ ] **Step 2: Run the smoke test to verify it fails (script does not exist yet)**

```bash
.venv/Scripts/python -m pytest tests/psycholinguistic/test_cli_smoke.py -q
```
Expected: FAIL — non-zero return code because `scripts/test_psycholinguistic.py` does not exist.

- [ ] **Step 3: Create `scripts/test_psycholinguistic.py`**

```python
#!/usr/bin/env python
"""CLI smoke test for PsycholinguisticAnalyzer.

The analyzer lives under ``backend/ml-inference/`` (a hyphenated service root
that cannot be imported via a dotted path), so we insert that root onto
``sys.path`` exactly as ``tests/psycholinguistic/conftest.py`` does, then import
``from app.pipelines...``.

Usage:
    python scripts/test_psycholinguistic.py --text "I think maybe I never did that."
    python scripts/test_psycholinguistic.py --file path/to/statements.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ML_INFERENCE_ROOT = _REPO_ROOT / "backend" / "ml-inference"
if str(_ML_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_INFERENCE_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the psycholinguistic analyzer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text string to analyze")
    group.add_argument("--file", type=Path, help="Path to a text file (one statement per line)")
    args = parser.parse_args()

    if args.text:
        statements = [args.text]
    else:
        if not args.file.exists():
            print(f"ERROR: file not found: {args.file}", file=sys.stderr)
            return 1
        statements = [
            line.strip()
            for line in args.file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not statements:
        print("ERROR: no statements to analyze.", file=sys.stderr)
        return 1

    from app.pipelines.psycholinguistic.analyzer import PsycholinguisticAnalyzer

    analyzer = PsycholinguisticAnalyzer()
    result = analyzer.analyze(statements)

    # (label, dimension) pairs in CLAUDE.md order, using the real field names.
    dimensions = [
        ("Pronoun Shift", result.pronoun_shift_score),
        ("Hedging", result.hedging_score),
        ("Cognitive Complexity", result.cognitive_complexity_score),
        ("Emotional Distribution", result.emotional_distribution_score),
        ("Disfluency", result.disfluency_score),
        ("Negation", result.negation_score),
        ("Detail Specificity", result.detail_specificity_score),
        ("Certainty", result.certainty_score),
    ]

    print(f"\nPsycholinguistic Analysis — {result.statement_count} statement(s)")
    print(
        f"Confidence: {result.confidence} | "
        f"Baseline: {'available' if result.baseline_available else 'not yet established'}"
    )
    print("-" * 64)
    for name, dim in dimensions:
        bar = "#" * int(dim.score / 5)
        print(f"  {name:<24} {dim.score:6.1f}/100  {bar}")
        if dim.evidence:
            print(f"      {', '.join(dim.evidence[:3])}")
    print("-" * 64)
    bar = "#" * int(result.composite_score / 5)
    print(f"  {'COMPOSITE':<24} {result.composite_score:6.1f}/100  {bar}")
    print()
    print("NOTE: scores are behavioral anomaly signals, not ground truth.")
    print("      ~75% F1 realistic ceiling. ALICE is not a verdict engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the smoke test to verify it passes**

```bash
.venv/Scripts/python -m pytest tests/psycholinguistic/test_cli_smoke.py -q
```
Expected: PASS.

- [ ] **Step 5: Eyeball the output manually**

```bash
.venv/Scripts/python scripts/test_psycholinguistic.py --text "I never said that. I was not there. I definitely, absolutely did not do anything, you know, um, I think."
```
Expected: all 8 dimensions + a COMPOSITE line; high-ish composite (negation + hedging + disfluency + certainty all firing). No traceback.

- [ ] **Step 6: Commit**

```bash
git add scripts/test_psycholinguistic.py tests/psycholinguistic/test_cli_smoke.py
git commit -m "feat(psycholinguistic): add test_psycholinguistic.py CLI + smoke test"
```

---

## Task C: `scripts/test_compress_and_analyze.py` Integration Stub (P2-S10, part 2)

**Why:** Demonstrates the end-to-end path — video → compression (FLAC + JSONL landmarks) →
the point where WhisperX transcription will feed the analyzer. Transcription is not built
yet, so the script stops at a clearly-labelled stub rather than fabricating statements.

**Files:**
- Create: `scripts/test_compress_and_analyze.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python
"""Integration stub: video -> compression pipeline -> (pending) psycholinguistic.

Runs the real compression pipeline on a video, then shows exactly where WhisperX
transcription will plug in to feed PsycholinguisticAnalyzer. Transcription is not
yet implemented, so the script stops at a labelled stub.

Usage:
    python scripts/test_compress_and_analyze.py path/to/video.mp4
    python scripts/test_compress_and_analyze.py path/to/video.mp4 --mode edge_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compression -> psycholinguistic stub")
    parser.add_argument("video", type=Path, help="Path to a video file")
    parser.add_argument(
        "--mode",
        choices=["raw", "roi", "edge_full", "edge_minimal"],
        default="edge_full",
        help="Compression mode (default: edge_full)",
    )
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

    print(f"\nStep 1 — Compression Pipeline ({mode.value})")
    print("-" * 64)
    pipeline = CompressionPipeline()
    result = pipeline.process(args.video, output_dir, mode)

    mb = 1_048_576
    print(f"  Input:              {result.input_size_bytes / mb:6.1f} MB")
    print(f"  FLAC audio (ML):    {result.flac_size_bytes / mb:6.2f} MB  -> {result.flac_audio_path}")
    if result.roi_video_path:
        print(f"  ROI video:          {(result.roi_video_size_bytes or 0) / mb:6.1f} MB  -> {result.roi_video_path}")
    if result.landmarks_path:
        print(f"  Landmarks (JSONL):  {(result.landmarks_size_bytes or 0) / mb:6.2f} MB  -> {result.landmarks_path}")
    if result.features_path:
        print(f"  Audio features:     {(result.features_size_bytes or 0) / mb:6.2f} MB  -> {result.features_path}")
    print(f"  Face detected:      {result.face_detected_pct:5.1f}% of frames")
    print(f"  Total time:         {result.processing_times.get('total', 0.0):5.1f}s")

    print("\nStep 2 — Transcription (PENDING)")
    print("-" * 64)
    print(f"  Would run: WhisperX on {result.flac_audio_path}")
    print("  Would produce: speaker-attributed statement strings")
    print("  Status: WhisperX not yet integrated (next plan)")

    print("\nStep 3 — Psycholinguistic Analysis (READY, awaiting transcript)")
    print("-" * 64)
    print("  Would run: PsycholinguisticAnalyzer.analyze(statements)")
    print("  Would produce: PsycholinguisticScore (8 dimensions + composite)")
    print("  Status: analyzer is built and tested — needs the WhisperX feed above.")

    print("\nEnd-to-end path verified through compression. Add WhisperX to unlock scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run against a demo video if one is present**

```bash
# List candidates first:
ls demo_data/honest/demo_file_1/ 2>/dev/null
# Then, substituting a real file:
.venv/Scripts/python scripts/test_compress_and_analyze.py demo_data/honest/demo_file_1/<your-video>.mp4 --mode edge_full
```
Expected: a compression summary (FLAC + a `_landmarks.jsonl` path), then the two PENDING
stubs. Exit code 0. If no demo video exists, skip — this script has no unit test because it
requires a real media file and ffmpeg.

- [ ] **Step 3: Commit**

```bash
git add scripts/test_compress_and_analyze.py
git commit -m "feat(psycholinguistic): add compress->analyze integration stub (JSONL + pending WhisperX)"
```

---

## Task D: Sync CLAUDE.md IMPLEMENTATION STATUS

**Why:** The status section still lists the psycholinguistic stack as *Pending* and describes
landmark output as "JSON" — both are now wrong. Bring it in line with what shipped.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Day-1 feature-extraction row to say JSONL**

In the Day-1 table, change the Feature extraction row from:

```
| Feature extraction | `compression/feature_extractor.py` | shipped — 478-pt landmarks JSON + MFCC/Chroma/Mel/Contrast/Tonnetz `.npz` |
```

to:

```
| Feature extraction | `compression/feature_extractor.py` | shipped — 478-pt landmarks **streaming JSONL** (`_landmarks.jsonl`, flushed every N frames) + MFCC/Chroma/Mel/Contrast/Tonnetz `.npz` |
```

- [ ] **Step 2: Add a Phase 1-Bridge + Phase 2 status block**

Immediately after the "Notable deviations from the original Day 1 spec" list (before
"### Pending (not yet implemented)"), insert:

```markdown
### Phase 1-Bridge — Mobile-Critical Fixes (complete)

Shipped in commit `1fef136`. Driven by a mobile high-usage analysis (see
`docs/superpowers/specs/2026-04-27-compression-pipeline-and-psycholinguistic-design.md`).

| Fix | Path | Status |
|---|---|---|
| Streaming JSONL landmarks (P1-S6) | `compression/feature_extractor.py` | shipped — peak RAM O(flush_interval), not O(total_frames) |
| Platform-aware model cache (P1-S7) | `compression/models.py` | shipped — `ALICE_MODEL_CACHE` > Windows `%LOCALAPPDATA%` > Android `$XDG_DATA_HOME` > XDG home |
| Mid-session tier switching (P1-S8) | `compression/pipeline.py` | shipped — `update_bandwidth()` + `on_mode_change` callback + `mode_transitions` audit |

### Phase 2 — Psycholinguistic Analyzer (complete, Day 1)

The first analysis vector. Lives under `backend/ml-inference/` (hyphenated service
root, imported via `sys.path`, NOT a dotted `ml_inference` path).

| Component | Path | Status |
|---|---|---|
| Schemas | `backend/shared/schemas/psycholinguistic.py` | shipped — `PsycholinguisticDimension`, `PsycholinguisticScore` (fields are `*_score`-suffixed) |
| Analyzer | `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py` | shipped — all 8 dimension scorers + equal-weighted composite, lazy spaCy/VADER load |
| Tooling | spaCy `en_core_web_sm` + NRCLex + vaderSentiment | hedging is a Day-1 word list (59% FP); replace with BERT in Phase 3 |
| Tests | `tests/psycholinguistic/` | 28 tests; full suite 43 passing |
| CLI | `scripts/test_psycholinguistic.py`, `scripts/test_compress_and_analyze.py` | shipped |
```

- [ ] **Step 3: Remove the now-false "Pending" line for psycholinguistic**

In the "### Pending (not yet implemented)" list, delete this line:

```
- Psycholinguistic analysis stack (spaCy + Empath + NRCLex + VADER + hedging BERT)
```

(The remaining four analysis vectors and infrastructure stay pending.)

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): mark Phase 1-Bridge + psycholinguistic vector complete; JSON->JSONL"
```

---

## Task E: Commit the Untracked Analyzer and Push

**Why:** The analyzer, schemas, and tests exist on disk but are untracked — a crash or
`git clean` would lose them. Get them under version control, then publish the branch.

**Files (already on disk, currently untracked):**
- `backend/shared/schemas/psycholinguistic.py`
- `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py` (+ package `__init__.py` files if present)
- `tests/psycholinguistic/__init__.py`, `conftest.py`, `test_schemas.py`, `test_analyzer.py`

- [ ] **Step 1: Confirm `.gitignore` excludes bytecode, then stage only source**

```bash
grep -q "__pycache__" .gitignore || echo "__pycache__/" >> .gitignore
git add backend/shared/schemas/psycholinguistic.py
git add "backend/ml-inference"
git add tests/psycholinguistic
git status --short
```
Expected: the analyzer, schema, and test `.py` files staged; **no** `.pyc` or `__pycache__`
entries staged. If any appear, unstage with `git reset <path>` and fix `.gitignore`.

- [ ] **Step 2: Run the full suite one more time before committing**

```bash
.venv/Scripts/python -m pytest tests/ -p no:cacheprovider
```
Expected: `44 passed` (43 existing + the new CLI smoke test from Task B).

- [ ] **Step 3: Commit the analyzer**

```bash
git commit -m "feat(psycholinguistic): add analyzer, schemas, and tests (8 dimensions, Day 1)

Implements P2-S1 through P2-S9: PsycholinguisticAnalyzer with pronoun, hedging,
cognitive complexity, emotional distribution, disfluency, negation, detail
specificity, and certainty scorers plus an equal-weighted composite. spaCy +
NRCLex + VADER, lazy-loaded. Lives under backend/ml-inference (sys.path import)."
```

- [ ] **Step 4: Push the branch**

```bash
git push origin dev-build
```
Expected: `dev-build -> dev-build` updated on `AEGIS-Medical/Project-ALICE`. If push is
rejected for auth, run `gh auth login` (or `gh auth setup-git`) and retry — `git fetch`
currently works via a cached credential helper, but `gh` itself is logged out.

---

## Final Verification

- [ ] **Full test suite**

```bash
.venv/Scripts/python -m pytest tests/ -p no:cacheprovider
```
Expected: all pass (44 with the new smoke test).

- [ ] **Both CLIs run**

```bash
.venv/Scripts/python scripts/test_psycholinguistic.py --text "I guess maybe I wasn't really sure, you know."
```
Expected: 8 dimensions + COMPOSITE, no traceback.

- [ ] **Working tree clean**

```bash
git status --short
```
Expected: empty (everything committed).

---

## Self-Review Checklist

- [x] Task 1 / P1-S6 (streaming JSONL) — verified DONE in `feature_extractor.py`, not re-planned
- [x] P1-S7, P1-S8 — verified DONE in `models.py` / `pipeline.py`, not re-planned
- [x] P2-S1…S9 analyzer + schemas — verified DONE & passing; scoped to commit (Task E), not rebuild
- [x] Field names corrected to the real `*_score` suffix throughout the CLI code
- [x] Hyphen directory + `sys.path` import pattern preserved (no rename to `ml_inference`)
- [x] Task A closes the pyproject dependency gap (spacy/nrclex/vaderSentiment)
- [x] Tasks B, C deliver the two missing P2-S10 CLI scripts with a real smoke test
- [x] Task D fixes the two stale facts in CLAUDE.md (JSON→JSONL; psycholinguistic no longer "pending")
- [x] Task E gets untracked work under version control and pushes; `.pyc` excluded
- [x] No "lie detector" phrasing introduced (CLAUDE.md invariant #6); CLIs carry the anomaly disclaimer (#5)
- [x] No placeholders — every step has concrete code/commands and expected output
