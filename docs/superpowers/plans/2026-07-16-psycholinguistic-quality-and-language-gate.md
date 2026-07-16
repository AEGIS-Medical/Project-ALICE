# Psycholinguistic Quality + Language Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count each linguistic signal exactly once (disjoint hedging/certainty lists), hard-fail non-English transcripts before any scoring (`UnsupportedLanguageError` at analyzer + stream entry + CLIs), and upgrade spaCy sm→md for better NER — closing CLAUDE.md gaps #4, #8-near-term, and the #7 spaCy item.

**Architecture:** All three changes live in or at the boundary of the existing psycholinguistic vector. The gate is exception-based and fires before any scoring work, so the ScoreEvent streaming contract is untouched (an empty stream still means only "zero statements"). Composite values shift (double-count removed; md NER differs), and the Session 5 convergence gate re-baselines automatically because stream and batch share the analyzer.

**Tech Stack:** Python 3.13, spaCy 3.x (`en_core_web_md`), Pydantic v2, pytest. No new dependencies beyond the md model download (~40 MB).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-16-psycholinguistic-quality-and-language-gate-design.md`. Spec governs on conflict.
- Run everything from `C:\Users\ryanh\ALICE\Project-ALICE` with `.venv/Scripts/python`.
- Hyphenated service root: analyzer/streaming code imports via `sys.path` insert of `backend/ml-inference` then `from app.pipelines...` (tests' conftests already do this).
- **Never translate-then-score.** The gate rejects; it never routes through MT.
- Supported languages constant: `SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en"})`. Accepted iff the primary subtag (before `-` or `_`, case-insensitive) is in the set — so `en`, `EN`, `en-US`, `en_GB` pass; `es`, `fr`, `zh` raise.
- Gate precedes the empty-statement check in `analyze()` (a Spanish empty transcript reports the language error — the more actionable fact).
- Invariant #3: error messages and logs carry language codes only — NEVER transcript text. Invariant #5: outputs remain score+confidence+label (no verdicts). Invariant #6: no "lie detector" anywhere.
- Existing range-based tests are re-verified after each behavior change; a test is adjusted ONLY with a recorded justification in the task report — no silent threshold nudging.
- The Session 5 convergence test (`tests/streaming/test_convergence.py`) must be green at the end of every task that changes scoring — it proves batch and stream moved together.

---

## File Map

| File | Action | Task |
|---|---|---|
| `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py` | Modify — delete `_TENTATIVE_MARKERS`, rework `_score_certainty` | 1 |
| `backend/shared/schemas/psycholinguistic.py` | Modify — dimension-8 docstring wording | 1 |
| `tests/psycholinguistic/test_analyzer.py` | Extend — disjointness guard + certainty rescoring tests | 1 |
| `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py` | Modify — `UnsupportedLanguageError`, `SUPPORTED_LANGUAGES`, `analyze(..., language=None)` | 2 |
| `backend/ml-inference/app/pipelines/psycholinguistic/__init__.py` | Create — re-exports (mirrors transcription package) | 2 |
| `tests/psycholinguistic/test_language_gate.py` | Create | 2 |
| `backend/ml-inference/app/pipelines/streaming/windowed_scorer.py` | Modify — entry gate on `transcript.language` | 3 |
| `tests/streaming/test_windowed_scorer.py` | Extend — gate tests | 3 |
| `scripts/replay_scores.py` | Modify — catch → stderr + exit 1 | 4 |
| `scripts/test_compress_and_analyze.py` | Modify — pass `transcript.language`, catch → exit 1 | 4 |
| `tests/streaming/test_cli_smoke.py` | Extend — non-en transcript CLI smoke | 4 |
| `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py` | Modify — `_SPACY_MODEL` sm→md | 5 |
| `Makefile`, `pyproject.toml`, `CLAUDE.md` | Modify — model refs + gap statuses | 5 |

---

## Task 1: Disjoint the Hedging/Certainty Word Lists (gap #4)

**Files:**
- Modify: `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py` (constants ~lines 110-143; `_score_certainty` ~lines 435-465)
- Modify: `backend/shared/schemas/psycholinguistic.py` (module docstring dimension list, item 8)
- Test: `tests/psycholinguistic/test_analyzer.py`

**Interfaces:**
- Consumes: existing `_HEDGE_PHRASES`, `_CERTAINTY_MARKERS`, `_score_certainty(doc, text)`.
- Produces: `_TENTATIVE_MARKERS` no longer exists; `_score_certainty` counts only `_CERTAINTY_MARKERS` + VADER term. Dimension 8 semantics: "over-certainty / emphatic assertion." No signature changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/psycholinguistic/test_analyzer.py`:

```python
# ---- Session 6 / gap #4: hedging-certainty disjoint ------------------------


def test_hedge_and_certainty_lists_are_disjoint():
    """Regression guard for gap #4: no marker may appear in both lists."""
    from app.pipelines.psycholinguistic import analyzer as mod

    overlap = set(mod._HEDGE_PHRASES) & set(mod._CERTAINTY_MARKERS)
    assert not overlap, f"markers scored twice: {sorted(overlap)}"


def test_tentative_markers_list_is_gone():
    """The tentative list was deleted; a revert must be loud."""
    from app.pipelines.psycholinguistic import analyzer as mod

    assert not hasattr(mod, "_TENTATIVE_MARKERS")


def test_tentative_text_no_longer_raises_certainty(analyzer):
    """Tentative-only text is hedging's job now: certainty stays low while
    hedging stays high (each signal counted exactly once)."""
    text = "I think maybe it was sort of possibly like that, I guess."
    doc = analyzer.nlp(text)
    certainty = analyzer._score_certainty(doc, text)
    hedging = analyzer._score_hedging(doc)
    assert certainty.score <= 35.0, (
        f"tentative text should barely move over-certainty "
        f"(got {certainty.score})"
    )
    assert hedging.score > 50.0, f"hedging must still fire (got {hedging.score})"


def test_emphatic_text_still_maxes_certainty(analyzer):
    """The over-certainty signal is untouched by the disjoint."""
    text = "I absolutely definitely swear, 100%, I never ever did it, no doubt."
    doc = analyzer.nlp(text)
    certainty = analyzer._score_certainty(doc, text)
    assert certainty.score >= 60.0, f"emphatic text must score high (got {certainty.score})"
```

- [ ] **Step 2: Run to verify the right ones fail**

Run: `.venv/Scripts/python -m pytest tests/psycholinguistic/test_analyzer.py -q -k "disjoint or tentative or emphatic"`
Expected: `test_hedge_and_certainty_lists_are_disjoint` FAILS (8 overlapping markers via `_TENTATIVE_MARKERS`... note: the overlap test compares `_HEDGE_PHRASES` vs `_CERTAINTY_MARKERS`, which are already disjoint sets of *words* — the true failure signal is `test_tentative_markers_list_is_gone` FAILS and `test_tentative_text_no_longer_raises_certainty` FAILS with a high certainty score). At least those two must fail; `test_emphatic_text_still_maxes_certainty` should already pass.

- [ ] **Step 3: Apply the disjoint in `analyzer.py`**

Read the current file first. Three edits:

(a) Delete the `_TENTATIVE_MARKERS = (...)` tuple (lines ~130-139) and its comment block. Replace the combined comment above `_CERTAINTY_MARKERS` (lines ~111-113) with:

```python
# Over-certainty / emphatic-assertion markers ("protesting too much").
# Session 6 / gap #4: tentative markers were REMOVED from this dimension --
# every one of them also appeared in _HEDGE_PHRASES, double-counting
# tentative language across two of the eight equally-weighted dimensions.
# Tentative language is owned solely by the hedging dimension; this
# dimension fires on emphatic absolutes only. Matched case-insensitively
# as substrings.
```

(b) Replace `_score_certainty` (keep the method name and signature) with:

```python
    def _score_certainty(self, doc: Doc, text: str) -> PsycholinguisticDimension:
        """Score over-certainty / emphatic assertion (dimension 8).

        Counts absolute-certainty markers ("definitely", "i swear", "100%")
        and folds in VADER's sentiment intensity (|compound|). Tentative
        language is deliberately NOT counted here -- it is the hedging
        dimension's signal (gap #4 disjoint; each marker counted once).
        ``doc`` is accepted for interface symmetry with the other scorers.
        """
        text_lower = text.lower()
        certainty_hits = [m for m in _CERTAINTY_MARKERS if m in text_lower]
        marker_count = sum(text_lower.count(m) for m in certainty_hits)

        compound = self.vader.polarity_scores(text)["compound"]
        intensity = abs(compound)

        score = min(
            100.0,
            marker_count * _CERTAINTY_POINTS_PER_MARKER
            + intensity * _CERTAINTY_VADER_WEIGHT,
        )
        evidence = [
            f"overcertainty_markers={marker_count}",
            f"vader_compound={compound:.3f}",
        ]
        if certainty_hits:
            evidence.append("markers=" + ", ".join(sorted(set(certainty_hits))))
        return PsycholinguisticDimension(score=score, evidence=evidence)
```

(c) In the module docstring's dimension list (and any "certainty vs tentative" wording in the file), rename dimension 8 to "over-certainty / emphatic assertion".

- [ ] **Step 4: Update the schema docstring**

In `backend/shared/schemas/psycholinguistic.py`, module docstring dimension list, change:

```
    8. certainty              over-certain vs. tentative language
```

to:

```
    8. certainty              over-certainty / emphatic assertion (tentative
                              language is the hedging dimension's signal --
                              gap #4 disjoint, Session 6)
```

(Match the exact current wording when editing — read the file first.)

- [ ] **Step 5: Run the psycholinguistic + streaming suites; re-verify ranges**

Run: `.venv/Scripts/python -m pytest tests/psycholinguistic/ tests/streaming/ -q`
Expected: the four new tests pass. If any PRE-EXISTING certainty/composite test fails, inspect it: if it asserted the old double-count behavior (e.g., a tentative-marker text expected to raise certainty), update the assertion to the new semantics and RECORD the justification in your task report. The convergence test must be green (stream and batch move together by construction).

- [ ] **Step 6: Commit**

```bash
git add backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py backend/shared/schemas/psycholinguistic.py tests/psycholinguistic/test_analyzer.py
git commit -m "fix(psycholinguistic): disjoint hedging/certainty lists -- each signal counted once (gap #4)"
```

---

## Task 2: UnsupportedLanguageError + Analyzer Gate (gap #8 near-term, part 1)

**Files:**
- Modify: `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py`
- Create: `backend/ml-inference/app/pipelines/psycholinguistic/__init__.py`
- Test: `tests/psycholinguistic/test_language_gate.py`

**Interfaces:**
- Produces:
  - `SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en"})` (module constant)
  - `class UnsupportedLanguageError(ValueError)` in `analyzer.py`
  - `PsycholinguisticAnalyzer.analyze(self, statements: list[str], language: str | None = None) -> PsycholinguisticScore` — `language=None` preserves legacy behavior; non-None non-English raises BEFORE the empty-statement check and before any spaCy work.
  - Package `__init__.py` re-exporting `PsycholinguisticAnalyzer`, `UnsupportedLanguageError`, `SUPPORTED_LANGUAGES` (mirrors the transcription package pattern).

- [ ] **Step 1: Write the failing tests**

Create `tests/psycholinguistic/test_language_gate.py`:

```python
"""Language gate tests (gap #8 near-term): non-English never flows silently."""
from __future__ import annotations

import pytest

from app.pipelines.psycholinguistic.analyzer import (
    SUPPORTED_LANGUAGES,
    PsycholinguisticAnalyzer,
    UnsupportedLanguageError,
)

_STMTS = ["I think I was at home.", "I never went there."]


@pytest.fixture(scope="module")
def gate_analyzer():
    return PsycholinguisticAnalyzer()


def test_supported_languages_constant():
    assert SUPPORTED_LANGUAGES == frozenset({"en"})


@pytest.mark.parametrize("lang", ["en", "EN", "en-US", "en_GB", "en-au"])
def test_english_variants_accepted(gate_analyzer, lang):
    result = gate_analyzer.analyze(_STMTS, language=lang)
    assert result.statement_count == 2


@pytest.mark.parametrize("lang", ["es", "fr", "zh", "de", "ja"])
def test_non_english_raises(gate_analyzer, lang):
    with pytest.raises(UnsupportedLanguageError) as exc_info:
        gate_analyzer.analyze(_STMTS, language=lang)
    msg = str(exc_info.value)
    assert lang.lower() in msg
    assert "en" in msg  # names the supported set
    # Invariant #3: the error must not leak transcript text.
    assert "home" not in msg and "never went" not in msg


def test_none_language_preserves_legacy_behavior(gate_analyzer):
    result = gate_analyzer.analyze(_STMTS)  # no language kwarg at all
    assert result.statement_count == 2


def test_gate_precedes_empty_statement_check(gate_analyzer):
    """A Spanish EMPTY transcript reports the language error -- the more
    actionable fact -- not 'No statements provided'."""
    with pytest.raises(UnsupportedLanguageError):
        gate_analyzer.analyze([], language="es")


def test_empty_statements_still_raise_valueerror_when_english(gate_analyzer):
    with pytest.raises(ValueError, match="No statements provided"):
        gate_analyzer.analyze([], language="en")


def test_error_is_a_valueerror_subclass():
    assert issubclass(UnsupportedLanguageError, ValueError)


def test_package_reexports():
    from app.pipelines.psycholinguistic import (  # noqa: F401
        PsycholinguisticAnalyzer as A,
        SUPPORTED_LANGUAGES as S,
        UnsupportedLanguageError as E,
    )
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/psycholinguistic/test_language_gate.py -q`
Expected: FAIL — `ImportError: cannot import name 'SUPPORTED_LANGUAGES'`.

- [ ] **Step 3: Implement the gate in `analyzer.py`**

(a) Add near the top-of-module constants:

```python
# Languages this vector can score. English-only for v1 (gap #8): the spaCy
# model, NRCLex, VADER, and every word list above are English, and the
# research base (Newman et al., Perez-Rosas corpora) is English. NEVER
# translate-then-score: MT destroys the surface stylometry being measured.
# A future per-language resource pack registers its code here.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en"})


class UnsupportedLanguageError(ValueError):
    """Transcript language cannot be scored by this English-only vector.

    The message names the language code and the supported set -- never
    transcript text (CLAUDE.md invariant #3).
    """


def _primary_subtag(language: str) -> str:
    """'en-US' / 'en_GB' / 'EN' -> 'en' (BCP-47 primary subtag, lowered)."""
    return language.strip().lower().replace("_", "-").split("-", 1)[0]
```

(b) Change `analyze`'s signature and add the gate as its FIRST statements (before the existing `if not statements:` check):

```python
    def analyze(
        self,
        statements: list[str],
        language: str | None = None,
    ) -> PsycholinguisticScore:
```

Docstring: add an `Args:` line — `language: BCP-47 code from Transcript.language. None (default) skips the gate for legacy text-only callers; a non-English value raises UnsupportedLanguageError before any scoring (gap #8).` And a `Raises:` line for `UnsupportedLanguageError`.

Body, first lines:

```python
        if language is not None and _primary_subtag(language) not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"language {language!r} is not supported by the "
                f"psycholinguistic vector; supported: "
                f"{', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )
        if not statements:
            raise ValueError("No statements provided")
```

(c) Create `backend/ml-inference/app/pipelines/psycholinguistic/__init__.py`:

```python
"""Psycholinguistic pipeline package."""
from app.pipelines.psycholinguistic.analyzer import (
    SUPPORTED_LANGUAGES,
    PsycholinguisticAnalyzer,
    UnsupportedLanguageError,
)

__all__ = [
    "PsycholinguisticAnalyzer",
    "UnsupportedLanguageError",
    "SUPPORTED_LANGUAGES",
]
```

- [ ] **Step 4: Run the gate tests, then both suites**

Run: `.venv/Scripts/python -m pytest tests/psycholinguistic/test_language_gate.py -q`
Expected: all pass.
Run: `.venv/Scripts/python -m pytest tests/psycholinguistic/ tests/streaming/ -q`
Expected: all pass (the new `__init__.py` must not break the existing direct-module imports in conftest/scorer).

- [ ] **Step 5: Commit**

```bash
git add backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py backend/ml-inference/app/pipelines/psycholinguistic/__init__.py tests/psycholinguistic/test_language_gate.py
git commit -m "feat(psycholinguistic): hard-fail language gate -- UnsupportedLanguageError before any scoring (gap #8 near-term)"
```

---

## Task 3: Stream-Entry Gate (gap #8 near-term, part 2)

**Files:**
- Modify: `backend/ml-inference/app/pipelines/streaming/windowed_scorer.py`
- Test: `tests/streaming/test_windowed_scorer.py` (extend)

**Interfaces:**
- Consumes: `UnsupportedLanguageError`, `SUPPORTED_LANGUAGES`, `_primary_subtag` — import the first two from `app.pipelines.psycholinguistic.analyzer`; the gate check in the scorer reuses the analyzer's helper (import `_primary_subtag` too — module-private by convention but this is the same service).
- Produces: `stream_scores` raises `UnsupportedLanguageError` on first iteration for non-English transcripts; zero events emitted. ScoreEvent schema and `validate_event_stream` untouched.

- [ ] **Step 1: Write the failing tests**

Append to `tests/streaming/test_windowed_scorer.py` (match its existing transcript-builder helpers — read the file first; it has fixtures/helpers for building synthetic Transcripts; reuse them, adjusting only the `language=` field):

```python
# ---- Session 6 / gap #8: language gate at stream entry ---------------------


def test_non_english_transcript_raises_on_first_iteration():
    from app.pipelines.psycholinguistic.analyzer import UnsupportedLanguageError

    transcript = _make_transcript(  # reuse this module's existing helper
        texts=["Hola, estaba en casa.", "Nunca fui alli."],
        language="es",
    )
    gen = stream_scores(transcript)
    with pytest.raises(UnsupportedLanguageError, match="es"):
        next(gen)


def test_non_english_transcript_emits_zero_events():
    from app.pipelines.psycholinguistic.analyzer import UnsupportedLanguageError

    transcript = _make_transcript(
        texts=["Hola, estaba en casa."], language="es"
    )
    events = []
    with pytest.raises(UnsupportedLanguageError):
        for ev in stream_scores(transcript):
            events.append(ev)
    assert events == []


def test_english_regional_variant_streams_normally():
    transcript = _make_transcript(
        texts=["I think I was at home.", "I never went there."],
        language="en-US",
    )
    events = list(stream_scores(transcript))
    assert events, "en-US must stream"
    assert events[-1].kind.value == "final"
```

> NOTE to implementer: `_make_transcript` is the name of whatever transcript-building helper already exists in this test module — read the file and use the actual helper/fixture name and signature, passing the language through. If the existing helper hardcodes `language="en"`, add a `language: str = "en"` parameter to it.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_windowed_scorer.py -q -k "language or english or non_english"`
Expected: the two non-English tests FAIL (no error raised — events stream through the English pipeline); the en-US test may already pass.

- [ ] **Step 3: Add the entry gate to `stream_scores`**

In `windowed_scorer.py`:

(a) Extend the psycholinguistic import (line ~32):

```python
from app.pipelines.psycholinguistic.analyzer import (
    SUPPORTED_LANGUAGES,
    PsycholinguisticAnalyzer,
    UnsupportedLanguageError,
    _primary_subtag,
)
```

(b) At the top of `stream_scores`'s body — BEFORE `cfg = config or StreamScorerConfig()` — insert:

```python
    # Gap #8 gate: never let a non-English transcript flow silently through
    # the English-only analyzer. Raised before any event is emitted, so the
    # ScoreEvent contract is untouched: an empty stream still means exactly
    # one thing (zero statements). NOTE: generators defer execution -- this
    # raise surfaces on the caller's first next()/iteration.
    if _primary_subtag(transcript.language) not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(
            f"language {transcript.language!r} is not supported by the "
            f"psycholinguistic vector; supported: "
            f"{', '.join(sorted(SUPPORTED_LANGUAGES))}"
        )
```

Also update the function docstring: add a `Raises: UnsupportedLanguageError` note stating it fires on first iteration for non-English transcripts.

(c) While there, pass the language down so batch and stream share one gate path: in `_analyze_slice`, no change (slices come from an already-gated transcript). Do NOT double-gate per tick.

- [ ] **Step 4: Run the streaming suite including convergence**

Run: `.venv/Scripts/python -m pytest tests/streaming/ -q`
Expected: all pass, including `test_convergence.py`.

- [ ] **Step 5: Commit**

```bash
git add backend/ml-inference/app/pipelines/streaming/windowed_scorer.py tests/streaming/test_windowed_scorer.py
git commit -m "feat(streaming): gate non-English transcripts at stream entry -- zero events, contract untouched (gap #8)"
```

---

## Task 4: CLI Handling (gap #8 near-term, part 3)

**Files:**
- Modify: `scripts/replay_scores.py`
- Modify: `scripts/test_compress_and_analyze.py`
- Test: `tests/streaming/test_cli_smoke.py` (extend)

**Interfaces:**
- Consumes: `UnsupportedLanguageError` (from `app.pipelines.psycholinguistic.analyzer`), `Transcript.language` (existing field).
- Produces: both CLIs exit 1 with a friendly stderr message naming the language; `test_compress_and_analyze.py` now passes `language=transcript.language` into `analyze()`.

- [ ] **Step 1: Write the failing CLI smoke test**

Append to `tests/streaming/test_cli_smoke.py` (read the file first — it already builds a Transcript JSON in tmp_path and runs `scripts/replay_scores.py` as a subprocess; mirror that setup):

```python
def test_replay_cli_rejects_non_english_transcript(tmp_path):
    from backend.shared.schemas.transcription import Transcript, TranscriptSegment

    transcript = Transcript(
        segments=[
            TranscriptSegment(
                text="Hola, estaba en casa.", start_seconds=0.0, end_seconds=2.0
            )
        ],
        language="es",
        audio_duration_seconds=2.0,
        model_name="fake-distil",
        backend="fake",
    )
    p = tmp_path / "es_transcript.json"
    p.write_text(transcript.model_dump_json(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "replay_scores.py"),
            "--transcript", str(p), "--pace", "0",
        ],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "es" in result.stderr
    assert "not supported" in result.stderr
    # Invariant #3: no transcript text in the error surface.
    assert "Hola" not in result.stderr and "casa" not in result.stderr
```

(Use the module's existing `subprocess`/`sys`/`_REPO_ROOT` imports — they exist already.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_cli_smoke.py -q -k "non_english"`
Expected: FAIL — today the CLI streams the Spanish transcript through the English pipeline and exits 0 (or crashes with the new gate's raw traceback if Tasks 2-3 landed: either way not the clean exit-1 + message).

- [ ] **Step 3: Catch in `replay_scores.py`**

Wrap the replay loop (currently `for event in ScoreReplayer(transcript, config).replay(pace=args.pace):` at line ~144). Add the import inside `main()` next to the existing `from app.pipelines.streaming import ScoreReplayer`:

```python
    from app.pipelines.psycholinguistic.analyzer import UnsupportedLanguageError
    from app.pipelines.streaming import ScoreReplayer
```

and wrap:

```python
    count = 0
    try:
        for event in ScoreReplayer(transcript, config).replay(pace=args.pace):
            print(_format_event(event), flush=True)
            count += 1
    except UnsupportedLanguageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "The psycholinguistic vector is English-only for v1 (CLAUDE.md "
            "gap #8); transcripts are never translate-then-scored.",
            file=sys.stderr,
        )
        return 1
```

- [ ] **Step 4: Pass the language + catch in `test_compress_and_analyze.py`**

In Step 3 of that script (psycholinguistic analysis section), change:

```python
    score = PsycholinguisticAnalyzer().analyze(statements)
```

to:

```python
    try:
        score = PsycholinguisticAnalyzer().analyze(
            statements, language=transcript.language
        )
    except UnsupportedLanguageError as exc:
        print(f"  Skipped: {exc}", file=sys.stderr)
        return 1
```

with the import updated to:

```python
    from app.pipelines.psycholinguistic.analyzer import (
        PsycholinguisticAnalyzer,
        UnsupportedLanguageError,
    )
```

- [ ] **Step 5: Run the CLI smokes, then the full suite**

Run: `.venv/Scripts/python -m pytest tests/streaming/test_cli_smoke.py -q`
Expected: all pass (old smokes + the new rejection test).
Run: `.venv/Scripts/python -m pytest tests/ -p no:cacheprovider 2>&1 | tail -3`
Expected: full suite green, 1 deselected (slow).

- [ ] **Step 6: Commit**

```bash
git add scripts/replay_scores.py scripts/test_compress_and_analyze.py tests/streaming/test_cli_smoke.py
git commit -m "feat(cli): friendly exit-1 on unsupported transcript language (gap #8)"
```

---

## Task 5: spaCy sm → md + Doc Sync (gap #7 item + gap statuses)

**Files:**
- Modify: `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py` (`_SPACY_MODEL`, install-hint string)
- Modify: `Makefile` (`install-spacy-model` target)
- Modify: `pyproject.toml` (spaCy model comment)
- Modify: `CLAUDE.md` (tooling row; gap #4 resolved; gap #8 near-term done; gap #7 spaCy item done)

**Interfaces:**
- Produces: `_SPACY_MODEL = "en_core_web_md"`. No API changes.

- [ ] **Step 1: Download the md model FIRST (so the suite can run)**

```bash
.venv/Scripts/python -m spacy download en_core_web_md
.venv/Scripts/python -c "import spacy; spacy.load('en_core_web_md'); print('md ok')"
```
Expected: `md ok` (~40 MB download — acceptable per project stance).

- [ ] **Step 2: Flip the constant + hint**

In `analyzer.py`: `_SPACY_MODEL = "en_core_web_sm"` → `_SPACY_MODEL = "en_core_web_md"`, and the comment above it: `# spaCy model used for all parsing. md (~40 MB): better vectors/NER than sm -- detail-specificity depends on NER quality (gap #7).` The lazy-loader's RuntimeError hint already interpolates `_SPACY_MODEL` — verify, don't duplicate.

In `Makefile`, `install-spacy-model` target: `en_core_web_sm` → `en_core_web_md`.

In `pyproject.toml`, the dependencies comment mentioning `en_core_web_sm`: update to `en_core_web_md` (install via `python -m spacy download en_core_web_md`).

- [ ] **Step 3: Run the FULL suite and re-verify range tests**

Run: `.venv/Scripts/python -m pytest tests/ -p no:cacheprovider 2>&1 | tail -5`
Expected: green. md's NER is strictly better, so directional assertions (specific text scores LOW on the specificity-anomaly dimension, vague text HIGH) should hold. If a range test genuinely breaks: inspect whether md's entity extraction legitimately moved the value across the asserted threshold; adjust the threshold ONLY with a one-line justification recorded in your task report (e.g., "md finds 2 more entities in the fixture sentence; direction unchanged"). The convergence test must be green.

- [ ] **Step 4: CLAUDE.md sync (four edits)**

1. Psycholinguistic Analysis Stack table (and the Phase 2 status table's Tooling row): `spaCy en_core_web_sm` → `spaCy en_core_web_md`.
2. Known gaps item 4 → struck through + resolved:
```markdown
4. ~~Hedging and certainty scorers double-count tentative markers~~ **RESOLVED
   (Session 6):** `_TENTATIVE_MARKERS` deleted; dimension 8 is over-certainty /
   emphatic assertion only; tentative language is owned solely by hedging. A
   list-disjointness regression test pins it.
```
3. Known gaps item 8: mark the near-term step done:
```markdown
   ... **Near-term gate SHIPPED (Session 6):** `UnsupportedLanguageError` raised
   before any scoring at `analyze(language=...)` and `stream_scores` entry
   (zero events; ScoreEvent schema untouched); CLIs exit 1 with the language
   named. Hard-fail is the v1 behavior; the late-fusion session adds graceful
   per-vector degradation. Expansion path (per-language packs) unchanged below.
```
(Insert into the existing item 8 text after the "Near term:" sentence, adapting to the exact current wording — read the file first.)
4. Known gaps item 7: remove the spaCy clause (now done), keeping the remaining smaller items.

- [ ] **Step 5: Final verification + commit**

Run: `.venv/Scripts/python -m pytest tests/ -p no:cacheprovider 2>&1 | tail -3`
Expected: green, 1 deselected.
Run: `.venv/Scripts/python scripts/test_psycholinguistic.py --text "I absolutely swear I never did that, 100 percent."`
Expected: certainty dimension high, output clean, exit 0.

```bash
git add backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py Makefile pyproject.toml CLAUDE.md
git commit -m "feat(psycholinguistic): spaCy sm->md for NER quality; CLAUDE.md gap sync (gaps #4, #7, #8)"
```

---

## Final Verification

- [ ] `.venv/Scripts/python -m pytest tests/ -p no:cacheprovider` → all green, 1 deselected (slow).
- [ ] `.venv/Scripts/python -m pytest tests/streaming/test_convergence.py -q` → green (batch and stream moved together through all scoring changes).
- [ ] Replay CLI happy path: `.venv/Scripts/python scripts/replay_scores.py --video demo_data/honest/trial_truth_001.mp4 --fake --pace 0` → events + final, exit 0.
- [ ] `git status --short` → clean.

---

## Self-Review Checklist

- [x] Spec change 1 (disjoint) → Task 1, with the literal disjointness guard AND the `_TENTATIVE_MARKERS`-absence guard (the honest failing test, since the two surviving lists never overlapped directly).
- [x] Spec change 2 (gate) → Tasks 2-4: analyzer gate (precedes empty-check), stream entry gate (raises on first iteration, zero events, schema untouched), CLI catch (exit 1, invariant #3 — no transcript text in stderr, asserted in tests).
- [x] Spec change 3 (sm→md) → Task 5, model downloaded before the constant flips; range-test policy (justify-or-leave) stated verbatim.
- [x] Hard-fail-for-now + ensemble-successor decision recorded in CLAUDE.md item 8 edit.
- [x] Convergence gate required green in every scoring-affecting task (1, 3, 5).
- [x] Type consistency: `UnsupportedLanguageError`, `SUPPORTED_LANGUAGES`, `_primary_subtag`, `analyze(statements, language=None)` identical across Tasks 2-4.
- [x] No placeholders; every step has code or exact commands with expected outcomes.
- [x] Namespace-package convention respected: new `psycholinguistic/__init__.py` mirrors the transcription package precedent (regular package inside the namespace tree — same as `transcription/`).
