# ALICE — Psycholinguistic Quality + Language Gate (Session 6)
## Design Spec · 2026-07-16

---

## Overview

Three contained improvements to the psycholinguistic vector, closing CLAUDE.md
gap #4, the near-term step of gap #8, and the spaCy item of gap #7:

1. **Disjoint the hedging/certainty word lists (gap #4).** Every one of the 8
   `_TENTATIVE_MARKERS` also appears in `_HEDGE_PHRASES`, so tentative words
   ("maybe", "i think", "sort of") raise TWO of the eight equally-weighted
   dimensions — a hidden ~2× weight on tentative language. Fix: dimension 8
   becomes **over-certainty / emphatic assertion** only; tentative language is
   owned solely by the hedging dimension. Each signal counted exactly once.
2. **Transcript-language gate (gap #8 near-term).** The analyzer's tooling and
   research base are English-only, but nothing stops a Spanish transcript from
   flowing silently through English lexicons today. Fix: a typed
   **hard fail** (`UnsupportedLanguageError`) raised BEFORE any scoring — at the
   analyzer when a language is supplied, and at `stream_scores` entry from
   `transcript.language`. **Never translate-then-score** (MT destroys the surface
   stylometry being measured).
3. **spaCy `en_core_web_sm` → `en_core_web_md` (gap #7 item).** Better NER for
   the detail-specificity dimension (~40 MB download — acceptable per the
   project's large-download stance).

**Decisions locked during brainstorming:**

- **Hard fail, for now.** Today psycholinguistic is the only implemented vector,
  so a gated transcript means no analysis exists — an error is the only honest
  output. Successor decision recorded: when the late-fusion ensemble ships, the
  fusion layer catches `UnsupportedLanguageError` and degrades gracefully (marks
  this vector unavailable, scores the others) instead of failing the job.
- **No ScoreEvent schema change.** The gate fires before the first event is
  emitted, so `cumulative`-is-required never comes into play. An empty stream
  continues to mean exactly one thing (zero-statement transcript / silence);
  unsupported language is an exception, never an empty stream.
- **Scoring philosophy unchanged (explicitly reaffirmed):** ALICE emits
  continuous 0–100 anomaly scores + confidence + qualitative labels — never
  hard deceptive/truthful verdicts (that is CyberQ's model, rejected: at a ~75%
  F1 ceiling a verdict is wrong 1-in-4). Nothing in this session touches that:
  dimension scores stay 0–100 into a 0–100 composite, ensemble/dev-facing
  (invariant #5).

---

## Change 1 — Word-list disjoint (`analyzer.py`)

Current `_score_certainty`:

```python
certainty_hits = [m for m in _CERTAINTY_MARKERS if m in text_lower]   # over-certainty
tentative_hits = [m for m in _TENTATIVE_MARKERS if m in text_lower]   # ALSO all in _HEDGE_PHRASES
marker_count   = counts of BOTH
score = min(100, marker_count * _CERTAINTY_POINTS_PER_MARKER
                 + |vader compound| * _CERTAINTY_VADER_WEIGHT)
```

After:

- **Delete `_TENTATIVE_MARKERS`** entirely.
- `_score_certainty` counts `_CERTAINTY_MARKERS` only (unchanged list:
  "definitely", "absolutely", "certainly", "undoubtedly", "no doubt",
  "without a doubt", "100%", "for sure", "i guarantee", "i swear", "totally",
  "completely", "always", "never") plus the existing VADER-intensity term.
- Docstrings updated: dimension 8 is "over-certainty / emphatic assertion
  (protesting-too-much signal)"; module docstring and the schema docstring in
  `backend/shared/schemas/psycholinguistic.py` (dimension list) note the
  disjoint and why (gap #4). Evidence strings keep the same format.
- **Signal accounting after the fix:** "I think maybe I saw him" → hedging
  rises (2 hedges), certainty flat. "I absolutely swear I never did it, 100%"
  → certainty maxes exactly as before. No signal lost; each counted once.
- **Composite effect:** hedgy text scores drop by roughly one dimension's
  double-count; the Session 5 convergence gate moves with it automatically
  (batch and stream share the analyzer — that was the point of the contract).

### Regression guard

A literal disjointness test so the bug cannot return:

```python
def test_hedge_and_certainty_lists_are_disjoint():
    assert not (set(_HEDGE_PHRASES) & set(_CERTAINTY_MARKERS))
```

(plus: `_TENTATIVE_MARKERS` no longer exists — an import test asserts its
absence so a revert is loud.)

---

## Change 2 — Language gate

### New exception (`backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py`)

```python
class UnsupportedLanguageError(ValueError):
    """Raised when a transcript's language cannot be scored by this
    English-only vector. Message names the language code and the supported
    set -- never transcript text (invariant #3)."""
```

Exported from the package `__init__` alongside the analyzer.

### Analyzer API

```python
def analyze(self, statements: list[str],
            language: str | None = None) -> PsycholinguisticScore:
```

- `language=None` (default): no gate — preserves existing text-only callers
  (dev CLIs fed raw strings have no language metadata).
- `language` supplied: normalized (case-insensitive; `en`, `en-us`, `en-GB`…
  → accepted iff the primary subtag is `en`); anything else raises
  `UnsupportedLanguageError("language 'es' is not supported by the
  psycholinguistic vector; supported: en")` BEFORE any spaCy/lexicon work.

### Gate points (every Transcript-driven path passes the language)

| Path | Change |
|---|---|
| `stream_scores(transcript, ...)` | Gates at generator entry on `transcript.language` — raises before the first event. ScoreEvent schema untouched. |
| `scripts/replay_scores.py` | Catches `UnsupportedLanguageError` → stderr message, exit 1 |
| `scripts/test_compress_and_analyze.py` | Passes `transcript.language` to `analyze()`; catches → friendly message, exit 1 |
| `scripts/test_psycholinguistic.py` | Unchanged (raw text input, no language metadata — documented) |

Note on the generator: the gate check runs in `stream_scores` **before the
first `yield`-containing loop executes**; because generators defer execution,
the raise surfaces on first `next()`/iteration — tests assert the error fires
on iteration, and the replayer/CLI naturally propagate it.

### What "supported" means going forward

Constant `SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en"})` — the
per-language expansion path (resource packs, dimension masks, per-language
validation corpora) is gap #8's long-term plan; this constant is where a new
pack registers.

---

## Change 3 — spaCy sm → md

- `_SPACY_MODEL = "en_core_web_md"` in `analyzer.py`; the RuntimeError install
  hint and Makefile `install-spacy-model` target updated to match; pyproject
  comment updated.
- Model installed locally during implementation
  (`python -m spacy download en_core_web_md`, ~40 MB).
- **Test policy:** the detail-specificity and other range-based analyzer tests
  are re-run against md. Directional assertions (specific text < 50,
  vague text > 50, etc.) are expected to hold — md's NER is strictly better.
  Any test that genuinely breaks is adjusted ONLY with a recorded justification
  in the plan's task report (no silent threshold nudging).
- CLAUDE.md tooling row updated (`en_core_web_sm` → `en_core_web_md`).

---

## Error Handling Summary

| Condition | Behavior |
|---|---|
| `analyze(stmts)` — no language | Unchanged (legacy text-only path) |
| `analyze(stmts, language="en"/"en-US")` | Scores normally |
| `analyze(stmts, language="es")` | `UnsupportedLanguageError` before any scoring |
| `stream_scores` on non-en transcript | Raises on first iteration, zero events emitted |
| CLI on non-en input | stderr: language + supported set, exit 1 — no transcript text in the message |
| Empty statements | Existing `ValueError("No statements provided")` unchanged; gate check runs first (a Spanish empty transcript reports the language error, the more actionable fact) |

Invariants: #3 (errors/logs carry language codes, never text), #5 (outputs stay
score+confidence+label — no verdicts), #6 (no forbidden phrase).

---

## Testing

| Suite | Coverage |
|---|---|
| `tests/psycholinguistic/test_analyzer.py` (extended) | List-disjointness regression guard; `_TENTATIVE_MARKERS` absence; certainty rescoring (emphatic text still high, tentative-only text now low on certainty and unchanged-high on hedging); composite shift sanity |
| `tests/psycholinguistic/test_language_gate.py` (new) | en / en-US / EN accepted; es/fr/None-vs-supplied semantics; error message contains code + supported set and no statement text; gate precedes empty-statement check |
| `tests/streaming/` (extended) | Non-en transcript → raises on first iteration, zero events; en transcript unchanged; convergence gate re-baselined and green |
| CLI smokes (extended) | replay + compress-analyze on a non-en transcript JSON → exit 1 + message |
| Full suite | Green; the convergence gate proves batch and stream moved together |

---

## Out of Scope

- Per-language resource packs / dimension masks / validation corpora (gap #8
  long-term)
- Ensemble graceful-degradation on `UnsupportedLanguageError` (successor
  decision — lands with the late-fusion session)
- Disfluency relocation to the vocal-tonality vector (gap #3)
- Learned (XGBoost) dimension weights replacing equal weights (Phase 3)
