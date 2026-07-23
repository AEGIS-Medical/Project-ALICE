# CLAUDE.md — Project ALICE Master Guide

> **This file is the single source of truth for all Claude Code agents working on this project.**
> Every agent — regardless of role — must read this file in full before writing any code.
> If this file conflicts with any other documentation, this file wins.

---

## PROJECT IDENTITY

**Name:** Project ALICE
**Full Name:** Advanced Logic Integrity & Consistency Examiner
**Type:** Mobile-first multimodal behavioral analysis platform
**Tagline:** Behavioral anomaly detection for video calls and uploaded recordings
**License:** Proprietary (all rights reserved)
**Primary Languages:** Python (backend/ML), Kotlin (mobile), TypeScript (tooling)

---

## WHAT THIS PROJECT DOES

ALICE analyzes video calls and uploaded video files for behavioral indicators of deception using five complementary analysis vectors, informed by Pérez-Rosas et al. (EMNLP 2015, ICMI 2015) and Li & Abouelenien (IEEE ISPDS 2024), adapted with modern components.

### The Five Analysis Vectors

1. **Facial Action Unit Analysis (Visual — Primary Weight: 0.30)** — Automated FACS Action Unit detection from video. Tracks AU12 without AU6 (non-Duchenne smile), AU1+AU4 (cognitive load brow activity), AU14 (contempt dimpler), bilateral asymmetry, and microexpression onset timing. Pérez-Rosas found removing facial displays dropped accuracy from 77% to 64% — the single largest contributor.

2. **Psycholinguistic Analysis (Linguistic — Primary Weight: 0.30)** — Full psycholinguistic profiling beyond simple contradiction detection: pronoun pattern shifts (first-person singular drops during deception per Newman et al. 2003), hedging frequency, cognitive complexity (subordinate clause depth), emotional word distribution (deceivers use more Anxiety/Anger words, truth-tellers use more Family/Home/Humans words per Pérez-Rosas EMNLP 2015), speech disfluencies (um/uh/er frequency), negation frequency, and detail specificity (named entity density). Li & Abouelenien confirmed linguistic features are the strongest single modality (~80% accuracy).

3. **Vocal Tonality Flux (Audio — Weight: 0.20)** — Speech emotion embeddings (emotion2vec+) and acoustic features (F0, jitter, shimmer, HNR) measuring rate of change relative to subject's personal baseline. Supplemented by speaking rate variability and pause pattern analysis.

4. **Statement Contradiction Detection (NLP — Weight: 0.15)** — Speaker-attributed transcription via WhisperX, semantic embedding of statements, cross-session contradiction detection using DeBERTa-v3 NLI against per-contact statement history in pgvector.

5. **Subject Identification & Tracking (Infrastructure — Weight: 0.05 for gaze subcomponent)** — Active speaker detection (LR-ASD: which face is speaking), speaker diarization (pyannote/WhisperX: who said what), and privacy-preserving face re-identification (EdgeFace-XS: 512-D embeddings, on-device, never store raw faces) for per-contact baseline maintenance.

### Research-Informed Architectural Decisions

These are non-negotiable choices backed by published evidence:

- **Late fusion, not early fusion.** Li & Abouelenien demonstrated independently-trained modality-specific models with score-level fusion outperform early fusion. Each vector produces an independent score; the ensemble combines them via XGBoost + Platt calibration.
- **Linguistic analysis is a primary channel.** Both Li & Abouelenien and Pérez-Rosas found linguistic features to be the most robust single modality. Our architecture co-weights linguistic and facial AU signals highest.
- **Gaze direction is LOW-WEIGHT.** A 2023 expert consensus survey (Luke et al.) found >80% of experts agreed liars and truth-tellers do not meaningfully differ in gaze aversion. Pérez-Rosas found motivated liars INCREASE eye contact. We score both extremes as anomalous but assign low ensemble weight (0.05). Pupil dilation is more promising but requires specialized hardware.
- **Microexpression detection is experimental only.** Microexpressions last 40-200ms, require ≥60fps, occur infrequently, and yield near-chance accuracy alone. Supplementary R&D signal, not primary.
- **Per-subject baseline is ALICE's key differentiator.** No competitor offers this. Behavioral patterns are highly individual. First 2-3 interactions establish baseline; accuracy improves with each session.
- **~75% F1 is the realistic accuracy ceiling.** On DOLOS (1,675 clips, largest realistic benchmark), best published result is 75.5% F1 (Vallabhaneni 2025). Claims of >90% are overfitted on tiny datasets. ALICE must be honest about this.

### What ALICE Is NOT

ALICE is **not a lie detector.** It is a behavioral anomaly detection system. Scores represent deviation from an individual's established baseline — not ground truth.

**Never use the phrase "lie detector"** in any code comment, UI string, API response, or documentation.

---

## COMPETITIVE LANDSCAPE

| Competitor | Modalities | Per-Subject Baseline | Real-Time | Call Integration | Claim |
|-----------|-----------|---------------------|-----------|-----------------|-------|
| CyberQ.ai | Text only (LLM prompt on Azure OpenAI) | No | No | No | 92% (unvalidated) |
| Converus EyeDetect | Eye tracking only | No | Semi | No | 86-90% |
| Clearspeed | Voice only | Unclear | Yes | No | Undisclosed |
| Deceptio.AI | Text only | No | No | No | Beta |
| **ALICE** | **5-vector multimodal** | **Yes** | **Yes** | **Yes** | **~75% F1 (honest)** |

ALICE's advantages: only genuine multimodal fusion, only per-subject baselines, only live video call platform integration, only edge-first privacy architecture, only honest accuracy claims.

---

## AGENT TEAM STRUCTURE

### 🏗️ Lead Developer (`@lead`)
**Scope:** Architecture, cross-service integration, dependencies, monorepo structure, CI/CD, conflict resolution.
**Owns:** `/`, `pyproject.toml`, `docker-compose.yml`, `Makefile`, `/proto/`, `/docs/architecture/`
**Authority:** Final say on all technical disputes.

### 🔍 Code Review Agent (`@reviewer`)
**Scope:** Code quality, type safety, test coverage, style enforcement.
**Owns:** `.pre-commit-config.yaml`, `ruff.toml`, `mypy.ini`, `/tests/`
**Rules:** `ruff` + `black` + `mypy --strict` on Python. `ktlint` + `detekt` on Kotlin. 80% min coverage, 90% for ML pipelines. Every public function has a docstring. Every ML input/output has shape assertions.

### 🔐 API Security Agent (`@security`)
**Scope:** Auth, encryption, secrets, OWASP compliance, consent flow, biometric data.
**Owns:** `/backend/auth/`, `/backend/api-gateway/middleware/`, `/infra/vault/`
**Rules:** JWT on all endpoints. Never log PII. Secrets from Vault only. Face embeddings are biometric data (BIPA/GDPR). Raw face images NEVER stored/transmitted/logged — only 512-D embeddings, encrypted.

### 🎨 UI Designer Agent (`@ui`)
**Scope:** All user-facing screens, components, design system.
**Owns:** `/mobile/shared/ui/`, `/mobile/androidApp/ui/`
**Rules:** Material 3. Dark theme only v1. Scores always include confidence + qualitative labels. Never show bare numbers.

### 🚀 DevOps Agent (`@devops`)
**Scope:** Docker, K8s, Terraform, CI/CD, monitoring, GPU management.
**Owns:** `/infra/`, `/docker/`, `/.github/workflows/`
**Rules:** Multi-stage Docker builds. Helm charts. GPU spots for retraining, reserved for inference. Zero-downtime deploys.

---

## TECH STACK — SELECTED COMPONENTS

### Subject Identification Pipeline (~13MB on-device)

| Component | Model | Size | License | Runs |
|-----------|-------|------|---------|------|
| Face Detection | SCRFD (via MediaPipe) | ~2MB | Apache 2.0 | Device |
| Active Speaker Detection | LR-ASD | ~4MB (0.84M params) | Train custom | Device |
| Face Re-ID | EdgeFace-XS | ~7MB (1.77M params) | Apache 2.0 | Device |
| Speaker Diarization (batch) | pyannote community-1 | ~50M params | MIT | Server |
| Speaker Diarization (stream) | diart wrapper | uses pyannote | MIT | Server |
| Who-Said-What | WhisperX | ~1.5B total | BSD-4 | Server |

Platform shortcut: Zoom/Teams/Meet provide per-participant audio streams = perfect diarization without a model.

### Facial Action Unit Pipeline

| Component | Model | Size | License | Runs |
|-----------|-------|------|---------|------|
| Face Alignment | MediaPipe Face Mesh | ~5MB | Apache 2.0 | Device |
| AU Detection (mobile) | Custom ResNet-18 INT8 TFLite | ~3-4MB | Own | Device |
| AU Detection (server) | ME-GraphAU architecture, own weights | ~50MB | Own | Server |
| Bilateral asymmetry | Custom post-processing | Code | Own | Both |

Deception-relevant AUs: AU12−AU6 (fake smile), AU1+AU4 (cognitive load), AU14 (contempt), AU15 (suppression), AU20 (fear), bilateral asymmetry (voluntary expression). Train on DISFA + BP4D datasets using LibreFace architecture as reference.

### Psycholinguistic Analysis Stack

| Tool | License | Purpose |
|------|---------|---------|
| spaCy v3.x | MIT | POS, deps, NER, pronouns, negation, complexity |
| Empath v0.89 | MIT | 200+ lexical categories, custom category generation |
| NRCLex v4.0 | MIT | 8 granular emotion categories |
| VADER v3.3.2 | MIT | Valence-aware sentiment |
| Custom BERT hedging classifier | Own | Modal verbs + epistemic phrases (NOT regex, 59% FP rate) |

Eight deception-relevant linguistic dimensions: (1) pronoun patterns, (2) hedging frequency, (3) cognitive complexity, (4) emotional word distribution, (5) speech disfluencies, (6) negation frequency, (7) detail specificity, (8) over-certainty / emphatic assertion (tentative language is measured by the hedging dimension).

### Vocal Tonality Stack

| Component | Model | License | Runs |
|-----------|-------|---------|------|
| Emotion embeddings | emotion2vec+ large (~300MB) | MIT | Server |
| Acoustic features | parselmouth/Praat | GPL-3.0 | Server |
| Handcrafted features | librosa | ISC | Both |
| Flux scorer | XGBoost per-contact (~2MB) | Apache 2.0 | Server |

### Transcription & Contradiction Stack

| Component | Model | License | Runs |
|-----------|-------|---------|------|
| Transcription (fast) | Whisper distil-large-v3 | MIT | Server |
| Transcription (accurate) | Whisper large-v3 | MIT | Server |
| Speaker-attributed | WhisperX | BSD-4 | Server |
| Statement embedding | all-mpnet-base-v2 | Apache 2.0 | Server |
| Contradiction NLI | DeBERTa-v3-large fine-tuned | MIT | Server |
| Vector store | pgvector | PostgreSQL | Server |

### Ensemble

- **Fusion:** Late fusion (score-level). Each vector produces independent 0-100 score.
- **Classifier:** XGBoost + Platt scaling, per-contact calibration.
- **Weights:** AU: 0.30, Psycholinguistic: 0.30, Tonality: 0.20, Contradiction: 0.15, Gaze: 0.05.
- **Explainability:** SHAP values per vector.
- **Confidence:** Low (1 session), Medium (2-3 sessions), High (4+ sessions).

---

## COMPRESSION ARCHITECTURE

### Tier 1: Lossless Audio (Always)
FLAC 48kHz mono. ALL ML models receive lossless. Opus generated separately for playback only.

### Tier 2: ROI Video
Face region QP -12 (near-lossless SSIM >0.98). Background CRF 28-32. 60-70% size reduction.

### Tier 3: Edge-First (Preferred for Live)
On-device: landmarks + AU data + face embeddings + audio features. Transmitted: landmark telemetry ~1.05 MB/min at 30 fps (ALTM protobuf; ~70 KB/min remains the target for the future on-device AU-activation payload) + FLAC audio (~2.5MB/min). No raw video leaves device.

Adaptive: ≥10Mbps→RAW, 5-10→ROI, 1-5→EDGE_FULL, <1→EDGE_MINIMAL.

### Storage Lifecycle
Hot (0-30d): full payload ~$1.15/user/mo. Warm (30-90d): no video ~$0.22. Cold (90-365d): scores+text ~$0.04. Purged (365+d): $0.

---

## PLATFORM CONNECTORS

```python
class PlatformConnector(ABC):
    async def authorize(self, user_id, auth_code) -> OAuthTokens: ...
    async def detect_meetings(self, user_id) -> list[MeetingEvent]: ...
    async def join_meeting(self, meeting_id, consent_tokens) -> MediaSession: ...
    async def stream_media(self, session) -> AsyncIterator[MediaChunk]: ...
    async def get_participant_audio_streams(self, session) -> dict[str, AudioStream]: ...
    async def get_recording(self, meeting_id) -> RecordingDownload: ...
    async def revoke_session(self, session) -> None: ...
```

Supported: Zoom (Meeting SDK), Teams (Graph API + Bot), Google Meet (REST API + Add-ons), Webex (Meetings API), Slack (Events API, audio-only), LiveKit (native WebRTC).

---

## UPSTREAM DEPENDENCIES

**jsugg/ser (MIT):** Adapted FeatureBackend protocol, Emotion2VecBackend, HandcraftedBackend, faster-whisper transcription, temporal pooling, post-processing hysteresis.

**MichiganNLP/deceptiondetection (Research Reference):** Papers and dataset descriptions for feature engineering. No code copied. Dataset (121 trial videos) available from authors for benchmarking.

**Available evaluation datasets:** Real-life Trial (121 videos), DOLOS (1,675 clips), MU3D (320 videos), Bag-of-Lies (325 samples), Box of Lies (game segments).

---

## LEGAL & COMPLIANCE

- Dual consent gate mandatory (two-party consent jurisdictions)
- Face embeddings + voice embeddings = biometric data under BIPA/GDPR
- Edge-first mode: raw biometric video never leaves device (reduced legal exposure)
- Accuracy disclaimer required: "approximately 75% accuracy on realistic benchmarks, not a replacement for human judgment"
- Storage lifecycle enforced at infrastructure level (MinIO ILM), not application level

---

## CRITICAL INVARIANTS

1. **Never feed lossy audio to ML models.** FLAC or PCM only.
2. **Never begin recording without dual consent tokens.**
3. **Never log PII.** Opaque IDs only.
4. **Never hardcode secrets.**
5. **Never show raw scores to users.** Calibrate + confidence + qualitative labels.
6. **Never use "lie detector"** anywhere.
7. **Never store raw face images.** 512-D embeddings only, encrypted.
8. **Never skip storage lifecycle.**
9. **Never transmit raw video in edge-first mode.**
10. **Never claim >80% accuracy** without peer-reviewed validation on ≥500 held-out samples.
11. **Always use late fusion.** Score-level combination, not feature concatenation.
12. **Always include baseline quality with every score.**

---

## IMPLEMENTATION STATUS

Snapshot as of 2026-07-07. The architecture sections above remain authoritative; this section is the truth about what code actually exists today.

### Day 1 — Compression Pipeline (complete)

The entry-point pipeline every uploaded or recorded video flows through. Lives under `backend/workers/app/compression/`.

| Component | Path | Status |
|---|---|---|
| Schemas | `backend/shared/schemas/media.py` | shipped — `CompressionMode`, `CompressionConfig`, `CompressionResult` |
| Audio extraction | `compression/audio_extractor.py` | shipped — FLAC 48k/mono (ML) + Opus 32k/mono (playback) |
| ROI video encoding | `compression/roi_encoder.py` | shipped — simplified v1 (single-CRF) |
| Feature extraction | `compression/feature_extractor.py` | shipped — 478-pt landmarks **ALTM protobuf telemetry** (`_landmarks.pb`, keyframe/delta + zlib chunks; see proto/landmarks.proto) + MFCC/Chroma/Mel/Contrast/Tonnetz `.npz` |
| Pipeline orchestrator | `compression/pipeline.py` | shipped — all four modes; graceful per-stage error handling |
| Model file management | `compression/models.py` | shipped — lazy download + cache for MediaPipe Tasks API models |
| CLI smoke test | `scripts/test_compression.py` | shipped |
| Build config | `pyproject.toml`, `Makefile` | shipped |

Notable deviations from the original Day 1 spec:

1. **MediaPipe Tasks API**, not the legacy Solutions API. mediapipe ≥ 0.10.30 — the only versions with Python 3.13 wheels — removed `mp.solutions` entirely. Code uses `vision.FaceDetector` and `vision.FaceLandmarker`; model `.tflite` / `.task` files are downloaded lazily on first use and cached under `%LOCALAPPDATA%/project-alice/models` (or `$ALICE_MODEL_CACHE` if set).
2. **ROI v1 is single-CRF** (CRF 22 if a face is detected anywhere, CRF 26 + warning otherwise), not a per-pixel QP map. The spec authorized this Day 1 trade-off; the per-frame bbox track is already produced by the scanner and will feed v2's x265 zones for true ROI encoding.
3. **CRF 26, not 32, for the no-face fallback.** Hedges toward higher quality when a no-face result might be a detector miss rather than a true negative.

### Phase 1-Bridge — Mobile-Critical Fixes (complete)

Shipped in commit `1fef136`. Driven by a mobile high-usage analysis (see `docs/superpowers/specs/2026-04-27-compression-pipeline-and-psycholinguistic-design.md`).

| Fix | Path | Status |
|---|---|---|
| Streaming JSONL landmarks (P1-S6) | `compression/feature_extractor.py` | shipped (superseded by ALTM protobuf in Session 4) — peak RAM O(flush_interval), not O(total_frames) |
| Platform-aware model cache (P1-S7) | `compression/models.py` | shipped — `ALICE_MODEL_CACHE` > Windows `%LOCALAPPDATA%` > Android `$XDG_DATA_HOME` > XDG home |
| Mid-session tier switching (P1-S8) | `compression/pipeline.py` | shipped — `update_bandwidth()` + `on_mode_change` callback + `mode_transitions` audit |

### Phase 2 — Psycholinguistic Analyzer (complete, Day 1)

The first analysis vector. Lives under `backend/ml-inference/` (hyphenated service root, imported via `sys.path`, NOT a dotted `ml_inference` path).

| Component | Path | Status |
|---|---|---|
| Schemas | `backend/shared/schemas/psycholinguistic.py` | shipped — `PsycholinguisticDimension`, `PsycholinguisticScore` (fields are `*_score`-suffixed) |
| Analyzer | `backend/ml-inference/app/pipelines/psycholinguistic/analyzer.py` | shipped — all 8 dimension scorers + equal-weighted composite, lazy spaCy/VADER load |
| Tooling | spaCy `en_core_web_md` + NRCLex + vaderSentiment | hedging is a Day-1 word list (59% FP); replace with BERT in Phase 3 |
| Tests | `tests/psycholinguistic/` | 29 tests (incl. CLI smoke); full suite 44 passing |
| CLI | `scripts/test_psycholinguistic.py`, `scripts/test_compress_and_analyze.py` | shipped |

### Session 3 — WhisperX Transcription Vector (complete)

Bridges the compression FLAC output to the psycholinguistic analyzer. Design spec:
`docs/superpowers/specs/2026-06-25-whisperx-transcription-vector-design.md`.

| Component | Path | Status |
|---|---|---|
| Schemas | `backend/shared/schemas/transcription.py` | shipped — `TranscriptSegment`, `Transcript` (with billable `audio_duration_seconds`, reserved `speaker` field), `TranscriptionConfig` |
| Backend protocol + fake | `backend/ml-inference/app/pipelines/transcription/backends.py` | shipped — `TranscriptionBackend` Protocol + deterministic `FakeTranscriptionBackend` (default test suite needs no torch / no downloads) |
| Real backend | same file, `WhisperXBackend` | shipped — lazy-imports whisperx inside `transcribe()` only; alignment ON, diarization OFF; `device="auto"` → cuda/float16 else cpu/int8 |
| Facade | `backend/ml-inference/app/pipelines/transcription/transcriber.py` | shipped — FLAC/WAV-only gate (invariant #1), opaque-facts-only logging (invariant #3) |
| Tests | `tests/transcription/` | shipped — schema/fake/facade/bridge/CLI suites; real-model test gated `@pytest.mark.slow` (deselected by default via `addopts`) |
| CLIs | `scripts/test_transcribe.py`; `scripts/test_compress_and_analyze.py` | shipped — the latter is now a LIVE video → FLAC → transcript → psycholinguistic score path (`--fake` runs it offline) |

Key decisions: one WhisperX segment == one statement (the analyzer re-joins all
statements before parsing, so segmentation never costs linguistic context);
`whisperx>=3.1` is an optional extra (`pip install -e ".[transcription]"`), NOT
installed by default — Windows + Py3.13 torch installs are rough; WSL/Linux is the
documented fallback runner for the real backend. Full suite: 74 passed, 1 deselected.

### Session 5 — ScoreEvent Streaming Contract + Replayer (complete)

Closes gap #2 at the contract layer (scope A: no live infra). Design spec:
`docs/superpowers/specs/2026-07-08-scoreevent-streaming-design.md`.

| Component | Path | Status |
|---|---|---|
| Schemas | `backend/shared/schemas/score_event.py` | shipped — `ScoreEventKind`, `StreamScorerConfig`, `ScoreEvent` (cumulative + recent per event), `validate_event_stream` |
| Windowed scorer | `backend/ml-inference/app/pipelines/streaming/windowed_scorer.py` | shipped — `stream_scores()` strictly causal sync generator; ticks every `tick_seconds`, skips silent ticks, `recent=None` when sparse |
| Replayer | `backend/ml-inference/app/pipelines/streaming/replayer.py` | shipped — `ScoreReplayer` wall-clock pacing (pace 0 = instant), injectable sleep, clean cancellation via generator close |
| CLI | `scripts/replay_scores.py` | shipped — `--transcript json` or `--video` (`--fake` offline), `--pace/--tick/--recent-window` |
| Tests | `tests/streaming/` | shipped — schema contract, causality gate (future-mutation invariance), **batch-convergence acceptance gate** (`test_convergence.py`: FINAL == batch field-for-field), pacing/cancellation, CLI smoke |

Key decisions (locked in the spec — do not re-litigate): sync generator core
with the async FastAPI/WebSocket shell as a documented future session
(decision #4: per-tick work is CPU-bound; the shell drives the generator via
`asyncio.to_thread(next, gen)`); each interim carries cumulative + recent;
exactly one FINAL, equal to batch; gap #4 (hedging/certainty double-count)
deliberately deferred until after convergence landed, so the gate's baseline
stayed stable.

### Session 7 — Live Service Async Shell (complete)

The real-time surface pre-designed in Session 5 (decision #4). Design spec:
`docs/superpowers/specs/2026-07-19-live-service-async-shell-design.md`.

| Component | Path | Status |
|---|---|---|
| Config | `backend/ml-inference/app/service/config.py` | shipped — frozen `LiveServiceConfig` |
| Sessions + reaper | `app/service/sessions.py` | shipped — detached lifecycle (CREATED/RUNNING/FINISHED/CANCELLED/FAILED), TTL reaper |
| Publisher (Kafka seam) | `app/service/publisher.py` | shipped — seq, ring buffer, fan-out, slow-client drop (4408); v2 swaps in a bus publisher |
| Runner | `app/service/runner.py` | shipped — one worker thread/session, sliced-sleep cancellation |
| REST + WS | `app/service/app.py` | shipped — POST/GET/DELETE /sessions, /healthz, WS /sessions/{id}/events?last_seq= (4404/4408) |
| Launch + demo | `scripts/run_live_service.py`, `scripts/live_client.py`, `make live` | shipped |

Wire contract: data frames `{session_id, seq, event}`; exactly one terminal
frame `{session_id, state, reason}`. Auth: none in v1 (localhost bind; JWT is
the api-gateway session's job). In update to "Known gaps" item 2: the async
shell is now SHIPPED; remaining for full live: incremental transcription +
platform media ingest.

### Known gaps & next-session priorities (review of 2026-07-03)

Ordered by business impact; items 1-2 gate the mobile/live story.

1. ~~Landmark telemetry is ~170× over budget~~ **RESOLVED (Session 4):** ALTM
   protobuf format (proto/landmarks.proto) — 12-bit quantization (0.13 px @1080p,
   below detector jitter), keyframe/delta, zlib chunks. Measured ~1,042 KB/min at
   30 fps synthetic (≈0.14 Mbps; real demo clip 1196 KB/min) vs ~12 MB/min
   JSONL. Gate ≤1.2 MB/min is bandwidth-derived (FLAC 0.33 Mbps + landmarks 0.16
   Mbps < 0.5 Mbps on a <1 Mbps EDGE_MINIMAL uplink) and enforced by
   tests/telemetry/test_budget.py. The ~70 KB/min figure now applies to the future
   AU-activation payload.
2. ~~No real-time path exists~~ **RESOLVED (Session 5 contract, Session 7
   shell):** `ScoreEvent` schema + causal windowed scorer + replayer shipped
   (Session 5); batch and stream converge on one contract, enforced by
   tests/streaming/test_convergence.py (FINAL == batch, field-for-field). The
   async FastAPI/WebSocket shell around the sync generator
   (`asyncio.to_thread(next, gen)`, per-session cancellation) shipped in
   Session 7 — `app/service/{config,sessions,publisher,runner,app}.py` +
   `scripts/run_live_service.py` / `scripts/live_client.py`. Remaining for a
   true live surface: incremental transcription and platform media ingest.
   The windowed events already power the report's scrubbable score timeline
   directly.
3. **Disfluency dimension will degrade on real transcripts.** Whisper-family models
   suppress filled pauses (um/uh), so the disfluency scorer reads near-zero once real
   WhisperX output replaces the fake backend. Validate on real audio; likely move
   disfluency detection to the vocal-tonality vector (audio-side pause analysis).
4. ~~Hedging and certainty scorers double-count tentative markers~~ **RESOLVED
   (Session 6):** `_TENTATIVE_MARKERS` deleted; dimension 8 is over-certainty /
   emphatic assertion only; tentative language is owned solely by hedging. A
   list-disjointness regression test pins it.
5. **WhatsApp has no live-call API** (E2E-encrypted, no bot join, Business API is
   messaging-only). Market it as upload-only ("analyze your recorded calls" — the
   AudioExtractor already ingests .mp4/.m4a/.ogg/.opus); live integration claims
   apply to Zoom / Teams / Meet / Webex / Slack / LiveKit only. Zoom/Teams
   per-participant audio streams give perfect speaker attribution without pyannote —
   a shortcut to speaker-attributed analysis before the diarization session.
6. **Billing hooks exist; metering does not.** `audio_duration_seconds` (content
   sold) and `mode_transitions` (compute cost) are recorded per job, but there is no
   MinIO, no ILM, no tenant model, no usage pipeline. Storage lifecycle must be
   enforced at infrastructure level (invariant #8) when built. Frame cold storage as
   baseline-continuity (per-contact history is the moat), not archive fees.
7. Smaller: `TranscriptionConfig.vad_chunk_seconds` is reserved/not wired; ROI v2
   x265 zones still pending (bbox track already produced); prototype auth file has
   a hardcoded secret — never ship it.
8. **Psycholinguistic vector is English-only — deliberate for v1.** Tooling
   (spaCy `en_core_web_md`, NRCLex, VADER, the hedging/certainty/filler word
   lists) and the underlying research base
   (Newman et al. pronoun effects, Pérez-Rosas corpora) are English. Pro-drop
   languages (es/ja/tr/it) hollow out the pronoun-shift dimension; fillers,
   negation norms, and clause-depth baselines differ per language; cross-language
   cue transfer is weak, so each language needs its own validation — not just a
   tokenizer swap. **Never translate-then-score:** MT destroys the surface
   stylometry being measured (normalizes hedges, deletes disfluencies, inserts
   pronouns the speaker never uttered, substitutes the translator's syntax).
   Near term: add a transcript-language gate so non-`"en"` transcripts never flow
   silently through the English pipeline (contract note: `ScoreEvent.cumulative`
   is required, so gating needs a small schema decision). **Near-term gate SHIPPED
   (Session 6):** `UnsupportedLanguageError` raised before any scoring at
   `analyze(language=...)` and `stream_scores` entry (zero events; ScoreEvent
   schema untouched); CLIs exit 1 with the language named. Hard-fail is the v1
   behavior; the late-fusion session adds graceful per-vector degradation.
   Expansion path (per-language packs) unchanged below: a resource pack
   (spaCy model + emotion/hedging lexicons + filler list), per-language
   dimension masks and ensemble weights, and a validation corpus. What
   already travels: WhisperX transcription (~99 languages;
   `Transcript.language` is recorded), contradiction detection once
   embeddings/NLI go multilingual (mDeBERTa-v3/XNLI), acoustic features, facial
   AUs, and the per-subject baseline — deviation-from-self partially factors out
   language and culture by construction.

### Pending (not yet implemented)

The remaining four analysis vectors and surrounding infrastructure remain to be built:

- Facial Action Unit detection (custom ResNet-18 + ME-GraphAU)
- Vocal tonality flux + emotion2vec+ + Praat acoustics
- Statement contradiction (DeBERTa-v3 NLI + pgvector — WhisperX transcription now shipped; diarization still pending)
- Subject identification (LR-ASD, EdgeFace, pyannote)
- Late-fusion ensemble (XGBoost + Platt calibration + SHAP)
- API gateway, auth, Triton serving, Celery workers, Kafka, Postgres
- Mobile client (Kotlin Multiplatform)
- Platform connectors (Zoom / Teams / Meet / Webex / Slack / LiveKit; WhatsApp = upload-only, see gap #5)
- Storage lifecycle ILM, consent gate, retention purge, usage metering
- Testing infrastructure (pytest at 80% / 90% coverage, integration suite, lint config)

### Operational notes

- **Runtime venv** lives at `.venv/` (Python 3.13.9 installed locally; not committed).
- **System binary requirement**: `ffmpeg` must be on PATH (on Windows: `winget install Gyan.FFmpeg`; on macOS: `brew install ffmpeg`; on Linux: distro package).
- **Verified end-to-end** on `trial_lie_001.mp4` from the Real-Life Trial Deception Detection 2016 dataset. RAW mode: 4.5 s wall-clock, 100% face-detection rate, ROI ratio 0.58. EDGE_FULL mode: 6.4 s wall-clock, 98% face-detection rate, landmark telemetry ~1.05 MB/min at 30 fps (ALTM protobuf; Session 4).
