<p align="center">
  <img src=".github/assets/alice-banner.png" alt="ALICE Banner" width="100%" />
</p>

<h1 align="center">Project ALICE</h1>
<h3 align="center">Advanced Logic Integrity & Consistency Examiner</h3>

<p align="center">
  <strong>Multimodal behavioral anomaly detection for video calls and uploaded recordings.</strong><br>
  Five-vector analysis: facial action units · psycholinguistic profiling · vocal tonality flux · statement contradiction · subject identification
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Kotlin-2.0+-7F52FF?style=flat-square&logo=kotlin&logoColor=white" />
  <img src="https://img.shields.io/badge/Platform-Android%20(iOS%20planned)-34A853?style=flat-square&logo=android&logoColor=white" />
  <img src="https://img.shields.io/badge/Fusion-Late%20(Score--Level)-EE4C2C?style=flat-square" />
  <img src="https://img.shields.io/badge/License-Proprietary-red?style=flat-square" />
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#research-foundation">Research</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#tech-stack">Tech Stack</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#competitive-landscape">Competition</a> ·
  <a href="#legal--compliance">Legal</a>
</p>

---

## Overview

ALICE is a mobile-first platform that analyzes video calls (live or uploaded) for behavioral indicators of deception using five independent analysis vectors fused through a late-fusion ensemble. It works by establishing a per-contact behavioral baseline over multiple interactions, then detecting statistically significant deviations across all vectors simultaneously.

**ALICE is not a lie detector.** It is a behavioral anomaly detection system. Scores represent deviation from an individual's established baseline — not ground truth. The best published accuracy on realistic multimodal deception datasets is approximately 75% F1. ALICE is transparent about this ceiling.

### Key Capabilities

- **Five-vector multimodal analysis** — Facial Action Units, psycholinguistic profiling, vocal tonality flux, statement contradiction, and subject identification fused via late fusion
- **Per-contact personalization** — Models adapt to each individual's behavioral patterns over time, the single most impactful accuracy improvement over population-level baselines
- **Live video call analysis** — Works alongside Zoom, Teams, Google Meet, Webex, and Slack via platform API integrations, or natively via LiveKit
- **Subject identification** — Active speaker detection (which face is talking), speaker diarization (who said what), and face re-identification across sessions for baseline continuity
- **Edge-first privacy** — On-device ML extracts facial landmarks, Action Units, and face embeddings locally. Raw face images never leave the device. Only 512-D embeddings and telemetry are transmitted
- **Three-tier compression** — Lossless audio for ML fidelity, ROI-encoded video preserving facial detail, edge-mode telemetry at ~70KB/min

---

## Implementation Status

> ⚠️ **Early development.** The full architecture below describes the target system. Today, only the **Day 1 compression pipeline** is implemented and verified end-to-end. The five analysis vectors, API gateway, mobile client, and platform connectors are not yet built. See [`CLAUDE.md`](CLAUDE.md#implementation-status) for the authoritative status snapshot.

**Working today:**
- Three-tier compression pipeline (RAW / ROI_ENCODED / EDGE_FULL / EDGE_MINIMAL)
- FLAC + Opus audio extraction (FLAC for ML, Opus for playback)
- MediaPipe Tasks API face detection + 478-point face landmarker
- librosa-based audio features (MFCC, Chroma, Mel, Spectral Contrast, Tonnetz)
- CLI smoke test (`make test-compress VIDEO=...`)
- Verified on Real-Life Trial Deception Detection 2016 dataset

**Not yet built:** Action Unit detection, psycholinguistic analysis, vocal tonality flux, statement contradiction, subject identification, late-fusion ensemble, mobile client, platform connectors, storage lifecycle, API gateway.

---

## Research Foundation

ALICE's architecture is informed by three primary research lineages. We do not copy their architectures — we use their findings to select modern, higher-accuracy components.

### Pérez-Rosas et al. (EMNLP 2015 & ICMI 2015)

**Dataset:** 121 real courtroom trial videos (defendants, witnesses, exonerees). **Result:** 75-82% accuracy with SVM/Decision Tree fusion of verbal and nonverbal features.

**Key findings adopted by ALICE:**
- Facial displays are the highest-contributing modality (removing them dropped accuracy from 77% to 64%)
- The most predictive nonverbal features: side turns, up gaze, blinking patterns, smiling, lip corner position, frowning, eyebrow raising
- Truth-tellers use more Family/Home/Humans words; deceivers use more Anxiety/Anger/negative emotion words
- Deceivers in high-stakes settings (trials) INCREASE eye contact — contradicting the popular belief that liars look away

**What ALICE does differently:** They used manual MUMIN gesture annotation. ALICE automates this via FACS Action Unit detection (custom ResNet-18 on-device, ME-GraphAU on server). They had no per-subject baseline. ALICE establishes baselines over the first 2-3 interactions.

### Li & Abouelenien (IEEE ISPDS 2024)

**Dataset:** 104 subjects, lab-elicited deception across two topics. **Result:** ~80% accuracy for linguistic features, ~65% for physiological.

**Key findings adopted by ALICE:**
- Late fusion (score-level) outperforms early fusion (feature concatenation) — ALICE uses late fusion exclusively
- Linguistic features dramatically outperform physiological features as a single modality
- Linguistic deception cues are domain-specific while physiological patterns generalize better across topics — ALICE addresses this with per-contact baselines
- Word2vec embeddings + CNN architecture worked for linguistic encoding

**What ALICE does differently:** They used physiological sensors (blood volume pulse, skin conductance). ALICE uses contactless analysis only (facial AU + audio + text). Their bimodal CNN used only two channels; ALICE fuses five.

### Current State-of-the-Art (2024-2026)

- **DOLOS dataset** (1,675 clips): best published F1 is 75.5% (Wav2Vec2 + BERT + BiLSTM, Vallabhaneni 2025)
- **SVC 2025 Challenge**: first standardized multimodal deception detection benchmark, targeting cross-domain generalization
- **Expert consensus (Luke et al. 2023):** >80% of experts agree gaze aversion is NOT a reliable deception indicator
- **Foundation models:** LLMs achieve SOTA on textual deception detection, but multimodal LMMs struggle to leverage multimodal cues effectively
- **RoBERTa + XGBoost** outperformed GPT-4 for emotion-based deception features (LieXBerta, Scientific Reports 2025)

---

## How It Works

```
┌───────────────────────────────────────────────────────────────────────┐
│                      VIDEO / AUDIO INPUT                              │
│           (Live call stream OR uploaded recording)                     │
└─────┬──────────┬──────────┬──────────┬──────────┬─────────────────────┘
      │          │          │          │          │
      ▼          ▼          ▼          ▼          ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────────────────┐
│  FACIAL  ││ PSYCHO-  ││ VOCAL    ││STATEMENT ││ SUBJECT              │
│  ACTION  ││LINGUISTIC││ TONALITY ││CONTRADIC-││ IDENTIFICATION       │
│  UNITS   ││ ANALYSIS ││  FLUX    ││  TION    ││                      │
│          ││          ││          ││          ││ LR-ASD (who's talking)│
│ MediaPipe││ spaCy    ││emotion2v+││ WhisperX ││ EdgeFace (face re-ID)│
│ +Custom  ││ Empath   ││ Praat    ││ DeBERTa  ││ pyannote (diarize)   │
│ ResNet-18││ NRCLex   ││ librosa  ││ pgvector ││                      │
│          ││ BERT     ││          ││          ││                      │
│ Weight:  ││ Weight:  ││ Weight:  ││ Weight:  ││ Weight:              │
│  0.30    ││  0.30    ││  0.20    ││  0.15    ││  0.05 (gaze only)    │
└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└──────────┬───────────┘
     │           │           │           │                  │
     ▼           ▼           ▼           ▼                  ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    LATE FUSION ENSEMBLE                                │
│                                                                       │
│  Each vector produces independent 0-100 score                         │
│  XGBoost combines scores (NOT features) + Platt calibration           │
│  Per-contact weights learned after baseline established               │
│  SHAP explainability per vector                                       │
│  Confidence: Low (1 session) / Medium (2-3) / High (4+)              │
└───────────────────────────────────────────────────────────────────────┘
```

### Vector Details

**1. Facial Action Units** — MediaPipe Face Mesh extracts 478 landmarks on-device. A custom ResNet-18 (INT8 TFLite, ~3-4MB) classifies FACS Action Units. Key deception signals: AU12 without AU6 (non-Duchenne/fake smile), AU1+AU4 (cognitive load brow activity), AU14 (contempt dimpler), and bilateral asymmetry of any AU (indicating voluntary rather than spontaneous expression). Server-side ME-GraphAU provides higher-accuracy analysis for batch processing.

**2. Psycholinguistic Analysis** — Eight dimensions extracted from transcribed speech: pronoun patterns (I/me/my frequency drops during deception), hedging frequency (via fine-tuned BERT, not regex — regex yields 59% false positive rate), cognitive complexity (subordinate clause depth), emotional word distribution (Empath 200+ categories + NRCLex 8 emotions), speech disfluencies (um/uh/er count), negation frequency, detail specificity (named entity density), and certainty vs. tentative language.

**3. Vocal Tonality Flux** — Dual pipeline: emotion2vec+ large (768-dim embeddings per 1-second window) for deep speech emotion representation, plus Praat acoustic features (F0 fundamental frequency, jitter, shimmer, HNR) for interpretable stress indicators. The flux score measures rate of change relative to the contact's personal baseline.

**4. Statement Contradiction** — Speaker-attributed transcription via WhisperX (Whisper + pyannote diarization + forced alignment). Each statement embedded via sentence-transformers, stored in pgvector. New statements compared against top-5 most similar historical statements using DeBERTa-v3 NLI classification.

**5. Subject Identification** — Three-stage on-device pipeline (~13MB total): face detection (MediaPipe), active speaker detection (LR-ASD, 0.84M params, 94.5% mAP — determines which face is speaking), and face re-identification (EdgeFace-XS, 512-D embeddings, 99.73% LFW accuracy). Speaker diarization (pyannote) runs server-side for audio-only scenarios. When Zoom/Teams provide per-participant audio streams, model-based diarization is bypassed entirely.

---

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         MOBILE CLIENT                            │
│   Kotlin Multiplatform · Compose UI · On-Device ML (~25MB)      │
│                                                                  │
│   MediaPipe → Custom AU Model → EdgeFace-XS → LR-ASD           │
│   + librosa features + FLAC audio extraction                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │ protobuf + FLAC audio
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        API GATEWAY                               │
│            FastAPI · JWT Auth · Rate Limiting · WebSocket        │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────────┘
       ▼          ▼          ▼          ▼          ▼
  ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
  │Keycloak │ │Triton  │ │WhisperX│ │DeBERTa │ │ Celery   │
  │ Auth    │ │AU +    │ │Speaker │ │NLI +   │ │ Workers  │
  │SSO/MFA  │ │Tonal.  │ │Diarize │ │Psycho- │ │Compress +│
  │JWT      │ │emotion │ │Transcr.│ │linguis.│ │Retrain   │
  └─────────┘ └────────┘ └────────┘ └────────┘ └──────────┘
       │          │          │          │          │
       └──────────┴──────────┴──────────┴──────────┘
                           │
┌──────────────────────────────────────────────────────────────────┐
│  PostgreSQL+pgvector · MinIO (S3) · Redis · Kafka · Vault       │
└──────────────────────────────────────────────────────────────────┘
```

### Compression Tiers

| Tier | Method | Bandwidth | ML Fidelity |
|------|--------|-----------|-------------|
| **1: Lossless Audio** | FLAC 48kHz mono | ~2.5 MB/min | Perfect — all models receive mathematically lossless audio |
| **2: ROI Video** | Face QP -12 / Background CRF 28-32 | ~5-10 MB/min | Near-lossless face (SSIM >0.98), lossy background |
| **3: Edge-First** | On-device AU + landmarks + embeddings | ~0.07 MB/min (+ 2.5 FLAC) | High — raw video never leaves device |

Auto-selection: ≥10Mbps→RAW, 5-10→ROI, 1-5→EDGE_FULL, <1→EDGE_MINIMAL.

### Platform Integration

ALICE works alongside existing video call platforms. Users don't switch apps.

| Platform | Real-Time Path | Post-Call Path | Per-Participant Audio |
|----------|---------------|----------------|----------------------|
| Zoom | Meeting SDK bot | recording.completed webhook | Yes (bypass diarization) |
| Teams | Bot Framework | Graph API callRecording | Yes |
| Google Meet | Add-ons SDK | Drive API recording | Limited |
| Webex | Bot media | Meetings API download | Yes |
| Slack | Events API | N/A (audio-only) | No |
| LiveKit | Direct WebRTC | Server recording | Yes |

### Storage Lifecycle

| Tier | Retention | Contents | Cost/User/Mo |
|------|-----------|----------|-------------|
| 🔥 Hot | 0-30 days | AV1 video + FLAC + AU data + transcript + scores | ~$1.15 |
| ♨️ Warm | 30-90 days | FLAC + AU data + transcript + scores (video deleted) | ~$0.22 |
| ❄️ Cold | 90-365 days | Transcript + scores + embeddings only | ~$0.04 |
| 🗑️ Purge | 365+ days | All PII deleted | $0.00 |

---

## Tech Stack

### On-Device ML Pipeline (~25MB)

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Face Detection + Alignment | MediaPipe Face Mesh | ~5MB | 478 landmarks at 30fps |
| Action Unit Detection | Custom ResNet-18 (INT8 TFLite) | ~3-4MB | 12+ AUs with intensity |
| Active Speaker Detection | LR-ASD | ~4MB | Which face is talking (94.5% mAP) |
| Face Re-ID | EdgeFace-XS | ~7MB | 512-D embeddings (99.73% LFW) |
| Audio Features | librosa (code only) | 0 | MFCC, chroma, mel, spectral |

### Server ML Pipeline

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| AU Detection (accurate) | ME-GraphAU architecture | ~50MB | 41 AU categories via GNN |
| Emotion Embeddings | emotion2vec+ large | ~300MB | Speech emotion representation |
| Acoustic Features | parselmouth (Praat) | 0 | F0, jitter, shimmer, HNR |
| Transcription + Diarization | WhisperX | ~1.5GB | Speaker-attributed word timestamps |
| Statement Embedding | all-mpnet-base-v2 | ~420MB | 768-D statement vectors |
| Contradiction NLI | DeBERTa-v3-large | ~400MB | Entailment/contradiction classification |
| Psycholinguistic | spaCy + Empath + NRCLex + VADER + custom BERT | ~500MB | 8 linguistic dimensions |
| Ensemble | XGBoost + Platt scaling | ~2MB/contact | Late fusion + calibration |

### Infrastructure

| Service | Technology |
|---------|-----------|
| API Gateway | FastAPI (Python 3.12+) |
| Auth | Keycloak (OIDC/SAML, MFA) |
| ML Serving | Triton Inference Server |
| Task Queue | Celery + Redis |
| Event Stream | Apache Kafka |
| Database | PostgreSQL 16 + pgvector + TimescaleDB |
| Object Storage | MinIO (S3-compatible, tiered lifecycle) |
| Secrets | HashiCorp Vault |
| Orchestration | Kubernetes (EKS/GKE) |

---

## Getting Started

### Prerequisites (Day 1)

- **Python 3.12+** (verified on 3.13)
- **FFmpeg** on PATH — Windows: `winget install Gyan.FFmpeg` · macOS: `brew install ffmpeg` · Linux: distro package
- **Git** for cloning

Full-stack prerequisites (JDK 17+, Docker, Android Studio, etc.) only become necessary once the additional services land — they are not required to run the compression pipeline today.

### Quick Start — Compression Pipeline

```bash
git clone https://github.com/AEGIS-Medical/Project-ALICE.git
cd Project-ALICE

# Create a venv (recommended) and install
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
make install                    # pip install -e ".[dev]"

# Run the pipeline against any video
make test-compress VIDEO=path/to/your/video.mp4
make test-compress VIDEO=path/to/your/video.mp4 MODE=edge_full
```

On first run the script downloads two MediaPipe model files (~4 MB total) to `%LOCALAPPDATA%/project-alice/models` (Windows) or `~/.cache/project-alice/models` (POSIX). Override with the `ALICE_MODEL_CACHE` env var.

Valid `MODE` values: `raw`, `roi`, `edge_full`, `edge_minimal` (default `raw`).

### Future Quick Start (not yet wired)

Once the API gateway, ML services, and orchestration land, the full-stack start will look like:

```bash
cp .env.example .env       # Edit with your config
make up                    # Start all services
make setup-models          # Download all ML models (~3 GB)
make analyze VIDEO=...     # Run full five-vector analysis
```

These targets are not present in the current Makefile.

### Development Commands

Currently shipped Makefile targets:

```bash
make install                                   # pip install -e ".[dev]"
make test-compress VIDEO=path/to/video.mp4     # smoke-test the compression pipeline
```

Planned (not yet shipped): `make up`, `make down`, `make test-backend`, `make lint`, `make build-android`, `make test-e2e`, `make train-au`, `make evaluate-all`, `make deploy-staging`.

> **For AI-assisted development:** Read `CLAUDE.md` before writing any code. It defines agent roles, component selections, research rationale, critical invariants, and the current implementation status snapshot.

---

## Competitive Landscape

| System | Modalities | Baseline | Real-Time | Call Integration | Validated Accuracy |
|--------|-----------|----------|-----------|------------------|--------------------|
| CyberQ.ai | Text only (LLM prompt) | None | No | None | 92% (unvalidated, no paper) |
| Converus EyeDetect | Eye tracking | None | Semi | None | 86-90% (proprietary validation) |
| Clearspeed | Voice only | Unclear | Yes | None | Undisclosed |
| **ALICE** | **5-vector fusion** | **Per-contact** | **Yes** | **Zoom/Teams/Meet** | **Targeting ~75% F1 (honest)** |

ALICE is the only system that combines genuine multimodal fusion, per-subject calibration, live video call integration, and edge-first privacy architecture.

---

## Legal & Compliance

| Regulation | ALICE Compliance |
|-----------|-----------------|
| Two-Party Consent | Dual consent gate — recording blocked until both JWT consent tokens verified |
| BIPA (Illinois) | Face embeddings = biometric data. Consent modal, published retention, no sale |
| GDPR | Edge-first minimization, 365-day auto-purge, export endpoint, explicit consent |
| Recording Disclosure | Bot announces presence; native calls show recording indicator |
| Accuracy Claims | "~75% on realistic benchmarks, not a replacement for human judgment" |

---

## Acknowledgments

- **[Pérez-Rosas et al.](https://github.com/MichiganNLP/deceptiondetection)** — Multimodal deception detection methodology using real-life trial data (EMNLP 2015, ICMI 2015). Feature engineering and evaluation methodology referenced.
- **[Li & Abouelenien](https://arxiv.org/abs/2311.10944)** — Bimodal CNN approach establishing late fusion superiority for deception detection (IEEE ISPDS 2024).
- **[jsugg/ser](https://github.com/jsugg/ser)** — Speech Emotion Recognition package (MIT). Feature backend protocols, emotion2vec integration, faster-whisper transcription adapted.
- **[MediaPipe](https://github.com/google/mediapipe)**, **[emotion2vec](https://github.com/ddlBoJack/emotion2vec)**, **[LiveKit](https://github.com/livekit)**, **[pyannote](https://github.com/pyannote/pyannote-audio)**, **[WhisperX](https://github.com/m-bain/whisperX)**.

---

## License

Proprietary. All rights reserved. See `LICENSE` for terms.

<p align="center"><sub>ALICE: Because consistency reveals character.</sub></p>
