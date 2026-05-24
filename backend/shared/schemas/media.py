"""Media compression schemas for Project ALICE.

Defines the typed contracts produced and consumed by the ingestion / compression
pipeline. Every video processed by ALICE -- whether uploaded or captured live
from a call platform -- passes through one of the four tiers below before any
downstream analysis vector (AU, psycholinguistic, tonality, contradiction,
gaze) sees it.

Tier reference (see CLAUDE.md "COMPRESSION ARCHITECTURE" and
docs/architecture/compression-tiers.md):

    RAW            Lossless input retained for batch / >=10 Mbps uplink.
    ROI_ENCODED    Face region near-lossless (QP -12), background CRF 28-32.
                   60-70% size reduction with SSIM > 0.98 over the face.
    EDGE_FULL      On-device extraction: landmarks + AU activations + face
                   embeddings + audio features. ~70 KB/min telemetry.
    EDGE_MINIMAL   Reduced-cadence edge extraction for < 1 Mbps uplinks.

Critical invariants enforced by these schemas (CLAUDE.md "CRITICAL INVARIANTS"):
    - Audio is ALWAYS Tier 1 lossless (FLAC 48 kHz mono) regardless of video
      tier. Lossy audio must NEVER reach an ML model. (Invariant #1.)
    - In edge-first modes (EDGE_FULL / EDGE_MINIMAL) raw video MUST NOT leave
      the device; only the landmarks/features artifacts may be transmitted.
      (Invariant #9.)
    - All filesystem locations are pathlib.Path, never bare strings, so callers
      cannot accidentally mix separators across Win/POSIX hosts.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CompressionMode(str, Enum):
    """Compression tier selected for a given media payload.

    The active mode is normally chosen adaptively from the available uplink
    bandwidth via ``CompressionConfig.select_mode``. The string values are the
    on-the-wire identifiers used in protobuf telemetry and DB rows; do not
    rename them without a migration.
    """

    RAW = "raw"
    ROI_ENCODED = "roi_encoded"
    EDGE_FULL = "edge_full"
    EDGE_MINIMAL = "edge_minimal"


class CompressionConfig(BaseModel):
    """Tunable parameters for the compression pipeline.

    Defaults are sourced from CLAUDE.md ("COMPRESSION ARCHITECTURE") and chosen
    to preserve signal quality for downstream ML while meeting bandwidth and
    storage budgets. Instances are frozen so a config can be safely shared
    across worker threads without locking.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
    )

    # ---- Tier 1: lossless audio (CLAUDE.md invariant #1) -------------------
    # These defaults are non-negotiable for ML inputs. Only adjust if you are
    # generating a *separate* playback artifact (e.g. Opus) downstream.
    audio_sample_rate_hz: int = Field(
        default=48_000,
        ge=8_000,
        le=192_000,
        description="FLAC sample rate. 48 kHz is the project standard.",
    )
    audio_channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Mono by default; downstream ML expects single-channel input.",
    )
    audio_bit_depth: int = Field(
        default=16,
        ge=16,
        le=32,
        description="FLAC bit depth. 16-bit is sufficient for speech analysis.",
    )
    flac_compression_level: int = Field(
        default=5,
        ge=0,
        le=12,
        description="FLAC effort level (0=fastest, 12=smallest). 5 is the sweet spot.",
    )

    # ---- Tier 2: ROI video encoding ---------------------------------------
    face_roi_qp_offset: int = Field(
        default=-12,
        ge=-30,
        le=0,
        description=(
            "QP offset applied inside the face ROI relative to the background "
            "CRF. -12 keeps SSIM > 0.98 over the face per CLAUDE.md."
        ),
    )
    background_crf: int = Field(
        default=30,
        ge=18,
        le=40,
        description="x264/x265 CRF for the non-face background (28-32 range).",
    )
    target_video_fps: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description=(
            "Target output frame rate. NOTE: microexpression analysis (40-200 "
            "ms events) requires >= 60 fps; raise this when feeding that path."
        ),
    )
    keyframe_interval_seconds: float = Field(
        default=2.0,
        gt=0.0,
        le=10.0,
        description="Forced GOP boundary cadence; supports random-access analysis.",
    )

    # ---- Face detection / ROI geometry ------------------------------------
    face_detection_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum SCRFD/MediaPipe confidence to treat a region as a face.",
    )
    roi_padding_pct: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description=(
            "Padding around the detected face bbox as a fraction of bbox size. "
            "0.20 leaves room for brow / chin movement without losing background."
        ),
    )
    roi_smoothing_window_frames: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Temporal smoothing window for the ROI bbox to suppress jitter.",
    )

    # ---- Tier 3: edge-first feature cadence -------------------------------
    edge_full_landmark_fps: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description="Landmark / AU emission rate in EDGE_FULL mode.",
    )
    edge_minimal_landmark_fps: float = Field(
        default=10.0,
        gt=0.0,
        le=60.0,
        description="Reduced landmark rate for EDGE_MINIMAL (< 1 Mbps uplinks).",
    )

    # ---- Adaptive bandwidth thresholds (Mbps) -----------------------------
    # Boundaries between tiers. Must be strictly decreasing -- enforced below.
    bandwidth_threshold_raw_mbps: float = Field(
        default=10.0,
        gt=0.0,
        description="At or above this uplink, the pipeline keeps RAW.",
    )
    bandwidth_threshold_roi_mbps: float = Field(
        default=5.0,
        gt=0.0,
        description="At or above this uplink, the pipeline uses ROI_ENCODED.",
    )
    bandwidth_threshold_edge_full_mbps: float = Field(
        default=1.0,
        gt=0.0,
        description="At or above this uplink, the pipeline uses EDGE_FULL.",
    )

    @model_validator(mode="after")
    def _bandwidth_thresholds_strictly_decreasing(self) -> "CompressionConfig":
        """Tier boundaries must be ordered RAW > ROI > EDGE_FULL.

        A non-monotonic ordering would make ``select_mode`` ambiguous, so we
        reject it at construction time rather than letting it surface as a
        silently-wrong tier choice in production.
        """
        if not (
            self.bandwidth_threshold_raw_mbps
            > self.bandwidth_threshold_roi_mbps
            > self.bandwidth_threshold_edge_full_mbps
        ):
            raise ValueError(
                "Bandwidth thresholds must be strictly decreasing: "
                f"raw={self.bandwidth_threshold_raw_mbps} > "
                f"roi={self.bandwidth_threshold_roi_mbps} > "
                f"edge_full={self.bandwidth_threshold_edge_full_mbps}"
            )
        return self

    def select_mode(self, available_uplink_mbps: float) -> CompressionMode:
        """Return the highest-fidelity tier that fits the given uplink budget.

        Args:
            available_uplink_mbps: Measured upstream bandwidth in megabits/sec.
                Negative values are treated as zero (offline / no link).

        Returns:
            The CompressionMode the pipeline should run for this session.
        """
        mbps = max(0.0, available_uplink_mbps)
        if mbps >= self.bandwidth_threshold_raw_mbps:
            return CompressionMode.RAW
        if mbps >= self.bandwidth_threshold_roi_mbps:
            return CompressionMode.ROI_ENCODED
        if mbps >= self.bandwidth_threshold_edge_full_mbps:
            return CompressionMode.EDGE_FULL
        return CompressionMode.EDGE_MINIMAL


class CompressionResult(BaseModel):
    """Outputs and telemetry from one run of the compression pipeline.

    Path fields marked Optional are populated only when the active tier emits
    that artifact:

        RAW            -> flac_audio_path only (video is the original input).
        ROI_ENCODED    -> flac_audio_path + roi_video_path.
        EDGE_FULL      -> flac_audio_path + landmarks_path + features_path.
        EDGE_MINIMAL   -> flac_audio_path + landmarks_path.

    Sizes are in bytes and are populated alongside their corresponding path.
    The pipeline is responsible for keeping each ``*_path`` and ``*_size_bytes``
    pair consistent; ``model_validator`` below enforces this at write time.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    mode: CompressionMode = Field(
        description="Compression tier that actually ran (post adaptive selection)."
    )

    input_path: Path = Field(description="Source media file submitted to the pipeline.")
    output_dir: Path = Field(description="Directory containing all artifacts below.")
    input_size_bytes: int = Field(ge=0)

    # ---- Tier 1 audio (always present) ------------------------------------
    flac_audio_path: Path = Field(
        description="Lossless FLAC audio. Always populated -- ML never sees lossy."
    )
    flac_size_bytes: int = Field(ge=0)

    # ---- Tier 2 ROI video (ROI_ENCODED only) ------------------------------
    roi_video_path: Optional[Path] = Field(
        default=None,
        description="ROI-encoded video. None outside ROI_ENCODED mode.",
    )
    roi_video_size_bytes: Optional[int] = Field(default=None, ge=0)

    # ---- Tier 3 edge artifacts (EDGE_FULL / EDGE_MINIMAL) -----------------
    landmarks_path: Optional[Path] = Field(
        default=None,
        description="Per-frame face landmarks + AU activations (parquet/protobuf).",
    )
    landmarks_size_bytes: Optional[int] = Field(default=None, ge=0)
    features_path: Optional[Path] = Field(
        default=None,
        description="Audio features + face embeddings. EDGE_FULL only.",
    )
    features_size_bytes: Optional[int] = Field(default=None, ge=0)

    # ---- Telemetry --------------------------------------------------------
    compression_ratios: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Stage -> output_size / input_size. Common keys: 'audio', 'video', "
            "'overall'. Values < 1.0 indicate compression."
        ),
    )
    processing_times: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Stage -> wall-clock seconds. Common keys: 'face_detection', "
            "'audio_encode', 'video_encode', 'edge_extract', 'total'."
        ),
    )
    face_detected_pct: float = Field(
        ge=0.0,
        le=100.0,
        description=(
            "Percent of analyzed frames where at least one face was detected. "
            "Low values mean ROI encoding fell back to a uniform CRF and "
            "downstream confidence should be reduced."
        ),
    )

    # ---- Path coercion ----------------------------------------------------
    # Pydantic v2 already accepts str for Path, but we coerce explicitly so the
    # internal type is *always* pathlib.Path -- callers comparing fields with
    # ``isinstance(..., Path)`` get a deterministic answer.
    @field_validator("input_path", "output_dir", "flac_audio_path", mode="before")
    @classmethod
    def _coerce_required_path(cls, v: object) -> Path:
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            return Path(v)
        raise TypeError(f"Expected Path or str, got {type(v).__name__}")

    @field_validator(
        "roi_video_path", "landmarks_path", "features_path", mode="before"
    )
    @classmethod
    def _coerce_optional_path(cls, v: object) -> Optional[Path]:
        if v is None:
            return None
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            return Path(v)
        raise TypeError(f"Expected Path, str, or None; got {type(v).__name__}")

    @model_validator(mode="after")
    def _path_size_pairs_consistent(self) -> "CompressionResult":
        """Each optional artifact must have its path and size populated together.

        Catches caller bugs where one field is filled but the matching size is
        forgotten -- which would otherwise show up as a None propagating into
        analytics dashboards.
        """
        pairs: tuple[tuple[str, Optional[Path], Optional[int]], ...] = (
            ("roi_video", self.roi_video_path, self.roi_video_size_bytes),
            ("landmarks", self.landmarks_path, self.landmarks_size_bytes),
            ("features", self.features_path, self.features_size_bytes),
        )
        for name, path, size in pairs:
            if (path is None) != (size is None):
                raise ValueError(
                    f"{name}_path and {name}_size_bytes must both be set or both None "
                    f"(got path={path!r}, size={size!r})"
                )
        return self
