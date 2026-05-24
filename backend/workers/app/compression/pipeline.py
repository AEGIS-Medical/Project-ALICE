"""End-to-end orchestrator for the compression pipeline.

This module wires the four stage classes (AudioExtractor, ROIEncoder,
FeatureExtractor) into a single ``process()`` call that takes a media file
plus a ``CompressionMode`` and returns a fully-populated ``CompressionResult``.

Mode -> stage map (per spec):

    RAW            audio + ROI video + audio features
    ROI_ENCODED    audio + ROI video + audio features
    EDGE_FULL      audio + landmarks + audio features
    EDGE_MINIMAL   audio + landmarks                   (NO audio features)

Stage failure policy:
    Audio extraction is the foundation of every downstream analyzer; if it
    fails, ``CompressionResult`` cannot be built (``flac_audio_path`` is
    required), so the exception is allowed to propagate.

    Every other stage (ROI, landmarks, audio features) is wrapped in a
    broad ``except Exception`` -- a stage failure is logged with full
    traceback, the corresponding result fields are left as ``None``, and
    the pipeline continues. The spec calls for this behavior explicitly so
    that, e.g., a MediaPipe crash on a tricky frame does not destroy the
    audio + ROI work that already succeeded.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from backend.shared.schemas.media import (
    CompressionConfig,
    CompressionMode,
    CompressionResult,
)
from backend.workers.app.compression.audio_extractor import AudioExtractor
from backend.workers.app.compression.feature_extractor import FeatureExtractor
from backend.workers.app.compression.roi_encoder import (
    SUPPORTED_VIDEO_EXTENSIONS,
    ROIEncoder,
)

logger = logging.getLogger(__name__)


# Stage-eligibility sets. Centralized here so the dispatch logic in
# ``process`` is a single membership check per stage.
_MODES_WITH_ROI_VIDEO: frozenset[CompressionMode] = frozenset({
    CompressionMode.RAW,
    CompressionMode.ROI_ENCODED,
})
_MODES_WITH_LANDMARKS: frozenset[CompressionMode] = frozenset({
    CompressionMode.EDGE_FULL,
    CompressionMode.EDGE_MINIMAL,
})
_MODES_WITHOUT_AUDIO_FEATURES: frozenset[CompressionMode] = frozenset({
    CompressionMode.EDGE_MINIMAL,
})


class PipelineInputError(ValueError):
    """Raised when the pipeline's gate-keeper input validation fails.

    Distinct from the per-stage validators (UnsupportedMediaError, etc.) so
    callers can tell "the file the user handed us is unusable" apart from
    "stage X choked on a valid-looking file".
    """


class CompressionPipeline:
    """Orchestrates audio extraction, ROI encoding, and feature extraction.

    The four stage objects are composed in ``__init__`` and reused across
    every ``process()`` call. They are also overridable via constructor
    arguments so tests can inject fakes without monkey-patching.

    Thread-safety: the pipeline shares its three stage instances across
    calls. AudioExtractor and FeatureExtractor are stateless across calls
    apart from telemetry attributes; ROIEncoder mutates ``last_*`` per
    call. Construct one CompressionPipeline per worker -- do not share
    across threads.

    Args:
        config: Project compression config. Defaults to
            ``CompressionConfig()`` (project-wide defaults).
        audio_extractor / roi_encoder / feature_extractor: Optional
            pre-built stages, primarily for tests.
    """

    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        audio_extractor: Optional[AudioExtractor] = None,
        roi_encoder: Optional[ROIEncoder] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self.config: CompressionConfig = config or CompressionConfig()
        self.audio_extractor: AudioExtractor = (
            audio_extractor or AudioExtractor(config=self.config)
        )
        self.roi_encoder: ROIEncoder = roi_encoder or ROIEncoder()
        self.feature_extractor: FeatureExtractor = (
            feature_extractor or FeatureExtractor()
        )

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        mode: CompressionMode = CompressionMode.RAW,
    ) -> CompressionResult:
        """Run the full pipeline against ``input_path``.

        Args:
            input_path: Source video file. Must exist, be a regular file,
                be non-empty, and have a video extension on the whitelist.
            output_dir: Destination root. Four subdirectories
                (``audio/``, ``video/``, ``landmarks/``, ``features/``) are
                always created here, even when a given mode does not write
                into all of them -- predictable layout helps downstream
                tooling.
            mode: Which compression tier to run. Defaults to RAW.

        Returns:
            A populated ``CompressionResult``. Optional path fields will be
            ``None`` for stages that did not run for ``mode`` OR that ran
            but failed gracefully.

        Raises:
            FileNotFoundError: ``input_path`` does not exist.
            PipelineInputError: ``input_path`` is not a regular file, is
                empty, or has an unsupported extension.
            Audio extraction errors: anything raised by AudioExtractor
                propagates -- audio is the only mandatory stage.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        # ---- Step 1: gate-keeper validation -------------------------------
        self._validate_input(input_path)

        # ---- Step 2: subdirectories --------------------------------------
        audio_dir, video_dir, landmarks_dir, features_dir = self._make_subdirs(
            output_dir,
        )

        input_size = input_path.stat().st_size
        processing_times: dict[str, float] = {}
        compression_ratios: dict[str, float] = {}

        logger.info(
            "pipeline_start input=%s mode=%s input_size=%d output_dir=%s",
            input_path, mode.value, input_size, output_dir,
        )

        # ---- Step 3: audio extraction (ALWAYS) ---------------------------
        # This is the only stage allowed to bring the pipeline down. Without
        # FLAC we cannot satisfy CompressionResult.flac_audio_path, and
        # downstream analyzers have nothing to consume.
        t0 = time.perf_counter()
        flac_path, _opus_path = self.audio_extractor.extract(input_path, audio_dir)
        processing_times["audio_extract"] = time.perf_counter() - t0
        flac_size = flac_path.stat().st_size
        compression_ratios["audio"] = (
            (flac_size / input_size) if input_size else 0.0
        )

        # ---- Step 4: ROI video encoding (RAW / ROI_ENCODED) --------------
        roi_video_path: Optional[Path] = None
        roi_video_size: Optional[int] = None
        roi_face_pct: Optional[float] = None
        if mode in _MODES_WITH_ROI_VIDEO:
            t0 = time.perf_counter()
            try:
                roi_video_path = self.roi_encoder.encode(
                    input_path, video_dir, self.config,
                )
                roi_video_size = roi_video_path.stat().st_size
                roi_face_pct = self.roi_encoder.last_face_detected_pct
                compression_ratios["video"] = (
                    (roi_video_size / input_size) if input_size else 0.0
                )
            except Exception as exc:
                logger.error(
                    "stage_failed stage=roi_encode input=%s mode=%s err=%s",
                    input_path, mode.value, exc, exc_info=True,
                )
                roi_video_path = None
                roi_video_size = None
            processing_times["roi_encode"] = time.perf_counter() - t0

        # ---- Step 5: landmark extraction (EDGE_FULL / EDGE_MINIMAL) ------
        landmarks_path: Optional[Path] = None
        landmarks_size: Optional[int] = None
        landmarks_face_pct: Optional[float] = None
        if mode in _MODES_WITH_LANDMARKS:
            t0 = time.perf_counter()
            try:
                landmarks_path = self.feature_extractor.extract_landmarks(
                    input_path, landmarks_dir,
                )
                landmarks_size = landmarks_path.stat().st_size
                processed = self.feature_extractor.last_frames_processed
                with_face = self.feature_extractor.last_frames_with_face
                landmarks_face_pct = (
                    (with_face / processed * 100.0) if processed else 0.0
                )
            except Exception as exc:
                logger.error(
                    "stage_failed stage=landmarks input=%s mode=%s err=%s",
                    input_path, mode.value, exc, exc_info=True,
                )
                landmarks_path = None
                landmarks_size = None
            processing_times["landmarks_extract"] = time.perf_counter() - t0

        # ---- Step 6: audio features (all modes EXCEPT EDGE_MINIMAL) ------
        # Note we feed the EXTRACTED FLAC, not the original input -- this
        # automatically satisfies FeatureExtractor's lossless-only check
        # (CLAUDE.md invariant #1) without having to plumb format awareness
        # through the orchestrator.
        features_path: Optional[Path] = None
        features_size: Optional[int] = None
        if mode not in _MODES_WITHOUT_AUDIO_FEATURES:
            t0 = time.perf_counter()
            try:
                features_path = self.feature_extractor.extract_audio_features(
                    flac_path, features_dir,
                )
                features_size = features_path.stat().st_size
            except Exception as exc:
                logger.error(
                    "stage_failed stage=audio_features input=%s mode=%s err=%s",
                    input_path, mode.value, exc, exc_info=True,
                )
                features_path = None
                features_size = None
            processing_times["audio_features"] = time.perf_counter() - t0

        # ---- Step 7: assemble the result ---------------------------------
        # face_detected_pct sourcing: prefer the ROI scanner's value (it
        # samples at native fps and is what was actually used to choose the
        # ROI CRF). Fall back to the landmark count when only edge stages
        # ran. If neither produced a value, default to 0.0 -- the schema
        # requires a float, and downstream analytics already treat 0.0 as
        # "low confidence, no face data".
        if roi_face_pct is not None:
            face_pct = roi_face_pct
        elif landmarks_face_pct is not None:
            face_pct = landmarks_face_pct
        else:
            face_pct = 0.0

        total_kept = flac_size
        for size in (roi_video_size, landmarks_size, features_size):
            if size is not None:
                total_kept += size
        compression_ratios["overall"] = (
            (total_kept / input_size) if input_size else 0.0
        )

        # 'total' is recorded last so it doesn't accidentally get re-summed
        # into itself if the dict is later iterated for aggregation.
        processing_times["total"] = sum(processing_times.values())

        result = CompressionResult(
            mode=mode,
            input_path=input_path,
            output_dir=output_dir,
            input_size_bytes=input_size,
            flac_audio_path=flac_path,
            flac_size_bytes=flac_size,
            roi_video_path=roi_video_path,
            roi_video_size_bytes=roi_video_size,
            landmarks_path=landmarks_path,
            landmarks_size_bytes=landmarks_size,
            features_path=features_path,
            features_size_bytes=features_size,
            compression_ratios=compression_ratios,
            processing_times=processing_times,
            face_detected_pct=face_pct,
        )

        logger.info(
            "pipeline_complete input=%s mode=%s total_seconds=%.2f "
            "face_pct=%.2f overall_ratio=%.4f "
            "flac=%s roi=%s landmarks=%s features=%s",
            input_path, mode.value, processing_times["total"], face_pct,
            compression_ratios.get("overall", 0.0),
            "ok" if flac_path else "missing",
            "ok" if roi_video_path else ("skipped" if mode not in _MODES_WITH_ROI_VIDEO else "failed"),
            "ok" if landmarks_path else ("skipped" if mode not in _MODES_WITH_LANDMARKS else "failed"),
            "ok" if features_path else ("skipped" if mode in _MODES_WITHOUT_AUDIO_FEATURES else "failed"),
        )
        return result

    # ---- internals --------------------------------------------------------

    def _validate_input(self, input_path: Path) -> None:
        """Gate-keeper validation: exists + regular + non-empty + video ext."""
        if not input_path.exists():
            raise FileNotFoundError(f"Pipeline input not found: {input_path}")
        if not input_path.is_file():
            raise PipelineInputError(
                f"Pipeline input is not a regular file: {input_path}"
            )
        if input_path.stat().st_size == 0:
            raise PipelineInputError(
                f"Pipeline input is empty (0 bytes): {input_path}"
            )
        ext = input_path.suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise PipelineInputError(
                f"Unsupported pipeline input extension {ext!r} for {input_path}. "
                f"Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
            )

    @staticmethod
    def _make_subdirs(
        output_dir: Path,
    ) -> tuple[Path, Path, Path, Path]:
        """Create the four pipeline output subdirectories.

        We create all four every run -- not just the ones the active mode
        will write into -- so the on-disk layout is predictable for
        downstream tooling regardless of which mode produced it.
        """
        audio_dir = output_dir / "audio"
        video_dir = output_dir / "video"
        landmarks_dir = output_dir / "landmarks"
        features_dir = output_dir / "features"
        for d in (audio_dir, video_dir, landmarks_dir, features_dir):
            d.mkdir(parents=True, exist_ok=True)
        return audio_dir, video_dir, landmarks_dir, features_dir
