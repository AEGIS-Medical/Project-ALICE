"""ROI-aware video encoder for the compression pipeline (Tier 2).

This is the **simplified ROI v1** implementation. It does NOT yet build a
per-pixel QP map; it picks a single CRF for the whole clip based on whether
MediaPipe detects a face anywhere in it. See "Future v2" below for the full
per-frame ROI plan.

Why simplified for Day 1:
    Per-frame ROI encoding requires either (a) x265 zone encoding driven by a
    moving-bbox track, or (b) a dual-encode-and-overlay pass. Both are
    delicate to get right (drift between detector and encoder timestamps,
    overlay alignment, dual-pass timing). The spec explicitly authorizes the
    simpler heuristic for Day 1, so we ship that and unblock the rest of the
    pipeline.

Heuristic (per spec):
    1. Sample frames at ~1 fps, run MediaPipe Face Detection (the lightweight
       detector, NOT Face Mesh -- detection is roughly 10x faster).
    2. face_detected_pct == 0  -> log WARNING, encode whole clip at CRF 26.
       (CRF 26 not 32: a "no face detected" result frequently means the
       detector missed the face, not that there was no face. We hedge toward
       higher quality.)
    3. face_detected_pct  > 0  -> encode whole clip at CRF 22.

Future v2 (NOT implemented here):
    - Per-frame face bbox track (already produced by step 1) is exported as
      an x265 zones string, giving QP -12 inside the ROI and CRF 32 outside.
    - Or: dual-encode the cropped face region at CRF 18 and overlay it on a
      CRF 32 background. Picks up SSIM > 0.98 on the face per CLAUDE.md.

Audio is intentionally dropped (``-an``) -- it is handled by AudioExtractor
and is the only ML-bound copy. Mixing audio back in here would risk a lossy
re-encode reaching downstream analyzers (CLAUDE.md invariant #1).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import ffmpeg
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from backend.shared.schemas.media import CompressionConfig
from backend.workers.app.compression.audio_extractor import UnsupportedMediaError
from backend.workers.app.compression.models import face_detector_model

logger = logging.getLogger(__name__)


# Video container extensions accepted as input. We deliberately exclude the
# audio-only formats that AudioExtractor accepts -- ROI encoding makes no
# sense without a video stream.
SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
    ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".m2ts", ".3gp",
})


class ROIEncodingError(RuntimeError):
    """Raised when ffmpeg fails to produce the ROI-encoded video."""


class ROIEncoder:
    """Day-1 simplified ROI video encoder.

    Thread-safe in the sense that it owns no mutable shared state across
    threads -- but the ``last_*`` telemetry attributes ARE mutated by each
    call to :meth:`encode`, so a single instance must not be shared across
    concurrent encodes. Construct one per worker.

    The class exposes telemetry from the most recent encode through public
    attributes (``last_face_detected_pct``, ``last_frames_sampled``) because
    the spec fixes ``encode``'s return type to ``Path``. Downstream
    orchestrators read these to populate ``CompressionResult.face_detected_pct``.
    """

    # ---- Encoder configuration (Day 1 fixed values per spec) -------------
    CRF_FACE_PRESENT: int = 22
    CRF_NO_FACE: int = 26
    VIDEO_CODEC: str = "libx264"
    ENCODER_PRESET: str = "medium"
    PIXEL_FORMAT: str = "yuv420p"

    # ---- Face-scan configuration ----------------------------------------
    SAMPLE_INTERVAL_SECONDS: float = 1.0
    # We hard-code BlazeFace short-range (<= ~2 m, selfie / video-call
    # distance) via the model file resolved by ``face_detector_model()``.
    # If ALICE later adds room-camera support, swap the model file and
    # expose the choice on CompressionConfig.

    def __init__(self) -> None:
        self.last_face_detected_pct: float = 0.0
        self.last_frames_sampled: int = 0

    def encode(
        self,
        video_path: Path,
        output_dir: Path,
        config: CompressionConfig,
    ) -> Path:
        """Run the simplified ROI encode on ``video_path``.

        Args:
            video_path: Source video file. Must contain a video stream;
                audio-only inputs are rejected because there is nothing to
                ROI-encode.
            output_dir: Destination directory. Created if it does not exist.
            config: Project compression config. Only ``face_detection_confidence``
                is read in v1 (the per-frame QP map fields will be used in v2).

        Returns:
            Path to the encoded MP4. After return, ``self.last_face_detected_pct``
            and ``self.last_frames_sampled`` reflect this run.

        Raises:
            FileNotFoundError: ``video_path`` does not exist.
            UnsupportedMediaError: extension is not on the video whitelist,
                or OpenCV cannot open the file for face scanning.
            ROIEncodingError: ffmpeg failed during the encode pass.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        self._validate_input(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        face_detected_pct, frames_sampled = self._scan_for_faces(video_path, config)
        self.last_face_detected_pct = face_detected_pct
        self.last_frames_sampled = frames_sampled

        if face_detected_pct == 0.0:
            crf = self.CRF_NO_FACE
            logger.warning(
                "roi_no_face_detected path=%s frames_sampled=%d "
                "fallback_crf=%d note=detector_may_have_missed_face",
                video_path, frames_sampled, crf,
            )
        else:
            crf = self.CRF_FACE_PRESENT

        output_path = output_dir / f"{video_path.stem}_roi.mp4"
        encode_seconds = self._encode_video(video_path, output_path, crf)

        input_size = video_path.stat().st_size
        output_size = output_path.stat().st_size
        logger.info(
            "roi_encoded input=%s input_size=%d output=%s output_size=%d "
            "ratio=%.4f crf=%d face_detected_pct=%.2f frames_sampled=%d "
            "encode_seconds=%.2f",
            video_path, input_size, output_path, output_size,
            (output_size / input_size) if input_size else 0.0,
            crf, face_detected_pct, frames_sampled, encode_seconds,
        )
        return output_path

    # ---- internals --------------------------------------------------------

    def _validate_input(self, video_path: Path) -> None:
        """Existence + video-extension whitelist check.

        We do NOT ffprobe here for a video-stream presence check because the
        face-scan step opens the file with OpenCV and will raise a clean
        ``UnsupportedMediaError`` if there is no decodable video.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        if not video_path.is_file():
            raise UnsupportedMediaError(f"Not a regular file: {video_path}")
        ext = video_path.suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise UnsupportedMediaError(
                f"Unsupported video extension {ext!r} for {video_path}. "
                f"Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
            )

    def _scan_for_faces(
        self,
        video_path: Path,
        config: CompressionConfig,
    ) -> tuple[float, int]:
        """Sample frames at ~1 fps and count those containing >= 1 face.

        Returns:
            ``(face_detected_pct, frames_sampled)``. Percent is in [0, 100].
            If the file has no readable frames, returns ``(0.0, 0)``; the
            caller treats that the same as "no face anywhere".

        Raises:
            UnsupportedMediaError: OpenCV could not open the file.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise UnsupportedMediaError(
                f"OpenCV could not open {video_path} for face scanning."
            )

        # CAP_PROP_FPS occasionally returns 0 on malformed containers;
        # default to 30 fps so we still get a sane sampling stride.
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_stride = max(
            1, int(round(source_fps * self.SAMPLE_INTERVAL_SECONDS))
        )

        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(face_detector_model()),
            ),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=config.face_detection_confidence,
        )
        # Tasks API uses explicit lifecycle (no context-manager protocol),
        # so we close in a finally to guarantee resource release.
        detector = vision.FaceDetector.create_from_options(options)
        try:
            faces_seen = 0
            frames_sampled = 0
            frame_idx = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % sample_stride == 0:
                    # MediaPipe expects RGB; cv2 reads BGR.
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    # detect_for_video requires monotonically increasing
                    # timestamps in ms; derive from the source frame index.
                    timestamp_ms = int((frame_idx / source_fps) * 1000)
                    result = detector.detect_for_video(mp_image, timestamp_ms)
                    frames_sampled += 1
                    if result.detections:
                        faces_seen += 1
                frame_idx += 1

            if frames_sampled == 0:
                return 0.0, 0
            return (faces_seen / frames_sampled) * 100.0, frames_sampled
        finally:
            detector.close()
            cap.release()

    def _encode_video(
        self,
        video_path: Path,
        output_path: Path,
        crf: int,
    ) -> float:
        """Re-encode the video at the given CRF, dropping audio. Returns seconds."""
        start = time.perf_counter()
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_path),
                    an=None,                      # drop audio (handled separately)
                    vcodec=self.VIDEO_CODEC,
                    crf=crf,
                    preset=self.ENCODER_PRESET,
                    pix_fmt=self.PIXEL_FORMAT,
                    map_metadata=-1,              # strip container PII
                    movflags="+faststart",        # moov atom up front for streaming
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise ROIEncodingError(
                f"ROI encode failed for {video_path}: {stderr}"
            ) from exc
        return time.perf_counter() - start
