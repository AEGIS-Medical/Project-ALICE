"""Feature extraction stage of the compression pipeline (Tier 3 inputs).

Two distinct extractors live here. Both produce artifacts that downstream
analyzers consume directly -- nothing in the pipeline re-encodes them:

  * :meth:`FeatureExtractor.extract_landmarks`
        MediaPipe Face Mesh -> per-frame JSON of 478 (x, y, z) landmarks.
        Consumed by the AU detector and the gaze sub-vector.
  * :meth:`FeatureExtractor.extract_audio_features`
        librosa MFCC / Chroma / Mel / Spectral-Contrast / Tonnetz, windowed
        at 1.0 s with 0.5 s stride by default. Consumed by the vocal tonality
        analyzer and the per-contact XGBoost flux scorer.

CRITICAL INVARIANT (CLAUDE.md #1): the audio-feature path REJECTS lossy
input. Only ``.flac`` and ``.wav`` are accepted -- ``.mp3``, ``.opus``,
``.aac``, ``.m4a``, ``.ogg`` etc. raise ``ValueError`` and the error message
names invariant #1 by number so the failure mode is unmistakable in logs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import librosa
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from backend.workers.app.compression.audio_extractor import UnsupportedMediaError
from backend.workers.app.compression.models import face_landmarker_model

logger = logging.getLogger(__name__)


# Lossless audio formats accepted by extract_audio_features. Anything else
# violates CLAUDE.md invariant #1 and is rejected with a ValueError that
# names the invariant explicitly.
LOSSLESS_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".flac", ".wav"})

# Video containers accepted by extract_landmarks.
SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
    ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".m2ts", ".3gp",
})


class FeatureExtractor:
    """Per-frame face landmarks and per-window audio features.

    The class itself owns no heavy ML state -- both extractors open their own
    short-lived MediaPipe / librosa contexts and release them on exit. That
    means a single instance is safe to reuse sequentially, but should NOT be
    shared across concurrent calls because the ``last_*`` telemetry
    attributes are mutated each call.

    Telemetry is exposed via attributes (``last_frames_processed``,
    ``last_frames_with_face``, ``last_audio_windows``, ``last_audio_sample_rate``)
    rather than widening the return types -- the file's contract pins both
    methods to ``Path`` returns.

    Args:
        frame_skip: Process every Nth frame for landmarks. ``1`` = every
            frame (the default and what AU analysis wants). Higher values
            trade temporal resolution for speed; raise only when wall-clock
            matters more than microexpression timing.
    """

    # ---- Audio feature dimensions (matched to spec) ----------------------
    N_MFCC: int = 40
    N_CHROMA: int = 12
    N_MEL: int = 128
    # librosa.feature.spectral_contrast returns (n_bands + 1, t); n_bands=6
    # -> 7 output rows, which is what the spec calls for.
    N_SPECTRAL_CONTRAST_BANDS: int = 6
    # Tonnetz is fixed at 6 dimensions by definition.
    N_TONNETZ: int = 6

    # ---- Face Mesh defaults ---------------------------------------------
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5
    MEDIAPIPE_MAX_FACES: int = 1

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

    # =========================================================================
    # Landmarks (MediaPipe Face Mesh)
    # =========================================================================

    def extract_landmarks(
        self,
        video_path: Path,
        output_dir: Path,
    ) -> Path:
        """Extract 478-point Face Mesh landmarks per frame to a JSON file.

        ``refine_landmarks=True`` is set so we get the full 478-point mesh
        (468 face + 10 iris). Without it MediaPipe returns 468 only.

        Frames where no face is detected are still emitted in the output
        with ``"landmarks": null`` so downstream analyzers can align by
        ``frame_number`` without re-deriving the timeline.

        Args:
            video_path: Source video. Must be a video container (audio-only
                inputs are rejected).
            output_dir: Destination. Created if missing.

        Returns:
            Path to the JSON file:
            ``{output_dir}/{stem}_landmarks.json``.

            JSON schema::

                [
                  {
                    "frame_number": int,
                    "timestamp_seconds": float,
                    "landmarks": [[x, y, z], ...]   // 478 entries, or null
                  },
                  ...
                ]

            Coordinates are normalized in ``[0, 1]`` for x/y; z is the
            depth value MediaPipe reports (negative = closer to camera).

        Raises:
            FileNotFoundError: ``video_path`` does not exist.
            UnsupportedMediaError: not a video extension, or OpenCV cannot
                open the file.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        self._validate_video_input(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise UnsupportedMediaError(
                f"OpenCV could not open {video_path} for landmark extraction."
            )

        # CAP_PROP_FPS occasionally returns 0 on malformed containers; fall
        # back to 30 fps so timestamps remain monotonic and roughly correct.
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        records: list[dict] = []
        frames_processed = 0
        frames_with_face = 0
        frame_idx = 0

        # ``output_face_blendshapes=False`` keeps the model emitting the full
        # 478-point mesh (468 face + 10 iris). Blendshapes / transformation
        # matrices would add cost we don't use in v1.
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
        # Tasks API objects use explicit close(), not a context manager.
        landmarker = vision.FaceLandmarker.create_from_options(options)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % self.frame_skip == 0:
                    # MediaPipe expects RGB; cv2 reads BGR.
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    # detect_for_video requires monotonically increasing
                    # timestamps in ms; derive from the source frame index.
                    timestamp_ms = int((frame_idx / fps) * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    landmarks: list[list[float]] | None = None
                    if result.face_landmarks:
                        # face_landmarks is a list-of-lists; one inner list
                        # per detected face. We pin num_faces=1, so [0] is safe.
                        mesh = result.face_landmarks[0]
                        landmarks = [[lm.x, lm.y, lm.z] for lm in mesh]
                        frames_with_face += 1

                    records.append({
                        "frame_number": frame_idx,
                        "timestamp_seconds": frame_idx / fps,
                        "landmarks": landmarks,
                    })
                    frames_processed += 1
                frame_idx += 1
        finally:
            landmarker.close()
            cap.release()

        output_path = output_dir / f"{video_path.stem}_landmarks.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(records, fh)

        self.last_frames_processed = frames_processed
        self.last_frames_with_face = frames_with_face

        face_pct = (
            (frames_with_face / frames_processed * 100.0)
            if frames_processed else 0.0
        )
        size = output_path.stat().st_size
        logger.info(
            "landmarks_extracted input=%s output=%s fps=%.2f frame_skip=%d "
            "frames_processed=%d frames_with_face=%d face_pct=%.2f output_size=%d",
            video_path, output_path, fps, self.frame_skip,
            frames_processed, frames_with_face, face_pct, size,
        )
        return output_path

    # =========================================================================
    # Audio features (librosa)
    # =========================================================================

    def extract_audio_features(
        self,
        flac_path: Path,
        output_dir: Path,
        window_sec: float = 1.0,
        stride_sec: float = 0.5,
    ) -> Path:
        """Compute per-window audio features and save them as a ``.npz``.

        Per the spec, each window's per-STFT-frame feature matrix is mean-
        pooled along the time axis to produce one vector per window, so the
        saved arrays all have shape ``(n_windows, feature_dim)``.

        Args:
            flac_path: Source audio. **Must be ``.flac`` or ``.wav``** -- any
                lossy format raises ``ValueError`` (see invariant #1 below).
            output_dir: Destination. Created if missing.
            window_sec: Analysis window length in seconds. Default 1.0.
            stride_sec: Hop between window starts in seconds. Default 0.5
                (50% overlap).

        Returns:
            Path to ``{output_dir}/{stem}_audio_features.npz`` with arrays:

                ``mfcc``                 (n_windows, 40)
                ``chroma``               (n_windows, 12)
                ``mel``                  (n_windows, 128)
                ``spectral_contrast``    (n_windows, 7)   # n_bands + 1
                ``tonnetz``              (n_windows, 6)
                ``timestamps_seconds``   (n_windows,)     # window start times
                ``sample_rate``          scalar (int)
                ``window_sec``           scalar (float)
                ``stride_sec``           scalar (float)

        Raises:
            FileNotFoundError: ``flac_path`` does not exist.
            ValueError: lossy input format (CLAUDE.md invariant #1), or
                non-positive window/stride.
        """
        flac_path = Path(flac_path)
        output_dir = Path(output_dir)
        self._validate_audio_input(flac_path)
        if window_sec <= 0 or stride_sec <= 0:
            raise ValueError(
                f"window_sec and stride_sec must be > 0 "
                f"(got window_sec={window_sec}, stride_sec={stride_sec})"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        # sr=None preserves the source rate (FLAC is 48 kHz per CLAUDE.md);
        # mono=True is a no-op when the source is already mono and a safe
        # mean-collapse otherwise.
        y, sr = librosa.load(str(flac_path), sr=None, mono=True)
        sr = int(sr)
        self.last_audio_sample_rate = sr

        window_samples = int(window_sec * sr)
        stride_samples = int(stride_sec * sr)
        if window_samples == 0 or stride_samples == 0:
            raise ValueError(
                f"window_sec / stride_sec are too small for sr={sr}: produced "
                f"window_samples={window_samples}, stride_samples={stride_samples}"
            )

        n_windows = (
            0 if len(y) < window_samples
            else (len(y) - window_samples) // stride_samples + 1
        )

        mfccs = np.empty((n_windows, self.N_MFCC), dtype=np.float32)
        chromas = np.empty((n_windows, self.N_CHROMA), dtype=np.float32)
        mels = np.empty((n_windows, self.N_MEL), dtype=np.float32)
        contrasts = np.empty(
            (n_windows, self.N_SPECTRAL_CONTRAST_BANDS + 1), dtype=np.float32,
        )
        tonnetzs = np.empty((n_windows, self.N_TONNETZ), dtype=np.float32)
        timestamps = np.empty(n_windows, dtype=np.float64)

        for i in range(n_windows):
            start = i * stride_samples
            win = y[start:start + window_samples]

            mfccs[i] = librosa.feature.mfcc(
                y=win, sr=sr, n_mfcc=self.N_MFCC,
            ).mean(axis=1)
            chromas[i] = librosa.feature.chroma_stft(
                y=win, sr=sr, n_chroma=self.N_CHROMA,
            ).mean(axis=1)
            mels[i] = librosa.feature.melspectrogram(
                y=win, sr=sr, n_mels=self.N_MEL,
            ).mean(axis=1)
            contrasts[i] = librosa.feature.spectral_contrast(
                y=win, sr=sr, n_bands=self.N_SPECTRAL_CONTRAST_BANDS,
            ).mean(axis=1)
            tonnetzs[i] = librosa.feature.tonnetz(y=win, sr=sr).mean(axis=1)
            timestamps[i] = start / sr

        output_path = output_dir / f"{flac_path.stem}_audio_features.npz"
        np.savez_compressed(
            output_path,
            mfcc=mfccs,
            chroma=chromas,
            mel=mels,
            spectral_contrast=contrasts,
            tonnetz=tonnetzs,
            timestamps_seconds=timestamps,
            sample_rate=np.int32(sr),
            window_sec=np.float32(window_sec),
            stride_sec=np.float32(stride_sec),
        )

        self.last_audio_windows = n_windows
        size = output_path.stat().st_size
        logger.info(
            "audio_features_extracted input=%s output=%s sr=%d "
            "duration_seconds=%.2f window_sec=%.2f stride_sec=%.2f "
            "n_windows=%d output_size=%d",
            flac_path, output_path, sr, len(y) / sr if sr else 0.0,
            window_sec, stride_sec, n_windows, size,
        )
        return output_path

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_video_input(self, video_path: Path) -> None:
        """Existence + video-extension whitelist."""
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

    def _validate_audio_input(self, flac_path: Path) -> None:
        """Enforce CLAUDE.md CRITICAL INVARIANT #1.

        The error message names the invariant by number on purpose: when this
        ValueError surfaces in a stack trace, the reader should not have to
        guess why the input was rejected.
        """
        if not flac_path.exists():
            raise FileNotFoundError(f"Input audio not found: {flac_path}")
        if not flac_path.is_file():
            raise ValueError(f"Not a regular file: {flac_path}")
        ext = flac_path.suffix.lower()
        if ext not in LOSSLESS_AUDIO_EXTENSIONS:
            raise ValueError(
                f"FeatureExtractor.extract_audio_features refuses {ext!r} input "
                f"({flac_path}). CLAUDE.md CRITICAL INVARIANT #1: lossy audio "
                f"must NEVER be fed to an ML model. "
                f"Accepted extensions: {sorted(LOSSLESS_AUDIO_EXTENSIONS)}. "
                f"Re-extract from the original media via AudioExtractor "
                f"(which always produces FLAC) and pass that path instead."
            )
