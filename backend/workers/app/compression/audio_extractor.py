"""Audio extraction stage of the compression pipeline.

Splits each ingested media file into TWO separate audio artifacts whose
purposes are strictly different and must NOT be confused:

    1. FLAC  (48 kHz / mono / lossless) -- the ML-ingest copy.
    2. Opus  (32 kbps / mono / lossy)   -- the user-playback copy.

CRITICAL INVARIANT (CLAUDE.md "CRITICAL INVARIANTS" #1):
Lossy audio must NEVER be fed to an ML model. The Opus file exists solely so
the mobile app can stream a small audio preview alongside its UI. Every
downstream analyzer -- WhisperX transcription, DeBERTa contradiction NLI,
emotion2vec+ vocal tonality, parselmouth acoustic features -- reads the FLAC
path. The return-tuple ordering of ``AudioExtractor.extract`` is part of the
contract: index 0 is ML-safe, index 1 is playback-only. Do not reverse them.

Implementation notes:
- ``map_metadata=-1`` is passed to both ffmpeg invocations so container-level
  metadata (camera serial numbers, GPS tags, author fields) is NOT propagated
  into the artifacts. This supports CLAUDE.md invariant #3 (never log PII).
- Both encodes always go through ffmpeg even if the source is already FLAC,
  because the source may not be at 48 kHz / mono / 16-bit. Re-encoding from
  FLAC -> FLAC at the canonical params is itself lossless.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import ffmpeg

from backend.shared.schemas.media import CompressionConfig

logger = logging.getLogger(__name__)


# Container/codec extensions we accept as input. ffmpeg can demux many more,
# but we whitelist the common ones so an unsupported format fails fast with a
# clear message instead of surfacing as an obscure ffmpeg decode error mid-run.
SUPPORTED_INPUT_EXTENSIONS: frozenset[str] = frozenset({
    # Video containers.
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
    ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".m2ts", ".3gp",
    # Audio-only containers. Legitimate when re-processing a stored FLAC or
    # ingesting an audio-only stream (e.g. Slack Huddles).
    ".wav", ".flac", ".m4a", ".mp3", ".aac", ".ogg", ".opus",
})


class UnsupportedMediaError(ValueError):
    """Raised when the input file is not a supported container, or has no
    decodable audio stream."""


class AudioExtractionError(RuntimeError):
    """Raised when ffmpeg fails to produce one of the requested artifacts."""


class AudioExtractor:
    """Extract lossless (ML) and lossy (playback) audio from a media file.

    Construction is cheap and the same instance is safe to share across
    threads -- it owns no mutable state beyond the immutable config.

    Args:
        config: Optional CompressionConfig. Defaults to the project-wide
            audio defaults (48 kHz, mono, 16-bit, FLAC level 5). Only the
            audio-related fields are read; video fields are ignored here.
        opus_bitrate: Bitrate string for the playback-only Opus copy.
            Defaults to "32k" per the project spec. The Opus rate is NOT in
            CompressionConfig because Opus is a playback artifact, not part
            of the analysis pipeline.
    """

    DEFAULT_OPUS_BITRATE: str = "32k"

    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        opus_bitrate: Optional[str] = None,
    ) -> None:
        self.config: CompressionConfig = config or CompressionConfig()
        self.opus_bitrate: str = opus_bitrate or self.DEFAULT_OPUS_BITRATE

        # Per-format encode timings from the most recent extract() call.
        # Exposed as attributes (rather than widening extract()'s return) so
        # downstream UIs (e.g. scripts/test_compression.py) can show separate
        # FLAC/Opus rows without us coupling them to the result schema.
        self.last_flac_seconds: float = 0.0
        self.last_opus_seconds: float = 0.0

    def extract(
        self,
        video_path: Path,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Extract FLAC + Opus audio from ``video_path`` into ``output_dir``.

        Args:
            video_path: Source media file (video container or audio-only).
            output_dir: Destination directory. Created if it does not exist.

        Returns:
            ``(flac_path, opus_path)``. Index 0 (FLAC) is the ONLY artifact
            that may be passed to an ML model. Index 1 (Opus) is for the
            mobile playback UI only.

        Raises:
            FileNotFoundError: ``video_path`` does not exist.
            UnsupportedMediaError: extension not whitelisted, file is not a
                regular file, or the file contains no audio stream.
            AudioExtractionError: ffmpeg failed on one of the encodes.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        self._validate_input(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        flac_path = output_dir / f"{video_path.stem}.flac"
        opus_path = output_dir / f"{video_path.stem}.opus"

        input_size = video_path.stat().st_size

        flac_seconds = self._encode_flac(video_path, flac_path)
        opus_seconds = self._encode_opus(video_path, opus_path)
        self.last_flac_seconds = flac_seconds
        self.last_opus_seconds = opus_seconds

        flac_size = flac_path.stat().st_size
        opus_size = opus_path.stat().st_size

        # Compression ratios are output_size / input_size; values < 1 indicate
        # space saved. opus_vs_flac shows how much smaller the playback copy
        # is than the ML copy -- useful for storage tier accounting.
        logger.info(
            "audio_extracted "
            "input=%s input_size=%d "
            "flac=%s flac_size=%d flac_ratio=%.4f flac_seconds=%.2f "
            "opus=%s opus_size=%d opus_ratio=%.4f opus_seconds=%.2f "
            "opus_vs_flac=%.4f",
            video_path, input_size,
            flac_path, flac_size,
            (flac_size / input_size) if input_size else 0.0,
            flac_seconds,
            opus_path, opus_size,
            (opus_size / input_size) if input_size else 0.0,
            opus_seconds,
            (opus_size / flac_size) if flac_size else 0.0,
        )

        return flac_path, opus_path

    # ---- internals --------------------------------------------------------

    def _validate_input(self, video_path: Path) -> None:
        """Existence + extension whitelist + audio-stream presence check.

        We probe with ffprobe rather than just trusting the extension because
        a .mp4 with no audio track is a real and recurring case (silent
        screen recordings). Catching it here returns a clean error before we
        spawn ffmpeg twice.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Input media file not found: {video_path}")
        if not video_path.is_file():
            raise UnsupportedMediaError(f"Not a regular file: {video_path}")

        ext = video_path.suffix.lower()
        if ext not in SUPPORTED_INPUT_EXTENSIONS:
            raise UnsupportedMediaError(
                f"Unsupported extension {ext!r} for {video_path}. "
                f"Supported: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
            )

        try:
            probe = ffmpeg.probe(str(video_path))
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise UnsupportedMediaError(
                f"ffprobe could not read {video_path}: {stderr}"
            ) from exc

        audio_streams = [
            s for s in probe.get("streams", [])
            if s.get("codec_type") == "audio"
        ]
        if not audio_streams:
            raise UnsupportedMediaError(
                f"No audio stream found in {video_path}; cannot extract."
            )

    def _encode_flac(self, video_path: Path, flac_path: Path) -> float:
        """Encode the lossless ML-bound FLAC copy. Returns wall-clock seconds.

        Parameters are sourced from CompressionConfig so the canonical 48 kHz
        / mono / 16-bit / level-5 contract lives in one place (media.py).
        """
        # FLAC's sample format is s16 for 16-bit and s32 for 24/32-bit; FFmpeg
        # stores 24-bit samples in s32 with the upper byte unused.
        sample_fmt = "s16" if self.config.audio_bit_depth == 16 else "s32"

        start = time.perf_counter()
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(flac_path),
                    vn=None,                                          # drop any video
                    acodec="flac",
                    ar=self.config.audio_sample_rate_hz,              # 48000 Hz
                    ac=self.config.audio_channels,                    # mono
                    sample_fmt=sample_fmt,
                    compression_level=self.config.flac_compression_level,
                    map_metadata=-1,                                  # strip PII
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise AudioExtractionError(
                f"FLAC encode failed for {video_path}: {stderr}"
            ) from exc
        return time.perf_counter() - start

    def _encode_opus(self, video_path: Path, opus_path: Path) -> float:
        """Encode the playback-only Opus copy. Returns wall-clock seconds.

        WARNING: This output is LOSSY. It must NEVER be passed to an ML model
        (CLAUDE.md invariant #1). It exists only so the mobile app can stream
        a small audio preview alongside its UI.

        ``application=voip`` selects libopus's speech-tuned mode, which gives
        better intelligibility at 32 kbps mono than the default ``audio``
        mode. ``ar=48000`` matches Opus's native internal sample rate so no
        resampling artefacts are introduced.
        """
        start = time.perf_counter()
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(opus_path),
                    vn=None,
                    acodec="libopus",
                    ac=self.config.audio_channels,                    # mono
                    ar=48_000,                                        # Opus native rate
                    application="voip",                               # speech-tuned
                    map_metadata=-1,                                  # strip PII
                    **{"b:a": self.opus_bitrate},                     # 32k
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise AudioExtractionError(
                f"Opus encode failed for {video_path}: {stderr}"
            ) from exc
        return time.perf_counter() - start
