"""Model-file management for the compression pipeline's MediaPipe stages.

The MediaPipe Tasks API requires ``.tflite`` / ``.task`` model files to be
on disk. We download them lazily from Google's official MediaPipe model
storage on first use and cache them per-user. This keeps the repo lean
(no vendored binaries in git) and the dependency story honest -- the
weights belong to MediaPipe, not us.

Cost: a one-time ~4 MB network fetch the first time the pipeline runs on
a given machine. Subsequent runs hit the cache and never touch the
network.

The cache location is, in priority order:
    1. ``ALICE_MODEL_CACHE`` env var, if set (CI, Android, custom deploy).
    2. Windows: ``%LOCALAPPDATA%/project-alice/models``.
    3. Android (Termux / KMP bridge sets ``XDG_DATA_HOME`` to
       ``Context.filesDir``): ``$XDG_DATA_HOME/project-alice/models``.
    4. macOS / Linux XDG standard: ``~/.local/share/project-alice/models``.

P1-S7: the previous implementation resolved only ``%LOCALAPPDATA%``, which is
undefined on Android, Linux, and macOS -- the pipeline crashed on first use
off-Windows. The chain below resolves on every platform the KMP mobile client
bridges to.
"""

from __future__ import annotations

import logging
import os
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


# Pinned to /latest/ on Google's MediaPipe model storage. Both files are
# under Apache 2.0 (the license MediaPipe ships them with). For strict
# reproducibility later, swap the 'latest' path segment for a dated version.
_MODEL_URLS: dict[str, str] = {
    "face_detector.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
}


def _cache_dir() -> Path:
    """Return the platform-appropriate model cache directory.

    Resolution order (first match wins): ``ALICE_MODEL_CACHE`` override ->
    Windows ``%LOCALAPPDATA%`` -> Android ``$XDG_DATA_HOME`` -> XDG fallback
    under the home directory. See module docstring for the rationale.
    """
    override = os.environ.get("ALICE_MODEL_CACHE")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "project-alice" / "models"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "project-alice" / "models"
    return Path.home() / ".local" / "share" / "project-alice" / "models"


def ensure_model(name: str) -> Path:
    """Return the local path to ``name``, downloading on first use.

    Args:
        name: A key in :data:`_MODEL_URLS` (e.g. ``"face_detector.tflite"``).

    Returns:
        Absolute path to the cached file.

    Raises:
        KeyError: ``name`` is not a recognized model.
        RuntimeError: the download failed; a partial file is removed.
    """
    if name not in _MODEL_URLS:
        raise KeyError(
            f"Unknown model {name!r}. Known: {sorted(_MODEL_URLS)}"
        )

    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / name
    if dest.exists():
        return dest

    url = _MODEL_URLS[name]
    logger.info("model_download_start name=%s url=%s dest=%s", name, url, dest)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        # Don't leave a half-written file behind -- a corrupt cache entry
        # would fail every subsequent run silently.
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download MediaPipe model {name!r} from {url}: {exc}"
        ) from exc

    logger.info(
        "model_download_complete name=%s size=%d dest=%s",
        name, dest.stat().st_size, dest,
    )
    return dest


def face_detector_model() -> Path:
    """Cached path to the BlazeFace short-range face-detection model."""
    return ensure_model("face_detector.tflite")


def face_landmarker_model() -> Path:
    """Cached path to the 478-point Face Landmarker bundle."""
    return ensure_model("face_landmarker.task")
