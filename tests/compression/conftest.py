"""Shared fakes for the compression test suite.

These let the landmark-extraction tests run without MediaPipe model
downloads or real video decoding. We patch the two heavy boundaries --
``cv2.VideoCapture`` (frame source) and ``vision.FaceLandmarker``
(inference) -- with deterministic fakes. ``cv2.cvtColor`` and ``mp.Image``
are left real because they are cheap on the tiny synthetic frames below.
"""

from __future__ import annotations

import numpy as np


class FakeCapture:
    """Stand-in for ``cv2.VideoCapture`` that yields N blank frames."""

    def __init__(self, n_frames: int, fps: float = 30.0) -> None:
        self._n = n_frames
        self._i = 0
        self._fps = fps
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API name)
        return True

    def get(self, _prop: int) -> float:
        return self._fps

    def read(self):
        if self._i >= self._n:
            return False, None
        self._i += 1
        # Tiny real array so cv2.cvtColor / mp.Image accept it cheaply.
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class _FakeLandmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FakeResult:
    def __init__(self, faces: list[list[_FakeLandmark]]) -> None:
        self.face_landmarks = faces


class FakeLandmarker:
    """Stand-in for ``vision.FaceLandmarker``.

    Args:
        with_face: emit a face on every frame when True.
        n_landmarks: mesh size (478 for the real refined mesh).
        raise_after: if set, raise RuntimeError once this many frames have
            been processed -- simulates a mid-stream interruption/kill.
    """

    def __init__(
        self,
        with_face: bool = True,
        n_landmarks: int = 478,
        raise_after: int | None = None,
    ) -> None:
        self.with_face = with_face
        self.n_landmarks = n_landmarks
        self.raise_after = raise_after
        self.closed = False
        self._calls = 0

    def detect_for_video(self, _image, _timestamp_ms: int) -> _FakeResult:
        self._calls += 1
        if self.raise_after is not None and self._calls > self.raise_after:
            raise RuntimeError("simulated mid-stream interruption")
        if self.with_face:
            mesh = [_FakeLandmark(0.1, 0.2, 0.0) for _ in range(self.n_landmarks)]
            return _FakeResult([mesh])
        return _FakeResult([])

    def close(self) -> None:
        self.closed = True
