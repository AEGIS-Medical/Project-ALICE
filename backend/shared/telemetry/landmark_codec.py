"""Landmark telemetry codec: quantization, keyframe/delta encoding, chunk framing.

Wire format spec: docs/superpowers/specs/2026-07-03-protobuf-landmark-telemetry-design.md
Schema: proto/landmarks.proto (codegen committed at backend/shared/proto_gen).

CLAUDE.md invariant #3: this module logs counts/bytes/ratios only -- never
coordinate values.
"""
from __future__ import annotations

import logging
import struct
import zlib
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Sequence

from backend.shared.proto_gen import landmarks_pb2 as pb

logger = logging.getLogger(__name__)

# Quantization scales. x,y are MediaPipe-normalized [0,1] -> uint12 (4095);
# z is approximately [-1,1] -> int12 (signed range 2047). Max x/y reconstruction
# error is 1/(2*4095) ≈ 1.2e-4 normalized ≈ 0.13 px at 1080p — below MediaPipe's
# own detector jitter (~0.2 px), so quantization is invisible downstream; finer
# precision (uint16) only encodes incompressible detector noise (measured 5.3x over
# the bandwidth envelope).
XY_SCALE: int = 4095
Z_SCALE: int = 2047


def quantize_frame(
    landmarks: list[list[float]],
) -> tuple[list[int], list[int], int]:
    """Quantize one frame of [x, y, z] landmarks.

    Returns:
        (xy, z, clamped_count): ``xy`` interleaves x0,y0,x1,y1,... as ints in
        [0, XY_SCALE]; ``z`` holds ints in [-Z_SCALE, Z_SCALE];
        ``clamped_count`` is how many input values fell outside their legal
        range and were clamped (MediaPipe emits slight overshoot at frame
        edges -- clamping is expected, but counted for telemetry).
    """
    xy: list[int] = []
    z: list[int] = []
    clamped = 0
    for point in landmarks:
        px, py, pz = point[0], point[1], point[2]
        for v in (px, py):
            if v < 0.0 or v > 1.0:
                clamped += 1
                v = 0.0 if v < 0.0 else 1.0
            xy.append(round(v * XY_SCALE))
        if pz < -1.0 or pz > 1.0:
            clamped += 1
            pz = -1.0 if pz < -1.0 else 1.0
        z.append(round(pz * Z_SCALE))
    return xy, z, clamped


def dequantize_frame(
    xy: Sequence[int], z: Sequence[int]
) -> list[list[float]]:
    """Inverse of :func:`quantize_frame` -- returns [[x, y, z], ...] floats."""
    return [
        [xy[2 * i] / XY_SCALE, xy[2 * i + 1] / XY_SCALE, z[i] / Z_SCALE]
        for i in range(len(z))
    ]


# ---- File framing constants --------------------------------------------
MAGIC: bytes = b"ALTM"
COMPRESSION_NONE: int = 0
COMPRESSION_ZLIB: int = 1

_LEN = struct.Struct("<I")
_METHOD = struct.Struct("<B")


class LandmarkDecodeError(RuntimeError):
    """Raised for real damage inside a length-valid chunk (corruption or a
    spec-violating frame sequence). Truncation is NOT an error -- see
    LandmarkDecoder.frames()."""


class DecodedFrame(NamedTuple):
    frame_number: int
    timestamp_seconds: float
    landmarks: Optional[list[list[float]]]  # 478 x [x, y, z], or None (no face)


class LandmarkEncoder:
    """Streaming writer for the ALTM landmark telemetry format.

    The header is written at construction, so the file is decodable from
    frame 0. Frames buffer into a LandmarkChunk that is zlib-compressed and
    flushed every ``keyframe_interval`` frames; every chunk therefore starts
    at a keyframe (or no-face frames followed by one), which is what makes
    any complete-chunk prefix of the file independently decodable.
    """

    def __init__(
        self,
        path: Path,
        *,
        landmark_count: int = 478,
        source_fps: float,
        frame_skip: int = 1,
        keyframe_interval: int = 30,
        zlib_level: int = 9,
    ) -> None:
        if source_fps <= 0:
            raise ValueError(f"source_fps must be > 0 (got {source_fps})")
        if keyframe_interval < 1:
            raise ValueError(
                f"keyframe_interval must be >= 1 (got {keyframe_interval})"
            )
        self._landmark_count = landmark_count
        self._keyframe_interval = keyframe_interval
        self._zlib_level = zlib_level
        self._closed = False

        # Previous emitted face frame's quantized values (delta reference).
        self._prev_xy: Optional[list[int]] = None
        self._prev_z: Optional[list[int]] = None

        self._chunk = pb.LandmarkChunk()
        self._frames_in_chunk = 0

        # Telemetry (final values valid after close()).
        self.frames_written = 0
        self.chunks_written = 0
        self.bytes_written = 0
        self.clamped_values = 0

        self._fh = open(path, "wb")
        header = pb.LandmarkStreamHeader(
            version=1,
            landmark_count=landmark_count,
            source_fps=source_fps,
            keyframe_interval=keyframe_interval,
            frame_skip=frame_skip,
        )
        hb = header.SerializeToString()
        self._fh.write(MAGIC)
        self._fh.write(_LEN.pack(len(hb)))
        self._fh.write(hb)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "LandmarkEncoder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- public API ----------------------------------------------------------
    def add_frame(
        self, frame_number: int, landmarks: Optional[list[list[float]]]
    ) -> None:
        if self._closed:
            raise ValueError("LandmarkEncoder is closed")

        frame = self._chunk.frames.add()
        if landmarks is None:
            frame.no_face.frame_number = frame_number
            self._prev_xy = None
            self._prev_z = None
        else:
            if len(landmarks) != self._landmark_count:
                raise ValueError(
                    f"Expected {self._landmark_count} landmarks, got "
                    f"{len(landmarks)} (frame {frame_number})"
                )
            xy, z, clamped = quantize_frame(landmarks)
            self.clamped_values += clamped

            # Delta iff we have a reference AND this is not a chunk start.
            if self._prev_xy is not None and self._frames_in_chunk > 0:
                frame.delta.frame_number = frame_number
                frame.delta.dxy.extend(
                    c - p for c, p in zip(xy, self._prev_xy)
                )
                frame.delta.dz.extend(
                    c - p for c, p in zip(z, self._prev_z)
                )
            else:
                frame.key.frame_number = frame_number
                frame.key.xy.extend(xy)
                frame.key.z.extend(z)
            self._prev_xy = xy
            self._prev_z = z

        self.frames_written += 1
        self._frames_in_chunk += 1
        if self._frames_in_chunk >= self._keyframe_interval:
            self._flush_chunk()

    def close(self) -> None:
        if self._closed:
            return
        self._flush_chunk()
        self.bytes_written = self._fh.tell()
        self._fh.close()
        self._closed = True
        logger.info(
            "landmark_encode_done frames=%d chunks=%d bytes=%d clamped=%d",
            self.frames_written, self.chunks_written,
            self.bytes_written, self.clamped_values,
        )

    # -- internals ------------------------------------------------------------
    def _flush_chunk(self) -> None:
        if not self._chunk.frames:
            return
        payload = self._chunk.SerializeToString()
        compressed = zlib.compress(payload, self._zlib_level)
        self._fh.write(_LEN.pack(len(compressed)))
        self._fh.write(_METHOD.pack(COMPRESSION_ZLIB))
        self._fh.write(compressed)
        self._fh.flush()
        self.chunks_written += 1
        self._chunk = pb.LandmarkChunk()
        self._frames_in_chunk = 0


class LandmarkDecoder:
    """Streaming reader for the ALTM landmark telemetry format.

    A truncated final chunk (crash mid-write) is skipped with a warning --
    recovery is the feature. Corruption inside a length-valid chunk raises
    LandmarkDecodeError.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self.chunks_read = 0
        with open(self._path, "rb") as fh:
            magic = fh.read(4)
            if magic != MAGIC:
                raise ValueError(
                    f"not an ALICE landmark telemetry file: {self._path}"
                )
            raw_len = fh.read(_LEN.size)
            if len(raw_len) < _LEN.size:
                raise ValueError(f"truncated header in {self._path}")
            (hlen,) = _LEN.unpack(raw_len)
            hb = fh.read(hlen)
            if len(hb) < hlen:
                raise ValueError(f"truncated header in {self._path}")
            self._header = pb.LandmarkStreamHeader.FromString(hb)
            if self._header.version != 1:
                raise ValueError(
                    f"unsupported landmark telemetry version "
                    f"{self._header.version} in {self._path}"
                )
            self._body_offset = fh.tell()

    @property
    def header(self) -> "pb.LandmarkStreamHeader":
        return self._header

    def frames(self) -> Iterator[DecodedFrame]:
        fps = self._header.source_fps
        chunk_index = 0
        with open(self._path, "rb") as fh:
            fh.seek(self._body_offset)
            while True:
                raw_len = fh.read(_LEN.size)
                if not raw_len:
                    return  # clean EOF
                if len(raw_len) < _LEN.size:
                    logger.warning(
                        "landmark_decode_truncated path=%s chunk=%d",
                        self._path, chunk_index,
                    )
                    return
                (clen,) = _LEN.unpack(raw_len)
                raw_method = fh.read(_METHOD.size)
                payload = fh.read(clen) if raw_method else b""
                if len(raw_method) < _METHOD.size or len(payload) < clen:
                    logger.warning(
                        "landmark_decode_truncated path=%s chunk=%d",
                        self._path, chunk_index,
                    )
                    return
                (method,) = _METHOD.unpack(raw_method)
                try:
                    if method == COMPRESSION_ZLIB:
                        payload = zlib.decompress(payload)
                    elif method != COMPRESSION_NONE:
                        raise LandmarkDecodeError(
                            f"unknown compression method {method} "
                            f"(chunk {chunk_index}) in {self._path}"
                        )
                    chunk = pb.LandmarkChunk.FromString(payload)
                except LandmarkDecodeError:
                    raise
                except Exception as exc:
                    raise LandmarkDecodeError(
                        f"corrupt chunk {chunk_index} in {self._path}: {exc}"
                    ) from exc

                # Delta reference resets at every chunk boundary by spec.
                prev_xy: Optional[list[int]] = None
                prev_z: Optional[list[int]] = None
                for f in chunk.frames:
                    kind = f.WhichOneof("kind")
                    if kind == "no_face":
                        prev_xy = None
                        prev_z = None
                        yield DecodedFrame(
                            f.no_face.frame_number,
                            f.no_face.frame_number / fps,
                            None,
                        )
                    elif kind == "key":
                        prev_xy = list(f.key.xy)
                        prev_z = list(f.key.z)
                        yield DecodedFrame(
                            f.key.frame_number,
                            f.key.frame_number / fps,
                            dequantize_frame(prev_xy, prev_z),
                        )
                    elif kind == "delta":
                        if prev_xy is None:
                            raise LandmarkDecodeError(
                                f"delta frame {f.delta.frame_number} with no "
                                f"prior keyframe (chunk {chunk_index}) in "
                                f"{self._path}"
                            )
                        prev_xy = [p + d for p, d in zip(prev_xy, f.delta.dxy)]
                        prev_z = [p + d for p, d in zip(prev_z, f.delta.dz)]
                        yield DecodedFrame(
                            f.delta.frame_number,
                            f.delta.frame_number / fps,
                            dequantize_frame(prev_xy, prev_z),
                        )
                    else:  # pragma: no cover - proto3 empty oneof
                        raise LandmarkDecodeError(
                            f"frame with no kind (chunk {chunk_index}) in "
                            f"{self._path}"
                        )
                chunk_index += 1
                self.chunks_read = chunk_index
