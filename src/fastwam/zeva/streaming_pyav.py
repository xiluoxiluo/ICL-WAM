"""Opt-in worker-local sequential PyAV decoding for CTE latent windows.

The same timestamp-distance/tolerance rule as the original LeRobot decoder is
used. Readers and a three-frame lookbehind persist within a DataLoader worker.
This module never changes preprocessing, VAE weights, or mmap format.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class _VideoCursor:
    reader: Any
    tail: deque = field(default_factory=lambda: deque(maxlen=3))

    def close(self) -> None:
        container = getattr(self.reader, "container", None)
        if container is not None:
            container.close()


class StreamingPyAVDecoder:
    """Reuse video readers for monotonic H32 windows in one worker.

    Out-of-order timestamps force a preceding-keyframe seek. Optional periodic
    checks run the original reference decoder and fail closed on ANY pixel
    mismatch, before writing the affected batch.
    """

    def __init__(self, *, max_open: int = 3, verify_every: int = 0) -> None:
        if type(max_open) is not int or max_open < 1:
            raise ValueError("max_open must be a positive integer")
        if type(verify_every) is not int or verify_every < 0:
            raise ValueError("verify_every must be a nonnegative integer")
        self.max_open = max_open
        self.verify_every = verify_every
        self._readers: OrderedDict[str, _VideoCursor] = OrderedDict()
        self.calls = 0
        self.new_readers = 0
        self.reseeks = 0

    def close(self) -> None:
        for cursor in self._readers.values():
            cursor.close()
        self._readers.clear()

    def _reader(self, path: str, first_timestamp: float) -> _VideoCursor:
        import torchvision

        cursor = self._readers.get(path)
        if cursor is None:
            torchvision.set_video_backend("pyav")
            reader = torchvision.io.VideoReader(path, "video")
            cursor = _VideoCursor(reader=reader)
            self._readers[path] = cursor
            self.new_readers += 1
            if len(self._readers) > self.max_open:
                _old_path, old = self._readers.popitem(last=False)
                old.close()
            reader.seek(first_timestamp, keyframes_only=True)
            return cursor
        self._readers.move_to_end(path)
        # The next requested boundary is no longer in the short lookbehind.
        if cursor.tail and float(cursor.tail[0][0]) > first_timestamp:
            cursor.reader.seek(first_timestamp, keyframes_only=True)
            cursor.tail.clear()
            self.reseeks += 1
        return cursor

    def decode(
        self, video_path: Path | str, timestamps: list[float],
        tolerance_s: float, backend: str | None = "pyav",
    ) -> torch.Tensor:
        if backend != "pyav":
            raise ValueError("sequential CTE decoder requires explicit pyav")
        if not timestamps:
            raise ValueError("at least one timestamp is required")
        if tolerance_s is None or float(tolerance_s) <= 0:
            raise ValueError("video timestamp tolerance must be positive")
        query_first, query_last = min(timestamps), max(timestamps)
        path = str(video_path)
        cursor = self._reader(path, query_first)
        available = list(cursor.tail)

        if not available or available[-1][0] < query_last:
            for frame in cursor.reader:
                current = float(frame["pts"])
                available.append((current, frame["data"]))
                if current >= query_last:
                    break
        if not available:
            raise RuntimeError(f"video has no decodable frames: {path}")

        query_pts = torch.tensor(timestamps, dtype=torch.float32)
        loaded_pts = torch.tensor([item[0] for item in available], dtype=torch.float32)
        distances = torch.cdist(query_pts[:, None], loaded_pts[:, None], p=1)
        minimum, indices = distances.min(1)
        if not bool((minimum < tolerance_s).all()):
            raise AssertionError(
                f"streaming decoder timestamp tolerance failed for {path}: "
                f"minimum={minimum.tolist()} tolerance={tolerance_s}"
            )
        output = torch.stack([available[int(i)][1] for i in indices]).to(torch.float32) / 255
        cursor.tail = deque(available[-3:], maxlen=3)
        self.calls += 1
        if self.verify_every and self.calls % self.verify_every == 0:
            from fastwam.datasets.lerobot.lerobot.datasets.video_utils import (
                decode_video_frames_torchvision,
            )

            reference = decode_video_frames_torchvision(
                path, list(timestamps), tolerance_s, backend="pyav",
            )
            if not torch.equal(output, reference):
                raise RuntimeError(
                    f"Streaming PyAV differs from reference decoder on {path}; "
                    "stop this job and resume with LATENT_STREAMING_PYAV=0."
                )
        return output
