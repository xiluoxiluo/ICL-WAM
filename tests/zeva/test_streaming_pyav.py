"""Synthetic exact-frame regression for opt-in sequential PyAV decoder."""

from types import SimpleNamespace

import pytest
import torch

from fastwam.zeva.streaming_pyav import StreamingPyAVDecoder


class FakeVideoReader:
    instances = []
    videos = {}

    def __init__(self, video_path, stream):
        self.path = str(video_path)
        self.frames = self.videos[self.path]
        self.pos = 0
        self.container = SimpleNamespace(close=lambda: None)
        self.instances.append(self)

    def seek(self, time, keyframes_only=True):
        index = next((i for i, (t, _) in enumerate(self.frames) if t >= time), len(self.frames) - 1)
        self.pos = index // 4 * 4

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.frames):
            raise StopIteration
        pts, frame = self.frames[self.pos]
        self.pos += 1
        return {"pts": pts, "data": frame}


def baseline(path, times, tolerance):
    reader = FakeVideoReader(path, "video")
    reader.seek(min(times))
    loaded = []
    for row in reader:
        loaded.append(row)
        if row["pts"] >= max(times):
            break
    values = torch.tensor(times, dtype=torch.float32)
    timestamps = torch.tensor([row["pts"] for row in loaded], dtype=torch.float32)
    distance = torch.cdist(values[:, None], timestamps[:, None], p=1)
    minimum, index = distance.min(1)
    assert (minimum < tolerance).all()
    return torch.stack([loaded[int(i)]["data"] for i in index]).float() / 255


@pytest.fixture
def fake_video(monkeypatch):
    import torchvision

    FakeVideoReader.instances = []
    FakeVideoReader.videos = {
        "video-a": [(i / 10, torch.full((3, 4, 4), i, dtype=torch.uint8)) for i in range(100)],
        "video-b": [(i / 10, torch.full((3, 4, 4), i * 2, dtype=torch.uint8)) for i in range(100)],
    }
    monkeypatch.setattr(torchvision.io, "VideoReader", FakeVideoReader)
    monkeypatch.setattr(torchvision, "set_video_backend", lambda backend: None)


def test_adjacent_h32_windows_match_reference_and_reuse_reader(fake_video):
    original = [baseline("video-a", [j / 10 for j in range(i, i + 9)], .05)
                for i in range(0, 64, 8)]
    FakeVideoReader.instances = []
    decoder = StreamingPyAVDecoder()
    outputs = [decoder.decode("video-a", [j / 10 for j in range(i, i + 9)], .05)
               for i in range(0, 64, 8)]
    for actual, expected in zip(outputs, original, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert decoder.new_readers == 1 and decoder.reseeks == 0
    decoder.close()


def test_backwards_seek_and_lru_eviction(fake_video):
    decoder = StreamingPyAVDecoder(max_open=1)
    for name, start in [("video-a", 10), ("video-a", 18), ("video-a", 4), ("video-b", 5)]:
        times = [j / 10 for j in range(start, start + 9)]
        torch.testing.assert_close(decoder.decode(name, times, .05), baseline(name, times, .05), rtol=0, atol=0)
    assert decoder.reseeks >= 1 and decoder.new_readers == 2
    decoder.close()


def test_variable_frame_times_match_reference(fake_video):
    import random
    rng = random.Random(123)
    t = 0.0
    timestamps = []
    for i in range(100):
        t += rng.choice([.07, .09, .10, .11])
        timestamps.append(t)
    FakeVideoReader.videos["video-c"] = [
        (value, torch.full((3, 4, 4), index, dtype=torch.uint8))
        for index, value in enumerate(timestamps)
    ]
    decoder = StreamingPyAVDecoder()
    for start in range(0, 84, 7):
        times = timestamps[start : start + 8]
        torch.testing.assert_close(decoder.decode("video-c", times, .08),
                                   baseline("video-c", times, .08), rtol=0, atol=0)


def test_invalid_decoder_settings():
    with pytest.raises(ValueError):
        StreamingPyAVDecoder(max_open=0)
    with pytest.raises(ValueError):
        StreamingPyAVDecoder(verify_every=-1)
    with pytest.raises(ValueError):
        StreamingPyAVDecoder().decode("unknown", [], 1)
