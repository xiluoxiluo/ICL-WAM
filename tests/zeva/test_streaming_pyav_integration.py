"""The optional sequential decoder stays private to CTE-cache workers."""

from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.precompute_zeva_cte_latents import _WindowSourceDataset
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset


def test_install_decoder_is_lazy_idempotent_and_disabled_by_default():
    inner = SimpleNamespace()
    base = SimpleNamespace(
        lerobot_dataset=SimpleNamespace(multi_dataset=SimpleNamespace(_datasets=[inner]))
    )
    wrapped = SimpleNamespace(base_dataset=base)
    original = _WindowSourceDataset(wrapped, [])
    original._enable_worker_decoder()
    assert not hasattr(inner, "_cte_streaming_pyav_decoder")

    optimized = _WindowSourceDataset(
        wrapped, [], streaming_pyav=True, streaming_verify_every=7,
        streaming_max_open=5,
    )
    optimized._enable_worker_decoder()
    reader = inner._cte_streaming_pyav_decoder
    assert reader.verify_every == 7
    assert reader.max_open == 5
    optimized._enable_worker_decoder()
    assert inner._cte_streaming_pyav_decoder is reader
    reader.close()


def test_generic_lerobot_path_is_not_changed_unless_decoder_installed(monkeypatch):
    import fastwam.datasets.lerobot.lerobot.lerobot_dataset as implementation

    observations = torch.ones((2, 3, 4, 4), dtype=torch.float32)
    calls = []
    monkeypatch.setattr(
        implementation,
        "decode_video_frames",
        lambda path, times, tol, backend: (
            calls.append(("legacy", str(path), list(times), backend)) or observations
        ),
    )
    meta = SimpleNamespace(get_video_file_path=lambda ep, camera: f"{ep}/{camera}.mp4")
    owner = SimpleNamespace(root=Path("/synthetic"), meta=meta, tolerance_s=.05, video_backend="pyav")
    original = LeRobotDataset._query_videos(owner, {"head":[0.0, 0.1]}, 2)
    assert torch.equal(original["head"], observations)
    assert calls[0] == ("legacy", "/synthetic/2/head.mp4", [0.0, 0.1], "pyav")

    class Decoder:
        def decode(self, path, times, tol, backend):
            calls.append(("stream", str(path), list(times), backend))
            return observations * 2

    owner._cte_streaming_pyav_decoder = Decoder()
    streamed = LeRobotDataset._query_videos(owner, {"head":[0.0, 0.1]}, 2)
    assert torch.equal(streamed["head"], observations * 2)
    assert calls[-1][0] == "stream"
