"""06 first-frame fast path: restore data state and preserve output values.

These tests require no GPUs or source video files.
"""

from types import SimpleNamespace

import pytest
import torch

from scripts.build_zeva_behavior_bank_optimized import (
    _InitialReadoutDataset,
    _only_initial_video_timestamps,
    _readout_tensor_collate,
)


class _Inner:
    def __init__(self):
        self.meta = SimpleNamespace(video_keys=["head", "left", "right"])
        self.delta_indices = {
            cam: list(range(33)) for cam in self.meta.video_keys
        }


class _Base:
    def __init__(self):
        self.inner = _Inner()
        self.processor = SimpleNamespace(num_image_steps=33)
        self.lerobot_dataset = SimpleNamespace(
            multi_dataset=SimpleNamespace(_datasets=[self.inner]),
            processor=self.processor,
        )
        self.observed = []

    def _get(self, index, *, initial_image_only=False):
        self.observed.append((
            index, initial_image_only, self.processor.num_image_steps,
            [len(v) for v in self.inner.delta_indices.values()],
        ))
        assert initial_image_only
        assert self.processor.num_image_steps == 1
        assert [len(v) for v in self.inner.delta_indices.values()] == [1, 1, 1]
        return {
            "dataset_index": int(index),
            "episode_id": f"test::episode-{index}",
            "episode_step": 0,
            "task_id": "task-1",
            "prompt": "Pick up block",
            "video": torch.ones(3, 1, 8, 8),
            "context": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "context_mask": torch.tensor([True, False, True]),
        }


class _Wrapped:
    def __init__(self, base):
        self.base_dataset = base

    def __getitem__(self, index):
        # Synthetic reference receives nine RGB observations but only the
        # initial one is consumed for the readout.
        return {
            "dataset_index": index,
            "episode": SimpleNamespace(
                episode_id=f"test::episode-{index}",
                episode_step=0,
                task_id="task-1",
                instruction="Pick up block",
            ),
            "video": torch.ones(3, 9, 8, 8),
            "context": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "context_mask": torch.tensor([True, False, True]),
        }


def test_temporary_video_window_restored_on_success_and_error():
    base = _Base()
    for raises in [False, True]:
        with pytest.raises(RuntimeError) if raises else _does_not_raise():
            with _only_initial_video_timestamps(base):
                assert base.processor.num_image_steps == 1
                assert len(base.inner.delta_indices["head"]) == 1
                if raises:
                    raise RuntimeError("intentional")
        assert base.processor.num_image_steps == 33
        assert len(base.inner.delta_indices["head"]) == 33


class _does_not_raise:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def test_fast_frame_values_and_immutable_episode_metadata_match_baseline():
    base = _Base()
    dataset = _Wrapped(base)
    plan = SimpleNamespace(dataset_start=17, episode_id="test::episode-17", task_id="task-1")
    original = _InitialReadoutDataset(dataset, [plan])
    optimized = _InitialReadoutDataset(dataset, [plan], first_frame_only=True)
    fast, baseline = optimized[0], original[0]
    for name in ("episode_id", "task_id", "dataset_start", "instruction"):
        assert fast[name] == baseline[name]
    for name in ("image", "context", "context_mask"):
        assert torch.equal(fast[name], baseline[name])
    assert base.observed == [(17, True, 1, [1, 1, 1])]
    assert base.processor.num_image_steps == 33
    assert all(len(x) == 33 for x in base.inner.delta_indices.values())


def test_worker_batch_preserves_order_and_shapes():
    base = _Base()
    wrapped = _Wrapped(base)
    plans = [
        SimpleNamespace(dataset_start=i, episode_id=f"test::episode-{i}", task_id="task-1")
        for i in (2, 5, 7)
    ]
    source = _InitialReadoutDataset(wrapped, plans, first_frame_only=True)
    merged = _readout_tensor_collate([source[i] for i in range(len(source))])
    assert merged["images"].shape == (3, 3, 8, 8)
    assert merged["contexts"].shape == (3, 3, 4)
    assert merged["masks"].shape == (3, 3)
    assert merged["episode_ids"] == [f"test::episode-{i}" for i in (2, 5, 7)]
    assert merged["task_ids"] == ["task-1"] * 3
    assert all(name == "Pick up block" for name in merged["instructions"])


def test_short_or_nonzero_video_offsets_rejected_without_persistent_mutation():
    base = _Base()
    base.inner.delta_indices["left"][0] = 1
    with pytest.raises(ValueError, match="first video offset"):
        with _only_initial_video_timestamps(base):
            pass
    assert base.processor.num_image_steps == 33
    assert all(len(x) == 33 for x in base.inner.delta_indices.values())
