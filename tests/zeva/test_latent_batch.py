"""Numerical/ordering invariants for worker-side CTE batching."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fastwam.zeva.latent_batch import collate_latent_windows


class _Windows(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return SimpleNamespace(
            frames=torch.full((9, 3, 8, 8), float(index)),
            transition_actions=torch.full((8, 4, 14), float(index) / 2),
            frame_valid=torch.tensor([True] * 8 + [index % 2 == 0]),
            transition_valid=torch.ones((8, 4), dtype=torch.bool),
        )


@pytest.mark.parametrize("workers", [0, 2])
def test_batches_keep_frames_actions_masks_and_order(workers):
    ds = _Windows()
    batches = list(DataLoader(ds, batch_size=4, num_workers=workers,
                              collate_fn=collate_latent_windows, pin_memory=False))
    assert [len(batch["frames"]) for batch in batches] == [4, 4, 2]
    assert batches[0]["frames"].shape == (4, 9, 3, 8, 8)
    assert batches[0]["actions"].shape == (4, 8, 4, 14)
    for start, batch in zip((0, 4, 8), batches):
        items = [ds[i] for i in range(start, min(start + 4, len(ds)))]
        for key, attr in (("frames", "frames"), ("actions", "transition_actions"),
                          ("frame_valid", "frame_valid"), ("transition_valid", "transition_valid")):
            torch.testing.assert_close(batch[key], torch.stack([getattr(item, attr) for item in items]), rtol=0, atol=0)


def test_empty_batch_rejected():
    with pytest.raises(ValueError, match="at least one"):
        collate_latent_windows([])
