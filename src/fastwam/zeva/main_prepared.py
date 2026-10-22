"""Episode-disjoint Stage-2 views over one observed-effect prepared cache.

The source cache covers all successful query episodes. Only training episodes
may supply retrieved demonstrations or update the model. These views preserve
the original query/window index, and never decode RGB just to find the split.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from .main_data import validate_main_artifact, validate_main_episode_ids


class MainPreparedSplitDataset(Dataset):
    def __init__(self, source: Dataset, indices: np.ndarray, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("main prepared split must be train or validation")
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        if (self.indices.ndim != 1 or not len(self.indices)
                or np.any(self.indices < 0) or np.any(self.indices >= len(source))
                or np.any(self.indices[1:] <= self.indices[:-1])):
            raise ValueError("main split requires nonempty, ordered, unique source query indices")
        self.indices.flags.writeable = False
        self.split = split
        self.split_identity = {
            "split": split,
            "query_indices_sha256": hashlib.sha256(self.indices.astype("<i8").tobytes()).hexdigest(),
            "episode_split_sha256": self.pim.manifest["success_data_identity"]["split_sha256"],
        }

    @property
    def pim(self):
        return self.source.pim

    @property
    def static_context(self):
        return self.source.static_context

    @property
    def history_schedule(self):
        return self.source.history_schedule

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.source[int(self.indices[index])]


def split_main_prepared_dataset(source: Dataset) -> tuple[MainPreparedSplitDataset, MainPreparedSplitDataset]:
    """Read only mmap episode indices; reject missing or extraneous episodes."""
    cache = source.pim
    if cache._cross_attempt is not None:
        raise ValueError("main observed splits do not accept Ego cross-attempt caches")
    validate_main_artifact(cache.manifest)
    if cache.manifest["data_role"] != "all":
        raise ValueError("main prepared query cache must contain both train and validation episodes")
    validate_main_episode_ids(cache.manifest, cache.episode_ids)
    splits = cache.manifest["episode_splits"]
    train_ids, validation_ids = set(splits["train"]), set(splits["validation"])
    role = np.asarray([0 if identity in train_ids else 1 if identity in validation_ids else -1
                       for identity in cache.episode_ids], dtype=np.int8)
    if np.any(role < 0):
        raise ValueError("main prepared query belongs to neither episode split")
    pieces: list[list[np.ndarray]] = [[], []]
    coverage = np.zeros(len(role), dtype=np.int64)
    for shard in cache.shards:
        values = np.load(cache.root / shard.path / "episode_index.npy", mmap_mode="r")
        if values.dtype != np.int32 or values.shape != (shard.size,):
            raise ValueError("main prepared episode_index shard has an invalid dtype or shape")
        if np.any(values < 0) or np.any(values >= len(role)):
            raise ValueError("main prepared episode_index is outside the declared episode list")
        coverage += np.bincount(values, minlength=len(role))
        membership = role[values]
        for split_index in (0, 1):
            pieces[split_index].append(np.flatnonzero(membership == split_index) + shard.global_start)
    if np.any(coverage == 0):
        missing = [cache.episode_ids[index] for index in np.flatnonzero(coverage == 0)]
        raise ValueError(f"main prepared cache has no query windows for declared episodes: {missing[:5]}")
    train_indices, validation_indices = (np.concatenate(part) for part in pieces)
    if len(train_indices) + len(validation_indices) != len(source):
        raise ValueError("main prepared split query coverage mismatch")
    return (MainPreparedSplitDataset(source, train_indices, "train"),
            MainPreparedSplitDataset(source, validation_indices, "validation"))
