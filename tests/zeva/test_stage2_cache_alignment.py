import torch
from torch.utils.data import Dataset

from fastwam.datasets.zeva_robotwin_dataset import ZevaStage2Dataset
from fastwam.zeva import CacheManifest, PhaseEffectCache, save_phase_effect_cache


class _Windows(Dataset):
    def __init__(self):
        self.samples = []
        for source_index in range(3):
            self.samples.append({
                "video": torch.zeros(3, 9, 8, 8),
                "action": torch.zeros(32, 14),
                "proprio": torch.zeros(32, 14),
                "prompt": "task",
                "image_is_pad": torch.zeros(9, dtype=torch.bool),
                "action_is_pad": torch.zeros(32, dtype=torch.bool),
                "dataset_index": source_index,
                "episode_id": "episode-0",
                "episode_step": source_index * 32,
                "task_id": 0,
                "task_name": "task",
                "context": torch.zeros(8, 4),
                "context_mask": torch.ones(8, dtype=torch.bool),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_stage2_uses_sparse_cache_window_indices(tmp_path):
    root = tmp_path / "cache"
    manifest = CacheManifest()
    records = []
    for window_index, phase_value in ((0, 1.0), (2, 2.0)):
        for effect_index, transition_index in enumerate((0, 4)):
            records.append({
                "episode_id": "episode-0",
                "task_id": 0,
                "window_index": window_index,
                "episode_step": window_index * 32,
                "transition_index": transition_index,
                "effect_index": effect_index,
                "phase_pre": torch.full((128,), phase_value),
                "phase_post": torch.full((128,), phase_value),
                "effect": torch.full((128,), phase_value),
                "valid": True,
            })
    save_phase_effect_cache(root, records, manifest, shard_size=16)
    cache = PhaseEffectCache.load(root)
    dataset = ZevaStage2Dataset(_Windows(), cache)

    assert len(dataset) == 2
    first, second = dataset[0], dataset[1]
    torch.testing.assert_close(first["phase"], torch.ones(128))
    torch.testing.assert_close(second["phase"], torch.full((128,), 2.0))
    assert bool(second["bit_mask"].any())
