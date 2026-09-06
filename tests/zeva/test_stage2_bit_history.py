import torch
from torch.utils.data import Dataset

from fastwam.datasets.zeva_robotwin_dataset import ZevaStage2Dataset
from fastwam.zeva import CacheManifest, PhaseEffectCache, save_phase_effect_cache


class _TinyWindows(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "video": torch.zeros(3, 9, 8, 8),
            "action": torch.zeros(32, 14),
            "proprio": torch.zeros(32, 14),
            "prompt": "task",
            "image_is_pad": torch.zeros(9, dtype=torch.bool),
            "action_is_pad": torch.zeros(32, dtype=torch.bool),
            "dataset_index": index,
            "episode_id": "episode-0",
            "episode_step": index * 32,
            "task_id": 0,
            "task_name": "task",
            "context": torch.zeros(8, 4),
            "context_mask": torch.ones(8, dtype=torch.bool),
        }


def test_stage2_bit_prefix_is_built_once_and_causal(tmp_path):
    root = tmp_path / "cache"
    records = []
    for window_index in range(2):
        for effect_index, transition_index in enumerate((0, 4)):
            value = float(window_index * 10 + effect_index + 1)
            records.append({
                "episode_id": "episode-0", "task_id": 0,
                "window_index": window_index, "episode_step": window_index * 32,
                "transition_index": transition_index,
                "effect_index": effect_index,
                "phase_pre": torch.ones(128), "phase_post": torch.ones(128),
                "effect": torch.full((128,), value), "valid": True,
            })
    save_phase_effect_cache(root, records, CacheManifest(), shard_size=16)
    dataset = ZevaStage2Dataset(_TinyWindows(), PhaseEffectCache.load(root), bit_size=4)
    assert not bool(dataset[0]["bit_mask"].any())
    assert int(dataset[1]["bit_mask"].sum()) == 2
    torch.testing.assert_close(dataset[1]["bit_effects"][-1], torch.full((128,), 2.0))
