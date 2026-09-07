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


def test_stage2_reads_v4_phase_queries_and_effect_rows(tmp_path):
    root = tmp_path / "cache-v4"
    manifest = CacheManifest(
        schema_version="zeva_fastwam_robotwin_cache_v4",
        history_semantics="full_episode_prefix",
        query_step_unit="raw_action_step",
    )
    records = [
        {
            "record_type": "phase_query",
            "episode_id": "episode-0",
            "task_id": 0,
            "window_index": 0,
            "episode_step": 0,
            "raw_step": 0,
            "start_raw_step": 0,
            "end_raw_step": 0,
            "transition_index": 0,
            "effect_index": 0,
            "phase_pre": torch.ones(128),
            "phase_post": torch.ones(128),
            "effect": torch.zeros(128),
            "valid": True,
        },
        {
            "record_type": "effect",
            "episode_id": "episode-0",
            "task_id": 0,
            "window_index": None,
            "episode_step": 0,
            "raw_step": 16,
            "start_raw_step": 0,
            "end_raw_step": 16,
            "transition_index": 0,
            "effect_index": 0,
            "phase_pre": torch.ones(128),
            "phase_post": torch.ones(128),
            "effect": torch.full((128,), 2.0),
            "valid": True,
        },
    ]
    save_phase_effect_cache(root, records, manifest, shard_size=16)
    dataset = ZevaStage2Dataset(_Windows(), PhaseEffectCache.load(root), bit_size=4)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["phase"].shape == (128,)
    assert not bool(sample["bit_mask"].any())


def test_stage2_v4_effect_at_query_boundary_is_visible(tmp_path):
    class _BoundaryWindow(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {
                "video": torch.zeros(3, 9, 8, 8),
                "action": torch.zeros(32, 14),
                "proprio": torch.zeros(32, 14),
                "prompt": "task",
                "image_is_pad": torch.zeros(9, dtype=torch.bool),
                "action_is_pad": torch.zeros(32, dtype=torch.bool),
                "dataset_index": 0,
                "episode_id": "episode-0",
                "episode_step": 16,
                "task_id": 0,
                "task_name": "task",
            }

    root = tmp_path / "cache-v4-boundary"
    manifest = CacheManifest(
        schema_version="zeva_fastwam_robotwin_cache_v4",
        history_semantics="full_episode_prefix",
        query_step_unit="raw_action_step",
    )
    records = [
        {
            "record_type": "effect",
            "episode_id": "episode-0",
            "task_id": 0,
            "window_index": None,
            "episode_step": 0,
            "raw_step": 16,
            "start_raw_step": 0,
            "end_raw_step": 16,
            "transition_index": 0,
            "effect_index": 0,
            "phase_pre": torch.ones(128),
            "phase_post": torch.ones(128),
            "effect": torch.full((128,), 3.0),
            "valid": True,
        },
        {
            "record_type": "phase_query",
            "episode_id": "episode-0",
            "task_id": 0,
            "window_index": 0,
            "episode_step": 16,
            "raw_step": 16,
            "start_raw_step": 16,
            "end_raw_step": 16,
            "transition_index": 4,
            "effect_index": 0,
            "phase_pre": torch.ones(128),
            "phase_post": torch.ones(128),
            "effect": torch.zeros(128),
            "valid": True,
        },
    ]
    save_phase_effect_cache(root, records, manifest)
    sample = ZevaStage2Dataset(_BoundaryWindow(), PhaseEffectCache.load(root))[0]
    assert bool(sample["bit_mask"].any())
    torch.testing.assert_close(sample["bit_effects"][-1], torch.full((128,), 3.0))
