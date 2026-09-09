from pathlib import Path
import runpy

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fastwam.datasets.zeva_robotwin_dataset import ZevaStage2Dataset
from fastwam.zeva import (
    CacheManifest, PhaseEffectCache, TaskContextBank,
    retrieve_task_context, save_phase_effect_cache,
)


def _bank() -> TaskContextBank:
    return TaskContextBank(
        [
            {"retrieval_key": [1.0, 0.0, 0.0, 0.0], "behavior_value": [1.0, 2.0, 3.0], "task_id": "a"},
            {"retrieval_key": [0.0, 1.0, 0.0, 0.0], "behavior_value": [4.0, 5.0, 6.0], "task_id": "b"},
        ],
        key_dim=4,
        value_dim=3,
    )


def test_task_context_bank_retrieves_single_prototype():
    result = _bank().retrieve(torch.tensor([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert result.indices.tolist() == [0]
    torch.testing.assert_close(result.values, torch.tensor([1.0, 2.0, 3.0]))
    assert result.sources[0]["top_k"][0]["task_id"] == "a"


def test_task_context_bank_softmax_weights_use_cosine_scores():
    bank = TaskContextBank(_bank().entries, key_dim=4, value_dim=3, temperature=1.0)
    result = bank.retrieve(torch.tensor([3.0, 4.0, 0.0, 0.0]), top_k=2)
    assert result.indices.tolist() == [1, 0]
    torch.testing.assert_close(result.scores, torch.tensor([0.8, 0.6]))
    weight_b = 1.0 / (1.0 + torch.exp(torch.tensor(-0.2)))
    expected = weight_b * torch.tensor([4.0, 5.0, 6.0]) + (1.0 - weight_b) * torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(result.values, expected)


def test_task_context_bank_uses_fastwam_context_query_and_roundtrips(tmp_path):
    bank = _bank()
    path = tmp_path / "task_context.pt"
    bank.save(path, metadata={"source": "test"})
    loaded = TaskContextBank.load(path, expected_key_dim=4, expected_value_dim=3)
    values, result = retrieve_task_context(
        torch.tensor([[[2.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0]]]),
        torch.tensor([[True, False]]),
        loaded,
        output_dim=3,
        top_k=1,
    )
    assert values.shape == (1, 3)
    assert result.indices.tolist() == [[0]]
    torch.testing.assert_close(values, torch.tensor([[1.0, 2.0, 3.0]]))
    assert loaded.temperature == bank.temperature


def test_task_context_bank_rejects_dimension_mismatch(tmp_path):
    path = tmp_path / "task_context.pt"
    _bank().save(path)
    with pytest.raises(ValueError, match="key_dim"):
        TaskContextBank.load(path, expected_key_dim=5)
    with pytest.raises(ValueError, match="value_dim"):
        TaskContextBank.load(path, expected_value_dim=4)


def test_task_context_bank_returns_one_source_group_per_batch_item():
    result = _bank().retrieve(torch.eye(4)[:2], top_k=1)
    assert result.values.shape == (2, 3)
    assert len(result.sources) == 2
    assert [source["top_k"][0]["task_id"] for source in result.sources] == ["a", "b"]


def test_task_context_bank_owns_frozen_copies():
    key = torch.tensor([1.0, 0.0], requires_grad=True)
    value = torch.tensor([2.0, 3.0], requires_grad=True)
    bank = TaskContextBank([{"retrieval_key": key, "behavior_value": value}], key_dim=2, value_dim=2)
    with torch.no_grad():
        key.fill_(0.0)
        value.fill_(0.0)
        bank.entries[0]["behavior_value"].fill_(10.0)
    result = bank.retrieve(torch.tensor([1.0, 0.0], requires_grad=True))
    torch.testing.assert_close(result.values, torch.tensor([2.0, 3.0]))
    assert not result.values.requires_grad
    assert not bank.entries[0]["behavior_value"].requires_grad


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_task_context_bank_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        TaskContextBank(_bank().entries, key_dim=4, value_dim=3, temperature=temperature)


@pytest.mark.parametrize("top_k", [0, 3])
def test_task_context_bank_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        _bank().retrieve(torch.ones(4), top_k=top_k)


def test_task_context_bank_validates_artifacts_and_entries(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save({"format": "cosmos_task_bank"}, path)
    with pytest.raises(ValueError, match="format"):
        TaskContextBank.load(path)
    with pytest.raises(ValueError, match="at least one"):
        TaskContextBank([], key_dim=4, value_dim=3)
    entries = list(_bank().entries)
    entries[0]["behavior_value"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        TaskContextBank(entries, key_dim=4, value_dim=3)


class _Windows(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        key = torch.tensor([2.0, 0.0, 0.0, 0.0]) if index < 2 else torch.tensor([0.0, 2.0, 0.0, 0.0])
        return {
            "video": torch.zeros(3, 9, 8, 8),
            "action": torch.zeros(32, 14),
            "prompt": "a" if index < 2 else "b",
            "dataset_index": index,
            "episode_id": f"episode-{index}",
            "episode_step": 0,
            "task_id": 0 if index < 2 else 1,
            "context": torch.stack([key * (index + 1), torch.full((4,), 100.0)]),
            "context_mask": torch.tensor([True, False]),
        }


def test_builder_averages_unbatched_contexts_and_preserves_groups():
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_zeva_task_context_bank.py"
    build_bank = runpy.run_path(str(script))["build_bank"]
    bank = build_bank(_Windows(), key_dim=4, value_dim=2, temperature=0.5)
    assert len(bank) == 2
    assert [entry["num_samples"] for entry in bank.entries] == [2, 1]
    assert [entry["instruction"] for entry in bank.entries] == ["a", "b"]
    torch.testing.assert_close(bank.entries[0]["retrieval_key"], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(bank.entries[0]["behavior_value"], torch.tensor([1.5, 0.0]))
    assert bank.temperature == 0.5


def test_stage2_collation_matches_deployment_retrieval(tmp_path):
    root = tmp_path / "cache"
    records = [{
        "episode_id": f"episode-{index}", "task_id": 0 if index < 2 else 1,
        "window_index": index, "episode_step": 0,
        "transition_index": 0, "effect_index": 0,
        "phase_pre": torch.ones(128), "phase_post": torch.ones(128),
        "effect": torch.ones(128), "valid": True,
    } for index in range(3)]
    save_phase_effect_cache(root, records, CacheManifest())
    cache = PhaseEffectCache.load(root)
    bank = _bank()
    dataset = ZevaStage2Dataset(_Windows(), cache, task_context_bank=bank, task_context_top_k=2)
    batch = next(iter(DataLoader(dataset, batch_size=3)))
    expected, retrieval = retrieve_task_context(
        batch["context"], batch["context_mask"], bank, output_dim=3, top_k=2,
    )
    assert batch["task_context"].shape == (3, 3)
    torch.testing.assert_close(batch["task_context"], expected)
    torch.testing.assert_close(batch["task_context_retrieval_indices"], retrieval.indices)
    torch.testing.assert_close(batch["task_context_retrieval_scores"], retrieval.scores)
    assert "task_context" not in ZevaStage2Dataset(_Windows(), cache)[0]
    with pytest.raises(ValueError, match="bank size"):
        ZevaStage2Dataset(_Windows(), cache, task_context_bank=bank, task_context_top_k=3)
