"""Task-memory holdout and initial-query artifact contracts."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.ego_task_memory import (
    build_ego_static_context_cache, load_ego_query_readouts,
    ordered_episode_ids, pool_train_behavior_entries, save_ego_task_memory,
)
from fastwam.zeva.prepared_stage2 import StaticEpisodeContextCache
from fastwam.zeva.static_task_context import (
    HEAD_FORMAT, READOUT_KIND, StaticTaskContextRetrievalConfig,
    StaticTaskContextRetrievalHead, StaticTaskContextRetriever, load_readout_cache,
)
from fastwam.zeva.task_context import TaskContextBank


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Cache:
    def __init__(self):
        self.manifest = SimpleNamespace(retrieval_dim=4, global_dim=6, executed_horizon=24)
        self.splits = {"train": [f"episode-{i}" for i in range(4)], "validation": ["episode-4", "episode-5"]}
        self.episodes = {f"episode-{i}": {"dataset_start": i * 49, "episode_length": 49,
                                        "task_id": str(i % 2), "split": "train" if i < 4 else "validation"}
                         for i in range(6)}
        self.rows = [{"episode_id": identity, "task_id": meta["task_id"], "raw_step": step,
                      "split": meta["split"]} for identity, meta in self.episodes.items() for step in (0, 24)]
        self.read = []

    def get(self, index):
        row = self.rows[index]
        self.read.append(row["episode_id"])
        number = int(row["episode_id"].split("-")[-1])
        return {**row, "retrieval": torch.tensor([1., float(number), float(row["raw_step"]), 1.]),
                "global": torch.full((6,), float(number + row["raw_step"] / 24))}


def artifacts(tmp_path):
    cache = Cache()
    generator = torch.Generator().manual_seed(10)
    readouts = {identity: torch.randn(7, generator=generator) for identity in ordered_episode_ids(cache)}
    paths = {"bank_path": tmp_path / "bank.pt", "train_readout_path": tmp_path / "train.pt",
             "query_readout_path": tmp_path / "queries.pt"}
    metadata = {"base_checkpoint_sha256": "base", "cte_checkpoint_sha256": "cte",
                "dataset_stats_sha256": "stats", "semantic_task_map_sha256": "semantic",
                "phase_cache_manifest_sha256": "boundaries", "episode_splits_sha256": "splits",
                "video_size": [384, 320], "context_len": 128}
    save_ego_task_memory(cache, readouts, metadata=metadata, **paths)
    return cache, paths, metadata


def head_checkpoint(path, paths, cache):
    torch.manual_seed(5)
    cfg = StaticTaskContextRetrievalConfig(input_dim=7, hidden_dim=8, output_dim=4, dropout=0)
    head = StaticTaskContextRetrievalHead(cfg)
    payload = {"format": HEAD_FORMAT, "readout_kind": READOUT_KIND,
               "bank_sha256": checkpoint_sha256(paths["bank_path"]),
               "readout_cache_sha256": checkpoint_sha256(paths["train_readout_path"]),
               "step": 1, "config": cfg.as_dict(), "model": head.state_dict(),
               "train_episode_ids": cache.splits["train"][:2],
               "validation_episode_ids": cache.splits["train"][2:]}
    torch.save(payload, path)
    return payload


def test_bank_pools_training_causal_states_without_reading_heldout_features():
    cache = Cache()
    entries = pool_train_behavior_entries(cache)
    assert [entry["episode_id"] for entry in entries] == cache.splits["train"]
    assert set(cache.read) == set(cache.splits["train"])
    for index, entry in enumerate(entries):
        torch.testing.assert_close(entry["behavior_value"], F.normalize(torch.full((6,), index + .5), dim=0))
        torch.testing.assert_close(entry["retrieval_key"], F.normalize(torch.tensor([1., float(index), 12., 1.]), dim=0))
        assert entry["num_decision_boundaries"] == 2


def test_bank_excludes_partial_final_decision_that_has_no_full_stage1_target():
    cache = Cache()
    cache.episodes["episode-0"]["episode_length"] = 50
    cache.rows.append({"episode_id": "episode-0", "task_id": "0", "raw_step": 48, "split": "train"})
    entry = pool_train_behavior_entries(cache)[0]
    assert entry["num_decision_boundaries"] == 2
    torch.testing.assert_close(entry["retrieval_key"], F.normalize(torch.tensor([1., 0., 12., 1.]), dim=0))


def test_bank_and_head_readouts_are_train_only_but_queries_cover_heldout(tmp_path):
    cache, paths, metadata = artifacts(tmp_path)
    bank = TaskContextBank.load(paths["bank_path"])
    training = load_readout_cache(paths["train_readout_path"], paths["bank_path"], bank)
    query = load_ego_query_readouts(paths["query_readout_path"], paths["bank_path"], paths["train_readout_path"],
                                     bank, cache.splits, expected_metadata=metadata)
    assert training["episode_ids"] == cache.splits["train"]
    assert query["episode_ids"] == ordered_episode_ids(cache)
    assert bank.metadata["semantic_task_count"] == 2
    assert bank.metadata["memory_semantics"] == "predicted_effect_trace"
    assert bank.key_dim == 4 and bank.value_dim == 6


def test_static_context_contains_heldout_queries_and_matches_public_retrieval(tmp_path):
    cache, paths, metadata = artifacts(tmp_path)
    head_path = tmp_path / "head.pt"
    head_checkpoint(head_path, paths, cache)
    output_path = tmp_path / "static.pt"
    result = build_ego_static_context_cache(cache=cache, head_path=head_path, output_path=output_path,
                                             expected_metadata=metadata, top_k=2, batch_size=3, **paths)
    assert result["train_queries"] == 4 and result["validation_queries"] == 2
    output = StaticEpisodeContextCache(output_path)
    assert set(output.episode_ids) == set(cache.episodes)
    retriever = StaticTaskContextRetriever.load(paths["bank_path"], head_path, top_k=2)
    query = torch.load(paths["query_readout_path"], weights_only=False)
    serialized = torch.load(output_path, weights_only=False)
    for index, identity in enumerate(query["episode_ids"]):
        exclusion = [identity] if identity in cache.splits["train"] else None
        expected = retriever.retrieve(query["readouts"][index:index + 1], exclude_episode_ids=exclusion)
        torch.testing.assert_close(output.get(identity), expected.values[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(serialized["retrieval_indices"][index], expected.indices[0])
        retrieved_ids = [retriever.bank.entries[i]["episode_id"] for i in expected.indices[0]]
        assert set(retrieved_ids) <= set(cache.splits["train"])
        if exclusion:
            assert identity not in retrieved_ids
    assert output.metadata["query_readout_cache_sha256"] == checkpoint_sha256(paths["query_readout_path"])
    assert output.metadata["phase_cache_manifest_sha256"] == "boundaries"


def test_head_selection_cannot_use_cte_heldout_episodes(tmp_path):
    cache, paths, metadata = artifacts(tmp_path)
    head_path = tmp_path / "head.pt"
    head = head_checkpoint(head_path, paths, cache)
    head["validation_episode_ids"] = cache.splits["validation"]
    torch.save(head, head_path)
    with pytest.raises(ValueError, match="inside the CTE training split"):
        build_ego_static_context_cache(cache=cache, head_path=head_path, output_path=tmp_path / "static.pt",
                                         expected_metadata=metadata, top_k=2, **paths)


def test_query_artifact_rejects_split_and_train_readout_drift(tmp_path):
    cache, paths, metadata = artifacts(tmp_path)
    bank = TaskContextBank.load(paths["bank_path"])
    payload = torch.load(paths["query_readout_path"], weights_only=False)
    changed = deepcopy(payload)
    changed["readouts"][0] += 1
    torch.save(changed, paths["query_readout_path"])
    with pytest.raises(ValueError, match="disagree"):
        load_ego_query_readouts(paths["query_readout_path"], paths["bank_path"], paths["train_readout_path"], bank, cache.splits)
    payload["episode_splits"]["validation"] = ["different episode"]
    torch.save(payload, paths["query_readout_path"])
    with pytest.raises(ValueError, match="held-out episode split"):
        load_ego_query_readouts(paths["query_readout_path"], paths["bank_path"], paths["train_readout_path"], bank, cache.splits)


def test_frozen_readout_extraction_receives_only_initial_image_and_instruction():
    script = Path(__file__).resolve().parents[2] / "scripts/build_zeva_ego_task_memory.py"
    spec = importlib.util.spec_from_file_location("ego_memory_builder_test", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    cache = Cache()

    class Base:
        def __getitem__(self, index):
            number = index // 49
            video = torch.full((3, 9, 4, 4), float("nan"))
            video[:, 0] = number
            return {"dataset_index": index, "episode_id": f"episode-{number}", "episode_step": 0,
                    "task_id": str(number % 2), "prompt": f"instruction {number}",
                    "video": video, "action": torch.zeros(32, 14),
                    "context": torch.ones(2, 7), "context_mask": torch.tensor([True, False])}

        def __len__(self):
            return 294

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

        def extract_task_context_readout(self, image, context, mask):
            assert image.ndim == 4 and bool(torch.isfinite(image).all())
            assert torch.equal(mask, torch.tensor([[True, False]]).expand(len(image), -1))
            return image.mean((1, 2, 3))[:, None].expand(-1, 7) * self.weight + context[:, 0]

    policy = Policy()
    readouts, instructions = builder.extract_initial_readouts(policy, Base(), cache, device="cpu", batch_size=2)
    assert not policy.training and not policy.weight.requires_grad
    assert set(readouts) == set(cache.episodes)
    for index in range(6):
        torch.testing.assert_close(readouts[f"episode-{index}"], torch.full((7,), float(index + 1)))
        assert instructions[f"episode-{index}"] == f"instruction {index}"
