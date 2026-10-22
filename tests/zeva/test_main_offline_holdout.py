import json

import numpy as np
import torch
from omegaconf import OmegaConf

from fastwam.zeva.main_data import build_success_manifest, select_main_success_rows
from fastwam.zeva.stage1_sampling import CTETrainIndex
from fastwam.zeva.static_task_context import (
    BEHAVIOR_KEY_SPACE, BEHAVIOR_VALUE_SPACE, HEAD_FORMAT, READOUT_FORMAT, READOUT_KIND,
    StaticTaskContextRetrievalConfig, StaticTaskContextRetrievalHead,
)
from fastwam.zeva.task_context import TaskContextBank
from fastwam.zeva.task_context_data import split_main_behavior_artifacts
from fastwam.zeva.task_context_training import train_retrieval_head


def _metadata(tmp_path):
    root = tmp_path / "source"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text('{"total_episodes":8}')
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": i, "success": True}) for i in range(8)))
    (root / "stats.json").write_text("{}")
    (root / "meta/tasks.jsonl").write_text("{}")
    manifest = root / "success.json"
    manifest.write_text(json.dumps(build_success_manifest([root])))
    cfg = OmegaConf.create({"model": {"zeva": {"profile": "main_observed", "offline": {"success_manifest_path": str(manifest)}}},
        "data": {"train": {"dataset_dirs": [str(root)], "pretrained_norm_stats": str(root / "stats.json"),
            "semantic_task_map_path": str(root / "meta/tasks.jsonl"), "is_training_set": True}}})
    rows = [CTETrainIndex(i, f"{root}::episode-{i}", f"task-{i // 4}", 0) for i in range(8)]
    return rows, select_main_success_rows(rows, cfg)[1]


def test_validation_prototypes_are_absent_from_bank_but_initial_queries_survive(tmp_path):
    rows, metadata = _metadata(tmp_path)
    entries = [{"episode_id": row.episode_id, "task_id": row.task_id,
                "retrieval_key": torch.randn(4), "behavior_value": torch.randn(6)} for row in rows]
    bank = TaskContextBank(entries, key_dim=4, value_dim=6, metadata=metadata)
    readouts = torch.arange(8 * 5).reshape(8, 5).float()
    train_bank, train_readouts, validation = split_main_behavior_artifacts(bank, readouts)
    assert {entry["episode_id"] for entry in train_bank.entries} == set(metadata["episode_splits"]["train"])
    assert set(validation["validation_episode_ids"]) == set(metadata["episode_splits"]["validation"])
    assert train_readouts.shape == (6, 5) and validation["validation_readouts"].shape == (2, 5)
    path = tmp_path / "bank.pt"
    train_bank.save(path)
    restored = TaskContextBank.load(path)
    assert restored.metadata["data_role"] == "train"


def test_pim_candidate_mask_excludes_every_heldout_outcome(tmp_path):
    from scripts.build_zeva_pim_retrieval_cache import _main_candidate_mask

    rows, metadata = _metadata(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "manifest.json").write_text(json.dumps(metadata))
    episode_ids = [row.episode_id for row in rows]
    (staging / "episode_ids.json").write_text(json.dumps(episode_ids))
    # Two effect rows per episode, including held-out outcomes that remain
    # usable only in that episode's causal BIT.
    effect_episode = np.repeat(np.arange(8), 2)
    allowed = _main_candidate_mask(staging, effect_episode)
    for index, episode_id in enumerate(episode_ids):
        assert allowed[2 * index:2 * index + 2].tolist() == [episode_id in metadata["episode_splits"]["train"]] * 2


def test_retrieval_uses_external_holdout_without_training_on_its_readouts():
    torch.manual_seed(23)
    head = StaticTaskContextRetrievalHead(StaticTaskContextRetrievalConfig(input_dim=5, hidden_dim=8, output_dim=4, dropout=0))
    readouts, keys = torch.randn(6, 5), torch.randn(6, 4)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    heldout = torch.randn(2, 5, requires_grad=True)
    metrics = train_retrieval_head(head, readouts, keys, labels, steps=2, batch_size=4, eval_every=1,
        validation_readouts=heldout, validation_semantic_ids=torch.tensor([0, 1]))
    assert metrics["external_validation"]
    assert metrics["train_indices"] == list(range(6))
    assert metrics["validation_indices"] == [0, 1]
    assert heldout.grad is None


def test_main_offline_builders_join_prepared_train_validation_views_on_cpu(tmp_path, monkeypatch):
    from fastwam.zeva.cache import save_phase_effect_cache
    from fastwam.zeva.checkpoint import checkpoint_sha256
    from fastwam.zeva.main_data import validate_main_artifact
    from fastwam.zeva.main_prepared import split_main_prepared_dataset
    from fastwam.zeva.prepared_stage2 import PreparedPIMCache, PreparedZevaStage2Dataset, StaticEpisodeContextCache
    from fastwam.zeva.schemas import CacheManifest
    from fastwam.zeva.cte_windows import MAIN_CTE_WINDOW_SEMANTICS
    from scripts.build_zeva_pim_retrieval_cache import main as build_pim
    from scripts.build_zeva_static_episode_context_cache import main

    torch.manual_seed(41)
    rows, metadata = _metadata(tmp_path)
    root = tmp_path / "source"
    bank_metadata = {**metadata, "dataset_stats_sha256": checkpoint_sha256(root / "stats.json"),
        "semantic_task_map_sha256": checkpoint_sha256(root / "meta/tasks.jsonl"),
        "video_size": [384, 320], "context_len": 128, "history_semantics": "full_episode_prefix",
        "semantic_task_count": 2, "readout_dim": 5, "key_space": BEHAVIOR_KEY_SPACE,
        "value_space": BEHAVIOR_VALUE_SPACE, "readout_kind": READOUT_KIND, "split": "train"}
    bank = TaskContextBank([{"episode_id": row.episode_id, "task_id": row.task_id,
        "retrieval_key": torch.randn(4), "behavior_value": torch.randn(6)} for row in rows],
        key_dim=4, value_dim=6, metadata=bank_metadata)
    bank, readouts, heldout = split_main_behavior_artifacts(bank, torch.randn(8, 5))
    bank_path, readout_path, head_path, output_path = [tmp_path / name for name in ("bank.pt", "readout.pt", "head.pt", "static.pt")]
    bank.save(bank_path)
    torch.save({**bank.metadata, **heldout, "format": READOUT_FORMAT, "readout_kind": READOUT_KIND,
        "bank_sha256": checkpoint_sha256(bank_path), "episode_ids": [entry["episode_id"] for entry in bank.entries],
        "readouts": readouts}, readout_path)
    head_cfg = StaticTaskContextRetrievalConfig(input_dim=5, hidden_dim=8, output_dim=4, dropout=0)
    head = StaticTaskContextRetrievalHead(head_cfg)
    torch.save({**bank.metadata, "format": HEAD_FORMAT, "readout_kind": READOUT_KIND,
        "bank_sha256": checkpoint_sha256(bank_path), "step": 1, "config": head_cfg.as_dict(), "model": head.state_dict()}, head_path)
    cfg = OmegaConf.create({"model": {"tokenizer_max_len": 128, "zeva": {"profile": "main_observed",
        "offline": {"success_manifest_path": str(root / "success.json")}, "task_context": {
            "bank_path": str(bank_path), "readout_cache_path": str(readout_path), "retrieval_checkpoint": str(head_path),
            "static_episode_cache_path": str(output_path), "top_k": 2}}},
        "data": {"train": {"dataset_dirs": [str(root)], "pretrained_norm_stats": str(root / "stats.json"),
            "semantic_task_map_path": str(root / "meta/tasks.jsonl"), "video_size": [384, 320], "is_training_set": True}}})
    main.__wrapped__(cfg)
    output = torch.load(output_path, weights_only=False)
    assert output["data_role"] == "all"
    assert set(output["episode_ids"]) == {row.episode_id for row in rows}
    assert output["retrieval_indices"].max().item() < len(bank)
    loaded = StaticEpisodeContextCache(output_path)
    assert all(loaded.get(episode_id).shape == (6,) for episode_id in heldout["validation_episode_ids"])

    # Exercise the actual PIM entry point, including source/split identity,
    # safetensor staging, cosine retrieval, mmap export and random equivalence
    # checks. Only frozen feature extraction uses small synthetic vectors.
    cte_path = tmp_path / "cte.pt"
    torch.save({**metadata, "data_role": "train", "semantic_task_identity": {
        "sha256": checkpoint_sha256(root / "meta/tasks.jsonl")}}, cte_path)
    phase_root, pim_root = tmp_path / "phase", tmp_path / "pim"
    phase_manifest = CacheManifest(**metadata,
        cte_window_semantics=MAIN_CTE_WINDOW_SEMANTICS,
        schema_version="zeva_fastwam_robotwin_cache_v4", history_semantics="full_episode_prefix",
        query_step_unit="raw_action_step", dataset_path=str(root), phase_dim=4, effect_dim=16,
        cte_checkpoint_sha256=checkpoint_sha256(cte_path),
        dataset_stats_sha256=checkpoint_sha256(root / "stats.json"))
    records, samples = [], {}
    for episode_index, row in enumerate(rows):
        for raw_step in (0, 16, 32):
            phase = torch.tensor([1.0, float(episode_index + 1), float(raw_step / 16), 0.5])
            if raw_step:
                effect = torch.zeros(16)
                effect[episode_index * 2 + raw_step // 16 - 1] = 1.0
                records.append({"record_type": "effect", "episode_id": row.episode_id,
                    "task_id": row.task_id, "window_index": None, "episode_step": raw_step - 16,
                    "raw_step": raw_step, "start_raw_step": raw_step - 16, "end_raw_step": raw_step,
                    "phase_pre": phase, "phase_post": phase, "effect": effect, "valid": True})
            # Deliberately sparse original source indices: no filtering or split
            # view may replace these with densely renumbered prepared indices.
            source_index = row.dataset_index * 100 + raw_step
            records.append({"record_type": "phase_query", "episode_id": row.episode_id,
                "task_id": row.task_id, "window_index": source_index, "episode_step": raw_step,
                "raw_step": raw_step, "start_raw_step": raw_step, "end_raw_step": raw_step,
                "phase_pre": phase, "phase_post": phase, "effect": torch.zeros(16), "valid": True})
            samples[source_index] = {"dataset_index": source_index, "episode_id": row.episode_id,
                "task_id": row.task_id, "episode_step": raw_step, "episode_num_steps": 80,
                "video": torch.zeros(3, 9, 8, 8), "action": torch.zeros(32, 14)}
    save_phase_effect_cache(phase_root, records, phase_manifest, shard_size=7)
    OmegaConf.update(cfg, "device", "cpu", force_add=True)
    for key, value in {
        "cte.checkpoint": str(cte_path), "cache.path": str(phase_root),
        "prepared_stage2.pim_cache_path": str(pim_root), "memory.pim_retrieval_mode": "phase",
        "memory.pim_top_k": 4, "memory.bit_size": 4, "prompt.persistent_length": 4,
        "prompt.brief_length": 4, "prompt.phase_dim": 4, "prompt.effect_dim": 16,
    }.items():
        OmegaConf.update(cfg, f"model.zeva.{key}", value, force_add=True)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("ZEVA_PIM_QUERY_BATCH", "5")
    build_pim.__wrapped__(cfg)
    pim = PreparedPIMCache(pim_root, expected={"data_role": "all", "candidate_split": "train",
        "dataset_path": str(root), "semantic_task_count": 2,
        "phase_cache_manifest_sha256": checkpoint_sha256(phase_root / "manifest.json"),
        "phase_cache_index_sha256": checkpoint_sha256(phase_root / "episode_index.json")})
    validate_main_artifact(pim.manifest, bank.metadata)
    validate_main_artifact(loaded.metadata, bank.metadata)

    class RawDataset:
        def __len__(self):
            return max(samples) + 1

        def __getitem__(self, index):
            return samples[index]

    prepared = PreparedZevaStage2Dataset(RawDataset(), pim, loaded)
    train, validation = split_main_prepared_dataset(prepared)
    assert len(train) == 18 and len(validation) == 6
    observed_windows = set()
    for view in (train, validation):
        for index in range(len(view)):
            sample = view[index]
            source_index = sample["dataset_index"]
            observed_windows.add(source_index)
            assert sample["episode_id"] == samples[source_index]["episode_id"]
            assert sample["episode_id"] in metadata["episode_splits"][view.split]
            assert sample["bit_mask"].sum().item() == sample["raw_step"] // 16
            if sample["raw_step"] == 0:
                assert not sample["bit_effects"].any()
            own_episode = int(sample["episode_id"].rsplit("-", 1)[1])
            selected_episodes = sample["pim_effects"].argmax(dim=-1).div(2, rounding_mode="floor")
            for selected_episode in selected_episodes.tolist():
                candidate = rows[selected_episode]
                assert candidate.episode_id in metadata["episode_splits"]["train"]
                assert candidate.task_id == sample["task_id"]
                assert selected_episode != own_episode
            if sample["raw_step"]:
                visible_effects = sample["bit_effects"][sample["bit_mask"]].argmax(dim=-1)
                assert visible_effects.tolist() == [own_episode * 2 + i for i in range(sample["raw_step"] // 16)]
            assert sample["task_context"].shape == (6,)
    assert observed_windows == set(samples)
