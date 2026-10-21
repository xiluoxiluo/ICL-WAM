import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from fastwam.zeva.cross_attempt_cache import (
    CROSS_ATTEMPT_CACHE_FORMAT, CrossAttemptPIMCache, file_sha256,
    validate_episode_splits,
)
from fastwam.zeva.prepared_stage2 import PreparedPIMCache, PreparedZevaStage2Dataset, StaticEpisodeContextCache, STATIC_CONTEXT_FORMAT
from fastwam.zeva.schemas import CacheManifest


SPEC = importlib.util.spec_from_file_location("build_cross", Path(__file__).resolve().parents[2] / "scripts/build_zeva_cross_attempt_cache.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def make_source(tmp_path):
    root = tmp_path / "phase"
    root.mkdir()
    manifest = CacheManifest(
        schema_version="zeva_fastwam_robotwin_cache_v4", history_semantics="full_episode_prefix",
        query_step_unit="raw_action_step", phase_dim=2, effect_dim=2,
        cte_checkpoint_sha256="cte", dataset_stats_sha256="stats", dataset_path="/dataset",
    )
    (root / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    rows, phases, effects = [], [], []
    splits = {"train": [], "validation": []}
    condition_map = {}
    semantic_mapping = {}
    for task_no, task in enumerate(("beat_block_hammer", "stack_bowls_three")):
        for episode_no in range(4):
            episode = f"dataset::episode-{task_no * 4 + episode_no}"
            semantic_mapping[str(task_no * 4 + episode_no)] = task
            splits["train" if episode_no < 3 else "validation"].append(episode)
            condition_map[episode] = "other" if episode_no == 2 else "same"
            for raw in (0, 8, 16, 80):
                rows.append({"episode_id": episode, "task_id": task, "record_type": "phase_query",
                             "raw_step": raw, "window_index": (task_no * 4 + episode_no) * 1000 + raw,
                             "start_raw_step": raw, "end_raw_step": raw,
                             "shard": "shard.safetensors", "offset": len(phases)})
                phases.append([1. + episode_no, raw / 80.])
                effects.append([0., 0.])
            for index in range(5):
                rows.append({"episode_id": episode, "task_id": task, "record_type": "effect",
                             "raw_step": (index + 1) * 16, "start_raw_step": index * 16,
                             "end_raw_step": (index + 1) * 16,
                             "shard": "shard.safetensors", "offset": len(phases)})
                phases.append([1. + episode_no, float(index)])
                effects.append([10. + episode_no, float(index + 1)])
    # Source order intentionally puts some future effects before earlier
    # queries; the builder must use explicit raw end times for causal BIT.
    rows.reverse()
    (root / "episode_index.json").write_text(json.dumps(rows))
    save_file({"phase_pre": torch.tensor(phases), "effect": torch.tensor(effects), "valid": torch.ones(len(phases), dtype=torch.bool)}, str(root / "shard.safetensors"))
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps(splits))
    condition_path = tmp_path / "conditions.json"
    condition_path.write_text(json.dumps(condition_map))
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({"format": "robotwin_semantic_task_map_v1", "episode_to_semantic_task": semantic_mapping}))
    return root, split_path, condition_path, semantic


def build(tmp_path, split="train", *, use_conditions=False):
    phase, splits, conditions, semantic = make_source(tmp_path)
    output = tmp_path / "prepared"
    BUILDER.build_cache(phase_cache=phase, output=output, semantic_map=semantic, split=split,
                        episode_splits=splits, conditions=conditions if use_conditions else None,
                        max_past_attempts=3, max_entries_per_attempt=3)
    return output


def test_compact_histories_are_train_only_and_query_bit_is_causal(tmp_path):
    output = build(tmp_path)
    cache = CrossAttemptPIMCache(output)
    assert cache._arrays["history_phases"].shape == (6, 3, 2)
    assert not (output / "pim_phases.npy").exists()
    rows = [cache.get(i, history_kind="matched", history_count=1) for i in range(len(cache))]
    for row in rows:
        assert row["episode_id"] not in row["pim_source_episode_ids"]
        assert all(source in cache.splits["train"] for source in row["pim_source_episode_ids"])
        assert row["pim_mask"].sum() == 3
        assert row["pim_effects"][row["pim_mask"]][:, 0].min() >= 10.  # raw, not normalized
        if row["raw_step"] in (0, 8):
            assert not row["bit_mask"].any()
        elif row["raw_step"] == 16:
            assert row["bit_mask"].sum() == 1
            assert row["bit_effects"][-1, 1] == 1.
        else:
            assert row["bit_effects"][:, 1].tolist() == [2., 3., 4., 5.]
    assert cache.manifest["eligible_pairing_episodes"]["cross_condition"] == 0
    with pytest.raises(ValueError, match="no cross_condition"):
        cache.get(0, history_kind="cross_condition")


def test_validation_sampling_is_repeatable_and_uses_real_conditions(tmp_path):
    cache = CrossAttemptPIMCache(build(tmp_path, "validation", use_conditions=True))
    for kind in ("matched", "far", "cross_condition"):
        row = cache.get(0, history_kind=kind, history_count=1)
        source = next(item for item in cache.history_episodes if item["episode_id"] == row["pim_source_episode_ids"][0])
        query = cache.query_episodes[row["episode_id"]]
        assert (source["condition"] != query["condition"]) == (kind == "cross_condition")
    first = cache.get(0)
    torch.manual_seed(9999)
    second = cache.get(0)
    assert first["pim_source_episode_ids"] == second["pim_source_episode_ids"]
    assert torch.equal(first["pim_phases"], second["pim_phases"])
    empty = cache.get(0, history_kind="empty")
    assert not empty["pim_mask"].any()
    assert not empty["pim_effects"].any()


def test_reader_checks_provenance_even_if_metadata_checksum_is_updated(tmp_path):
    output = build(tmp_path)
    pairings = json.loads((output / "pairings.json").read_text())
    histories = json.loads((output / "history_episodes.json").read_text())
    episode = next(iter(pairings))
    pairings[episode]["matched"] = [next(i for i, row in enumerate(histories) if row["episode_id"] == episode)]
    (output / "pairings.json").write_text(json.dumps(pairings))
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["metadata_sha256"]["pairings.json"] = file_sha256(output / "pairings.json")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="history leakage"):
        CrossAttemptPIMCache(output)


def test_prepared_dataset_accepts_new_schema_and_preserves_exact_window_identity(tmp_path):
    output = build(tmp_path)
    cache = PreparedPIMCache(output, expected={"format": CROSS_ATTEMPT_CACHE_FORMAT, "split": "train"})
    with pytest.raises(ValueError, match="identity mismatch"):
        PreparedPIMCache(output, expected={"retrieval_mode": "phase"})
    static_path = tmp_path / "static.pt"
    torch.save({"format": STATIC_CONTEXT_FORMAT, "episode_ids": list(cache.episode_ids),
                "task_contexts": torch.ones(len(cache.episode_ids), 256)}, static_path)
    index = {cache.get(i)["window_index"]: cache.get(i) for i in range(len(cache))}

    class Base:
        def __len__(self):
            return max(index) + 1

        def __getitem__(self, window):
            row = index[window]
            return {"video": torch.zeros(3, 9, 2, 2), "action": torch.zeros(32, 14),
                    "dataset_index": window, "episode_id": row["episode_id"],
                    "task_id": row["task_id"], "episode_step": row["raw_step"], "episode_num_steps": 112}

    dataset = PreparedZevaStage2Dataset(Base(), cache, StaticEpisodeContextCache(static_path))
    sample = dataset[0]
    assert sample["pim_phases"].shape == (9, 2)
    assert sample["task_context"].shape == (256,)
    assert sample["pim_history_count"] >= 1
    assert 0 <= sample["pim_history_kind"] < 3


def test_splits_and_incomplete_effects_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        validate_episode_splits({"train": ["a"], "validation": ["a"]}, {"a"})
    phase, splits, _conditions, semantic = make_source(tmp_path)
    rows = json.loads((phase / "episode_index.json").read_text())
    train_ids = json.loads(splits.read_text())["train"]
    bad = next(row for row in rows if row["record_type"] == "effect" and row["episode_id"] in train_ids)
    bad["start_raw_step"] += 1
    (phase / "episode_index.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="completed full effect"):
        BUILDER.build_cache(phase_cache=phase, output=tmp_path / "bad", semantic_map=semantic,
                            episode_splits=splits)
    assert not (tmp_path / "bad").exists()


def test_cache_does_not_overwrite_existing_artifact(tmp_path):
    output = build(tmp_path)
    with pytest.raises(FileExistsError):
        BUILDER.build_cache(phase_cache=tmp_path / "phase", output=output, semantic_map=tmp_path / "semantic.json")


def test_payload_corruption_and_wrong_semantic_map_are_rejected(tmp_path):
    output = build(tmp_path)
    values = np.load(output / "history_effects.npy", mmap_mode="r+")
    values[0, 0, 0] += 1
    values.flush()
    with pytest.raises(ValueError, match="tensor checksum"):
        CrossAttemptPIMCache(output)
    semantic_path = tmp_path / "semantic.json"
    semantic = json.loads(semantic_path.read_text())
    semantic["episode_to_semantic_task"]["0"] = "stack_bowls_three"
    semantic_path.write_text(json.dumps(semantic))
    with pytest.raises(ValueError, match="semantic task mismatch"):
        BUILDER.build_cache(phase_cache=tmp_path / "phase", output=tmp_path / "mismatch", semantic_map=semantic_path)


def test_cross_attempt_training_preset_has_consistent_serving_capacity():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[2] / "configs")):
        cfg = compose(config_name="train", overrides=[
            "task=robotwin_zeva_fastwam_cross_attempt_3cam_384",
            "ckpt=/tmp/base.pt",
            "model.zeva.policy_checkpoint=/tmp/stage2a.pt",
            "model.zeva.prepared_stage2.pim_cache_path=/tmp/pim_train",
            "model.zeva.prepared_stage2.validation_pim_cache_path=/tmp/pim_validation",
        ])
    z = cfg.model.zeva
    assert z.training_stage == "pim_adapter"
    assert z.prompt.pim_variant == z.adapter.pim_variant == "cross_attempt"
    assert z.memory.cross_attempt_only is True
    assert z.memory.pim_max_entries == z.memory.pim_top_k == z.prompt.persistent_length
    assert z.memory.pim_max_entries == z.memory.max_past_attempts * z.memory.max_entries_per_attempt
    assert z.memory.pim_retrieval_mode == "cross_attempt"
    assert z.prompt.pim_context_dropout == z.prompt.pim_support_dropout == 0.0
    assert z.freeze_fastwam and z.freeze_cte_stage2 and cfg.data.train.use_text_embed_cache
    assert cfg.ckpt == "/tmp/base.pt"
    assert z.policy_checkpoint == "/tmp/stage2a.pt"
    assert z.prepared_stage2.validation_pim_cache_path == "/tmp/pim_validation"
