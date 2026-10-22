"""Source-population and holdout contracts checked against pinned Zeva-Ego."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from fastwam.zeva.ego_cte_training import (
    PairedEpochSampler,
    build_episode_splits,
    canonical_sha256,
    episode_metadata,
    select_ego_episode_scope,
    verify_ego_episode_scope,
)
from fastwam.zeva.ego_episode_data import EGO_CAMERA_KEYS, EgoEpisode, RoboTwinEgoEpisodeDataset


def source_scope(tmp_path, *, tasks=("beta", "alpha"), short=False):
    episodes, conditions = [], {}
    for condition in ("Clean", "Randomized"):
        for task in ("alpha", "beta"):
            for position in range(21):
                index = len(episodes)
                identity = f"{condition}/{task}::episode-{position}"
                length = 8 if short and (condition, task, position) == ("Clean", "beta", 0) else 49
                episodes.append(EgoEpisode(identity, task, index, index * 49, length))
                conditions[identity] = condition
    # Input interleaving differs from reference partition order. The split and
    # training Subset must nevertheless keep the reference partition order.
    # Keep each partition's real source-table sequence while interleaving
    # partitions: positions remain 0..20 in their actual source order.
    source = tuple(episodes[partition * 21 + position]
                   for position in range(21) for partition in (2, 0, 3, 1))
    dataset = SimpleNamespace(source_episodes=source,
                              episodes=tuple(episode for episode in source if episode.episode_length > 24),
                              executed_horizon=24)
    subset_path, conditions_path = tmp_path / "tasks.json", tmp_path / "conditions.json"
    subset_path.write_text(json.dumps({"task_names": list(tasks)}))
    conditions_path.write_text(json.dumps(conditions))
    return dataset, subset_path, conditions_path


def select_scope(dataset, subset_path, conditions_path):
    return select_ego_episode_scope(dataset, task_subset_path=subset_path,
                                    conditions_path=conditions_path, conditions=["Clean", "Randomized"])


def scope_manifest(dataset, scope, population):
    return {"data_scope": scope, "tasks": scope["task_names"],
            "recipe": {"split_policy": "zeva_ego", "seed": 2026, "split_seed": 1000, "validation_fraction": .05},
            "data_identity": {"episode_metadata_sha256": canonical_sha256(episode_metadata(dataset.episodes))},
            "episode_splits": build_episode_splits(dataset.episodes, partition_rows=population)}


def test_condition_task_sha_round_matches_pinned_reference_membership(tmp_path):
    dataset, subset, conditions = source_scope(tmp_path)
    scope, population = select_scope(dataset, subset, conditions)
    result = build_episode_splits(dataset.episodes, partition_rows=population)
    assert scope["task_names"] == ["beta", "alpha"]
    assert scope["partition_keys"] == [["Clean", "alpha"], ["Clean", "beta"],
                                       ["Randomized", "alpha"], ["Randomized", "beta"]]
    # Captured from robotwin_validation_episode_mask at reference bae85ad:
    # seed1000, fraction.05, four source partitions of21 => positions4,20,17,0.
    assert result["validation"] == ["Clean/alpha::episode-4", "Clean/beta::episode-20",
                                     "Randomized/alpha::episode-17", "Randomized/beta::episode-0"]
    assert len(result["train"]) == 80  # round(21 * .05) == 1, not ceil == 2.
    assert result["train"][:4] == [f"Clean/alpha::episode-{index}" for index in range(4)]
    verify_ego_episode_scope(dataset, scope_manifest(dataset, scope, population))


def test_task_subset_and_short_episode_filter_happen_after_source_partition_split(tmp_path):
    dataset, subset, conditions = source_scope(tmp_path, tasks=("beta",), short=True)
    scope, population = select_scope(dataset, subset, conditions)
    assert len(population) == 84
    assert len(dataset.episodes) == 41
    assert {row["task_id"] for row in population} == {"alpha", "beta"}
    assert {episode.task_id for episode in dataset.episodes} == {"beta"}
    assert any(row["episode_length"] == 8 for row in population)
    result = build_episode_splits(dataset.episodes, partition_rows=population)
    assert result["validation"] == ["Clean/beta::episode-20", "Randomized/beta::episode-0"]
    assert "Clean/beta::episode-0" not in result["train"]
    verify_ego_episode_scope(dataset, scope_manifest(dataset, scope, population))


def test_reference_scope_requires_real_mapping_and_rejects_unknown_empty_positions(tmp_path):
    dataset, subset, conditions = source_scope(tmp_path)
    with pytest.raises(FileNotFoundError, match="explicit task-subset"):
        select_scope(dataset, tmp_path / "missing.json", conditions)
    values = json.loads(conditions.read_text())
    del values[dataset.source_episodes[0].episode_id]
    conditions.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="whole source split population"):
        select_scope(dataset, subset, conditions)
    dataset.empty_source_episode_ids = ("unknown-empty-episode",)
    with pytest.raises(ValueError, match="empty source episodes"):
        select_scope(dataset, subset, conditions)


def test_episode_index_retains_short_source_and_detects_empty_slots_without_decoding():
    class MetadataTable:
        column_names = ["task_index"]

        def with_format(self, value):
            assert value is None
            return self

        def __getitem__(self, starts):
            assert starts == [0, 49, 57]  # No nonexistent empty-episode first frame.
            return {"task_index": [0, 0, 0]}

    inner = SimpleNamespace(root="/source", episodes=None, num_frames=106,
                            meta=SimpleNamespace(total_episodes=4), hf_dataset=MetadataTable(),
                            episode_data_index={"from": [0, 49, 57, 57], "to": [49, 57, 57, 106]})
    source = SimpleNamespace(multi_dataset=SimpleNamespace(_datasets=[inner]),
                             global_sample_stride=1, action_size=32,
                             processor=SimpleNamespace(use_stepwise_action_norm=False),
                             image_meta=[{"key": key, "lerobot_key": key} for key in EGO_CAMERA_KEYS])
    base = SimpleNamespace(lerobot_dataset=source, semantic_task_identity={},
                           resolve_semantic_task_id=lambda **_: "alpha")
    dataset = RoboTwinEgoEpisodeDataset(base, episode_ids=["/source::episode-3"])
    assert [episode.episode_length for episode in dataset.source_episodes] == [49, 8, 49]
    assert [episode.episode_id for episode in dataset.episodes] == ["/source::episode-3"]
    assert dataset.empty_source_episode_ids == ("/source::episode-2",)


def test_reference_source_binding_rejects_scope_relabel_and_unselected_source_drift(tmp_path):
    dataset, subset, conditions = source_scope(tmp_path, tasks=("beta",))
    scope, population = select_scope(dataset, subset, conditions)
    manifest = scope_manifest(dataset, scope, population)
    verify_ego_episode_scope(dataset, manifest)
    legacy = deepcopy(manifest)
    del legacy["data_scope"]
    with pytest.raises(ValueError, match="versioned condition/task"):
        verify_ego_episode_scope(dataset, legacy)
    changed = deepcopy(manifest)
    changed["data_scope"]["partition_population"][0]["episode_position"] = 200
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_ego_episode_scope(dataset, changed)
    changed = deepcopy(manifest)
    train, validation = changed["episode_splits"]["train"], changed["episode_splits"]["validation"]
    train[0], validation[0] = validation[0], train[0]
    with pytest.raises(ValueError, match="SHA/round"):
        verify_ego_episode_scope(dataset, changed)
    changed = deepcopy(manifest)
    changed["data_scope"]["partition_population"][0]["episode_length"] += 1
    changed["data_scope"]["partition_population_sha256"] = canonical_sha256(changed["data_scope"]["partition_population"])
    with pytest.raises(ValueError, match="actual source metadata/order"):
        verify_ego_episode_scope(dataset, changed)
    source = list(dataset.source_episodes)
    assert source[0].task_id == "alpha"
    source[0] = replace(source[0], episode_length=source[0].episode_length + 1)
    dataset.source_episodes = tuple(source)
    with pytest.raises(ValueError, match="full source episode population"):
        verify_ego_episode_scope(dataset, manifest)


def test_reference_singleton_partition_and_explicit_split_validation(tmp_path):
    rows = [EgoEpisode(f"episode-{index}", "alpha", index, index * 49, 49) for index in range(4)]
    population = [{**row, "partition_id": int(index > 0), "episode_position": max(0, index - 1)}
                  for index, row in enumerate(episode_metadata(rows))]
    split = build_episode_splits(rows, partition_rows=population)
    assert "episode-0" in split["train"]  # Singleton partitions keep their sole episode.
    assert len(split["validation"]) == 1
    supplied = tmp_path / "split.json"
    supplied.write_text(json.dumps(split))
    assert build_episode_splits(rows, partition_rows=population, split_path=supplied) == split
    changed = deepcopy(split)
    changed["train"][0], changed["validation"][0] = changed["validation"][0], changed["train"][0]
    supplied.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs from the reference"):
        build_episode_splits(rows, partition_rows=population, split_path=supplied)
    with pytest.raises(ValueError, match="explicit condition/task"):
        build_episode_splits(rows)


def test_sampler_preserves_external_task_subset_order_and_absent_task_slots():
    labels = ["alpha"] * 5 + ["beta"] * 4
    sampler = PairedEpochSampler(labels, seed=1000, task_order=["beta", "absent", "alpha"])
    # Captured from reference train_cte.PairedEpochSampler at bae85ad with
    # the same dataset ordering and the same external task_names list.
    assert list(sampler) == [6, 7, 8, 5, 3, 4, 1, 0, 2]
    sampler.set_epoch(1)
    assert list(sampler) == [6, 5, 8, 7, 2, 1, 0, 3, 4]
