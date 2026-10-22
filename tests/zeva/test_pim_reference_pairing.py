"""Match the reference's one partner per episode and complete 5:2:1 curriculum."""

import json

import pytest
import torch

from fastwam.zeva.cross_attempt_cache import CrossAttemptPIMCache, build_episode_pairings, file_sha256
from fastwam.zeva.ego_boundary_cache import build_ego_boundary_cache
from fastwam.zeva.ego_pim_schedule import select_history_kind
from test_ego_effect_pipeline import BUILDER, _CausalCTE, make_boundary_cache, make_dataset


def test_pairing_uses_unique_nearest_farthest_and_excludes_query_episode():
    query = {"episode_id": "query", "task_id": "task", "condition": "same", "initial_phase": [1., 0.]}
    histories = [
        {**query, "split": "train"},
        {**query, "episode_id": "near", "initial_phase": [.9, .1], "split": "train"},
        {**query, "episode_id": "middle", "initial_phase": [.5, .5], "split": "train"},
        {**query, "episode_id": "far", "initial_phase": [-1., 0.], "split": "train"},
        {**query, "episode_id": "cross_near", "condition": "other", "split": "train"},
        {**query, "episode_id": "cross_far", "condition": "other", "initial_phase": [-1., 0.], "split": "train"},
    ]
    reference = build_episode_pairings({"query": query}, histories, max_histories=3,
        use_conditions=True, history_pairing="reference_nearest", allow_reduced_curriculum=False)
    assert reference["query"] == {"matched": [1], "far": [3], "cross_condition": [4]}
    legacy = build_episode_pairings({"query": query}, histories, max_histories=3, use_conditions=True)
    assert legacy["query"] == {"matched": [1, 2, 3], "far": [3, 2, 1], "cross_condition": [4, 5]}
    with pytest.raises(ValueError, match="real condition metadata"):
        build_episode_pairings({"query": query}, histories, max_histories=3, use_conditions=False,
            history_pairing="reference_nearest", allow_reduced_curriculum=False)
    with pytest.raises(ValueError, match="same- and cross-condition"):
        build_episode_pairings({"query": query}, histories[:4], max_histories=3, use_conditions=True,
            history_pairing="reference_nearest", allow_reduced_curriculum=False)


def _conditioned_source(tmp_path):
    source, _ = make_dataset((57,) * 5)
    ids = [episode.episode_id for episode in source.episodes]
    splits = {"train": ids[:4], "validation": ids[4:]}
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({"format": "robotwin_semantic_task_map_v1",
        "episode_to_semantic_task": {str(index): "beat_block_hammer" for index in range(5)}}))
    boundary = tmp_path / "boundary"
    build_ego_boundary_cache(source, _CausalCTE(), boundary,
        identity={"cte_checkpoint_sha256": "cte", "dataset_stats_sha256": "stats",
                  "semantic_task_map_sha256": file_sha256(semantic), "dataset_path": "/source"},
        episode_splits=splits)
    conditions = tmp_path / "conditions.json"
    conditions.write_text(json.dumps({identity: ("clean" if index in (0, 1, 4) else "randomized")
                                      for index, identity in enumerate(ids)}))
    return boundary, semantic, conditions


def test_reference_cache_freezes_partners_for_all_windows_and_keeps_full_cycle(tmp_path):
    boundary, semantic, conditions = _conditioned_source(tmp_path)
    output = tmp_path / "pim"
    manifest = BUILDER.build_cache(phase_cache=boundary, output=output, semantic_map=semantic,
                                   conditions=conditions, max_entries_per_attempt=3)
    assert manifest["history_pairing"] == "reference_nearest"
    assert manifest["allow_reduced_curriculum"] is False and manifest["reduced_curriculum"] is False
    assert manifest["globally_available_history_kinds"] == ["matched", "far", "cross_condition"]
    cache = CrossAttemptPIMCache(output)
    cache.validate_training_curriculum(history_pairing="reference_nearest", allow_reduced_curriculum=False)
    seen = {}
    for index in range(len(cache)):
        for kind in ("matched", "far", "cross_condition"):
            # Changing the global RNG and query window cannot select a
            # different demonstration for this episode/mode.
            torch.manual_seed(171 * index)
            row = cache.get(index, history_kind=kind, history_count=1)
            identity = (row["episode_id"], kind)
            partner = tuple(row["pim_source_episode_ids"])
            assert len(partner) == 1
            assert seen.setdefault(identity, partner) == partner
            assert len(cache.pairings[row["episode_id"]][kind]) == 1
    kinds = manifest["globally_available_history_kinds"]
    assert [select_history_kind(step, seed=9, warmup_steps=500, available_kinds=kinds)
            for step in range(8)] == [0] * 5 + [1] * 2 + [2]
    assert [select_history_kind(step, seed=9, warmup_steps=500, available_kinds=kinds)
            for step in range(500, 510)] == [0] * 5 + [1] * 2 + [2] + [3] * 2
    with pytest.raises(ValueError, match="history_pairing differs"):
        cache.validate_training_curriculum(history_pairing="legacy_candidates", allow_reduced_curriculum=False)


def test_missing_conditions_need_explicit_cache_and_training_ablation(tmp_path):
    boundary, semantic = make_boundary_cache(tmp_path)
    output = tmp_path / "pim"
    with pytest.raises(ValueError, match="real --conditions"):
        BUILDER.build_cache(phase_cache=boundary, output=output, semantic_map=semantic)
    assert not output.exists()
    manifest = BUILDER.build_cache(phase_cache=boundary, output=output, semantic_map=semantic,
                                   allow_reduced_curriculum=True)
    assert manifest["reduced_curriculum"] is True and manifest["allow_reduced_curriculum"] is True
    assert manifest["globally_available_history_kinds"] == ["matched", "far"]
    cache = CrossAttemptPIMCache(output)
    with pytest.raises(ValueError, match="allow_reduced_curriculum=true"):
        cache.validate_training_curriculum(history_pairing="reference_nearest", allow_reduced_curriculum=False)
    cache.validate_training_curriculum(history_pairing="reference_nearest", allow_reduced_curriculum=True)
    manifest["allow_reduced_curriculum"] = False
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="real condition metadata"):
        CrossAttemptPIMCache(output)


def test_incomplete_real_conditions_are_rejected_and_legacy_mode_stays_explicit(tmp_path):
    boundary, semantic, conditions = _conditioned_source(tmp_path)
    mapping = json.loads(conditions.read_text())
    conditions.write_text(json.dumps({identity: "only_one_condition" for identity in mapping}))
    output = tmp_path / "strict"
    with pytest.raises(ValueError, match="same- and cross-condition"):
        BUILDER.build_cache(phase_cache=boundary, output=output, semantic_map=semantic, conditions=conditions)
    assert not output.exists()
    legacy = tmp_path / "legacy"
    manifest = BUILDER.build_cache(phase_cache=boundary, output=legacy, semantic_map=semantic,
        history_pairing="legacy_candidates", allow_reduced_curriculum=True)
    cache = CrossAttemptPIMCache(legacy)
    assert manifest["history_pairing"] == "legacy_candidates" and manifest["reduced_curriculum"] is True
    assert all(len(row["matched"]) == 3 for row in cache.pairings.values())
    with pytest.raises(ValueError, match="history_pairing differs"):
        cache.validate_training_curriculum(history_pairing="reference_nearest", allow_reduced_curriculum=True)
