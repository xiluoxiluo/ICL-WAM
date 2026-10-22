"""Reference defaults bind tails, commit boundaries, precision and RNG identity."""

from collections import deque
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
from experiments.robotwin.multiattempt import DEVICE_CONTINUOUS_NOISE_PROTOCOL, validate_attempts
from fastwam.zeva.attempt_protocol import ContinuousModelNoiseStream
from fastwam.zeva.cross_attempt_cache import (
    ALL_CTE_FRAMES, COMPLETE_POLICY_WINDOWS, REFERENCE_PAIRING_METRIC, LEGACY_PAIRING_METRIC,
    CrossAttemptPIMCache, build_episode_pairings, file_sha256,
)
from fastwam.zeva.ego_boundary_cache import build_ego_boundary_cache, EgoBoundaryCache
from fastwam.zeva.lifecycle import CausalMemoryLifecycle, LifecycleConfig
from fastwam.zeva.memory import PersistentInteractionMemory, PersistentInteractionMemoryConfig
from fastwam.zeva.prepared_stage2 import PreparedPIMCache, PreparedZevaStage2Dataset, StaticEpisodeContextCache, STATIC_CONTEXT_FORMAT
from fastwam.zeva_memory_contract import pim_memory_contract, validate_checkpoint_pim_memory_contract
from test_ego_effect_pipeline import BUILDER, _CausalCTE, make_dataset
from test_slot_rng_evaluation import _audited_rows


def _reference_lifecycle():
    memory = PersistentInteractionMemory(PersistentInteractionMemoryConfig(
        phase_dim=2, effect_dim=2, capacity=256, top_k=256, cross_attempt_only=True,
        max_past_attempts=4, max_entries_per_attempt=64, compression="similarity_merge",
        uniform_indexing="reference_round", commit_timing="attempt_reset", storage_device="policy",
    ))
    lifecycle = CausalMemoryLifecycle(memory, LifecycleConfig(memory_semantics="predicted_effect_trace"))
    lifecycle.reset_episode("task", "scene")
    return lifecycle


def test_reset_publishes_only_the_previous_attempt_and_terminal_discards_pending(monkeypatch):
    lifecycle = _reference_lifecycle()
    phase, effect = torch.tensor([2., 0.]), torch.tensor([7., 0.])
    # Even on the CPU test host, fail any forced host-transfer call: policy
    # storage must retain the input device through append, merge and readout.
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("forced CPU transfer")))
    lifecycle.observe_predicted_effect(phase, effect)
    effect.add_(100)  # Detached storage also owns its tensors.
    lifecycle.end_attempt()
    assert len(lifecycle.pim) == 0 and lifecycle.pim.pending_count == 1
    assert lifecycle.bit.tensors().valid.sum() == 1
    lifecycle.reset_attempt(1)
    assert len(lifecycle.pim) == 1 and lifecycle.pim.pending_count == 0
    assert not lifecycle.bit.tensors().valid.any()
    inputs = lifecycle.memory_inputs(phase, torch.zeros(1, 2))
    torch.testing.assert_close(inputs["pim_effects"][0, 0], torch.tensor([7., 0.]))
    assert inputs["pim_phases"].device == phase.device
    lifecycle.observe_predicted_effect(phase, torch.tensor([11., 0.]))
    lifecycle.end_attempt()
    assert lifecycle.pim.pending_count == 1 and lifecycle.pim.past_attempt_count == 1
    lifecycle.reset_episode("task", "next_scene")
    assert len(lifecycle.pim) == lifecycle.pim.pending_count == lifecycle.pim.past_attempt_count == 0


def test_reference_storage_retains_four_completed_attempts_and_evicts_whole_oldest():
    lifecycle = _reference_lifecycle()
    for attempt in range(5):
        lifecycle.observe_predicted_effect(torch.tensor([float(attempt + 1), 0.]), torch.tensor([float(attempt + 1), 0.]))
        lifecycle.end_attempt()
        lifecycle.reset_attempt(attempt + 1)
    assert lifecycle.pim.past_attempt_count == 4
    assert lifecycle.pim.entries[0].observation_count == 4
    assert lifecycle.pim.entries[0].first_attempt_id == 1
    assert lifecycle.pim.entries[0].last_attempt_id == 4
    torch.testing.assert_close(lifecycle.pim.entries[0].effect, torch.tensor([3.5, 0.]))


def test_predictive_finalization_does_not_take_a_terminal_observation_or_commit():
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy._attempt_finalized, policy._ego_cte = False, True
    policy.zeva_mode, policy.cte = "pim_on", SimpleNamespace()
    policy.lifecycle = _reference_lifecycle()
    policy.lifecycle.observe_predicted_effect(torch.ones(2), torch.ones(2))
    policy._transition_actions = [torch.zeros(14)]
    policy.finalize_attempt(SimpleNamespace())  # No get_obs API is needed.
    assert policy._attempt_finalized and not policy._transition_actions
    assert policy.lifecycle.pim.pending_count == 1 and len(policy.lifecycle.pim) == 0


def test_memory_v2_cannot_be_replayed_as_v1_or_changed_by_a_compression_ablation():
    common = dict(cross_attempt_only=True, compression="uniform", uniform_indexing="reference_round",
                  max_past_attempts=4, pim_max_entries=256)
    legacy = pim_memory_contract(common)
    reference = pim_memory_contract({**common, "commit_timing": "attempt_reset", "storage_device": "policy"})
    assert reference["schema"] == "iclwam_cross_attempt_pim_memory_v2"
    assert legacy["schema"] == "iclwam_cross_attempt_pim_memory_v1"
    for target in (legacy, {**reference, "commit_timing": "attempt_end"}):
        with pytest.raises(ValueError, match="mismatch"):
            validate_checkpoint_pim_memory_contract({"pim_memory_contract": reference}, target, allow_compression_override=True)
    with pytest.raises(ValueError, match="explicitly record"):
        validate_checkpoint_pim_memory_contract({"config": {"model": {"zeva": {"memory": {
            **common, "commit_timing": "attempt_reset", "storage_device": "policy"}}}}}, reference)


def _short_source(tmp_path, *, legacy=False):
    dataset, _ = make_dataset((49, 49, 49))
    ids = [episode.episode_id for episode in dataset.episodes]
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({"format": "robotwin_semantic_task_map_v1",
        "episode_to_semantic_task": {str(i): "beat_block_hammer" for i in range(3)}}))
    boundary = tmp_path / "boundary"
    build_ego_boundary_cache(dataset, _CausalCTE(), boundary,
        identity={"cte_checkpoint_sha256": "cte", "dataset_stats_sha256": "stats",
                  "semantic_task_map_sha256": file_sha256(semantic), "dataset_path": "/source",
                  "data_scope": {"task_names": ["beat_block_hammer"]}, "export_tf32": False,
                  **({"schema_version": "zeva_ego_boundary_cache_v1", "export_scope": "decision_only_v1"} if legacy else {})},
        episode_splits={"train": ids[:2], "validation": ids[2:]})
    return boundary, semantic


def test_all_cte_frames_survive_export_pim_and_native_padded_policy_join(tmp_path):
    boundary, semantic = _short_source(tmp_path)
    source = EgoBoundaryCache(boundary)
    assert source.manifest.schema_version == "zeva_ego_boundary_cache_v2"
    assert source.manifest.export_scope == ALL_CTE_FRAMES
    assert [row["raw_step"] for row in source.rows[:3]] == [0, 24, 48]
    assert source.rows[2]["decision_valid"] is False and source.rows[2]["policy_valid"] is False
    output = tmp_path / "pim"
    manifest = BUILDER.build_cache(phase_cache=boundary, output=output, semantic_map=semantic,
                                   allow_reduced_curriculum=True)
    assert manifest["format"] == "zeva_stage2_cross_attempt_pim_v4"
    assert manifest["num_queries"] == 6 and manifest["max_past_attempts"] == 4
    assert manifest["data_scope"] == source.manifest.data_scope
    pim = PreparedPIMCache(output)
    row_index = next(i for i in range(len(pim)) if pim.get(i)["raw_step"] == 48)
    row = pim.get(row_index)
    assert row["current_effect_mask"] and row["current_effect"].tolist() == [23., 24.]
    assert row["pim_mask"].sum() == 3
    assert row["pim_effects"][row["pim_mask"], 0].tolist() == [21., 22., 23.]

    class NativePaddedBase:
        def __getitem__(self, index):
            assert index == row["window_index"]
            return {"video": torch.zeros(3, 9, 2, 2), "action": torch.full((32, 14), 2.75),
                    "action_is_pad": torch.ones(32, dtype=torch.bool),
                    "image_is_pad": torch.tensor([False] + [True] * 8),
                    "dataset_index": index, "episode_id": row["episode_id"], "task_id": row["task_id"],
                    "episode_step": 48, "episode_num_steps": 49}

    static = tmp_path / "static.pt"
    torch.save({"format": STATIC_CONTEXT_FORMAT, "episode_ids": list(pim.episode_ids),
                "task_contexts": torch.ones(len(pim.episode_ids), 2)}, static)
    joined = PreparedZevaStage2Dataset(NativePaddedBase(), pim, StaticEpisodeContextCache(static))[row_index]
    assert not joined["action_valid"].any()
    assert torch.all(joined["action"] == 2.75)  # Terminal-repeat targets remain available to full-H32 losses.
    assert joined["current_effect_mask"]
    with pytest.raises(ValueError, match="query_scope differs"):
        pim._cross_attempt.validate_training_curriculum(history_pairing="reference_nearest",
            allow_reduced_curriculum=True, query_scope=COMPLETE_POLICY_WINDOWS)


def test_old_complete_window_cache_remains_explicit_and_readable(tmp_path):
    boundary, semantic = _short_source(tmp_path, legacy=True)
    output = tmp_path / "legacy"
    manifest = BUILDER.build_cache(phase_cache=boundary, output=output, semantic_map=semantic,
        query_scope=COMPLETE_POLICY_WINDOWS, pairing_metric=LEGACY_PAIRING_METRIC,
        max_past_attempts=3, allow_reduced_curriculum=True)
    assert manifest["format"] == "zeva_stage2_cross_attempt_pim_v3"
    old = CrossAttemptPIMCache(output)
    assert len(old) == 2  # Only raw_step=0 has a complete H32 target.
    assert old.get(0, history_kind="matched")["pim_mask"].sum() == 2  # Pure terminal frame was never exported.
    manifest["format"] = "zeva_stage2_cross_attempt_pim_v4"
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="all CTE frames"):
        CrossAttemptPIMCache(output)


def test_reference_ties_use_source_order_while_legacy_uses_episode_names():
    query = {"episode_id": "query", "task_id": "task", "condition": "clean", "initial_phase": [1., 0.]}
    histories = [{**query, "episode_id": name, "split": "train"} for name in ("z_first", "a_second")]
    strict = build_episode_pairings({"query": query}, histories, max_histories=4, use_conditions=True,
        history_pairing="reference_nearest", pairing_metric=REFERENCE_PAIRING_METRIC)
    legacy = build_episode_pairings({"query": query}, histories, max_histories=4, use_conditions=True,
        history_pairing="reference_nearest", pairing_metric=LEGACY_PAIRING_METRIC)
    assert strict["query"]["matched"] == strict["query"]["far"] == [0]
    assert legacy["query"]["matched"] == legacy["query"]["far"] == [1]


def test_device_stream_uses_policy_device_and_cannot_load_legacy_state(tmp_path):
    state = tmp_path / "slot.json"
    identity = {"policy_device": "cpu"}
    stream = ContinuousModelNoiseStream(19, protocol=DEVICE_CONTINUOUS_NOISE_PROTOCOL, device="cpu",
        state_path=state, run_id="run", task_name="a", checkpoint_identity=identity)
    expected = torch.Generator(device="cpu").manual_seed(19)
    torch.testing.assert_close(stream.next_noise((1, 32, 14)), torch.randn(1, 32, 14, generator=expected))
    stream.save_state()
    assert stream.snapshot()["protocol"] == DEVICE_CONTINUOUS_NOISE_PROTOCOL
    with pytest.raises(ValueError, match="identity mismatch"):
        ContinuousModelNoiseStream(19, state_path=state, run_id="run", task_name="b", checkpoint_identity=identity)
    resumed = ContinuousModelNoiseStream(19, protocol=DEVICE_CONTINUOUS_NOISE_PROTOCOL, device="cpu",
        state_path=state, run_id="run", task_name="b", checkpoint_identity=identity)
    torch.testing.assert_close(resumed.next_noise((1, 32, 14)), torch.randn(1, 32, 14, generator=expected))


def test_device_rng_audit_binds_noise_to_the_loaded_policy_device(tmp_path):
    manifest, sha, _, rows = _audited_rows(tmp_path)
    manifest["evaluation_protocol"]["model_rng_protocol"] = DEVICE_CONTINUOUS_NOISE_PROTOCOL
    for row in rows:
        row["checkpoint_identity"]["policy_device"] = "cpu"
        for key in ("seed_stream_before", "seed_stream"):
            row[key]["protocol"] = DEVICE_CONTINUOUS_NOISE_PROTOCOL
    validate_attempts(manifest, sha, rows)
    for row in rows:
        row["checkpoint_identity"]["policy_device"] = "cuda:0"
    with pytest.raises(ValueError, match="actual policy device"):
        validate_attempts(manifest, sha, rows)


def test_reference_task_language_uses_first_instruction_until_attempt_reset():
    calls = []
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.model = SimpleNamespace(device="cpu", zeva_numerical_contract="zeva_ego_v1")
    policy._task_language_memory = SimpleNamespace(
        encoder=SimpleNamespace(encode=lambda text, **kwargs: calls.append(text) or torch.ones(1, 2)),
        retrieve=lambda features: SimpleNamespace(values=features))
    first = policy._resolve_task_language("first")
    assert policy._resolve_task_language("later wording") is first and calls == ["first"]
    policy.pending_actions, policy.lifecycle = deque(), None
    policy.reset_timing_rollout = lambda: None
    policy.begin_attempt(1)
    assert policy._resolve_task_language("later wording") is not first
    assert calls == ["first", "later wording"]


def test_v2_boundary_cannot_be_silently_relabelled_as_old_complete_window_history(tmp_path):
    boundary, semantic = _short_source(tmp_path)
    with pytest.raises(ValueError, match="v1 decision-only boundary"):
        BUILDER.build_cache(phase_cache=boundary, output=tmp_path / "wrong", semantic_map=semantic,
            query_scope=COMPLETE_POLICY_WINDOWS, pairing_metric=LEGACY_PAIRING_METRIC,
            max_past_attempts=3, allow_reduced_curriculum=True)


@pytest.mark.parametrize("problem", [None, "runtime", "checkpoint", "missing_checkpoint_marker"])
def test_rollout_audit_rejects_numerical_recipe_changes(tmp_path, problem):
    manifest, sha, _, rows = _audited_rows(tmp_path)
    contract = {"prefix_layout": "zeva_ego", "effect_source": "current_prediction",
                "memory_semantics": "predicted_effect_trace", "prior_variant": "zeva_ego",
                "skip_get_obs_within_replan": True, "numerical_contract": "zeva_ego_v1"}
    manifest["evaluation_protocol"]["policy_contracts"] = {"new_on": contract}
    for row in rows:
        row.update(contract)
        row["checkpoint_identity"].update(numerical_contract="zeva_ego_v1", checkpoint_numerical_contract="zeva_ego_v1")
        if problem == "runtime":
            row["numerical_contract"] = "legacy_v1"
        elif problem == "checkpoint":
            row["checkpoint_identity"]["checkpoint_numerical_contract"] = "legacy_v1"
        elif problem == "missing_checkpoint_marker":
            row["checkpoint_identity"].pop("checkpoint_numerical_contract")
    if problem is None:
        validate_attempts(manifest, sha, rows)
    else:
        with pytest.raises(ValueError, match="numerical_contract"):
            validate_attempts(manifest, sha, rows)


@pytest.mark.parametrize("family", ["old", "new"])
def test_cte_epoch40_cli_selects_only_one_family_and_never_base(tmp_path, monkeypatch, capsys, family):
    import runpy
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "eval", "--checkpoint", "base.pt", "--dataset-stats", "stats.json",
        "--old-addon-checkpoint", "old.pt", "--new-addon-checkpoint", "new.pt",
        "--old-cte-checkpoint", "old_cte.pt", "--new-cte-checkpoint", "new_cte.pt",
        "--task-context-bank", "bank.pt", "--task-context-retrieval-checkpoint", "head.pt",
        "--task-language-encoder", "language.pt", "--old-effect-source", "current_prediction",
        "--old-task-config", "robotwin_zeva_ego_pim_4e95_3cam_384",
        f"--{family}-cte-exploratory-epoch40", "--conditions", "base", "old", "new_off", "new_on",
        "--output-dir", str(output), "--dry-run",
    ])
    runpy.run_path(str(root / "scripts/eval_zeva_robotwin_manifest.py"))["main"]()
    capsys.readouterr()
    plan = json.loads((output / "command_plan.json").read_text())
    expected = {"base": False, "old": family == "old", "new_off": family == "new", "new_on": family == "new"}
    assert {condition: value["cte_exploratory_epoch40"] for condition, value in plan["policy_contracts"].items()} == expected
    for item in plan["commands"]:
        fields = {key: json.loads(value) for arg in item["command"] if "=" in arg for key, value in [arg.split("=", 1)]}
        assert fields.get("++model.zeva.cte.exploratory_epoch40", False) is expected[item["condition"]]
