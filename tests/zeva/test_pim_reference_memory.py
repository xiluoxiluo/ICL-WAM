"""Reference compression preserves raw effects and cannot silently change old runs."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
import torch

from fastwam.zeva.memory import (
    PersistentInteractionMemory, PersistentInteractionMemoryConfig, uniform_history_indices,
)
from fastwam.zeva.cross_attempt_cache import CrossAttemptPIMCache, LEGACY_EGO_CROSS_ATTEMPT_CACHE_FORMAT
from fastwam.zeva_memory_contract import (
    pim_memory_contract, validate_checkpoint_pim_memory_contract, validate_pim_memory_contract,
)
from experiments.robotwin.multiattempt import validate_attempts
from test_ego_effect_pipeline import BUILDER, make_boundary_cache
from test_slot_rng_evaluation import _audited_rows


ROOT = Path(__file__).resolve().parents[2]


def _reference_contract(**overrides):
    return pim_memory_contract({
        "cross_attempt_only": True, "compression": "similarity_merge",
        "uniform_indexing": "reference_round", "merge_threshold": .95, **overrides,
    })


def test_reference_uniform_rounds_midpoints_and_preserves_explicit_legacy_floor():
    assert uniform_history_indices(8, 3, rounding="reference_round") == [0, 4, 7]
    assert uniform_history_indices(8, 3) == [0, 3, 7]
    assert uniform_history_indices(4, 3, rounding="reference_round") == [0, 2, 3]
    assert uniform_history_indices(2, 4, rounding="reference_round") == [0, 1]


def test_two_level_merge_keeps_raw_weighted_centroids_and_evicts_whole_attempts():
    memory = PersistentInteractionMemory(PersistentInteractionMemoryConfig(
        phase_dim=2, effect_dim=2, capacity=6, top_k=6, cross_attempt_only=True,
        max_past_attempts=2, max_entries_per_attempt=3, compression="similarity_merge",
        uniform_indexing="reference_round", merge_threshold=.95,
    ))
    memory.reset_episode("task", episode_id="scene")

    def append(attempt, index, phase, effect):
        memory.append_completed(task_cluster="task", phase=torch.tensor(phase).float(),
                                effect=torch.tensor(effect).float(), attempt_id=attempt,
                                transition_index=index)

    append(0, 0, [2, 0], [10, 0])
    append(0, 1, [4, 0], [20, 0])
    append(0, 2, [0, 2], [0, 4])
    assert len(memory) == 0 and memory.pending_count == 3
    memory.commit_attempt()
    assert len(memory) == 2
    assert memory.entries[0].observation_count == 2
    torch.testing.assert_close(memory.entries[0].phase, torch.tensor([3., 0.]))
    torch.testing.assert_close(memory.entries[0].effect, torch.tensor([15., 0.]))

    append(1, 0, [6, 0], [30, 0])
    assert memory.entries[0].observation_count == 2  # Current trace is still private.
    memory.commit_attempt()
    assert memory.entries[0].observation_count == 3
    torch.testing.assert_close(memory.entries[0].phase, torch.tensor([4., 0.]))
    torch.testing.assert_close(memory.entries[0].effect, torch.tensor([20., 0.]))
    assert (memory.entries[0].first_attempt_id, memory.entries[0].last_attempt_id) == (0, 1)

    append(2, 0, [10, 0], [50, 0])
    memory.commit_attempt()
    memory.commit_attempt()  # Idempotent; no historical count is duplicated.
    assert len(memory) == 1 and memory.past_attempt_count == 2
    assert memory.entries[0].observation_count == 2
    assert (memory.entries[0].first_attempt_id, memory.entries[0].last_attempt_id) == (1, 2)
    torch.testing.assert_close(memory.entries[0].phase, torch.tensor([8., 0.]))
    torch.testing.assert_close(memory.entries[0].effect, torch.tensor([40., 0.]))
    memory.reset_episode("task", episode_id="next_scene")
    assert len(memory) == memory.pending_count == memory.past_attempt_count == 0


def test_predictive_cache_versions_bind_actual_rounding_and_keep_legacy_readable(tmp_path):
    boundary, semantic = make_boundary_cache(tmp_path)
    expected = {"reference_round": ([21., 23., 24.], "zeva_stage2_cross_attempt_pim_v3"),
                "legacy_floor": ([21., 22., 24.], LEGACY_EGO_CROSS_ATTEMPT_CACHE_FORMAT)}
    for indexing, (effects, format_name) in expected.items():
        output = tmp_path / indexing
        options = {} if indexing == "reference_round" else {"uniform_indexing": indexing}
        manifest = BUILDER.build_cache(phase_cache=boundary, output=output, semantic_map=semantic,
                                       max_entries_per_attempt=3, allow_reduced_curriculum=True, **options)
        assert manifest["format"] == format_name
        assert manifest["history_compression"] == "uniform" and manifest["uniform_indexing"] == indexing
        cache = CrossAttemptPIMCache(output)
        row = cache.get(0, history_kind="matched", history_count=1)
        assert row["pim_effects"][row["pim_mask"], 0].tolist() == effects
        manifest.pop("uniform_indexing")
        manifest.pop("history_compression")
        (output / "manifest.json").write_text(json.dumps(manifest))
        if indexing == "reference_round":
            with pytest.raises(ValueError, match="explicit uniform/reference_round"):
                CrossAttemptPIMCache(output)
        else:
            assert CrossAttemptPIMCache(output).uniform_indexing == "legacy_floor"


def test_old_artifacts_need_legacy_target_and_ablation_cannot_change_rounding():
    reference = _reference_contract()
    legacy = _reference_contract(compression="uniform", uniform_indexing="legacy_floor")
    assert validate_checkpoint_pim_memory_contract({}, legacy) == legacy
    with pytest.raises(ValueError, match="no PIM memory contract"):
        validate_checkpoint_pim_memory_contract({}, reference)
    uniform = _reference_contract(compression="uniform")
    payload = {"pim_memory_contract": reference}
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_checkpoint_pim_memory_contract(payload, uniform)
    assert validate_checkpoint_pim_memory_contract(payload, uniform, allow_compression_override=True) == reference
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_checkpoint_pim_memory_contract(payload, legacy, allow_compression_override=True)
    corrupted = deepcopy(reference)
    corrupted["phase_weight"] = 1.5
    with pytest.raises(ValueError):
        validate_pim_memory_contract(corrupted)
    with pytest.raises(ValueError, match="disagrees with its saved configuration"):
        validate_checkpoint_pim_memory_contract({**payload, "config": {
            "model": {"zeva": {"memory": {"cross_attempt_only": True}}}}}, reference)


@pytest.mark.parametrize("problem", [None, "served_compression", "checkpoint_compression", "override", "missing_foundation", "required_flag"])
def test_audit_binds_pim_semantics_and_frozen_stage_b_inherited_foundation(tmp_path, problem):
    manifest, sha, _, rows = _audited_rows(tmp_path)
    memory = _reference_contract()
    contract = {
        "prefix_layout": "zeva_ego", "effect_source": "current_prediction",
        "memory_semantics": "predicted_effect_trace", "prior_variant": "zeva_ego",
        "skip_get_obs_within_replan": True, "training_recipe": "ego_adapter_warmup",
        "task_context_mode": "pooling", "foundation_mode": "frozen",
        "foundation_checkpoint_required": True, "foundation_sha256": "e" * 64,
        "pim_memory_contract": memory, "checkpoint_pim_memory_contract": memory,
        "pim_compression_override": False,
    }
    manifest["evaluation_protocol"]["policy_contracts"] = {"new_on": contract}
    for row in rows:
        row.update(deepcopy(contract))
        row["checkpoint_identity"].update({
            "training_recipe": "ego_adapter_warmup", "foundation_checkpoint_required": True,
            "foundation_sha256": "e" * 64, "pim_memory_contract": deepcopy(memory),
            "checkpoint_pim_memory_contract": deepcopy(memory), "pim_compression_override": False,
        })
        if problem == "served_compression":
            row["pim_memory_contract"] = _reference_contract(compression="uniform")
        elif problem == "checkpoint_compression":
            row["checkpoint_identity"]["checkpoint_pim_memory_contract"] = _reference_contract(compression="uniform")
        elif problem == "override":
            row["checkpoint_identity"]["pim_compression_override"] = True
        elif problem == "missing_foundation":
            del row["checkpoint_identity"]["foundation_sha256"]
        elif problem == "required_flag":
            row["checkpoint_identity"]["foundation_checkpoint_required"] = False
    if problem is None:
        assert len(validate_attempts(manifest, sha, rows)) == 3
    else:
        with pytest.raises(ValueError):
            validate_attempts(manifest, sha, rows)


def test_manifest_records_explicit_uniform_ablation_without_weakening_legacy_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "eval", "--checkpoint", "base.pt", "--dataset-stats", "stats.json",
        "--new-addon-checkpoint", "addon.pt", "--new-cte-checkpoint", "cte.pt",
        "--new-task-context-bank", "bank.pt", "--new-task-context-retrieval-checkpoint", "classifier.pt",
        "--new-task-language-encoder", "encoder.pt", "--new-pim-compression", "uniform",
        "--conditions", "new_off", "new_on", "--output-dir", str(tmp_path / "run"), "--dry-run",
    ])
    runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["main"]()
    capsys.readouterr()
    plan = json.loads((tmp_path / "run/command_plan.json").read_text())
    assert plan["policy_contracts"]["new_on"] == plan["policy_contracts"]["new_off"]
    contract = plan["policy_contracts"]["new_on"]
    assert contract["pim_compression_override"] is True
    assert contract["pim_memory_contract"] == _reference_contract(compression="uniform")
    for item in plan["commands"]:
        fields = {key: json.loads(value) for arg in item["command"] if "=" in arg
                  for key, value in [arg.split("=", 1)]}
        assert fields["++model.zeva.memory.compression"] == "uniform"
        assert fields["++model.zeva.memory.allow_compression_override"] is True
