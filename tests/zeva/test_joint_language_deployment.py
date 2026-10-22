"""Deploy the actual joint bundle with instruction-only task retrieval."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
from experiments.robotwin.multiattempt import validate_attempts
from test_joint_foundation import _save_joint_bundle
from test_slot_rng_evaluation import _audited_rows


ROOT = Path(__file__).resolve().parents[2]


def test_task_language_session_reads_only_instruction_and_resets_for_retry():
    calls = []
    result = SimpleNamespace(values=torch.randn(1, 256))
    def encode(instruction, *, device):
        calls.append(instruction)
        return torch.ones(1, 4, device=device)
    def retrieve(features):
        assert features.shape == (1, 4)
        return result
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.model = SimpleNamespace(device="cpu")
    policy._task_language_memory = SimpleNamespace(encoder=SimpleNamespace(encode=encode), retrieve=retrieve)
    assert policy._resolve_task_language("pick the cup") is result
    # Foundation changes do not enter the task-query feature path.
    policy.model.some_finetuned_weights = torch.randn(99)
    assert policy._resolve_task_language("pick the cup") is result
    assert calls == ["pick the cup"]
    with pytest.raises(ValueError, match="instruction changed"):
        policy._resolve_task_language("open the drawer")
    policy.pending_actions, policy.lifecycle = deque(), None
    policy.reset_timing_rollout = lambda: None
    policy.begin_attempt(1)
    assert policy._resolve_task_language("pick the cup") is result
    assert calls == ["pick the cup", "pick the cup"]


def test_manifest_validates_joint_foundation_before_launching_rollouts(tmp_path):
    check = runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["_foundation_artifact"]
    addon = tmp_path / "step_000501_addon.pt"
    _save_joint_bundle(addon)
    identity = check(str(addon), "ego_joint_warmup")
    assert Path(identity["foundation_checkpoint"]).is_file()
    assert len(identity["foundation_sha256"]) == 64
    with pytest.raises(ValueError, match="recipe differs"):
        check(str(addon), "ego_adapter_warmup")
    Path(identity["foundation_checkpoint"]).unlink()
    with pytest.raises(FileNotFoundError, match="foundation"):
        check(str(addon), "ego_joint_warmup")


@pytest.mark.parametrize("problem", [None, "missing_foundation", "wrong_foundation", "encoder", "query_scope", "mode"])
def test_rollout_audit_binds_joint_foundation_and_task_language_artifacts(tmp_path, problem):
    manifest, sha, _, rows = _audited_rows(tmp_path)
    contract = {"prefix_layout": "zeva_ego", "effect_source": "current_prediction",
                "memory_semantics": "predicted_effect_trace", "prior_variant": "zeva_ego",
                "skip_get_obs_within_replan": True, "training_recipe": "ego_joint_warmup",
                "task_context_mode": "task_language", "foundation_sha256": "e" * 64}
    manifest["evaluation_protocol"]["policy_contracts"] = {"new_on": contract}
    identity = {"mode": "task_language", "bank_sha256": "1" * 64, "head_sha256": "2" * 64,
                "language_encoder_sha256": "3" * 64, "readout_kind": "frozen_task_input_embedding_mean_v1",
                "query_scope": "language_predicted_task", "temperature": 1.0, "top_k": 5}
    for row in rows:
        row.update({key: value for key, value in contract.items() if key != "foundation_sha256"})
        row["checkpoint_identity"].update(training_recipe="ego_joint_warmup", foundation_sha256="e" * 64,
                                           task_context=deepcopy(identity))
    # Keep the recorded identity consistent across rows: otherwise a generic
    # mixed-checkpoint check could mask the new contract-specific validation.
    for row in rows:
        if problem == "missing_foundation":
            row["checkpoint_identity"].pop("foundation_sha256")
        elif problem == "wrong_foundation":
            row["checkpoint_identity"]["foundation_sha256"] = "f" * 64
        elif problem == "encoder":
            row["checkpoint_identity"]["task_context"]["language_encoder_sha256"] = ""
        elif problem == "query_scope":
            row["checkpoint_identity"]["task_context"]["query_scope"] = "ground_truth_task"
        elif problem == "mode":
            row["task_context_mode"] = "static"
    if problem is None:
        validate_attempts(manifest, sha, rows)
    else:
        expected = {
            "missing_foundation": "joint foundation SHA256",
            "wrong_foundation": "Loaded foundation differs",
            "encoder": "language_encoder_sha256",
            "query_scope": "Task-language retrieval contract differs",
            "mode": "task_context_mode differs",
        }
        with pytest.raises(ValueError, match=expected[problem]):
            validate_attempts(manifest, sha, rows)


def test_joint_evaluation_cannot_select_stale_initial_readout_retrieval():
    resolve = runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["_policy_contract"]
    with pytest.raises(ValueError, match="task_language"):
        resolve("robotwin_zeva_ego_pim_3cam_384", [], "zeva_ego", "current_prediction", task_context_mode="static")
    old = resolve("robotwin_zeva_ego_pim_frozen_3cam_384", [], "zeva_ego", "current_prediction")
    assert old["training_recipe"] == "ego_adapter_warmup" and old["task_context_mode"] == "static"
