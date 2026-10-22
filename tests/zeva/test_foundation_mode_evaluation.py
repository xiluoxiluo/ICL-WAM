"""The single foundation switch survives the real evaluation entry points."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
from hydra import compose, initialize_config_dir

from experiments.robotwin.multiattempt import validate_attempts
from test_slot_rng_evaluation import _audited_rows


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode,recipe", [("frozen", "ego_adapter_warmup"), ("joint", "ego_joint_warmup")])
def test_manifest_uses_same_language_pipeline_and_explicit_foundation_mode(tmp_path, monkeypatch, capsys, mode, recipe):
    output = tmp_path / mode
    monkeypatch.setattr(sys, "argv", [
        "eval", "--checkpoint", "base.pt", "--dataset-stats", "stats.json",
        "--new-cte-checkpoint", "cte.pt", "--new-addon-checkpoint", "addon.pt",
        "--new-task-context-bank", "bank.pt", "--new-task-context-retrieval-checkpoint", "classifier.pt",
        "--new-task-language-encoder", "encoder.pt", "--new-foundation-mode", mode,
        "--conditions", "new_off", "new_on", "--policy-text-cache-size", "7",
        "--output-dir", str(output), "--dry-run",
    ])
    runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["main"]()
    capsys.readouterr()
    plan = json.loads((output / "command_plan.json").read_text())
    assert plan["policy_contracts"]["new_on"] == plan["policy_contracts"]["new_off"]
    contract = plan["policy_contracts"]["new_on"]
    assert contract["foundation_mode"] == mode
    assert contract["training_recipe"] == recipe
    assert contract["task_context_mode"] == "task_language"
    assert contract["policy_text_encoder_device"] == "cpu"
    assert contract["policy_text_cache_size"] == 7
    for item in plan["commands"]:
        overrides = {key: json.loads(value) for arg in item["command"] if "=" in arg
                     for key, value in [arg.split("=", 1)]}
        assert overrides["++foundation_mode"] == mode
        assert overrides["EVALUATION.policy_text_encoder_device"] == "cpu"
        assert overrides["EVALUATION.policy_text_cache_size"] == 7
        assert overrides["EVALUATION.cross_memory_enabled"] is (item["condition"] == "new_on")


@pytest.mark.parametrize("mode", ["frozen", "joint"])
def test_sixteen_rank_recipe_preserves_effective_batch_and_query(mode):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[
            "task=robotwin_zeva_ego_pim_16gpu_3cam_384", f"foundation_mode={mode}",
        ])
    assert cfg.batch_size * cfg.gradient_accumulation_steps * 16 == 256
    assert cfg.zeva_optimizer_sharding == "zero1"
    assert cfg.model.zeva.task_context.mode == "task_language"
    assert cfg.model.zeva.freeze_fastwam is (mode == "frozen")


def test_base_stays_original_frozen_foundation_when_old_task_defaults_to_joint(tmp_path, monkeypatch, capsys):
    output = tmp_path / "base"
    monkeypatch.setattr(sys, "argv", [
        "eval", "--checkpoint", "base.pt", "--dataset-stats", "stats.json",
        "--old-task-config", "robotwin_zeva_ego_pim_3cam_384", "--old-foundation-mode", "frozen",
        "--old-effect-source", "current_prediction",
        "--conditions", "base", "--output-dir", str(output), "--dry-run",
    ])
    runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["main"]()
    capsys.readouterr()
    plan = json.loads((output / "command_plan.json").read_text())
    contract = plan["policy_contracts"]["base"]
    assert contract["foundation_mode"] == "frozen"
    assert contract["training_recipe"] == "frozen_parent"
    assert all(not arg.startswith("++foundation_mode=")
               for item in plan["commands"] for arg in item["command"])


@pytest.mark.parametrize("override", ["foundation_mode=frozen", "model.zeva.freeze_fastwam=true",
                                      "model.zeva.training_recipe=ego_adapter_warmup"])
def test_manifest_rejects_hidden_foundation_overrides(tmp_path, monkeypatch, capsys, override):
    monkeypatch.setattr(sys, "argv", [
        "eval", "--checkpoint", "base.pt", "--dataset-stats", "stats.json",
        "--conditions", "base", "--output-dir", str(tmp_path / "run"),
        "--dry-run", "--override", override,
    ])
    with pytest.raises(SystemExit):
        runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["main"]()
    assert "foundation-mode" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("problem", [None, "foundation_mode", "device", "cache", "missing_identity", "weights", "dtype"])
def test_rollout_audit_checks_actual_mode_and_policy_text_identity(tmp_path, problem):
    manifest, sha, _, rows = _audited_rows(tmp_path)
    contract = {"prefix_layout": "zeva_ego", "effect_source": "current_prediction",
                "memory_semantics": "predicted_effect_trace", "prior_variant": "zeva_ego",
                "skip_get_obs_within_replan": True, "training_recipe": "ego_adapter_warmup",
                "task_context_mode": "pooling", "foundation_mode": "frozen",
                "policy_text_encoder_device": "cpu", "policy_text_cache_size": 16}
    manifest["evaluation_protocol"]["policy_contracts"] = {"new_on": contract}
    text_identity = {"schema": "iclwam_policy_text_v1",
                     "prompt_semantics": "wan_whitespace_special_tokens_zero_padding_visible_v1",
                     "encoder_device": "cpu", "encoder_dtype": "torch.float32",
                     "encoder_sha256": "e" * 64, "tokenizer_sha256": "f" * 64,
                     "tokenizer_max_length": 128, "cache_size": 16, "cache_storage_device": "cpu"}
    for row in rows:
        row.update(contract)
        row["checkpoint_identity"]["policy_text"] = deepcopy(text_identity)
        if problem == "foundation_mode":
            row["foundation_mode"] = "joint"
        elif problem == "device":
            row["policy_text_encoder_device"] = "cuda"
        elif problem == "cache":
            row["checkpoint_identity"]["policy_text"]["cache_size"] = 0
        elif problem == "missing_identity":
            del row["checkpoint_identity"]["policy_text"]
        elif problem == "weights":
            row["checkpoint_identity"]["policy_text"]["encoder_sha256"] = ""
        elif problem == "dtype":
            row["checkpoint_identity"]["policy_text"]["encoder_dtype"] = "unreported"
    if problem is None:
        assert len(validate_attempts(manifest, sha, rows)) == 3
    else:
        with pytest.raises(ValueError):
            validate_attempts(manifest, sha, rows)
