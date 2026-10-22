"""Evaluation plans preserve the selected prefix topology and reader capacity."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("layout", ["legacy_merged", "zeva_ego"])
def test_fixed_attempt_cli_passes_explicit_prefix_contract(monkeypatch, layout):
    main = runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_fixed_attempts.py"))["main"]
    monkeypatch.setattr(sys, "argv", [
        "eval", "--task", "test", "--seed", "0", "--ckpt", "base.pt",
        "--cte-checkpoint", "cte.pt", "--addon-checkpoint", "addon.pt",
        "--sim-task", "robotwin_zeva_fastwam_cross_attempt_3cam_384",
        "--prefix-layout", layout, "--effect-source", "latest_completed",
    ])
    commands = []
    monkeypatch.setattr(subprocess, "call", lambda cmd, cwd: commands.append(cmd) or 0)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
    assert f"model.zeva.adapter.prefix_layout={layout}" in commands[0]
    assert "model.zeva.adapter.effect_source=latest_completed" in commands[0]
    assert "EVALUATION.pim_top_k=192" in commands[0]


@pytest.mark.parametrize("old_task,expected_old_slots", [
    ("robotwin_zeva_fastwam_pim_3cam_384", 4),
    ("robotwin_zeva_fastwam_cross_attempt_3cam_384", 192),
])
def test_manifest_uses_per_condition_layout_and_configured_reader_capacity(
    tmp_path, monkeypatch, capsys, old_task, expected_old_slots,
):
    main = runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["main"]
    monkeypatch.setattr(sys, "argv", [
        "eval", "--checkpoint", "base.pt", "--dataset-stats", "stats.json",
        "--cte-checkpoint", "cte.pt", "--old-addon-checkpoint", "old.pt",
        "--new-addon-checkpoint", "new.pt", "--task-context-mode", "pooling",
        "--old-task-config", old_task, "--conditions", "old", "new_off", "new_on",
        "--output-dir", str(tmp_path / "run"), "--dry-run",
    ])
    main()
    capsys.readouterr()
    plan = json.loads((tmp_path / "run" / "command_plan.json").read_text())
    contracts = plan["policy_contracts"]
    assert contracts["old"]["prefix_layout"] == "zeva_ego"
    assert contracts["old"]["pim_top_k"] == expected_old_slots
    assert contracts["new_off"] == contracts["new_on"]
    assert contracts["new_on"] == {
        "prefix_layout": "zeva_ego", "effect_source": "current_prediction",
        "pim_variant": "cross_attempt", "pim_top_k": 192,
        "memory_semantics": "predicted_effect_trace", "prior_variant": "zeva_ego",
        "skip_get_obs_within_replan": True,
    }
    for item in plan["commands"]:
        fields = dict(argument.split("=", 1) for argument in item["command"] if "=" in argument)
        contract = contracts[item["condition"]]
        assert json.loads(fields["model.zeva.adapter.prefix_layout"]) == contract["prefix_layout"]
        expected_source = "latest_completed" if item["condition"] == "old" else "current_prediction"
        assert json.loads(fields["model.zeva.adapter.effect_source"]) == expected_source
        assert int(fields["EVALUATION.pim_top_k"]) == contract["pim_top_k"]
        assert json.loads(fields["EVALUATION.cross_memory_enabled"]) == (item["condition"] != "new_off")
        assert json.loads(fields["EVALUATION.skip_get_obs_within_replan"]) == (item["condition"] != "old")


def test_manifest_resolves_reader_overrides_instead_of_task_name_heuristics():
    resolve = runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["_policy_contract"]
    contract = resolve(
        "robotwin_zeva_fastwam_cross_attempt_3cam_384",
        [
            "model.zeva.memory.pim_top_k=96", "model.zeva.prompt.persistent_length=96",
            "model.zeva.memory.pim_max_entries=96", "model.zeva.memory.max_entries_per_attempt=32",
        ],
        "legacy_merged", "latest_completed",
    )
    assert contract["pim_top_k"] == 96
    assert contract["prefix_layout"] == "legacy_merged"
    assert contract["effect_source"] is None
    with pytest.raises(ValueError, match="persistent_length"):
        resolve(
            "robotwin_zeva_fastwam_cross_attempt_3cam_384",
            ["model.zeva.memory.pim_top_k=96"], "zeva_ego", "latest_completed",
        )


def test_manifest_rejects_predictive_source_on_legacy_cte_or_prior():
    resolve = runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["_policy_contract"]
    with pytest.raises(ValueError, match="full Zeva-Ego"):
        resolve("robotwin_zeva_fastwam_cross_attempt_3cam_384", [], "zeva_ego", "current_prediction")
    with pytest.raises(ValueError, match="requires current_prediction"):
        resolve("robotwin_zeva_ego_pim_3cam_384", [], "zeva_ego", "latest_completed")


def test_manifest_uses_condition_specific_cte_and_context_artifacts(tmp_path, monkeypatch, capsys):
    main = runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["main"]
    monkeypatch.setattr(sys, "argv", [
        "eval", "--checkpoint", "base.pt", "--dataset-stats", "stats.json",
        "--old-addon-checkpoint", "old.pt", "--new-addon-checkpoint", "new.pt",
        "--old-cte-checkpoint", "old_cte.pt", "--new-cte-checkpoint", "new_cte.pt",
        "--old-task-context-bank", "old_bank.pt", "--new-task-context-bank", "new_bank.pt",
        "--old-task-context-retrieval-checkpoint", "old_head.pt",
        "--new-task-context-retrieval-checkpoint", "new_head.pt",
        "--conditions", "old", "new_off", "new_on", "--output-dir", str(tmp_path / "run"), "--dry-run",
    ])
    main()
    capsys.readouterr()
    plan = json.loads((tmp_path / "run" / "command_plan.json").read_text())
    assert plan["policy_artifacts"]["new_off"] == plan["policy_artifacts"]["new_on"]
    for item in plan["commands"]:
        family = "old" if item["condition"] == "old" else "new"
        fields = dict(argument.split("=", 1) for argument in item["command"] if "=" in argument)
        assert Path(json.loads(fields["EVALUATION.cte_checkpoint"])).name == f"{family}_cte.pt"
        assert Path(json.loads(fields["model.zeva.task_context.bank_path"])).name == f"{family}_bank.pt"
        assert Path(json.loads(fields["model.zeva.task_context.retrieval_checkpoint"])).name == f"{family}_head.pt"
