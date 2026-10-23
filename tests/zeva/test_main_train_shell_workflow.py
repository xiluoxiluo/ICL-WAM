"""Main launchers: provenance, real Hydra commands, isolation and no-op dry runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "scripts/train"


def _env(**kwargs):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", **kwargs}


def _launch(name, env, cwd, *args):
    return subprocess.run(["bash", str(WORKFLOW / name), *args], env=env, cwd=cwd,
                          capture_output=True, text=True)


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def source_inputs(tmp_path):
    data = tmp_path / 'dataset, "double" \'single\' $(printf DO_NOT_EXECUTE)'
    (data / "meta").mkdir(parents=True)
    rows = [{"episode_index": i, "length": 49, "success": i != 4} for i in range(5)]
    (data / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    _json(data / "meta/info.json", {"total_episodes": 5})
    stats = tmp_path / "actual_stats_alias.json"
    semantic = tmp_path / "semantic.json"
    _json(stats, {"action": {"mean": [0.] * 14, "std": [1.] * 14}})
    _json(semantic, {"format": "robotwin_semantic_task_map_v1",
                     "episode_to_semantic_task": {str(i): "adjust_bottle" if i < 2 else "beat_block_hammer" for i in range(5)}})
    namespace = runpy.run_path(str(ROOT / "src/fastwam/zeva/main_data.py"))
    success = tmp_path / "verified_successes.json"
    _json(success, namespace["build_success_manifest"]([data]))
    base, action, assets = tmp_path / "base.pt", tmp_path / "action.pt", tmp_path / "assets"
    base.write_bytes(b"metadata checks never deserialize this")
    action.write_bytes(b"not a checkpoint")
    assets.mkdir()
    return {"dataset-root": data, "stats": stats, "semantic-map": semantic,
            "success-manifest": success, "base-checkpoint": base,
            "action-dit-checkpoint": action, "assets-root": assets}


def _preflight(paths):
    command = [sys.executable, "-I", "-S", "-B", str(WORKFLOW / "_preflight.py")]
    for key, value in paths.items():
        command += ["--" + key, str(value)]
    return subprocess.run(command, cwd="/tmp", capture_output=True, text=True)


def test_main_preflight_is_stdlib_read_only_and_accepts_actual_two_task_scope(source_inputs, tmp_path):
    before = sorted(tmp_path.rglob("*"))
    result = _preflight(source_inputs)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 successful demonstrations" in result.stdout
    assert "canonical tasks: 2; no fixed 50-task restriction" in result.stdout
    assert "dependencies and CUDA were not imported" in result.stdout
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("defect", ["missing", "source_changed", "unknown_evidence", "wrong_root", "empty_declaration"])
def test_main_preflight_fails_closed_on_success_provenance(source_inputs, defect):
    path = source_inputs["success-manifest"]
    payload = json.loads(path.read_text())
    if defect == "missing":
        path.unlink()
    elif defect == "source_changed":
        _json(source_inputs["dataset-root"] / "meta/info.json", {"total_episodes": 5, "changed": True})
    elif defect == "unknown_evidence":
        payload["evidence"]["kind"] = "assumed_from_task_name"
        _json(path, payload)
    elif defect == "wrong_root":
        payload["sources"][0]["dataset_root"] = "/other/source"
        _json(path, payload)
    else:
        payload["evidence"] = {"kind": "explicit_success_only_declaration", "description": ""}
        _json(path, payload)
    result = _preflight(source_inputs)
    assert result.returncode == 1 and "Preflight FAILED" in result.stdout


@pytest.mark.parametrize("first,last,settings_file", [("01", "11", False), ("03", "09", False), ("01", "11", True)])
def test_main_dry_run_executes_nothing_and_creates_nothing(tmp_path, first, last, settings_file):
    env = _env(PYTHON_BIN=str(tmp_path / "nonexistent python"), DATA_ROOT=str(tmp_path / "missing data"),
               RUN_ROOT=str(tmp_path / "new run"), DIFFSYNTH_MODEL_BASE_PATH=str(tmp_path / "missing assets"))
    if settings_file:
        config = tmp_path / "my settings.sh"
        config.write_text("DRY_RUN=0\n")
        env["ICLWAM_TRAIN_ENV"] = str(config)
    before = sorted(tmp_path.rglob("*"))
    result = _launch("12_run_all.sh", env, tmp_path, "--from", first, "--to", last, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(tmp_path.rglob("*")) == before
    headers = [line for line in result.stdout.splitlines() if line.startswith("========== ")]
    assert [int(line.split()[1][:2]) for line in headers] == list(range(int(first), int(last) + 1))
    assert "train_zeva_ego_cte.py" not in result.stdout and "task_language" not in result.stdout
    assert "--declare-success-only" not in result.stdout
    if last == "11":
        assert "world=16, per-rank batch=1, accumulation=8, effective batch=128, optimizer=none" in result.stdout
        assert "world=16, per-rank batch=1, accumulation=4, effective batch=64, optimizer=none" in result.stdout


def test_every_generated_hydra_stage_composes_the_main_lineage(tmp_path):
    from hydra import compose, initialize_config_dir
    from fastwam.zeva_numerical_contract import config_numerical_contract

    env = _env(PYTHON_BIN="not-executed", RUN_ROOT=str(tmp_path / "run"))
    result = _launch("12_run_all.sh", env, tmp_path, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]
    assert len(commands) == 11
    entries = []
    for command in commands[1:]:
        index = next(i for i, token in enumerate(command) if token.endswith(".py"))
        entry = Path(command[index]).name
        entries.append(entry)
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="train", overrides=command[index + 1:])
        assert cfg.model.zeva.profile == "main_observed"
        assert config_numerical_contract(cfg.model.zeva) == "zeva_main_v2"
        assert cfg.model.zeva.offline.validation_fraction == .1 and cfg.model.zeva.offline.split_seed == 42
        assert cfg.data.train.val_set_proportion == 0 and cfg.data.val is None
        assert cfg.model.zeva.task_context.mode == "static"
        if entry == "precompute_zeva_cte_latents.py":
            assert cfg.cte_latent_cache.reserve_space is True
            assert cfg.cte_latent_cache.process_group_timeout_seconds == 86400
        if entry == "train_zeva_cte.py":
            assert cfg.max_steps == 5000 and cfg.learning_rate == 2e-4
            assert cfg.model.zeva.cte.latent_cache_path.endswith("/artifacts/cte_latents")
            assert cfg.cte_tasks_per_batch * cfg.cte_samples_per_task == cfg.batch_size == 8
        if entry == "train_zeva_fastwam_prepared.py":
            assert cfg.foundation_mode == "frozen" and cfg.zeva_optimizer_sharding == "none"
            assert cfg.model.zeva.prepared_stage2.validation_pim_cache_path is None
            assert cfg.mixed_precision == "bf16"
            if cfg.model.zeva.training_stage == "policy_injection":
                assert (cfg.max_steps, cfg.learning_rate, cfg.gradient_accumulation_steps) == (5000, 2e-4, 8)
            else:
                assert (cfg.max_steps, cfg.learning_rate, cfg.gradient_accumulation_steps) == (500, 5e-5, 4)
    assert entries == ["precompute_text_embeds.py", "precompute_zeva_cte_latents.py", "train_zeva_cte.py",
                       "build_zeva_robotwin_cache_from_latents.py", "build_zeva_behavior_bank_optimized.py",
                       "train_zeva_task_context_retrieval_optimized.py", "build_zeva_static_episode_context_cache.py",
                       "build_zeva_pim_retrieval_cache.py", "train_zeva_fastwam_prepared.py", "train_zeva_fastwam_prepared.py"]


@pytest.fixture
def captured_launch(source_inputs, tmp_path):
    python = tmp_path / "fake python"
    python.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                      "with open(os.environ['ARGV_LOG'],'a') as f:\n"
                      "    f.write(json.dumps({'argv':sys.argv[1:],'cwd':os.getcwd(),'gpus':os.environ.get('CUDA_VISIBLE_DEVICES')})+'\\n')\n")
    python.chmod(0o755)
    variables = {"DATA_ROOT": "dataset-root", "STATS_PATH": "stats", "SEMANTIC_TASK_MAP": "semantic-map",
                 "SUCCESS_MANIFEST": "success-manifest", "BASE_CKPT": "base-checkpoint",
                 "ACTION_DIT_CKPT": "action-dit-checkpoint", "DIFFSYNTH_MODEL_BASE_PATH": "assets-root"}
    env = _env(**{key: str(source_inputs[value]) for key, value in variables.items()})
    env.update(PYTHON_BIN=str(python), ARGV_LOG=str(tmp_path / "argv.jsonl"),
               RUN_ROOT=str(tmp_path / 'new run, "quoted"'), CUDA_VISIBLE_DEVICES="5")
    for name in ("TEXT_CACHE", "LATENT_CACHE", "PHASE_CACHE", "PIM_CACHE"):
        path = tmp_path / name
        path.mkdir()
        env[name] = str(path)
    for name in ("CTE_CKPT", "BEHAVIOR_BANK", "READOUT_CACHE", "RETRIEVAL_HEAD", "STATIC_CONTEXT_CACHE"):
        path = tmp_path / (name + ".pt")
        path.write_bytes(b"only launcher argv is executed")
        env[name] = str(path)
    return env


def _captured_cfg(env):
    from hydra import compose, initialize_config_dir
    rows = [json.loads(line) for line in Path(env["ARGV_LOG"]).read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    index = next(i for i, token in enumerate(row["argv"]) if token.endswith(".py"))
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=row["argv"][index + 1:])
    return row, cfg


def test_main_launcher_preserves_literal_paths_and_actual_single_gpu_batch(captured_launch, tmp_path):
    env = captured_launch
    result = _launch("10_train_stage2a.sh", env, tmp_path, "log_every=1")
    assert result.returncode == 0, result.stdout + result.stderr
    row, cfg = _captured_cfg(env)
    assert row["cwd"] == str(ROOT) and row["gpus"] == "5"
    assert cfg.gradient_accumulation_steps == 128 and cfg.foundation_mode == "frozen"
    assert list(cfg.data.train.dataset_dirs) == [env["DATA_ROOT"]]
    assert cfg.data.train.pretrained_norm_stats == env["STATS_PATH"]
    assert cfg.model.zeva.offline.success_manifest_path == env["SUCCESS_MANIFEST"]
    assert cfg.log_every == 1 and "$(printf DO_NOT_EXECUTE)" in cfg.data.train.dataset_dirs[0]


def test_main_frozen_b_uses_selected_joint_a_parent_and_stage_specific_batch(captured_launch, tmp_path):
    env = captured_launch
    env.update(GPU_IDS="0,1,2,3,4,5,6,7", NNODES="2", NODE_RANK="1", MASTER_ADDR="10.0.0.1",
               STAGE2A_FOUNDATION_MODE="joint", STAGE2B_FOUNDATION_MODE="frozen", STAGE2A_STEPS="375")
    parent = Path(env["RUN_ROOT"]) / "stage2a_joint/checkpoints/weights/step_000375_addon.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"fake parent")
    result = _launch("11_train_stage2b.sh", env, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    row, cfg = _captured_cfg(env)
    assert "--nnodes=2" in row["argv"] and "--node-rank=1" in row["argv"]
    assert cfg.foundation_mode == "frozen" and cfg.model.zeva.policy_checkpoint == str(parent)
    assert cfg.gradient_accumulation_steps == 4 and cfg.max_steps == 500
    assert cfg.zeva_optimizer_sharding == "none"


def test_main_resume_requires_full_state_and_does_not_reload_old_a(captured_launch, tmp_path):
    env = captured_launch
    state = tmp_path / "trainer state"
    state.mkdir()
    env["STAGE2B_RESUME"] = str(state)
    result = _launch("11_train_stage2b.sh", env, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    _, cfg = _captured_cfg(env)
    assert cfg.resume == str(state) and cfg.model.zeva.policy_checkpoint is None
    Path(env["ARGV_LOG"]).unlink()
    env["STAGE2B_RESUME"] = env["CTE_CKPT"]
    result = _launch("11_train_stage2b.sh", env, tmp_path)
    assert result.returncode != 0 and "Required directory missing" in result.stderr
    assert not Path(env["ARGV_LOG"]).exists()


@pytest.mark.parametrize("reserve,expected", [("1", True), ("0", False)])
def test_latent_resume_preserves_path_and_uses_boolean_reservation(captured_launch, tmp_path, reserve, expected):
    env = {**captured_launch, "LATENT_RESUME": "1", "LATENT_RESERVE_SPACE": reserve,
           "PREP_GPU_IDS": "2,3", "PREP_NPROC_PER_NODE": "2",
           "LATENT_PROCESS_GROUP_TIMEOUT_SECONDS": "172800"}
    result = _launch("03_precompute_cte_latents.sh", env, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    row, cfg = _captured_cfg(env)
    assert row["gpus"] == "2,3" and "--nproc-per-node=2" in row["argv"]
    assert cfg.cte_latent_cache.reserve_space is expected
    assert cfg.cte_latent_cache.process_group_timeout_seconds == 172800
    assert cfg.cte_latent_cache.overwrite is False
    assert cfg.model.zeva.cte.latent_cache_path == env["LATENT_CACHE"]


def test_invalid_latent_reservation_switch_does_not_launch_python(captured_launch, tmp_path):
    env = {**captured_launch, "LATENT_RESUME": "1", "LATENT_RESERVE_SPACE": "2"}
    result = _launch("03_precompute_cte_latents.sh", env, tmp_path)
    assert result.returncode != 0 and "LATENT_RESERVE_SPACE must be 0 or 1" in result.stderr
    assert not Path(env["ARGV_LOG"]).exists()


def test_invalid_latent_process_group_timeout_does_not_launch_python(captured_launch, tmp_path):
    env = {**captured_launch, "LATENT_RESUME": "1", "LATENT_PROCESS_GROUP_TIMEOUT_SECONDS": "0"}
    result = _launch("03_precompute_cte_latents.sh", env, tmp_path)
    assert result.returncode != 0 and "LATENT_PROCESS_GROUP_TIMEOUT_SECONDS" in result.stderr
    assert not Path(env["ARGV_LOG"]).exists()


def test_missing_success_manifest_never_runs_python_or_declares_success(captured_launch, tmp_path):
    env = captured_launch
    env["SUCCESS_MANIFEST"] = str(tmp_path / "unknown_success.json")
    result = _launch("04_train_cte.sh", env, tmp_path)
    assert result.returncode != 0 and "build_zeva_success_manifest.py" in result.stderr
    assert "--outcomes-json" in result.stderr and "--declare-success-only" not in result.stderr
    assert not Path(env["ARGV_LOG"]).exists() and not Path(env["SUCCESS_MANIFEST"]).exists()


@pytest.mark.parametrize("settings,overrides,match", [
    ({"GPU_IDS": "0,1,2"}, [], "divisible"),
    ({"GPU_IDS": "0,0"}, [], "Duplicate"),
    ({"STAGE2A_GRAD_ACCUM_STEPS": "4"}, [], "effective global batch"),
    ({"STAGE2A_OPTIMIZER_SHARDING": "zero1"}, [], "at least two ranks"),
    ({}, ["model.zeva.profile=zeva_ego"], "Workflow-controlled"),
    ({}, ["++model.zeva.offline.validation_fraction=0.3"], "Workflow-controlled"),
])
def test_invalid_main_topology_or_lineage_rejected_before_python(captured_launch, tmp_path, settings, overrides, match):
    env = {**captured_launch, **settings}
    result = _launch("10_train_stage2a.sh", env, tmp_path, "--dry-run", *overrides)
    assert result.returncode != 0 and match in result.stderr
    assert not Path(env["ARGV_LOG"]).exists()


def test_only_new_main_numbering_remains_and_ego_is_preserved():
    main_names = sorted(path.name for path in WORKFLOW.glob("[0-9][0-9]*.sh"))
    assert len(main_names) == 12 and [name[:2] for name in main_names] == [f"{i:02d}" for i in range(1, 13)]
    assert "03_train_cte.sh" not in main_names and "06_train_task_language.sh" not in main_names
    assert (ROOT / "scripts/train_ego/03_train_cte.sh").is_file()
    assert (ROOT / "scripts/train_ego/06_train_task_language.sh").is_file()
