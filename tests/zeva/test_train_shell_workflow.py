"""Offline launchers: commands and source checks without models or GPU work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "scripts/train"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def metadata_inputs(tmp_path):
    dataset = tmp_path / 'dataset source, "quoted" \'single\' $(printf SHELL_EXPANSION)'
    dataset.mkdir()
    (dataset / "meta").mkdir()
    # The last short episode is outside the selected task, but must still be
    # represented in the real condition map used to build source partitions.
    episodes = [{"episode_index": 0, "length": 49},
                {"episode_index": 1, "length": 49},
                {"episode_index": 2, "length": 3}]
    (dataset / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8")
    paths = {"dataset-root": dataset, "stats": tmp_path / "stats.json",
             "semantic-map": tmp_path / "semantic.json", "task-subset": tmp_path / "subset.json",
             "conditions": tmp_path / "conditions.json", "base-checkpoint": tmp_path / "base.pt",
             "action-dit-checkpoint": tmp_path / "action.pt", "assets-root": tmp_path / "assets"}
    _write_json(paths["stats"], {"action": {"mean": [0.0] * 14, "std": [1.0] * 14}})
    _write_json(paths["semantic-map"], {
        "format": "robotwin_semantic_task_map_v1",
        "episode_to_semantic_task": {"0": "adjust_bottle", "1": "adjust_bottle", "2": "beat_block_hammer"},
    })
    _write_json(paths["task-subset"], {"task_names": ["adjust_bottle"]})
    _write_json(paths["conditions"], {
        f"{dataset}::episode-{index}": "Randomized" if index == 1 else "Clean" for index in range(3)
    })
    paths["base-checkpoint"].write_bytes(b"Not deserialized by a metadata-only check")
    paths["action-dit-checkpoint"].write_bytes(b"Not deserialized by a metadata-only check")
    paths["assets-root"].mkdir()
    return paths


def _preflight(paths):
    # -S deliberately removes site-packages: any accidental Torch / dataset
    # import makes this fail. -B and a directory snapshot verify no pycache.
    args = [sys.executable, "-I", "-S", "-B", str(WORKFLOW / "_preflight.py")]
    for name, path in paths.items():
        args.extend([f"--{name}", str(path)])
    return subprocess.run(args, cwd="/tmp", capture_output=True, text=True, check=False)


def test_metadata_preflight_is_standard_library_only_and_read_only(metadata_inputs, tmp_path):
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    result = _preflight(metadata_inputs)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Preflight PASSED" in result.stdout
    assert "training dependencies and CUDA were not imported" in result.stdout
    assert "covers all 3 positive-length source episodes" in result.stdout
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("defect", [
    "missing_unselected_short_episode", "wrong_root", "wrong_semantic_schema",
    "duplicate_task", "all_tasks_empty", "zero_length_source",
])
def test_metadata_preflight_rejects_ambiguous_or_incomplete_sources(metadata_inputs, defect):
    paths = metadata_inputs
    if defect == "missing_unselected_short_episode":
        conditions = json.loads(paths["conditions"].read_text())
        del conditions[f"{paths['dataset-root']}::episode-2"]
        _write_json(paths["conditions"], conditions)
    elif defect == "wrong_root":
        _write_json(paths["conditions"], {"/wrong::episode-0": "Clean", "/wrong::episode-1": "Randomized"})
    elif defect == "wrong_semantic_schema":
        _write_json(paths["semantic-map"], {"adjust_bottle": [0, 1]})
    elif defect == "duplicate_task":
        _write_json(paths["task-subset"], {"task_names": ["adjust_bottle", "adjust_bottle"]})
    elif defect == "all_tasks_empty":
        _write_json(paths["task-subset"], {"task_names": ["press_stapler"]})
    elif defect == "zero_length_source":
        with (paths["dataset-root"] / "meta/episodes.jsonl").open("a") as stream:
            stream.write(json.dumps({"episode_index": 3, "length": 0}) + "\n")
    result = _preflight(paths)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Preflight FAILED" in result.stdout


def test_metadata_preflight_retains_empty_task_slots(metadata_inputs):
    path = metadata_inputs["task-subset"]
    _write_json(path, {"task_names": ["press_stapler", "adjust_bottle"]})
    original = path.read_bytes()
    result = _preflight(metadata_inputs)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "empty task slots are retained in the supplied task order" in result.stdout
    assert path.read_bytes() == original


def _clean_env(**overrides):
    # Do not accidentally depend on the developer's CUDA/rank/resume settings.
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", **overrides}


def _launch(name, env, cwd, *args):
    return subprocess.run(
        ["bash", str(WORKFLOW / name), *args], env=env, cwd=cwd,
        capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("first,last,config_disables_dry_run", [
    ("01", "11", False), ("08", "09", False), ("01", "11", True),
])
def test_dry_run_range_never_invokes_python_or_creates_outputs(tmp_path, first, last, config_disables_dry_run):
    env = _clean_env(
        PYTHON_BIN=str(tmp_path / "nonexistent python"), DATA_ROOT=str(tmp_path / "missing dataset"),
        RUN_ROOT=str(tmp_path / "new experiment"), DIFFSYNTH_MODEL_BASE_PATH=str(tmp_path / "missing assets"),
    )
    if config_disables_dry_run:
        # The explicit CLI option must win in every child, even when sourcing
        # the user's configuration resets DRY_RUN to its normal execution mode.
        config = tmp_path / "settings with spaces.sh"
        config.write_text(f"DRY_RUN=0\nPYTHON_BIN={shlex.quote(env['PYTHON_BIN'])}\n", encoding="utf-8")
        env["ICLWAM_TRAIN_ENV"] = str(config)
    before = sorted(tmp_path.rglob("*"))
    result = _launch("12_run_all.sh", env, tmp_path, "--from", first, "--to", last, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    headers = [line for line in result.stdout.splitlines() if line.startswith("========== ")]
    assert [int(line.split()[1][:2]) for line in headers] == list(range(int(first), int(last) + 1))
    assert "nonexistent\\ python" in result.stdout
    assert sorted(tmp_path.rglob("*")) == before
    if first == "01":
        assert "world=16, per-rank batch=1, accumulation=16, effective batch=256" in result.stdout


@pytest.fixture
def captured_launch(metadata_inputs, tmp_path):
    fake_python = tmp_path / "fake python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['ARGV_LOG'], 'a', encoding='utf-8') as stream:\n"
        "    json.dump({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'env': {\n"
        "        key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH',\n"
        "        'DIFFSYNTH_MODEL_BASE_PATH', 'PRECOMPUTE_NUM_GPUS')}}, stream)\n"
        "    stream.write('\\n')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    names = {
        "DATA_ROOT": "dataset-root", "STATS_PATH": "stats", "SEMANTIC_TASK_MAP": "semantic-map",
        "CTE_TASK_SUBSET": "task-subset", "EPISODE_CONDITIONS": "conditions",
        "BASE_CKPT": "base-checkpoint", "ACTION_DIT_CKPT": "action-dit-checkpoint",
        "DIFFSYNTH_MODEL_BASE_PATH": "assets-root",
    }
    env = _clean_env(**{name: str(metadata_inputs[key]) for name, key in names.items()})
    env.update(PYTHON_BIN=str(fake_python), ARGV_LOG=str(tmp_path / "argv.jsonl"),
               RUN_ROOT=str(tmp_path / 'experiment, "quoted"'), CUDA_VISIBLE_DEVICES="5")
    artifact_root = tmp_path / "prepared inputs"
    artifact_root.mkdir()
    for name in ("TEXT_CACHE", "BOUNDARY_CACHE", "PIM_TRAIN", "PIM_VALIDATION"):
        path = artifact_root / name
        path.mkdir()
        env[name] = str(path)
    for name in ("CTE_CKPT", "TASK_MEMORY_BANK", "LANGUAGE_ENCODER", "LANGUAGE_FEATURES",
                 "LANGUAGE_CLASSIFIER", "QUERY_CONTEXT_CACHE"):
        path = artifact_root / f"{name}.pt"
        path.write_bytes(b"Not a model: the fake Python records argv only.")
        env[name] = str(path)
    return env


def _one_capture(env):
    records = [json.loads(line) for line in Path(env["ARGV_LOG"]).read_text().splitlines()]
    assert len(records) == 1
    return records[0]


def _compose_capture(record, entry):
    from hydra import compose, initialize_config_dir

    argv = record["argv"]
    index = next(i for i, value in enumerate(argv) if Path(value).name == entry)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train", overrides=argv[index + 1:])


def test_single_gpu_stage2_preserves_paths_and_hydra_overrides(captured_launch, tmp_path):
    env = captured_launch
    result = _launch("10_train_stage2a.sh", env, tmp_path, "log_every=1")
    assert result.returncode == 0, result.stdout + result.stderr
    record = _one_capture(env)
    assert record["cwd"] == str(ROOT)
    assert record["env"]["CUDA_VISIBLE_DEVICES"] == "5"
    assert record["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT / "src")
    assert record["env"]["DIFFSYNTH_MODEL_BASE_PATH"] == env["DIFFSYNTH_MODEL_BASE_PATH"]
    assert record["argv"][:4] == ["-m", "torch.distributed.run", "--nproc-per-node=1", "--standalone"]
    cfg = _compose_capture(record, "train_zeva_fastwam_prepared.py")
    assert list(cfg.data.train.dataset_dirs) == [env["DATA_ROOT"]]
    assert cfg.output_dir == str(Path(env["RUN_ROOT"]) / "stage2a_joint")
    assert cfg.foundation_mode == "joint"
    assert cfg.batch_size == 1 and cfg.gradient_accumulation_steps == 256
    assert cfg.zeva_optimizer_sharding == "none"
    assert cfg.log_every == 1
    assert "$(printf SHELL_EXPANSION)" in cfg.data.train.dataset_dirs[0]


def test_multinode_frozen_stage2b_keeps_joint_parent(captured_launch, tmp_path):
    env = captured_launch
    env.update(GPU_IDS="0,1,2,3,4,5,6,7", NNODES="2", NODE_RANK="1",
               MASTER_ADDR="10.0.0.1", MASTER_PORT="29601", STAGE2B_FOUNDATION_MODE="frozen",
               STAGE2A_STEPS="375")
    parent = Path(env["RUN_ROOT"]) / "stage2a_joint/checkpoints/weights/step_000375_addon.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"Fake parent")
    result = _launch("11_train_stage2b.sh", env, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    record = _one_capture(env)
    assert "--standalone" not in record["argv"]
    for flag in ("--nproc-per-node=8", "--nnodes=2", "--node-rank=1",
                 "--master-addr=10.0.0.1", "--master-port=29601"):
        assert flag in record["argv"]
    cfg = _compose_capture(record, "train_zeva_fastwam_prepared.py")
    assert cfg.foundation_mode == "frozen"
    assert cfg.model.zeva.policy_checkpoint == str(parent)
    assert cfg.output_dir == str(Path(env["RUN_ROOT"]) / "stage2b_frozen")
    assert cfg.batch_size == 1 and cfg.gradient_accumulation_steps == 16
    assert cfg.zeva_optimizer_sharding == "zero1"


def test_stage2b_resume_needs_state_directory_but_not_old_parent(captured_launch, tmp_path):
    env = captured_launch
    state = tmp_path / "trainer state"
    state.mkdir()
    env["STAGE2B_RESUME"] = str(state)
    # No Stage2A directory or checkpoint exists in this fixture.
    result = _launch("11_train_stage2b.sh", env, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    record = _one_capture(env)
    cfg = _compose_capture(record, "train_zeva_fastwam_prepared.py")
    assert cfg.resume == str(state)
    assert cfg.model.zeva.policy_checkpoint is None
    assert not (Path(env["RUN_ROOT"]) / "stage2a_joint").exists()

    Path(env["ARGV_LOG"]).unlink()
    env["STAGE2B_RESUME"] = env["CTE_CKPT"]  # A weights file cannot stand in for trainer state.
    result = _launch("11_train_stage2b.sh", env, tmp_path)
    assert result.returncode != 0
    assert "Required directory missing" in result.stderr
    assert not Path(env["ARGV_LOG"]).exists()


def test_cte_resume_is_single_process_and_uses_first_allocated_gpu(captured_launch, tmp_path):
    env = captured_launch
    env.update(GPU_IDS="7,4", CTE_RESUME=env["CTE_CKPT"])
    result = _launch("03_train_cte.sh", env, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    record = _one_capture(env)
    assert record["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert Path(record["argv"][0]).name == "train_zeva_ego_cte.py"
    cfg = _compose_capture(record, "train_zeva_ego_cte.py")
    assert cfg.resume == env["CTE_RESUME"]
    assert cfg.num_epochs == 80 and cfg.max_steps is None
    assert cfg.gradient_accumulation_steps == 1


@pytest.mark.parametrize("settings,args,expected", [
    ({"GPU_IDS": "0,1,2"}, [], "divisible"),
    ({"GPU_IDS": "0,0"}, [], "duplicate"),
    ({"NNODES": "2", "NODE_RANK": "2"}, [], "NODE_RANK"),
    ({"ZEVA_OPTIMIZER_SHARDING": "zero1"}, [], "at least two ranks"),
    ({}, ["++ckpt=/wrong-parent.pt"], "Workflow-controlled override"),
])
def test_invalid_launch_is_rejected_before_python(captured_launch, tmp_path, settings, args, expected):
    env = {**captured_launch, **settings}
    result = _launch("10_train_stage2a.sh", env, tmp_path, "--dry-run", *args)
    assert result.returncode != 0, result.stdout + result.stderr
    assert expected.lower() in result.stderr.lower()
    assert not Path(env["ARGV_LOG"]).exists()
