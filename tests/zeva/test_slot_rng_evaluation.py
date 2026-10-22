"""The fresh-run slot protocol remains reproducible across task subprocesses."""

from copy import deepcopy
import json
from pathlib import Path
import random
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.robotwin.multiattempt import (
    ATTEMPT_SCHEMA, CONTINUOUS_NOISE_PROTOCOL, EVALUATION_PROTOCOL_SCHEMA,
    MANIFEST_SCHEMA, append_attempt, load_manifest, summarize_attempts,
    validate_attempts,
)
from fastwam.zeva.attempt_protocol import ContinuousModelNoiseStream


ROOT = Path(__file__).resolve().parents[2]


def test_continuous_slot_noise_matches_uninterrupted_generator_across_tasks(tmp_path):
    state = tmp_path / "slot_0.json"
    uninterrupted = ContinuousModelNoiseStream(20260907)
    first = ContinuousModelNoiseStream(20260907, state_path=state, run_id="run:on", task_name="task_a")
    for _ in range(3):
        assert torch.equal(first.next_noise((1, 32, 14)), uninterrupted.next_noise((1, 32, 14)))
        random.seed(11)
        torch.manual_seed(99)  # Environment/global resets cannot reseed this stream.
        torch.randn(37)
    first.save_state()
    second = ContinuousModelNoiseStream(20260907, state_path=state, run_id="run:on", task_name="task_b")
    assert second.snapshot() == uninterrupted.snapshot()
    for _ in range(4):
        assert torch.equal(second.next_noise((1, 32, 14)), uninterrupted.next_noise((1, 32, 14)))
    second.save_state()
    assert json.loads(state.read_text())["completed_tasks"] == ["task_a", "task_b"]
    assert second.snapshot() == uninterrupted.snapshot()
    assert second.snapshot()["draws"] == 7
    with pytest.raises(RuntimeError, match="after completing"):
        second.next_noise((1, 32, 14))


@pytest.mark.parametrize("problem", ["seed", "slot", "run", "checkpoint", "duplicate", "checksum", "draws", "shape"])
def test_slot_handoff_rejects_wrong_identity_or_corruption(tmp_path, problem):
    path = tmp_path / "state.json"
    first = ContinuousModelNoiseStream(17, state_path=path, run_id="run", task_name="a", slot_count=2)
    first.next_noise((1, 32, 14))
    first.save_state()
    args = {"seed": 17, "state_path": path, "run_id": "run", "task_name": "b", "slot_count": 2}
    if problem == "seed":
        args["seed"] = 18
    elif problem == "slot":
        args["slot_index"] = 1
    elif problem == "run":
        args["run_id"] = "other"
    elif problem == "checkpoint":
        args["checkpoint_identity"] = {"cte_sha256": "different"}
    elif problem == "duplicate":
        args["task_name"] = "a"
    else:
        data = json.loads(path.read_text())
        if problem == "checksum":
            data["state_sha256"] = "0" * 64
        elif problem == "draws":
            data["draws"] = True
        else:
            data["noise_shape"] = [2, 32, 14]
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        ContinuousModelNoiseStream(**args)


def test_slot_stream_refuses_shape_changes_and_uncommitted_partial_file(tmp_path):
    path = tmp_path / "state.json"
    stream = ContinuousModelNoiseStream(9, state_path=path, run_id="run", task_name="a")
    stream.next_noise((1, 32, 14))
    with pytest.raises(ValueError, match="shape changed"):
        stream.next_noise((1, 16, 14))
    path.with_name(path.name + ".partial").write_text("interrupted write")
    with pytest.raises(FileExistsError):
        stream.save_state()
    assert not path.exists()


def _audited_rows(tmp_path):
    protocol = {
        "schema": EVALUATION_PROTOCOL_SCHEMA, "model_rng_protocol": CONTINUOUS_NOISE_PROTOCOL,
        "source_manifest_sha256": "1" * 64, "manifest_scope": "custom",
        "model_seed": 20260907, "slots": 2, "num_inference_steps": 10,
        "task_slot_assignment": {"a": 0, "b": 1, "c": 0}, "task_order": ["a", "c", "b"],
    }
    manifest = {
        "schema": MANIFEST_SCHEMA, "max_attempts": 4,
        "action_dim": 14, "action_horizon": 32, "replan_steps": 24,
        "task_config": "demo_randomized", "instruction_type": "seen",
        "evaluation_protocol": protocol,
        "episodes": [{"task": task, "episode_id": f"{task}:7", "seed": 7, "instruction": f"do {task}"} for task in ("a", "b", "c")],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    manifest, sha = load_manifest(path)
    streams = {slot: ContinuousModelNoiseStream(20260907, slot_index=slot, slot_count=2) for slot in (0, 1)}
    rows = []
    for task in protocol["task_order"]:
        stream = streams[protocol["task_slot_assignment"][task]]
        tries = 2 if task == "a" else 1
        for attempt in range(1, tries + 1):
            success = attempt == tries
            steps = 24 if success else 48
            before = stream.snapshot()
            for _ in range(steps // 24):
                stream.next_noise((1, 32, 14))
            rows.append({
                "schema": ATTEMPT_SCHEMA, "manifest_sha256": sha, "run_id": "run", "condition": "new_on",
                "task": task, "episode_id": f"{task}:7", "scene_seed": 7, "instruction": f"do {task}",
                "task_config": "demo_randomized", "instruction_type": "seen", "action_horizon": 32, "replan_steps": 24,
                "max_attempts": 4, "attempt": attempt, "success": success, "status": "success" if success else "timeout",
                "step_count": steps, "step_limit": 48, "model_seed": 20260907, "num_inference_steps": 10,
                "zeva_mode": "pim_on", "cross_memory_enabled": True,
                "checkpoint_identity": {key: char * 64 for key, char in zip(("base_sha256", "cte_sha256", "addon_sha256", "stats_sha256"), "abcd")},
                "reset_scope": "episode" if attempt == 1 else "attempt", "pim_entries_before": attempt - 1,
                "evaluation_slot": {"index": protocol["task_slot_assignment"][task], "count": 2},
                "seed_stream_before": before, "seed_stream": stream.snapshot(),
            })
    return manifest, sha, path, rows


def test_audit_verifies_continuous_noise_across_tasks_and_fixed_denominator(tmp_path):
    manifest, sha, path, rows = _audited_rows(tmp_path)
    report = summarize_attempts(manifest, sha, rows)
    assert report["protocol_audit"]["status"] == "passed"
    assert report["protocol_audit"]["model_noise_draws"] == 5
    assert report["manifest_scope"] == "custom"
    assert report["cumulative_success_rate"] == [
        {"attempt": k, "solved": 2 if k == 1 else 3, "total": 3, "rate": 2 / 3 if k == 1 else 1.0}
        for k in range(1, 5)
    ]
    assert load_manifest(path, expected_sha256=sha)[1] == sha
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_manifest(path, expected_sha256="f" * 64)


@pytest.mark.parametrize("problem", ["reset", "memory", "slot", "seed", "reseed", "extra_draw", "order", "checkpoint", "budget"])
def test_protocol_audit_rejects_hidden_evaluation_changes(tmp_path, problem):
    manifest, sha, _, rows = _audited_rows(tmp_path)
    if problem == "reset":
        rows[1]["reset_scope"] = "episode"
    elif problem == "memory":
        rows[0]["pim_entries_before"] = 1
    elif problem == "slot":
        rows[2]["evaluation_slot"]["index"] = 1
    elif problem == "seed":
        rows[0]["seed_stream_before"]["seed"] = 0
    elif problem == "reseed":
        rows[2]["seed_stream_before"] = deepcopy(rows[0]["seed_stream_before"])
    elif problem == "extra_draw":
        rows[-1]["seed_stream"]["draws"] += 1
    elif problem == "order":
        rows = [rows[-1], *rows[:-1]]
    elif problem == "checkpoint":
        for row in rows:
            row["checkpoint_identity"]["cte_sha256"] = ""
    else:
        rows[1]["step_limit"] += 1
    with pytest.raises(ValueError):
        validate_attempts(manifest, sha, rows)


def test_successful_task_handoff_must_match_last_record(tmp_path):
    verify = runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["_verify_rng_handoff"]
    state, log = tmp_path / "slot.json", tmp_path / "log.jsonl"
    identity = {"base_sha256": "a" * 64, "cte_sha256": "b" * 64, "task_context": {"bank_sha256": "c" * 64}}
    stream = ContinuousModelNoiseStream(3, state_path=state, run_id="run:on", task_name="task", checkpoint_identity=identity)
    stream.next_noise((1, 32, 14))
    stream.save_state()
    row = {"task": "task", "seed_stream": stream.snapshot(), "checkpoint_identity": identity}
    log.write_text(json.dumps(row) + "\n")
    kwargs = {"seed": 3, "slot": 0, "slots": 1, "run_id": "run:on", "completed_tasks": ["task"], "log_path": log}
    verify(state, **kwargs)
    with pytest.raises(ValueError, match="identity/order"):
        verify(state, **{**kwargs, "completed_tasks": ["earlier_task", "task"]})
    row["seed_stream"]["draws"] += 1
    log.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="final attempt snapshot"):
        verify(state, **kwargs)
    state.unlink()
    with pytest.raises(ValueError, match="without committing"):
        verify(state, **kwargs)


def _dry_run(monkeypatch, output, *extra):
    main = runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["main"]
    monkeypatch.setattr(sys, "argv", [
        "eval", "--checkpoint", "base.pt", "--dataset-stats", "stats.json",
        "--conditions", "base", "--output-dir", str(output), "--dry-run", *extra,
    ])
    main()
    return json.loads((output / "command_plan.json").read_text())


def test_manifest_plan_freezes_slot_order_hash_and_development_scope(tmp_path, monkeypatch, capsys):
    full = _dry_run(monkeypatch, tmp_path / "full")
    capsys.readouterr()
    protocol = full["evaluation_protocol"]
    assert protocol["manifest_scope"] == "frozen_full"
    assert protocol["model_seed"] == 20260907 and protocol["slots"] == 8
    assert full["episodes"] == 200
    assert full["model_rng_protocol"] == CONTINUOUS_NOISE_PROTOCOL
    assert [row["task"] for row in full["commands"]] == protocol["task_order"]
    for item in full["commands"]:
        fields = dict(part.split("=", 1) for part in item["command"] if "=" in part)
        assert json.loads(fields["EVALUATION.expected_manifest_sha256"]) == full["manifest_sha256"]
        assert int(fields["EVALUATION.evaluation_slot_index"]) == item["slot"]
    selected = sorted(protocol["task_slot_assignment"])[-1]
    dev = _dry_run(monkeypatch, tmp_path / "dev", "--tasks", selected)
    capsys.readouterr()
    assert dev["episodes"] == 20
    assert dev["evaluation_protocol"]["manifest_scope"] == "development_subset"
    assert dev["evaluation_protocol"]["task_slot_assignment"][selected] == protocol["task_slot_assignment"][selected]
    assert dev["manifest_sha256"] != full["manifest_sha256"]
    with pytest.raises(SystemExit):
        _dry_run(monkeypatch, tmp_path / "dev", "--tasks", selected)
    assert "must not exist" in capsys.readouterr().err


def test_custom_manifest_needs_explicit_hash(tmp_path, monkeypatch, capsys):
    _, sha, path, _ = _audited_rows(tmp_path)
    with pytest.raises(SystemExit):
        _dry_run(monkeypatch, tmp_path / "run", "--manifest", str(path))
    assert "require --expected-manifest-sha256" in capsys.readouterr().err
    plan = _dry_run(monkeypatch, tmp_path / "run", "--manifest", str(path), "--expected-manifest-sha256", sha)
    capsys.readouterr()
    assert plan["evaluation_protocol"]["manifest_scope"] == "custom"


@pytest.mark.parametrize("fail_task", [None, "c"])
def test_manifest_runner_exports_complete_only_after_all_task_handoffs_and_audits(
    tmp_path, monkeypatch, capsys, fail_task,
):
    """Exercise orchestration with a fake simulator; this is not an SR result."""
    _, source_sha, source_path, _ = _audited_rows(tmp_path)
    base, stats = tmp_path / "base.pt", tmp_path / "stats.json"
    base.write_bytes(b"test-checkpoint")
    stats.write_text("{}")
    output = tmp_path / "run"
    main = runpy.run_path(str(ROOT / "scripts/eval_zeva_robotwin_manifest.py"))["main"]
    monkeypatch.setattr(sys, "argv", [
        "eval", "--checkpoint", str(base), "--dataset-stats", str(stats), "--slots", "2",
        "--manifest", str(source_path), "--expected-manifest-sha256", source_sha,
        "--conditions", "base", "--output-dir", str(output),
    ])
    visited = []
    identity = {"base_sha256": "a" * 64, "stats_sha256": "b" * 64}

    def child(command, *, cwd, check):
        settings = {key: json.loads(value) for part in command if "=" in part for key, value in [part.split("=", 1)]}
        get = lambda key: settings[f"EVALUATION.{key}"]
        task = get("task_name")
        visited.append(task)
        manifest, sha = load_manifest(get("episode_manifest"), expected_sha256=get("expected_manifest_sha256"))
        stream = ContinuousModelNoiseStream(
            get("model_seed"), slot_index=get("evaluation_slot_index"), slot_count=get("evaluation_slot_count"),
            state_path=get("model_rng_state_path"), run_id=get("model_rng_run_id"),
            task_name=task, checkpoint_identity=identity,
        )
        for episode in manifest["episodes"]:
            if episode["task"] != task:
                continue
            for attempt in (1, 2):
                success = attempt == 2
                steps = 24 if success else 48
                before = stream.snapshot()
                for _ in range(steps // 24):
                    stream.next_noise((1, 32, 14))
                video = output / f"{task}_{attempt}.mp4"
                video.write_bytes(b"fake simulator recording")
                append_attempt(get("attempt_log_path"), {
                    "schema": ATTEMPT_SCHEMA, "manifest_sha256": sha,
                    "task": task, "episode_id": episode["episode_id"], "scene_seed": episode["seed"],
                    "instruction": episode["instruction"], "task_config": manifest["task_config"],
                    "instruction_type": manifest["instruction_type"], "action_horizon": 32, "replan_steps": 24,
                    "max_attempts": 4, "attempt": attempt, "success": success, "status": "success" if success else "timeout",
                    "step_count": steps, "step_limit": 48, "model_seed": get("model_seed"), "num_inference_steps": 10,
                    "run_id": get("run_id"), "condition": "base", "zeva_mode": "base", "cross_memory_enabled": True,
                    "skip_get_obs_within_replan": get("skip_get_obs_within_replan"),
                    "checkpoint_identity": identity, "reset_scope": "episode" if attempt == 1 else "attempt",
                    "pim_entries_before": 0, "evaluation_slot": {"index": get("evaluation_slot_index"), "count": 2},
                    "seed_stream_before": before, "seed_stream": stream.snapshot(), "video_path": str(video),
                })
                if task == fail_task:
                    return SimpleNamespace(returncode=42)  # No export on task failure.
        stream.save_state()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", child)
    if fail_task:
        with pytest.raises(SystemExit, match="partial logs are not a completed benchmark"):
            main()
        assert visited == ["a", "c"]
        assert not (output / "COMPLETE").exists()
        assert json.loads((output / "run_failure.json").read_text())["returncode"] == 42
        assert json.loads((output / "rng_state/base/slot_0.json").read_text())["completed_tasks"] == ["a"]
        with pytest.raises(SystemExit):
            main()
        assert "must not exist" in capsys.readouterr().err
    else:
        main()
        capsys.readouterr()
        assert visited == ["a", "c", "b"]
        assert (output / "COMPLETE").is_file()
        report = json.loads((output / "base.summary.json").read_text())
        assert report["protocol_audit"]["model_noise_draws"] == 9
        assert report["video_audit"]["videos"] == 6
        assert report["cumulative_success_rate"][1] == {"attempt": 2, "solved": 3, "total": 3, "rate": 1.0}

