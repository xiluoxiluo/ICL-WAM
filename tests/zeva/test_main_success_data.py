import json

import pytest
from omegaconf import OmegaConf

from fastwam.zeva.main_data import (
    build_success_manifest, select_main_success_rows, validate_main_artifact,
    validate_main_episode_ids, normalize_main_dataset_paths,
)
from fastwam.zeva.stage1_sampling import CTETrainIndex


def _source(tmp_path, outcomes=(True, False, True)):
    root = tmp_path / "demonstrations"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_episodes": len(outcomes)}))
    rows = [{"episode_index": i, "length": 65, **({"success": value} if value is not None else {})}
            for i, value in enumerate(outcomes)]
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (root / "stats.json").write_text("{}")
    (root / "meta/semantic_task_map.json").write_text("{}")
    return root


def _config(root, manifest):
    return OmegaConf.create({
        "model": {"zeva": {"profile": "main_observed", "offline": {"success_manifest_path": str(manifest)}}},
        "data": {"train": {"dataset_dirs": [str(root)], "is_training_set": True,
                           "pretrained_norm_stats": str(root / "stats.json"),
                           "semantic_task_map_path": str(root / "meta/semantic_task_map.json")}},
    })


def _rows(root):
    return [CTETrainIndex(i * 65, f"{root}::episode-{i}", "task", 0) for i in range(3)]


def test_success_filter_excludes_failure_actions_and_binds_actual_episode_set(tmp_path):
    root = _source(tmp_path)
    manifest = tmp_path / "success.json"
    manifest.write_text(json.dumps(build_success_manifest([root])))
    selected, metadata = select_main_success_rows(_rows(root), _config(root, manifest))
    assert [row.dataset_index for row in selected] == [0, 130]
    validate_main_artifact(metadata)
    validate_main_episode_ids(metadata, [row.episode_id for row in selected])
    with pytest.raises(ValueError, match="membership"):
        validate_main_episode_ids(metadata, [row.episode_id for row in _rows(root)])


def test_unknown_outcome_cannot_be_inferred_from_dataset_name(tmp_path):
    root = _source(tmp_path, (None, None, None))
    with pytest.raises(ValueError, match="unknown collection outcome"):
        build_success_manifest([root])
    with pytest.raises(ValueError, match="requires evidence"):
        build_success_manifest([root], declare_success_only=True)
    declared = build_success_manifest([root], declare_success_only=True, evidence="Operator checked successful-only collection export log")
    assert declared["evidence"]["kind"] == "explicit_success_only_declaration"
    assert declared["sources"][0]["success_episode_indices"] == [0, 1, 2]


def test_declared_success_only_cannot_overwrite_observed_failure(tmp_path):
    root = _source(tmp_path)
    with pytest.raises(ValueError, match="known failed"):
        build_success_manifest([root], declare_success_only=True, evidence="A label must not be overridden")


def test_manifest_rejects_source_metadata_mutation_and_resume_stats_change(tmp_path):
    root = _source(tmp_path)
    manifest = tmp_path / "success.json"
    manifest.write_text(json.dumps(build_success_manifest([root])))
    cfg = _config(root, manifest)
    _, old = select_main_success_rows(_rows(root), cfg)
    (root / "stats.json").write_text('{"changed": true}')
    _, new = select_main_success_rows(_rows(root), cfg)
    with pytest.raises(ValueError, match="different success data"):
        validate_main_artifact(new, old)
    (root / "meta/info.json").write_text('{"total_episodes": 3, "fps": 15}')
    with pytest.raises(ValueError, match="source changed"):
        select_main_success_rows(_rows(root), cfg)


def test_main_refuses_validation_split_and_predictive_artifacts(tmp_path):
    root = _source(tmp_path)
    manifest = tmp_path / "success.json"
    manifest.write_text(json.dumps(build_success_manifest([root])))
    cfg = _config(root, manifest)
    cfg.data.train.is_training_set = False
    with pytest.raises(ValueError, match="training episode split"):
        select_main_success_rows(_rows(root), cfg)
    with pytest.raises(ValueError, match="main artifact"):
        validate_main_artifact({"memory_semantics": "predicted_effect_trace"})


def test_fixed_split_is_task_stratified_disjoint_and_bound_to_identity(tmp_path):
    root = _source(tmp_path, (True,) * 8)
    manifest = tmp_path / "success.json"
    manifest.write_text(json.dumps(build_success_manifest([root])))
    rows = [CTETrainIndex(i * 65, f"{root}::episode-{i}", f"task-{i // 4}", 0) for i in range(8)]
    _, metadata = select_main_success_rows(rows, _config(root, manifest))
    _, reordered = select_main_success_rows(reversed(rows), _config(root, manifest))
    assert metadata == reordered
    splits = metadata["episode_splits"]
    assert len(splits["train"]) == 6 and len(splits["validation"]) == 2
    assert not set(splits["train"]) & set(splits["validation"])
    task_by_id = {row.episode_id: row.task_id for row in rows}
    assert {task_by_id[episode_id] for episode_id in splits["validation"]} == {"task-0", "task-1"}
    train_metadata = {**metadata, "data_role": "train"}
    validate_main_episode_ids(train_metadata, splits["train"])
    with pytest.raises(ValueError, match="membership"):
        validate_main_episode_ids(train_metadata, splits["train"] + splits["validation"])
    altered = {**metadata, "episode_splits": {**splits, "train": splits["validation"]}}
    with pytest.raises(ValueError, match="overlapping"):
        validate_main_artifact(altered)


def test_task_with_one_successful_episode_fails_instead_of_reusing_validation(tmp_path):
    root = _source(tmp_path, (True, True, False))
    manifest = tmp_path / "success.json"
    manifest.write_text(json.dumps(build_success_manifest([root])))
    rows = [CTETrainIndex(i * 65, f"{root}::episode-{i}", f"task-{i}", 0) for i in range(3)]
    with pytest.raises(ValueError, match="at least 2"):
        select_main_success_rows(rows, _config(root, manifest))


def test_main_canonicalizes_dataset_root_before_sample_identity_is_constructed(tmp_path, monkeypatch):
    root = _source(tmp_path)
    manifest = tmp_path / "success.json"
    manifest.write_text(json.dumps(build_success_manifest([root])))
    monkeypatch.chdir(tmp_path)
    cfg = _config(root, manifest)
    cfg.data.train.dataset_dirs = ["./demonstrations"]
    normalize_main_dataset_paths(cfg)
    assert list(cfg.data.train.dataset_dirs) == [str(root.resolve())]
    selected, metadata = select_main_success_rows(_rows(cfg.data.train.dataset_dirs[0]), cfg)
    assert [row.dataset_index for row in selected] == [0, 130]
    validate_main_episode_ids(metadata, [row.episode_id for row in selected])
    cfg.model.zeva.profile = "legacy"
    cfg.data.train.dataset_dirs = ["./demonstrations"]
    normalize_main_dataset_paths(cfg)
    assert list(cfg.data.train.dataset_dirs) == ["./demonstrations"]
