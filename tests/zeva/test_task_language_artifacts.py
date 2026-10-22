"""Language-only artifact construction, split isolation and prepared validation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.ego_task_memory import (
    build_ego_task_language_context_cache, episode_task_instructions, load_ego_task_language_features,
    ordered_episode_ids, save_ego_task_language_artifacts, validate_ego_task_language_static_context,
)
from fastwam.zeva.prepared_stage2 import StaticEpisodeContextCache
from fastwam.zeva.task_context import TaskContextBank
from fastwam.zeva.task_language import (
    CLASSIFIER_FORMAT, TASK_LANGUAGE_FEATURE_KIND, FrozenTaskLanguageEncoder,
    TaskLanguageHead, TaskLanguageHeadConfig, TaskLanguageMemory, export_task_language_encoder,
)
from fastwam.zeva.task_language_training import train_task_language_classifier


class Cache:
    def __init__(self):
        self.manifest = SimpleNamespace(retrieval_dim=128, global_dim=256, executed_horizon=24)
        self.splits = {"train": [f"episode-{i}" for i in range(6)], "validation": ["episode-6", "episode-7"]}
        self.episodes = {f"episode-{i}": {"dataset_start": i * 49, "episode_length": 49,
                                        "task_id": str(i % 2), "split": "train" if i < 6 else "validation"}
                         for i in range(8)}
        self.rows = [{"episode_id": identity, "task_id": info["task_id"], "raw_step": step,
                      "split": info["split"]} for identity, info in self.episodes.items() for step in (0, 24)]
        self.read = []

    def get(self, index):
        row = self.rows[index]
        self.read.append(row["episode_id"])
        number = int(row["episode_id"].split("-")[-1])
        key, value = torch.zeros(128), torch.zeros(256)
        key[number % 2], key[2] = 1.0, 0.1 * number
        value[number % 2], value[2], value[3] = 1.0, 0.2 * number, row["raw_step"] / 24
        return {**row, "retrieval": key, "global": value}


def encoder_artifact(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    table = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.]])
    export_task_language_encoder(path, table, fast, max_length=8)
    return FrozenTaskLanguageEncoder.load(path)


@pytest.fixture
def artifacts(tmp_path):
    cache = Cache()
    paths = {"bank_path": tmp_path / "bank.pt", "feature_path": tmp_path / "language_features.pt",
             "encoder_path": tmp_path / "encoder.pt", "head_path": tmp_path / "classifier.pt"}
    encoder = encoder_artifact(paths["encoder_path"])
    instructions = {identity: "alpha" if i % 2 == 0 else "beta" for i, identity in enumerate(ordered_episode_ids(cache))}
    # Deliberately make a held-out task-1 episode say alpha: retrieval must
    # follow language prediction rather than its ground-truth task metadata.
    instructions["episode-7"] = "alpha"
    features = {identity: encoder.encode(text)[0] for identity, text in instructions.items()}
    metadata = {"cte_checkpoint_sha256": "cte", "dataset_stats_sha256": "stats",
                "semantic_task_map_sha256": "semantic", "phase_cache_manifest_sha256": "boundaries",
                "episode_splits_sha256": "splits"}
    save_ego_task_language_artifacts(cache, features, instructions=instructions, metadata=metadata,
                                      **{k: paths[k] for k in ("bank_path", "feature_path", "encoder_path")})
    bank = TaskContextBank.load(paths["bank_path"])
    training = cache.splits["train"]
    labels = torch.tensor([int(cache.episodes[identity]["task_id"]) for identity in training])
    keys = torch.stack([entry["retrieval_key"] for entry in bank.entries])
    torch.manual_seed(9)
    config = TaskLanguageHeadConfig(input_dim=4, hidden_dim=16, output_dim=128, dropout=0.0)
    head = TaskLanguageHead(config)
    trained = train_task_language_classifier(head, torch.stack([features[identity] for identity in training]),
                                             keys, labels, steps=40, batch_size=4, learning_rate=.03,
                                             eval_every=10, seed=3)
    assert trained["best_task_top1"] == 1.0
    checkpoint = {"format": CLASSIFIER_FORMAT, "source_feature": "task_language",
                  "feature_kind": TASK_LANGUAGE_FEATURE_KIND, "config": config.as_dict(), "model": head.state_dict(),
                  "task_names": ["0", "1"], "task_prototypes": trained["task_prototypes"], "step": trained["step"],
                  "bank_sha256": checkpoint_sha256(paths["bank_path"]),
                  "encoder_sha256": checkpoint_sha256(paths["encoder_path"]),
                  "feature_cache_sha256": checkpoint_sha256(paths["feature_path"]),
                  "prototype_scope": "classifier_training_episodes_only",
                  "train_episode_ids": [training[index] for index in trained["train_indices"]],
                  "validation_episode_ids": [training[index] for index in trained["validation_indices"]]}
    torch.save(checkpoint, paths["head_path"])
    return cache, paths, metadata


def test_language_bank_reads_only_train_trajectories_but_features_cover_all_queries(artifacts):
    cache, paths, metadata = artifacts
    assert set(cache.read) == set(cache.splits["train"])
    bank = TaskContextBank.load(paths["bank_path"])
    payload = load_ego_task_language_features(paths["feature_path"], paths["bank_path"], paths["encoder_path"],
                                              cache, expected_metadata=metadata)
    assert payload["episode_ids"] == ordered_episode_ids(cache)
    assert {entry["episode_id"] for entry in bank.entries} == set(cache.splits["train"])
    assert bank.temperature == 1.0
    assert bank.metadata["foundation_independent_query"] is True
    assert "base_checkpoint_sha256" not in bank.metadata


def test_static_context_uses_predicted_task_and_excludes_self_from_mean_and_support(artifacts, tmp_path):
    cache, paths, metadata = artifacts
    target = tmp_path / "static.pt"
    build_ego_task_language_context_cache(cache=cache, output_path=target, expected_metadata=metadata,
                                           top_k=5, batch_size=3, **paths)
    static = StaticEpisodeContextCache(target)
    payload = torch.load(target, weights_only=False)
    query = validate_ego_task_language_static_context(static, cache=cache, expected_metadata=metadata, top_k=5, **paths)
    assert payload["retrieval_sources"][-1]["predicted_task"] == "0"
    assert cache.episodes["episode-7"]["task_id"] == "1"
    bank = TaskContextBank.load(paths["bank_path"])
    keys = torch.stack([entry["retrieval_key"] for entry in bank.entries])
    values = torch.stack([entry["behavior_value"] for entry in bank.entries])
    for i, identity in enumerate(query["episode_ids"]):
        predicted = payload["retrieval_sources"][i]["predicted_task"]
        available = [j for j, entry in enumerate(bank.entries)
                     if entry["task_id"] == predicted and entry["episode_id"] != identity]
        mean = F.normalize(keys[available].mean(0), dim=0)
        scores = mean @ keys[available].T
        expected = (scores.softmax(-1)[:, None] * values[available]).sum(0)
        torch.testing.assert_close(static.get(identity), expected)
        valid_indices = payload["retrieval_indices"][i]
        assert set(valid_indices[valid_indices >= 0].tolist()) == set(available)
        assert len(payload["retrieval_sources"][i]["top_k"]) == len(available)


def test_reference_training_context_uses_the_same_train_memory_as_inference(artifacts, tmp_path):
    from fastwam.zeva.task_language import TaskLanguageMemory

    cache, paths, metadata = artifacts
    target = tmp_path / "reference_static.pt"
    build_ego_task_language_context_cache(cache=cache, output_path=target, expected_metadata=metadata,
                                         top_k=5, batch_size=3, train_query_exclusion="none", **paths)
    static = StaticEpisodeContextCache(target)
    query = validate_ego_task_language_static_context(
        static, cache=cache, expected_metadata=metadata, train_query_exclusion="none", **paths)
    memory = TaskLanguageMemory.load(paths["bank_path"], paths["head_path"], encoder_path=paths["encoder_path"])
    expected = memory.retrieve(query["features"])
    payload = torch.load(target, weights_only=False)
    torch.testing.assert_close(payload["task_contexts"], expected.values)
    assert static.metadata["train_query_exclusion"] == "none"
    assert static.metadata["validation_query_sources"] == "train_only"
    for row, identity in enumerate(query["episode_ids"]):
        sources = payload["retrieval_sources"][row]["top_k"]
        assert all(item["episode_id"] in cache.splits["train"] for item in sources)
        # In this small fixture each task has fewer than five demos, so the
        # reference includes the query's own train demo in its fixed task bank.
        if identity in cache.splits["train"]:
            assert identity in {item["episode_id"] for item in sources}
    # The intentionally contradictory held-out GT task still cannot route
    # language queries or supply a validation trajectory to the memory.
    assert payload["retrieval_sources"][-1]["predicted_task"] == "0"
    assert cache.episodes["episode-7"]["task_id"] == "1"
    with pytest.raises(ValueError, match="train_query_exclusion"):
        validate_ego_task_language_static_context(
            static, cache=cache, expected_metadata=metadata, train_query_exclusion="leave_one_episode_out", **paths)


def test_unknown_training_exclusion_is_rejected_before_artifact_creation(artifacts, tmp_path):
    cache, paths, metadata = artifacts
    output = tmp_path / "invalid_exclusion.pt"
    with pytest.raises(ValueError, match="train_query_exclusion"):
        build_ego_task_language_context_cache(cache=cache, output_path=output, expected_metadata=metadata,
                                             train_query_exclusion="ground_truth_task", **paths)
    assert not output.exists()


def test_classifier_selection_cannot_consume_cte_heldout_episode(artifacts, tmp_path):
    cache, paths, metadata = artifacts
    checkpoint = torch.load(paths["head_path"], weights_only=False)
    checkpoint["validation_episode_ids"] = cache.splits["validation"]
    torch.save(checkpoint, paths["head_path"])
    with pytest.raises(ValueError, match="inside the CTE training split"):
        build_ego_task_language_context_cache(cache=cache, output_path=tmp_path / "invalid.pt",
                                               expected_metadata=metadata, **paths)


def test_language_features_reject_old_readouts_and_encoder_drift(artifacts):
    cache, paths, metadata = artifacts
    payload = torch.load(paths["feature_path"], weights_only=False)
    wrong = deepcopy(payload)
    wrong["feature_kind"] = "frozen_fastwam_initial_readout"
    torch.save(wrong, paths["feature_path"])
    with pytest.raises(ValueError, match="task-only"):
        load_ego_task_language_features(paths["feature_path"], paths["bank_path"], paths["encoder_path"], cache)
    torch.save(payload, paths["feature_path"])
    with paths["encoder_path"].open("ab") as handle:
        handle.write(b"changed encoder")
    with pytest.raises(ValueError, match="encoder_sha256"):
        load_ego_task_language_features(paths["feature_path"], paths["bank_path"], paths["encoder_path"], cache)


def test_prepared_context_rejects_artifact_or_query_scope_drift(artifacts, tmp_path):
    cache, paths, metadata = artifacts
    target = tmp_path / "static.pt"
    build_ego_task_language_context_cache(cache=cache, output_path=target, expected_metadata=metadata, **paths)
    static = StaticEpisodeContextCache(target)
    static.metadata["scope_selection"] = "ground_truth_task"
    with pytest.raises(ValueError, match="scope_selection"):
        validate_ego_task_language_static_context(static, cache=cache, expected_metadata=metadata, **paths)
    static.metadata["scope_selection"] = "predicted_task_only"
    static.metadata["base_checkpoint_sha256"] = "mutable foundation"
    with pytest.raises(ValueError, match="mutable foundation"):
        validate_ego_task_language_static_context(static, cache=cache, expected_metadata=metadata, **paths)


class Table:
    def __init__(self, rows):
        self.rows, self.column_names = rows, list(rows[0])

    def with_format(self, _):
        return self

    def select_columns(self, names):
        assert all(name in {"episode_index", "frame_index", "task_index"} for name in names)
        return Table([{key: row[key] for key in names} for row in self.rows])

    def __getitem__(self, selection):
        selected = self.rows[selection]
        return {key: [row[key] for row in selected] for key in self.column_names}


def metadata_source():
    records = [{"episode_index": i, "frame_index": frame, "task_index": i % 2}
               for i in range(2) for frame in range(49)]
    inner = SimpleNamespace(root="/dataset", hf_dataset=Table(records), meta=SimpleNamespace(tasks={0: "pick alpha", 1: "place beta"}))
    base = SimpleNamespace(lerobot_dataset=SimpleNamespace(multi_dataset=SimpleNamespace(_datasets=[inner])),
                           resolve_semantic_task_id=lambda **kwargs: str(kwargs["raw_task_index"]))
    cache = SimpleNamespace(episodes={f"/dataset::episode-{i}": {"dataset_start": i * 49, "episode_length": 49,
                                                               "episode_index": i, "task_id": str(i % 2)} for i in range(2)})
    return base, cache, inner


def test_instruction_extraction_uses_only_raw_metadata_without_video_prompt_or_images():
    base, cache, _ = metadata_source()
    actual = episode_task_instructions(base, cache)
    assert actual == {"/dataset::episode-0": "pick alpha", "/dataset::episode-1": "place beta"}


def test_static_language_queries_reject_changed_instruction_within_episode():
    base, cache, inner = metadata_source()
    inner.hf_dataset.rows[24]["task_index"] = 1
    with pytest.raises(ValueError, match="fixed instruction"):
        episode_task_instructions(base, cache)


def test_language_bank_builder_runs_on_cpu_without_foundation_checkpoint_or_image_readout(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    module_path = Path(__file__).resolve().parents[2] / "scripts/build_zeva_ego_task_memory.py"
    spec = importlib.util.spec_from_file_location("task_language_memory_builder_test", module_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    cache = Cache()
    rename = {old: f"/dataset::episode-{i}" for i, old in enumerate(ordered_episode_ids(cache))}
    cache.episodes = {rename[old]: {**meta, "episode_index": int(old.split("-")[-1])}
                      for old, meta in cache.episodes.items()}
    cache.splits = {split: [rename[old] for old in ids] for split, ids in cache.splits.items()}
    cache.rows = [{**row, "episode_id": rename[row["episode_id"]]} for row in cache.rows]
    # Cache.get uses the final numeric identity component and remains usable.
    metadata_rows = [{"episode_index": i, "frame_index": frame, "task_index": i % 2}
                     for i in range(8) for frame in range(49)]
    inner = SimpleNamespace(root="/dataset", hf_dataset=Table(metadata_rows),
                            meta=SimpleNamespace(tasks={0: "alpha", 1: "beta"}))
    base = SimpleNamespace(lerobot_dataset=SimpleNamespace(multi_dataset=SimpleNamespace(_datasets=[inner])),
                           resolve_semantic_task_id=lambda **kwargs: str(kwargs["raw_task_index"]))
    stats, semantic, cte, boundary = (tmp_path / name for name in ("stats.json", "tasks.json", "cte.pt", "boundary"))
    stats.write_text("{}")
    semantic.write_text("{}")
    stats_sha, semantic_sha = checkpoint_sha256(stats), checkpoint_sha256(semantic)
    torch.save({"episode_splits": cache.splits, "dataset_stats_sha256": stats_sha,
                "semantic_task_map_sha256": semantic_sha}, cte)
    boundary.mkdir()
    (boundary / "manifest.json").write_text("{}")
    (boundary / "episode_splits.json").write_text("{}")
    cache.manifest.dataset_stats_sha256 = stats_sha
    cache.manifest.semantic_task_map_sha256 = semantic_sha
    cache.manifest.cte_checkpoint_sha256 = checkpoint_sha256(cte)
    cache.manifest.dataset_path = "/dataset"
    encoder = tmp_path / "encoder.pt"
    encoder_artifact(encoder)
    cfg = OmegaConf.create({
        "ckpt": None, "device": "cpu", "model": {"tokenizer_max_len": 128, "zeva": {
            "cte": {"variant": "zeva_ego", "input_type": "rgb_multiview", "checkpoint": str(cte),
                    "retrieval_dim": 128, "hidden_dim": 256}, "cache": {"path": str(boundary)},
            "task_context": {"mode": "task_language", "build_stage": "bank", "temperature": 1.0,
                             "bank_path": str(tmp_path / "new_bank.pt"), "language_encoder_path": str(encoder),
                             "language_feature_cache_path": str(tmp_path / "new_features.pt")}}},
        "data": {"train": {"_target_": "metadata_source_only", "pretrained_norm_stats": str(stats),
                           "semantic_task_map_path": str(semantic), "dataset_dirs": ["/dataset"],
                           "video_size": [384, 320], "context_len": 128, "is_training_set": True,
                           "val_set_proportion": 0.0, "global_sample_stride": 1, "use_text_embed_cache": True}},
    })
    calls = []

    def metadata_only(config):
        assert config._target_ == "metadata_source_only"
        assert config.use_text_embed_cache is False
        calls.append(config._target_)
        return base

    monkeypatch.setattr(builder, "instantiate", metadata_only)
    monkeypatch.setattr(builder, "EgoBoundaryCache", lambda _: cache)
    monkeypatch.setattr(builder, "require_ego_checkpoint_ready", lambda _: None)
    monkeypatch.setattr(builder, "extract_initial_readouts", lambda *a, **k: pytest.fail("image readout was used"))
    result = builder.build(cfg)
    assert calls == ["metadata_source_only"]
    assert result["train_episodes"] == 6 and result["query_episodes"] == 8
