import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from fastwam.zeva.main_data import build_success_manifest, select_main_success_rows
from fastwam.zeva.main_prepared import split_main_prepared_dataset
from fastwam.zeva.pim_training import PIMTrainingDropoutConfig, PIMTrainingDropoutDataset
from fastwam.zeva.prepared_stage2 import PreparedPIMShard
from fastwam.zeva.stage1_sampling import CTETrainIndex


def _prepared(tmp_path, *, missing_episode=False):
    root = tmp_path / "source"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_episodes": 4}))
    (root / "meta/episodes.jsonl").write_text("\n".join(
        json.dumps({"episode_index": i, "length": 80, "success": True}) for i in range(4)
    ))
    (root / "stats.json").write_text("{}")
    (root / "meta/semantic_task_map.json").write_text("{}")
    manifest = tmp_path / "success.json"
    manifest.write_text(json.dumps(build_success_manifest([root])))
    cfg = OmegaConf.create({
        "model": {"zeva": {"profile": "main_observed", "offline": {"success_manifest_path": str(manifest)}}},
        "data": {"train": {"dataset_dirs": [str(root)], "is_training_set": True,
                           "pretrained_norm_stats": str(root / "stats.json"),
                           "semantic_task_map_path": str(root / "meta/semantic_task_map.json")}},
    })
    episode_ids = tuple(f"{root}::episode-{i}" for i in range(4))
    rows = [CTETrainIndex(i * 80, episode_ids[i], "task", 0) for i in range(4)]
    _, metadata = select_main_success_rows(rows, cfg)
    cache_root = tmp_path / "cache"
    episodes = [2, 0, 3, 1, 0, 2, 1, 3]
    if missing_episode:
        episodes = [episode for episode in episodes if episode != 3]
    shards = []
    for i, start in enumerate(range(0, len(episodes), 3)):
        end = min(start + 3, len(episodes))
        name = f"shard_{i:03d}"
        (cache_root / name).mkdir(parents=True)
        np.save(cache_root / name / "episode_index.npy", np.asarray(episodes[start:end], dtype=np.int32))
        shards.append(PreparedPIMShard(name, start, end))
    cache = SimpleNamespace(root=cache_root, manifest=metadata, episode_ids=episode_ids,
                            _cross_attempt=None, shards=tuple(shards))

    class Source:
        pim = cache
        static_context = object()
        history_schedule = "legacy"

        def __len__(self):
            return len(episodes)

        def __getitem__(self, index):
            return {"source_query_index": index, "episode_id": episode_ids[episodes[index]]}

    return Source()


def test_fixed_episode_views_preserve_global_window_mapping_and_holdout(tmp_path):
    source = _prepared(tmp_path)
    train, validation = split_main_prepared_dataset(source)
    assert len(train) + len(validation) == len(source)
    for view in (train, validation):
        observed = {view[i]["episode_id"] for i in range(len(view))}
        assert observed == set(source.pim.manifest["episode_splits"][view.split])
        assert [view[i]["source_query_index"] for i in range(len(view))] == view.indices.tolist()
    assert not set(train.indices) & set(validation.indices)
    assert train.split_identity["query_indices_sha256"] != validation.split_identity["query_indices_sha256"]
    assert train.split_identity["episode_split_sha256"] == validation.split_identity["episode_split_sha256"]
    with pytest.raises(ValueError):
        train.indices[0] = 99
    wrapped = PIMTrainingDropoutDataset(train, PIMTrainingDropoutConfig())
    assert wrapped.split_identity == train.split_identity
    assert wrapped.pim is source.pim
    assert wrapped.static_context is source.static_context


def test_split_fails_instead_of_silently_dropping_an_episode(tmp_path):
    source = _prepared(tmp_path, missing_episode=True)
    with pytest.raises(ValueError, match="no query windows"):
        split_main_prepared_dataset(source)


def test_split_rejects_invalid_index_and_ego_semantics(tmp_path):
    source = _prepared(tmp_path)
    first = source.pim.root / source.pim.shards[0].path / "episode_index.npy"
    np.save(first, np.asarray([0, 1, 4], dtype=np.int32))
    with pytest.raises(ValueError, match="outside"):
        split_main_prepared_dataset(source)
    source.pim._cross_attempt = object()
    with pytest.raises(ValueError, match="Ego"):
        split_main_prepared_dataset(source)


def test_main_dropout_is_applied_before_mce_exactly_once():
    path = Path(__file__).resolve().parents[2] / "scripts/train_zeva_fastwam_prepared.py"
    spec = importlib.util.spec_from_file_location("main_prepared_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = OmegaConf.create({"seed": 7, "model": {"zeva": {
        "profile": "main_observed",
        "prompt": {"pim_context_dropout": 0.0, "pim_support_dropout": 0.0},
        "prepared_stage2": {"context_dropout": 0.2, "support_dropout": 0.3},
    }}})
    result = module._dropout_config(cfg)
    assert (result.context_dropout, result.support_dropout, result.seed) == (0.2, 0.3, 7)
