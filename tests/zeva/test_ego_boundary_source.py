"""The cache entry point binds source trajectories before RGB feature export."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.ego_boundary_cache import ALL_CTE_FRAMES, EGO_BOUNDARY_CACHE_FORMAT, EgoBoundaryCache
from fastwam.zeva.ego_cte_training import build_episode_splits, canonical_sha256, select_ego_episode_scope
from test_ego_effect_pipeline import _CausalCTE, make_dataset


@pytest.mark.parametrize("drift", [None, "length", "offset", "task", "missing_identity"])
def test_cache_export_requires_same_stage1_episode_metadata_before_decoding(tmp_path, monkeypatch, drift):
    script = Path(__file__).resolve().parents[2] / "scripts/build_zeva_ego_boundary_cache.py"
    spec = importlib.util.spec_from_file_location("ego_boundary_source_test", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    source, inner = make_dataset()
    episodes = [asdict(episode) for episode in source.episodes]
    ids = [episode["episode_id"] for episode in episodes]
    splits = {"train": ids[:2], "validation": ids[2:]}
    cte_path, stats_path, semantic_path = [tmp_path / name for name in ("cte.pt", "stats.json", "semantic.json")]
    for path in (cte_path, stats_path, semantic_path):
        path.write_text("test artifact")
    metadata = {
        "data_identity": {"episode_metadata_sha256": canonical_sha256(episodes)},
        "dataset_stats_sha256": checkpoint_sha256(stats_path),
        "semantic_task_map_sha256": checkpoint_sha256(semantic_path),
    }
    # Keep every episode ID and normalization artifact unchanged. Only the
    # trajectory metadata drifts, which an ID-set check cannot detect.
    if drift == "length":
        inner.episode_data_index["to"][0] += 1
    elif drift == "offset":
        inner.episode_data_index["from"][1] += 1
    elif drift == "task":
        source.base_dataset.resolve_semantic_task_id = lambda **_: "different_task"
    elif drift == "missing_identity":
        metadata["data_identity"] = {"data_config_sha256": "unrelated_config_hash"}
    cte = _CausalCTE()
    monkeypatch.setattr(builder, "load_ego_cte_checkpoint", lambda *args, **kwargs: (
        cte, {"manifest": metadata, "episode_splits": splits},
    ))
    monkeypatch.setattr(builder, "instantiate", lambda _: source.base_dataset)
    output = tmp_path / "boundary_cache"
    cfg = OmegaConf.create({
        "device": "cpu", "model": {"zeva": {"cte": {"checkpoint": str(cte_path), "require_reference_recipe": False},
                                               "cache": {"path": str(output)}}},
        "data": {"train": {"pretrained_norm_stats": str(stats_path), "semantic_task_map_path": str(semantic_path),
                            "dataset_dirs": [str(inner.root)]}},
    })
    if drift is None:
        builder.main.__wrapped__(cfg)
        cache = EgoBoundaryCache(output)
        assert cache.splits == splits and set(cache.episodes) == set(ids)
        assert inner.calls and cte.step_calls
    else:
        with pytest.raises(ValueError, match="episode metadata differs"):
            builder.main.__wrapped__(cfg)
        assert not output.exists() and not inner.calls and not cte.step_calls


def test_reference_cache_entry_binds_actual_scope_and_disables_export_tf32(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/build_zeva_ego_boundary_cache.py"
    spec = importlib.util.spec_from_file_location("ego_reference_boundary_source_test", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source, inner = make_dataset()
    subset_path, conditions_path = tmp_path / "tasks.json", tmp_path / "conditions.json"
    subset_path.write_text(json.dumps({"task_names": ["beat_block_hammer"]}))
    conditions_path.write_text(json.dumps({episode.episode_id: "Clean" for episode in source.source_episodes}))
    scope, population = select_ego_episode_scope(source, task_subset_path=subset_path,
                                                 conditions_path=conditions_path, conditions=["Clean"])
    splits = build_episode_splits(source.episodes, partition_rows=population)
    cte_path, stats_path, semantic_path = [tmp_path / name for name in ("cte.pt", "stats.json", "semantic.json")]
    for path in (cte_path, stats_path, semantic_path):
        path.write_text("test artifact")
    metadata = {
        "tasks": scope["task_names"], "data_scope": scope, "episode_splits": splits,
        "recipe": {"split_policy": "zeva_ego", "seed": 11, "split_seed": 1000, "validation_fraction": .05},
        "data_identity": {"episode_metadata_sha256": canonical_sha256([asdict(episode) for episode in source.episodes])},
        "dataset_stats_sha256": checkpoint_sha256(stats_path),
        "semantic_task_map_sha256": checkpoint_sha256(semantic_path),
    }
    cte, calls = _CausalCTE(), []

    def load(path, **kwargs):
        calls.append(kwargs)
        return cte, {"manifest": metadata, "episode_splits": splits}

    monkeypatch.setattr(builder, "load_ego_cte_checkpoint", load)
    monkeypatch.setattr(builder, "instantiate", lambda _: source.base_dataset)
    output = tmp_path / "reference_boundaries"
    cfg = OmegaConf.create({
        "device": "cpu", "model": {"zeva": {
            "cte": {"checkpoint": str(cte_path), "require_reference_recipe": True, "exploratory_epoch40": False},
            "cache": {"path": str(output), "export_scope": ALL_CTE_FRAMES},
        }},
        "data": {"train": {"pretrained_norm_stats": str(stats_path), "semantic_task_map_path": str(semantic_path),
                            "dataset_dirs": [str(inner.root)]}},
    })
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        builder.main.__wrapped__(cfg)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul_tf32, cudnn_tf32
    assert calls == [{"device": "cpu", "require_ready": True,
                      "require_reference_recipe": True, "exploratory_epoch40": False}]
    cache = EgoBoundaryCache(output)
    assert cache.manifest.schema_version == EGO_BOUNDARY_CACHE_FORMAT
    assert cache.manifest.export_scope == ALL_CTE_FRAMES
    assert cache.manifest.export_tf32 is False
    assert cache.manifest.data_scope == scope
    assert cache.splits == splits and inner.calls and cte.step_calls
