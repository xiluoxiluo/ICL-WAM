"""The cache entry point binds source trajectories before RGB feature export."""

from dataclasses import asdict
import importlib.util
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.ego_boundary_cache import EgoBoundaryCache
from fastwam.zeva.ego_cte_training import canonical_sha256
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
        "device": "cpu", "model": {"zeva": {"cte": {"checkpoint": str(cte_path)}, "cache": {"path": str(output)}}},
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
