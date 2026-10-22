"""Cache full-prefix CTE predictions on RoboTwin's actual H24 replan grid.

  PYTHONPATH=src python scripts/build_zeva_ego_boundary_cache.py \
    task=robotwin_zeva_ego_policy_3cam_384 \
    model.zeva.cte.checkpoint=/runs/cte/cte.pt \
    model.zeva.cache.path=/data/ego_boundary_cache

The source dataset must contain the CTE checkpoint's complete episode split.
An independent split is never silently regenerated at this stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.ego_boundary_cache import build_ego_boundary_cache
from fastwam.zeva.ego_cte_training import load_ego_cte_checkpoint
from fastwam.zeva.ego_episode_data import RoboTwinEgoEpisodeDataset


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    zeva = cfg.model.zeva
    cte_path, output = str(zeva.cte.checkpoint), str(zeva.cache.path)
    if cte_path in ("", "None", "null") or output in ("", "None", "null"):
        raise ValueError("set model.zeva.cte.checkpoint and model.zeva.cache.path")
    device = str(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    cte, payload = load_ego_cte_checkpoint(cte_path, device=device, require_ready=True)
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    with open_dict(data_cfg):
        data_cfg.val_set_proportion = 0.0
        data_cfg.global_sample_stride = 1
        data_cfg.skip_padding_as_possible = False
        data_cfg.use_text_embed_cache = False
    stats_path = Path(str(data_cfg.pretrained_norm_stats))
    semantic_path = Path(str(data_cfg.semantic_task_map_path))
    identity = {"cte_checkpoint_sha256": checkpoint_sha256(cte_path),
                "dataset_stats_sha256": checkpoint_sha256(stats_path),
                "semantic_task_map_sha256": checkpoint_sha256(semantic_path),
                "dataset_path": str(Path(data_cfg.dataset_dirs[0]).expanduser().resolve())}
    metadata = payload.get("metadata", payload.get("manifest", {}))
    for key in ("dataset_stats_sha256", "semantic_task_map_sha256"):
        checkpoint_value = payload.get(key, metadata.get(key))
        if checkpoint_value != identity[key]:
            raise ValueError(f"boundary cache source identity differs from CTE training: {key}")
    splits = payload.get("episode_splits", metadata.get("episode_splits"))
    if not isinstance(splits, dict):
        raise ValueError("ready CTE checkpoint must preserve its explicit episode_splits")
    dataset = RoboTwinEgoEpisodeDataset(instantiate(data_cfg),
        executed_horizon=cte.cfg.executed_horizon, image_size=cte.cfg.image_size, action_dim=cte.cfg.action_dim,
        episode_ids=[episode for values in splits.values() for episode in values])
    manifest = build_ego_boundary_cache(dataset, cte, output, identity=identity, episode_splits=splits)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
