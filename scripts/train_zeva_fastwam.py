"""Stage 2: train only CausalPromptEncoder/BehaviorPrefixAdapter/gate."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.datasets.zeva_robotwin_dataset import ZevaStage2Dataset
from fastwam.runtime import _mixed_precision_to_model_dtype
from fastwam.trainer import Wan22Trainer
from fastwam.zeva.cache import PhaseEffectCache
from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.schemas import sha256_file


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Zeva Stage 2 requires CUDA for the frozen Wan2.2 FastWAM action path")
    zeva = cfg.model.get("zeva", {})
    cache_path = str(zeva.get("cache", {}).get("path"))
    if not cache_path or cache_path in {"None", "null"}:
        raise ValueError("Set model.zeva.cache.path for Stage 2")
    cte_path = str(zeva.get("cte", {}).get("checkpoint"))
    if not cte_path or cte_path in {"None", "null"}:
        raise ValueError("Set model.zeva.cte.checkpoint for Stage 2 compatibility checks")
    checkpoint = cfg.get("ckpt")
    if checkpoint in (None, "", "None", "null") or not Path(str(checkpoint)).is_file():
        raise FileNotFoundError(
            "Zeva Stage 2 requires an existing frozen FastWAM base checkpoint via ckpt"
        )
    cache_cfg = zeva.get("cache", {})
    if int(cfg.data.train.get("global_sample_stride", 1)) != 1:
        raise ValueError(
            "RoboTwin Zeva V1 requires data.train.global_sample_stride=1 for exact "
            f"frame/action alignment; got {cfg.data.train.global_sample_stride}"
        )
    expected = {
        "action_dim": 14,
        "action_group_size": 4,
        "action_horizon": 32,
        "video_frames": 9,
        "camera_keys": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        "action_video_freq_ratio": int(cfg.data.train.action_video_freq_ratio),
        "action_normalization": "fastwam_processor_output",
    }
    prompt_cfg = cfg.model.get("zeva", {}).get("prompt", {})
    expected["phase_dim"] = int(prompt_cfg.get("phase_dim", 128))
    expected["effect_dim"] = int(prompt_cfg.get("effect_dim", 128))
    expected["cte_checkpoint_sha256"] = checkpoint_sha256(cte_path)
    stats_path = str(cfg.data.train.get("pretrained_norm_stats", ""))
    if stats_path in {"", "None", "null"} or not Path(stats_path).is_file():
        raise FileNotFoundError(
            "Zeva Stage 2 requires an existing data.train.pretrained_norm_stats file"
        )
    expected["dataset_stats_sha256"] = sha256_file(stats_path)
    if bool(cache_cfg.get("strict_manifest", True)):
        expected["dataset_path"] = str(cfg.data.train.dataset_dirs[0])
    cache = PhaseEffectCache.load(cache_path, expected=expected)
    base = instantiate(cfg.data.train)
    top_k = int(zeva.get("memory", {}).get("pim_top_k", 4))
    bit_size = int(zeva.get("memory", {}).get("bit_size", 4))
    if top_k != int(prompt_cfg.get("persistent_length", 4)):
        raise ValueError(
            "zeva.memory.pim_top_k must match zeva.prompt.persistent_length; "
            f"got {top_k} vs {prompt_cfg.get('persistent_length', 4)}"
        )
    if bit_size != int(prompt_cfg.get("brief_length", 4)):
        raise ValueError(
            "zeva.memory.bit_size must match zeva.prompt.brief_length; "
            f"got {bit_size} vs {prompt_cfg.get('brief_length', 4)}"
        )
    dataset = ZevaStage2Dataset(base, cache, top_k=top_k, bit_size=bit_size)
    model = instantiate(cfg.model, model_dtype=_mixed_precision_to_model_dtype(str(cfg.mixed_precision)), device="cuda")
    model.load_checkpoint(str(checkpoint))
    trainer = Wan22Trainer(cfg=cfg, model=model, train_dataset=dataset, val_dataset=None)
    trainer.train()
    print(f"Stage 2 complete; addon checkpoints are under {Path(str(cfg.output_dir)) / 'checkpoints'}")


if __name__ == "__main__":
    main()
