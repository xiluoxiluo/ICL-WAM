"""Build phase/effect shards from a frozen Stage 1 CTE."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva import CausalTransitionEncoder, CausalTransitionEncoderConfig
from fastwam.zeva.cache import save_phase_effect_cache
from fastwam.zeva.checkpoint import checkpoint_sha256, load_cte_checkpoint
from fastwam.zeva.schemas import CacheManifest, sha256_file


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # The Zeva block belongs to the FastWAM model config.  Keep all training
    # and cache entry points on the same ``model.zeva`` namespace so Hydra
    # overrides also reach ``create_fastwam``.
    zeva = cfg.model.get("zeva", {})
    cte_path = str(zeva.get("cte", {}).get("checkpoint"))
    cache_path = str(zeva.get("cache", {}).get("path"))
    if not cte_path or cte_path in {"None", "null"} or not cache_path or cache_path in {"None", "null"}:
        raise ValueError("Set model.zeva.cte.checkpoint and model.zeva.cache.path")
    stats_path = str(cfg.data.train.get("pretrained_norm_stats", ""))
    if stats_path in {"", "None", "null"} or not Path(stats_path).is_file():
        raise FileNotFoundError(
            "Zeva cache construction requires an existing data.train.pretrained_norm_stats file"
        )
    base = instantiate(cfg.data.train)
    dataset = ZevaRobotWinDataset(base)
    sample_stride = int(cfg.data.train.get("global_sample_stride", 1))
    if sample_stride != 1:
        raise ValueError(
            "RoboTwin Zeva V1 requires data.train.global_sample_stride=1 for exact "
            f"frame/action alignment; got {sample_stride}"
        )
    payload = torch.load(cte_path, map_location="cpu", weights_only=False)
    if int(payload.get("action_dim", -1)) != 14 or tuple(payload.get("camera_keys", ())) != ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        raise ValueError(
            "CTE checkpoint metadata is incompatible with RoboTwin V1 "
            "(action_dim=14, cameras=cam_high/cam_left_wrist/cam_right_wrist)"
        )
    cte_config = dict(payload.get("config", {}))
    # Older checkpoints stored the CTE config nested under ``cte``.
    cte_config = dict(cte_config.get("cte", cte_config))
    allowed = set(CausalTransitionEncoderConfig.__dataclass_fields__)
    model = CausalTransitionEncoder(CausalTransitionEncoderConfig(**{k: v for k, v in cte_config.items() if k in allowed}))
    if model.cfg.action_dim != 14 or model.cfg.transition_steps != 4:
        raise ValueError(
            "RoboTwin Zeva V1 cache requires CTE action_dim=14 and transition_steps=4; "
            f"got action_dim={model.cfg.action_dim}, transition_steps={model.cfg.transition_steps}"
        )
    if int(cfg.data.train.action_video_freq_ratio) != 4:
        raise ValueError(
            "RoboTwin Zeva V1 requires action_video_freq_ratio=4; "
            f"got {cfg.data.train.action_video_freq_ratio}"
        )
    load_cte_checkpoint(cte_path, model, map_location="cpu")
    model.eval()
    records = []
    # RobotVideoDataset exposes overlapping windows (normally stride=1).  The
    # CTE is recurrent, so cache construction must consume the same ordered,
    # non-overlapping 32-action chunks as Stage 1 and carry the final state to
    # the next chunk in an episode.  Encoding every sliding window from the
    # default BOS state would duplicate transitions in PIM and break the
    # offline/online state machine.
    next_episode_step: dict[str, int] = {}
    carried_state: dict[str, torch.Tensor] = {}
    seen_source_indices: set[int] = set()
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            episode = sample["episode"]
            episode_id = str(episode.episode_id)
            episode_step = int(episode.episode_step)
            expected_step = next_episode_step.get(episode_id)
            if expected_step is not None:
                if episode_step < expected_step:
                    # Overlapping source windows are deliberately not cached;
                    # their start state is already represented by the previous
                    # non-overlapping chunk.
                    continue
                if episode_step != expected_step:
                    # A gap means the recurrent state cannot be joined safely.
                    # Start a new causal segment rather than silently carrying
                    # state across an unknown interval.
                    carried_state.pop(episode_id, None)
                    next_episode_step.pop(episode_id, None)

            source_index = int(sample.get("dataset_index", index))
            if source_index in seen_source_indices:
                raise ValueError(
                    f"dataset returned duplicate source index {source_index}; "
                    "cache construction cannot preserve deterministic joins"
                )
            frames = torch.cat((sample["before_frames"], sample["after_frames"][-1:]), dim=0).unsqueeze(0)
            actions = sample["transition_actions"].unsqueeze(0)
            initial_state = torch.zeros((1, model.cfg.hidden_dim), dtype=frames.dtype)
            initial_state_mask = torch.zeros((1,), dtype=torch.bool)
            if episode_id in carried_state:
                initial_state[0] = carried_state[episode_id]
                initial_state_mask[0] = True
            output = model(
                frames,
                actions,
                valid_mask=sample["frame_valid"].unsqueeze(0),
                transition_valid=sample["transition_valid"].unsqueeze(0),
                initial_state=initial_state,
                initial_state_mask=initial_state_mask,
            )
            # Stage 2 queries phase_pre for the window's first action. If t=0
            # is invalid, later valid records cannot be safely joined to this
            # 32-action target, so discard the whole window.
            if not bool(output["transition_complete"][0, 0]):
                carried_state.pop(episode_id, None)
                next_episode_step.pop(episode_id, None)
                continue
            seen_source_indices.add(source_index)
            for t in range(8):
                if not bool(output["transition_complete"][0, t]):
                    continue
                records.append({
                    "episode_id": episode.episode_id,
                    "task_id": episode.task_id,
                    "attempt_id": 0,
                    "window_index": source_index,
                    "episode_step": episode_step + t * model.cfg.transition_steps * sample_stride,
                    "transition_index": t,
                    "phase_pre": output["phase"][0, t],
                    "phase_post": output["phase"][0, t + 1],
                    "effect": output["transition_effect"][0, t],
                    "valid": True,
                })
            if bool(sample["frame_valid"].all()) and bool(sample["transition_valid"].all()):
                carried_state[episode_id] = output["causal_interaction_state"][0, -1].detach().clone()
                next_episode_step[episode_id] = episode_step + 32 * sample_stride
            else:
                # A partial/padded window cannot establish the next causal
                # state. Do not let it contaminate a later segment.
                carried_state.pop(episode_id, None)
                next_episode_step.pop(episode_id, None)
    cache_cfg = zeva.get("cache", {})
    stats_hash = sha256_file(stats_path)
    manifest = CacheManifest(
        cte_checkpoint_sha256=checkpoint_sha256(cte_path),
        dataset_stats_sha256=stats_hash,
        dataset_path=str(cfg.data.train.dataset_dirs[0]),
        action_dim=model.cfg.action_dim,
        action_group_size=model.cfg.transition_steps,
        action_horizon=model.cfg.transition_steps * 8,
        video_frames=9,
        phase_dim=model.cfg.phase_dim,
        effect_dim=model.cfg.effect_dim,
        feature_dtype=str(cache_cfg.get("feature_dtype", "float32")),
        action_video_freq_ratio=int(cfg.data.train.action_video_freq_ratio),
    )
    if not records:
        raise RuntimeError("CTE cache construction produced no valid non-overlapping windows")
    save_phase_effect_cache(cache_path, records, manifest)
    print(f"saved {len(records)} phase/effect records to {cache_path}")


if __name__ == "__main__":
    main()
