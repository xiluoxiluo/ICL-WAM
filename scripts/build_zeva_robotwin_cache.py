"""Build phase/effect shards from a frozen Stage 1 CTE."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva import CausalTransitionEncoder, CausalTransitionEncoderConfig, FastWAMCTELatentEncoder, load_frozen_wan_vae, validate_vae_metadata
from fastwam.zeva.cache import save_phase_effect_cache
from fastwam.zeva.checkpoint import checkpoint_sha256, load_cte_checkpoint
from fastwam.zeva.schemas import CacheManifest, sha256_file


def _video_size_hw(cfg: DictConfig) -> tuple[int, int]:
    value = cfg.data.train.get("video_size")
    if value is None or len(value) != 2:
        raise ValueError("data.train.video_size must be [H, W] for Zeva cache construction")
    size = tuple(int(v) for v in value)
    if min(size) < 1:
        raise ValueError(f"data.train.video_size must be positive, got {size}")
    return size


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
    cte_config = dict(payload.get("config", payload.get("model_config", {})))
    # Older checkpoints stored the CTE config nested under ``cte``.
    cte_config = dict(cte_config.get("cte", cte_config))
    cte_input_type = str(payload.get("cte_input_type", cte_config.get("input_type", "rgb_frame")))
    checkpoint_action_dim = int(payload.get("action_dim", cte_config.get("action_dim", -1)))
    if (
        checkpoint_action_dim != 14
        or cte_input_type not in {"rgb_frame", "wan_vae_latent"}
        or tuple(payload.get("camera_keys", ())) != ("cam_high", "cam_left_wrist", "cam_right_wrist")
    ):
        raise ValueError(
            "CTE checkpoint metadata is incompatible with RoboTwin V1 "
            "(input_type=rgb_frame|wan_vae_latent, action_dim=14, cameras=cam_high/cam_left_wrist/cam_right_wrist)"
        )
    checkpoint_vae_metadata = dict(payload.get("vae_metadata", {}))
    cte_vae_input_size = _video_size_hw(cfg)
    checkpoint_input_size = payload.get("cte_vae_input_size")
    if cte_input_type == "wan_vae_latent":
        if checkpoint_input_size is None or len(checkpoint_input_size) != 2:
            raise ValueError("wan_vae_latent CTE checkpoints must include cte_vae_input_size")
        checkpoint_input_size = tuple(int(v) for v in checkpoint_input_size)
        if checkpoint_input_size != cte_vae_input_size:
            raise ValueError(
                "CTE/cache VAE input size mismatch: checkpoint declares "
                f"{checkpoint_input_size}, data config uses {cte_vae_input_size}"
            )
    if cte_input_type == "wan_vae_latent" and not checkpoint_vae_metadata:
        raise ValueError("wan_vae_latent CTE checkpoints must include vae_metadata")
    if cte_input_type == "wan_vae_latent":
        required_vae_metadata = {
            "model_id",
            "vae_path",
            "z_dim",
            "temporal_downsample_factor",
            "upsampling_factor",
        }
        if not required_vae_metadata.issubset(checkpoint_vae_metadata):
            raise ValueError(
                "wan_vae_latent CTE checkpoints must record complete VAE "
                f"identity metadata: {sorted(required_vae_metadata)}"
            )
    allowed = set(CausalTransitionEncoderConfig.__dataclass_fields__)
    model = CausalTransitionEncoder(CausalTransitionEncoderConfig(**{k: v for k, v in cte_config.items() if k in allowed}))
    if (
        model.cfg.action_dim != 14
        or model.cfg.transition_steps != 4
        or model.cfg.effect_window_transitions != 4
    ):
        raise ValueError(
            "RoboTwin Zeva V1 cache requires CTE action_dim=14, "
            "transition_steps=4, and effect_window_transitions=4; "
            f"got action_dim={model.cfg.action_dim}, "
            f"transition_steps={model.cfg.transition_steps}, "
            f"effect_window_transitions={model.cfg.effect_window_transitions}"
        )
    if int(cfg.data.train.action_video_freq_ratio) != 4:
        raise ValueError(
            "RoboTwin Zeva V1 requires action_video_freq_ratio=4; "
            f"got {cfg.data.train.action_video_freq_ratio}"
        )
    load_cte_checkpoint(cte_path, model, map_location="cpu")
    configured_device = cfg.get("device")
    if configured_device is None or str(configured_device).strip().lower() in {"", "none", "null"}:
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    cte_device = torch.device(str(configured_device))
    if cte_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("latent CTE cache requested CUDA but CUDA is unavailable")
    frame_encoder = None
    vae_metadata: dict[str, object] = {}
    if cte_input_type == "wan_vae_latent":
        model_values = dict(cfg.model)
        vae, vae_metadata = load_frozen_wan_vae(
            model_id=str(model_values.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
            tokenizer_model_id=str(model_values.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")),
            device=str(cte_device),
            torch_dtype=torch.float32 if cte_device.type == "cpu" else torch.bfloat16,
            redirect_common_files=bool(model_values.get("redirect_common_files", True)),
        )
        validate_vae_metadata(checkpoint_vae_metadata, vae_metadata)
        frame_encoder = FastWAMCTELatentEncoder(
            vae, resize=cte_vae_input_size,
            expected_channels=model.cfg.image_channels,
            input_range="minus_one_one",
        ).encode_history
    model.to(cte_device)
    model.eval()
    records = []
    # RobotVideoDataset exposes overlapping windows (normally stride=1).  Cache
    # only ordered, non-overlapping 32-action source windows.  Each source
    # window is passed through the canonical full-history Zeva CTE once; no
    # recurrent hidden state is carried between windows.
    next_episode_step: dict[str, int] = {}
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
                    next_episode_step.pop(episode_id, None)

            source_index = int(sample.get("dataset_index", index))
            if source_index in seen_source_indices:
                raise ValueError(
                    f"dataset returned duplicate source index {source_index}; "
                    "cache construction cannot preserve deterministic joins"
                )
            frames = sample.get("cte_frames")
            preencoded = frames is not None
            if frames is None:
                frames = torch.cat((sample["before_frames"], sample["after_frames"][-1:]), dim=0)
            elif frames.ndim != 4 or frames.shape[0] != 9:
                raise ValueError("sample['cte_frames'] must be [9,C,H,W]")
            if frame_encoder is not None and not preencoded:
                frames = frame_encoder(frames.unsqueeze(0))[0].cpu()
            frames = frames.unsqueeze(0).to(cte_device)
            actions = sample["transition_actions"].unsqueeze(0).to(cte_device)
            output = model(
                frames,
                actions,
                valid_mask=sample["frame_valid"].unsqueeze(0).to(cte_device),
                transition_valid=sample["transition_valid"].unsqueeze(0).to(cte_device),
            )
            # Stage 2 queries phase_pre for the window's first action. If t=0
            # is invalid, later valid records cannot be safely joined to this
            # 32-action target, so discard the whole window.
            if not bool(output["transition_complete"][0, 0]):
                next_episode_step.pop(episode_id, None)
                continue
            seen_source_indices.add(source_index)
            # Zeva observes one effect after each four-transition (16-action)
            # window.  Store exactly the two effect rows produced by the
            # 32-action source window, with the phase captured at window start.
            effect_count = output["effect_post"].shape[1]
            for effect_index in range(effect_count):
                if not bool(output["effect_complete"][0, effect_index]):
                    continue
                transition_index = effect_index * model.cfg.effect_window_transitions
                transition_end = transition_index + model.cfg.effect_window_transitions
                records.append({
                    "episode_id": episode.episode_id,
                    "task_id": episode.task_id,
                    "attempt_id": 0,
                    "window_index": source_index,
                    "episode_step": episode_step + transition_index * model.cfg.transition_steps * sample_stride,
                    "transition_index": transition_index,
                    "effect_index": effect_index,
                    "phase_pre": output["phase"][0, transition_index],
                    "phase_post": output["phase"][0, transition_end],
                    "effect": output["effect_post"][0, effect_index],
                    "valid": True,
                })
            if bool(sample["frame_valid"].all()) and bool(sample["transition_valid"].all()):
                next_episode_step[episode_id] = episode_step + 32 * sample_stride
            else:
                # A partial/padded window cannot establish the next ordered
                # source position.
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
        image_channels=model.cfg.image_channels,
        feature_dtype=str(cache_cfg.get("feature_dtype", "float32")),
        action_video_freq_ratio=int(cfg.data.train.action_video_freq_ratio),
        cte_input_type=cte_input_type,
        latent_channels=model.cfg.image_channels if cte_input_type == "wan_vae_latent" else 0,
        vae_metadata=dict(vae_metadata),
        cte_vae_input_size=cte_vae_input_size if cte_input_type == "wan_vae_latent" else None,
    )
    if not records:
        raise RuntimeError("CTE cache construction produced no valid non-overlapping windows")
    save_phase_effect_cache(cache_path, records, manifest)
    print(f"saved {len(records)} phase/effect records to {cache_path}")


if __name__ == "__main__":
    main()
