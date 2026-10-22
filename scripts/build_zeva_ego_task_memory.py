"""Build train-only task memory, then static contexts for train + validation.

Use task_context.build_stage=bank first. In task_language mode this exports
the frozen input-embedding encoder and raw-instruction features; train its
classifier with train_zeva_task_language.py, then build_stage=static_context.
Legacy static mode retains initial-image/text retrieval. Both modes retain
the exact CTE episode split and boundary-cache identity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset

from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.ego_boundary_cache import EgoBoundaryCache
from fastwam.zeva.ego_cte_training import require_ego_checkpoint_ready, sha256_file
from fastwam.zeva.ego_episode_data import EGO_CAMERA_KEYS, RoboTwinEgoEpisodeDataset
from fastwam.zeva.ego_task_memory import (
    build_ego_static_context_cache, ordered_episode_ids, save_ego_task_memory,
    build_ego_task_language_context_cache, episode_task_instructions, save_ego_task_language_artifacts,
)


def _path(value, name: str, *, existing: bool = False) -> Path:
    if value in (None, "", "None", "null"):
        raise ValueError(f"Configure {name}")
    path = Path(str(value)).expanduser().resolve()
    if existing and not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return path


class _InitialEpisodeReadouts(Dataset):
    def __init__(self, base, cache) -> None:
        from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset

        self.base, self.cache = ZevaRobotWinDataset(base), cache
        self.episode_ids = ordered_episode_ids(cache)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        identity = self.episode_ids[index]
        metadata = self.cache.episodes[identity]
        source_index = int(metadata["dataset_start"])
        sample = self.base[source_index]
        episode = sample["episode"]
        if (int(sample.get("dataset_index", source_index)) != source_index or str(episode.episode_id) != identity
                or int(episode.episode_step) != 0 or str(episode.task_id) != str(metadata["task_id"])):
            raise ValueError("Initial-readout source indexing/episode/task does not match the CTE cache")
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("Initial task queries require frozen initial image and cached instruction embeddings")
        return {"episode_id": identity, "instruction": str(episode.instruction),
                "image": sample["video"][:, 0].float(), "context": sample["context"].float(),
                "context_mask": sample["context_mask"].bool()}


def _identity_collate(samples):
    return samples


@torch.no_grad()
def extract_initial_readouts(policy, base, cache, *, device, batch_size=2, workers=0):
    if batch_size < 1 or workers < 0:
        raise ValueError("Readout batch size must be positive and workers nonnegative")
    source = _InitialEpisodeReadouts(base, cache)
    loader = DataLoader(source, batch_size=batch_size, num_workers=workers, shuffle=False,
                        collate_fn=_identity_collate, pin_memory=torch.device(device).type == "cuda")
    readouts, instructions = {}, {}
    policy.eval().requires_grad_(False)
    for items in loader:
        images = torch.stack([item["image"] for item in items]).to(device)
        context = torch.stack([item["context"] for item in items]).to(device)
        mask = torch.stack([item["context_mask"] for item in items]).to(device)
        values = policy.extract_task_context_readout(images, context, mask).float().cpu()
        if values.ndim != 2 or values.shape[0] != len(items) or not bool(torch.isfinite(values).all()):
            raise ValueError("Frozen FastWAM initial readout must be finite [batch,hidden]")
        for item, value in zip(items, values, strict=True):
            readouts[item["episode_id"]] = value
            instructions[item["episode_id"]] = item["instruction"]
        print(json.dumps({"initial_readouts": len(readouts), "total_episodes": len(source)}), flush=True)
    return readouts, instructions


def build(cfg: DictConfig) -> dict:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This split-aware task-memory builder uses one process; do not launch multi-rank torchrun")
    zeva, data = cfg.model.zeva, cfg.data.train
    tc = zeva.task_context
    task_language = str(tc.get("mode", "static")) == "task_language"
    if str(tc.get("mode", "static")) not in {"static", "task_language"}:
        raise ValueError("Predictive task-memory builder supports static or task_language mode")
    stage = str(tc.get("build_stage", "bank"))
    if stage not in {"bank", "static_context"}:
        raise ValueError("task_context.build_stage must be bank or static_context")
    if str(zeva.cte.get("variant")) != "zeva_ego" or str(zeva.cte.get("input_type")) != "rgb_multiview":
        raise ValueError("New task memory requires the predictive multiview Zeva-Ego CTE")
    cache_path = _path(zeva.cache.path, "model.zeva.cache.path", existing=True)
    cte_path = _path(zeva.cte.checkpoint, "model.zeva.cte.checkpoint", existing=True)
    base_path = None if task_language else _path(cfg.get("ckpt"), "ckpt (frozen FastWAM base)", existing=True)
    stats_path = _path(data.pretrained_norm_stats, "data.train.pretrained_norm_stats", existing=True)
    semantic_path = _path(data.semantic_task_map_path, "data.train.semantic_task_map_path", existing=True)
    bank_path = _path(tc.bank_path, "task_context.bank_path", existing=stage == "static_context")
    if task_language:
        encoder_path = _path(tc.get("language_encoder_path"), "task_context.language_encoder_path",
                             existing=stage == "static_context")
        feature_path = _path(tc.get("language_feature_cache_path"), "task_context.language_feature_cache_path",
                             existing=stage == "static_context")
        if float(tc.temperature) != 1.0:
            raise ValueError("Zeva-Ego task-language retrieval requires temperature=1.0")
    else:
        train_readout_path = _path(tc.readout_cache_path, "task_context.readout_cache_path", existing=stage == "static_context")
        query_path = _path(tc.query_readout_cache_path, "task_context.query_readout_cache_path", existing=stage == "static_context")
    cache = EgoBoundaryCache(cache_path)
    checkpoint = torch.load(cte_path, map_location="cpu", weights_only=False)
    require_ego_checkpoint_ready(checkpoint)
    if checkpoint["episode_splits"] != cache.splits:
        raise ValueError("CTE and boundary cache use different held-out episode splits")
    stats_sha, semantic_sha, cte_sha = sha256_file(stats_path), sha256_file(semantic_path), checkpoint_sha256(cte_path)
    for name, expected in (("dataset_stats_sha256", stats_sha), ("semantic_task_map_sha256", semantic_sha),
                           ("cte_checkpoint_sha256", cte_sha)):
        if getattr(cache.manifest, name) != expected:
            raise ValueError(f"Boundary-cache artifact identity mismatch: {name}")
    if checkpoint["dataset_stats_sha256"] != stats_sha or checkpoint["semantic_task_map_sha256"] != semantic_sha:
        raise ValueError("CTE stats/semantic identity differs from configured readout data")
    if (cache.manifest.retrieval_dim, cache.manifest.global_dim) != (int(zeva.cte.retrieval_dim), int(zeva.cte.hidden_dim)):
        raise ValueError("Task-memory key/value dimensions differ from the configured CTE")
    video_size = [int(value) for value in data.video_size]
    if video_size != [384, 320] or int(data.context_len) != int(cfg.model.tokenizer_max_len):
        raise ValueError("Formal RoboTwin readouts require video_size=[384,320] and the policy's context length")
    dataset_paths = [str(Path(path).expanduser().resolve()) for path in data.dataset_dirs]
    cached_paths = list(getattr(cache.manifest, "dataset_paths", []) or [cache.manifest.dataset_path])
    if dataset_paths != [str(Path(path).expanduser().resolve()) for path in cached_paths]:
        raise ValueError("Readout dataset paths differ from the CTE boundary cache")
    metadata = {
        "cte_checkpoint_sha256": cte_sha,
        "dataset_stats_sha256": stats_sha, "semantic_task_map_sha256": semantic_sha,
        "phase_cache_manifest_sha256": sha256_file(cache_path / "manifest.json"),
        "episode_splits_sha256": sha256_file(cache_path / "episode_splits.json"),
        "dataset_dirs": dataset_paths, "camera_keys": list(EGO_CAMERA_KEYS),
        "executed_horizon": 24, "action_dim": 14, "history_semantics": "full_episode_prefix",
        "memory_semantics": "predicted_effect_trace", "cte_variant": "zeva_ego", "effect_source": "current_prediction",
        "pooling_rule": "complete_transition_starts_mean_then_l2_keys_and_values",
        "retrieval_trace_space": "cte_task_head_raw_v1",
    }
    if not task_language:
        metadata.update({"base_checkpoint_sha256": checkpoint_sha256(base_path),
                         "video_size": video_size, "context_len": int(cfg.model.tokenizer_max_len)})
    del checkpoint
    configured_device = cfg.get("device")
    device = torch.device(str(configured_device) if configured_device not in (None, "", "None", "null")
                          else "cuda" if torch.cuda.is_available() else "cpu")
    if task_language:
        from fastwam.zeva.task_language import (
            FrozenTaskLanguageEncoder, export_task_language_encoder_from_fastwam_config,
        )

        if stage == "static_context":
            head_path = _path(tc.retrieval_checkpoint, "task_context.retrieval_checkpoint", existing=True)
            output = _path(tc.static_episode_cache_path, "task_context.static_episode_cache_path")
            result = build_ego_task_language_context_cache(
                cache=cache, bank_path=bank_path, feature_path=feature_path, encoder_path=encoder_path,
                head_path=head_path, output_path=output, expected_metadata=metadata,
                top_k=int(tc.top_k), device=device, batch_size=int(tc.get("static_context_batch_size", 512)),
            )
        else:
            if bank_path.exists() or feature_path.exists():
                raise ValueError("Task-language bank/features require unused output paths")
            if not bool(data.get("is_training_set", False)) or float(data.get("val_set_proportion", -1)) != 0:
                raise ValueError("Read all CTE source episodes with is_training_set=true and val_set_proportion=0")
            if int(data.global_sample_stride) != 1:
                raise ValueError("Task-language metadata requires source stride1")
            # Only Arrow metadata is read; cached policy embeddings, RGB and
            # FastWAM model construction are unnecessary for language queries.
            source_cfg = OmegaConf.create(OmegaConf.to_container(data, resolve=True))
            source_cfg.use_text_embed_cache = False
            base = instantiate(source_cfg)
            instructions = episode_task_instructions(base, cache)
            if not encoder_path.exists():
                export_task_language_encoder_from_fastwam_config(
                    cfg.model, encoder_path, max_length=int(tc.get("language_max_length", 200)),
                )
            encoder = FrozenTaskLanguageEncoder.load(encoder_path)
            ids = ordered_episode_ids(cache)
            feature_batch = int(tc.get("language_feature_batch_size", 256))
            if feature_batch < 1:
                raise ValueError("language_feature_batch_size must be positive")
            features = {}
            for start in range(0, len(ids), feature_batch):
                selected = ids[start:start + feature_batch]
                encoded = encoder.encode([instructions[identity] for identity in selected], device="cpu")
                features.update(zip(selected, encoded, strict=True))
            result = save_ego_task_language_artifacts(
                cache, features, instructions=instructions, bank_path=bank_path, feature_path=feature_path,
                encoder_path=encoder_path, metadata={**metadata, "builder": Path(__file__).name},
            )
    elif stage == "static_context":
        head_path = _path(tc.retrieval_checkpoint, "task_context.retrieval_checkpoint", existing=True)
        output = _path(tc.static_episode_cache_path, "task_context.static_episode_cache_path")
        result = build_ego_static_context_cache(cache=cache, bank_path=bank_path, train_readout_path=train_readout_path,
                                                query_readout_path=query_path, head_path=head_path, output_path=output,
                                                expected_metadata=metadata, top_k=int(tc.top_k), device=device,
                                                batch_size=int(tc.get("static_context_batch_size", 512)))
    else:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Building frozen FastWAM initial readouts requires CUDA")
        if not bool(data.get("is_training_set", False)) or float(data.get("val_set_proportion", -1)) != 0:
            raise ValueError("Read all source episodes with data.train.is_training_set=true and val_set_proportion=0")
        if (not bool(data.get("use_text_embed_cache", False)) or int(data.global_sample_stride) != 1
                or int(data.action_video_freq_ratio) != 4):
            raise ValueError("Initial readouts require cached text, source stride1 and video/action ratio4")
        base = instantiate(data)
        source = RoboTwinEgoEpisodeDataset(base)
        if {episode.episode_id for episode in source.episodes} != set(cache.episodes):
            raise ValueError("Readout dataset and boundary cache have different source episodes")
        for episode in source.episodes:
            recorded = cache.episodes[episode.episode_id]
            for name in ("task_id", "dataset_start", "episode_length"):
                if getattr(episode, name) != recorded[name]:
                    raise ValueError(f"Initial-readout episode metadata mismatch: {episode.episode_id} {name}")
        from fastwam.runtime import _mixed_precision_to_model_dtype

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.zeva.enabled = False
        policy = instantiate(model_cfg, model_dtype=_mixed_precision_to_model_dtype(str(cfg.get("mixed_precision", "bf16"))),
                              device=str(device))
        policy.load_checkpoint(str(base_path))
        readouts, instructions = extract_initial_readouts(policy, base, cache, device=device,
                                                           batch_size=int(tc.get("readout_batch_size", 2)),
                                                           workers=int(cfg.get("num_workers", 4)))
        if any(value.shape != (int(cfg.model.video_dit_config.hidden_dim),) for value in readouts.values()):
            raise ValueError("FastWAM readout dimension differs from the configured frozen backbone")
        result = save_ego_task_memory(cache, readouts, bank_path=bank_path, train_readout_path=train_readout_path,
                                       query_readout_path=query_path, metadata={**metadata, "builder": Path(__file__).name},
                                       instructions=instructions, temperature=float(tc.temperature))
    print(json.dumps({"build_stage": stage, **result}, sort_keys=True), flush=True)
    return result


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    build(cfg)


if __name__ == "__main__":
    main()
