"""Build Zeva phase/effect cache-v4 from the Stage-1 VAE latent cache.

This preserves the existing cache-v4 semantics:
- full_episode_prefix history;
- raw_action_step queries every 4 actions;
- 16-action effect windows;
- canonical semantic_task_id;
- exact CTE checkpoint / stats / VAE identity in the manifest.

Most RGB decoding and Wan-VAE work is eliminated. The Stage-1 latent cache
contains non-overlapping 32-action windows. If an episode's final cache-v4
coverage requires one additional non-32-aligned source window, only that tail
window is decoded and VAE-encoded. Tail windows are batched per rank.
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Running ``python scripts/...`` puts scripts/ rather than the repo root at
# sys.path[0]. Add the repository root so the existing cache builder can be
# imported as a module without copying its writer/merge logic.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_zeva_robotwin_cache import (  # noqa: E402
    AsyncCacheWriter,
    CacheManifest,
    DecodedEpisode,
    EpisodePlan,
    StreamingCacheWriter,
    _barrier,
    _build_episode_plans_metadata_only,
    _compute_episode,
    _destroy_process_group,
    _init_distributed,
    _merge_rank_caches,
    _partition_episode_plans,
    _resolve_device,
    _video_size_hw,
)
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset  # noqa: E402
from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset  # noqa: E402
from fastwam.zeva import (  # noqa: E402
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    FastWAMCTELatentEncoder,
    load_frozen_wan_vae,
    validate_vae_metadata,
)
from fastwam.zeva.checkpoint import checkpoint_sha256, load_cte_checkpoint  # noqa: E402
from fastwam.zeva.cte_latent_cache import CachedCTELatentWindowDataset  # noqa: E402
from fastwam.zeva.schemas import sha256_file  # noqa: E402


def _cfg_dict(value: Any) -> dict[str, Any]:
    return {} if value is None else dict(OmegaConf.to_container(value, resolve=True))


def _identity_collate(batch):
    return batch


def _canonicalize_plans(
    plans: list[EpisodePlan],
    latent_cache: CachedCTELatentWindowDataset,
) -> list[EpisodePlan]:
    out: list[EpisodePlan] = []
    missing: list[str] = []
    for plan in plans:
        task_id = latent_cache.task_by_episode.get(plan.episode_id)
        if task_id is None:
            missing.append(plan.episode_id)
            continue
        out.append(replace(plan, task_id=str(task_id)))
    if missing:
        raise ValueError(
            "phase-cache metadata includes episodes missing from the Stage-1 latent cache. "
            f"count={len(missing)}, examples={missing[:5]}. Ensure train split/seed are identical."
        )
    return out


def _cached_end_raw_step(
    latent_cache: CachedCTELatentWindowDataset,
    episode_id: str,
) -> int:
    rows = latent_cache.rows_by_episode.get(str(episode_id), [])
    if not rows:
        return -1
    # Every Stage-1 row spans exactly 32 raw actions and has 9 boundaries.
    return max(int(row["episode_step"]) + 32 for row in rows)


def _tail_required(plan: EpisodePlan, latent_cache: CachedCTELatentWindowDataset) -> bool:
    # Existing cache-v4 builder's coverage_starts() extends the reconstructed
    # prefix to last_query_start + 32. Match it exactly.
    required_end = int(plan.max_query_start) + int(plan.source_window_actions)
    return _cached_end_raw_step(latent_cache, plan.episode_id) < required_end


class _TailSourceDataset(Dataset):
    """Decode only the final source window missing from Stage-1 latent cache."""

    def __init__(self, dataset: ZevaRobotWinDataset, plans: list[EpisodePlan]) -> None:
        self.dataset = dataset
        self.plans = plans

    def __len__(self) -> int:
        return len(self.plans)

    def __getitem__(self, index: int) -> dict[str, Any]:
        plan = self.plans[int(index)]
        source_index = int(plan.dataset_start + plan.max_query_start)
        sample = self.dataset[source_index]
        actual_index = int(sample.get("dataset_index", source_index))
        if actual_index != source_index:
            raise RuntimeError(
                f"tail fallback requires deterministic index {source_index}, got {actual_index}"
            )
        episode = sample.get("episode")
        if episode is None:
            raise ValueError(f"tail sample {source_index} has no episode metadata")
        actual_episode = str(getattr(episode, "episode_id", ""))
        actual_step = int(getattr(episode, "episode_step", -1))
        if actual_episode != plan.episode_id or actual_step != int(plan.max_query_start):
            raise ValueError(
                "tail fallback episode mismatch: "
                f"plan=({plan.episode_id}, {plan.max_query_start}), "
                f"sample=({actual_episode}, {actual_step})"
            )
        frames = sample.get("cte_frames")
        if frames is None:
            frames = torch.cat((sample["before_frames"], sample["after_frames"][-1:]), dim=0)
        if frames.ndim != 4 or tuple(frames.shape[:2]) != (9, 3):
            raise ValueError(f"tail RGB must be [9,3,H,W], got {tuple(frames.shape)}")
        return {
            "episode_id": plan.episode_id,
            "episode_step": int(plan.max_query_start),
            "frames": frames,
            "actions": sample["transition_actions"].float(),
        }


def _precompute_tail_supplements(
    *,
    plans: list[EpisodePlan],
    dataset: ZevaRobotWinDataset,
    encoder: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    rank: int,
) -> dict[str, tuple[int, torch.Tensor, torch.Tensor]]:
    """Return episode_id -> (start_raw, BF16 latent[9,...], actions[8,4,14])."""
    if not plans:
        return {}
    source = _TailSourceDataset(dataset, plans)
    kwargs: dict[str, Any] = {
        "dataset": source,
        "batch_size": max(1, int(batch_size)),
        "shuffle": False,
        "num_workers": max(0, int(num_workers)),
        "collate_fn": _identity_collate,
        "pin_memory": True,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        kwargs["persistent_workers"] = True
    loader = DataLoader(**kwargs)

    out: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {}
    progress = tqdm(
        loader,
        total=(len(plans) + max(1, batch_size) - 1) // max(1, batch_size),
        desc=f"Tail VAE rank {rank}",
        unit="batch",
        dynamic_ncols=True,
        position=rank,
        leave=True,
    )
    with torch.inference_mode():
        for items in progress:
            frames = torch.stack([item["frames"] for item in items]).to(device, non_blocking=True)
            latent = encoder(frames)
            # The Stage-1 cache stores BF16 bits. Cast here so overlapping
            # boundaries compare against exactly the representation CTE saw.
            latent = latent.to(torch.bfloat16).cpu()
            for i, item in enumerate(items):
                episode_id = str(item["episode_id"])
                if episode_id in out:
                    raise ValueError(f"duplicate tail supplement for {episode_id}")
                out[episode_id] = (
                    int(item["episode_step"]),
                    latent[i].clone(),
                    item["actions"].float().cpu().clone(),
                )
    return out


def _insert_consistent(
    mapping: dict[int, torch.Tensor],
    raw_step: int,
    value: torch.Tensor,
    *,
    episode_id: str,
    kind: str,
) -> None:
    raw_step = int(raw_step)
    value = value.detach().cpu().clone()
    old = mapping.get(raw_step)
    if old is None:
        mapping[raw_step] = value
        return
    if old.dtype == torch.bfloat16 or value.dtype == torch.bfloat16:
        equal = torch.equal(old.to(torch.bfloat16), value.to(torch.bfloat16))
    else:
        equal = torch.allclose(old.float(), value.float(), atol=1e-6, rtol=0.0)
    if not equal:
        raise ValueError(
            f"inconsistent {kind}: episode={episode_id}, raw_step={raw_step}"
        )


def _materialize_episode(
    plan: EpisodePlan,
    *,
    latent_cache: CachedCTELatentWindowDataset,
    tail: tuple[int, torch.Tensor, torch.Tensor] | None,
) -> DecodedEpisode:
    boundary_frames: dict[int, torch.Tensor] = {}
    action_groups: dict[int, torch.Tensor] = {}
    rows = latent_cache.rows_by_episode.get(plan.episode_id, [])
    if not rows:
        raise ValueError(f"latent cache has no rows for {plan.episode_id}")

    for row in rows:
        sample = latent_cache.get_row(row)
        start = int(row["episode_step"])
        for local, frame in enumerate(sample["cte_frames"]):
            _insert_consistent(
                boundary_frames,
                start + 4 * local,
                frame,
                episode_id=plan.episode_id,
                kind="latent boundary",
            )
        for local, action in enumerate(sample["transition_actions"]):
            _insert_consistent(
                action_groups,
                start + 4 * local,
                action,
                episode_id=plan.episode_id,
                kind="action group",
            )

    if tail is not None:
        start, frames, actions = tail
        for local, frame in enumerate(frames):
            _insert_consistent(
                boundary_frames,
                start + 4 * local,
                frame,
                episode_id=plan.episode_id,
                kind="tail latent boundary",
            )
        for local, action in enumerate(actions):
            _insert_consistent(
                action_groups,
                start + 4 * local,
                action,
                episode_id=plan.episode_id,
                kind="tail action group",
            )

    required_end = int(plan.max_query_start) + int(plan.source_window_actions)
    required_steps = list(range(0, required_end + 1, int(plan.query_stride)))
    missing_frames = [step for step in required_steps if step not in boundary_frames]
    missing_actions = [step for step in required_steps[:-1] if step not in action_groups]
    if missing_frames or missing_actions:
        raise RuntimeError(
            f"latent reconstruction incomplete for {plan.episode_id}: "
            f"missing_frames={missing_frames[:8]}, missing_actions={missing_actions[:8]}"
        )

    # Trim any Stage-1 cached values beyond the exact cache-v4 prefix.
    boundary_frames = {step: boundary_frames[step] for step in required_steps}
    action_groups = {step: action_groups[step] for step in required_steps[:-1]}

    return DecodedEpisode(
        plan=plan,
        boundary_frames=boundary_frames,
        action_groups=action_groups,
        segments=[required_steps],
        preencoded=True,
        num_decoded_windows=len(rows) + (1 if tail is not None else 0),
        decode_seconds=0.0,
    )


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    info = _init_distributed()
    try:
        zeva = cfg.model.get("zeva", {})
        cte_cfg = _cfg_dict(zeva.get("cte"))
        cache_cfg = _cfg_dict(zeva.get("cache"))
        local_cfg = _cfg_dict(cfg.get("phase_cache_from_latents"))

        cte_value = cte_cfg.get("checkpoint")
        latent_value = local_cfg.get("latent_cache_path", cte_cfg.get("latent_cache_path"))
        output_value = cache_cfg.get("path")
        if cte_value in (None, "", "None", "null"):
            raise ValueError("set model.zeva.cte.checkpoint")
        if latent_value in (None, "", "None", "null"):
            raise ValueError("set model.zeva.cte.latent_cache_path")
        if output_value in (None, "", "None", "null"):
            raise ValueError("set model.zeva.cache.path")

        cte_path = Path(str(cte_value)).expanduser().resolve()
        cache_path = Path(str(output_value)).expanduser().resolve()
        if not cte_path.is_file():
            raise FileNotFoundError(cte_path)

        stats_path = Path(str(cfg.data.train.get("pretrained_norm_stats", ""))).expanduser()
        semantic_path = Path(str(cfg.data.train.get("semantic_task_map_path", ""))).expanduser()
        if not stats_path.is_file() or not semantic_path.is_file():
            raise FileNotFoundError("phase/effect cache requires stats and semantic task map")

        sample_stride = int(cfg.data.train.get("global_sample_stride", 1))
        if sample_stride != 1:
            raise ValueError("RoboTwin Zeva V1 requires global_sample_stride=1")

        payload = torch.load(cte_path, map_location="cpu", weights_only=False)
        raw_cfg = dict(payload.get("config", payload.get("model_config", {})))
        model_cfg_dict = dict(raw_cfg.get("cte", raw_cfg))
        cte_input_type = str(payload.get("cte_input_type", model_cfg_dict.get("input_type", "")))
        if cte_input_type != "wan_vae_latent":
            raise ValueError("latent-backed cache builder requires wan_vae_latent CTE")

        allowed = set(CausalTransitionEncoderConfig.__dataclass_fields__)
        model = CausalTransitionEncoder(
            CausalTransitionEncoderConfig(
                **{k: v for k, v in model_cfg_dict.items() if k in allowed}
            )
        )
        if (
            model.cfg.action_dim != 14
            or model.cfg.transition_steps != 4
            or model.cfg.effect_window_transitions != 4
        ):
            raise ValueError(
                "RoboTwin cache-v4 requires action_dim=14, transition_steps=4, "
                "effect_window_transitions=4"
            )
        load_cte_checkpoint(cte_path, model, map_location="cpu")

        latent_cache = CachedCTELatentWindowDataset(
            str(latent_value),
            expected_dataset_stats_sha256=sha256_file(stats_path),
            expected_semantic_task_sha256=sha256_file(semantic_path),
            expected_video_size=_video_size_hw(cfg),
            expected_action_dim=model.cfg.action_dim,
            expected_transition_steps=model.cfg.transition_steps,
            expected_latent_channels=model.cfg.image_channels,
        )
        checkpoint_vae_metadata = dict(payload.get("vae_metadata", {}))
        validate_vae_metadata(checkpoint_vae_metadata, latent_cache.vae_metadata)

        # Instantiate the normal dataset only for metadata planning and the
        # small set of tail windows that are not present in Stage-1 cache.
        with open_dict(cfg.data.train):
            cfg.data.train.video_backend = "pyav"
            cfg.data.train.use_text_embed_cache = False
        BaseLerobotDataset.presample_images = True

        base = None
        if info.distributed:
            if info.rank == 0:
                base = instantiate(cfg.data.train)
            _barrier(info)
            if info.rank != 0:
                base = instantiate(cfg.data.train)
            _barrier(info)
        else:
            base = instantiate(cfg.data.train)
        assert base is not None
        dataset = ZevaRobotWinDataset(base)

        plans = _build_episode_plans_metadata_only(
            dataset,
            sample_stride=sample_stride,
            transition_steps=model.cfg.transition_steps,
            source_window_actions=model.cfg.transition_steps * 8,
            show_progress=(info.rank == 0),
        )
        if not plans:
            raise RuntimeError("cache planner found no complete episodes")
        plans = _canonicalize_plans(plans, latent_cache)
        assignments, loads = _partition_episode_plans(plans, info.world_size)
        local_plans = assignments[info.rank]

        if info.rank == 0:
            print("========== phase/effect cache from latent cache ==========")
            print(f"CTE checkpoint: {cte_path}")
            print(f"CTE sha256: {checkpoint_sha256(cte_path)}")
            print(f"latent cache: {latent_cache.root}")
            print(f"output: {cache_path}")
            print(f"episodes: {len(plans):,}")
            print(f"phase queries: {sum(plan.query_count for plan in plans):,}")
            print(f"world size: {info.world_size}")
            for rank_id, load in enumerate(loads):
                print(f"rank {rank_id}: phase_queries={load:,}")

        tail_plans = [plan for plan in local_plans if _tail_required(plan, latent_cache)]
        local_tail_count = len(tail_plans)
        tail_counts = [local_tail_count]
        if info.distributed:
            gathered = [None for _ in range(info.world_size)] if info.rank == 0 else None
            dist.gather_object(local_tail_count, gathered, dst=0)
            if info.rank == 0:
                assert gathered is not None
                tail_counts = [int(v) for v in gathered]
        if info.rank == 0:
            print(f"tail fallback windows total: {sum(tail_counts):,}")
            print(f"tail fallback by rank: {tail_counts}")

        cte_device = _resolve_device(cfg, info)
        model.to(cte_device).eval().requires_grad_(False)

        tail_supplements: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {}
        if tail_plans:
            model_values = _cfg_dict(cfg.model)
            vae, vae_metadata = load_frozen_wan_vae(
                model_id=str(model_values.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
                tokenizer_model_id=str(
                    model_values.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")
                ),
                device=str(cte_device),
                torch_dtype=torch.bfloat16,
                redirect_common_files=bool(model_values.get("redirect_common_files", True)),
            )
            validate_vae_metadata(checkpoint_vae_metadata, vae_metadata)
            encoder = FastWAMCTELatentEncoder(
                vae,
                resize=_video_size_hw(cfg),
                expected_channels=model.cfg.image_channels,
                input_range="minus_one_one",
            ).encode_history
            tail_supplements = _precompute_tail_supplements(
                plans=tail_plans,
                dataset=dataset,
                encoder=encoder,
                device=cte_device,
                batch_size=int(local_cfg.get("tail_batch_size", 4)),
                num_workers=int(local_cfg.get("tail_num_workers", 2)),
                prefetch_factor=int(local_cfg.get("tail_prefetch_factor", 2)),
                rank=info.rank,
            )
            del vae, encoder
            if cte_device.type == "cuda":
                torch.cuda.empty_cache()

        manifest = CacheManifest(
            schema_version="zeva_fastwam_robotwin_cache_v4",
            history_semantics="full_episode_prefix",
            query_step_unit="raw_action_step",
            cte_checkpoint_sha256=checkpoint_sha256(cte_path),
            dataset_stats_sha256=sha256_file(stats_path),
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
            cte_input_type="wan_vae_latent",
            latent_channels=model.cfg.image_channels,
            # Stage-2 compares this field exactly to the CTE checkpoint.
            vae_metadata=checkpoint_vae_metadata,
            cte_vae_input_size=_video_size_hw(cfg),
        )

        final_root = cache_path
        work_root = final_root.parent / f".{final_root.name}.building"
        rank_root = work_root / f"rank_{info.rank:03d}"
        if info.rank == 0:
            if work_root.exists():
                import shutil
                shutil.rmtree(work_root)
            work_root.mkdir(parents=True, exist_ok=True)
        _barrier(info)
        rank_root.mkdir(parents=True, exist_ok=True)

        writer = StreamingCacheWriter(
            rank_root,
            manifest,
            shard_size=int(cache_cfg.get("shard_size", 4096)),
        )
        async_writer = AsyncCacheWriter(
            writer,
            enabled=bool(cache_cfg.get("async_write", True)),
            queue_size=int(cache_cfg.get("writer_queue_size", 4)),
        )

        progress = tqdm(
            total=sum(plan.query_count for plan in local_plans),
            desc=f"Zeva latent->cache rank {info.rank}/{info.world_size}",
            unit="query",
            dynamic_ncols=True,
            position=info.rank,
            leave=True,
        )
        started = time.perf_counter()
        record_count = 0
        try:
            with torch.inference_mode():
                for plan in local_plans:
                    decoded = _materialize_episode(
                        plan,
                        latent_cache=latent_cache,
                        tail=tail_supplements.get(plan.episode_id),
                    )
                    records, _stats = _compute_episode(
                        decoded,
                        model=model,
                        cte_device=cte_device,
                        cte_input_type="wan_vae_latent",
                        frame_encoder=None,
                        profile_gpu_timing=False,
                        pin_memory=False,
                    )
                    async_writer.submit(records)
                    record_count += len(records)
                    progress.update(plan.query_count)
                    progress.set_postfix(records=record_count, refresh=False)
        finally:
            progress.close()
            async_writer.close()

        print(
            f"[rank {info.rank}] complete: episodes={len(local_plans):,}, "
            f"records={record_count:,}, tail_windows={len(tail_supplements):,}, "
            f"wall={time.perf_counter() - started:.1f}s"
        )
        _barrier(info)

        if info.rank == 0:
            total_records = _merge_rank_caches(
                work_root=work_root,
                final_root=final_root,
                manifest=manifest,
                world_size=info.world_size,
                overwrite=bool(cache_cfg.get("overwrite", False)),
            )
            print("========== phase/effect cache ==========")
            print(f"saved records: {total_records:,}")
            print(f"output: {final_root}")
            print("history_semantics: full_episode_prefix")
            print("query_step_unit: raw_action_step")
            print("semantic identity source: Stage-1 canonical latent cache")
            print("PHASE/EFFECT CACHE V4: PASSED")
            if work_root.exists():
                import shutil
                shutil.rmtree(work_root)
        _barrier(info)

    finally:
        _destroy_process_group(info)


if __name__ == "__main__":
    main()
