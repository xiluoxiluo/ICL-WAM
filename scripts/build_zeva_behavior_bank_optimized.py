"""Build Zeva static behavior bank + FastWAM initial readouts efficiently.

Formal RoboTwin path:
* metadata-only episode planning (no million-window RGB pass);
* Stage-1 BF16 latent-cache reuse for full-prefix CTE trajectories;
* only missing final tail windows use the frozen Wan VAE;
* batched FastWAM clean initial readout extraction;
* torchrun episode sharding across GPUs;
* strict CTE/base/stats/semantic/VAE identity checks;
* deterministic rank-local artifacts merged by rank 0.

The algorithmic definition is unchanged from the original builder:
  key   = normalized mean of CTE retrieval states over one demonstration
  value = mean of CTE causal_interaction_state over one demonstration
  readout = frozen FastWAM clean initial image/text final-video hidden mean
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_zeva_robotwin_cache import (  # noqa: E402
    EpisodePlan,
    _barrier,
    _build_episode_plans_metadata_only,
    _destroy_process_group,
    _init_distributed,
    _partition_episode_plans,
    _resolve_device,
)
from scripts.build_zeva_robotwin_cache_from_latents import (  # noqa: E402
    _canonicalize_plans,
    _materialize_episode,
    _precompute_tail_supplements,
    _tail_required,
    _validate_plan_alignment,
)
from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset  # noqa: E402
from fastwam.runtime import _mixed_precision_to_model_dtype  # noqa: E402
from fastwam.zeva import CausalTransitionEncoder, CausalTransitionEncoderConfig  # noqa: E402
from fastwam.zeva.checkpoint import checkpoint_sha256, load_cte_checkpoint  # noqa: E402
from fastwam.zeva.main_data import is_main_profile, normalize_main_dataset_paths, select_main_success_rows, validate_main_artifact  # noqa: E402
from fastwam.zeva.cte_windows import MAIN_CTE_WINDOW_SEMANTICS  # noqa: E402
from fastwam.zeva.task_context_data import split_main_behavior_artifacts  # noqa: E402
from fastwam.zeva.cte_latent_cache import CachedCTELatentWindowDataset  # noqa: E402
from fastwam.zeva.schemas import sha256_file  # noqa: E402
from fastwam.zeva.static_task_context import (  # noqa: E402
    BEHAVIOR_KEY_SPACE,
    BEHAVIOR_VALUE_SPACE,
    READOUT_FORMAT,
    READOUT_KIND,
    load_readout_cache,
    trajectory_prototype,
    validate_behavior_bank,
)
from fastwam.zeva.task_context import TaskContextBank  # noqa: E402
from fastwam.zeva.vae_adapter import FastWAMCTELatentEncoder, validate_vae_metadata  # noqa: E402


def _cfg_dict(value: Any) -> dict[str, Any]:
    return {} if value is None else dict(OmegaConf.to_container(value, resolve=True))


def _identity_collate(batch):
    return batch


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default if value in (None, "") else value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be boolean, got {value!r}")


@contextmanager
def _only_initial_video_timestamps(base_dataset):
    """Temporarily request timestamp zero for each image stream, not 33 frames.

    Readout generation is an independent DataLoader. Restoring these settings
    after each sample prevents state leaking into any ordinary RGB/action
    dataset access. With multiple workers, each process owns its own copy.
    """
    lerobot = base_dataset.lerobot_dataset
    processor = lerobot.processor
    if processor is None or not hasattr(processor, "num_image_steps"):
        raise ValueError("first-frame readout requires the normal FastWAM image processor")
    modifications = []
    try:
        for source in lerobot.multi_dataset._datasets:
            if source.delta_indices is None:
                raise ValueError("first-frame readout requires LeRobot delta indices")
            for camera in source.meta.video_keys:
                if camera not in source.delta_indices:
                    raise ValueError(f"missing video delta indices for {camera}")
                offsets = source.delta_indices[camera]
                if len(offsets) < 1 or int(offsets[0]) != 0:
                    raise ValueError(f"first video offset must be 0 for {camera}")
                modifications.append((source, camera, offsets))
                source.delta_indices[camera] = offsets[:1]
        old_num_steps = int(processor.num_image_steps)
        processor.num_image_steps = 1
        try:
            yield
        finally:
            processor.num_image_steps = old_num_steps
    finally:
        for source, camera, offsets in modifications:
            source.delta_indices[camera] = offsets


def _readout_tensor_collate(items):
    if not items:
        raise ValueError("initial readout batch must not be empty")
    return {
        "images": torch.stack([item["image"] for item in items]),
        "contexts": torch.stack([item["context"] for item in items]),
        "masks": torch.stack([item["context_mask"] for item in items]),
        "episode_ids": [item["episode_id"] for item in items],
        "task_ids": [item["task_id"] for item in items],
        "instructions": [item["instruction"] for item in items],
    }


def _readout_wait_times(loader):
    iterator = iter(loader)
    while True:
        before = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        yield batch, time.perf_counter() - before


class _InitialReadoutDataset(Dataset):
    def __init__(
        self, dataset: ZevaRobotWinDataset, plans: list[EpisodePlan], *,
        first_frame_only: bool = False,
    ) -> None:
        self.dataset = dataset
        self.plans = plans
        self.first_frame_only = bool(first_frame_only)

    def __len__(self) -> int:
        return len(self.plans)

    def __getitem__(self, index: int) -> dict[str, Any]:
        plan = self.plans[int(index)]
        source_index = int(plan.dataset_start)
        if self.first_frame_only:
            # Preserve the exact FastWAM RobotVideoDataset image transforms and
            # text cache path; only video query timestamps are shortened.
            base_dataset = self.dataset.base_dataset
            with _only_initial_video_timestamps(base_dataset):
                sample = base_dataset._get(source_index, initial_image_only=True)
            episode_id = str(sample.get("episode_id", ""))
            episode_step = int(sample.get("episode_step", -1))
            task_id = str(sample.get("task_id", ""))
            instruction = str(sample.get("prompt", ""))
            expected_time = 1
        else:
            sample = self.dataset[source_index]
            episode = sample["episode"]
            episode_id = str(episode.episode_id)
            episode_step = int(episode.episode_step)
            task_id = str(episode.task_id)
            instruction = str(episode.instruction)
            expected_time = 9
        if int(sample.get("dataset_index", source_index)) != source_index:
            raise RuntimeError("initial-readout dataset requires deterministic source indexing")
        if episode_id != str(plan.episode_id) or episode_step != 0:
            raise ValueError(
                f"initial-readout episode mismatch: plan={plan.episode_id}, "
                f"sample=({episode_id}, step={episode_step})"
            )
        if task_id != str(plan.task_id):
            raise ValueError(
                f"initial-readout semantic task mismatch: plan={plan.task_id}, sample={task_id}"
            )
        if "context" not in sample or "context_mask" not in sample:
            raise KeyError(
                "behavior-bank readout requires cached text context; "
                "use the Stage-2 task config with data.train.use_text_embed_cache=true"
            )
        video = sample["video"]
        if video.ndim != 4 or tuple(video.shape[:2]) != (3, expected_time):
            raise ValueError(
                f"initial readout expects video [3,{expected_time},H,W], got {tuple(video.shape)}"
            )
        return {
            "episode_id": episode_id,
            "task_id": task_id,
            "dataset_start": source_index,
            "instruction": instruction,
            "image": video[:, 0].float(),
            "context": sample["context"].float(),
            "context_mask": sample["context_mask"].bool(),
        }


def _cte_model_from_checkpoint(path: Path) -> tuple[CausalTransitionEncoder, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    values = dict(payload.get("config", payload.get("model_config", {})))
    values = dict(values.get("cte", values))
    allowed = CausalTransitionEncoderConfig.__dataclass_fields__
    model = CausalTransitionEncoder(
        CausalTransitionEncoderConfig(**{k: v for k, v in values.items() if k in allowed})
    )
    load_cte_checkpoint(path, model)
    return model, payload


def _semantic_sha(payload: dict[str, Any]) -> str:
    identity = payload.get("semantic_task_identity") or {}
    value = identity.get("sha256")
    if not value:
        raise ValueError("CTE checkpoint is missing semantic_task_identity.sha256")
    return str(value)


def _compute_cte_prototype(
    plan: EpisodePlan,
    *,
    latent_cache: CachedCTELatentWindowDataset,
    tail: tuple[int, torch.Tensor, torch.Tensor] | None,
    cte: CausalTransitionEncoder,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    decoded = _materialize_episode(plan, latent_cache=latent_cache, tail=tail)
    if len(decoded.segments) != 1:
        raise RuntimeError(f"full-prefix trajectory unexpectedly has {len(decoded.segments)} segments")
    steps = decoded.segments[0]
    frames = torch.stack([decoded.boundary_frames[s] for s in steps]).unsqueeze(0)
    actions = torch.stack([decoded.action_groups[s] for s in steps[:-1]]).unsqueeze(0)
    expected_actions = (1, len(steps) - 1, cte.cfg.transition_steps, cte.cfg.action_dim)
    if tuple(actions.shape) != expected_actions:
        raise ValueError(f"CTE trajectory actions mismatch: {tuple(actions.shape)} != {expected_actions}")
    frames = frames.to(device=device, dtype=torch.float32, non_blocking=True)
    actions = actions.to(device=device, dtype=torch.float32, non_blocking=True)
    valid = torch.ones((1, len(steps)), device=device, dtype=torch.bool)
    transition_valid = torch.ones(actions.shape[:-1], device=device, dtype=torch.bool)
    with torch.inference_mode():
        output = cte(frames, actions, valid_mask=valid, transition_valid=transition_valid)
        key, value = trajectory_prototype(output, valid)
    if key.shape != (1, cte.cfg.retrieval_dim) or value.shape != (1, cte.cfg.hidden_dim):
        raise RuntimeError(
            f"trajectory prototype shape mismatch: key={tuple(key.shape)}, value={tuple(value.shape)}"
        )
    if not bool(torch.isfinite(key).all() and torch.isfinite(value).all()):
        raise FloatingPointError("trajectory prototype contains NaN/Inf")
    return key[0].float().cpu(), value[0].float().cpu(), len(steps) - 1


def _build_readouts(
    *,
    dataset: ZevaRobotWinDataset,
    plans: list[EpisodePlan],
    policy: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    rank: int,
    first_frame_only: bool = False,
    fast_collate: bool = False,
    profile_every: int = 0,
) -> tuple[list[str], list[str], list[str], torch.Tensor]:
    source = _InitialReadoutDataset(dataset, plans, first_frame_only=first_frame_only)
    if profile_every < 0:
        raise ValueError("ZEVA_BEHAVIOR_READOUT_PROFILE_EVERY must be >= 0")
    kwargs: dict[str, Any] = {
        "dataset": source,
        "batch_size": max(1, int(batch_size)),
        "shuffle": False,
        "num_workers": max(0, int(num_workers)),
        "collate_fn": _readout_tensor_collate if fast_collate else _identity_collate,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(**kwargs)
    episode_ids: list[str] = []
    task_ids: list[str] = []
    instructions: list[str] = []
    readouts: list[torch.Tensor] = []
    timed = _readout_wait_times(loader) if profile_every else ((batch, 0.0) for batch in loader)
    progress = tqdm(
        timed, total=len(loader), desc=f"FastWAM readout rank {rank}",
        unit="batch", position=rank, dynamic_ncols=True,
    )
    timing = {"wait": 0.0, "forward": 0.0, "episodes": 0}
    batches = 0
    with torch.inference_mode():
        for items, wait_seconds in progress:
            started = time.perf_counter() if profile_every else 0.0
            if fast_collate:
                images = items["images"].to(device, non_blocking=True)
                contexts = items["contexts"].to(device, non_blocking=True)
                masks = items["masks"].to(device, non_blocking=True)
                this_episode_ids = items["episode_ids"]
                this_task_ids = items["task_ids"]
                this_instructions = items["instructions"]
            else:
                images = torch.stack([item["image"] for item in items]).to(device, non_blocking=True)
                contexts = torch.stack([item["context"] for item in items]).to(device, non_blocking=True)
                masks = torch.stack([item["context_mask"] for item in items]).to(device, non_blocking=True)
                this_episode_ids = [str(item["episode_id"]) for item in items]
                this_task_ids = [str(item["task_id"]) for item in items]
                this_instructions = [str(item["instruction"]) for item in items]
            readout = policy.extract_task_context_readout(images, contexts, masks)
            if readout.ndim != 2 or readout.shape[0] != len(this_episode_ids):
                raise RuntimeError(f"FastWAM readout shape mismatch: {tuple(readout.shape)}")
            if not bool(torch.isfinite(readout).all()):
                raise FloatingPointError("FastWAM clean readout contains NaN/Inf")
            readouts.extend(readout.float().cpu().unbind(0))
            episode_ids.extend(this_episode_ids)
            task_ids.extend(this_task_ids)
            instructions.extend(this_instructions)
            if profile_every:
                batches += 1
                timing["wait"] += wait_seconds
                timing["forward"] += time.perf_counter() - started
                timing["episodes"] += len(this_episode_ids)
                if batches % profile_every == 0:
                    avg_wait = timing["wait"] / profile_every
                    avg_forward = timing["forward"] / profile_every
                    total = timing["wait"] + timing["forward"]
                    print(
                        f"[behavior readout rank {rank}] last {profile_every} batches "
                        f"wait_data={avg_wait:.3f}s/batch "
                        f"h2d_forward_d2h={avg_forward:.3f}s/batch "
                        f"throughput={timing['episodes']/max(total, 1.e-9):.2f} episodes/s",
                        flush=True,
                    )
                    timing = {"wait": 0.0, "forward": 0.0, "episodes": 0}
    return episode_ids, task_ids, instructions, torch.stack(readouts)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    normalize_main_dataset_paths(cfg)
    info = _init_distributed()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("optimized behavior-bank construction requires CUDA")
        if not bool(cfg.data.train.get("is_training_set", False)):
            raise ValueError("behavior bank must be built from data.train")
        if not bool(cfg.data.train.get("use_text_embed_cache", False)):
            raise ValueError(
                "behavior bank requires data.train.use_text_embed_cache=true; "
                "do not reuse the Stage-1 CTE task config without this override"
            )
        if int(cfg.data.train.get("global_sample_stride", -1)) != 1:
            raise ValueError("behavior bank requires global_sample_stride=1")
        if int(cfg.data.train.get("action_video_freq_ratio", -1)) != 4:
            raise ValueError("behavior bank requires action_video_freq_ratio=4")
        if int(cfg.data.train.get("context_len", -1)) != int(cfg.model.tokenizer_max_len):
            raise ValueError("data.train.context_len must equal model.tokenizer_max_len")

        zeva = cfg.model.zeva
        tc = zeva.task_context
        cte_path = Path(str(zeva.cte.checkpoint)).expanduser().resolve()
        latent_path = Path(str(zeva.cte.get("latent_cache_path"))).expanduser().resolve()
        base_ckpt = Path(str(cfg.get("ckpt"))).expanduser().resolve()
        bank_path = Path(str(tc.bank_path)).expanduser().resolve()
        readout_path = Path(str(tc.readout_cache_path)).expanduser().resolve()
        for name, path in {
            "CTE checkpoint": cte_path,
            "latent cache": latent_path,
            "FastWAM base checkpoint": base_ckpt,
            "dataset stats": Path(str(cfg.data.train.pretrained_norm_stats)),
            "semantic task map": Path(str(cfg.data.train.semantic_task_map_path)),
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"missing {name}: {path}")
        if bank_path == readout_path:
            raise ValueError("behavior bank and readout cache must use different paths")
        overwrite = _env_bool("ZEVA_BEHAVIOR_OVERWRITE", False)
        if info.rank == 0 and not overwrite:
            existing = [str(path) for path in (bank_path, readout_path) if path.exists()]
            if existing:
                raise FileExistsError(f"behavior artifacts already exist: {existing}")
        _barrier(info)

        cte, payload = _cte_model_from_checkpoint(cte_path)
        if (
            cte.cfg.action_dim != 14
            or cte.cfg.transition_steps != 4
            or cte.cfg.effect_window_transitions != 4
            or cte.cfg.image_channels != 48
        ):
            raise ValueError("CTE checkpoint is incompatible with RoboTwin latent Stage-1 contract")
        if int(cte.cfg.retrieval_dim) != int(tc.key_dim):
            raise ValueError("behavior-bank key_dim must match CTE retrieval_dim")
        if int(cte.cfg.hidden_dim) != int(tc.value_dim):
            raise ValueError("behavior-bank value_dim must match CTE hidden_dim")
        if int(tc.value_dim) != int(zeva.prompt.global_dim):
            raise ValueError("behavior-bank value_dim must match CausalPrompt global_dim")
        if tuple(payload.get("camera_keys", ())) != ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            raise ValueError("CTE checkpoint camera order mismatch")
        if str(payload.get("cte_input_type")) != "wan_vae_latent":
            raise ValueError("optimized behavior bank requires wan_vae_latent CTE")

        stats_sha = sha256_file(str(cfg.data.train.pretrained_norm_stats))
        semantic_sha = sha256_file(str(cfg.data.train.semantic_task_map_path))
        if _semantic_sha(payload) != semantic_sha:
            raise ValueError("current semantic task map SHA differs from Stage-1 CTE checkpoint")
        video_size = tuple(int(v) for v in cfg.data.train.video_size)
        if tuple(int(v) for v in payload.get("cte_vae_input_size", ())) != video_size:
            raise ValueError("CTE checkpoint VAE input size differs from current data.video_size")

        latent_cache = CachedCTELatentWindowDataset(
            latent_path,
            expected_dataset_stats_sha256=stats_sha,
            expected_semantic_task_sha256=semantic_sha,
            expected_video_size=video_size,
            expected_action_dim=14,
            expected_transition_steps=4,
            expected_latent_channels=48,
        )
        latent_cache.validate_storage_layout(
            require_complete_windows=latent_cache.manifest.get("profile") != "main_observed",
        )
        if str(latent_cache.semantic_task_identity.get("sha256")) != semantic_sha:
            raise ValueError("latent cache semantic-task identity differs from current map")

        # Instantiate the ordinary dataset only once.  Full trajectory CTE input
        # comes from the latent cache; RGB is decoded only for episode starts and
        # the final non-aligned tail when needed.
        base = instantiate(cfg.data.train)
        if getattr(base, "semantic_task_identity", None) is None:
            raise ValueError("RobotVideoDataset did not load semantic_task_map_path")
        if str(base.semantic_task_identity.get("sha256")) != semantic_sha:
            raise ValueError("dataset semantic-task identity differs from CTE/latent cache")
        dataset = ZevaRobotWinDataset(base)
        plans = _build_episode_plans_metadata_only(
            dataset,
            sample_stride=1,
            transition_steps=cte.cfg.transition_steps,
            source_window_actions=32,
            show_progress=info.rank == 0,
            include_observed_tail=is_main_profile(cfg),
        )
        plans, main_metadata = select_main_success_rows(plans, cfg)
        if main_metadata:
            validate_main_artifact(payload, main_metadata)
            validate_main_artifact(latent_cache.manifest, main_metadata)
            main_metadata["cte_window_semantics"] = MAIN_CTE_WINDOW_SEMANTICS
        plans = _canonicalize_plans(plans, latent_cache)
        _validate_plan_alignment(plans, latent_cache)
        buckets, loads = _partition_episode_plans(plans, info.world_size)
        local_plans = buckets[info.rank]
        if info.rank == 0:
            print("========== optimized behavior bank ==========")
            print(f"episodes: {len(plans):,}")
            print(f"semantic tasks: {len(set(str(p.task_id) for p in plans))}")
            print(f"rank query loads: {loads}")
            print(f"CTE sha256: {checkpoint_sha256(cte_path)}")
            print(f"base sha256: {checkpoint_sha256(base_ckpt)}")
            print(f"semantic map sha256: {semantic_sha}")

        device = _resolve_device(cfg, info)
        cte.to(device).eval().requires_grad_(False)
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.zeva.enabled = False
        policy = instantiate(
            model_cfg,
            model_dtype=_mixed_precision_to_model_dtype(str(cfg.mixed_precision)),
            device=str(device),
        )
        policy.load_checkpoint(str(base_ckpt))
        policy.eval().requires_grad_(False)
        validate_vae_metadata(
            dict(payload.get("vae_metadata", {})),
            {
                "model_id": str(cfg.model.model_id),
                "z_dim": int(getattr(policy.vae, "z_dim", -1)),
                "temporal_downsample_factor": int(getattr(policy.vae, "temporal_downsample_factor", -1)),
                "upsampling_factor": int(getattr(policy.vae, "upsampling_factor", -1)),
            },
        )
        frame_encoder = FastWAMCTELatentEncoder(
            policy,
            resize=video_size,
            expected_channels=cte.cfg.image_channels,
            input_range="minus_one_one",
        ).encode_history

        # Fail before long tail processing if the FastWAM readout contract or
        # batched execution is bad.  Probe two episodes when possible because
        # the production path deliberately uses batched readout extraction.
        if local_plans:
            probe_count = min(2, len(local_plans))
            probe_source = _InitialReadoutDataset(dataset, local_plans[:probe_count])
            probes = [probe_source[i] for i in range(probe_count)]
            with torch.inference_mode():
                probe_readout = policy.extract_task_context_readout(
                    torch.stack([v["image"] for v in probes]).to(device),
                    torch.stack([v["context"] for v in probes]).to(device),
                    torch.stack([v["context_mask"] for v in probes]).to(device),
                )
            expected_readout_dim = int(cfg.model.video_dit_config.hidden_dim)
            expected_shape = (probe_count, expected_readout_dim)
            if tuple(probe_readout.shape) != expected_shape:
                raise RuntimeError(
                    f"FastWAM batched readout preflight mismatch: "
                    f"{tuple(probe_readout.shape)} != {expected_shape}"
                )
            if not bool(torch.isfinite(probe_readout).all()):
                raise FloatingPointError("FastWAM batched readout preflight produced NaN/Inf")
            print(f"[rank {info.rank}] batched readout preflight PASSED: {tuple(probe_readout.shape)}")
        if _env_bool("ZEVA_BEHAVIOR_READOUT_FIRST_FRAME_ONLY", False) and local_plans:
            # Compare the *real* first frames and frozen text-cache values,
            # before any expensive CTE tail/prototype/readout work. Failure
            # is fail-closed, not silently tolerated.
            count = min(max(1, _env_int("ZEVA_BEHAVIOR_READOUT_VERIFY_SAMPLES", 2)), len(local_plans))
            original = _InitialReadoutDataset(dataset, local_plans[:count])
            optimized = _InitialReadoutDataset(dataset, local_plans[:count], first_frame_only=True)
            for idx in range(count):
                reference, observed = original[idx], optimized[idx]
                for key in ("episode_id", "task_id", "dataset_start", "instruction"):
                    if reference[key] != observed[key]:
                        raise RuntimeError(
                            f"initial-readout first-frame identity mismatch at {idx}: {key}"
                        )
                for key in ("image", "context", "context_mask"):
                    if not torch.equal(reference[key], observed[key]):
                        error = float((reference[key].float() - observed[key].float()).abs().max())
                        raise RuntimeError(
                            f"initial-readout first-frame difference at {idx}: {key} max_abs={error}; "
                            "resume with ZEVA_BEHAVIOR_READOUT_FIRST_FRAME_ONLY=0"
                        )
            print(f"[rank {info.rank}] first-frame readout equivalence PASSED: {count} samples", flush=True)
        _barrier(info)

        tail_plans = [plan for plan in local_plans if _tail_required(plan, latent_cache)]
        tail_batch_size = _env_int("ZEVA_BEHAVIOR_TAIL_BATCH_SIZE", 4)
        tail_workers = _env_int("ZEVA_BEHAVIOR_TAIL_WORKERS", 2)
        tails = _precompute_tail_supplements(
            plans=tail_plans,
            dataset=dataset,
            encoder=frame_encoder,
            device=device,
            batch_size=tail_batch_size,
            num_workers=tail_workers,
            prefetch_factor=2,
            rank=info.rank,
            expected_latent_shape=latent_cache.latent_shape,
        )

        # Full-prefix CTE prototypes.  This is small relative to the FastWAM
        # readout path and is kept episode-local to avoid variable-length padding.
        prototypes: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        progress = tqdm(
            local_plans,
            desc=f"CTE prototypes rank {info.rank}/{info.world_size}",
            unit="episode",
            position=info.rank,
            dynamic_ncols=True,
        )
        with torch.inference_mode():
            for plan in progress:
                prototypes[plan.episode_id] = _compute_cte_prototype(
                    plan,
                    latent_cache=latent_cache,
                    tail=tails.get(plan.episode_id),
                    cte=cte,
                    device=device,
                )

        readout_batch = _env_int("ZEVA_BEHAVIOR_READOUT_BATCH_SIZE", 2)
        readout_workers = _env_int("ZEVA_BEHAVIOR_READOUT_WORKERS", 2)
        episode_ids, task_ids, instructions, readouts = _build_readouts(
            dataset=dataset,
            plans=local_plans,
            policy=policy,
            device=device,
            batch_size=readout_batch,
            num_workers=readout_workers,
            rank=info.rank,
            first_frame_only=_env_bool("ZEVA_BEHAVIOR_READOUT_FIRST_FRAME_ONLY", False),
            fast_collate=_env_bool("ZEVA_BEHAVIOR_READOUT_FAST_COLLATE", False),
            profile_every=_env_int("ZEVA_BEHAVIOR_READOUT_PROFILE_EVERY", 0),
        )
        if episode_ids != [plan.episode_id for plan in local_plans]:
            raise RuntimeError("FastWAM readout order differs from local episode plan order")
        if task_ids != [str(plan.task_id) for plan in local_plans]:
            raise RuntimeError("FastWAM readout task IDs differ from local episode plan order")

        local_entries = []
        for plan, episode_id, task_id, instruction in zip(
            local_plans, episode_ids, task_ids, instructions, strict=True
        ):
            key, value, transitions = prototypes[episode_id]
            local_entries.append(
                {
                    "retrieval_key": key,
                    "behavior_value": value,
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "instruction": instruction,
                    "num_transitions": int(transitions),
                    "dataset_start": int(plan.dataset_start),
                }
            )

        work_root = bank_path.parent / f".{bank_path.stem}.building"
        if info.rank == 0:
            if work_root.exists():
                shutil.rmtree(work_root)
            work_root.mkdir(parents=True, exist_ok=True)
        _barrier(info)
        rank_file = work_root / f"rank_{info.rank:03d}.pt"
        torch.save(
            {
                "entries": local_entries,
                "episode_ids": episode_ids,
                "readouts": readouts.float().cpu(),
                "tail_windows": len(tails),
            },
            rank_file,
        )
        _barrier(info)

        if info.rank == 0:
            merged: list[tuple[dict[str, Any], torch.Tensor]] = []
            total_tail = 0
            for rank in range(info.world_size):
                payload_rank = torch.load(work_root / f"rank_{rank:03d}.pt", map_location="cpu", weights_only=False)
                entries = payload_rank["entries"]
                values = payload_rank["readouts"]
                if values.ndim != 2 or values.shape[0] != len(entries):
                    raise RuntimeError(f"rank {rank} readout/entry count mismatch")
                merged.extend(zip(entries, values.unbind(0), strict=True))
                total_tail += int(payload_rank.get("tail_windows", 0))
            merged.sort(key=lambda pair: int(pair[0]["dataset_start"]))
            entries = [pair[0] for pair in merged]
            all_readouts = torch.stack([pair[1] for pair in merged]).float().contiguous()
            ids = [str(entry["episode_id"]) for entry in entries]
            if len(entries) != len(plans) or len(set(ids)) != len(ids):
                raise RuntimeError("merged behavior bank does not contain one unique entry per episode")
            task_set = {str(entry["task_id"]) for entry in entries}
            if not main_metadata and len(task_set) != 50:
                raise RuntimeError(f"formal RoboTwin behavior bank expected 50 semantic tasks, got {len(task_set)}")
            if not bool(torch.isfinite(all_readouts).all()):
                raise FloatingPointError("merged FastWAM readouts contain NaN/Inf")

            metadata = {
                **main_metadata,
                "builder": "build_zeva_behavior_bank_optimized.py",
                "base_checkpoint_sha256": checkpoint_sha256(base_ckpt),
                "cte_checkpoint_sha256": checkpoint_sha256(cte_path),
                "dataset_stats_sha256": stats_sha,
                "semantic_task_map_sha256": semantic_sha,
                "latent_cache_manifest_sha256": sha256_file(latent_path / "manifest.json"),
                "video_size": list(video_size),
                "context_len": int(cfg.model.tokenizer_max_len),
                "dataset_dirs": [str(v) for v in cfg.data.train.dataset_dirs],
                "key_space": BEHAVIOR_KEY_SPACE,
                "value_space": BEHAVIOR_VALUE_SPACE,
                "readout_kind": READOUT_KIND,
                "split": "train",
                "readout_dim": int(all_readouts.shape[1]),
                "history_semantics": "full_episode_prefix",
                "num_episodes": len(entries),
                "semantic_task_count": len(task_set),
            }
            bank = TaskContextBank(
                entries,
                key_dim=cte.cfg.retrieval_dim,
                value_dim=cte.cfg.hidden_dim,
                temperature=float(tc.temperature),
                metadata=metadata,
            )
            bank, all_readouts, validation_readouts = split_main_behavior_artifacts(bank, all_readouts)
            if main_metadata:
                main_metadata = {**main_metadata, "data_role": "train"}
            entries = list(bank.entries)
            ids = [str(entry["episode_id"]) for entry in entries]
            metadata = dict(bank.metadata)
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            readout_path.parent.mkdir(parents=True, exist_ok=True)
            # Build both final files under the private work directory first.
            # A crash must never leave a partially-written artifact at the
            # configured production path.  os.replace preserves file bytes, so
            # the bank SHA embedded in the readout remains valid after commit.
            tmp_bank = bank_path.with_name(f".{bank_path.name}.tmp")
            tmp_readout = readout_path.with_name(f".{readout_path.name}.tmp")
            bank.save(tmp_bank)
            bank_hash = checkpoint_sha256(tmp_bank)
            torch.save(
                {
                    **main_metadata,
                    **validation_readouts,
                    "format": READOUT_FORMAT,
                    "readout_kind": READOUT_KIND,
                    "bank_sha256": bank_hash,
                    "episode_ids": ids,
                    "readouts": all_readouts,
                    "metadata": metadata,
                },
                tmp_readout,
            )
            # Reload through public readers before committing either artifact.
            reloaded = TaskContextBank.load(
                tmp_bank,
                expected_key_dim=cte.cfg.retrieval_dim,
                expected_value_dim=cte.cfg.hidden_dim,
            )
            validate_behavior_bank(reloaded)
            load_readout_cache(tmp_readout, tmp_bank, reloaded)
            if len(reloaded) != len(entries):
                raise RuntimeError("behavior-bank reload count mismatch")
            os.replace(tmp_bank, bank_path)
            os.replace(tmp_readout, readout_path)
            if checkpoint_sha256(bank_path) != bank_hash:
                raise RuntimeError("behavior-bank hash changed during atomic commit")
            print("========== behavior bank complete ==========")
            print(f"entries: {len(entries):,}")
            print(f"readouts: {tuple(all_readouts.shape)} float32")
            print(f"semantic tasks: {len(task_set)}")
            print(f"tail windows: {total_tail:,}")
            print(f"bank: {bank_path}")
            print(f"readouts: {readout_path}")
            print("BEHAVIOR BANK: PASSED")
            shutil.rmtree(work_root)
        _barrier(info)
    finally:
        _destroy_process_group(info)


if __name__ == "__main__":
    main()
