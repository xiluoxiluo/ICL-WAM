"""Precompute RoboTwin Stage-1 Wan-VAE latent windows with torchrun.

Recommended launch on one 16-GPU node:

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m torch.distributed.run \
  --standalone --nproc_per_node=16 \
  scripts/precompute_zeva_cte_latents.py \
  task=robotwin_zeva_fastwam_3cam_384 \
  +model.zeva.cte.latent_cache_path=/ABS/PATH/zeva_cte_latent_cache_v1 \
  +cte_latent_cache.batch_size=4 \
  +cte_latent_cache.num_workers=2 \
  +cte_latent_cache.flush_every_batches=8

Design:
- force PyAV; TorchCodec is never attempted;
- use the exact RobotVideoDataset preprocessing used by Stage-1;
- keep non-overlapping H32 storage windows, including main's masked real tail;
- assign whole episodes to ranks: episode_index % world_size;
- store latent values losslessly as raw BF16 bits in mmap .npy files;
- also cache normalized actions/masks, so Stage-1 training needs neither RGB
  decoding nor the frozen Wan VAE;
- resumable per rank via progress.json.
"""

from __future__ import annotations

import json
import errno
import io
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva.cte_latent_cache import (
    CACHE_FORMAT,
    LATENT_STORAGE_DTYPE,
    bf16_tensor_to_u16_numpy,
)
from fastwam.zeva.schemas import sha256_file
from fastwam.zeva.semantic_tasks import parse_episode_index_from_id
from fastwam.zeva.stage1_sampling import CTETrainIndex, build_cte_training_index
from fastwam.zeva.main_data import is_main_profile, normalize_main_dataset_paths, select_main_success_rows, validate_main_artifact
from fastwam.zeva.cte_windows import (
    MAIN_CTE_WINDOW_SEMANTICS, observed_prefix_transitions, validate_main_cte_window_semantics,
)
from fastwam.zeva.vae_adapter import (
    FastWAMCTELatentEncoder, load_frozen_wan_vae, validate_vae_metadata,
)
from fastwam.zeva_latent_storage import (
    CacheStorageError, atomic_write_json, cache_window_payload_bytes,
    check_storage_capacity, read_npy_header, recommended_storage_margin_bytes,
    reserve_file_storage, storage_usage,
)


def _cfg_dict(value) -> dict:
    return {} if value is None else dict(OmegaConf.to_container(value, resolve=True))


def _video_size_hw(cfg: DictConfig) -> tuple[int, int]:
    value = cfg.data.train.get("video_size")
    if value is None or len(value) != 2:
        raise ValueError("data.train.video_size must be [H, W]")
    size = tuple(int(v) for v in value)
    if min(size) < 1:
        raise ValueError(f"invalid video_size: {size}")
    return size


def _distributed_context(*, timeout_seconds: int = 86400) -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if timeout_seconds < 1:
        raise ValueError("cte_latent_cache.process_group_timeout_seconds must be positive")

    if world_size < 1:
        raise ValueError(
            f"invalid WORLD_SIZE={world_size}"
        )

    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"invalid rank={rank}, world_size={world_size}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "VAE latent precompute requires CUDA"
        )

    visible_gpu_count = torch.cuda.device_count()

    if local_rank < 0 or local_rank >= visible_gpu_count:
        raise RuntimeError(
            "LOCAL_RANK exceeds visible CUDA devices: "
            f"local_rank={local_rank}, "
            f"visible_gpu_count={visible_gpu_count}, "
            f"CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )

    # Important: bind rank -> GPU BEFORE NCCL initialization.
    torch.cuda.set_device(local_rank)

    device = torch.device(
        "cuda",
        local_rank,
    )

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
            # Completed/empty shards can wait while other GPUs encode for
            # hours. torchrun still terminates peers promptly if a rank exits.
            timeout=timedelta(seconds=timeout_seconds),
        )

    return (
        rank,
        local_rank,
        world_size,
        rank == 0,
    )


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _atomic_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _all_rank_values(value: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    values: list[Any] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def _storage_recovery_error(exc: OSError) -> OSError:
    if isinstance(exc, CacheStorageError) or exc.errno not in (errno.ENOSPC, getattr(errno, "EDQUOT", 122)):
        return exc
    cause = "Disk quota exceeded" if exc.errno == getattr(errno, "EDQUOT", 122) else "No space left on device"
    return CacheStorageError(
        exc.errno,
        f"{cause} while writing the CTE latent cache. Preserve existing arrays and the last "
        "committed progress.json; temporary JSON files are not recovery checkpoints. "
        "Stop all writers, extend/free the relevant quota or move the complete cache to a "
        "filesystem with enough space and quota, then resume at the same cache path with "
        "overwrite=false and the original generation world size. After migration, set the "
        "cache path to the copied cache root. Filesystem free space alone does not verify quota.",
        exc.filename,
    )


def _rank_phase(label: str, callback, world_size: int):
    """Report local failures before any rank enters the next storage phase.

    DONE and empty ranks must also call this function. Keep callbacks free of
    collectives: one local EDQUOT must not leave peers waiting in a barrier.
    Large objects such as mmap arrays and model instances stay on their rank.
    """
    result = None
    error = None
    try:
        result = callback()
    except Exception as exc:
        error = _storage_recovery_error(exc) if isinstance(exc, OSError) else exc
    errors = _all_rank_values(
        None if error is None else f"{type(error).__name__}: {error}", world_size,
    )
    if any(message is not None for message in errors):
        if world_size == 1 and error is not None:
            raise error
        failures = "\n".join(
            f"  rank {rank_id}: {message}"
            for rank_id, message in enumerate(errors) if message is not None
        )
        raise RuntimeError(f"CTE latent cache {label} failed:\n{failures}") from error
    return result


def _identity_collate(batch):
    return batch


def _semantic_remap_rows(rows: list[CTETrainIndex], base_dataset) -> list[CTETrainIndex]:
    resolver = getattr(base_dataset, "resolve_semantic_task_id", None)
    if not callable(resolver):
        raise RuntimeError("RobotVideoDataset.resolve_semantic_task_id() is required")
    out: list[CTETrainIndex] = []
    for row in rows:
        episode_index = parse_episode_index_from_id(row.episode_id)
        semantic_task = resolver(
            episode_index=episode_index,
            raw_task_index=row.task_id,
            strict=True,
        )
        out.append(
            CTETrainIndex(
                dataset_index=int(row.dataset_index),
                episode_id=str(row.episode_id),
                task_id=str(semantic_task),
                episode_step=int(row.episode_step),
            )
        )
    return out


def _semantic_stats(rows: list[CTETrainIndex]) -> dict[str, Any]:
    by_task: dict[str, set[str]] = defaultdict(set)
    windows_per_task: Counter[str] = Counter()
    for row in rows:
        task = str(row.task_id)
        by_task[task].add(str(row.episode_id))
        windows_per_task[task] += 1
    episodes_per_task = {task: len(value) for task, value in by_task.items()}
    return {
        "semantic_task_count": len(by_task),
        "singleton_semantic_tasks": sum(v == 1 for v in episodes_per_task.values()),
        "min_episodes_per_task": min(episodes_per_task.values()) if episodes_per_task else 0,
        "max_episodes_per_task": max(episodes_per_task.values()) if episodes_per_task else 0,
        "episodes_per_task": dict(sorted(episodes_per_task.items())),
        "windows_per_task": dict(sorted(windows_per_task.items())),
    }


@dataclass(frozen=True)
class _BuildItem:
    dataset_index: int
    episode_id: str
    episode_step: int
    frames: torch.Tensor
    transition_actions: torch.Tensor
    frame_valid: torch.Tensor
    transition_valid: torch.Tensor
    task_id: str


class _WindowSourceDataset(Dataset):
    """Materialize exact Stage-1 windows from the normal RGB dataset."""

    def __init__(self, dataset: ZevaRobotWinDataset, rows: list[CTETrainIndex]) -> None:
        self.dataset = dataset
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> _BuildItem:
        row = self.rows[int(index)]
        sample = self.dataset[int(row.dataset_index)]
        source_index = int(sample.get("dataset_index", -1))
        if source_index != int(row.dataset_index):
            raise RuntimeError(
                f"cache source mismatch: requested {row.dataset_index}, got {source_index}"
            )

        episode = sample["episode"]
        if str(episode.episode_id) != str(row.episode_id):
            raise ValueError(
                f"episode mismatch for dataset_index={row.dataset_index}: "
                f"index={row.episode_id}, sample={episode.episode_id}"
            )
        if int(episode.episode_step) != int(row.episode_step):
            raise ValueError(
                f"episode_step mismatch for dataset_index={row.dataset_index}: "
                f"index={row.episode_step}, sample={episode.episode_step}"
            )

        frames = sample.get("cte_frames")
        if frames is None:
            frames = torch.cat(
                (sample["before_frames"], sample["after_frames"][-1:]),
                dim=0,
            )
        if frames.ndim != 4 or frames.shape[0] != 9 or frames.shape[1] != 3:
            raise ValueError(
                "precompute source must be RGB [9,3,H,W]; "
                f"got {tuple(frames.shape)}"
            )
        if observed_prefix_transitions(sample["frame_valid"], sample["transition_valid"]) < 1:
            raise ValueError("CTE latent cache source has no real completed control interval")

        return _BuildItem(
            dataset_index=int(row.dataset_index),
            episode_id=str(episode.episode_id),
            episode_step=int(episode.episode_step),
            frames=frames,
            transition_actions=sample["transition_actions"].float(),
            frame_valid=sample["frame_valid"].bool(),
            transition_valid=sample["transition_valid"].bool(),
            task_id=str(episode.task_id),
        )


def _load_progress(rank_dir: Path, local_count: int) -> int:
    path = rank_dir / "progress.json"
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload.get("total")) is not int or payload["total"] != local_count:
        raise ValueError(
            f"resume count mismatch in {path}: expected {local_count}, "
            f"got {payload.get('total')}"
        )
    next_offset = payload.get("next_offset")
    if type(next_offset) is not int or not 0 <= next_offset <= local_count:
        raise ValueError(f"invalid resume offset {next_offset} in {path}")
    return next_offset


def _array_layout(count: int, latent_shape: tuple[int, ...]) -> dict:
    return {
        "latents": ("latents_bf16_u16.npy", np.dtype(np.uint16), (count, *latent_shape)),
        "actions": ("transition_actions.npy", np.dtype(np.float32), (count, 8, 4, 14)),
        "frame_valid": ("frame_valid.npy", np.dtype(np.uint8), (count, 9)),
        "transition_valid": ("transition_valid.npy", np.dtype(np.uint8), (count, 8, 4)),
    }


def _validate_arrays(
    rank_dir: Path, *, count: int, latent_shape: tuple[int, ...],
    allow_missing: bool = False,
) -> None:
    """Read only headers/stat before opening any existing array for writes."""
    for filename, dtype, shape in _array_layout(count, latent_shape).values():
        path = rank_dir / filename
        if not path.exists() and allow_missing:
            continue
        header = read_npy_header(path)
        if tuple(header["shape"]) != shape or np.dtype(header["dtype"]) != dtype:
            raise ValueError(
                f"resume array shape/dtype mismatch in {path}: expected {shape}/{dtype}, "
                f"got {header['shape']}/{header['dtype']}; existing files were not changed"
            )
        if header["fortran_order"]:
            raise ValueError(f"resume cache requires C-order array: {path}")
        if header["logical_bytes"] != header["expected_file_bytes"]:
            raise ValueError(
                f"resume array file length mismatch in {path}: expected "
                f"{header['expected_file_bytes']}, got {header['logical_bytes']}; "
                "preserve this cache and restore the complete file before resuming"
            )


def _open_existing_arrays(
    rank_dir: Path, *, count: int, latent_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    _validate_arrays(rank_dir, count=count, latent_shape=latent_shape)
    return {
        name: np.load(rank_dir / filename, mmap_mode="r+", allow_pickle=False)
        for name, (filename, _, _) in _array_layout(count, latent_shape).items()
    }


def _create_arrays(
    rank_dir: Path,
    *,
    count: int,
    latent_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    frame_valid_shape: tuple[int, ...],
    transition_valid_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    if (action_shape, frame_valid_shape, transition_valid_shape) != ((8, 4, 14), (9,), (8, 4)):
        raise ValueError("unexpected CTE action/mask shape; refusing to create incompatible cache")
    _validate_arrays(rank_dir, count=count, latent_shape=latent_shape, allow_missing=True)
    # Reuse valid zero-progress files left by an interrupted allocation. Never
    # use w+ on an existing file, including when its progress was not committed.
    for filename, dtype, shape in _array_layout(count, latent_shape).values():
        path = rank_dir / filename
        if path.exists():
            continue
        # Exclusive create also protects an existing cache from a second
        # writer racing this one. A partially created invalid file is reported
        # on restart; no existing file is repaired/deleted automatically.
        with path.open("xb") as handle:
            np.lib.format.write_array_header_1_0(handle, {
                "descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": False, "shape": shape,
            })
            handle.truncate(handle.tell() + int(np.prod(shape)) * dtype.itemsize)
    return _open_existing_arrays(rank_dir, count=count, latent_shape=latent_shape)


def _validate_generation_layout(cache_root: Path, world_size: int) -> None:
    for directory in cache_root.glob("rank_*"):
        if not directory.is_dir():
            continue
        try:
            rank_id = int(directory.name.removeprefix("rank_"))
        except ValueError:
            continue
        if not 0 <= rank_id < world_size and any(directory.iterdir()):
            raise ValueError(
                f"cache contains {directory.name} outside current world_size={world_size}; "
                "resume with the original generation world size, or use a new cache path"
            )
        path = directory / "rank_meta.json"
        if path.is_file():
            meta = json.loads(path.read_text(encoding="utf-8"))
            if type(meta.get("world_size")) is not int or meta["world_size"] != world_size:
                raise ValueError(
                    f"resume world_size mismatch in {path}: saved {meta.get('world_size')}, "
                    f"current {world_size}; resume with the same generation world size"
                )


def _inspect_rank_state(
    rank_dir: Path, *, rank: int, world_size: int, count: int,
    main_metadata: dict, input_size: tuple[int, int], expected_channels: int,
) -> dict:
    offset = _load_progress(rank_dir, count)
    done = (rank_dir / "DONE").is_file()
    path = rank_dir / "rank_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if (offset > 0 or done) and meta is None:
        raise ValueError(f"committed cache is missing rank_meta.json: {rank_dir}")
    shape = None
    if meta is not None:
        for key, expected in (("rank", rank), ("world_size", world_size), ("num_windows", count)):
            if type(meta.get(key)) is not int or meta[key] != expected:
                raise ValueError(
                    f"resume {key} mismatch in {path}: expected {expected}, got {meta.get(key)}; "
                    "preserve the cache and resume with its original generation world size"
                )
        if main_metadata:
            validate_main_artifact(meta, main_metadata)
            validate_main_cte_window_semantics(meta)
        values = meta.get("latent_shape")
        if count > 0:
            if (not isinstance(values, list) or len(values) != 4
                    or any(type(v) is not int or v < 1 for v in values)
                    or values[0] != 9 or values[1] != expected_channels):
                raise ValueError(f"invalid stored latent_shape in {path}: {values}")
            shape = tuple(values)
            if meta.get("latent_storage_dtype") != LATENT_STORAGE_DTYPE:
                raise ValueError(f"latent storage dtype mismatch in {path}")
            if tuple(meta.get("cte_vae_input_size", ())) != input_size:
                raise ValueError(f"CTE VAE input size mismatch in {path}")
            if not isinstance(meta.get("vae_metadata"), dict):
                raise ValueError(f"missing VAE identity metadata in {path}")
            _validate_arrays(rank_dir, count=count, latent_shape=shape,
                             allow_missing=offset == 0 and not done)
        elif values != []:
            raise ValueError(f"empty rank must have empty latent_shape: {path}")
    if done and (not (rank_dir / "progress.json").is_file() or offset != count):
        raise ValueError(f"DONE cache does not have a fully committed progress.json: {rank_dir}")
    return {"metadata": meta, "offset": offset, "done": done, "shape": shape, "count": count}


def _storage_plan(rank_dir: Path, count: int, latent_shape: tuple[int, ...]) -> dict:
    layout = _array_layout(count, latent_shape) if count else {}
    expected = 0
    additional = 0
    for filename, dtype, shape in layout.values():
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False, "shape": shape,
        })
        file_bytes = header.tell() + int(np.prod(shape)) * dtype.itemsize
        allocated = storage_usage([rank_dir / filename])["allocated_bytes"]
        expected += file_bytes
        additional += max(0, file_bytes - allocated)
    usage = storage_usage(rank_dir / values[0] for values in layout.values())
    return {**usage, "expected_array_bytes": expected, "required_additional_bytes": additional}


def _encode_batch(items, frame_encoder, device, expected_shape=None) -> np.ndarray:
    frames = torch.stack([item.frames for item in items], dim=0).to(device, non_blocking=True)
    with torch.inference_mode():
        latent = frame_encoder(frames)
    if latent.ndim != 5 or latent.shape[0] != len(items) or latent.shape[1] != 9:
        raise ValueError(
            "FastWAMCTELatentEncoder must return [B,9,C,H,W], "
            f"got {tuple(latent.shape)}"
        )
    actual_shape = tuple(int(v) for v in latent.shape[1:])
    if expected_shape is not None and actual_shape != tuple(expected_shape):
        raise ValueError(
            f"actual VAE latent shape {actual_shape} differs from cache shape {expected_shape}; "
            "existing cache values and committed progress were preserved"
        )
    return bf16_tensor_to_u16_numpy(latent)


def _flush_arrays(arrays: dict[str, np.ndarray]) -> None:
    for value in arrays.values():
        value.flush()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    normalize_main_dataset_paths(cfg)
    rank, local_rank, world_size, is_main = _distributed_context(
        timeout_seconds=int(_cfg_dict(cfg.get("cte_latent_cache")).get("process_group_timeout_seconds", 86400)),
    )
    device = torch.device("cuda", local_rank)

    try:
        zeva = cfg.model.get("zeva", {})
        cte_values = _cfg_dict(zeva.get("cte"))
        cache_cfg = _cfg_dict(cfg.get("cte_latent_cache"))

        cache_value = cache_cfg.get("path", cte_values.get("latent_cache_path"))
        if cache_value in (None, "", "None", "null"):
            raise ValueError(
                "Set +model.zeva.cte.latent_cache_path=/absolute/cache/path "
                "or +cte_latent_cache.path=/absolute/cache/path"
            )
        cache_root = Path(str(cache_value)).expanduser().resolve()

        batch_size = max(int(cache_cfg.get("batch_size", 4)), 1)
        num_workers = max(int(cache_cfg.get("num_workers", 2)), 0)
        prefetch_factor = max(int(cache_cfg.get("prefetch_factor", 2)), 1)
        flush_every = max(int(cache_cfg.get("flush_every_batches", 8)), 1)
        overwrite = bool(cache_cfg.get("overwrite", False))
        reserve_space = bool(cache_cfg.get("reserve_space", True))
        presample_images = bool(cache_cfg.get("presample_images", True))

        stats_path = Path(str(cfg.data.train.get("pretrained_norm_stats", "")))
        if not stats_path.is_file():
            raise FileNotFoundError(
                "VAE latent cache requires data.train.pretrained_norm_stats"
            )
        semantic_map_path = Path(str(cfg.data.train.get("semantic_task_map_path", "")))
        if not semantic_map_path.is_file():
            raise FileNotFoundError(
                "VAE latent cache requires data.train.semantic_task_map_path"
            )

        # Force the exact requested decoder.  This bypasses get_safe_default_codec
        # and therefore never attempts TorchCodec.
        OmegaConf.update(cfg, "data.train.video_backend", "pyav", force_add=True)
        OmegaConf.update(cfg, "data.train.use_text_embed_cache", False, force_add=True)

        # Only request the 9 image timestamps needed by the 32-action CTE
        # window. PyAV still decodes inter-frame dependencies internally, but
        # this avoids carrying/resize-processing 33 RGB tensors in Python.
        BaseLerobotDataset.presample_images = presample_images

        def clear_requested_cache():
            if rank == 0 and overwrite and cache_root.exists():
                shutil.rmtree(cache_root)
        _rank_phase("requested overwrite", clear_requested_cache, world_size)
        _rank_phase("directory creation", lambda: cache_root.mkdir(parents=True, exist_ok=True), world_size)
        _rank_phase("generation layout validation", lambda: (
            _validate_generation_layout(cache_root, world_size) if is_main else None
        ), world_size)

        base = instantiate(cfg.data.train)
        if not bool(getattr(base, "require_semantic_task_id", False)):
            raise ValueError("formal cache generation requires require_semantic_task_id=true")
        semantic_identity = getattr(base, "semantic_task_identity", None)
        if semantic_identity is None:
            raise ValueError("semantic_task_identity is missing")
        dataset = ZevaRobotWinDataset(base)

        sample_stride = int(cfg.data.train.get("global_sample_stride", 1))
        if sample_stride != 1:
            raise ValueError("RoboTwin Zeva V1 requires global_sample_stride=1")

        all_rows = build_cte_training_index(
            dataset,
            source_window_actions=32,
            sample_stride=sample_stride,
            include_observed_tail=is_main_profile(cfg),
        )
        all_rows, main_metadata = select_main_success_rows(all_rows, cfg)
        if main_metadata:
            main_metadata["cte_window_semantics"] = MAIN_CTE_WINDOW_SEMANTICS
        if not all_rows:
            raise RuntimeError("no Stage-1 CTE windows found")

        # A resumed cache must refer to the same successful demonstrations;
        # equal window counts alone cannot establish that identity.
        provenance_path = cache_root / "success_data_contract.json"
        if main_metadata:
            def validate_provenance():
                if not is_main:
                    return
                if provenance_path.exists():
                    previous = json.loads(provenance_path.read_text())
                    validate_main_artifact(previous, main_metadata)
                    validate_main_cte_window_semantics(previous)
                elif any(cache_root.glob("rank_*/progress.json")):
                    raise ValueError("cannot resume a legacy latent cache as main_observed; rebuild in a new path")
                else:
                    _atomic_json(provenance_path, main_metadata)
            _rank_phase("success data contract", validate_provenance, world_size)

        # Keep all windows from one episode on one GPU.  This is deterministic
        # and prevents duplicate window generation across ranks.
        local_rows = [
            row
            for row in all_rows
            if parse_episode_index_from_id(row.episode_id) % world_size == rank
        ]
        local_count = len(local_rows)
        rank_dir = cache_root / f"rank_{rank:03d}"
        expected_channels = int(cte_values.get("image_channels", 48))
        cte_vae_input_size = _video_size_hw(cfg)
        state = _rank_phase("read-only resume validation", lambda: _inspect_rank_state(
            rank_dir, rank=rank, world_size=world_size, count=local_count,
            main_metadata=main_metadata, input_size=cte_vae_input_size,
            expected_channels=expected_channels,
        ), world_size)
        # Do not broadcast the potentially large success episode/split lists
        # from every rank's metadata; they were validated locally above.
        states = _all_rank_values({
            "shape": state["shape"], "count": local_count, "done": state["done"],
            "vae_metadata": (state["metadata"] or {}).get("vae_metadata"),
        }, world_size)
        shapes = {tuple(item["shape"]) for item in states if item["shape"] is not None}
        if len(shapes) > 1:
            raise ValueError(f"stored ranks disagree on latent shape: {sorted(shapes)}")
        stored_vaes = [item["vae_metadata"] for item in states if item["shape"] is not None]
        canonical_vae_metadata = stored_vaes[0] if stored_vaes else None
        for metadata in stored_vaes:
            validate_vae_metadata(canonical_vae_metadata, metadata)
        progress_offset = state["offset"]
        remaining_rows = local_rows[progress_offset:]
        runtime: dict[str, Any] = {}

        def prepare_encoder_and_loader():
            if state["done"] or not remaining_rows or runtime:
                return
            model_values = _cfg_dict(cfg.model)
            vae, vae_metadata = load_frozen_wan_vae(
                model_id=str(model_values.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
                tokenizer_model_id=str(
                    model_values.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")
                ),
                device=str(device),
                torch_dtype=torch.bfloat16,
                redirect_common_files=bool(model_values.get("redirect_common_files", True)),
            )
            if canonical_vae_metadata is not None:
                validate_vae_metadata(canonical_vae_metadata, vae_metadata)
            frame_encoder = FastWAMCTELatentEncoder(
                vae,
                resize=cte_vae_input_size,
                expected_channels=expected_channels,
                input_range="minus_one_one",
            ).encode_history
            source_dataset = _WindowSourceDataset(dataset, remaining_rows)
            loader_kwargs: dict[str, Any] = {
                "dataset": source_dataset,
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": num_workers,
                "collate_fn": _identity_collate,
                "pin_memory": True,
            }
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = prefetch_factor
                loader_kwargs["persistent_workers"] = True
            loader = DataLoader(**loader_kwargs)
            runtime.update(encoder=frame_encoder, vae_metadata=vae_metadata,
                           loader=loader, iterator=iter(loader))

        if shapes:
            latent_shape = next(iter(shapes))
        else:
            # The designated rank encodes its actual first pending batch only
            # once. Keep that batch/result for the normal writer so probing
            # neither discards a sample nor advances its preprocessing twice.
            probe_rank = next(i for i, item in enumerate(states) if item["count"] > 0)

            def probe_shape():
                if rank != probe_rank:
                    return None
                prepare_encoder_and_loader()
                items = next(runtime["iterator"])
                bits = _encode_batch(items, runtime["encoder"], device)
                runtime["first_batch"] = (items, bits)
                return {"shape": tuple(int(v) for v in bits.shape[1:]),
                        "vae_metadata": runtime["vae_metadata"]}

            probe_result = _rank_phase("first-batch shape probe", probe_shape, world_size)
            probe_results = _all_rank_values(probe_result, world_size)
            latent_shape = tuple(probe_results[probe_rank]["shape"])
            canonical_vae_metadata = probe_results[probe_rank]["vae_metadata"]
            if (len(latent_shape) != 4 or latent_shape[0] != 9
                    or latent_shape[1] != expected_channels or min(latent_shape) < 1):
                raise ValueError(f"unexpected actual VAE latent shape: {latent_shape}")

        # No writable mmap or bulk encoding happens before this whole-cache
        # accounting. Sparse files contribute allocated blocks, not st_size.
        def inspect_storage():
            if local_count:
                _validate_arrays(rank_dir, count=local_count, latent_shape=latent_shape,
                                 allow_missing=progress_offset == 0 and not state["done"])
            return _storage_plan(rank_dir, local_count, latent_shape)

        local_plan = _rank_phase("array storage inspection", inspect_storage, world_size)
        plans = _all_rank_values(local_plan, world_size)
        payload_bytes = len(all_rows) * cache_window_payload_bytes(latent_shape)
        additional_bytes = sum(item["required_additional_bytes"] for item in plans)
        all_done = all(item["done"] for item in states)
        margin_bytes = 0 if all_done else recommended_storage_margin_bytes(payload_bytes)

        def check_whole_cache_capacity():
            if not is_main:
                return
            print("========== CTE latent cache storage preflight ==========", flush=True)
            print(f"windows: {len(all_rows):,}; actual latent shape/window: {latent_shape}", flush=True)
            print(f"full array payload: {payload_bytes / 1024**3:.2f} GiB; "
                  f"already allocated: {sum(p['allocated_bytes'] for p in plans) / 1024**3:.2f} GiB", flush=True)
            print(f"additional array allocation: {additional_bytes / 1024**3:.2f} GiB; "
                  f"metadata headroom: {margin_bytes / 1024**3:.2f} GiB", flush=True)
            if all_done:
                print("All rank arrays are complete; array reservation and payload headroom are skipped. "
                      "Final metadata may still need space/quota.", flush=True)
            report = check_storage_capacity(cache_root, additional_bytes if not all_done else 0,
                                            margin_bytes=margin_bytes)
            print(f"filesystem available: {report['available_bytes'] / 1024**3:.2f} GiB; "
                  "user/project quota is not established by filesystem free space", flush=True)

        _rank_phase("whole-cache capacity preflight", check_whole_cache_capacity, world_size)
        _rank_phase("VAE initialization", prepare_encoder_and_loader, world_size)

        def prepare_rank_files():
            if state["done"]:
                return None, None
            rank_dir.mkdir(parents=True, exist_ok=True)
            if state["metadata"] is None:
                rank_meta = {**main_metadata, "rank": rank, "world_size": world_size,
                             "num_windows": local_count, "latent_shape": list(latent_shape) if local_count else []}
                if local_count:
                    rank_meta.update(vae_metadata=runtime["vae_metadata"],
                                     cte_vae_input_size=list(cte_vae_input_size),
                                     latent_storage_dtype=LATENT_STORAGE_DTYPE)
                _atomic_json(rank_dir / "rank_meta.json", rank_meta)
            if not local_count:
                return None, None
            arrays = _create_arrays(
                rank_dir, count=local_count, latent_shape=latent_shape,
                action_shape=(8, 4, 14), frame_valid_shape=(9,), transition_valid_shape=(8, 4),
            )
            paths = [rank_dir / values[0] for values in _array_layout(local_count, latent_shape).values()]
            reservation = reserve_file_storage(paths) if reserve_space else None
            return arrays, reservation

        arrays, reservation = _rank_phase("array preparation/reservation", prepare_rank_files, world_size)
        reservations = _all_rank_values(reservation, world_size)
        if is_main and not all_done:
            if not reserve_space:
                print("WARNING: reserve_space=false; array ranges were not reserved. "
                      "Quota/full-disk errors can still occur during encoding.", flush=True)
            else:
                unsupported = [path for report in reservations if report is not None
                               for path in report["unsupported_paths"]]
                if unsupported:
                    print(f"WARNING: filesystem allocation reservation is unsupported for {len(unsupported)} files; "
                          "continuing with filesystem-only checks. User/project quota remains unverified.", flush=True)
                else:
                    print("Array file ranges reserved. Later metadata and other jobs' quota use are not guaranteed.", flush=True)

        if state["done"]:
            print(f"[latent-cache rank {rank}] already complete: {local_count:,} windows")
        else:
            def pending_batches():
                if "first_batch" in runtime:
                    yield runtime.pop("first_batch")
                for items in runtime.get("iterator", ()):
                    yield items, None

            progress = tqdm(
                pending_batches(),
                total=(len(remaining_rows) + batch_size - 1) // batch_size,
                desc=f"VAE latent cache GPU{local_rank} rank {rank}/{world_size}",
                unit="batch",
                dynamic_ncols=True,
                position=rank,
                leave=True,
            )

            write_offset = progress_offset
            batches_since_flush = 0
            with torch.inference_mode():
                for items, latent_bits in progress:
                    if latent_bits is None:
                        latent_bits = _encode_batch(items, runtime["encoder"], device, latent_shape)

                    actions = torch.stack(
                        [item.transition_actions for item in items], dim=0
                    ).numpy()
                    frame_valid = torch.stack(
                        [item.frame_valid for item in items], dim=0
                    ).to(torch.uint8).numpy()
                    transition_valid = torch.stack(
                        [item.transition_valid for item in items], dim=0
                    ).to(torch.uint8).numpy()

                    if (tuple(actions.shape[1:]), tuple(frame_valid.shape[1:]),
                            tuple(transition_valid.shape[1:])) != ((8, 4, 14), (9,), (8, 4)):
                        raise ValueError("CTE source action/mask dimensions differ from the allocated cache")

                    assert arrays is not None
                    count = len(items)
                    end = write_offset + count
                    if end > local_count:
                        raise RuntimeError("rank cache write exceeded allocated window count")
                    arrays["latents"][write_offset:end] = latent_bits
                    arrays["actions"][write_offset:end] = actions
                    arrays["frame_valid"][write_offset:end] = frame_valid
                    arrays["transition_valid"][write_offset:end] = transition_valid
                    write_offset = end
                    batches_since_flush += 1

                    if batches_since_flush >= flush_every:
                        _flush_arrays(arrays)
                        _atomic_json(
                            rank_dir / "progress.json",
                            {"next_offset": write_offset, "total": local_count},
                        )
                        batches_since_flush = 0

                    progress.set_postfix(
                        windows=f"{write_offset:,}/{local_count:,}",
                        refresh=False,
                    )

            if local_count > 0 and arrays is None:
                raise RuntimeError("rank produced no latent batch despite non-empty row assignment")
            if arrays is not None:
                _flush_arrays(arrays)
            if write_offset != local_count:
                raise RuntimeError(
                    f"rank {rank} cache incomplete: wrote {write_offset}, expected {local_count}"
                )
            _atomic_json(
                rank_dir / "progress.json",
                {"next_offset": local_count, "total": local_count},
            )
            (rank_dir / "DONE").write_text("ok\n", encoding="utf-8")
            print(f"[latent-cache rank {rank}] complete: {local_count:,} windows")

        _barrier(world_size)

        def finalize_cache():
            if not is_main:
                return
            canonical_rows = _semantic_remap_rows(all_rows, base)
            semantic_stats = _semantic_stats(canonical_rows)

            rank_metas: list[dict[str, Any]] = []
            for rank_id in range(world_size):
                rank_dir = cache_root / f"rank_{rank_id:03d}"
                if not (rank_dir / "DONE").is_file():
                    raise RuntimeError(f"rank {rank_id} did not finish latent cache generation")
                rank_metas.append(
                    json.loads((rank_dir / "rank_meta.json").read_text(encoding="utf-8"))
                )
                if main_metadata:
                    validate_main_artifact(rank_metas[-1], main_metadata)
                    validate_main_cte_window_semantics(rank_metas[-1])

            nonempty_shapes = {
                tuple(int(v) for v in meta["latent_shape"])
                for meta in rank_metas
                if int(meta.get("num_windows", 0)) > 0
            }
            if len(nonempty_shapes) != 1:
                raise ValueError(f"ranks disagree on latent shape: {sorted(nonempty_shapes)}")
            latent_shape = next(iter(nonempty_shapes))

            rank_counts = [0] * world_size
            index_rows: list[dict[str, Any]] = []
            for row in canonical_rows:
                episode_index = parse_episode_index_from_id(row.episode_id)
                rank_id = episode_index % world_size
                offset = rank_counts[rank_id]
                rank_counts[rank_id] += 1
                index_rows.append(
                    {
                        "dataset_index": int(row.dataset_index),
                        "episode_id": str(row.episode_id),
                        "episode_index": int(episode_index),
                        "episode_step": int(row.episode_step),
                        "task_id": str(row.task_id),
                        "rank": int(rank_id),
                        "offset": int(offset),
                    }
                )

            declared_counts = [int(meta["num_windows"]) for meta in rank_metas]
            if rank_counts != declared_counts:
                raise ValueError(
                    f"rank cache counts do not match deterministic partition: "
                    f"index={rank_counts}, files={declared_counts}"
                )

            first_meta = next(
                meta for meta in rank_metas if int(meta.get("num_windows", 0)) > 0
            )
            vae_metadata = dict(first_meta["vae_metadata"])
            for meta in rank_metas:
                if int(meta.get("num_windows", 0)) == 0:
                    continue
                validate_vae_metadata(vae_metadata, dict(meta["vae_metadata"]))

            bytes_per_window = cache_window_payload_bytes(latent_shape)
            payload_bytes_estimate = int(len(index_rows) * bytes_per_window)

            manifest = {
                **main_metadata,
                "format": CACHE_FORMAT,
                "dataset_dirs": [str(v) for v in cfg.data.train.dataset_dirs],
                "dataset_stats": str(stats_path.resolve()),
                "dataset_stats_sha256": sha256_file(stats_path),
                "semantic_task_identity": semantic_identity,
                "semantic_task_stats": semantic_stats,
                "semantic_task_map_sha256": sha256_file(semantic_map_path),
                "video_backend": "pyav",
                "presample_images": presample_images,
                "video_size": list(_video_size_hw(cfg)),
                "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                "source_window_actions": 32,
                "sample_stride": sample_stride,
                "video_frames": 9,
                "transition_count": 8,
                "transition_steps": 4,
                "action_dim": 14,
                "latent_channels": int(latent_shape[1]),
                "latent_shape": list(latent_shape),
                "latent_storage_dtype": LATENT_STORAGE_DTYPE,
                "vae_metadata": vae_metadata,
                "cte_vae_input_size": list(_video_size_hw(cfg)),
                "num_windows": len(index_rows),
                "bytes_per_window_estimate": bytes_per_window,
                "payload_bytes_estimate": payload_bytes_estimate,
                "generation_world_size": world_size,
                "rank_window_counts": rank_counts,
                "split": {
                    "is_training_set": bool(cfg.data.train.get("is_training_set", True)),
                    "val_set_proportion": float(cfg.data.train.get("val_set_proportion", 0.0)),
                    "seed": int(cfg.get("seed", 42)),
                },
            }
            _atomic_json(cache_root / "index.json", index_rows)
            _atomic_json(cache_root / "manifest.json", manifest)

            print("========== VAE latent cache ==========")
            print(f"path: {cache_root}")
            print(f"windows: {len(index_rows):,}")
            print(f"latent shape/window: {latent_shape}")
            print(f"storage: {LATENT_STORAGE_DTYPE}")
            print(
                "payload estimate: "
                f"{payload_bytes_estimate / (1024 ** 3):.2f} GiB"
            )
            print(f"semantic tasks: {semantic_stats['semantic_task_count']}")
            print(f"rank counts: {rank_counts}")
            print("VAE LATENT CACHE: PASSED")

        _rank_phase("final index/manifest commit", finalize_cache, world_size)

    except OSError as exc:
        error = _storage_recovery_error(exc)
        if error is exc:
            raise
        raise error from exc
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
