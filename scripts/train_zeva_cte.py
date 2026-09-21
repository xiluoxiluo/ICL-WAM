"""Stage 1 CTE training on ordered RoboTwin windows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tqdm import tqdm

import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva import (
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    CTELossConfig,
    TaskBalancedCTEBatchSampler,
    build_cte_training_index,
    causal_transition_encoder_loss,
)
from fastwam.zeva.checkpoint import load_cte_checkpoint, save_cte_checkpoint
from fastwam.zeva.schemas import sha256_file
from fastwam.zeva.vae_adapter import FastWAMCTELatentEncoder, load_frozen_wan_vae


def _cfg_dict(value) -> dict:
    return {} if value is None else dict(OmegaConf.to_container(value, resolve=True))

def _identity_collate(batch):
    """Keep Zeva samples as a list of dicts."""
    return batch

def _video_size_hw(cfg: DictConfig) -> tuple[int, int]:
    value = cfg.data.train.get("video_size")
    if value is None or len(value) != 2:
        raise ValueError("data.train.video_size must be [H, W] for Zeva CTE training")
    size = tuple(int(v) for v in value)
    if min(size) < 1:
        raise ValueError(f"data.train.video_size must be positive, got {size}")
    return size


def _distributed_context() -> tuple[int, int, int, bool]:
    """Initialize torch.distributed when launched by torchrun."""

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"invalid distributed environment: rank={rank}, world_size={world_size}"
        )
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size, rank == 0


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _reduce_mean(values: list[float], device: torch.device, world_size: int) -> list[float]:
    if world_size == 1:
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(world_size)
    return [float(value) for value in tensor.cpu()]


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size, is_main_process = _distributed_context()
    configured_device = cfg.get("device")
    if configured_device is None or str(configured_device).strip().lower() in {"", "none", "null"}:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = str(configured_device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Stage 1 requested CUDA but torch.cuda.is_available() is false; "
            "set device=cpu only for a small debug run"
        )
    if device_name == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    stats_value = str(cfg.data.train.get("pretrained_norm_stats", ""))
    if stats_value in {"", "None", "null"} or not Path(stats_value).is_file():
        raise FileNotFoundError(
            "Zeva Stage 1 requires an existing data.train.pretrained_norm_stats "
            "file so CTE/cache normalization is reproducible"
        )
    base = instantiate(cfg.data.train)
    dataset = ZevaRobotWinDataset(base)
    if len(dataset) == 0:
        raise ValueError("Stage 1 dataset is empty")
    zeva = cfg.model.get("zeva", {})
    cte_values = _cfg_dict(zeva.get("cte"))
    allowed = set(CausalTransitionEncoderConfig.__dataclass_fields__)
    model = CausalTransitionEncoder(CausalTransitionEncoderConfig(**{k: v for k, v in cte_values.items() if k in allowed}))
    cte_input_type = str(cte_values.get("input_type", "rgb_frame"))
    if cte_input_type not in {"rgb_frame", "wan_vae_latent"}:
        raise ValueError("zeva.cte.input_type must be rgb_frame or wan_vae_latent")
    if cte_input_type == "rgb_frame" and model.cfg.image_channels != 3:
        raise ValueError("rgb_frame CTE training requires image_channels=3")
    frame_encoder = None
    vae_metadata: dict[str, object] = {}
    cte_vae_input_size = _video_size_hw(cfg)
    if cte_input_type == "wan_vae_latent":
        model_values = _cfg_dict(cfg.model)
        vae, vae_metadata = load_frozen_wan_vae(
            model_id=str(model_values.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
            tokenizer_model_id=str(model_values.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")),
            device=str(device),
            torch_dtype=torch.float32 if device.type == "cpu" else torch.bfloat16,
            redirect_common_files=bool(model_values.get("redirect_common_files", True)),
        )
        frame_encoder = FastWAMCTELatentEncoder(
            vae, resize=cte_vae_input_size, expected_channels=model.cfg.image_channels,
            input_range="minus_one_one",
        ).encode_history
    if (
        model.cfg.action_dim != 14
        or model.cfg.transition_steps != 4
        or model.cfg.effect_window_transitions != 4
    ):
        raise ValueError(
            "RoboTwin Zeva V1 Stage 1 requires CTE action_dim=14, "
            "transition_steps=4, and effect_window_transitions=4; "
            f"got action_dim={model.cfg.action_dim}, "
            f"transition_steps={model.cfg.transition_steps}, "
            f"effect_window_transitions={model.cfg.effect_window_transitions}"
        )
    model.to(device)
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    batch_size = max(int(cfg.get("batch_size") or 1), 1)
    sample_stride = int(cfg.data.train.get("global_sample_stride", 1))
    if sample_stride != 1:
        raise ValueError(
            "RoboTwin Zeva V1 requires data.train.global_sample_stride=1 for exact "
            f"frame/action alignment; got {sample_stride}"
        )
    steps = int(cfg.get("max_steps") or 1000)
    training_index = build_cte_training_index(
        dataset,
        source_window_actions=32,
        sample_stride=sample_stride,
    )
    if not training_index:
        raise ValueError("Stage 1 contains no complete, non-overlapping CTE windows")
    samples_per_task = int(cfg.get("cte_samples_per_task") or 4)
    configured_tasks = cfg.get("cte_tasks_per_batch")
    tasks_per_batch = None if configured_tasks in (None, "", "None", "null") else int(configured_tasks)
    if tasks_per_batch is None:
        tasks_per_batch = max(1, min(4, batch_size // max(samples_per_task, 1)))
    batch_sampler = TaskBalancedCTEBatchSampler(
        training_index,
        batch_size=batch_size,
        tasks_per_batch=tasks_per_batch,
        samples_per_task=samples_per_task,
        seed=seed,
    )
    if tasks_per_batch < 2 and is_main_process:
        print(
            "Stage 1 warning: configured batch_size cannot provide two tasks "
            f"with {samples_per_task} samples per task; task contrastive loss may be sparse."
        )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    model.train(); step = 0
    train_model = model
    if world_size > 1:
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            # Task/effect contrastive terms can be empty for a rank-local
            # remainder batch, so parameter usage is intentionally dynamic.
            find_unused_parameters=True,
        )
    output_dir = Path(str(cfg.output_dir))
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    _barrier(world_size)
    # Keep the exact resolved run configuration and data identity alongside the
    # weights so a cache cannot be mistaken for a different CTE/data pair.
    if is_main_process:
        OmegaConf.save(config=cfg, f=str(output_dir / "config.yaml"), resolve=True)
    dataset_manifest = {
        "dataset_dirs": [str(value) for value in cfg.data.train.dataset_dirs],
        "dataset_stats": stats_value,
        "dataset_stats_sha256": sha256_file(stats_value),
        "action_dim": model.cfg.action_dim,
        "transition_steps": model.cfg.transition_steps,
        "action_horizon": model.cfg.transition_steps * 8,
        "video_frames": 9,
        "image_channels": model.cfg.image_channels,
        "cte_input_type": cte_input_type,
        "latent_channels": model.cfg.image_channels if cte_input_type == "wan_vae_latent" else 0,
        "vae_metadata": vae_metadata,
        "cte_vae_input_size": list(cte_vae_input_size),
        "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    }
    if is_main_process:
        (output_dir / "dataset_manifest.json").write_text(
            json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _barrier(world_size)
    resume_path = cfg.get("resume")
    if resume_path not in (None, "", "None", "null"):
        resume_payload = load_cte_checkpoint(
            str(resume_path), model, optimizer=optimizer, scheduler=scheduler, map_location=str(device)
        )
        step = int(resume_payload.get("step", 0))
        if step >= steps and is_main_process:
            print(f"Stage 1 checkpoint already reached max_steps={steps}: {resume_path}")
        metrics_file = (output_dir / "metrics.jsonl").open("a", encoding="utf-8") if is_main_process else None
    else:
        metrics_file = (output_dir / "metrics.jsonl").open("w", encoding="utf-8") if is_main_process else None
    semantic_lookup: dict[str, int] = {}

    def train_batch(batch: list[dict]) -> None:
        frames_list, actions_list, valid_list, transition_valid_list = [], [], [], []
        preencoded_flags = []

        semantic_ids = []
        batch_task_keys: list[str] = []
        batch_episode_ids: list[str] = []
        batch_window_starts: list[int] = []

        # ------------------------------------------------------------
        # 1. Collect the whole batch first.
        #
        # IMPORTANT:
        # Do NOT run Wan VAE inside this per-sample loop.
        # ------------------------------------------------------------
        for sample in batch:
            frames = sample.get("cte_frames")
            preencoded = frames is not None

            if frames is None:
                frames = torch.cat(
                    (
                        sample["before_frames"],
                        sample["after_frames"][-1:],
                    ),
                    dim=0,
                )
            elif frames.ndim != 4 or frames.shape[0] != 9:
                raise ValueError(
                    "sample['cte_frames'] must be [9,C,H,W]"
                )

            frames_list.append(frames)
            preencoded_flags.append(preencoded)

            actions_list.append(sample["transition_actions"])
            valid_list.append(sample["frame_valid"])
            transition_valid_list.append(sample["transition_valid"])

            episode = sample["episode"]

            task_key = (
                f"id:{episode.task_id}"
                if episode.task_id not in (None, 0, "0")
                else f"name:{episode.task_name}"
            )

            semantic_lookup.setdefault(
                task_key,
                len(semantic_lookup),
            )

            semantic_ids.append(
                semantic_lookup[task_key]
            )

            batch_task_keys.append(task_key)
            batch_episode_ids.append(
                str(episode.episode_id)
            )
            batch_window_starts.append(
                int(episode.episode_step)
            )

        # ------------------------------------------------------------
        # 2. Stack RGB/latent histories.
        #
        # RGB case:
        #   [B, 9, 3, H, W]
        #
        # preencoded case:
        #   [B, 9, C_latent, H_latent, W_latent]
        # ------------------------------------------------------------
        frames = torch.stack(
            frames_list,
            dim=0,
        )

        # A batch must not mix raw RGB and pre-encoded CTE latents.
        if any(preencoded_flags) and not all(preencoded_flags):
            raise ValueError(
                "A CTE batch cannot mix raw RGB frames "
                "and pre-encoded CTE frames."
            )

        # ------------------------------------------------------------
        # 3. Encode the WHOLE batch with the frozen Wan VAE.
        #
        # OLD:
        #
        #   for sample in batch:
        #       frame_encoder(frames.unsqueeze(0))
        #
        #   => B independent VAE calls
        #
        # NEW:
        #
        #   frame_encoder([B, 9, 3, H, W])
        #
        #   => one batched VAE call
        #
        # FastWAMCTELatentEncoder.encode_history() internally:
        #
        #   [B,9,3,H,W]
        #       ↓ flatten
        #   [B*9,3,H,W]
        #       ↓ Wan VAE
        #   [B*9,C,H',W']
        #       ↓ reshape
        #   [B,9,C,H',W']
        #
        # Do NOT move the result back to CPU here.
        # ------------------------------------------------------------
        if frame_encoder is not None and not all(preencoded_flags):
            frames = frame_encoder(frames)

        # frame_encoder normally already returns a CUDA tensor when the
        # frozen Wan VAE lives on CUDA. Keep this .to(device) for both
        # RGB and pre-encoded paths.
        frames = frames.to(device)

        actions = torch.stack(
            actions_list,
            dim=0,
        ).to(device)

        valid = torch.stack(
            valid_list,
            dim=0,
        ).to(device)

        transition_valid = torch.stack(
            transition_valid_list,
            dim=0,
        ).to(device)

        # ------------------------------------------------------------
        # 4. CTE forward
        # ------------------------------------------------------------
        output = train_model(
            frames,
            actions,
            valid_mask=valid,
            transition_valid=transition_valid,
        )

        losses = causal_transition_encoder_loss(
            output,
            actions,
            valid,
            torch.tensor(
                semantic_ids,
                dtype=torch.long,
                device=device,
            ),
            CTELossConfig(),
        )

        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(
                f"non-finite CTE loss at step {step + 1}: "
                f"{losses['total'].item()}"
            )

        # ------------------------------------------------------------
        # 5. Backward
        # ------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)

        losses["total"].backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            train_model.parameters(),
            float(cfg.get("max_grad_norm", 1.0)),
        )

        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite CTE gradient norm at step {step + 1}: "
                f"{grad_norm.item()}"
            )

        optimizer.step()
        scheduler.step()

        model.update_ema_target()

        # ------------------------------------------------------------
        # 6. DDP metric reduction
        # ------------------------------------------------------------
        reduced_loss_values = _reduce_mean(
            [
                float(losses["total"].detach()),
                float(losses["action"].detach()),
                float(losses["vision"].detach()),
                float(losses["loss_task"].detach()),
                float(losses["loss_phase"].detach()),
                float(losses["loss_effect"].detach()),
                float(grad_norm.detach()),
            ],
            device,
            world_size,
        )

        metric_payload = {
            "step": int(step + 1),
            "loss": reduced_loss_values[0],
            "action": reduced_loss_values[1],
            "vision": reduced_loss_values[2],
            "task": reduced_loss_values[3],
            "phase": reduced_loss_values[4],
            "effect": reduced_loss_values[5],
            "grad_norm": reduced_loss_values[6],

            "actual_batch_size": len(batch) * world_size,

            "distinct_task_count": len(
                set(batch_task_keys)
            ),

            "task_positive_anchor_count": sum(
                count
                for count in {
                    key: batch_task_keys.count(key)
                    for key in set(batch_task_keys)
                }.values()
                if count > 1
            ),

            "task_positive_pair_count": sum(
                count * (count - 1) // 2
                for count in {
                    key: batch_task_keys.count(key)
                    for key in set(batch_task_keys)
                }.values()
            ),

            "distinct_episode_count": len(
                set(batch_episode_ids)
            ),

            "dataset_index": [
                int(sample.get("dataset_index", -1))
                for sample in batch
            ],

            "episode_id": batch_episode_ids,
            "task_id": batch_task_keys,
            "episode_step": batch_window_starts,
        }

        metric_payload.update(
            {
                "cte/loss": metric_payload["loss"],
                "cte/loss_action": metric_payload["action"],
                "cte/loss_vision": metric_payload["vision"],
                "cte/loss_task": metric_payload["task"],
                "cte/loss_phase": metric_payload["phase"],
                "cte/loss_effect": metric_payload["effect"],
                "cte/actual_batch_size": metric_payload[
                    "actual_batch_size"
                ],
                "cte/distinct_task_count": metric_payload[
                    "distinct_task_count"
                ],
                "cte/task_positive_anchor_count": metric_payload[
                    "task_positive_anchor_count"
                ],
                "cte/task_positive_pair_count": metric_payload[
                    "task_positive_pair_count"
                ],
                "cte/distinct_episode_count": metric_payload[
                    "distinct_episode_count"
                ],
            }
        )

        if is_main_process:
            metrics_file.write(
                json.dumps(metric_payload) + "\n"
            )
            metrics_file.flush()

    # ================================================================
    # Training loop
    # ================================================================
    epoch = 0

    while step < steps:
        # Temporal selection is complete before batching. This prevents a
        # same-episode overlap row (e.g. step 1) from being flushed into the
        # next optimizer batch after step 0 was selected.

        batch_sampler.set_epoch(epoch)
        epoch += 1

        progressed = False

        all_batches = list(batch_sampler)

        if not all_batches:
            raise RuntimeError(
                "Stage 1 sampler produced no training batch"
            )

        # ------------------------------------------------------------
        # DDP batch sharding
        # ------------------------------------------------------------
        if world_size > 1:
            batches_per_rank = (
                len(all_batches) + world_size - 1
            ) // world_size

            padded_batches = (
                all_batches
                + [all_batches[0]]
                * (
                    batches_per_rank * world_size
                    - len(all_batches)
                )
            )

            local_batches = padded_batches[
                rank::world_size
            ]

        else:
            local_batches = all_batches

        # ------------------------------------------------------------
        # Progress bar
        # ------------------------------------------------------------
                # ------------------------------------------------------------
        # Multi-process RGB/video loading
        # ------------------------------------------------------------
        local_index_batches = [
            [int(row.dataset_index) for row in index_batch]
            for index_batch in local_batches
        ]

        num_workers = max(
            int(cfg.get("num_workers") or 0),
            0,
        )

        loader_kwargs = {
            "dataset": dataset,
            "batch_sampler": local_index_batches,
            "num_workers": num_workers,
            "collate_fn": _identity_collate,
            "pin_memory": device.type == "cuda",
        }

        # prefetch_factor is valid only when num_workers > 0
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        train_loader = DataLoader(
            **loader_kwargs,
        )

        iterator = tqdm(
            train_loader,
            total=len(local_index_batches),
            desc=f"CTE training rank {rank}",
            disable=not is_main_process,
            dynamic_ncols=True,
        )

        for batch in iterator:
            if step >= steps:
                break

            if is_main_process:
                print(
                    f"[step {step + 1}] "
                    f"RGB batch loaded by {num_workers} workers, "
                    "running batched VAE + CTE...",
                    flush=True,
                )

            train_batch(batch)

            step += 1
            progressed = True

            if is_main_process:
                iterator.set_postfix(
                    step=step
                )

        if not progressed:
            raise RuntimeError(
                "Stage 1 sampler produced no training batch"
            )

    # ================================================================
    # Finish training
    # ================================================================
    if is_main_process:
        metrics_file.close()

    _barrier(world_size)

    # ================================================================
    # Save Stage-1 CTE checkpoint
    # ================================================================
    if is_main_process:
        save_cte_checkpoint(
            output_dir / "cte.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            config=_cfg_dict(zeva),
            cte_input_type=cte_input_type,
            vae_metadata=vae_metadata,
            cte_vae_input_size=cte_vae_input_size,
        )

        print(
            f"saved Stage 1 checkpoint: "
            f"{output_dir / 'cte.pt'}"
        )

    _barrier(world_size)


if __name__ == "__main__":
    main()
