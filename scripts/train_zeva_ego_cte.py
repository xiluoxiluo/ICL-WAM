"""Train the Zeva-Ego CTE on full RoboTwin episodes at FastWAM H24 boundaries.

Example:
    python scripts/train_zeva_ego_cte.py \
      task=robotwin_zeva_fastwam_ego_cte_3cam_384 output_dir=/path/to/cte

The legacy window/VAE trainer remains scripts/train_zeva_cte.py. This entry
point is deliberately single-process: no implicit repeated-episode DDP sampler.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

from fastwam.zeva.ego_cte import ZevaEgoCTE, ZevaEgoCTEConfig, zeva_ego_cte_loss
from fastwam.zeva.ego_cte_training import (
    FORMAL_EPOCHS, MIN_PREDICTION_GAIN, REFERENCE_REVISION,
    PairedEpochSampler, build_ego_cte_optimizer, build_episode_splits,
    canonical_sha256, episode_metadata, load_ego_cte_checkpoint,
    select_ego_episode_scope,
    restore_ego_training_checkpoint, save_ego_cte_checkpoint,
    sha256_file, validate_ego_cte,
)
from fastwam.zeva.ego_episode_data import RoboTwinEgoEpisodeDataset, collate_ego_episodes


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def train(cfg: DictConfig) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Zeva-Ego CTE uses one process and full no-replacement epochs; multi-rank torchrun is not supported")
    epochs, batch_size = int(cfg.get("num_epochs", FORMAL_EPOCHS)), int(cfg.get("batch_size", 8))
    workers, seed = int(cfg.get("num_workers", 4)), int(cfg.get("seed", 1000))
    evaluation_every = int(cfg.get("cte_eval_every_epochs", 5))
    smoke_steps = int(cfg.get("cte_smoke_steps", 0))
    log_every = int(cfg.get("log_every", 10))
    if min(epochs, batch_size, evaluation_every, log_every) < 1 or workers < 0 or smoke_steps < 0:
        raise ValueError("Epochs, batch size and intervals must be positive; workers/smoke steps must be nonnegative")
    if batch_size % 2:
        raise ValueError("Use an even episode batch size so same-task contrastive pairs stay together")
    if int(cfg.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("This CTE recipe updates once per full episode batch; gradient accumulation is not implemented")
    if cfg.get("max_steps") not in (None, "", "None", "null"):
        raise ValueError("Formal CTE training is epoch-based; use max_steps=null and cte_smoke_steps only for smoke checks")
    gradient_clip = float(cfg.get("max_grad_norm", 1.0))
    effect_weight = float(cfg.get("cte_effect_weight", 0.2))
    if not np.isfinite(gradient_clip) or gradient_clip <= 0 or not np.isfinite(effect_weight) or effect_weight < 0:
        raise ValueError("Gradient clipping must be positive and effect_weight finite and nonnegative")

    raw_model = dict(OmegaConf.to_container(cfg.model.zeva.cte, resolve=True))
    model_config = ZevaEgoCTEConfig(**{key: value for key, value in raw_model.items()
                                      if key in ZevaEgoCTEConfig.__dataclass_fields__})
    if not smoke_steps and (not model_config.formal_architecture or not model_config.vision_pretrained):
        raise ValueError("Formal CTE training requires pretrained ResNet18/Mamba; explicit test backends require cte_smoke_steps>0")
    configured_device = cfg.get("device")
    device = torch.device(str(configured_device) if configured_device not in (None, "", "None", "null")
                          else "cuda" if torch.cuda.is_available() else "cpu")
    if model_config.formal_architecture and device.type != "cuda":
        raise ValueError("Formal Mamba CTE requires a CUDA device; CPU smoke uses explicit gru_test/tiny_test backends")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_config = cfg.data.train
    if float(data_config.get("val_set_proportion", -1)) != 0.0:
        raise ValueError("data.train.val_set_proportion must be 0; this trainer splits whole episodes explicitly")
    if not bool(data_config.get("require_semantic_task_id", False)):
        raise ValueError("CTE requires canonical semantic task IDs, not language task_index labels")
    if bool(data_config.get("use_text_embed_cache", True)):
        raise ValueError("Stage1 has no language input; set data.train.use_text_embed_cache=false")
    stats_path = Path(str(data_config.get("pretrained_norm_stats", ""))).expanduser()
    semantic_path = Path(str(data_config.get("semantic_task_map_path", ""))).expanduser()
    for name, path in (("pretrained_norm_stats", stats_path), ("semantic_task_map_path", semantic_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Existing {name} file is required: {path}")

    output = Path(str(cfg.output_dir)).expanduser().resolve()
    resume = cfg.get("resume")
    resume = None if resume in (None, "", "None", "null") else str(resume)
    if smoke_steps and resume:
        raise ValueError("Smoke runs start independently and cannot resume formal checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise ValueError("Fresh CTE training requires an empty output directory")

    base = instantiate(data_config)
    dataset = RoboTwinEgoEpisodeDataset(base, executed_horizon=model_config.executed_horizon,
                                        image_size=model_config.image_size, action_dim=model_config.action_dim)
    split_policy = str(cfg.get("cte_split_policy", "zeva_ego"))
    if split_policy not in {"zeva_ego", "legacy_semantic_task"}:
        raise ValueError("cte_split_policy must be zeva_ego or legacy_semantic_task")
    split_seed = int(cfg.get("cte_split_seed", 1000)) if split_policy == "zeva_ego" else seed
    data_scope, partition_rows = None, None
    if split_policy == "zeva_ego":
        task_subset_path, conditions_path = cfg.get("cte_task_subset_path"), cfg.get("cte_episode_conditions_path")
        if not task_subset_path or not conditions_path:
            raise ValueError("Reference CTE training requires cte_task_subset_path and cte_episode_conditions_path")
        data_scope, partition_rows = select_ego_episode_scope(
            dataset, task_subset_path=str(task_subset_path), conditions_path=str(conditions_path),
            conditions=list(cfg.get("cte_conditions", ["Clean", "Randomized"])),
        )
    rows = episode_metadata(dataset.episodes)
    split_path = cfg.get("cte_episode_split_path")
    split_path = None if split_path in (None, "", "None", "null") else str(split_path)
    splits = build_episode_splits(dataset.episodes, seed=split_seed,
                                  validation_fraction=float(cfg.get("cte_validation_fraction", 0.05)),
                                  split_path=split_path, split_policy=split_policy, partition_rows=partition_rows)
    positions = {row["episode_id"]: index for index, row in enumerate(rows)}
    indices = {key: [positions[identity] for identity in ids] for key, ids in splits.items()}
    train_dataset, validation_dataset = Subset(dataset, indices["train"]), Subset(dataset, indices["validation"])
    tasks = data_scope["task_names"] if data_scope is not None else sorted({row["task_id"] for row in rows})
    sampler = PairedEpochSampler([rows[index]["task_id"] for index in indices["train"]], seed, task_order=tasks)
    task_ids = {task: i for i, task in enumerate(tasks)}
    loader_generator = torch.Generator().manual_seed(seed)
    loader_options = dict(batch_size=batch_size, num_workers=workers, collate_fn=collate_ego_episodes,
                          pin_memory=device.type == "cuda", generator=loader_generator,
                          worker_init_fn=_seed_worker, persistent_workers=False, drop_last=False)
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    if resume:
        model, previous = load_ego_cte_checkpoint(resume, device=device, require_ready=False)
        if model.cfg != model_config:
            raise ValueError("Resume checkpoint config differs from this training configuration")
    else:
        model, previous = ZevaEgoCTE(model_config).to(device), None
    optimizer = build_ego_cte_optimizer(model, learning_rate=float(cfg.get("learning_rate", 1e-4)),
                                        vision_learning_rate=float(cfg.get("cte_vision_learning_rate", 1e-5)),
                                        weight_decay=float(cfg.get("weight_decay", 1e-4)))
    source_root = Path(__file__).resolve().parents[1]
    source_paths = {"trainer": Path(__file__).resolve(),
                    **{name: source_root / "src/fastwam/zeva" / f"{name}.py"
                       for name in ("ego_cte", "ego_cte_training", "ego_episode_data")}}
    recipe = {"name": "zeva_ego_epoch80_with_partition_holdout" if data_scope is not None else "legacy_ego_epoch80_with_episode_holdout",
              "formal_endpoint_epoch": FORMAL_EPOCHS,
              "split_policy": split_policy, "split_seed": split_seed,
              "validation_fraction": float(cfg.get("cte_validation_fraction", 0.05)),
              "batch_size": batch_size, "seed": seed, "num_workers": workers,
              "learning_rate": float(cfg.get("learning_rate", 1e-4)),
              "vision_learning_rate": float(cfg.get("cte_vision_learning_rate", 1e-5)),
              "weight_decay": float(cfg.get("weight_decay", 1e-4)), "gradient_clip": gradient_clip,
              "effect_weight": effect_weight, "validation_every_epochs": evaluation_every,
              "prediction_gain_threshold": MIN_PREDICTION_GAIN, "scheduler": None,
              "sampler": "same_task_pairs_full_episode_no_replacement", "world_size": 1}
    resolved_data = OmegaConf.to_container(data_config, resolve=True)
    manifest = {
        "schema": model.schema, "reference_revision": REFERENCE_REVISION, "config": model_config.to_dict(),
        "tasks": tasks, "episode_splits": splits,
        "dataset_path": [str(Path(path).expanduser().resolve()) for path in data_config.dataset_dirs],
        "dataset_stats_sha256": sha256_file(stats_path), "semantic_task_map_sha256": sha256_file(semantic_path),
        "data_identity": {"episode_metadata_sha256": canonical_sha256(rows), "data_config_sha256": canonical_sha256(resolved_data)},
        "source_sha256": {key: sha256_file(path) for key, path in source_paths.items()},
        "recipe": recipe, "normalization": "FastWAM dataset_stats joint14; RGB independently resized to [-1,1]",
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
        "promotable": bool(not smoke_steps and model_config.formal_architecture and model_config.vision_pretrained),
    }
    if data_scope is not None:
        manifest["data_scope"] = data_scope
    start_epoch, step = (0, 0) if previous is None else restore_ego_training_checkpoint(previous, model, optimizer, manifest, loader_generator)
    if start_epoch >= epochs:
        raise ValueError(f"Resume checkpoint already completed epoch {start_epoch}; requested num_epochs={epochs}")
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "episode_splits.json", splits)
    OmegaConf.save(cfg, output / "resolved_config.yaml")
    print(json.dumps({"train_episodes": len(train_dataset), "validation_episodes": len(validation_dataset),
                      "semantic_tasks": len(task_ids), "start_epoch": start_epoch, "promotable": manifest["promotable"]}), flush=True)

    metrics_path = output / "metrics.jsonl"
    with metrics_path.open("a") as metrics_stream:
        for epoch in range(start_epoch, epochs):
            sampler.set_epoch(epoch)
            model.train()
            seen: set[str] = set()
            last_end = time.perf_counter()
            for batch in train_loader:
                batch_ready = time.perf_counter()
                identities = list(map(str, batch["episode_id"]))
                if len(set(identities)) != len(identities) or seen.intersection(identities):
                    raise RuntimeError("The no-replacement episode sampler repeated an episode")
                seen.update(identities)
                images = batch["images"].to(device, non_blocking=True)
                actions = batch["executed_actions"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                labels = torch.tensor([task_ids[str(task)] for task in batch["task_id"]], device=device)
                outputs = model(images, actions, valid)
                losses = zeva_ego_cte_loss(outputs, actions, valid, labels, effect_weight=effect_weight)
                if not bool(torch.isfinite(losses["total"])):
                    raise FloatingPointError("Non-finite CTE training loss")
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                if float(norm) == 0:
                    raise RuntimeError("CTE backward produced zero gradients")
                optimizer.step()
                model.update_ema_target()
                step += 1
                if step % log_every == 0 or smoke_steps:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    metrics = {"epoch": epoch + 1, "step": step, "grad_norm": float(norm),
                               "data_seconds": batch_ready - last_end,
                               "update_seconds": time.perf_counter() - batch_ready,
                               **{name: float(value.detach()) for name, value in losses.items()}}
                    line = json.dumps(metrics, allow_nan=False)
                    print(line, flush=True)
                    metrics_stream.write(line + "\n")
                    metrics_stream.flush()
                last_end = time.perf_counter()
                if smoke_steps and step >= smoke_steps:
                    _write_json(output / "SMOKE_COMPLETED.json", {"steps": step, "promotable": False, "checkpoint_written": False})
                    print("SMOKE_COMPLETED; no promotable checkpoint was written", flush=True)
                    return
            if seen != set(splits["train"]):
                raise RuntimeError("A training epoch did not cover every training episode exactly once")
            validation = None
            if (epoch + 1) % evaluation_every == 0 or epoch + 1 == epochs:
                validation = validate_ego_cte(model, validation_loader, device)
                validation["epoch"] = epoch + 1
                if not validation["finite"]:
                    raise FloatingPointError("Non-finite or empty held-out CTE validation")
            # A smoke budget can exceed one epoch. It must never emit formal checkpoints.
            if smoke_steps:
                continue
            path = output / f"cte_epoch_{epoch + 1:03d}.pt"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite an existing completed epoch: {path}")
            checkpoint = save_ego_cte_checkpoint(path, model, optimizer, epoch=epoch + 1, step=step,
                                                  manifest=manifest, validation=validation, loader_generator=loader_generator)
            result = {"completed_epoch": epoch + 1, "step": step, "validation": validation,
                      "ready_for_stage2": checkpoint["ready_for_stage2"], "checkpoint": str(path)}
            line = json.dumps(result, allow_nan=False)
            print(line, flush=True)
            metrics_stream.write(line + "\n")
            metrics_stream.flush()
        if smoke_steps:
            raise ValueError("Requested smoke steps exceed the available configured epochs; no checkpoint was written")
    _write_json(output / "COMPLETED.json", {"epoch": epochs, "step": step,
                                            "ready_for_stage2": checkpoint["ready_for_stage2"],
                                            "checkpoint": str(path)})


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
