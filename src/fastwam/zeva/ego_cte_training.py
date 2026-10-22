"""Episode training, validation and checkpoint contracts for the Zeva-Ego CTE.

The recipe follows Zeva ``scripts/robotwin/train_cte.py`` at
bae85ad753d3afe4f2e56762373d48ac0267e101.  Dataset normalization and action
coordinates remain FastWAM's joint14 contract; legacy window CTEs are separate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Sampler

from .ego_cte import EGO_CTE_SCHEMA, ZevaEgoCTE, ZevaEgoCTEConfig


REFERENCE_REVISION = "bae85ad753d3afe4f2e56762373d48ac0267e101"
MIN_PREDICTION_GAIN = 0.05
FORMAL_EPOCHS = 80
_ERROR_KEYS = ("action", "action_zero", "vision", "vision_persistence", "effect", "effect_zero")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def episode_metadata(episodes: Iterable[Any]) -> list[dict[str, Any]]:
    result = []
    for episode in episodes:
        row = asdict(episode) if is_dataclass(episode) else dict(episode)
        if not row.get("episode_id") or not row.get("task_id"):
            raise ValueError("Episode metadata must include nonempty canonical episode_id and task_id")
        result.append(row)
    ids = [row["episode_id"] for row in result]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source episode IDs would invalidate no-replacement training")
    return result


def build_episode_splits(episodes: Iterable[Any], *, seed: int = 1000,
                         validation_fraction: float = 0.05,
                         split_path: str | Path | None = None) -> dict[str, list[str]]:
    """Deterministic semantic-task split, before any RGB is decoded.

    A supplied JSON uses ``{"train": [episode_ids], "validation": [...]}``.
    All source episodes must occur exactly once, and each task needs at least
    two training episodes for contrastive positives plus one held-out episode.
    """
    rows = episode_metadata(episodes)
    if not rows or not 0 < validation_fraction < 1:
        raise ValueError("Nonempty episode metadata and validation_fraction in (0,1) are required")
    by_task: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(str(row["episode_id"]))
    sparse = {task: len(ids) for task, ids in by_task.items() if len(ids) < 3}
    if sparse:
        raise ValueError(f"Each semantic task needs >=3 episodes (>=2 train, >=1 validation); insufficient: {sparse}")
    if split_path is None:
        generator = torch.Generator().manual_seed(int(seed))
        splits: dict[str, list[str]] = {"train": [], "validation": []}
        for task in sorted(by_task):
            ids = sorted(by_task[task])
            ids = [ids[i] for i in torch.randperm(len(ids), generator=generator).tolist()]
            count = min(len(ids) - 2, max(1, math.ceil(len(ids) * validation_fraction)))
            splits["validation"].extend(ids[:count])
            splits["train"].extend(ids[count:])
    else:
        splits = json.loads(Path(split_path).read_text())
        if not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
            raise ValueError("Episode split JSON must contain exactly train and validation lists")
        if any(not isinstance(splits[key], list) or any(not isinstance(v, str) for v in splits[key]) for key in splits):
            raise ValueError("Episode split entries must be lists of canonical episode ID strings")
    for key in splits:
        if len(set(splits[key])) != len(splits[key]):
            raise ValueError(f"Duplicate episode IDs in {key} split")
    train, validation = set(splits["train"]), set(splits["validation"])
    if train & validation:
        raise ValueError("Train/validation episode overlap is forbidden")
    if train | validation != {str(row["episode_id"]) for row in rows}:
        raise ValueError("Train/validation splits must cover exactly the indexed source episodes")
    for task, ids in by_task.items():
        if len(train.intersection(ids)) < 2 or not validation.intersection(ids):
            raise ValueError(f"Task {task!r} needs >=2 training and >=1 held-out validation episodes")
    return {key: sorted(splits[key]) for key in ("train", "validation")}


class PairedEpochSampler(Sampler[int]):
    """Full episode coverage, adjacent same-task pairs, never replacement.

    Use an even batch size (reference: 8), so pairs do not cross batch borders.
    Odd task leftovers are appended and retained, just as in Zeva-Ego.
    """

    def __init__(self, task_ids: Iterable[str], seed: int = 1000) -> None:
        self.task_ids = tuple(map(str, task_ids))
        if not self.task_ids:
            raise ValueError("Cannot sample an empty episode dataset")
        self.seed, self.epoch = int(seed), 0

    def __len__(self) -> int:
        return len(self.task_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, task in enumerate(self.task_ids):
            groups[task].append(index)
        pairs, leftovers = [], []
        for task in sorted(groups):
            group = groups[task]
            order = [group[i] for i in torch.randperm(len(group), generator=generator).tolist()]
            pairs.extend(order[i:i + 2] for i in range(0, len(order) - 1, 2))
            if len(order) % 2:
                leftovers.append(order[-1])
        result = [index for i in torch.randperm(len(pairs), generator=generator).tolist() for index in pairs[i]]
        return iter(result + leftovers)


def build_ego_cte_optimizer(model: ZevaEgoCTE, *, learning_rate: float = 1e-4,
                             vision_learning_rate: float = 1e-5,
                             weight_decay: float = 1e-4) -> torch.optim.AdamW:
    if min(learning_rate, vision_learning_rate) <= 0 or weight_decay < 0:
        raise ValueError("Optimizer learning rates must be positive and weight decay nonnegative")
    vision = [parameter for parameter in model.vision_encoder.parameters() if parameter.requires_grad]
    vision_ids = {id(parameter) for parameter in vision}
    other = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in vision_ids]
    return torch.optim.AdamW([
        {"params": vision, "lr": vision_learning_rate, "name": "online_vision"},
        {"params": other, "lr": learning_rate, "name": "cte_heads_and_streams"},
    ], weight_decay=weight_decay)


def prediction_gain_metrics(errors: dict[str, float], transitions: int) -> dict[str, Any]:
    """Recompute the gate from measured MSEs, never from a claimed pass flag."""
    result: dict[str, Any] = {key: float(errors[key]) for key in _ERROR_KEYS}
    result["transitions"] = int(transitions)
    finite = transitions > 0 and all(math.isfinite(result[key]) and result[key] >= 0 for key in _ERROR_KEYS)
    for name, baseline in (("action", "action_zero"), ("vision", "vision_persistence"), ("effect", "effect_zero")):
        result[f"{name}_gain"] = 1.0 - result[name] / max(result[baseline], 1e-12)
    result["finite"] = finite
    result["stage1_gate"] = finite and all(result[f"{name}_gain"] >= MIN_PREDICTION_GAIN for name in ("action", "vision", "effect"))
    return result


@torch.no_grad()
def validate_ego_cte(model: ZevaEgoCTE, loader, device: str | torch.device) -> dict[str, Any]:
    """Held-out episode validation against zero actions/effects and persistence."""
    was_training = model.training
    model.eval()
    sums, count = dict.fromkeys(_ERROR_KEYS, 0.0), 0
    try:
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            actions = batch["executed_actions"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, dtype=torch.bool, non_blocking=True)
            outputs = model(images, actions, valid)
            # Select first: NaN or huge padding must never contaminate metrics.
            act, target = actions[valid], outputs["target_effect"][valid]
            rows = {
                "action": (outputs["pred_act"][valid] - act).square().mean((-1, -2)),
                "action_zero": act.square().mean((-1, -2)),
                "vision": (outputs["pred_vis"][valid] - outputs["target_vis"][valid]).square().mean(-1),
                "vision_persistence": target.square().mean(-1),
                "effect": (outputs["pred_effect"][valid] - target).square().mean(-1),
                "effect_zero": target.square().mean(-1),
            }
            count += int(valid.sum())
            for name, value in rows.items():
                sums[name] += float(value.sum())
    finally:
        model.train(was_training)
    return prediction_gain_metrics({key: value / max(count, 1) for key, value in sums.items()}, count)


def capture_rng_state(loader_generator: torch.Generator | None = None) -> dict[str, Any]:
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": np.random.get_state(),
            "loader_generator": None if loader_generator is None else loader_generator.get_state()}


def restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator | None = None) -> None:
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same CUDA RNG device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if loader_generator is not None:
        if state.get("loader_generator") is None:
            raise ValueError("Checkpoint has no DataLoader RNG state for exact resume")
        loader_generator.set_state(state["loader_generator"].cpu())


def _checkpoint_config(payload: dict[str, Any]) -> ZevaEgoCTEConfig:
    if payload.get("schema") != EGO_CTE_SCHEMA:
        raise ValueError("Not an iclwam_zeva_ego_cte_v1 checkpoint; legacy CTEs require their original loader")
    config = payload.get("model_config", payload.get("config"))
    if not isinstance(config, dict):
        raise ValueError("Zeva-Ego checkpoint must declare its complete model config")
    if payload.get("config", config) != config or payload.get("manifest", {}).get("config", config) != config:
        raise ValueError("Conflicting checkpoint model/config/manifest identities")
    unknown = set(config) - set(ZevaEgoCTEConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unrecognized Zeva-Ego model configuration fields: {sorted(unknown)}")
    if set(config) != set(ZevaEgoCTEConfig.__dataclass_fields__):
        raise ValueError("Zeva-Ego checkpoint model configuration must be complete")
    return ZevaEgoCTEConfig(**config)


def require_ego_checkpoint_ready(payload: dict[str, Any]) -> None:
    """Prevent smoke, CPU surrogate or unvalidated Stage1 weights from caching."""
    cfg = _checkpoint_config(payload)
    manifest = payload.get("manifest", {})
    if (payload.get("cte_input_type") != "rgb_multiview"
            or payload.get("camera_keys") != ["cam_high", "cam_left_wrist", "cam_right_wrist"]
            or payload.get("memory_semantics") != "predicted_effect_trace"):
        raise ValueError("Formal CTE checkpoint must declare multiview camera order and predicted-effect semantics")
    if not cfg.formal_architecture or not cfg.vision_pretrained or not manifest.get("promotable", False):
        raise ValueError("CTE checkpoint is non-promotable: formal pretrained ResNet18/Mamba training is required")
    if (cfg.action_dim, cfg.executed_horizon, cfg.num_views, cfg.image_size) != (14, 24, 3, 224):
        raise ValueError("Formal FastWAM CTE requires joint14, H24 and three independent 224 RGB views")
    epoch = int(payload.get("epoch", 0))
    if epoch < FORMAL_EPOCHS:
        raise ValueError(f"Stage1 checkpoint is not ready: complete at least {FORMAL_EPOCHS} full epochs")
    required = ("episode_splits", "dataset_stats_sha256", "semantic_task_map_sha256", "data_identity", "source_sha256")
    if any(not manifest.get(key) for key in required):
        raise ValueError("Stage1 checkpoint lacks dataset, split, normalization or source identities")
    splits = manifest["episode_splits"]
    if not isinstance(splits, dict) or set(splits) != {"train", "validation"} or not all(splits.values()):
        raise ValueError("Stage1 checkpoint needs nonempty train/validation episode lists")
    if set(splits["train"]) & set(splits["validation"]):
        raise ValueError("Checkpoint records train/validation episode overlap")
    if payload.get("episode_splits") != splits:
        raise ValueError("Checkpoint split identity mismatch")
    for key in ("dataset_stats_sha256", "semantic_task_map_sha256"):
        if payload.get(key) != manifest[key]:
            raise ValueError(f"Checkpoint {key} identity mismatch")
    validation = payload.get("validation")
    if not isinstance(validation, dict) or int(validation.get("epoch", -1)) != epoch:
        raise ValueError("Ready CTE checkpoints need held-out validation of this exact epoch")
    try:
        metrics = prediction_gain_metrics(validation, int(validation["transitions"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Incomplete CTE prediction-gain validation") from exc
    if not metrics["stage1_gate"]:
        gains = {key: metrics[key] for key in ("action_gain", "vision_gain", "effect_gain")}
        raise ValueError(f"Stage1 prediction gate failed; each held-out gain must be >= {MIN_PREDICTION_GAIN}: {gains}")


def save_ego_cte_checkpoint(path: str | Path, model: ZevaEgoCTE, optimizer,
                            *, epoch: int, step: int, manifest: dict[str, Any],
                            validation: dict[str, Any] | None,
                            loader_generator: torch.Generator | None = None) -> dict[str, Any]:
    path = Path(path)
    config = model.cfg.to_dict()
    if manifest.get("config") != config:
        raise ValueError("Checkpoint manifest/config mismatch")
    state = model.state_dict()
    payload: dict[str, Any] = {
        "schema": EGO_CTE_SCHEMA, "manifest": manifest, "config": config, "model_config": config,
        "model": state, "model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch), "step": int(step), "validation": validation,
        "rng": capture_rng_state(loader_generator), "cte_input_type": "rgb_multiview",
        "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "action_dim": model.cfg.action_dim, "image_channels": 3,
        "memory_semantics": "predicted_effect_trace",
    }
    for key in ("episode_splits", "dataset_stats_sha256", "semantic_task_map_sha256", "dataset_path"):
        payload[key] = manifest.get(key)
    try:
        require_ego_checkpoint_ready(payload)
        payload["ready_for_stage2"] = True
    except ValueError:
        payload["ready_for_stage2"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def load_ego_cte_checkpoint(path: str | Path, *, device: str | torch.device = "cpu",
                            map_location: Any = "cpu", require_ready: bool = True) -> tuple[ZevaEgoCTE, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = _checkpoint_config(payload)
    if require_ready:
        require_ego_checkpoint_ready(payload)
    # Strict restore includes all pretrained and EMA weights; no download on load.
    model = ZevaEgoCTE(replace(config, vision_pretrained=False))
    model.cfg = model.config = config
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def restore_ego_training_checkpoint(payload: dict[str, Any], model: ZevaEgoCTE, optimizer,
                                     manifest: dict[str, Any],
                                     loader_generator: torch.Generator | None = None) -> tuple[int, int]:
    """Resume only completed epochs with identical data, recipe and code."""
    if _checkpoint_config(payload) != model.cfg:
        raise ValueError("Exact resume model config mismatch")
    previous = payload.get("manifest", {})
    identity_keys = ("config", "tasks", "episode_splits", "dataset_stats_sha256", "semantic_task_map_sha256",
                     "data_identity", "source_sha256", "recipe", "promotable")
    for key in identity_keys:
        if key not in previous or previous[key] != manifest.get(key):
            raise ValueError(f"Exact resume identity mismatch: {key}")
    if not previous["promotable"]:
        raise ValueError("Smoke/test checkpoints cannot resume formal CTE training")
    if int(payload.get("epoch", 0)) < 1 or int(payload.get("step", 0)) < 1:
        raise ValueError("Exact resume requires a completed epoch checkpoint")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore_rng_state(payload["rng"], loader_generator)
    return int(payload["epoch"]), int(payload["step"])
