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
EXPLORATORY_EPOCHS = 40
EXPLORATORY_STEPS = 26160
CTE_DATA_SCOPE_SCHEMA = "iclwam_zeva_ego_data_scope_v1"
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


def _reference_partition_population(source: list[dict[str, Any]], condition_map: dict[str, str],
                                    selected_conditions: Iterable[str]) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    population = [{**row, "condition": condition_map[row["episode_id"]]}
                  for row in source if condition_map[row["episode_id"]] in selected_conditions]
    keys = sorted({(row["condition"], row["task_id"]) for row in population})
    if not keys:
        raise ValueError("Configured CTE conditions select no source partitions")
    partitions = {key: index for index, key in enumerate(keys)}
    positions: dict[tuple[str, str], int] = defaultdict(int)
    for row in population:
        key = (row["condition"], row["task_id"])
        row["partition_id"] = partitions[key]
        row["episode_position"] = positions[key]
        positions[key] += 1
    population.sort(key=lambda row: (row["partition_id"], row["episode_position"]))
    return keys, population


def select_ego_episode_scope(dataset: Any, *, task_subset_path: str | Path,
                             conditions_path: str | Path,
                             conditions: Iterable[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select an explicit task/condition scope without decoding any image.

    The adapter's split population includes short positive-length episodes
    and all configured source tasks. The CTE task subset is applied after
    holdout, as in the reference adapter -> Episodes construction sequence.
    Partition positions are the source-table order within condition/task.
    """
    if getattr(dataset, "empty_source_episode_ids", ()):
        raise ValueError("Reference CTE cannot infer partition positions across empty source episodes; provide a source with its real positive episode ordering preserved")
    task_subset_path, conditions_path = Path(task_subset_path), Path(conditions_path)
    if not task_subset_path.is_file() or not conditions_path.is_file():
        raise FileNotFoundError("Reference CTE needs explicit task-subset and real episode-conditions JSON files")
    subset = json.loads(task_subset_path.read_text())
    tasks = subset.get("task_names") if isinstance(subset, dict) else None
    if (not isinstance(tasks, list) or not tasks or any(not isinstance(task, str) or not task for task in tasks)
            or len(set(tasks)) != len(tasks)):
        raise ValueError("CTE task subset must contain a nonempty, unique ordered task_names list")
    selected_conditions = list(conditions)
    if (not selected_conditions or any(not isinstance(value, str) or not value for value in selected_conditions)
            or len(set(selected_conditions)) != len(selected_conditions)):
        raise ValueError("CTE conditions must be a nonempty list of unique real condition names")
    condition_map = json.loads(conditions_path.read_text())
    if (not isinstance(condition_map, dict)
            or any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in condition_map.items())):
        raise ValueError("CTE episode conditions must map canonical episode IDs to nonempty real condition names")
    source = episode_metadata(dataset.source_episodes)
    missing = {row["episode_id"] for row in source} - set(condition_map)
    if missing:
        raise ValueError(f"CTE conditions must cover the whole source split population; missing {len(missing)} episodes")
    keys, population = _reference_partition_population(source, condition_map, selected_conditions)
    selected_ids = {row["episode_id"] for row in population if row["task_id"] in tasks}
    selected = tuple(episode for episode in dataset.episodes if episode.episode_id in selected_ids)
    if not selected:
        raise ValueError("The frozen task/condition subset contains no complete CTE episodes")
    dataset.episodes = selected
    scope = {
        "schema": CTE_DATA_SCOPE_SCHEMA, "split_policy": "zeva_ego",
        "task_names": tasks, "conditions": selected_conditions,
        "task_subset_sha256": sha256_file(task_subset_path),
        "episode_conditions_sha256": sha256_file(conditions_path),
        "episode_conditions": {row["episode_id"]: condition_map[row["episode_id"]] for row in source},
        "source_episode_metadata_sha256": canonical_sha256(source),
        "partition_population_sha256": canonical_sha256(population),
        "partition_population": population,
        "eligible_episode_ids": [episode.episode_id for episode in selected],
        "partition_keys": [list(key) for key in keys],
        "partition_position_semantics": "source_table_order_within_condition_task",
    }
    return scope, population


def _legacy_episode_splits(episodes: Iterable[Any], *, seed: int = 1000,
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


def build_episode_splits(episodes: Iterable[Any], *, seed: int = 1000,
                         validation_fraction: float = 0.05,
                         split_path: str | Path | None = None,
                         split_policy: str = "zeva_ego",
                         partition_rows: Iterable[dict[str, Any]] | None = None) -> dict[str, list[str]]:
    """Reference SHA/round holdout within each condition/task partition.

    The eligible CTE episodes are selected only after splitting the full
    positive-length source population. A supplied split cannot silently turn
    a different holdout into a reference run; legacy partitions are explicit.
    """
    if split_policy == "legacy_semantic_task":
        return _legacy_episode_splits(episodes, seed=seed, validation_fraction=validation_fraction,
                                      split_path=split_path)
    if split_policy != "zeva_ego":
        raise ValueError("cte_split_policy must be zeva_ego or legacy_semantic_task")
    rows = episode_metadata(episodes)
    if not rows or not 0 < validation_fraction < 1:
        raise ValueError("Nonempty episodes and validation_fraction in (0,1) are required")
    if partition_rows is None:
        raise ValueError("Reference CTE split requires explicit condition/task source partition metadata")
    population = episode_metadata(partition_rows)
    target_ids = {row["episode_id"] for row in rows}
    if not target_ids <= {row["episode_id"] for row in population}:
        raise ValueError("CTE source partition population omits an eligible episode")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_positions: set[tuple[int, int]] = set()
    for row in population:
        partition, position = row.get("partition_id"), row.get("episode_position")
        if type(partition) is not int or type(position) is not int or min(partition, position) < 0:
            raise ValueError("Reference partition IDs and episode positions must be nonnegative integers")
        if (partition, position) in seen_positions:
            raise ValueError("Duplicate reference episode position within a source partition")
        seen_positions.add((partition, position))
        groups[partition].append(row)
    held_out: set[str] = set()
    ordered = []
    for partition in sorted(groups):
        candidates = sorted(groups[partition], key=lambda row: row["episode_position"])
        ordered.extend(candidates)
        if len(candidates) < 2:
            continue
        count = min(len(candidates) - 1, max(1, round(len(candidates) * validation_fraction)))
        scores = [int.from_bytes(hashlib.sha256(
            f"{int(seed)}:{partition}:{row['episode_position']}".encode()).digest()[:8], "little")
                  for row in candidates]
        selected = np.argsort(np.asarray(scores, dtype=np.uint64))[:count]
        held_out.update(candidates[index]["episode_id"] for index in selected)
    result = {
        "train": [row["episode_id"] for row in ordered if row["episode_id"] in target_ids and row["episode_id"] not in held_out],
        "validation": [row["episode_id"] for row in ordered if row["episode_id"] in target_ids and row["episode_id"] in held_out],
    }
    if not result["train"] or not result["validation"]:
        raise ValueError("The reference task/condition subset needs nonempty train and validation CTE episodes")
    if split_path is not None:
        supplied = json.loads(Path(split_path).read_text())
        if (not isinstance(supplied, dict) or set(supplied) != set(result)
                or any(not isinstance(supplied[key], list) or len(supplied[key]) != len(set(supplied[key]))
                       or set(supplied[key]) != set(result[key]) for key in result)):
            raise ValueError("Explicit episode split differs from the reference SHA/round partition split")
    return result


def _require_reference_data_scope(manifest: dict[str, Any], *, executed_horizon: int = 24) -> dict[str, Any]:
    """Recompute the recorded partition/split contract, not just its label."""
    scope, recipe = manifest.get("data_scope"), manifest.get("recipe", {})
    if (not isinstance(scope, dict) or scope.get("schema") != CTE_DATA_SCOPE_SCHEMA
            or scope.get("split_policy") != "zeva_ego" or recipe.get("split_policy") != "zeva_ego"):
        raise ValueError("Reference CTE recipe requires a versioned condition/task source scope; use an explicit legacy route for old artifacts")
    tasks, conditions = scope.get("task_names"), scope.get("conditions")
    if (not isinstance(tasks, list) or not tasks or any(not isinstance(value, str) or not value for value in tasks)
            or len(set(tasks)) != len(tasks)
            or not isinstance(conditions, list) or not conditions
            or any(not isinstance(value, str) or not value for value in conditions) or len(set(conditions)) != len(conditions)
            or manifest.get("tasks") != tasks):
        raise ValueError("Reference CTE task/condition scope is inconsistent")
    condition_map = scope.get("episode_conditions")
    if (not isinstance(condition_map, dict) or not condition_map
            or any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
                   for key, value in condition_map.items())
            or scope.get("partition_position_semantics") != "source_table_order_within_condition_task"):
        raise ValueError("Reference CTE needs its real source condition mapping and partition-position contract")
    for key in ("task_subset_sha256", "episode_conditions_sha256", "source_episode_metadata_sha256", "partition_population_sha256"):
        digest = scope.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Reference CTE scope requires a SHA256 identity for {key}")
    population = scope.get("partition_population")
    if not isinstance(population, list) or not population or canonical_sha256(population) != scope["partition_population_sha256"]:
        raise ValueError("Reference CTE source partition population fingerprint mismatch")
    population = episode_metadata(population)
    if any(condition_map.get(row["episode_id"]) != row.get("condition") for row in population):
        raise ValueError("Reference CTE partition population disagrees with its source condition mapping")
    keys = sorted({(row.get("condition"), row["task_id"]) for row in population})
    if scope.get("partition_keys") != [list(key) for key in keys] or any(key[0] not in conditions for key in keys):
        raise ValueError("Reference CTE condition/task partition keys disagree with source records")
    positions: dict[tuple[str, str], int] = defaultdict(int)
    for row in population:
        key = (row["condition"], row["task_id"])
        if row.get("partition_id") != keys.index(key) or row.get("episode_position") != positions[key]:
            raise ValueError("Reference CTE partition IDs or source episode positions are inconsistent")
        positions[key] += 1
        if type(row.get("episode_length")) is not int or row["episode_length"] < 1:
            raise ValueError("Reference CTE partition population needs positive source episode lengths")
    eligible = scope.get("eligible_episode_ids")
    expected = {row["episode_id"] for row in population
                if row["task_id"] in tasks and row["episode_length"] > executed_horizon}
    if (not isinstance(eligible, list) or len(set(eligible)) != len(eligible) or set(eligible) != expected):
        raise ValueError("Reference CTE eligible episodes disagree with task subset or execution horizon")
    by_id = {row["episode_id"]: {key: value for key, value in row.items()
                                 if key not in {"condition", "partition_id", "episode_position"}} for row in population}
    selected_rows = [by_id[identity] for identity in eligible]
    if canonical_sha256(selected_rows) != manifest.get("data_identity", {}).get("episode_metadata_sha256"):
        raise ValueError("Reference CTE selected episode metadata fingerprint mismatch")
    if "split_seed" not in recipe or "validation_fraction" not in recipe:
        raise ValueError("Reference CTE recipe must record split_seed and validation_fraction")
    splits = build_episode_splits(selected_rows, seed=int(recipe["split_seed"]),
                                  validation_fraction=float(recipe["validation_fraction"]),
                                  split_policy="zeva_ego", partition_rows=population)
    if splits != manifest.get("episode_splits"):
        raise ValueError("Reference CTE checkpoint split differs from its SHA/round source population")
    return scope


def verify_ego_episode_scope(dataset: Any, manifest: dict[str, Any]) -> None:
    """Bind reference scope to the actual full source before feature export."""
    if getattr(dataset, "empty_source_episode_ids", ()):
        raise ValueError("Reference CTE cannot verify partition positions across empty source episodes")
    scope = _require_reference_data_scope(manifest, executed_horizon=dataset.executed_horizon)
    source = episode_metadata(dataset.source_episodes)
    if canonical_sha256(source) != scope["source_episode_metadata_sha256"]:
        raise ValueError("Reference CTE full source episode population differs from Stage1 training")
    if set(scope["episode_conditions"]) != {row["episode_id"] for row in source}:
        raise ValueError("Reference CTE condition mapping does not cover exactly the actual source population")
    _, population = _reference_partition_population(source, scope["episode_conditions"], scope["conditions"])
    if population != scope["partition_population"]:
        raise ValueError("Reference CTE saved partition population differs from actual source metadata/order")
    if canonical_sha256(episode_metadata(dataset.episodes)) != manifest.get("data_identity", {}).get("episode_metadata_sha256"):
        raise ValueError("Reference CTE selected source episode metadata differs from Stage1 training")


class PairedEpochSampler(Sampler[int]):
    """Full episode coverage, adjacent same-task pairs, never replacement.

    Use an even batch size (reference: 8), so pairs do not cross batch borders.
    Odd task leftovers are appended and retained, just as in Zeva-Ego.
    """

    def __init__(self, task_ids: Iterable[str], seed: int = 1000,
                 task_order: Iterable[str] | None = None) -> None:
        self.task_ids = tuple(map(str, task_ids))
        if not self.task_ids:
            raise ValueError("Cannot sample an empty episode dataset")
        self.task_order = tuple(sorted(set(self.task_ids))) if task_order is None else tuple(map(str, task_order))
        if len(set(self.task_order)) != len(self.task_order) or not set(self.task_ids) <= set(self.task_order):
            raise ValueError("Sampler task order must uniquely contain every observed task")
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
        for task in self.task_order:
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


def require_ego_checkpoint_ready(payload: dict[str, Any], *, exploratory_epoch40: bool = False,
                                  require_reference_recipe: bool = True) -> None:
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
    epoch_value = payload.get("epoch", 0)
    epoch = int(epoch_value)
    expected_epoch = EXPLORATORY_EPOCHS if exploratory_epoch40 else FORMAL_EPOCHS
    wrong_epoch = epoch_value != expected_epoch if require_reference_recipe or exploratory_epoch40 else epoch < FORMAL_EPOCHS
    if wrong_epoch or (exploratory_epoch40 and payload.get("step") != EXPLORATORY_STEPS):
        route = f"explicit epoch40/step{EXPLORATORY_STEPS} experiment" if exploratory_epoch40 else "exactly 80 full epochs"
        raise ValueError(f"Stage1 checkpoint does not satisfy the reference endpoint: {route}")
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
    if require_reference_recipe:
        _require_reference_data_scope(manifest, executed_horizon=cfg.executed_horizon)
    for key in ("dataset_stats_sha256", "semantic_task_map_sha256"):
        if payload.get(key) != manifest[key]:
            raise ValueError(f"Checkpoint {key} identity mismatch")
    validation = payload.get("validation")
    if not isinstance(validation, dict) or int(validation.get("epoch", -1)) != epoch:
        raise ValueError("Ready CTE checkpoints need held-out validation of this exact epoch")
    if (require_reference_recipe or exploratory_epoch40) and not validation.get("stage1_gate"):
        raise ValueError("Reference CTE checkpoint must record a passed Stage1 validation gate")
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
        require_ego_checkpoint_ready(payload, require_reference_recipe=manifest.get("recipe", {}).get("split_policy") != "legacy_semantic_task")
        payload["ready_for_stage2"] = True
    except ValueError:
        payload["ready_for_stage2"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def load_ego_cte_checkpoint(path: str | Path, *, device: str | torch.device = "cpu",
                            map_location: Any = "cpu", require_ready: bool = True,
                            exploratory_epoch40: bool = False,
                            require_reference_recipe: bool = True) -> tuple[ZevaEgoCTE, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = _checkpoint_config(payload)
    if exploratory_epoch40 and not require_ready:
        raise ValueError("The explicit epoch40 route cannot bypass validation")
    if require_ready:
        require_ego_checkpoint_ready(payload, exploratory_epoch40=exploratory_epoch40,
                                      require_reference_recipe=require_reference_recipe)
    # Strict restore includes all pretrained and EMA weights; no download on load.
    model = ZevaEgoCTE(replace(config, vision_pretrained=False))
    model.cfg = model.config = config
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    if require_ready:
        model.requires_grad_(False)
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
    if "data_scope" in previous or "data_scope" in manifest:
        identity_keys += ("data_scope",)
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
