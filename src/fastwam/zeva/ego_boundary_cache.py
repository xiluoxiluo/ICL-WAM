"""Versioned causal prediction cache for the RGB, replan-cadence Zeva-Ego CTE."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .cross_attempt_cache import file_sha256, validate_episode_splits
from .ego_episode_data import EGO_CAMERA_KEYS, EGO_MEMORY_SEMANTICS, RoboTwinEgoEpisodeDataset


EGO_BOUNDARY_CACHE_FORMAT = "zeva_ego_boundary_cache_v1"


@dataclass(frozen=True)
class EgoBoundaryManifest:
    cte_checkpoint_sha256: str
    dataset_stats_sha256: str
    semantic_task_map_sha256: str
    dataset_path: str
    phase_dim: int = 256
    effect_dim: int = 256
    global_dim: int = 256
    retrieval_dim: int = 128
    executed_horizon: int = 24
    action_dim: int = 14
    policy_horizon: int = 32
    image_size: int = 224
    camera_keys: tuple[str, ...] | list[str] = EGO_CAMERA_KEYS
    schema_version: str = EGO_BOUNDARY_CACHE_FORMAT
    history_semantics: str = "full_episode_prefix"
    memory_semantics: str = EGO_MEMORY_SEMANTICS
    effect_source: str = "current_prediction"
    query_step_unit: str = "raw_action_step"
    feature_dtype: str = "float32"
    cte_input_type: str = "rgb_multiview"
    cte_variant: str = "zeva_ego"
    retrieval_space: str = "cte_task_head_raw_v1"
    encoding_path: str = "recurrent_step_v1"
    num_records: int = 0
    num_episodes: int = 0
    index_sha256: str = ""
    episode_metadata_sha256: str = ""
    episode_splits_sha256: str = ""
    shard_sha256: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if (self.schema_version != EGO_BOUNDARY_CACHE_FORMAT or self.history_semantics != "full_episode_prefix"
                or self.memory_semantics != EGO_MEMORY_SEMANTICS or self.effect_source != "current_prediction"
                or self.query_step_unit != "raw_action_step" or self.feature_dtype != "float32"
                or self.cte_input_type != "rgb_multiview" or self.cte_variant != "zeva_ego"
                or self.retrieval_space != "cte_task_head_raw_v1" or self.encoding_path != "recurrent_step_v1"):
            raise ValueError("incompatible Zeva-Ego boundary cache semantics")
        if (self.executed_horizon, self.action_dim, self.policy_horizon, self.image_size) != (24, 14, 32, 224):
            raise ValueError("RoboTwin Zeva-Ego cache requires H24/joint14, policy H32, RGB224")
        if tuple(self.camera_keys) != EGO_CAMERA_KEYS:
            raise ValueError("Zeva-Ego boundary cache camera order mismatch")
        if min(self.phase_dim, self.effect_dim, self.global_dim, self.retrieval_dim) < 1:
            raise ValueError("invalid CTE feature dimensions")
        if not all((self.cte_checkpoint_sha256, self.dataset_stats_sha256, self.semantic_task_map_sha256)):
            raise ValueError("CTE, normalization and semantic-task identities are required")


class EgoBoundaryCache:
    """Verified reader; effect is explicitly the prediction made at this row."""
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = EgoBoundaryManifest(**json.loads((self.root / "manifest.json").read_text()))
        self.manifest.validate()
        for name, digest in (("episode_index.json", self.manifest.index_sha256),
                             ("episodes.json", self.manifest.episode_metadata_sha256),
                             ("episode_splits.json", self.manifest.episode_splits_sha256)):
            if not digest or file_sha256(self.root / name) != digest:
                raise ValueError(f"boundary cache metadata checksum mismatch: {name}")
        self.rows = json.loads((self.root / "episode_index.json").read_text())
        self.episodes = json.loads((self.root / "episodes.json").read_text())
        self.splits = validate_episode_splits(json.loads((self.root / "episode_splits.json").read_text()), set(self.episodes))
        if len(self.rows) != self.manifest.num_records or len(self.episodes) != self.manifest.num_episodes:
            raise ValueError("boundary cache record/episode count mismatch")
        self._verified: set[str] = set()
        self._shard_name = None
        self._tensors = None
        if set(self.episodes) != set().union(*(set(ids) for ids in self.splits.values())):
            raise ValueError("boundary cache split metadata must cover every episode")
        episode_split = {identity: name for name, ids in self.splits.items() for identity in ids}
        seen = set()
        for row in self.rows:
            identity = str(row["episode_id"]), int(row["raw_step"])
            if (identity in seen or identity[0] not in self.episodes or identity[1] < 0
                    or identity[1] % self.manifest.executed_horizon
                    or row.get("record_type") != "phase_query"
                    or row.get("memory_semantics") != EGO_MEMORY_SEMANTICS
                    or int(row["start_raw_step"]) != identity[1] or int(row["end_raw_step"]) != identity[1]
                    or int(row["available_at_raw_step"]) != identity[1]
                    or not bool(row["decision_valid"])):
                raise ValueError("invalid/duplicate causal boundary record")
            episode = self.episodes[identity[0]]
            split = episode_split[identity[0]]
            if (row["task_id"] != episode["task_id"] or row.get("split") != split or episode.get("split") != split
                    or identity[1] >= int(episode["episode_length"]) - 1
                    or int(row["window_index"]) != int(episode["dataset_start"]) + identity[1]
                    or bool(row["policy_valid"]) != (identity[1] + self.manifest.policy_horizon < int(episode["episode_length"]))
                    or int(row["offset"]) < 0 or row["shard"] != episode["shard"]
                    or Path(row["shard"]).name != row["shard"]):
                raise ValueError("boundary row differs from its episode/split/window provenance")
            seen.add(identity)

    def get(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        name = row["shard"]
        if name not in self._verified:
            expected = self.manifest.shard_sha256.get(name)
            if not expected or file_sha256(self.root / name) != expected:
                raise ValueError(f"boundary cache shard checksum mismatch: {name}")
            self._verified.add(name)
        if name != self._shard_name:
            self._tensors = load_file(str(self.root / name), device="cpu")
            self._shard_name = name
        offset = int(row["offset"])
        if not 0 <= offset < self._tensors["valid"].shape[0]:
            raise ValueError("boundary cache shard offset is out of range")
        value = dict(row)
        for field, dimension in (("phase_pre", self.manifest.phase_dim), ("effect", self.manifest.effect_dim),
                                 ("global", self.manifest.global_dim), ("retrieval", self.manifest.retrieval_dim)):
            tensor = self._tensors[field][offset]
            if tensor.dtype != torch.float32 or tensor.shape != (dimension,) or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"invalid boundary cache feature: {field}")
            value[field] = tensor.clone()
        if self._tensors["valid"].dtype != torch.bool or not bool(self._tensors["valid"][offset]):
            raise ValueError("boundary predictions must have valid observed inputs")
        value["phase"] = value["phase_pre"]
        value["current_effect"] = value["effect"]
        return value

    def __len__(self) -> int:
        return len(self.rows)


@torch.no_grad()
def build_ego_boundary_cache(dataset: RoboTwinEgoEpisodeDataset, cte: torch.nn.Module,
                             output: str | Path, *, identity: dict[str, Any],
                             episode_splits: dict[str, list[str]]) -> dict[str, Any]:
    """Write one shard per episode; only current/past RGB/actions reach CTE.

    ``global`` and ``retrieval`` traces are retained for task-memory building.
    The split file is copied verbatim after validation; downstream task memory
    and PIM must select only its train episode IDs.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite boundary cache: {output}")
    episodes_by_id = {episode.episode_id: episode for episode in dataset.episodes}
    splits = validate_episode_splits(episode_splits, set(episodes_by_id))
    if set(episodes_by_id) != set().union(*(set(ids) for ids in splits.values())):
        raise ValueError("boundary cache splits must cover every encoded episode")
    temporary = output.with_name(output.name + f".building-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        cte.eval()
        episode_split = {identity: name for name, ids in splits.items() for identity in ids}
        parameter = next(cte.parameters())
        rows, metadata, hashes = [], {}, {}
        dimensions = None
        for index, episode in enumerate(dataset.episodes):
            sample = dataset[index]
            images = sample["images"].unsqueeze(0).to(device=parameter.device, dtype=parameter.dtype)
            actions = sample["executed_actions"].unsqueeze(0).to(device=parameter.device, dtype=parameter.dtype)
            if not bool(sample["valid"].all()):
                raise ValueError("cache export accepts only observed complete execution prefixes")
            # Match reference export and deployment exactly. No batched Mamba
            # kernel or EMA target encoder participates in feature extraction.
            state, phases, predictions = None, [], []
            for boundary in range(images.shape[1]):
                previous = None if boundary == 0 else actions[:, boundary - 1]
                phase, predicted_effect, state = cte.step(images[:, boundary], previous, state)
                phases.append(phase)
                predictions.append(predicted_effect)
            phase = torch.stack(phases, dim=1)[0]
            effect = torch.stack(predictions, dim=1)[0]
            retrieval = cte.task_head(phase)
            features = {"phase_pre": phase, "effect": effect, "global": phase, "retrieval": retrieval}
            if any(value.ndim != 2 or value.shape[0] != images.shape[1] for value in features.values()):
                raise ValueError("CTE must return a feature for every observed boundary, including the last")
            shape = tuple(features[key].shape[-1] for key in ("phase_pre", "effect", "global", "retrieval"))
            if dimensions is not None and dimensions != shape:
                raise ValueError("CTE output dimensions changed between episodes")
            dimensions = shape
            selected = sample["decision_valid"].nonzero().flatten()
            shard = {name: value[selected.to(value.device)].detach().float().cpu().contiguous() for name, value in features.items()}
            shard["valid"] = torch.ones(selected.numel(), dtype=torch.bool)
            if not all(bool(torch.isfinite(value).all()) for value in shard.values()):
                raise ValueError("CTE produced non-finite cache features")
            name = f"episode-{index:07d}.safetensors"
            save_file(shard, str(temporary / name))
            hashes[name] = file_sha256(temporary / name)
            split = episode_split[episode.episode_id]
            metadata[episode.episode_id] = {"task_id": episode.task_id, "episode_index": episode.episode_index,
                "dataset_start": episode.dataset_start, "episode_length": episode.episode_length,
                "split": split, "shard": name, "num_predictions": selected.numel(),
                "trailing_action_count": sample["trailing_action_count"]}
            for offset, boundary in enumerate(selected.tolist()):
                raw = int(sample["raw_steps"][boundary])
                rows.append({"episode_id": episode.episode_id, "task_id": episode.task_id, "attempt_id": 0,
                    "record_type": "phase_query", "memory_semantics": EGO_MEMORY_SEMANTICS,
                    "raw_step": raw, "start_raw_step": raw, "end_raw_step": raw, "available_at_raw_step": raw,
                    "window_index": int(sample["window_indices"][boundary]), "transition_index": boundary,
                    "episode_step": raw, "policy_valid": bool(sample["policy_valid"][boundary]),
                    "decision_valid": True, "split": split, "shard": name, "offset": offset})
        for name, value in (("episode_index.json", rows), ("episodes.json", metadata), ("episode_splits.json", splits)):
            (temporary / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        dp, de, dg, dr = dimensions
        manifest = EgoBoundaryManifest(**identity, phase_dim=dp, effect_dim=de, global_dim=dg, retrieval_dim=dr,
            executed_horizon=dataset.executed_horizon, image_size=dataset.image_size, action_dim=dataset.action_dim,
            policy_horizon=dataset.policy_horizon, num_records=len(rows), num_episodes=len(metadata),
            index_sha256=file_sha256(temporary / "episode_index.json"),
            episode_metadata_sha256=file_sha256(temporary / "episodes.json"),
            episode_splits_sha256=file_sha256(temporary / "episode_splits.json"), shard_sha256=hashes)
        manifest.validate()
        (temporary / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
        checked = EgoBoundaryCache(temporary)
        for index in range(len(checked)):
            checked.get(index)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
        return asdict(manifest)
    except Exception:
        shutil.rmtree(temporary)
        raise
