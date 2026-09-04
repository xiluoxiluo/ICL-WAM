"""Episode-ordered phase/effect cache and deterministic shard IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file

from .schemas import CacheManifest


class PhaseEffectCache:
    def __init__(self, root: str | Path, manifest: CacheManifest, rows: list[dict]):
        self.root, self.manifest, self.rows = Path(root), manifest, rows
        self._shards: dict[str, dict[str, torch.Tensor]] = {}
        required = {"episode_id", "task_id", "attempt_id", "transition_index", "window_index", "episode_step", "shard", "offset"}
        for index, row in enumerate(rows):
            missing = required - set(row)
            if missing:
                raise ValueError(f"cache row {index} missing fields: {sorted(missing)}")
            shard = self.root / str(row["shard"])
            if not shard.is_file():
                raise FileNotFoundError(f"cache row {index} references missing shard: {shard}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, name: str) -> dict[str, torch.Tensor]:
        # Stage 2 repeatedly reads the same ordered shards while constructing
        # BIT and the offline memory bank. Cache the opened tensor map per
        # process instead of reopening a safetensors file for every row.
        if name not in self._shards:
            self._shards[name] = load_file(str(self.root / name), device="cpu")
        return self._shards[name]

    def get(self, index: int) -> dict:
        row = self.rows[int(index)]
        shard = self._load_shard(str(row["shard"]))
        offset = int(row["offset"])
        return {
            "episode_id": row["episode_id"], "task_id": row["task_id"],
            "attempt_id": row["attempt_id"], "transition_index": row["transition_index"],
            "episode_step": row.get("episode_step"),
            "window_index": row.get("window_index"),
            "phase_pre": shard["phase_pre"][offset], "phase_post": shard["phase_post"][offset],
            "effect": shard["effect"][offset], "valid": bool(shard["valid"][offset].item()),
        }

    @classmethod
    def load(cls, root: str | Path, expected: dict[str, object] | None = None) -> "PhaseEffectCache":
        root = Path(root)
        manifest = CacheManifest(**json.loads((root / "manifest.json").read_text()))
        manifest.validate(expected)
        rows = json.loads((root / "episode_index.json").read_text())
        return cls(root, manifest, rows)


def save_phase_effect_cache(root: str | Path, records: Iterable[dict], manifest: CacheManifest, shard_size: int = 4096) -> None:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    rows: list[dict] = []; pending: list[dict] = []
    seen_keys: set[tuple[int, int]] = set()

    def flush(shard_id: int, values: list[dict]) -> None:
        if not values:
            return
        name = f"phase_effect-{shard_id:05d}.safetensors"
        phase_pre = torch.stack([v["phase_pre"].float().cpu() for v in values])
        phase_post = torch.stack([v["phase_post"].float().cpu() for v in values])
        effect = torch.stack([v["effect"].float().cpu() for v in values])
        if phase_pre.ndim != 2 or phase_pre.shape[1] != manifest.phase_dim or phase_post.shape != phase_pre.shape:
            raise ValueError("phase feature shape does not match cache manifest")
        if effect.ndim != 2 or effect.shape[1] != manifest.effect_dim:
            raise ValueError("effect feature shape does not match cache manifest")
        if not (torch.isfinite(phase_pre).all() and torch.isfinite(phase_post).all() and torch.isfinite(effect).all()):
            raise ValueError("cache refuses non-finite phase/effect features")
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[manifest.feature_dtype]
        save_file({
            "phase_pre": phase_pre.to(dtype=dtype),
            "phase_post": phase_post.to(dtype=dtype),
            "effect": effect.to(dtype=dtype),
            "valid": torch.tensor([bool(v["valid"]) for v in values], dtype=torch.bool),
        }, str(root / name))
        for offset, value in enumerate(values):
            rows.append({"episode_id": str(value["episode_id"]), "task_id": value.get("task_id", 0),
                         "attempt_id": int(value.get("attempt_id", 0)),
                         "episode_step": None if value.get("episode_step") is None else int(value["episode_step"]),
                         "transition_index": int(value.get("transition_index", offset)),
                         "window_index": None if value.get("window_index") is None else int(value["window_index"]),
                         "shard": name, "offset": offset})

    shard_id = 0
    for record in records:
        if record.get("window_index") is not None:
            key = (int(record["window_index"]), int(record.get("transition_index", 0)))
            if key in seen_keys:
                raise ValueError(f"duplicate cache row for window/transition {key}")
            seen_keys.add(key)
        pending.append(record)
        if len(pending) >= shard_size:
            flush(shard_id, pending); shard_id += 1; pending = []
    flush(shard_id, pending)
    (root / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    (root / "episode_index.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
