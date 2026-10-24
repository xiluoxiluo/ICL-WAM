"""Atomic, source-bound rank snapshots for Zeva's behavior-bank builder.

Only completed per-rank artifacts are eligible for reuse. A legacy rank
snapshot has no source hashes and therefore requires explicit opt-in after
its episode order, tensor dimensions and finite values have been checked.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def rank_contract(
    *, plans: Iterable[Any], source: dict[str, Any], rank: int, world_size: int
) -> dict[str, Any]:
    """Bind a saved rank to the same source and deterministic episode partition."""
    coordinates = [
        (str(plan.episode_id), str(plan.task_id), int(plan.dataset_start))
        for plan in plans
    ]
    if not coordinates:
        raise ValueError("cannot checkpoint an empty behavior-bank rank")
    if len({row[0] for row in coordinates}) != len(coordinates):
        raise ValueError("behavior-bank rank has duplicate episode identities")
    return {
        "format": "zeva_behavior_rank_resume_v1",
        "rank": int(rank),
        "world_size": int(world_size),
        "num_episodes": len(coordinates),
        "episode_order_sha256": _sha256_json(coordinates),
        "source_sha256": _sha256_json(source),
    }


def validate_rank_snapshot(
    payload: dict[str, Any], *, plans: list[Any], contract: dict[str, Any],
    key_dim: int, value_dim: int, readout_dim: int,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Reject stale/corrupt/incompatible data before any rank can skip work."""
    if not isinstance(payload, dict):
        raise ValueError("rank snapshot must be a dict")
    saved_contract = payload.get("resume_contract")
    if saved_contract is None:
        if not allow_legacy:
            raise ValueError(
                "legacy rank snapshot has no checkpoint/source hashes; "
                "set ZEVA_BEHAVIOR_ACCEPT_LEGACY_RANK=1 only if it belongs "
                "to the same CTE/base/data run"
            )
    elif saved_contract != contract:
        raise ValueError("behavior rank checkpoint source/episode/world-size identity mismatch")
    entries, ids, readouts = (
        payload.get("entries"), payload.get("episode_ids"), payload.get("readouts")
    )
    if not isinstance(entries, list) or len(entries) != len(plans):
        raise ValueError("rank checkpoint entry count does not match current episode plan")
    if not isinstance(ids, list) or ids != [str(p.episode_id) for p in plans]:
        raise ValueError("rank checkpoint episode order differs from current plan")
    if not torch.is_tensor(readouts) or readouts.dtype != torch.float32 or (
        readouts.ndim != 2 or tuple(readouts.shape) != (len(plans), int(readout_dim))
    ):
        raise ValueError("rank checkpoint readout tensor shape/dtype is incompatible")
    if not bool(torch.isfinite(readouts).all()):
        raise ValueError("rank checkpoint readouts contain nonfinite values")
    for i, (entry, plan) in enumerate(zip(entries, plans, strict=True)):
        if not isinstance(entry, dict):
            raise ValueError(f"rank checkpoint entry {i} must be a dict")
        if (
            str(entry.get("episode_id")) != str(plan.episode_id)
            or str(entry.get("task_id")) != str(plan.task_id)
            or int(entry.get("dataset_start", -1)) != int(plan.dataset_start)
        ):
            raise ValueError(f"rank checkpoint entry {i} identity mismatch")
        if not isinstance(entry.get("instruction"), str):
            raise ValueError(f"rank checkpoint entry {i} instruction missing")
        if int(entry.get("num_transitions", 0)) < 1:
            raise ValueError(f"rank checkpoint entry {i} has invalid transition count")
        for name, dim in (("retrieval_key", key_dim), ("behavior_value", value_dim)):
            tensor = entry.get(name)
            if (not torch.is_tensor(tensor) or tuple(tensor.shape) != (int(dim),)
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError(f"rank checkpoint entry {i} {name} invalid")
    if int(payload.get("tail_windows", -1)) < 0:
        raise ValueError("rank checkpoint tail-window count invalid")
    return payload


def atomic_save_rank_snapshot(path: Path, payload: dict[str, Any]) -> None:
    """Never advertise a half-written rank file as resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    check = torch.load(temp, map_location="cpu", weights_only=False)
    if not isinstance(check, dict) or check.get("episode_ids") != payload.get("episode_ids"):
        raise RuntimeError("rank snapshot serialization validation failed")
    os.replace(temp, path)


def wait_for_files(
    paths: Iterable[Path], *, phase: str, timeout_seconds: float = 3600.0,
    report_every_seconds: float = 30.0, poll_seconds: float = 5.0,
) -> None:
    """Filesystem rendezvous avoids a timed NCCL collective during CPU merge."""
    wanted = tuple(Path(path) for path in paths)
    if not wanted:
        raise ValueError("at least one synchronization file is required")
    if timeout_seconds <= 0 or report_every_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("wait/poll intervals must be positive")
    deadline = time.monotonic() + timeout_seconds
    next_report = time.monotonic() + report_every_seconds
    while True:
        missing = [str(path) for path in wanted if not path.is_file()]
        if not missing:
            return
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(f"{phase}: timed out waiting for files: {missing}")
        if now >= next_report:
            print(f"[behavior-bank sync] {phase}: awaiting {missing}", flush=True)
            next_report = now + report_every_seconds
        time.sleep(min(poll_seconds, deadline - now))
