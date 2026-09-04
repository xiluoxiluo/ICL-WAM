"""Separated CTE/addon checkpoints and strict compatibility checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .schemas import CacheManifest


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_cte_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, step: int = 0, config: dict | None = None, camera_keys: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")) -> None:
    payload = {"model": model.state_dict(), "optimizer": None if optimizer is None else optimizer.state_dict(), "scheduler": None if scheduler is None else scheduler.state_dict(), "step": int(step), "config": config or model.cfg.to_dict(), "action_dim": int(model.cfg.action_dim), "camera_keys": list(camera_keys)}
    torch.save(payload, path)


def load_cte_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def save_addon_checkpoint(path: str | Path, causal_prompt_encoder, behavior_prefix_adapter, step: int, config: dict, base_checkpoint_sha256: str, cte_checkpoint_sha256: str) -> None:
    torch.save({"causal_prompt_encoder": causal_prompt_encoder.state_dict(), "behavior_prefix_adapter": behavior_prefix_adapter.state_dict(), "pim_gate": behavior_prefix_adapter.pim_gate.detach().cpu(), "step": int(step), "config": config, "base_checkpoint_sha256": base_checkpoint_sha256, "cte_checkpoint_sha256": cte_checkpoint_sha256}, path)


def load_addon_checkpoint(path: str | Path, causal_prompt_encoder, behavior_prefix_adapter, *, base_checkpoint_sha256: str | None = None, cte_checkpoint_sha256: str | None = None, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    for name, expected in (("base_checkpoint_sha256", base_checkpoint_sha256), ("cte_checkpoint_sha256", cte_checkpoint_sha256)):
        if expected is not None and payload.get(name) != expected:
            raise ValueError(f"addon checkpoint {name} mismatch")
    causal_prompt_encoder.load_state_dict(payload["causal_prompt_encoder"], strict=True)
    behavior_prefix_adapter.load_state_dict(payload["behavior_prefix_adapter"], strict=True)
    return payload
