"""Ordinary DDP with optional ZeRO-1 Adam state sharding.

Parameters and gradients remain replicated. This is PyTorch's
ZeroRedundancyOptimizer, not FSDP or a DeepSpeed checkpoint implementation.
Every rank participates in consolidation; only rank zero writes the CPU
checkpoint. Resume reads that complete state and restores each rank's shard.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.distributed as dist


OPTIMIZER_CHECKPOINT_FORMAT = "fastwam_zeva_optimizer_full_v1"


def optimizer_sharding_mode(value) -> str:
    mode = "none" if value is None else str(value).strip().lower()
    if mode not in {"none", "zero1"}:
        raise ValueError("zeva_optimizer_sharding must be none or zero1")
    return mode


def optimizer_betas(value, *, predictive: bool) -> tuple[float, float]:
    if value is None:
        return (0.9, 0.999) if predictive else (0.9, 0.95)
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError("optimizer_betas must contain two finite values in [0,1)") from error
    if len(values) != 2 or any(not math.isfinite(item) or not 0 <= item < 1 for item in values):
        raise ValueError("optimizer_betas must contain two finite values in [0,1)")
    return values


def unwrap_optimizer(optimizer):
    # Accelerate wraps an ordinary optimizer without changing its parameters.
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    return optimizer


def _is_zero1(optimizer) -> bool:
    from torch.distributed.optim import ZeroRedundancyOptimizer

    return isinstance(unwrap_optimizer(optimizer), ZeroRedundancyOptimizer)


def create_training_optimizer(parameters, *, sharding: str, predictive: bool,
                              lr: float, weight_decay: float, betas):
    mode = optimizer_sharding_mode(sharding)
    options = dict(lr=lr, weight_decay=weight_decay, betas=tuple(betas))
    if predictive:
        # Keep the Adam implementation equal in frozen/joint experiments and
        # avoid foreach tensor-list temporaries on large FP32 parameter groups.
        options["foreach"] = False
    if mode == "none":
        return torch.optim.AdamW(parameters, **options)
    if not predictive:
        raise ValueError("zero1 is supported only for predictive Zeva Stage2 training")
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
        raise ValueError("zero1 requires ordinary DDP with at least two processes; use none for one process")
    from torch.distributed.optim import ZeroRedundancyOptimizer

    optimizer = ZeroRedundancyOptimizer(parameters, optimizer_class=torch.optim.AdamW,
                                        overlap_with_ddp=False, parameters_as_bucket_view=False, **options)
    params = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if any(parameter.dtype != torch.float32 or not parameter.requires_grad for parameter in params):
        raise ValueError("zero1 predictive Adam requires an explicit FP32 trainable parameter whitelist")
    if len({parameter.device for parameter in params}) != 1:
        raise ValueError("zero1 parameters must already reside on this rank's single training device")
    return optimizer


def optimizer_checkpoint_contract(optimizer, *, world_size: int) -> dict:
    raw = unwrap_optimizer(optimizer)
    return {
        "format": OPTIMIZER_CHECKPOINT_FORMAT,
        "sharding": "zero1" if _is_zero1(raw) else "none",
        "optimizer_class": "AdamW", "world_size": int(world_size),
        "groups": [{
            "name": group.get("name"),
            "shapes": [list(parameter.shape) for parameter in group["params"]],
            "dtypes": [str(parameter.dtype) for parameter in group["params"]],
            "betas": list(group["betas"]), "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
        } for group in raw.param_groups],
    }


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def collect_optimizer_checkpoint(optimizer, scheduler, *, world_size: int, rank: int) -> dict | None:
    """Collectively snapshot current state; return a CPU payload on rank zero.

    This must be called outside an accumulation window by *every* rank. The
    complete Adam state temporarily occupies host memory on rank zero only.
    ``release_optimizer_checkpoint`` must follow the synchronous file write.
    """
    raw = unwrap_optimizer(optimizer)
    if _is_zero1(raw):
        if raw.world_size != world_size or raw.rank != rank:
            raise ValueError("ZeroRedundancyOptimizer and Accelerator rank/world size disagree")
        raw.consolidate_state_dict(to=0)
    if rank != 0:
        return None
    return {
        "optimizer": _cpu_tree(raw.state_dict()), "scheduler": _cpu_tree(scheduler.state_dict()),
        "optimizer_contract": optimizer_checkpoint_contract(raw, world_size=world_size),
    }


def release_optimizer_checkpoint(optimizer) -> None:
    raw = unwrap_optimizer(optimizer)
    if _is_zero1(raw):
        # PyTorch keeps the consolidation snapshot for subsequent state_dict()
        # calls. It is disposable after the synchronous save and would retain
        # a full CPU Adam state throughout training otherwise (PyTorch 2.7).
        raw._all_state_dicts.clear()


def load_optimizer_checkpoint(path: str | Path, optimizer, scheduler, *,
                              world_size: int, predictive: bool) -> None:
    """Restore complete CPU state, retaining only local ZeRO-1 moments.

    mmap avoids allocating the full multi-billion-parameter CPU checkpoint in
    each rank. ZeroRedundancyOptimizer.load_state_dict copies only owned
    moments to the parameter device and keeps Adam scalar steps on CPU.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    raw = unwrap_optimizer(optimizer)
    contract = payload.get("optimizer_contract")
    expected = optimizer_checkpoint_contract(raw, world_size=world_size)
    if contract is not None and contract != expected:
        raise ValueError("Predictive Stage2 optimizer checkpoint contract changed (sharding/world size/groups/betas/precision)")
    if contract is None and _is_zero1(raw):
        raise ValueError("zero1 exact resume requires a complete optimizer checkpoint with its sharding contract")
    state = payload["optimizer"]
    groups = state["param_groups"]
    if len(groups) != len(raw.param_groups):
        raise ValueError("Optimizer checkpoint parameter group count changed")
    for saved, current in zip(groups, raw.param_groups, strict=True):
        if len(saved["params"]) != len(current["params"]) or tuple(saved["betas"]) != tuple(current["betas"]):
            raise ValueError("Optimizer checkpoint parameter layout or optimizer_betas changed")
        if predictive:
            for index in saved["params"]:
                values = state["state"].get(index, {})
                if any(isinstance(values.get(key), torch.Tensor) and values[key].dtype != torch.float32
                       for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")):
                    raise ValueError("Predictive Stage2 resume requires FP32 Adam moments; a BF16 optimizer checkpoint cannot be silently upgraded")
    optimizer.load_state_dict(state)
    if _is_zero1(raw):
        # PyTorch 2.7 restores the owned state into both ``raw.optim.state``
        # and the inherited wrapper ``state``. On CUDA those may be separate
        # copies. Only the local Adam instance updates moments, so alias the
        # wrapper to it rather than retaining stale duplicate GPU moments.
        # Global state_dict() is still built by collective consolidation.
        raw.state = raw.optim.state
    scheduler.load_state_dict(payload["scheduler"])
