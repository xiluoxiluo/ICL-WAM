"""Repeated-attempt protocol shared by RoboTwin Zeva evaluation.

Zeva keeps the environment instance fixed across attempts but changes the
policy sampling seed deterministically for every attempt/replan pair.  Reusing
one diffusion seed across all retries can cause a frozen policy with a weak PIM
residual to repeat essentially the same failure indefinitely.
"""

from __future__ import annotations


ATTEMPT_SEED_STRIDE = 1_000


def inference_seed(base_seed: int | None, attempt_id: int, replan_index: int) -> int | None:
    """Return Zeva-compatible deterministic seed for one policy request."""
    if base_seed is None:
        return None
    base_seed = int(base_seed)
    attempt_id = int(attempt_id)
    replan_index = int(replan_index)
    if base_seed < 0 or attempt_id < 0 or replan_index < 0:
        raise ValueError("base_seed, attempt_id and replan_index must be non-negative")
    if replan_index >= ATTEMPT_SEED_STRIDE:
        raise ValueError(
            f"replan_index must be < {ATTEMPT_SEED_STRIDE} to avoid cross-attempt seed collisions"
        )
    value = base_seed + ATTEMPT_SEED_STRIDE * attempt_id + replan_index
    if value >= 2**32:
        raise ValueError("derived inference seed exceeds uint32")
    return value
