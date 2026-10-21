"""Model sampling streams independent of RoboTwin scene/reset randomness.

The FastWAM API accepts a seed per inference request. ``ModelSeedStream`` is
therefore an adapted request-seed stream, not an exact reproduction of Zeva's
continuously consumed torch generator. All policy conditions use the same
stream rule, and it is never reset at an attempt or episode boundary.
"""

from __future__ import annotations

import hashlib
import json
import random


ATTEMPT_SEED_STRIDE = 1_000


class ModelSeedStream:
    """A local RNG whose consumption is unaffected by environment resets."""

    def __init__(self, seed: int | None) -> None:
        if seed is not None and (isinstance(seed, bool) or int(seed) < 0):
            raise ValueError("model seed must be non-negative")
        self.seed = None if seed is None else int(seed)
        self._rng = random.Random(self.seed)
        self.draws = 0

    def next_seed(self) -> int:
        value = self._rng.randrange(2**32)
        self.draws += 1
        return value

    def snapshot(self) -> dict[str, object]:
        state = json.dumps(self._rng.getstate(), separators=(",", ":"))
        return {
            "protocol": "request_seed_stream_v1",
            "seed": self.seed,
            "draws": self.draws,
            "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
        }


def inference_seed(base_seed: int | None, attempt_id: int, replan_index: int) -> int | None:
    """Legacy deterministic schedule, retained for old experiment replay."""
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
