"""Dependency-free Zeva-main request seeds and their auditable history.

The published main ``attempt_protocol.inference_seed`` keys a request by its
attempt and replan indices. Unlike the separate Zeva-Ego continuous generator,
the seed for a retry cannot shift when a preceding attempt terminates early.
The simulator scene seed never enters this model-noise schedule.
"""

from __future__ import annotations


MAIN_MODEL_RNG_PROTOCOL = "zeva_main_attempt_replan_seed_v1"
ATTEMPT_SEED_STRIDE = 1_000


def main_inference_seed(base_seed: int, attempt_id: int, replan_index: int) -> int:
    """Match main17ca9fa's uint32 ``base + 1000 * attempt + replan``."""
    if any(type(value) is not int or value < 0 for value in (base_seed, attempt_id, replan_index)):
        raise ValueError("base_seed, attempt_id, and replan_index must be non-negative integers")
    if replan_index >= ATTEMPT_SEED_STRIDE:
        raise ValueError("replan_index must be < 1000")
    value = base_seed + ATTEMPT_SEED_STRIDE * attempt_id + replan_index
    if value >= 2**32:
        raise ValueError("derived inference seed exceeds uint32")
    return value


class MainAttemptSeedSchedule:
    """Record only the actual policy requests in the current attempt.

    This is a stateless seed formula, not a persistent Torch RNG. The model
    creates a private generator from each returned seed on ``device``. A
    diagnostic reusing that request must not call ``next_seed`` a second time.
    """

    def __init__(self, seed: int, *, device: str = "cpu") -> None:
        main_inference_seed(seed, 0, 0)
        if not isinstance(device, str) or not device:
            raise ValueError("request-noise device must be a nonempty string")
        self.seed, self.device = seed, device
        self.reset_episode()

    def reset_episode(self) -> None:
        self.attempt_id = 0
        self.request_seeds: list[int] = []

    def begin_attempt(self, attempt_id: int) -> None:
        main_inference_seed(self.seed, attempt_id, 0)
        if attempt_id not in {self.attempt_id, self.attempt_id + 1}:
            raise ValueError("request seed attempt_id must stay constant or increment by one")
        if attempt_id == self.attempt_id and self.request_seeds:
            raise RuntimeError("cannot reset an active request-seed attempt")
        self.attempt_id = attempt_id
        self.request_seeds = []

    def next_seed(self, attempt_id: int, replan_index: int) -> int:
        if attempt_id != self.attempt_id or replan_index != len(self.request_seeds):
            raise ValueError("request seed must match the active attempt and next replan index")
        seed = main_inference_seed(self.seed, attempt_id, replan_index)
        self.request_seeds.append(seed)
        return seed

    def snapshot(self) -> dict[str, object]:
        return {
            "protocol": MAIN_MODEL_RNG_PROTOCOL, "seed": self.seed,
            "attempt_id": self.attempt_id, "draws": len(self.request_seeds),
            "request_seeds": list(self.request_seeds),
            "device": self.device, "dtype": "float32",
        }
