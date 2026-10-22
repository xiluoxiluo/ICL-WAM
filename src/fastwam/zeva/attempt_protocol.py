"""Model sampling streams independent of RoboTwin scene/reset randomness.

Formal evaluation consumes one private CPU Torch generator per slot, including
across task subprocesses. This preserves Zeva-Ego's continuous-stream rule;
FastWAM's CPU float32 noise is not Zeva's CUDA numerical sequence. The previous
per-request seed stream remains available for replaying legacy experiments.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
from pathlib import Path


ATTEMPT_SEED_STRIDE = 1_000
CONTINUOUS_NOISE_PROTOCOL = "torch_continuous_slot_v1"


class ContinuousModelNoiseStream:
    """One CPU Torch generator per evaluation slot, independent of resets.

    State can be handed to the next task subprocess in the same fresh run.
    This is not a partial-run resume facility: the manifest runner refuses
    existing output directories and verifies each completed task handoff.
    CPU float32 sampling is FastWAM's contract, not Zeva's CUDA bit sequence.
    """

    def __init__(
        self, seed: int, *, slot_index: int = 0, slot_count: int = 1,
        run_id: str | None = None, state_path: str | Path | None = None,
        task_name: str | None = None,
        checkpoint_identity: dict[str, object] | None = None,
    ) -> None:
        import torch

        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("continuous model seed must be an integer in [0,2**63)")
        if type(slot_count) is not int or slot_count < 1 or type(slot_index) is not int or not 0 <= slot_index < slot_count:
            raise ValueError("invalid evaluation slot index/count")
        self.seed, self.slot_index, self.slot_count = seed, slot_index, slot_count
        self.run_id, self.task_name = run_id, task_name
        if checkpoint_identity is not None and not isinstance(checkpoint_identity, dict):
            raise ValueError("checkpoint_identity must be a JSON object")
        # Freeze all base/addon/CTE/stats and static bank/readout identities;
        # later mutation of a policy metadata dict cannot alter a handoff.
        self.checkpoint_identity = json.loads(json.dumps(checkpoint_identity, sort_keys=True, allow_nan=False))
        self.state_path = None if state_path is None else Path(state_path)
        if self.state_path is not None and (not run_id or not task_name):
            raise ValueError("slot state handoff requires run_id and task_name")
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.draws = 0
        self.noise_shape: tuple[int, ...] | None = None
        self.completed_tasks: list[str] = []
        self._saved = False
        if self.state_path is not None and self.state_path.exists():
            payload = json.loads(self.state_path.read_text())
            identity = self._identity()
            if any(payload.get(key) != value for key, value in identity.items()):
                raise ValueError("continuous model RNG handoff identity mismatch")
            self.completed_tasks = payload.get("completed_tasks", [])
            if (not isinstance(self.completed_tasks, list) or not self.completed_tasks
                    or any(not isinstance(task, str) or not task for task in self.completed_tasks)
                    or len(set(self.completed_tasks)) != len(self.completed_tasks)
                    or task_name in self.completed_tasks):
                raise ValueError("invalid or duplicate completed task in RNG handoff")
            draws = payload.get("draws")
            shape = payload.get("noise_shape")
            if type(draws) is not int or draws < 0 or (shape is not None and not self._valid_shape(shape)):
                raise ValueError("invalid RNG draw count or noise shape")
            try:
                raw = base64.b64decode(payload["generator_state"], validate=True)
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError("invalid encoded Torch RNG state") from exc
            if hashlib.sha256(raw).hexdigest() != payload.get("state_sha256"):
                raise ValueError("continuous model RNG state checksum mismatch")
            try:
                self.generator.set_state(torch.tensor(list(raw), dtype=torch.uint8))
            except RuntimeError as exc:
                raise ValueError("invalid Torch generator state") from exc
            self.draws = draws
            self.noise_shape = None if shape is None else tuple(shape)

    @staticmethod
    def _valid_shape(shape) -> bool:
        return isinstance(shape, (list, tuple)) and len(shape) == 3 and shape[0] == 1 and all(type(n) is int and n > 0 for n in shape)

    def _identity(self) -> dict[str, object]:
        return {
            "protocol": CONTINUOUS_NOISE_PROTOCOL, "seed": self.seed,
            "slot_index": self.slot_index, "slot_count": self.slot_count,
            "run_id": self.run_id, "device": "cpu", "dtype": "float32",
            "checkpoint_identity": self.checkpoint_identity,
        }

    def _state_bytes(self) -> bytes:
        return bytes(self.generator.get_state().tolist())

    def next_noise(self, shape):
        import torch

        if self._saved:
            raise RuntimeError("cannot sample after completing this task's RNG handoff")
        if not self._valid_shape(shape):
            raise ValueError("action noise shape must be [1,horizon,action_dim]")
        shape = tuple(shape)
        if self.noise_shape is not None and shape != self.noise_shape:
            raise ValueError("action noise shape changed within an evaluation slot")
        self.noise_shape = shape
        noise = torch.randn(shape, generator=self.generator, device="cpu", dtype=torch.float32)
        self.draws += 1
        return noise

    def snapshot(self) -> dict[str, object]:
        return {
            "protocol": CONTINUOUS_NOISE_PROTOCOL, "seed": self.seed,
            "draws": self.draws, "state_sha256": hashlib.sha256(self._state_bytes()).hexdigest(),
            "device": "cpu", "dtype": "float32",
        }

    def save_state(self) -> None:
        if self.state_path is None or self._saved:
            return
        raw = self._state_bytes()
        payload = {
            **self._identity(), **self.snapshot(),
            "generator_state": base64.b64encode(raw).decode("ascii"),
            "noise_shape": self.noise_shape,
            "completed_tasks": [*self.completed_tasks, self.task_name],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(self.state_path.name + ".partial")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(self.state_path)
        self._saved = True


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
