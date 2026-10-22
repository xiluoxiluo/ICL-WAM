"""Local implementation of the context/support dropout declared by Zeva main.

The release specifies both dropout rates but omits the complete data builder.
Here the whole persistent context and individual support entries are dropped
so the frozen policy sees empty, sparse, and full memory states. This local
implementation is explicit and deterministic per sample,
which is important for multi-worker/multi-rank Stage-2 training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PIMTrainingDropoutConfig:
    context_dropout: float = 0.2
    support_dropout: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        for name, value in (
            ("context_dropout", self.context_dropout),
            ("support_dropout", self.support_dropout),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


def _sample_generator(seed: int, sample_index: int) -> torch.Generator:
    # A large odd multiplier keeps neighbouring dataset indices decorrelated
    # while making the result independent of dataloader worker/rank ordering.
    value = (int(seed) + 1_000_003 * int(sample_index)) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(value)
    return generator


def apply_pim_training_dropout(
    *,
    phases: Tensor,
    effects: Tensor,
    valid: Tensor,
    scores: Tensor | None,
    sample_index: int,
    config: PIMTrainingDropoutConfig,
) -> dict[str, Tensor]:
    """Apply Zeva-style context/support dropout to one top-k PIM sample.

    Dropped entries are physically zeroed in addition to clearing the valid
    mask.  This prevents masked values from leaking through future refactors or
    attention implementations.  Scores for dropped entries are set to -inf.
    """

    if phases.ndim != 2 or effects.ndim != 2 or valid.ndim != 1:
        raise ValueError("PIM tensors must be phases/effects [K,D] and valid [K]")
    if phases.shape[0] != effects.shape[0] or phases.shape[0] != valid.shape[0]:
        raise ValueError("PIM top-k dimension mismatch")
    if scores is not None and (scores.ndim != 1 or scores.shape[0] != valid.shape[0]):
        raise ValueError("PIM scores must have shape [K]")

    phases = phases.clone()
    effects = effects.clone()
    valid = valid.bool().clone()
    scores_out = None if scores is None else scores.clone()
    count_before = valid.sum(dtype=torch.int64)

    generator = _sample_generator(config.seed, sample_index)
    context_dropped = torch.rand((), generator=generator).item() < float(config.context_dropout)
    if context_dropped:
        valid.zero_()
    elif float(config.support_dropout) > 0.0 and bool(valid.any()):
        keep = torch.rand(valid.shape, generator=generator) >= float(config.support_dropout)
        valid &= keep

    dropped = ~valid
    phases[dropped] = 0
    effects[dropped] = 0
    if scores_out is not None:
        scores_out[dropped] = -torch.inf

    return {
        "phases": phases,
        "effects": effects,
        "valid": valid,
        "scores": scores_out if scores_out is not None else torch.empty(0),
        "count_before": count_before,
        "count_after": valid.sum(dtype=torch.int64),
        "context_dropped": torch.tensor(bool(context_dropped), dtype=torch.bool),
    }


class PIMTrainingDropoutDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies PIM dropout without altering source caches."""

    def __init__(self, dataset, config: PIMTrainingDropoutConfig) -> None:
        self.dataset = dataset
        self.config = config

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def pim(self):
        return self.dataset.pim

    @property
    def static_context(self):
        return self.dataset.static_context

    @property
    def split_identity(self):
        return getattr(self.dataset, "split_identity", None)

    def __getitem__(self, index: int):
        sample = dict(self.dataset[index])
        required = ("pim_phases", "pim_effects", "pim_mask")
        if any(name not in sample for name in required):
            raise KeyError(f"Stage-2B sample is missing required PIM tensors: {required}")
        result = apply_pim_training_dropout(
            phases=sample["pim_phases"],
            effects=sample["pim_effects"],
            valid=sample["pim_mask"],
            scores=sample.get("pim_scores"),
            sample_index=int(index),
            config=self.config,
        )
        sample["pim_phases"] = result["phases"]
        sample["pim_effects"] = result["effects"]
        sample["pim_mask"] = result["valid"]
        if "pim_scores" in sample:
            sample["pim_scores"] = result["scores"]
        # Expose scalar diagnostics so the trainer can verify that the YAML
        # dropout settings are actually active instead of silently ignored.
        sample["pim_count_before"] = result["count_before"]
        sample["pim_count_after"] = result["count_after"]
        sample["pim_context_dropped"] = result["context_dropped"]
        return sample
