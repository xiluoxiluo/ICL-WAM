from __future__ import annotations

import torch

from fastwam.zeva.attempt_protocol import inference_seed
from fastwam.zeva.pim_training import (
    PIMTrainingDropoutConfig,
    apply_pim_training_dropout,
)


def _sample(index: int, *, context_dropout: float = 0.2, support_dropout: float = 0.2):
    phases = torch.arange(16, dtype=torch.float32).view(4, 4)
    effects = phases + 100
    valid = torch.ones(4, dtype=torch.bool)
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)
    return apply_pim_training_dropout(
        phases=phases,
        effects=effects,
        valid=valid,
        scores=scores,
        sample_index=index,
        config=PIMTrainingDropoutConfig(
            context_dropout=context_dropout,
            support_dropout=support_dropout,
            seed=42,
        ),
    )


def test_pim_dropout_is_deterministic_per_sample() -> None:
    first = _sample(17)
    second = _sample(17)
    for key in ("phases", "effects", "valid", "scores", "count_after", "context_dropped"):
        assert torch.equal(first[key], second[key])


def test_context_dropout_can_produce_empty_memory() -> None:
    result = _sample(0, context_dropout=1.0, support_dropout=0.0)
    assert int(result["count_before"]) == 4
    assert int(result["count_after"]) == 0
    assert not bool(result["valid"].any())
    assert torch.count_nonzero(result["phases"]) == 0
    assert torch.count_nonzero(result["effects"]) == 0
    assert torch.isneginf(result["scores"]).all()


def test_support_dropout_zeroes_dropped_support() -> None:
    result = _sample(3, context_dropout=0.0, support_dropout=1.0)
    assert int(result["count_after"]) == 0
    assert torch.count_nonzero(result["phases"]) == 0
    assert torch.count_nonzero(result["effects"]) == 0


def test_retry_seed_matches_zeva_attempt_replan_schedule() -> None:
    assert inference_seed(197, 0, 0) == 197
    assert inference_seed(197, 0, 3) == 200
    assert inference_seed(197, 1, 0) == 1197
    assert inference_seed(197, 2, 7) == 2204
    values = {
        inference_seed(197, attempt, replan)
        for attempt in range(4)
        for replan in range(20)
    }
    assert len(values) == 80


def test_retry_seed_allows_unseeded_runtime() -> None:
    assert inference_seed(None, 9, 4) is None
