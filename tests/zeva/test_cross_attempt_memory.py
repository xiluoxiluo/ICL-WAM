import pytest
import torch

from fastwam.zeva.lifecycle import CausalMemoryLifecycle
from fastwam.zeva.memory import PersistentInteractionMemory, PersistentInteractionMemoryConfig


def make_memory():
    return PersistentInteractionMemory(PersistentInteractionMemoryConfig(
        phase_dim=2, effect_dim=2, capacity=6, top_k=6,
        cross_attempt_only=True, max_past_attempts=2, max_entries_per_attempt=3,
    ))


def test_current_attempt_is_private_but_bit_is_live_and_commit_is_idempotent():
    memory = make_memory()
    lifecycle = CausalMemoryLifecycle(memory)
    lifecycle.reset_episode("task", "episode")
    for index in range(5):
        lifecycle.observe_completed_effect(torch.tensor([2., float(index)]), torch.tensor([3., float(index)]))
    current = lifecycle.memory_inputs(torch.tensor([1., 0.]), torch.zeros(1, 256))
    assert current["bit_mask"].sum() == 4
    assert current["pim_mask"].sum() == 0
    assert memory.pending_count == 5
    lifecycle.end_attempt()
    lifecycle.end_attempt()
    assert memory.pending_count == 0
    assert [entry.transition_index for entry in memory.entries] == [0, 8, 16]
    assert memory.entries[0].effect.tolist() == [3., 0.]  # raw values, same as training
    lifecycle.reset_attempt(1)
    assert lifecycle.bit.tensors().valid.sum() == 0
    assert len(memory) == 3
    lifecycle.observe_completed_effect(torch.tensor([2., 0.]), torch.tensor([99., 0.]))
    previous = lifecycle.memory_inputs(torch.tensor([1., 0.]), torch.zeros(1, 256))
    assert previous["pim_mask"].sum() == 3
    assert all(source["attempt_id"] == 0 for source in lifecycle.last_retrieval_sources)
    assert not torch.any(previous["pim_effects"] == 99.)


def test_capacity_evicts_whole_old_attempts_and_episode_reset_clears_pending():
    memory = make_memory()
    memory.reset_episode("task")
    for attempt in range(4):
        memory.begin_attempt(attempt)
        for step in range(5):
            memory.append_completed(task_cluster="task", phase=torch.tensor([1., 0.]),
                                    effect=torch.tensor([float(attempt), float(step)]),
                                    attempt_id=attempt, transition_index=step)
        memory.commit_attempt()
    assert len(memory) == 6
    assert {entry.attempt_id for entry in memory.entries} == {2, 3}
    assert memory.past_attempt_count == 2
    memory.reset_episode("task", episode_id="new")
    assert len(memory) == memory.pending_count == memory.past_attempt_count == 0


def test_cross_attempt_rejects_truncated_retrieval_and_late_writes():
    memory = make_memory()
    memory.reset_episode("task")
    with pytest.raises(ValueError, match="all historical"):
        memory.query_tensors(torch.ones(2), top_k=2)
    memory.commit_attempt()
    with pytest.raises(RuntimeError, match="after commit"):
        memory.append_completed(task_cluster="task", phase=torch.ones(2), effect=torch.ones(2), attempt_id=0)
    with pytest.raises(ValueError, match="full capacity"):
        PersistentInteractionMemoryConfig(capacity=192, top_k=4, cross_attempt_only=True)


def test_invalid_effect_does_not_mutate_bit():
    memory = make_memory()
    lifecycle = CausalMemoryLifecycle(memory)
    lifecycle.reset_episode("task")
    with pytest.raises(ValueError, match="non-finite"):
        lifecycle.observe_completed_effect(torch.ones(2), torch.tensor([float("nan"), 0.]))
    assert not lifecycle.bit.tensors().valid.any()
    assert memory.pending_count == 0
