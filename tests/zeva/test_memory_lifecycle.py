import torch

from fastwam.zeva import CausalMemoryLifecycle, PersistentInteractionMemory, PersistentInteractionMemoryConfig


def test_bit_clears_and_pim_exposes_completed_current_attempt():
    pim = PersistentInteractionMemory(PersistentInteractionMemoryConfig(phase_dim=2, effect_dim=2, capacity=8, top_k=2))
    lifecycle = CausalMemoryLifecycle(pim); lifecycle.set_effect_dim(2); lifecycle.reset_episode("task", episode_id="ep")
    lifecycle.observe_completed_transition(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    # A completed effect is persistent immediately and is causally visible to
    # the next query in the same attempt.
    assert lifecycle.memory_inputs(torch.tensor([1.0, 0.0]), torch.zeros(256))["pim_mask"].sum() == 1
    lifecycle.reset_attempt(1)
    assert lifecycle.bit.tensors().valid.sum() == 0
    assert lifecycle.memory_inputs(torch.tensor([1.0, 0.0]), torch.zeros(256))["pim_mask"].sum() == 1
    assert lifecycle.last_retrieval_sources[0]["episode_id"] == "ep"


def test_lifecycle_uses_pim_effect_dimension():
    pim = PersistentInteractionMemory(
        PersistentInteractionMemoryConfig(phase_dim=3, effect_dim=5, capacity=4, top_k=1)
    )
    lifecycle = CausalMemoryLifecycle(pim)
    lifecycle.reset_episode("task")
    lifecycle.observe_completed_transition(torch.ones(3), torch.ones(5))
    assert lifecycle.bit.tensors().effects.shape[-1] == 5


def test_reset_attempt_requires_episode_before_mutating_lifecycle():
    pim = PersistentInteractionMemory(PersistentInteractionMemoryConfig(phase_dim=2, effect_dim=2))
    lifecycle = CausalMemoryLifecycle(pim)
    try:
        lifecycle.reset_attempt(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("reset_attempt accepted an uninitialized PIM")
    assert lifecycle._attempt_id == 0
    assert lifecycle._transition_index == 0


def test_observe_effect_requires_episode_before_mutating_bit():
    pim = PersistentInteractionMemory(PersistentInteractionMemoryConfig(phase_dim=2, effect_dim=2))
    lifecycle = CausalMemoryLifecycle(pim)
    try:
        lifecycle.observe_completed_effect(torch.ones(2), torch.ones(2))
    except RuntimeError:
        pass
    else:
        raise AssertionError("observe_completed_effect accepted an uninitialized PIM")
    assert lifecycle.bit.tensors().valid.sum() == 0


def test_effect_index_metadata_is_canonical():
    pim = PersistentInteractionMemory(PersistentInteractionMemoryConfig(phase_dim=2, effect_dim=2))
    lifecycle = CausalMemoryLifecycle(pim)
    lifecycle.reset_episode("task")
    lifecycle.observe_completed_effect(
        torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), metadata={"effect_index": 99}
    )
    assert pim.entries[0].metadata["effect_index"] == 0


def test_cross_attempt_pim_merge_is_hidden_until_next_attempt():
    pim = PersistentInteractionMemory(
        PersistentInteractionMemoryConfig(phase_dim=2, effect_dim=2, capacity=8, top_k=2)
    )
    pim.reset_episode("task", episode_id="ep")
    pim.append_completed(
        task_cluster="task", phase=torch.tensor([1.0, 0.0]),
        effect=torch.tensor([1.0, 0.0]), attempt_id=0,
    )
    pim.begin_attempt(1)
    pim.append_completed(
        task_cluster="task", phase=torch.tensor([1.0, 0.0]),
        effect=torch.tensor([1.0, 0.0]), attempt_id=1,
    )
    # The matching observation is merged into the persistent prototype, but
    # the prototype is tagged with the active attempt and cannot leak back
    # into a prediction made during that same attempt.
    _phases, _effects, valid, _scores, sources = pim.query_tensors(
        torch.tensor([1.0, 0.0]), top_k=2, exclude_attempt_id=1
    )
    assert valid.tolist() == [False, False]
    pim.begin_attempt(2)
    _phases, _effects, valid, _scores, sources = pim.query_tensors(
        torch.tensor([1.0, 0.0]), top_k=2, exclude_attempt_id=2
    )
    assert valid.tolist() == [True, False]
    assert sources[0]["attempt_id"] == 1
    assert sources[0]["first_attempt_id"] == 0
    assert sources[0]["last_attempt_id"] == 1


def test_effect_lifecycle_keeps_window_offset_and_writes_immediately():
    pim = PersistentInteractionMemory(
        PersistentInteractionMemoryConfig(phase_dim=2, effect_dim=2, capacity=8, top_k=2)
    )
    lifecycle = CausalMemoryLifecycle(pim)
    lifecycle.reset_episode("task", episode_id="ep")
    lifecycle.observe_completed_effect(
        torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])
    )
    lifecycle.observe_completed_effect(
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])
    )
    # Effects are indexed 0/1, but their source transition windows begin at
    # transition 0/4 in the 8-transition source window. Both are persistent
    # before the attempt boundary, while BIT is still attempt-scoped.
    assert len(pim) == 2
    lifecycle.end_attempt()
    assert [entry.transition_index for entry in pim.entries] == [0, 4]
    assert [entry.metadata["effect_index"] for entry in pim.entries] == [0, 1]
    assert lifecycle.bit.tensors().valid.sum() == 0
