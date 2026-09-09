import torch

from fastwam.zeva import MemoryBank, PersistentInteractionMemory, PersistentInteractionMemoryConfig


def test_pim_merge_and_retrieval_are_deterministic():
    pim = PersistentInteractionMemory(
        PersistentInteractionMemoryConfig(phase_dim=2, effect_dim=2, capacity=4, top_k=2, merge_threshold=0.9)
    )
    pim.reset_episode("task", episode_id="episode-1")
    first, merged = pim.append_completed(
        task_cluster="task", phase=torch.tensor([1.0, 0.0]), effect=torch.tensor([1.0, 0.0]), attempt_id=0
    )
    second, merged = pim.append_completed(
        task_cluster="task", phase=torch.tensor([1.0, 0.0]), effect=torch.tensor([1.0, 0.0]), attempt_id=0
    )
    assert first == second and merged and pim.entries[0].observation_count == 2

    bank = MemoryBank(phase_dim=2, effect_dim=2, top_k=2)
    bank.add(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), episode_id="other-a", task_id=3)
    bank.add(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), episode_id="other-b", task_id=3)
    bank.add(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), episode_id="same", task_id=3)
    result = bank.retrieve(torch.tensor([1.0, 0.0]), episode_id="same", task_id=3)
    assert result.mask.tolist() == [True, True]
    assert result.sources[0]["episode_id"] == "other-a"
    assert result.sources[0]["transition_index"] == 0


def test_memory_rejects_non_finite_configuration_and_features():
    try:
        PersistentInteractionMemoryConfig(phase_merge_weight=float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite PIM merge weight was accepted")

    bank = MemoryBank(phase_dim=2, effect_dim=2, top_k=1)
    try:
        bank.add(torch.tensor([float("nan"), 0.0]), torch.ones(2), episode_id="e", task_id=1)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite memory feature was accepted")


def test_cross_task_effect_retrieval_uses_effect_similarity_and_scope():
    bank = MemoryBank(phase_dim=2, effect_dim=2, top_k=2)
    # The phase ranking intentionally disagrees with the effect ranking.
    bank.add(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), episode_id="a", task_id="task-a")
    bank.add(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]), episode_id="b", task_id="task-b")
    bank.add(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), episode_id="c", task_id="task-a")

    result = bank.retrieve_by_effect(
        torch.tensor([1.0, 0.0]),
        episode_id="query",
        task_id="task-a",
        top_k=1,
    )
    assert result.mask.tolist() == [True]
    assert result.sources[0]["episode_id"] == "b"
    assert result.sources[0]["retrieval_key"] == "effect"
    assert result.sources[0]["retrieval_scope"] == "cross_task"


def test_cross_task_effect_retrieval_requires_task_id():
    bank = MemoryBank(phase_dim=2, effect_dim=2, top_k=1)
    try:
        bank.retrieve_by_effect(torch.ones(2), episode_id="query")
    except ValueError:
        pass
    else:
        raise AssertionError("cross-task effect retrieval accepted a missing task_id")
