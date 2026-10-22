"""Small counterexamples found by a fresh audit against Zeva main17ca9fa."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastwam.zeva.memory import PersistentInteractionMemory, PersistentInteractionMemoryConfig
from fastwam.zeva_main_protocol import MainAttemptSeedSchedule, main_inference_seed


def _memory(*, capacity=3, top_k=3):
    memory = PersistentInteractionMemory(PersistentInteractionMemoryConfig(
        phase_dim=3, effect_dim=3, capacity=capacity, top_k=top_k, merge_threshold=.99,
    ))
    memory.reset_episode("task")
    return memory


def _write(memory, vector, index):
    return memory.append_completed(task_cluster="task", phase=vector, effect=vector,
                                   attempt_id=0, transition_index=index)


def test_main_eviction_uses_original_insertion_tie_break_after_reversed_merges():
    memory = _memory(capacity=2, top_k=2)
    a, b, c = torch.eye(3)
    _write(memory, a, 0)
    _write(memory, b, 4)
    _write(memory, b, 8)
    _write(memory, a, 12)
    assert [entry.observation_count for entry in memory.entries] == [2, 2]
    _write(memory, c, 16)
    # main17ca9fa uses (count, last_attempt, insertion index), not the last
    # transition timestamp. The old local key incorrectly evicted b here.
    torch.testing.assert_close(torch.stack([entry.phase for entry in memory.entries]), torch.stack((b, c)))


def test_main_retrieval_uses_reference_torch_topk_order_for_equal_scores():
    memory = _memory(capacity=8, top_k=4)
    # Distinct effects keep same-phase entries from merging. With threshold
    # one, repeated exact keys remain independent when effects differ.
    memory.config = PersistentInteractionMemoryConfig(
        phase_dim=3, effect_dim=3, capacity=8, top_k=4, merge_threshold=1,
    )
    phase = torch.tensor([1., 0., 0.])
    for index, effect in enumerate((torch.tensor([0., 1., float(n)]) for n in range(8))):
        memory.append_completed(task_cluster="task", phase=phase, effect=effect,
                                attempt_id=0, transition_index=index)
    assert len(memory.entries) == 8
    _, _, valid, scores, sources = memory.query_tensors(phase)
    expected_scores, expected_order = torch.topk(torch.ones(8), k=4, largest=True, sorted=True)
    assert [source["transition_index"] for source in sources] == expected_order.tolist()
    assert valid.all()
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)


def test_main_schedule_is_request_indexed_and_unaffected_by_simulator_reseeding():
    on, off = MainAttemptSeedSchedule(20260824), MainAttemptSeedSchedule(20260824)
    on.next_seed(0, 0)
    for replan in range(19):
        off.next_seed(0, replan)
    on.begin_attempt(1)
    off.begin_attempt(1)
    torch.manual_seed(1000)
    on_noise = torch.randn((1, 32, 14), generator=torch.Generator().manual_seed(on.next_seed(1, 0)))
    torch.manual_seed(9999)
    off_noise = torch.randn((1, 32, 14), generator=torch.Generator().manual_seed(off.next_seed(1, 0)))
    torch.testing.assert_close(on_noise, off_noise, rtol=0, atol=0)
    assert on.snapshot()["request_seeds"] == off.snapshot()["request_seeds"] == [20261824]
    with pytest.raises(ValueError, match="next replan"):
        on.next_seed(1, 0)
    with pytest.raises(ValueError, match="1000"):
        main_inference_seed(1, 0, 1000)
    with pytest.raises(ValueError, match="uint32"):
        main_inference_seed(2**32 - 1, 1, 0)


def test_actual_deployment_inference_passes_main_seed_to_model_without_continuous_noise():
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy

    requests = []

    def infer_action(**kwargs):
        requests.append(kwargs)
        generator = torch.Generator(device=kwargs["rand_device"]).manual_seed(kwargs["seed"])
        return {"action": torch.randn(1, 32, 14, generator=generator)}

    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy._main_observed = True
    policy._main_warmup_requests = 0
    policy.model = SimpleNamespace(device="cpu", torch_dtype=torch.float32, infer_action=infer_action)
    policy._main_seed_schedule = MainAttemptSeedSchedule(21)
    policy._main_seed_schedule.begin_attempt(1)
    policy._model_noise_stream = None
    policy._model_seed_stream = SimpleNamespace(next_seed=lambda: pytest.fail("main consumed the old random seed stream"))
    policy._attempt_id, policy._replan_index = 1, 0
    policy.zeva_mode, policy.action_horizon = "base", 32
    policy._build_robotwin_image_tensor = lambda observation: torch.zeros(1, 3, 16, 16)
    policy._normalize_state = lambda state: torch.as_tensor(state).unsqueeze(0)
    policy._policy_prompt_cache = SimpleNamespace(encode=lambda *args, **kwargs: (torch.zeros(1, 1, 1), None))
    policy._denormalize_action = lambda action: action.numpy()
    policy.negative_prompt, policy.text_cfg_scale, policy.num_inference_steps = "", 1., 10
    policy.sigma_shift, policy.rand_device, policy.tiled, policy.timing_enabled = None, "cpu", False, False
    observation = {"joint_action": {"vector": np.zeros(14)}}
    first = policy._infer_action_chunk(observation, "task")
    second = policy._infer_action_chunk(observation, "task")
    assert [request["seed"] for request in requests] == [1021, 1021, 1022]
    assert all("initial_action_noise" not in request for request in requests)
    assert policy._main_seed_schedule.snapshot()["request_seeds"] == [1021, 1022]
    assert policy._main_warmup_requests == 1 and policy._main_warmup_seed == 1021
    # A control produced after warmup is the unmodified no-warmup result for
    # this seed; the discarded forward never advanced the request schedule.
    expected = torch.randn(1, 32, 14, generator=torch.Generator().manual_seed(1021)).numpy()[0]
    np.testing.assert_array_equal(first, expected)
    assert not np.array_equal(first, second)


def test_main_rgb_normalizes_before_bfloat16_cast_matching_training(monkeypatch):
    from experiments.robotwin.fastwam_policy import deploy_policy

    # Constant tiles isolate dtype ordering from the separately defined
    # bilinear resize. This invokes the actual policy mosaic/normalization.
    policy = deploy_policy.WorldActionRobotWinPolicy.__new__(deploy_policy.WorldActionRobotWinPolicy)
    policy._main_observed = True
    policy.model = SimpleNamespace(device="cpu", torch_dtype=torch.bfloat16)
    observation = {"observation": {
        name: {"rgb": np.full((1, 1, 3), 128, dtype=np.uint8)}
        for name in ("head_camera", "left_camera", "right_camera")
    }}
    # Preserve zero-one float tiles on the second resize pass, like the real
    # resize helper; it only divides by 255 for raw byte/range inputs.
    def resize(image, size_wh):
        value = np.asarray(image, dtype=np.float32)[0, 0]
        if float(value.max()) > 1:
            value = value / 255.
        return np.broadcast_to(value, (size_wh[1], size_wh[0], 3)).copy()

    monkeypatch.setattr(deploy_policy, "_resize_rgb", resize)
    actual = policy._build_robotwin_image_tensor(observation)
    expected = (torch.tensor(128 / 255., dtype=torch.float32) * 2 - 1).bfloat16()
    assert actual.dtype == torch.bfloat16 and actual.shape == (1, 3, 384, 320)
    assert torch.equal(actual, expected.expand_as(actual))
    assert expected != torch.tensor(128 / 255., dtype=torch.bfloat16) * 2 - 1
