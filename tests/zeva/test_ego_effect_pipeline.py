"""Causal prediction semantics across RGB episodes, cache, BIT/PIM and serving."""
from collections import deque
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fastwam.zeva.cross_attempt_cache import EGO_CROSS_ATTEMPT_CACHE_FORMAT, CrossAttemptPIMCache, file_sha256
from fastwam.zeva.ego_boundary_cache import EgoBoundaryCache, build_ego_boundary_cache
from fastwam.zeva.ego_episode_data import EGO_CAMERA_KEYS, RoboTwinEgoEpisodeDataset, collate_ego_episodes
from fastwam.zeva.lifecycle import CausalMemoryLifecycle, LifecycleConfig
from fastwam.zeva.memory import PersistentInteractionMemory, PersistentInteractionMemoryConfig
from fastwam.zeva.prepared_stage2 import PreparedPIMCache, PreparedZevaStage2Dataset, StaticEpisodeContextCache, STATIC_CONTEXT_FORMAT

SPEC = importlib.util.spec_from_file_location("ego_build_cross", Path(__file__).resolve().parents[2] / "scripts/build_zeva_cross_attempt_cache.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class _Table:
    column_names = ["task_index"]

    def with_format(self, _):
        return self

    def __getitem__(self, indices):
        return {"task_index": [0] * len(indices)}


class _Inner:
    def __init__(self, lengths):
        self.lengths = lengths
        self.meta = SimpleNamespace(total_episodes=len(lengths))
        self.episodes = None
        self.num_frames = sum(lengths)
        self.root = "/source"
        ends = np.cumsum(lengths)
        self.episode_data_index = {"from": np.r_[0, ends[:-1]], "to": ends}
        self.hf_dataset = _Table()
        self.delta_indices = {f"observation.images.{key}": list(range(33)) for key in EGO_CAMERA_KEYS}
        self.delta_timestamps = {key: list(map(float, value)) for key, value in self.delta_indices.items()}
        self.calls = []
        self.substitute = False

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.episode_data_index["to"], index, side="right"))
        raw = index - int(self.episode_data_index["from"][episode])
        self.calls.append((episode, raw))
        result = {"episode_index": episode + int(self.substitute), "frame_index": raw,
                  "action": torch.arange(raw, raw + 32).float()[:, None].expand(32, 14).clone(),
                  "observation.state": torch.zeros(33, 14),
                  "action_is_pad": torch.arange(raw, raw + 32) >= self.lengths[episode] - 1}
        for view, key in enumerate(EGO_CAMERA_KEYS):
            camera = f"observation.images.{key}"
            result[camera] = torch.full((len(self.delta_indices[camera]), 3, 7, 9), (view + 1) / 4)
            result[camera + "_is_pad"] = torch.zeros(len(self.delta_indices[camera]), dtype=torch.bool)
        return result


class _Multi:
    def __init__(self, inner):
        self._datasets = [inner]

    def __getitem__(self, index):
        return self._datasets[0][index]


class _Processor:
    use_stepwise_action_norm = False

    def __init__(self):
        self.normalizer = SimpleNamespace(forward=self.normalize)
        self.action_state_merger = SimpleNamespace(forward=self.merge)

    def action_state_transform(self, value):
        return value

    def normalize(self, value):
        value["action"]["default"] = (value["action"]["default"] - 2) / 4
        return value

    def merge(self, value):
        value["action"] = value["action"]["default"]
        value["state"] = value["state"]["default"]
        return value


def make_dataset(lengths=(81, 81, 81)):
    inner = _Inner(lengths)
    source = SimpleNamespace(
        multi_dataset=_Multi(inner), processor=_Processor(), action_size=32, global_sample_stride=1,
        image_meta=[{"key": key, "lerobot_key": f"observation.images.{key}"} for key in EGO_CAMERA_KEYS],
        action_meta=[{"key": "default", "lerobot_key": "action"}],
        state_meta=[{"key": "default", "lerobot_key": "observation.state"}],
        _split_lerobot_sample=lambda raw: raw,
        _get_action=lambda meta, raw: raw[meta["lerobot_key"]],
        _get_state=lambda meta, raw: raw[meta["lerobot_key"]],
    )
    base = SimpleNamespace(lerobot_dataset=source, semantic_task_identity={"sha256": "semantic"},
                           resolve_semantic_task_id=lambda **_: "beat_block_hammer")
    return RoboTwinEgoEpisodeDataset(base), inner


class _CausalCTE(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(executed_horizon=24, transition_steps=24, effect_window_transitions=1,
                                   image_size=224, action_dim=14)
        self.step_calls = []
        self.task_head = lambda phase: phase + 40

    def forward(self, images, executed_actions, valid):
        raise AssertionError("formal feature cache must use the same recurrent step API as deployment")

    def step(self, images, previous, cache):
        self.step_calls.append((images.clone(), None if previous is None else previous.clone()))
        index = 1 if cache is None else cache["index"] + 1
        phase = self.marker.new_tensor([[index, index + 1]])
        return phase, phase + 20, {"index": index}


def make_boundary_cache(tmp_path):
    dataset, _ = make_dataset()
    ids = [episode.episode_id for episode in dataset.episodes]
    splits = {"train": ids[:2], "validation": ids[2:]}
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({"format": "robotwin_semantic_task_map_v1",
                                   "episode_to_semantic_task": {str(i): "beat_block_hammer" for i in range(3)}}))
    output = tmp_path / "boundaries"
    build_ego_boundary_cache(dataset, _CausalCTE(), output,
        identity={"cte_checkpoint_sha256": "cte", "dataset_stats_sha256": "stats",
                  "semantic_task_map_sha256": file_sha256(semantic), "dataset_path": "/source"},
        episode_splits=splits)
    return output, semantic


def test_episode_reader_uses_independent_views_and_only_full_executed_chunks():
    dataset, original = make_dataset((81, 49))
    first, short = dataset[0], dataset[1]
    assert first["images"].shape == (4, 3, 3, 224, 224)
    torch.testing.assert_close(first["images"][0].mean((1, 2, 3)), torch.tensor([-.5, 0., .5]))
    assert first["raw_steps"].tolist() == [0, 24, 48, 72]
    assert first["trailing_action_count"] == 8
    assert first["policy_valid"].tolist() == [True, True, True, False]
    assert short["decision_valid"].tolist() == [True, True, False]
    assert torch.equal(first["executed_actions"][0, :, 0], (torch.arange(24) - 2) / 4)
    assert all(len(value) == 33 for value in original.delta_indices.values())
    assert all(len(value) == 1 for value in dataset._raw_dataset._datasets[0].delta_indices.values())
    batch = collate_ego_episodes([first, short])
    assert batch["valid"].tolist() == [[True, True, True], [True, True, False]]
    assert not batch["action_valid"][1, -1].any()
    assert not batch["boundary_valid"][1, -1]
    dataset._raw_dataset._datasets[0].substitute = True
    with pytest.raises(ValueError, match="substituted"):
        dataset[0]


def test_prediction_cache_preserves_current_effect_and_train_only_history(tmp_path):
    output, semantic = make_boundary_cache(tmp_path)
    boundary = EgoBoundaryCache(output)
    assert boundary.get(0)["current_effect"].tolist() == [21., 22.]
    assert boundary.get(0)["global"].shape == (2,)
    assert boundary.manifest.encoding_path == "recurrent_step_v1"
    assert boundary.get(0)["retrieval"].shape == (2,)
    prepared_root = tmp_path / "prepared"
    manifest = BUILDER.build_cache(phase_cache=output, output=prepared_root, semantic_map=semantic,
                                   max_entries_per_attempt=4, allow_reduced_curriculum=True)
    assert manifest["format"] == EGO_CROSS_ATTEMPT_CACHE_FORMAT
    assert manifest["memory_semantics"] == "predicted_effect_trace"
    assert manifest["globally_available_history_kinds"] == ["matched", "far"]
    cache = CrossAttemptPIMCache(prepared_root)
    for index in range(len(cache)):
        row = cache.get(index, history_kind="matched", history_count=1)
        assert row["current_effect_mask"]
        assert row["bit_mask"].sum() == row["raw_step"] // 24
        assert row["episode_id"] not in row["pim_source_episode_ids"]
        assert set(row["pim_source_episode_ids"]) <= set(boundary.splits["train"])
        assert row["pim_mask"].sum() == 4
        assert not torch.any(row["pim_effects"] == 999)
        if row["raw_step"] == 24:
            torch.testing.assert_close(row["bit_effects"][-1], torch.tensor([21., 22.]))
            assert not torch.equal(row["current_effect"], row["bit_effects"][-1])
    rng_before = torch.get_rng_state().clone()
    bundle = cache.get_history_candidates(0)
    assert torch.equal(torch.get_rng_state(), rng_before)
    torch.manual_seed(987654)  # a resumed worker may have unrelated RNG state
    resumed = cache.get_history_candidates(0)
    for name in ("pim_history_phases", "pim_history_effects", "pim_history_masks"):
        torch.testing.assert_close(bundle[name], resumed[name])
    assert bundle["pim_history_available"].tolist() == [True, True, False]
    assert bundle["pim_history_phases"].shape == (3, 12, 2)
    assert bundle["pim_history_masks"].sum(-1).tolist() == [4, 4, 0]
    assert not bundle["pim_mask"].any()
    wrong = tmp_path / "split.json"
    wrong.write_text(json.dumps({"train": boundary.splits["train"] + boundary.splits["validation"], "validation": []}))
    with pytest.raises(ValueError, match="preserve"):
        BUILDER.build_cache(phase_cache=output, output=tmp_path / "bad", semantic_map=semantic, episode_splits=wrong)

    # The public prepared dataset carries both explicit current effect and the
    # optimizer-step candidates to the trainer, without substituting a sample.
    prepared = PreparedPIMCache(prepared_root)
    lookup = {prepared.get(i)["window_index"]: prepared.get(i) for i in range(len(prepared))}
    class Base:
        def __getitem__(self, index):
            row = lookup[index]
            return {"video": torch.zeros(3, 9, 2, 2), "action": torch.zeros(32, 14),
                    "dataset_index": index, "episode_id": row["episode_id"], "task_id": row["task_id"],
                    "episode_step": row["raw_step"], "episode_num_steps": 81}
    static = tmp_path / "static.pt"
    torch.save({"format": STATIC_CONTEXT_FORMAT, "episode_ids": list(prepared.episode_ids),
                "task_contexts": torch.ones(len(prepared.episode_ids), 3)}, static)
    sample = PreparedZevaStage2Dataset(Base(), prepared, StaticEpisodeContextCache(static),
                                      history_schedule="ego_optimizer_step")[0]
    assert sample["current_effect_mask"] and sample["pim_history_available"].tolist() == [True, True, False]
    assert sample["current_effect"].tolist() == [21., 22.]


def _lifecycle():
    memory = PersistentInteractionMemory(PersistentInteractionMemoryConfig(
        phase_dim=2, effect_dim=2, capacity=6, top_k=6, cross_attempt_only=True,
        max_past_attempts=3, max_entries_per_attempt=2))
    return CausalMemoryLifecycle(memory, LifecycleConfig(transition_steps=24, effect_window_transitions=1,
                                                         memory_semantics="predicted_effect_trace"))


def test_predicted_memory_is_private_until_attempt_end_and_cannot_mix_observed_effects():
    lifecycle = _lifecycle()
    lifecycle.reset_episode("task")
    phase, predicted = torch.tensor([1., 2.]), torch.tensor([21., 22.])
    before = lifecycle.memory_inputs(phase[None], torch.zeros(1, 4))
    assert not before["bit_mask"].any() and not before["pim_mask"].any()
    lifecycle.observe_predicted_effect(phase, predicted)
    after = lifecycle.memory_inputs(phase[None], torch.zeros(1, 4))
    assert after["bit_mask"].sum() == 1 and not after["pim_mask"].any()
    assert lifecycle.pim.pending_count == 1
    with pytest.raises(ValueError, match="observed completed effects"):
        lifecycle.observe_completed_effect(phase, predicted)
    lifecycle.end_attempt()
    lifecycle.reset_attempt(1)
    retry = lifecycle.memory_inputs(phase[None], torch.zeros(1, 4))
    assert not retry["bit_mask"].any() and retry["pim_mask"].sum() == 1
    assert lifecycle.pim.entries[0].metadata["memory_semantics"] == "predicted_effect_trace"


def test_deployment_predicts_once_per_replan_reuses_noise_and_commits_no_terminal_query():
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
    from fastwam.zeva.attempt_protocol import ModelSeedStream, ContinuousModelNoiseStream
    requests = []
    class Adapter:
        config = SimpleNamespace(pim_variant="cross_attempt", prefix_layout="zeva_ego",
                                 effect_source="current_prediction", prior_variant="zeva_ego")
        def prior_for_prompt(self, task, *args, **kwargs):
            return torch.zeros(1, 32, 14), torch.ones(1, 32, 14)
        def action_prior_residual(self, mean, **kwargs):
            return mean
        def behavior_prefix_tokens(self, task, prompt, pim_mask, *, bit_effects, bit_mask,
                                   current_effect, current_effect_mask, enable_pim=True):
            assert current_effect_mask.all()
            valid = torch.stack((current_effect_mask, current_effect_mask,
                                 pim_mask.any(-1) & enable_pim), dim=1)
            return torch.ones(1, 3, 8), valid
    class Prompt:
        config = SimpleNamespace(global_dim=4, hidden_dim=6)
        def __call__(self, **kwargs):
            return torch.ones(1, 6)
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.model = SimpleNamespace(device="cpu", torch_dtype=torch.float32, zeva_injection_mode="exact_zeva",
        zeva_behavior_prefix_adapter=Adapter(), zeva_prompt_encoder=Prompt(),
        encode_prompt=lambda _: (torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)),
        infer_action_zeva=lambda **kw: requests.append(kw) or {"action": torch.full((1, 32, 14), 2.)})
    from fastwam.zeva.prompt_cache import PolicyPromptCache
    policy._policy_prompt_cache = PolicyPromptCache(policy.model.encode_prompt)
    policy.cte, policy.lifecycle = _CausalCTE(), _lifecycle()
    # Policy inference may be bf16, but persisted CTE traces must retain the
    # original float32 features exported by the offline recurrent cache.
    policy.model.torch_dtype = torch.bfloat16
    append_prediction = policy.lifecycle.observe_predicted_effect
    def preserve_cte_precision(phase, effect, **kwargs):
        assert phase.dtype == effect.dtype == torch.float32
        append_prediction(phase, effect, **kwargs)
    policy.lifecycle.observe_predicted_effect = preserve_cte_precision
    policy._ego_cte, policy._ego_has_started, policy._cte_state = True, False, None
    policy._cte_history, policy._static_task_context_session, policy.task_context_bank = None, None, None
    policy._needs_episode_reset, policy._attempt_finalized = True, False
    policy.zeva_mode, policy.cross_memory_enabled, policy.pim_diagnostic_interval = "pim_on", True, 1
    policy._transition_actions, policy.pending_actions = [], deque()
    policy._attempt_id = policy._replan_index = policy.episode_count = policy.step_count = 0
    policy.action_horizon, policy.replan_steps, policy.num_inference_steps = 32, 24, 10
    policy.negative_prompt, policy.text_cfg_scale, policy.sigma_shift = "", 1., None
    policy.rand_device, policy.tiled, policy.timing_enabled = "cpu", False, False
    policy._model_seed_stream, policy._model_noise_stream = ModelSeedStream(11), ContinuousModelNoiseStream(11)
    policy._build_robotwin_image_tensor = lambda _: torch.zeros(1, 3, 8, 8)
    policy._normalize_state = lambda _: torch.zeros(1, 14)
    policy._normalize_action = lambda action, state: torch.as_tensor(action) / 2
    policy._denormalize_action = lambda action: action.numpy()
    observation = {"joint_action": {"vector": np.zeros(14)}, "observation": {
        name: {"rgb": np.full((7, 9, 3), 64 + index * 64, np.uint8)}
        for index, name in enumerate(("head_camera", "left_camera", "right_camera"))}}
    env = SimpleNamespace(get_instruction=lambda: "task", take_action=lambda *a, **kw: None,
                          get_obs=lambda: observation)
    for step in range(25):
        policy.step(env, observation if step % 24 == 0 else None)
    assert len(policy.cte.step_calls) == 2
    assert policy.cte.step_calls[0][1] is None
    torch.testing.assert_close(policy.cte.step_calls[1][1], torch.ones(1, 24, 14))
    assert len(requests) == 4  # two replans, each with one paired diagnostic
    assert not requests[0]["zeva_bit_mask"].any()
    assert requests[2]["zeva_bit_mask"].sum() == 1
    assert requests[0]["zeva_current_effect"].tolist() == [[21., 22.]]
    assert requests[2]["zeva_current_effect"].tolist() == [[22., 23.]]
    assert requests[0]["initial_action_noise"] is requests[1]["initial_action_noise"]
    assert requests[2]["initial_action_noise"] is requests[3]["initial_action_noise"]
    assert all(request["seed"] is None for request in requests)
    assert policy._model_noise_stream.draws == 2 and policy._model_seed_stream.draws == 0
    assert policy.lifecycle.pim.pending_count == 2 and len(policy.lifecycle.pim) == 0
    policy.finalize_attempt(env)
    assert len(policy.cte.step_calls) == 2  # terminal observation is not a new decision
    assert policy._terminal_observation_step == 25
    assert policy.lifecycle.pim.pending_count == 0 and len(policy.lifecycle.pim) == 2


def test_skipped_observations_require_state_independent_prediction_mode():
    from experiments.robotwin.fastwam_policy.deploy_policy import _supports_boundary_observations
    model = {"zeva": {"adapter": {"effect_source": "current_prediction"}}}
    assert _supports_boundary_observations(model, {"action_state_transforms": None})
    assert not _supports_boundary_observations(model, {"action_state_transforms": ["delta"]})
    assert not _supports_boundary_observations(model, {"use_stepwise_action_norm": True})
    assert not _supports_boundary_observations({"zeva": {}}, {"action_state_transforms": None})
