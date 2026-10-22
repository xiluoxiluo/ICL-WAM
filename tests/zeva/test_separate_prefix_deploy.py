"""Deployment keeps completed BIT effects separate from persistent memory."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
from fastwam.zeva.attempt_protocol import ModelSeedStream
from fastwam.zeva.prompt_cache import PolicyPromptCache


class _PrefixAdapter:
    config = SimpleNamespace(
        pim_variant="cross_attempt", prefix_layout="zeva_ego", effect_source="latest_completed",
    )

    def __init__(self):
        self.prefix_calls = []

    def prior_for_prompt(self, task, phase, bit, bit_mask, prompt, pim_mask, *, enable_pim=True):
        mean = torch.zeros(task.shape[0], 32, 14)
        if enable_pim:
            mean += pim_mask.any(-1)[:, None, None]
        return mean, torch.ones_like(mean)

    def action_prior_residual(self, mean, *, training):
        return mean + 0.25

    def behavior_prefix_slot(self, *args, **kwargs):
        raise AssertionError("separate-prefix diagnostics must not merge its three tokens")

    def behavior_prefix_tokens(self, task, prompt, pim_mask, *, bit_effects, bit_mask, enable_pim=True):
        self.prefix_calls.append((bit_effects.clone(), bit_mask.clone(), enable_pim))
        batch = task.shape[0]
        valid = torch.stack((
            torch.ones(batch, dtype=torch.bool),
            bit_mask.any(-1),
            pim_mask.any(-1) & enable_pim,
        ), dim=1)
        # Invalid slots deliberately contain NaNs: diagnostics should use
        # their attention validity instead of reporting unavailable content.
        tokens = torch.full((batch, 3, 8), float("nan"))
        tokens[:, 0] = 1.0 + valid[:, 2, None]
        tokens[valid[:, 1], 1] = 3.0
        tokens[valid[:, 2], 2] = 5.0
        return tokens, valid


def _runtime_policy(memory, *, mode="pim_on", cross_memory=True):
    requests = []
    adapter = _PrefixAdapter()

    class PromptEncoder:
        config = SimpleNamespace(global_dim=4, hidden_dim=6)

        def __call__(self, **kwargs):
            if mode != "pim_on" or not cross_memory:
                raise AssertionError("disabled PIM must not compute its readout")
            return torch.ones(1, 6)

    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.model = SimpleNamespace(
        device="cpu", torch_dtype=torch.float32, zeva_injection_mode="exact_zeva",
        zeva_prompt_encoder=PromptEncoder(), zeva_behavior_prefix_adapter=adapter,
        encode_prompt=lambda _: (torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)),
        infer_action_zeva=lambda **kwargs: requests.append(kwargs) or {"action": torch.zeros(1, 32, 14)},
    )
    policy._policy_prompt_cache = PolicyPromptCache(policy.model.encode_prompt)
    policy.zeva_mode, policy.cross_memory_enabled = mode, cross_memory
    policy._static_task_context_session, policy.task_context_bank = None, None
    # This sentinel must never be mistaken for the completed effect used
    # by the prefix; only lifecycle.memory_inputs owns the causal BIT.
    policy._cte_history = SimpleNamespace(forward=lambda: {
        "phase": torch.ones(1, 2, 3), "effect_post": torch.full((1, 1, 2), 999.0),
    })
    policy._needs_episode_reset = False
    policy.lifecycle = SimpleNamespace(memory_inputs=lambda phase, task: memory)
    policy._build_robotwin_image_tensor = lambda obs: torch.zeros(1, 3, 8, 8)
    policy._normalize_state = lambda state: torch.zeros(1, 14)
    policy._denormalize_action = lambda action: action.numpy()
    policy._model_seed_stream = ModelSeedStream(12)
    policy._attempt_id, policy._replan_index = 1, 0
    policy.action_horizon, policy.replan_steps, policy.num_inference_steps = 32, 24, 10
    policy.negative_prompt, policy.text_cfg_scale, policy.sigma_shift = "", 1.0, None
    policy.rand_device, policy.tiled, policy.timing_enabled = "cpu", False, False
    policy.pim_diagnostic_interval = 0
    return policy, requests


@pytest.mark.parametrize("mode,cross_memory", [("pim_on", True), ("pim_on", False), ("zeva_stage2", True)])
@pytest.mark.parametrize("has_completed_effect", [False, True])
def test_action_request_passes_completed_bit_even_when_pim_is_off(mode, cross_memory, has_completed_effect):
    memory = {
        "bit_effects": torch.tensor([[[0.0, 0.0], [7.0, 8.0], [0.0, 0.0], [9.0, 10.0]]]),
        "bit_mask": torch.tensor([[False, has_completed_effect, False, has_completed_effect]]),
        "pim_mask": torch.ones(1, 192, dtype=torch.bool),
    }
    policy, requests = _runtime_policy(memory, mode=mode, cross_memory=cross_memory)
    action = policy._infer_action_chunk({"joint_action": {"vector": np.zeros(14)}}, "task")
    assert action.shape == (32, 14)
    assert len(requests) == 1
    torch.testing.assert_close(requests[0]["zeva_bit_effects"], memory["bit_effects"])
    torch.testing.assert_close(requests[0]["zeva_bit_mask"], memory["bit_mask"])
    assert requests[0]["enable_prefix_injection"] is True
    assert requests[0]["enable_action_prior"] is True
    assert bool(requests[0]["pim_mask"].any()) == (mode == "pim_on" and cross_memory)
    assert policy._model_seed_stream.draws == 1


def test_paired_probe_keeps_effect_visibility_and_handles_masked_prefix_slots():
    adapter, requests = _PrefixAdapter(), []
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.replan_steps = 24
    policy.model = SimpleNamespace(
        device="cpu", zeva_injection_mode="exact_zeva", zeva_behavior_prefix_adapter=adapter,
        infer_action_zeva=lambda **kwargs: requests.append(kwargs) or {"action": torch.zeros(2, 32, 14)},
    )
    memory = {
        "bit_effects": torch.ones(2, 4, 2),
        "bit_mask": torch.tensor([[False, True, False, True], [False, False, False, False]]),
    }
    pim_mask = torch.tensor([[True, False], [False, False]])
    policy._measure_pim_effect(
        {"seed": 123}, {"action": torch.zeros(2, 32, 14)},
        torch.zeros(2, 4), torch.zeros(2, 3), memory, torch.ones(2, 6), pim_mask,
    )
    assert len(requests) == 1 and requests[0]["seed"] == 123
    assert not requests[0]["pim_mask"].any()
    torch.testing.assert_close(requests[0]["zeva_bit_effects"], memory["bit_effects"])
    torch.testing.assert_close(requests[0]["zeva_bit_mask"], memory["bit_mask"])
    assert [entry[2] for entry in adapter.prefix_calls] == [True, False]
    for effects, valid, _ in adapter.prefix_calls:
        torch.testing.assert_close(effects, memory["bit_effects"])
        torch.testing.assert_close(valid, memory["bit_mask"])
    metrics = policy._last_pim_diagnostics
    assert metrics["prefix_layout"] == "zeva_ego"
    assert metrics["effect_source"] == "latest_completed"
    assert metrics["prefix_valid_tokens_on"] == 2.0
    assert metrics["prefix_valid_tokens_off"] == 1.5
    assert metrics["effect_token_valid_samples"] == 1
    assert metrics["pim_token_valid_samples"] == 1
    assert np.isfinite(metrics["prefix_delta_rms"])
    assert np.isfinite(metrics["prefix_relative_delta"])


def test_rollout_metadata_records_loaded_prefix_contract():
    policy, _ = _runtime_policy({})
    policy.seed = 12
    policy.checkpoint_identity = {"addon_sha256": "three-token-checkpoint"}
    policy.lifecycle = None
    policy._terminal_observation_step = None
    policy._last_pim_diagnostics = {}
    metadata = policy.evaluation_metadata()
    assert metadata["prefix_layout"] == "zeva_ego"
    assert metadata["effect_source"] == "latest_completed"
    assert metadata["checkpoint_identity"]["addon_sha256"] == "three-token-checkpoint"


@pytest.mark.parametrize("resolved_config", [False, True])
@pytest.mark.parametrize("layout_override", [None, "legacy_merged"])
def test_plugin_entrypoint_preserves_or_explicitly_overrides_prefix_contract(
    tmp_path, monkeypatch, resolved_config, layout_override,
):
    from omegaconf import OmegaConf
    from experiments.robotwin.fastwam_policy import deploy_policy

    cfg = OmegaConf.create({
        "runtime_schema": "iclwam-robotwin-runtime-v1",
        "model": {"zeva": {"adapter": {"prefix_layout": "zeva_ego", "effect_source": "latest_completed"}}},
        "data": {"train": {
            "processor": {}, "num_frames": 33, "action_video_freq_ratio": 4, "video_size": [384, 320],
        }},
        "EVALUATION": {"device": "cpu", "replan_steps": 24, "zeva_mode": "base"},
        "mixed_precision": "no", "eval_num_inference_steps": 10,
    })
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    args = {"ckpt_setting": "base.pt", "dataset_stats_path": str(stats)}
    if resolved_config:
        path = tmp_path / "runtime.yaml"
        OmegaConf.save(cfg, path)
        args["runtime_config_path"] = str(path)
        monkeypatch.setattr(deploy_policy, "_compose_sim_cfg", lambda **_: pytest.fail("must use resolved model config"))
    else:
        monkeypatch.setattr(deploy_policy, "_compose_sim_cfg", lambda **_: cfg)
    if layout_override is not None:
        args.update(prefix_layout=layout_override, effect_source="latest_completed")
    monkeypatch.setattr(deploy_policy, "WorldActionRobotWinPolicy", lambda **kwargs: kwargs)
    result = deploy_policy.get_model(args)
    assert result["model_cfg"].zeva.adapter.prefix_layout == (layout_override or "zeva_ego")
    assert result["model_cfg"].zeva.adapter.effect_source == "latest_completed"
