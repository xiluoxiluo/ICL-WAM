"""Audit original-main semantics through the real WAM training/inference calls."""

from dataclasses import replace

import pytest
import torch

from fastwam.zeva.behavior_prefix_adapter import gaussian_prior_nll
from test_main_model_contract import _prior, _prompt, main_model
from test_separate_prefix_integration import _sample


def test_main_initial_readout_pools_before_widening_policy_hidden(monkeypatch):
    model = main_model().eval()
    # The FP32 mean has low bits which BF16 pooling intentionally rounds in
    # the original extract_batch call. Equal token values would miss this.
    hidden = torch.tensor([1.0, 1.0078125, 1.0, 1.0], dtype=torch.bfloat16).reshape(1, 4, 1).expand(1, 4, 12)
    monkeypatch.setattr(model.mot, "prefill_video_cache_tensor", lambda **kwargs: ([], [], hidden))
    arguments = (torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 6), torch.ones(1, 3, dtype=torch.bool))
    actual = model.extract_task_context_readout(*arguments)
    torch.testing.assert_close(actual, hidden.mean(dim=1).float(), rtol=0, atol=0)
    assert actual.dtype == torch.float32 and not actual.requires_grad
    model.zeva_profile = "legacy"
    legacy = model.extract_task_context_readout(*arguments)
    torch.testing.assert_close(legacy, hidden.float().mean(dim=1), rtol=0, atol=0)
    assert not torch.equal(actual, legacy)


@pytest.mark.parametrize("stage", ["policy_injection", "pim_adapter"])
def test_main_direct_forward_nll_preserves_bf16_target_and_prior_compute_dtype(monkeypatch, stage):
    from fastwam.models.wan22 import fastwam as implementation

    model, sample = main_model(stage).bfloat16(), _sample()
    model.torch_dtype = torch.bfloat16
    model.prepare_zeva_training_precision()
    clean = torch.full_like(sample["action"], 0.123456)
    clean[:, 8:] = 1.012345
    valid = sample["action_valid"].clone()
    valid[:, 8:] = False
    mean = torch.full_like(clean, 0.05, dtype=torch.bfloat16)
    std = torch.full_like(clean, 0.7, dtype=torch.bfloat16)
    captured = []

    def nll(target, actual_mean, actual_std, actual_valid, **kwargs):
        captured.append((target, actual_mean, actual_std, actual_valid))
        return gaussian_prior_nll(target, actual_mean, actual_std, actual_valid, **kwargs)

    monkeypatch.setattr(implementation, "gaussian_prior_nll", nll)
    _, metrics = model.forward_zeva_action_train(
        input_image=sample["video"][:, :, 0], clean_action=clean,
        context=sample["context"], context_mask=sample["context_mask"],
        causal_prompt=torch.zeros(3, 8), pim_mask=torch.zeros_like(sample["pim_mask"]),
        action_valid=valid, zeva_prior_mean=mean, zeva_prior_std=std,
        zeva_action_residual=torch.zeros(3, 32, 8, dtype=torch.bfloat16),
        zeva_task_tokens=sample["task_context"],
    )
    target, actual_mean, actual_std, actual_valid = captured[0]
    assert target.dtype == actual_mean.dtype == actual_std.dtype == torch.bfloat16
    torch.testing.assert_close(target, clean.bfloat16(), rtol=0, atol=0)
    assert actual_mean is mean and actual_std is std and torch.equal(actual_valid, valid)
    expected = gaussian_prior_nll(clean.bfloat16(), mean, std, valid)
    old_fp32_target = gaussian_prior_nll(clean, mean.float(), std.float(), valid)
    assert expected.item() != old_fp32_target.item()
    assert metrics["behavior_prior_nll"] == expected.item()


@pytest.mark.parametrize("joint", [False, True])
def test_main_pim_keeps_parent_nll_as_a_constant_without_changing_gradients(joint):
    model, sample = main_model(joint=joint), _sample()
    adapter = model.zeva_behavior_prefix_adapter
    with torch.no_grad():
        adapter.pim_gate.fill_(0.2)
    rng = torch.get_rng_state()
    loss, metrics = model(sample)
    loss.backward()
    gradients = {name: None if value.grad is None else value.grad.clone()
                 for name, value in model.named_parameters() if value.requires_grad}
    assert metrics["behavior_prior_nll"] != 0
    assert loss.item() == pytest.approx(metrics["loss_flow"] + 0.01 * metrics["behavior_prior_nll"])
    assert all(parameter.grad is None for parameter in adapter.policy_injection_parameters())

    model.zero_grad(set_to_none=True)
    adapter.config = replace(adapter.config, prior_loss_weight=0.0)
    torch.set_rng_state(rng)
    flow_only, flow_metrics = model(sample)
    flow_only.backward()
    assert flow_only.item() == flow_metrics["loss_flow"]
    assert flow_metrics["behavior_prior_nll"] == metrics["behavior_prior_nll"]
    for name, value in model.named_parameters():
        if value.requires_grad:
            expected = gradients[name]
            if expected is None:
                assert value.grad is None
            else:
                torch.testing.assert_close(value.grad, expected, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in adapter.policy_injection_parameters())


@pytest.mark.parametrize("stage", ["policy_injection", "pim_adapter"])
def test_main_training_passes_mean_and_example_dropout_to_the_real_action_branch(monkeypatch, stage):
    model, sample = main_model(stage), _sample()
    adapter = model.zeva_behavior_prefix_adapter
    adapter.config = replace(adapter.config, prior_dropout_rate=0.4)
    with torch.no_grad():
        adapter.action_prior_adapter.weight.normal_(std=0.1)
    captured = {}
    adapter.prior.register_forward_hook(
        lambda module, args, output: captured.update(mean=output[0].detach(), std=output[1].detach()))
    adapter.action_prior_adapter.register_forward_hook(
        lambda module, args, output: captured.update(adapter_input=args[0].detach(), raw_residual=output.detach()))
    native_rand = torch.rand
    draws = []

    def draw(*args, **kwargs):
        if args and args[0] == (3, 1, 1):
            draws.append(True)
            return torch.tensor([0.399, 0.4, 0.99], device=kwargs.get("device")).reshape(3, 1, 1)
        return native_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", draw)
    native_denoise = model._denoise_action_with_video_cache_zeva

    def denoise(**kwargs):
        captured["action_residual"] = kwargs["zeva_action_residual"].detach()
        return native_denoise(**kwargs)

    monkeypatch.setattr(model, "_denoise_action_with_video_cache_zeva", denoise)
    _, metrics = model(sample)
    assert len(draws) == 1
    torch.testing.assert_close(captured["adapter_input"], captured["mean"], rtol=0, atol=0)
    expected = captured["raw_residual"] * torch.tensor([0, 1, 1]).reshape(3, 1, 1)
    torch.testing.assert_close(captured["action_residual"], expected, rtol=0, atol=0)
    assert metrics["behavior_prior_nll"] == pytest.approx(gaussian_prior_nll(
        sample["action"], captured["mean"], captured["std"], sample["action_valid"]).item())
    # NLL includes the dropped row. Serving always uses the mean with a
    # 0.5 multiplier, with neither resampling nor inverted-dropout scaling.
    result = adapter.action_prior_residual(captured["mean"], training=False)
    torch.testing.assert_close(result, captured["raw_residual"] * 0.5, rtol=0, atol=0)
    assert len(draws) == 1


@pytest.mark.parametrize("stage", ["policy_injection", "pim_adapter"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_main_training_and_inference_share_conditioned_first_denoise(monkeypatch, stage, dtype):
    model = main_model(stage).eval()
    model.to(dtype=dtype)
    model.torch_dtype = dtype
    model.prepare_zeva_inference_precision()
    sample = {key: value[-1:] if isinstance(value, torch.Tensor) else value for key, value in _sample().items()}
    reader, adapter = model.zeva_prompt_encoder, model.zeva_behavior_prefix_adapter
    with torch.no_grad():
        adapter.action_prior_adapter.weight.normal_(std=0.1)
        adapter.pim_gate.fill_(0.2)
    if stage == "policy_injection":
        prompt = torch.zeros(1, 8)
        pim_mask = torch.zeros_like(sample["pim_mask"])
    else:
        prompt = _prompt(reader, sample)
        pim_mask = sample["pim_mask"]
    mean, std = _prior(adapter, sample, prompt)
    residual = adapter.action_prior_residual(mean, training=False)
    noise = torch.randn_like(sample["action"])
    calls = []
    native_denoise = model._denoise_action_with_video_cache_zeva

    def denoise(**kwargs):
        result = native_denoise(**kwargs)
        calls.append((kwargs, result.detach().clone()))
        return result

    monkeypatch.setattr(model, "_denoise_action_with_video_cache_zeva", denoise)
    condition = dict(
        input_image=sample["video"][:, :, 0], context=sample["context"],
        context_mask=sample["context_mask"], causal_prompt=prompt, pim_mask=pim_mask,
        zeva_task_tokens=sample["task_context"], zeva_action_residual=residual,
    )
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        model.forward_zeva_action_train(
            **condition, clean_action=sample["action"], action_valid=sample["action_valid"],
            noise=noise, timestep=torch.tensor([1000.0]), zeva_prior_mean=mean, zeva_prior_std=std,
        )
        model.infer_action_zeva(
            **condition, prompt=None, action_horizon=32, num_inference_steps=2,
            initial_action_noise=noise, zeva_mode="zeva_stage2" if stage == "policy_injection" else "pim_on",
        )
    assert len(calls) == 3 and torch.equal(torch.get_rng_state(), rng)
    train_call, first_inference_call = calls[:2]
    for name in ("latents_action", "timestep_action", "action_attention_mask", "zeva_action_residual"):
        torch.testing.assert_close(train_call[0][name], first_inference_call[0][name], rtol=0, atol=0)
    for name in ("video_cache_k", "video_cache_v"):
        for train_value, inference_value in zip(train_call[0][name], first_inference_call[0][name], strict=True):
            torch.testing.assert_close(train_value, inference_value, rtol=0, atol=0)
    torch.testing.assert_close(train_call[1], first_inference_call[1], rtol=0, atol=0)
    for arguments, _ in calls[1:]:
        torch.testing.assert_close(arguments["zeva_action_residual"], residual, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["policy_injection", "pim_adapter"])
def test_frozen_main_uses_activation_checkpointing_without_changing_policy_modes(monkeypatch, stage):
    from fastwam.models.wan22 import mot as implementation

    model, sample = main_model(stage), _sample()
    with torch.no_grad():
        model.zeva_behavior_prefix_adapter.pim_gate.fill_(0.2)
    assert not model.mot.training and not model.zeva_joint_foundation
    rng = torch.get_rng_state()
    loss, _ = model(sample)
    loss.backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    calls = []
    checkpoint = implementation.checkpoint

    def record(function, *args, **kwargs):
        calls.append(function.__name__)
        return checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(implementation, "checkpoint", record)
    model.mot.mot_checkpoint_mixed_attn = True
    torch.set_rng_state(rng)
    checkpointed_loss, _ = model(sample)
    checkpointed_loss.backward()
    assert calls.count("_prefill_video_tensor_layer") == model.mot.num_layers
    assert calls.count("_action_cache_tensor_layer") == model.mot.num_layers
    torch.testing.assert_close(checkpointed_loss, loss, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    assert not model.mot.training
    assert all(parameter.grad is None for parameter in model.zeva_foundation_parameters())
    # Evaluation never creates checkpoints, even when the option is enabled.
    with torch.no_grad():
        model(sample)
    assert len(calls) == 2 * model.mot.num_layers
