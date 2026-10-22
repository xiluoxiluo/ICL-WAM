"""Numerical contracts at the EAP/PIM ↔ FastWAM boundary.

The reference computes small addon modules in FP32 during evaluation, keeps
training autocast, and supervises the original full action horizon. These
tests exercise real adapters and the real tiny WAM training/inference paths.
"""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from fastwam.zeva import CausalPromptEncoder
from fastwam.zeva.behavior_prefix_adapter import ExactZevaPolicyInjectionAdapter
from fastwam.zeva.checkpoint import load_addon_checkpoint, save_addon_checkpoint
from fastwam.zeva_numerical_contract import (
    checkpoint_numerical_contract,
    config_numerical_contract,
    validate_checkpoint_numerical_contract,
)
from test_ego_policy_alignment import predictive_model, predictive_sample


def reference_model(stage="policy_injection"):
    model = predictive_model(stage)
    prompt_config = replace(model.zeva_prompt_encoder.config,
                            initialization="zeva_ego", numerical_contract="zeva_ego_v1")
    adapter_config = replace(model.zeva_behavior_prefix_adapter.config,
                             initialization="zeva_ego", numerical_contract="zeva_ego_v1")
    model.attach_zeva_addon(CausalPromptEncoder(prompt_config), ExactZevaPolicyInjectionAdapter(adapter_config))
    model.configure_zeva_trainable_state()
    return model


def prompt_args(sample):
    return (sample["task_context"], sample["phase"], sample["bit_effects"],
            sample["bit_mask"], sample["pim_phases"], sample["pim_effects"], sample["pim_mask"])


def prior_args(sample, prompt):
    return (sample["task_context"], sample["phase"], sample["bit_effects"],
            sample["bit_mask"], prompt, sample["pim_mask"])


@pytest.mark.parametrize("contract,need_weights", [("zeva_ego_v1", True), ("legacy_v1", False)])
def test_prior_attention_uses_reference_math_path(monkeypatch, contract, need_weights):
    prior = (reference_model() if contract == "zeva_ego_v1" else predictive_model()).zeva_behavior_prefix_adapter.prior
    original = prior.progress_attention.forward
    seen = []

    def forward(*args, **kwargs):
        seen.append(kwargs["need_weights"])
        return original(*args, **kwargs)

    monkeypatch.setattr(prior.progress_attention, "forward", forward)
    sample = predictive_sample()
    mean, std = prior(sample["task_context"], sample["phase"])
    assert seen == [need_weights]
    assert mean.shape == std.shape == (3, 32, 14)


@pytest.mark.parametrize("contract", ["zeva_ego_v1", "legacy_v1"])
def test_seeded_prior_dropout_preserves_the_versioned_reference_mask(contract):
    parent = predictive_model().zeva_behavior_prefix_adapter
    adapter = ExactZevaPolicyInjectionAdapter(replace(
        parent.config, numerical_contract=contract, prior_dropout_rate=0.4,
    ))
    with torch.no_grad():
        adapter.action_prior_adapter.weight.zero_()
        adapter.action_prior_adapter.bias.fill_(1.0)
    torch.manual_seed(402)
    draw = torch.rand(64, 1, 1)
    expected = (draw < 0.6) if contract == "zeva_ego_v1" else (draw >= 0.4)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(402)
    residual = adapter.action_prior_residual(torch.zeros(64, 32, 14), training=True)
    torch.testing.assert_close(residual, expected.float().expand_as(residual), rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(adapter.eval().action_prior_residual(torch.zeros(64, 32, 14), training=False),
                               torch.full_like(residual, 0.5), rtol=0, atol=0)


@pytest.mark.parametrize("contract,normalization_dtype", [
    ("zeva_ego_v1", torch.bfloat16), ("legacy_v1", torch.float32),
])
def test_pim_training_normalizes_in_reference_compute_dtype(monkeypatch, contract, normalization_dtype):
    model = reference_model("pim_adapter") if contract == "zeva_ego_v1" else predictive_model("pim_adapter")
    reader, sample = model.zeva_prompt_encoder, predictive_sample()
    sample["pim_mask"].fill_(True)
    seen = []
    normalize = F.normalize

    def capture(tensor, *args, **kwargs):
        seen.append(tensor.dtype)
        return normalize(tensor, *args, **kwargs)

    monkeypatch.setattr(F, "normalize", capture)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = reader(*prompt_args(sample))
        query = normalize(reader.pim_query(sample["phase"]).to(normalization_dtype), dim=-1)
        keys = normalize(reader.pim_key(sample["pim_phases"]).to(normalization_dtype), dim=-1)
        weights = torch.einsum("bd,bnd->bn", query, keys).softmax(-1)
        expected = torch.einsum("bn,bnd->bd", weights, reader.pim_value(sample["pim_effects"]))
    assert seen == [normalization_dtype, normalization_dtype]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_global_dtype_moves_preserve_fp32_addon_values_and_gradients():
    model = reference_model("pim_adapter").eval()
    adapter, reader = model.zeva_behavior_prefix_adapter, model.zeva_prompt_encoder
    weight = adapter.behavior_effect_projector.weight
    with torch.no_grad():
        weight.fill_(1.000123)
    weight.grad = torch.full_like(weight, 0.000123)
    originals = {name: value.detach().clone() for name, value in model.named_parameters()
                 if name.startswith(("zeva_prompt_encoder.", "zeva_behavior_prefix_adapter."))}
    original_grad = weight.grad.clone()
    model.to(device="cpu", dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in model.mot.parameters())
    assert model.zeva_condition_dtype == torch.float32
    for name, value in model.named_parameters():
        if name in originals:
            torch.testing.assert_close(value, originals[name], rtol=0, atol=0)
            assert value.dtype == torch.float32
    torch.testing.assert_close(weight.grad, original_grad, rtol=0, atol=0)
    # Module-specific dtype conversions must obey the same contract.
    adapter.half()
    reader.bfloat16()
    assert all(p.dtype == torch.float32 for p in (*adapter.parameters(), *reader.parameters()))
    torch.testing.assert_close(weight, originals["zeva_behavior_prefix_adapter.behavior_effect_projector.weight"],
                               rtol=0, atol=0)


def test_reference_eval_is_fp32_under_outer_amp_and_training_keeps_amp():
    model, sample = reference_model("pim_adapter").eval(), predictive_sample()
    adapter, reader = model.zeva_behavior_prefix_adapter, model.zeva_prompt_encoder
    with torch.no_grad():
        adapter.pim_to_global.weight.normal_(std=0.1)
        adapter.action_prior_adapter.weight.normal_(std=0.1)

    def compute():
        context = reader(*prompt_args(sample))
        mean, std = adapter.prior_for_prompt(*prior_args(sample, context))
        residual = adapter.action_prior_residual(mean, training=False)
        tokens, _ = adapter.behavior_prefix_tokens(
            sample["task_context"], context, sample["pim_mask"],
            current_effect=sample["current_effect"], current_effect_mask=sample["current_effect_mask"],
        )
        return context, mean, std, residual, tokens

    expected = compute()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = compute()
    for left, right in zip(actual, expected, strict=True):
        assert left.dtype == torch.float32
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    model.configure_zeva_trainable_state()
    assert not adapter.training and adapter.prior.training
    with torch.autocast("cpu", dtype=torch.bfloat16):
        context = reader(*prompt_args(sample))
        mean, _ = adapter.prior_for_prompt(*prior_args(sample, context))
    assert context.dtype == mean.dtype == torch.bfloat16
    # Freezing EAP parameters/modules must not disable AMP through a trained PIM.
    adapter.prior.eval().requires_grad_(False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mean, _ = adapter.prior_for_prompt(*prior_args(sample, context))
    assert mean.dtype == torch.bfloat16


def test_real_wam_training_uses_original_full_horizon_targets(monkeypatch):
    from fastwam.models.wan22 import fastwam as implementation

    model, sample = reference_model(), predictive_sample()
    model.torch_dtype = torch.bfloat16
    model.to(dtype=torch.bfloat16)
    model.configure_zeva_trainable_state()
    model.prepare_zeva_training_precision()
    # The valid prefix and padded tail differ enough to detect an accidental
    # EAP mask; the FP32 fractions also detect round-trip BF16 targets.
    sample["action"][:, :8] = 0.123456
    sample["action"][:, 8:] = 1.012345
    sample["action_valid"][:, 8:] = False
    original_target = sample["action"].clone()
    seen = []
    gaussian_prior_nll = implementation.gaussian_prior_nll

    def capture(target, mean, std, valid=None, **kwargs):
        seen.append((target.detach().clone(), mean.detach().clone(), std.detach().clone(), valid))
        return gaussian_prior_nll(target, mean, std, valid, **kwargs)

    monkeypatch.setattr(implementation, "gaussian_prior_nll", capture)
    rng = torch.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, masked_metrics = model(sample)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.zeva_behavior_prefix_adapter.prior.dist_head[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.mot.parameters())
    target, mean, std, mask = seen[0]
    assert mask is None
    assert target.dtype == torch.float32 and mean.dtype == std.dtype == torch.bfloat16
    torch.testing.assert_close(target, original_target, rtol=0, atol=0)
    assert not torch.equal(target, original_target.bfloat16().float())
    reference = -torch.distributions.Normal(mean, std).log_prob(original_target).sum(-1).mean()
    assert masked_metrics["behavior_prior_nll"] == pytest.approx(reference.item(), abs=1e-6)

    # Padding provenance cannot remove repeated terminal targets from either
    # reference objective; the underlying WAM flow formula stays unchanged.
    sample["action_valid"].fill_(True)
    torch.set_rng_state(rng)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, full_metrics = model(sample)
    assert full_metrics["behavior_prior_nll"] == masked_metrics["behavior_prior_nll"]
    assert full_metrics["loss_flow"] == masked_metrics["loss_flow"]


@pytest.mark.parametrize("contract", ["zeva_ego_v1", "legacy_v1"])
def test_terminal_repeat_targets_train_joint_foundation_only_in_reference_contract(contract):
    model = reference_model() if contract == "zeva_ego_v1" else predictive_model()
    model.zeva_training_recipe = "ego_joint_parent"
    model.zeva_task_context_mode = "task_language"
    model.configure_zeva_trainable_state()
    model.prepare_zeva_training_precision()
    sample = predictive_sample()
    # A final recorded observation has a full repeated action target even
    # though the dataset's provenance mask marks every future index padded.
    sample["action"][:] = 0.376543
    sample["action_valid"].zero_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, metrics = model(sample)
    loss.backward()
    foundation_gradient = sum(float(p.grad.detach().float().abs().sum())
                              for p in model.zeva_foundation_parameters() if p.grad is not None)
    if contract == "zeva_ego_v1":
        assert metrics["loss_flow"] > 0 and foundation_gradient > 0
    else:
        assert metrics["loss_flow"] == foundation_gradient == 0


def test_reference_checkpoint_reloads_fp32_addon_into_bf16_wam_and_infers(tmp_path):
    source = reference_model("pim_adapter").eval()
    with torch.no_grad():
        source.zeva_behavior_prefix_adapter.behavior_effect_projector.weight.fill_(1.000123)
    path = tmp_path / "reference_addon.pt"
    source.save_zeva_addon_checkpoint(path)
    assert torch.load(path, weights_only=False)["numerical_contract"] == "zeva_ego_v1"
    target = reference_model("pim_adapter").eval()
    target.torch_dtype = torch.bfloat16
    target.to(dtype=torch.bfloat16)
    target.load_zeva_addon_checkpoint(path)
    torch.testing.assert_close(target.zeva_behavior_prefix_adapter.behavior_effect_projector.weight,
                               source.zeva_behavior_prefix_adapter.behavior_effect_projector.weight, rtol=0, atol=0)
    sample = predictive_sample()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        context = target.zeva_prompt_encoder(*prompt_args(sample))
        mean, _ = target.zeva_behavior_prefix_adapter.prior_for_prompt(*prior_args(sample, context))
        residual = target.zeva_behavior_prefix_adapter.action_prior_residual(mean, training=False)
        output = target.infer_action_zeva(
            prompt=None, input_image=torch.zeros(1, 3, 16, 16),
            context=sample["context"][:1], context_mask=sample["context_mask"][:1],
            action_horizon=32, num_inference_steps=2, causal_prompt=context[:1],
            pim_mask=sample["pim_mask"][:1], zeva_task_tokens=sample["task_context"][:1],
            zeva_current_effect=sample["current_effect"][:1],
            zeva_current_effect_mask=sample["current_effect_mask"][:1],
            zeva_action_residual=residual[:1], initial_action_noise=torch.randn(1, 32, 14),
        )
    assert context.dtype == mean.dtype == residual.dtype == torch.float32
    assert output["action"].shape == (32, 14) and torch.isfinite(output["action"]).all()
    assert all(p.dtype == torch.bfloat16 for p in target.mot.parameters())


@pytest.mark.parametrize("saved,target", [("legacy_v1", "zeva_ego_v1"), ("zeva_ego_v1", "legacy_v1")])
def test_numerical_checkpoint_mismatch_rejected_before_mutating_weights(tmp_path, saved, target):
    source = reference_model() if saved == "zeva_ego_v1" else predictive_model()
    destination = reference_model() if target == "zeva_ego_v1" else predictive_model()
    path = tmp_path / "addon.pt"
    save_addon_checkpoint(path, source.zeva_prompt_encoder, source.zeva_behavior_prefix_adapter,
                          1, {}, "base", "cte")
    if saved == "legacy_v1":
        payload = torch.load(path, weights_only=False)
        del payload["numerical_contract"]  # Exactly the pre-contract checkpoint format.
        torch.save(payload, path)
        load_addon_checkpoint(path, source.zeva_prompt_encoder, source.zeva_behavior_prefix_adapter)
    original = deepcopy(destination.state_dict())
    with pytest.raises(ValueError, match="numerical_contract mismatch"):
        destination.load_zeva_addon_checkpoint(path)
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_numerical_contract_metadata_requires_matching_configuration_and_marker():
    assert checkpoint_numerical_contract({}) == "legacy_v1"
    assert config_numerical_contract({}) == "legacy_v1"
    zeva = {name: {"numerical_contract": "zeva_ego_v1"} for name in ("prompt", "adapter")}
    payload = {"numerical_contract": "zeva_ego_v1", "config": {"model": {"zeva": zeva}}}
    assert validate_checkpoint_numerical_contract(payload, "zeva_ego_v1") == "zeva_ego_v1"
    with pytest.raises(ValueError, match="missing its numerical_contract marker"):
        checkpoint_numerical_contract({"config": payload["config"]})
    with pytest.raises(ValueError, match="disagrees with its saved configuration"):
        checkpoint_numerical_contract(dict(payload, numerical_contract="legacy_v1"))
    with pytest.raises(ValueError, match="must agree"):
        config_numerical_contract({"prompt": zeva["prompt"], "adapter": {}})
