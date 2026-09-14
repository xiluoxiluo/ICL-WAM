import math
from copy import deepcopy
from types import MethodType

import pytest
import torch
from torch import nn

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.zeva import (
    BehaviorPrefixAdapter,
    BehaviorPrefixAdapterConfig,
    CausalPromptConfig,
    CausalPromptEncoder,
    ExactZevaPolicyInjectionAdapter,
    ExactZevaPolicyInjectionConfig,
)
from fastwam.zeva.checkpoint import load_addon_checkpoint


class _ActionExpert(nn.Module):
    action_dim = 14

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(14, 8)

    def prepare(self, action_tokens, timestep, context, context_mask):
        tokens = self.encoder(action_tokens)
        return tokens, timestep, torch.ones_like(tokens), context, context_mask, torch.zeros(tokens.shape[1], 1)

    def post(self, tokens):
        return tokens


class _MoT(nn.Module):
    def forward_action_with_video_cache_tensor(self, action_tokens, **kwargs):
        return action_tokens * 2.0 + 1.0


class _InferenceActionExpert(nn.Module):
    action_dim = 14
    hidden_dim = 8

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(14, 8, bias=False)
        self.decoder = nn.Linear(8, 14, bias=False)

    def prepare(self, action_tokens, timestep, context, context_mask):
        context_scalar = context[:, :1].mean(dim=-1, keepdim=True)
        tokens = self.encoder(action_tokens) + context_scalar
        return (
            tokens,
            timestep,
            torch.ones_like(tokens),
            context,
            context_mask,
            torch.zeros(tokens.shape[1], 1),
        )

    def post(self, tokens):
        return self.decoder(tokens)


class _InferenceVideoExpert(nn.Module):
    video_attention_mask_mode = "first_frame_causal"
    fuse_vae_embedding_in_latents = False

    def prepare(self, x, timestep, context, context_mask, **kwargs):
        tokens = x.new_zeros((x.shape[0], 1, 8))
        return (
            tokens,
            timestep,
            torch.ones_like(tokens),
            context,
            context_mask,
            torch.zeros(1, 1),
            1,
            1,
            1,
            1,
        )


class _InferenceMoT(nn.Module):
    def prefill_video_cache_tensor(self, video_tokens, **kwargs):
        cache = [video_tokens.new_zeros((video_tokens.shape[0], 1, 1, 1))]
        return cache, [value.clone() for value in cache]

    def forward_action_with_video_cache_tensor(self, action_tokens, **kwargs):
        return action_tokens


class _InferenceScheduler:
    def build_inference_schedule(self, **kwargs):
        return torch.tensor([1.0]), torch.tensor([1.0])

    def step(self, prediction, _delta, _sample):
        return prediction


def _model():
    model = FastWAM.__new__(FastWAM)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.action_expert = _ActionExpert()
    model.mot = _MoT()
    model.zeva_enabled = True
    model.zeva_prompt_encoder = CausalPromptEncoder()
    model.zeva_behavior_prefix_adapter = BehaviorPrefixAdapter(
        BehaviorPrefixAdapterConfig(action_hidden_dim=8, num_heads=2)
    )
    model.zeva_injection_mode = "memory_residual"
    model.zeva_training_stage = "policy_injection"
    model.zeva_task_context_mode = "pooling"
    return model


def _inference_model():
    model = FastWAM.__new__(FastWAM)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.action_expert = _InferenceActionExpert()
    model.video_expert = _InferenceVideoExpert()
    model.mot = _InferenceMoT()
    model.infer_action_scheduler = _InferenceScheduler()
    model.proprio_dim = None
    model.proprio_encoder = None
    model.zeva_enabled = True
    model.zeva_prompt_encoder = CausalPromptEncoder(
        CausalPromptConfig(
            global_dim=8,
            phase_dim=4,
            effect_dim=4,
            brief_length=2,
            persistent_length=2,
            hidden_dim=8,
            num_heads=2,
        )
    )
    model.zeva_behavior_prefix_adapter = ExactZevaPolicyInjectionAdapter(
        ExactZevaPolicyInjectionConfig(
            memory_dim=8,
            context_dim=4,
            global_dim=8,
            phase_dim=4,
            effect_dim=4,
            effect_history_length=2,
            action_dim=14,
            action_horizon=32,
            prior_hidden_dim=8,
            prior_num_heads=2,
            action_hidden_dim=8,
            prior_dropout_rate=0.0,
        )
    )
    model.zeva_injection_mode = "exact_zeva"
    model.zeva_training_stage = "pim_adapter"
    model._encode_input_image_latents_tensor = MethodType(
        lambda self, input_image, tiled=False: input_image.new_zeros((1, 4, 1, 1, 1)),
        model,
    )
    model._build_mot_attention_mask = MethodType(
        lambda self, video_seq_len, action_seq_len, **kwargs: torch.zeros(
            video_seq_len + action_seq_len,
            video_seq_len + action_seq_len,
        ),
        model,
    )
    with torch.no_grad():
        adapter = model.zeva_behavior_prefix_adapter
        adapter.behavior_global_projector.weight.fill_(0.25)
        adapter.behavior_global_projector.bias.zero_()
        adapter.prefix_project.weight.fill_(0.125)
        adapter.prefix_project.bias.zero_()
    return model.eval()


def _native_inference_model():
    """A true FastWAM action path with no Zeva addon attached."""
    model = FastWAM.__new__(FastWAM)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.action_expert = _InferenceActionExpert()
    model.video_expert = _InferenceVideoExpert()
    model.mot = _InferenceMoT()
    model.infer_action_scheduler = _InferenceScheduler()
    model.proprio_dim = None
    model.proprio_encoder = None
    model.zeva_enabled = False
    model.zeva_prompt_encoder = None
    model.zeva_behavior_prefix_adapter = None
    model.zeva_injection_mode = "exact_zeva"
    model.zeva_training_stage = "policy_injection"
    model._encode_input_image_latents_tensor = MethodType(
        lambda self, input_image, tiled=False: input_image.new_zeros((1, 4, 1, 1, 1)),
        model,
    )
    model._build_mot_attention_mask = MethodType(
        lambda self, video_seq_len, action_seq_len, **kwargs: torch.zeros(
            video_seq_len + action_seq_len, video_seq_len + action_seq_len
        ),
        model,
    )
    return model.eval()


def _copy_fastwam_test_weights(source, target):
    target.action_expert.load_state_dict(source.action_expert.state_dict())
    if any(True for _ in source.video_expert.parameters()):
        target.video_expert.load_state_dict(source.video_expert.state_dict())
    if any(True for _ in source.mot.parameters()):
        target.mot.load_state_dict(source.mot.state_dict())


def _infer_mode(model, mode, causal_prompt, *, seed=17):
    return model.infer_action(
        prompt=None,
        context=torch.zeros(1, 2, 4),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        input_image=torch.zeros(1, 3, 16, 16),
        action_horizon=32,
        num_inference_steps=1,
        seed=seed,
        causal_prompt=causal_prompt,
        pim_mask=torch.ones(1, 2, dtype=torch.bool),
        zeva_action_residual=torch.full((1, 32, 8), 0.25),
        zeva_task_tokens=torch.ones(1, 8),
        zeva_mode=mode,
    )["action"]


def test_gate_zero_action_path_matches_base_and_addon_gets_gradients():
    model = _model()
    memory = torch.randn(2, 256)
    memory_mask = torch.ones(2, 4, dtype=torch.bool)
    action = torch.randn(2, 32, 14)
    timestep = torch.ones(2)
    context = torch.randn(2, 3, 4)
    context_mask = torch.ones(2, 3, dtype=torch.bool)
    attention = torch.zeros(2, 32 + 1)
    cache = [torch.zeros(2, 1, 1, 1)]

    model.eval()
    base = model._denoise_action_with_video_cache(
        action, timestep, context, context_mask, cache, cache, attention
    )
    shadow = model._denoise_action_with_video_cache_zeva(
        action, timestep, context, context_mask, cache, cache, attention,
        memory, memory_mask, gate_override=0.0,
    )
    torch.testing.assert_close(base, shadow, rtol=0.0, atol=0.0)

    model.zeva_prompt_encoder.train()
    model.zeva_behavior_prefix_adapter.train()
    model.requires_grad_(False)
    model.zeva_prompt_encoder.requires_grad_(True)
    model.zeva_behavior_prefix_adapter.requires_grad_(True)
    task_args = (
        torch.randn(2, 256), torch.randn(2, 128), torch.randn(2, 4, 128),
        torch.ones(2, 4, dtype=torch.bool), torch.randn(2, 4, 128),
        torch.randn(2, 4, 128), torch.ones(2, 4, dtype=torch.bool),
    )
    memory = model.zeva_prompt_encoder(*task_args)
    memory_mask = task_args[-1]
    pred = model._denoise_action_with_video_cache_zeva(
        action, timestep, context, context_mask, cache, cache, attention,
        memory, memory_mask,
    )
    pred.square().mean().backward()
    assert model.zeva_behavior_prefix_adapter.pim_gate.grad is not None
    # The exact zero gate intentionally makes the first residual update a
    # gate-only step; once the gate opens, the projector and prompt receive
    # ordinary upstream gradients.
    assert model.zeva_behavior_prefix_adapter.output.weight.grad is not None
    assert model.zeva_behavior_prefix_adapter.output.weight.grad.norm() == 0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.zeva_prompt_encoder.parameters())
    assert all(p.grad is None for p in model.action_expert.parameters())


def test_nonzero_addon_gate_changes_action_path():
    """A loaded/trained addon must influence ActionDiT before the MoT block."""
    model = _model()
    memory = torch.randn(1, 256)
    memory_mask = torch.ones(1, 4, dtype=torch.bool)
    action = torch.randn(1, 32, 14)
    timestep = torch.ones(1)
    context = torch.randn(1, 3, 4)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    attention = torch.zeros(1, 33)
    cache = [torch.zeros(1, 1, 1, 1)]

    model.eval()
    with torch.no_grad():
        # Simulate a checkpoint whose residual head and scalar gate have moved
        # away from their exact no-op initialization.
        model.zeva_behavior_prefix_adapter.output.bias.fill_(1.0)
        model.zeva_behavior_prefix_adapter.pim_gate.fill_(math.atanh(0.5))
    base = model._denoise_action_with_video_cache(
        action, timestep, context, context_mask, cache, cache, attention
    )
    conditioned = model._denoise_action_with_video_cache_zeva(
        action,
        timestep,
        context,
        context_mask,
        cache,
        cache,
        attention,
        memory,
        memory_mask,
    )
    assert not torch.equal(base, conditioned)


def test_exact_model_owns_the_independent_behavior_prefix_slot():
    model = _model()
    model.zeva_injection_mode = "exact_zeva"
    model.zeva_behavior_prefix_adapter = ExactZevaPolicyInjectionAdapter(
        ExactZevaPolicyInjectionConfig(
            memory_dim=8,
            context_dim=4,
            global_dim=8,
            phase_dim=4,
            effect_dim=4,
            effect_history_length=2,
            action_dim=3,
            action_horizon=5,
            prior_hidden_dim=8,
            prior_num_heads=2,
            action_hidden_dim=8,
        )
    )
    with torch.no_grad():
        model.zeva_behavior_prefix_adapter.pim_gate.fill_(0.5)
    context = torch.randn(1, 3, 4)
    context_mask = torch.tensor([[True, True, False]])
    memory = torch.randn(1, 8)
    memory_mask = torch.ones(1, 4, dtype=torch.bool)
    task = torch.randn(1, 8)
    augmented, augmented_mask = model._prepend_exact_zeva_behavior_slot(
        context, context_mask, memory, memory_mask, task_tokens=task
    )
    assert augmented.shape == (1, 4, 4)
    assert augmented_mask.tolist() == [[True, True, True, False]]
    torch.testing.assert_close(augmented[:, 1:], context)
    torch.testing.assert_close(augmented_mask[:, 1:], context_mask)

    # Legacy adapters keep their old action-hidden path and do not acquire a
    # context slot implicitly.
    model.zeva_injection_mode = "memory_residual"
    same_context, same_mask = model._prepend_exact_zeva_behavior_slot(
        context, context_mask, memory, memory_mask, task
    )
    assert same_context is context
    assert same_mask is context_mask


def test_conditioned_action_path_reports_residual_debug_metrics():
    model = _model()
    memory = torch.randn(1, 256)
    memory_mask = torch.ones(1, 4, dtype=torch.bool)
    action = torch.randn(1, 32, 14)
    timestep = torch.ones(1)
    context = torch.randn(1, 3, 4)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    attention = torch.zeros(1, 33)
    cache = [torch.zeros(1, 1, 1, 1)]
    with torch.no_grad():
        model.zeva_behavior_prefix_adapter.output.weight.normal_(std=0.01)
        model.zeva_behavior_prefix_adapter.pim_gate.fill_(math.atanh(0.5))
    debug = {}
    model._denoise_action_with_video_cache_zeva(
        action, timestep, context, context_mask, cache, cache, attention,
        memory, memory_mask, debug=debug,
    )
    assert debug["base_action_hidden_norm"] > 0
    assert debug["memory_delta_hidden_norm"] > 0
    assert debug["conditioned_action_hidden_norm"] > 0
    assert torch.isfinite(debug["memory_residual_ratio"])


def test_memory_content_changes_action_prediction():
    """Different PIM content must affect actions, not merely a constant bias."""
    torch.manual_seed(7)
    model = _model().eval()
    with torch.no_grad():
        model.zeva_behavior_prefix_adapter.output.weight.normal_(mean=0.0, std=0.01)
        model.zeva_behavior_prefix_adapter.pim_gate.fill_(math.atanh(0.5))

    action = torch.randn(1, 32, 14)
    timestep = torch.ones(1)
    context = torch.randn(1, 3, 4)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    attention = torch.zeros(1, 33)
    cache = [torch.zeros(1, 1, 1, 1)]
    task = torch.randn(1, 256)
    phase = torch.randn(1, 128)
    bit = torch.randn(1, 4, 128)
    bit_mask = torch.ones(1, 4, dtype=torch.bool)
    pim_phase_a = torch.randn(1, 4, 128)
    pim_effect_a = torch.randn(1, 4, 128)
    pim_phase_b = pim_phase_a.clone()
    pim_effect_b = -pim_effect_a
    pim_mask = torch.ones(1, 4, dtype=torch.bool)

    memory_a = model.zeva_prompt_encoder(
        task, phase, bit, bit_mask, pim_phase_a, pim_effect_a, pim_mask
    )
    memory_b = model.zeva_prompt_encoder(
        task, phase, bit, bit_mask, pim_phase_b, pim_effect_b, pim_mask
    )
    assert torch.linalg.vector_norm(memory_a - memory_b) > 1e-6

    action_a = model._denoise_action_with_video_cache_zeva(
        action, timestep, context, context_mask, cache, cache, attention, memory_a, pim_mask
    )
    action_b = model._denoise_action_with_video_cache_zeva(
        action, timestep, context, context_mask, cache, cache, attention, memory_b, pim_mask
    )
    delta = torch.linalg.vector_norm(action_a - action_b)
    assert torch.isfinite(delta)
    assert delta > 1e-5


def test_formal_inference_mode_equivalences_and_pim_effect():
    model = _inference_model()
    prompt_a = torch.ones(1, 8)
    prompt_b = -prompt_a

    native = _infer_mode(model, "base", prompt_a)
    addon_base = _infer_mode(model, "base", prompt_b)
    torch.testing.assert_close(native, addon_base, rtol=0.0, atol=0.0)

    stage2 = _infer_mode(model, "zeva_stage2", prompt_a)
    shadow = _infer_mode(model, "pim_shadow", prompt_b)
    torch.testing.assert_close(stage2, shadow, rtol=0.0, atol=0.0)
    assert not torch.equal(native, stage2)

    with torch.no_grad():
        model.zeva_behavior_prefix_adapter.pim_gate.zero_()
    gate_zero = _infer_mode(model, "pim_on", prompt_b)
    torch.testing.assert_close(stage2, gate_zero, rtol=0.0, atol=0.0)

    with torch.no_grad():
        model.zeva_behavior_prefix_adapter.pim_gate.fill_(math.atanh(0.5))
    pim_a = _infer_mode(model, "pim_on", prompt_a)
    pim_b = _infer_mode(model, "pim_on", prompt_b)
    assert not torch.equal(pim_a, pim_b)


def test_base_mode_matches_true_native_fastwam():
    zeva_model = _inference_model()
    native_model = _native_inference_model()
    _copy_fastwam_test_weights(zeva_model, native_model)
    kwargs = {
        "prompt": None,
        "context": torch.zeros(1, 2, 4),
        "context_mask": torch.ones(1, 2, dtype=torch.bool),
        "input_image": torch.zeros(1, 3, 16, 16),
        "action_horizon": 32,
        "num_inference_steps": 1,
        "seed": 17,
    }
    native_action = native_model.infer_action(zeva_mode="base", **kwargs)["action"]
    addon_base_action = zeva_model.infer_action(zeva_mode="base", **kwargs)["action"]
    torch.testing.assert_close(native_action, addon_base_action, rtol=0.0, atol=0.0)


def test_parameter_report_rejects_trainable_base():
    model = _model()
    model.zeva_behavior_prefix_adapter = ExactZevaPolicyInjectionAdapter(
        ExactZevaPolicyInjectionConfig(
            context_dim=4,
            action_hidden_dim=8,
            prior_hidden_dim=16,
            prior_num_heads=2,
            prior_dropout_rate=0.0,
        )
    )
    model.zeva_injection_mode = "exact_zeva"
    model.base_probe = nn.Linear(2, 2)
    model.configure_zeva_trainable_state()
    report = model.zeva_parameter_report()
    assert report["trainable_count"] > 0
    model.base_probe.weight.requires_grad_(True)
    try:
        model.zeva_parameter_report()
    except AssertionError:
        pass
    else:
        raise AssertionError("base parameter was not rejected by Zeva whitelist")


def test_attach_zeva_addon_freezes_base_modules():
    model = _model()
    # _model() emulates an already-constructed FastWAM with an attached addon;
    # exercise the same invariant directly on its public attach method.
    model.zeva_enabled = False
    model.zeva_prompt_encoder = None
    model.zeva_behavior_prefix_adapter = None
    model.attach_zeva_addon(CausalPromptEncoder(), BehaviorPrefixAdapter(
        BehaviorPrefixAdapterConfig(action_hidden_dim=8, num_heads=2)
    ))
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith(("zeva_prompt_encoder.", "zeva_behavior_prefix_adapter."))
    )
    assert any(parameter.requires_grad for parameter in model.zeva_prompt_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.zeva_behavior_prefix_adapter.parameters())


@pytest.mark.parametrize("use_bank", [False, True])
def test_forward_routes_stage2_through_model_entrypoint(use_bank):
    model = _model()
    called = {}

    def fake_forward(self, **kwargs):
        called.update(kwargs)
        return torch.tensor(2.0), {"loss_action": 2.0}

    model.forward_zeva_action_train = MethodType(fake_forward, model)
    sample = {
        "_training_mode": "zeva_stage2",
        "video": torch.zeros(1, 3, 1, 8, 8),
        "action": torch.zeros(1, 32, 14),
        "context": torch.zeros(1, 3, 256),
        "context_mask": torch.ones(1, 3, dtype=torch.bool),
        "phase": torch.zeros(1, 128),
        "bit_effects": torch.zeros(1, 4, 128),
        "bit_mask": torch.zeros(1, 4, dtype=torch.bool),
        "pim_phases": torch.zeros(1, 4, 128),
        "pim_effects": torch.zeros(1, 4, 128),
        "pim_mask": torch.zeros(1, 4, dtype=torch.bool),
    }
    expected_task = torch.arange(256).float().unsqueeze(0) if use_bank else torch.zeros(1, 256)
    if use_bank:
        sample["task_context"] = expected_task.clone()

    loss, metrics = model(sample)
    torch.testing.assert_close(called["zeva_task_tokens"], expected_task)
    assert loss.item() == 2.0
    assert metrics["memory/bit_count"] == 0.0
    assert metrics["memory/pim_count"] == 0.0
    assert called["causal_prompt"].shape == (1, 256)
    assert torch.count_nonzero(called["causal_prompt"]) == 0
    if use_bank:
        sample["task_context"] = torch.zeros(1, 1, 256)
        with pytest.raises(ValueError, match="task_context must be"):
            model(sample)
        model.zeva_task_context_mode = "static"
        sample.pop("task_context")
        with pytest.raises(ValueError, match="static task_context"):
            model(sample)


def _stage_model():
    model = _model()
    model.base_probe = nn.Linear(3, 3)
    model.zeva_behavior_prefix_adapter = ExactZevaPolicyInjectionAdapter(
        ExactZevaPolicyInjectionConfig(
            context_dim=16,
            action_hidden_dim=8,
            prior_hidden_dim=16,
            prior_num_heads=2,
            prior_dropout_rate=0.0,
        )
    )
    model.zeva_injection_mode = "exact_zeva"
    return model


@pytest.mark.parametrize(
    ("stage", "allowed"),
    [
        (
            "policy_injection",
            (
                "zeva_behavior_prefix_adapter.prior.",
                "zeva_behavior_prefix_adapter.action_prior_adapter.",
                "zeva_behavior_prefix_adapter.behavior_global_projector.",
            ),
        ),
        (
            "pim_adapter",
            (
                "zeva_prompt_encoder.",
                "zeva_behavior_prefix_adapter.prefix_project.",
                "zeva_behavior_prefix_adapter.pim_gate",
            ),
        ),
    ],
)
def test_stage_specific_trainable_whitelist(stage, allowed):
    model = _stage_model().set_zeva_training_stage(stage)
    model.configure_zeva_trainable_state()
    report = model.zeva_parameter_report()
    assert report["training_stage"] == stage
    assert report["trainable_names"]
    assert all(name.startswith(allowed) for name in report["trainable_names"])
    assert {id(parameter) for parameter in model.zeva_trainable_parameters()} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_pim_stage_loads_only_policy_injection_checkpoint(tmp_path):
    source = _stage_model()
    with torch.no_grad():
        for parameter in source.zeva_prompt_encoder.parameters():
            parameter.fill_(2.0)
        for parameter in source.zeva_behavior_prefix_adapter.policy_injection_parameters():
            parameter.fill_(1.0)
        source.zeva_behavior_prefix_adapter.prefix_project.weight.fill_(3.0)
        source.zeva_behavior_prefix_adapter.prefix_project.bias.fill_(3.0)
        source.zeva_behavior_prefix_adapter.pim_gate.fill_(0.7)
    path = tmp_path / "policy.pt"
    torch.save(
        {
            "causal_prompt_encoder": source.zeva_prompt_encoder.state_dict(),
            "behavior_prefix_adapter": source.zeva_behavior_prefix_adapter.state_dict(),
            "training_stage": "policy_injection",
            "task_context_identity": None,
        },
        path,
    )

    target = _stage_model()
    prompt_before = deepcopy(target.zeva_prompt_encoder.state_dict())
    prefix_before = deepcopy(
        target.zeva_behavior_prefix_adapter.prefix_project.state_dict()
    )
    gate_before = target.zeva_behavior_prefix_adapter.pim_gate.detach().clone()
    payload = load_addon_checkpoint(
        path,
        target.zeva_prompt_encoder,
        target.zeva_behavior_prefix_adapter,
        load_scope="policy_injection",
    )

    assert payload["training_stage"] == "policy_injection"
    for source_parameter, target_parameter in zip(
        source.zeva_behavior_prefix_adapter.policy_injection_parameters(),
        target.zeva_behavior_prefix_adapter.policy_injection_parameters(),
    ):
        torch.testing.assert_close(target_parameter, source_parameter)
    for name, value in target.zeva_prompt_encoder.state_dict().items():
        torch.testing.assert_close(value, prompt_before[name])
    for name, value in target.zeva_behavior_prefix_adapter.prefix_project.state_dict().items():
        torch.testing.assert_close(value, prefix_before[name])
    torch.testing.assert_close(
        target.zeva_behavior_prefix_adapter.pim_gate, gate_before
    )


def _clone_non_zeva_state(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(("zeva_prompt_encoder.", "zeva_behavior_prefix_adapter."))
    }


def test_policy_injection_step_does_not_modify_fastwam():
    model = _stage_model()
    model.set_zeva_training_stage("policy_injection")
    model.configure_zeva_trainable_state()
    before = _clone_non_zeva_state(model)
    optimizer = torch.optim.AdamW(model.zeva_trainable_parameters(), lr=1e-4)
    adapter = model.zeva_behavior_prefix_adapter
    task_tokens = torch.randn(2, adapter.config.global_dim)
    phase = torch.randn(2, adapter.config.phase_dim)
    bit_effects = torch.randn(2, adapter.config.effect_history_length, adapter.config.effect_dim)
    bit_mask = torch.ones(2, adapter.config.effect_history_length, dtype=torch.bool)
    prior_mean, prior_std = adapter.prior(task_tokens, phase, bit_effects, bit_mask)
    residual = adapter.action_prior_residual(prior_mean, training=True)
    loss = residual.float().square().mean() + prior_std.float().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = _clone_non_zeva_state(model)
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0.0, atol=0.0)
