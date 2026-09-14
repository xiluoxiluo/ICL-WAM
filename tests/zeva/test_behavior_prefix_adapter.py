import torch

from fastwam.zeva import (
    BehaviorPrefixAdapter, BehaviorPrefixAdapterConfig, CausalPromptEncoder,
    ExactZevaPolicyInjectionAdapter, ExactZevaPolicyInjectionConfig,
    gaussian_prior_nll,
)


def test_gate_zero_is_exact_zero_and_training_has_gradients():
    prompt = CausalPromptEncoder()
    adapter = BehaviorPrefixAdapter(BehaviorPrefixAdapterConfig())
    args = (torch.randn(2, 256), torch.randn(2, 128), torch.randn(2, 4, 128), torch.ones(2, 4, dtype=torch.bool), torch.randn(2, 4, 128), torch.randn(2, 4, 128), torch.ones(2, 4, dtype=torch.bool))
    causal_prompt = prompt(*args)
    memory = causal_prompt[:, None]
    mask = args[-1].any(dim=-1, keepdim=True)
    adapter.eval()
    assert torch.equal(adapter.gated(memory, mask), torch.zeros(2, 32, 1024))
    # Open the gate for the gradient check. With a mathematically exact
    # zero gate, no upstream residual parameter can receive a first-step
    # gradient; the gate itself is intentionally the parameter that moves.
    with torch.no_grad():
        adapter.pim_gate.fill_(0.5)
    adapter.train(); prompt.train()
    loss = adapter.gated(memory, mask).square().mean(); loss.backward()
    assert adapter.pim_gate.grad is not None and torch.isfinite(adapter.pim_gate.grad)
    assert adapter.output.weight.grad is not None
    assert adapter.output.weight.grad.norm() > 0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in prompt.parameters())


def test_exact_zeva_adapter_has_separate_prefix_and_action_prior_branches():
    cfg = ExactZevaPolicyInjectionConfig(
        memory_dim=8, context_dim=16, global_dim=8, phase_dim=4, effect_dim=4,
        effect_history_length=2, action_dim=3, action_horizon=5,
        prior_hidden_dim=8, prior_num_heads=2, action_hidden_dim=6,
    )
    adapter = ExactZevaPolicyInjectionAdapter(cfg)
    task = torch.randn(2, 8)
    phase = torch.randn(2, 4)
    bit = torch.randn(2, 2, 4)
    bit_mask = torch.ones(2, 2, dtype=torch.bool)
    causal_prompt = torch.randn(2, 8)
    pim_mask = torch.ones(2, 4, dtype=torch.bool)
    mean, std = adapter.prior(task, phase, bit, bit_mask)
    assert mean.shape == std.shape == (2, 5, 3)
    assert torch.isfinite(gaussian_prior_nll(torch.randn_like(mean), mean, std))
    prefix = adapter.causal_prompt_prefix(causal_prompt, pim_mask)
    residual = adapter.action_prior_residual(mean, training=True)
    assert prefix.shape == (2, 1, 16)
    assert residual.shape == (2, 5, 6)
    # Exact Zeva initialization keeps both policy injection branches harmless.
    assert torch.equal(residual, torch.zeros_like(residual))
    with torch.no_grad():
        adapter.pim_gate.fill_(0.0)
    assert torch.equal(prefix, torch.zeros_like(prefix))

    # The first two prompt tokens are valid even without persistent evidence;
    # only the PIM token is allowed to open the causal-prompt branch.
    with torch.no_grad():
        adapter.pim_gate.fill_(0.5)
    no_pim_mask = torch.zeros(2, 4, dtype=torch.bool)
    no_pim_prefix = adapter.causal_prompt_prefix(causal_prompt, no_pim_mask)
    assert torch.equal(no_pim_prefix, torch.zeros_like(no_pim_prefix))
    pim_prefix = adapter.causal_prompt_prefix(causal_prompt, pim_mask)
    assert torch.count_nonzero(pim_prefix) > 0


def test_exact_zeva_prompt_uses_an_independent_context_slot():
    cfg = ExactZevaPolicyInjectionConfig(
        memory_dim=8, context_dim=16, global_dim=8, phase_dim=4, effect_dim=4,
        effect_history_length=2, action_dim=3, action_horizon=5,
        prior_hidden_dim=8, prior_num_heads=2, action_hidden_dim=6,
    )
    adapter = ExactZevaPolicyInjectionAdapter(cfg)
    with torch.no_grad():
        adapter.pim_gate.fill_(0.5)
    context = torch.randn(2, 3, 16)
    original_context = context.clone()
    context_mask = torch.tensor([[True, True, False], [True, False, False]])
    causal_prompt = torch.randn(2, 8)
    pim_mask = torch.tensor([[True, True, True, True], [False, False, False, False]])
    task = torch.randn(2, 8)
    augmented, augmented_mask = adapter.prepend_behavior_prefix_slot(
        context, context_mask, causal_prompt, pim_mask, task_tokens=task
    )
    assert augmented.shape == (2, 4, 16)
    assert augmented_mask.shape == (2, 4)
    expected_slot = adapter.behavior_prefix_slot(task, causal_prompt, pim_mask)
    torch.testing.assert_close(augmented[:, :1], expected_slot)
    torch.testing.assert_close(augmented[:, 1:], context)
    torch.testing.assert_close(augmented_mask[:, 1:], context_mask)
    assert torch.equal(augmented_mask[:, 0], torch.ones(2, dtype=torch.bool))
    # The instruction context remains byte-for-byte unchanged.
    assert torch.equal(context, original_context)


def test_pim_disable_preserves_the_global_behavior_prefix():
    cfg = ExactZevaPolicyInjectionConfig(
        memory_dim=8, context_dim=16, global_dim=8, phase_dim=4, effect_dim=4,
        effect_history_length=2, action_dim=3, action_horizon=5,
        prior_hidden_dim=8, prior_num_heads=2, action_hidden_dim=6,
    )
    adapter = ExactZevaPolicyInjectionAdapter(cfg)
    task = torch.randn(2, 8)
    causal_prompt = torch.randn(2, 8)
    pim_mask = torch.ones(2, 4, dtype=torch.bool)
    expected = adapter.behavior_global_projector(task)[:, None]
    actual = adapter.behavior_prefix_slot(
        task, causal_prompt, pim_mask, enable_pim=False
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
