import torch

from fastwam.zeva import BehaviorPrefixAdapter, BehaviorPrefixAdapterConfig, CausalPromptEncoder


def test_gate_zero_is_exact_zero_and_training_has_gradients():
    prompt = CausalPromptEncoder()
    adapter = BehaviorPrefixAdapter(BehaviorPrefixAdapterConfig())
    args = (torch.randn(2, 256), torch.randn(2, 128), torch.randn(2, 4, 128), torch.ones(2, 4, dtype=torch.bool), torch.randn(2, 4, 128), torch.randn(2, 4, 128), torch.ones(2, 4, dtype=torch.bool))
    memory, mask = prompt(*args)
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
