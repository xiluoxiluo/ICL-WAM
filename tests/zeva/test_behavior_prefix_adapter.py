import torch

from fastwam.zeva import BehaviorPrefixAdapter, BehaviorPrefixAdapterConfig, CausalPromptEncoder


def test_gate_zero_is_exact_zero_and_training_has_gradients():
    prompt = CausalPromptEncoder()
    adapter = BehaviorPrefixAdapter(BehaviorPrefixAdapterConfig())
    args = (torch.randn(2, 256), torch.randn(2, 128), torch.randn(2, 4, 128), torch.ones(2, 4, dtype=torch.bool), torch.randn(2, 4, 128), torch.randn(2, 4, 128), torch.ones(2, 4, dtype=torch.bool))
    memory, mask = prompt(*args)
    adapter.eval()
    assert torch.equal(adapter.gated(memory, mask), torch.zeros(2, 32, 1024))
    adapter.train(); prompt.train()
    loss = adapter.gated(memory, mask).square().mean(); loss.backward()
    assert adapter.pim_gate.grad is not None and torch.isfinite(adapter.pim_gate.grad)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in prompt.parameters())
