import torch

from fastwam.zeva import CausalPromptConfig, CausalPromptEncoder, task_tokens_from_context


def test_task_token_pooling_uses_full_masked_context():
    context = torch.zeros(1, 2, 8)
    context[0, 0, 0] = 2.0
    context[0, 1, 7] = 4.0
    tokens = task_tokens_from_context(context, torch.ones(1, 2, dtype=torch.bool), 2)
    assert tokens.shape == (1, 2)
    assert float(tokens[0, 1]) > 0.0


def test_task_token_pooling_rejects_bad_mask():
    context = torch.zeros(1, 2, 8)
    try:
        task_tokens_from_context(context, torch.ones(1, 3, dtype=torch.bool), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid context mask was accepted")


def test_causal_prompt_returns_single_fused_vector():
    encoder = CausalPromptEncoder()
    args = (
        torch.randn(2, 256),
        torch.randn(2, 128),
        torch.zeros(2, 4, 128),
        torch.zeros(2, 4, dtype=torch.bool),
        torch.zeros(2, 4, 128),
        torch.zeros(2, 4, 128),
        torch.zeros(2, 4, dtype=torch.bool),
    )
    causal_prompt = encoder(*args)
    assert causal_prompt.shape == (2, encoder.config.hidden_dim)


def test_pim_dropout_applies_only_in_training_and_keeps_bos_slot(monkeypatch):
    config = CausalPromptConfig(
        global_dim=8,
        phase_dim=4,
        effect_dim=4,
        brief_length=2,
        persistent_length=2,
        hidden_dim=8,
        num_heads=2,
        pim_context_dropout=0.5,
        pim_support_dropout=0.5,
    )
    encoder = CausalPromptEncoder(config)
    args = (
        torch.randn(1, 8),
        torch.randn(1, 4),
        torch.randn(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, 2, 4),
        torch.randn(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
    )

    # Force every slot to be dropped. The encoder must retain one BOS key so
    # MultiheadAttention never receives an all-masked row.
    monkeypatch.setattr(torch, "rand", lambda shape, device=None: torch.zeros(shape, device=device))
    masks = []

    def capture(_module, _inputs, kwargs):
        masks.append(kwargs["key_padding_mask"].detach().clone())

    handles = [
        encoder.brief_attention.register_forward_pre_hook(capture, with_kwargs=True),
        encoder.persistent_attention.register_forward_pre_hook(capture, with_kwargs=True),
    ]
    try:
        encoder.train()
        encoder(*args)
    finally:
        for handle in handles:
            handle.remove()
    assert len(masks) == 2
    assert torch.equal(masks[0], torch.tensor([[False, True]]))
    assert torch.equal(masks[1], torch.tensor([[False, True]]))

    # Evaluation must bypass dropout entirely, including its random-number call.
    def fail_rand(*_args, **_kwargs):
        raise AssertionError("dropout RNG was used during inference")

    monkeypatch.setattr(torch, "rand", fail_rand)
    encoder.eval()
    encoder(*args)
