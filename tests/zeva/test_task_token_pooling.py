import torch

from fastwam.zeva import CausalPromptEncoder, task_tokens_from_context


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


def test_prompt_output_mask_matches_token_order_for_empty_memory():
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
    _tokens, mask = encoder(*args)
    assert mask.tolist() == [[True, True, False, False], [True, True, False, False]]
