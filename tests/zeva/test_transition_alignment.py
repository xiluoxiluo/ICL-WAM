import torch

from fastwam.zeva.schemas import build_transition_view


def test_32_actions_align_to_eight_transitions_and_padding_is_masked():
    actions = torch.zeros(1, 32, 14)
    frames = torch.zeros(1, 9, 3, 8, 8)
    action_valid = torch.ones(1, 32, dtype=torch.bool)
    action_valid[:, 8:12] = False
    result = build_transition_view(actions, frames, action_valid=action_valid)
    assert result["transition_actions"].shape == (1, 8, 4, 14)
    assert result["transition_valid"].tolist() == [[True, True, False, True, True, True, True, True]]
