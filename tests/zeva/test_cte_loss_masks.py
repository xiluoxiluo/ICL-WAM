import torch

from fastwam.zeva import (
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    CTELossConfig,
    causal_transition_encoder_loss,
)


def test_cte_default_weights_follow_zeva_reference():
    cfg = CTELossConfig()
    assert cfg.action_weight == 1.0
    assert cfg.vision_weight == 1.0
    assert cfg.task_weight == 0.2
    assert cfg.phase_weight == 0.1
    assert cfg.effect_weight == 0.25


def test_invalid_transition_is_excluded_from_action_and_effect_losses():
    model = CausalTransitionEncoder(
        CausalTransitionEncoderConfig(
            hidden_dim=16, retrieval_dim=8, phase_dim=8, effect_dim=8,
            num_layers=1, num_heads=4,
        )
    )
    frames = torch.randn(1, 9, 3, 24, 24)
    actions = torch.randn(1, 8, 4, 14)
    frame_valid = torch.ones(1, 9, dtype=torch.bool)
    transition_valid = torch.ones(1, 8, 4, dtype=torch.bool)
    transition_valid[:, 2] = False
    outputs = model(frames, actions, frame_valid, transition_valid)
    assert not bool(outputs["transition_complete"][:, 2].any())
    assert not bool(outputs["effect_complete"][:, 0].any())
    losses = causal_transition_encoder_loss(
        outputs, actions, frame_valid, torch.tensor([0]), CTELossConfig()
    )
    assert torch.isfinite(losses["total"])
