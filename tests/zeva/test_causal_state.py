import torch

from fastwam.zeva import CausalTransitionEncoder, CausalTransitionEncoderConfig


def test_online_update_uses_state_and_returns_one_effect():
    model = CausalTransitionEncoder(CausalTransitionEncoderConfig(hidden_dim=16, retrieval_dim=8, phase_dim=8, effect_dim=8, num_layers=1, num_heads=4))
    action = torch.randn(1, 4, 14)
    before, after = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
    state_a, _, effect = model.update(torch.zeros(1, 16), action, before, after)
    state_b, _, _ = model.update(torch.ones(1, 16), action, before, after)
    assert effect.shape == (1, 8)
    assert not torch.equal(state_a, state_b)


def test_online_effect_is_action_conditioned():
    model = CausalTransitionEncoder(
        CausalTransitionEncoderConfig(hidden_dim=16, retrieval_dim=8, phase_dim=8, effect_dim=8, num_layers=1, num_heads=4)
    )
    before, after = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
    state = torch.zeros(1, 16)
    action_a = torch.zeros(1, 4, 14)
    action_b = torch.randn(1, 4, 14)
    effect_a = model.update(state, action_a, before, after)[2]
    effect_b = model.update(state, action_b, before, after)[2]
    assert not torch.equal(effect_a, effect_b)


def test_online_update_matches_two_frame_causal_forward_handoff():
    model = CausalTransitionEncoder(
        CausalTransitionEncoderConfig(
            hidden_dim=16, retrieval_dim=8, phase_dim=8, effect_dim=8,
            num_layers=1, num_heads=4,
        )
    )
    model.eval()
    state = torch.randn(1, 16)
    action = torch.randn(1, 4, 14)
    before, after = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
    online_state, online_phase, _ = model.update(state, action, before, after)
    frames = torch.cat((before, after), dim=0).unsqueeze(0)
    transitions = action.unsqueeze(1)
    offline = model(
        frames,
        transitions,
        initial_state=state,
        initial_state_mask=torch.ones(1, dtype=torch.bool),
    )
    torch.testing.assert_close(online_state, offline["causal_interaction_state"][:, 1])
    torch.testing.assert_close(online_phase, offline["phase"][:, 1])


def test_observed_effect_uses_after_frame_but_prediction_does_not():
    model = CausalTransitionEncoder(
        CausalTransitionEncoderConfig(hidden_dim=16, retrieval_dim=8, phase_dim=8, effect_dim=8, num_layers=1, num_heads=4)
    )
    frames = torch.randn(1, 9, 3, 32, 32)
    actions = torch.randn(1, 8, 4, 14)
    altered = frames.clone()
    altered[:, -1] = torch.randn_like(altered[:, -1])
    a = model(frames, actions)
    b = model(altered, actions)
    torch.testing.assert_close(a["transition_effect_prediction"][:, :-1], b["transition_effect_prediction"][:, :-1])
    assert not torch.equal(a["transition_effect"][:, -1], b["transition_effect"][:, -1])
