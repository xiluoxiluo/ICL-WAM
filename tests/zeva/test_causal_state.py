import torch

from fastwam.zeva import CausalTransitionEncoder, CausalTransitionEncoderConfig


def _model() -> CausalTransitionEncoder:
    return CausalTransitionEncoder(
        CausalTransitionEncoderConfig(
            hidden_dim=16, retrieval_dim=8, phase_dim=8, effect_dim=8,
            num_layers=1, num_heads=4,
        )
    ).eval()


def test_full_history_contract_and_effect_cadence():
    model = _model()
    frames = torch.randn(1, 9, 3, 32, 32)
    actions = torch.randn(1, 8, 4, 14)
    output = model(frames, actions)
    assert output["phase"].shape == (1, 9, 8)
    assert output["effect_post"].shape == (1, 2, 8)
    assert output["effect_complete"].tolist() == [[True, True]]


def test_right_shift_hides_current_transition_from_current_phase():
    model = _model()
    frames = torch.randn(1, 9, 3, 32, 32)
    actions = torch.randn(1, 8, 4, 14)
    altered = actions.clone()
    altered[:, 3] += 100.0
    a, b = model(frames, actions), model(frames, altered)
    # Transition k is injected at state k+1, so states through k are stable.
    torch.testing.assert_close(a["phase"][:, :4], b["phase"][:, :4])
    assert not torch.equal(a["phase"][:, 4], b["phase"][:, 4])


def test_future_frames_do_not_change_earlier_phase_states():
    """The recurrent CTE path must not read observations after the query state."""
    model = _model()
    frames = torch.randn(1, 9, 3, 32, 32)
    actions = torch.randn(1, 8, 4, 14)
    altered = frames.clone()
    altered[:, 5:] += 100.0
    a, b = model(frames, actions), model(altered, actions)
    torch.testing.assert_close(a["phase"][:, :5], b["phase"][:, :5])
    assert not torch.equal(a["phase"][:, 5], b["phase"][:, 5])


def test_partial_history_has_no_complete_effect_window():
    model = _model()
    frames = torch.randn(1, 4, 3, 24, 24)
    actions = torch.randn(1, 3, 4, 14)
    output = model(frames, actions)
    assert output["effect_post"].shape == (1, 0, 8)
    assert output["effect_complete"].shape == (1, 0)
