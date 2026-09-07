import torch

from fastwam.zeva import CausalCTEHistory, CausalTransitionEncoder, CausalTransitionEncoderConfig


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


def test_cte_history_encodes_only_new_boundaries():
    calls = []

    def encoder(history):
        calls.append(int(history.shape[1]))
        return history + 1.0

    cte = _model()
    history = CausalCTEHistory(cte, frame_encoder=encoder)
    first = torch.zeros(3, 8, 8)
    history.reset(first)
    action = torch.zeros(4, 14)
    history.append_transition(action, torch.ones(3, 8, 8))
    history.append_transition(action, torch.full((3, 8, 8), 2.0))
    assert calls == [1, 1, 1]
    output = history.forward()
    assert output["phase"].shape[1] == 3
    torch.testing.assert_close(history.phase_at_raw_step(4), output["phase"][:, 1])
    try:
        history.phase_at_raw_step(2)
    except KeyError:
        pass
    else:
        raise AssertionError("unobserved raw-step query was accepted")


def test_cte_history_matches_direct_full_prefix_encoding():
    model = _model()
    frames = torch.randn(1, 9, 3, 32, 32)
    actions = torch.randn(1, 8, 4, 14)
    direct = model(frames, actions)
    history = CausalCTEHistory(model)
    history.reset(frames[0, 0])
    for index in range(8):
        history.append_transition(actions[0, index], frames[0, index + 1])
    online = history.forward()
    torch.testing.assert_close(online["phase"], direct["phase"], rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(online["effect_post"], direct["effect_post"], rtol=0.0, atol=1.0e-6)


def test_cte_history_does_not_commit_failed_encoded_boundary():
    calls = []

    def encoder(history):
        calls.append(int(history.shape[1]))
        if len(calls) == 2:
            raise RuntimeError("synthetic encoder failure")
        return history

    history = CausalCTEHistory(_model(), frame_encoder=encoder)
    history.reset(torch.zeros(3, 8, 8))
    try:
        history.append_transition(torch.zeros(4, 14), torch.ones(3, 8, 8))
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed frame encoding was accepted")
    assert history._raw_steps == [0]
    assert history._actions == []
