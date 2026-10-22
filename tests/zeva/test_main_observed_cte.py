import pytest
import torch

from fastwam.datasets.zeva_robotwin_dataset import EpisodeMetadata
from fastwam.zeva.causal_transition_encoder import CausalTransitionEncoder, CausalTransitionEncoderConfig
from fastwam.zeva.cte_losses import causal_transition_encoder_loss
from fastwam.zeva.cte_training import MainCTEEpisodeDataset, pad_cte_trajectories
from fastwam.zeva.stage1_sampling import CTETrainIndex


def _model():
    return CausalTransitionEncoder(CausalTransitionEncoderConfig(
        hidden_dim=16, phase_dim=8, effect_dim=8, retrieval_dim=8, num_layers=1, num_heads=4,
    )).eval()


def test_post_effect_requires_real_future_observation_but_never_changes_earlier_phase():
    torch.manual_seed(8)
    model = _model()
    frames, actions = torch.randn(1, 9, 3, 16, 16), torch.randn(1, 8, 4, 14)
    original = model(frames, actions)
    changed_actions = actions.clone()
    changed_actions[:, :4] += 10
    action_changed = model(frames, changed_actions)
    # A posterior outcome cannot be an action-only prediction in disguise.
    torch.testing.assert_close(original["effect_post"], action_changed["effect_post"])
    assert not torch.equal(original["effect_pre"][:, 0], action_changed["effect_pre"][:, 0])
    changed_frames = frames.clone()
    changed_frames[:, 4] += torch.randn_like(changed_frames[:, 4]) * 5
    observed_changed = model(changed_frames, actions)
    torch.testing.assert_close(original["phase"][:, :4], observed_changed["phase"][:, :4])
    assert not torch.equal(original["effect_post"][:, 0], observed_changed["effect_post"][:, 0])
    prefix = model(frames[:, :4], actions[:, :3])
    assert prefix["effect_post"].shape[1] == 0


def test_missing_intermediate_observation_never_becomes_complete_effect():
    model = _model()
    frames, actions = torch.randn(1, 5, 3, 16, 16), torch.randn(1, 4, 4, 14)
    valid = torch.ones(1, 5, dtype=torch.bool)
    valid[:, 2] = False
    result = model(frames, actions, valid)
    assert result["effect_complete"].tolist() == [[False]]


def _window(frames, actions, start):
    return {
        "dataset_index": start, "cte_frames": frames[start // 4:start // 4 + 9],
        "transition_actions": actions[start // 4:start // 4 + 8],
        "frame_valid": torch.ones(9, dtype=torch.bool),
        "transition_valid": torch.ones(8, 4, dtype=torch.bool),
        "episode": EpisodeMetadata("/source::episode-0", "task", "task", "do it", start, 65),
    }


def test_episode_stitching_preserves_boundary_actions_and_recurrent_prefix():
    torch.manual_seed(14)
    frames, actions = torch.randn(17, 3, 16, 16), torch.randn(16, 4, 14)
    base = {start: _window(frames, actions, start) for start in (0, 32)}
    rows = [CTETrainIndex(start, "/source::episode-0", "task", start) for start in (0, 32)]
    sample = MainCTEEpisodeDataset(base, rows, preencoded=True)[0]
    torch.testing.assert_close(sample["cte_frames"], frames)
    torch.testing.assert_close(sample["transition_actions"], actions)
    model = _model()
    actual = model(sample["cte_frames"][None], sample["transition_actions"][None])
    direct = model(frames[None], actions[None])
    torch.testing.assert_close(actual["phase"], direct["phase"])
    with pytest.raises(ValueError, match="contiguous"):
        MainCTEEpisodeDataset(base, rows[1:], preencoded=True)


def test_variable_episodes_padding_has_no_effect_target_and_can_backpropagate():
    frames = [torch.randn(length, 3, 16, 16) for length in (5, 9)]
    actions = [torch.randn(length - 1, 4, 14) for length in (5, 9)]
    padded = pad_cte_trajectories(frames, actions,
        [torch.ones(length, dtype=torch.bool) for length in (5, 9)],
        [torch.ones(length - 1, 4, dtype=torch.bool) for length in (5, 9)])
    observations, controls, valid, completed = padded
    model = _model()
    result = model(observations, controls, valid, completed)
    assert result["effect_complete"].tolist() == [[True, False], [True, True]]
    losses = causal_transition_encoder_loss(result, controls, valid, torch.tensor([0, 0]))
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.effect_post_head[1].weight.grad is not None
