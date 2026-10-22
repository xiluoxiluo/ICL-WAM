"""Exercise separate behavior slots through actual Wan/MoT attention on CPU."""

from types import SimpleNamespace

import pytest
import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.zeva import (
    CausalPromptConfig,
    CausalPromptEncoder,
    ExactZevaPolicyInjectionAdapter,
    ExactZevaPolicyInjectionConfig,
)


def _model(*, stage="pim_adapter"):
    # Different expert widths and B != heads expose accidental broadcasting
    # that a single-example or same-width attention mock would miss.
    torch.manual_seed(103)
    video = WanVideoDiT(
        hidden_dim=12, in_dim=4, out_dim=4, ffn_dim=24, text_dim=6,
        freq_dim=8, eps=1e-6, patch_size=(1, 1, 1), num_heads=2,
        attn_head_dim=6, num_layers=2, has_image_input=False,
        seperated_timestep=True, video_attention_mask_mode="first_frame_causal",
    )
    action = ActionDiT(
        hidden_dim=8, action_dim=14, ffn_dim=24, text_dim=6, freq_dim=8,
        eps=1e-6, num_heads=2, attn_head_dim=6, num_layers=2,
    )
    vae = SimpleNamespace(
        model=SimpleNamespace(encode=lambda x, scale: x.new_zeros((x.shape[0], 4, 1, 2, 2))),
        scale=None,
        to=lambda *args, **kwargs: None,
    )
    model = FastWAM(video, action, MoT({"video": video, "action": action}), vae, text_dim=6)
    variant = "legacy" if stage == "policy_injection" else "cross_attempt"
    model.attach_zeva_addon(
        CausalPromptEncoder(CausalPromptConfig(
            global_dim=8, phase_dim=4, effect_dim=4, brief_length=2,
            persistent_length=3, hidden_dim=8, num_heads=2, pim_variant=variant,
        )),
        ExactZevaPolicyInjectionAdapter(ExactZevaPolicyInjectionConfig(
            memory_dim=8, context_dim=12, global_dim=8, phase_dim=4, effect_dim=4,
            effect_history_length=2, action_dim=14, action_horizon=32,
            prior_hidden_dim=8, prior_num_heads=2, action_hidden_dim=8,
            prior_dropout_rate=0.0, pim_variant=variant, prefix_layout="zeva_ego",
        )),
    )
    model.zeva_task_context_mode = "static"
    model.set_zeva_training_stage(stage)
    model.configure_zeva_trainable_state()
    return model


def _prepared_video(model, *, batch=3, frames=1):
    context = torch.randn(batch, 3, 6)
    context_mask = torch.tensor([[True, True, False]]).expand(batch, -1)
    prepared = model.video_expert.prepare(
        x=torch.randn(batch, 4, frames, 2, 2), timestep=torch.ones(batch),
        context=context, context_mask=context_mask, fuse_vae_embedding_in_latents=True,
    )
    return prepared, context, context_mask


def test_separate_slots_have_causal_per_example_attention_visibility():
    model = _model()
    valid = torch.tensor([[True, True, True], [True, False, True], [True, True, False]])
    mask = model._build_mot_attention_mask(
        video_seq_len=11, action_seq_len=32, video_tokens_per_frame=4,
        device=torch.device("cpu"), behavior_prefix_len=3,
        behavior_prefix_mask=valid, behavior_prefix_layout="zeva_ego",
    )
    assert mask.shape == (3, 1, 43, 43)
    for b in range(3):
        for slot in range(3):
            if not valid[b, slot]:
                assert not mask[b, :, :, slot].any()
            else:
                assert mask[b, 0, :7, slot].all()
                assert mask[b, 0, 11:, slot].all()
    # The three conditions and clean first frame form one visible block.
    assert mask[:, :, :3, 3:7].all()
    assert not mask[:, :, :7, 7:].any()
    # Action sees the first frame and its own tokens, but no future video.
    assert mask[:, :, 11:, 3:7].all()
    assert mask[:, :, 11:, 11:].all()
    assert not mask[:, :, 11:, 7:11].any()


def test_clean_time_text_and_physical_positions_survive_prepend():
    model = _model()
    prepared, _, _ = _prepared_video(model, frames=2)
    video, _, time, _, text_mask, freqs, *_ = prepared
    valid = torch.tensor([[True, True, True], [True, False, True], [True, True, False]])
    prefix = torch.randn(3, 3, 12)
    prefix[~valid] = float("nan")
    combined, combined_time, combined_freqs, combined_text, length = (
        model._prepend_exact_zeva_behavior_token_to_video_stream(
            prefix, video, time, freqs, text_mask,
            behavior_prefix_mask=valid, prefix_layout="zeva_ego",
        )
    )
    assert length == 3
    assert torch.isfinite(combined).all()
    torch.testing.assert_close(combined_time[:, :3], time[:, :1].expand(-1, 3, -1, -1))
    torch.testing.assert_close(combined_text[:, :3], text_mask[:, :1].expand(-1, 3, -1) & valid[..., None])
    torch.testing.assert_close(combined_freqs[:3], torch.ones_like(combined_freqs[:3]))
    torch.testing.assert_close(combined[:, 3:], video, rtol=0, atol=0)
    torch.testing.assert_close(combined_time[:, 3:], time, rtol=0, atol=0)
    torch.testing.assert_close(combined_freqs[3:], freqs, rtol=0, atol=0)
    torch.testing.assert_close(combined_text[:, 3:], text_mask)


def test_real_mot_masked_slots_match_physically_removed_slots_in_mixed_batch():
    model = _model().eval()
    prepared, context, context_mask = _prepared_video(model)
    video, _, time, video_context, text_mask, freqs, *_, per_frame = prepared
    valid = torch.tensor([[True, False, False], [True, True, False], [True, False, True]])
    prefix = torch.randn(3, 3, 12)
    prefix[~valid] = float("nan")
    video, time, freqs, text_mask, _ = model._prepend_exact_zeva_behavior_token_to_video_stream(
        prefix, video, time, freqs, text_mask,
        behavior_prefix_mask=valid, prefix_layout="zeva_ego",
    )
    video_len = video.shape[1]
    mask = model._build_mot_attention_mask(
        video_len, 32, per_frame, torch.device("cpu"), 3, valid, "zeva_ego",
    )
    action, _, action_time, action_context, action_text, action_freqs = model.action_expert.prepare(
        torch.randn(3, 32, 14), torch.full((3,), 0.5), context, context_mask,
    )

    def run(v, vt, vf, vc, vm, a, at, ac, am, full_mask):
        k, values = model.mot.prefill_video_cache_tensor(
            v, vf, vt, vc, vm, full_mask[..., :v.shape[1], :v.shape[1]],
        )
        return model.mot.forward_action_with_video_cache_tensor(
            a, action_freqs, at, ac, am, k, values, full_mask[..., v.shape[1]:, :],
        )

    with torch.no_grad():
        padded = run(video, time, freqs, video_context, text_mask,
                     action, action_time, action_context, action_text, mask)
        for b in range(3):
            keep_video = torch.cat((valid[b].nonzero().flatten(), torch.arange(3, video_len)))
            keep_all = torch.cat((keep_video, torch.arange(video_len, video_len + 32)))
            compact_mask = mask[b:b + 1].index_select(-2, keep_all).index_select(-1, keep_all)
            compact = run(
                video[b:b + 1, keep_video], time[b:b + 1, keep_video], freqs[keep_video],
                video_context[b:b + 1], text_mask[b:b + 1, keep_video],
                action[b:b + 1], action_time[b:b + 1], action_context[b:b + 1], action_text[b:b + 1],
                compact_mask,
            )
            torch.testing.assert_close(padded[b:b + 1], compact, rtol=2e-5, atol=2e-6)


def _sample():
    return {
        "_training_mode": "zeva_stage2",
        "video": torch.zeros(3, 3, 1, 16, 16),
        "action": torch.randn(3, 32, 14),
        "action_valid": torch.ones(3, 32, dtype=torch.bool),
        "context": torch.randn(3, 3, 6),
        "context_mask": torch.ones(3, 3, dtype=torch.bool),
        "task_context": torch.randn(3, 8), "phase": torch.randn(3, 4),
        "bit_effects": torch.randn(3, 2, 4),
        "bit_mask": torch.tensor([[False, False], [False, True], [True, True]]),
        "pim_phases": torch.randn(3, 3, 4), "pim_effects": torch.randn(3, 3, 4),
        "pim_mask": torch.tensor([[False, False, False], [False, True, True], [True, True, True]]),
    }


@pytest.mark.parametrize("stage", ["policy_injection", "pim_adapter"])
def test_action_loss_backpropagates_through_frozen_mot_to_correct_stage(stage):
    model = _model(stage=stage)
    model.zeva_parameter_report()
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    sample = _sample()
    optimizer = torch.optim.AdamW(model.zeva_trainable_parameters(), lr=5e-4)
    loss, metrics = model(sample)
    assert torch.isfinite(loss)
    assert metrics["prefix/token_capacity"] == 3
    assert metrics["prefix/global_valid_fraction"] == 1
    assert metrics["prefix/effect_valid_fraction"] == pytest.approx(2 / 3)
    assert metrics["prefix/pim_valid_fraction"] == pytest.approx(0 if stage == "policy_injection" else 2 / 3)
    assert "gate" not in metrics
    loss.backward()
    adapter = model.zeva_behavior_prefix_adapter
    target = adapter.behavior_effect_projector if stage == "policy_injection" else adapter.prefix_project
    assert target.weight.grad is not None and target.weight.grad.norm() > 0
    if stage == "pim_adapter":
        assert adapter.pim_to_global.weight.grad is not None and adapter.pim_to_global.weight.grad.norm() > 0
        assert any(p.grad is not None and p.grad.norm() > 0 for p in model.zeva_prompt_encoder.parameters())
        assert adapter.behavior_effect_projector.weight.grad is None
    optimizer.step()
    for name, original in frozen.items():
        p = model.get_parameter(name)
        assert p.grad is None
        torch.testing.assert_close(p, original, rtol=0, atol=0)


def test_same_noise_inference_preserves_parent_and_reads_effect_and_pim_independently():
    model = _model().eval()
    adapter = model.zeva_behavior_prefix_adapter
    task, phase, prompt = torch.randn(1, 8), torch.randn(1, 4), torch.randn(1, 8)
    bit = torch.randn(1, 2, 4)
    valid_bit = torch.ones(1, 2, dtype=torch.bool)

    def infer(*, mode="pim_on", has_pim=False, effects=bit, has_effect=True, prior=False):
        pim_mask = torch.full((1, 3), has_pim, dtype=torch.bool)
        bit_mask = valid_bit if has_effect else torch.zeros_like(valid_bit)
        mean, _ = adapter.prior_for_prompt(
            task, phase, effects, bit_mask, prompt, pim_mask, enable_pim=(mode == "pim_on"),
        )
        return model.infer_action_zeva(
            prompt=None, input_image=torch.zeros(1, 3, 16, 16),
            context=torch.zeros(1, 3, 6), context_mask=torch.ones(1, 3, dtype=torch.bool),
            action_horizon=32, num_inference_steps=2, seed=107,
            causal_prompt=prompt, pim_mask=pim_mask, zeva_task_tokens=task,
            zeva_bit_effects=effects, zeva_bit_mask=bit_mask,
            zeva_action_residual=adapter.action_prior_residual(mean, training=False),
            zeva_mode=mode, enable_action_prior=prior,
        )["action"]

    # PIM off/empty is exactly this topology's Stage2A parent with identical noise.
    torch.testing.assert_close(infer(mode="zeva_stage2", has_pim=True, prior=True), infer(prior=True), rtol=0, atol=0)
    off = infer()
    assert not torch.allclose(infer(has_pim=True), off, rtol=1e-6, atol=1e-7)
    changed_effect = bit.clone()
    changed_effect[:, -1] += 2.0
    assert not torch.allclose(infer(effects=changed_effect), off, rtol=1e-6, atol=1e-7)
    # Masked data cannot enter via prefix even if the unavailable effect is NaN.
    empty_effect = infer(has_effect=False)
    torch.testing.assert_close(
        infer(has_effect=False, effects=torch.full_like(bit, float("nan"))),
        empty_effect, rtol=0, atol=0,
    )
