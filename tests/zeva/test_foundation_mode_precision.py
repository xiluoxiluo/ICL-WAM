"""Exercise the real Hydra/runtime constructor and predictive AMP contract."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.zeva.checkpoint import (
    checkpoint_foundation_mode,
    resolve_joint_foundation_checkpoint,
)
from test_ego_policy_alignment import predictive_model, predictive_sample
from test_joint_foundation import TinyFrozenVAE


PIM_TASK = "robotwin_zeva_ego_pim_3cam_384"
POLICY_TASK = "robotwin_zeva_ego_policy_3cam_384"


def _compose(task=PIM_TASK, *overrides):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        return compose(config_name="train", overrides=[f"task={task}", *overrides])


def test_foundation_mode_is_the_only_protocol_difference():
    joint = _compose(PIM_TASK, "foundation_mode=joint")
    frozen = _compose(PIM_TASK, "foundation_mode=frozen")
    for cfg, mode, recipe in (
        (joint, "joint", "ego_joint_warmup"),
        (frozen, "frozen", "ego_adapter_warmup"),
    ):
        assert cfg.foundation_mode == cfg.model.zeva.foundation_mode == mode
        # Plain composition resolves these values, with no runtime resolver.
        assert cfg.model.zeva.training_recipe == recipe
        assert cfg.model.zeva.freeze_fastwam == (mode == "frozen")
        assert cfg.model.zeva.task_context.mode == "task_language"
        assert cfg.model.mot_checkpoint_mixed_attn
        assert not cfg.model.video_dit_config.use_gradient_checkpointing
        assert not cfg.model.action_dit_config.use_gradient_checkpointing
    resolved = [OmegaConf.to_container(cfg, resolve=True) for cfg in (joint, frozen)]
    for cfg in resolved:
        cfg.pop("foundation_mode")
        for key in ("foundation_mode", "training_recipe", "freeze_fastwam"):
            cfg["model"]["zeva"].pop(key)
    assert resolved[0] == resolved[1]


def test_language_frozen_alias_and_legacy_static_are_distinct():
    frozen = _compose("robotwin_zeva_ego_pim_language_frozen_3cam_384")
    override = _compose(PIM_TASK, "foundation_mode=frozen")
    assert OmegaConf.to_container(frozen, resolve=True) == OmegaConf.to_container(override, resolve=True)
    legacy = _compose("robotwin_zeva_ego_pim_frozen_3cam_384")
    assert legacy.foundation_mode == legacy.model.zeva.foundation_mode == "frozen"
    assert legacy.model.zeva.training_recipe == "ego_adapter_warmup"
    assert legacy.model.zeva.freeze_fastwam
    assert legacy.model.zeva.task_context.mode == "static"


def _tiny_runtime_model(monkeypatch, cfg):
    cfg = deepcopy(cfg)
    for key, value in dict(
        hidden_dim=12, in_dim=4, out_dim=4, ffn_dim=24, text_dim=6,
        freq_dim=8, patch_size=[1, 1, 1], num_heads=2, attn_head_dim=6,
        num_layers=2,
    ).items():
        cfg.model.video_dit_config[key] = value
    for key, value in dict(
        hidden_dim=8, ffn_dim=24, text_dim=6, freq_dim=8,
        num_heads=2, attn_head_dim=6, num_layers=2,
    ).items():
        cfg.model.action_dit_config[key] = value
    for key, value in dict(
        global_dim=8, phase_dim=8, effect_dim=8, hidden_dim=8,
        num_heads=2, brief_length=2, persistent_length=3,
    ).items():
        cfg.model.zeva.prompt[key] = value
    for key, value in dict(
        memory_dim=8, global_dim=8, phase_dim=8, effect_dim=8,
        effect_history_length=2, action_hidden_dim=8, prior_hidden_dim=8,
        prior_num_heads=2, prior_dropout_rate=0.0,
    ).items():
        cfg.model.zeva.adapter[key] = value

    def components(*, dit_config, torch_dtype, device, **kwargs):
        # Only external checkpoint IO is replaced. The resolved expert flags
        # still reach the real Wan/ActionDiT/MoT constructors.
        return SimpleNamespace(
            dit=WanVideoDiT(**dit_config).to(device=device, dtype=torch_dtype),
            vae=TinyFrozenVAE().to(device=device, dtype=torch_dtype),
            text_encoder=None, tokenizer=None,
            dit_path="tiny", vae_path="tiny", text_encoder_path=None, tokenizer_path=None,
        )

    monkeypatch.setattr("fastwam.models.wan22.fastwam.load_wan22_ti2v_5b_components", components)
    return instantiate(
        cfg.model, device="cpu", model_dtype=torch.bfloat16,
        skip_dit_load_from_pretrain=True, action_dit_pretrained_path=None,
        proprio_dim=None,
    )


@pytest.mark.parametrize("task,mode", [
    (POLICY_TASK, "frozen"), (PIM_TASK, "frozen"), (PIM_TASK, "joint"),
])
def test_hydra_runtime_constructor_amp_updates_and_checkpoint_precision(monkeypatch, tmp_path, task, mode):
    cfg = _compose(task, *([f"foundation_mode={mode}"] if task == PIM_TASK else []))
    model = _tiny_runtime_model(monkeypatch, cfg)
    model.configure_zeva_trainable_state()
    model.prepare_zeva_training_precision()
    assert model.zeva_foundation_mode == mode
    assert model.mot.mot_checkpoint_mixed_attn == cfg.model.mot_checkpoint_mixed_attn
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in model.vae.parameters())
    assert all(p.requires_grad == (mode == "joint") for p in model.mot.parameters())
    if mode == "frozen":
        assert all(p.dtype == torch.bfloat16 for p in model.mot.parameters())
    assert model.zeva_parameter_report()["trainable_dtypes"] == ["torch.float32"]

    parameters = model.zeva_trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.999))
    effect = model.zeva_behavior_prefix_adapter.behavior_effect_projector.weight
    original_effect = effect.detach().clone()
    original_backbone = model.video_expert.patch_embedding.weight.detach().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = model(predictive_sample())
    assert torch.isfinite(loss)
    loss.backward()
    assert effect.grad is not None and effect.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(effect, original_effect)
    assert all(state[key].dtype == torch.float32 for state in optimizer.state.values()
               for key in ("exp_avg", "exp_avg_sq"))
    if mode == "frozen":
        assert all(p.grad is None for p in model.mot.parameters())
        torch.testing.assert_close(model.video_expert.patch_embedding.weight, original_backbone, rtol=0, atol=0)
    else:
        assert not torch.equal(model.video_expert.patch_embedding.weight, original_backbone)

    model.to(device="cpu")
    with pytest.raises(ValueError, match="must remain FP32"):
        model.to(dtype=torch.bfloat16)
    assert effect.dtype == torch.float32

    path = tmp_path / "step_000001_addon.pt"
    model.save_zeva_addon_checkpoint(path, step=1)
    payload = torch.load(path, weights_only=False)
    assert payload["foundation_mode"] == mode
    assert payload["trainable_precision"] == "float32"
    assert checkpoint_foundation_mode(payload) == mode
    sidecar = resolve_joint_foundation_checkpoint(path, payload)
    assert (sidecar is not None) == (mode == "joint")
    if sidecar is not None:
        assert torch.load(sidecar, weights_only=False)["trainable_precision"] == "float32"

    target = _tiny_runtime_model(monkeypatch, cfg)
    target.configure_zeva_trainable_state()
    target.prepare_zeva_training_precision()
    target.load_zeva_addon_checkpoint(path)
    torch.testing.assert_close(target.zeva_behavior_prefix_adapter.behavior_effect_projector.weight,
                               effect, rtol=0, atol=0)
    restored_optimizer = torch.optim.AdamW(target.zeva_trainable_parameters(), lr=1e-4, betas=(0.9, 0.999))
    restored_optimizer.load_state_dict(optimizer.state_dict())
    assert all(state[key].dtype == torch.float32 for state in restored_optimizer.state.values()
               for key in ("exp_avg", "exp_avg_sq"))


@pytest.mark.parametrize("override,match", [
    ("model.zeva.foundation_mode=unknown", "must be 'frozen' or 'joint'"),
    ("model.zeva.foundation_mode=frozen", "training_recipe disagree"),
    ("model.zeva.freeze_fastwam=true", "must agree with foundation_mode"),
])
def test_conflicting_mode_is_rejected_before_loading_large_backbone(monkeypatch, override, match):
    cfg = _compose(PIM_TASK, override)

    def unexpected_load(**kwargs):
        raise AssertionError("large backbone must not load before validation")

    monkeypatch.setattr("fastwam.models.wan22.fastwam.load_wan22_ti2v_5b_components", unexpected_load)
    with pytest.raises(InstantiationException, match=match):
        instantiate(cfg.model, device="cpu")


def test_checkpoint_mode_accepts_legacy_and_rejects_contradictions():
    assert checkpoint_foundation_mode({}) == "frozen"
    assert checkpoint_foundation_mode({"training_recipe": "ego_joint_warmup"}) == "joint"
    with pytest.raises(ValueError, match="training_recipe disagree"):
        checkpoint_foundation_mode({"training_recipe": "ego_joint_warmup", "foundation_mode": "frozen"})


@pytest.mark.parametrize("options", [{"text_cfg_scale": 2.0}, {"negative_prompt": "unsafe"}])
def test_action_inference_rejects_unimplemented_text_guidance(options):
    with pytest.raises(ValueError, match="classifier-free text guidance is not implemented"):
        predictive_model().infer_action(None, torch.zeros(1, 3, 16, 16), 32, **options)
