"""Original-main conditioning, numerics, and complete FastWAM bundles on CPU."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from omegaconf import OmegaConf

from fastwam.zeva import CausalPromptEncoder
from fastwam.zeva.behavior_prefix_adapter import ExactZevaPolicyInjectionAdapter, gaussian_prior_nll
from fastwam.zeva.checkpoint import checkpoint_sha256, resolve_joint_foundation_checkpoint
from fastwam.zeva.main_data import build_success_manifest, select_main_success_rows, validate_main_artifact
from fastwam.zeva.stage1_sampling import CTETrainIndex
from fastwam.zeva_memory_contract import main_pim_memory_contract
from test_separate_prefix_integration import _model, _sample


def main_model(stage="pim_adapter", *, joint=False, metadata=None):
    model = _model(stage="policy_injection")
    reader = replace(model.zeva_prompt_encoder.config, numerical_contract="zeva_main_v1")
    adapter = replace(model.zeva_behavior_prefix_adapter.config,
                      numerical_contract="zeva_main_v1", prefix_layout="legacy_merged")
    model.attach_zeva_addon(CausalPromptEncoder(reader), ExactZevaPolicyInjectionAdapter(adapter))
    model.zeva_profile = "main_observed"
    model.zeva_training_recipe = f"main_{'joint' if joint else 'frozen'}_{'parent' if stage == 'policy_injection' else 'pim'}"
    model.zeva_pim_memory_contract = main_pim_memory_contract({"bit_size": 2, "pim_top_k": 3})
    if metadata is not None:
        model.zeva_main_artifact_metadata = deepcopy(metadata)
        model.zeva_success_data_identity = deepcopy(metadata["success_data_identity"])
    model.set_zeva_training_stage(stage).configure_zeva_trainable_state()
    return model


def main_metadata(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text('{"total_episodes":4}')
    (source / "meta/episodes.jsonl").write_text("\n".join(
        json.dumps({"episode_index": index, "success": True}) for index in range(4)))
    (source / "stats.json").write_text("{}")
    (source / "meta/tasks.jsonl").write_text("{}")
    manifest = source / "success.json"
    manifest.write_text(json.dumps(build_success_manifest([source])))
    config = OmegaConf.create({"model": {"zeva": {"profile": "main_observed", "offline": {
        "success_manifest_path": str(manifest)}}}, "data": {"train": {
            "dataset_dirs": [str(source)], "pretrained_norm_stats": str(source / "stats.json"),
            "semantic_task_map_path": str(source / "meta/tasks.jsonl"), "is_training_set": True}}})
    rows = [CTETrainIndex(index, f"{source}::episode-{index}", "task", 0) for index in range(4)]
    metadata = select_main_success_rows(rows, config)[1]
    metadata["data_role"] = "train"
    return metadata


def _prompt(reader, sample):
    return reader(sample["task_context"], sample["phase"], sample["bit_effects"], sample["bit_mask"],
                  sample["pim_phases"], sample["pim_effects"], sample["pim_mask"])


def _prior(adapter, sample, prompt):
    return adapter.prior_for_prompt(sample["task_context"], sample["phase"], sample["bit_effects"],
                                    sample["bit_mask"], prompt, sample["pim_mask"])


def main_inference_kwargs():
    return dict(prompt=None, input_image=torch.zeros(1, 3, 16, 16),
                context=torch.randn(1, 3, 6), context_mask=torch.ones(1, 3, dtype=torch.bool),
                action_horizon=32, num_inference_steps=2, causal_prompt=torch.randn(1, 8),
                pim_mask=torch.ones(1, 3, dtype=torch.bool), zeva_task_tokens=torch.randn(1, 8),
                initial_action_noise=torch.randn(1, 32, 14), enable_action_prior=False)


def test_main_prior_reads_completed_bit_while_pim_changes_only_the_merged_prefix():
    model, sample = main_model().eval(), _sample()
    reader, adapter = model.zeva_prompt_encoder, model.zeva_behavior_prefix_adapter
    sample["bit_mask"].fill_(True)
    sample["pim_mask"].fill_(True)
    with torch.no_grad():
        adapter.pim_gate.fill_(0.35)
    prompt = _prompt(reader, sample)
    mean, std = _prior(adapter, sample, prompt)
    changed = {**sample, "pim_effects": sample["pim_effects"] * -3 + 2}
    other_prompt = _prompt(reader, changed)
    assert not torch.equal(prompt, other_prompt)
    other_mean, other_std = _prior(adapter, changed, other_prompt)
    torch.testing.assert_close(other_mean, mean, rtol=0, atol=0)
    torch.testing.assert_close(other_std, std, rtol=0, atol=0)
    tokens, valid = adapter.behavior_prefix_tokens(sample["task_context"], prompt, sample["pim_mask"])
    changed_tokens, _ = adapter.behavior_prefix_tokens(sample["task_context"], other_prompt, sample["pim_mask"])
    assert tokens.shape == (3, 1, 12) and valid.all()
    assert not torch.equal(tokens, changed_tokens)
    changed_bit = {**sample, "bit_effects": sample["bit_effects"].flip(-1) * 2}
    bit_mean, _ = _prior(adapter, changed_bit, prompt)
    assert not torch.equal(bit_mean, mean)
    empty = {**sample, "bit_mask": torch.zeros_like(sample["bit_mask"])}
    empty_nan = {**empty, "bit_effects": torch.full_like(sample["bit_effects"], float("nan"))}
    torch.testing.assert_close(_prior(adapter, empty, prompt), _prior(adapter, empty_nan, prompt), rtol=0, atol=0)


def test_zero_gate_is_exact_parent_action_and_gate_learns_before_the_reader():
    model, sample = main_model(), _sample()
    adapter, reader = model.zeva_behavior_prefix_adapter, model.zeva_prompt_encoder
    sample["pim_mask"].fill_(True)
    kwargs = main_inference_kwargs()
    rng = torch.get_rng_state().clone()
    parent = model.infer_action_zeva(**kwargs, zeva_mode="zeva_stage2")["action"]
    initial = model.infer_action_zeva(**kwargs, zeva_mode="pim_on")["action"]
    torch.testing.assert_close(initial, parent, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    model.configure_zeva_trainable_state()
    optimizer = torch.optim.SGD(model.zeva_trainable_parameters(), lr=0.05)
    loss, _ = model(sample)
    loss.backward()
    assert adapter.pim_gate.grad.abs() > 0
    assert adapter.prefix_project.weight.grad is not None and not adapter.prefix_project.weight.grad.any()
    assert all(parameter.grad is not None and not parameter.grad.any() for parameter in reader.parameters())
    assert all(parameter.grad is None for parameter in adapter.policy_injection_parameters())
    assert all(parameter.grad is None for parameter in model.zeva_foundation_parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = model(sample)
    loss.backward()
    assert adapter.prefix_project.weight.grad.norm() > 0
    assert sum(parameter.grad.abs().sum() for parameter in reader.parameters()) > 0
    changed = model.infer_action_zeva(**kwargs, zeva_mode="pim_on")["action"]
    assert not torch.equal(changed, parent)
    empty = {**kwargs, "pim_mask": torch.zeros_like(kwargs["pim_mask"])}
    torch.testing.assert_close(model.infer_action_zeva(**empty, zeva_mode="pim_on")["action"], parent,
                               rtol=0, atol=0)


def test_main_prior_initialization_resets_attention_before_position_parameters(monkeypatch):
    prior = main_model().zeva_behavior_prefix_adapter.prior
    events = []
    for name in ("progress_attention", "effect_attention"):
        attention = getattr(prior, name)
        original = attention._reset_parameters
        def capture(original=original, name=name):
            events.append(name)
            return original()
        monkeypatch.setattr(attention, "_reset_parameters", capture)
    original_normal = torch.nn.init.normal_
    names = {id(getattr(prior, name)): name for name in ("anchor_position", "effect_position", "bos_effect")}
    def normal(tensor, *args, **kwargs):
        if id(tensor) in names:
            events.append(names[id(tensor)])
        return original_normal(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.nn.init, "normal_", normal)
    prior.reset_parameters()
    assert events == ["progress_attention", "effect_attention", "anchor_position", "effect_position", "bos_effect"]


def test_main_fp32_parameters_and_eval_survive_dtype_moves_while_training_uses_amp():
    model, sample = main_model().eval(), _sample()
    reader, adapter = model.zeva_prompt_encoder, model.zeva_behavior_prefix_adapter
    with torch.no_grad():
        adapter.prefix_project.weight.fill_(1.000123)
        adapter.pim_gate.fill_(0.2)
        adapter.action_prior_adapter.weight.normal_(std=0.1)
    original = {name: value.clone() for name, value in adapter.state_dict().items()}
    model.bfloat16()
    model.torch_dtype = torch.bfloat16
    assert all(parameter.dtype == torch.float32 for parameter in (*reader.parameters(), *adapter.parameters()))
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.mot.parameters())
    for name, value in adapter.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    def compute():
        prompt = _prompt(reader, sample)
        mean, std = _prior(adapter, sample, prompt)
        residual = adapter.action_prior_residual(mean, training=False)
        tokens, _ = adapter.behavior_prefix_tokens(sample["task_context"], prompt, sample["pim_mask"])
        return prompt, mean, std, residual, tokens
    expected = compute()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = compute()
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.dtype == torch.float32
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    model.configure_zeva_trainable_state()
    assert not adapter.prior.training and reader.training
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prompt = _prompt(reader, sample)
        mean, _ = _prior(adapter, sample, prompt)
    assert prompt.dtype == mean.dtype == torch.bfloat16


def test_main_nll_retains_original_coordinate_mean_and_valid_action_mask(monkeypatch):
    from fastwam.models.wan22 import fastwam as implementation
    model, sample = main_model("policy_injection"), _sample()
    model.bfloat16()
    model.torch_dtype = torch.bfloat16
    model.prepare_zeva_training_precision()
    sample["action"][:, :8] = 0.123456
    sample["action"][:, 8:] = 1.012345
    sample["action_valid"][:, 8:] = False
    captured = []
    def capture(target, mean, std, valid=None, **kwargs):
        captured.append((target.detach(), mean.detach(), std.detach(), valid, kwargs))
        return gaussian_prior_nll(target, mean, std, valid, **kwargs)
    monkeypatch.setattr(implementation, "gaussian_prior_nll", capture)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, metrics = model(sample)
    loss.backward()
    target, mean, std, valid, options = captured[0]
    torch.testing.assert_close(target, sample["action"], rtol=0, atol=0)
    assert target.dtype == torch.float32 and options["reference_reduction"] is False
    assert torch.equal(valid, sample["action_valid"])
    expected = (0.5 * (((target - mean) / std).square() + 2 * std.log()).mean(-1))[valid].mean()
    assert metrics["behavior_prior_nll"] == pytest.approx(expected.item(), abs=1e-6)
    assert model.zeva_behavior_prefix_adapter.prior.distribution_head[-1].weight.grad.norm() > 0


@pytest.mark.parametrize("stage", ["policy_injection", "pim_adapter"])
@pytest.mark.parametrize("joint", [False, True])
def test_main_training_scope_and_bundle_roundtrip(tmp_path, stage, joint):
    metadata = main_metadata(tmp_path)
    model = main_model(stage, joint=joint, metadata=metadata)
    model.prepare_zeva_training_precision()
    adapter = model.zeva_behavior_prefix_adapter
    assert all(parameter.requires_grad == joint for parameter in model.zeva_foundation_parameters())
    assert all(parameter.requires_grad == (stage == "policy_injection") for parameter in adapter.policy_injection_parameters())
    assert all(parameter.requires_grad == (stage == "pim_adapter") for parameter in adapter.pim_adapter_parameters())
    model.zeva_parameter_report()
    sample = _sample()
    with torch.no_grad():
        adapter.pim_gate.fill_(0.2)
    loss, _ = model(sample)
    loss.backward()
    if joint:
        assert model.video_expert.patch_embedding.weight.grad is not None
        assert model.action_expert.head.weight.grad.norm() > 0
    path = tmp_path / "policy_addon.pt"
    model.save_zeva_addon_checkpoint(path, step=7, base_checkpoint_sha256="base", cte_checkpoint_sha256="cte")
    payload = torch.load(path, weights_only=False)
    validate_main_artifact(payload, expected=metadata)
    assert payload["episode_splits"] == metadata["episode_splits"] and payload["data_role"] == "train"
    assert (resolve_joint_foundation_checkpoint(path) is not None) == joint
    restored = main_model(stage, joint=joint, metadata=metadata)
    restored.load_zeva_addon_checkpoint(path, base_checkpoint_sha256="base", cte_checkpoint_sha256="cte")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_main_joint_parent_is_shipped_by_inherited_frozen_pim_child(tmp_path):
    metadata = main_metadata(tmp_path)
    parent = main_model("policy_injection", joint=True, metadata=metadata).bfloat16()
    parent.torch_dtype = torch.bfloat16
    parent.prepare_zeva_training_precision()
    with torch.no_grad():
        parent.video_expert.patch_embedding.weight.add_(1e-6)
        parent.action_expert.head.weight.add_(0.123456)
    parent_path = tmp_path / "parent_addon.pt"
    parent.save_zeva_addon_checkpoint(parent_path, step=9)
    parent_foundation = resolve_joint_foundation_checkpoint(parent_path)
    child = main_model(metadata=metadata).bfloat16()
    child.torch_dtype = torch.bfloat16
    child.prepare_zeva_training_precision()
    original_reader = deepcopy(child.zeva_prompt_encoder.state_dict())
    child.load_zeva_addon_checkpoint(parent_path, load_scope="policy_injection")
    assert child.zeva_foundation_mode == "frozen" and child.zeva_requires_foundation_checkpoint
    assert child.video_expert.patch_embedding.weight.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in child.zeva_foundation_parameters())
    for name, value in parent.mot.state_dict().items():
        torch.testing.assert_close(child.mot.state_dict()[name], value, rtol=0, atol=0)
    for name, value in original_reader.items():
        torch.testing.assert_close(child.zeva_prompt_encoder.state_dict()[name], value, rtol=0, atol=0)
    output = tmp_path / "child_addon.pt"
    child.save_zeva_addon_checkpoint(output, step=15)
    payload = torch.load(output, weights_only=False)
    assert payload["foundation_parent_identity"] == {"addon_sha256": checkpoint_sha256(parent_path),
        "foundation_sha256": checkpoint_sha256(parent_foundation), "training_stage": "policy_injection",
        "training_recipe": "main_joint_parent"}
    parent_path.unlink()
    parent_foundation.unlink()
    restored = main_model(metadata=metadata).bfloat16()
    restored.torch_dtype = torch.bfloat16
    restored.prepare_zeva_training_precision()
    restored.load_zeva_addon_checkpoint(output)
    for name, value in child.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["missing_metadata", "other_success_data", "other_profile", "wrong_effect", "wrong_recipe"])
def test_main_loading_rejects_incompatible_profile_or_artifact_contract(tmp_path, problem):
    metadata = main_metadata(tmp_path)
    model = main_model(metadata=metadata)
    path = tmp_path / "addon.pt"
    model.save_zeva_addon_checkpoint(path)
    payload = torch.load(path, weights_only=False)
    if problem == "missing_metadata":
        del model.zeva_main_artifact_metadata
    elif problem == "other_success_data":
        model.zeva_main_artifact_metadata = main_metadata(tmp_path / "other")
    elif problem == "other_profile":
        payload["profile"] = "zeva_ego"
    elif problem == "wrong_effect":
        payload["effect_source"] = "current_prediction"
    else:
        payload["training_recipe"] = "ego_adapter_warmup"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        model.load_zeva_addon_checkpoint(path)


def test_main_and_ego_addons_cannot_cross_load_even_with_identical_external_ids(tmp_path):
    from test_reference_numerics import reference_model
    metadata = main_metadata(tmp_path)
    main, ego = main_model("policy_injection", metadata=metadata), reference_model("policy_injection")
    main_path, ego_path = tmp_path / "main.pt", tmp_path / "ego.pt"
    main.save_zeva_addon_checkpoint(main_path)
    ego.save_zeva_addon_checkpoint(ego_path)
    with pytest.raises(ValueError, match="different policy profile"):
        ego.load_zeva_addon_checkpoint(main_path)
    with pytest.raises(ValueError, match="main artifact|PIM memory contract"):
        main.load_zeva_addon_checkpoint(ego_path)


@pytest.mark.parametrize("problem", ["profile", "success_data", "data_role", "missing_split", "missing_weight"])
def test_main_sidecar_is_bound_to_complete_addon_metadata_before_backbone_mutation(tmp_path, problem):
    metadata = main_metadata(tmp_path)
    model = main_model("policy_injection", joint=True, metadata=metadata)
    path = tmp_path / "addon.pt"
    model.save_zeva_addon_checkpoint(path)
    payload = torch.load(path, weights_only=False)
    sidecar = resolve_joint_foundation_checkpoint(path, payload)
    foundation = torch.load(sidecar, weights_only=False)
    if problem == "profile":
        foundation["profile"] = "zeva_ego"
    elif problem == "success_data":
        foundation.update(main_metadata(tmp_path / "other"))
    elif problem == "data_role":
        foundation["data_role"] = "all"
    elif problem == "missing_split":
        del foundation["episode_splits"]
    else:
        del foundation["mot"][next(iter(foundation["mot"]))]
    torch.save(foundation, sidecar)
    payload["foundation_checkpoint"]["sha256"] = checkpoint_sha256(sidecar)
    torch.save(payload, path)
    target = main_model("policy_injection", joint=True, metadata=metadata)
    target.zeva_freeze_readout_on_load = True
    before = deepcopy(target.mot.state_dict())
    with pytest.raises(ValueError):
        target.load_zeva_addon_checkpoint(path)
    assert not hasattr(target, "_main_frozen_readout_snapshot")
    for name, value in before.items():
        torch.testing.assert_close(target.mot.state_dict()[name], value, rtol=0, atol=0)
