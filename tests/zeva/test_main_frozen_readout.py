"""Joint main deployment reads original base features and restores live weights."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from fastwam.zeva.frozen_readout import FrozenReadoutSnapshot
from test_main_model_contract import main_metadata, main_model


def test_frozen_snapshot_restores_parameters_buffers_and_child_modes_on_exception():
    module = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    module.train()
    module[1].eval()
    frozen = deepcopy(module.state_dict())
    snapshot = FrozenReadoutSnapshot(module)
    parameter_ids = [id(parameter) for parameter in module.parameters()]
    with torch.no_grad():
        for value in module.state_dict(keep_vars=True).values():
            value.add_(3)
    live = deepcopy(module.state_dict())
    modes = [child.training for child in module.modules()]
    with pytest.raises(RuntimeError, match="readout failed"):
        with snapshot.activate(module):
            assert not any(child.training for child in module.modules())
            for name, value in frozen.items():
                torch.testing.assert_close(module.state_dict()[name], value, rtol=0, atol=0)
            raise RuntimeError("readout failed")
    assert [id(parameter) for parameter in module.parameters()] == parameter_ids
    assert [child.training for child in module.modules()] == modes
    assert all(parameter.requires_grad for parameter in module.parameters())
    assert all(value.device.type == "cpu" for value in snapshot.state.values())
    for name, value in live.items():
        torch.testing.assert_close(module.state_dict()[name], value, rtol=0, atol=0)
    for name, value in frozen.items():
        torch.testing.assert_close(snapshot.state[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("inherited", [False, True])
def test_joint_load_captures_base_before_overwrite_and_reads_without_query_drift(tmp_path, monkeypatch, inherited):
    metadata = main_metadata(tmp_path)
    source = main_model("policy_injection", joint=True, metadata=metadata)
    with torch.no_grad():
        source.video_expert.patch_embedding.bias[0].add_(0.34567)
        source.action_expert.head.weight.add_(0.125)
    path = tmp_path / "joint_parent_addon.pt"
    source.save_zeva_addon_checkpoint(path)
    if inherited:
        child = main_model(metadata=metadata)
        child.load_zeva_addon_checkpoint(path, load_scope="policy_injection")
        path = tmp_path / "frozen_child_addon.pt"
        child.save_zeva_addon_checkpoint(path)
        source = child
    stage = "pim_adapter" if inherited else "policy_injection"
    target = main_model(stage, joint=not inherited, metadata=metadata).eval().requires_grad_(False)
    base = deepcopy(target.video_expert.state_dict())
    args = (torch.zeros(1, 3, 16, 16), torch.randn(1, 3, 6), torch.ones(1, 3, dtype=torch.bool))
    expected = target.extract_task_context_readout(*args)
    target.zeva_freeze_readout_on_load = True
    target.load_zeva_addon_checkpoint(path)
    snapshot = target._main_frozen_readout_snapshot
    for name, value in base.items():
        torch.testing.assert_close(snapshot.state[name], value, rtol=0, atol=0)
        torch.testing.assert_close(target.video_expert.state_dict()[name], source.video_expert.state_dict()[name],
                                   rtol=0, atol=0)
    trained = deepcopy(target.video_expert.state_dict())
    actual = target.extract_task_context_readout(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(target._extract_task_context_readout(*args), expected)
    for name, value in trained.items():
        torch.testing.assert_close(target.video_expert.state_dict()[name], value, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in target.parameters())
    target.load_zeva_addon_checkpoint(path)
    assert target._main_frozen_readout_snapshot is snapshot
    def fail(*args):
        for name, value in base.items():
            torch.testing.assert_close(target.video_expert.state_dict()[name], value, rtol=0, atol=0)
        raise RuntimeError("frozen readout failed")
    monkeypatch.setattr(target, "_extract_task_context_readout", fail)
    with pytest.raises(RuntimeError, match="frozen readout failed"):
        target.extract_task_context_readout(*args)
    for name, value in trained.items():
        torch.testing.assert_close(target.video_expert.state_dict()[name], value, rtol=0, atol=0)


def test_base_reload_clears_snapshot_only_after_a_valid_complete_restore(tmp_path):
    metadata = main_metadata(tmp_path)
    base = main_model("policy_injection", joint=True, metadata=metadata)
    base_path = tmp_path / "base.pt"
    base.save_checkpoint(base_path)
    with torch.no_grad():
        base.video_expert.patch_embedding.bias[0].add_(0.25)
    path = tmp_path / "joint_addon.pt"
    base.save_zeva_addon_checkpoint(path)
    target = main_model("policy_injection", joint=True, metadata=metadata)
    target.zeva_freeze_readout_on_load = True
    target.load_zeva_addon_checkpoint(path)
    snapshot = target._main_frozen_readout_snapshot
    current = deepcopy(target.state_dict())
    invalid = tmp_path / "partial.pt"
    torch.save({"mot": {"invalid_key": torch.ones(1)}}, invalid)
    with pytest.raises(ValueError, match="complete mot"):
        target.load_checkpoint(invalid)
    assert target._main_frozen_readout_snapshot is snapshot
    for name, value in current.items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    target.load_checkpoint(base_path)
    assert not hasattr(target, "_main_frozen_readout_snapshot")
    assert target.zeva_foundation_identity is None
    target.load_zeva_addon_checkpoint(path)
    assert target._main_frozen_readout_snapshot is not snapshot
