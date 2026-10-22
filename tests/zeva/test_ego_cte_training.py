"""Training recipe, held-out gate and exact-resume provenance checks."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from fastwam.zeva.ego_cte import EGO_CTE_SCHEMA, ZevaEgoCTE, ZevaEgoCTEConfig
from fastwam.zeva.ego_cte_training import (
    PairedEpochSampler, build_ego_cte_optimizer, build_episode_splits,
    capture_rng_state, load_ego_cte_checkpoint, prediction_gain_metrics,
    require_ego_checkpoint_ready, restore_ego_training_checkpoint,
    restore_rng_state, save_ego_cte_checkpoint, validate_ego_cte,
)
from fastwam.zeva.ego_episode_data import EgoEpisode


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return ZevaEgoCTEConfig(hidden_dim=8, phase_dim=8, effect_dim=8, retrieval_dim=4,
                            aux_phase_dim=4, num_layers=1, image_size=8, dropout=0,
                            temporal_backend="gru_test", vision_backbone="tiny_test", vision_pretrained=False)


def episodes():
    return [EgoEpisode(f"task{task}::episode-{index}", str(task), index, index * 49, 49)
            for task in range(2) for index in range(5)]


def metrics(error=0.5):
    return prediction_gain_metrics({"action": error, "action_zero": 1., "vision": error,
                                    "vision_persistence": 1., "effect": error, "effect_zero": 1.}, 8)


def manifest(cfg, promotable=False):
    return {"config": cfg.to_dict(), "promotable": promotable, "tasks": ["task"],
            "episode_splits": {"train": ["train1", "train2"], "validation": ["validation1"]},
            "dataset_stats_sha256": "stats", "semantic_task_map_sha256": "semantic",
            "data_identity": {"episodes": "identity"}, "source_sha256": {"model": "code"},
            "dataset_path": ["/dataset"], "recipe": {"seed": 1000, "batch_size": 8}}


def formal_payload():
    cfg = ZevaEgoCTEConfig()
    identity = manifest(cfg, True)
    value = metrics()
    value["epoch"] = 80
    return {"schema": EGO_CTE_SCHEMA, "config": cfg.to_dict(), "model_config": cfg.to_dict(),
            "manifest": identity, "epoch": 80, "step": 1000, "validation": value,
            "cte_input_type": "rgb_multiview", "memory_semantics": "predicted_effect_trace",
            "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            **{key: identity[key] for key in ("episode_splits", "dataset_stats_sha256", "semantic_task_map_sha256")}}


def test_episode_split_is_disjoint_stable_complete_and_semantically_balanced(tmp_path):
    source = episodes()
    split = build_episode_splits(source)
    assert split == build_episode_splits(reversed(source))
    assert not set(split["train"]) & set(split["validation"])
    assert set(split["train"] + split["validation"]) == {row.episode_id for row in source}
    assert len(split["train"]) == 8 and len(split["validation"]) == 2
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    assert build_episode_splits(source, split_path=path) == split
    bad = deepcopy(split)
    bad["train"].append(bad["validation"][0])
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="overlap"):
        build_episode_splits(source, split_path=path)
    with pytest.raises(ValueError, match=">=3 episodes"):
        build_episode_splits(source[:2])


def test_paired_sampler_covers_each_episode_once_even_with_odd_task_counts():
    labels = ["a"] * 5 + ["b"] * 4 + ["c"] * 3
    sampler = PairedEpochSampler(labels)
    first = list(sampler)
    assert sorted(first) == list(range(len(labels)))
    assert first == list(sampler)
    for start in range(0, 10, 2):
        assert labels[first[start]] == labels[first[start + 1]]
    sampler.set_epoch(1)
    assert list(sampler) != first
    assert sorted(sampler) == list(range(len(labels)))


def test_optimizer_excludes_ema_and_uses_two_reference_learning_rates():
    model = ZevaEgoCTE(tiny_config())
    optimizer = build_ego_cte_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    assert all(group["weight_decay"] == 1e-4 for group in optimizer.param_groups)
    included = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in included}) == len(included)
    assert {id(parameter) for parameter in included} == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert not {id(parameter) for parameter in included} & {id(parameter) for parameter in model.target_vision_encoder.parameters()}


def test_gain_gate_requires_all_three_measured_gains_and_finite_data():
    assert metrics(0.9)["stage1_gate"]
    assert not metrics(0.96)["stage1_gate"]
    assert not metrics(float("nan"))["stage1_gate"]
    errors = {"action": .9, "action_zero": 1., "vision": .96,
              "vision_persistence": 1., "effect": .9, "effect_zero": 1.}
    assert not prediction_gain_metrics(errors, 12)["stage1_gate"]
    assert not prediction_gain_metrics(errors, 0)["stage1_gate"]


def test_validation_selects_valid_chunks_and_restores_model_mode():
    class Predictor(torch.nn.Module):
        def forward(self, images, actions, valid):
            target = torch.ones(*valid.shape, 3)
            return {"pred_act": actions * .5, "target_effect": target,
                    "target_vis": target * 2, "pred_vis": target * 1.5,
                    "pred_effect": target * .5}

    model = Predictor().train()
    actions = torch.ones(2, 2, 24, 14)
    actions[0, 1] = float("nan")
    batch = {"images": torch.zeros(2, 3, 3, 3, 8, 8), "executed_actions": actions,
             "valid": torch.tensor([[True, False], [True, True]])}
    result = validate_ego_cte(model, [batch], "cpu")
    assert result["transitions"] == 3 and result["stage1_gate"]
    assert result["action"] == .25 and result["vision_gain"] == .75
    assert model.training


def test_ready_gate_recomputes_metrics_and_requires_same_epoch_validation():
    payload = formal_payload()
    require_ego_checkpoint_ready(payload)
    payload["validation"]["effect"] = 2.
    payload["validation"]["effect_gain"] = 1.
    payload["validation"]["stage1_gate"] = True
    with pytest.raises(ValueError, match="prediction gate failed"):
        require_ego_checkpoint_ready(payload)
    payload = formal_payload()
    payload["epoch"] = 79
    with pytest.raises(ValueError, match="80 full epochs"):
        require_ego_checkpoint_ready(payload)
    payload = formal_payload()
    payload["validation"]["epoch"] = 75
    with pytest.raises(ValueError, match="this exact epoch"):
        require_ego_checkpoint_ready(payload)
    payload = formal_payload()
    payload["manifest"]["promotable"] = False
    with pytest.raises(ValueError, match="non-promotable"):
        require_ego_checkpoint_ready(payload)
    payload = formal_payload()
    payload["camera_keys"].reverse()
    with pytest.raises(ValueError, match="camera order"):
        require_ego_checkpoint_ready(payload)


def test_test_backend_checkpoint_cannot_be_promoted_even_with_forged_flag(tmp_path):
    model = ZevaEgoCTE(tiny_config())
    optimizer = build_ego_cte_optimizer(model)
    path = tmp_path / "test.pt"
    value = metrics()
    value["epoch"] = 80
    payload = save_ego_cte_checkpoint(path, model, optimizer, epoch=80, step=80,
                                       manifest=manifest(model.cfg, True), validation=value)
    assert not payload["ready_for_stage2"]
    assert payload["cte_input_type"] == "rgb_multiview"
    with pytest.raises(ValueError, match="non-promotable"):
        load_ego_cte_checkpoint(path)
    restored, _ = load_ego_cte_checkpoint(path, require_ready=False)
    assert restored.cfg == model.cfg and not restored.training
    for key in model.state_dict():
        torch.testing.assert_close(restored.state_dict()[key], model.state_dict()[key])


def test_rng_restore_and_resume_rejects_source_or_split_drift(tmp_path):
    generator = torch.Generator().manual_seed(7)
    state = capture_rng_state(generator)
    expected = (torch.rand(3), random.random(), np.random.rand(), torch.rand(3, generator=generator))
    restore_rng_state(state, generator)
    observed = (torch.rand(3), random.random(), np.random.rand(), torch.rand(3, generator=generator))
    torch.testing.assert_close(expected[0], observed[0])
    assert expected[1:3] == observed[1:3]
    torch.testing.assert_close(expected[3], observed[3])
    model = ZevaEgoCTE(tiny_config())
    optimizer = build_ego_cte_optimizer(model)
    identity = manifest(model.cfg, True)
    payload = save_ego_cte_checkpoint(tmp_path / "resume.pt", model, optimizer, epoch=1, step=2,
                                       manifest=identity, validation=None, loader_generator=generator)
    changed = deepcopy(identity)
    changed["source_sha256"]["model"] = "different code"
    with pytest.raises(ValueError, match="source_sha256"):
        restore_ego_training_checkpoint(payload, model, optimizer, changed, generator)
    changed = deepcopy(identity)
    changed["episode_splits"]["train"] = ["different episode"]
    with pytest.raises(ValueError, match="episode_splits"):
        restore_ego_training_checkpoint(payload, model, optimizer, changed, generator)
    assert restore_ego_training_checkpoint(payload, model, optimizer, identity, generator) == (1, 2)


def test_real_epoch_entrypoint_smoke_is_nonpromotable_and_writes_no_checkpoint(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/train_zeva_ego_cte.py"
    spec = importlib.util.spec_from_file_location("ego_cte_train_test", script)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    class Dataset(torch.utils.data.Dataset):
        episodes = tuple(EgoEpisode(f"dataset::episode-{i}", "task", i, i * 49, 49) for i in range(6))

        def __len__(self):
            return len(self.episodes)

        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(index)
            return {"images": torch.rand(3, 3, 3, 8, 8, generator=generator) * 2 - 1,
                    "executed_actions": torch.randn(2, 24, 14, generator=generator),
                    "valid": torch.ones(2, dtype=torch.bool), "action_valid": torch.ones(2, 24, dtype=torch.bool),
                    "raw_steps": torch.tensor([0, 24, 48]), "window_indices": torch.tensor([0, 24, 48]) + index * 49,
                    "policy_valid": torch.tensor([True, False, False]), "decision_valid": torch.tensor([True, True, False]),
                    "episode_id": self.episodes[index].episode_id, "task_id": "task",
                    "episode_index": index, "episode_length": 49, "trailing_action_count": 0}

    monkeypatch.setattr(trainer, "instantiate", lambda _: object())
    monkeypatch.setattr(trainer, "RoboTwinEgoEpisodeDataset", lambda *args, **kwargs: Dataset())
    stats, semantic = tmp_path / "stats.json", tmp_path / "semantic.json"
    stats.write_text("{}")
    semantic.write_text("{}")
    output = tmp_path / "run"
    cfg = OmegaConf.create({"num_epochs": 1, "batch_size": 2, "num_workers": 0,
                           "device": "cpu", "output_dir": str(output), "cte_smoke_steps": 1,
                           "model": {"zeva": {"cte": tiny_config().to_dict()}},
                           "data": {"train": {"val_set_proportion": 0., "require_semantic_task_id": True,
                                               "use_text_embed_cache": False, "dataset_dirs": [str(tmp_path)],
                                               "pretrained_norm_stats": str(stats), "semantic_task_map_path": str(semantic)}}})
    trainer.train(cfg)
    assert json.loads((output / "SMOKE_COMPLETED.json").read_text())["promotable"] is False
    assert json.loads((output / "manifest.json").read_text())["promotable"] is False
    assert not list(output.glob("*.pt"))
