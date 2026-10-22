"""Check runnable presets encode the reference and explicit legacy contracts."""

from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest

from fastwam.zeva_memory_contract import pim_memory_contract


def _config(task, *overrides):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        return compose(config_name="train", overrides=[f"task={task}", *overrides])


@pytest.mark.parametrize("mode,recipe", [("joint", "ego_joint_parent"), ("frozen", "frozen_parent")])
def test_stage2a_reference_schedule_and_effective_batch(mode, recipe):
    for task, world_size in (
        ("robotwin_zeva_ego_policy_3cam_384", 4),
        ("robotwin_zeva_ego_policy_16gpu_3cam_384", 16),
    ):
        cfg = _config(task, f"foundation_mode={mode}")
        assert cfg.model.zeva.training_recipe == recipe
        assert cfg.model.zeva.freeze_fastwam == (mode == "frozen")
        assert cfg.max_steps == 5000
        assert cfg.learning_rate == 5e-5
        assert cfg.zeva_foundation_learning_rate == 5e-6
        assert cfg.lr_scheduler_type == "constant"
        assert cfg.zeva_stage2a_schedule == "zeva_ego"
        assert cfg.batch_size * cfg.gradient_accumulation_steps * world_size == 256


def test_predictive_defaults_and_two_explicit_legacy_families():
    current = _config("robotwin_zeva_ego_pim_3cam_384")
    language_legacy = _config("robotwin_zeva_ego_pim_language_legacy_3cam_384")
    static_legacy = _config("robotwin_zeva_ego_pim_frozen_3cam_384")
    assert current.model.zeva.task_context.mode == language_legacy.model.zeva.task_context.mode == "task_language"
    assert static_legacy.model.zeva.task_context.mode == "static"
    assert current.model.zeva.task_context.train_query_exclusion == "none"
    assert current.model.zeva.prompt.initialization == "zeva_ego"
    assert current.model.zeva.adapter.initialization == "zeva_ego"
    assert current.model.zeva.prepared_stage2.history_pairing == "reference_nearest"
    assert not current.model.zeva.prepared_stage2.allow_reduced_curriculum
    contract = pim_memory_contract(current.model.zeva.memory)
    assert contract["compression"] == "similarity_merge"
    assert contract["uniform_indexing"] == "reference_round"
    assert contract["merge_threshold"] == 0.95
    assert contract["phase_weight"] == 0.5
    for old in (language_legacy, static_legacy):
        assert old.model.zeva.task_context.train_query_exclusion == "leave_one_episode_out"
        assert old.model.zeva.prompt.initialization == "legacy_xavier"
        assert old.model.zeva.adapter.initialization == "legacy_xavier"
        assert old.model.zeva.prepared_stage2.history_pairing == "legacy_candidates"
        assert old.model.zeva.prepared_stage2.allow_reduced_curriculum
        old_contract = pim_memory_contract(old.model.zeva.memory)
        assert old_contract["compression"] == "uniform"
        assert old_contract["uniform_indexing"] == "legacy_floor"
