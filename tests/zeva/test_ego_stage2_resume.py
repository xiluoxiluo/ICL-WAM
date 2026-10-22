"""Regression checks for predictive Stage2's resumable checkpoint contract."""
from types import SimpleNamespace

import pytest

from fastwam.trainer import Wan22Trainer


def test_incomplete_predictive_state_never_falls_back_to_generic_accelerate(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.resume = str(tmp_path)
    trainer.predictive_zeva = trainer.zeva_training = True
    trainer.accelerator = SimpleNamespace(process_index=0)
    trainer.load_training_state = lambda _: pytest.fail("incomplete state reached generic loader")
    # A stale COMPLETE alone must not make this a resumable checkpoint.
    (tmp_path / "COMPLETE").write_text("complete\n")
    with pytest.raises(ValueError, match="optimizer_scheduler.pt"):
        trainer._resume_or_load_checkpoint()


def test_interrupted_checkpoint_overwrite_invalidates_old_complete_before_weights(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.global_step, trainer.state_dir, trainer.predictive_zeva = 500, str(tmp_path), True
    trainer.accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=lambda: None)
    state = tmp_path / "step_000500"
    state.mkdir()
    marker = state / "COMPLETE"
    marker.write_text("old completion\n")

    def interrupted_save(**_):
        assert not marker.exists()
        raise OSError("simulated interrupted write")

    trainer._save_weights_checkpoint = interrupted_save
    with pytest.raises(OSError, match="interrupted"):
        trainer.save_checkpoint()
    assert not marker.exists()


@pytest.mark.parametrize("key", ["seed", "batch_size", "gradient_accumulation_steps", "mixed_precision",
                                  "learning_rate", "weight_decay", "max_grad_norm", "max_steps"])
def test_predictive_stage2a_also_rejects_changed_resume_recipe(key):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.predictive_zeva = True
    trainer.zeva_training_recipe = "frozen_parent"
    trainer.cfg = {"seed": 1000, "batch_size": 1, "gradient_accumulation_steps": 1,
                   "mixed_precision": "bf16", "learning_rate": 1e-4, "weight_decay": 1e-4,
                   "max_grad_norm": 1.0, "max_steps": 5000}
    saved = {"training_recipe": "frozen_parent", "config": dict(trainer.cfg)}
    trainer._validate_zeva_resume_recipe(saved)
    saved["config"][key] = "changed"
    with pytest.raises(ValueError, match=key):
        trainer._validate_zeva_resume_recipe(saved)
