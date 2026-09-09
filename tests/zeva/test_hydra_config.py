from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def test_zeva_overrides_live_under_model_namespace():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path("configs").resolve())
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=robotwin_zeva_fastwam_3cam_384",
                "model.zeva.cte.checkpoint=/tmp/cte.pt",
                "model.zeva.cache.path=/tmp/cache",
                "model.zeva.task_context.mode=bank",
                "model.zeva.task_context.bank_path=/tmp/task-context.pt",
                "model.zeva.task_context.temperature=0.1",
                "model.zeva.prompt.global_dim=64",
            ],
        )
    assert cfg.model.zeva.enabled is True
    assert cfg.model.zeva.cte.checkpoint == "/tmp/cte.pt"
    assert cfg.model.zeva.cache.path == "/tmp/cache"
    assert cfg.model.zeva.task_context.mode == "bank"
    assert cfg.model.zeva.task_context.bank_path == "/tmp/task-context.pt"
    assert cfg.model.zeva.task_context.temperature == 0.1
    assert cfg.model.zeva.task_context.value_dim == 64


def test_zeva_fastwam_defaults_to_two_branch_policy_injection():
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        cfg = compose(
            config_name="train",
            overrides=["task=robotwin_zeva_fastwam_3cam_384"],
        )
    assert cfg.model.zeva.adapter.mode == "exact_zeva"
    assert cfg.model.zeva.adapter.leading_condition_steps == 0


def test_static_task_context_preset_matches_fastwam_and_cte_spaces():
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        cfg = compose(config_name="train", overrides=[
            "task=robotwin_zeva_fastwam_static_3cam_384", "ckpt=/tmp/base.pt",
        ])
    assert cfg.ckpt == "/tmp/base.pt"
    tc = cfg.model.zeva.task_context
    assert tc.mode == "static" and tc.top_k == 5
    assert tc.key_dim == tc.retrieval.output_dim == cfg.model.zeva.cte.retrieval_dim
    assert tc.value_dim == cfg.model.zeva.cte.hidden_dim == cfg.model.zeva.prompt.global_dim
    assert tc.retrieval.input_dim == cfg.model.video_dit_config.hidden_dim
