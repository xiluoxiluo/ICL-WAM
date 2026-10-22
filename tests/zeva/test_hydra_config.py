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
    assert cfg.model.zeva.training_stage == "policy_injection"
    assert cfg.model.zeva.task_context.mode == "static"
    assert cfg.model.zeva.memory.pim_max_entries == 64
    assert cfg.model.zeva.adapter.prior_dropout_rate == 0.4
    assert cfg.model.zeva.adapter.leading_condition_steps == 0
    assert cfg.model.zeva.adapter.prefix_layout == "legacy_merged"
    assert cfg.model.zeva.policy_warmstart_checkpoint is None


def test_stage2_presets_select_disjoint_training_stages():
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        policy = compose(
            config_name="train",
            overrides=["task=robotwin_zeva_fastwam_policy_3cam_384"],
        )
        pim = compose(
            config_name="train",
            overrides=["task=robotwin_zeva_fastwam_pim_3cam_384"],
        )
    assert policy.model.zeva.training_stage == "policy_injection"
    assert pim.model.zeva.training_stage == "pim_adapter"


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


def test_ego_policy_preset_uses_new_prefix_with_original_stage2a_cache_contract():
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        cfg = compose(config_name="train", overrides=[
            "task=robotwin_zeva_fastwam_ego_policy_3cam_384",
            "ckpt=/tmp/base.pt",
            "model.zeva.policy_warmstart_checkpoint=/tmp/old-stage2a.pt",
            "model.zeva.prepared_stage2.pim_cache_path=/tmp/original-stage2a-cache",
        ])
    z = cfg.model.zeva
    assert cfg.ckpt == "/tmp/base.pt"
    assert z.training_stage == "policy_injection"
    assert z.adapter.prefix_layout == "zeva_ego"
    assert z.adapter.effect_source == "latest_completed"
    assert z.policy_warmstart_checkpoint == "/tmp/old-stage2a.pt"
    assert z.policy_checkpoint is None
    assert z.prompt.get("pim_variant", "legacy") == "legacy"
    assert z.adapter.get("pim_variant", "legacy") == "legacy"
    assert z.memory.pim_retrieval_mode == "phase"
    assert z.prepared_stage2.pim_cache_path == "/tmp/original-stage2a-cache"
    assert z.task_context.mode == "static"
    assert z.freeze_fastwam and z.freeze_cte_stage2 and cfg.data.train.use_text_embed_cache


def test_cross_attempt_new_prefix_default_and_explicit_legacy_reproduction():
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        new = compose(config_name="train", overrides=[
            "task=robotwin_zeva_fastwam_cross_attempt_3cam_384",
        ])
        old = compose(config_name="train", overrides=[
            "task=robotwin_zeva_fastwam_cross_attempt_3cam_384",
            "model.zeva.adapter.prefix_layout=legacy_merged",
        ])
    assert new.model.zeva.adapter.prefix_layout == "zeva_ego"
    assert new.model.zeva.adapter.effect_source == "latest_completed"
    assert old.model.zeva.adapter.prefix_layout == "legacy_merged"
    for cfg in (new, old):
        z = cfg.model.zeva
        assert z.training_stage == "pim_adapter"
        assert z.prompt.pim_variant == z.adapter.pim_variant == "cross_attempt"
        assert z.memory.pim_retrieval_mode == "cross_attempt"
        assert z.policy_warmstart_checkpoint is None
