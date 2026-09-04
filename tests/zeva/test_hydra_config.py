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
            ],
        )
    assert cfg.model.zeva.enabled is True
    assert cfg.model.zeva.cte.checkpoint == "/tmp/cte.pt"
    assert cfg.model.zeva.cache.path == "/tmp/cache"
