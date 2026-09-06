import torch

from fastwam.zeva import CacheManifest, PhaseEffectCache, save_phase_effect_cache


def test_cache_round_trip_and_manifest(tmp_path):
    root = tmp_path / "cache"
    manifest = CacheManifest(cte_checkpoint_sha256="a" * 64, dataset_stats_sha256="b" * 64)
    save_phase_effect_cache(root, [{"episode_id": "e", "task_id": 1, "phase_pre": torch.ones(128), "phase_post": torch.zeros(128), "effect": torch.ones(128), "valid": True}], manifest, shard_size=1)
    loaded = PhaseEffectCache.load(root, expected={"action_dim": 14, "action_horizon": 32, "action_normalization": "fastwam_processor_output"})
    assert len(loaded) == 1 and loaded.get(0)["valid"]


def test_cache_manifest_rejects_inconsistent_shape():
    try:
        CacheManifest(action_horizon=31)
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent action/video schema was accepted")


def test_cache_manifest_tracks_latent_contract_separately():
    manifest = CacheManifest(
        image_channels=48,
        latent_channels=48,
        cte_input_type="wan_vae_latent",
        vae_metadata={
            "model_id": "wan",
            "vae_path": "/models/wan.vae",
            "z_dim": 48,
            "temporal_downsample_factor": 4,
            "upsampling_factor": 8,
        },
        cte_vae_input_size=(384, 320),
    )
    assert manifest.image_channels == manifest.latent_channels == 48
    assert manifest.cte_vae_input_size == (384, 320)
