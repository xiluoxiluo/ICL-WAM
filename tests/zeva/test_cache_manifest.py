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


def test_cache_v4_requires_full_prefix_raw_step_metadata(tmp_path):
    root = tmp_path / "cache-v4"
    manifest = CacheManifest(
        schema_version="zeva_fastwam_robotwin_cache_v4",
        history_semantics="full_episode_prefix",
        query_step_unit="raw_action_step",
    )
    records = [
        {
            "record_type": "phase_query",
            "episode_id": "e",
            "task_id": 1,
            "window_index": 0,
            "episode_step": 0,
            "raw_step": 0,
            "start_raw_step": 0,
            "end_raw_step": 0,
            "transition_index": 0,
            "effect_index": 0,
            "phase_pre": torch.ones(128),
            "phase_post": torch.ones(128),
            "effect": torch.zeros(128),
            "valid": True,
        },
        {
            "record_type": "effect",
            "episode_id": "e",
            "task_id": 1,
            "window_index": None,
            "episode_step": 0,
            "raw_step": 16,
            "start_raw_step": 0,
            "end_raw_step": 16,
            "transition_index": 0,
            "effect_index": 0,
            "phase_pre": torch.ones(128),
            "phase_post": torch.ones(128),
            "effect": torch.ones(128),
            "valid": True,
        },
    ]
    save_phase_effect_cache(root, records, manifest)
    loaded = PhaseEffectCache.load(root, expected={"schema_version": "zeva_fastwam_robotwin_cache_v4"})
    assert loaded.get(0)["record_type"] == "phase_query"
    assert loaded.get(1)["end_raw_step"] == 16


def test_cache_v4_rejects_mismatched_temporal_role(tmp_path):
    manifest = CacheManifest(
        schema_version="zeva_fastwam_robotwin_cache_v4",
        history_semantics="full_episode_prefix",
        query_step_unit="raw_action_step",
    )
    record = {
        "record_type": "phase_query",
        "episode_id": "e",
        "task_id": 1,
        "window_index": 0,
        "episode_step": 0,
        "raw_step": 4,
        "start_raw_step": 0,
        "end_raw_step": 0,
        "transition_index": 0,
        "effect_index": 0,
        "phase_pre": torch.ones(128),
        "phase_post": torch.ones(128),
        "effect": torch.zeros(128),
        "valid": True,
    }
    try:
        save_phase_effect_cache(tmp_path / "invalid", [record], manifest)
    except ValueError:
        pass
    else:
        raise AssertionError("v4 query accepted mismatched raw-step metadata")
