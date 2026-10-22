"""Keep real terminal outcomes without treating fixed-H32 storage as evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva.causal_transition_encoder import CausalTransitionEncoder, CausalTransitionEncoderConfig
from fastwam.zeva.cte_latent_cache import (
    CACHE_FORMAT, LATENT_STORAGE_DTYPE, CachedCTELatentWindowDataset, bf16_tensor_to_u16_numpy,
)
from fastwam.zeva.cte_losses import causal_transition_encoder_loss
from fastwam.zeva.cte_training import MainCTEEpisodeDataset
from fastwam.zeva.cte_windows import MAIN_CTE_WINDOW_SEMANTICS, observed_prefix_transitions
from fastwam.zeva.lifecycle import CausalCTEHistory
from fastwam.zeva.main_data import select_main_success_rows, validate_main_episode_ids
from fastwam.zeva.stage1_sampling import build_cte_training_index
from test_main_offline_holdout import _metadata


class _RawEpisodes:
    """Native masks and real post-control image timestamps, no simulator mock."""

    def __init__(self, root, controls):
        self.root, self.controls = Path(root), list(controls)
        lengths = np.asarray(self.controls, dtype=np.int64) + 1
        self.ends = np.cumsum(lengths)
        self.starts = self.ends - lengths
        outer = self

        class TaskTable:
            column_names = ("task_index",)

            def with_format(self, _):
                return self

            def __getitem__(self, indices):
                return {"task_index": [int(np.searchsorted(outer.ends, i, side="right")) // 4 for i in indices]}

        inner = SimpleNamespace(root=self.root, episodes=None, num_frames=len(self),
            meta=SimpleNamespace(total_episodes=len(self.controls)),
            episode_data_index={"from": self.starts, "to": self.ends}, hf_dataset=TaskTable())
        self.lerobot_dataset = SimpleNamespace(multi_dataset=SimpleNamespace(_datasets=[inner]))

    def resolve_semantic_task_id(self, *, episode_index, raw_task_index, strict):
        return f"task-{episode_index // 4}"

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.ends, index, side="right"))
        start, maximum = int(index - self.starts[episode]), self.controls[episode]
        raw_observations = torch.arange(9) * 4 + start
        raw_actions = torch.arange(32) + start
        observed = raw_observations.clamp_max(maximum).float()
        video = (observed[None, :, None, None].expand(3, -1, 16, 16) / 8).clone()
        # Distinct spatial/channel changes survive the fixed target's layer
        # norm, while real four-step boundaries are exact BF16 values.
        video[1] = (observed / 4).square()[:, None, None] / 8
        video[2] = torch.arange(16).float()[None, None, :] / 16
        action = raw_actions.clamp_max(max(0, maximum - 1)).float()[:, None].expand(-1, 14) / 8
        return {"dataset_index": int(index), "episode_id": f"{self.root}::episode-{episode}",
            "task_id": f"task-{episode // 4}", "episode_step": start, "episode_num_steps": maximum + 1,
            "video": video.clone(), "action": action.clone(), "context": torch.zeros(3, 6),
            "context_mask": torch.ones(3, dtype=torch.bool), "prompt": "do it",
            "image_is_pad": raw_observations > maximum, "action_is_pad": raw_actions >= maximum}


def _cte():
    return CausalTransitionEncoder(CausalTransitionEncoderConfig(
        hidden_dim=16, retrieval_dim=8, phase_dim=8, effect_dim=8, num_layers=1, num_heads=4,
        vision_gradient_checkpointing=False,
    ))


@pytest.mark.parametrize("controls", [4, 7, 16, 20, 31, 32, 36, 48, 63, 64, 67])
def test_stage1_keeps_every_observed_four_step_boundary_and_terminal_effect(tmp_path, controls):
    raw = _RawEpisodes(tmp_path, [controls])
    dataset = ZevaRobotWinDataset(raw)
    rows = build_cte_training_index(dataset, include_observed_tail=True)
    assert [r.episode_step for r in rows] == list(range(0, (controls // 4) * 4, 32))
    sample = MainCTEEpisodeDataset(dataset, rows, preencoded=False)[0]
    assert sample["cte_frames"].shape[0] == controls // 4 + 1
    assert sample["transition_actions"].shape[0] == controls // 4
    assert sample["cte_frames"][-1, 0, 0, 0] == (controls // 4) / 2
    model = _cte().eval()
    with torch.no_grad():
        output = model(sample["cte_frames"][None], sample["transition_actions"][None])
    assert output["effect_complete"].tolist() == [[True] * (controls // 16)]


def test_tail_padding_never_enters_cte_history_or_receives_training_gradient(tmp_path):
    dataset = ZevaRobotWinDataset(_RawEpisodes(tmp_path, [48]))
    rows = build_cte_training_index(dataset, include_observed_tail=True)
    sources = {}
    for row in rows:
        sample = dict(dataset[row.dataset_index])
        sample["cte_frames"] = sample["video"].permute(1, 0, 2, 3).clone().requires_grad_()
        sample["transition_actions"] = sample["transition_actions"].clone().requires_grad_()
        sources[row.dataset_index] = sample
    # Explicitly poison the storage padding: it must be trimmed before CTE.
    tail = sources[32]
    with torch.no_grad():
        tail["cte_frames"][5:] = float("nan")
        tail["transition_actions"][4:] = float("nan")
    sample = MainCTEEpisodeDataset(sources, rows, preencoded=True)[0]
    model = _cte()
    output = model(sample["cte_frames"][None], sample["transition_actions"][None])
    losses = causal_transition_encoder_loss(output, sample["transition_actions"][None],
        sample["frame_valid"][None], torch.tensor([0]))
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert output["effect_complete"].tolist() == [[True, True, True]]
    assert torch.equal(tail["cte_frames"].grad[5:], torch.zeros_like(tail["cte_frames"].grad[5:]))
    assert torch.equal(tail["transition_actions"].grad[4:], torch.zeros_like(tail["transition_actions"].grad[4:]))
    assert model.effect_post_head[1].weight.grad.norm() > 0


def _write_latent_cache(path, dataset, rows, metadata):
    rank = path / "rank_000"
    rank.mkdir(parents=True)
    samples = [dataset[row.dataset_index] for row in rows]
    frames = torch.stack([s["video"].permute(1, 0, 2, 3) for s in samples])
    np.save(rank / "latents_bf16_u16.npy", bf16_tensor_to_u16_numpy(frames))
    np.save(rank / "transition_actions.npy", torch.stack([s["transition_actions"] for s in samples]).numpy())
    np.save(rank / "frame_valid.npy", torch.stack([s["frame_valid"] for s in samples]).numpy().astype(np.uint8))
    np.save(rank / "transition_valid.npy", torch.stack([s["transition_valid"] for s in samples]).numpy().astype(np.uint8))
    index = [{"dataset_index": row.dataset_index, "episode_id": row.episode_id, "episode_step": row.episode_step,
              "task_id": row.task_id, "rank": 0, "offset": i} for i, row in enumerate(rows)]
    manifest = {**metadata, "cte_window_semantics": MAIN_CTE_WINDOW_SEMANTICS,
        "format": CACHE_FORMAT, "latent_storage_dtype": LATENT_STORAGE_DTYPE, "num_windows": len(rows),
        "latent_shape": list(frames.shape[1:]), "latent_channels": 3, "cte_vae_input_size": [16, 16],
        "transition_steps": 4, "action_dim": 14}
    (path / "index.json").write_text(json.dumps(index))
    (path / "manifest.json").write_text(json.dumps(manifest))
    return CachedCTELatentWindowDataset(path)


def test_raw_and_latent_cache_keep_short_episodes_splits_and_match_online_history(tmp_path):
    from scripts.build_zeva_robotwin_cache import _build_episode_plans_metadata_only, _decode_episode, _compute_episode
    from scripts.build_zeva_robotwin_cache_from_latents import (
        _expected_effect_count, _materialize_episode, _tail_required, _validate_plan_alignment,
    )
    from scripts.validate_zeva_cte_readiness import _reconstruct_episode

    _, original_metadata = _metadata(tmp_path)
    root = tmp_path / "source"
    controls = [48, 20, 4, 67, 32, 17, 7, 15]
    raw = _RawEpisodes(root, controls)
    dataset = ZevaRobotWinDataset(raw)
    rows = build_cte_training_index(dataset, include_observed_tail=True)
    cfg = OmegaConf.create({"model": {"zeva": {"profile": "main_observed", "offline": {
        "success_manifest_path": str(root / "success.json")}}}, "data": {"train": {
            "dataset_dirs": [str(root)], "pretrained_norm_stats": str(root / "stats.json"),
            "semantic_task_map_path": str(root / "meta/tasks.jsonl"), "is_training_set": True}}})
    rows, metadata = select_main_success_rows(rows, cfg)
    assert metadata["success_data_identity"] == original_metadata["success_data_identity"]
    cache = _write_latent_cache(tmp_path / "latents", dataset, rows, metadata)
    cache.validate_storage_layout()
    with pytest.raises(ValueError, match="invalid/padded"):
        cache.validate_storage_layout(require_complete_windows=True)
    plans = _build_episode_plans_metadata_only(dataset, sample_stride=1, transition_steps=4,
        show_progress=False, include_observed_tail=True)
    _validate_plan_alignment(plans, cache)
    validate_main_episode_ids(metadata, [plan.episode_id for plan in plans])
    assert len(plans) == 8
    model = _cte().eval().requires_grad_(False)
    model.zeva_main_features = True
    for plan, count in zip(plans, controls, strict=True):
        assert plan.history_end_raw_step == count // 4 * 4
        assert plan.max_query_start == max(0, ((count - 32) // 4) * 4)
        assert not _tail_required(plan, cache)
        decoded = _materialize_episode(plan, latent_cache=cache, tail=None)
        direct = _decode_episode(plan, dataset=dataset, transition_steps=4, sample_stride=1, cte_input_type="rgb_frame")
        assert decoded.segments == direct.segments == [list(range(0, count // 4 * 4 + 1, 4))]
        diagnostic_steps, diagnostic_frames, _ = _reconstruct_episode(cache, plan.episode_id)
        assert diagnostic_steps == decoded.segments[0]
        assert diagnostic_frames.shape[0] == count // 4 + 1
        for step in decoded.boundary_frames:
            torch.testing.assert_close(decoded.boundary_frames[step], direct.boundary_frames[step], rtol=0, atol=0)
        with torch.no_grad():
            records, _ = _compute_episode(decoded, model=model, cte_device=torch.device("cpu"),
                cte_input_type="rgb_frame", frame_encoder=None, profile_gpu_timing=False, pin_memory=False)
        effects = [r for r in records if r["record_type"] == "effect"]
        queries = [r for r in records if r["record_type"] == "phase_query"]
        assert len(effects) == _expected_effect_count(plan, 4) == count // 16
        assert len(queries) == plan.query_count and queries[0]["raw_step"] == 0
        history = CausalCTEHistory(model)
        history.reset(decoded.boundary_frames[0])
        for step in decoded.segments[0][:-1]:
            history.append_transition(decoded.action_groups[step], decoded.boundary_frames[step + 4])
        online = history.forward()
        for i, effect in enumerate(effects):
            torch.testing.assert_close(effect["effect"], online["effect_post"][0, i], rtol=0, atol=0)
            torch.testing.assert_close(effect["phase_pre"], online["phase"][0, i * 4], rtol=0, atol=0)

    manifest_path = cache.root / "manifest.json"
    old = json.loads(manifest_path.read_text())
    old.pop("cte_window_semantics")
    manifest_path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="rebuild"):
        CachedCTELatentWindowDataset(cache.root)


def test_main_feature_eval_uses_cpu_fp32_even_inside_amp_and_does_not_cast_weights():
    model = _cte().eval()
    model.zeva_main_features = True
    frames, actions = torch.randn(1, 5, 3, 16, 16), torch.randn(1, 4, 4, 14)
    with torch.no_grad():
        expected = model(frames, actions)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model(frames, actions)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_observed_prefix_masks_refuse_observation_holes():
    valid = torch.ones(9, dtype=torch.bool)
    valid[3] = False
    with pytest.raises(ValueError, match="contiguous"):
        observed_prefix_transitions(valid, torch.ones(8, 4, dtype=torch.bool))


def test_main_cte_checkpoint_binds_tail_semantics_and_restores_feature_precision(tmp_path):
    from fastwam.zeva.checkpoint import load_cte_checkpoint, save_cte_checkpoint

    _, metadata = _metadata(tmp_path)
    metadata = {**metadata, "data_role": "train", "cte_window_semantics": MAIN_CTE_WINDOW_SEMANTICS}
    path = tmp_path / "cte.pt"
    save_cte_checkpoint(path, _cte(), config=metadata)
    restored = _cte()
    payload = load_cte_checkpoint(path, restored)
    assert restored.zeva_main_features is True
    assert payload["cte_window_semantics"] == MAIN_CTE_WINDOW_SEMANTICS
    payload.pop("cte_window_semantics")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="rebuild"):
        load_cte_checkpoint(path, _cte())


def test_short_native_target_masks_survive_prepared_join_and_actual_flow_nll_loss(tmp_path, monkeypatch):
    from fastwam.models.wan22 import fastwam as implementation
    from fastwam.zeva.main_prepared import split_main_prepared_dataset
    from fastwam.zeva.prepared_stage2 import (
        PIM_CACHE_FORMAT, STATIC_CONTEXT_FORMAT, PreparedPIMCache, PreparedZevaStage2Dataset,
        StaticEpisodeContextCache,
    )
    from test_main_model_contract import main_model

    _, metadata = _metadata(tmp_path)
    raw = _RawEpisodes(tmp_path / "source", [20, 4, 16, 24, 20, 4, 16, 24])
    cache_root = tmp_path / "prepared"
    shard = cache_root / "shard_000"
    shard.mkdir(parents=True)
    count = len(raw.controls)
    episode_ids = [f"{raw.root}::episode-{i}" for i in range(count)]
    arrays = {
        "window_index": raw.starts.astype(np.int64), "raw_step": np.zeros(count, dtype=np.int32),
        "episode_index": np.arange(count, dtype=np.int32), "task_index": np.arange(count, dtype=np.int16) // 4,
        "phase": np.ones((count, 4), dtype=np.float32),
        "bit_effects": np.zeros((count, 2, 4), dtype=np.float32), "bit_mask": np.zeros((count, 2), dtype=np.uint8),
        "pim_phases": np.zeros((count, 3, 4), dtype=np.float32),
        "pim_effects": np.zeros((count, 3, 4), dtype=np.float32),
        "pim_mask": np.zeros((count, 3), dtype=np.uint8), "pim_scores": np.zeros((count, 3), dtype=np.float32),
    }
    for key, value in arrays.items():
        np.save(shard / f"{key}.npy", value)
    (cache_root / "episode_ids.json").write_text(json.dumps(episode_ids))
    (cache_root / "task_ids.json").write_text(json.dumps(["task-0", "task-1"]))
    (cache_root / "manifest.json").write_text(json.dumps({**metadata, "format": PIM_CACHE_FORMAT,
        "phase_dim": 4, "effect_dim": 4, "top_k": 3, "bit_size": 2, "num_queries": count,
        "num_episodes": count, "semantic_task_count": 2, "retrieval_mode": "phase",
        "shards": [{"path": "shard_000", "global_start": 0, "global_end": count}]}))
    static_path = tmp_path / "static.pt"
    torch.save({"format": STATIC_CONTEXT_FORMAT, "episode_ids": episode_ids,
                "task_contexts": torch.ones(count, 8), "metadata": metadata}, static_path)
    dataset = PreparedZevaStage2Dataset(raw, PreparedPIMCache(cache_root), StaticEpisodeContextCache(static_path))
    train, validation = split_main_prepared_dataset(dataset)
    assert len(train) + len(validation) == count
    assert {train[i]["episode_id"] for i in range(len(train))} == set(metadata["episode_splits"]["train"])
    assert {validation[i]["episode_id"] for i in range(len(validation))} == set(metadata["episode_splits"]["validation"])
    item = dataset[0]
    assert item["action_valid"].tolist() == [True] * 20 + [False] * 12
    sample = {key: value[None] for key, value in item.items() if isinstance(value, torch.Tensor)}
    sample["_training_mode"] = "zeva_stage2"
    model = main_model("policy_injection")
    predictions, nll_checks = [], []
    original_denoise = model._denoise_action_with_video_cache_zeva
    original_nll = implementation.gaussian_prior_nll

    def denoise(*args, **kwargs):
        result = original_denoise(*args, **kwargs)
        result.retain_grad()
        predictions.append(result)
        return result

    def nll(target, mean, std, valid=None, **kwargs):
        result = original_nll(target, mean, std, valid, **kwargs)
        torch.testing.assert_close(valid, sample["action_valid"], rtol=0, atol=0)
        # Isolate direct NLL supervision; EAP also conditions the action
        # transformer, whose valid tokens legitimately have other gradients.
        gradients = torch.autograd.grad(result, (mean, std), retain_graph=True)
        for gradient in gradients:
            assert torch.equal(gradient[~valid], torch.zeros_like(gradient[~valid]))
        nll_checks.append(True)
        return result

    monkeypatch.setattr(model, "_denoise_action_with_video_cache_zeva", denoise)
    monkeypatch.setattr(implementation, "gaussian_prior_nll", nll)
    loss, metrics = model(sample)
    assert torch.isfinite(loss) and "behavior_prior_nll" in metrics
    loss.backward()
    assert nll_checks == [True] and len(predictions) == 1
    valid = sample["action_valid"]
    assert predictions[0].grad[valid].abs().sum() > 0
    assert torch.equal(predictions[0].grad[~valid], torch.zeros_like(predictions[0].grad[~valid]))
