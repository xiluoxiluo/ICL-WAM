from copy import deepcopy
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset, ZevaStage2Dataset
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.zeva import CausalTransitionEncoder, CausalTransitionEncoderConfig, TaskContextBank, CacheManifest, PhaseEffectCache, save_phase_effect_cache
from fastwam.zeva.checkpoint import checkpoint_sha256, load_addon_checkpoint
from fastwam.zeva.static_task_context import (
    BEHAVIOR_KEY_SPACE, BEHAVIOR_VALUE_SPACE, HEAD_FORMAT, READOUT_KIND, READOUT_FORMAT,
    StaticTaskContextRetrievalConfig, StaticTaskContextRetrievalHead,
    StaticTaskContextRetriever, StaticTaskContextSession,
    bidirectional_supervised_contrastive_loss, trajectory_prototype,
    load_readout_cache, stage2_task_contexts,
)
from fastwam.zeva.task_context_data import build_behavior_bank, iter_demo_trajectories
from fastwam.zeva.task_context_training import train_retrieval_head, split_retrieval_episodes


def _bank():
    return TaskContextBank([
        {"episode_id": f"e{i}", "task_id": str(i // 3), "instruction": str(i // 3),
         "retrieval_key": [1., 0.] if i < 3 else [0., 1.],
         "behavior_value": [float(i), float(i + 1), float(i + 2)]}
        for i in range(6)
    ], key_dim=2, value_dim=3, metadata={
        "key_space": BEHAVIOR_KEY_SPACE, "value_space": BEHAVIOR_VALUE_SPACE,
        "readout_kind": READOUT_KIND, "split": "train", "readout_dim": 4,
    })


def _head():
    return StaticTaskContextRetrievalHead(StaticTaskContextRetrievalConfig(4, 8, 2, 0.0))


def test_zeva_symmetric_multi_positive_loss_and_head_gradients():
    queries = torch.eye(2).repeat_interleave(2, dim=0).requires_grad_()
    labels = torch.tensor([0, 0, 1, 1])
    loss = bidirectional_supervised_contrastive_loss(queries, queries.detach(), labels, temperature=1.0)
    torch.testing.assert_close(loss, torch.log1p(torch.exp(torch.tensor(-1.0))))
    loss.backward()
    assert queries.grad is not None and bool(torch.isfinite(queries.grad).all())
    head = _head()
    output = head(torch.randn(6, 4))
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(6))
    bidirectional_supervised_contrastive_loss(output, torch.eye(2).repeat(3, 1), torch.arange(6) % 2).backward()
    assert any(parameter.grad is not None and parameter.grad.norm() > 0 for parameter in head.parameters())


def test_trajectory_prototype_uses_cte_states_and_valid_mask():
    output = {
        "retrieval": torch.tensor([[[1., 0.], [0., 1.], [float("nan"), 0.]]]),
        "causal_interaction_state": torch.tensor([[[1., 2., 3.], [3., 4., 5.], [100., 100., 100.]]]),
    }
    key, value = trajectory_prototype(output, torch.tensor([[True, True, False]]))
    torch.testing.assert_close(key, torch.ones(1, 2) / 2**0.5)
    torch.testing.assert_close(value, torch.tensor([[2., 3., 4.]]))
    with pytest.raises(ValueError, match="valid CTE state"):
        trajectory_prototype(output, torch.zeros(1, 3, dtype=torch.bool))


def test_retrieval_excludes_own_demonstration_and_freezes_modules():
    retriever = StaticTaskContextRetriever(_bank(), _head(), top_k=50)
    result = retriever.retrieve(torch.randn(1, 4), exclude_episode_ids=["e0"])
    assert 0 not in result.indices[0].tolist()
    assert result.values.shape == (1, 3)
    assert not result.values.requires_grad
    assert all(not parameter.requires_grad for parameter in retriever.head.parameters())
    with pytest.raises(ValueError, match="key_space"):
        StaticTaskContextRetriever(TaskContextBank(_bank().entries, key_dim=2, value_dim=3), _head())


def test_trained_head_and_cache_are_bound_to_bank(tmp_path):
    bank_path, head_path, cache_path = [tmp_path / name for name in ("bank.pt", "head.pt", "cache.pt")]
    bank = _bank()
    bank.save(bank_path)
    head = _head()
    payload = {
        "format": HEAD_FORMAT, "readout_kind": READOUT_KIND, "step": 1,
        "bank_sha256": checkpoint_sha256(bank_path),
        "config": head.config.as_dict(), "model": head.state_dict(),
    }
    torch.save(payload, head_path)
    cache = {
        "format": READOUT_FORMAT, "readout_kind": READOUT_KIND,
        "bank_sha256": checkpoint_sha256(bank_path), "readouts": torch.randn(6, 4),
        "episode_ids": [f"e{i}" for i in range(6)],
    }
    torch.save(cache, cache_path)
    retriever = StaticTaskContextRetriever.load(bank_path, head_path)
    loaded = load_readout_cache(cache_path, bank_path, retriever.bank)
    contexts = stage2_task_contexts(retriever, loaded)
    assert set(contexts) == set(cache["episode_ids"])
    for index, episode_id in enumerate(cache["episode_ids"]):
        expected = retriever.retrieve(cache["readouts"][index:index + 1], exclude_episode_ids=[episode_id])
        torch.testing.assert_close(contexts[episode_id], expected.values[0])
    payload["step"] = 0
    torch.save(payload, head_path)
    with pytest.raises(ValueError, match="trained"):
        StaticTaskContextRetriever.load(bank_path, head_path)
    payload["step"] = 1
    payload["bank_sha256"] = "wrong"
    torch.save(payload, head_path)
    with pytest.raises(ValueError, match="identity"):
        StaticTaskContextRetriever.load(bank_path, head_path)
    cache["episode_ids"] = list(reversed(cache["episode_ids"]))
    torch.save(cache, cache_path)
    with pytest.raises(ValueError, match="episode order"):
        load_readout_cache(cache_path, bank_path, bank)


def test_session_only_reads_initial_observation_and_resets_each_attempt():
    class Policy:
        def __init__(self):
            self.calls = []

        def extract_task_context_readout(self, image, context, mask):
            self.calls.append(image.clone())
            return image

    policy = Policy()
    session = StaticTaskContextSession(StaticTaskContextRetriever(_bank(), _head(), top_k=2))
    initial = torch.randn(1, 4)
    first = session.resolve(policy, initial, None, None, "task")
    later = session.resolve(policy, torch.randn(1, 4), None, None, "task")
    assert first is later and len(policy.calls) == 1
    torch.testing.assert_close(policy.calls[0], initial)
    with pytest.raises(ValueError, match="instruction changed"):
        session.resolve(policy, initial, None, None, "changed")
    session.reset()
    session.resolve(policy, initial, None, None, "changed")
    assert len(policy.calls) == 2


class _VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv3d(3, 4, 1)

    def encode(self, video, scale):
        assert video.shape[2] == 1
        return self.projection(video)


def _tiny_policy():
    model = FastWAM.__new__(FastWAM)
    nn.Module.__init__(model)
    model.device, model.torch_dtype = torch.device("cpu"), torch.float32
    model.video_expert = WanVideoDiT(
        hidden_dim=16, in_dim=4, ffn_dim=32, out_dim=4, text_dim=6,
        freq_dim=8, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
        attn_head_dim=8, num_layers=2, has_image_input=False,
        seperated_timestep=True, fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
    )
    model.mot = MoT({"video": model.video_expert, "action": deepcopy(model.video_expert)})
    model.vae = SimpleNamespace(model=_VAE().eval(), scale=None)
    model.zeva_prompt_encoder = nn.Linear(6, 16)
    return model.eval()


def test_real_video_mot_readout_is_clean_frozen_and_multimodal():
    torch.manual_seed(7)
    model = _tiny_policy()
    image, text = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 6)
    mask = torch.ones(2, 3, dtype=torch.bool)
    output = model.extract_task_context_readout(image, text, mask)
    assert output.shape == (2, 16) and not output.requires_grad
    changed_image = model.extract_task_context_readout(image + 0.5, text, mask)
    changed_text = model.extract_task_context_readout(image, text + 1.0, mask)
    assert not torch.allclose(output, changed_image)
    assert not torch.allclose(output, changed_text)
    with torch.no_grad():
        model.zeva_prompt_encoder.weight.fill_(999.0)
    torch.testing.assert_close(model.extract_task_context_readout(image, text, mask), output)
    assert all(parameter.grad is None for parameter in model.parameters())


class _DemoWindows:
    def __init__(self):
        self.video = torch.arange(10).float()[None, :, None, None].expand(3, -1, 8, 8) / 10
        self.actions = torch.arange(36).float()[:, None].expand(-1, 14) / 36

    def __len__(self):
        return 2

    def __getitem__(self, index):
        start = index * 4
        return {
            "dataset_index": index, "episode_id": "demo", "episode_step": start,
            "task_id": "task", "prompt": "instruction",
            "video": self.video[:, index:index + 9].clone(),
            "action": self.actions[start:start + 32].clone(),
            "context": torch.ones(3, 6), "context_mask": torch.ones(3, dtype=torch.bool),
        }


def test_demo_join_deduplicates_overlaps_and_bank_contains_behavior():
    torch.manual_seed(11)
    source = _DemoWindows()
    dataset = ZevaRobotWinDataset(source)
    demo = list(iter_demo_trajectories(dataset))[0]
    torch.testing.assert_close(demo["frames"], source.video.permute(1, 0, 2, 3))
    torch.testing.assert_close(demo["actions"].flatten(0, 1), source.actions)
    cte = CausalTransitionEncoder(CausalTransitionEncoderConfig(
        hidden_dim=16, retrieval_dim=8, phase_dim=8, effect_dim=8, num_layers=1, num_heads=4,
    ))
    policy = _tiny_policy()
    bank, readouts = build_behavior_bank(dataset, cte, policy)
    assert bank.key_dim == 8 and bank.value_dim == 16
    assert bank.entries[0]["num_transitions"] == 9
    with torch.no_grad():
        encoded = cte(demo["frames"].unsqueeze(0), demo["actions"].unsqueeze(0))
        expected_key, expected_value = trajectory_prototype(encoded, torch.ones(1, 10, dtype=torch.bool))
    torch.testing.assert_close(bank.entries[0]["behavior_value"], expected_value[0])
    torch.testing.assert_close(bank.entries[0]["retrieval_key"], expected_key[0])
    # A changed demonstration future changes the bank, never its initial query input.
    source.actions = -source.actions
    changed_bank, changed_readouts = build_behavior_bank(dataset, cte, policy)
    torch.testing.assert_close(readouts, changed_readouts)
    assert not torch.allclose(bank.entries[0]["behavior_value"], changed_bank.entries[0]["behavior_value"])


def test_query_head_training_learns_and_validation_episodes_are_disjoint():
    torch.manual_seed(4)
    labels = torch.tensor([0] * 5 + [1] * 5)
    readouts = torch.nn.functional.one_hot(labels, 4).float() + torch.randn(10, 4) * 0.01
    keys = torch.nn.functional.one_hot(labels, 2).float()
    readouts.requires_grad_(True)
    keys.requires_grad_(True)
    head = _head()
    initial = deepcopy(head.state_dict())
    metrics = train_retrieval_head(
        head, readouts, keys, labels, steps=50, batch_size=8,
        learning_rate=0.03, eval_every=25, seed=4,
    )
    assert not set(metrics["train_indices"]) & set(metrics["validation_indices"])
    assert any(not torch.equal(initial[name], value) for name, value in head.state_dict().items())
    assert metrics["history"][-1]["task_top1"] == 1.0
    assert readouts.grad is None and keys.grad is None
    with pytest.raises(ValueError, match="two tasks"):
        split_retrieval_episodes(torch.zeros(5, dtype=torch.long))


def test_stage2_reuses_one_initial_context_for_all_episode_windows(tmp_path):
    root = tmp_path / "cache"
    records = [{
        "episode_id": "demo", "task_id": "task", "window_index": index,
        "episode_step": index * 4, "transition_index": 0, "effect_index": 0,
        "phase_pre": torch.ones(128), "phase_post": torch.ones(128),
        "effect": torch.ones(128), "valid": True,
    } for index in range(2)]
    save_phase_effect_cache(root, records, CacheManifest())
    cache = PhaseEffectCache.load(root)
    context = torch.arange(16).float()
    dataset = ZevaStage2Dataset(_DemoWindows(), cache, task_context_by_episode={"demo": context})
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    torch.testing.assert_close(batch["task_context"], context.repeat(2, 1))
    with pytest.raises(ValueError, match="missing initial"):
        ZevaStage2Dataset(_DemoWindows(), cache, task_context_by_episode={})


def test_addon_rejects_wrong_or_missing_task_context_identity(tmp_path):
    prompt, adapter = nn.Linear(2, 2), nn.Linear(2, 2)
    identity = {"mode": "static", "bank_sha256": "bank", "head_sha256": "head", "top_k": 5}
    path = tmp_path / "addon.pt"
    torch.save({
        "causal_prompt_encoder": prompt.state_dict(), "behavior_prefix_adapter": adapter.state_dict(),
        "task_context_identity": identity,
    }, path)
    load_addon_checkpoint(path, prompt, adapter, task_context_identity=identity)
    for mismatch in (None, {**identity, "top_k": 1}, {**identity, "bank_sha256": "wrong"}):
        with pytest.raises(ValueError, match="task-context"):
            load_addon_checkpoint(path, prompt, adapter, task_context_identity=mismatch)


def test_retrieval_training_cli_produces_a_loadable_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[2]
    bank_path, cache_path, head_path = [tmp_path / name for name in ("bank.pt", "readouts.pt", "head.pt")]
    bank = _bank()
    bank.save(bank_path)
    torch.save({
        "format": READOUT_FORMAT, "readout_kind": READOUT_KIND,
        "bank_sha256": checkpoint_sha256(bank_path),
        "episode_ids": [entry["episode_id"] for entry in bank.entries],
        "readouts": torch.eye(4)[:2].repeat_interleave(3, dim=0),
    }, cache_path)
    result = subprocess.run([
        sys.executable, str(root / "scripts/train_zeva_task_context_retrieval.py"),
        "task=robotwin_zeva_fastwam_static_3cam_384", "device=cpu",
        f"model.zeva.task_context.bank_path={bank_path}",
        f"model.zeva.task_context.readout_cache_path={cache_path}",
        f"model.zeva.task_context.retrieval_checkpoint={head_path}",
        "model.zeva.task_context.key_dim=2", "model.zeva.task_context.retrieval.input_dim=4",
        "model.zeva.task_context.retrieval.hidden_dim=8",
        "model.zeva.task_context.training.steps=2",
    ], cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    retriever = StaticTaskContextRetriever.load(bank_path, head_path)
    assert retriever.retrieve(torch.ones(1, 4)).values.shape == (1, 3)


def test_evaluation_wrapper_selects_static_head_and_default_top_five(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    main = runpy.run_path(str(root / "scripts/eval_zeva_robotwin_fixed_attempts.py"))["main"]
    monkeypatch.setattr(sys, "argv", [
        "eval", "--task", "test", "--seed", "0", "--ckpt", "base.pt",
        "--cte-checkpoint", "cte.pt", "--addon-checkpoint", "addon.pt",
        "--task-context-bank", "bank.pt", "--task-context-retrieval-checkpoint", "head.pt",
    ])
    commands = []
    monkeypatch.setattr(subprocess, "call", lambda cmd, cwd: commands.append(cmd) or 0)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
    assert "model.zeva.task_context.mode=static" in commands[0]
    assert "model.zeva.task_context.top_k=5" in commands[0]
