"""Actual tiny Wan text encoding, bounded caching, and isolated device use."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from experiments.robotwin.fastwam_policy.deploy_policy import (
    WorldActionRobotWinPolicy, _cuda_memory_metadata, _inference_model_config,
)
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.wan_video_text_encoder import WanTextEncoder
from fastwam.zeva.attempt_protocol import ModelSeedStream
from fastwam.zeva.prompt_cache import (
    FrozenPolicyTextEncoder, PolicyPromptCache, load_policy_prompt_cache,
)


class _Tokenizer:
    def __init__(self, snapshot=b'{"vocabulary":"test"}'):
        self.calls = []
        self.snapshot = snapshot

    def save_pretrained(self, folder):
        (Path(folder) / "tokenizer.json").write_bytes(self.snapshot)

    def __call__(self, prompt, *, return_mask, add_special_tokens):
        assert return_mask and add_special_tokens
        self.calls.append(prompt)
        # Special tokens + variable content + padding, just like the policy
        # tokenizer contract. Vocabulary is deliberately tiny for real UMT5.
        token = 3 + sum(prompt.encode()) % 20
        return torch.tensor([[1, token, 2, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]])


class _TinyWan(WanTextEncoder):
    def __init__(self):
        super().__init__(vocab=32, dim=8, dim_attn=8, dim_ffn=16, num_heads=2,
                         num_layers=1, num_buckets=8, dropout=0.25)
        self.calls = 0
        self.input_devices = []

    def forward(self, ids, mask=None):
        self.calls += 1
        self.input_devices.append(str(ids.device))
        return super().forward(ids, mask)


def _cache(size=2, dtype=torch.float32):
    encoder, tokenizer = _TinyWan(), _Tokenizer()
    frozen = FrozenPolicyTextEncoder(encoder, tokenizer, device="cpu", dtype=dtype)
    return PolicyPromptCache(frozen.encode, max_entries=size), frozen, tokenizer


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_wan_encoding_and_cached_result_preserve_original_features_and_mask(dtype):
    cache, frozen, tokenizer = _cache(dtype=dtype)
    original = SimpleNamespace(text_encoder=frozen.text_encoder, tokenizer=tokenizer, device="cpu")
    expected, expected_mask = FastWAM.encode_prompt(original, "put the cup on the plate")
    actual, actual_mask = cache.encode("put the cup on the plate", device="cpu", dtype=dtype)
    repeated, repeated_mask = cache.encode("put the cup on the plate", device="cpu", dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    assert torch.equal(actual_mask, expected_mask) and torch.equal(repeated_mask, expected_mask)
    assert actual_mask.all() and not actual[:, 3:].any()
    assert frozen.text_encoder.calls == 2  # old reference + one cache miss
    assert not frozen.text_encoder.training
    assert not any(parameter.requires_grad for parameter in frozen.text_encoder.parameters())
    assert not actual.requires_grad


def test_cache_is_bounded_lru_instruction_specific_and_return_values_are_not_shared():
    cache, frozen, _ = _cache(size=2)
    def encode(prompt):
        return cache.encode(prompt, device="cpu", dtype=torch.float32)
    first, first_mask = encode("a")
    expected, expected_mask = first.clone(), first_mask.clone()
    second, _ = encode("b")
    assert not torch.equal(expected, second)
    first.zero_(); first_mask.zero_()
    again, mask = encode("a")  # a becomes newest; caller mutations cannot reach the cache.
    assert torch.equal(again, expected) and torch.equal(mask, expected_mask)
    encode("c")  # evicts b, not a.
    assert torch.equal(encode("a")[0], expected)
    encode("b")
    assert frozen.text_encoder.calls == 4
    assert cache.snapshot() == {"entries": 2, "capacity": 2, "hits": 2, "misses": 4}
    assert all(context.device.type == mask.device.type == "cpu" for context, mask in cache._entries.values())


def test_zero_capacity_is_uncached_reference_and_bad_sizes_are_rejected():
    cache, frozen, _ = _cache(size=0)
    for _ in range(2):
        cache.encode("same", device="cpu", dtype=torch.float32)
    assert frozen.text_encoder.calls == 2 and cache.snapshot()["entries"] == 0
    for size in (-1, True, 1.5):
        with pytest.raises(ValueError, match="non-negative integer"):
            PolicyPromptCache(frozen.encode, max_entries=size)


def test_gpu_request_transports_only_features_and_mask_while_encoder_stays_cpu(monkeypatch):
    cache, frozen, _ = _cache()
    original_to, transfers = torch.Tensor.to, []
    def intercept_cuda(value, *args, **kwargs):
        device = kwargs.get("device")
        if device is not None and torch.device(device).type == "cuda":
            transfers.append(tuple(value.shape))
            kwargs = {**kwargs, "device": "cpu"}
        return original_to(value, *args, **kwargs)
    # CPU CI observes the transport boundary without pretending to test a
    # real CUDA kernel. The encoder computation below is real tiny Wan.
    monkeypatch.setattr(torch.Tensor, "to", intercept_cuda)
    cache.encode("one instruction", device="cuda:0", dtype=torch.bfloat16)
    cache.encode("one instruction", device="cuda:0", dtype=torch.bfloat16)
    assert transfers == [(1, 5, 8), (1, 5), (1, 5, 8), (1, 5)]
    assert frozen.text_encoder.input_devices == ["cpu"]
    assert all(parameter.device.type == "cpu" for parameter in frozen.text_encoder.parameters())


def test_inference_config_disables_training_options_and_never_registers_text_encoder():
    config = OmegaConf.create({"load_text_encoder": True, "mot_checkpoint_mixed_attn": True,
        "compile_training_denoise": True,
        "video_dit_config": {"use_gradient_checkpointing": "${mot_checkpoint_mixed_attn}"},
        "action_dit_config": {"use_gradient_checkpointing": "${mot_checkpoint_mixed_attn}"}})
    inference = _inference_model_config(config)
    assert config.load_text_encoder and config.mot_checkpoint_mixed_attn
    assert not inference.load_text_encoder and not inference.mot_checkpoint_mixed_attn
    assert not inference.compile_training_denoise
    assert not inference.video_dit_config.use_gradient_checkpointing
    assert not inference.action_dit_config.use_gradient_checkpointing
    cache, frozen, _ = _cache()
    # Exercise FastWAM.to itself without allocating the real multi-billion
    # parameter policy. Its independent prompt encoder remains on CPU.
    model = FastWAM.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.mot, model.vae, model.text_encoder = torch.nn.Linear(2, 2), torch.nn.Identity(), None
    model.to("meta")
    assert next(model.mot.parameters()).device.type == "meta"
    assert all(parameter.device.type == "cpu" for parameter in frozen.text_encoder.parameters())
    cache.encode("still usable", device="cpu", dtype=torch.float32)


def test_factory_loads_only_text_directly_on_cpu_and_binds_weight_tokenizer_identity(tmp_path, monkeypatch):
    from fastwam.models.wan22.helpers import loader
    from fastwam.models.wan22 import wan_video_text_encoder
    weights = tmp_path / "text.pt"
    torch.save(_TinyWan().state_dict(), weights)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text('{"vocabulary":"test"}')
    loaded, tokenizers, downloads = [], [], []
    def config(path, name):
        return SimpleNamespace(path=str(path), download_if_necessary=lambda: downloads.append(name))
    configs = (config("unused_dit", "dit"), config(weights, "text"),
               config("unused_vae", "vae"), config(tokenizer_path, "tokenizer"))
    monkeypatch.setattr(loader, "_resolve_configs", lambda **kwargs: configs)
    def load(path, name, *, torch_dtype, device):
        loaded.append((path, name, torch_dtype, device))
        return _TinyWan()
    monkeypatch.setattr(loader, "_load_registered_model", load)
    def tokenizer(**kwargs):
        tokenizers.append(kwargs)
        return _Tokenizer((Path(kwargs["name"]) / "tokenizer.json").read_bytes())
    monkeypatch.setattr(wan_video_text_encoder, "HuggingfaceTokenizer", tokenizer)
    cache = load_policy_prompt_cache({"model_id": "wan", "tokenizer_model_id": "t5", "tokenizer_max_len": 128},
                                    encoder_device="cpu", dtype=torch.bfloat16, max_entries=16)
    assert downloads == ["text", "tokenizer"]
    assert loaded == [(str(weights), "wan_video_text_encoder", torch.bfloat16, "cpu")]
    assert tokenizers == [{"name": str(tokenizer_path), "seq_len": 128, "clean": "whitespace"}]
    identity = cache.identity
    assert identity["encoder_device"] == "cpu" and identity["encoder_dtype"] == "torch.bfloat16"
    assert len(identity["encoder_sha256"]) == len(identity["tokenizer_sha256"]) == 64
    identity["encoder_device"] = "cuda"
    assert cache.identity["encoder_device"] == "cpu"
    (tokenizer_path / "download.metadata").write_text("volatile local cache timestamp")
    unchanged = load_policy_prompt_cache({"model_id": "wan", "tokenizer_model_id": "t5"}, dtype=torch.bfloat16)
    assert unchanged.identity["tokenizer_sha256"] == cache.identity["tokenizer_sha256"]
    (tokenizer_path / "tokenizer.json").write_text('{"vocabulary":"changed"}')
    changed = load_policy_prompt_cache({"model_id": "wan", "tokenizer_model_id": "t5"}, dtype=torch.bfloat16)
    assert changed.identity["tokenizer_sha256"] != cache.identity["tokenizer_sha256"]


def test_base_deployment_uses_same_cache_across_replans_without_prompt_fallback():
    cache, frozen, tokenizer = _cache()
    requests = []
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.model = SimpleNamespace(device="cpu", torch_dtype=torch.float32,
        infer_action=lambda **kwargs: requests.append(kwargs) or {"action": torch.zeros(32, 14)})
    policy._policy_prompt_cache = cache
    policy.zeva_mode = "base"
    policy._build_robotwin_image_tensor = lambda observation: torch.zeros(1, 3, 16, 16)
    policy._normalize_state = lambda state: torch.zeros(1, 14)
    policy._denormalize_action = lambda action: action[None].numpy()
    policy._model_seed_stream = ModelSeedStream(7)
    policy._replan_index = 0
    policy.action_horizon, policy.num_inference_steps = 32, 10
    policy.negative_prompt, policy.text_cfg_scale, policy.sigma_shift = "", 1.0, None
    policy.rand_device, policy.tiled, policy.timing_enabled = "cpu", False, False
    observation = {"joint_action": {"vector": np.zeros(14)}}
    policy._infer_action_chunk(observation, "pick the cup")
    policy._infer_action_chunk(observation, "pick the cup")
    policy._infer_action_chunk(observation, "open the drawer")
    assert frozen.text_encoder.calls == 2 and cache.snapshot()["hits"] == 1
    assert all(request["prompt"] is None for request in requests)
    assert all(request["context_mask"].all() for request in requests)
    assert len(tokenizer.calls) == 2 and policy._model_seed_stream.draws == 3


def test_base_constructor_with_joint_recipe_records_loaded_base_identity(tmp_path, monkeypatch):
    from experiments.robotwin.fastwam_policy import deploy_policy
    from fastwam.zeva.attempt_protocol import CONTINUOUS_NOISE_PROTOCOL
    from fastwam.zeva.checkpoint import checkpoint_sha256

    class TinyPolicy(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.device, self.torch_dtype = "cpu", torch.float32
            self.zeva_training_recipe = config.zeva.training_recipe
        def load_checkpoint(self, path):
            self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        def load_zeva_addon_checkpoint(self, *args, **kwargs):
            raise AssertionError("base evaluation must not load the joint addon/foundation")

    class TinyProcessor(torch.nn.Module):
        def set_normalizer_from_stats(self, stats):
            self.stats = stats

    config = OmegaConf.create({"zeva": {"training_recipe": "ego_joint_warmup",
                                        "task_context": {"mode": "task_language"}}})
    base, stats = tmp_path / "base.pt", tmp_path / "stats.json"
    torch.save({"weight": torch.tensor([7.0])}, base)
    stats.write_text("{}")
    monkeypatch.setattr(deploy_policy, "instantiate",
                        lambda cfg, **kwargs: TinyPolicy(cfg) if kwargs else TinyProcessor())
    monkeypatch.setattr(deploy_policy, "load_dataset_stats_from_json", lambda path: {})
    cache, _, _ = _cache(size=16)
    monkeypatch.setattr(deploy_policy, "load_policy_prompt_cache", lambda *args, **kwargs: cache)
    policy = WorldActionRobotWinPolicy(
        model_cfg=config, processor_cfg=OmegaConf.create({}), checkpoint_path=str(base),
        dataset_stats_path=stats, device="cpu", model_dtype=torch.float32,
        action_horizon=32, replan_steps=24, num_inference_steps=10, sigma_shift=None,
        seed=7, text_cfg_scale=1.0, negative_prompt="", rand_device="cpu", tiled=False,
        timing_enabled=False, num_video_frames=9, zeva_mode="base",
        model_rng_protocol=CONTINUOUS_NOISE_PROTOCOL,
    )
    # Exercise actual construction, base state loading, rollout metadata, and
    # RNG identity binding even though the inherited model recipe says joint.
    assert policy.model.zeva_training_recipe == "ego_joint_warmup"
    assert policy.model.weight.item() == 7.0 and not policy.model.weight.requires_grad
    metadata = policy.evaluation_metadata()
    assert metadata["training_recipe"] == "frozen_parent"
    assert metadata["foundation_mode"] == "frozen"
    assert metadata["checkpoint_identity"]["base_sha256"] == checkpoint_sha256(base)
    assert metadata["checkpoint_identity"]["addon_sha256"] is None
    assert "foundation_sha256" not in metadata["checkpoint_identity"]
    assert policy._model_noise_stream.checkpoint_identity == metadata["checkpoint_identity"]


def test_zeva_deployment_uses_same_cache_for_repeated_instruction():
    from test_separate_prefix_deploy import _runtime_policy
    memory = {"bit_effects": torch.zeros(1, 4, 2), "bit_mask": torch.zeros(1, 4, dtype=torch.bool),
              "pim_mask": torch.ones(1, 192, dtype=torch.bool)}
    policy, requests = _runtime_policy(memory)
    cache, frozen, _ = _cache()
    policy._policy_prompt_cache = cache
    observation = {"joint_action": {"vector": np.zeros(14)}}
    for _ in range(2):
        policy._infer_action_chunk(observation, "pick the cup")
    assert frozen.text_encoder.calls == 1 and cache.snapshot()["hits"] == 1
    assert all(request["prompt"] is None for request in requests)
    assert torch.equal(requests[0]["context"], requests[1]["context"])


def test_memory_telemetry_reports_allocator_and_device_scope_without_resetting_peaks(monkeypatch):
    assert _cuda_memory_metadata("cpu") is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (600, 1000))
    for name, value in (("memory_allocated", 100), ("memory_reserved", 200),
                        ("max_memory_allocated", 300), ("max_memory_reserved", 400)):
        monkeypatch.setattr(torch.cuda, name, lambda device, result=value: result)
    values = _cuda_memory_metadata("cuda:0")
    assert values["allocated_bytes"] == 100 and values["peak_reserved_bytes"] == 400
    assert values["device_free_bytes"] == 600 and values["device_total_bytes"] == 1000
