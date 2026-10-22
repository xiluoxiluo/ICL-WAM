"""Frozen policy text encoding independent of the action model's device.

The complete UMT5 encoder can stay on CPU while only its small contextual
features move to the policy GPU. This is separate from task-language retrieval:
it preserves FastWAM's contextual prompt encoding and visible zero padding.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Callable

import torch
from torch import Tensor

from .checkpoint import checkpoint_sha256


POLICY_TEXT_SCHEMA = "iclwam_policy_text_v1"
PROMPT_SEMANTICS = "wan_whitespace_special_tokens_zero_padding_visible_v1"


class FrozenPolicyTextEncoder:
    """Own a frozen UMT5 outside the action model's registered modules."""

    def __init__(self, text_encoder, tokenizer, *, device: str = "cpu", dtype=torch.bfloat16):
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("policy text encoder device must be cpu or cuda")
        self.dtype = dtype
        self.text_encoder = text_encoder.to(device=self.device, dtype=dtype).eval().requires_grad_(False)
        self.tokenizer = tokenizer

    @torch.no_grad()
    def encode(self, prompt: str) -> tuple[Tensor, Tensor]:
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(device=self.device, dtype=torch.bool)
        context = self.text_encoder(ids, mask)
        # Match FastWAM.encode_prompt exactly: padded features are zero, but
        # those zero slots remain visible to the policy's cross attention.
        for row, length in enumerate(mask.gt(0).sum(dim=1).long()):
            context[row, length:] = 0
        return context, torch.ones_like(mask)


class PolicyPromptCache:
    """Bounded LRU of immutable CPU features, with explicit output transport."""

    def __init__(self, encode: Callable[[str], tuple[Tensor, Tensor]], *, max_entries: int = 16,
                 identity: dict | None = None):
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("policy text cache size must be a non-negative integer")
        self._encode = encode
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[Tensor, Tensor]] = OrderedDict()
        self._identity = {
            **deepcopy(identity or {}), "schema": POLICY_TEXT_SCHEMA,
            "prompt_semantics": PROMPT_SEMANTICS, "cache_size": max_entries,
            "cache_storage_device": "cpu",
        }
        self.hits, self.misses = 0, 0

    @property
    def identity(self) -> dict:
        return deepcopy(self._identity)

    def snapshot(self) -> dict:
        return {"entries": len(self._entries), "capacity": self.max_entries,
                "hits": self.hits, "misses": self.misses}

    @torch.no_grad()
    def encode(self, prompt: str, *, device, dtype) -> tuple[Tensor, Tensor]:
        if not isinstance(prompt, str):
            raise TypeError("policy prompt cache requires one prompt string")
        value = self._entries.get(prompt)
        if value is not None:
            self.hits += 1
            self._entries.move_to_end(prompt)
        else:
            self.misses += 1
            context, mask = self._encode(prompt)
            if (context.ndim != 3 or mask.ndim != 2 or context.shape[:2] != mask.shape
                    or context.shape[0] != 1 or not context.is_floating_point()
                    or not bool(torch.isfinite(context).all())):
                raise ValueError("policy text encoder must return finite [1,L,D] features and [1,L] mask")
            # Never retain encoder activations or a GPU tensor across replans.
            value = (context.detach().to(device="cpu", copy=True).contiguous(),
                     mask.detach().to(device="cpu", dtype=torch.bool, copy=True).contiguous())
            if self.max_entries:
                self._entries[prompt] = value
                if len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
        context, mask = value
        # copy=True also protects CPU cache entries when the caller runs on CPU
        # and modifies its returned context (e.g. by appending proprio features).
        return (context.to(device=device, dtype=dtype, copy=True),
                mask.to(device=device, dtype=torch.bool, copy=True))


def _artifact_digest(path: str | list[str]) -> str:
    """Content identity independent of a model cache's absolute host path."""
    if isinstance(path, list):
        parts = [(Path(value).name, _artifact_digest(value)) for value in path]
        return hashlib.sha256(json.dumps(sorted(parts), separators=(",", ":")).encode()).hexdigest()
    root = Path(path)
    if root.is_file():
        return checkpoint_sha256(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing policy text artifact: {root}")
    parts = [(str(value.relative_to(root)), checkpoint_sha256(value))
             for value in sorted(root.rglob("*")) if value.is_file()]
    if not parts:
        raise ValueError("Policy tokenizer directory is empty")
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def _tokenizer_snapshot_digest(tokenizer) -> str:
    # Save the tokenizer actually used for inference, not its download folder.
    # HF strips original host paths from this portable snapshot. Download locks,
    # timestamps and unrelated model-cache files cannot change the identity.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    with tempfile.TemporaryDirectory(prefix="iclwam_policy_tokenizer_") as folder:
        tokenizer.save_pretrained(folder)
        return _artifact_digest(folder)


def load_policy_prompt_cache(model_cfg, *, encoder_device: str = "cpu", dtype=torch.bfloat16,
                             max_entries: int = 16) -> PolicyPromptCache:
    """Load only text components, directly onto their configured device.

    The action model must be constructed with load_text_encoder=False. In
    particular this never creates a GPU UMT5 and offloads it after the fact.
    """
    # Imports are deferred so cache use/tests do not initialize model loaders.
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    device = torch.device(encoder_device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("policy text encoder device must be cpu or cuda")
    if type(max_entries) is not int or max_entries < 0:
        raise ValueError("policy text cache size must be a non-negative integer")
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=str(model_cfg["model_id"]), tokenizer_model_id=str(model_cfg["tokenizer_model_id"]),
        redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    encoder = _load_registered_model(text_config.path, "wan_video_text_encoder", torch_dtype=dtype,
                                     device=str(device))
    max_length = int(model_cfg.get("tokenizer_max_len", 512))
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=max_length, clean="whitespace")
    frozen = FrozenPolicyTextEncoder(encoder, tokenizer, device=str(device), dtype=dtype)
    return PolicyPromptCache(frozen.encode, max_entries=max_entries, identity={
        "encoder_device": str(device), "encoder_dtype": str(dtype),
        "encoder_sha256": _artifact_digest(text_config.path),
        "tokenizer_sha256": _tokenizer_snapshot_digest(tokenizer),
        "tokenizer_max_length": max_length,
    })
