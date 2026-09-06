# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Explicit RGB-to-CTE input adapters.

The Zeva CTE itself is deliberately kept generic: it consumes a tensor with
shape ``[B,T,C,H,W]`` and does not know how the tensor was produced.  Zeva's
released serving path obtains that tensor by encoding each RGB observation
with the frozen Wan VAE.  Keeping that operation here (rather than in the
CTE) makes the boundary auditable and prevents a checkpoint trained on one
representation from being silently used with another one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


_VAE_IDENTITY_KEYS = (
    "model_id",
    "z_dim",
    "temporal_downsample_factor",
    "upsampling_factor",
)


def validate_vae_metadata(expected: dict[str, object], actual: dict[str, object]) -> None:
    """Check semantic VAE identity while allowing paths to differ per host."""
    for key in _VAE_IDENTITY_KEYS:
        if key not in expected or key not in actual:
            raise ValueError(f"VAE metadata is missing required identity field {key!r}")
        if str(expected[key]) != str(actual[key]):
            raise ValueError(
                f"VAE metadata mismatch for {key}: expected {expected[key]!r}, "
                f"got {actual[key]!r}"
            )


class FastWAMCTELatentEncoder:
    """Encode RGB frames with an already-loaded, frozen FastWAM/Wan model.

    ``model`` may expose the public FastWAM ``_encode_video_latents`` helper,
    the single-image ``_encode_input_image_latents_tensor`` helper, or Zeva's
    ``encode`` method.  No model parameters are modified.  The result is
    always the CTE convention ``[B,C_latent,H_latent,W_latent]``.
    """

    def __init__(
        self,
        model: Any,
        *,
        resize: tuple[int, int] = (480, 832),
        expected_channels: int | None = None,
        input_range: str = "auto",
    ) -> None:
        self.model = model
        self.resize = tuple(int(value) for value in resize)
        self.expected_channels = None if expected_channels is None else int(expected_channels)
        self.input_range = str(input_range)
        if len(self.resize) != 2 or min(self.resize) < 1:
            raise ValueError("resize must contain two positive dimensions")
        if self.expected_channels is not None and self.expected_channels < 1:
            raise ValueError("expected_channels must be positive")
        if self.input_range not in {"auto", "zero_one", "minus_one_one", "uint8"}:
            raise ValueError("input_range must be auto, zero_one, minus_one_one, or uint8")

    def _as_rgb_batch(self, rgb: Tensor) -> Tensor:
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must be [B,3,H,W] or [3,H,W], got {tuple(rgb.shape)}")
        if self.input_range == "uint8" or rgb.dtype == torch.uint8:
            rgb = rgb.float() / 127.5 - 1.0
        elif self.input_range == "minus_one_one":
            rgb = rgb.float().clamp(-1.0, 1.0)
        elif self.input_range == "zero_one":
            rgb = rgb.float().clamp(0.0, 1.0) * 2.0 - 1.0
        else:
            rgb = rgb.float()
            minimum, maximum = float(rgb.detach().amin()), float(rgb.detach().amax())
            if minimum < -1e-4:
                if maximum > 1.0001:
                    raise ValueError("RGB values must be uint8/[0,255], [0,1], or [-1,1]")
                return rgb.clamp(-1.0, 1.0)
            if maximum > 1.0001:
                rgb = rgb / 127.5 - 1.0
            else:
                rgb = rgb * 2.0 - 1.0
        if self.input_range == "auto" and float(rgb.detach().amax()) > 1.0001:
            rgb = rgb / 127.5 - 1.0
        return rgb.clamp(-1.0, 1.0)

    def _encode(self, normalized: Tensor) -> Tensor:
        module = self.model if isinstance(self.model, torch.nn.Module) else getattr(self.model, "model", None)
        if isinstance(module, torch.nn.Module):
            try:
                parameter = next(module.parameters())
                normalized = normalized.to(device=parameter.device, dtype=parameter.dtype)
            except StopIteration:
                pass
        # Wan's video VAE expects a temporal axis.  The one-frame call is
        # causal and yields a single latent frame for every RGB input.
        video = normalized.unsqueeze(2)
        if hasattr(self.model, "vae") and hasattr(self.model.vae, "model"):
            latent = self.model.vae.model.encode(video, self.model.vae.scale)
        elif hasattr(self.model, "_encode_video_latents"):
            latent = self.model._encode_video_latents(video, tiled=False)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "encode"):
            scale = getattr(self.model, "scale", None)
            if scale is None:
                raise TypeError("VAE wrapper does not expose its normalization scale")
            latent = self.model.model.encode(video, scale)
        elif hasattr(self.model, "encode"):
            latent = self.model.encode(video)
        elif hasattr(self.model, "_encode_input_image_latents_tensor"):
            outputs = [self.model._encode_input_image_latents_tensor(frame, tiled=False) for frame in normalized]
            latent = torch.cat(outputs, dim=0)
        else:
            raise TypeError(
                "model must expose _encode_video_latents, encode, or "
                "_encode_input_image_latents_tensor"
            )
        if latent.ndim != 5 or latent.shape[2] != 1:
            raise ValueError(f"VAE encoder must return [B,C,1,H,W], got {tuple(latent.shape)}")
        latent = latent[:, :, 0].float().contiguous()
        if latent.shape[0] != normalized.shape[0]:
            raise ValueError("VAE encoder changed the RGB batch size")
        if self.expected_channels is not None and latent.shape[1] != self.expected_channels:
            raise ValueError(
                f"VAE latent channels mismatch: expected {self.expected_channels}, got {latent.shape[1]}"
            )
        return latent

    @torch.no_grad()
    def encode(self, rgb: Tensor) -> Tensor:
        """Return one CTE latent frame per RGB input."""
        normalized = self._as_rgb_batch(rgb)
        normalized = F.interpolate(normalized, size=self.resize, mode="bilinear", align_corners=False)
        return self._encode(normalized)

    @torch.no_grad()
    def encode_history(self, rgb_history: Tensor) -> Tensor:
        """Encode ``[B,T,3,H,W]`` RGB history into ``[B,T,C,H',W']``."""
        if rgb_history.ndim != 5 or rgb_history.shape[2] != 3:
            raise ValueError(f"rgb_history must be [B,T,3,H,W], got {tuple(rgb_history.shape)}")
        batch, steps = rgb_history.shape[:2]
        latent = self.encode(rgb_history.flatten(0, 1))
        return latent.reshape(batch, steps, *latent.shape[1:])


def load_frozen_wan_vae(
    *,
    model_id: str,
    tokenizer_model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.float32,
    redirect_common_files: bool = True,
) -> tuple[Any, dict[str, object]]:
    """Load only the VAE component through FastWAM's registered loader."""
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs

    _dit_config, _text_config, vae_config, _tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch_dtype,
        device=device,
    )
    vae.eval().requires_grad_(False)
    return vae, {
        "model_id": str(model_id),
        "vae_path": str(vae_config.path),
        "z_dim": int(getattr(vae, "z_dim", getattr(getattr(vae, "model", None), "z_dim", 0))),
        "temporal_downsample_factor": int(getattr(vae, "temporal_downsample_factor", 0)),
        "upsampling_factor": int(getattr(vae, "upsampling_factor", 0)),
    }


def make_frame_encoder(
    input_type: str,
    *,
    model: Any | None = None,
    resize: tuple[int, int] = (480, 832),
    expected_channels: int | None = None,
) -> Callable[[Tensor], Tensor] | None:
    """Build an explicit encoder for a configured CTE input contract."""
    if input_type == "rgb_frame":
        return None
    if input_type != "wan_vae_latent":
        raise ValueError("input_type must be 'rgb_frame' or 'wan_vae_latent'")
    if model is None:
        raise ValueError("wan_vae_latent input requires an explicit frozen VAE model")
    return FastWAMCTELatentEncoder(
        model, resize=resize, expected_channels=expected_channels,
        input_range="minus_one_one",
    ).encode_history
