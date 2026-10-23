"""Worker-side tensor batching for the two-GPU CTE latent cache.

Stack inputs in source order. A plain dict lets DataLoader pin tensors before
nonblocking host-to-device transfer; no model or frame processing is changed.
"""

from __future__ import annotations

from typing import Any

import torch


def collate_latent_windows(items: list[Any]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("CTE latent batching requires at least one window")
    return {
        "frames": torch.stack([item.frames for item in items], dim=0),
        "actions": torch.stack([item.transition_actions for item in items], dim=0),
        "frame_valid": torch.stack([item.frame_valid for item in items], dim=0),
        "transition_valid": torch.stack([item.transition_valid for item in items], dim=0),
    }
