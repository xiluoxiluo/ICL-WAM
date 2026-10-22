"""Original-main CTE/static-query compute precision, separate from policy dtype.

The released main server keeps these feature models in FP32 and calls them
under CUDA BF16 autocast, then widens the returned features to FP32. The
policy's EAP/MCE has a different rule: it follows the full policy's dtype.
CPU-only development uses explicit FP32, never an inherited AMP context.
"""

from contextlib import nullcontext

import torch


def main_feature_autocast(device, *, enabled: bool):
    """Use the main serving compute rule without changing stored parameters.

    Callers must opt in only for validated main-v2 artifacts. Legacy/Ego
    callers retain their existing ambient context and numerical semantics.
    """
    if not enabled:
        return nullcontext()
    device = torch.device(device)
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type == "cuda")
