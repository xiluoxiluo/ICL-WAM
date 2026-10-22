"""Separate FP32 training masters from each reference's serving precision."""

from contextvars import ContextVar
from functools import wraps

import torch
from torch import nn

from fastwam.zeva_numerical_contract import FP32_NUMERICAL_CONTRACTS, MAIN_NUMERICAL_CONTRACT


_reference_training = ContextVar("zeva_reference_training", default=None)


class ReferenceFloat32Module(nn.Module):
    """Preserve FP32 values until an explicit main inference conversion.

    Calling the original conversion and then .float() would already round
    the weights. Device selection is taken from the converted tensor, while
    the actual FP32 copy is made from the original tensor instead.
    Ego also serves in FP32. Original main serves in the policy dtype, so
    FastWAM.prepare_zeva_inference_precision explicitly releases its master
    weight protection only on a separate deployment model.
    """

    def _apply(self, fn, recurse=True):
        contract = getattr(getattr(self, "config", None), "numerical_contract", None)
        strict = contract in FP32_NUMERICAL_CONTRACTS or (
            contract == MAIN_NUMERICAL_CONTRACT
            and not getattr(self, "_main_inference_precision", False)
        )
        if not strict:
            return super()._apply(fn, recurse=recurse)

        def preserve_float32(tensor):
            converted = fn(tensor)
            if tensor.is_floating_point() and converted.dtype != torch.float32:
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(preserve_float32, recurse=recurse)


def reference_inference_precision(method):
    """Keep Ego eval FP32; main follows its weights and caller's autocast."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        strict = getattr(getattr(self, "config", None), "numerical_contract", None) in FP32_NUMERICAL_CONTRACTS
        if not strict:
            return method(self, *args, **kwargs)
        # FastWAM stage freezing toggles individual child modules, rather
        # than the adapter's top-level .training flag. Any trained child
        # means the caller's AMP context must still govern this forward.
        # A nested frozen EAP must inherit that decision: eval() alone does
        # not exempt a frozen module from the reference training autocast.
        training = _reference_training.get()
        if training is None:
            training = kwargs.get("training", any(module.training for module in self.modules()))
        token = _reference_training.set(training)
        try:
            if not training:
                parameter = next(self.parameters())
                with torch.autocast(device_type=parameter.device.type, enabled=False):
                    return method(self, *args, **kwargs)
            return method(self, *args, **kwargs)
        finally:
            _reference_training.reset(token)

    return wrapped
