"""Compatibility import for the canonical RoboTwin policy implementation.

The HTTP server and in-process evaluator intentionally share one class so
normalization, PIM off semantics, attempt lifecycle and RNG cannot drift.
Existing public and private imports continue to resolve through __getattr__.
"""

from experiments.robotwin.fastwam_policy import deploy_policy as _runtime
from experiments.robotwin.fastwam_policy.deploy_policy import (
    WorldActionRobotWinPolicy,
    _mixed_precision_to_model_dtype,
    encode_obs,
    eval,
    get_model,
    reset_model,
)


def __getattr__(name):
    return getattr(_runtime, name)


__all__ = [name for name in dir(_runtime) if not name.startswith("_")]
