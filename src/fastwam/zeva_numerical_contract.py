"""Versioned EAP/PIM numerical semantics, usable by lightweight eval tools."""

from collections.abc import Mapping
from fastwam.zeva_main_contract import MAIN_NUMERICAL_CONTRACT


LEGACY_NUMERICAL_CONTRACT = "legacy_v1"
REFERENCE_NUMERICAL_CONTRACT = "zeva_ego_v1"
# Ego explicitly preserves FP32 addon inference. Original main instead casts
# the full policy (including EAP/MCE) to its configured inference precision.
FP32_NUMERICAL_CONTRACTS = frozenset({REFERENCE_NUMERICAL_CONTRACT})
REFERENCE_NUMERICAL_CONTRACTS = FP32_NUMERICAL_CONTRACTS | {MAIN_NUMERICAL_CONTRACT}


def _validated(value) -> str:
    if value not in {LEGACY_NUMERICAL_CONTRACT, *REFERENCE_NUMERICAL_CONTRACTS}:
        raise ValueError("unsupported Zeva numerical_contract")
    return value


def config_numerical_contract(zeva_config: Mapping) -> str:
    """The reader and policy adapter must implement the same recorded recipe."""
    prompt = _validated(zeva_config.get("prompt", {}).get("numerical_contract", LEGACY_NUMERICAL_CONTRACT))
    adapter = _validated(zeva_config.get("adapter", {}).get("numerical_contract", LEGACY_NUMERICAL_CONTRACT))
    if prompt != adapter:
        raise ValueError("prompt and adapter numerical_contract must agree")
    return adapter


def checkpoint_numerical_contract(payload: Mapping) -> str:
    recorded = payload.get("numerical_contract")
    config = payload.get("config", {})
    zeva = config.get("model", {}).get("zeva") if isinstance(config, Mapping) else None
    from_config = config_numerical_contract(zeva) if isinstance(zeva, Mapping) else None
    if recorded is None:
        if from_config in REFERENCE_NUMERICAL_CONTRACTS:
            raise ValueError("reference checkpoint is missing its numerical_contract marker")
        return LEGACY_NUMERICAL_CONTRACT
    recorded = _validated(recorded)
    if from_config is not None and recorded != from_config:
        raise ValueError("checkpoint numerical_contract disagrees with its saved configuration")
    return recorded


def validate_checkpoint_numerical_contract(payload: Mapping, expected: str) -> str:
    saved = checkpoint_numerical_contract(payload)
    if saved != _validated(expected):
        raise ValueError("checkpoint numerical_contract mismatch; use the matching legacy configuration or train the reference recipe")
    return saved
