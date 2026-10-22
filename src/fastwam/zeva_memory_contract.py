"""Versioned online PIM semantics, without importing Torch or model code."""

from __future__ import annotations

from collections.abc import Mapping
import math


PIM_MEMORY_CONTRACT_SCHEMA = "iclwam_cross_attempt_pim_memory_v1"


def pim_memory_contract(config: Mapping) -> dict | None:
    """Resolve only behavior-relevant memory settings; legacy defaults stay explicit."""
    if not bool(config.get("cross_attempt_only", False)):
        return None
    compression = str(config.get("compression", "uniform"))
    indexing = str(config.get("uniform_indexing", "legacy_floor"))
    if compression not in {"uniform", "similarity_merge"}:
        raise ValueError("PIM compression must be uniform or similarity_merge")
    if indexing not in {"legacy_floor", "reference_round"}:
        raise ValueError("PIM uniform_indexing must be legacy_floor or reference_round")
    counts = {
        "max_past_attempts": config.get("max_past_attempts", 3),
        "max_entries_per_attempt": config.get("max_entries_per_attempt", 64),
        "capacity": config.get("pim_max_entries", config.get("capacity", 192)),
    }
    for name, value in counts.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"PIM {name} must be a positive integer")
    if counts["capacity"] < counts["max_past_attempts"] * counts["max_entries_per_attempt"]:
        raise ValueError("PIM capacity must retain all bounded complete attempts")
    threshold = float(config.get("merge_threshold", 0.85))
    phase = float(config.get("beta_phase", config.get("phase_merge_weight", 0.5)))
    effect = float(config.get("beta_effect", config.get("effect_merge_weight", 0.5)))
    if (not all(math.isfinite(value) for value in (threshold, phase, effect))
            or not -1 <= threshold <= 1 or min(phase, effect) < 0 or phase + effect <= 0):
        raise ValueError("Invalid PIM similarity-merge threshold or weights")
    return {
        "schema": PIM_MEMORY_CONTRACT_SCHEMA, "compression": compression,
        "uniform_indexing": indexing, **counts,
        # These values do not affect uniform compression; canonicalize them
        # so changing an unused legacy threshold does not invent a new policy.
        "merge_threshold": threshold if compression == "similarity_merge" else None,
        "phase_weight": phase / (phase + effect) if compression == "similarity_merge" else None,
    }


def validate_pim_memory_contract(contract: dict) -> dict:
    if not isinstance(contract, dict) or contract.get("schema") != PIM_MEMORY_CONTRACT_SCHEMA:
        raise ValueError("Missing or unsupported PIM memory contract")
    phase_weight = contract.get("phase_weight")
    if phase_weight is not None and (type(phase_weight) not in {int, float}
                                     or not math.isfinite(phase_weight)
                                     or not 0 <= phase_weight <= 1):
        raise ValueError("PIM phase_weight must be finite and in [0,1]")
    config = {
        **contract, "cross_attempt_only": True,
        "merge_threshold": 0.85 if contract.get("merge_threshold") is None else contract["merge_threshold"],
        "beta_phase": 0.5 if phase_weight is None else phase_weight,
        "beta_effect": 0.5 if phase_weight is None else 1.0 - phase_weight,
    }
    canonical = pim_memory_contract(config)
    if contract != canonical:
        raise ValueError("PIM memory contract contains inconsistent or unversioned fields")
    return dict(canonical)


def checkpoint_pim_memory_contract(payload: dict) -> dict | None:
    recorded = payload.get("pim_memory_contract")
    stored = payload.get("config", {})
    memory = stored.get("model", {}).get("zeva", {}).get("memory") if isinstance(stored, dict) else None
    from_config = pim_memory_contract(memory) if isinstance(memory, dict) else None
    if recorded is not None:
        recorded = validate_pim_memory_contract(recorded)
        if from_config is not None and recorded != from_config:
            raise ValueError("Checkpoint PIM metadata disagrees with its saved configuration")
        return recorded
    # Old full training configs establish the historical uniform/floor policy.
    # Do not accept unmarked new compression semantics as an old checkpoint.
    if from_config is not None:
        if (from_config["compression"], from_config["uniform_indexing"]) != ("uniform", "legacy_floor"):
            raise ValueError("Checkpoint must explicitly record its non-legacy PIM memory contract")
        return from_config
    return None


def validate_checkpoint_pim_memory_contract(payload: dict, expected: dict | None, *,
                                            allow_compression_override: bool = False) -> dict | None:
    """Reject silent online policy changes; allow a recorded compression-only ablation."""
    saved = checkpoint_pim_memory_contract(payload)
    if expected is None:
        return saved
    expected = validate_pim_memory_contract(expected)
    if saved is None:
        if (expected["compression"], expected["uniform_indexing"]) == ("uniform", "legacy_floor"):
            # Early addons omitted config metadata. Only an explicitly legacy
            # target can supply the otherwise implicit old serving behavior.
            return expected
        raise ValueError("Legacy addon has no PIM memory contract; select explicit uniform/legacy_floor configuration")
    if saved != expected:
        changed = {key for key in saved if saved[key] != expected[key]}
        if not allow_compression_override or not changed <= {"compression", "merge_threshold", "phase_weight"}:
            raise ValueError("Checkpoint PIM memory contract mismatch; use matching compression/indexing or an explicit compression ablation")
    return saved
