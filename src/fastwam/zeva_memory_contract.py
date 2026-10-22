"""Versioned online PIM semantics, without importing Torch or model code."""

from __future__ import annotations

from collections.abc import Mapping
import math


LEGACY_PIM_MEMORY_CONTRACT_SCHEMA = "iclwam_cross_attempt_pim_memory_v1"
PIM_MEMORY_CONTRACT_SCHEMA = "iclwam_cross_attempt_pim_memory_v2"
MAIN_PIM_MEMORY_CONTRACT_SCHEMA = "iclwam_main_observed_pim_v1"


def main_pim_memory_contract(config: Mapping) -> dict:
    if config.get("cross_attempt_only", False):
        raise ValueError("original Zeva PIM accepts completed current-attempt interactions")
    capacity = config.get("pim_max_entries", config.get("capacity", 64))
    top_k = config.get("pim_top_k", config.get("top_k", 4))
    bit_size = config.get("bit_size", 4)
    if any(type(x) is not int or x < 1 for x in (capacity, top_k, bit_size)) or top_k > capacity:
        raise ValueError("invalid main PIM capacity/top_k/BIT size")
    threshold = float(config.get("merge_threshold", 0.85))
    phase = float(config.get("beta_phase", config.get("phase_weight", 0.5)))
    effect = float(config.get("beta_effect", config.get("effect_weight", 0.5)))
    if (not all(math.isfinite(x) for x in (threshold, phase, effect))
            or not -1 <= threshold <= 1 or min(phase, effect) < 0 or phase + effect <= 0):
        raise ValueError("invalid main similarity merge parameters")
    if config.get("pim_retrieval_mode", "phase") != "phase":
        raise ValueError("original Zeva PIM retrieves by current phase")
    return {"schema": MAIN_PIM_MEMORY_CONTRACT_SCHEMA,
            "memory_semantics": "completed_effect_trace", "capacity": capacity,
            "top_k": top_k, "bit_size": bit_size, "merge_threshold": threshold,
            "phase_weight": phase / (phase + effect), "effect_weight": effect / (phase + effect),
            "retrieval": "phase", "commit_timing": "completed_window",
            "episode_scoped": True, "cross_attempt_only": False}


def pim_memory_contract(config: Mapping) -> dict | None:
    """Resolve only behavior-relevant memory settings; legacy defaults stay explicit."""
    if config.get("memory_semantics") == "completed_effect_trace":
        return main_pim_memory_contract(config)
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
    timing = str(config.get("commit_timing", "attempt_end"))
    storage = str(config.get("storage_device", "cpu"))
    if timing not in {"attempt_end", "attempt_reset"} or storage not in {"cpu", "policy"}:
        raise ValueError("Invalid PIM commit_timing or storage_device")
    reference_lifecycle = (timing, storage) != ("attempt_end", "cpu")
    contract = {
        "schema": PIM_MEMORY_CONTRACT_SCHEMA if reference_lifecycle else LEGACY_PIM_MEMORY_CONTRACT_SCHEMA,
        "compression": compression,
        "uniform_indexing": indexing, **counts,
        # These values do not affect uniform compression; canonicalize them
        # so changing an unused legacy threshold does not invent a new policy.
        "merge_threshold": threshold if compression == "similarity_merge" else None,
        "phase_weight": phase / (phase + effect) if compression == "similarity_merge" else None,
    }

    if reference_lifecycle:
        contract.update(commit_timing=timing, storage_device=storage)
    return contract


def validate_pim_memory_contract(contract: dict) -> dict:
    if isinstance(contract, dict) and contract.get("schema") == MAIN_PIM_MEMORY_CONTRACT_SCHEMA:
        canonical = main_pim_memory_contract(contract)
        if canonical != contract:
            raise ValueError("main PIM contract contains inconsistent or unversioned fields")
        return canonical
    if not isinstance(contract, dict) or contract.get("schema") not in {PIM_MEMORY_CONTRACT_SCHEMA, LEGACY_PIM_MEMORY_CONTRACT_SCHEMA}:
        raise ValueError("Missing or unsupported PIM memory contract")
    if contract["schema"] == PIM_MEMORY_CONTRACT_SCHEMA and not {"commit_timing", "storage_device"} <= contract.keys():
        raise ValueError("PIM memory v2 must declare commit timing and storage device")
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
        if from_config["schema"] == MAIN_PIM_MEMORY_CONTRACT_SCHEMA:
            raise ValueError("main observed-effect checkpoint must record its PIM memory contract")
        if (from_config["schema"] != LEGACY_PIM_MEMORY_CONTRACT_SCHEMA
                or (from_config["compression"], from_config["uniform_indexing"]) != ("uniform", "legacy_floor")):
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
        if (expected["schema"] == LEGACY_PIM_MEMORY_CONTRACT_SCHEMA
                and (expected["compression"], expected["uniform_indexing"]) == ("uniform", "legacy_floor")):
            # Early addons omitted config metadata. Only an explicitly legacy
            # target can supply the otherwise implicit old serving behavior.
            return expected
        raise ValueError("Legacy addon has no PIM memory contract; select explicit uniform/legacy_floor configuration")
    if saved != expected:
        changed = {key for key in saved.keys() | expected.keys() if saved.get(key) != expected.get(key)}
        if not allow_compression_override or not changed <= {"compression", "merge_threshold", "phase_weight"}:
            raise ValueError("Checkpoint PIM memory contract mismatch; use matching compression/indexing or an explicit compression ablation")
    return saved
