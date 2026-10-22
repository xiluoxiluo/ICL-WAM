"""Lightweight contracts for original Zeva observed-interaction adaptation.

The pinned public main branch defines the CTE/MCE/prior and runtime. Its full
CTE optimization/data pipeline is not released; local recipes are recorded as
such rather than borrowing the incompatible Zeva-Ego recipe identity.
"""
from collections.abc import Mapping

MAIN_PROFILE = "main_observed"
MAIN_REFERENCE_COMMIT = "17ca9fa0c68dca1fc7915dd6433de70a100e93a1"
MAIN_ARTIFACT_LINEAGE = "zeva_main_observed_effect_v1"
MAIN_NUMERICAL_CONTRACT = "zeva_main_v1"
MAIN_ADDON_SCHEMA = "zeva_main_observed_policy_v1"
MAIN_JOINT_RECIPES = frozenset({"main_joint_parent", "main_joint_pim"})
MAIN_RECIPES = MAIN_JOINT_RECIPES | {"main_frozen_parent", "main_frozen_pim"}


def validate_main_model_config(zeva: Mapping) -> None:
    if zeva.get("profile") != MAIN_PROFILE:
        raise ValueError("Expected explicit main_observed profile")
    cte, memory = zeva.get("cte", {}), zeva.get("memory", {})
    prompt, adapter = zeva.get("prompt", {}), zeva.get("adapter", {})
    if cte.get("variant", "window") == "zeva_ego":
        raise ValueError("main_observed requires the completed-effect CTE, not predictive Ego")
    required = {"mode": "exact_zeva", "prefix_layout": "legacy_merged",
                "effect_source": "latest_completed", "prior_variant": "legacy_bit",
                "pim_variant": "legacy", "numerical_contract": MAIN_NUMERICAL_CONTRACT}
    for name, value in required.items():
        if adapter.get(name) != value:
            raise ValueError(f"main_observed adapter requires {name}={value}")
    if prompt.get("pim_variant") != "legacy" or prompt.get("numerical_contract") != MAIN_NUMERICAL_CONTRACT:
        raise ValueError("main_observed requires the original task/phase/BIT/PIM MCE")
    if memory.get("memory_semantics") != "completed_effect_trace" or memory.get("cross_attempt_only", False):
        raise ValueError("main_observed publishes completed interactions online, including the current attempt")
    if zeva.get("task_context", {}).get("mode") != "static":
        raise ValueError("main_observed requires frozen initial image/text static retrieval")
    if zeva.get("training_recipe") not in MAIN_RECIPES:
        raise ValueError("main_observed requires an explicit main training recipe")
    if prompt.get("pim_context_dropout", 0) or prompt.get("pim_support_dropout", 0):
        raise ValueError("The released main MCE has no internal dropout; augment offline support explicitly")


def main_profile_metadata(success_data_identity: Mapping) -> dict:
    if not isinstance(success_data_identity, Mapping) or not success_data_identity:
        raise ValueError("main checkpoint requires success-only data identity")
    return {"profile": MAIN_PROFILE, "reference_commit": MAIN_REFERENCE_COMMIT,
            "artifact_lineage": MAIN_ARTIFACT_LINEAGE,
            "memory_semantics": "completed_effect_trace",
            "success_data_identity": dict(success_data_identity)}
