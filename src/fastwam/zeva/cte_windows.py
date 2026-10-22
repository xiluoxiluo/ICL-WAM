"""Real observed-prefix boundaries for the main CTE's fixed-size disk windows."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor


MAIN_CTE_WINDOW_SEMANTICS = "full_observed_prefix_masked_tail_v1"


def validate_main_cte_window_semantics(payload: Mapping[str, object]) -> None:
    """Old complete-H32 caches omitted real terminal transitions; rebuild them."""
    if (payload.get("profile") == "main_observed"
            and payload.get("cte_window_semantics") != MAIN_CTE_WINDOW_SEMANTICS):
        raise ValueError(
            "main CTE artifact lacks full observed-prefix/masked-tail semantics; "
            "rebuild the main latent/phase caches and retrain CTE from real terminal boundaries"
        )


def observed_prefix_transitions(frame_valid: Tensor, transition_valid: Tensor) -> int:
    """Count contiguous four-control intervals whose endpoints were observed.

    The disk window keeps its native H32 shape. Its padding is storage only:
    callers must trim at this returned boundary before constructing a causal
    history. A partial final control group never establishes a transition.
    """
    if frame_valid.ndim != 1 or transition_valid.ndim != 2:
        raise ValueError("CTE window masks must be [T] and [T-1,4]")
    if frame_valid.shape[0] != transition_valid.shape[0] + 1 or transition_valid.shape[1] != 4:
        raise ValueError("CTE window observation/control masks are not aligned")
    frames, controls = frame_valid.bool(), transition_valid.bool()
    if not bool(frames[0]):
        raise ValueError("a real CTE window must begin with an observed image")
    for name, values in (("observation", frames), ("control", controls.flatten())):
        if bool((values[1:] & ~values[:-1]).any()):
            raise ValueError(f"CTE {name} validity must be a contiguous observed prefix")
    complete = frames[:-1] & frames[1:] & controls.all(dim=-1)
    if bool((complete[1:] & ~complete[:-1]).any()):
        raise ValueError("CTE completed transitions cannot contain an observation gap")
    return int(complete.sum().item())
