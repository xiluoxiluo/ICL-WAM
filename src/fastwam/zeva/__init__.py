"""Zeva causal-memory addon for FastWAM RoboTwin."""

from .behavior_prefix_adapter import BehaviorPrefixAdapter, BehaviorPrefixAdapterConfig
from .causal_prompt import CausalPromptConfig, CausalPromptEncoder, task_tokens_from_context
from .causal_transition_encoder import CausalTransitionEncoder, CausalTransitionEncoderConfig
from .cte_losses import CTELossConfig, causal_transition_encoder_loss
from .lifecycle import CausalCTEHistory, CausalMemoryLifecycle, LifecycleConfig
from .memory import BriefInteractionTrace, PersistentInteractionMemory, PersistentInteractionMemoryConfig
from .retrieval import MemoryBank, RetrievalResult
from .schemas import CacheManifest, TransitionRecord, build_transition_view, transition_valid_mask
from .cache import PhaseEffectCache, save_phase_effect_cache
from .vae_adapter import FastWAMCTELatentEncoder, load_frozen_wan_vae, make_frame_encoder, validate_vae_metadata

__all__ = [
    "BehaviorPrefixAdapter",
    "BehaviorPrefixAdapterConfig",
    "CausalPromptConfig",
    "CausalPromptEncoder",
    "task_tokens_from_context",
    "CausalTransitionEncoder",
    "CausalTransitionEncoderConfig",
    "CTELossConfig",
    "CausalMemoryLifecycle",
    "CausalCTEHistory",
    "LifecycleConfig",
    "BriefInteractionTrace",
    "PersistentInteractionMemory",
    "PersistentInteractionMemoryConfig",
    "MemoryBank",
    "RetrievalResult",
    "CacheManifest",
    "TransitionRecord",
    "build_transition_view",
    "transition_valid_mask",
    "causal_transition_encoder_loss",
    "PhaseEffectCache",
    "save_phase_effect_cache",
    "FastWAMCTELatentEncoder",
    "make_frame_encoder",
    "load_frozen_wan_vae",
    "validate_vae_metadata",
]
