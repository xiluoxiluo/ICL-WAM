import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.constants import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.zeva import (
    CausalCTEHistory,
    CausalMemoryLifecycle,
    FastWAMCTELatentEncoder,
    LifecycleConfig,
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    validate_vae_metadata,
    PersistentInteractionMemory,
    PersistentInteractionMemoryConfig,
    task_tokens_from_context,
    TaskContextBank,
    retrieve_task_context,
)
from fastwam.zeva.attempt_protocol import (ModelSeedStream, ContinuousModelNoiseStream,
    CONTINUOUS_NOISE_PROTOCOL, DEVICE_CONTINUOUS_NOISE_PROTOCOL, CONTINUOUS_NOISE_PROTOCOLS,
    MAIN_MODEL_RNG_PROTOCOL, MainAttemptSeedSchedule)
from fastwam.zeva.ego_episode_data import prepare_ego_rgb_views
from fastwam.zeva_memory_contract import pim_memory_contract, validate_checkpoint_pim_memory_contract
from fastwam.zeva_numerical_contract import config_numerical_contract, validate_checkpoint_numerical_contract
from fastwam.zeva.checkpoint import checkpoint_sha256, load_cte_checkpoint
from fastwam.zeva.static_task_context import (
    StaticTaskContextRetriever, StaticTaskContextSession, task_context_identity,
)
from fastwam.zeva.task_language import TaskLanguageMemory, task_language_identity
from fastwam.zeva.prompt_cache import load_policy_prompt_cache
from fastwam.zeva.main_runtime import MainObservedRuntime, PhaseMatchedSuccessReplay

logger = logging.getLogger(__name__)


def _inference_model_config(model_cfg: Any) -> DictConfig:
    """Keep training memory options and the large text encoder out of policy init."""
    config = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
    config.load_text_encoder = False
    config.mot_checkpoint_mixed_attn = False
    config.compile_training_denoise = False
    for name in ("video_dit_config", "action_dit_config"):
        if name in config and config[name] is not None:
            config[name].use_gradient_checkpointing = False
    return config


def _prefix_metadata(model: Any) -> dict[str, Any]:
    """Record the prefix contract used by the loaded model in rollout logs."""
    config = getattr(getattr(model, "zeva_behavior_prefix_adapter", None), "config", None)
    if config is None:
        return {"prefix_layout": None, "effect_source": None}
    layout = getattr(config, "prefix_layout", "legacy_merged")
    return {
        "prefix_layout": layout,
        "effect_source": (getattr(config, "effect_source", None)
                          if layout == "zeva_ego" or getattr(model, "zeva_profile", None) == "main_observed" else None),
    }


def _cuda_memory_metadata(device: Any) -> dict[str, Any] | None:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(device)
    return {
        "scope": "process_allocator_peak_since_start",
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        # Device-wide free space also reflects the simulator/renderer and
        # other processes, which Torch's allocator counters cannot include.
        "device_free_bytes": int(free), "device_total_bytes": int(total),
    }


def _supports_boundary_observations(model_cfg: Any, processor_cfg: Any) -> bool:
    effect_source = model_cfg.get("zeva", {}).get("adapter", {}).get("effect_source", "latest_completed")
    transforms = processor_cfg.get("action_state_transforms")
    return (effect_source == "current_prediction" and (transforms is None or len(transforms) == 0)
            and not bool(processor_cfg.get("use_stepwise_action_norm", False)))


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    # Match RobotVideoDataset exactly: tensor bilinear resize with antialias
    # enabled (the processor first maps each raw camera to 240x320, then the
    # RoboTwin mosaic maps it to its 256/128-pixel tile).
    import torchvision.transforms.functional as transforms_F

    array = np.asarray(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div(255.0)
    else:
        tensor = tensor.float()
        if float(tensor.detach().amax()) > 1.0:
            tensor = tensor.div(255.0)
    resized = transforms_F.resize(
        tensor,
        size=[int(size_wh[1]), int(size_wh[0])],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return resized.permute(1, 2, 0).contiguous().numpy()


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        video_size: tuple[int, int] | list[int] = (384, 320),
        zeva_mode: str = "base",
        cte_checkpoint: Optional[str] = None,
        addon_checkpoint: Optional[str] = None,
        pim_top_k: int = 4,
        max_attempts: int = 1,
        cross_memory_enabled: bool = True,
        pim_diagnostic_interval: int = 0,
        model_rng_protocol: str = "request_seed_stream_v1",
        evaluation_slot_index: int = 0,
        evaluation_slot_count: int = 1,
        model_rng_state_path: Optional[str] = None,
        model_rng_task_name: Optional[str] = None,
        model_rng_run_id: Optional[str] = None,
        skip_get_obs_within_replan: bool = False,
        policy_text_encoder_device: str = "cpu",
        policy_text_cache_size: int = 16,
        pim_success_replay_enabled: bool = False,
        pim_success_replay_phase_threshold: float = 0.80,
        pim_success_replay_blend: float = 1.0,
    ) -> None:
        model_cfg_copy = _inference_model_config(model_cfg)
        self.zeva_profile = str(model_cfg_copy.get("zeva", {}).get("profile", "legacy"))
        self._main_observed = self.zeva_profile == "main_observed"
        # The published main protocol declares one discarded model request.
        # Our local client scopes it to this model instance, across all resets.
        self._main_warmup_requests = 0
        self._main_warmup_seed: int | None = None
        if pim_success_replay_enabled and not self._main_observed:
            raise ValueError("successful-action replay requires the explicit main_observed profile")
        self._success_replay = PhaseMatchedSuccessReplay(
            enabled=pim_success_replay_enabled,
            phase_threshold=pim_success_replay_phase_threshold, blend=pim_success_replay_blend,
        )
        self._main_runtime: MainObservedRuntime | None = None
        self._main_episode_instruction: str | None = None
        self._pending_replay_actions: deque[bool] = deque()
        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval().requires_grad_(False)
        if self._main_observed:
            # Joint main weights must not change the frozen initial-observation
            # retrieval feature space. The model preserves the base readout on CPU.
            self.model.zeva_freeze_readout_on_load = True
        # Build UMT5 independently on CPU by default. It is never registered
        # under self.model, so policy.to(cuda) cannot move it or create a peak
        # GPU allocation before offloading. Only cached text features move.
        self._policy_prompt_cache = load_policy_prompt_cache(
            model_cfg_copy, encoder_device=policy_text_encoder_device,
            dtype=model_dtype, max_entries=policy_text_cache_size,
        )
        self.policy_text_encoder_device = torch.device(policy_text_encoder_device).type
        self.policy_text_cache_size = policy_text_cache_size
        requested_zeva_mode = str(zeva_mode)
        if requested_zeva_mode not in {
            "base",
            "zeva_stage2",
            "pim_shadow",
            "pim_on",
        }:
            raise ValueError(
                "zeva_mode must be base, zeva_stage2, pim_shadow, or pim_on"
            )
        self.skip_get_obs_within_replan = bool(skip_get_obs_within_replan)
        if (requested_zeva_mode != "base" and self.skip_get_obs_within_replan
                and not _supports_boundary_observations(model_cfg_copy, processor_cfg)):
            raise ValueError("skipping non-boundary observations requires current_prediction with state-independent joint normalization")
        zeva_memory_cfg = model_cfg_copy.get("zeva", {}).get("memory", {})
        if self._main_observed and (
            zeva_memory_cfg.get("memory_semantics") != "completed_effect_trace"
            or bool(zeva_memory_cfg.get("cross_attempt_only", False))
            or model_cfg_copy.get("zeva", {}).get("adapter", {}).get("effect_source", "latest_completed") != "latest_completed"
        ):
            raise ValueError("main_observed requires observed completed effects and immediate phase-retrieved PIM")
        task_context_cfg = model_cfg_copy.get("zeva", {}).get("task_context", {})
        self.task_context_bank = None
        self._static_task_context_session = None
        self._task_language_memory = None
        self._task_language_instruction = None
        self._task_language_result = None
        self._last_task_context_retrieval = None
        self.task_context_mode = str(task_context_cfg.get("mode", "static"))
        self.task_context_top_k = int(task_context_cfg.get("top_k", 5))
        if requested_zeva_mode not in {"base", "pim_shadow"} and self.task_context_mode == "task_language":
            for name in ("bank_path", "retrieval_checkpoint", "language_encoder_path"):
                if _is_none_like(task_context_cfg.get(name)):
                    raise ValueError(f"task-language context requires zeva.task_context.{name}")
            if _is_none_like(cte_checkpoint):
                raise ValueError("task-language context requires the matching CTE checkpoint")
            if float(task_context_cfg.get("temperature", 1.0)) != 1.0:
                raise ValueError("task-language top-5 retrieval uses the reference unscaled cosine softmax")
            bank_path = str(task_context_cfg["bank_path"])
            head_path = str(task_context_cfg["retrieval_checkpoint"])
            encoder_path = str(task_context_cfg["language_encoder_path"])
            self._task_language_memory = TaskLanguageMemory.load(
                bank_path, head_path, encoder_path=encoder_path,
                top_k=self.task_context_top_k, device=device,
                expected={
                    "cte_checkpoint_sha256": checkpoint_sha256(str(cte_checkpoint)),
                    "dataset_stats_sha256": checkpoint_sha256(dataset_stats_path),
                    "memory_semantics": "predicted_effect_trace", "effect_source": "current_prediction",
                    "cte_variant": "zeva_ego",
                },
            )
            if (self.model.zeva_prompt_encoder is None
                    or self._task_language_memory.bank.value_dim != self.model.zeva_prompt_encoder.config.global_dim):
                raise ValueError("task-language memory value_dim must match the Zeva prompt encoder")
            self.model.zeva_task_context_identity = task_language_identity(
                bank_path, head_path, encoder_path, top_k=self.task_context_top_k,
            )
        elif requested_zeva_mode not in {"base", "pim_shadow"} and self.task_context_mode == "bank":
            if self.model.zeva_prompt_encoder is None:
                raise ValueError("task-context bank requires an enabled Zeva prompt encoder")
            bank_path = task_context_cfg.get("bank_path")
            if _is_none_like(bank_path):
                raise ValueError("zeva.task_context.bank_path is required when mode=bank")
            self.task_context_bank = TaskContextBank.load(
                str(bank_path),
                expected_key_dim=int(task_context_cfg.get("key_dim", 256)),
                expected_value_dim=int(
                    task_context_cfg.get(
                        "value_dim",
                        self.model.zeva_prompt_encoder.config.global_dim,
                    )
                ),
            )
            if self.task_context_bank.value_dim != int(self.model.zeva_prompt_encoder.config.global_dim):
                raise ValueError("zeva.task_context.value_dim must match zeva.prompt.global_dim")
            if self.task_context_top_k < 1 or self.task_context_top_k > len(self.task_context_bank):
                raise ValueError(
                    "zeva.task_context.top_k must be within the task-context bank size; "
                    f"got {self.task_context_top_k}, bank_size={len(self.task_context_bank)}"
                )
        elif requested_zeva_mode not in {"base", "pim_shadow"} and self.task_context_mode == "static":
            for name in ("bank_path", "retrieval_checkpoint"):
                if _is_none_like(task_context_cfg.get(name)):
                    raise ValueError(f"static task context requires zeva.task_context.{name}")
            if _is_none_like(cte_checkpoint):
                raise ValueError("static task context requires the matching CTE checkpoint")
            bank_path = str(task_context_cfg["bank_path"])
            head_path = str(task_context_cfg["retrieval_checkpoint"])
            retriever = StaticTaskContextRetriever.load(
                bank_path, head_path, top_k=self.task_context_top_k, device=device,
                expected={
                    "base_checkpoint_sha256": checkpoint_sha256(checkpoint_path),
                    "cte_checkpoint_sha256": checkpoint_sha256(str(cte_checkpoint)),
                    "dataset_stats_sha256": checkpoint_sha256(dataset_stats_path),
                    "video_size": [int(value) for value in video_size],
                    "context_len": int(model_cfg_copy.get("tokenizer_max_len", 128)),
                    "readout_dim": int(model_cfg_copy.video_dit_config.hidden_dim),
                    **({"memory_semantics": "predicted_effect_trace", "effect_source": "current_prediction", "cte_variant": "zeva_ego"}
                       if model_cfg_copy.get("zeva", {}).get("adapter", {}).get("effect_source") == "current_prediction" else {}),
                },
            )
            if self.model.zeva_prompt_encoder is None or retriever.bank.value_dim != self.model.zeva_prompt_encoder.config.global_dim:
                raise ValueError("static behavior bank value_dim must match the Zeva prompt encoder")
            self._static_task_context_session = StaticTaskContextSession(retriever)
            self.model.zeva_task_context_identity = task_context_identity(
                "static", bank_path, head_path, self.task_context_top_k,
            )
        elif requested_zeva_mode not in {"base", "pim_shadow"} and self.task_context_mode != "pooling":
            raise ValueError("zeva.task_context.mode must be pooling, bank, static, or task_language")

        self.zeva_mode = requested_zeva_mode
        self.cross_memory_enabled = bool(cross_memory_enabled)
        self.pim_diagnostic_interval = int(pim_diagnostic_interval)
        if self.pim_diagnostic_interval < 0:
            raise ValueError("pim_diagnostic_interval must be non-negative")
        self._last_pim_diagnostics = {}
        self._terminal_observation_step = None
        numerical_contract = config_numerical_contract(model_cfg_copy.get("zeva", {}))
        policy_device = torch.device(self.model.device)
        if policy_device.type == "cuda" and policy_device.index is None:
            policy_device = torch.device("cuda", torch.cuda.current_device())
        self.checkpoint_identity = {
            "policy_device": str(policy_device),
            "numerical_contract": numerical_contract if requested_zeva_mode != "base" else None,
            "cte_exploratory_epoch40": bool(model_cfg_copy.get("zeva", {}).get("cte", {}).get("exploratory_epoch40", False)),
            "cte_require_reference_recipe": bool(model_cfg_copy.get("zeva", {}).get("cte", {}).get("require_reference_recipe", numerical_contract == "zeva_ego_v1")),
            "training_query_scope": str(model_cfg_copy.get("zeva", {}).get("task_context", {}).get("training_query_scope", "episode_static")),
            "inference_task_context_scope": "reference_first_instruction" if numerical_contract == "zeva_ego_v1" else "legacy_constant_instruction",
            "base_sha256": checkpoint_sha256(checkpoint_path),
            "stats_sha256": checkpoint_sha256(dataset_stats_path),
            "cte_sha256": checkpoint_sha256(cte_checkpoint) if cte_checkpoint else None,
            "addon_sha256": checkpoint_sha256(addon_checkpoint) if addon_checkpoint else None,
            "task_context": getattr(self.model, "zeva_task_context_identity", None),
            "policy_text": self._policy_prompt_cache.identity,
            # A base evaluation may reuse a joint Stage2B architecture config,
            # but it loaded only the original base weights above. Bind the
            # identity to those deployed weights, not that unused recipe.
            "training_recipe": ("frozen_parent" if requested_zeva_mode == "base"
                                else str(getattr(self.model, "zeva_training_recipe", "frozen_parent"))),
            "foundation_checkpoint_required": False,
            "pim_memory_contract": pim_memory_contract(zeva_memory_cfg) if requested_zeva_mode != "base" else None,
            "pim_compression_override": bool(zeva_memory_cfg.get("allow_compression_override", False)) if requested_zeva_mode != "base" else False,
            **_prefix_metadata(self.model),
        }
        self.cte = None
        self._ego_cte = False
        self.cte_schema = None
        self._cte_state = None
        self._ego_has_started = False
        self.lifecycle = None
        self._cte_history = None
        self._cte_frame_encoder = None
        self._observed_effect_count = 0
        self._transition_actions: list[torch.Tensor] = []
        self._attempt_finalized = False
        if len(video_size) != 2 or min(int(v) for v in video_size) < 1:
            raise ValueError(f"video_size must be [H, W] with positive dimensions, got {video_size}")
        self.video_size = tuple(int(v) for v in video_size)
        self.max_attempts = int(max_attempts)
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._needs_episode_reset = True
        if self.zeva_mode != "base":
            if not getattr(self.model, "zeva_enabled", False):
                raise ValueError("Zeva evaluation requires model config zeva.enabled=true")
            if not cte_checkpoint:
                raise ValueError("cte_checkpoint is required for Zeva evaluation")
            if requested_zeva_mode in {"zeva_stage2", "pim_on"} and not addon_checkpoint:
                raise ValueError(
                    "addon_checkpoint is required for zeva_stage2 and pim_on evaluation"
                )
            cte_payload = torch.load(str(cte_checkpoint), map_location="cpu", weights_only=False)
            if self._main_observed:
                from fastwam.zeva.main_data import inherit_main_artifact, validate_main_artifact
                validate_main_artifact(cte_payload)
                if cte_payload["success_data_identity"]["dataset_stats_sha256"] != checkpoint_sha256(dataset_stats_path):
                    raise ValueError("main_observed deployment normalization differs from the verified training data")
                self.model.zeva_success_data_identity = dict(cte_payload["success_data_identity"])
                self.model.zeva_main_artifact_metadata = inherit_main_artifact(cte_payload)
                self.checkpoint_identity["success_data_identity"] = dict(cte_payload["success_data_identity"])
                for artifact_key in ("bank_path", "retrieval_checkpoint"):
                    artifact_path = task_context_cfg.get(artifact_key)
                    if not _is_none_like(artifact_path):
                        validate_main_artifact(
                            torch.load(str(artifact_path), map_location="cpu", weights_only=False),
                            expected=cte_payload,
                        )
            cte_values = dict(cte_payload.get("config", cte_payload.get("model_config", {})))
            cte_values = dict(cte_values.get("cte", cte_values))
            checkpoint_action_dim = int(cte_payload.get("action_dim", cte_values.get("action_dim", -1)))
            if checkpoint_action_dim != 14 or tuple(cte_payload.get("camera_keys", ())) != ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                raise ValueError(
                    "CTE checkpoint metadata is incompatible with RoboTwin V1 "
                    "(action_dim=14, cameras=cam_high/cam_left_wrist/cam_right_wrist)"
                )
            self.cte_schema = str(cte_payload.get("schema", "legacy_cte"))
            self.checkpoint_identity["cte_schema"] = self.cte_schema
            self._ego_cte = self.cte_schema == "iclwam_zeva_ego_cte_v1"
            if self._main_observed and self._ego_cte:
                raise ValueError("main_observed cannot consume a predictive Zeva-Ego CTE checkpoint")
            if self._ego_cte:
                from fastwam.zeva.ego_cte_training import load_ego_cte_checkpoint
                cte_options = model_cfg_copy.get("zeva", {}).get("cte", {})
                exploratory = bool(cte_options.get("exploratory_epoch40", False))
                strict_recipe = bool(cte_options.get("require_reference_recipe", numerical_contract == "zeva_ego_v1"))
                self.cte, cte_payload = load_ego_cte_checkpoint(
                    str(cte_checkpoint), device=device, require_ready=True,
                    exploratory_epoch40=exploratory, require_reference_recipe=strict_recipe)
                self.checkpoint_identity.update(
                    cte_exploratory_epoch40=exploratory, cte_require_reference_recipe=strict_recipe,
                    cte_epoch=int(cte_payload["epoch"]), cte_data_scope=cte_payload.get("manifest", {}).get("data_scope"))
                cte_cfg = self.cte.cfg
                if (cte_cfg.executed_horizon, cte_cfg.action_dim, cte_cfg.image_size) != (24, 14, 224):
                    raise ValueError("RoboTwin Zeva-Ego requires H24/joint14/RGB224")
                if cte_payload.get("dataset_stats_sha256") != checkpoint_sha256(dataset_stats_path):
                    raise ValueError("Zeva-Ego deployment action stats differ from CTE training")
                adapter_cfg = self.model.zeva_behavior_prefix_adapter.config
                if getattr(adapter_cfg, "effect_source", "latest_completed") != "current_prediction":
                    raise ValueError("Zeva-Ego CTE requires current_prediction prefix conditioning")
                if zeva_memory_cfg.get("memory_semantics") != "predicted_effect_trace":
                    raise ValueError("Zeva-Ego CTE requires explicit predicted_effect_trace runtime memory")
                if requested_zeva_mode != "pim_shadow":
                    if self._task_language_memory is not None:
                        bank = self._task_language_memory.bank
                    elif self._static_task_context_session is not None:
                        bank = self._static_task_context_session.retriever.bank
                    else:
                        raise ValueError("formal Zeva-Ego deployment requires a trained task-memory retriever")
                    if bank.metadata.get("semantic_task_map_sha256") != cte_payload["semantic_task_map_sha256"]:
                        raise ValueError("static task-memory semantic map differs from CTE training")
                    train_episodes = set(cte_payload["episode_splits"]["train"])
                    if any(entry["episode_id"] not in train_episodes for entry in bank.entries):
                        raise ValueError("static task-memory contains a non-training CTE episode")
            else:
                allowed_cte = set(CausalTransitionEncoderConfig.__dataclass_fields__)
                cte_cfg = CausalTransitionEncoderConfig(**{k: v for k, v in cte_values.items() if k in allowed_cte})
                if getattr(self.model.zeva_behavior_prefix_adapter.config, "effect_source", "latest_completed") == "current_prediction":
                    raise ValueError("legacy CTE checkpoints do not provide a causal current predicted effect")
            if not self._ego_cte and (
                cte_cfg.action_dim != 14
                or cte_cfg.transition_steps != 4
                or cte_cfg.effect_window_transitions != 4
            ):
                raise ValueError(
                    "RoboTwin Zeva V1 requires CTE action_dim=14, "
                    "transition_steps=4, and effect_window_transitions=4; "
                    f"got action_dim={cte_cfg.action_dim}, "
                    f"transition_steps={cte_cfg.transition_steps}, "
                    f"effect_window_transitions={cte_cfg.effect_window_transitions}"
                )
            if cte_cfg.phase_dim != int(self.model.zeva_prompt_encoder.config.phase_dim) or cte_cfg.effect_dim != int(self.model.zeva_prompt_encoder.config.effect_dim):
                raise ValueError(
                    "CTE phase/effect dimensions must match the attached CausalPromptEncoder: "
                    f"cte=({cte_cfg.phase_dim},{cte_cfg.effect_dim}), "
                    f"prompt=({self.model.zeva_prompt_encoder.config.phase_dim},"
                    f"{self.model.zeva_prompt_encoder.config.effect_dim})"
                )
            prompt_cfg = self.model.zeva_prompt_encoder.config
            cross_attempt = getattr(prompt_cfg, "pim_variant", "legacy") == "cross_attempt"
            if cross_attempt != bool(zeva_memory_cfg.get("cross_attempt_only", False)):
                raise ValueError("Cross-attempt model and runtime memory configuration disagree")
            if int(prompt_cfg.persistent_length) != int(pim_top_k):
                raise ValueError(
                    "pim_top_k must match the trained CausalPromptEncoder persistent_length; "
                    f"got pim_top_k={pim_top_k}, persistent_length={prompt_cfg.persistent_length}"
                )
            configured_top_k = int(zeva_memory_cfg.get("pim_top_k", pim_top_k))
            if configured_top_k != int(pim_top_k):
                raise ValueError(
                    "Evaluation pim_top_k must match the trained model.zeva.memory.pim_top_k; "
                    f"got {pim_top_k} vs {configured_top_k}"
                )
            if not self._ego_cte:
                self.cte = CausalTransitionEncoder(cte_cfg).to(device).eval()
                load_cte_checkpoint(str(cte_checkpoint), self.cte, map_location="cpu")
            self.cte.requires_grad_(False)
            cte_input_type = str(cte_payload.get("cte_input_type", cte_values.get("input_type", "rgb_frame")))
            if self._main_observed and cte_input_type != "wan_vae_latent":
                raise ValueError("main_observed requires the frozen Wan-VAE observed-effect CTE input contract")
            if self._ego_cte:
                if cte_input_type != "rgb_multiview":
                    raise ValueError("Zeva-Ego CTE checkpoint must declare rgb_multiview")
            elif cte_input_type == "rgb_frame":
                if cte_cfg.image_channels != 3:
                    raise ValueError("rgb_frame CTE checkpoints must set image_channels=3")
            elif cte_input_type == "wan_vae_latent":
                if int(cte_payload.get("latent_channels", cte_cfg.image_channels)) != cte_cfg.image_channels:
                    raise ValueError("CTE latent channel metadata does not match image_channels")
                vae_metadata = dict(cte_payload.get("vae_metadata", {}))
                required_vae_metadata = {
                    "model_id",
                    "vae_path",
                    "z_dim",
                    "temporal_downsample_factor",
                    "upsampling_factor",
                }
                if not required_vae_metadata.issubset(vae_metadata):
                    raise ValueError(
                        "wan_vae_latent CTE checkpoints must record complete "
                        f"VAE identity metadata: {sorted(required_vae_metadata)}"
                    )
                runtime_vae_metadata = {
                    "model_id": str(model_cfg.get("model_id", "")),
                    "z_dim": int(getattr(self.model.vae, "z_dim", -1)),
                    "temporal_downsample_factor": int(
                        getattr(self.model.vae, "temporal_downsample_factor", -1)
                    ),
                    "upsampling_factor": int(getattr(self.model.vae, "upsampling_factor", -1)),
                }
                if runtime_vae_metadata["model_id"]:
                    validate_vae_metadata(vae_metadata, runtime_vae_metadata)
                checkpoint_size = cte_payload.get("cte_vae_input_size")
                if checkpoint_size is None or len(checkpoint_size) != 2:
                    raise ValueError(
                        "wan_vae_latent CTE checkpoints must record cte_vae_input_size as [H, W]"
                    )
                checkpoint_size = tuple(int(v) for v in checkpoint_size)
                if checkpoint_size != self.video_size:
                    raise ValueError(
                        "CTE VAE input size mismatch: checkpoint declares "
                        f"{checkpoint_size}, evaluation data uses {self.video_size}"
                    )
                self._cte_frame_encoder = FastWAMCTELatentEncoder(
                    self.model,
                    resize=self.video_size,
                    expected_channels=cte_cfg.image_channels,
                    input_range="minus_one_one",
                ).encode_history
            else:
                raise ValueError(
                    f"Unsupported CTE input type {cte_input_type!r}; expected rgb_frame or wan_vae_latent"
                )
            if addon_checkpoint and requested_zeva_mode in {"zeva_stage2", "pim_on"}:
                addon_payload = self.model.load_zeva_addon_checkpoint(
                    str(addon_checkpoint),
                    base_checkpoint_sha256=checkpoint_sha256(checkpoint_path),
                    cte_checkpoint_sha256=checkpoint_sha256(cte_checkpoint),
                )
                if self._main_observed:
                    from fastwam.zeva.main_data import validate_main_artifact
                    validate_main_artifact(addon_payload, expected=cte_payload)
                self.checkpoint_identity["checkpoint_numerical_contract"] = validate_checkpoint_numerical_contract(
                    addon_payload, numerical_contract)
                self.checkpoint_identity["training_recipe"] = addon_payload.get("training_recipe", "frozen_parent")
                from fastwam.zeva.checkpoint import checkpoint_requires_foundation
                self.checkpoint_identity["foundation_checkpoint_required"] = checkpoint_requires_foundation(addon_payload)
                self.checkpoint_identity["foundation_contract_schema"] = addon_payload.get("foundation_contract_schema")
                self.checkpoint_identity["checkpoint_pim_memory_contract"] = validate_checkpoint_pim_memory_contract(
                    addon_payload, self.checkpoint_identity["pim_memory_contract"],
                    allow_compression_override=self.checkpoint_identity["pim_compression_override"],
                )
                if addon_payload.get("foundation_checkpoint") is not None:
                    self.checkpoint_identity["foundation_sha256"] = addon_payload["foundation_checkpoint"]["sha256"]
            # Loading a joint checkpoint can restore trainability. Inference
            # updates only runtime memory, never foundation/addon parameters.
            self.model.eval().requires_grad_(False)
            if self._main_observed:
                self.model.prepare_zeva_inference_precision()
            pim = PersistentInteractionMemory(
                PersistentInteractionMemoryConfig(
                    phase_dim=cte_cfg.phase_dim,
                    effect_dim=cte_cfg.effect_dim,
                    capacity=int(zeva_memory_cfg.get("pim_max_entries", 64)),
                    top_k=int(pim_top_k),
                    merge_threshold=float(zeva_memory_cfg.get("merge_threshold", 0.85)),
                    phase_merge_weight=float(zeva_memory_cfg.get("beta_phase", 0.5)),
                    effect_merge_weight=float(zeva_memory_cfg.get("beta_effect", 0.5)),
                    cross_attempt_only=bool(zeva_memory_cfg.get("cross_attempt_only", False)),
                    max_past_attempts=int(zeva_memory_cfg.get("max_past_attempts", 3)),
                    max_entries_per_attempt=int(zeva_memory_cfg.get("max_entries_per_attempt", 64)),
                    compression=str(zeva_memory_cfg.get("compression", "uniform")),
                    uniform_indexing=str(zeva_memory_cfg.get("uniform_indexing", "legacy_floor")),
                    commit_timing=str(zeva_memory_cfg.get("commit_timing", "attempt_end")),
                    storage_device=str(zeva_memory_cfg.get("storage_device", "cpu")),
                )
            )
            self.lifecycle = CausalMemoryLifecycle(
                pim,
                LifecycleConfig(
                    bit_size=int(prompt_cfg.brief_length),
                    transition_steps=cte_cfg.transition_steps,
                    effect_window_transitions=cte_cfg.effect_window_transitions,
                    action_dim=cte_cfg.action_dim,
                    memory_semantics="predicted_effect_trace" if self._ego_cte else "completed_effect_trace",
                ),
            )

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        if self.zeva_mode != "base" and self.action_horizon != 32:
            raise ValueError(
                "Zeva V1 requires action_horizon=32 to preserve the trained 8x4 transition alignment"
            )
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        if self.zeva_mode != "base":
            if self._ego_cte and self.replan_steps != int(self.cte.cfg.executed_horizon):
                raise ValueError("Zeva-Ego replan_steps must equal the trained executed_horizon=24")
            transition_steps = int(self.cte.cfg.transition_steps)
            if self.replan_steps % transition_steps:
                raise ValueError(
                    "Zeva evaluation requires replan_steps to be a multiple of "
                    f"the CTE transition_steps ({transition_steps}); got {self.replan_steps}"
                )
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self._model_seed_stream = ModelSeedStream(seed)
        self._model_noise_stream = None
        self._main_seed_schedule = None
        self.evaluation_slot_index, self.evaluation_slot_count = int(evaluation_slot_index), int(evaluation_slot_count)
        if not 0 <= self.evaluation_slot_index < self.evaluation_slot_count:
            raise ValueError("evaluation slot index must be within slot count")
        if self._main_observed:
            if model_rng_protocol != MAIN_MODEL_RNG_PROTOCOL:
                raise ValueError("main_observed requires Zeva-main attempt/replan request seeds")
            if any(value is not None for value in (model_rng_state_path, model_rng_task_name, model_rng_run_id)):
                raise ValueError("Zeva-main request seeds do not use a continuous RNG handoff")
            self._main_seed_schedule = MainAttemptSeedSchedule(seed, device=str(torch.device(rand_device)))
        elif model_rng_protocol in CONTINUOUS_NOISE_PROTOCOLS:
            if seed is None:
                raise ValueError("continuous model RNG requires an explicit model seed")
            self._model_noise_stream = ContinuousModelNoiseStream(seed, slot_index=self.evaluation_slot_index,
                slot_count=self.evaluation_slot_count, run_id=model_rng_run_id,
                state_path=model_rng_state_path, task_name=model_rng_task_name,
                checkpoint_identity=self.checkpoint_identity, protocol=model_rng_protocol,
                device=policy_device if model_rng_protocol == DEVICE_CONTINUOUS_NOISE_PROTOCOL else "cpu")
        elif model_rng_protocol != "request_seed_stream_v1":
            raise ValueError("unsupported model RNG protocol")
        self._attempt_id = 0
        self._replan_index = 0
        self._last_inference_seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]
        action = action.to(dtype=torch.float32, device="cpu")
        dummy_state = torch.zeros(
            (action.shape[0], 1, state_meta[0]["shape"]), dtype=action.dtype
        )
        batch = {
            # ActionStateMerger.backward consumes its concatenated outputs.
            "action": action,
            "state": dummy_state,
        }
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        if self.processor.action_state_transforms is not None:
            for transform in reversed(self.processor.action_state_transforms):
                batch = transform.backward(batch)
        return batch["action"][action_key].numpy()

    def _normalize_action(self, action: np.ndarray, state: Optional[np.ndarray] = None) -> torch.Tensor:
        action_meta = self.processor.shape_meta["action"]
        state_meta = self.processor.shape_meta["state"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        key = action_meta[0]["key"]
        state_key = state_meta[0]["key"]
        value = torch.as_tensor(action, dtype=torch.float32).reshape(1, -1)
        if value.shape[-1] != action_meta[0]["raw_shape"]:
            raise ValueError(f"Expected raw action dim {action_meta[0]['raw_shape']}, got {value.shape[-1]}")
        state_value = torch.zeros((1, state_meta[0]["raw_shape"]), dtype=value.dtype) if state is None else torch.as_tensor(state, dtype=value.dtype).reshape(1, -1)
        if state_value.shape[-1] != state_meta[0]["raw_shape"]:
            raise ValueError(f"Expected raw state dim {state_meta[0]['raw_shape']}, got {state_value.shape[-1]}")
        batch = {"action": {key: value}, "state": {state_key: state_value}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        return batch["action"][0]

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        required = ("head_camera", "left_camera", "right_camera")
        if any(key not in obs_data or "rgb" not in obs_data[key] for key in required):
            raise KeyError(
                "RoboTwin observation must expose RGB arrays under head_camera, "
                "left_camera, and right_camera"
            )
        for key in required:
            rgb = np.asarray(obs_data[key]["rgb"])
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"{key}.rgb must be an HxWx3 array, got {rgb.shape}")
        head_raw = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 240))
        left_raw = _resize_rgb(obs_data["left_camera"]["rgb"], (320, 240))
        right_raw = _resize_rgb(obs_data["right_camera"]["rgb"], (320, 240))
        head = _resize_rgb(head_raw, (320, 256))
        left = _resize_rgb(left_raw, (160, 128))
        right = _resize_rgb(right_raw, (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        if getattr(self, "_main_observed", False):
            # Dataset preprocessing and original main normalize RGB in FP32
            # before the VAE/policy dtype conversion. Casting [0,1] to BF16
            # first introduces a second rounding, especially around grey .5.
            return (image_tensor.float() * 2.0 - 1.0).to(
                device=self.model.device, dtype=self.model.torch_dtype,
            )
        image_tensor = image_tensor.to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * 2.0 - 1.0
        return image_tensor

    def _build_robotwin_cte_views(self, observation: Dict[str, Any]) -> torch.Tensor:
        views = []
        for key in ("head_camera", "left_camera", "right_camera"):
            rgb = np.asarray(observation["observation"][key]["rgb"])
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("CTE expects original three HxWx3 RGB views")
            views.append(torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1))
        return prepare_ego_rgb_views(views, self.cte.cfg.image_size)

    def _step_ego_cte(self, observation: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        previous = None
        if self._ego_has_started:
            if len(self._transition_actions) != self.cte.cfg.executed_horizon:
                raise ValueError("CTE replan requires exactly the previous completed H24 action chunk")
            previous = torch.stack(self._transition_actions).unsqueeze(0)
        elif self._transition_actions:
            raise ValueError("the first CTE boundary must use action SOS")
        parameter = next(self.cte.parameters())
        images = self._build_robotwin_cte_views(observation).unsqueeze(0).to(parameter)
        if previous is not None:
            previous = previous.to(parameter)
        phase, predicted_effect, self._cte_state = self.cte.step(images, previous, self._cte_state)
        self._ego_has_started = True
        self._transition_actions.clear()
        return phase, predicted_effect

    @torch.no_grad()
    def _resolve_task_language(self, instruction: str):
        """Resolve once per attempt from raw instruction alone.

        No image, online CTE state, current episode identity or environment
        task label is exposed to the frozen task-language classifier.
        """
        if getattr(self, "_task_language_result", None) is None:
            features = self._task_language_memory.encoder.encode(instruction, device=self.model.device)
            self._task_language_result = self._task_language_memory.retrieve(features)
            self._task_language_instruction = instruction
        elif (instruction != self._task_language_instruction
              and getattr(self.model, "zeva_numerical_contract", "legacy_v1") != "zeva_ego_v1"):
            raise ValueError("task-language instruction changed within an attempt; reset before changing tasks")
        return self._task_language_result

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        context, _context_mask = self._policy_prompt_cache.encode(
            prompt, device=self.model.device, dtype=self.model.torch_dtype,
        )
        noise_stream = getattr(self, "_model_noise_stream", None)
        initial_noise = None if noise_stream is None else noise_stream.next_noise((1, self.action_horizon, 14))
        main_seeds = getattr(self, "_main_seed_schedule", None)
        request_seed = (main_seeds.next_seed(self._attempt_id, self._replan_index) if main_seeds is not None
                        else self._model_seed_stream.next_seed() if noise_stream is None else None)
        noise_kwargs = {} if initial_noise is None else {"initial_action_noise": initial_noise}
        self._last_inference_seed = request_seed
        if self.zeva_mode != "base":
            if self._needs_episode_reset:
                self.lifecycle.reset_episode(instruction, episode_id=f"episode-{self.episode_count}")
                self._needs_episode_reset = False
            if not getattr(self, "_ego_cte", False) and self._cte_history is None:
                self._cte_history = CausalCTEHistory(self.cte, frame_encoder=self._cte_frame_encoder)
                self._cte_history.reset(image_tensor[0].float())
                self._observed_effect_count = 0
            with torch.no_grad():
                task_dim = int(self.model.zeva_prompt_encoder.config.global_dim)
                if self.zeva_mode == "pim_shadow":
                    task_tokens = torch.zeros(
                        (1, task_dim), device=self.model.device, dtype=getattr(self.model, "zeva_condition_dtype", self.model.torch_dtype)
                    )
                elif getattr(self, "_task_language_memory", None) is not None:
                    task_context_result = self._resolve_task_language(instruction)
                    task_tokens = task_context_result.values
                    self._last_task_context_retrieval = {
                        "indices": task_context_result.indices.cpu(),
                        "scores": task_context_result.scores.cpu(),
                        "sources": task_context_result.sources,
                    }
                elif self._static_task_context_session is not None:
                    task_context_result = self._static_task_context_session.resolve(
                        self.model, image_tensor, context, _context_mask, instruction,
                    )
                    task_tokens = task_context_result.values
                    self._last_task_context_retrieval = {
                        "indices": task_context_result.indices.cpu(),
                        "scores": task_context_result.scores.cpu(),
                        "sources": task_context_result.sources,
                    }
                elif self.task_context_bank is None:
                    task_tokens = task_tokens_from_context(context, _context_mask, task_dim)
                else:
                    task_tokens, task_context_result = retrieve_task_context(
                        context,
                        _context_mask,
                        self.task_context_bank,
                        output_dim=task_dim,
                        top_k=self.task_context_top_k,
                    )
                    self._last_task_context_retrieval = {
                        "indices": task_context_result.indices.detach().cpu(),
                        "scores": task_context_result.scores.detach().cpu(),
                        "sources": task_context_result.sources,
                    }
                task_tokens = task_tokens.to(device=self.model.device, dtype=getattr(self.model, "zeva_condition_dtype", self.model.torch_dtype))
                if getattr(self, "_ego_cte", False):
                    phase, current_effect = self._step_ego_cte(observation)
                    trace_phase, trace_effect = phase, current_effect
                    current_effect = current_effect.to(device=self.model.device, dtype=getattr(self.model, "zeva_condition_dtype", self.model.torch_dtype))
                    current_effect_mask = torch.ones(phase.shape[0], dtype=torch.bool, device=self.model.device)
                else:
                    if getattr(self, "_main_observed", False):
                        phase = self._main_runtime.current_phase()
                    else:
                        encoded = self._cte_history.forward()
                        phase = encoded["phase"][:, -1]
                    current_effect, current_effect_mask = None, None
                memory = self.lifecycle.memory_inputs(phase, task_tokens)
                phase = phase.to(device=self.model.device, dtype=getattr(self.model, "zeva_condition_dtype", self.model.torch_dtype))
                memory = {
                    key: value.to(
                        device=self.model.device,
                        dtype=getattr(self.model, "zeva_condition_dtype", self.model.torch_dtype) if value.is_floating_point() else value.dtype,
                    )
                    for key, value in memory.items()
                }
                if self.zeva_mode == "pim_shadow":
                    causal_prompt = torch.zeros(
                        (1, int(self.model.zeva_prompt_encoder.config.hidden_dim)),
                        device=self.model.device,
                        dtype=getattr(self.model, "zeva_condition_dtype", self.model.torch_dtype),
                    )
                    pim_mask = memory["pim_mask"]
                elif self.zeva_mode == "zeva_stage2" or not self.cross_memory_enabled:
                    causal_prompt = torch.zeros(
                        (1, int(self.model.zeva_prompt_encoder.config.hidden_dim)),
                        device=self.model.device,
                        dtype=getattr(self.model, "zeva_condition_dtype", self.model.torch_dtype),
                    )
                    pim_mask = torch.zeros_like(memory["pim_mask"], dtype=torch.bool)
                else:
                    causal_prompt = self.model.zeva_prompt_encoder(**memory)
                    pim_mask = memory["pim_mask"]
                zeva_action_residual = None
                if (
                    self.zeva_mode != "pim_shadow"
                    and getattr(self.model, "zeva_injection_mode", "memory_residual") == "exact_zeva"
                ):
                    adapter = self.model.zeva_behavior_prefix_adapter
                    prior_mean, _prior_std = adapter.prior_for_prompt(
                        task_tokens,
                        phase,
                        memory["bit_effects"],
                        memory["bit_mask"],
                        causal_prompt,
                        pim_mask,
                        enable_pim=self.zeva_mode == "pim_on" and self.cross_memory_enabled,
                    )
                    zeva_action_residual = adapter.action_prior_residual(
                        prior_mean, training=False
                    )
            infer_kwargs = {
                "prompt": None,
                "context": context,
                "context_mask": _context_mask,
                "causal_prompt": causal_prompt,
                "pim_mask": pim_mask,
                "zeva_action_residual": zeva_action_residual,
                "zeva_task_tokens": task_tokens,
                # Legacy reads completed BIT; the new version supplies a
                # separate causal current prediction below.
                "zeva_bit_effects": memory["bit_effects"],
                "zeva_bit_mask": memory["bit_mask"],
                "zeva_mode": self.zeva_mode,
                "enable_prefix_injection": self.zeva_mode in {"zeva_stage2", "pim_on"},
                "enable_action_prior": self.zeva_mode in {"zeva_stage2", "pim_on"},
                "input_image": image_tensor,
                "action_horizon": self.action_horizon,
                "proprio": proprio,
                "negative_prompt": self.negative_prompt,
                "text_cfg_scale": self.text_cfg_scale,
                "num_inference_steps": self.num_inference_steps,
                "sigma_shift": self.sigma_shift,
                "seed": request_seed,
                "rand_device": self.rand_device,
                "tiled": self.tiled,
            }
            infer_kwargs.update(noise_kwargs)
            if current_effect is not None:
                infer_kwargs.update(zeva_current_effect=current_effect, zeva_current_effect_mask=current_effect_mask)
            infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
            with torch.no_grad():
                self._warmup_main_model(self.model.infer_action_zeva, infer_kwargs)
                pred = self.model.infer_action_zeva(**infer_kwargs)
                if (
                    self.zeva_mode == "pim_on" and self.cross_memory_enabled
                    and self.pim_diagnostic_interval
                    and self._replan_index % self.pim_diagnostic_interval == 0
                ):
                    self._measure_pim_effect(
                        infer_kwargs, pred, task_tokens, phase, memory,
                        causal_prompt, pim_mask,
                    )
            if self.timing_enabled:
                self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
            self._last_inference_seed = request_seed
            adapter = self.model.zeva_behavior_prefix_adapter
            gate = (
                float(torch.tanh(adapter.pim_gate.detach()).cpu())
                if getattr(adapter.config, "pim_variant", "legacy") != "cross_attempt" else None
            )
            logger.info(
                "Zeva request | attempt=%d replan=%d seed=%s pim_count=%d gate=%s mode=%s cross_memory=%s",
                self._attempt_id,
                self._replan_index,
                str(request_seed),
                int(pim_mask.sum().item()),
                gate,
                self.zeva_mode,
                self.cross_memory_enabled,
            )
            action_chunk = self._denormalize_action(pred["action"])[0]
            if getattr(self, "_ego_cte", False):
                if action_chunk.shape[0] < self.replan_steps:
                    raise ValueError("Zeva-Ego action prediction is shorter than its H24 execution chunk")
                # The query saw only earlier predictions in BIT/PIM. This
                # boundary enters BIT and private pending PIM after the action.
                self.lifecycle.observe_predicted_effect(trace_phase, trace_effect,
                    metadata={"source_step": self.step_count, "decision_boundary": True})
            self._replan_index += 1
            return action_chunk
        infer_kwargs = {
            "prompt": None,
            "context": context,
            "context_mask": _context_mask,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": request_seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_kwargs.update(noise_kwargs)
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            self._warmup_main_model(self.model.infer_action, infer_kwargs)
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]
        action_chunk = self._denormalize_action(action_tensor)[0]
        self._replan_index += 1
        return action_chunk

    @torch.no_grad()
    def _warmup_main_model(self, infer, infer_kwargs: dict[str, Any]) -> None:
        """Discard one same-seed forward without a control or memory request.

        Conditions and the actual request seed have already been prepared.
        Calling only the frozen model avoids action execution, lifecycle
        writes, replay bookkeeping, or an extra seed/replan counter update.
        The once-only flag deliberately survives episode and attempt resets.
        """
        if not getattr(self, "_main_observed", False) or getattr(self, "_main_warmup_requests", 0):
            return
        if infer_kwargs.get("seed") is None or "initial_action_noise" in infer_kwargs:
            raise ValueError("main warmup requires the actual deterministic request seed")
        discarded = infer(**infer_kwargs)
        del discarded
        self._main_warmup_seed = int(infer_kwargs["seed"])
        self._main_warmup_requests = 1

    @torch.no_grad()
    def _measure_pim_effect(self, infer_kwargs, pred, task_tokens, phase, memory, causal_prompt, pim_mask):
        """Same-state, same-noise probe; consumes no model seed stream draws."""
        adapter = self.model.zeva_behavior_prefix_adapter
        off_kwargs = dict(infer_kwargs)
        off_kwargs["causal_prompt"] = torch.zeros_like(causal_prompt)
        off_kwargs["pim_mask"] = torch.zeros_like(pim_mask)
        # Memory-off changes persistent history only. Keep the observed
        # current-attempt effect context identical in the paired probe.
        off_kwargs["zeva_bit_effects"] = memory["bit_effects"]
        off_kwargs["zeva_bit_mask"] = memory["bit_mask"]
        exact = getattr(self.model, "zeva_injection_mode", "memory_residual") == "exact_zeva"
        if exact:
            off_mean, _ = adapter.prior_for_prompt(
                task_tokens, phase, memory["bit_effects"], memory["bit_mask"],
                off_kwargs["causal_prompt"], off_kwargs["pim_mask"], enable_pim=False,
            )
            off_kwargs["zeva_action_residual"] = adapter.action_prior_residual(off_mean, training=False)
        # Preserve global torch RNG as well as reusing the explicit request
        # seed, so optional diagnostics cannot change subsequent rollout noise.
        device = torch.device(self.model.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            pred_off = self.model.infer_action_zeva(**off_kwargs)
        action_on, action_off = pred["action"].float(), pred_off["action"].float()
        if action_on.ndim == 2:
            action_on = action_on.unsqueeze(0)
        if action_off.ndim == 2:
            action_off = action_off.unsqueeze(0)
        if action_on.shape != action_off.shape or action_on.ndim != 3:
            raise ValueError("PIM action diagnostics require matching [B,T,D] actions")
        n_exec = min(self.replan_steps, action_on.shape[1])
        delta = action_on[:, :n_exec] - action_off[:, :n_exec]
        self._last_pim_diagnostics = {
            "valid_entries": int(pim_mask.sum()),
            "executed_action_steps": n_exec,
            "action_exec_delta_rms": float(delta.square().mean().sqrt()),
            "action_exec_relative_delta": float(delta.norm() / action_off[:, :n_exec].norm().clamp_min(1e-8)),
        }
        if exact:
            layout = getattr(getattr(adapter, "config", None), "prefix_layout", "legacy_merged")
            if layout == "zeva_ego":
                prefix_kwargs = {
                    "bit_effects": memory["bit_effects"],
                    "bit_mask": memory["bit_mask"],
                }
                if "zeva_current_effect" in infer_kwargs:
                    prefix_kwargs.update(current_effect=infer_kwargs["zeva_current_effect"],
                                         current_effect_mask=infer_kwargs["zeva_current_effect_mask"])
                on_prefix, on_valid = adapter.behavior_prefix_tokens(
                    task_tokens, causal_prompt, pim_mask, **prefix_kwargs,
                )
                off_prefix, off_valid = adapter.behavior_prefix_tokens(
                    task_tokens, causal_prompt, pim_mask, enable_pim=False, **prefix_kwargs,
                )
                # Invalid slots have no attention visibility. Mask them in
                # diagnostics too, including a projector's possible bias.
                on_prefix = torch.where(on_valid[..., None], on_prefix, torch.zeros_like(on_prefix))
                off_prefix = torch.where(off_valid[..., None], off_prefix, torch.zeros_like(off_prefix))
                self._last_pim_diagnostics.update({
                    "prefix_layout": layout,
                    "effect_source": adapter.config.effect_source,
                    "prefix_valid_tokens_on": float(on_valid.float().sum(-1).mean()),
                    "prefix_valid_tokens_off": float(off_valid.float().sum(-1).mean()),
                    "effect_token_valid_samples": int(on_valid[:, 1].sum()),
                    "pim_token_valid_samples": int(on_valid[:, 2].sum()),
                })
            else:
                on_prefix = adapter.behavior_prefix_slot(task_tokens, causal_prompt, pim_mask)
                off_prefix = adapter.behavior_prefix_slot(task_tokens, causal_prompt, pim_mask, enable_pim=False)
            prefix_delta = on_prefix.float() - off_prefix.float()
            on_mean, _ = adapter.prior_for_prompt(
                task_tokens, phase, memory["bit_effects"], memory["bit_mask"], causal_prompt, pim_mask,
            )
            self._last_pim_diagnostics.update({
                "prefix_delta_rms": float(prefix_delta.square().mean().sqrt()),
                "prefix_relative_delta": float(prefix_delta.norm() / off_prefix.float().norm().clamp_min(1e-8)),
                "prior_mean_delta_rms": float((on_mean.float() - off_mean.float()).square().mean().sqrt()),
            })
        logger.info("PIM same-noise diagnostics: %s", self._last_pim_diagnostics)

    def evaluation_metadata(self) -> dict[str, Any]:
        pim = self.lifecycle.pim if self.lifecycle is not None else None
        if self._needs_episode_reset:
            # reset() defers task identity until the first real observation;
            # the old episode's store will be cleared before its first read.
            pim = None
        return {
            "evaluation_profile": getattr(self, "zeva_profile", "legacy"),
            "zeva_profile": getattr(self, "zeva_profile", "legacy"),
            "inference_parameters_frozen": self._inference_parameters_frozen(),
            "model_seed": self.seed,
            "seed_stream": (getattr(self, "_main_seed_schedule", None)
                            or getattr(self, "_model_noise_stream", None) or self._model_seed_stream).snapshot(),
            "discarded_warmup_requests": int(getattr(self, "_main_warmup_requests", 0)),
            "discarded_warmup_scope": "policy_instance" if getattr(self, "_main_observed", False) else None,
            "discarded_warmup_inference_seed": getattr(self, "_main_warmup_seed", None),
            "evaluation_slot": {"index": getattr(self, "evaluation_slot_index", 0), "count": getattr(self, "evaluation_slot_count", 1)},
            "memory_semantics": "predicted_effect_trace" if getattr(self, "_ego_cte", False) else "completed_effect_trace",
            "observed_effect_source": "observed_post" if getattr(self, "_main_observed", False) else None,
            "checkpoint_identity": self.checkpoint_identity,
            "training_recipe": self.checkpoint_identity.get("training_recipe", "frozen_parent"),
            "foundation_mode": ("joint" if self.checkpoint_identity.get("training_recipe") in {"ego_joint_parent", "ego_joint_warmup", "main_joint_parent", "main_joint_pim"} else "frozen"),
            "foundation_checkpoint_required": self.checkpoint_identity.get("foundation_checkpoint_required", False),
            "foundation_contract_schema": self.checkpoint_identity.get("foundation_contract_schema"),
            "pim_memory_contract": self.checkpoint_identity.get("pim_memory_contract"),
            "numerical_contract": self.checkpoint_identity.get("numerical_contract"),
            "training_query_scope": self.checkpoint_identity.get("training_query_scope"),
            "inference_task_context_scope": self.checkpoint_identity.get("inference_task_context_scope"),
            "cte_exploratory_epoch40": self.checkpoint_identity.get("cte_exploratory_epoch40"),
            "cte_require_reference_recipe": self.checkpoint_identity.get("cte_require_reference_recipe"),
            "pim_compression_override": self.checkpoint_identity.get("pim_compression_override", False),
            "policy_text_encoder_device": getattr(self, "policy_text_encoder_device", None),
            "policy_text_cache_size": getattr(self, "policy_text_cache_size", None),
            "policy_text_cache": (self._policy_prompt_cache.snapshot()
                                  if getattr(self, "_policy_prompt_cache", None) is not None else None),
            "cuda_memory": _cuda_memory_metadata(getattr(self.model, "device", "cpu")),
            "task_context_mode": getattr(self, "task_context_mode", "static"),
            "task_language_predictions": (getattr(getattr(self, "_task_language_memory", None), "last_diagnostics", [])
                                          if getattr(self, "_task_language_result", None) is not None else []),
            "cte_schema": getattr(self, "cte_schema", None),
            "prior_variant": getattr(getattr(getattr(self.model, "zeva_behavior_prefix_adapter", None), "config", None), "prior_variant", "legacy_bit"),
            **_prefix_metadata(self.model),
            "zeva_mode": self.zeva_mode,
            "cross_memory_enabled": self.cross_memory_enabled,
            "action_horizon": self.action_horizon,
            "replan_steps": self.replan_steps,
            "skip_get_obs_within_replan": bool(getattr(self, "skip_get_obs_within_replan", False)),
            "num_inference_steps": self.num_inference_steps,
            "pim_entries": len(pim) if pim is not None else 0,
            "pim_pending_entries": getattr(pim, "pending_count", 0),
            "pim_source_attempts": sorted({entry.attempt_id for entry in pim.entries}) if pim is not None else [],
            "replans": self._replan_index,
            "terminal_observation_step": self._terminal_observation_step,
            "pim_diagnostics_last": dict(self._last_pim_diagnostics),
            "main_observed_runtime": (self._main_runtime.snapshot() if getattr(self, "_main_runtime", None) is not None else None),
            **(self._success_replay.snapshot() if getattr(self, "_success_replay", None) is not None else {}),
        }

    def _inference_parameters_frozen(self) -> bool | None:
        modules = [self.model]
        if getattr(self, "cte", None) is not None:
            modules.append(self.cte)
        if any(not callable(getattr(module, "parameters", None)) for module in modules):
            return None
        return all(not module.training and not any(p.requires_grad for p in module.parameters()) for module in modules)

    def _ensure_main_runtime(self, observation: Dict[str, Any], instruction: str) -> None:
        if self._main_episode_instruction is not None and instruction != self._main_episode_instruction:
            raise ValueError("main_observed retries must preserve the episode instruction; reset for a new episode")
        if self._main_runtime is not None:
            return
        if self._needs_episode_reset:
            self.lifecycle.reset_episode(instruction, episode_id=f"episode-{self.episode_count}")
            self._needs_episode_reset = False
        self._main_episode_instruction = instruction
        runtime = MainObservedRuntime(self.cte, self.lifecycle, frame_encoder=self._cte_frame_encoder)
        runtime.start(self._build_robotwin_image_tensor(observation)[0].float())
        self._main_runtime = runtime
        self._cte_history = runtime.history

    def save_model_rng_state(self) -> None:
        stream = getattr(self, "_model_noise_stream", None)
        if stream is not None:
            stream.save_state()

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        replayed_steps = 0
        if getattr(self, "_main_observed", False) and self.zeva_mode != "base":
            # Preserve the policy's normal sampling/RNG call even when a later
            # attempt can replay the first successful trace, as original main does.
            action_tensor, replayed_steps = self._success_replay.apply(
                self._main_runtime.current_phase(), torch.as_tensor(action_chunk),
                replan_index=self._replan_index - 1,
                conditioning=self.zeva_mode == "pim_on" and self.cross_memory_enabled,
            )
            action_chunk = action_tensor.numpy()
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))
            if getattr(self, "_main_observed", False) and self.zeva_mode != "base":
                self._pending_replay_actions.append(i < replayed_steps)

    def should_request_observation(self) -> bool:
        if getattr(self, "_main_observed", False) and self.zeva_mode != "base":
            return not self.pending_actions or self._main_runtime is None or self._main_runtime.needs_boundary_observation
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        main_observed = getattr(self, "_main_observed", False) and self.zeva_mode != "base"
        if main_observed:
            if self._attempt_finalized:
                raise RuntimeError("begin a new attempt before executing more controls")
            if self._main_runtime is None:
                if observation is None:
                    raise ValueError("main_observed requires its initial real observation")
                self._ensure_main_runtime(observation, task_env.get_instruction())
            if self._main_runtime.needs_boundary_observation:
                if observation is None:
                    raise ValueError("main_observed requires a fresh observation after each four executed actions")
                self._main_runtime.observe(
                    self._build_robotwin_image_tensor(observation)[0].float(), raw_step=self.step_count,
                )
        if not main_observed and self.zeva_mode != "base" and not getattr(self, "_ego_cte", False) and observation is not None and self._cte_history is not None:
            next_image = self._build_robotwin_image_tensor(observation)[0].float()
            if len(self._transition_actions) >= self.cte.cfg.transition_steps:
                action_group = torch.stack(self._transition_actions[: self.cte.cfg.transition_steps]).to(self.model.device)
                self._cte_history.append_transition(action_group, next_image)
                self._transition_actions = self._transition_actions[self.cte.cfg.transition_steps :]
                encoded = self._cte_history.forward()
                complete = encoded["effect_complete"][0]
                while self._observed_effect_count < complete.shape[0]:
                    effect_index = self._observed_effect_count
                    if bool(complete[effect_index]):
                        start = effect_index * self.cte.cfg.effect_window_transitions
                        self.lifecycle.observe_completed_effect(
                            encoded["phase"][0, start],
                            encoded["effect_post"][0, effect_index],
                            metadata={"source_step": start * self.cte.cfg.transition_steps},
                        )
                    self._observed_effect_count += 1
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions[0]
        normalized_action = None
        if self.zeva_mode != "base":
            raw_state = None if observation is None else np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
            normalized_action = self._normalize_action(action, raw_state)
            if main_observed:
                normalized_action = self._main_runtime.validate_action(normalized_action)
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        self.pending_actions.popleft()
        if self.zeva_mode != "base":
            if main_observed:
                self._main_runtime.record_executed_action(normalized_action)
                self._success_replay.record_executed_action(
                    torch.as_tensor(action), replayed=self._pending_replay_actions.popleft(),
                )
            else:
                self._transition_actions.append(normalized_action)
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        if getattr(self, "_main_seed_schedule", None) is not None:
            self._main_seed_schedule.reset_episode()
        self._main_runtime = None
        self._main_episode_instruction = None
        if getattr(self, "_success_replay", None) is not None:
            self._success_replay.reset_episode()
            self._pending_replay_actions.clear()
        if getattr(self, "_static_task_context_session", None) is not None:
            self._static_task_context_session.reset()
        self._task_language_instruction = self._task_language_result = None
        self._last_task_context_retrieval = None
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()
        self._cte_history = None
        self._cte_state = None
        self._ego_has_started = False
        self._observed_effect_count = 0
        self._transition_actions.clear()
        self._attempt_finalized = False
        self._attempt_id = 0
        self._replan_index = 0
        self._last_inference_seed = self.seed
        self._last_pim_diagnostics = {}
        self._terminal_observation_step = None
        if self.zeva_mode != "base":
            self._needs_episode_reset = True

    def begin_attempt(self, attempt_id: int) -> None:
        """Reset transient CTE/BIT while preserving episode PIM and seed schedule."""
        attempt_id = int(attempt_id)
        if getattr(self, "_main_observed", False) and self.zeva_mode != "base":
            if attempt_id not in {self._attempt_id, self._attempt_id + 1}:
                raise ValueError("main_observed attempt IDs must increment by one")
            if attempt_id == self._attempt_id and self.step_count:
                raise RuntimeError("cannot erase a running main_observed attempt without episode reset")
            if attempt_id == self._attempt_id + 1 and not self._attempt_finalized:
                raise RuntimeError("finalize the previous attempt with its terminal observation before retrying")
            if attempt_id == 0:
                self._success_replay.reset_episode()
                self._main_episode_instruction = None
            else:
                self._success_replay.begin_attempt(attempt_id)
            self._main_runtime = None
            self._pending_replay_actions.clear()
        if getattr(self, "_main_seed_schedule", None) is not None:
            self._main_seed_schedule.begin_attempt(attempt_id)
        if getattr(self, "_static_task_context_session", None) is not None:
            self._static_task_context_session.reset()
        self._task_language_instruction = self._task_language_result = None
        self._last_task_context_retrieval = None
        attempt_id = int(attempt_id)
        if attempt_id < 0:
            raise ValueError("attempt_id must be non-negative")
        self._attempt_id = attempt_id
        self._replan_index = 0
        self._last_inference_seed = None
        self.pending_actions.clear()
        self.step_count = 0
        self.reset_timing_rollout()
        self._last_pim_diagnostics = {}
        self._terminal_observation_step = None
        self._attempt_finalized = False
        self._cte_state = None
        self._ego_has_started = False
        if self.lifecycle is None:
            return
        if attempt_id == 0:
            self._cte_history = None
            self._observed_effect_count = 0
            self._transition_actions.clear()
            self._attempt_finalized = False
            self._needs_episode_reset = True
            return
        if self.lifecycle.pim.task_cluster is not None:
            self.lifecycle.reset_attempt(attempt_id)
        self._cte_history = None
        self._cte_state = None
        self._ego_has_started = False
        self._observed_effect_count = 0
        self._transition_actions.clear()
        self._attempt_finalized = False
        self._needs_episode_reset = self.lifecycle.pim.task_cluster is None

    def finalize_attempt(self, task_env, success: bool = False) -> None:
        """Finalize one normally completed attempt, including failures."""
        if self._attempt_finalized:
            return

        if self.zeva_mode == "base" or self.cte is None or self.lifecycle is None:
            self._transition_actions.clear()
            self._attempt_finalized = True
            return

        if (getattr(self, "_ego_cte", False)
                and self.lifecycle.pim.config.commit_timing == "attempt_reset"):
            # The reference trace contains predictions at actual decision
            # boundaries only. No terminal observation or extra prediction is
            # needed; the next attempt reset publishes this private trace.
            self._transition_actions.clear()
            if self.lifecycle.pim.task_cluster is not None:
                self.lifecycle.end_attempt()
            self._attempt_finalized = True
            return

        try:
            # now_obs can still be the pre-action frame on timeout. Fetch a
            # fresh observation before the environment/video writer is closed.
            getter = getattr(task_env, "get_obs", None)
            if not callable(getter):
                raise RuntimeError("Attempt finalization requires a fresh terminal get_obs()")
            terminal_obs = getter()
            has_terminal_obs = isinstance(terminal_obs, dict) and "observation" in terminal_obs
            if not has_terminal_obs:
                raise RuntimeError("Missing terminal observation; refusing to commit stale PIM")
            self._terminal_observation_step = self.step_count
            if getattr(self, "_main_observed", False):
                if self._main_runtime is None:
                    self._ensure_main_runtime(terminal_obs, task_env.get_instruction())
                self._main_runtime.observe(
                    self._build_robotwin_image_tensor(terminal_obs)[0].float(),
                    raw_step=self.step_count, terminal=True,
                )
                self._success_replay.finalize_attempt(success=success, terminal_observed=True)
                self.lifecycle.end_attempt()
                self.pending_actions.clear()
                self._pending_replay_actions.clear()
                self._transition_actions.clear()
                self._attempt_finalized = True
                return
            has_complete_tail = len(self._transition_actions) == self.cte.cfg.transition_steps

            if not getattr(self, "_ego_cte", False) and has_terminal_obs and has_complete_tail and self._cte_history is not None:
                next_image = self._build_robotwin_image_tensor(terminal_obs)[0].float()
                action_group = torch.stack(self._transition_actions).to(self.model.device)
                self._cte_history.append_transition(action_group, next_image)
                encoded = self._cte_history.forward()
                complete = encoded["effect_complete"][0]
                while self._observed_effect_count < complete.shape[0]:
                    effect_index = self._observed_effect_count
                    if bool(complete[effect_index]):
                        start = effect_index * self.cte.cfg.effect_window_transitions
                        self.lifecycle.observe_completed_effect(
                            encoded["phase"][0, start],
                            encoded["effect_post"][0, effect_index],
                            metadata={
                                "source_step": start * self.cte.cfg.transition_steps,
                                "terminal_attempt": True,
                                "terminal_observation_used": True,
                            },
                        )
                    self._observed_effect_count += 1

            self._transition_actions.clear()
            if self.lifecycle.pim.task_cluster is not None:
                self.lifecycle.end_attempt()
            self._attempt_finalized = True
        except Exception:
            raise


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    runtime_config_path = usr_args.get("runtime_config_path")
    if not _is_none_like(runtime_config_path):
        cfg = OmegaConf.load(str(runtime_config_path))
        if cfg.get("runtime_schema") != "iclwam-robotwin-runtime-v1":
            raise ValueError("Unsupported resolved RoboTwin runtime configuration")
    else:
        # Keep direct plugin/HTTP and old entrypoint calls compatible.
        cfg = _compose_sim_cfg(
            sim_cfg_path=sim_cfg_path,
            sim_cfg_name=sim_cfg_name,
            sim_task=sim_task,
        )
    for key, config_key in (
        ("task_context_bank", "bank_path"),
        ("task_context_retrieval_checkpoint", "retrieval_checkpoint"),
        ("task_language_encoder", "language_encoder_path"),
        ("task_context_top_k", "top_k"),
        ("task_context_mode", "mode"),
    ):
        if not _is_none_like(usr_args.get(key)):
            cfg.model.zeva.task_context[config_key] = usr_args[key]
    # Direct RoboTwin plugin calls can explicitly select the old topology
    # when reusing a merged-prefix checkpoint. Resolved Hydra evaluation
    # configs already carry these fields and need no reconstruction.
    for key in ("prefix_layout", "effect_source"):
        if not _is_none_like(usr_args.get(key)):
            cfg.model.zeva.adapter[key] = str(usr_args[key])

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    if device == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        logger.warning("Using float32 for CPU evaluation because mixed precision=%s is CUDA-oriented.", mixed_precision)
        model_dtype = torch.float32

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("model_seed"))
    if seed is None:
        seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    zeva_mode = str(usr_args.get("zeva_mode", cfg.EVALUATION.get("zeva_mode", "base")))
    cte_checkpoint = usr_args.get("cte_checkpoint", cfg.EVALUATION.get("cte_checkpoint"))
    addon_checkpoint = usr_args.get("addon_checkpoint", cfg.EVALUATION.get("addon_checkpoint"))
    pim_top_k = int(usr_args.get("pim_top_k", cfg.EVALUATION.get("pim_top_k", 4)))
    max_attempts = int(usr_args.get("max_attempts", cfg.EVALUATION.get("max_attempts", 1)))
    skip_observations = _parse_bool(usr_args.get("skip_get_obs_within_replan", cfg.EVALUATION.get("skip_get_obs_within_replan", False)))
    if zeva_mode != "base" and skip_observations and not _supports_boundary_observations(cfg.model, cfg.data.train.processor):
        raise ValueError(
            "Zeva evaluation requires skip_get_obs_within_replan=false unless the model uses "
            "current_prediction with no action-state transforms or stepwise normalization"
        )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        video_size=tuple(int(v) for v in cfg.data.train.video_size),
        zeva_mode=zeva_mode,
        cte_checkpoint=None if _is_none_like(cte_checkpoint) else str(cte_checkpoint),
        addon_checkpoint=None if _is_none_like(addon_checkpoint) else str(addon_checkpoint),
        pim_top_k=pim_top_k,
        max_attempts=max_attempts,
        cross_memory_enabled=_parse_bool(usr_args.get("cross_memory_enabled", cfg.EVALUATION.get("cross_memory_enabled", True))),
        pim_diagnostic_interval=int(usr_args.get("pim_diagnostic_interval", cfg.EVALUATION.get("pim_diagnostic_interval", 0))),
        model_rng_protocol=str(usr_args.get("model_rng_protocol", cfg.EVALUATION.get("model_rng_protocol", "request_seed_stream_v1"))),
        evaluation_slot_index=int(usr_args.get("evaluation_slot_index", 0)),
        evaluation_slot_count=int(usr_args.get("evaluation_slot_count", 1)),
        model_rng_state_path=usr_args.get("model_rng_state_path"),
        model_rng_task_name=usr_args.get("model_rng_task_name"),
        model_rng_run_id=usr_args.get("model_rng_run_id"),
        skip_get_obs_within_replan=skip_observations,
        policy_text_encoder_device=str(usr_args.get("policy_text_encoder_device")
                                       or cfg.EVALUATION.get("policy_text_encoder_device", "cpu")),
        policy_text_cache_size=int(usr_args.get("policy_text_cache_size",
                                               cfg.EVALUATION.get("policy_text_cache_size", 16))),
        pim_success_replay_enabled=_parse_bool(usr_args.get("pim_success_replay_enabled",
                                                         cfg.EVALUATION.get("pim_success_replay_enabled", False))),
        pim_success_replay_phase_threshold=float(usr_args.get("pim_success_replay_phase_threshold",
                                                             cfg.EVALUATION.get("pim_success_replay_phase_threshold", 0.80))),
        pim_success_replay_blend=float(usr_args.get("pim_success_replay_blend",
                                                   cfg.EVALUATION.get("pim_success_replay_blend", 1.0))),
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
