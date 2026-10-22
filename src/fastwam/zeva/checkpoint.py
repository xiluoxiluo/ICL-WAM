"""Separated CTE/addon checkpoints and strict compatibility checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .schemas import CacheManifest
from fastwam.zeva_main_contract import MAIN_ADDON_SCHEMA, MAIN_NUMERICAL_CONTRACT

JOINT_FOUNDATION_SCHEMA = "iclwam_joint_foundation_v1"
FOUNDATION_CONTRACT_SCHEMA = "iclwam_foundation_contract_v1"


def checkpoint_foundation_mode(payload: dict) -> str:
    """Read the explicit mode, retaining recipe-based legacy compatibility."""
    inferred = "joint" if payload.get("training_recipe") in {"ego_joint_parent", "ego_joint_warmup", "main_joint_parent", "main_joint_pim"} else "frozen"
    mode = payload.get("foundation_mode", inferred)
    if mode not in {"frozen", "joint"}:
        raise ValueError("checkpoint foundation_mode must be 'frozen' or 'joint'")
    if mode != inferred:
        raise ValueError("checkpoint foundation_mode and training_recipe disagree")
    precision = payload.get("trainable_precision")
    if precision is not None and precision != "float32":
        raise ValueError("unsupported checkpoint trainable_precision")
    return mode


def checkpoint_requires_foundation(payload: dict) -> bool:
    """Separate current-stage trainability from inherited backbone weights.

    Frozen Stage2B can inherit the trained Stage2A foundation. It must ship
    that backbone even though it never updates it. Historical joint Stage2B
    bundles infer the requirement from their recipe; old frozen addons must
    never silently acquire a sidecar.
    """
    from fastwam.zeva_memory_contract import checkpoint_pim_memory_contract

    memory = checkpoint_pim_memory_contract(payload)
    new_contract = (
        any(name in payload for name in (
            "foundation_contract_schema", "foundation_checkpoint_required",
            "foundation_parent_identity", "pim_memory_contract",
        ))
        or payload.get("training_recipe") == "ego_joint_parent"
        or payload.get("profile") == "main_observed"
        or (memory is not None and (memory.get("compression"), memory.get("uniform_indexing"))
            != ("uniform", "legacy_floor"))
    )
    if new_contract:
        if payload.get("foundation_contract_schema") != FOUNDATION_CONTRACT_SCHEMA:
            raise ValueError("new checkpoint requires a complete versioned foundation contract")
        missing = {"training_stage", "training_recipe", "foundation_mode",
                   "foundation_checkpoint_required"} - payload.keys()
        if missing:
            raise ValueError(f"incomplete foundation contract: missing {sorted(missing)}")
    mode = checkpoint_foundation_mode(payload)
    if payload.get("profile") == "main_observed":
        from fastwam.zeva_main_contract import MAIN_RECIPES
        recipe = payload.get("training_recipe")
        if recipe not in MAIN_RECIPES:
            raise ValueError("main checkpoint requires an explicit main training recipe")
        expected_stage = "policy_injection" if recipe.endswith("parent") else "pim_adapter"
        if payload.get("training_stage") != expected_stage:
            raise ValueError("main checkpoint training stage and recipe disagree")
    required = payload.get("foundation_checkpoint_required", mode == "joint")
    if type(required) is not bool:
        raise ValueError("foundation_checkpoint_required must be boolean")
    if mode == "joint" and not required:
        raise ValueError("joint foundation checkpoint cannot disable its required weights")
    if new_contract and required and not isinstance(payload.get("foundation_checkpoint"), dict):
        raise ValueError("foundation contract is missing its required foundation checkpoint identity")
    if not required:
        if payload.get("foundation_checkpoint") is not None or payload.get("foundation_parent_identity") is not None:
            raise ValueError("foundation checkpoint metadata requires trained or inherited foundation weights")
        return False
    stage, recipe = payload.get("training_stage"), payload.get("training_recipe")
    valid = ((stage == "policy_injection" and recipe == "ego_joint_parent")
             or (stage == "pim_adapter" and recipe in {"ego_joint_warmup", "ego_adapter_warmup"}))
    main = payload.get("profile") == "main_observed"
    if main:
        valid = ((stage == "policy_injection" and recipe == "main_joint_parent")
                 or (stage == "pim_adapter" and recipe in {"main_joint_pim", "main_frozen_pim"}))
    expected_schema = MAIN_ADDON_SCHEMA if main else "zeva_ego_prediction_v1"
    if payload.get("addon_schema") != expected_schema or not valid:
        raise ValueError("foundation bundle requires a predictive joint Stage2A/Stage2B or inherited frozen Stage2B")
    if mode == "frozen":
        parent = payload.get("foundation_parent_identity")
        if (not isinstance(parent, dict) or parent.get("training_stage") != "policy_injection"
                or parent.get("training_recipe") != ("main_joint_parent" if main else "ego_joint_parent")):
            raise ValueError("frozen foundation bundle requires its joint Stage2A parent identity")
        for name in ("addon_sha256", "foundation_sha256"):
            digest = parent.get(name)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"foundation parent requires a valid {name}")
    return True


def resolve_joint_foundation_checkpoint(
    addon_path: str | Path, payload: dict | None = None, *, verify_hash: bool = True,
) -> Path | None:
    """Resolve required sibling weights, including a frozen inherited parent.

    Paths are bundle-relative basenames, so copying the pair to another host
    preserves its identity. A joint addon can never silently use the original
    base instead of its updated policy backbone.
    """
    addon_path = Path(addon_path)
    if payload is None:
        payload = torch.load(addon_path, map_location="cpu", weights_only=False)
    identity = payload.get("foundation_checkpoint")
    if not checkpoint_requires_foundation(payload):
        return None
    if not isinstance(identity, dict) or identity.get("schema") != JOINT_FOUNDATION_SCHEMA:
        raise ValueError("joint addon is missing its required foundation checkpoint identity")
    filename = identity.get("filename")
    if (not isinstance(filename, str) or not filename or Path(filename).name != filename
            or filename in {".", ".."} or "\\" in filename):
        raise ValueError("foundation checkpoint filename must be a bundle-relative basename")
    if identity.get("step") != payload.get("step"):
        raise ValueError("foundation checkpoint/addon step mismatch")
    if identity.get("base_checkpoint_sha256") != payload.get("base_checkpoint_sha256"):
        raise ValueError("foundation checkpoint/addon base parent mismatch")
    digest = identity.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("foundation checkpoint requires a SHA256 identity")
    path = addon_path.parent / filename
    if path == addon_path or not path.is_file():
        raise FileNotFoundError(f"joint addon requires its foundation checkpoint: {path}")
    if verify_hash and checkpoint_sha256(path) != digest:
        raise ValueError("foundation checkpoint SHA256 mismatch")
    return path


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_cte_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    config: dict | None = None,
    camera_keys: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist"),
    cte_input_type: str | None = None,
    vae_metadata: dict | None = None,
    cte_vae_input_size: tuple[int, int] | list[int] | None = None,
) -> None:
    model_config = dict(model.cfg.to_dict())
    if config:
        # Keep the resolved training namespace, while ensuring the constructor
        # shape and input contract are available at the top level for cache
        # builders that do not have to instantiate the whole training config.
        model_config.update(config)
    configured_input_type = cte_input_type or (config or {}).get("cte_input_type") or (config or {}).get("input_type")
    if configured_input_type is None and isinstance(config, dict):
        configured_input_type = (config.get("cte") or {}).get("input_type")
    configured_input_type = str(configured_input_type or "rgb_frame")
    if configured_input_type not in {"rgb_frame", "wan_vae_latent"}:
        raise ValueError("cte_input_type must be 'rgb_frame' or 'wan_vae_latent'")
    payload = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "step": int(step),
        "config": model_config,
        # Match Zeva's released loader naming while retaining the local
        # ``config`` field used by older ICL-WAM checkpoints.
        "model_config": dict(model.cfg.to_dict()),
        "action_dim": int(model.cfg.action_dim),
        "image_channels": int(model.cfg.image_channels),
        # Keep the upstream representation explicit so an RGB checkpoint and
        # a Wan-latent checkpoint cannot be mixed by accident.
        "cte_input_type": configured_input_type,
        "camera_keys": list(camera_keys),
    }
    if configured_input_type == "wan_vae_latent":
        if cte_vae_input_size is None or len(cte_vae_input_size) != 2:
            raise ValueError(
                "wan_vae_latent checkpoints must record cte_vae_input_size as [H, W]"
            )
        cte_vae_input_size = tuple(int(value) for value in cte_vae_input_size)
        if min(cte_vae_input_size) < 1:
            raise ValueError("cte_vae_input_size must contain positive dimensions")
        required_vae_metadata = {
            "model_id",
            "vae_path",
            "z_dim",
            "temporal_downsample_factor",
            "upsampling_factor",
        }
        if not vae_metadata or not required_vae_metadata.issubset(vae_metadata):
            raise ValueError(
                "wan_vae_latent checkpoints must record complete VAE identity "
                f"metadata: {sorted(required_vae_metadata)}"
            )
        payload["latent_channels"] = int(model.cfg.image_channels)
        payload["vae_metadata"] = dict(vae_metadata or {})
        payload["cte_vae_input_size"] = list(cte_vae_input_size)
    from .main_data import inherit_main_artifact
    payload.update(inherit_main_artifact(config or {}))
    if payload.get("profile") == "main_observed":
        from .cte_windows import validate_main_cte_window_semantics
        payload["cte_window_semantics"] = (config or {}).get("cte_window_semantics")
        validate_main_cte_window_semantics(payload)
    torch.save(payload, path)


def load_cte_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    from .main_data import MAIN_PROFILE, validate_main_artifact
    if payload.get("profile") == MAIN_PROFILE:
        validate_main_artifact(payload)
        from .cte_windows import validate_main_cte_window_semantics
        validate_main_cte_window_semantics(payload)
    model.load_state_dict(payload["model"], strict=True)
    model.zeva_main_features = payload.get("profile") == MAIN_PROFILE
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def addon_layout_metadata(behavior_prefix_adapter) -> dict[str, str | None]:
    """Describe the policy-side topology independently of the PIM reader.

    Missing layout metadata in historical checkpoints means ``legacy_merged``.
    A new effect projector must never be silently attached to those weights at
    deployment, or frozen as an allegedly trained Stage2A parent.
    """
    config = getattr(behavior_prefix_adapter, "config", None)
    variant = str(getattr(config, "pim_variant", "legacy"))
    layout = str(getattr(config, "prefix_layout", "legacy_merged"))
    if variant not in {"legacy", "cross_attempt"}:
        raise ValueError(f"unsupported addon pim_variant: {variant!r}")
    if layout not in {"legacy_merged", "zeva_ego"}:
        raise ValueError(f"unsupported addon prefix_layout: {layout!r}")
    effect_source = None
    prior_variant = str(getattr(config, "prior_variant", "legacy_bit"))
    if layout == "zeva_ego":
        effect_source = str(getattr(config, "effect_source", "latest_completed"))
        if effect_source not in {"latest_completed", "current_prediction"}:
            raise ValueError("unsupported zeva_ego effect_source")
    predictive = effect_source == "current_prediction"
    main = getattr(config, "numerical_contract", None) == MAIN_NUMERICAL_CONTRACT
    if predictive != (prior_variant == "zeva_ego"):
        raise ValueError("predictive CTE and zeva_ego action prior must be used together")
    return {
        "pim_variant": variant,
        "prefix_layout": layout,
        "effect_source": effect_source,
        **({"profile": "main_observed", "effect_source": "latest_completed",
            "prior_variant": "legacy_bit", "memory_semantics": "completed_effect_trace",
            "numerical_contract": MAIN_NUMERICAL_CONTRACT} if main else {}),
        **({"prior_variant": prior_variant, "memory_semantics": "predicted_effect_trace",
            "numerical_contract": getattr(config, "numerical_contract", "legacy_v1")} if predictive else {}),
        "addon_schema": (
            MAIN_ADDON_SCHEMA if main else "zeva_ego_prediction_v1" if predictive
            else "zeva_ego_prefix_v1" if layout == "zeva_ego"
            else "cross_attempt_v1" if variant == "cross_attempt"
            else "legacy"
        ),
    }


def _checkpoint_prefix_layout(payload: dict) -> str:
    layout = payload.get("prefix_layout", "legacy_merged")
    if layout not in {"legacy_merged", "zeva_ego"}:
        raise ValueError(f"unsupported checkpoint prefix_layout: {layout!r}")
    if layout == "zeva_ego":
        schema = payload.get("addon_schema")
        if schema not in {"zeva_ego_prefix_v1", "zeva_ego_prediction_v1"}:
            raise ValueError("zeva_ego checkpoint requires a zeva_ego_prefix_v1 or zeva_ego_prediction_v1 schema marker")
        expected_source = "current_prediction" if schema == "zeva_ego_prediction_v1" else "latest_completed"
        if payload.get("effect_source") != expected_source:
            raise ValueError(f"zeva_ego checkpoint effect_source must be {expected_source!r}")
        if schema == "zeva_ego_prediction_v1" and (
            payload.get("prior_variant") != "zeva_ego"
            or payload.get("memory_semantics") != "predicted_effect_trace"
        ):
            raise ValueError("predictive checkpoint is missing its prior/memory semantics")
    elif payload.get("addon_schema") in {"zeva_ego_prefix_v1", "zeva_ego_prediction_v1"}:
        raise ValueError("zeva_ego checkpoint must declare prefix_layout='zeva_ego'")
    elif payload.get("pim_variant", "legacy") == "cross_attempt":
        if payload.get("addon_schema") != "cross_attempt_v1":
            raise ValueError("cross_attempt checkpoint is missing its cross_attempt_v1 schema marker")
    return layout


def save_addon_checkpoint(
    path: str | Path,
    causal_prompt_encoder,
    behavior_prefix_adapter,
    step: int,
    config: dict,
    base_checkpoint_sha256: str,
    cte_checkpoint_sha256: str,
    training_stage: str = "policy_injection",
) -> None:
    torch.save(
        {
            "causal_prompt_encoder": causal_prompt_encoder.state_dict(),
            "behavior_prefix_adapter": behavior_prefix_adapter.state_dict(),
            "pim_gate": behavior_prefix_adapter.pim_gate.detach().cpu(),
            **addon_layout_metadata(behavior_prefix_adapter),
            "training_stage": str(training_stage),
            "training_recipe": "frozen_parent",
            "foundation_contract_schema": FOUNDATION_CONTRACT_SCHEMA,
            "foundation_mode": "frozen",
            "foundation_checkpoint_required": False,
            "step": int(step),
            "config": config,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "cte_checkpoint_sha256": cte_checkpoint_sha256,
        },
        path,
    )


def load_addon_checkpoint(
    path: str | Path,
    causal_prompt_encoder,
    behavior_prefix_adapter,
    *,
    base_checkpoint_sha256: str | None = None,
    cte_checkpoint_sha256: str | None = None,
    task_context_identity: dict | None = None,
    map_location: str = "cpu",
    load_scope: str = "all",
    target_training_stage: str | None = None,
    target_training_recipe: str | None = None,
    foundation_loader=None,
    pim_memory_contract: dict | None = None,
    allow_pim_compression_override: bool = False,
    loaded_foundation_identity: dict | None = None,
    main_artifact_metadata: dict | None = None,
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    foundation_path = resolve_joint_foundation_checkpoint(path, payload)
    if foundation_path is None and loaded_foundation_identity is not None:
        raise ValueError("load the original FastWAM base before loading an addon that does not require trained foundation weights")
    if load_scope == "all" and target_training_recipe is not None:
        target_joint = target_training_recipe in {"ego_joint_parent", "ego_joint_warmup", "main_joint_parent", "main_joint_pim"}
        checkpoint_joint = checkpoint_foundation_mode(payload) == "joint"
        if target_joint != checkpoint_joint:
            raise ValueError("joint/frozen foundation recipe mismatch; use the matching complete checkpoint bundle")
    if load_scope == "all" and target_training_stage is not None and payload.get("training_stage") != target_training_stage:
        raise ValueError("addon checkpoint training stage mismatch; initialize Stage2B with load_scope='policy_injection'")
    if load_scope == "all" and pim_memory_contract is not None:
        from fastwam.zeva_memory_contract import validate_checkpoint_pim_memory_contract
        validate_checkpoint_pim_memory_contract(
            payload, pim_memory_contract, allow_compression_override=allow_pim_compression_override,
        )
    if foundation_path is not None and foundation_loader is None:
        raise ValueError("joint addon requires the complete FastWAM foundation loader")
    if payload.get("task_context_identity") != task_context_identity:
        raise ValueError("addon checkpoint task-context artifacts or retrieval settings mismatch")
    for name, expected in (("base_checkpoint_sha256", base_checkpoint_sha256), ("cte_checkpoint_sha256", cte_checkpoint_sha256)):
        if expected is not None and payload.get(name) != expected:
            raise ValueError(f"addon checkpoint {name} mismatch")
    expected_layout = addon_layout_metadata(behavior_prefix_adapter)
    if expected_layout.get("profile") == "main_observed":
        from .main_data import validate_main_artifact
        if main_artifact_metadata is None:
            raise ValueError("main addon loading requires the target CTE success-data and episode-split metadata")
        validate_main_artifact(payload, expected=main_artifact_metadata)
        for name, expected in expected_layout.items():
            if payload.get(name) != expected:
                raise ValueError(f"main_observed addon contract mismatch: {name}; train the observed-effect main recipe")
    elif payload.get("profile") == "main_observed":
        raise ValueError("main_observed addon cannot be loaded into a different policy profile")
    from fastwam.zeva_numerical_contract import validate_checkpoint_numerical_contract
    expected_numerics = expected_layout.get("numerical_contract", "legacy_v1")
    if getattr(getattr(causal_prompt_encoder, "config", None), "numerical_contract", "legacy_v1") != expected_numerics:
        raise ValueError("prompt and adapter numerical_contract must agree")
    validate_checkpoint_numerical_contract(payload, expected_numerics)
    checkpoint_layout = _checkpoint_prefix_layout(payload)
    if payload.get("prior_variant", "legacy_bit") != expected_layout.get("prior_variant", "legacy_bit"):
        raise ValueError("addon checkpoint prior_variant mismatch; train a new predictive Stage2A")
    if checkpoint_layout == "zeva_ego" and expected_layout["prefix_layout"] == "zeva_ego":
        if payload.get("effect_source") != expected_layout["effect_source"]:
            raise ValueError("addon checkpoint effect_source mismatch; rebuild CTE, caches and Stage2A")
    if load_scope in {"all", "policy_injection"} and checkpoint_layout != expected_layout["prefix_layout"]:
        raise ValueError(
            "addon checkpoint prefix_layout mismatch: "
            f"{checkpoint_layout} vs {expected_layout['prefix_layout']}; "
            "warmstart a new zeva_ego Stage2A with policy_warmstart and train "
            "its effect projector before initializing Stage2B"
        )
    if load_scope == "all":
        expected_variant = getattr(getattr(behavior_prefix_adapter, "config", None), "pim_variant", "legacy")
        checkpoint_variant = payload.get("pim_variant", "legacy")
        if checkpoint_variant != expected_variant:
            raise ValueError(
                f"addon checkpoint pim_variant mismatch: {checkpoint_variant} vs {expected_variant}; "
                "initialize the new variant with load_scope='policy_injection' from a Stage2A checkpoint"
            )
        causal_prompt_encoder.load_state_dict(
            payload["causal_prompt_encoder"], strict=True
        )
        behavior_prefix_adapter.load_state_dict(
            payload["behavior_prefix_adapter"], strict=True
        )
    elif load_scope in {"policy_injection", "policy_warmstart"}:
        if payload.get("training_stage") != "policy_injection":
            raise ValueError(f"{load_scope} initialization requires a Stage2A policy_injection checkpoint")
        if load_scope == "policy_warmstart":
            if target_training_stage != "policy_injection":
                raise ValueError("policy_warmstart is only valid when training a new Stage2A policy_injection model")
            if checkpoint_layout != "legacy_merged" or expected_layout["prefix_layout"] != "zeva_ego":
                raise ValueError("policy_warmstart only supports legacy_merged Stage2A -> zeva_ego Stage2A")
        adapter_state = payload["behavior_prefix_adapter"]
        keep_prefixes = (
            "prior.",
            "action_prior_adapter.",
            "behavior_global_projector.",
        )
        if load_scope == "policy_injection" and expected_layout["prefix_layout"] == "zeva_ego":
            keep_prefixes += ("behavior_effect_projector.",)
        policy_state = {
            key: value
            for key, value in adapter_state.items()
            if key.startswith(keep_prefixes)
        }
        expected_policy_keys = {
            key for key in behavior_prefix_adapter.state_dict() if key.startswith(keep_prefixes)
        }
        if set(policy_state) != expected_policy_keys:
            raise ValueError(
                "incomplete policy-injection checkpoint: "
                f"missing={sorted(expected_policy_keys - set(policy_state))}, "
                f"unexpected={sorted(set(policy_state) - expected_policy_keys)}"
            )
        _missing, unexpected = behavior_prefix_adapter.load_state_dict(
            policy_state, strict=False
        )
        if unexpected:
            raise ValueError(
                f"unexpected policy-injection checkpoint keys: {unexpected}"
            )
    else:
        raise ValueError("load_scope must be 'all', 'policy_injection', or 'policy_warmstart'")
    if foundation_path is not None:
        foundation_loader(foundation_path, payload)
    return payload
