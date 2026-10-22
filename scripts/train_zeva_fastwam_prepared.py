"""Formal Stage-2 Zeva/FastWAM training from prepared offline conditioning.

This entrypoint preserves the model/trainer semantics of train_zeva_fastwam.py
but replaces expensive per-sample Python PIM retrieval and per-startup static
retrieval with immutable, identity-checked prepared artifacts.

For ``pim_adapter`` training the prepared tensors are wrapped with Zeva's
context/support dropout.  Keeping this here, at the formal training entrypoint,
prevents the YAML knobs from becoming dead configuration again.
"""

from __future__ import annotations

from pathlib import Path
import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.runtime import _mixed_precision_to_model_dtype
from fastwam.trainer import Wan22Trainer
from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.pim_training import (
    PIMTrainingDropoutConfig,
    PIMTrainingDropoutDataset,
    apply_pim_training_dropout,
)
from fastwam.zeva.prepared_stage2 import (
    PreparedPIMCache,
    PreparedZevaStage2Dataset,
    StaticEpisodeContextCache,
    validate_static_artifact_hashes,
)
from fastwam.zeva.schemas import sha256_file
from fastwam.zeva.static_task_context import task_context_identity
from fastwam.zeva.cross_attempt_cache import (CROSS_ATTEMPT_CACHE_FORMAT, EGO_CROSS_ATTEMPT_CACHE_FORMAT,
                                            LEGACY_EGO_CROSS_ATTEMPT_CACHE_FORMAT, LEGACY_REFERENCE_EGO_CROSS_ATTEMPT_CACHE_FORMAT,
                                            ALL_CTE_FRAMES, COMPLETE_POLICY_WINDOWS, LEGACY_PAIRING_METRIC, HISTORY_KINDS)
from fastwam.zeva.ego_boundary_cache import EgoBoundaryCache
from fastwam.zeva.ego_cte_training import require_ego_checkpoint_ready
from fastwam.zeva.ego_task_memory import (
    decision_task_instructions, episode_task_instructions, validate_ego_task_language_static_context,
)


def _require_file(value, name: str) -> Path:
    if value in (None, "", "None", "null"):
        raise ValueError(f"Set {name}")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    return path


def _resolve_process_device() -> torch.device:
    """Bind torchrun/Accelerate workers before a multi-billion-param model is built."""
    if not torch.cuda.is_available():
        raise RuntimeError("Zeva Stage-2 requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= count:
        raise ValueError(f"LOCAL_RANK={local_rank} outside visible CUDA device count={count}")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def _dropout_config(cfg: DictConfig) -> PIMTrainingDropoutConfig:
    prompt = cfg.model.zeva.prompt
    return PIMTrainingDropoutConfig(
        context_dropout=float(prompt.get("pim_context_dropout", 0.2)),
        support_dropout=float(prompt.get("pim_support_dropout", 0.2)),
        seed=int(cfg.seed),
    )


def _print_pim_dropout_probe(
    pim_cache: PreparedPIMCache,
    config: PIMTrainingDropoutConfig,
    *,
    count: int = 256,
) -> None:
    """Verify empty/sparse/full occupancy without decoding any training video."""
    probe = min(int(count), len(pim_cache))
    if probe <= 0:
        raise ValueError("Stage-2B prepared PIM cache is empty")
    histogram: dict[int, int] = {}
    context_drops = 0
    for index in range(probe):
        prepared = pim_cache.get(index)
        result = apply_pim_training_dropout(
            phases=prepared["pim_phases"],
            effects=prepared["pim_effects"],
            valid=prepared["pim_mask"],
            scores=prepared["pim_scores"],
            sample_index=index,
            config=config,
        )
        after = int(result["count_after"].item())
        histogram[after] = histogram.get(after, 0) + 1
        context_drops += int(bool(result["context_dropped"].item()))
    print(
        f"PIM dropout probe ({probe} prepared queries): "
        f"counts={dict(sorted(histogram.items()))} context_drops={context_drops}"
    )
    if len(histogram) < 2:
        raise RuntimeError(
            "PIM dropout preflight produced only one occupancy value. "
            "Stage2B would not see the empty/sparse/full memory distribution required by Zeva."
        )


def _print_cross_attempt_probe(pim_cache: PreparedPIMCache, *, seed: int, count: int = 256) -> None:
    """Report actual mixture/capacity without changing training RNG state."""
    kinds = {kind: 0 for kind in HISTORY_KINDS}
    history_counts: dict[int, int] = {}
    valid_counts: dict[int, int] = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for index in range(min(count, len(pim_cache))):
            row = pim_cache.get(index)
            kinds[HISTORY_KINDS[int(row["pim_history_kind"])]] += 1
            histories, entries = int(row["pim_history_count"]), int(row["pim_mask"].sum())
            history_counts[histories] = history_counts.get(histories, 0) + 1
            valid_counts[entries] = valid_counts.get(entries, 0) + 1
            if pim_cache.manifest["split"] == "train" and entries == 0:
                raise ValueError("cross-attempt frozen-parent training produced an empty-history sample")
    print(f"Cross-attempt probe: kind_counts={kinds} history_counts={history_counts} valid_entry_counts={valid_counts}")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Zeva Stage-2 requires CUDA")
    device = _resolve_process_device()
    if not bool(cfg.data.train.get("use_text_embed_cache", False)):
        raise ValueError("Stage-2 requires data.train.use_text_embed_cache=true")
    if int(cfg.data.train.get("global_sample_stride", -1)) != 1:
        raise ValueError("Stage-2 requires global_sample_stride=1")
    if int(cfg.data.train.get("action_video_freq_ratio", -1)) != 4:
        raise ValueError("Stage-2 requires action_video_freq_ratio=4")

    zeva = cfg.model.zeva
    predictive = str(zeva.adapter.get("effect_source", "latest_completed")) == "current_prediction"
    cross_attempt = str(zeva.prompt.get("pim_variant", "legacy")) == "cross_attempt"
    if cross_attempt != (str(zeva.adapter.get("pim_variant", "legacy")) == "cross_attempt"):
        raise ValueError("prompt and adapter PIM variants must agree")
    training_stage = str(zeva.training_stage)
    if training_stage not in {"policy_injection", "pim_adapter"}:
        raise ValueError(f"unsupported Zeva Stage-2 training_stage={training_stage!r}")
    task_mode = str(zeva.task_context.mode)
    task_language = task_mode == "task_language"
    if task_mode not in {"static", "task_language"}:
        raise ValueError("formal prepared Stage-2 requires static or task_language task context")
    if task_language and not predictive:
        raise ValueError("task-language retrieval requires the predictive Zeva-Ego CTE")
    if str(zeva.get("training_recipe")) == "ego_joint_warmup" and not task_language:
        raise ValueError("Joint foundation training requires independent frozen task-language retrieval")
    retrieval_mode = "cross_attempt" if cross_attempt else "phase"
    if str(zeva.memory.pim_retrieval_mode) != retrieval_mode:
        raise ValueError(f"prepared Stage-2 variant requires pim_retrieval_mode={retrieval_mode}")
    if cross_attempt and training_stage != "pim_adapter" and not predictive:
        raise ValueError("cross-attempt variant is initialized from an existing Stage2A parent")
    if predictive:
        if len(cfg.data.train.dataset_dirs) != 1:
            raise ValueError("predictive RoboTwin pipeline currently requires one dataset root")
        if not cross_attempt or str(zeva.adapter.get("prior_variant")) != "zeva_ego":
            raise ValueError("predictive Stage2 requires the Zeva-Ego prior and cross-attempt reader")
        if str(zeva.memory.get("memory_semantics")) != "predicted_effect_trace":
            raise ValueError("predictive Stage2 requires predicted_effect_trace memory")
        if training_stage == "pim_adapter" and str(zeva.get("training_recipe")) not in {"ego_adapter_warmup", "ego_joint_warmup"}:
            raise ValueError("predictive Stage2B requires an explicit ego_adapter_warmup or ego_joint_warmup recipe")

    base_ckpt = _require_file(cfg.get("ckpt"), "ckpt")
    cte_ckpt = _require_file(zeva.cte.checkpoint, "model.zeva.cte.checkpoint")
    bank_path = _require_file(zeva.task_context.bank_path, "model.zeva.task_context.bank_path")
    if task_language:
        encoder_path = _require_file(zeva.task_context.get("language_encoder_path"),
                                      "model.zeva.task_context.language_encoder_path")
        feature_path = _require_file(zeva.task_context.get("language_feature_cache_path"),
                                      "model.zeva.task_context.language_feature_cache_path")
        if float(zeva.task_context.temperature) != 1.0:
            raise ValueError("Task-scoped language retrieval uses reference softmax temperature=1.0")
    else:
        readout_path = _require_file(
            zeva.task_context.readout_cache_path,
            "model.zeva.task_context.readout_cache_path",
        )
    head_path = _require_file(
        zeva.task_context.retrieval_checkpoint,
        "model.zeva.task_context.retrieval_checkpoint",
    )
    static_path = _require_file(
        zeva.task_context.static_episode_cache_path,
        "model.zeva.task_context.static_episode_cache_path",
    )
    pim_root = Path(str(zeva.prepared_stage2.pim_cache_path)).expanduser().resolve()
    if not pim_root.is_dir():
        raise FileNotFoundError(f"missing model.zeva.prepared_stage2.pim_cache_path: {pim_root}")
    phase_root = Path(str(zeva.cache.path)).expanduser().resolve()
    if not phase_root.is_dir():
        raise FileNotFoundError(f"missing validated model.zeva.cache.path: {phase_root}")
    phase_manifest_path = phase_root / "manifest.json"
    phase_index_path = phase_root / "episode_index.json"
    if not phase_manifest_path.is_file() or not phase_index_path.is_file():
        raise FileNotFoundError(f"incomplete phase/effect cache: {phase_root}")

    stats_path = _require_file(cfg.data.train.pretrained_norm_stats, "data.train.pretrained_norm_stats")
    semantic_path = _require_file(cfg.data.train.semantic_task_map_path, "data.train.semantic_task_map_path")
    base_sha = checkpoint_sha256(base_ckpt)
    cte_sha = checkpoint_sha256(cte_ckpt)
    stats_sha = sha256_file(stats_path)
    semantic_sha = sha256_file(semantic_path)
    top_k = int(zeva.memory.pim_top_k)
    bit_size = int(zeva.memory.bit_size)
    if top_k != int(zeva.prompt.persistent_length):
        raise ValueError("PIM top_k must match prompt persistent_length")
    if bit_size != int(zeva.prompt.brief_length):
        raise ValueError("BIT size must match prompt brief_length")

    cte_payload = torch.load(cte_ckpt, map_location="cpu", weights_only=False)
    video_size = [int(v) for v in cfg.data.train.video_size]
    boundary = None
    if predictive:
        require_ego_checkpoint_ready(
            cte_payload, require_reference_recipe=bool(zeva.cte.get("require_reference_recipe", True)),
            exploratory_epoch40=bool(zeva.cte.get("exploratory_epoch40", False)),
        )
        if cte_payload["semantic_task_map_sha256"] != semantic_sha or cte_payload["dataset_stats_sha256"] != stats_sha:
            raise ValueError("predictive CTE semantic-task map or joint normalization mismatch")
        boundary = EgoBoundaryCache(phase_root)
        for key, expected in (("cte_checkpoint_sha256", cte_sha), ("dataset_stats_sha256", stats_sha),
                              ("semantic_task_map_sha256", semantic_sha),
                              ("phase_dim", int(zeva.prompt.phase_dim)), ("effect_dim", int(zeva.prompt.effect_dim)),
                              ("dataset_path", str(Path(cfg.data.train.dataset_dirs[0]).expanduser().resolve()))):
            if getattr(boundary.manifest, key) != expected:
                raise ValueError(f"predictive boundary cache identity mismatch: {key}")
        if boundary.splits != cte_payload["episode_splits"]:
            raise ValueError("boundary cache must reuse the CTE episode train/validation split")
        if bool(zeva.cte.get("require_reference_recipe", True)):
            if boundary.manifest.data_scope != cte_payload["manifest"]["data_scope"]:
                raise ValueError("boundary cache must preserve the reference CTE source scope")
            if boundary.manifest.export_tf32 is not False:
                raise ValueError("reference CTE boundary export must explicitly disable TF32")
        if float(cfg.data.train.get("val_set_proportion", -1)) != 0:
            raise ValueError("explicit predictive episode splits require data.train.val_set_proportion=0")
    else:
        cte_identity = cte_payload.get("semantic_task_identity") or {}
        if str(cte_identity.get("sha256")) != semantic_sha:
            raise ValueError("Stage-2 semantic map differs from CTE checkpoint")
        if list(cte_payload.get("cte_vae_input_size", ())) != video_size:
            raise ValueError("Stage-2 video size differs from CTE VAE input size")
    if tuple(cte_payload.get("camera_keys", ())) != ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        raise ValueError("Stage-2 camera order differs from CTE checkpoint")

    pim_expected = {
            "retrieval_mode": retrieval_mode,
            "history_semantics": "full_episode_prefix",
            "query_step_unit": "raw_action_step",
            "phase_dim": int(zeva.prompt.phase_dim),
            "effect_dim": int(zeva.prompt.effect_dim),
            "top_k": top_k,
            "bit_size": bit_size,
            "cte_checkpoint_sha256": cte_sha,
            "dataset_stats_sha256": stats_sha,
            "dataset_path": str(Path(cfg.data.train.dataset_dirs[0]).expanduser().resolve()) if predictive else str(cfg.data.train.dataset_dirs[0]),
            "semantic_task_map_sha256": semantic_sha,
            "phase_cache_manifest_sha256": sha256_file(phase_manifest_path),
            "phase_cache_index_sha256": sha256_file(phase_index_path),
            "feature_dtype": "float32",
        }
    if cross_attempt:
        if not bool(zeva.memory.get("cross_attempt_only", False)) or int(zeva.memory.pim_max_entries) != top_k:
            raise ValueError("cross-attempt training/serving memory capacity must agree")
        uniform_indexing = str(zeva.memory.get("uniform_indexing", "legacy_floor"))
        query_scope = str(zeva.prepared_stage2.get("query_scope", COMPLETE_POLICY_WINDOWS))
        pairing_metric = str(zeva.prepared_stage2.get("pairing_metric", LEGACY_PAIRING_METRIC))
        predictive_cache_format = (EGO_CROSS_ATTEMPT_CACHE_FORMAT if query_scope == ALL_CTE_FRAMES else
                                   LEGACY_REFERENCE_EGO_CROSS_ATTEMPT_CACHE_FORMAT if uniform_indexing == "reference_round"
                                   else LEGACY_EGO_CROSS_ATTEMPT_CACHE_FORMAT)
        pim_expected.update({
            "format": predictive_cache_format if predictive else CROSS_ATTEMPT_CACHE_FORMAT, "pim_variant": "cross_attempt",
            "source_subset": "train-only", "split": "train",
            "max_past_attempts": int(zeva.memory.max_past_attempts),
            "max_entries_per_attempt": int(zeva.memory.max_entries_per_attempt),
            "history_source_semantics": "other_train_episode",
            "candidate_semantics": "complete_episode_uniform",
        })
        if predictive:
            pim_expected.update({"memory_semantics": "predicted_effect_trace", "effect_source": "current_prediction",
                                 "executed_horizon": 24})
            if query_scope == ALL_CTE_FRAMES:
                pim_expected.update({"query_scope": query_scope, "pairing_metric": pairing_metric,
                                     "data_scope": boundary.manifest.data_scope,
                                     "source_export_scope": boundary.manifest.export_scope})
            if uniform_indexing == "reference_round":
                pim_expected.update({"uniform_indexing": "reference_round", "history_compression": "uniform"})
    pim = PreparedPIMCache(pim_root, expected=pim_expected)
    if predictive and pim._cross_attempt.splits != boundary.splits:
        raise ValueError("prepared PIM cache must use the exact CTE/boundary episode split")
    if predictive:
        pim._cross_attempt.validate_training_curriculum(
            history_pairing=str(zeva.prepared_stage2.get("history_pairing", "legacy_candidates")),
            allow_reduced_curriculum=zeva.prepared_stage2.get("allow_reduced_curriculum", False),
            query_scope=query_scope, pairing_metric=pairing_metric,
        )
    static_context = StaticEpisodeContextCache(static_path)
    static_expected = {
        "cte_checkpoint_sha256": cte_sha,
        "dataset_stats_sha256": stats_sha,
        "semantic_task_map_sha256": semantic_sha,
    }
    if not task_language:
        validate_static_artifact_hashes(
            static_context, bank_path=bank_path, head_path=head_path, readout_path=readout_path,
        )
        static_expected.update({"base_checkpoint_sha256": base_sha, "video_size": video_size,
                                "context_len": int(cfg.model.tokenizer_max_len)})
    if predictive:
        static_expected.update({
            "memory_semantics": "predicted_effect_trace",
            "phase_cache_manifest_sha256": sha256_file(phase_manifest_path),
            "episode_splits_sha256": sha256_file(phase_root / "episode_splits.json"),
        })
    language_queries = None
    if task_language:
        language_queries = validate_ego_task_language_static_context(
            static_context, cache=boundary, bank_path=bank_path, feature_path=feature_path,
            encoder_path=encoder_path, head_path=head_path, expected_metadata=static_expected,
            top_k=int(zeva.task_context.top_k),
            train_query_exclusion=str(zeva.task_context.get("train_query_exclusion", "leave_one_episode_out")),
            training_query_scope=str(zeva.task_context.get("training_query_scope", "episode_static")),
        )
    for key, expected in static_expected.items():
        if static_context.metadata.get(key) != expected:
            raise ValueError(
                f"static context source identity mismatch for {key}: "
                f"expected={expected!r}, got={static_context.metadata.get(key)!r}"
            )
    if int(static_context.metadata.get("top_k", -1)) != int(zeva.task_context.top_k):
        raise ValueError("static context top_k differs from Stage-2 task_context.top_k")
    if static_context.value_dim != int(zeva.prompt.global_dim):
        raise ValueError("static context value dim differs from prompt.global_dim")

    base = instantiate(cfg.data.train)
    if getattr(base, "semantic_task_identity", None) is None:
        raise ValueError("Stage-2 dataset did not load semantic_task_map_path")
    if str(base.semantic_task_identity.get("sha256")) != semantic_sha:
        raise ValueError("Stage-2 dataset semantic identity mismatch")
    if task_language:
        language_query_scope = str(zeva.task_context.get("training_query_scope", "episode_static"))
        current_instructions = episode_task_instructions(base, boundary, require_fixed=language_query_scope == "episode_static")
        cached_instructions = dict(zip(language_queries["episode_ids"], language_queries["instructions"], strict=True))
        if current_instructions != cached_instructions:
            raise ValueError("Stage-2 raw instructions changed after task-language feature construction")
        if language_query_scope == "per_decision" and decision_task_instructions(base, boundary) != static_context.query_records:
            raise ValueError("Stage-2 raw per-decision instructions changed after task-context construction")
    history_schedule = "ego_optimizer_step" if predictive and training_stage == "pim_adapter" else "legacy"
    dataset = PreparedZevaStage2Dataset(base, pim, static_context, history_schedule=history_schedule)
    validation_dataset = None
    validation_path = zeva.prepared_stage2.get("validation_pim_cache_path")
    if validation_path not in (None, "", "None", "null"):
        if not cross_attempt:
            raise ValueError("explicit prepared validation split requires the cross-attempt schema")
        validation_pim = PreparedPIMCache(str(validation_path), expected={**pim_expected, "split": "validation"})
        if predictive:
            validation_pim._cross_attempt.validate_training_curriculum(
                history_pairing=str(zeva.prepared_stage2.get("history_pairing", "legacy_candidates")),
                allow_reduced_curriculum=zeva.prepared_stage2.get("allow_reduced_curriculum", False),
                query_scope=query_scope, pairing_metric=pairing_metric,
            )
        split_hash = pim.manifest["metadata_sha256"]["episode_splits.json"]
        if validation_pim.manifest["metadata_sha256"]["episode_splits.json"] != split_hash:
            raise ValueError("train and validation prepared caches use different episode splits")
        for key in ("use_conditions", "conditions_sha256"):
            if validation_pim.manifest[key] != pim.manifest[key]:
                raise ValueError(f"train and validation history conditions differ: {key}")
        if set(pim.episode_ids) & set(validation_pim.episode_ids):
            raise ValueError("train and validation prepared episodes overlap")
        validation_dataset = PreparedZevaStage2Dataset(base, validation_pim, static_context, history_schedule=history_schedule)
        if predictive and history_schedule == "ego_optimizer_step":
            if not set(pim.manifest["globally_available_history_kinds"]).issubset(validation_pim.manifest["globally_available_history_kinds"]):
                raise ValueError("validation queries lack a training history condition; rebuild condition pairings")
    elif predictive:
        raise ValueError("predictive Stage2 requires an independent prepared validation cache")

    dropout = None
    if training_stage == "pim_adapter":
        dropout = _dropout_config(cfg)
        if cross_attempt:
            if dropout.context_dropout != 0.0 or dropout.support_dropout != 0.0:
                raise ValueError("cross-attempt cache owns history sampling; disable stacked prompt/context/support dropout")
            if not predictive and pim.manifest["history_weights"].get("empty", 0.0) != 0.0:
                raise ValueError("frozen-parent training requires nonempty histories")
            if predictive:
                label = "explicit reduced ablation" if pim._cross_attempt.reduced_curriculum else "reference full 5:2:1 then 5:2:1:2"
                print(f"History curriculum: {label}; pairing={pim._cross_attempt.history_pairing}; available modes:",
                      pim.manifest["globally_available_history_kinds"])
            elif int(os.environ.get("RANK", "0")) == 0:
                _print_cross_attempt_probe(pim, seed=int(cfg.seed))
            dropout = None
        elif dropout.context_dropout <= 0.0 and dropout.support_dropout <= 0.0:
            raise ValueError(
                "pim_adapter training requires non-zero PIM context/support dropout; "
                "otherwise every sample sees the same dense top-k memory."
            )
        if dropout is not None:
            if int(os.environ.get("RANK", "0")) == 0:
                _print_pim_dropout_probe(pim, dropout)
            dataset = PIMTrainingDropoutDataset(dataset, dropout)

    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(str(cfg.mixed_precision)),
        device=str(device),
    )
    model.load_checkpoint(str(base_ckpt))
    if task_language:
        from fastwam.zeva.task_language import task_language_identity

        model.zeva_task_context_identity = task_language_identity(
            bank_path, head_path, encoder_path, top_k=int(zeva.task_context.top_k),
        )
    else:
        model.zeva_task_context_identity = task_context_identity(
            "static", str(bank_path), str(head_path), int(zeva.task_context.top_k),
        )

    print("========== prepared Stage-2 ==========")
    print(f"training_stage: {training_stage}")
    print(f"dataset queries: {len(dataset):,}")
    print(f"prepared PIM: {pim_root}")
    print(f"prepared validation queries: {len(validation_dataset) if validation_dataset is not None else 0}")
    print(f"static context: {static_path}")
    if dropout is not None:
        print(
            "PIM training dropout: "
            f"context={dropout.context_dropout:.3f} "
            f"support={dropout.support_dropout:.3f} seed={dropout.seed}"
        )
    print(f"base sha256: {base_sha}")
    print(f"CTE sha256: {cte_sha}")
    print(f"semantic map sha256: {semantic_sha}")
    print("PREPARED STAGE-2 PREFLIGHT: PASSED")

    trainer = Wan22Trainer(cfg=cfg, model=model, train_dataset=dataset, val_dataset=validation_dataset)
    trainer.train()
    print(f"Stage-2 complete; checkpoints are under {Path(str(cfg.output_dir)) / 'checkpoints'}")


if __name__ == "__main__":
    main()
