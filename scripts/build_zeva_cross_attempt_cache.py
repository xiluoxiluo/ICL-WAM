"""Build compact whole-episode PIM from an existing full-prefix v4 cache.

Example:
  PYTHONPATH=src python scripts/build_zeva_cross_attempt_cache.py \
    --phase-cache /data/phase_effect_v4 --semantic-map /data/semantic_task_map.json \
    --output /data/pim_cross_train --split train

Use the emitted episode_splits.json with --episode-splits when building the
validation cache. Predictive reference caches require real condition metadata
as an episode-ID -> condition string JSON mapping. Missing history modes need
the explicit --allow-reduced-curriculum ablation; no condition is inferred.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from fastwam.zeva.cross_attempt_cache import (
    CROSS_ATTEMPT_CACHE_FORMAT, EGO_CROSS_ATTEMPT_CACHE_FORMAT, LEGACY_EGO_CROSS_ATTEMPT_CACHE_FORMAT,
    CrossAttemptPIMCache, HISTORY_KINDS,
    build_episode_pairings, file_sha256, validate_episode_splits,
)
from fastwam.zeva.ego_boundary_cache import EGO_BOUNDARY_CACHE_FORMAT, EgoBoundaryCache
from fastwam.zeva.memory import uniform_history_indices
from fastwam.zeva.schemas import CacheManifest
from fastwam.zeva.semantic_tasks import SemanticTaskMap, parse_episode_index_from_id


def _automatic_splits(tasks: dict[str, str], fraction: float, seed: int) -> dict[str, list[str]]:
    if not 0 <= fraction < 1:
        raise ValueError("validation_fraction must be in [0,1)")
    by_task: dict[str, list[str]] = defaultdict(list)
    for episode, task in tasks.items():
        by_task[task].append(episode)
    rng = np.random.default_rng(seed)
    result: dict[str, list[str]] = {"train": [], "validation": []}
    for task in sorted(by_task):
        episodes = sorted(by_task[task])
        order = rng.permutation(len(episodes)).tolist()
        # At least two training episodes are necessary for leave-one-out PIM.
        n_val = min(len(episodes) - 2, max(1, round(len(episodes) * fraction))) if fraction > 0 and len(episodes) >= 3 else 0
        result["validation"].extend(episodes[i] for i in order[:n_val])
        result["train"].extend(episodes[i] for i in order[n_val:])
    return {name: sorted(ids) for name, ids in result.items()}


def build_cache(
    *, phase_cache: Path, output: Path, semantic_map: Path, split: str = "train",
    episode_splits: Path | None = None, conditions: Path | None = None,
    validation_fraction: float = 0.1, seed: int = 42, max_past_attempts: int = 3,
    max_entries_per_attempt: int = 64, bit_size: int = 4,
    history_weights: dict[str, float] | None = None,
    uniform_indexing: str | None = None,
    history_pairing: str | None = None,
    allow_reduced_curriculum: bool | None = None,
) -> dict[str, Any]:
    phase_cache, output, semantic_map = phase_cache.resolve(), output.resolve(), semantic_map.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cache: {output}")
    if split not in {"train", "validation"} or min(max_past_attempts, max_entries_per_attempt, bit_size) < 1:
        raise ValueError("invalid split/history dimensions")
    if not semantic_map.is_file():
        raise FileNotFoundError(semantic_map)
    semantic = SemanticTaskMap.load(semantic_map)
    source_values = json.loads((phase_cache / "manifest.json").read_text())
    current_prediction = source_values.get("schema_version") == EGO_BOUNDARY_CACHE_FORMAT
    uniform_indexing = uniform_indexing or ("reference_round" if current_prediction else "legacy_floor")
    if uniform_indexing not in {"reference_round", "legacy_floor"}:
        raise ValueError("unknown PIM uniform_indexing")
    if not current_prediction and uniform_indexing != "legacy_floor":
        raise ValueError("reference_round caches require predictive Zeva-Ego features")
    reference_cache = current_prediction and uniform_indexing == "reference_round"
    history_pairing = history_pairing or ("reference_nearest" if reference_cache else "legacy_candidates")
    if history_pairing not in {"reference_nearest", "legacy_candidates"}:
        raise ValueError("unknown PIM history_pairing")
    if not reference_cache and history_pairing != "legacy_candidates":
        raise ValueError("reference_nearest requires a predictive reference_round cache")
    if allow_reduced_curriculum is None:
        allow_reduced_curriculum = not reference_cache
    if type(allow_reduced_curriculum) is not bool:
        raise ValueError("allow_reduced_curriculum must be boolean")
    ego_source = EgoBoundaryCache(phase_cache) if current_prediction else None
    if current_prediction:
        source_manifest = ego_source.manifest
        if source_manifest.semantic_task_map_sha256 != file_sha256(semantic_map):
            raise ValueError("boundary cache semantic map identity mismatch")
    else:
        source_manifest = CacheManifest(**source_values)
        source_manifest.validate({
            "schema_version": "zeva_fastwam_robotwin_cache_v4",
            "history_semantics": "full_episode_prefix", "query_step_unit": "raw_action_step",
            "feature_dtype": "float32",
        })
    rows = json.loads((phase_cache / "episode_index.json").read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("source phase/effect cache has no records")
    tasks: dict[str, str] = {}
    for row in rows:
        episode, task = str(row["episode_id"]), str(row["task_id"])
        resolved = semantic.resolve(episode_index=parse_episode_index_from_id(episode), strict=True)
        if task != resolved:
            raise ValueError(f"source cache semantic task mismatch for {episode}: {task!r} != {resolved!r}")
        if episode in tasks and tasks[episode] != task:
            raise ValueError("an episode cannot belong to multiple tasks")
        tasks[episode] = task
    splits = validate_episode_splits(
        json.loads(episode_splits.read_text()) if episode_splits else (
            ego_source.splits if current_prediction else _automatic_splits(tasks, validation_fraction, seed)),
        set(tasks),
    )
    if current_prediction and splits != ego_source.splits:
        raise ValueError("predicted-effect PIM must preserve the CTE/boundary cache episode split")
    query_set, train_set = set(splits[split]), set(splits["train"])
    if not query_set:
        raise ValueError(f"episode split {split} is empty")
    condition_map = json.loads(conditions.read_text()) if conditions else {}
    if conditions and (not isinstance(condition_map, dict) or (query_set | train_set) - set(condition_map)):
        raise ValueError("conditions must cover every query and training episode")
    if any(not isinstance(value, str) or not value for value in condition_map.values()):
        raise ValueError("condition values must be non-empty strings")
    if current_prediction and not allow_reduced_curriculum and conditions is None:
        raise ValueError("Reference PIM curriculum requires real --conditions metadata; use --allow-reduced-curriculum only for an explicit non-reference ablation")
    weights = history_weights or {"matched": 5.0, "far": 2.0, "cross_condition": 1.0, "empty": 0.0}
    if set(weights) - set(HISTORY_KINDS):
        raise ValueError("unknown history mixture kind")
    weights = {kind: float(weights.get(kind, 0.0)) for kind in HISTORY_KINDS}
    if any(not np.isfinite(value) or value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("invalid history mixture weights")
    if split == "train" and weights["empty"] > 0:
        raise ValueError("frozen-parent training requires nonempty histories; use empty diagnostics on validation")
    queries = [row for row in rows if row["record_type"] == "phase_query" and str(row["episode_id"]) in query_set and (not current_prediction or row["policy_valid"])]
    if not queries:
        raise ValueError("no query rows in selected split")
    query_episode_ids = sorted({str(row["episode_id"]) for row in queries})
    if set(query_episode_ids) != query_set:
        raise ValueError("a split episode has no phase queries")
    task_ids = sorted({tasks[episode] for episode in query_episode_ids})
    if len(task_ids) > np.iinfo(np.int16).max:
        raise ValueError("too many tasks for query task index")
    ep_index = {value: index for index, value in enumerate(query_episode_ids)}
    task_index = {value: index for index, value in enumerate(task_ids)}
    temporary = output.with_name(output.name + f".building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        n, dp, de = len(queries), source_manifest.phase_dim, source_manifest.effect_dim
        arrays: dict[str, np.ndarray] = {}
        specs = {
            "window_index": (np.int64, (n,)), "raw_step": (np.int32, (n,)),
            "episode_index": (np.int32, (n,)), "task_index": (np.int16, (n,)),
            "phase": (np.float32, (n, dp)), "bit_effects": (np.float32, (n, bit_size, de)),
            "bit_mask": (np.uint8, (n, bit_size)),
        }
        if current_prediction:
            specs["current_effect"] = (np.float32, (n, de))
            specs["current_effect_mask"] = (np.uint8, (n,))
        for name, (dtype, shape) in specs.items():
            arrays[name] = np.lib.format.open_memmap(temporary / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            arrays[name][:] = 0
        initial_phases: dict[str, np.ndarray] = {}
        effects: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        bit: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=bit_size))
        last_effect_end: dict[str, int] = {}
        seen_queries: set[tuple[str, int]] = set()
        seen_windows: set[int] = set()
        shard_name, tensors = None, None
        verified_shards: set[str] = set()
        cursor = 0
        # Completed effects at time t are available to the query at t. Sorting
        # by explicit raw time prevents accidental future BIT leakage.
        ordered = sorted(rows, key=lambda row: (str(row["episode_id"]), int(row["raw_step"]), 0 if row["record_type"] == "effect" else 1))
        for row in ordered:
            episode = str(row["episode_id"])
            if episode not in query_set | train_set:
                continue
            name = str(row["shard"])
            if shard_name != name:
                if current_prediction and name not in verified_shards:
                    if file_sha256(phase_cache / name) != source_manifest.shard_sha256.get(name):
                        raise ValueError("boundary prediction shard checksum mismatch")
                    verified_shards.add(name)
                tensors = load_file(str(phase_cache / name), device="cpu")
                shard_name = name
                if (tensors["phase_pre"].dtype != torch.float32 or tensors["effect"].dtype != torch.float32
                        or tensors["valid"].dtype != torch.bool
                        or tensors["phase_pre"].ndim != 2 or tensors["phase_pre"].shape[1] != dp
                        or tensors["effect"].shape != (tensors["phase_pre"].shape[0], de)
                        or tensors["valid"].shape != (tensors["phase_pre"].shape[0],)):
                    raise ValueError("source shard shape/dtype differs from v4 manifest")
            offset = int(row["offset"])
            if not 0 <= offset < len(tensors["valid"]):
                raise ValueError("source row offset is outside its shard")
            if not bool(tensors["valid"][offset].item()):
                raise ValueError("v4 phase/effect source contains invalid record")
            phase = tensors["phase_pre"][offset].float().numpy().copy()
            effect = tensors["effect"][offset].float().numpy().copy()
            if phase.shape != (dp,) or effect.shape != (de,) or not np.isfinite(phase).all() or not np.isfinite(effect).all():
                raise ValueError("invalid source phase/effect feature")
            raw_step = int(row["raw_step"])
            if raw_step < 0:
                raise ValueError("source raw_step must be non-negative")
            if row["record_type"] == "effect":
                start, end = int(row["start_raw_step"]), int(row["end_raw_step"])
                window = source_manifest.action_group_size * source_manifest.effect_window_transitions
                if end != raw_step or end - start != window or start < 0:
                    raise ValueError("PIM source must be a completed full effect window")
                if end <= last_effect_end.get(episode, -1):
                    raise ValueError("duplicate or non-increasing completed effect")
                last_effect_end[episode] = end
                if episode in train_set:
                    effects[episode].append((phase, effect))
                if episode in query_set:
                    bit[episode].append(effect)
            elif row["record_type"] == "phase_query":
                if int(row["start_raw_step"]) != raw_step or int(row["end_raw_step"]) != raw_step:
                    raise ValueError("phase query must identify exactly one raw boundary")
                if raw_step == 0:
                    if episode in initial_phases:
                        raise ValueError("duplicate initial phase query")
                    initial_phases[episode] = phase
                if episode in query_set and (not current_prediction or row["policy_valid"]):
                    identity = (episode, raw_step)
                    window_index = int(row["window_index"])
                    if window_index < 0 or identity in seen_queries or window_index in seen_windows:
                        raise ValueError("duplicate query identity/window index")
                    seen_queries.add(identity)
                    seen_windows.add(window_index)
                    arrays["window_index"][cursor] = window_index
                    arrays["raw_step"][cursor] = raw_step
                    arrays["episode_index"][cursor] = ep_index[episode]
                    arrays["task_index"][cursor] = task_index[tasks[episode]]
                    arrays["phase"][cursor] = phase
                    past = list(bit[episode])
                    if past:
                        arrays["bit_effects"][cursor, -len(past):] = np.stack(past)
                        arrays["bit_mask"][cursor, -len(past):] = 1
                    if current_prediction:
                        arrays["current_effect"][cursor] = effect
                        arrays["current_effect_mask"][cursor] = 1
                    cursor += 1
                if current_prediction:
                    # Match deployment: condition the current action first, then
                    # append its prediction. The query never reads itself in BIT.
                    if episode in train_set:
                        effects[episode].append((phase, effect))
                    if episode in query_set:
                        bit[episode].append(effect)
            else:
                raise ValueError(f"unsupported record type: {row['record_type']}")
        if cursor != n:
            raise RuntimeError("query count changed during cache construction")
        required_initials = query_set | {episode for episode in train_set if effects[episode]}
        if required_initials - set(initial_phases):
            raise ValueError("initial raw_step=0 phase is required for episode pairing")
        history_ids = sorted(episode for episode in train_set if effects[episode])
        if not history_ids:
            raise ValueError("no completed training history exists")
        query_meta = {episode: {"episode_id": episode, "task_id": tasks[episode], "split": split,
                                "condition": condition_map.get(episode), "initial_phase": initial_phases[episode]}
                      for episode in query_episode_ids}
        history_meta = [{"episode_id": episode, "task_id": tasks[episode], "split": "train",
                         "condition": condition_map.get(episode), "initial_phase": initial_phases[episode]}
                        for episode in history_ids]
        pairings = build_episode_pairings(
            query_meta, history_meta, max_histories=max_past_attempts, use_conditions=conditions is not None,
            history_pairing=history_pairing, allow_reduced_curriculum=allow_reduced_curriculum,
        )
        h = len(history_ids)
        for name, shape, dtype in (
            ("history_phases", (h, max_entries_per_attempt, dp), np.float32),
            ("history_effects", (h, max_entries_per_attempt, de), np.float32),
            ("history_mask", (h, max_entries_per_attempt), np.uint8),
        ):
            arrays[name] = np.lib.format.open_memmap(temporary / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            arrays[name][:] = 0
        for index, episode in enumerate(history_ids):
            trace = effects[episode]
            selected = uniform_history_indices(len(trace), max_entries_per_attempt, rounding=uniform_indexing)
            arrays["history_phases"][index, :len(selected)] = np.stack([trace[i][0] for i in selected])
            arrays["history_effects"][index, :len(selected)] = np.stack([trace[i][1] for i in selected])
            arrays["history_mask"][index, :len(selected)] = 1
        for array in arrays.values():
            array.flush()
        # Initial phases are only used for pairing; omit them from JSON to keep
        # provenance small and avoid a second copy of every representation.
        for row in (*query_meta.values(), *history_meta):
            row.pop("initial_phase")
        metadata = {
            "episode_ids.json": query_episode_ids, "task_ids.json": task_ids,
            "query_episodes.json": query_meta, "history_episodes.json": history_meta,
            "pairings.json": pairings, "episode_splits.json": splits,
        }
        for name, value in metadata.items():
            (temporary / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        manifest = {
            "format": ((EGO_CROSS_ATTEMPT_CACHE_FORMAT if uniform_indexing == "reference_round"
                        else LEGACY_EGO_CROSS_ATTEMPT_CACHE_FORMAT) if current_prediction
                       else CROSS_ATTEMPT_CACHE_FORMAT), "pim_variant": "cross_attempt",
            "history_compression": "uniform", "uniform_indexing": uniform_indexing,
            "history_pairing": history_pairing, "allow_reduced_curriculum": allow_reduced_curriculum,
            "retrieval_mode": "cross_attempt", "source_subset": "train-only", "split": split,
            "history_semantics": "full_episode_prefix", "query_step_unit": "raw_action_step",
            "history_source_semantics": "other_train_episode", "candidate_semantics": "complete_episode_uniform",
            "pairing_basis": "initial_phase_cosine", "use_conditions": conditions is not None,
            "phase_dim": dp, "effect_dim": de, "bit_size": bit_size,
            "top_k": max_past_attempts * max_entries_per_attempt,
            "max_past_attempts": max_past_attempts, "max_entries_per_attempt": max_entries_per_attempt,
            "num_queries": n, "num_episodes": len(query_episode_ids), "num_history_episodes": h,
            "semantic_task_count": len(task_ids), "feature_dtype": "float32", "seed": seed,
            "history_weights": weights,
            "eligible_pairing_episodes": {kind: sum(bool(row[kind]) for row in pairings.values()) for kind in HISTORY_KINDS[:3]},
            "cte_checkpoint_sha256": source_manifest.cte_checkpoint_sha256,
            "dataset_stats_sha256": source_manifest.dataset_stats_sha256,
            "dataset_path": source_manifest.dataset_path,
            "semantic_task_map_sha256": file_sha256(semantic_map),
            "phase_cache_manifest_sha256": file_sha256(phase_cache / "manifest.json"),
            "phase_cache_index_sha256": file_sha256(phase_cache / "episode_index.json"),
            "metadata_sha256": {name: file_sha256(temporary / name) for name in metadata},
            "array_sha256": {f"{name}.npy": file_sha256(temporary / f"{name}.npy") for name in arrays},
            "conditions_sha256": file_sha256(conditions) if conditions else None,
        }
        if current_prediction:
            global_kinds = [kind for kind in HISTORY_KINDS[:3] if all(row[kind] for row in pairings.values())]
            if not global_kinds:
                raise ValueError("no nonempty history kind is available for every query episode")
            manifest.update(globally_available_history_kinds=global_kinds,
                            reduced_curriculum=global_kinds != list(HISTORY_KINDS[:3]),
                            memory_semantics="predicted_effect_trace", effect_source="current_prediction",
                            executed_horizon=source_manifest.executed_horizon,
                            cte_input_type=source_manifest.cte_input_type,
                            source_episode_splits_sha256=source_manifest.episode_splits_sha256)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        reader = CrossAttemptPIMCache(temporary)
        # Validate every query's episode/task mapping without decoding RGB.
        for index in range(len(reader)):
            ep = int(reader._arrays["episode_index"][index])
            task = int(reader._arrays["task_index"][index])
            if reader.query_episodes[reader.episode_ids[ep]]["task_id"] != reader.task_ids[task]:
                raise ValueError("constructed query task identity mismatch")
        del reader, arrays
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-map", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--episode-splits", type=Path)
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-past-attempts", type=int, default=3)
    parser.add_argument("--max-entries-per-attempt", type=int, default=64)
    parser.add_argument("--bit-size", type=int, default=4)
    parser.add_argument("--uniform-indexing", choices=("reference_round", "legacy_floor"),
                        help="Predictive default is reference_round; legacy_floor only replays old caches")
    parser.add_argument("--history-pairing", choices=("reference_nearest", "legacy_candidates"),
                        help="Predictive v3 defaults to one fixed nearest/farthest episode per mode")
    parser.add_argument("--allow-reduced-curriculum", action="store_true", default=None,
                        help="Explicit non-reference ablation permitting missing real condition/history modes")
    parser.add_argument("--history-weights", type=json.loads,
                        help='JSON weights, e.g. {"matched":5,"far":2,"cross_condition":1,"empty":0}; train requires empty=0')
    args = parser.parse_args()
    manifest = build_cache(**vars(args))
    print(json.dumps({key: manifest[key] for key in ("format", "split", "num_queries", "num_episodes", "num_history_episodes", "eligible_pairing_episodes", "history_weights")}, indent=2))


if __name__ == "__main__":
    main()
