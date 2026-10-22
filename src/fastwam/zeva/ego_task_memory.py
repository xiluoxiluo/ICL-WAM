"""Train-only demonstration memory and all-episode initial-query artifacts.

Validation episodes contribute initial observation/text queries only. Their
trajectory features never enter the demonstration bank or retrieval-head data.
"""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .checkpoint import checkpoint_sha256
from .prepared_stage2 import STATIC_CONTEXT_FORMAT, StaticEpisodeContextCache
from .static_task_context import (
    BEHAVIOR_KEY_SPACE, BEHAVIOR_VALUE_SPACE, READOUT_FORMAT, READOUT_KIND,
    StaticTaskContextRetriever, load_readout_cache, validate_behavior_bank,
)
from .task_context import TaskContextBank


EGO_QUERY_READOUT_FORMAT = "zeva_ego_initial_query_readouts_v1"


def episode_task_instructions(base, cache) -> dict[str, str]:
    """Read raw task text from metadata only, without image/action access.

    This mirrors LeRobot's task-index lookup, not the FastWAM video prompt.
    A static episode context requires one instruction throughout an episode.
    Semantic task labels are checked for artifact integrity but never passed
    into the language classifier as query inputs.
    """
    sources = base.lerobot_dataset.multi_dataset._datasets
    if len(sources) != 1:
        raise ValueError("Task-language artifacts require exactly one RoboTwin dataset root")
    source = sources[0]
    table = source.hf_dataset.with_format(None)
    required = ("episode_index", "frame_index", "task_index")
    if not set(required).issubset(table.column_names):
        raise ValueError("Task-language metadata needs episode_index, frame_index and task_index")
    table = table.select_columns(list(required))
    output = {}
    for identity in ordered_episode_ids(cache):
        record = cache.episodes[identity]
        start, length = int(record["dataset_start"]), int(record["episode_length"])
        rows = table[start:start + length]
        if len(rows["task_index"]) != length:
            raise ValueError("Task-language episode extends beyond source metadata")
        episode_index = int(record["episode_index"])
        if (identity != f"{source.root}::episode-{episode_index}"
                or any(int(value) != episode_index for value in rows["episode_index"])
                or list(map(int, rows["frame_index"])) != list(range(length))):
            raise ValueError("Task-language source episode/frame identity differs from boundary cache")
        raw_indices = list(map(int, rows["task_index"]))
        semantic = str(base.resolve_semantic_task_id(episode_index=episode_index,
                                                     raw_task_index=raw_indices[0], strict=True))
        if semantic != str(record["task_id"]):
            raise ValueError("Task-language source semantic task differs from boundary cache")
        texts = {str(source.meta.tasks[index]) for index in set(raw_indices)}
        if len(texts) != 1:
            raise ValueError("Task-language static context requires a fixed instruction within each episode")
        override = getattr(base, "override_instruction", None)
        instruction = str(override) if override is not None else next(iter(texts))
        if not instruction.strip():
            raise ValueError("Task-language queries cannot use empty task instructions")
        output[identity] = instruction
    return output


def _language_bank_metadata(metadata: dict[str, Any], *, encoder_sha256: str) -> dict[str, Any]:
    from .task_language import TASK_LANGUAGE_FEATURE_KIND

    if "base_checkpoint_sha256" in metadata:
        raise ValueError("Task-language query artifacts must not depend on a mutable foundation checkpoint")
    return {
        **metadata, "key_space": BEHAVIOR_KEY_SPACE, "value_space": BEHAVIOR_VALUE_SPACE,
        "readout_kind": TASK_LANGUAGE_FEATURE_KIND, "query_mode": "task_language", "split": "train",
        "foundation_independent_query": True, "encoder_sha256": encoder_sha256,
        "history_semantics": "full_episode_prefix", "memory_semantics": "predicted_effect_trace",
        "pooling_rule": "complete_transition_starts_mean_then_l2_keys_and_values",
        "retrieval_trace_space": "cte_task_head_raw_v1",
    }


def save_ego_task_language_artifacts(cache, features: dict[str, Tensor], *, instructions: dict[str, str],
                                      bank_path: str | Path, feature_path: str | Path,
                                      encoder_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Save a train-only CTE bank plus pure-language queries for both splits."""
    from .task_language import FEATURE_FORMAT, TASK_LANGUAGE_FEATURE_KIND

    bank_path, feature_path, encoder_path = (Path(path).expanduser().resolve()
                                             for path in (bank_path, feature_path, encoder_path))
    if len({bank_path, feature_path, encoder_path}) != 3 or bank_path.exists() or feature_path.exists():
        raise ValueError("Language bank/features need distinct unused paths separate from the encoder")
    if not encoder_path.is_file():
        raise FileNotFoundError(f"Frozen task-language encoder artifact is missing: {encoder_path}")
    ids = ordered_episode_ids(cache)
    if set(features) != set(ids) or set(instructions) != set(ids):
        raise ValueError("Pure-language queries must cover exactly the CTE train and validation episodes")
    values = torch.stack([features[identity].detach().float().cpu() for identity in ids])
    if values.ndim != 2 or values.shape[1] < 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("Task-language features must be finite float32 [episode,feature]")
    if any(not isinstance(instructions[identity], str) or not instructions[identity].strip() for identity in ids):
        raise ValueError("Each language query requires a nonempty raw instruction")
    entries = pool_train_behavior_entries(cache, instructions)
    encoder_sha = checkpoint_sha256(encoder_path)
    full_metadata = {
        **_language_bank_metadata(metadata, encoder_sha256=encoder_sha),
        "num_episodes": len(entries), "num_query_episodes": len(ids), "language_feature_dim": values.shape[1],
        "semantic_task_count": len({entry["task_id"] for entry in entries}),
    }
    bank = TaskContextBank(entries, key_dim=cache.manifest.retrieval_dim, value_dim=cache.manifest.global_dim,
                           temperature=1.0, metadata=full_metadata)
    for path in (bank_path, feature_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = [path.with_name(f".{path.name}.building-{os.getpid()}") for path in (bank_path, feature_path)]
    try:
        bank.save(temporary[0])
        payload = {
            "format": FEATURE_FORMAT, "feature_kind": TASK_LANGUAGE_FEATURE_KIND,
            "source_feature": "task_language", "bank_sha256": checkpoint_sha256(temporary[0]),
            "encoder_sha256": encoder_sha, "episode_ids": ids,
            "task_ids": [str(cache.episodes[identity]["task_id"]) for identity in ids],
            "instructions": [instructions[identity] for identity in ids], "features": values,
            "episode_splits": cache.splits, "metadata": full_metadata,
        }
        torch.save(payload, temporary[1])
        load_ego_task_language_features(temporary[1], temporary[0], encoder_path, cache,
                                         expected_metadata=metadata)
        for source, destination in zip(temporary, (bank_path, feature_path), strict=True):
            source.replace(destination)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return {"train_episodes": len(entries), "query_episodes": len(ids),
            "language_feature_dim": values.shape[1], "bank_sha256": checkpoint_sha256(bank_path),
            "language_feature_cache_sha256": checkpoint_sha256(feature_path)}


def load_ego_task_language_features(path: str | Path, bank_path: str | Path,
                                     encoder_path: str | Path, cache, *,
                                     expected_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    from .task_language import FEATURE_FORMAT, TASK_LANGUAGE_FEATURE_KIND

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format") != FEATURE_FORMAT or payload.get("feature_kind") != TASK_LANGUAGE_FEATURE_KIND
            or payload.get("source_feature") != "task_language"):
        raise ValueError("Expected task-only frozen input-embedding feature cache")
    encoder_sha = checkpoint_sha256(encoder_path)
    for key, expected in (("bank_sha256", checkpoint_sha256(bank_path)),
                          ("encoder_sha256", encoder_sha), ("episode_splits", cache.splits)):
        if payload.get(key) != expected:
            raise ValueError(f"Task-language feature identity mismatch: {key}")
    ids, features = payload.get("episode_ids"), payload.get("features")
    if not isinstance(ids, list) or len(set(ids)) != len(ids) or set(ids) != set(cache.episodes):
        raise ValueError("Task-language feature cache must cover every CTE query episode exactly once")
    if (not isinstance(features, Tensor) or features.ndim != 2 or features.shape[0] != len(ids)
            or features.dtype != torch.float32 or not bool(torch.isfinite(features).all())):
        raise ValueError("Invalid task-language feature tensor")
    tasks, instructions = payload.get("task_ids"), payload.get("instructions")
    if tasks != [str(cache.episodes[identity]["task_id"]) for identity in ids]:
        raise ValueError("Task-language task labels disagree with CTE episode metadata")
    if (not isinstance(instructions, list) or len(instructions) != len(ids)
            or any(not isinstance(value, str) or not value.strip() for value in instructions)):
        raise ValueError("Task-language feature cache requires one raw instruction per episode")
    expected = _language_bank_metadata(dict(expected_metadata or {}), encoder_sha256=encoder_sha)
    bank = TaskContextBank.load(bank_path, expected_key_dim=cache.manifest.retrieval_dim,
                                expected_value_dim=cache.manifest.global_dim)
    validate_behavior_bank(bank, expected)
    if bank.temperature != 1.0:
        raise ValueError("Reference task-scoped cosine/softmax retrieval uses temperature=1.0")
    if {entry["episode_id"] for entry in bank.entries} != set(cache.splits["train"]):
        raise ValueError("Task-language memory must contain exactly the CTE training episodes")
    for key, value in expected.items():
        if payload.get("metadata", {}).get(key) != value:
            raise ValueError(f"Task-language feature metadata mismatch: {key}")
    if features.shape[1] != bank.metadata.get("language_feature_dim"):
        raise ValueError("Task-language feature dimension differs from the bank/encoder contract")
    return payload


def build_ego_task_language_context_cache(*, cache, bank_path: str | Path, feature_path: str | Path,
                                           encoder_path: str | Path, head_path: str | Path,
                                           output_path: str | Path, expected_metadata: dict[str, Any],
                                           top_k: int = 5, device: str | torch.device = "cpu",
                                           batch_size: int = 512) -> dict[str, Any]:
    """Classify language, then aggregate that task's train demos without GT scope."""
    from .task_language import TaskLanguageMemory, task_language_identity

    inputs = tuple(Path(path).expanduser().resolve() for path in (bank_path, feature_path, encoder_path, head_path))
    output_path = Path(output_path).expanduser().resolve()
    if output_path in inputs or output_path.exists() or batch_size < 1:
        raise ValueError("Task-language static contexts need a positive batch and an unused output path")
    bank_path, feature_path, encoder_path, head_path = inputs
    query = load_ego_task_language_features(feature_path, bank_path, encoder_path, cache,
                                            expected_metadata=expected_metadata)
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_cache_sha256") != checkpoint_sha256(feature_path):
        raise ValueError("Language classifier was trained with a different feature artifact")
    train, validation = set(checkpoint.get("train_episode_ids", [])), set(checkpoint.get("validation_episode_ids", []))
    if train & validation or train | validation != set(cache.splits["train"]):
        raise ValueError("Language classifier train/selection episodes must stay inside the CTE training split")
    memory = TaskLanguageMemory.load(bank_path, head_path, encoder_path=encoder_path, top_k=top_k,
                                     device=device, expected=expected_metadata)
    contexts, indices, scores, sources = [], [], [], []
    ids, train_ids = query["episode_ids"], set(cache.splits["train"])
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            exclusions = [identity if identity in train_ids else None for identity in ids[start:stop]]
            result = memory.retrieve(query["features"][start:stop].to(device), exclude_episode_ids=exclusions)
            contexts.append(result.values.float().cpu())
            indices.append(result.indices.cpu())
            scores.append(result.scores.float().cpu())
            sources.extend(result.sources)
    identity = task_language_identity(bank_path, head_path, encoder_path, top_k=top_k)
    metadata = {
        **memory.bank.metadata, "bank_sha256": checkpoint_sha256(bank_path),
        "retrieval_checkpoint_sha256": checkpoint_sha256(head_path),
        "language_encoder_sha256": identity["language_encoder_sha256"],
        "language_feature_cache_sha256": checkpoint_sha256(feature_path), "top_k": top_k, "temperature": 1.0,
        "query_scope": "train_and_validation_task_language_only", "task_context_identity": identity,
        "train_query_exclusion": "leave_one_episode_out_in_task_mean_and_candidates",
        "validation_query_sources": "train_only", "scope_selection": "predicted_task_only",
    }
    payload = {"format": STATIC_CONTEXT_FORMAT, "episode_ids": ids, "task_contexts": torch.cat(contexts),
               "retrieval_indices": torch.cat(indices), "retrieval_scores": torch.cat(scores),
               "retrieval_sources": sources, "episode_splits": cache.splits, "metadata": metadata}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.building-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        checked = StaticEpisodeContextCache(temporary)
        if set(checked.episode_ids) != set(cache.episodes):
            raise ValueError("Task-language static cache omitted held-out query episodes")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"episodes": len(ids), "train_queries": len(train_ids),
            "validation_queries": len(cache.splits["validation"]), "top_k": top_k,
            "static_cache_sha256": checkpoint_sha256(output_path)}


def validate_ego_task_language_static_context(static_context: StaticEpisodeContextCache, *, cache,
                                               bank_path: str | Path, feature_path: str | Path,
                                               encoder_path: str | Path, head_path: str | Path,
                                               expected_metadata: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """Validate immutable query artifacts independently of a trainable foundation."""
    from .task_language import TASK_LANGUAGE_FEATURE_KIND, task_language_identity

    query = load_ego_task_language_features(feature_path, bank_path, encoder_path, cache,
                                            expected_metadata=expected_metadata)
    identity = task_language_identity(bank_path, head_path, encoder_path, top_k=top_k)
    expected = {
        **expected_metadata, "bank_sha256": identity["bank_sha256"],
        "retrieval_checkpoint_sha256": identity["head_sha256"],
        "language_encoder_sha256": identity["language_encoder_sha256"],
        "language_feature_cache_sha256": checkpoint_sha256(feature_path), "top_k": top_k, "temperature": 1.0,
        "readout_kind": TASK_LANGUAGE_FEATURE_KIND, "query_mode": "task_language",
        "foundation_independent_query": True, "scope_selection": "predicted_task_only",
        "train_query_exclusion": "leave_one_episode_out_in_task_mean_and_candidates",
        "validation_query_sources": "train_only",
        "task_context_identity": identity,
    }
    for key, value in expected.items():
        if static_context.metadata.get(key) != value:
            raise ValueError(f"Task-language static-context identity mismatch: {key}")
    if "base_checkpoint_sha256" in static_context.metadata:
        raise ValueError("Task-language static queries cannot depend on a mutable foundation checkpoint")
    if set(static_context.episode_ids) != set(cache.episodes):
        raise ValueError("Task-language static context must cover the exact CTE train/validation queries")
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("source_feature") != "task_language"
            or checkpoint.get("feature_cache_sha256") != checkpoint_sha256(feature_path)):
        raise ValueError("Static task-language context does not use the expected trained classifier")
    train, validation = set(checkpoint.get("train_episode_ids", [])), set(checkpoint.get("validation_episode_ids", []))
    if train & validation or train | validation != set(cache.splits["train"]):
        raise ValueError("Language classifier train/selection episodes must stay inside the CTE training split")
    return query


def ordered_episode_ids(cache) -> list[str]:
    return sorted(cache.episodes, key=lambda identity: (int(cache.episodes[identity]["dataset_start"]), identity))


def pool_train_behavior_entries(cache, instructions: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Reference export pools complete-transition starts, then normalizes.

    Zeva's export_cte_artifacts.py uses task_head(phase[:-1]).mean and
    phase[:-1].mean, L2-normalizing both trajectory prototypes. Partial final
    decision states may be useful online/PIM, but have no Stage1 action target
    and must not enter this pool. Validation trajectories are never read.
    """
    train_ids = set(cache.splits["train"])
    by_episode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(cache.rows):
        identity = row["episode_id"]
        if (identity in train_ids
                and int(row["raw_step"]) + int(cache.manifest.executed_horizon)
                <= int(cache.episodes[identity]["episode_length"]) - 1):
            by_episode[row["episode_id"]].append(index)
    if set(by_episode) != train_ids:
        raise ValueError("Every training episode needs cached causal decision states")
    entries = []
    for identity in ordered_episode_ids(cache):
        if identity not in train_ids:
            continue
        metadata = cache.episodes[identity]
        indices = sorted(by_episode[identity], key=lambda index: int(cache.rows[index]["raw_step"]))
        if int(cache.rows[indices[0]]["raw_step"]) != 0:
            raise ValueError("CTE task-memory trajectories must begin at the initial SOS boundary")
        rows = [cache.get(index) for index in indices]
        if any(str(row["task_id"]) != str(metadata["task_id"]) or row.get("split") != "train" for row in rows):
            raise ValueError("Task-memory source task/split identity mismatch")
        keys = torch.stack([row["retrieval"] for row in rows]).float()
        values = torch.stack([row["global"] for row in rows]).float()
        if not bool(torch.isfinite(keys).all() and torch.isfinite(values).all()):
            raise ValueError("Task-memory prototypes must be finite")
        entries.append({"retrieval_key": F.normalize(keys.mean(0), dim=0), "behavior_value": F.normalize(values.mean(0), dim=0),
                        "episode_id": identity, "task_id": str(metadata["task_id"]),
                        "instruction": (instructions or {}).get(identity, ""),
                        "dataset_start": int(metadata["dataset_start"]),
                        "num_decision_boundaries": len(rows), "split": "train"})
    return entries


def save_ego_task_memory(cache, readouts: dict[str, Tensor], *, bank_path: str | Path,
                          train_readout_path: str | Path, query_readout_path: str | Path,
                          metadata: dict[str, Any], instructions: dict[str, str] | None = None,
                          temperature: float = 0.07) -> dict[str, Any]:
    paths = tuple(Path(path).expanduser().resolve() for path in (bank_path, train_readout_path, query_readout_path))
    if len(set(paths)) != 3 or any(path.exists() for path in paths):
        raise ValueError("Bank, train readouts and all-query readouts require three distinct unused paths")
    bank_path, train_readout_path, query_readout_path = paths
    episode_ids = ordered_episode_ids(cache)
    if set(readouts) != set(episode_ids):
        raise ValueError("Initial readouts must cover exactly train plus held-out validation episodes")
    all_readouts = torch.stack([readouts[identity].detach().float().cpu() for identity in episode_ids])
    if all_readouts.ndim != 2 or not bool(torch.isfinite(all_readouts).all()):
        raise ValueError("Initial query readouts must be finite [episode,feature]")
    entries = pool_train_behavior_entries(cache, instructions)
    train_ids = [entry["episode_id"] for entry in entries]
    train_readouts = torch.stack([readouts[identity].detach().float().cpu() for identity in train_ids])
    full_metadata = {
        **metadata, "key_space": BEHAVIOR_KEY_SPACE, "value_space": BEHAVIOR_VALUE_SPACE,
        "readout_kind": READOUT_KIND, "split": "train", "readout_dim": all_readouts.shape[1],
        "history_semantics": "full_episode_prefix", "memory_semantics": "predicted_effect_trace",
        "pooling_rule": "complete_transition_starts_mean_then_l2_keys_and_values",
        "retrieval_trace_space": "cte_task_head_raw_v1",
        "num_episodes": len(entries), "num_query_episodes": len(episode_ids),
        "semantic_task_count": len({entry["task_id"] for entry in entries}),
    }
    bank = TaskContextBank(entries, key_dim=cache.manifest.retrieval_dim, value_dim=cache.manifest.global_dim,
                           temperature=temperature, metadata=full_metadata)
    temporary = [path.with_name(f".{path.name}.building-{os.getpid()}") for path in paths]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bank.save(temporary[0])
        bank_sha = checkpoint_sha256(temporary[0])
        torch.save({"format": READOUT_FORMAT, "readout_kind": READOUT_KIND, "bank_sha256": bank_sha,
                    "episode_ids": train_ids, "readouts": train_readouts, "metadata": full_metadata}, temporary[1])
        torch.save({"format": EGO_QUERY_READOUT_FORMAT, "readout_kind": READOUT_KIND, "bank_sha256": bank_sha,
                    "train_readout_cache_sha256": checkpoint_sha256(temporary[1]),
                    "episode_ids": episode_ids, "task_ids": [str(cache.episodes[identity]["task_id"]) for identity in episode_ids],
                    "readouts": all_readouts, "episode_splits": cache.splits,
                    "metadata": {**full_metadata, "query_scope": "train_and_validation_initial_only"}}, temporary[2])
        reloaded = TaskContextBank.load(temporary[0], expected_key_dim=cache.manifest.retrieval_dim,
                                        expected_value_dim=cache.manifest.global_dim)
        validate_behavior_bank(reloaded)
        load_readout_cache(temporary[1], temporary[0], reloaded)
        load_ego_query_readouts(temporary[2], temporary[0], temporary[1], reloaded, cache.splits,
                                expected_metadata=metadata)
        for source, destination in zip(temporary, paths, strict=True):
            source.replace(destination)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return {"train_episodes": len(train_ids), "query_episodes": len(episode_ids),
            "semantic_task_count": full_metadata["semantic_task_count"], "readout_dim": all_readouts.shape[1],
            "bank_sha256": checkpoint_sha256(bank_path)}


def load_ego_query_readouts(path: str | Path, bank_path: str | Path, train_readout_path: str | Path,
                            bank: TaskContextBank, episode_splits: dict[str, list[str]], *,
                            expected_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != EGO_QUERY_READOUT_FORMAT or payload.get("readout_kind") != READOUT_KIND:
        raise ValueError("Expected Zeva-Ego all-episode initial query readouts")
    if payload.get("bank_sha256") != checkpoint_sha256(bank_path):
        raise ValueError("Query-readout bank identity mismatch")
    if payload.get("train_readout_cache_sha256") != checkpoint_sha256(train_readout_path):
        raise ValueError("Query-readout training readout identity mismatch")
    if payload.get("episode_splits") != episode_splits:
        raise ValueError("Query readouts do not use the CTE held-out episode split")
    for key, expected in (expected_metadata or {}).items():
        if payload.get("metadata", {}).get(key) != expected:
            raise ValueError(f"Query-readout metadata mismatch: {key}")
    ids, values = payload.get("episode_ids"), payload.get("readouts")
    expected_ids = set(episode_splits["train"]) | set(episode_splits["validation"])
    if not isinstance(ids, list) or len(set(ids)) != len(ids) or set(ids) != expected_ids:
        raise ValueError("All-query readout episode coverage mismatch")
    if (not isinstance(values, Tensor) or values.shape != (len(ids), bank.metadata["readout_dim"])
            or values.dtype != torch.float32 or not bool(torch.isfinite(values).all())):
        raise ValueError("Invalid all-query initial readout tensor")
    train = load_readout_cache(train_readout_path, bank_path, bank)
    if set(train["episode_ids"]) != set(episode_splits["train"]):
        raise ValueError("Only CTE training demonstrations may enter the bank/head training data")
    positions = {identity: index for index, identity in enumerate(ids)}
    selected = torch.stack([values[positions[identity]] for identity in train["episode_ids"]])
    if not torch.equal(selected, train["readouts"]):
        raise ValueError("Train/all-query readouts disagree for the same initial observations")
    return payload


@torch.no_grad()
def retrieve_episode_contexts(bank: TaskContextBank, head, episode_ids: list[str], readouts: Tensor,
                               train_ids: set[str], *, top_k: int = 5,
                               batch_size: int = 512) -> tuple[Tensor, Tensor, Tensor]:
    """Train queries exclude their own demo; held-out queries search all train demos."""
    if not 1 <= top_k < len(bank) or batch_size < 1:
        raise ValueError("Static context requires positive batch size and 1 <= top_k < train bank size")
    if readouts.ndim != 2 or len(readouts) != len(episode_ids) or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("One initial readout is required per unique query episode")
    entries = bank.entries
    positions = {entry["episode_id"]: index for index, entry in enumerate(entries)}
    if set(positions) != set(train_ids):
        raise ValueError("Retrieval bank must contain exactly the CTE training episodes")
    parameter = next(head.parameters())
    device = parameter.device
    keys = F.normalize(torch.stack([entry["retrieval_key"] for entry in entries]).to(device), dim=-1)
    values = torch.stack([entry["behavior_value"] for entry in entries]).to(device)
    contexts = torch.empty(len(episode_ids), bank.value_dim, dtype=torch.float32)
    indices = torch.empty(len(episode_ids), top_k, dtype=torch.long)
    scores_output = torch.empty(len(episode_ids), top_k, dtype=torch.float32)
    head.eval()
    for start in range(0, len(episode_ids), batch_size):
        stop = min(start + batch_size, len(episode_ids))
        query = head(readouts[start:stop].to(device=device, dtype=parameter.dtype))
        scores = query.float() @ keys.T
        for local, identity in enumerate(episode_ids[start:stop]):
            if identity in train_ids:
                scores[local, positions[identity]] = -torch.inf
        top_scores, top_indices = scores.topk(top_k, dim=-1)
        context = torch.einsum("bk,bkd->bd", F.softmax(top_scores / bank.temperature, dim=-1), values[top_indices])
        contexts[start:stop], indices[start:stop], scores_output[start:stop] = context.cpu(), top_indices.cpu(), top_scores.cpu()
    if not bool(torch.isfinite(contexts).all() and torch.isfinite(scores_output).all()):
        raise ValueError("Non-finite retrieved static episode context")
    return contexts, indices, scores_output


def build_ego_static_context_cache(*, cache, bank_path: str | Path, train_readout_path: str | Path,
                                    query_readout_path: str | Path, head_path: str | Path,
                                    output_path: str | Path, expected_metadata: dict[str, Any],
                                    top_k: int = 5, device: str | torch.device = "cpu",
                                    batch_size: int = 512) -> dict[str, Any]:
    inputs = tuple(Path(path).expanduser().resolve() for path in (bank_path, train_readout_path, query_readout_path, head_path))
    output_path = Path(output_path).expanduser().resolve()
    if output_path in inputs or output_path.exists():
        raise ValueError("Static contexts require an unused path distinct from all source artifacts")
    bank_path, train_readout_path, query_readout_path, head_path = inputs
    bank = TaskContextBank.load(bank_path, expected_key_dim=cache.manifest.retrieval_dim,
                                expected_value_dim=cache.manifest.global_dim)
    validate_behavior_bank(bank, expected_metadata)
    query = load_ego_query_readouts(query_readout_path, bank_path, train_readout_path, bank,
                                    cache.splits, expected_metadata=expected_metadata)
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
    if checkpoint.get("readout_cache_sha256") != checkpoint_sha256(train_readout_path):
        raise ValueError("Retrieval head was trained with a different training readout artifact")
    head_train, head_validation = set(checkpoint.get("train_episode_ids", [])), set(checkpoint.get("validation_episode_ids", []))
    if head_train & head_validation or head_train | head_validation != set(cache.splits["train"]):
        raise ValueError("Retrieval-head training/selection episodes must stay inside the CTE training split")
    retriever = StaticTaskContextRetriever.load(bank_path, head_path, top_k=top_k, device=device,
                                                expected=expected_metadata)
    contexts, indices, scores = retrieve_episode_contexts(bank, retriever.head, query["episode_ids"], query["readouts"],
                                                         set(cache.splits["train"]), top_k=top_k, batch_size=batch_size)
    metadata = {**bank.metadata, "bank_sha256": checkpoint_sha256(bank_path),
                "retrieval_checkpoint_sha256": checkpoint_sha256(head_path),
                "readout_cache_sha256": checkpoint_sha256(train_readout_path),
                "query_readout_cache_sha256": checkpoint_sha256(query_readout_path),
                "top_k": top_k, "temperature": bank.temperature,
                "query_scope": "train_and_validation_initial_only",
                "train_query_exclusion": "leave_one_episode_out", "validation_query_sources": "train_only"}
    payload = {"format": STATIC_CONTEXT_FORMAT, "episode_ids": query["episode_ids"],
               "task_contexts": contexts, "retrieval_indices": indices, "retrieval_scores": scores,
               "episode_splits": cache.splits, "metadata": metadata}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.building-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        checked = StaticEpisodeContextCache(temporary)
        if set(checked.episode_ids) != set(cache.episodes):
            raise ValueError("Static context cache omitted held-out query episodes")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"episodes": len(contexts), "train_queries": len(cache.splits["train"]),
            "validation_queries": len(cache.splits["validation"]), "top_k": top_k,
            "static_cache_sha256": checkpoint_sha256(output_path)}
