"""Compact complete-history PIM cache, with train-only episode provenance.

Each bounded source episode is stored once. Query rows retain only causal
phase/BIT inputs; whole historical episodes are packed on demand, avoiding a
192-entry copy for every query window. No success labels or actions are stored.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


CROSS_ATTEMPT_CACHE_FORMAT = "zeva_stage2_cross_attempt_pim_v1"
LEGACY_EGO_CROSS_ATTEMPT_CACHE_FORMAT = "zeva_stage2_cross_attempt_pim_v2"
LEGACY_REFERENCE_EGO_CROSS_ATTEMPT_CACHE_FORMAT = "zeva_stage2_cross_attempt_pim_v3"
EGO_CROSS_ATTEMPT_CACHE_FORMAT = "zeva_stage2_cross_attempt_pim_v4"
CROSS_ATTEMPT_CACHE_FORMATS = (CROSS_ATTEMPT_CACHE_FORMAT, LEGACY_EGO_CROSS_ATTEMPT_CACHE_FORMAT,
                             LEGACY_REFERENCE_EGO_CROSS_ATTEMPT_CACHE_FORMAT, EGO_CROSS_ATTEMPT_CACHE_FORMAT)
ALL_CTE_FRAMES = "all_cte_frames_v1"
COMPLETE_POLICY_WINDOWS = "complete_policy_windows_v1"
REFERENCE_PAIRING_METRIC = "torch_float32_first_in_source_order"
LEGACY_PAIRING_METRIC = "numpy_float32_episode_id_v1"
HISTORY_KINDS = ("matched", "far", "cross_condition", "empty")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_episode_splits(splits: dict[str, list[str]], available: set[str]) -> dict[str, list[str]]:
    if not isinstance(splits, dict) or "train" not in splits or "validation" not in splits:
        raise ValueError("episode splits require train and validation lists")
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    for name, ids in splits.items():
        if name not in {"train", "validation", "test"} or not isinstance(ids, list):
            raise ValueError("unsupported split name or non-list episode IDs")
        ids = [str(value) for value in ids]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("episode splits overlap or contain duplicate IDs")
        if set(ids) - available:
            raise ValueError(f"split {name} contains unknown episode IDs")
        seen.update(ids)
        result[name] = sorted(ids)
    if not result["train"]:
        raise ValueError("at least one train episode is required")
    return result


def build_episode_pairings(
    queries: dict[str, dict[str, Any]],
    histories: list[dict[str, Any]],
    *,
    max_histories: int,
    use_conditions: bool,
    history_pairing: str = "legacy_candidates",
    allow_reduced_curriculum: bool = True,
    pairing_metric: str = LEGACY_PAIRING_METRIC,
) -> dict[str, dict[str, list[int]]]:
    """Pair initial phases only; later query outcomes/actions cannot affect it."""
    if pairing_metric not in {REFERENCE_PAIRING_METRIC, LEGACY_PAIRING_METRIC}:
        raise ValueError("unknown PIM pairing_metric")
    if max_histories < 1:
        raise ValueError("max_histories must be positive")
    if history_pairing not in {"reference_nearest", "legacy_candidates"}:
        raise ValueError("unknown PIM history_pairing")
    if not allow_reduced_curriculum and not use_conditions:
        raise ValueError("Reference PIM curriculum requires real condition metadata")
    if any(row.get("split") != "train" for row in histories):
        raise ValueError("PIM history sources must be train-only")
    if len({row["episode_id"] for row in histories}) != len(histories):
        raise ValueError("duplicate history episode IDs")
    by_task: dict[str, list[tuple[int, dict[str, Any], np.ndarray]]] = defaultdict(list)
    for index, history in enumerate(histories):
        key = np.asarray(history["initial_phase"], dtype=np.float32)
        if not np.isfinite(key).all():
            raise ValueError("initial phase must be finite")
        if pairing_metric == LEGACY_PAIRING_METRIC:
            key = key / max(float(np.linalg.norm(key)), 1e-12)
        by_task[history["task_id"]].append((index, history, key))
    result: dict[str, dict[str, list[int]]] = {}
    for episode_id, query in queries.items():
        q = np.asarray(query["initial_phase"], dtype=np.float32)
        if not np.isfinite(q).all():
            raise ValueError("initial phase must be finite")
        if pairing_metric == LEGACY_PAIRING_METRIC:
            q = q / max(float(np.linalg.norm(q)), 1e-12)
        else:
            # Use the same float32 normalize + matrix-vector product as the
            # reference builder. Source order, not episode names, breaks ties.
            candidates = [(i, h, k) for i, h, k in by_task[query["task_id"]] if h["episode_id"] != episode_id]
            scores = {}
            for is_cross in (False, True):
                group = [(i, k) for i, h, k in candidates
                         if bool(use_conditions and h["condition"] != query["condition"]) == is_cross]
                if group:
                    keys = torch.from_numpy(np.stack([key for _, key in group])).float()
                    values = torch.nn.functional.normalize(keys, dim=-1) @ torch.nn.functional.normalize(torch.from_numpy(q).float(), dim=-1)
                    scores.update({i: float(value) for (i, _), value in zip(group, values, strict=True)})
        same: list[tuple[float, str, int]] = []
        cross: list[tuple[float, str, int]] = []
        for index, history, key in by_task[query["task_id"]]:
            if history["episode_id"] == episode_id:
                continue
            item = ((float(q @ key) if pairing_metric == LEGACY_PAIRING_METRIC else scores[index]),
                    str(history["episode_id"]) if pairing_metric == LEGACY_PAIRING_METRIC else index, index)
            if use_conditions and history["condition"] != query["condition"]:
                cross.append(item)
            else:
                same.append(item)
        matched = sorted(same, key=lambda item: (-item[0], item[1]))
        far = sorted(same, key=lambda item: (item[0], item[1]))
        across = sorted(cross, key=lambda item: (-item[0], item[1]))
        if not matched and not across:
            raise ValueError(f"no other complete train episode for query {episode_id}")
        if not allow_reduced_curriculum and (not matched or not across):
            raise ValueError(f"Reference PIM curriculum needs same- and cross-condition train histories for {episode_id}")
        keep = 1 if history_pairing == "reference_nearest" else max_histories
        result[episode_id] = {
            "matched": [item[2] for item in matched[:keep]],
            "far": [item[2] for item in far[:keep]],
            "cross_condition": [item[2] for item in across[:keep]],
        }
    return result


class CrossAttemptPIMCache:
    """PreparedPIMCache-compatible reader for complete historical attempts."""

    def __init__(self, root: str | Path, *, expected: dict[str, Any] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") not in CROSS_ATTEMPT_CACHE_FORMATS:
            raise ValueError("not a cross-attempt PIM cache")
        for key, value in dict(expected or {}).items():
            if self.manifest.get(key) != value:
                raise ValueError(f"prepared PIM identity mismatch for {key}: expected={value!r}, actual={self.manifest.get(key)!r}")
        if self.manifest.get("pim_variant") != "cross_attempt" or self.manifest.get("source_subset") != "train-only":
            raise ValueError("cross-attempt cache requires explicit train-only provenance")
        self.current_prediction = self.manifest["format"] in {
            EGO_CROSS_ATTEMPT_CACHE_FORMAT, LEGACY_REFERENCE_EGO_CROSS_ATTEMPT_CACHE_FORMAT, LEGACY_EGO_CROSS_ATTEMPT_CACHE_FORMAT}
        self.query_scope = self.manifest.get("query_scope", COMPLETE_POLICY_WINDOWS)
        self.pairing_metric = self.manifest.get("pairing_metric", LEGACY_PAIRING_METRIC)
        if self.manifest["format"] == EGO_CROSS_ATTEMPT_CACHE_FORMAT:
            if (self.query_scope != ALL_CTE_FRAMES or self.pairing_metric != REFERENCE_PAIRING_METRIC
                    or self.manifest.get("source_export_scope") != ALL_CTE_FRAMES):
                raise ValueError("PIM cache v4 requires all CTE frames and reference float32/source-order pairing")
            if not {"query_scope", "pairing_metric"} <= self.manifest.keys():
                raise ValueError("PIM cache v4 requires explicit query/pairing contracts")
        elif self.query_scope != COMPLETE_POLICY_WINDOWS or self.pairing_metric != LEGACY_PAIRING_METRIC:
            raise ValueError("Legacy PIM cache query/pairing semantics cannot be relabeled")
        self.uniform_indexing = self.manifest.get("uniform_indexing", "legacy_floor")
        self.history_pairing = self.manifest.get("history_pairing", "legacy_candidates")
        self.allow_reduced_curriculum = self.manifest.get("allow_reduced_curriculum", True)
        if self.history_pairing not in {"reference_nearest", "legacy_candidates"}:
            raise ValueError("unknown PIM history_pairing")
        if type(self.allow_reduced_curriculum) is not bool:
            raise ValueError("allow_reduced_curriculum must be boolean")
        if self.manifest["format"] in {EGO_CROSS_ATTEMPT_CACHE_FORMAT, LEGACY_REFERENCE_EGO_CROSS_ATTEMPT_CACHE_FORMAT}:
            if (self.uniform_indexing != "reference_round"
                    or self.manifest.get("history_compression") != "uniform"):
                raise ValueError("PIM reference cache requires explicit uniform/reference_round history compression")
            if "history_pairing" not in self.manifest or "allow_reduced_curriculum" not in self.manifest:
                raise ValueError("PIM reference cache must declare its pairing and curriculum policy")
        elif self.uniform_indexing != "legacy_floor":
            raise ValueError("Legacy PIM cache formats require legacy_floor indexing")
        elif self.history_pairing != "legacy_candidates":
            raise ValueError("Legacy PIM cache formats require legacy_candidates pairing")
        if self.current_prediction:
            for key, required in {"memory_semantics": "predicted_effect_trace", "effect_source": "current_prediction",
                                  "executed_horizon": 24, "cte_input_type": "rgb_multiview"}.items():
                if self.manifest.get(key) != required:
                    raise ValueError(f"predicted-effect cache semantic mismatch: {key}")
        elif self.manifest.get("memory_semantics", "completed_effect_trace") != "completed_effect_trace":
            raise ValueError("legacy cross-attempt cache cannot contain predicted-effect semantics")
        self.phase_dim = int(self.manifest["phase_dim"])
        self.effect_dim = int(self.manifest["effect_dim"])
        self.bit_size = int(self.manifest["bit_size"])
        self.max_histories = int(self.manifest["max_past_attempts"])
        self.max_entries = int(self.manifest["max_entries_per_attempt"])
        self.top_k = int(self.manifest["top_k"])
        self.num_queries = int(self.manifest["num_queries"])
        self.retrieval_mode = "cross_attempt"
        if min(self.phase_dim, self.effect_dim, self.bit_size, self.max_histories, self.max_entries, self.num_queries) < 1:
            raise ValueError("invalid cross-attempt cache dimensions")
        if self.top_k != self.max_histories * self.max_entries:
            raise ValueError("cross-attempt candidate capacity does not match history limits")
        self.split = str(self.manifest["split"])
        if self.split not in {"train", "validation"}:
            raise ValueError("query cache split must be train or validation")
        self.seed = int(self.manifest.get("seed", 42))
        self.use_conditions = bool(self.manifest["use_conditions"])
        if self.current_prediction and not self.allow_reduced_curriculum and not self.use_conditions:
            raise ValueError("Reference PIM curriculum requires real condition metadata")

        def read_metadata(name: str):
            path = self.root / name
            expected_sha = self.manifest.get("metadata_sha256", {}).get(name)
            if not expected_sha or file_sha256(path) != expected_sha:
                raise ValueError(f"cross-attempt metadata checksum mismatch: {name}")
            return json.loads(path.read_text())

        self.episode_ids = tuple(read_metadata("episode_ids.json"))
        self.task_ids = tuple(read_metadata("task_ids.json"))
        self.query_episodes = read_metadata("query_episodes.json")
        self.history_episodes = read_metadata("history_episodes.json")
        self.pairings = read_metadata("pairings.json")
        splits = read_metadata("episode_splits.json")
        all_ids = set().union(*(set(ids) for ids in splits.values()))
        self.splits = validate_episode_splits(splits, all_ids)
        if len(set(self.episode_ids)) != len(self.episode_ids) or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("duplicate episode/task IDs in cross-attempt cache")
        if set(self.episode_ids) != set(self.query_episodes) or set(self.pairings) != set(self.episode_ids):
            raise ValueError("cross-attempt query metadata coverage mismatch")
        if len(self.episode_ids) != int(self.manifest["num_episodes"]):
            raise ValueError("cross-attempt query episode count mismatch")
        if len(self.history_episodes) != int(self.manifest["num_history_episodes"]):
            raise ValueError("cross-attempt source episode count mismatch")
        if len(self.task_ids) != int(self.manifest["semantic_task_count"]):
            raise ValueError("cross-attempt task count mismatch")
        train_ids = set(self.splits["train"])
        query_ids = set(self.splits[self.split])
        history_ids = [row["episode_id"] for row in self.history_episodes]
        if len(history_ids) != len(set(history_ids)):
            raise ValueError("duplicate history source episodes")
        if set(history_ids) - train_ids:
            raise ValueError("history source outside training split")
        for episode_id, query in self.query_episodes.items():
            if episode_id not in query_ids or query["split"] != self.split:
                raise ValueError("query episode has wrong split")
            if query["task_id"] not in self.task_ids:
                raise ValueError("unknown query task")
            if set(self.pairings[episode_id]) != set(HISTORY_KINDS[:3]):
                raise ValueError("cross-attempt pairing kinds are incomplete")
            for kind, indices in self.pairings[episode_id].items():
                if kind not in HISTORY_KINDS[:3] or len(indices) > self.max_histories or len(set(indices)) != len(indices):
                    raise ValueError("invalid cross-attempt pairing")
                if self.history_pairing == "reference_nearest" and len(indices) > 1:
                    raise ValueError("reference_nearest must select exactly one fixed partner per available mode")
                for index in indices:
                    if not isinstance(index, int) or not 0 <= index < len(self.history_episodes):
                        raise ValueError("history source index out of range")
                    source = self.history_episodes[index]
                    if source["split"] != "train" or source["episode_id"] == episode_id or source["task_id"] != query["task_id"]:
                        raise ValueError("history leakage: sources must be other same-task train episodes")
                    if self.use_conditions:
                        same = source["condition"] == query["condition"]
                        if same == (kind == "cross_condition"):
                            raise ValueError("history condition does not match pairing kind")
                    elif kind == "cross_condition" and indices:
                        raise ValueError("cross-condition pairing requires actual condition metadata")
        if self.current_prediction:
            global_kinds = [kind for kind in HISTORY_KINDS[:3] if all(row[kind] for row in self.pairings.values())]
            if not global_kinds or self.manifest.get("globally_available_history_kinds") != global_kinds:
                raise ValueError("globally available history kinds do not match pairing provenance")
            reduced = global_kinds != list(HISTORY_KINDS[:3])
            if self.manifest["format"] in {EGO_CROSS_ATTEMPT_CACHE_FORMAT, LEGACY_REFERENCE_EGO_CROSS_ATTEMPT_CACHE_FORMAT} and self.manifest.get("reduced_curriculum") is not reduced:
                raise ValueError("PIM cache reduced_curriculum does not match its available history modes")
            if reduced and not self.allow_reduced_curriculum:
                raise ValueError("Reference PIM curriculum requires all three history modes for every query")
            self.reduced_curriculum = reduced
        else:
            self.reduced_curriculum = None
        self.weights = {kind: float(self.manifest["history_weights"].get(kind, 0.0)) for kind in HISTORY_KINDS}
        if any(not np.isfinite(weight) or weight < 0 for weight in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("invalid history mixture weights")
        for episode_id in self.episode_ids:
            available_weight = self.weights["empty"] + sum(self.weights[k] for k in HISTORY_KINDS[:3] if self.pairings[episode_id].get(k))
            if available_weight <= 0:
                raise ValueError(f"no eligible weighted history for {episode_id}")
        self._arrays: dict[str, np.ndarray] = {}
        n = self.num_queries
        specs = {
            "window_index": (np.int64, (n,)), "raw_step": (np.int32, (n,)),
            "episode_index": (np.int32, (n,)), "task_index": (np.int16, (n,)),
            "phase": (np.float32, (n, self.phase_dim)),
            "bit_effects": (np.float32, (n, self.bit_size, self.effect_dim)),
            "bit_mask": (np.uint8, (n, self.bit_size)),
            "history_phases": (np.float32, (len(history_ids), self.max_entries, self.phase_dim)),
            "history_effects": (np.float32, (len(history_ids), self.max_entries, self.effect_dim)),
            "history_mask": (np.uint8, (len(history_ids), self.max_entries)),
        }
        if self.current_prediction:
            specs["current_effect"] = (np.float32, (n, self.effect_dim))
            specs["current_effect_mask"] = (np.uint8, (n,))
        for name, (dtype, shape) in specs.items():
            path = self.root / f"{name}.npy"
            expected_sha = self.manifest.get("array_sha256", {}).get(path.name)
            if not expected_sha or file_sha256(path) != expected_sha:
                raise ValueError(f"cross-attempt tensor checksum mismatch: {path.name}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != dtype or array.shape != shape:
                raise ValueError(f"cross-attempt array shape/dtype mismatch: {name}")
            self._arrays[name] = array
        if self.query_scope == ALL_CTE_FRAMES:
            seen_queries = set()
            for index in range(n):
                ep = int(self._arrays["episode_index"][index])
                if not 0 <= ep < len(self.episode_ids):
                    raise ValueError("all-frame PIM query episode index is invalid")
                episode_id = self.episode_ids[ep]
                meta = self.query_episodes[episode_id]
                raw = int(self._arrays["raw_step"][index])
                if (raw < 0 or raw >= int(meta.get("episode_length", 0)) or raw % 24
                        or int(self._arrays["window_index"][index]) != int(meta.get("dataset_start", -1)) + raw
                        or (episode_id, raw) in seen_queries):
                    raise ValueError("all-frame PIM query differs from its source boundary/window provenance")
                seen_queries.add((episode_id, raw))
            expected_queries = {(episode_id, raw) for episode_id, meta in self.query_episodes.items()
                                for raw in range(0, ((int(meta["episode_length"]) - 1) // 24) * 24 + 1, 24)}
            if seen_queries != expected_queries:
                raise ValueError("all-frame PIM cache must retain every observed CTE frame, including padded tails")
        if not np.all(np.asarray(self._arrays["history_mask"]).sum(axis=1) > 0):
            raise ValueError("history catalog contains an empty source episode")
        for name in (("history_mask", "bit_mask", "current_effect_mask") if self.current_prediction else ("history_mask", "bit_mask")):
            if not np.all((self._arrays[name] == 0) | (self._arrays[name] == 1)):
                raise ValueError(f"non-binary cross-attempt mask: {name}")

    def __len__(self) -> int:
        return self.num_queries

    def validate_training_curriculum(self, *, history_pairing: str, allow_reduced_curriculum: bool,
                                     query_scope: str | None = None, pairing_metric: str | None = None) -> None:
        """Require the training config to opt into any non-reference reduction."""
        if not self.current_prediction:
            raise ValueError("predictive curriculum requires a predicted-effect PIM cache")
        if query_scope is not None and query_scope != self.query_scope:
            raise ValueError("prepared PIM query_scope differs from the selected training recipe")
        if pairing_metric is not None and pairing_metric != self.pairing_metric:
            raise ValueError("prepared PIM pairing_metric differs from the selected training recipe")
        if history_pairing != self.history_pairing:
            raise ValueError("prepared PIM history_pairing differs from the selected training recipe")
        if type(allow_reduced_curriculum) is not bool:
            raise ValueError("allow_reduced_curriculum must be boolean")
        if self.reduced_curriculum and not allow_reduced_curriculum:
            raise ValueError("Reduced PIM curriculum requires explicit prepared_stage2.allow_reduced_curriculum=true")

    def get(self, index: int, *, history_kind: str | None = None, history_count: int | None = None,
            generator: torch.Generator | None = None) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        arrays = self._arrays
        ep_index, task_index = int(arrays["episode_index"][index]), int(arrays["task_index"][index])
        if not 0 <= ep_index < len(self.episode_ids) or not 0 <= task_index < len(self.task_ids):
            raise ValueError("cross-attempt query index out of range")
        episode_id, task_id = self.episode_ids[ep_index], self.task_ids[task_index]
        if self.query_episodes[episode_id]["task_id"] != task_id:
            raise ValueError("cross-attempt query task mismatch")
        # Training uses the worker's seeded torch RNG and varies by visit.
        # Validation uses a local generator, independent of visit/worker order.
        if generator is None and self.split == "validation":
            generator = torch.Generator().manual_seed(self.seed + index)
        pairings = self.pairings[episode_id]
        if history_kind is None:
            weights = torch.tensor([self.weights[k] if k == "empty" or pairings.get(k) else 0.0 for k in HISTORY_KINDS])
            history_kind = HISTORY_KINDS[int(torch.multinomial(weights, 1, generator=generator).item())]
        if history_kind not in HISTORY_KINDS:
            raise ValueError("unknown history kind")
        candidates = pairings.get(history_kind, [])
        selected: list[int] = []
        if history_kind != "empty":
            if not candidates:
                raise ValueError(f"no {history_kind} history for {episode_id}")
            count = int(history_count) if history_count is not None else int(torch.randint(1, len(candidates) + 1, (), generator=generator).item())
            if not 1 <= count <= min(self.max_histories, len(candidates)):
                raise ValueError("invalid requested history count")
            order = (list(range(count)) if self.history_pairing == "reference_nearest"
                     else torch.randperm(len(candidates), generator=generator)[:count].tolist())
            selected = [candidates[i] for i in order]
        phases = np.zeros((self.top_k, self.phase_dim), dtype=np.float32)
        effects = np.zeros((self.top_k, self.effect_dim), dtype=np.float32)
        mask = np.zeros(self.top_k, dtype=np.bool_)
        for slot, source_index in enumerate(selected):
            sl = slice(slot * self.max_entries, (slot + 1) * self.max_entries)
            phases[sl] = arrays["history_phases"][source_index]
            effects[sl] = arrays["history_effects"][source_index]
            mask[sl] = arrays["history_mask"][source_index].astype(bool)
        phase = np.array(arrays["phase"][index], copy=True)
        bit_effects = np.array(arrays["bit_effects"][index], copy=True)
        if not all(np.isfinite(value).all() for value in (phases, effects, phase, bit_effects)):
            raise ValueError("non-finite cross-attempt conditioning")
        norms = np.linalg.norm(phases, axis=1) * max(float(np.linalg.norm(phase)), 1e-12)
        scores = (phases @ phase) / np.maximum(norms, 1e-12)
        scores[~mask] = -np.inf
        result = {
            "window_index": int(arrays["window_index"][index]), "raw_step": int(arrays["raw_step"][index]),
            "episode_id": episode_id, "task_id": task_id,
            "phase": torch.from_numpy(phase), "bit_effects": torch.from_numpy(bit_effects),
            "bit_mask": torch.from_numpy(np.array(arrays["bit_mask"][index], copy=True)).bool(),
            "pim_phases": torch.from_numpy(phases), "pim_effects": torch.from_numpy(effects),
            "pim_mask": torch.from_numpy(mask), "pim_scores": torch.from_numpy(scores),
            "pim_history_kind": torch.tensor(HISTORY_KINDS.index(history_kind), dtype=torch.long),
            "pim_history_count": torch.tensor(len(selected), dtype=torch.long),
            "pim_nonempty_history_weight": torch.tensor(sum(self.weights[k] for k in HISTORY_KINDS[:3] if pairings.get(k)), dtype=torch.float32),
            "pim_source_episode_ids": [self.history_episodes[i]["episode_id"] for i in selected],
        }
        if self.current_prediction:
            effect = np.array(arrays["current_effect"][index], copy=True)
            if not np.isfinite(effect).all():
                raise ValueError("non-finite current predicted effect")
            result["current_effect"] = torch.from_numpy(effect)
            result["current_effect_mask"] = torch.tensor(bool(arrays["current_effect_mask"][index]))
        return result

    def get_history_candidates(self, index: int) -> dict[str, Any]:
        """Bundle one prior attempt per mode for optimizer-step selection.

        No training-step state lives in dataset workers: prefetch cannot move
        the PIM-only warmup boundary or independently choose microbatch modes.
        """
        if not self.current_prediction:
            raise ValueError("optimizer-step history schedule requires a predicted-effect cache (v2 legacy or v3 reference)")
        result = self.get(index, history_kind="empty")
        kinds = self.manifest["globally_available_history_kinds"]
        bundles = {"pim_history_phases": [], "pim_history_effects": [],
                   "pim_history_masks": [], "pim_history_scores": []}
        available = []
        for kind_index, kind in enumerate(HISTORY_KINDS[:3]):
            enabled = kind in kinds
            # The reference cache has one fixed partner per episode/mode.
            # Legacy top-N candidates retain their query-local seeded draw.
            # Both stay stable across workers, prefetch depth and resume.
            generator = torch.Generator().manual_seed(self.seed + 3 * int(index) + kind_index)
            candidate = self.get(index, history_kind=kind, history_count=1, generator=generator) if enabled else result
            available.append(enabled)
            for target, source in (("pim_history_phases", "pim_phases"), ("pim_history_effects", "pim_effects"),
                                   ("pim_history_masks", "pim_mask"), ("pim_history_scores", "pim_scores")):
                bundles[target].append(candidate[source])
        result.update({name: torch.stack(values) for name, values in bundles.items()})
        result["pim_history_available"] = torch.tensor(available, dtype=torch.bool)
        return result
