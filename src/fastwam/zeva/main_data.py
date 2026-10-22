"""Success-demonstration provenance for Zeva main's observed-effect lineage.

Success is a property of the collection record, never inferred from a task
name, a condition, an action tensor or an absence of failure labels.  A source
without outcomes requires an explicit, recorded success-only declaration.
Deployment experience (including failures) does not pass through this offline
demonstration selector and is never an imitation-learning target here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar


MAIN_PROFILE = "main_observed"
MAIN_ARTIFACT_LINEAGE = "zeva_main_observed_effect_v1"
MAIN_MEMORY_SEMANTICS = "completed_effect_trace"
SUCCESS_MANIFEST_FORMAT = "zeva_success_demonstrations_v1"
MAIN_REFERENCE_COMMIT = "17ca9fa0c68dca1fc7915dd6433de70a100e93a1"
_IDENTITY_KEYS = (
    "manifest_sha256", "source_fingerprint", "success_episode_ids_sha256",
    "selected_episode_ids_sha256", "dataset_stats_sha256", "semantic_task_map_sha256",
    "split_sha256",
)
T = TypeVar("T")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def is_main_profile(cfg: Any) -> bool:
    return str(cfg.get("model", {}).get("zeva", {}).get("profile", "")) == MAIN_PROFILE


def normalize_main_dataset_paths(cfg: Any) -> None:
    """Use one source-qualified episode identity before constructing datasets.

    LeRobot and the FastWAM wrapper embed their configured root in episode
    IDs. Resolving the root only in a manifest would leave relative/symlink
    spellings in actual samples, breaking their joins to canonical split IDs.
    This changes no source row indices and leaves historical profiles alone.
    """
    if is_main_profile(cfg):
        cfg.data.train.dataset_dirs = [
            str(Path(str(root)).expanduser().resolve())
            for root in cfg.data.train.dataset_dirs
        ]


def canonical_episode_id(value: str) -> str:
    root, sep, index = str(value).rpartition("::episode-")
    if not sep or not root or not index.isdecimal():
        raise ValueError(f"expected a source-qualified episode id, got {value!r}")
    return f"{Path(root).expanduser().resolve()}::episode-{int(index)}"


def source_metadata(root: str | Path) -> dict[str, Any]:
    """Bind collection metadata, including the complete episode membership.

    This is explicitly a metadata/content-manifest identity, not a claim that
    every video byte was independently checksummed or that declared outcomes
    were verified in a simulator. Stats and the task map are bound separately.
    """
    root = Path(root).expanduser().resolve()
    required = ("meta/info.json", "meta/episodes.jsonl")
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"success provenance requires {root / relative}")
    metadata = {relative: _sha(root / relative) for relative in required}
    for relative in ("meta/tasks.jsonl", "meta/episodes_stats.jsonl", "meta/semantic_task_map.json"):
        if (root / relative).is_file():
            metadata[relative] = _sha(root / relative)
    records = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    ids = [int(row["episode_index"]) for row in records]
    if len(set(ids)) != len(ids) or any(value < 0 for value in ids):
        raise ValueError(f"invalid/duplicate episode indices in {root}")
    info = json.loads((root / "meta/info.json").read_text())
    if "total_episodes" in info and int(info["total_episodes"]) != len(ids):
        raise ValueError("info.json and episodes.jsonl disagree on episode count")
    return {"dataset_root": str(root), "metadata_sha256": metadata, "episode_indices": sorted(ids)}


def _bool_outcome(record: Mapping[str, Any]) -> bool | None:
    values = [record[key] for key in ("success", "is_success", "episode_success") if key in record]
    if "outcome" in record:
        outcome = record["outcome"]
        if outcome in ("success", "failure"):
            values.append(outcome == "success")
        else:
            raise ValueError("outcome must explicitly be 'success' or 'failure'")
    if not values:
        return None
    if any(type(value) is not bool for value in values) or len(set(values)) != 1:
        raise ValueError("success fields must be unambiguous JSON booleans")
    return values[0]


def build_success_manifest(
    dataset_roots: Iterable[str | Path], *, outcomes: Mapping[str, bool] | None = None,
    declare_success_only: bool = False, evidence: str = "",
) -> dict[str, Any]:
    """Construct an auditable selector; unknown/mixed outcomes fail closed."""
    roots = list(dataset_roots)
    if not roots:
        raise ValueError("at least one dataset root is required")
    if declare_success_only and (not evidence.strip() or outcomes is not None):
        raise ValueError("success-only declaration requires evidence and cannot override supplied outcomes")
    normalized_outcomes = None
    if outcomes is not None:
        normalized_outcomes = {canonical_episode_id(key): value for key, value in outcomes.items()}
        if len(normalized_outcomes) != len(outcomes) or any(type(value) is not bool for value in normalized_outcomes.values()):
            raise ValueError("outcomes must contain unique source-qualified IDs with JSON boolean success")
    sources, seen_ids = [], set()
    for root in roots:
        source = source_metadata(root)
        records = [json.loads(line) for line in (Path(source["dataset_root"]) / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        successes = []
        for record in records:
            episode_id = f"{source['dataset_root']}::episode-{int(record['episode_index'])}"
            if episode_id in seen_ids:
                raise ValueError("dataset roots must not repeat")
            seen_ids.add(episode_id)
            embedded = _bool_outcome(record)
            status = normalized_outcomes.get(episode_id) if normalized_outcomes is not None else embedded
            if normalized_outcomes is not None and embedded is not None and status != embedded:
                raise ValueError(f"outcome record contradicts collection metadata: {episode_id}")
            if declare_success_only:
                if embedded is False:
                    raise ValueError(f"cannot declare known failed demonstration successful: {episode_id}")
                status = True
            if status is None:
                raise ValueError(
                    f"unknown collection outcome for {episode_id}; provide outcomes JSON or an explicit "
                    "--declare-success-only --evidence declaration after checking the collection source"
                )
            if status:
                successes.append(int(record["episode_index"]))
        source["success_episode_indices"] = sorted(successes)
        sources.append(source)
    if normalized_outcomes is not None and set(normalized_outcomes) != seen_ids:
        raise ValueError("outcomes JSON must exactly cover the specified source datasets")
    if not any(source["success_episode_indices"] for source in sources):
        raise ValueError("source contains no successful demonstrations")
    return {
        "format": SUCCESS_MANIFEST_FORMAT, "sources": sources,
        "source_binding": "collection_metadata_sha256",
        "evidence": {"kind": "explicit_success_only_declaration" if declare_success_only else "episode_outcomes",
                     "description": evidence.strip(),
                     "outcomes_sha256": _digest(normalized_outcomes) if normalized_outcomes is not None else None},
    }


def load_success_selection(cfg: Any) -> tuple[set[str], dict[str, Any]]:
    zeva = cfg.get("model", {}).get("zeva", {})
    path = zeva.get("offline", {}).get("success_manifest_path")
    if path in (None, "", "None", "null"):
        raise ValueError("main_observed requires model.zeva.offline.success_manifest_path; no success outcome is inferred")
    path = Path(str(path)).expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("format") != SUCCESS_MANIFEST_FORMAT:
        raise ValueError("unsupported success-demonstration manifest")
    evidence = payload.get("evidence", {})
    if evidence.get("kind") not in {"episode_outcomes", "explicit_success_only_declaration"}:
        raise ValueError("success manifest requires explicit collection provenance")
    if evidence["kind"] == "explicit_success_only_declaration" and not str(evidence.get("description", "")).strip():
        raise ValueError("success-only declaration is missing its evidence/source description")
    sources = payload.get("sources", [])
    configured_roots = {str(Path(str(root)).expanduser().resolve()) for root in cfg.data.train.dataset_dirs}
    if len(sources) != len(configured_roots) or {source.get("dataset_root") for source in sources} != configured_roots:
        raise ValueError("success manifest and configured dataset roots differ")
    allowed: set[str] = set()
    for source in sources:
        actual = source_metadata(source["dataset_root"])
        for key in ("dataset_root", "metadata_sha256", "episode_indices"):
            if source.get(key) != actual[key]:
                raise ValueError(f"success manifest source changed: {source['dataset_root']} ({key})")
        selected = source.get("success_episode_indices", [])
        if not isinstance(selected, list) or len(selected) != len(set(selected)) or not set(selected) <= set(actual["episode_indices"]):
            raise ValueError("success manifest has invalid episode membership")
        allowed.update(f"{source['dataset_root']}::episode-{int(index)}" for index in selected)
    if not allowed:
        raise ValueError("success manifest selects no demonstrations")
    stats = cfg.data.train.get("pretrained_norm_stats")
    semantic = cfg.data.train.get("semantic_task_map_path")
    if not stats or not semantic:
        raise ValueError("main success data requires normalization statistics and canonical task map")
    identity = {
        "manifest_sha256": _sha(path), "source_fingerprint": _digest(sources),
        "success_episode_ids_sha256": _digest(sorted(allowed)),
        "num_success_episodes": len(allowed), "evidence_kind": evidence["kind"],
        "source_binding": "collection_metadata_sha256",
        "dataset_stats_sha256": _sha(str(stats)), "semantic_task_map_sha256": _sha(str(semantic)),
    }
    return allowed, identity


def select_main_success_rows(rows: Iterable[T], cfg: Any) -> tuple[list[T], dict[str, Any]]:
    """Filter training indices or cache episode plans, retaining source indices."""
    rows = list(rows)
    if not is_main_profile(cfg):
        return rows, {}
    if not bool(cfg.data.train.get("is_training_set", True)):
        raise ValueError("main offline training/cache must use the training episode split")
    allowed, identity = load_success_selection(cfg)
    selected = [row for row in rows if canonical_episode_id(
        str(row["episode_id"] if isinstance(row, Mapping) else getattr(row, "episode_id"))
    ) in allowed]
    episode_ids = sorted({canonical_episode_id(str(
        row["episode_id"] if isinstance(row, Mapping) else getattr(row, "episode_id")
    )) for row in selected})
    if not episode_ids:
        raise ValueError("no complete successful demonstrations remain in the training split")
    task_by_episode: dict[str, str] = {}
    for row in selected:
        row_id = canonical_episode_id(str(row["episode_id"] if isinstance(row, Mapping) else getattr(row, "episode_id")))
        task_id = str(row["task_id"] if isinstance(row, Mapping) else getattr(row, "task_id"))
        if row_id in task_by_episode and task_by_episode[row_id] != task_id:
            raise ValueError("main split requires one canonical semantic task per episode")
        task_by_episode[row_id] = task_id
    offline = cfg.get("model", {}).get("zeva", {}).get("offline", {})
    fraction = float(offline.get("validation_fraction", 0.1))
    split_seed = int(offline.get("split_seed", 42))
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("main offline validation_fraction must lie strictly between 0 and 1")
    train_ids, validation_ids = [], []
    for task_id in sorted(set(task_by_episode.values())):
        task_episodes = [episode_id for episode_id, task in task_by_episode.items() if task == task_id]
        if len(task_episodes) < 2:
            raise ValueError(f"main holdout needs at least 2 successful complete episodes for task {task_id!r}")
        ordered = sorted(task_episodes, key=lambda episode_id: _digest([split_seed, task_id, episode_id]))
        count = max(1, min(len(ordered) - 1, math.ceil(len(ordered) * fraction)))
        validation_ids.extend(ordered[:count])
        train_ids.extend(ordered[count:])
    splits = {"format": "zeva_main_success_episode_split_v1", "seed": split_seed,
              "validation_fraction": fraction, "train": sorted(train_ids), "validation": sorted(validation_ids)}
    identity.update({"selected_episode_ids_sha256": _digest(episode_ids), "num_selected_episodes": len(episode_ids)})
    identity["split_sha256"] = _digest(splits)
    metadata = {
        "profile": MAIN_PROFILE, "artifact_lineage": MAIN_ARTIFACT_LINEAGE,
        "memory_semantics": MAIN_MEMORY_SEMANTICS, "success_data_identity": identity,
        "episode_splits": splits, "data_role": "all",
    }
    validate_main_artifact(metadata)
    return selected, metadata


def validate_main_artifact(payload: Mapping[str, Any], expected: Mapping[str, Any] | None = None) -> None:
    """Reject predictive/legacy artifacts when selecting the observed main path."""
    required = {"profile": MAIN_PROFILE, "artifact_lineage": MAIN_ARTIFACT_LINEAGE,
                "memory_semantics": MAIN_MEMORY_SEMANTICS}
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"main artifact {key} mismatch; rebuild the observed-effect main artifacts")
    identity = payload.get("success_data_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("main artifact is missing success-demonstration provenance")
    for key in _IDENTITY_KEYS:
        digest = identity.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid main success-data identity {key}")
    if (type(identity.get("num_selected_episodes")) is not int or identity["num_selected_episodes"] < 1
            or type(identity.get("num_success_episodes")) is not int
            or identity["num_success_episodes"] < identity["num_selected_episodes"]):
        raise ValueError("invalid main success episode counts")
    if identity.get("evidence_kind") not in {"episode_outcomes", "explicit_success_only_declaration"}:
        raise ValueError("main success data requires outcome evidence or an explicit source declaration")
    splits = payload.get("episode_splits")
    if not isinstance(splits, Mapping) or splits.get("format") != "zeva_main_success_episode_split_v1":
        raise ValueError("main artifact is missing its fixed episode-disjoint holdout")
    train, validation = splits.get("train"), splits.get("validation")
    if (not isinstance(train, list) or not train or not isinstance(validation, list) or not validation
            or len(set(train)) != len(train) or len(set(validation)) != len(validation)
            or set(train) & set(validation)):
        raise ValueError("main artifact has empty, duplicate or overlapping episode splits")
    if (any(canonical_episode_id(value) != value for value in train + validation)
            or _digest(sorted(train + validation)) != identity["selected_episode_ids_sha256"]
            or len(train) + len(validation) != identity["num_selected_episodes"]
            or _digest(dict(splits)) != identity["split_sha256"]):
        raise ValueError("main episode splits do not match the bound success-data identity")
    if payload.get("data_role") not in {"train", "validation", "all"}:
        raise ValueError("main artifact must declare train/validation/all data_role")
    if expected is not None:
        validate_main_artifact(expected)
        if dict(identity) != dict(expected["success_data_identity"]):
            raise ValueError("main artifacts use different success data, split, statistics or task mapping")


def inherit_main_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Carry provenance through derived banks/caches; do not upgrade old files."""
    if payload.get("profile") != MAIN_PROFILE:
        return {}
    validate_main_artifact(payload)
    return {key: payload[key] for key in (
        "profile", "artifact_lineage", "memory_semantics", "success_data_identity", "episode_splits", "data_role",
    )}


def validate_main_episode_ids(payload: Mapping[str, Any], episode_ids: Iterable[str]) -> None:
    """Verify that an artifact really contains the selected success split."""
    if payload.get("profile") != MAIN_PROFILE:
        return
    validate_main_artifact(payload)
    actual = sorted({canonical_episode_id(value) for value in episode_ids})
    identity = payload["success_data_identity"]
    role = payload["data_role"]
    splits = payload["episode_splits"]
    expected_ids = sorted(splits["train"] + splits["validation"] if role == "all" else splits[role])
    if actual != expected_ids:
        raise ValueError("main artifact episode membership differs from its successful training split")


def main_split_rows(rows: Iterable[T], metadata: Mapping[str, Any], split: str) -> list[T]:
    """Select an already established holdout; never repartition a filtered bank."""
    validate_main_artifact(metadata)
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    allowed = set(metadata["episode_splits"][split])
    return [row for row in rows if canonical_episode_id(str(
        row["episode_id"] if isinstance(row, Mapping) else getattr(row, "episode_id")
    )) in allowed]


def validate_main_source(cfg: Any, metadata: Mapping[str, Any]) -> None:
    """Validate source identity of an existing partition without splitting it again."""
    validate_main_artifact(metadata)
    _allowed, actual = load_success_selection(cfg)
    expected = metadata["success_data_identity"]
    if any(actual[key] != expected[key] for key in actual):
        raise ValueError("main artifact source, success declaration, statistics or task map changed")
    offline = cfg.get("model", {}).get("zeva", {}).get("offline", {})
    splits = metadata["episode_splits"]
    if (int(offline.get("split_seed", 42)) != splits["seed"]
            or float(offline.get("validation_fraction", 0.1)) != splits["validation_fraction"]):
        raise ValueError("main artifact holdout configuration changed; rebuild dependent artifacts")
