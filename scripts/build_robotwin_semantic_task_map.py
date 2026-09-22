"""Build a strict episode -> canonical RoboTwin task mapping.

The converted RoboTwin LeRobot dataset can contain many natural-language task
strings (and therefore many ``task_index`` values) even though RoboTwin has a
small canonical task set.  Zeva CTE/PIM must use the canonical task identity,
not the language-string index.

This builder intentionally refuses fuzzy/LLM guessing.  It derives canonical
identity from task-cache provenance (preferred), direct episode metadata when
present, and optional user-supplied overrides.  Training should start only
when every episode used by the dataset is mapped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from fastwam.zeva.semantic_tasks import (
    ROBOTWIN_CANONICAL_TASKS,
    SemanticTaskMap,
    normalize_task_name,
    normalize_text,
)

CANONICAL = frozenset(ROBOTWIN_CANONICAL_TASKS)


def _canonical_from_path(path: Path) -> str | None:
    candidates: list[str] = []
    for part in path.parts:
        value = normalize_task_name(Path(part).stem)
        if value in CANONICAL:
            candidates.append(value)
        # Accept filenames such as ``beat_block_hammer_tasks.json``.
        for task in ROBOTWIN_CANONICAL_TASKS:
            if value == task or value.startswith(task + "_") or value.endswith("_" + task):
                candidates.append(task)
    unique = sorted(set(candidates))
    if len(unique) > 1:
        raise ValueError(f"ambiguous canonical task in path {path}: {unique}")
    return unique[0] if unique else None


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_strings(item)


def _read_strings(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    values: list[str] = []
    try:
        if suffix == ".json":
            values.extend(_iter_strings(json.loads(path.read_text(encoding="utf-8"))))
        elif suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    values.extend(_iter_strings(json.loads(line)))
        elif suffix in {".txt", ".md"}:
            values.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError:
                return values
            values.extend(_iter_strings(yaml.safe_load(path.read_text(encoding="utf-8"))))
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    values.extend(_iter_strings(row))
        elif suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            for column in df.columns:
                if df[column].dtype == object:
                    for item in df[column].dropna().tolist():
                        values.extend(_iter_strings(item))
    except Exception as exc:
        raise RuntimeError(f"failed to parse task-cache file {path}: {exc}") from exc
    return values


def _load_tasks_table(dataset_root: Path) -> tuple[dict[int, str], dict[str, int]]:
    import pandas as pd

    candidates = [
        dataset_root / "meta" / "tasks.parquet",
        dataset_root / "meta" / "tasks.jsonl",
    ]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            f"cannot find meta/tasks.parquet or meta/tasks.jsonl under {dataset_root}"
        )

    if source.suffix == ".parquet":
        df = pd.read_parquet(source)
    else:
        df = pd.read_json(source, lines=True)

    if "task" not in df.columns:
        if getattr(df.index, "name", None) == "task":
            df = df.reset_index()
        elif "instruction" in df.columns:
            df = df.rename(columns={"instruction": "task"})
        else:
            object_columns = [c for c in df.columns if df[c].dtype == object]
            if len(object_columns) == 1:
                df = df.rename(columns={object_columns[0]: "task"})
            else:
                raise ValueError(
                    f"unable to identify task text column in {source}; columns={list(df.columns)}"
                )

    if "task_index" not in df.columns:
        raise ValueError(f"{source} has no task_index column")

    by_index: dict[int, str] = {}
    by_text: dict[str, int] = {}
    for row in df[["task_index", "task"]].itertuples(index=False):
        idx = int(row.task_index)
        text = str(row.task)
        by_index[idx] = text
        norm = normalize_text(text)
        existing = by_text.get(norm)
        if existing is not None and existing != idx:
            raise ValueError(f"duplicate normalized task text maps to multiple task indices: {text!r}")
        by_text[norm] = idx
    return by_index, by_text


def _task_cache_text_map(task_cache_dir: Path) -> dict[str, str]:
    if not task_cache_dir.exists():
        return {}

    text_to_tasks: dict[str, set[str]] = defaultdict(set)
    supported = {".json", ".jsonl", ".txt", ".md", ".yaml", ".yml", ".csv", ".parquet"}
    files = [p for p in task_cache_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported]
    if not files:
        return {}

    for path in files:
        strings = _read_strings(path)
        canonical = _canonical_from_path(path.relative_to(task_cache_dir))
        if canonical is None:
            embedded = sorted({
                normalize_task_name(value)
                for value in strings
                if normalize_task_name(value) in CANONICAL
            })
            if len(embedded) == 1:
                canonical = embedded[0]
            elif len(embedded) > 1:
                raise ValueError(
                    f"ambiguous canonical tasks embedded in task-cache file {path}: {embedded}"
                )
        if canonical is None:
            # No canonical provenance: do not guess from prompt semantics.
            continue
        for value in strings:
            norm = normalize_text(value)
            if norm:
                text_to_tasks[norm].add(canonical)

    ambiguous = {text: sorted(tasks) for text, tasks in text_to_tasks.items() if len(tasks) > 1}
    if ambiguous:
        preview = list(ambiguous.items())[:20]
        raise ValueError(f"task_cache contains ambiguous instruction ownership: {preview}")

    return {text: next(iter(tasks)) for text, tasks in text_to_tasks.items()}


def _load_manual_mapping(path: Path | None) -> tuple[dict[int, str], dict[int, str], dict[str, str]]:
    """Return episode overrides, task-index overrides, task-text overrides."""
    if path is None:
        return {}, {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode = {int(k): normalize_task_name(v) for k, v in payload.get("episode_to_semantic_task", {}).items()}
    task_index = {int(k): normalize_task_name(v) for k, v in payload.get("task_index_to_semantic_task", {}).items()}
    task_text = {normalize_text(k): normalize_task_name(v) for k, v in payload.get("task_text_to_semantic_task", {}).items()}
    for mapping in (episode, task_index, task_text):
        bad = sorted(set(mapping.values()) - CANONICAL)
        if bad:
            raise ValueError(f"manual mapping contains non-canonical tasks: {bad}")
    return episode, task_index, task_text


def _episode_task_indices_from_data(dataset_root: Path) -> dict[int, int]:
    """Scan only episode_index/task_index columns; never decode video."""
    import pyarrow.parquet as pq

    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {dataset_root / 'data'}")

    episode_to_raw: dict[int, int] = {}
    conflicts: dict[int, set[int]] = defaultdict(set)

    for file_index, path in enumerate(files, start=1):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        episodes = table.column("episode_index").to_pylist()
        tasks = table.column("task_index").to_pylist()
        for episode, task_index in zip(episodes, tasks, strict=True):
            ep = int(episode)
            raw = int(task_index)
            previous = episode_to_raw.get(ep)
            if previous is None:
                episode_to_raw[ep] = raw
            elif previous != raw:
                conflicts[ep].update((previous, raw))
        print(f"[semantic-map] scanned parquet {file_index}/{len(files)}: {path.name}", end="\r", flush=True)
    print()

    if conflicts:
        preview = [(ep, sorted(values)) for ep, values in list(conflicts.items())[:20]]
        raise ValueError(f"episodes with multiple task_index values are unsupported: {preview}")
    return episode_to_raw


def _direct_episode_metadata(dataset_root: Path) -> dict[int, str]:
    """Use direct canonical columns from meta/episodes if the conversion kept them."""
    import pandas as pd

    root = dataset_root / "meta" / "episodes"
    if not root.exists():
        return {}
    files = sorted(root.rglob("*.parquet")) + sorted(root.rglob("*.jsonl"))
    result: dict[int, str] = {}
    canonical_columns = (
        "semantic_task_id",
        "robotwin_task",
        "task_name",
        "env_task",
        "source_task",
        "canonical_task",
    )
    for path in files:
        try:
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_json(path, lines=True)
        except Exception:
            continue
        if "episode_index" not in df.columns:
            continue
        column = next((c for c in canonical_columns if c in df.columns), None)
        if column is None:
            continue
        for episode, value in zip(df["episode_index"], df[column], strict=True):
            task = normalize_task_name(value)
            if task in CANONICAL:
                result[int(episode)] = task
    return result


def build_mapping(dataset_root: Path, task_cache_dir: Path, manual_path: Path | None) -> dict[str, Any]:
    tasks_by_index, _tasks_by_text = _load_tasks_table(dataset_root)
    cache_text_to_semantic = _task_cache_text_map(task_cache_dir)
    manual_episode, manual_index, manual_text = _load_manual_mapping(manual_path)
    direct_episode = _direct_episode_metadata(dataset_root)

    task_index_to_semantic: dict[int, str] = {}
    for raw_index, text in tasks_by_index.items():
        semantic = manual_index.get(raw_index)
        if semantic is None:
            semantic = manual_text.get(normalize_text(text))
        if semantic is None:
            semantic = cache_text_to_semantic.get(normalize_text(text))
        if semantic is not None:
            task_index_to_semantic[raw_index] = semantic

    episode_to_raw = _episode_task_indices_from_data(dataset_root)
    episode_to_semantic: dict[int, str] = {}
    unmapped: list[dict[str, Any]] = []

    for episode, raw_index in sorted(episode_to_raw.items()):
        semantic = manual_episode.get(episode)
        if semantic is None:
            semantic = direct_episode.get(episode)
        if semantic is None:
            semantic = task_index_to_semantic.get(raw_index)
        if semantic is None:
            unmapped.append({
                "episode_index": episode,
                "task_index": raw_index,
                "task": tasks_by_index.get(raw_index, "<missing task text>"),
            })
            continue
        episode_to_semantic[episode] = semantic

    counts = Counter(episode_to_semantic.values())
    payload = {
        "format": SemanticTaskMap.FORMAT,
        "dataset_root": str(dataset_root.resolve()),
        "canonical_tasks": list(ROBOTWIN_CANONICAL_TASKS),
        "episode_to_semantic_task": {str(k): v for k, v in sorted(episode_to_semantic.items())},
        "task_index_to_semantic_task": {str(k): v for k, v in sorted(task_index_to_semantic.items())},
        "stats": {
            "episodes_total": len(episode_to_raw),
            "episodes_mapped": len(episode_to_semantic),
            "episodes_unmapped": len(unmapped),
            "semantic_task_count": len(counts),
            "episodes_per_semantic_task": dict(sorted(counts.items())),
            "task_index_total": len(tasks_by_index),
            "task_index_mapped": len(task_index_to_semantic),
            "task_cache_text_entries": len(cache_text_to_semantic),
        },
        "unmapped_preview": unmapped[:200],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-cache-dir", type=Path, default=None)
    parser.add_argument("--manual-map", type=Path, default=None)
    parser.add_argument("--allow-subset-of-canonical-tasks", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    task_cache_dir = (args.task_cache_dir or dataset_root / "meta" / "task_cache").resolve()
    payload = build_mapping(dataset_root, task_cache_dir, args.manual_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stats = payload["stats"]
    print("========== Semantic task map ==========")
    for key, value in stats.items():
        if key != "episodes_per_semantic_task":
            print(f"{key}: {value}")
    print("episodes_per_semantic_task:")
    for task, count in stats["episodes_per_semantic_task"].items():
        print(f"  {task}: {count}")

    if stats["episodes_unmapped"]:
        diagnostics = args.output.with_suffix(args.output.suffix + ".unmapped.json")
        diagnostics.write_text(
            json.dumps(payload["unmapped_preview"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(
            f"ERROR: {stats['episodes_unmapped']} episodes are unmapped. "
            f"See {diagnostics}. Do NOT train CTE yet."
        )

    if not args.allow_subset_of_canonical_tasks and stats["semantic_task_count"] != len(ROBOTWIN_CANONICAL_TASKS):
        raise SystemExit(
            "ERROR: mapping covers every episode but resolves "
            f"{stats['semantic_task_count']} semantic tasks instead of expected "
            f"{len(ROBOTWIN_CANONICAL_TASKS)}. Verify dataset scope before training."
        )

    # Strictly reload what we wrote. This validates canonical names and format.
    mapping = SemanticTaskMap.load(args.output)
    print(f"semantic_task_map_sha256: {mapping.sha256}")
    print("SEMANTIC TASK MAP: PASSED")


if __name__ == "__main__":
    main()
