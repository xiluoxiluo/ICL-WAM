#!/usr/bin/env python3
"""Read-only input and optional environment checks for the offline curriculum.

Metadata checks use only the Python standard library.  In particular, this
script does not construct a dataset, read RGB/actions, deserialize checkpoints,
download weights, or create the task subset / episode conditions files.  The
CTE and prepared-training entry points remain responsible for their complete
source, shape, checkpoint, and Stage1-readiness contracts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


_MISSING = object()
_REQUIRED_CONDITIONS = frozenset(("Clean", "Randomized"))
_ENVIRONMENT_MODULES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("hydra", "hydra-core"),
    ("omegaconf", "omegaconf"),
    ("accelerate", "accelerate"),
    ("av", "av"),
    ("datasets", "datasets"),
    ("transformers", "transformers"),
    ("mamba_ssm", "mamba-ssm"),
    ("causal_conv1d", "causal-conv1d"),
)


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []

    @staticmethod
    def info(message: str) -> None:
        print(f"[INFO] {message}")

    @staticmethod
    def ok(message: str) -> None:
        print(f"[OK] {message}")

    @staticmethod
    def warning(message: str) -> None:
        print(f"[WARN] {message}")

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"[ERROR] {message}")


def _existing_path(raw: str, label: str, *, directory: bool, report: Report) -> Path | None:
    try:
        path = Path(raw).expanduser().resolve()
        exists = path.is_dir() if directory else path.is_file()
    except (OSError, RuntimeError, ValueError) as exc:
        report.error(f"{label}: cannot resolve {raw!r}: {exc}")
        return None
    if not exists:
        kind = "directory" if directory else "file"
        report.error(f"{label}: required input {kind} does not exist: {path}")
        return None
    report.ok(f"{label}: {path}")
    return path


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not a valid JSON number")


def _read_json(path: Path | None, label: str, report: Report) -> Any:
    if path is None:
        return _MISSING
    try:
        result = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        report.error(f"{label}: invalid or unreadable JSON in {path}: {exc}")
        return _MISSING
    report.ok(f"{label}: JSON parses")
    return result


def _task_names(payload: Any, report: Report) -> list[str] | None:
    if payload is _MISSING:
        return None
    tasks = payload.get("task_names") if isinstance(payload, dict) else None
    if (not isinstance(tasks, list) or not tasks
            or any(not isinstance(task, str) or not task.strip() for task in tasks)):
        report.error("task-subset: expected a nonempty ordered task_names array of nonempty strings")
        return None
    if len(set(tasks)) != len(tasks):
        report.error("task-subset: task_names must be unique; preserve the intended task order")
        return None
    report.ok(f"task-subset: {len(tasks)} unique tasks in supplied order: {', '.join(tasks)}")
    return tasks


def _semantic_map(path: Path | None, payload: Any, tasks: list[str] | None, report: Report) -> None:
    if path is None or payload is _MISSING:
        return
    if not isinstance(payload, dict):
        report.error("semantic-map: expected a JSON object accepted by SemanticTaskMap")
        return
    try:
        # The resolver is standard-library-only.  Loading its source directly
        # avoids fastwam.zeva.__init__, which imports training dependencies.
        # compile/exec also avoids emitting a __pycache__ during a read-only check.
        source = Path(__file__).resolve().parents[2] / "src/fastwam/zeva/semantic_tasks.py"
        name = "_iclwam_train_preflight_semantic_tasks"
        module = ModuleType(name)
        module.__file__ = str(source)
        previous = sys.modules.get(name)
        sys.modules[name] = module  # dataclasses resolves annotations via this registry.
        try:
            exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), module.__dict__)
            mapping = module.SemanticTaskMap(path, payload)
        finally:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    except (OSError, UnicodeError, TypeError, ValueError, KeyError, AttributeError) as exc:
        report.error(f"semantic-map: {exc}")
        return
    report.ok(
        f"semantic-map: {len(mapping.episode_to_task)} episode mappings, "
        f"{len(mapping.semantic_tasks)} canonical tasks ({mapping.FORMAT})"
    )
    if tasks is not None:
        missing = [task for task in tasks if task not in mapping.semantic_tasks]
        if len(missing) == len(tasks):
            report.error("task-subset: no selected tasks occur in the semantic map's episode mappings")
        elif missing:
            report.warning(
                f"task-subset: tasks absent from the semantic map's episode mappings: {missing}; "
                "empty task slots are retained in the supplied task order"
            )
        else:
            report.ok("task-subset: every task occurs in the actual semantic episode mappings")


def _conditions(payload: Any, root: Path | None, report: Report) -> dict[str, str] | None:
    if payload is _MISSING:
        return None
    if (not isinstance(payload, dict) or not payload
            or any(not isinstance(key, str) or not key
                   or not isinstance(value, str) or not value.strip()
                   for key, value in payload.items())):
        report.error("conditions: expected a nonempty JSON object mapping canonical episode IDs to real nonempty condition names")
        return None
    if root is not None:
        prefix = f"{root}::episode-"
        invalid = []
        for key in payload:
            suffix = key[len(prefix):] if key.startswith(prefix) else ""
            canonical_index = (suffix.isascii() and suffix.isdecimal()
                               and (suffix == "0" or not suffix.startswith("0")))
            if not canonical_index:
                invalid.append(key)
        if invalid:
            report.error(
                f"conditions: {len(invalid)} IDs must use the resolved dataset root and "
                f"canonical nonnegative episode index, e.g. {prefix}0; invalid examples: {invalid[:5]}"
            )
    counts = Counter(payload.values())
    report.info("conditions: " + ", ".join(f"{name}={counts[name]}" for name in sorted(counts)))
    missing = sorted(_REQUIRED_CONDITIONS - set(counts))
    if missing:
        report.error(f"conditions: the default full curriculum requires both Clean and Randomized; missing {missing}")
    extras = sorted(set(counts) - _REQUIRED_CONDITIONS)
    if extras:
        report.info(f"conditions: additional real conditions are allowed and retained: {extras}; default CTE conditions select Clean and Randomized")
    return payload


def _episode_coverage(root: Path | None, conditions: dict[str, str] | None, report: Report) -> None:
    if root is None:
        return
    path = root / "meta/episodes.jsonl"
    if not path.exists():
        report.warning("meta/episodes.jsonl is absent; full-source condition coverage is deferred to the CTE loader")
        return
    positive_ids: set[str] = set()
    seen: set[int] = set()
    nonpositive = 0
    invalid_count = 0
    invalid_examples: list[str] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=_reject_json_constant)
                    if not isinstance(row, dict):
                        raise ValueError("episode row must be a JSON object")
                    index, length = row.get("episode_index"), row.get("length")
                    if type(index) is not int or index < 0 or type(length) is not int or length < 0:
                        raise ValueError("episode_index and length must be nonnegative integers")
                    if index in seen:
                        raise ValueError(f"duplicate episode_index {index}")
                    seen.add(index)
                    if length > 0:
                        positive_ids.add(f"{root}::episode-{index}")
                    else:
                        nonpositive += 1
                except ValueError as exc:
                    invalid_count += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(f"line {line_number}: {exc}")
    except (OSError, UnicodeError) as exc:
        report.error(f"episode metadata: cannot read {path}: {exc}")
        return
    if invalid_count:
        report.error(f"episode metadata: {invalid_count} invalid rows in {path}: {'; '.join(invalid_examples)}")
    if nonpositive:
        report.error(
            f"episode metadata: {nonpositive} zero-length rows; the strict CTE loader rejects "
            "empty source slots because their reference partition positions cannot be inferred"
        )
    if not positive_ids:
        report.error(f"episode metadata: no positive-length source episodes in {path}")
        return
    report.info(f"episode metadata: {len(positive_ids)} positive-length source episodes; coverage includes short episodes and tasks outside the selected subset")
    if conditions is None:
        return
    missing = sorted(positive_ids - set(conditions))
    if missing:
        report.error(f"conditions: missing {len(missing)} positive-length source episodes (not just task subset); examples: {missing[:5]}")
    else:
        report.ok(f"conditions: covers all {len(positive_ids)} positive-length source episodes")
    source_conditions = {conditions[key] for key in positive_ids if key in conditions}
    absent = sorted(_REQUIRED_CONDITIONS - source_conditions)
    if absent:
        report.error(f"conditions: default curriculum conditions absent from positive-length source episodes: {absent}")
    extra_ids = set(conditions) - positive_ids
    if extra_ids:
        report.info(f"conditions: {len(extra_ids)} additional IDs are outside the positive-length source table; they do not contribute to this coverage check")


def _check_environment(report: Report) -> None:
    modules: dict[str, Any] = {}
    report.info(f"Python {sys.version.split()[0]}; executable={sys.executable}")
    report.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    for name, distribution in _ENVIRONMENT_MODULES:
        try:
            module = importlib.import_module(name)
            modules[name] = module
            version = getattr(module, "__version__", None)
            if version is None:
                try:
                    version = metadata.version(distribution)
                except metadata.PackageNotFoundError:
                    version = "unknown"
            report.ok(f"import {name}: {version}")
        except Exception as exc:
            report.error(f"import {name}: {type(exc).__name__}: {exc}")
    torch = modules.get("torch")
    if torch is None:
        return
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        report.info(f"Torch CUDA runtime={torch.version.cuda}; CUDA available={available}; visible GPUs={count}")
        if not available or count < 1:
            report.error("CUDA is required for the default offline GPU training curriculum")
        else:
            for index in range(count):
                properties = torch.cuda.get_device_properties(index)
                report.ok(f"visible GPU {index}: {properties.name}, {properties.total_memory / (1024 ** 3):.1f} GiB")
    except Exception as exc:
        report.error(f"CUDA inspection: {type(exc).__name__}: {exc}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, help_text in (
        ("dataset-root", "Existing LeRobot dataset directory; resolved path is the source identity"),
        ("stats", "Existing normalization statistics JSON"),
        ("semantic-map", "Existing RoboTwin semantic task map JSON"),
        ("task-subset", "User-supplied ordered CTE task_names JSON; never generated here"),
        ("conditions", "User-supplied real episode-to-condition JSON; never inferred here"),
        ("base-checkpoint", "Existing FastWAM foundation checkpoint"),
        ("action-dit-checkpoint", "Existing ActionDiT initialization checkpoint"),
        ("assets-root", "Existing Wan model/tokenizer asset directory"),
    ):
        parser.add_argument(f"--{flag}", required=True, help=help_text)
    parser.add_argument("--check-environment", action="store_true", help="Import training dependencies and inspect CUDA, without constructing models or downloading assets")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Imports requested by --check-environment must not leave Python bytecode
    # artifacts either.  This does not change the caller's environment.
    sys.dont_write_bytecode = True
    report = Report()
    report.info("Task-subset and conditions paths are required input conventions, not presumed existing or generated training artifacts; provide the real metadata files")
    paths: dict[str, Path | None] = {}
    for name in ("dataset_root", "stats", "semantic_map", "task_subset", "conditions",
                 "base_checkpoint", "action_dit_checkpoint", "assets_root"):
        paths[name] = _existing_path(getattr(args, name), name.replace("_", "-"),
                                     directory=name in {"dataset_root", "assets_root"}, report=report)
    payloads = {name: _read_json(paths[name], name.replace("_", "-"), report)
                for name in ("stats", "semantic_map", "task_subset", "conditions")}
    tasks = _task_names(payloads["task_subset"], report)
    _semantic_map(paths["semantic_map"], payloads["semantic_map"], tasks, report)
    conditions = _conditions(payloads["conditions"], paths["dataset_root"], report)
    _episode_coverage(paths["dataset_root"], conditions, report)
    if args.check_environment:
        _check_environment(report)
    else:
        report.info("Metadata-only mode: training dependencies and CUDA were not imported or checked")
    report.info("Checkpoint files were not loaded; tensor shapes, complete source identity, normalization semantics and Stage1 gate are checked by the actual CTE/training consumers")
    if report.errors:
        print(f"Preflight FAILED: {len(report.errors)} error(s); no training or metadata generation was performed.")
        return 1
    print("Preflight PASSED: input metadata checks completed; no training or metadata generation was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
