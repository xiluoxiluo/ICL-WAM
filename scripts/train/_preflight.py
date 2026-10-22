#!/usr/bin/env python3
"""Read-only main success-provenance checks; optional environment imports.

Without --check-environment this entry uses only the standard library, reads
collection metadata, and never opens video, tensors, checkpoints, or CUDA.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


class _Config(dict):
    def __getattr__(self, key):
        return self[key]


def _source_module(relative: str, name: str):
    """Read stdlib-only validators without the heavyweight zeva package init."""
    path = ROOT / relative
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "stats", "semantic-map", "success-manifest", "base-checkpoint", "action-dit-checkpoint", "assets-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--check-environment", action="store_true")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    # Isolated `python -I -S` must still resolve the shared, stdlib-only main
    # contract. Importing fastwam itself is lightweight; zeva's package init
    # remains bypassed by _source_module below.
    sys.path.insert(0, str(ROOT / "src"))
    errors = []
    paths = {}
    for name in ("dataset_root", "stats", "semantic_map", "success_manifest", "base_checkpoint", "action_dit_checkpoint", "assets_root"):
        path = getattr(args, name).expanduser().resolve()
        paths[name] = path
        exists = path.is_dir() if name in {"dataset_root", "assets_root"} else path.is_file()
        if not exists:
            errors.append(f"Required {name} missing: {path}")
        else:
            print(f"[OK] {name}: {path}")
    if not errors:
        try:
            stats = json.loads(paths["stats"].read_text())
            if not isinstance(stats, dict) or not stats:
                raise ValueError("Normalization statistics must be a nonempty JSON object")
            semantics = _source_module("src/fastwam/zeva/semantic_tasks.py", "_main_preflight_semantics")
            mapping = semantics.SemanticTaskMap(paths["semantic_map"], json.loads(paths["semantic_map"].read_text()))
            print(f"[OK] canonical tasks: {len(mapping.semantic_tasks)}; no fixed 50-task restriction")
            data = _source_module("src/fastwam/zeva/main_data.py", "_main_preflight_success")
            cfg = _Config(model=_Config(zeva=_Config(profile="main_observed", offline=_Config(success_manifest_path=str(paths["success_manifest"])))),
                          data=_Config(train=_Config(dataset_dirs=[str(paths["dataset_root"])],
                              pretrained_norm_stats=str(paths["stats"]), semantic_task_map_path=str(paths["semantic_map"]))))
            selected, identity = data.load_success_selection(cfg)
            print(f"[OK] {len(selected)} successful demonstrations; source={identity['evidence_kind']}")
            print(f"[OK] collection metadata fingerprint: {identity['source_fingerprint']}")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(str(exc))
    if args.check_environment and not errors:
        print(f"[INFO] Python={sys.executable}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
        loaded = {}
        for name in ("torch", "torchvision", "hydra", "omegaconf", "accelerate", "av", "datasets", "transformers", "safetensors"):
            try:
                loaded[name] = importlib.import_module(name)
                print(f"[OK] {name}: {getattr(loaded[name], '__version__', 'imported')}")
            except Exception as exc:
                errors.append(f"Environment import {name}: {exc}")
        if "torch" in loaded:
            torch = loaded["torch"]
            if not torch.cuda.is_available():
                errors.append("CUDA is unavailable in the selected Python environment")
            else:
                print(f"[OK] visible CUDA devices: {torch.cuda.device_count()}")
    elif not args.check_environment:
        print("[INFO] Metadata-only: training dependencies and CUDA were not imported")
    print("[INFO] CTE/cache consumers validate complete episodes, per-task train/validation membership, tensor shapes and actual checkpoint lineage.")
    print("[INFO] Metadata provenance is not a simulator rerun or a checksum of every video byte; no checkpoints were deserialized.")
    for error in errors:
        print(f"[ERROR] {error}")
    print(f"Preflight {'FAILED' if errors else 'PASSED'}: no training or metadata generation was performed.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
