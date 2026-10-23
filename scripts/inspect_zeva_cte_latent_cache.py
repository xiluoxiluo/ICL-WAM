#!/usr/bin/env python3
"""Read-only storage/progress inspection; no NumPy, Torch, CUDA or payload reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.zeva_latent_storage import cache_window_payload_bytes, read_npy_header


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return value


def _inspect_rank(directory: Path) -> dict:
    result = {
        "directory": str(directory), "rank": int(directory.name.split("_")[1]),
        "world_size": None, "num_windows": None, "committed_windows": None,
        "done_marker": (directory / "DONE").is_file(), "latent_shape": None,
        "payload_bytes": 0, "arrays": [], "errors": [], "warnings": [],
        "ignored_temporary_files": sorted(p.name for p in directory.glob("*.tmp")),
    }
    try:
        meta = _object(directory / "rank_meta.json")
        rank = _integer(meta.get("rank"), "rank")
        world = _integer(meta.get("world_size"), "world_size", 1)
        count = _integer(meta.get("num_windows"), "num_windows")
        if rank != result["rank"] or rank >= world:
            raise ValueError("rank directory, rank metadata and world_size disagree")
        result.update(world_size=world, num_windows=count)
        shape = meta.get("latent_shape")
        if not isinstance(shape, list) or (count and len(shape) != 4):
            raise ValueError("latent_shape must describe [frames, channels, height, width]")
        if count:
            for dimension in shape:
                _integer(dimension, "latent_shape dimension", 1)
            if shape[0] != 9:
                raise ValueError("Expected 9 stored latent frames per CTE window")
            result["payload_bytes"] = count * cache_window_payload_bytes(shape)
        elif shape:
            raise ValueError("An empty rank must have latent_shape=[]")
        result["latent_shape"] = shape
        progress_path = directory / "progress.json"
        if progress_path.is_file():
            progress = _object(progress_path)
            total = _integer(progress.get("total"), "progress.total")
            offset = _integer(progress.get("next_offset"), "progress.next_offset")
            if total != count or offset > count:
                raise ValueError("progress.json total/next_offset is inconsistent with rank_meta.json")
        else:
            offset = 0
            result["warnings"].append("No committed progress.json; no windows are counted as reusable.")
        result["committed_windows"] = offset
        if result["done_marker"] and (not progress_path.is_file() or offset != count):
            result["errors"].append("DONE exists without a complete committed progress.json")
    except (OSError, ValueError, TypeError) as exc:
        result["errors"].append(str(exc))
        return result

    if not count:
        if any(directory.glob("*.npy")):
            result["errors"].append("Unexpected arrays for an empty rank")
        return result
    native = "<" if sys.byteorder == "little" else ">"
    layouts = {
        "latents_bf16_u16.npy": ([count, *shape], {native + "u2", "=u2"}),
        "transition_actions.npy": ([count, 8, 4, 14], {native + "f4", "=f4"}),
        "frame_valid.npy": ([count, 9], {"|u1"}),
        "transition_valid.npy": ([count, 8, 4], {"|u1"}),
    }
    for filename, (expected_shape, expected_dtypes) in layouts.items():
        try:
            header = read_npy_header(directory / filename)
            result["arrays"].append(header)
            if header["shape"] != expected_shape or header["dtype"] not in expected_dtypes:
                raise ValueError(f"{filename}: unexpected shape/dtype {header['shape']}/{header['dtype']}")
            if header["fortran_order"]:
                raise ValueError(f"{filename}: expected C-contiguous storage")
            if header["is_truncated"] or header["trailing_bytes"]:
                raise ValueError(f"{filename}: file length disagrees with its NPY header")
        except (OSError, ValueError, TypeError) as exc:
            result["errors"].append(str(exc))
    return result


def inspect_cache(cache_root: str | Path) -> dict:
    """Inspect saved offsets and array headers, without asserting payload integrity."""
    root = Path(cache_root).expanduser().resolve()
    report = {
        "cache_root": str(root), "scope": "read-only metadata, NPY headers and filesystem allocation",
        "payload_contents_verified": False, "source_provenance_verified": False,
        "quota_verified": False, "quota_available_bytes": None,
        "errors": [], "warnings": [], "ranks": [],
    }
    if not root.is_dir():
        report["errors"].append(f"Cache directory does not exist: {root}")
        return report
    directories = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"rank_\d+", p.name))
    report["ranks"] = ranks = [_inspect_rank(directory) for directory in directories]
    for rank in ranks:
        report["errors"].extend(f"rank {rank['rank']}: {error}" for error in rank["errors"])
        report["warnings"].extend(f"rank {rank['rank']}: {warning}" for warning in rank["warnings"])
    worlds = {rank["world_size"] for rank in ranks if rank["world_size"] is not None}
    world = next(iter(worlds)) if len(worlds) == 1 else None
    if len(worlds) > 1:
        report["errors"].append("Rank metadata contains different generation world sizes")
    ids = [rank["rank"] for rank in ranks]
    if len(set(ids)) != len(ids):
        report["errors"].append("Duplicate rank directory identities")
    missing = sorted(set(range(world)) - set(ids)) if world is not None else []
    if missing:
        report["warnings"].append(f"Missing rank directories: {missing}; totals exclude missing metadata")
    if not ranks:
        report["warnings"].append("No rank metadata yet; actual latent shape and total capacity cannot be determined")
    shapes = {tuple(rank["latent_shape"]) for rank in ranks if rank["num_windows"] and rank["latent_shape"]}
    if len(shapes) > 1:
        report["errors"].append("Nonempty ranks contain different latent shapes")
    arrays = [array for rank in ranks for array in rank["arrays"]]
    payload = sum(rank["payload_bytes"] for rank in ranks)
    logical = sum(array["logical_bytes"] for array in arrays)
    allocated = sum(array["allocated_bytes"] for array in arrays)
    headers = sum(array["data_offset"] for array in arrays)
    report.update(
        generation_world_size=world, missing_ranks=missing,
        num_windows=sum(rank["num_windows"] or 0 for rank in ranks),
        committed_windows=sum(rank["committed_windows"] or 0 for rank in ranks),
        full_array_payload_bytes=payload, observed_array_header_bytes=headers,
        array_logical_bytes=logical, array_allocated_bytes=allocated,
        allocated_blocks_reported=bool(arrays) and all(array["allocated_bytes_exact"] for array in arrays),
        estimated_additional_array_allocation_bytes=max(0, payload + headers - allocated),
        root_manifest_present=(root / "manifest.json").is_file(),
        root_index_present=(root / "index.json").is_file(),
    )
    complete_layout = bool(world) and not missing and len(ranks) == world and not report["errors"]
    report["capacity_totals_cover_all_ranks"] = complete_layout
    report["completion_markers_present"] = bool(
        complete_layout and all(rank["done_marker"] for rank in ranks)
        and report["root_manifest_present"] and report["root_index_present"]
    )
    try:
        report["filesystem_available_bytes"] = int(shutil.disk_usage(root).free)
    except OSError as exc:
        report["filesystem_available_bytes"] = None
        report["warnings"].append(f"Cannot read filesystem availability: {exc}")
    report["warnings"].append(
        "Allocated blocks and filesystem free space do not establish user/project quota. "
        "Sparse file length and preallocated blocks do not establish completed windows."
    )
    return report


def _size(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 1024**3:.3f} GiB ({value:,} bytes)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="Print a structured read-only report")
    args = parser.parse_args()
    report = inspect_cache(args.cache_root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"CTE cache inspection (read-only): {report['cache_root']}")
        print("No array payloads, models, CUDA or source dataset were loaded.")
        for rank in report["ranks"]:
            print(f"rank {rank['rank']}: committed={rank['committed_windows']}/{rank['num_windows']}, "
                  f"DONE={rank['done_marker']}, latent_shape={rank['latent_shape']}")
            if rank["ignored_temporary_files"]:
                print(f"  Ignored temporary files: {', '.join(rank['ignored_temporary_files'])}")
        if "full_array_payload_bytes" in report:
            print(f"Generation world size: {report['generation_world_size']}")
            print(f"Committed windows: {report['committed_windows']:,}/{report['num_windows']:,}")
            print(f"Capacity totals cover all ranks: {report['capacity_totals_cover_all_ranks']}")
            print(f"Full array payload: {_size(report['full_array_payload_bytes'])}")
            print(f"Array logical length: {_size(report['array_logical_bytes'])}")
            print(f"Array allocated blocks: {_size(report['array_allocated_bytes'])}")
            print(f"Estimated additional array allocation: {_size(report['estimated_additional_array_allocation_bytes'])}")
            print(f"Filesystem available: {_size(report.get('filesystem_available_bytes'))}")
            print("User/project quota available: UNKNOWN (check the storage platform/quota tools)")
            print(f"Completion markers present: {report['completion_markers_present']}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        print("Payload integrity and source/checkpoint compatibility were not verified; "
              "the normal resume/training validation still applies.")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
