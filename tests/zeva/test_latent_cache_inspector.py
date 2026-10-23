"""Inspect interrupted caches without GPU dependencies or cache mutations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/inspect_zeva_cte_latent_cache.py"
inspect_cache = runpy.run_path(str(SCRIPT))["inspect_cache"]


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _rank(root, *, rank=0, world=1, count=3, offset=1):
    directory = root / f"rank_{rank:03d}"
    directory.mkdir(parents=True)
    shape = [9, 2, 1, 1] if count else []
    _json(directory / "rank_meta.json", {"rank": rank, "world_size": world,
                                         "num_windows": count, "latent_shape": shape})
    _json(directory / "progress.json", {"total": count, "next_offset": offset})
    if count:
        for name, tail, dtype in [
            ("latents_bf16_u16", shape, np.uint16),
            ("transition_actions", [8, 4, 14], np.float32),
            ("frame_valid", [9], np.uint8),
            ("transition_valid", [8, 4], np.uint8),
        ]:
            np.save(directory / (name + ".npy"), np.zeros((count, *tail), dtype=dtype))
    return directory


def _snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns,
                                      hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob("*") if p.is_file()}


def test_partial_cache_reports_saved_progress_and_actual_dimensions(tmp_path):
    directory = _rank(tmp_path)
    (directory / "progress.json.tmp").write_text('{"next_offset": 3, "total":')
    before = _snapshot(tmp_path)
    report = inspect_cache(tmp_path)
    assert report["errors"] == []
    assert report["generation_world_size"] == 1
    assert report["num_windows"] == 3 and report["committed_windows"] == 1
    assert report["full_array_payload_bytes"] == 3 * (9 * 2 * 2 + 8 * 4 * 14 * 4 + 9 + 8 * 4)
    assert report["capacity_totals_cover_all_ranks"] is True
    assert report["completion_markers_present"] is False
    assert report["quota_available_bytes"] is None and not report["quota_verified"]
    assert report["payload_contents_verified"] is False
    assert report["ranks"][0]["ignored_temporary_files"] == ["progress.json.tmp"]
    assert _snapshot(tmp_path) == before


def test_cli_runs_with_stdlib_only_and_changes_no_files(tmp_path):
    _rank(tmp_path)
    before = _snapshot(tmp_path)
    result = subprocess.run([sys.executable, "-I", "-S", "-B", str(SCRIPT),
                             "--cache-root", str(tmp_path), "--json"],
                            cwd="/tmp", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["committed_windows"] == 1 and not report["source_provenance_verified"]
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("defect", ["dtype", "shape", "truncated", "progress", "done", "wrong_rank"])
def test_corruption_is_reported_without_repair(tmp_path, defect):
    directory = _rank(tmp_path)
    array = directory / "transition_actions.npy"
    if defect == "dtype":
        np.save(array, np.zeros((3, 8, 4, 14), dtype=np.float64))
    elif defect == "shape":
        np.save(array, np.zeros((3, 8, 4, 13), dtype=np.float32))
    elif defect == "truncated":
        with array.open("r+b") as handle:
            handle.truncate(array.stat().st_size - 4)
    elif defect == "progress":
        _json(directory / "progress.json", {"total": 3, "next_offset": 4})
    elif defect == "done":
        (directory / "DONE").write_text("ok\n")
    else:
        meta = json.loads((directory / "rank_meta.json").read_text())
        meta["rank"] = 1
        _json(directory / "rank_meta.json", meta)
    before = _snapshot(tmp_path)
    report = inspect_cache(tmp_path)
    assert report["errors"] and report["capacity_totals_cover_all_ranks"] is False
    assert _snapshot(tmp_path) == before


def test_done_marker_requires_saved_progress_even_if_rank_is_empty(tmp_path):
    directory = _rank(tmp_path, count=0, offset=0)
    (directory / "DONE").write_text("ok\n")
    assert inspect_cache(tmp_path)["errors"] == []
    (directory / "progress.json").unlink()
    assert any("DONE" in error for error in inspect_cache(tmp_path)["errors"])


def test_partial_rank_coverage_and_inconsistent_world_are_visible(tmp_path):
    _rank(tmp_path, rank=0, world=2)
    report = inspect_cache(tmp_path)
    assert report["missing_ranks"] == [1] and report["num_windows"] == 3
    assert report["capacity_totals_cover_all_ranks"] is False
    assert any("totals exclude missing metadata" in warning for warning in report["warnings"])
    _rank(tmp_path, rank=1, world=3)
    report = inspect_cache(tmp_path)
    assert any("different generation world sizes" in error for error in report["errors"])


def test_no_cache_directory_is_created(tmp_path):
    missing = tmp_path / "absent"
    report = inspect_cache(missing)
    assert report["errors"] and not missing.exists()


def test_completion_markers_do_not_claim_payload_or_provenance_verification(tmp_path):
    directory = _rank(tmp_path, offset=3)
    (directory / "DONE").write_text("ok\n")
    _json(tmp_path / "manifest.json", {})
    _json(tmp_path / "index.json", {})
    report = inspect_cache(tmp_path)
    assert report["completion_markers_present"] is True
    assert not report["payload_contents_verified"] and not report["source_provenance_verified"]
