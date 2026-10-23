"""Dry-run assertions for the optional two-GPU CTE cache performance switches."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _launch(fast_collate: str, profile: str = "50", streaming: str = "0", verify: str = "0"):
    env = {
        **os.environ,
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHON_BIN": "not-executed-dry-run",
        "GPU_IDS": "0,1",
        "PREP_GPU_IDS": "0,1",
        "PREP_NPROC_PER_NODE": "2",
        "NNODES": "1",
        "NODE_RANK": "0",
        "WORLD_SIZE": "1",
        "LATENT_RESUME": "1",
        "LATENT_FAST_COLLATE": fast_collate,
        "LATENT_STREAMING_PYAV": streaming,
        "LATENT_STREAMING_VERIFY_EVERY": verify,
        "LATENT_STREAMING_MAX_OPEN": "6",
        "LATENT_PROFILE_EVERY_BATCHES": profile,
        "LATENT_BATCH_SIZE": "8",
        "NUM_WORKERS": "8",
        "LATENT_PREFETCH_FACTOR": "2",
    }
    return subprocess.run(
        ["bash", str(ROOT / "scripts/train/03_precompute_cte_latents.sh"), "--dry-run"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )


def test_fast_collate_profile_and_two_processes_are_passed_to_hydra():
    result = _launch("1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--nproc-per-node=2" in result.stdout
    for name, value in (
        ("batch_size", "8"),
        ("num_workers", "8"),
        ("prefetch_factor", "2"),
        ("fast_collate", "true"),
        ("profile_every_batches", "50"),
        ("overwrite", "false"),
    ):
        assert f"++cte_latent_cache.{name}={value}" in result.stdout
    assert "--declare-success-only" not in result.stdout


def test_original_fallback_remains_available():
    result = _launch("0", profile="0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "++cte_latent_cache.fast_collate=false" in result.stdout
    assert "++cte_latent_cache.profile_every_batches=0" in result.stdout


@pytest.mark.parametrize("fast,profile", [("maybe", "50"), ("1", "-1"), ("1", "nan")])
def test_bad_switches_fail_without_starting_gpu_processes(fast, profile):
    result = _launch(fast, profile)
    assert result.returncode != 0
    assert "VAE latent cache GPU" not in result.stdout


def test_streaming_decoder_switches_reach_hydra_only_when_requested():
    result = _launch("1", streaming="1", verify="128")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "++cte_latent_cache.streaming_pyav=true" in result.stdout
    assert "++cte_latent_cache.streaming_verify_every=128" in result.stdout
    assert "++cte_latent_cache.streaming_max_open=6" in result.stdout


@pytest.mark.parametrize("streaming,verify", [("unknown", "0"), ("1", "-1"), ("1", "none")])
def test_bad_streaming_options_are_rejected_before_training(streaming, verify):
    result = _launch("1", streaming=streaming, verify=verify)
    assert result.returncode != 0
    assert "VAE latent cache GPU" not in result.stdout
