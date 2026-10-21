import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def _builder(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/build_zeva_robotwin_cache.py"
    spec = importlib.util.spec_from_file_location("_zeva_cache_source_index_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _plan():
    return SimpleNamespace(
        dataset_start=10, episode_id="episode-1", coverage_starts=lambda: [0],
    )


def test_cache_builder_refuses_dataset_retry_that_changes_source_index(monkeypatch):
    builder = _builder(monkeypatch)
    # Reject before decoding metadata or frames from a substituted sample.
    with pytest.raises(RuntimeError, match="requested 10, got 11"):
        builder._decode_episode(
            _plan(), dataset={10: {"dataset_index": 11}},
            transition_steps=4, sample_stride=1, cte_input_type="wan_vae_latent",
        )


def test_cache_builder_preserves_source_identity_and_raw_boundaries(monkeypatch):
    builder = _builder(monkeypatch)
    sample = {
        "dataset_index": 10,
        "episode": {"episode_id": "episode-1", "episode_step": 0},
        "cte_frames": torch.zeros(9, 3, 2, 2),
        "transition_actions": torch.zeros(8, 4, 14),
    }
    decoded = builder._decode_episode(
        _plan(), dataset={10: sample}, transition_steps=4, sample_stride=1,
        cte_input_type="wan_vae_latent",
    )
    assert decoded.num_decoded_windows == 1
    assert list(decoded.boundary_frames) == list(range(0, 33, 4))
    assert list(decoded.action_groups) == list(range(0, 32, 4))
    assert decoded.segments == [list(range(0, 33, 4))]
