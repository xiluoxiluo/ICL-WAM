"""Validate safe reuse of rank output after a late 06 merge failure."""

from types import SimpleNamespace

import pytest
import torch

from fastwam.zeva.behavior_rank_resume import (
    atomic_save_rank_snapshot,
    rank_contract,
    validate_rank_snapshot,
    wait_for_files,
)


def _plans(n=4):
    return [
        SimpleNamespace(
            episode_id=f"episode-{i}", task_id=str(i % 2), dataset_start=i * 111
        )
        for i in range(n)
    ]


def _payload(plans, contract):
    return {
        "resume_contract": contract,
        "entries": [
            {
                "episode_id": p.episode_id,
                "task_id": p.task_id,
                "dataset_start": p.dataset_start,
                "instruction": "pick",
                "num_transitions": 8,
                "retrieval_key": torch.ones(3),
                "behavior_value": torch.ones(4),
            }
            for p in plans
        ],
        "episode_ids": [p.episode_id for p in plans],
        "readouts": torch.ones(len(plans), 5),
        "tail_windows": 0,
    }


def _check(payload, plans, contract, allow_legacy=False):
    return validate_rank_snapshot(
        payload,
        plans=plans,
        contract=contract,
        key_dim=3,
        value_dim=4,
        readout_dim=5,
        allow_legacy=allow_legacy,
    )


def test_rank_checkpoint_is_atomic_and_roundtrips(tmp_path):
    plans = _plans()
    contract = rank_contract(plans=plans, source={"cte": "abc", "base": "def"}, rank=1, world_size=2)
    payload = _payload(plans, contract)
    assert _check(payload, plans, contract) is payload
    path = tmp_path / "rank_001.pt"
    atomic_save_rank_snapshot(path, payload)
    assert path.is_file()
    assert not (tmp_path / "rank_001.pt.tmp").exists()
    restored = torch.load(path, map_location="cpu", weights_only=False)
    _check(restored, plans, contract)


def test_different_source_and_episode_order_fail_closed():
    plans = _plans()
    expected = rank_contract(plans=plans, source={"cte": "first"}, rank=0, world_size=2)
    payload = _payload(plans, expected)
    changed = rank_contract(plans=plans, source={"cte": "other"}, rank=0, world_size=2)
    payload["resume_contract"] = changed
    with pytest.raises(ValueError, match="source/episode/world-size identity"):
        _check(payload, plans, expected)

    payload["resume_contract"] = expected
    payload["entries"][0]["dataset_start"] = -1
    with pytest.raises(ValueError, match="entry 0 identity mismatch"):
        _check(payload, plans, expected)


def test_legacy_checkpoint_requires_explicit_opt_in():
    plans = _plans()
    contract = rank_contract(plans=plans, source={"cte": "abc"}, rank=0, world_size=2)
    payload = _payload(plans, contract)
    del payload["resume_contract"]
    with pytest.raises(ValueError, match="legacy rank snapshot"):
        _check(payload, plans, contract)
    assert _check(payload, plans, contract, allow_legacy=True) is payload


def test_bad_shape_nan_and_empty_window_rejected():
    plans = _plans()
    contract = rank_contract(plans=plans, source={"cte": "abc"}, rank=1, world_size=2)
    payload = _payload(plans, contract)
    payload["readouts"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        _check(payload, plans, contract)


def test_filesystem_wait_handles_done_and_missing(tmp_path):
    (tmp_path / "rank_000.pt").write_bytes(b"ok")
    wait_for_files([tmp_path / "rank_000.pt"], phase="rank files", timeout_seconds=.01, poll_seconds=.005)
    with pytest.raises(TimeoutError, match="rank_001.pt"):
        wait_for_files([tmp_path / "rank_001.pt"], phase="rank files", timeout_seconds=.01, poll_seconds=.005)
