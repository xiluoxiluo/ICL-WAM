"""CPU coverage for cache integrity and distributed storage-phase ordering.

The threaded rank harness exercises the entrypoint with real NumPy files and
Torch tensors; it replaces CUDA/VAE/video I/O and NCCL transport, not recovery
or capacity orchestration. It is not a GPU/NCCL integration test.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import errno
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


LATENT_SHAPE = (9, 2, 2, 2)
VAE_METADATA = {"model_id": "test-vae", "z_dim": 2,
                "temporal_downsample_factor": 1, "upsampling_factor": 1,
                "vae_path": "/test/vae"}


@pytest.fixture
def precompute(monkeypatch):
    # LeRobot's optional video/HF dependencies are unrelated to storage. Keep
    # the real Zeva dataset, array helpers and encoder contract importable.
    video_module = ModuleType("fastwam.datasets.lerobot.base_lerobot_dataset")
    video_module.BaseLerobotDataset = type("BaseLerobotDataset", (), {})
    monkeypatch.setitem(sys.modules, video_module.__name__, video_module)
    path = Path(__file__).resolve().parents[2] / "scripts/precompute_zeva_cte_latents.py"
    spec = importlib.util.spec_from_file_location("_latent_precompute_recovery", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _make_rank(module, root, *, rank=0, world_size=2, count=3, offset=1, done=False):
    directory = root / f"rank_{rank:03d}"
    directory.mkdir(parents=True)
    if count:
        arrays = module._create_arrays(
            directory, count=count, latent_shape=LATENT_SHAPE,
            action_shape=(8, 4, 14), frame_valid_shape=(9,), transition_valid_shape=(8, 4),
        )
        arrays["latents"][:] = 173
        arrays["actions"][:] = 3.25
        arrays["frame_valid"][:] = 1
        arrays["transition_valid"][:] = 1
        module._flush_arrays(arrays)
    meta = {"rank": rank, "world_size": world_size, "num_windows": count,
            "latent_shape": list(LATENT_SHAPE) if count else [],
            "latent_storage_dtype": module.LATENT_STORAGE_DTYPE,
            "cte_vae_input_size": [2, 2], "vae_metadata": VAE_METADATA}
    (directory / "rank_meta.json").write_text(json.dumps(meta))
    (directory / "progress.json").write_text(json.dumps({"next_offset": offset, "total": count}))
    if done:
        (directory / "DONE").write_text("ok\n")
    return directory


def _snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _inspect(module, directory, *, count=3, rank=0, world_size=2):
    return module._inspect_rank_state(
        directory, rank=rank, world_size=world_size, count=count,
        main_metadata={}, input_size=(2, 2), expected_channels=2,
    )


def test_resume_uses_only_committed_progress_and_preserves_arrays(precompute, tmp_path):
    directory = _make_rank(precompute, tmp_path)
    (directory / "progress.json.tmp").write_text('{"next_offset": 3, "total": 3}')
    (directory / "progress.json.interrupted.tmp").write_text('{"next_offset": 2')
    before = _snapshot(tmp_path)
    state = _inspect(precompute, directory)
    assert state["offset"] == 1
    assert state["shape"] == LATENT_SHAPE
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("key,value", [
    ("rank", 1), ("world_size", 3), ("num_windows", 4),
    ("latent_shape", [9, 3, 2, 2]), ("latent_storage_dtype", "float32"),
    ("cte_vae_input_size", [4, 4]),
])
def test_resume_rejects_incompatible_metadata_without_writing(precompute, tmp_path, key, value):
    directory = _make_rank(precompute, tmp_path)
    path = directory / "rank_meta.json"
    meta = json.loads(path.read_text())
    meta[key] = value
    path.write_text(json.dumps(meta))
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError):
        _inspect(precompute, directory)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("damage", ["dtype", "shape", "truncated", "trailing", "fortran"])
def test_array_validation_precedes_all_writable_maps(precompute, tmp_path, monkeypatch, damage):
    directory = _make_rank(precompute, tmp_path)
    path = directory / "transition_actions.npy"
    if damage == "dtype":
        np.save(path, np.zeros((3, 8, 4, 14), dtype=np.float16))
    elif damage == "shape":
        np.save(path, np.zeros((3, 8, 4, 13), dtype=np.float32))
    elif damage == "fortran":
        np.save(path, np.asfortranarray(np.zeros((3, 8, 4, 14), dtype=np.float32)))
    elif damage == "truncated":
        with path.open("r+b") as handle:
            handle.truncate(path.stat().st_size - 1)
    else:
        with path.open("ab") as handle:
            handle.write(b"x")
    before = _snapshot(tmp_path)
    monkeypatch.setattr(precompute.np, "load", lambda *args, **kwargs: pytest.fail("opened writable map before validating all files"))
    with pytest.raises(ValueError):
        precompute._open_existing_arrays(directory, count=3, latent_shape=LATENT_SHAPE)
    assert _snapshot(tmp_path) == before


def test_reuses_valid_uncommitted_files_and_creates_only_missing_files(precompute, tmp_path):
    directory = _make_rank(precompute, tmp_path, offset=0)
    keep = directory / "latents_bf16_u16.npy"
    before = keep.read_bytes()
    (directory / "transition_actions.npy").unlink()
    _inspect(precompute, directory)
    arrays = precompute._create_arrays(
        directory, count=3, latent_shape=LATENT_SHAPE, action_shape=(8, 4, 14),
        frame_valid_shape=(9,), transition_valid_shape=(8, 4),
    )
    assert keep.read_bytes() == before
    assert np.all(arrays["latents"] == 173)
    assert arrays["actions"].shape == (3, 8, 4, 14)


def test_missing_committed_array_is_not_recreated(precompute, tmp_path):
    directory = _make_rank(precompute, tmp_path, offset=1)
    missing = directory / "frame_valid.npy"
    missing.unlink()
    before = _snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        _inspect(precompute, directory)
    assert _snapshot(tmp_path) == before


def test_done_requires_fully_committed_progress(precompute, tmp_path):
    directory = _make_rank(precompute, tmp_path, offset=1, done=True)
    with pytest.raises(ValueError, match="fully committed"):
        _inspect(precompute, directory)


def test_generation_check_rejects_changed_world_size_with_same_local_count(precompute, tmp_path):
    _make_rank(precompute, tmp_path, world_size=2, count=3)
    with pytest.raises(ValueError, match="world_size mismatch"):
        precompute._validate_generation_layout(tmp_path, world_size=1)


def test_generation_check_rejects_orphaned_out_of_range_rank(precompute, tmp_path):
    directory = tmp_path / "rank_003"
    directory.mkdir()
    (directory / "progress.json").write_text('{"next_offset": 1, "total": 1}')
    with pytest.raises(ValueError, match="outside current world_size"):
        precompute._validate_generation_layout(tmp_path, world_size=2)


def test_storage_plan_counts_sparse_allocation_not_logical_size(precompute, tmp_path):
    directory = tmp_path / "rank_000"
    directory.mkdir()
    shape = (9, 2, 64, 64)
    precompute._create_arrays(directory, count=8, latent_shape=shape,
                              action_shape=(8, 4, 14), frame_valid_shape=(9,), transition_valid_shape=(8, 4))
    plan = precompute._storage_plan(directory, 8, shape)
    assert plan["expected_array_bytes"] == sum(path.stat().st_size for path in directory.glob("*.npy"))
    if plan["allocated_bytes"] >= plan["logical_bytes"] / 2:
        pytest.skip("test filesystem does not preserve sparse file allocation")
    assert plan["required_additional_bytes"] > plan["logical_bytes"] / 2
    empty = precompute._storage_plan(tmp_path / "rank_001", 0, shape)
    assert empty["expected_array_bytes"] == empty["required_additional_bytes"] == 0


def test_actual_resume_encode_shape_must_match_before_any_assignment(precompute):
    items = [SimpleNamespace(frames=torch.zeros(9, 3, 2, 2))]
    with pytest.raises(ValueError, match="differs from cache shape"):
        precompute._encode_batch(items, lambda _: torch.zeros(1, 9, 2, 3, 2),
                                 torch.device("cpu"), LATENT_SHAPE)


def test_process_group_binds_gpu_and_forwards_long_timeout(precompute, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    events = []
    monkeypatch.setattr(precompute.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(precompute.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(precompute.torch.cuda, "set_device", lambda rank: events.append(("bind", rank)))
    monkeypatch.setattr(precompute.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(precompute.dist, "init_process_group", lambda **kwargs: events.append(("init", kwargs)))
    assert precompute._distributed_context() == (1, 1, 2, False)
    assert events[0] == ("bind", 1)
    assert events[1][1]["timeout"] == timedelta(seconds=86400)
    events.clear()
    precompute._distributed_context(timeout_seconds=123)
    assert events[1][1]["timeout"] == timedelta(seconds=123)


@pytest.mark.parametrize("seconds", [0, -1])
def test_invalid_process_group_timeout_rejected_before_cuda(precompute, seconds):
    with pytest.raises(ValueError, match="must be positive"):
        precompute._distributed_context(timeout_seconds=seconds)


class _ThreadRanks:
    """Bounded CPU transport to detect inconsistent collective participation."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.thread = threading.local()
        self.sync = threading.Barrier(world_size, timeout=10)
        self.lock = threading.Lock()
        self.values = {}

    def all_gather_object(self, output, value):
        sequence = getattr(self.thread, "sequence", 0)
        self.thread.sequence = sequence + 1
        with self.lock:
            self.values.setdefault(sequence, {})[self.thread.rank] = value
        self.sync.wait()
        output[:] = [self.values[sequence][i] for i in range(self.world_size)]
        self.sync.wait()

    def barrier(self):
        self.sync.wait()

    @staticmethod
    def is_available():
        return False

    def run(self, callback):
        def worker(rank):
            self.thread.rank = rank
            self.thread.sequence = 0
            return callback()
        with ThreadPoolExecutor(max_workers=self.world_size) as pool:
            futures = [pool.submit(worker, rank) for rank in range(self.world_size)]
            errors = []
            for future in futures:
                try:
                    future.result(timeout=20)
                except Exception as exc:
                    errors.append(exc)
        return errors


def _mock_entrypoint(module, monkeypatch, tmp_path, *, world_size=3):
    ranks = _ThreadRanks(world_size)
    read_indices, encodings, reservations, capacity_calls = [], [], [], []
    cache = tmp_path / "cache"
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    task_map = tmp_path / "tasks.json"
    task_map.write_text("{}")
    rows = [module.CTETrainIndex(0, "source::episode-0", "task", 0),
            module.CTETrainIndex(1, "source::episode-0", "task", 32),
            module.CTETrainIndex(2, "source::episode-1", "task", 0),
            module.CTETrainIndex(3, "source::episode-1", "task", 32)]

    class Source:
        def __getitem__(self, index):
            read_indices.append(index)
            row = rows[index]
            return {"dataset_index": index,
                    "episode": SimpleNamespace(episode_id=row.episode_id, episode_step=row.episode_step, task_id="task"),
                    "cte_frames": torch.full((9, 3, 2, 2), float(index + 1)),
                    "transition_actions": torch.full((8, 4, 14), float(index)),
                    "frame_valid": torch.ones(9, dtype=torch.bool),
                    "transition_valid": torch.ones(8, 4, dtype=torch.bool)}

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_history(self, frames):
            encodings.append((ranks.thread.rank, len(frames)))
            return frames[:, :, :2].to(torch.bfloat16)

    class TorchCPU:
        def __getattr__(self, key):
            return getattr(torch, key)

        @staticmethod
        def device(*args, **kwargs):
            return torch.device("cpu")

    base = SimpleNamespace(require_semantic_task_id=True, semantic_task_identity={"kind": "test"},
                           resolve_semantic_task_id=lambda **kwargs: "task")
    monkeypatch.setattr(module, "torch", TorchCPU())
    monkeypatch.setattr(module, "dist", ranks)
    monkeypatch.setattr(module, "_distributed_context", lambda **kwargs: (
        ranks.thread.rank, ranks.thread.rank, ranks.world_size, ranks.thread.rank == 0,
    ))
    monkeypatch.setattr(module, "normalize_main_dataset_paths", lambda cfg: None)
    monkeypatch.setattr(module, "instantiate", lambda cfg: base)
    monkeypatch.setattr(module, "ZevaRobotWinDataset", lambda base: Source())
    monkeypatch.setattr(module, "build_cte_training_index", lambda *args, **kwargs: rows)
    monkeypatch.setattr(module, "select_main_success_rows", lambda rows, cfg: (rows, {}))
    monkeypatch.setattr(module, "is_main_profile", lambda cfg: False)
    monkeypatch.setattr(module, "load_frozen_wan_vae", lambda **kwargs: (object(), dict(VAE_METADATA)))
    monkeypatch.setattr(module, "FastWAMCTELatentEncoder", Encoder)

    def check_capacity(root, additional, *, margin_bytes):
        capacity_calls.append((additional, margin_bytes))
        return {"available_bytes": 10**12}

    def reserve(paths):
        reservations.append(ranks.thread.rank)
        return {"unsupported_paths": [], "reservation_complete": True}

    monkeypatch.setattr(module, "check_storage_capacity", check_capacity)
    monkeypatch.setattr(module, "reserve_file_storage", reserve)
    cfg = {"model": {"zeva": {"cte": {"latent_cache_path": str(cache), "image_channels": 2}}},
           "data": {"train": {"dataset_dirs": ["source"], "video_size": [2, 2],
                                "pretrained_norm_stats": str(stats), "semantic_task_map_path": str(task_map)}},
           "cte_latent_cache": {"batch_size": 1, "num_workers": 0, "flush_every_batches": 1}}
    run = lambda: ranks.run(lambda: module.main.__wrapped__(OmegaConf.create(cfg)))
    return SimpleNamespace(run=run, ranks=ranks, cache=cache, reads=read_indices,
                           encodings=encodings, reservations=reservations, capacity=capacity_calls)


def test_fresh_probe_is_reused_and_empty_rank_joins_all_storage_phases(precompute, monkeypatch, tmp_path):
    job = _mock_entrypoint(precompute, monkeypatch, tmp_path)
    assert job.run() == []
    assert sorted(job.reads) == [0, 1, 2, 3]
    assert sum(count for _, count in job.encodings) == 4
    assert sorted(job.reservations) == [0, 1]
    assert (job.cache / "rank_002/DONE").is_file()
    assert json.loads((job.cache / "manifest.json").read_text())["rank_window_counts"] == [2, 2, 0]
    assert job.capacity[0][0] > 0 and job.capacity[0][1] >= 256 * 1024**2


def test_mixed_done_empty_and_pending_resume_does_not_reencode_committed_rows(precompute, monkeypatch, tmp_path):
    job = _mock_entrypoint(precompute, monkeypatch, tmp_path)
    assert job.run() == []
    completed = _snapshot(job.cache / "rank_000")
    pending = job.cache / "rank_001"
    (pending / "DONE").unlink()
    (pending / "progress.json").write_text('{"next_offset": 1, "total": 2}')
    prefix = np.load(pending / "latents_bf16_u16.npy")[0].copy()
    job.reads.clear()
    job.encodings.clear()
    job.reservations.clear()
    assert job.run() == []
    assert job.reads == [3]
    assert job.encodings == [(1, 1)]
    assert job.reservations == [1]
    assert _snapshot(job.cache / "rank_000") == completed
    assert np.array_equal(np.load(pending / "latents_bf16_u16.npy")[0], prefix)


def test_all_done_skips_encoder_reservation_and_large_headroom(precompute, monkeypatch, tmp_path):
    job = _mock_entrypoint(precompute, monkeypatch, tmp_path)
    assert job.run() == []
    job.reads.clear()
    job.encodings.clear()
    job.reservations.clear()
    monkeypatch.setattr(precompute, "load_frozen_wan_vae", lambda **kwargs: pytest.fail("loaded VAE for a complete cache"))
    assert job.run() == []
    assert job.reads == job.encodings == job.reservations == []
    assert job.capacity[-1] == (0, 0)


def test_reservation_failure_reaches_done_and_empty_ranks_without_modifying_committed_cache(precompute, monkeypatch, tmp_path):
    job = _mock_entrypoint(precompute, monkeypatch, tmp_path)
    assert job.run() == []
    pending = job.cache / "rank_001"
    (pending / "DONE").unlink()
    (pending / "progress.json").write_text('{"next_offset": 1, "total": 2}')
    before = _snapshot(job.cache)
    job.encodings.clear()

    def quota_error(paths):
        assert job.ranks.thread.rank == 1
        raise OSError(errno.EDQUOT, "quota exhausted")

    monkeypatch.setattr(precompute, "reserve_file_storage", quota_error)
    errors = job.run()
    assert len(errors) == 3
    assert all(isinstance(exc, RuntimeError) and "rank 1" in str(exc) and "Disk quota exceeded" in str(exc) for exc in errors)
    assert all("original generation world size" in str(exc) for exc in errors)
    assert job.encodings == []
    assert _snapshot(job.cache) == before


def test_final_metadata_quota_failure_is_reported_to_every_rank(precompute, monkeypatch, tmp_path):
    job = _mock_entrypoint(precompute, monkeypatch, tmp_path)
    write_json = precompute._atomic_json

    def quota_on_index(path, payload):
        if path.name == "index.json":
            raise OSError(errno.EDQUOT, "quota exhausted")
        return write_json(path, payload)

    monkeypatch.setattr(precompute, "_atomic_json", quota_on_index)
    errors = job.run()
    assert len(errors) == 3
    assert all("final index/manifest commit" in str(exc) and "Disk quota exceeded" in str(exc) for exc in errors)
    for rank, count in enumerate((2, 2, 0)):
        directory = job.cache / f"rank_{rank:03d}"
        assert (directory / "DONE").is_file()
        assert json.loads((directory / "progress.json").read_text())["next_offset"] == count


@pytest.mark.parametrize("failure_stage", ["array_flush", "progress_commit"])
def test_write_failure_resumes_from_previous_committed_offset(precompute, monkeypatch, tmp_path, failure_stage):
    job = _mock_entrypoint(precompute, monkeypatch, tmp_path, world_size=1)
    flush = precompute._flush_arrays
    atomic = precompute._atomic_json
    flush_count = 0

    def fail_flush(arrays):
        nonlocal flush_count
        flush_count += 1
        if flush_count == 2:
            raise OSError(errno.EDQUOT, "quota exhausted")
        return flush(arrays)

    def fail_commit(path, payload):
        if path.name == "progress.json" and payload["next_offset"] == 2:
            raise OSError(errno.EDQUOT, "quota exhausted")
        return atomic(path, payload)

    if failure_stage == "array_flush":
        monkeypatch.setattr(precompute, "_flush_arrays", fail_flush)
    else:
        monkeypatch.setattr(precompute, "_atomic_json", fail_commit)
    errors = job.run()
    assert len(errors) == 1 and isinstance(errors[0], precompute.CacheStorageError)
    directory = job.cache / "rank_000"
    assert json.loads((directory / "progress.json").read_text())["next_offset"] == 1
    assert not (directory / "DONE").exists()
    prefix = np.load(directory / "latents_bf16_u16.npy")[0].copy()

    monkeypatch.setattr(precompute, "_flush_arrays", flush)
    monkeypatch.setattr(precompute, "_atomic_json", atomic)
    job.reads.clear()
    assert job.run() == []
    assert job.reads == [1, 2, 3]
    assert np.array_equal(np.load(directory / "latents_bf16_u16.npy")[0], prefix)
    assert json.loads((directory / "progress.json").read_text())["next_offset"] == 4
