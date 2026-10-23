"""Quota failures must not erase a committed mmap cache or its progress."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fastwam import zeva_latent_storage as storage


ROOT = Path(__file__).resolve().parents[2]


def test_window_payload_and_metadata_margin_use_exact_raw_storage_units():
    assert storage.cache_window_payload_bytes((9, 48, 24, 20)) == 416553
    assert storage.recommended_storage_margin_bytes(0) == 256 * 1024**2
    assert storage.recommended_storage_margin_bytes(400 * 1024**3) == 4 * 1024**3
    with pytest.raises(ValueError, match="positive"):
        storage.cache_window_payload_bytes((9, 48, 0, 20))


def test_sparse_file_accounting_does_not_count_logical_bytes_as_allocated(tmp_path):
    path = tmp_path / "latents.npy"
    with path.open("wb") as handle:
        handle.write(b"header")
        handle.truncate(8 * 1024**2)
    linked = tmp_path / "same_file.npy"
    os.link(path, linked)
    missing = tmp_path / "not_created.npy"
    report = storage.storage_usage([path, path, linked, missing])
    assert report["file_count"] == 1
    assert report["logical_bytes"] == 8 * 1024**2
    assert report["allocated_bytes"] == path.stat().st_blocks * 512
    assert storage.allocated_file_bytes(path) == report["allocated_bytes"]
    assert report["allocated_bytes_exact"]
    assert report["missing_paths"] == [str(missing)]


def test_filesystem_check_is_read_only_and_does_not_claim_quota_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=4096))
    absent_cache = tmp_path / "not_created" / "cache"
    report = storage.check_storage_capacity(absent_cache, 3000, margin_bytes=500)
    assert report["required_with_margin_bytes"] == 3500
    assert report["available_bytes"] == 4096
    assert report["quota_verified"] is False
    assert not absent_cache.parent.exists()
    with pytest.raises(storage.CacheStorageError) as caught:
        storage.check_storage_capacity(absent_cache, 3500, margin_bytes=1000)
    assert caught.value.errno == errno.ENOSPC
    assert "user/project quota" in str(caught.value)


@pytest.mark.parametrize("error", [errno.ENOSPC, errno.EDQUOT])
def test_reservation_capacity_failure_preserves_existing_array_and_errno(tmp_path, monkeypatch, error):
    path = tmp_path / "latents.npy"
    original = b"header and previously committed latent values"
    path.write_bytes(original)
    opened = []
    real_open = storage.os.open

    def checked_open(filename, flags):
        opened.append(flags)
        return real_open(filename, flags)

    def fail(descriptor, offset, length):
        assert offset == 0 and length == len(original)
        raise OSError(error, "simulated capacity failure")

    monkeypatch.setattr(storage.os, "open", checked_open)
    monkeypatch.setattr(storage.os, "posix_fallocate", fail, raising=False)
    with pytest.raises(storage.CacheStorageError) as caught:
        storage.reserve_file_storage([path])
    assert caught.value.errno == error
    assert "same generation world size" in str(caught.value)
    assert opened == [os.O_RDWR]
    assert path.read_bytes() == original


@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EOPNOTSUPP])
def test_only_explicitly_unsupported_reservation_is_reported_as_fallback(tmp_path, monkeypatch, error):
    path = tmp_path / "latents.npy"
    path.write_bytes(b"committed")

    def unsupported(*args):
        raise OSError(error, "unsupported")

    monkeypatch.setattr(storage.os, "posix_fallocate", unsupported, raising=False)
    report = storage.reserve_file_storage([path])
    assert report["unsupported_paths"] == [str(path)]
    assert report["unsupported_details"][0]["errno"] == error
    assert report["reservation_complete"] is False
    assert report["quota_verified"] is False
    assert path.read_bytes() == b"committed"


def test_missing_reservation_api_is_explicit_and_zero_length_files_are_skipped(tmp_path, monkeypatch):
    path, empty = tmp_path / "latents.npy", tmp_path / "empty"
    path.write_bytes(b"committed")
    empty.touch()
    monkeypatch.delattr(storage.os, "posix_fallocate", raising=False)
    report = storage.reserve_file_storage([path, empty])
    assert report["unsupported_paths"] == [str(path)]
    assert report["empty_paths"] == [str(empty)]
    assert report["reservation_complete"] is False


def test_unexpected_allocation_error_is_not_silently_ignored(tmp_path, monkeypatch):
    path = tmp_path / "latents.npy"
    path.write_bytes(b"committed")

    def invalid(*args):
        raise OSError(errno.EINVAL, "unexpected invalid argument")

    monkeypatch.setattr(storage.os, "posix_fallocate", invalid, raising=False)
    with pytest.raises(OSError) as caught:
        storage.reserve_file_storage([path])
    assert caught.value.errno == errno.EINVAL
    assert path.read_bytes() == b"committed"


def test_small_real_reservation_preserves_entire_existing_file(tmp_path):
    if not hasattr(os, "posix_fallocate"):
        pytest.skip("POSIX file allocation is unavailable")
    path = tmp_path / "latents.npy"
    with path.open("wb") as handle:
        handle.write(b"cached header and committed prefix")
        handle.truncate(16 * 1024)
    before = path.read_bytes()
    report = storage.reserve_file_storage([path])
    if report["unsupported_paths"]:
        pytest.skip("This filesystem does not support POSIX file allocation")
    assert report["reserved_paths"] == [str(path)]
    assert report["reserved_logical_bytes"] == len(before)
    assert report["reservation_complete"] is True
    assert report["quota_verified"] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("stage", ["create", "write", "flush", "close", "fsync", "replace"])
@pytest.mark.parametrize("error", [errno.ENOSPC, errno.EDQUOT])
def test_failed_atomic_progress_commit_keeps_old_record_and_cleans_own_temp(
    tmp_path, monkeypatch, stage, error,
):
    path = tmp_path / "progress.json"
    original = b'{"next_offset": 11456, "total": 99736}\n'
    path.write_bytes(original)

    def fail(*args, **kwargs):
        raise OSError(error, "simulated metadata quota failure")

    if stage == "create":
        monkeypatch.setattr(storage.tempfile, "mkstemp", fail)
    elif stage in {"write", "flush", "close"}:
        real_fdopen = storage.os.fdopen

        class InterruptedFile:
            def __init__(self, *args, **kwargs):
                self.handle = real_fdopen(*args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.handle.close()
                if stage == "close":
                    fail()

            def write(self, text):
                if stage == "write":
                    self.handle.write(text[:10])
                    self.handle.flush()
                    fail()
                return self.handle.write(text)

            def flush(self):
                self.handle.flush()
                if stage == "flush":
                    fail()

            def fileno(self):
                return self.handle.fileno()

        monkeypatch.setattr(storage.os, "fdopen", InterruptedFile)
    else:
        monkeypatch.setattr(storage.os, stage, fail)
    with pytest.raises(storage.CacheStorageError) as caught:
        storage.atomic_write_json(path, {"next_offset": 11520, "total": 99736})
    assert caught.value.errno == error
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_successful_atomic_commit_and_invalid_payload_do_not_leave_temporary_files(tmp_path):
    path = tmp_path / "rank_000" / "progress.json"
    payload = {"next_offset": 11520, "total": 99736}
    storage.atomic_write_json(path, payload)
    before = path.read_bytes()
    assert json.loads(before) == payload
    with pytest.raises(ValueError):
        storage.atomic_write_json(path, {"next_offset": float("nan")})
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("dtype,itemsize", [("uint16", 2), ("float32", 4), ("uint8", 1)])
def test_header_reader_matches_real_numpy_arrays_without_reading_payload(tmp_path, dtype, itemsize):
    import numpy as np

    path = tmp_path / "array.npy"
    np.save(path, np.arange(24, dtype=dtype).reshape(2, 3, 4))
    report = storage.read_npy_header(path)
    assert report["shape"] == [2, 3, 4]
    assert report["itemsize"] == itemsize
    assert report["payload_bytes"] == 24 * itemsize
    assert report["expected_file_bytes"] == path.stat().st_size
    assert report["logical_bytes"] == path.stat().st_size
    assert not report["is_truncated"] and report["trailing_bytes"] == 0


def _manual_npy(path, version=(2, 0), descriptor="<u2", shape=(2, 3), payload_bytes=12):
    header = repr({"descr": descriptor, "fortran_order": False, "shape": shape}).encode("utf-8") + b"\n"
    length_format = "<H" if version == (1, 0) else "<I"
    path.write_bytes(b"\x93NUMPY" + bytes(version) + struct.pack(length_format, len(header)) + header + bytes(payload_bytes))


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_header_reader_reports_truncated_payload_without_reading_or_repairing_it(tmp_path, version):
    path = tmp_path / "array.npy"
    _manual_npy(path, version=version, payload_bytes=2)
    before = path.read_bytes()
    report = storage.read_npy_header(path)
    assert report["format_version"] == list(version)
    assert report["payload_bytes"] == 12
    assert report["expected_file_bytes"] - report["logical_bytes"] == 10
    assert report["is_truncated"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("defect", ["magic", "short_header", "object", "negative_shape", "oversize_header"])
def test_corrupt_or_unsafe_header_is_rejected(tmp_path, defect):
    path = tmp_path / "array.npy"
    _manual_npy(path)
    kwargs = {}
    if defect == "magic":
        path.write_bytes(b"not an array")
    elif defect == "short_header":
        path.write_bytes(path.read_bytes()[:20])
    elif defect == "object":
        _manual_npy(path, descriptor="|O")
    elif defect == "negative_shape":
        _manual_npy(path, shape=(-1, 3))
    else:
        kwargs["max_header_bytes"] = 4
    with pytest.raises(ValueError):
        storage.read_npy_header(path, **kwargs)


def test_storage_helpers_import_under_stdlib_only_isolated_python():
    command = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "from fastwam import zeva_latent_storage; "
        "assert 'numpy' not in sys.modules and 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
