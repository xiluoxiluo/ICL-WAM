"""Storage checks for the existing mmap CTE cache, using only the stdlib.

Filesystem free space and a user's/project's remaining quota are different
limits. Reserving an existing array's file range can expose EDQUOT before the
expensive encoding loop, but does not guarantee space for future metadata or
protect against other processes consuming storage afterwards.
"""

from __future__ import annotations

import ast
import errno
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import tempfile
from typing import Any, Iterable, Sequence


DEFAULT_STORAGE_MARGIN_BYTES = 256 * 1024**2
_CAPACITY_ERRORS = {errno.ENOSPC, getattr(errno, "EDQUOT", 122)}
_UNSUPPORTED_RESERVATION_ERRORS = {
    errno.ENOSYS,
    getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    getattr(errno, "ENOTSUP", errno.ENOSYS),
}


class CacheStorageError(OSError):
    """An actionable cache capacity/quota failure, preserving its errno."""


def _nonnegative_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _shape_product(shape: Sequence[int], *, allow_zero: bool = False) -> int:
    if not isinstance(shape, (tuple, list)) or not shape:
        raise ValueError("array shape must be a nonempty tuple/list of integers")
    for dimension in shape:
        _nonnegative_integer(dimension, "shape dimension")
        if dimension == 0 and not allow_zero:
            raise ValueError("window dimensions must be positive")
    return math.prod(shape)


def cache_window_payload_bytes(
    latent_shape: Sequence[int],
    action_shape: Sequence[int] = (8, 4, 14),
    frame_valid_shape: Sequence[int] = (9,),
    transition_valid_shape: Sequence[int] = (8, 4),
) -> int:
    """Bytes per window for uint16 BF16 bits, float32 actions and uint8 masks.

    NPY headers, provenance, index JSON and filesystem allocation granularity
    are additional. Use actual latent dimensions, not a guessed VAE stride.
    """
    return (
        2 * _shape_product(latent_shape)
        + 4 * _shape_product(action_shape)
        + _shape_product(frame_valid_shape)
        + _shape_product(transition_valid_shape)
    )


def recommended_storage_margin_bytes(payload_bytes: int) -> int:
    """Conservative free-space headroom for metadata; not a quota reservation."""
    payload_bytes = _nonnegative_integer(payload_bytes, "payload_bytes")
    return max(DEFAULT_STORAGE_MARGIN_BYTES, (payload_bytes + 99) // 100)


def allocated_file_bytes(path: str | Path) -> int:
    """Filesystem-reported blocks, not sparse length or quota accounting.

    POSIX st_blocks uses 512-byte units. Shared extents, compression and
    filesystem-specific quota rules can make quota charges differ from this.
    """
    info = Path(path).stat()
    return int(info.st_blocks) * 512 if hasattr(info, "st_blocks") else int(info.st_size)


def storage_usage(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Inspect existing files without reading arrays or changing their contents.

    Missing paths are reported and contribute zero. Hard links are counted
    once so that an allocation estimate does not double-count the same file.
    On platforms without st_blocks, allocated_bytes is a logical-size fallback.
    allocated_bytes_exact distinguishes reported blocks from that fallback;
    it does not mean an exact user/project quota charge.
    """
    logical = allocated = count = 0
    allocated_exact = True
    missing: list[str] = []
    seen: set[tuple[int, int]] = set()
    for value in paths:
        path = Path(value)
        try:
            info = path.stat()
        except FileNotFoundError:
            missing.append(str(path))
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Expected a regular cache file: {path}")
        identity = (info.st_dev, info.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        count += 1
        logical += int(info.st_size)
        if hasattr(info, "st_blocks"):
            allocated += int(info.st_blocks) * 512
        else:
            allocated_exact = False
            allocated += int(info.st_size)
    return {
        "file_count": count,
        "logical_bytes": logical,
        "allocated_bytes": allocated,
        "allocated_bytes_exact": allocated_exact,
        "allocation_scope": "filesystem-reported blocks, not user/project quota accounting",
        "missing_paths": missing,
    }


def _existing_directory(path: str | Path) -> Path:
    current = Path(path).expanduser().resolve()
    while not current.exists():
        if current == current.parent:
            raise FileNotFoundError(f"No existing filesystem ancestor for {path}")
        current = current.parent
    return current if current.is_dir() else current.parent


def _gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def check_storage_capacity(
    cache_root: str | Path,
    required_additional_bytes: int,
    *,
    margin_bytes: int = DEFAULT_STORAGE_MARGIN_BYTES,
) -> dict[str, Any]:
    """Check filesystem availability only; this does not check project quota.

    ``required_additional_bytes`` must subtract already allocated blocks, not
    the logical size of pre-created sparse mmap files. The caller must account
    for all ranks together when they share the filesystem.
    """
    required = _nonnegative_integer(required_additional_bytes, "required_additional_bytes")
    margin = _nonnegative_integer(margin_bytes, "margin_bytes")
    directory = _existing_directory(cache_root)
    usage = shutil.disk_usage(directory)
    report = {
        "filesystem_path": str(directory),
        "available_bytes": int(usage.free),
        "required_additional_bytes": required,
        "margin_bytes": margin,
        "required_with_margin_bytes": required + margin,
        "quota_verified": False,
        "scope": "filesystem availability; user/project quota is not verified",
    }
    if required + margin > usage.free:
        raise CacheStorageError(
            errno.ENOSPC,
            f"CTE cache needs {_gib(required)} additional array storage plus "
            f"{_gib(margin)} metadata headroom, but the filesystem reports only "
            f"{_gib(usage.free)} available. Preserve the existing cache and "
            "progress.json; free storage or move the complete cache to a "
            "filesystem with enough space and quota before resuming with the "
            "same generation world size. Filesystem availability does not "
            "prove that your user/project quota is sufficient.",
            str(cache_root),
        )
    return report


def _capacity_failure(exc: OSError, path: str | Path, operation: str) -> CacheStorageError:
    cause = "Disk quota exceeded" if exc.errno == getattr(errno, "EDQUOT", 122) else "No space left on device"
    return CacheStorageError(
        exc.errno,
        f"{cause} while {operation}. Preserve existing cache files and the last "
        "committed progress.json. Free/extend the relevant user or project "
        "quota, or move the complete cache to a filesystem with enough space "
        "and quota, then resume with the same generation world size. df/free "
        "space alone does not establish remaining quota.",
        str(path),
    )


def reserve_file_storage(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Reserve existing array file ranges without truncating or changing values.

    Call before writing the mmap arrays and with no concurrent writer to the
    same files. Some libc implementations emulate posix_fallocate; concurrent
    writes during that emulation are unsafe. Other ranks may reserve distinct
    files concurrently. A failed reservation leaves the arrays and committed
    progress intact; some blocks may already have been allocated.

    Only an unavailable API or explicit unsupported-operation errno falls back
    to an unverified report. EDQUOT/ENOSPC and unexpected errors are never
    silently treated as unsupported. Metadata quota remains unverified even
    after every array range has been successfully reserved.
    """
    reserve = getattr(os, "posix_fallocate", None)
    reserved: list[str] = []
    empty: list[str] = []
    unsupported: list[str] = []
    details: list[dict[str, Any]] = []
    total = 0
    seen: set[tuple[int, int]] = set()
    for value in paths:
        path = Path(value)
        # O_RDWR is required for POSIX allocation. Never use O_TRUNC, O_CREAT
        # or O_APPEND: these files may contain a committed cache prefix.
        descriptor = os.open(path, os.O_RDWR)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"Expected a regular cache file: {path}")
            identity = (info.st_dev, info.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            if info.st_size == 0:
                empty.append(str(path))
                continue
            if reserve is None:
                unsupported.append(str(path))
                details.append({"path": str(path), "errno": None, "reason": "os.posix_fallocate is unavailable"})
                continue
            try:
                reserve(descriptor, 0, int(info.st_size))
            except OSError as exc:
                if exc.errno in _CAPACITY_ERRORS:
                    raise _capacity_failure(exc, path, "reserving the existing CTE array range") from exc
                if exc.errno in _UNSUPPORTED_RESERVATION_ERRORS:
                    unsupported.append(str(path))
                    details.append({"path": str(path), "errno": exc.errno, "reason": str(exc)})
                    continue
                raise
            reserved.append(str(path))
            total += int(info.st_size)
        finally:
            os.close(descriptor)
    return {
        "reserved_paths": reserved,
        "reserved_logical_bytes": total,
        "empty_paths": empty,
        "unsupported_paths": unsupported,
        "unsupported_details": details,
        "reservation_complete": not unsupported,
        "quota_verified": False,
        "scope": "existing array file ranges only; later metadata and concurrent quota use are not guaranteed",
    }


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Commit JSON only after a successful write and file fsync.

    A quota/full-filesystem error while creating, writing, syncing or replacing
    the temporary file leaves the previous target untouched until a successful
    atomic replace. This does not claim power-loss durability of directory
    entries on every filesystem. The cache writer must flush array data before
    committing a progress offset with this function.
    """
    target = Path(path)
    temporary: Path | None = None
    # Serialize before touching the filesystem; e.g. NaN cannot replace a valid
    # progress record or leave a half-written target behind.
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        if exc.errno in _CAPACITY_ERRORS:
            raise _capacity_failure(exc, target, "saving cache metadata/progress") from exc
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Do not hide the original write/commit error with a cleanup
                # error. A leftover uniquely named temp is never a checkpoint.
                pass


def read_npy_header(path: str | Path, *, max_header_bytes: int = 1024**2) -> dict[str, Any]:
    """Read a primitive NPY header and file allocation without importing NumPy.

    Cache payload is never read. Structured/object dtypes are intentionally
    rejected: the CTE cache uses plain uint16/float32/uint8 arrays. A truncated
    data payload is reported rather than read or repaired.
    """
    source = Path(path)
    _nonnegative_integer(max_header_bytes, "max_header_bytes")
    with source.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
            raise ValueError(f"Invalid or truncated NPY magic/version: {source}")
        version = tuple(prefix[6:8])
        if version == (1, 0):
            length_size, format_code, encoding = 2, "<H", "latin1"
        elif version in {(2, 0), (3, 0)}:
            length_size, format_code, encoding = 4, "<I", "utf-8" if version == (3, 0) else "latin1"
        else:
            raise ValueError(f"Unsupported NPY format version {version}: {source}")
        length_raw = handle.read(length_size)
        if len(length_raw) != length_size:
            raise ValueError(f"Truncated NPY header length: {source}")
        header_size = struct.unpack(format_code, length_raw)[0]
        if header_size > max_header_bytes:
            raise ValueError(f"NPY header exceeds {max_header_bytes} bytes: {source}")
        raw = handle.read(header_size)
        if len(raw) != header_size:
            raise ValueError(f"Truncated NPY header: {source}")
        try:
            header = ast.literal_eval(raw.decode(encoding).strip())
        except (ValueError, SyntaxError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid NPY header dictionary: {source}") from exc
        if not isinstance(header, dict) or set(header) != {"shape", "descr", "fortran_order"}:
            raise ValueError(f"Unexpected NPY header fields: {source}")
        shape = header["shape"]
        if not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid NPY shape: {source}")
        if type(header["fortran_order"]) is not bool:
            raise ValueError(f"Invalid NPY array order: {source}")
        descriptor = header["descr"]
        match = re.fullmatch(r"[<>=|]([biufcSUV])(\d+)", descriptor) if isinstance(descriptor, str) else None
        if match is None:
            raise ValueError(f"Unsupported nonprimitive/object NPY dtype {descriptor!r}: {source}")
        itemsize = int(match.group(2)) * (4 if match.group(1) == "U" else 1)
        if itemsize < 1:
            raise ValueError(f"Invalid NPY item size: {source}")
        data_offset = 8 + length_size + header_size
        payload_bytes = math.prod(shape) * itemsize
        info = os.fstat(handle.fileno())
    expected = data_offset + payload_bytes
    return {
        "path": str(source),
        "format_version": list(version),
        "shape": list(shape),
        "dtype": descriptor,
        "itemsize": itemsize,
        "fortran_order": header["fortran_order"],
        "data_offset": data_offset,
        "payload_bytes": payload_bytes,
        "expected_file_bytes": expected,
        "logical_bytes": int(info.st_size),
        "allocated_bytes": int(info.st_blocks) * 512 if hasattr(info, "st_blocks") else int(info.st_size),
        "allocated_bytes_exact": hasattr(info, "st_blocks"),
        "allocation_scope": "filesystem-reported blocks, not user/project quota accounting",
        "is_truncated": info.st_size < expected,
        "trailing_bytes": max(0, info.st_size - expected),
    }
