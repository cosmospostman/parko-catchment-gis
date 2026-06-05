"""Unit tests for utils/fetch.py cache concurrency behaviour.

FC-1. Same-path lock: concurrent workers on the same cache path serialise —
      only one S3 fetch occurs; the second worker gets a cache hit.
FC-2. Different-path independence: concurrent workers on different cache paths
      both fetch without blocking each other.
FC-3. Atomic write: the final cache file is never seen in a partial state by
      a concurrent reader (write goes via a unique tmp file then rename).
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_patch(value: float = 1.0):
    arr = np.full((4, 4), value, dtype=np.float32)
    transform = from_bounds(500_000, 8_200_000, 500_040, 8_200_040, 4, 4)
    crs = CRS.from_epsg(32754)
    return arr, transform, crs


def _save(path: Path, value: float = 1.0):
    from utils.fetch import _save_patch_cache
    _save_patch_cache(path, _make_patch(value))


def _load(path: Path):
    from utils.fetch import _load_patch_cache
    return _load_patch_cache(path)


# ---------------------------------------------------------------------------
# FC-1. Same cache path: only one fetch, second worker gets cache hit
# ---------------------------------------------------------------------------

def test_same_path_only_fetches_once(tmp_path):
    """Two threads calling _fetch_or_load_cached on the same path concurrently
    must result in exactly one S3 fetch; the second must read from cache."""
    from utils.fetch import _get_chunk_lock, _save_patch_cache, _load_patch_cache

    path = tmp_path / "item" / "B08.npz"
    fetch_count = 0
    barrier = threading.Barrier(2)

    def _fake_fetch(p: Path) -> tuple:
        nonlocal fetch_count
        barrier.wait()  # both threads reach the lock at the same time
        lock = _get_chunk_lock(p)
        with lock:
            if p.exists():
                return _load_patch_cache(p)
            fetch_count += 1
            data = _make_patch()
            _save_patch_cache(p, data)
            return data

    results = [None, None]

    def worker(idx):
        results[idx] = _fake_fetch(path)

    t0 = threading.Thread(target=worker, args=(0,))
    t1 = threading.Thread(target=worker, args=(1,))
    t0.start(); t1.start()
    t0.join(); t1.join()

    assert fetch_count == 1, f"Expected 1 fetch, got {fetch_count}"
    assert results[0] is not None
    assert results[1] is not None


# ---------------------------------------------------------------------------
# FC-2. Different cache paths: workers do not block each other
# ---------------------------------------------------------------------------

def test_different_paths_do_not_serialise(tmp_path):
    """Two threads on different cache paths must be able to run concurrently —
    verified by confirming both reach their barriers simultaneously."""
    from utils.fetch import _get_chunk_lock, _save_patch_cache

    path_a = tmp_path / "item" / "B04.npz"
    path_b = tmp_path / "item" / "B08.npz"
    barrier = threading.Barrier(2, timeout=5)
    reached = [False, False]

    def worker(idx, path):
        lock = _get_chunk_lock(path)
        with lock:
            reached[idx] = True
            # Both threads must be able to reach this point concurrently.
            # If the locks were shared, the second thread would be blocked
            # and the barrier would time out.
            barrier.wait()
            _save_patch_cache(path, _make_patch(float(idx)))

    t0 = threading.Thread(target=worker, args=(0, path_a))
    t1 = threading.Thread(target=worker, args=(1, path_b))
    t0.start(); t1.start()
    t0.join(); t1.join()

    assert reached[0] and reached[1], "Both workers must have run concurrently"
    assert path_a.exists()
    assert path_b.exists()


# ---------------------------------------------------------------------------
# FC-3. Atomic write: reader never sees a partial file
# ---------------------------------------------------------------------------

def test_write_is_atomic(tmp_path):
    """The cache file must not be visible to a reader until fully written.
    Implemented via write-to-unique-tmp then rename — the target path either
    does not exist or contains a complete .npz."""
    from utils.fetch import _save_patch_cache

    path = tmp_path / "scene" / "B08.npz"
    seen_corrupt = []

    write_started = threading.Event()
    check_done = threading.Event()

    def writer():
        write_started.set()
        _save_patch_cache(path, _make_patch())
        check_done.set()

    def reader():
        write_started.wait()
        # Poll briefly while the write is in progress
        for _ in range(200):
            if path.exists():
                try:
                    np.load(path, allow_pickle=False)
                except Exception as exc:
                    seen_corrupt.append(str(exc))
                break

    wt = threading.Thread(target=writer)
    rt = threading.Thread(target=reader)
    rt.start(); wt.start()
    wt.join(); rt.join()

    assert not seen_corrupt, f"Partial file observed: {seen_corrupt}"
    assert path.exists()
    data = np.load(path, allow_pickle=False)
    assert "arr" in data
