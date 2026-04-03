"""
Shared pipeline utilities for tiled analysis scripts.

Provides:
  setup_gdal_env()       — GDAL HTTP settings and AWS env vars
  setup_proj()           — PROJ_DATA bootstrap + pyproj set_data_dir()
  run_tiled_pipeline()   — Process-pool tiled pipeline for steps 01 and 03
  _pool_size()           — Recommended worker count for ProcessPoolExecutor
"""
import csv
import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level worker state, populated by the pool initializer.
_WORKER_STATE: dict = {}

_STAT_FIELDS = [
    "tile_idx", "fetch_s", "fetch_ok", "cached", "array_shape",
    "array_bytes", "chunk_size", "fetch_workers", "compute_workers",
    "compute_s", "compute_ok",
]


def _pool_size(n_points: int | None = None) -> int:
    """Return a sensible ProcessPoolExecutor worker count.

    Formula: min(cpu_count, n_points) capped at 8 to avoid process
    startup overhead dominating for small point sets.

    cpu_count() is None in some constrained environments; falls back to 4.
    n_points=None means "don't cap by point count".
    """
    cpus = multiprocessing.cpu_count() or 4
    workers = min(cpus, 8)
    if n_points is not None:
        workers = min(workers, n_points)
    return max(1, workers)


def setup_gdal_env() -> None:
    """Set GDAL HTTP env vars for robust COG fetching from public S3.

    Must be called before any ThreadPoolExecutor so worker threads inherit
    the environment. Uses setdefault so caller-supplied values are respected.
    """
    logging.getLogger("rasterio.session").setLevel(logging.WARNING)
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    gdal_env = {
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "GDAL_HTTP_RETRY_ON_HTTP_ERROR": "429,500,502,503,504",
        "GDAL_HTTP_PERSISTENT": "YES",
        "GDAL_HTTP_VERSION": "2",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "CPL_VSIL_CURL_CACHE_SIZE": "67108864",  # 64 MB connection cache
        "CPL_VSIL_CURL_CHUNK_SIZE": "10485760",  # 10 MB
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }
    for k, v in gdal_env.items():
        os.environ.setdefault(k, v)


def setup_proj() -> None:
    """Bootstrap PROJ_DATA so worker threads find proj.db.

    Prefers the rasterio-bundled proj_data (matches the libproj rasterio links
    against); falls back to pyproj's copy. Also calls pyproj set_data_dir()
    because pyproj caches the path at import time.
    """
    if "PROJ_DATA" not in os.environ:
        try:
            import rasterio as _rasterio
            proj_data = os.path.join(os.path.dirname(_rasterio.__file__), "proj_data")
            if os.path.isdir(proj_data):
                os.environ["PROJ_DATA"] = proj_data
        except Exception:
            pass
    if "PROJ_DATA" not in os.environ:
        try:
            from pyproj.datadir import get_data_dir
            os.environ["PROJ_DATA"] = get_data_dir()
        except Exception:
            pass
    try:
        from pyproj.datadir import set_data_dir
        set_data_dir(os.environ["PROJ_DATA"])
    except Exception:
        pass


def _worker_init():
    """No-op pool initializer — state is inherited via fork from _WORKER_STATE."""


def _populate_worker_state(fetch_fn, compute_fn, scratch_dir, fetch_workers, compute_workers):
    """Populate module-level worker state before forking so children inherit it."""
    _WORKER_STATE["fetch_fn"] = fetch_fn
    _WORKER_STATE["compute_fn"] = compute_fn
    _WORKER_STATE["scratch_dir"] = scratch_dir
    _WORKER_STATE["fetch_workers"] = fetch_workers
    _WORKER_STATE["compute_workers"] = compute_workers


def _run_tile(args):
    """Module-level worker: runs fetch + compute in a forked process. Returns (tile_idx, path, stat)."""
    tile_idx, tile_bbox = args
    fetch_fn     = _WORKER_STATE["fetch_fn"]
    compute_fn   = _WORKER_STATE["compute_fn"]
    scratch_dir  = _WORKER_STATE["scratch_dir"]
    fetch_workers  = _WORKER_STATE["fetch_workers"]
    compute_workers = _WORKER_STATE["compute_workers"]

    tile_path = scratch_dir / f"tile_{tile_idx:05d}.tif"
    stat = {
        "tile_idx": tile_idx,
        "fetch_s": 0.0,
        "fetch_ok": False,
        "cached": False,
        "array_shape": "",
        "array_bytes": 0,
        "chunk_size": os.environ.get("CPL_VSIL_CURL_CHUNK_SIZE", ""),
        "fetch_workers": fetch_workers,
        "compute_workers": compute_workers,
        "compute_s": 0.0,
        "compute_ok": False,
    }
    if tile_path.exists() and tile_path.stat().st_size > 0:
        stat["cached"] = True
        stat["fetch_ok"] = True
        stat["compute_ok"] = True
        return (tile_idx, tile_path, stat)

    t0 = time.monotonic()
    try:
        raw = fetch_fn(tile_idx, tile_bbox, tile_path)
        stat["fetch_s"] = round(time.monotonic() - t0, 3)
        if raw is None:
            return (tile_idx, None, stat)
        stat["fetch_ok"] = True
        try:
            import xarray as xr
            if isinstance(raw, xr.Dataset):
                dims = next(iter(raw.data_vars.values())).shape
                stat["array_shape"] = "x".join(str(d) for d in dims)
                stat["array_bytes"] = int(sum(v.nbytes for v in raw.data_vars.values()))
            else:
                stat["array_shape"] = "x".join(str(d) for d in raw.shape)
                stat["array_bytes"] = int(raw.nbytes)
        except Exception:
            pass
    except Exception as exc:
        stat["fetch_s"] = round(time.monotonic() - t0, 3)
        logger.warning("Fetch tile %d failed: %s", tile_idx, exc)
        return (tile_idx, None, stat)

    t1 = time.monotonic()
    try:
        path = compute_fn(tile_idx, raw, tile_path)
        stat["compute_s"] = round(time.monotonic() - t1, 3)
        stat["compute_ok"] = path is not None
        return (tile_idx, path, stat)
    except Exception as exc:
        stat["compute_s"] = round(time.monotonic() - t1, 3)
        logger.warning("Compute tile %d failed: %s", tile_idx, exc)
        return (tile_idx, None, stat)


def run_tiled_pipeline(
    *,
    tile_bboxes: List,
    scratch_dir: Path,
    fetch_fn: Callable,
    compute_fn: Callable,
    merge_fn: Callable,
    out_path: Path,
    nodata,
    crs: str,
    fetch_workers: int,
    compute_workers: int,
) -> None:
    """Process-pool tiled pipeline for raster processing.

    Each tile is processed in a forked worker process (fetch + compute),
    giving each worker its own GIL for true CPU parallelism.

    fetch_fn(tile_idx, tile_bbox, tile_path) -> xarray.DataArray or None
    compute_fn(tile_idx, raw, tile_path) -> Path or None
    merge_fn(valid_paths, out_path, nodata, crs)
    """
    n_tiles = len(tile_bboxes)
    results: List[Optional[Path]] = [None] * n_tiles
    results_lock = threading.Lock()

    # Stats writer thread
    stats_q: "multiprocessing.Queue" = multiprocessing.Queue()

    def _stats_writer():
        try:
            import config as _config
            csv_path = Path(_config.LOG_DIR) / f"tile_stats_{out_path.stem}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_STAT_FIELDS)
                writer.writeheader()
                while True:
                    row = stats_q.get()
                    if row is None:
                        break
                    writer.writerow(row)
                    f.flush()
            logger.info("Tile stats written: %s", csv_path)
        except Exception as exc:
            logger.warning("Could not write tile stats CSV: %s", exc)
            while True:
                if stats_q.get() is None:
                    break

    stats_thread = threading.Thread(target=_stats_writer, daemon=True)
    stats_thread.start()

    _populate_worker_state(fetch_fn, compute_fn, scratch_dir, fetch_workers, compute_workers)

    def _collect(tile_idx, path, stat):
        if stat.get("cached"):
            logger.info("Tile %d/%d skipped (cached)", tile_idx + 1, n_tiles)
        with results_lock:
            results[tile_idx] = path
        stats_q.put(stat)

    if fetch_workers <= 1:
        # Run tiles inline — no forking. Required for correctness when callables
        # are not picklable (e.g. during unit tests with mocked functions).
        for args in enumerate(tile_bboxes):
            tile_idx, path, stat = _run_tile(args)
            _collect(tile_idx, path, stat)
    else:
        mp_ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=fetch_workers,
            mp_context=mp_ctx,
            initializer=_worker_init,
        ) as pool:
            futures = {
                pool.submit(_run_tile, args): args[0]
                for args in enumerate(tile_bboxes)
            }
            for future in as_completed(futures):
                try:
                    tile_idx, path, stat = future.result()
                except Exception as exc:
                    tile_idx = futures[future]
                    logger.warning("Tile %d failed: %s", tile_idx, exc)
                    stats_q.put({
                        "tile_idx": tile_idx, "fetch_s": 0.0, "fetch_ok": False,
                        "cached": False, "array_shape": "", "array_bytes": 0,
                        "chunk_size": "", "fetch_workers": fetch_workers,
                        "compute_workers": compute_workers, "compute_s": 0.0,
                        "compute_ok": False,
                    })
                    continue
                _collect(tile_idx, path, stat)

    stats_q.put(None)
    stats_thread.join()

    valid_paths = [p for p in results if p is not None]
    if not valid_paths:
        raise RuntimeError("All tiles failed — no output produced")

    merge_fn(valid_paths, out_path, nodata=nodata, crs=crs)
    logger.info("Written: %s", out_path)

    for p in valid_paths:
        p.unlink(missing_ok=True)
    try:
        scratch_dir.rmdir()
    except OSError:
        pass
