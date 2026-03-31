"""
Shared pipeline utilities for tiled analysis scripts.

Provides:
  setup_gdal_env()       — GDAL HTTP settings and AWS env vars
  setup_proj()           — PROJ_DATA bootstrap + pyproj set_data_dir()
  run_tiled_pipeline()   — Two-pool fetch/compute orchestration for steps 01 and 03
"""
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def setup_gdal_env() -> None:
    """Set GDAL HTTP env vars for robust COG fetching from public S3.

    Must be called before any ThreadPoolExecutor so worker threads inherit
    the environment. Uses setdefault so caller-supplied values are respected.
    """
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
    """Two-pool fetch/compute pipeline for tiled raster processing.

    fetch_fn(tile_idx, tile_bbox, tile_path) -> xarray.DataArray or None
        Called in the fetch pool. Should materialise the raw band array
        (e.g. via stackstac + .compute()). Returns None on failure.
        If tile_path already exists with non-zero size, returns the sentinel
        string "cached" to signal the compute worker to use the on-disk tile.

    compute_fn(tile_idx, raw, tile_path) -> Path or None
        Called in the compute pool. Receives the materialised DataArray from
        fetch_fn, applies index math, writes tile_path, returns it. Returns
        None on failure.

    merge_fn(valid_paths, out_path, nodata, crs)
        Called once after all tiles complete.

    Queue convention — items on q are 3-tuples:
        (tile_idx, raw_or_None, existing_path_or_None)
        - (idx, DataArray, None)  — fetched OK, needs compute
        - (idx, None, Path)       — cached on disk, skip compute
        - (idx, None, None)       — fetch failed
    """
    n_tiles = len(tile_bboxes)
    q: queue.Queue = queue.Queue(maxsize=compute_workers * 2)
    results: List[Optional[Path]] = [None] * n_tiles
    results_lock = threading.Lock()

    def _compute_worker():
        while True:
            item = q.get()
            if item is None:
                break
            tile_idx, raw, existing_path = item
            if existing_path is not None:
                with results_lock:
                    results[tile_idx] = existing_path
                continue
            if raw is None:
                continue  # fetch failed; results[tile_idx] stays None
            path = compute_fn(tile_idx, raw, scratch_dir / f"tile_{tile_idx:05d}.tif")
            with results_lock:
                results[tile_idx] = path

    compute_threads = [
        threading.Thread(target=_compute_worker, daemon=True)
        for _ in range(compute_workers)
    ]
    for t in compute_threads:
        t.start()

    def _fetch_tile(args):
        tile_idx, tile_bbox = args
        tile_path = scratch_dir / f"tile_{tile_idx:05d}.tif"
        if tile_path.exists() and tile_path.stat().st_size > 0:
            logger.info("Tile %d/%d skipped (cached)", tile_idx + 1, n_tiles)
            q.put((tile_idx, None, tile_path))
            return
        try:
            raw = fetch_fn(tile_idx, tile_bbox, tile_path)
            if raw is None:
                q.put((tile_idx, None, None))
            else:
                q.put((tile_idx, raw, None))
        except Exception as exc:
            logger.warning("Fetch tile %d failed: %s", tile_idx, exc)
            q.put((tile_idx, None, None))

    with ThreadPoolExecutor(max_workers=fetch_workers) as fetch_pool:
        futures = [fetch_pool.submit(_fetch_tile, args) for args in enumerate(tile_bboxes)]
        for f in futures:
            f.result()  # re-raise any uncaught exception

    for _ in range(compute_workers):
        q.put(None)
    for t in compute_threads:
        t.join()

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
