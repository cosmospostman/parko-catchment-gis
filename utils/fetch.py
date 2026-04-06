"""utils/fetch.py — patch-based Sentinel-2 fetch.

Fetches one bbox-covering patch per (item, band) via COG range requests.
Designed for dense point grids within a small bbox where all points land
in the same COG tile — one request per (item, band) covers all points.

The result is a PatchData dict that can be passed directly to MemoryChipStore.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from pyproj import Transformer
from rasterio.windows import Window

from analysis.constants import SCL_BAND, SCL_CLEAR_VALUES
from analysis.timeseries.extraction import _scl_has_clear_pixels

logger = logging.getLogger(__name__)

# Type alias: (item_id, band) → (2D float32 array, windowed Affine, rasterio CRS)
PatchData = dict[tuple[str, str], tuple[np.ndarray, object, object]]


def _read_bbox_patch(
    href: str,
    bbox_wgs84: list[float],
    max_retries: int = 4,
) -> tuple[np.ndarray, object, object] | None:
    """Blocking: open a COG via range requests and read a patch covering bbox_wgs84.

    Returns (2D float32 array, windowed Affine transform, rasterio CRS), or
    None if the window falls outside the raster or all retries are exhausted.
    """
    import time
    for attempt in range(max_retries + 1):
        try:
            with rasterio.open(href) as src:
                t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                lon_min, lat_min, lon_max, lat_max = bbox_wgs84
                xs, ys = t.transform([lon_min, lon_max], [lat_min, lat_max])
                win = rasterio.windows.from_bounds(
                    min(xs), min(ys), max(xs), max(ys),
                    transform=src.transform,
                )
                win = win.intersection(Window(0, 0, src.width, src.height))
                if win.width <= 0 or win.height <= 0:
                    return None
                # Ensure at least 1×1 pixel — low-res bands (e.g. AOT at 60m)
                # may produce sub-pixel float windows over a small bbox.
                win = Window(
                    win.col_off, win.row_off,
                    max(1, win.width), max(1, win.height),
                )
                arr = src.read(1, window=win).astype(np.float32)
                patch_transform = rasterio.windows.transform(win, src.transform)
                return arr, patch_transform, src.crs
        except Exception as exc:
            if attempt < max_retries:
                wait = 2 ** attempt  # 1, 2, 4, 8 seconds
                logger.debug("Retry %d/%d for %s after error: %s (waiting %ds)",
                             attempt + 1, max_retries, href, exc, wait)
                time.sleep(wait)
            else:
                logger.debug("Giving up on %s after %d retries: %s", href, max_retries, exc)
                return None


def _cache_path(cache_dir: Path, item_id: str, band: str) -> Path:
    return cache_dir / item_id / f"{band}.npz"


def _load_patch_cache(path: Path) -> tuple[np.ndarray, object, object] | None:
    """Load a cached patch from a .npz file. Returns None if missing or corrupt."""
    import pickle
    try:
        z = np.load(path, allow_pickle=True)
        arr = z["arr"]
        transform = pickle.loads(z["transform"].tobytes())
        crs = pickle.loads(z["crs"].tobytes())
        return arr, transform, crs
    except Exception:
        return None


def _save_patch_cache(path: Path, data: tuple[np.ndarray, object, object]) -> None:
    """Save a patch to a .npz file, creating parent dirs as needed."""
    import pickle
    arr, transform, crs = data
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        arr=arr,
        transform=np.frombuffer(pickle.dumps(transform), dtype=np.uint8),
        crs=np.frombuffer(pickle.dumps(crs), dtype=np.uint8),
    )


async def fetch_patches(
    points: list[tuple[str, float, float]],
    items: list,
    bands: list[str],
    bbox_wgs84: list[float],
    scl_filter: bool = True,
    max_concurrent: int = 32,
    band_alias: dict[str, str] | None = None,
    cache_dir: Path | None = None,
) -> PatchData:
    """Fetch one bbox-covering patch per (item, band) instead of per-point chips.

    For point grids where all points fall within a small bbox, all points land
    in the same COG tile. This function issues one range request per (item, band)
    covering the entire bbox, then downstream code slices pixel values from the
    patch in memory via MemoryChipStore.

    Parameters
    ----------
    points:
        List of (point_id, lon, lat). Used only for the item bbox pre-filter.
    items:
        pystac.Item objects from a STAC search.
    bands:
        Canonical band names to fetch (e.g. ["B03", "B04", "SCL"]).
    bbox_wgs84:
        Bounding box [lon_min, lat_min, lon_max, lat_max] in EPSG:4326.
    scl_filter:
        If True, skip all band patches for items where the SCL patch contains
        no clear pixels (wholly-clouded acquisition over the bbox).
    max_concurrent:
        Maximum simultaneous in-flight network requests.
    band_alias:
        Optional mapping from canonical band name to STAC asset key.
    cache_dir:
        Optional directory for caching fetched patches as .npz files.
        On re-runs, cached patches are loaded from disk instead of re-fetched.
        Layout: {cache_dir}/{item_id}/{band}.npz

    Returns
    -------
    PatchData
        Mapping (item_id, band) → (2D float32 array, Affine transform, CRS).
        Cloud-filtered items and missing bands are absent from the dict.
    """
    import os
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "128000000")  # 128MB header cache

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max_concurrent)
    executor = ThreadPoolExecutor(max_workers=max_concurrent)
    _alias: dict[str, str] = band_alias or {}
    result: PatchData = {}
    fetched = cached = filtered = errors = 0
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    async def fetch_one_patch(item_id: str, band: str, href: str) -> tuple[np.ndarray, object, object] | None:
        nonlocal fetched, cached, errors
        if cache_dir is not None:
            path = _cache_path(cache_dir, item_id, band)
            if path.exists():
                data = await loop.run_in_executor(executor, _load_patch_cache, path)
                if data is not None:
                    cached += 1
                    return data
        async with sem:
            data = await loop.run_in_executor(executor, _read_bbox_patch, href, bbox_wgs84)
        if data is None:
            errors += 1
            return None
        if cache_dir is not None:
            await loop.run_in_executor(executor, _save_patch_cache,
                                       _cache_path(cache_dir, item_id, band), data)
        fetched += 1
        return data

    # --- Pre-filter items by bbox, collect hrefs ----------------------------
    item_specs = []
    for item in items:
        if item.bbox:
            ib = item.bbox
            if ib[0] > lon_max or ib[2] < lon_min or ib[1] > lat_max or ib[3] < lat_min:
                continue
        assets = item.assets
        scl_asset_key = _alias.get(SCL_BAND, SCL_BAND)
        scl_href = assets[scl_asset_key].href if (scl_filter and scl_asset_key in assets) else None
        band_hrefs = []
        for band in bands:
            if band == SCL_BAND:
                continue
            asset_key = _alias.get(band, band)
            if asset_key in assets:
                band_hrefs.append((band, assets[asset_key].href))
            else:
                logger.debug("Band %s (asset key %s) not in item %s assets",
                             band, asset_key, item.id)
        item_specs.append((item, scl_href, band_hrefs))

    # --- Phase 1: fetch all SCL patches concurrently ------------------------
    async def _none() -> None:
        return None

    logger.info("fetch_patches: phase 1 — fetching %d SCL patches", len(item_specs))
    scl_results = await asyncio.gather(*[
        fetch_one_patch(item.id, SCL_BAND, href) if href else _none()
        for (item, href, _) in item_specs
    ])
    logger.info("fetch_patches: SCL done, applying cloud filter")

    # --- Phase 2: apply cloud filter, fetch all spectral bands at once ------
    spectral_tasks: list[tuple[str, str, asyncio.Task]] = []
    for (item, scl_href, band_hrefs), scl_data in zip(item_specs, scl_results):
        item_id = item.id
        if scl_href is not None:
            if scl_data is None:
                continue
            scl_arr, _, _ = scl_data
            if not _scl_has_clear_pixels(scl_arr):
                filtered += 1
                logger.debug("Skipping wholly-clouded item %s", item_id)
                continue
            result[(item_id, SCL_BAND)] = scl_data

        for band, href in band_hrefs:
            spectral_tasks.append((item_id, band, asyncio.ensure_future(fetch_one_patch(item_id, band, href))))

    n_spectral = len(spectral_tasks)
    logger.info("fetch_patches: phase 2 — fetching %d spectral patches (%d items cloud-filtered)",
                n_spectral, filtered)

    completed = 0
    log_every = max(1, n_spectral // 10)

    async def tracked(item_id: str, band: str, task: asyncio.Task):
        nonlocal completed, fetched
        data = await task
        completed += 1
        if data is not None:
            result[(item_id, band)] = data
            fetched += 1
        if completed % log_every == 0 or completed == n_spectral:
            logger.info("fetch_patches: %d/%d spectral patches done", completed, n_spectral)

    await asyncio.gather(*[tracked(item_id, band, t) for item_id, band, t in spectral_tasks])

    executor.shutdown(wait=False)
    logger.info(
        "fetch_patches complete: %d fetched, %d from cache, "
        "%d items cloud-filtered, %d errors",
        fetched, cached, filtered, errors,
    )
    return result
