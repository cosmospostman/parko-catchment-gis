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
                # Expand by 4 pixels on each side.  make_pixel_grid snaps to
                # the UTM grid, so points can extend up to ~3 pixels beyond
                # the bbox edge after reprojection (measured empirically across
                # all training regions).  1 extra pixel gives headroom for
                # floating-point rounding in CachedNpzChipStore._pixel_coords.
                _EXPAND = 4
                win = Window(
                    win.col_off - _EXPAND, win.row_off - _EXPAND,
                    win.width + 2 * _EXPAND, win.height + 2 * _EXPAND,
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
    try:
        from affine import Affine
        from pyproj import CRS
        z = np.load(path, allow_pickle=False)
        arr = z["arr"]
        transform = Affine(*z["transform_coeffs"].tolist())
        crs = CRS.from_wkt(z["crs_wkt"].tobytes().decode("utf-8"))
        return arr, transform, crs
    except Exception:
        return None


def _load_patch_cache_bbox(path: Path) -> list[float] | None:
    """Return the bbox_wgs84 stored in a cached .npz, or None if absent/corrupt."""
    try:
        z = np.load(path, allow_pickle=False)
        if "fetch_bbox" not in z:
            return None
        return z["fetch_bbox"].tolist()
    except Exception:
        return None


def _patch_covers_bbox(
    data: tuple[np.ndarray, object, object],
    bbox_wgs84: list[float],
    path: Path | None = None,
) -> bool:
    """Return True if the cached patch was fetched for bbox_wgs84.

    Primary check: compare the stored fetch_bbox against the requested bbox (exact
    match within 1e-6 degrees).  This correctly handles border-tile items whose
    raster doesn't extend to the full bbox — they are cached as partial patches but
    are still valid for this bbox; points outside the raster come back as NaN.

    Fallback for legacy patches that lack fetch_bbox: check whether all four WGS84
    bbox corners project inside the patch extent (±1 pixel slop).
    """
    if path is not None:
        stored_bbox = _load_patch_cache_bbox(path)
        if stored_bbox is not None:
            return all(abs(a - b) < 1e-6 for a, b in zip(stored_bbox, bbox_wgs84))

    # Legacy fallback: geometric coverage check (patches written before fetch_bbox was stored)
    from pyproj import Transformer, CRS
    arr, transform, crs = data
    h, w = arr.shape
    crs_obj = CRS.from_wkt(crs.to_wkt()) if hasattr(crs, "to_wkt") else crs
    t = Transformer.from_crs("EPSG:4326", crs_obj, always_xy=True)
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    xs, ys = t.transform(
        [lon_min, lon_max, lon_min, lon_max],
        [lat_min, lat_min, lat_max, lat_max],
    )
    a = transform
    cols_f = (np.array(xs) - float(a.c)) / float(a.a)
    rows_f = (np.array(ys) - float(a.f)) / float(a.e)
    # Use floor() to match CachedNpzChipStore._pixel_coords, which floors
    # continuous pixel coordinates before checking bounds.
    _SLOP = 1
    rows_i = np.floor(rows_f).astype(int)
    cols_i = np.floor(cols_f).astype(int)
    return bool(cols_i.min() >= -_SLOP and cols_i.max() < w + _SLOP and
                rows_i.min() >= -_SLOP and rows_i.max() < h + _SLOP)


def _save_patch_cache(
    path: Path,
    data: tuple[np.ndarray, object, object],
    fetch_bbox: list[float] | None = None,
) -> None:
    """Save a patch to a .npz file, creating parent dirs as needed."""
    arr, transform, crs = data
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = dict(
        arr=arr,
        transform_coeffs=np.array(list(transform)[:6], dtype=np.float64),
        crs_wkt=np.frombuffer(crs.to_wkt().encode("utf-8"), dtype=np.uint8),
    )
    if fetch_bbox is not None:
        kwargs["fetch_bbox"] = np.array(fetch_bbox, dtype=np.float64)
    np.savez(path, **kwargs)


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
                    covers = await loop.run_in_executor(
                        executor, _patch_covers_bbox, data, bbox_wgs84, path
                    )
                    if covers:
                        cached += 1
                        return data
                    logger.debug(
                        "Cached patch for %s/%s does not cover bbox — re-fetching",
                        item_id, band,
                    )
        async with sem:
            data = await loop.run_in_executor(executor, _read_bbox_patch, href, bbox_wgs84)
        if data is None:
            errors += 1
            return None
        if cache_dir is not None:
            await loop.run_in_executor(
                executor, _save_patch_cache,
                _cache_path(cache_dir, item_id, band), data, bbox_wgs84,
            )
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
    # When cache_dir is set every fetched patch is written to disk immediately
    # inside fetch_one_patch.  We skip accumulating into `result` in that case
    # so the patch arrays are GC'd as soon as the write completes, preventing
    # the full fetch from materialising all patches in RAM at once.
    _accumulate = cache_dir is None

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
            if _accumulate:
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
            if _accumulate:
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
