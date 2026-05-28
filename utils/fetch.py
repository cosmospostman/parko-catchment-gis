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
                try:
                    win = win.intersection(Window(0, 0, src.width, src.height))
                except Exception:
                    return None  # bbox doesn't overlap this item's raster extent
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
            msg = str(exc)
            # 409 Conflict is non-transient (GDAL/S3 multipart conflict or
            # expired SAS token that signing won't fix here); don't retry.
            if "409" in msg:
                logger.warning("fetch 409 (non-retryable): %s  url=%s", msg, href)
                return None
            if attempt < max_retries:
                wait = 2 ** attempt  # 1, 2, 4, 8 seconds
                logger.debug("Retry %d/%d for %s after error: %s (waiting %ds)",
                             attempt + 1, max_retries, href, exc, wait)
                time.sleep(wait)
            else:
                logger.warning("fetch failed after %d retries (%s): %s  url=%s",
                               max_retries, exc.__class__.__name__, exc, href)
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
    item_signer: object | None = None,
    sensor_label: str = "S2",
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
        Set to False for sensors without an SCL band (e.g. Sentinel-1).
    max_concurrent:
        Maximum simultaneous in-flight network requests.
    band_alias:
        Optional mapping from canonical band name to STAC asset key.
    cache_dir:
        Optional directory for caching fetched patches as .npz files.
        On re-runs, cached patches are loaded from disk instead of re-fetched.
        Layout: {cache_dir}/{item_id}/{band}.npz
    item_signer:
        Optional callable ``(item) -> item`` applied to each item before its
        asset hrefs are read.  Use this for collections that require per-item
        authentication (e.g. ``planetary_computer.sign`` for MPC assets).

    Returns
    -------
    PatchData
        Mapping (item_id, band) → (2D float32 array, Affine transform, CRS).
        Cloud-filtered items and missing bands are absent from the dict.
    """
    import os
    # Let Python-level retry logic (in _read_bbox_patch) own all retries.
    # GDAL's internal curl retry duplicates that and hides transient errors.
    # Hard-set (not setdefault) so setup_gdal_env()'s "5" can't override us.
    os.environ["GDAL_HTTP_MAX_RETRY"] = "0"
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    # Do not set CPL_VSIL_CURL_CACHE_SIZE — the header cache can replay stale
    # multipart upload IDs under high concurrency, producing spurious 409s.

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max_concurrent)
    executor = ThreadPoolExecutor(max_workers=max_concurrent)
    _alias: dict[str, str] = band_alias or {}
    result: PatchData = {}
    fetched = cached = filtered = errors = 0
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    def _get_href(item, asset_key: str) -> str:
        """Sign item at call time (not upfront) so SAS tokens are fresh."""
        if item_signer is not None:
            try:
                item = item_signer(item)
            except Exception:
                pass
        return item.assets[asset_key].href

    async def fetch_one_patch(item, band: str, asset_key: str) -> tuple[np.ndarray, object, object] | None:
        nonlocal fetched, cached, errors
        item_id = item.id
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
        # Sign outside the semaphore: signing may block on the MPC token API
        # and should not hold a fetch concurrency slot while waiting.
        href = await loop.run_in_executor(executor, _get_href, item, asset_key)
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

    # --- Pre-filter items by bbox, collect asset keys (no signing yet) -------
    item_specs = []
    for item in items:
        if item.bbox:
            ib = item.bbox
            if ib[0] > lon_max or ib[2] < lon_min or ib[1] > lat_max or ib[3] < lat_min:
                continue
        assets = item.assets
        scl_asset_key = _alias.get(SCL_BAND, SCL_BAND)
        has_scl = scl_filter and scl_asset_key in assets
        band_asset_keys = []
        for band in bands:
            if band == SCL_BAND:
                continue
            asset_key = _alias.get(band, band)
            if asset_key in assets:
                band_asset_keys.append((band, asset_key))
            else:
                logger.debug("Band %s (asset key %s) not in item %s assets",
                             band, asset_key, item.id)
        item_specs.append((item, scl_asset_key if has_scl else None, band_asset_keys))

    # --- Phase 1: fetch SCL patches (skipped when scl_filter=False) ---------
    async def _none() -> None:
        return None

    if scl_filter:
        n_scl = len(item_specs)
        logger.info("fetch_patches: phase 1 — fetching %d SCL patches", n_scl)
        scl_done = 0

        async def fetch_scl_tracked(item, scl_asset_key):
            nonlocal scl_done
            result = await (fetch_one_patch(item, SCL_BAND, scl_asset_key) if scl_asset_key else _none())
            scl_done += 1
            logger.info("  %s chips  fetch %d/%d patches done", sensor_label, scl_done, n_scl)
            return result

        scl_results = await asyncio.gather(*[
            fetch_scl_tracked(item, scl_asset_key)
            for (item, scl_asset_key, _) in item_specs
        ])
        logger.info("fetch_patches: SCL done, applying cloud filter")
    else:
        scl_results = [None] * len(item_specs)

    # --- Phase 2: apply cloud filter, fetch all spectral bands at once ------
    # When cache_dir is set every fetched patch is written to disk immediately
    # inside fetch_one_patch.  We skip accumulating into `result` in that case
    # so the patch arrays are GC'd as soon as the write completes, preventing
    # the full fetch from materialising all patches in RAM at once.
    _accumulate = cache_dir is None

    spectral_tasks: list[tuple[str, str, asyncio.Task]] = []
    for (item, scl_asset_key, band_asset_keys), scl_data in zip(item_specs, scl_results):
        item_id = item.id
        if scl_filter and scl_asset_key is not None:
            if scl_data is None:
                continue
            scl_arr, _, _ = scl_data
            if not _scl_has_clear_pixels(scl_arr):
                filtered += 1
                logger.debug("Skipping wholly-clouded item %s", item_id)
                continue
            if _accumulate:
                result[(item_id, SCL_BAND)] = scl_data

        for band, asset_key in band_asset_keys:
            spectral_tasks.append((item_id, band, asyncio.ensure_future(fetch_one_patch(item, band, asset_key))))

    n_spectral = len(spectral_tasks)
    logger.info("fetch_patches: phase 2 — fetching %d spectral patches (%d items cloud-filtered)",
                n_spectral, filtered)

    completed = 0

    async def tracked(item_id: str, band: str, task: asyncio.Task):
        nonlocal completed
        data = await task
        completed += 1
        if data is not None and _accumulate:
            result[(item_id, band)] = data
        logger.info("  %s chips  fetch %d/%d patches done", sensor_label, completed, n_spectral)

    await asyncio.gather(*[tracked(item_id, band, t) for item_id, band, t in spectral_tasks])

    executor.shutdown(wait=False)
    _log = logger.warning if errors > 0 and fetched == 0 and cached == 0 else logger.info
    _log(
        "fetch_patches complete: %d fetched, %d from cache, "
        "%d items cloud-filtered, %d errors",
        fetched, cached, filtered, errors,
    )
    return result


async def fetch_patches_to_tiff(
    items: list,
    bands: list[str],
    bbox_wgs84: list[float],
    out_dir: Path,
    max_concurrent: int = 128,
    band_alias: dict[str, str] | None = None,
    item_signer: object | None = None,
) -> list[Path]:
    """Fetch one bbox-covering patch per (item, band) and write each to a GeoTIFF on disk.

    Unlike fetch_patches(), no patch arrays are accumulated in memory — each array is
    written to {out_dir}/{item_id}/{band}.tif immediately and then dereferenced.  This
    keeps peak RAM at O(one patch) regardless of item count, suitable for 8 GB machines.

    Applies the same SCL cloud filter as fetch_patches(): wholly-clouded items are skipped.
    Existing non-empty .tif files are skipped (resume support within a strip restart).

    Returns a list of all written .tif paths.
    """
    import os
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max_concurrent)
    # Two executors: fetch threads saturate the network link; write threads handle
    # disk I/O without stealing slots from in-flight range requests.
    fetch_executor = ThreadPoolExecutor(max_workers=max_concurrent)
    write_executor = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 8))
    _alias: dict[str, str] = band_alias or {}
    written: list[Path] = []
    written_lock = asyncio.Lock()
    fetched = cached = filtered = errors = 0
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    def _get_href(item, asset_key: str) -> str:
        if item_signer is not None:
            try:
                item = item_signer(item)
            except Exception:
                pass
        return item.assets[asset_key].href

    def _write_tif(path: Path, arr: np.ndarray, transform, crs) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with rasterio.open(
            tmp, "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(arr, 1)
        tmp.replace(path)

    async def _fetch_and_write(item, band: str, asset_key: str) -> Path | None:
        nonlocal fetched, cached, errors
        item_id = item.id
        path = out_dir / item_id / f"{band}.tif"
        if path.exists() and path.stat().st_size > 0:
            cached += 1
            return path
        href = await loop.run_in_executor(fetch_executor, _get_href, item, asset_key)
        async with sem:
            data = await loop.run_in_executor(fetch_executor, _read_bbox_patch, href, bbox_wgs84)
        if data is None:
            errors += 1
            return None
        arr, transform, crs = data
        await loop.run_in_executor(write_executor, _write_tif, path, arr, transform, crs)
        fetched += 1
        return path

    # Build item specs (same pre-filter as fetch_patches)
    item_specs = []
    for item in items:
        if item.bbox:
            ib = item.bbox
            if ib[0] > lon_max or ib[2] < lon_min or ib[1] > lat_max or ib[3] < lat_min:
                continue
        assets = item.assets
        scl_asset_key = _alias.get(SCL_BAND, SCL_BAND)
        has_scl = scl_asset_key in assets
        band_asset_keys = []
        for band in bands:
            if band == SCL_BAND:
                continue
            asset_key = _alias.get(band, band)
            if asset_key in assets:
                band_asset_keys.append((band, asset_key))
        item_specs.append((item, scl_asset_key if has_scl else None, band_asset_keys))

    # Phase 1: SCL patches
    n_scl = len(item_specs)
    logger.info("fetch_patches_to_tiff: phase 1 — fetching %d SCL patches", n_scl)
    scl_done = 0

    async def fetch_scl_tracked(item, scl_asset_key):
        nonlocal scl_done
        if scl_asset_key is None:
            return None
        result = await _fetch_and_write(item, SCL_BAND, scl_asset_key)
        scl_done += 1
        logger.info("  S2 chips  fetch %d/%d SCL done", scl_done, n_scl)
        return result

    scl_paths = await asyncio.gather(*[
        fetch_scl_tracked(item, scl_asset_key)
        for (item, scl_asset_key, _) in item_specs
    ])

    # Phase 2: spectral patches for non-clouded items
    spectral_tasks: list[tuple[asyncio.Task]] = []
    for (item, scl_asset_key, band_asset_keys), scl_path in zip(item_specs, scl_paths):
        if scl_asset_key is not None:
            if scl_path is None:
                continue
            # Load SCL from the written tif to apply cloud filter
            try:
                with rasterio.open(scl_path) as src:
                    scl_arr = src.read(1)
            except Exception:
                continue
            if not _scl_has_clear_pixels(scl_arr):
                filtered += 1
                logger.debug("Skipping wholly-clouded item %s", item.id)
                continue
        for band, asset_key in band_asset_keys:
            spectral_tasks.append(asyncio.ensure_future(_fetch_and_write(item, band, asset_key)))

    n_spectral = len(spectral_tasks)
    logger.info(
        "fetch_patches_to_tiff: phase 2 — fetching %d spectral patches (%d items cloud-filtered)",
        n_spectral, filtered,
    )
    completed = 0

    async def tracked(task):
        nonlocal completed
        path = await task
        completed += 1
        logger.info("  S2 chips  fetch %d/%d spectral done", completed, n_spectral)
        if path is not None:
            async with written_lock:
                written.append(path)

    await asyncio.gather(*[tracked(t) for t in spectral_tasks])

    fetch_executor.shutdown(wait=False)
    write_executor.shutdown(wait=False)
    _log = logger.warning if errors > 0 and fetched == 0 and cached == 0 else logger.info
    _log(
        "fetch_patches_to_tiff complete: %d fetched, %d cached, "
        "%d items cloud-filtered, %d errors",
        fetched, cached, filtered, errors,
    )
    return written
