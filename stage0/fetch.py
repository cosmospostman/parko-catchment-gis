"""Stage 0: async COG chip fetch.

Downloads one chip per (item, band, point) combination from the STAC archive
into inputs/{item_id}/{band}_{point_id}.tif. This is the most expensive
checkpoint to reconstruct — correctness and idempotency matter more than
speed tuning at this stage.

Concurrency model
-----------------
Network I/O is latency-bound, not bandwidth-bound. Each COG range-request
round-trip is ~50–150 ms regardless of connection speed. We amortise that
latency by running many requests concurrently via asyncio.

Because rasterio's windowed reads are blocking, they run in a
ThreadPoolExecutor managed by the event loop. The asyncio.Semaphore keeps
at most `max_concurrent` requests in-flight at the application level —
before the OS network stack or S3 sees load — preventing socket exhaustion
and rate-limit responses.

Default max_concurrent=32 is chosen for a 100 Mbps workstation connection.
Raise to 64–128 only on a dedicated EC2 instance with profiling.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import warnings

import numpy as np
import rasterio
import rasterio.windows
from pyproj import Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

from analysis.constants import SCL_BAND, SCL_CLEAR_VALUES
from analysis.timeseries.extraction import _scl_has_clear_pixels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _point_window(src, lon: float, lat: float, window_px: int) -> Window:
    """Return a rasterio Window centred on (lon, lat) with side window_px.

    Reprojects (lon, lat) from EPSG:4326 into the raster's CRS before
    applying the affine transform. Caller is responsible for clamping to
    raster bounds.
    """
    t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = t.transform(lon, lat)
    col, row = ~src.transform * (x, y)
    half = window_px // 2
    return Window(
        col_off=int(col) - half,
        row_off=int(row) - half,
        width=window_px,
        height=window_px,
    )


def _read_chip(href: str, lon: float, lat: float, window_px: int) -> np.ndarray | None:
    """Blocking: open a remote/local COG and read a windowed chip.

    Returns a 2-D float32 array of shape (window_px, window_px), or None if
    the window falls outside the raster bounds or the file cannot be read.
    """
    try:
        with rasterio.open(href) as src:
            win = _point_window(src, lon, lat, window_px)
            # Clamp window to raster bounds
            win = win.intersection(Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                return None
            arr = src.read(1, window=win, boundless=True,
                           fill_value=0, out_shape=(window_px, window_px))
            return arr.astype(np.float32)
    except Exception as exc:
        logger.debug("Could not read chip from %s: %s", href, exc)
        return None


def _read_chip_from_path(path: Path) -> np.ndarray:
    """Blocking: read band 1 from an existing chip file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            return src.read(1)


def _write_chip(path: Path, arr: np.ndarray) -> None:
    """Write a 2-D float32 array as a single-band GeoTIFF chip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path, "w",
            driver="GTiff",
            height=h, width=w,
            count=1, dtype="float32",
        ) as dst:
            dst.write(arr, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_chips(
    points: list[tuple[str, float, float]],
    items: list,
    bands: list[str],
    window_px: int = 5,
    inputs_dir: Path = Path("inputs/"),
    scl_filter: bool = True,
    max_concurrent: int = 32,
    band_alias: dict[str, str] | None = None,
) -> None:
    """Download COG chips for all (item, band, point) combinations.

    Applies only an SCL pre-filter to skip wholly-clouded acquisitions —
    no spectral computation, no quality scoring.

    Writes chips to::

        {inputs_dir}/{item_id}/{band}_{point_id}.tif

    This function is idempotent: existing chips are skipped without re-reading.

    Parameters
    ----------
    points:
        List of (point_id, lon, lat) tuples. lon/lat in EPSG:4326.
    items:
        pystac.Item objects from a STAC search.
    bands:
        Canonical band names to download (e.g. ["B03", "B04", "SCL"]).
        These are used as the filename stem — chip files are always written
        as ``{band}_{point_id}.tif`` using the canonical name.
    window_px:
        Chip side length in pixels. Use 1 for point extraction, larger for
        compositing.
    inputs_dir:
        Root directory for staged chips.
    scl_filter:
        If True, skip all band chips for acquisitions where the SCL chip
        contains no clear pixels (wholly clouded/shadowed scene).
    max_concurrent:
        Maximum number of simultaneous in-flight network requests.
        See module docstring for guidance on tuning.
    band_alias:
        Optional mapping from canonical band name to the asset key used by
        the STAC provider (e.g. ``{"B03": "green", "SCL": "scl"}``).
        When provided, asset lookup uses the alias key while the canonical
        name is still used for file paths and downstream code.
        Bands not in the mapping are looked up by their canonical name.
    """
    inputs_dir = Path(inputs_dir)
    sem = asyncio.Semaphore(max_concurrent)
    loop = asyncio.get_running_loop()
    _alias: dict[str, str] = band_alias or {}

    # Use a thread pool sized to max_concurrent — one thread per in-flight request
    executor = ThreadPoolExecutor(max_workers=max_concurrent)

    total = skipped = fetched = filtered = errors = 0

    async def fetch_one(item_id: str, band: str, point_id: str, href: str,
                        lon: float, lat: float) -> np.ndarray | None:
        nonlocal total, skipped, fetched, errors
        total += 1
        path = inputs_dir / item_id / f"{band}_{point_id}.tif"
        if path.exists():
            skipped += 1
            return None  # idempotent
        async with sem:
            arr = await loop.run_in_executor(
                executor, _read_chip, href, lon, lat, window_px
            )
        if arr is None:
            errors += 1
            return None
        await loop.run_in_executor(executor, _write_chip, path, arr)
        fetched += 1
        return arr

    async def process_point(
        item_id: str,
        point_id: str,
        lon: float,
        lat: float,
        has_scl: bool,
        scl_href: str | None,
        band_hrefs: list[tuple[str, str]],
    ) -> None:
        nonlocal filtered
        # SCL pre-filter: fetch or read existing chip, skip if wholly clouded
        if has_scl:
            scl_path = inputs_dir / item_id / f"{SCL_BAND}_{point_id}.tif"
            if scl_path.exists():
                scl_arr = await loop.run_in_executor(
                    executor, _read_chip_from_path, scl_path
                )
            else:
                scl_arr = await fetch_one(item_id, SCL_BAND, point_id,
                                          scl_href, lon, lat)
                if scl_arr is None and scl_path.exists():
                    scl_arr = await loop.run_in_executor(
                        executor, _read_chip_from_path, scl_path
                    )
            if scl_arr is not None and not _scl_has_clear_pixels(scl_arr):
                filtered += 1
                logger.debug("Skipping wholly-clouded acquisition %s at %s",
                             item_id, point_id)
                return

        # Fetch all bands for this point concurrently
        await asyncio.gather(*[
            fetch_one(item_id, band, point_id, href, lon, lat)
            for band, href in band_hrefs
        ])

    # Build all tasks across all items upfront so the semaphore can keep
    # max_concurrent requests in-flight across the entire job, not just
    # within a single item.
    all_tasks = []
    for item in items:
        item_id = item.id
        assets = item.assets

        # Pre-filter points to those inside this item's bbox (EPSG:4326).
        if item.bbox:
            minx, miny, maxx, maxy = item.bbox
            item_points = [
                (pid, lon, lat) for pid, lon, lat in points
                if minx <= lon <= maxx and miny <= lat <= maxy
            ]
        else:
            item_points = points

        if not item_points:
            continue

        scl_asset_key = _alias.get(SCL_BAND, SCL_BAND)
        has_scl = scl_filter and scl_asset_key in assets
        scl_href = assets[scl_asset_key].href if has_scl else None

        band_hrefs: list[tuple[str, str]] = []
        for band in bands:
            if band == SCL_BAND:
                continue
            asset_key = _alias.get(band, band)
            if asset_key not in assets:
                logger.debug("Band %s (asset key %s) not in item %s assets",
                             band, asset_key, item_id)
                continue
            band_hrefs.append((band, assets[asset_key].href))

        for pid, lon, lat in item_points:
            all_tasks.append(
                process_point(item_id, pid, lon, lat, has_scl, scl_href, band_hrefs)
            )

    logger.info("fetch_chips: dispatching %d point-item tasks across %d items",
                len(all_tasks), len(items))
    await asyncio.gather(*all_tasks)

    executor.shutdown(wait=False)
    logger.info(
        "fetch_chips complete: %d total, %d fetched, %d skipped (existing), "
        "%d filtered (clouded), %d errors",
        total, fetched, skipped, filtered, errors,
    )


# ---------------------------------------------------------------------------
# Patch-based fetch — efficient for dense point grids within a small bbox
# ---------------------------------------------------------------------------

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
    try:
        z = np.load(path, allow_pickle=True)
        arr = z["arr"]
        transform = z["transform"].item()   # stored as 0-d object array
        crs = z["crs"].item()
        return arr, transform, crs
    except Exception:
        return None


def _save_patch_cache(path: Path, data: tuple[np.ndarray, object, object]) -> None:
    """Save a patch to a .npz file, creating parent dirs as needed."""
    arr, transform, crs = data
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, arr=arr, transform=np.array(transform, dtype=object),
             crs=np.array(crs, dtype=object))


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

    This replaces fetch_chips + DiskChipStore for the dense-grid case. The
    result is a PatchData dict that can be passed directly to MemoryChipStore.

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
    # item_specs: list of (item, scl_href_or_None, [(band, href), ...])
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
    # SCL must resolve before we can decide which items pass the cloud filter,
    # but we fetch all of them at once rather than one-at-a-time.
    scl_hrefs = [scl_href for _, scl_href, _ in item_specs]
    async def _none() -> None:
        return None

    logger.info("fetch_patches: phase 1 — fetching %d SCL patches", len(item_specs))
    scl_results = await asyncio.gather(*[
        fetch_one_patch(item.id, SCL_BAND, href) if href else _none()
        for (item, href, _) in item_specs
    ])
    logger.info("fetch_patches: SCL done, applying cloud filter")

    # --- Phase 2: apply cloud filter, fetch all spectral bands at once ------
    spectral_tasks: list[tuple[str, str, asyncio.Task]] = []  # (item_id, band, task)
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
    log_every = max(1, n_spectral // 10)  # every 10%

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
