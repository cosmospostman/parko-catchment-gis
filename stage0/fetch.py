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
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.windows import Window

from analysis.constants import SCL_BAND, SCL_CLEAR_VALUES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _point_window(transform, lon: float, lat: float, window_px: int) -> Window:
    """Return a rasterio Window centred on (lon, lat) with side window_px.

    Uses the raster's affine transform to convert geographic coords to
    pixel coords. Caller is responsible for clamping to raster bounds.
    """
    col, row = ~transform * (lon, lat)
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
            win = _point_window(src.transform, lon, lat, window_px)
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


def _scl_has_clear_pixels(arr: np.ndarray) -> bool:
    """Return True if any pixel in the SCL chip is in SCL_CLEAR_VALUES."""
    return bool(np.any(np.isin(arr.astype(np.int32), list(SCL_CLEAR_VALUES))))


def _write_chip(path: Path, arr: np.ndarray) -> None:
    """Write a 2-D float32 array as a single-band GeoTIFF chip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
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
        Band asset keys to download (e.g. ["B03", "B04", "SCL"]).
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
    """
    inputs_dir = Path(inputs_dir)
    sem = asyncio.Semaphore(max_concurrent)
    loop = asyncio.get_running_loop()

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

    for item in items:
        item_id = item.id
        assets = item.assets

        # Ensure SCL is fetched before other bands when scl_filter is active
        band_order = bands[:]
        if scl_filter and SCL_BAND in assets and SCL_BAND not in band_order:
            band_order = [SCL_BAND] + band_order

        for point_id, lon, lat in points:
            # SCL pre-filter: fetch SCL chip first, skip acquisition if wholly clouded
            if scl_filter and SCL_BAND in assets:
                scl_href = assets[SCL_BAND].href
                scl_path = inputs_dir / item_id / f"{SCL_BAND}_{point_id}.tif"
                if scl_path.exists():
                    # Read existing SCL chip to check clarity
                    with rasterio.open(scl_path) as src:
                        scl_arr = src.read(1)
                else:
                    scl_arr = await fetch_one(item_id, SCL_BAND, point_id, scl_href, lon, lat)
                if scl_arr is None:
                    # Try reading from disk (may have been written in fetch_one)
                    scl_path = inputs_dir / item_id / f"{SCL_BAND}_{point_id}.tif"
                    if scl_path.exists():
                        with rasterio.open(scl_path) as src:
                            scl_arr = src.read(1)
                if scl_arr is not None and not _scl_has_clear_pixels(scl_arr):
                    filtered += 1
                    logger.debug(
                        "Skipping wholly-clouded acquisition %s at %s", item_id, point_id
                    )
                    continue

            # Fetch remaining bands concurrently for this (item, point)
            tasks = []
            for band in bands:
                if band == SCL_BAND:
                    continue  # already handled above
                if band not in assets:
                    logger.debug("Band %s not in item %s assets", band, item_id)
                    continue
                href = assets[band].href
                tasks.append(fetch_one(item_id, band, point_id, href, lon, lat))

            await asyncio.gather(*tasks)

    executor.shutdown(wait=False)
    logger.info(
        "fetch_chips complete: %d total, %d fetched, %d skipped (existing), "
        "%d filtered (clouded), %d errors",
        total, fetched, skipped, filtered, errors,
    )
