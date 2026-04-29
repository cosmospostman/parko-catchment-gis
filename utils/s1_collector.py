"""utils/s1_collector.py — Fetch Sentinel-1 GRD observations for a pixel grid.

Fetches S1 VH and VV backscatter (linear power) for a bbox and date window,
aligning observations to the same pixel grid used by pixel_collector.py.
Results are returned as a DataFrame with the same schema conventions as S2
rows, with `source="S1"` and S2-specific columns set to NaN.

Output rows (one per pixel per S1 acquisition):
    point_id  : str    — pixel identifier matching S2 grid ("px_0042_0031")
    lon       : float  — pixel centre longitude (EPSG:4326)
    lat       : float  — pixel centre latitude  (EPSG:4326)
    date      : date   — S1 acquisition date (UTC, date only)
    source    : str    — "S1"
    vh        : float  — VH backscatter (linear power, not dB)
    vv        : float  — VV backscatter (linear power, not dB)

All S2 band columns (B02…B12, scl_purity, scl, aot, etc.) are absent from S1
rows and will be NaN when the two DataFrames are concatenated.

The caller is responsible for sorting the combined DataFrame by (point_id, date)
after concatenation — sort_parquet_by_pixel handles this.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.stac import search_sentinel1, filter_items_by_bbox
from utils.pipeline import setup_gdal_env

logger = logging.getLogger(__name__)

_STAC_ENDPOINT  = "https://earth-search.aws.element84.com/v1"
_S1_COLLECTION  = "sentinel-1-grd"
_S1_BANDS       = ["vh", "vv"]

_DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "chips" / "s1"


def _reconstruct_affine(item):
    """Return rasterio Affine from a STAC item's proj:transform property, or None."""
    from rasterio.transform import Affine
    pt = item.properties.get("proj:transform")
    if pt is None or len(pt) < 6:
        return None
    # proj:transform: [xres, xskew, x_origin, yskew, yres, y_origin, ...]
    return Affine(pt[0], pt[1], pt[2], pt[3], pt[4], pt[5])


def _chip_cache_path(cache_dir: Path, item_id: str, band: str, bbox: list[float] | None = None) -> Path:
    """Return cache path for a (item, band, bbox) triple.

    bbox is included in the key because the windowed array and its win_affine
    are specific to the requested bbox — the same S1 item fetched for different
    bboxes produces different arrays.
    """
    if bbox is not None:
        bbox_key = "_".join(f"{v:.6f}" for v in bbox)
        return cache_dir / item_id / f"{band}_{bbox_key}.npz"
    return cache_dir / item_id / f"{band}.npz"


def _load_chip(path: Path) -> tuple[np.ndarray, object] | None:
    """Load a cached S1 band patch. Returns (arr, win_affine) or None."""
    try:
        from affine import Affine
        z = np.load(path, allow_pickle=False)
        arr = z["arr"]
        win_affine = Affine(*z["transform_coeffs"].tolist())
        return arr, win_affine
    except Exception:
        return None


def _save_chip(path: Path, arr: np.ndarray, win_affine) -> None:
    """Save a S1 band patch to .npz — same format as S2 chips (no CRS needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        arr=arr,
        transform_coeffs=np.array(list(win_affine)[:6], dtype=np.float64),
    )


def _pixel_window(affine, bbox_wgs84: list[float], src_width: int, src_height: int):
    """Return a rasterio Window for bbox_wgs84 clipped to raster bounds, or None.

    Returns None when the bbox falls outside the raster extent — this happens
    when filter_items_by_bbox accepts an item whose envelope overlaps but whose
    actual raster data does not reach the target area.
    """
    import rasterio.windows
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    win = rasterio.windows.from_bounds(
        lon_min, lat_min, lon_max, lat_max, transform=affine
    )
    try:
        return win.intersection(
            rasterio.windows.Window(0, 0, src_width, src_height)
        )
    except Exception:
        return None


def _read_band_array(
    href: str,
    affine,
    bbox_wgs84: list[float],
    cache_path: Path | None = None,
) -> tuple[np.ndarray, object] | None:
    """Read a single S1 band COG, returning (2D float32 array, win_affine) or None.

    Loads from cache_path if it exists; writes to cache_path after a successful read.
    """
    import rasterio
    import rasterio.windows

    if cache_path is not None:
        cached = _load_chip(cache_path)
        if cached is not None:
            return cached

    try:
        with rasterio.open(href) as src:
            win = _pixel_window(affine, bbox_wgs84, src.width, src.height)
            if win is None or win.width <= 0 or win.height <= 0:
                return None
            arr = src.read(1, window=win).astype(np.float32)
            arr[arr == 0] = np.nan
            win_affine = rasterio.windows.transform(win, affine)
    except Exception as exc:
        logger.debug("S1 band read failed (%s): %s", href, exc)
        return None

    if cache_path is not None:
        try:
            _save_chip(cache_path, arr, win_affine)
        except Exception as exc:
            logger.warning("S1 chip cache write failed (%s): %s", cache_path, exc)

    return arr, win_affine


def _extract_item(
    item,
    affine,
    bbox_wgs84: list[float],
    points: list[tuple[str, float, float]],
    cache_dir: Path | None = None,
) -> list[dict] | None:
    """Extract per-pixel VH/VV for one S1 item.

    Returns a list of row dicts (one per pixel with valid data), or None if
    the item has no overlap with bbox_wgs84.
    """
    date = pd.to_datetime(item.datetime).date()

    vh_arr = vv_arr = None
    win_affine = None

    for band in _S1_BANDS:
        if band not in item.assets:
            continue
        href = item.assets[band].href
        cache_path = _chip_cache_path(cache_dir, item.id, band, bbox_wgs84) if cache_dir else None
        result = _read_band_array(href, affine, bbox_wgs84, cache_path)
        if result is None:
            continue
        arr, band_win_affine = result
        if band == "vh":
            vh_arr = arr
            win_affine = band_win_affine
        else:
            vv_arr = arr
            if win_affine is None:
                win_affine = band_win_affine

    if vh_arr is None and vv_arr is None:
        return None

    ref_arr = vh_arr if vh_arr is not None else vv_arr
    nrows, ncols = ref_arr.shape

    rows_out = []
    for pid, lon, lat in points:
        col_f = (lon - win_affine.c) / win_affine.a
        row_f = (lat - win_affine.f) / win_affine.e
        c = int(np.floor(col_f))
        r = int(np.floor(row_f))
        if not (0 <= r < nrows and 0 <= c < ncols):
            continue

        vh_val = float(vh_arr[r, c]) if vh_arr is not None else np.nan
        vv_val = float(vv_arr[r, c]) if vv_arr is not None else np.nan

        if np.isnan(vh_val) and np.isnan(vv_val):
            continue

        rows_out.append({
            "point_id": pid,
            "lon":      lon,
            "lat":      lat,
            "date":     date,
            "source":   "S1",
            "vh":       vh_val,
            "vv":       vv_val,
        })

    return rows_out if rows_out else None


def collect_s1(
    bbox_wgs84: list[float],
    start: str,
    end: str,
    points: list[tuple[str, float, float]],
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch S1 VH/VV observations for a pixel grid and return as a DataFrame.

    Parameters
    ----------
    bbox_wgs84:
        [lon_min, lat_min, lon_max, lat_max] — same bbox used for S2 collect.
    start, end:
        ISO date strings ("YYYY-MM-DD") — same window used for S2 collect.
    points:
        List of (point_id, lon, lat) — same grid produced by make_pixel_grid().
    cache_dir:
        Directory for per-(item, band) .npz patch caches. Mirrors S2 chip cache
        layout: {cache_dir}/{item_id}/{band}.npz. Defaults to data/chips/s1/.

    Returns
    -------
    DataFrame with columns [point_id, lon, lat, date, source, vh, vv].
    Empty DataFrame if no S1 data is found.
    """
    setup_gdal_env()

    resolved_cache = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR

    items = search_sentinel1(
        bbox=bbox_wgs84,
        start=start,
        end=end,
        endpoint=_STAC_ENDPOINT,
        collection=_S1_COLLECTION,
    )
    items = filter_items_by_bbox(items, bbox_wgs84)
    if not items:
        logger.info("No S1 items found for bbox %s in %s/%s", bbox_wgs84, start, end)
        return pd.DataFrame()

    cached_count = sum(
        1 for item in items
        if all(_chip_cache_path(resolved_cache, item.id, b, bbox_wgs84).exists() for b in _S1_BANDS
               if b in item.assets)
    )
    logger.info(
        "S1: %d items (%d fully cached) for bbox %s",
        len(items), cached_count, bbox_wgs84,
    )

    all_rows: list[dict] = []
    for item in items:
        affine = _reconstruct_affine(item)
        if affine is None:
            logger.warning("S1 item %s has no proj:transform — skipping", item.id)
            continue
        rows = _extract_item(item, affine, bbox_wgs84, points, cache_dir=resolved_cache)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        logger.info("S1: no usable observations extracted")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("S1: %d observations for %d unique dates", len(df), df["date"].nunique())
    return df
