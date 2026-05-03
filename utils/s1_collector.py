"""utils/s1_collector.py — Fetch Sentinel-1 RTC observations for a pixel grid.

Fetches S1 VH and VV backscatter (linear power) from the Microsoft Planetary
Computer sentinel-1-rtc collection (gamma0 radiometrically terrain corrected).
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
    orbit     : str    — "ascending" or "descending" (sat:orbit_state from STAC item)

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

_STAC_ENDPOINT  = "https://planetarycomputer.microsoft.com/api/stac/v1"
_S1_COLLECTION  = "sentinel-1-rtc"
_S1_BANDS       = ["vh", "vv"]
_S1_NODATA      = -32768   # RTC nodata value (GRD used 0)

_DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "chips" / "s1"

_transformer_cache: dict[str, object] = {}


def _get_transformer(src_crs: str, dst_crs: str):
    """Return a cached pyproj Transformer for the given CRS pair."""
    from pyproj import Transformer
    key = (src_crs, dst_crs)
    if key not in _transformer_cache:
        _transformer_cache[key] = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return _transformer_cache[key]


def _reconstruct_affine(item):
    """Return rasterio Affine from a STAC item's proj:transform property, or None."""
    from rasterio.transform import Affine
    pt = item.properties.get("proj:transform")
    if pt is None or len(pt) < 6:
        return None
    # proj:transform: [xres, xskew, x_origin, yskew, yres, y_origin, ...]
    return Affine(pt[0], pt[1], pt[2], pt[3], pt[4], pt[5])


def _item_crs(item) -> str | None:
    """Return the CRS of a STAC item as an EPSG string, or None."""
    code = item.properties.get("proj:code") or item.properties.get("proj:epsg")
    if code:
        return str(code) if str(code).startswith("EPSG:") else f"EPSG:{code}"
    return None


def _reproject_bbox(bbox_wgs84: list[float], target_crs: str) -> list[float]:
    """Reproject [lon_min, lat_min, lon_max, lat_max] from WGS84 to target_crs."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    x_min, y_min = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
    x_max, y_max = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])
    return [min(x_min, x_max), min(y_min, y_max),
            max(x_min, x_max), max(y_min, y_max)]


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


def _pixel_window(affine, bbox_native: list[float], src_width: int, src_height: int):
    """Return a rasterio Window for bbox_native clipped to raster bounds, or None.

    bbox_native must be in the same CRS as the raster (may be projected).
    Returns None when the bbox falls outside the raster extent.
    """
    import rasterio.windows
    x_min, y_min, x_max, y_max = bbox_native
    win = rasterio.windows.from_bounds(
        x_min, y_min, x_max, y_max, transform=affine
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
    bbox_native: list[float],
    cache_path: Path | None = None,
    nodata: float = _S1_NODATA,
) -> tuple[np.ndarray, object] | None:
    """Read a single S1 band COG, returning (2D float32 array, win_affine) or None.

    bbox_native is in the raster's native CRS (projected for RTC items).
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
            win = _pixel_window(affine, bbox_native, src.width, src.height)
            if win is None or win.width <= 0 or win.height <= 0:
                return None
            arr = src.read(1, window=win).astype(np.float32)
            arr[arr == nodata] = np.nan
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
) -> tuple[list, list, list, list, list, list, list] | None:
    """Extract per-pixel VH/VV for one S1 item.

    Returns a 7-tuple of per-column lists
    (point_ids, lons, lats, dates, vh_vals, vv_vals, orbits)
    for all pixels with at least one valid observation, or None if the item
    has no overlap with bbox_wgs84.
    """
    date = pd.to_datetime(item.datetime).date()
    orbit_state = item.properties.get("sat:orbit_state", None)

    _MARGIN = 0.01
    bbox_fetch = [
        bbox_wgs84[0] - _MARGIN, bbox_wgs84[1] - _MARGIN,
        bbox_wgs84[2] + _MARGIN, bbox_wgs84[3] + _MARGIN,
    ]

    crs = _item_crs(item)
    if crs and crs != "EPSG:4326":
        bbox_native = _reproject_bbox(bbox_fetch, crs)
        pt_transformer = _get_transformer("EPSG:4326", crs)
    else:
        bbox_native = bbox_fetch
        pt_transformer = None

    vh_arr = vv_arr = None
    win_affine = None

    for band in _S1_BANDS:
        if band not in item.assets:
            continue
        href = item.assets[band].href
        cache_path = _chip_cache_path(cache_dir, item.id, band, bbox_wgs84) if cache_dir else None
        result = _read_band_array(href, affine, bbox_native, cache_path)
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

    pids_arr = [p[0] for p in points]
    lons_arr = np.array([p[1] for p in points], dtype=np.float64)
    lats_arr = np.array([p[2] for p in points], dtype=np.float64)

    if pt_transformer is not None:
        px, py = pt_transformer.transform(lons_arr, lats_arr)
    else:
        px, py = lons_arr, lats_arr

    cols = np.floor((px - win_affine.c) / win_affine.a).astype(np.int32)
    rows = np.floor((py - win_affine.f) / win_affine.e).astype(np.int32)

    in_bounds = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    idx = np.where(in_bounds)[0]
    if idx.size == 0:
        return None

    r_idx = rows[idx]
    c_idx = cols[idx]

    vh_vals = vh_arr[r_idx, c_idx] if vh_arr is not None else np.full(idx.size, np.nan, np.float32)
    vv_vals = vv_arr[r_idx, c_idx] if vv_arr is not None else np.full(idx.size, np.nan, np.float32)

    keep = ~(np.isnan(vh_vals) & np.isnan(vv_vals))
    idx = idx[keep]
    vh_vals = vh_vals[keep]
    vv_vals = vv_vals[keep]

    if idx.size == 0:
        return None

    n = idx.size
    return (
        [pids_arr[i] for i in idx],
        lons_arr[idx].tolist(),
        lats_arr[idx].tolist(),
        [date] * n,
        vh_vals.tolist(),
        vv_vals.tolist(),
        [orbit_state] * n,
    )


def collect_s1(
    bbox_wgs84: list[float],
    start: str,
    end: str,
    points: list[tuple[str, float, float]],
    cache_dir: Path | None = None,
    point_shard_size: int = 50_000,
    n_workers: int = 4,
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
    point_shard_size:
        Max points per shard. Each shard's rows are spilled to a temp parquet
        before the next shard is processed, bounding peak memory use. Defaults
        to 50 000 — at ~100 S1 items that's ~5 M rows peak per shard (~400 MB).
    n_workers:
        Number of parallel threads for item fetches. Each worker holds at most
        two band arrays (~15 MB each for a typical bbox) in memory, so peak
        array memory is roughly n_workers × 30 MB. Defaults to 4.

    Returns
    -------
    DataFrame with columns [point_id, lon, lat, date, source, vh, vv, orbit].
    Empty DataFrame if no S1 data is found.
    """
    import hashlib
    import pickle
    import tempfile
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pyarrow as pa
    import pyarrow.parquet as pq

    setup_gdal_env()

    resolved_cache = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR

    # Cache S1 STAC search results to avoid repeated round-trips on re-runs.
    stac_key = hashlib.md5(
        f"{bbox_wgs84}|{start}|{end}".encode()
    ).hexdigest()
    stac_cache_dir = resolved_cache / "stac"
    stac_cache_dir.mkdir(parents=True, exist_ok=True)
    stac_cache = stac_cache_dir / f"{stac_key}.pkl"

    items = None
    if stac_cache.exists():
        try:
            with stac_cache.open("rb") as fh:
                items = pickle.load(fh)
            logger.info("S1 STAC: %d items loaded from cache (%s)", len(items), stac_cache.name)
        except Exception:
            logger.warning("S1 STAC cache corrupt — re-fetching (%s)", stac_cache.name)
            stac_cache.unlink(missing_ok=True)

    if items is None:
        try:
            import planetary_computer as _pc
            _modifier = _pc.sign_inplace
        except ImportError:
            _modifier = None

        logger.info(
            "S1 STAC search: endpoint=%s collection=%s bbox=%s start=%s end=%s",
            _STAC_ENDPOINT, _S1_COLLECTION, bbox_wgs84, start, end,
        )
        items = search_sentinel1(
            bbox=bbox_wgs84,
            start=start,
            end=end,
            endpoint=_STAC_ENDPOINT,
            collection=_S1_COLLECTION,
            modifier=_modifier,
        )
        try:
            with stac_cache.open("wb") as fh:
                pickle.dump(items, fh)
            logger.info("S1 STAC: %d items found and cached (%s)", len(items), stac_cache.name)
        except Exception:
            logger.debug("S1 STAC: items not picklable — skipping cache write")

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

    # Import signer for per-item token refresh (tokens expire during long fetches)
    try:
        import planetary_computer as pc
        _sign = pc.sign
    except ImportError:
        _sign = None

    _ARROW_SCHEMA = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("lon",      pa.float64()),
        pa.field("lat",      pa.float64()),
        pa.field("date",     pa.timestamp("ms")),
        pa.field("source",   pa.string()),
        pa.field("vh",       pa.float32()),
        pa.field("vv",       pa.float32()),
        pa.field("orbit",    pa.string()),
    ])
    _FLUSH_ROWS = 500_000

    # Shard points to bound peak memory: process point_shard_size points at a
    # time across all items, spilling each shard to a temp parquet before moving
    # on. This mirrors S2's row-budget sharding strategy.
    n_shards = max(1, (len(points) + point_shard_size - 1) // point_shard_size)
    use_sharding = n_shards > 1

    with tempfile.TemporaryDirectory(prefix="s1_collect_") as _tmp:
        tmp_dir = Path(_tmp)
        shard_paths: list[Path] = []

        for shard_idx in range(n_shards):
            shard_points = points[shard_idx * point_shard_size : (shard_idx + 1) * point_shard_size]
            if use_sharding:
                logger.info(
                    "S1: shard %d/%d — %d points",
                    shard_idx + 1, n_shards, len(shard_points),
                )

            shard_path = tmp_dir / f"shard_{shard_idx:04d}.parquet"
            writer = pq.ParquetWriter(shard_path, _ARROW_SCHEMA)
            total_rows = 0
            n_items_done = 0
            log_interval = max(1, len(items) // 10)

            write_lock = threading.Lock()
            # Per-column lists: far lower Python object overhead than list-of-dicts.
            buf_pid:    list = []
            buf_lon:    list = []
            buf_lat:    list = []
            buf_date:   list = []
            buf_vh:     list = []
            buf_vv:     list = []
            buf_orbit:  list = []

            def _flush_locked() -> None:
                if not buf_pid:
                    return
                tbl = pa.table({
                    "point_id": pa.array(buf_pid,  pa.string()),
                    "lon":      pa.array(buf_lon,  pa.float64()),
                    "lat":      pa.array(buf_lat,  pa.float64()),
                    "date":     pa.array([pd.Timestamp(d) for d in buf_date], pa.timestamp("ms")),
                    "source":   pa.array(["S1"] * len(buf_pid), pa.string()),
                    "vh":       pa.array(buf_vh,   pa.float32()),
                    "vv":       pa.array(buf_vv,   pa.float32()),
                    "orbit":    pa.array(buf_orbit, pa.string()),
                })
                writer.write_table(tbl)
                buf_pid.clear(); buf_lon.clear(); buf_lat.clear(); buf_date.clear()
                buf_vh.clear();  buf_vv.clear();  buf_orbit.clear()

            def _fetch_item(item) -> int:
                """Fetch one item and append columns to buf. Returns row count."""
                if _sign is not None:
                    try:
                        item = _sign(item)
                    except TypeError:
                        pass
                affine = _reconstruct_affine(item)
                if affine is None:
                    logger.warning("S1 item %s has no proj:transform — skipping", item.id)
                    return 0
                cols = _extract_item(item, affine, bbox_wgs84, shard_points, cache_dir=resolved_cache)
                if cols is None:
                    return 0
                pids, lons, lats, dates, vhs, vvs, orbits = cols
                with write_lock:
                    buf_pid.extend(pids);   buf_lon.extend(lons);   buf_lat.extend(lats)
                    buf_date.extend(dates); buf_vh.extend(vhs);     buf_vv.extend(vvs)
                    buf_orbit.extend(orbits)
                    if len(buf_pid) >= _FLUSH_ROWS:
                        _flush_locked()
                return len(pids)

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_fetch_item, item): item for item in items}
                for fut in as_completed(futures):
                    n_items_done += 1
                    total_rows += fut.result()
                    if n_items_done % log_interval == 0 or n_items_done == len(items):
                        prefix = f"shard {shard_idx + 1}/{n_shards}  " if use_sharding else ""
                        logger.info(
                            "S1: %sitem %d/%d  %d rows so far",
                            prefix, n_items_done, len(items), total_rows,
                        )

            with write_lock:
                _flush_locked()
            writer.close()

            if use_sharding:
                logger.info(
                    "S1: shard %d/%d complete — %d rows",
                    shard_idx + 1, n_shards, total_rows,
                )
            if total_rows > 0:
                shard_paths.append(shard_path)
            else:
                shard_path.unlink(missing_ok=True)

        if not shard_paths:
            logger.info("S1: no usable observations extracted")
            return pd.DataFrame()

        if len(shard_paths) == 1:
            df = pq.read_table(shard_paths[0]).to_pandas()
        else:
            df = pq.read_table(shard_paths).to_pandas()

    df["date"] = pd.to_datetime(df["date"])
    logger.info("S1: %d observations for %d unique dates", len(df), df["date"].nunique())
    return df
