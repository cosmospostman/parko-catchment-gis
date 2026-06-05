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

import asyncio
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

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


# ---------------------------------------------------------------------------
# _read_band_array: kept for unit tests and cache-miss fallback
# ---------------------------------------------------------------------------

def _read_band_array(
    href: str,
    affine,
    bbox_native: list[float],
    cache_path: "Path | None" = None,
    nodata: float = _S1_NODATA,
) -> "tuple[np.ndarray, object] | None":
    """Read a single S1 band COG, returning (2D float32 array, win_affine) or None.

    bbox_native is in the raster's native CRS (projected for RTC items).
    Loads from cache_path if it exists; writes to cache_path after a successful read.

    This function is used by unit tests and by _extract_item (legacy public API).
    The main collection path uses fetch_patches instead.
    """
    import rasterio
    import rasterio.windows
    from utils.fetch import _load_patch_cache, _save_patch_cache

    if cache_path is not None and cache_path.exists():
        cached = _load_patch_cache(cache_path)
        if cached is not None:
            arr, win_affine, _ = cached
            return arr, win_affine

    try:
        with rasterio.open(href) as src:
            win = _pixel_window(affine, bbox_native, src.width, src.height)
            if win is None or win.width <= 0 or win.height <= 0:
                return None
            arr = src.read(1, window=win).astype(np.float32)
            arr[arr == nodata] = np.nan
            arr[arr == 0] = np.nan
            win_affine = rasterio.windows.transform(win, affine)
            crs = src.crs
    except Exception as exc:
        logger.debug("S1 band read failed (%s): %s", href, exc)
        return None

    if cache_path is not None:
        try:
            _save_patch_cache(cache_path, (arr, win_affine, crs), fetch_bbox=None)
        except Exception as exc:
            logger.warning("S1 chip cache write failed (%s): %s", cache_path, exc)

    return arr, win_affine


def _extract_item_projected(
    item,
    affine,
    bbox_wgs84: list[float],
    pids: list[str],
    lons: np.ndarray,
    lats: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    cache_dir: "Path | None" = None,
) -> "tuple[list, list, list, list, list, list, list] | None":
    """Extract per-pixel VH/VV for one S1 item using pre-projected point coordinates.

    px/py must already be in the item's native CRS.  Used by the legacy
    _extract_item public API and by unit tests.

    Returns a 7-tuple (point_ids, lons, lats, dates, vh_vals, vv_vals, orbits)
    for pixels with at least one valid observation, or None if no overlap.
    """
    from utils.fetch import _cache_path as _fc_path, _load_patch_cache

    _dt = item.datetime
    date = _dt.replace(tzinfo=None) if hasattr(_dt, "replace") else _dt
    orbit_state = item.properties.get("sat:orbit_state", None)

    _MARGIN = 0.01
    bbox_fetch = [
        bbox_wgs84[0] - _MARGIN, bbox_wgs84[1] - _MARGIN,
        bbox_wgs84[2] + _MARGIN, bbox_wgs84[3] + _MARGIN,
    ]

    crs = _item_crs(item)
    if crs and crs != "EPSG:4326":
        bbox_native = _reproject_bbox(bbox_fetch, crs)
    else:
        bbox_native = bbox_fetch

    vh_arr = vv_arr = None
    win_affine = None

    for band in _S1_BANDS:
        if band not in item.assets:
            continue
        href = item.assets[band].href
        # Check fetch.py cache first (new format), fall back to direct read
        cache_path = _fc_path(cache_dir, item.id, band) if cache_dir else None
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

    # px/py already in native CRS — no reprojection needed here
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
        [pids[i] for i in idx],
        lons[idx].tolist(),
        lats[idx].tolist(),
        [date] * n,
        vh_vals.tolist(),
        vv_vals.tolist(),
        [orbit_state] * n,
    )


def _extract_item(
    item,
    affine,
    bbox_wgs84: list[float],
    points: list[tuple[str, float, float]],
    cache_dir: "Path | None" = None,
) -> "tuple[list, list, list, list, list, list, list] | None":
    """Extract per-pixel VH/VV for one S1 item.

    Convenience wrapper used by unit tests and the legacy collect_s1 public API.
    The main collection path uses fetch_patches + CachedNpzChipStore instead.
    """
    from pyproj import Transformer
    pids = [p[0] for p in points]
    lons = np.array([p[1] for p in points], dtype=np.float64)
    lats = np.array([p[2] for p in points], dtype=np.float64)
    crs = _item_crs(item)
    if crs and crs != "EPSG:4326":
        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        px, py = t.transform(lons, lats)
        px = np.asarray(px, dtype=np.float64)
        py = np.asarray(py, dtype=np.float64)
    else:
        px, py = lons, lats
    return _extract_item_projected(
        item, affine, bbox_wgs84, pids, lons, lats, px, py, cache_dir=cache_dir,
    )


def _tile_bbox(bbox_wgs84: list[float]) -> list[float]:
    """Snap bbox outward to 1° grid — all regions in the same degree-tile share one cache entry."""
    import math
    minx, miny, maxx, maxy = bbox_wgs84
    return [math.floor(minx), math.floor(miny), math.ceil(maxx), math.ceil(maxy)]


def _resolve_s1_items(
    bbox_wgs84: list[float],
    start: str,
    end: str,
    resolved_cache: Path,
):
    """STAC search + cache for S1 items. Returns filtered item list or []."""
    import hashlib
    import pickle

    # Cache at tile granularity (1° grid) so nearby regions share one GeoParquet scan.
    # The exact bbox filter is still applied by filter_items_by_bbox after loading.
    search_bbox = _tile_bbox(bbox_wgs84)
    stac_key = hashlib.md5(
        f"{search_bbox}|{start}|{end}".encode()
    ).hexdigest()
    stac_cache_dir = resolved_cache / "stac"
    stac_cache_dir.mkdir(parents=True, exist_ok=True)
    stac_cache = stac_cache_dir / f"s1_{stac_key}.pkl"

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
        logger.info(
            "S1 STAC search: endpoint=%s collection=%s bbox=%s start=%s end=%s",
            _STAC_ENDPOINT, _S1_COLLECTION, search_bbox, start, end,
        )
        # Fetch unsigned so cached items don't carry expired SAS tokens.
        # Per-item signing happens in fetch_patches at read time.
        items = search_sentinel1(
            bbox=search_bbox,
            start=start,
            end=end,
            endpoint=_STAC_ENDPOINT,
            collection=_S1_COLLECTION,
            modifier=None,
        )
        try:
            with stac_cache.open("wb") as fh:
                pickle.dump(items, fh)
            logger.info("S1 STAC: %d items found and cached (%s)", len(items), stac_cache.name)
        except Exception as _e:
            logger.warning("S1 STAC: cache write failed (%s) — skipping", _e)

    items = filter_items_by_bbox(items, bbox_wgs84)
    return items


def _extract_s1_from_store(
    item,
    store,
    point_ids: list[str],
    lons: np.ndarray,
    lats: np.ndarray,
) -> "tuple[list[str], np.ndarray, np.ndarray, object, np.ndarray, np.ndarray, str | None] | None":
    """Extract per-pixel VH/VV for one S1 item from a CachedNpzChipStore.

    Returns a 7-tuple (point_ids, lons, lats, date, vh_vals, vv_vals, orbit_state)
    where lons/lats/vh/vv are numpy arrays and date/orbit_state are scalars.
    Returns None if nothing valid.
    """
    item_id = item.id
    _dt = item.datetime
    date = _dt.replace(tzinfo=None) if hasattr(_dt, "replace") else _dt
    orbit_state = item.properties.get("sat:orbit_state", None)

    vh_vals = store.get_all_points(item_id, "vh")
    vv_vals = store.get_all_points(item_id, "vv")

    if vh_vals is None and vv_vals is None:
        return None

    n = len(point_ids)
    if vh_vals is None:
        vh_vals = np.full(n, np.nan, dtype=np.float32)
    if vv_vals is None:
        vv_vals = np.full(n, np.nan, dtype=np.float32)

    # Zero is S1 no-data (in addition to the -32768 nodata stored as NaN by rasterio)
    vh_vals = np.where(vh_vals == 0, np.nan, vh_vals)
    vv_vals = np.where(vv_vals == 0, np.nan, vv_vals)

    keep = ~(np.isnan(vh_vals) & np.isnan(vv_vals))
    idx = np.where(keep)[0]
    if idx.size == 0:
        return None

    return (
        [point_ids[i] for i in idx],
        lons[idx],
        lats[idx],
        date,
        vh_vals[idx],
        vv_vals[idx],
        orbit_state,
    )


_S1_EXTRACT_WORKERS = 8


def _collect_s1_shards(
    out_dir: Path,
    items: list,
    bbox_wgs84: list[float],
    points: list[tuple[str, float, float]],
    resolved_cache: Path,
    point_shard_size: int = 500_000,
    max_concurrent: int = 32,
    phases: "set[str] | None" = None,
    on_fetch_tick: "Callable[[int], None] | None" = None,
    on_extract_tick: "Callable[[int], None] | None" = None,
) -> list[Path]:
    """Fetch S1 patches via fetch_patches (async, semaphore-bounded) then extract.

    Uses the same fetch_patches + CachedNpzChipStore pattern as the S2 pipeline,
    giving true async concurrency for COG range requests instead of a fixed thread
    pool. max_concurrent controls the semaphore passed to fetch_patches.

    on_fetch_tick(n_done) and on_extract_tick(n_done) are called as items complete
    in each phase (used by the pipeline to drive the progress display).

    Returns list of written shard paths (empty if no data). Caller owns out_dir.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from utils.fetch import fetch_patches
    from utils.chip_store import CachedNpzChipStore

    from utils.signing import make_mpc_signer
    _sign = make_mpc_signer()

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

    _do_fetch   = phases is None or "fetch"   in phases
    _do_extract = phases is None or "extract" in phases

    n_shards = max(1, (len(points) + point_shard_size - 1) // point_shard_size)
    use_sharding = n_shards > 1
    shard_paths: list[Path] = []

    if use_sharding:
        logger.info("S1: %d shards total (%d points each)", n_shards, point_shard_size)

    # Fetch patches once for all shards — NPZ cache is keyed by (item_id, band),
    # independent of which subset of points will be extracted.
    if _do_fetch:
        logger.info(
            "S1: fetching patches for %d items (%d concurrent)",
            len(items), max_concurrent,
        )
        _fetch_done = [0]
        _fetch_lock = threading.Lock()

        def _on_fetch_item(item_id: str) -> None:
            with _fetch_lock:
                _fetch_done[0] += 1
                n = _fetch_done[0]
            if on_fetch_tick is not None:
                on_fetch_tick(n)

        asyncio.run(fetch_patches(
            points=points,
            items=items,
            bands=_S1_BANDS,
            bbox_wgs84=bbox_wgs84,
            scl_filter=False,
            max_concurrent=max_concurrent,
            band_alias=None,
            cache_dir=resolved_cache,
            item_signer=_sign,
            sensor_label="S1",
            on_item_done=_on_fetch_item if on_fetch_tick is not None else None,
        ))

    for shard_idx in range(n_shards):
        shard_points = points[shard_idx * point_shard_size : (shard_idx + 1) * point_shard_size]
        shard_path = out_dir / f"shard_{shard_idx:04d}.parquet"
        if shard_path.exists():
            logger.info("S1: shard %d/%d — cached, skipping", shard_idx + 1, n_shards)
            shard_paths.append(shard_path)
            continue
        if use_sharding:
            logger.info("S1: shard %d/%d — %d points", shard_idx + 1, n_shards, len(shard_points))

        point_ids = [p[0] for p in shard_points]
        lons = np.array([p[1] for p in shard_points], dtype=np.float64)
        lats = np.array([p[2] for p in shard_points], dtype=np.float64)
        point_coords = {pid: (lon, lat) for pid, lon, lat in shard_points}

        if not _do_extract:
            continue

        # Phase 2: parallel extract — one CachedNpzChipStore per worker thread
        # (store is not thread-safe; each thread gets its own instance).
        _tls = threading.local()

        def _get_store() -> "CachedNpzChipStore":
            if not hasattr(_tls, "store"):
                _tls.store = CachedNpzChipStore(
                    cache_dir=resolved_cache,
                    point_coords=point_coords,
                    bands=_S1_BANDS,
                )
            return _tls.store

        def _extract_one(item):
            store = _get_store()
            result = _extract_s1_from_store(item, store, point_ids, lons, lats)
            store.release_item(item.id)
            return result

        # Collect results in submission order so parquet rows are date-ordered.
        writer = pq.ParquetWriter(shard_path, _ARROW_SCHEMA, compression="none", write_statistics=False)
        total_rows = 0
        log_interval = max(1, len(items) // 10)

        acc_pid:   list[list[str]]   = []
        acc_lon:   list[np.ndarray]  = []
        acc_lat:   list[np.ndarray]  = []
        acc_date:  list[np.ndarray]  = []
        acc_vh:    list[np.ndarray]  = []
        acc_vv:    list[np.ndarray]  = []
        acc_orbit: list[np.ndarray]  = []
        acc_rows = 0
        _FLUSH_ROWS = 500_000

        def _flush() -> None:
            nonlocal acc_rows
            if not acc_pid:
                return
            all_pids   = [p for chunk in acc_pid for p in chunk]
            all_lons   = np.concatenate(acc_lon)
            all_lats   = np.concatenate(acc_lat)
            all_dates  = np.concatenate(acc_date)
            all_vhs    = np.concatenate(acc_vh)
            all_vvs    = np.concatenate(acc_vv)
            all_orbits = np.concatenate(acc_orbit)
            tbl = pa.table({
                "point_id": pa.array(all_pids,   pa.string()),
                "lon":      pa.array(all_lons,   pa.float64()),
                "lat":      pa.array(all_lats,   pa.float64()),
                "date":     pa.array(all_dates,  pa.timestamp("ms")),
                "source":   pa.repeat("S1", len(all_pids)).cast(pa.string()),
                "vh":       pa.array(all_vhs,    pa.float32()),
                "vv":       pa.array(all_vvs,    pa.float32()),
                "orbit":    pa.array(all_orbits, pa.string()),
            })
            writer.write_table(tbl)
            acc_pid.clear(); acc_lon.clear(); acc_lat.clear(); acc_date.clear()
            acc_vh.clear();  acc_vv.clear();  acc_orbit.clear()
            acc_rows = 0

        _extract_done = [0]

        with ThreadPoolExecutor(max_workers=_S1_EXTRACT_WORKERS) as pool:
            futs = {pool.submit(_extract_one, item): item for item in items}

            for fut in as_completed(futs):
                result = fut.result()
                _extract_done[0] += 1
                if on_extract_tick is not None:
                    on_extract_tick(_extract_done[0])

                if result is None:
                    continue
                pids, i_lons, i_lats, date, vhs, vvs, orbit_state = result
                n_item = len(pids)
                acc_pid.append(pids)
                acc_lon.append(i_lons)
                acc_lat.append(i_lats)
                acc_date.append(np.full(n_item, np.datetime64(date, "ms")))
                acc_vh.append(vhs)
                acc_vv.append(vvs)
                acc_orbit.append(np.full(n_item, orbit_state if orbit_state is not None else "", dtype=object))
                acc_rows += n_item
                total_rows += n_item
                if acc_rows >= _FLUSH_ROWS:
                    _flush()

                done_n = _extract_done[0]
                if done_n % log_interval == 0 or done_n == len(items):
                    if use_sharding:
                        logger.info(
                            "S1 scenes  shard %d/%d  item %d/%d  %d rows",
                            shard_idx + 1, n_shards, done_n, len(items), total_rows,
                        )
                    else:
                        logger.info("S1: item %d/%d  %d rows so far", done_n, len(items), total_rows)

        _flush()
        writer.close()

        if use_sharding:
            logger.info("S1: shard %d/%d complete — %d rows", shard_idx + 1, n_shards, total_rows)
        if total_rows > 0:
            shard_paths.append(shard_path)
        else:
            shard_path.unlink(missing_ok=True)

    return shard_paths


def collect_s1_for_tile(
    s2_path: "Path | None",
    bbox_wgs84: list[float],
    start: str,
    end: str,
    out_path: Path,
    cache_dir: "Path | None" = None,
    max_concurrent: int = 32,
    points: "list[tuple[str, float, float]] | None" = None,
    phases: "set[str] | None" = None,
    on_fetch_tick: "Callable[[int], None] | None" = None,
    on_extract_tick: "Callable[[int], None] | None" = None,
    on_items_resolved: "Callable[[int], None] | None" = None,
) -> "Path | None":
    """Fetch S1 for a tile's pixel grid and write a sorted S1 parquet to out_path.

    When *points* is supplied, coord reading from *s2_path* is skipped entirely
    and *s2_path* may be None.  Schema is taken from COMBINED_PIXEL_SCHEMA.
    When *points* is absent, coords are read from *s2_path* and schema is
    derived from that file via _extend_schema() — existing callers unchanged.

    Idempotent: returns out_path immediately if it already exists and is non-empty.
    Returns out_path on success, or None if no S1 data is available.
    """
    import tempfile
    import pyarrow.parquet as pq
    from utils.parquet_utils import _extend_schema, _sort_s1_shards, COMBINED_PIXEL_SCHEMA

    _fetch_only = phases is not None and phases == {"fetch"}

    if not _fetch_only and out_path.exists() and out_path.stat().st_size > 0:
        try:
            pq.ParquetFile(out_path).metadata  # validates magic bytes and footer
            logger.info("%s already exists — skipping", out_path.name)
            return out_path
        except Exception:
            logger.warning("%s is corrupt — rebuilding", out_path.name)
            out_path.unlink()

    setup_gdal_env()

    resolved_cache = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR

    if points is not None:
        # Proxy path: caller supplies the pixel grid directly; no s2 file needed.
        combined_schema = COMBINED_PIXEL_SCHEMA
    else:
        # Local path: read points from s2 parquet (thread-safe via PyArrow group_by).
        if s2_path is None:
            raise ValueError("collect_s1_for_tile: s2_path required when points is not supplied")
        _coords_tbl = pq.ParquetFile(s2_path).read(columns=["point_id", "lon", "lat"])
        _coords_tbl = _coords_tbl.group_by("point_id").aggregate([("lon", "min"), ("lat", "min")])
        points = list(zip(
            _coords_tbl.column("point_id").to_pylist(),
            _coords_tbl.column("lon_min").to_pylist(),
            _coords_tbl.column("lat_min").to_pylist(),
        ))
        del _coords_tbl
        combined_schema = _extend_schema(pq.ParquetFile(s2_path).schema_arrow)

    items = _resolve_s1_items(bbox_wgs84, start, end, resolved_cache)
    if not items:
        logger.info("no items for bbox %s %s/%s", bbox_wgs84, start, end)
        return None

    if on_items_resolved is not None:
        on_items_resolved(len(items))

    with tempfile.TemporaryDirectory(prefix="s1_tile_") as _tmp:
        shard_paths = _collect_s1_shards(
            out_dir=Path(_tmp),
            items=items,
            bbox_wgs84=bbox_wgs84,
            points=points,
            resolved_cache=resolved_cache,
            max_concurrent=max_concurrent,
            phases=phases,
            on_fetch_tick=on_fetch_tick,
            on_extract_tick=on_extract_tick,
        )

        if _fetch_only:
            return None

        if not shard_paths:
            logger.info("no usable observations")
            return None

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _sort_s1_shards(shard_paths, out_path, combined_schema)

    logger.info("wrote %s", out_path.name)
    return out_path


def collect_s1(
    bbox_wgs84: list[float],
    start: str,
    end: str,
    points: list[tuple[str, float, float]],
    cache_dir: "Path | None" = None,
    point_shard_size: int = 50_000,
    max_concurrent: int = 32,
) -> pl.DataFrame:
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
    max_concurrent:
        Maximum simultaneous in-flight network requests passed to fetch_patches.
        Defaults to 32. Increase to saturate a high-bandwidth link; each
        in-flight request holds at most one band patch (~15 MB) in memory.

    Returns
    -------
    DataFrame with columns [point_id, lon, lat, date, source, vh, vv, orbit].
    Empty DataFrame if no S1 data is found.
    """
    import tempfile
    import pyarrow.parquet as pq

    setup_gdal_env()

    resolved_cache = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR

    items = _resolve_s1_items(bbox_wgs84, start, end, resolved_cache)
    if not items:
        logger.info("No S1 items found for bbox %s in %s/%s", bbox_wgs84, start, end)
        return pl.DataFrame()

    with tempfile.TemporaryDirectory(prefix="s1_collect_") as _tmp:
        shard_paths = _collect_s1_shards(
            out_dir=Path(_tmp),
            items=items,
            bbox_wgs84=bbox_wgs84,
            points=points,
            resolved_cache=resolved_cache,
            point_shard_size=point_shard_size,
            max_concurrent=max_concurrent,
        )

        if not shard_paths:
            logger.info("S1: no usable observations extracted")
            return pl.DataFrame()

        df = pl.read_parquet(shard_paths if len(shard_paths) > 1 else shard_paths[0])

    n_unique_dates = df["date"].n_unique()
    logger.info("S1: %d observations for %d unique dates", len(df), n_unique_dates)
    return df
