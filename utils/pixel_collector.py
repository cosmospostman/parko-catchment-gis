"""utils/pixel_collector.py — collect all S2 observations for a bbox.

Fetches every available Sentinel-2 L2A acquisition over a bounding box for a
given date range and writes a single Parquet file containing one row per
(pixel, date) observation. All spectral bands are retained as surface
reflectance values alongside per-observation quality scores.

The output is intentionally broad — every band that might be useful for signal
exploration is collected once so that downstream analysis scripts can iterate
quickly without re-fetching from the network.

Output schema
-------------
point_id      : str   — pixel grid identifier, e.g. "px_0042_0031"
lon           : float — pixel centre longitude (EPSG:4326)
lat           : float — pixel centre latitude  (EPSG:4326)
date          : date  — acquisition date (UTC, date only)
item_id       : str   — STAC item ID
tile_id       : str   — S2 MGRS tile identifier
B02           : float — blue            (surface reflectance 0–1)
B03           : float — green
B04           : float — red
B05           : float — red-edge 1
B06           : float — red-edge 2
B07           : float — red-edge 3
B08           : float — NIR broad
B8A           : float — NIR narrow
B11           : float — SWIR 1.6 µm
B12           : float — SWIR 2.2 µm
scl_purity    : float — fraction of clear pixels in the 5×5 chip window
scl           : int8  — raw SCL class value (4=vegetation, 5=bare soil, 6=water, 7=unclassified, 11=snow/ice)
aot           : float — inverse aerosol optical thickness  (1 = clean air)
view_zenith   : float — inverse view zenith angle          (1 = nadir)
sun_zenith    : float — inverse sun zenith angle           (1 = high sun)
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS, SCL_BAND, AOT_BAND, SCL_CLEAR_VALUES, UINT16_BAND_SCALE
from utils.chip_store import CachedNpzChipStore, MemoryChipStore
from utils.fetch import fetch_patches
from utils.stac import search_sentinel2

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Raised when a fetch pipeline step fails (replaces sys.exit(1))."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION  = "sentinel-2-l2a"

_TILE_ID_RE = re.compile(r"^S2[ABC]_(\d{2}[A-Z]{3})_")


def _utm_crs_for_bbox(bbox_wgs84: list[float]) -> str:
    """Return the UTM CRS EPSG code for the centre longitude of a WGS84 bbox."""
    lon_centre = (bbox_wgs84[0] + bbox_wgs84[2]) / 2
    lat_centre = (bbox_wgs84[1] + bbox_wgs84[3]) / 2
    zone = min(int((lon_centre + 180) / 6) + 1, 60)
    epsg = 32600 + zone if lat_centre >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

# earth-search asset key aliases for S2 L2A
BAND_ALIAS: dict[str, str] = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
    "AOT": "aot",
}

FETCH_BANDS = BANDS + [SCL_BAND, AOT_BAND]   # VZA/SZA not available at earth-search


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def make_pixel_grid(
    bbox_wgs84: list[float],
    utm_crs: str | None = None,
    resolution_m: float = 10.0,
    point_id_prefix: str = "px",
) -> list[tuple[str, float, float]]:
    """Generate one point per S2 pixel inside bbox_wgs84, aligned to a 10 m UTM grid.

    The grid origin is snapped to the nearest 10 m multiple so that points
    fall at S2 pixel centres rather than between pixels.

    Returns list of (point_id, lon, lat).
    """
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    if utm_crs is None:
        utm_crs = _utm_crs_for_bbox(bbox_wgs84)

    to_utm  = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs  = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    x0, y0 = to_utm.transform(lon_min, lat_min)
    x1, y1 = to_utm.transform(lon_max, lat_max)

    # Snap to nearest 10 m grid origin (aligns with S2 pixel grid)
    r = resolution_m
    x0_snap = np.floor(x0 / r) * r
    y0_snap = np.floor(y0 / r) * r

    xs = np.arange(x0_snap, x1, r)
    ys = np.arange(y0_snap, y1, r)

    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    lons, lats = to_wgs.transform(xx.ravel(), yy.ravel())
    ii, jj = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)), indexing="ij")
    pids = [f"{point_id_prefix}_{i:04d}_{j:04d}" for i, j in zip(ii.ravel(), jj.ravel())]
    points = list(zip(pids, lons.tolist(), lats.tolist()))

    logger.info(
        "Pixel grid: %d × %d = %d points at %.0f m spacing",
        len(xs), len(ys), len(points), r,
    )
    return points


# ---------------------------------------------------------------------------
# Vectorised per-item extraction → DataFrame (no Observation objects)
# ---------------------------------------------------------------------------

def _band_to_uint16(arr: np.ndarray) -> pl.Series:
    """Convert float32 reflectance [0, 1] to a nullable uint16 Series ×10000.

    NaN values (S1 rows carry no S2 bands) become null in the output column.
    """
    nan_mask = np.isnan(arr)
    safe = np.where(nan_mask, 0.0, arr)
    quantised = np.clip(np.round(safe * UINT16_BAND_SCALE), 0, 65535).astype(np.uint16)
    s = pl.Series(quantised, dtype=pl.UInt16)
    if nan_mask.any():
        s = s.scatter(np.where(nan_mask)[0].tolist(), None)
    return s


def _extract_item_from_tiffs(
    item,
    tiff_dir: Path,
    point_ids: list[str],
    lons: np.ndarray,
    lats: np.ndarray,
    apply_nbar: bool = True,
    utm_crs: str = "EPSG:32755",
) -> pl.DataFrame | None:
    """Extract all usable pixels for one STAC item from on-disk GeoTIFF patches.

    Counterpart to extract_item_to_df() for the network→disk pipeline.
    Reads from {tiff_dir}/{band}.tif files written by fetch_patches_to_tiff().
    Peak RAM = one item's bands sampled to n_points floats — a few MB per worker.

    Returns None if the item has no clear pixels or the SCL tif is missing.
    """
    import rasterio
    from pyproj import Transformer, CRS

    item_id = item.id
    m = _TILE_ID_RE.match(item_id)
    tile_id = m.group(1) if m else item.properties.get("s2:mgrs_tile", "")
    _dt = item.datetime.replace(tzinfo=None)
    item_date = _dt
    n = len(point_ids)

    def _sample_tif(band: str) -> np.ndarray | None:
        """Open {tiff_dir}/{band}.tif and sample pixel values at all points."""
        path = tiff_dir / f"{band}.tif"
        if not path.exists():
            return None
        try:
            with rasterio.open(path) as src:
                crs = src.crs
                transform = src.transform
                h, w = src.height, src.width
                t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                xs, ys = t.transform(lons, lats)
                cols_f, rows_f = ~transform * (xs, ys)
                rows_raw = np.floor(rows_f).astype(np.intp)
                cols_raw = np.floor(cols_f).astype(np.intp)
                _EDGE_SLOP = 1
                oob_mask = (
                    (rows_raw < -_EDGE_SLOP) | (rows_raw >= h + _EDGE_SLOP) |
                    (cols_raw < -_EDGE_SLOP) | (cols_raw >= w + _EDGE_SLOP)
                )
                rows = np.clip(rows_raw, 0, h - 1)
                cols = np.clip(cols_raw, 0, w - 1)
                arr = src.read(1)
                vals = arr[rows, cols].astype(np.float32)
                if oob_mask.any():
                    vals[oob_mask] = np.nan
                return vals
        except Exception as exc:
            logger.debug("_extract_item_from_tiffs: failed to read %s/%s: %s", item_id, band, exc)
            return None

    # SCL
    scl_vals = _sample_tif(SCL_BAND)
    if scl_vals is None:
        return None
    with np.errstate(invalid="ignore"):
        scl_int = scl_vals.astype(np.int32)
    clear_mask = np.isin(scl_int, list(SCL_CLEAR_VALUES))
    if not clear_mask.any():
        return None
    scl_purity = clear_mask.astype(np.float32)

    # AOT
    aot_vals = _sample_tif(AOT_BAND)
    if aot_vals is not None:
        aot_quality = np.clip(1.0 - aot_vals * 0.001, 0.0, 1.0)
    else:
        aot_quality = np.ones(n, dtype=np.float32)

    # Spectral bands
    band_arrays: dict[str, np.ndarray] = {}
    for band in BANDS:
        vals = _sample_tif(band)
        band_arrays[band] = vals / 10000.0 if vals is not None else np.full(n, np.nan, dtype=np.float32)

    # NBAR c-factor correction
    angles = None
    if apply_nbar:
        from utils.granule_angles import get_item_angles
        from utils.nbar import c_factor as compute_cf
        angles = get_item_angles(item, lons, lats, utm_crs=utm_crs, bands=list(BANDS))
        if angles is not None:
            for band in BANDS:
                if band not in angles or band_arrays[band] is None:
                    continue
                a = angles[band]
                raa = a["saa"] - a["vaa"]
                cf = compute_cf(a["sza"], a["vza"], raa, band)
                corrected = np.clip(band_arrays[band] * cf, 0.0, 1.0)
                band_arrays[band] = np.where(np.isnan(cf), band_arrays[band], corrected)

    # Zenith columns
    if angles is not None:
        sza_mean = np.mean(
            [angles[b]["sza"] for b in BANDS if b in angles], axis=0
        ).astype(np.float32)
        vza_mean = np.mean(
            [angles[b]["vza"] for b in BANDS if b in angles], axis=0
        ).astype(np.float32)
        sun_zenith_col  = np.where(np.isnan(sza_mean), 1.0, np.clip(1.0 - sza_mean / 90.0, 0.0, 1.0))
        view_zenith_col = np.where(np.isnan(vza_mean), 1.0, np.clip(1.0 - vza_mean / 90.0, 0.0, 1.0))
    else:
        sun_zenith_col  = np.ones(n, dtype=np.float32)
        view_zenith_col = np.ones(n, dtype=np.float32)

    idx = np.where(clear_mask)[0]
    n_clear = len(idx)
    if n_clear == 0:
        return None

    band_data = {band: band_arrays[band][idx] for band in BANDS}

    all_nan_mask = np.ones(n_clear, dtype=bool)
    for arr in band_data.values():
        all_nan_mask &= np.isnan(arr)
    if all_nan_mask.all():
        return None

    return pl.DataFrame({
        "point_id":    list(np.array(point_ids)[idx]),
        "lon":         lons[idx].astype(np.float64),
        "lat":         lats[idx].astype(np.float64),
        "date":        pl.Series([item_date] * n_clear),
        "item_id":     [item_id] * n_clear,
        "tile_id":     [tile_id] * n_clear,
        "scl_purity":  scl_purity[idx].astype(np.int8),
        "scl":         scl_int[idx].astype(np.int8),
        "aot":         np.round(aot_quality[idx] * 100).astype(np.uint8),
        "view_zenith": np.round(view_zenith_col[idx] * 100).astype(np.uint8),
        "sun_zenith":  np.round(sun_zenith_col[idx] * 100).astype(np.uint8),
        **{band: _band_to_uint16(arr) for band, arr in band_data.items()},
    })


def extract_item_to_df(
    item,
    store: MemoryChipStore,
    point_ids: list[str],
    lons: np.ndarray,
    lats: np.ndarray,
    apply_nbar: bool = True,
    utm_crs: str = "EPSG:32755",
) -> pl.DataFrame | None:
    """Extract all usable pixels for one STAC item into a DataFrame.

    Uses store.get_all_points() to fetch entire bands as numpy arrays,
    avoiding per-point Python loops. Returns None if the item has no
    clear pixels at all (cloud-filtered).
    """
    item_id = item.id
    m = _TILE_ID_RE.match(item_id)
    tile_id = m.group(1) if m else item.properties.get("s2:mgrs_tile", "")
    _dt = item.datetime.replace(tzinfo=None)
    item_date = _dt
    n = len(point_ids)

    # --- SCL: per-point clear-pixel mask ------------------------------------
    scl_vals = store.get_all_points(item_id, SCL_BAND)
    if scl_vals is None:
        return None
    with np.errstate(invalid="ignore"):  # NaN→int32 for OOB points is intentional
        scl_int = scl_vals.astype(np.int32)
    clear_mask = np.isin(scl_int, list(SCL_CLEAR_VALUES))
    if not clear_mask.any():
        return None
    scl_purity = clear_mask.astype(np.float32)  # 1×1 chip → purity = 0 or 1

    # --- AOT quality --------------------------------------------------------
    aot_vals = store.get_all_points(item_id, AOT_BAND)
    if aot_vals is not None:
        aot_quality = np.clip(1.0 - aot_vals * 0.001, 0.0, 1.0)
    else:
        aot_quality = np.ones(n, dtype=np.float32)

    # --- Spectral bands (surface reflectance = raw / 10000) -----------------
    band_arrays: dict[str, np.ndarray] = {}
    for band in BANDS:
        vals = store.get_all_points(item_id, band)
        band_arrays[band] = vals / 10000.0 if vals is not None else np.full(n, np.nan, dtype=np.float32)

    # --- NBAR c-factor correction (optional) --------------------------------
    angles = None
    if apply_nbar:
        from utils.granule_angles import get_item_angles
        from utils.nbar import c_factor as compute_cf
        angles = get_item_angles(item, lons, lats, utm_crs=utm_crs, bands=list(BANDS))
        if angles is not None:
            for band in BANDS:
                if band not in angles or band_arrays[band] is None:
                    continue
                a = angles[band]
                raa = a["saa"] - a["vaa"]
                cf = compute_cf(a["sza"], a["vza"], raa, band)
                corrected = np.clip(band_arrays[band] * cf, 0.0, 1.0)
                # Fall back to uncorrected value where cf is NaN (angle interpolation
                # outside granule grid — affects pixels near tile edges)
                band_arrays[band] = np.where(np.isnan(cf), band_arrays[band], corrected)
        # If angles is None (fetch failed), band_arrays are left uncorrected

    # --- Zenith quality columns ---------------------------------------------
    if angles is not None:
        sza_mean = np.mean(
            [angles[b]["sza"] for b in BANDS if b in angles], axis=0
        ).astype(np.float32)
        vza_mean = np.mean(
            [angles[b]["vza"] for b in BANDS if b in angles], axis=0
        ).astype(np.float32)
        sun_zenith_col  = np.where(np.isnan(sza_mean), 1.0, np.clip(1.0 - sza_mean / 90.0, 0.0, 1.0))
        view_zenith_col = np.where(np.isnan(vza_mean), 1.0, np.clip(1.0 - vza_mean / 90.0, 0.0, 1.0))
    else:
        sun_zenith_col  = np.ones(n, dtype=np.float32)
        view_zenith_col = np.ones(n, dtype=np.float32)

    # --- Filter to clear pixels only ----------------------------------------
    idx = np.where(clear_mask)[0]
    n_clear = len(idx)
    if n_clear == 0:
        return None

    band_data = {band: band_arrays[band][idx] for band in BANDS}

    # Drop rows where all spectral bands are NaN
    all_nan_mask = np.ones(n_clear, dtype=bool)
    for arr in band_data.values():
        all_nan_mask &= np.isnan(arr)
    if all_nan_mask.all():
        return None

    df = pl.DataFrame({
        "point_id":    list(np.array(point_ids)[idx]),
        "lon":         lons[idx].astype(np.float64),
        "lat":         lats[idx].astype(np.float64),
        "date":        pl.Series([item_date] * n_clear),
        "item_id":     [item_id] * n_clear,
        "tile_id":     [tile_id] * n_clear,
        "scl_purity":  scl_purity[idx].astype(np.int8),
        "scl":         scl_int[idx].astype(np.int8),
        "aot":         np.round(aot_quality[idx] * 100).astype(np.uint8),
        "view_zenith": np.round(view_zenith_col[idx] * 100).astype(np.uint8),
        "sun_zenith":  np.round(sun_zenith_col[idx] * 100).astype(np.uint8),
        **{band: _band_to_uint16(arr) for band, arr in band_data.items()},
    })

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _auto_n_workers(n_pixels: int, n_bands: int = 12, target_gb: float = 4.0) -> int:
    """Return a safe worker count for concurrent item extraction.

    Each worker holds one item's full band patches in RAM.  The dominant cost
    is the numpy arrays produced by extract_item_to_df: n_pixels × n_bands
    float32 values plus a similar-sized Polars DataFrame copy.  We budget
    ~2× for the DataFrame overhead.
    """
    bytes_per_item = n_pixels * n_bands * 4 * 2  # ×2 for numpy + polars copy
    target_bytes = int(target_gb * 1024 ** 3)
    workers = max(1, min(8, target_bytes // max(1, bytes_per_item)))
    return workers


def _collect_per_scene(
    points: list[tuple[str, float, float]],
    items: list,
    bbox_wgs84: list[float],
    out_dir: Path,
    cache_dir: Path,
    apply_nbar: bool,
    max_concurrent: int,
    n_workers: int | None,
    phases: set[str] | None,
    utm_crs: str,
) -> Iterator[tuple[str, Path]]:
    """per_scene=True implementation for collect().

    Yields (scene_id, path) for each item that produces at least one clear
    pixel.  Each output parquet is sorted by point_id.  No shard accumulation,
    no stale-cache check (VM is freshly launched), no cross-tile dedup.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from utils.parquet_utils import _optimise_schema, _WRITE_OPTS

    _phases = phases if phases is not None else {"fetch", "extract"}
    out_dir.mkdir(parents=True, exist_ok=True)

    point_ids  = [pid          for pid, _, _   in points]
    lons       = np.array([lon for _, lon, _   in points], dtype=np.float64)
    lats       = np.array([lat for _, _, lat   in points], dtype=np.float64)
    point_coords = {pid: (lon, lat) for pid, lon, lat in points}

    col_order = (
        ["point_id", "lon", "lat", "date", "item_id", "tile_id"]
        + list(BANDS)
        + ["scl_purity", "scl", "aot", "view_zenith", "sun_zenith"]
    )

    # --- Fetch phase ---------------------------------------------------------
    if "fetch" in _phases:
        from utils.fetch import _cache_path
        import os as _os
        cached_keys: set[tuple[str, str]] = set()
        try:
            for _entry in _os.scandir(cache_dir):
                if _entry.is_dir():
                    for _f in _os.scandir(_entry.path):
                        if _f.name.endswith(".npz"):
                            cached_keys.add((_entry.name, _f.name[:-4]))
        except FileNotFoundError:
            pass
        uncached_items = [
            item for item in items
            if any((item.id, band) not in cached_keys for band in FETCH_BANDS)
        ]
        if uncached_items:
            logger.info("per_scene fetch: %d/%d items not yet cached", len(uncached_items), len(items))
            asyncio.run(fetch_patches(
                points=points,
                items=uncached_items,
                bands=FETCH_BANDS,
                bbox_wgs84=bbox_wgs84,
                scl_filter=True,
                band_alias=BAND_ALIAS,
                max_concurrent=max_concurrent,
                cache_dir=cache_dir,
            ))
        else:
            logger.info("per_scene fetch: all %d items already cached", len(items))

    if "extract" not in _phases:
        return

    # --- Extract phase: one parquet per item, parallelised over n_workers ----
    import threading as _threading
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed
    _thread_local = _threading.local()

    def _get_store() -> CachedNpzChipStore:
        if not hasattr(_thread_local, "store"):
            _thread_local.store = CachedNpzChipStore(
                cache_dir=cache_dir,
                point_coords=point_coords,
                bands=FETCH_BANDS,
            )
        return _thread_local.store

    def _extract_one(item_idx: int, item) -> tuple[str, Path] | None:
        """Extract one scene to parquet. Returns (scene_id, path) or None."""
        scene_id = item.id
        n_items = len(items)
        out_path = out_dir / f"scene_{item_idx:04d}.parquet"
        if out_path.exists() and out_path.stat().st_size > 0:
            try:
                pq.ParquetFile(out_path).metadata
                logger.info("per_scene: scene %d/%d %s already extracted — skipping",
                            item_idx + 1, n_items, scene_id)
                return (scene_id, out_path)
            except Exception:
                out_path.unlink(missing_ok=True)

        store = _get_store()
        df = extract_item_to_df(
            item, store, point_ids, lons, lats,
            apply_nbar=apply_nbar, utm_crs=utm_crs,
        )
        store.release_item(item.id)

        if df is None or len(df) == 0:
            logger.info("per_scene: scene %d/%d %s — no clear pixels, skipping",
                        item_idx + 1, n_items, scene_id)
            return None

        tbl = df.select(col_order).to_arrow()
        tbl = _optimise_schema(tbl)
        tbl_pl = pl.from_arrow(tbl).with_columns(
            pl.col("point_id").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("_northing")
        ).sort("_northing").drop("_northing")
        tbl_sorted = tbl_pl.to_arrow().cast(tbl.schema)

        tmp_path = out_path.with_suffix(".tmp.parquet")
        tmp_path.unlink(missing_ok=True)
        writer = pq.ParquetWriter(str(tmp_path), tbl_sorted.schema, **_WRITE_OPTS)
        writer.write_table(tbl_sorted)
        writer.close()
        tmp_path.replace(out_path)

        logger.info(
            "per_scene: scene %d/%d %s — %d rows → %s",
            item_idx + 1, n_items, scene_id, len(tbl_sorted), out_path.name,
        )
        return (scene_id, out_path)

    _n_extract = max(1, n_workers) if n_workers else 1
    with _TPE(max_workers=_n_extract) as pool:
        futs = {pool.submit(_extract_one, idx, item): idx for idx, item in enumerate(items)}
        for fut in _as_completed(futs):
            result = fut.result()
            if result is not None:
                yield result


def collect(
    bbox_wgs84: list[float],
    start: str,
    end: str,
    out_dir: Path,
    cloud_max: int,
    cache_dir: Path | None = None,
    apply_nbar: bool = True,
    max_concurrent: int = 32,
    items=None,
    point_id_prefix: str = "px",
    calibration_out: Path | None = None,
    geometry=None,
    n_workers: int | None = None,
    phases: set[str] | None = None,
    target_extraction_gb: float = 4.0,
    per_scene: bool = False,
) -> "list[Path] | Iterator[tuple[str, Path]]":
    """Collect S2 observations for bbox_wgs84, writing one parquet per S2 tile.

    Output files are written to ``out_dir/<tile_id>.parquet``, one per MGRS
    tile that has observations.  Returns the list of written paths.

    If *items* is provided (a pre-fetched, deduplicated STAC item list), the
    STAC search step is skipped entirely.  This lets the caller share one STAC
    search result across multiple collect() calls for the same tile.

    If *geometry* is provided (a Shapely geometry), only pixels whose centres
    fall inside the geometry are fetched and stored.  The bbox_wgs84 is still
    used for STAC search and COG reads (unavoidable — rasterio reads rectangular
    windows), but the chip cache and parquet output will only contain
    polygon-interior pixels.

    *n_workers* controls how many items are extracted concurrently during the
    shard write phase.  Each worker holds one item's full band patches in RAM,
    so this is the primary memory knob for large locations.  ``None`` (default)
    auto-scales based on pixel count and *target_extraction_gb*.

    *target_extraction_gb* is the RAM budget passed to _auto_n_workers when
    *n_workers* is None.  Callers that derive this from a system memory budget
    (e.g. fetch_spec._budget_params) should pass it here; the default of 4 GB
    matches the historical behaviour for direct callers.

    *phases* selects which pipeline phases to run.  Defaults to both:
      ``{"fetch", "extract"}``
    Pass ``{"fetch"}`` to only populate the .npz patch cache (network I/O, low
    memory).  Pass ``{"extract"}`` to skip network fetch and only build parquets
    from an already-populated cache.  Both phases are independently idempotent.
    """
    _phases = phases if phases is not None else {"fetch", "extract"}
    # --- 1. Generate pixel grid -------------------------------------------
    utm_crs = _utm_crs_for_bbox(bbox_wgs84)
    points = make_pixel_grid(bbox_wgs84, utm_crs=utm_crs, point_id_prefix=point_id_prefix)

    if geometry is not None:
        from shapely.geometry import MultiPoint
        before = len(points)
        mp = MultiPoint([(lon, lat) for _, lon, lat in points])
        points = [pt for pt, contained in zip(points, [geometry.contains(p) for p in mp.geoms]) if contained]
        logger.info(
            "Polygon mask: %d / %d points retained (%.0f%%)",
            len(points), before, 100 * len(points) / before if before else 0,
        )

    # Resolve n_workers after the pixel grid so the auto-scale sees the real count.
    if n_workers is None:
        n_workers = _auto_n_workers(len(points), target_gb=target_extraction_gb)
    else:
        n_workers = max(1, n_workers)
    logger.info("Extraction workers: %d (pixel count %d, target %.0f GB)", n_workers, len(points), target_extraction_gb)


    point_coords = {pid: (lon, lat) for pid, lon, lat in points}

    import hashlib, pickle
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "data" / "chips" / out_dir.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. STAC search (cached, or use caller-supplied items) ---------------
    if items is not None:
        logger.info("Using %d pre-supplied STAC items (skipping search)", len(items))
    else:
        stac_key = hashlib.md5(
            f"{bbox_wgs84}|{start}|{end}|{cloud_max}".encode()
        ).hexdigest()
        stac_cache = cache_dir / f"stac_{stac_key}.pkl"

        if stac_cache.exists():
            logger.info("STAC search: loading cached result (%s)", stac_cache.name)
            with stac_cache.open("rb") as fh:
                items = pickle.load(fh)
            logger.info("%d items loaded from cache", len(items))
        else:
            logger.info("STAC search: %s → %s  cloud < %d%%", start, end, cloud_max)
            items = search_sentinel2(
                bbox=bbox_wgs84,
                start=start,
                end=end,
                cloud_cover_max=cloud_max,
                endpoint=STAC_ENDPOINT,
                collection=S2_COLLECTION,
            )
            if not items:
                raise FetchError("No STAC items found — check bbox and date range")
            with stac_cache.open("wb") as fh:
                pickle.dump(items, fh)
            logger.info("%d items found, cached to %s", len(items), stac_cache.name)
        if not items:
            raise FetchError("No STAC items found — check bbox and date range")

        # Deduplicate items: keep only one granule per (date, satellite, tile).
        # A bbox that straddles two adjacent S2 processing granules (e.g. _0_L2A
        # and _1_L2A from the same overpass) will otherwise produce duplicate
        # (point_id, date) rows with slightly different resampled band values.
        # Item IDs follow the pattern S2X_TTTTTT_YYYYMMDD_N_L2A where N is the
        # granule index — we strip it to get a canonical per-overpass key.
        _granule_re = re.compile(r"_\d+_L2A$")
        seen: set[tuple] = set()
        deduped_items = []
        for item in items:
            key = _granule_re.sub("", item.id)
            if key not in seen:
                seen.add(key)
                deduped_items.append(item)
        if len(deduped_items) < len(items):
            logger.info(
                "Deduplicated STAC items: %d → %d (removed %d duplicate granules)",
                len(items), len(deduped_items), len(items) - len(deduped_items),
            )
        items = deduped_items

    # --- per_scene mode: yield one sorted parquet per item -------------------
    # Used by proxy/server.py. Skips shard planning, stale-cache check, and
    # the concat/dedup step. Returns an Iterator so the caller can stream
    # results as each scene completes.
    if per_scene:
        return _collect_per_scene(
            points=points,
            items=items,
            bbox_wgs84=bbox_wgs84,
            out_dir=out_dir,
            cache_dir=cache_dir,
            apply_nbar=apply_nbar,
            max_concurrent=max_concurrent,
            n_workers=n_workers,
            phases=phases,
            utm_crs=utm_crs,
        )

    # --- Canonical tile assignment -------------------------------------------
    # Adjacent MGRS tiles have non-aligned 10 m pixel grids. A point sampled
    # from tile A vs tile B gets a different sub-pixel position, producing
    # spurious spectral variation. Assign each point one canonical tile (the
    # one whose grid it is natively aligned to) and suppress observations from
    col_order = (
        ["point_id", "lon", "lat", "date", "item_id", "tile_id"]
        + list(BANDS)
        + ["scl_purity", "scl", "aot", "view_zenith", "sun_zenith"]
    )

    # --- 3+4. Plan shards, then fetch only if needed -------------------------
    # Shard planning uses only the points list and item count — no patches needed.
    # If all shards are already complete we skip fetch_patches entirely.
    shard_row_budget = 200_000_000

    rc_to_points: dict[str, list[tuple[str, float, float]]] = {}
    for pid, lon, lat in points:
        rc = pid.split("_")[1]
        rc_to_points.setdefault(rc, []).append((pid, lon, lat))
    sorted_rcs = sorted(rc_to_points.keys())

    rows_per_point = len(items)
    shards: list[list[tuple[str, float, float]]] = []
    current_shard: list[tuple[str, float, float]] = []
    current_shard_rows = 0
    for rc in sorted_rcs:
        rc_points = rc_to_points[rc]
        rc_rows = len(rc_points) * rows_per_point
        if current_shard and current_shard_rows + rc_rows > shard_row_budget:
            shards.append(current_shard)
            current_shard = []
            current_shard_rows = 0
        current_shard.extend(rc_points)
        current_shard_rows += rc_rows
    if current_shard:
        shards.append(current_shard)

    n_shards = len(shards)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _shard_path(idx: int) -> Path:
        return out_dir / f"_collect_shard{idx:03d}.parquet"

    def _shard_complete(idx: int) -> bool:
        sp = _shard_path(idx)
        dp = sp.with_suffix(".done")
        if not dp.exists():
            return False
        return "__done__" in dp.read_text()

    # Pre-check: verify that cached spectral patches were fetched for the current bbox.
    # Patches written for a different bbox (e.g. a training region on the same tile)
    # silently corrupt the output — CachedNpzChipStore clips out-of-bounds coordinates
    # to the patch edge, making every pixel return the same value.
    # If stale patches are found, invalidate shard .done files so shards are rebuilt.
    # Note: border-tile patches (e.g. 55KBB for a site that spans 55KBB+55KCB) are NOT
    # stale — they were fetched for this bbox and OOB points are handled as NaN.
    from utils.fetch import _cache_path as _fc_path, _load_patch_cache as _lpc, _patch_covers_bbox as _pcb
    # Scan ALL item dirs in the cache, not just those matching current STAC item IDs.
    # Stale dirs may have been written by a previous fetch with a different date window
    # or a different region's bbox — their item IDs won't appear in the current STAC
    # list, so checking only current items silently misses them.
    _stale_ids: set[str] = set()
    if cache_dir.is_dir():
        for _item_dir in cache_dir.iterdir():
            if not _item_dir.is_dir():
                continue
            # Check ALL band patches — a single valid band is not sufficient;
            # different bands may have been cached at different bbox extents.
            _npzs = list(_item_dir.glob("*.npz"))
            if not _npzs:
                continue
            for _npz in _npzs:
                _data = _lpc(_npz)
                if _data is None or not _pcb(_data, bbox_wgs84, path=_npz):
                    _stale_ids.add(_item_dir.name)
                    break
    if _stale_ids:
        logger.warning(
            "Stale chip cache: %d items have patches that don't cover bbox %s "
            "— deleting stale .npz files, shard .done files, and tile parquets",
            len(_stale_ids), bbox_wgs84,
        )
        # Delete the stale .npz files so uncached_items picks them up for re-fetch.
        # fetch_patches would re-fetch if called, but it only gets called for items
        # missing from cache — files that exist but cover the wrong bbox are never
        # passed to fetch_patches without this deletion.
        _n_npz_deleted = 0
        for _sid in _stale_ids:
            _item_dir = cache_dir / _sid
            if _item_dir.is_dir():
                for _npz in _item_dir.glob("*.npz"):
                    _npz.unlink()
                    _n_npz_deleted += 1
        if _n_npz_deleted:
            logger.info("  Deleted %d stale .npz patch files", _n_npz_deleted)
        for _i in range(n_shards):
            _dp = _shard_path(_i).with_suffix(".done")
            if _dp.exists():
                _dp.unlink()
                logger.info("  Deleted stale done file: %s", _dp.name)
        # Also delete any tile parquets built from the stale data
        for _tp in out_dir.iterdir():
            if _tp.name.endswith(".s2.parquet") and not _tp.stem.startswith("_"):
                _tp.unlink()
                logger.info("  Deleted stale tile parquet: %s", _tp.name)

    all_shards_done = all(_shard_complete(i) for i in range(n_shards))

    if "fetch" in _phases:
        if all_shards_done:
            logger.info("All %d shards already complete — skipping fetch", n_shards)
        else:
            logger.info(
                "%d row-coords → %d shards (~%d coords/shard, budget %dM rows/shard)",
                len(sorted_rcs), n_shards,
                len(sorted_rcs) // n_shards if n_shards else 0,
                shard_row_budget // 1_000_000,
            )
            # Check how many items are already fully cached on disk.
            from utils.fetch import _cache_path
            import os as _os
            cached_keys: set[tuple[str, str]] = set()
            try:
                for _entry in _os.scandir(cache_dir):
                    if _entry.is_dir():
                        for _f in _os.scandir(_entry.path):
                            if _f.name.endswith(".npz"):
                                cached_keys.add((_entry.name, _f.name[:-4]))
            except FileNotFoundError:
                pass
            uncached_items = [
                item for item in items
                if any((item.id, band) not in cached_keys for band in FETCH_BANDS)
            ]
            if uncached_items:
                logger.info(
                    "Fetching %d/%d items not yet in cache for bbox %s",
                    len(uncached_items), len(items), bbox_wgs84,
                )
                asyncio.run(fetch_patches(
                    points=points,
                    items=uncached_items,
                    bands=FETCH_BANDS,
                    bbox_wgs84=bbox_wgs84,
                    scl_filter=True,
                    band_alias=BAND_ALIAS,
                    max_concurrent=max_concurrent,
                    cache_dir=cache_dir,
                ))
            else:
                logger.info("All %d items already in cache — no network fetch needed", len(items))

    if "extract" not in _phases:
        logger.info("collect: fetch-only phase complete for %s %s/%s", bbox_wgs84, start, end)
        return []

    from utils.parquet_utils import sort_parquet_by_pixel, _optimise_schema, _WRITE_OPTS

    sorted_shard_paths: list[Path] = []

    for shard_idx, shard_points in enumerate(shards):
        shard_path = _shard_path(shard_idx)
        done_path = shard_path.with_suffix(".done")

        sorted_sp = shard_path.with_name(shard_path.stem + "_sorted.parquet")
        if _shard_complete(shard_idx):
            logger.info("Shard %d/%d already complete, skipping", shard_idx + 1, n_shards)
            if sorted_sp.exists():
                sorted_shard_paths.append(sorted_sp)
            elif shard_path.exists():
                sorting_tmp = sorted_sp.with_suffix(".sorting.parquet")
                sorting_tmp.unlink(missing_ok=True)
                logger.info("  sorting shard %d/%d → %s ...", shard_idx + 1, n_shards, sorted_sp.name)
                sort_parquet_by_pixel(shard_path, sorting_tmp, row_group_size=5_000_000, ram_budget_gb=12.0, _skip_dict_rewrite=True)
                sorting_tmp.replace(sorted_sp)
                shard_path.unlink()
                logger.info("  shard %d/%d sorted (raw shard deleted)", shard_idx + 1, n_shards)
                sorted_shard_paths.append(sorted_sp)
            continue

        logger.info(
            "Shard %d/%d: %d points ...", shard_idx + 1, n_shards, len(shard_points)
        )

        shard_point_ids = [pid for pid, _, _ in shard_points]
        shard_lons = np.array([lon for _, lon, _ in shard_points], dtype=np.float64)
        shard_lats = np.array([lat for _, _, lat in shard_points], dtype=np.float64)

        shard_point_coords = {pid: (lon, lat) for pid, lon, lat in shard_points}

        # Per-shard checkpoint: item ids already written for this shard
        done_ids: set[str] = set()
        if done_path.exists():
            done_ids = set(done_path.read_text().splitlines())

        pending_items = [item for item in items if item.id not in done_ids]

        writer: pq.ParquetWriter | None = None
        shard_rows = 0
        shard_buf: list[pa.Table] = []
        shard_buf_rows = 0
        SHARD_BUF_SIZE = 500_000

        def _flush_shard_buf() -> None:
            nonlocal writer, shard_buf_rows
            if not shard_buf:
                return
            out = pa.concat_tables(shard_buf)
            shard_buf.clear()
            shard_buf_rows = 0
            if writer is None:
                writer = pq.ParquetWriter(shard_path, out.schema, compression="none")
            writer.write_table(out)

        # --- Concurrent item processing --------------------------------------
        # n_workers threads each process one item at a time: load .npz from
        # disk, fetch granule XML (HTTP), run c-factor, build DataFrame.
        # Disk reads and HTTP fetches release the GIL so threads genuinely
        # run in parallel.  Results are placed on an output queue in
        # submission order (via a per-item Future) so the writer stays serial.
        #
        # Memory bound: at most n_workers items' patches live in RAM at once.
        #
        # One CachedNpzChipStore per thread — stores cache the (lon,lat)→(row,col)
        # projection for each tile CRS, which is expensive at 2M points.  Creating
        # a new store per item call throws that cache away every call; thread-local
        # stores reuse projections across all items a given worker processes.
        import threading as _threading
        _thread_local = _threading.local()

        def _get_thread_store() -> CachedNpzChipStore:
            if not hasattr(_thread_local, "store"):
                _thread_local.store = CachedNpzChipStore(
                    cache_dir=cache_dir,
                    point_coords=shard_point_coords,
                    bands=FETCH_BANDS,
                )
            return _thread_local.store

        def _process_item(_item) -> tuple | None:
            """Run in a worker thread. Returns (item_id, df | None)."""
            store = _get_thread_store()
            df = extract_item_to_df(
                _item, store, shard_point_ids, shard_lons, shard_lats,
                apply_nbar=apply_nbar, utm_crs=utm_crs,
            )
            store.release_item(_item.id)
            return (_item.id, df)

        # Submit items in a sliding window of at most N_WORKERS in-flight at once.
        # Submitting all items upfront would leave completed DataFrames queued in
        # memory until the writer drains them — fatal for large shards.
        from concurrent.futures import wait, FIRST_COMPLETED
        with done_path.open("a") as done_fh, ThreadPoolExecutor(max_workers=n_workers) as pool:
            pending_queue = list(pending_items)
            in_flight: dict = {}  # future → item
            i = 0

            while pending_queue or in_flight:
                # Fill the window up to n_workers
                while pending_queue and len(in_flight) < n_workers:
                    item = pending_queue.pop(0)
                    in_flight[pool.submit(_process_item, item)] = item

                # Wait for at least one to finish
                done_futs, _ = wait(in_flight, return_when=FIRST_COMPLETED)

                for fut in done_futs:
                    in_flight.pop(fut)
                    item_id, batch_df = fut.result()

                    if batch_df is not None and len(batch_df) > 0:
                        table = batch_df.select(col_order).to_arrow()
                        shard_buf.append(table)
                        shard_buf_rows += len(table)
                        shard_rows += len(table)
                        if shard_buf_rows >= SHARD_BUF_SIZE:
                            _flush_shard_buf()

                    done_ids.add(item_id)
                    done_fh.write(item_id + "\n")
                    done_fh.flush()

                    i += 1
                    if i % 50 == 0 or i == len(pending_items):
                        logger.info(
                            "  S2 scenes  shard %d/%d  item %d/%d  %d rows  workers %d/%d",
                            shard_idx + 1, n_shards, i, len(pending_items), shard_rows,
                            len(in_flight), n_workers,
                        )

        _flush_shard_buf()
        if writer is not None:
            writer.close()
        # Mark shard complete with a sentinel line
        with done_path.open("a") as done_fh:
            done_fh.write("__done__\n")
        logger.info("  shard %d/%d complete: %d rows", shard_idx + 1, n_shards, shard_rows)

        # Sort immediately — overlaps fetch IO of the next shard with sort CPU.
        # Write to a .sorting temp file and rename atomically so an interrupted
        # sort never leaves a partial file that looks complete on restart.
        if sorted_sp.exists():
            logger.info(
                "  shard %d/%d sorted file already exists, reusing: %s",
                shard_idx + 1, n_shards, sorted_sp.name,
            )
        elif shard_path.exists():
            sorting_tmp = sorted_sp.with_suffix(".sorting.parquet")
            sorting_tmp.unlink(missing_ok=True)
            logger.info("  sorting shard %d/%d → %s ...", shard_idx + 1, n_shards, sorted_sp.name)
            sort_parquet_by_pixel(shard_path, sorting_tmp, row_group_size=5_000_000, ram_budget_gb=12.0, _skip_dict_rewrite=True)
            sorting_tmp.replace(sorted_sp)
            shard_path.unlink()
            logger.info("  shard %d/%d sorted (raw shard deleted)", shard_idx + 1, n_shards)
        else:
            logger.info("  shard %d/%d produced no rows, skipping", shard_idx + 1, n_shards)
        if sorted_sp.exists():
            sorted_shard_paths.append(sorted_sp)

    # --- 5. Concat + dedup in a single streaming pass, writing per-tile -------
    # Rows are sorted by point_id. Canonical tile assignment ensures each
    # point_id belongs to exactly one tile, so all rows for a given tile are
    # contiguous in the sorted stream. We maintain per-tile writers and open a
    # new writer whenever tile_id changes.
    #
    # Tile-boundary duplicates: a pixel near an MGRS boundary may appear in two
    # tiles on the same date. sort_parquet_by_pixel sorts by
    # (point_id, date, scl_purity desc) so the best row comes first; a simple
    # shift-compare drops duplicates in O(n).
    #
    # Sorted shards are deleted only after all output files are verified.

    from utils.tile_harmonisation import calibrate, load_corrections

    if not sorted_shard_paths:
        if all_shards_done:
            existing = sorted(
                p for p in out_dir.iterdir()
                if p.name.endswith(".s2.parquet")
                and not p.stem.startswith("_")
                and "_tmp" not in p.stem
            )
            if existing:
                logger.info(
                    "All shards already complete and tile outputs exist — returning %d existing tile(s).",
                    len(existing),
                )
                return existing
            logger.warning("All shards completed but produced no rows — returning empty result.")
            return []
        raise FetchError("No data collected: all shards are empty. Check STAC availability and date range.")

    _corrections: dict | None = None
    if calibration_out is not None:
        calibrate(sorted_shard_paths, calibration_out)
        _corrections = load_corrections(calibration_out)

    logger.info("Concatenating %d sorted shards → per-tile parquets in %s ...", n_shards, out_dir)

    # tile_id → ParquetWriter
    tile_writers: dict[str, pq.ParquetWriter] = {}
    tile_rows: dict[str, int] = {}
    total_rows = 0
    n_dedup_dropped = 0

    concat_rg_size = 5_000_000
    # Per-tile write buffers: tile_id → list[pa.Table]
    write_bufs: dict[str, list[pa.Table]] = {}
    write_buf_rows: dict[str, int] = {}
    total_rgs = sum(
        pq.ParquetFile(sp).metadata.num_row_groups for sp in sorted_shard_paths
    )
    rgs_done = 0

    _CORRECT_BANDS = ["B04", "B05", "B07", "B08", "B11"]

    import polars as _pl
    _corr_df = (
        _pl.DataFrame(
            [(t, b, y, s) for (t, b, y), s in _corrections.items()],
            schema={"tile_id": _pl.String, "band": _pl.String, "year": _pl.Int32, "scale": _pl.Float32},
            orient="row",
        ) if _corrections else None
    )

    def _apply_corrections(tbl: pa.Table) -> pa.Table:
        chunk = _pl.from_arrow(tbl).with_columns(
            _pl.col("date").dt.year().cast(_pl.Int32).alias("year")
        )
        for band in _CORRECT_BANDS:
            if band not in chunk.columns:
                continue
            band_corr = _corr_df.filter(_pl.col("band") == band).select(["tile_id", "year", "scale"])
            chunk = (
                chunk
                .join(band_corr, on=["tile_id", "year"], how="left")
                .with_columns(
                    _pl.when(_pl.col("scale").is_not_null())
                      .then(_pl.col(band) * _pl.col("scale"))
                      .otherwise(_pl.col(band))
                      .alias(band)
                )
                .drop("scale")
            )
        chunk = chunk.drop("year").with_columns([
            ((_pl.col("B08") - _pl.col("B04")) / (_pl.col("B08") + _pl.col("B04"))).alias("NDVI"),
            ((_pl.col("B03") - _pl.col("B08")) / (_pl.col("B03") + _pl.col("B08"))).alias("NDWI"),
            (2.5 * (_pl.col("B08") - _pl.col("B04")) /
             (_pl.col("B08") + 6.0 * _pl.col("B04") - 7.5 * _pl.col("B02") + 1.0)).alias("EVI"),
        ])
        return chunk.to_arrow()

    def _flush_tile(tid: str) -> int:
        buf = write_bufs.get(tid)
        if not buf:
            return 0
        out = pa.concat_tables(buf)
        write_bufs[tid] = []
        write_buf_rows[tid] = 0
        if _corrections:
            out = _apply_corrections(out)
        out = _optimise_schema(out)
        if tid not in tile_writers:
            tile_path = out_dir / f"{tid}.s2.parquet"
            tile_writers[tid] = pq.ParquetWriter(str(tile_path), out.schema, **_WRITE_OPTS)
            tile_rows[tid] = 0
        tile_writers[tid].write_table(out)
        n = len(out)
        tile_rows[tid] += n
        return n

    def _dedup_rg(tbl: pa.Table) -> pa.Table:
        """Drop tile-boundary duplicate (point_id, date) rows."""
        nonlocal n_dedup_dropped
        import pyarrow.compute as pc
        n_in = len(tbl)
        pid_col  = tbl.column("point_id").combine_chunks()
        date_col = tbl.column("date").combine_chunks()
        pid_prev  = pa.concat_arrays([pid_col[:1],  pid_col[:-1]])
        date_prev = pa.concat_arrays([date_col[:1], date_col[:-1]])
        keep = pc.or_(pc.not_equal(pid_col, pid_prev), pc.not_equal(date_col, date_prev))
        n_keep = pc.sum(keep).as_py()
        if n_keep < n_in:
            n_dedup_dropped += n_in - n_keep
            return tbl.filter(keep)
        return tbl

    import pyarrow.compute as pc

    try:
        for sorted_sp in sorted_shard_paths:
            pf = pq.ParquetFile(sorted_sp)
            n_rg = pf.metadata.num_row_groups
            for rg_idx in range(n_rg):
                rg = pf.read_row_group(rg_idx)
                rg = _dedup_rg(rg)
                if len(rg) == 0:
                    rgs_done += 1
                    continue
                # Split row group by tile_id and route to per-tile buffers.
                tile_col = rg.column("tile_id").combine_chunks()
                unique_tiles = pc.unique(tile_col).to_pylist()
                for tid in unique_tiles:
                    if not tid:
                        n_bad = pc.sum(pc.equal(tile_col, tid)).as_py()
                        logger.warning("Skipping %d rows with empty tile_id in shard", n_bad)
                        continue
                    mask = pc.equal(tile_col, tid)
                    subset = rg.filter(mask)
                    write_bufs.setdefault(tid, []).append(subset)
                    write_buf_rows[tid] = write_buf_rows.get(tid, 0) + len(subset)
                    if write_buf_rows[tid] >= concat_rg_size:
                        total_rows += _flush_tile(tid)
                rgs_done += 1
                if rgs_done % 50 == 0 or rgs_done == total_rgs:
                    logger.info(
                        "  S2 merge  concat %d/%d row groups (%.0f%%)  %d rows written",
                        rgs_done, total_rgs, 100 * rgs_done / total_rgs, total_rows,
                    )
            del pf

        for tid in list(write_bufs.keys()):
            total_rows += _flush_tile(tid)

        for writer in tile_writers.values():
            writer.close()

    except Exception:
        for writer in tile_writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for tid in tile_writers:
            p = out_dir / f"{tid}.s2.parquet"
            p.unlink(missing_ok=True)
        raise

    if n_dedup_dropped:
        logger.info("Cross-tile dedup: removed %d boundary duplicate rows", n_dedup_dropped)

    # Verify row count before deleting sorted intermediates.
    if total_rows == 0:
        raise FetchError("No usable observations — all pixels clouded or missing?")

    written_rows = sum(
        pq.ParquetFile(out_dir / f"{tid}.s2.parquet").metadata.num_rows
        for tid in tile_writers
    )
    if written_rows != total_rows:
        raise FetchError(
            f"Row count mismatch: counted {total_rows} but parquet reports {written_rows} — output may be corrupt"
        )

    # Only delete sorted intermediates after successful verification.
    for sorted_sp in sorted_shard_paths:
        sorted_sp.unlink(missing_ok=True)

    written_paths = [out_dir / f"{tid}.s2.parquet" for tid in sorted(tile_writers)]
    for p in written_paths:
        n = pq.ParquetFile(p).metadata.num_rows
        logger.info("Written: %s  (%d rows)", p.name, n)

    from utils.parquet_utils import is_pixel_sorted
    for p in written_paths:
        if not is_pixel_sorted(p, n_check=10):
            logger.warning("%s is NOT pixel-sorted — sorting now ...", p.name)
            sorting_tmp = p.with_name(p.stem + ".sorting.parquet")
            sorting_tmp.unlink(missing_ok=True)
            sort_parquet_by_pixel(p, sorting_tmp, row_group_size=5_000_000, ram_budget_gb=12.0, _skip_dict_rewrite=True)
            sorting_tmp.replace(p)
            logger.info("Pixel sort complete: %s", p.name)

    print(f"\nDone.")
    print(f"  Rows   : {total_rows}")
    print(f"  Tiles  : {len(written_paths)}")
    for p in written_paths:
        print(f"  Output : {p}")

    return written_paths
