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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS, SCL_BAND, AOT_BAND, SCL_CLEAR_VALUES, add_spectral_indices, SPECTRAL_INDEX_COLS
from utils.chip_store import CachedNpzChipStore, MemoryChipStore
from utils.fetch import fetch_patches
from utils.stac import search_sentinel2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION  = "sentinel-2-l2a"

_TILE_ID_RE = re.compile(r"^S2[AB]_(\d{2}[A-Z]{3})_")


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

def extract_item_to_df(
    item,
    store: MemoryChipStore,
    point_ids: list[str],
    lons: np.ndarray,
    lats: np.ndarray,
    apply_nbar: bool = True,
    utm_crs: str = "EPSG:32755",
) -> pd.DataFrame | None:
    """Extract all usable pixels for one STAC item into a DataFrame.

    Uses store.get_all_points() to fetch entire bands as numpy arrays,
    avoiding per-point Python loops. Returns None if the item has no
    clear pixels at all (cloud-filtered).
    """
    item_id = item.id
    m = _TILE_ID_RE.match(item_id)
    tile_id = m.group(1) if m else item.properties.get("s2:mgrs_tile", "")
    item_date = pd.Timestamp(item.datetime.replace(tzinfo=None))
    n = len(point_ids)

    # --- SCL: per-point clear-pixel mask ------------------------------------
    scl_vals = store.get_all_points(item_id, SCL_BAND)
    if scl_vals is None:
        return None
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
                band_arrays[band] = np.clip(band_arrays[band] * cf, 0.0, 1.0)
        # If angles is None (fetch failed), band_arrays are left uncorrected

    # --- Zenith quality columns ---------------------------------------------
    if angles is not None:
        sza_mean = np.mean(
            [angles[b]["sza"] for b in BANDS if b in angles], axis=0
        ).astype(np.float32)
        vza_mean = np.mean(
            [angles[b]["vza"] for b in BANDS if b in angles], axis=0
        ).astype(np.float32)
        sun_zenith_col  = np.clip(1.0 - sza_mean / 90.0, 0.0, 1.0)
        view_zenith_col = np.clip(1.0 - vza_mean / 90.0, 0.0, 1.0)
    else:
        sun_zenith_col  = np.ones(n, dtype=np.float32)
        view_zenith_col = np.ones(n, dtype=np.float32)

    # --- Filter to clear pixels only ----------------------------------------
    idx = np.where(clear_mask)[0]
    df = pd.DataFrame({
        "point_id":   np.array(point_ids)[idx],
        "lon":        lons[idx],
        "lat":        lats[idx],
        "date":       item_date,
        "item_id":    item_id,
        "tile_id":    tile_id,
        "scl_purity": scl_purity[idx],
        "scl":        scl_int[idx].astype(np.int8),
        "aot":        aot_quality[idx],
        "view_zenith": view_zenith_col[idx],
        "sun_zenith":  sun_zenith_col[idx],
        **{band: band_arrays[band][idx] for band in BANDS},
    })

    # Drop rows where all spectral bands are NaN
    if df[list(BANDS)].isna().all(axis=1).all():
        return None

    return add_spectral_indices(df)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

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
) -> list[Path]:
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
    """
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
                logger.error("No STAC items found — check bbox and date range")
                sys.exit(1)
            with stac_cache.open("wb") as fh:
                pickle.dump(items, fh)
            logger.info("%d items found, cached to %s", len(items), stac_cache.name)
        if not items:
            logger.error("No STAC items found — check bbox and date range")
            sys.exit(1)

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

    # --- Canonical tile assignment -------------------------------------------
    # Adjacent MGRS tiles have non-aligned 10 m pixel grids. A point sampled
    # from tile A vs tile B gets a different sub-pixel position, producing
    # spurious spectral variation. Assign each point one canonical tile (the
    # one whose grid it is natively aligned to) and suppress observations from
    col_order = (
        ["point_id", "lon", "lat", "date", "item_id", "tile_id"]
        + list(BANDS)
        + ["scl_purity", "scl", "aot", "view_zenith", "sun_zenith"]
        + SPECTRAL_INDEX_COLS
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

    all_shards_done = all(_shard_complete(i) for i in range(n_shards))

    if all_shards_done:
        logger.info("All %d shards already complete — skipping fetch", n_shards)
    else:
        logger.info(
            "%d row-coords → %d shards (~%d coords/shard, budget %dM rows/shard)",
            len(sorted_rcs), n_shards,
            len(sorted_rcs) // n_shards if n_shards else 0,
            shard_row_budget // 1_000_000,
        )
        # Check how many items are already fully cached on disk
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

    from signals._shared import sort_parquet_by_pixel, _optimise_schema, _WRITE_OPTS

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
                sort_parquet_by_pixel(shard_path, sorting_tmp, row_group_size=5_000_000, ram_budget_gb=20.0)
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
        # N_WORKERS threads each process one item at a time: load .npz from
        # disk, fetch granule XML (HTTP), run c-factor, build DataFrame.
        # Disk reads and HTTP fetches release the GIL so threads genuinely
        # run in parallel.  Results are placed on an output queue in
        # submission order (via a per-item Future) so the writer stays serial.
        #
        # Memory bound: at most N_WORKERS items' patches live in RAM at once.

        N_WORKERS = 8

        def _process_item(_item) -> tuple | None:
            """Run in a worker thread. Returns (item_id, df | None)."""
            store = CachedNpzChipStore(
                cache_dir=cache_dir,
                point_coords=shard_point_coords,
                bands=FETCH_BANDS,
            )
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
        with done_path.open("a") as done_fh, ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            pending_queue = list(pending_items)
            in_flight: dict = {}  # future → item
            i = 0

            while pending_queue or in_flight:
                # Fill the window up to N_WORKERS
                while pending_queue and len(in_flight) < N_WORKERS:
                    item = pending_queue.pop(0)
                    in_flight[pool.submit(_process_item, item)] = item

                # Wait for at least one to finish
                done_futs, _ = wait(in_flight, return_when=FIRST_COMPLETED)

                for fut in done_futs:
                    in_flight.pop(fut)
                    item_id, batch_df = fut.result()

                    if batch_df is not None and not batch_df.empty:
                        table = pa.Table.from_pandas(batch_df[col_order], preserve_index=False)
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
                            "  shard %d/%d  item %d/%d  %d rows",
                            shard_idx + 1, n_shards, i, len(pending_items), shard_rows,
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
            sort_parquet_by_pixel(shard_path, sorting_tmp, row_group_size=5_000_000, ram_budget_gb=20.0)
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
            logger.warning("All shards completed but produced no rows — returning empty result.")
            return []
        raise RuntimeError("No data collected: all shards are empty. Check STAC availability and date range.")

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
            tile_path = out_dir / f"{tid}.parquet"
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
                    mask = pc.equal(tile_col, tid)
                    subset = rg.filter(mask)
                    write_bufs.setdefault(tid, []).append(subset)
                    write_buf_rows[tid] = write_buf_rows.get(tid, 0) + len(subset)
                    if write_buf_rows[tid] >= concat_rg_size:
                        total_rows += _flush_tile(tid)
                rgs_done += 1
                if rgs_done % 50 == 0 or rgs_done == total_rgs:
                    logger.info(
                        "  concat %d/%d row groups (%.0f%%)  %d rows written",
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
            p = out_dir / f"{tid}.parquet"
            p.unlink(missing_ok=True)
        raise

    if n_dedup_dropped:
        logger.info("Cross-tile dedup: removed %d boundary duplicate rows", n_dedup_dropped)

    # Verify row count before deleting sorted intermediates.
    if total_rows == 0:
        logger.error("No usable observations — all pixels clouded or missing?")
        sys.exit(1)

    written_rows = sum(
        pq.ParquetFile(out_dir / f"{tid}.parquet").metadata.num_rows
        for tid in tile_writers
    )
    if written_rows != total_rows:
        logger.error(
            "Row count mismatch: counted %d but parquet reports %d — output may be corrupt",
            total_rows, written_rows,
        )
        sys.exit(1)

    # Only delete sorted intermediates after successful verification.
    for sorted_sp in sorted_shard_paths:
        sorted_sp.unlink(missing_ok=True)

    written_paths = [out_dir / f"{tid}.parquet" for tid in sorted(tile_writers)]
    for p in written_paths:
        n = pq.ParquetFile(p).metadata.num_rows
        logger.info("Written: %s  (%d rows)", p.name, n)

    from signals._shared import is_pixel_sorted
    for p in written_paths:
        if not is_pixel_sorted(p, n_check=10):
            logger.warning("%s is NOT pixel-sorted — sorting now ...", p.name)
            sorting_tmp = p.with_name(p.stem + ".sorting.parquet")
            sorting_tmp.unlink(missing_ok=True)
            sort_parquet_by_pixel(p, sorting_tmp, row_group_size=5_000_000, ram_budget_gb=20.0)
            sorting_tmp.replace(p)
            logger.info("Pixel sort complete: %s", p.name)

    print(f"\nDone.")
    print(f"  Rows   : {total_rows}")
    print(f"  Tiles  : {len(written_paths)}")
    for p in written_paths:
        print(f"  Output : {p}")

    return written_paths
