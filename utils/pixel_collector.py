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
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    stride: int = 1,
    point_id_prefix: str = "px",
) -> list[tuple[str, float, float]]:
    """Generate one point per S2 pixel inside bbox_wgs84, aligned to a 10 m UTM grid.

    The grid origin is snapped to the nearest 10 m multiple so that points
    fall at S2 pixel centres rather than between pixels.

    stride > 1 keeps every Nth pixel in both x and y, reducing point count by
    stride² while preserving uniform spatial coverage. Effective resolution
    becomes stride × 10 m (e.g. stride=3 → 30 m spacing).

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

    xs = np.arange(x0_snap, x1, r)[::stride]
    ys = np.arange(y0_snap, y1, r)[::stride]

    points: list[tuple[str, float, float]] = []
    for i, xi in enumerate(xs):
        for j, yj in enumerate(ys):
            lon, lat = to_wgs.transform(xi, yj)
            pid = f"{point_id_prefix}_{i:04d}_{j:04d}"
            points.append((pid, float(lon), float(lat)))

    logger.info(
        "Pixel grid: %d × %d = %d points at %.0f m spacing (stride=%d)",
        len(xs), len(ys), len(points), r * stride, stride,
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
    tile_id = item.properties.get("s2:mgrs_tile", "")
    item_date = pd.Timestamp(item.datetime.replace(tzinfo=None))
    n = len(point_ids)

    # --- SCL: per-point clear-pixel mask ------------------------------------
    scl_vals = store.get_all_points(item_id, SCL_BAND)
    if scl_vals is None:
        return None
    scl_int = scl_vals.astype(np.int32)
    clear_mask = np.zeros(n, dtype=bool)
    for v in SCL_CLEAR_VALUES:
        clear_mask |= (scl_int == v)
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
    out_path: Path,
    cloud_max: int,
    cache_dir: Path | None = None,
    stride: int = 1,
    apply_nbar: bool = True,
    max_concurrent: int = 32,
    items=None,
    point_id_prefix: str = "px",
) -> None:
    """Collect S2 observations for bbox_wgs84.

    If *items* is provided (a pre-fetched, deduplicated STAC item list), the
    STAC search step is skipped entirely.  This lets the caller share one STAC
    search result across multiple collect() calls for the same tile.
    """
    # --- 1. Generate pixel grid -------------------------------------------
    utm_crs = _utm_crs_for_bbox(bbox_wgs84)
    points = make_pixel_grid(bbox_wgs84, utm_crs=utm_crs, stride=stride, point_id_prefix=point_id_prefix)
    point_coords = {pid: (lon, lat) for pid, lon, lat in points}

    import hashlib, pickle
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "data" / "chips" / (out_path.stem + ".chips")
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
        import re as _re
        _granule_re = _re.compile(r"_\d+_L2A$")
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
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _shard_path(idx: int) -> Path:
        return out_path.with_name(f"{out_path.stem}.shard{idx:03d}.parquet")

    all_shards_done = all(
        _shard_path(i).exists() and _shard_path(i).with_suffix(".done").exists()
        for i in range(n_shards)
    )

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
        uncached_items = [
            item for item in items
            if not all(
                _cache_path(cache_dir, item.id, band).exists()
                for band in FETCH_BANDS
            )
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

    shard_paths: list[Path] = []

    for shard_idx, shard_points in enumerate(shards):
        shard_path = _shard_path(shard_idx)
        done_path = shard_path.with_suffix(".done")

        if shard_path.exists() and done_path.exists():
            logger.info("Shard %d/%d already complete, skipping", shard_idx + 1, n_shards)
            shard_paths.append(shard_path)
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
        done_fh = done_path.open("a")

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

        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            # Submit all items; keep futures in order for checkpoint tracking
            futures = {pool.submit(_process_item, item): item for item in pending_items}

            i = 0
            for fut in as_completed(futures):
                item_id, batch_df = fut.result()

                if batch_df is not None and not batch_df.empty:
                    table = pa.Table.from_pandas(batch_df[col_order], preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(shard_path, table.schema)
                    writer.write_table(table)
                    shard_rows += len(batch_df)

                done_ids.add(item_id)
                done_fh.write(item_id + "\n")
                done_fh.flush()

                i += 1
                if i % 50 == 0 or i == len(pending_items):
                    logger.info(
                        "  shard %d/%d  item %d/%d  %d rows",
                        shard_idx + 1, n_shards, i, len(pending_items), shard_rows,
                    )

        done_fh.close()
        if writer is not None:
            writer.close()
        # Mark shard complete with a sentinel line
        done_path.open("a").write("__done__\n")
        if shard_path.exists():
            shard_paths.append(shard_path)
        logger.info("  shard %d/%d complete: %d rows", shard_idx + 1, n_shards, shard_rows)

    # --- 5. Sort each shard by point_id, then concatenate → pixel-sorted output -
    # Each shard covers a disjoint set of row-coords, so the shards are already
    # pixel-disjoint from each other.  Within each shard rows are item-ordered
    # (not pixel-ordered), so we must sort by point_id before concatenating.
    #
    # We use sort_parquet_by_pixel() on each shard in turn (single-pass, bounded
    # RAM) writing to a <stem>_sorted.parquet sibling, then concatenate those
    # sorted intermediates row-group-by-row-group into the final output.
    # Original shard files are preserved.
    from signals._shared import sort_parquet_by_pixel

    sorted_shard_paths: list[Path] = []
    for shard_idx in range(n_shards):
        sp = _shard_path(shard_idx)
        sorted_sp = sp.with_name(sp.stem + "_sorted.parquet")
        if sorted_sp.exists():
            logger.info(
                "  shard %d/%d sorted file already exists, reusing: %s",
                shard_idx + 1, n_shards, sorted_sp.name,
            )
        elif sp.exists():
            logger.info(
                "  sorting shard %d/%d → %s ...", shard_idx + 1, n_shards, sorted_sp.name
            )
            sort_parquet_by_pixel(sp, sorted_sp, row_group_size=5_000_000)
            logger.info("  shard %d/%d sorted", shard_idx + 1, n_shards)
        else:
            logger.warning("  shard %d/%d: neither unsorted nor sorted parquet found, skipping", shard_idx + 1, n_shards)
            continue
        sorted_shard_paths.append(sorted_sp)

    logger.info("Concatenating %d sorted shards → %s ...", n_shards, out_path.name)
    final_writer: pq.ParquetWriter | None = None
    total_rows = 0
    for sorted_sp in sorted_shard_paths:
        pf = pq.ParquetFile(sorted_sp)
        schema = pf.schema_arrow
        n_rg = pf.metadata.num_row_groups
        for rg_idx in range(n_rg):
            tbl = pf.read_row_group(rg_idx)
            if final_writer is None:
                final_writer = pq.ParquetWriter(out_path, schema)
            final_writer.write_table(tbl)
            total_rows += len(tbl)
        del pf

    if final_writer is not None:
        final_writer.close()

    # Clean up sorted intermediates (original shards are kept)
    for sorted_sp in sorted_shard_paths:
        sorted_sp.unlink(missing_ok=True)

    if total_rows == 0:
        logger.error("No usable observations — all pixels clouded or missing?")
        sys.exit(1)

    logger.info("Written: %s  (%d rows)", out_path, total_rows)
    print(f"\nDone.")
    print(f"  Rows   : {total_rows}")
    print(f"  Output : {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Collect Sentinel-2 L2A pixel observations for a named location."
    )
    parser.add_argument("--location", required=True, help="Location name (matches data/locations/*.yaml)")
    parser.add_argument("--start",    required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--cloud-max", type=int, default=80, help="Max cloud cover %% (default 80)")
    parser.add_argument("--stride",    type=int, default=1,  help="Pixel stride (default 1)")
    parser.add_argument("--no-nbar",   action="store_true",  help="Disable BRDF NBAR c-factor correction")
    parser.add_argument("--out",       default=None,         help="Output parquet path (default data/pixels/<location>.parquet)")
    args = parser.parse_args()

    from utils.location import get as get_location

    try:
        loc = get_location(args.location)
    except KeyError:
        print(f"ERROR: unknown location '{args.location}'", file=sys.stderr)
        sys.exit(1)

    loc.fetch(
        out_path=Path(args.out) if args.out else None,
        start=args.start,
        end=args.end,
        cloud_max=args.cloud_max,
        stride=args.stride,
        apply_nbar=not args.no_nbar,
    )
