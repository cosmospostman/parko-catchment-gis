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
aot           : float — inverse aerosol optical thickness  (1 = clean air)
view_zenith   : float — inverse view zenith angle          (1 = nadir)
sun_zenith    : float — inverse sun zenith angle           (1 = high sun)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS, SCL_BAND, AOT_BAND, SCL_CLEAR_VALUES
from utils.chip_store import MemoryChipStore
from utils.fetch import fetch_patches
from utils.stac import search_sentinel2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION  = "sentinel-2-l2a"
UTM_CRS        = "EPSG:32755"   # WGS 84 / UTM zone 55S — covers eastern Australia

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
    utm_crs: str = UTM_CRS,
    resolution_m: float = 10.0,
    stride: int = 1,
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
            pid = f"px_{i:04d}_{j:04d}"
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
        "aot":        aot_quality[idx],
        "view_zenith": np.ones(len(idx), dtype=np.float32),  # not available at earth-search
        "sun_zenith":  np.ones(len(idx), dtype=np.float32),
        **{band: band_arrays[band][idx] for band in BANDS},
    })

    # Drop rows where all spectral bands are NaN
    if df[list(BANDS)].isna().all(axis=1).all():
        return None

    return df


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
) -> None:
    # --- 1. Generate pixel grid -------------------------------------------
    points = make_pixel_grid(bbox_wgs84, stride=stride)
    point_coords = {pid: (lon, lat) for pid, lon, lat in points}

    # --- 2. STAC search (cached) ---------------------------------------------
    import hashlib, pickle
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "data" / "chips" / (out_path.stem + ".chips")
    cache_dir.mkdir(parents=True, exist_ok=True)
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

    # --- 3. Fetch patches (one range request per item×band, not per point) --
    logger.info("Fetching patches for bbox %s", bbox_wgs84)
    patches = asyncio.run(fetch_patches(
        points=points,
        items=items,
        bands=FETCH_BANDS,
        bbox_wgs84=bbox_wgs84,
        scl_filter=True,
        band_alias=BAND_ALIAS,
        max_concurrent=32,
        cache_dir=cache_dir,
    ))
    logger.info("Fetched %d (item, band) patches", len(patches))

    # --- 4. Extract observations and write Parquet in item-batches -----------
    # Vectorised: fetch entire band arrays per item via get_all_points(),
    # apply SCL mask, build DataFrame directly — no per-point Python loops.
    store = MemoryChipStore(patches, point_coords)
    point_ids = [pid for pid, _, _ in points]
    lons_arr = np.array([lon for _, lon, _ in points], dtype=np.float64)
    lats_arr = np.array([lat for _, _, lat in points], dtype=np.float64)

    col_order = (
        ["point_id", "lon", "lat", "date", "item_id", "tile_id"]
        + list(BANDS)
        + ["scl_purity", "aot", "view_zenith", "sun_zenith"]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_path = out_path.with_suffix(".done")
    partial_path = out_path.with_suffix(".partial.parquet")

    # Load checkpoint: set of item_ids already written
    done_ids: set[str] = set()
    resuming = done_path.exists() and out_path.exists()
    if resuming:
        done_ids = set(done_path.read_text().splitlines())
        existing_rows = pq.read_metadata(out_path).num_rows
        logger.info(
            "Checkpoint: %d items already done (%d rows), resuming",
            len(done_ids), existing_rows,
        )

    writer: pq.ParquetWriter | None = None
    new_rows = 0

    logger.info("Extracting observations (vectorised, %d items) ...", len(items))
    for i, item in enumerate(items):
        if item.id in done_ids:
            store.release_item(item.id)
            continue

        batch_df = extract_item_to_df(item, store, point_ids, lons_arr, lats_arr)
        store.release_item(item.id)

        if batch_df is not None and not batch_df.empty:
            table = pa.Table.from_pandas(batch_df[col_order], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(partial_path, table.schema)
            writer.write_table(table)
            new_rows += len(batch_df)

        # Mark item done and flush checkpoint after every item
        done_ids.add(item.id)
        done_path.write_text("\n".join(done_ids))

        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            logger.info("  %d/%d items processed, %d new rows", i + 1, len(items), new_rows)

    if writer is not None:
        writer.close()

    # Merge partial into final output
    if resuming and partial_path.exists():
        logger.info("Merging checkpoint parquet with new rows (%d) ...", new_rows)
        merged = pa.concat_tables([
            pq.read_table(out_path),
            pq.read_table(partial_path),
        ])
        pq.write_table(merged, out_path)
        partial_path.unlink()
        total_rows = len(merged)
    elif partial_path.exists():
        partial_path.rename(out_path)
        total_rows = new_rows
    else:
        # Resumed and all items were already done
        total_rows = pq.read_metadata(out_path).num_rows if out_path.exists() else 0

    if total_rows == 0:
        logger.error("No usable observations — all pixels clouded or missing?")
        sys.exit(1)

    # Remove checkpoint file on clean completion
    done_path.unlink(missing_ok=True)

    logger.info("Written: %s  (%d rows)", out_path, total_rows)
    print(f"\nDone.")
    print(f"  Rows   : {total_rows}")
    print(f"  Output : {out_path}")
