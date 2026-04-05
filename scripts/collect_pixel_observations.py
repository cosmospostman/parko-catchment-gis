"""scripts/collect_pixel_observations.py — collect all S2 observations for a bbox.

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

Usage
-----
# Longreach high-density infestation patch, full 2020–2025 archive:
python scripts/collect_pixel_observations.py \\
    --bbox 145.4240,-22.7640,145.4250,-22.7610 \\
    --start 2020-01-01 --end 2025-12-31 \\
    --out data/longreach_pixels.parquet

# Generic usage:
python scripts/collect_pixel_observations.py \\
    --bbox LON_MIN,LAT_MIN,LON_MAX,LAT_MAX \\
    --start YYYY-MM-DD --end YYYY-MM-DD \\
    --out path/to/output.parquet \\
    --cloud-max 30

Notes
-----
- One bbox-covering patch is fetched per (item, band) — not per point.
  All points in the bbox are sliced from the same patch in memory, so
  network requests scale with items×bands, not items×bands×points.
- No chip files are written to disk. Each run re-fetches from the network.
- Points are placed on a 10 m UTM grid aligned to the S2 pixel grid,
  one point per pixel inside the bbox.
- Rows with no spectral bands (all NaN) are dropped before writing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS, SCL_BAND, AOT_BAND, SCL_CLEAR_VALUES
from stage0.chip_store import MemoryChipStore
from stage0.fetch import fetch_patches
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
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    logger.info("Extracting observations (vectorised, %d items) ...", len(items))
    for i, item in enumerate(items):
        batch_df = extract_item_to_df(item, store, point_ids, lons_arr, lats_arr)
        store.release_item(item.id)

        if batch_df is None or batch_df.empty:
            continue

        table = pa.Table.from_pandas(batch_df[col_order], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        total_rows += len(batch_df)

        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            logger.info("  %d/%d items processed, %d rows written", i + 1, len(items), total_rows)

    if writer is not None:
        writer.close()
    else:
        logger.error("No usable observations — all pixels clouded or missing?")
        sys.exit(1)

    logger.info("Written: %s  (%d rows)", out_path, total_rows)
    print(f"\nDone.")
    print(f"  Rows   : {total_rows}")
    print(f"  Output : {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect all S2 observations for a bbox into a Parquet file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--bbox", required=True,
        help="Bounding box as 'lon_min,lat_min,lon_max,lat_max' (EPSG:4326). "
             "Example: 145.4213,-22.7671,145.4287,-22.7597",
    )
    p.add_argument(
        "--start", default="2020-01-01",
        help="Start date YYYY-MM-DD (default: 2020-01-01)",
    )
    p.add_argument(
        "--end", default=date.today().isoformat(),
        help="End date YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Output Parquet file path, e.g. data/longreach_pixels.parquet",
    )
    p.add_argument(
        "--cloud-max", type=int, default=30,
        help="Maximum scene cloud cover %% (default: 30)",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory to cache fetched patches as .npz files. Re-runs skip "
             "already-cached patches. Default: <out>.cache/ next to output.",
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help="Keep every Nth pixel in x and y (default: 1 = all pixels). "
             "stride=3 gives ~30 m spacing and reduces point count by ~9×. "
             "Applied at grid generation, so fewer patches are fetched too.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("rasterio.session").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

    args = _parse_args()

    bbox = [float(x) for x in args.bbox.split(",")]
    if len(bbox) != 4:
        print("ERROR: --bbox must be 'lon_min,lat_min,lon_max,lat_max'", file=sys.stderr)
        sys.exit(1)

    cache_dir = args.cache_dir or args.out.parent / (args.out.stem + ".cache")

    collect(
        bbox_wgs84=bbox,
        start=args.start,
        end=args.end,
        out_path=args.out,
        cloud_max=args.cloud_max,
        cache_dir=cache_dir,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
