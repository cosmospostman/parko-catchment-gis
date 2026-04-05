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

# Generic usage with explicit working dir for chip cache:
python scripts/collect_pixel_observations.py \\
    --bbox LON_MIN,LAT_MIN,LON_MAX,LAT_MAX \\
    --start YYYY-MM-DD --end YYYY-MM-DD \\
    --out path/to/output.parquet \\
    --chips-dir path/to/chip/cache \\
    --cloud-max 30

Notes
-----
- Chip files are cached under --chips-dir and are never re-downloaded.
  Re-running the script is safe and fast if chips already exist.
- Points are placed on a 10 m UTM grid aligned to the S2 pixel grid,
  one point per pixel inside the bbox.
- The window_px=1 chip fetch reads the single centre pixel only — no
  spatial averaging. This is appropriate for a 10 m pixel grid where
  each point corresponds to exactly one S2 pixel.
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
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import BANDS, SCL_BAND, AOT_BAND, VZA_BAND, SZA_BAND
from analysis.timeseries.extraction import extract_observations
from stage0.chip_store import DiskChipStore
from stage0.fetch import fetch_chips
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
) -> list[tuple[str, float, float]]:
    """Generate one point per S2 pixel inside bbox_wgs84, aligned to a 10 m UTM grid.

    The grid origin is snapped to the nearest 10 m multiple so that points
    fall at S2 pixel centres rather than between pixels.

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

    xs = np.arange(x0_snap, x1, r)
    ys = np.arange(y0_snap, y1, r)

    points: list[tuple[str, float, float]] = []
    for i, xi in enumerate(xs):
        for j, yj in enumerate(ys):
            lon, lat = to_wgs.transform(xi, yj)
            pid = f"px_{i:04d}_{j:04d}"
            points.append((pid, float(lon), float(lat)))

    logger.info(
        "Pixel grid: %d × %d = %d points at %.0f m spacing",
        len(xs), len(ys), len(points), r,
    )
    return points


# ---------------------------------------------------------------------------
# Observation → DataFrame
# ---------------------------------------------------------------------------

def observations_to_dataframe(
    observations: list,
    point_coords: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Convert Observation objects to a flat DataFrame."""
    rows = []
    for obs in observations:
        lon, lat = point_coords.get(obs.point_id, (float("nan"), float("nan")))
        row: dict = {
            "point_id":   obs.point_id,
            "lon":        lon,
            "lat":        lat,
            "date":       obs.date.date(),
            "item_id":    obs.meta.get("item_id", ""),
            "tile_id":    obs.meta.get("tile_id", ""),
            "scl_purity": obs.quality.scl_purity,
            "aot":        obs.quality.aot,
            "view_zenith": obs.quality.view_zenith,
            "sun_zenith":  obs.quality.sun_zenith,
        }
        for band in BANDS:
            row[band] = obs.bands.get(band, float("nan"))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    col_order = (
        ["point_id", "lon", "lat", "date", "item_id", "tile_id"]
        + list(BANDS)
        + ["scl_purity", "aot", "view_zenith", "sun_zenith"]
    )
    df = pd.DataFrame(rows)[col_order]
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect(
    bbox_wgs84: list[float],
    start: str,
    end: str,
    out_path: Path,
    chips_dir: Path,
    cloud_max: int,
) -> None:
    # --- 1. Generate pixel grid -------------------------------------------
    points = make_pixel_grid(bbox_wgs84)
    point_coords = {pid: (lon, lat) for pid, lon, lat in points}

    # --- 2. STAC search ------------------------------------------------------
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
    logger.info("%d items found", len(items))

    # --- 3. Fetch chips (idempotent) ----------------------------------------
    chips_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching chips → %s", chips_dir)
    asyncio.run(fetch_chips(
        points=points,
        items=items,
        bands=FETCH_BANDS,
        window_px=1,          # single-pixel extraction — one point per S2 pixel
        inputs_dir=chips_dir,
        scl_filter=True,
        band_alias=BAND_ALIAS,
    ))

    # --- 4. Extract observations --------------------------------------------
    store = DiskChipStore(chips_dir)
    logger.info("Extracting observations ...")
    observations = extract_observations(items, points, store, bands=BANDS)
    logger.info("%d observations extracted", len(observations))

    if not observations:
        logger.error("No usable observations — all pixels clouded or missing?")
        sys.exit(1)

    # --- 5. Build DataFrame and write Parquet --------------------------------
    df = observations_to_dataframe(observations, point_coords)

    # Drop rows where all spectral bands are NaN
    band_cols = list(BANDS)
    df = df.dropna(subset=band_cols, how="all")
    logger.info("%d rows after dropping all-NaN band rows", len(df))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Written: %s  (%d rows × %d cols)", out_path, len(df), len(df.columns))

    # Summary
    n_points   = df["point_id"].nunique()
    n_dates    = df["date"].nunique()
    date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"
    print(f"\nDone.")
    print(f"  Points : {n_points}")
    print(f"  Dates  : {n_dates}  ({date_range})")
    print(f"  Rows   : {len(df)}")
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
        "--chips-dir", type=Path, default=None,
        help="Directory for cached chip files (default: <out>.chips/ next to output)",
    )
    p.add_argument(
        "--cloud-max", type=int, default=30,
        help="Maximum scene cloud cover %% (default: 30)",
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

    chips_dir = args.chips_dir or args.out.with_suffix("").parent / (args.out.stem + ".chips")

    collect(
        bbox_wgs84=bbox,
        start=args.start,
        end=args.end,
        out_path=args.out,
        chips_dir=chips_dir,
        cloud_max=args.cloud_max,
    )


if __name__ == "__main__":
    main()
