"""
scripts/build_dea_baseline.py — Download DEA Landsat scenes and build the NDVI baseline median TIF.

Decoupled from the main pipeline: produces only ndvi_baseline_median.tif and caches
the raw per-tile DEA data permanently so the script is resumable and re-runnable.

Outputs:
  {CACHE_DIR}/dea_tiles/tile_{N}.tif   — raw per-tile NDVI median (kept)
  {CACHE_DIR}/ndvi_baseline_median.tif — final baseline COG (used by stage 02)

Usage:
    source config.sh
    python scripts/build_dea_baseline.py --year 2024
    python scripts/build_dea_baseline.py --year 2024 --rebuild   # ignore existing cache
"""
import argparse
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr

DEA_BANDS = ["nbart_red", "nbart_nir"]

BASELINE_TILE_SIZE_PX = int(os.environ.get("BASELINE_TILE_SIZE_PX", "512"))
FETCH_WORKERS         = int(os.environ.get("FETCH_WORKERS",   "4"))
COMPUTE_WORKERS       = int(os.environ.get("COMPUTE_WORKERS", str(os.cpu_count() or 4)))

logger = logging.getLogger(__name__)


def build_baseline(bbox, config, tile_cache_dir: Path, rebuild: bool) -> None:
    from utils.stac import DEA_LANDSAT_COLLECTIONS, rewrite_dea_hrefs_to_s3
    from utils.tiling import make_tile_bboxes, merge_tile_rasters
    from utils.pipeline import setup_gdal_env, setup_proj
    from utils.io import write_cog
    import pystac_client

    setup_gdal_env()
    setup_proj()

    dry_start_mm_dd = config.COMPOSITE_START
    dry_end_mm_dd   = config.COMPOSITE_END
    start_year      = config.BASELINE_START_YEAR
    end_year        = config.YEAR - 1
    use_dea_s3      = os.environ.get("USE_DEA_S3", "false").lower() == "true"

    baseline_path = config.ndvi_baseline_path()
    expected_tag  = f"NDVI_BASELINE:{start_year}-{end_year}:{dry_start_mm_dd}/{dry_end_mm_dd}"

    if baseline_path.exists() and not rebuild:
        try:
            with rasterio.open(str(baseline_path)) as src:
                tag = src.tags().get("IMAGEDESCRIPTION", "")
            if tag == expected_tag:
                logger.info("Baseline already up to date: %s", baseline_path)
                return
            logger.info("Cache tag mismatch ('%s' vs '%s') — rebuilding", tag, expected_tag)
        except Exception as exc:
            logger.warning("Could not read baseline metadata: %s — rebuilding", exc)

    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    tile_bboxes = make_tile_bboxes(bbox, 30, BASELINE_TILE_SIZE_PX)
    n_tiles = len(tile_bboxes)
    logger.info(
        "DEA baseline  %d→%d  window: %s→%s  tiles=%d  fetch_workers=%d  compute_workers=%d  s3=%s",
        start_year, end_year, dry_start_mm_dd, dry_end_mm_dd,
        n_tiles, FETCH_WORKERS, COMPUTE_WORKERS, use_dea_s3,
    )

    t0 = time.monotonic()
    q: queue.Queue = queue.Queue(maxsize=COMPUTE_WORKERS * 2)
    results: List[Optional[Path]] = [None] * n_tiles
    results_lock  = threading.Lock()
    completed_count = [0]
    failed_count    = [0]
    counts_lock   = threading.Lock()

    def _compute_worker():
        while True:
            item = q.get()
            if item is None:
                break
            tile_idx, raw_ds, fetch_s, existing_path = item
            if existing_path is not None:
                with results_lock:
                    results[tile_idx] = existing_path
                with counts_lock:
                    completed_count[0] += 1
                    done = completed_count[0]
                logger.info("Tile %d/%d skipped (cached)  [%d%%]", tile_idx + 1, n_tiles, round(100 * done / n_tiles))
                continue
            if raw_ds is None:
                with counts_lock:
                    failed_count[0] += 1
                continue
            tile_path = tile_cache_dir / f"tile_{tile_idx:05d}.tif"
            t_compute = time.monotonic()
            try:
                nir  = raw_ds["nbart_nir"].astype(np.float32)
                red  = raw_ds["nbart_red"].astype(np.float32)
                ndvi = (nir - red) / (nir + red + 1e-10)
                ndvi = ndvi.clip(-1.0, 1.0)
                baseline_tile = ndvi.median(dim="time", skipna=True)
                baseline_tile = baseline_tile.assign_coords(
                    x=nir.coords["x"], y=nir.coords["y"]
                )
                baseline_tile = baseline_tile.rio.write_crs(config.TARGET_CRS)
                baseline_tile.rio.to_raster(
                    str(tile_path), driver="GTiff", dtype="float32", compress="deflate",
                )
                compute_s = round(time.monotonic() - t_compute, 1)
                with results_lock:
                    results[tile_idx] = tile_path
                with counts_lock:
                    completed_count[0] += 1
                    done = completed_count[0]
                logger.info(
                    "Tile %d/%d  fetch=%.1fs  compute=%.1fs  [%d%%]",
                    tile_idx + 1, n_tiles, fetch_s, compute_s, round(100 * done / n_tiles),
                )
            except Exception as exc:
                with counts_lock:
                    failed_count[0] += 1
                logger.warning("Compute tile %d failed: %s", tile_idx, exc)

    compute_threads = [threading.Thread(target=_compute_worker, daemon=True) for _ in range(COMPUTE_WORKERS)]
    for t in compute_threads:
        t.start()

    def _fetch_tile(args):
        tile_idx, tile_bbox = args
        tile_path = tile_cache_dir / f"tile_{tile_idx:05d}.tif"
        if tile_path.exists() and tile_path.stat().st_size > 0:
            q.put((tile_idx, None, 0.0, tile_path))
            return
        import dask
        import odc.stac
        t_fetch = time.monotonic()
        try:
            max_items_per_year = 12
            n_years = end_year - start_year + 1
            tile_catalog = pystac_client.Client.open("https://explorer.dea.ga.gov.au/stac")
            all_items = list(tile_catalog.search(
                collections=DEA_LANDSAT_COLLECTIONS,
                bbox=tile_bbox,
                datetime=f"{start_year}-{dry_start_mm_dd}/{end_year}-{dry_end_mm_dd}",
                max_items=max_items_per_year * n_years,
            ).items())

            dry_start_month = int(dry_start_mm_dd.split("-")[0])
            dry_end_month   = int(dry_end_mm_dd.split("-")[0])
            by_year: dict = {}
            for it in all_items:
                dt = it.datetime or it.properties.get("datetime", "")
                yr = int(str(dt)[:4])
                mo = int(str(dt)[5:7])
                if dry_start_month <= mo <= dry_end_month and start_year <= yr <= end_year:
                    by_year.setdefault(yr, []).append(it)

            tile_items = []
            for yr_items in by_year.values():
                if len(yr_items) > max_items_per_year:
                    step = len(yr_items) / max_items_per_year
                    yr_items = [yr_items[round(i * step)] for i in range(max_items_per_year)]
                tile_items.extend(yr_items)

            logger.info(
                "Tile %d/%d  STAC: %d items across %d years",
                tile_idx + 1, n_tiles, len(tile_items), n_years,
            )
            if not tile_items:
                q.put((tile_idx, None, 0.0, None))
                return

            if use_dea_s3:
                tile_items = rewrite_dea_hrefs_to_s3(tile_items, DEA_BANDS)

            ds = odc.stac.load(
                tile_items,
                bands=DEA_BANDS,
                resolution=30,
                bbox=tile_bbox,
                crs=config.TARGET_CRS,
                chunks={"x": 2048, "y": 2048},
            )
            with dask.config.set(scheduler="threads"):
                raw_ds = ds.compute()
            fetch_s = round(time.monotonic() - t_fetch, 1)
            q.put((tile_idx, raw_ds, fetch_s, None))
        except Exception as exc:
            fetch_s = round(time.monotonic() - t_fetch, 1)
            logger.warning("Fetch tile %d failed (%.1fs): %s", tile_idx, fetch_s, exc)
            q.put((tile_idx, None, fetch_s, None))

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = [pool.submit(_fetch_tile, args) for args in enumerate(tile_bboxes)]
        for f in futures:
            f.result()

    for _ in range(COMPUTE_WORKERS):
        q.put(None)
    for t in compute_threads:
        t.join()

    valid_paths = [p for p in results if p is not None]
    if not valid_paths:
        raise RuntimeError("All baseline tiles failed — no output produced")

    elapsed_s = time.monotonic() - t0
    logger.info(
        "Tiles done: %d/%d succeeded  %d failed  elapsed=%.0fs",
        len(valid_paths), n_tiles, failed_count[0], elapsed_s,
    )

    merged_path = tile_cache_dir / "baseline_merged.tif"
    logger.info("Merging %d tiles…", len(valid_paths))
    merge_tile_rasters(valid_paths, merged_path, nodata=np.nan, crs=config.TARGET_CRS)

    import rioxarray
    baseline = rioxarray.open_rasterio(str(merged_path), masked=True).load()
    if baseline.ndim == 3:
        baseline = baseline.squeeze()
    baseline = baseline.rio.write_crs(config.TARGET_CRS)

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    write_cog(baseline, baseline_path)
    with rasterio.open(str(baseline_path), "r+") as dst:
        dst.update_tags(IMAGEDESCRIPTION=expected_tag)

    merged_path.unlink(missing_ok=True)
    logger.info("Baseline written: %s", baseline_path)
    logger.info("Raw tile cache retained: %s  (%d tiles)", tile_cache_dir, len(valid_paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DEA Landsat NDVI baseline TIF")
    parser.add_argument("--year", type=int, required=True,
                        help="Analysis year — baseline covers BASELINE_START_YEAR to YEAR-1")
    parser.add_argument("--rebuild", action="store_true",
                        help="Ignore existing baseline cache and rebuild from scratch")
    parser.add_argument("--tile-cache-dir", default=None,
                        help="Directory for raw per-tile cache (default: {CACHE_DIR}/dea_tiles)")
    args = parser.parse_args()

    os.environ.setdefault("YEAR", str(args.year))
    os.environ["YEAR"] = str(args.year)

    import config
    from utils.io import configure_logging

    configure_logging()

    tile_cache_dir = Path(args.tile_cache_dir) if args.tile_cache_dir else config.CACHE_DIR / "dea_tiles"

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON)
    bbox = list(catchment.to_crs("EPSG:4326").total_bounds)

    build_baseline(bbox, config, tile_cache_dir, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
