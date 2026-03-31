"""
Step 02 — Long-term NDVI anomaly (DEA Landsat baseline).

Produces: ndvi_anomaly_{year}.tif  (COG, EPSG:7844, 30 m resampled to 10 m)

Note: this script does NOT use run_tiled_pipeline — it has two sequential tiled
passes (baseline build + anomaly compute) with an intermediate cache, and uses
odc-stac rather than stackstac. setup_gdal_env() and setup_proj() are shared
from utils.pipeline.
"""
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
import xarray as xr

# Script-level constants
DEA_BANDS = ["nbart_red", "nbart_nir"]
RESAMPLING_METHOD = Resampling.bilinear
BASELINE_CACHE_FILENAME = "ndvi_baseline_median.tif"

# Baseline tiles can be larger: only 2 bands at 30 m
BASELINE_TILE_SIZE_PX = int(os.environ.get("TILE_SIZE_PX",    "1024"))
FETCH_WORKERS         = int(os.environ.get("FETCH_WORKERS",   "16"))
COMPUTE_WORKERS       = int(os.environ.get("COMPUTE_WORKERS", str(os.cpu_count() or 4)))

logger = logging.getLogger(__name__)


def _build_baseline(bbox, config) -> xr.DataArray:
    """Download and compute the Landsat NDVI baseline median (1986 to year-1)."""
    from utils.stac import load_dea_landsat
    from utils.tiling import make_tile_bboxes, merge_tile_rasters
    from utils.pipeline import setup_gdal_env, setup_proj

    start = f"{config.BASELINE_START_YEAR}-01-01"
    end   = f"{config.YEAR - 1}-12-31"
    logger.info(
        "── Step 02 baseline  DEA Landsat %s → %s ──────────────────────────",
        start, end,
    )

    setup_gdal_env()
    setup_proj()

    tile_bboxes = make_tile_bboxes(bbox, 30, BASELINE_TILE_SIZE_PX)
    n_tiles = len(tile_bboxes)
    logger.info(
        "Baseline: %d tiles  tile_size=%dpx  fetch_workers=%d  compute_workers=%d",
        n_tiles, BASELINE_TILE_SIZE_PX, FETCH_WORKERS, COMPUTE_WORKERS,
    )

    baseline_t0 = time.monotonic()

    scratch_dir = Path(config.WORKING_DIR) / f"tiles_baseline_{config.YEAR}"
    scratch_dir.mkdir(exist_ok=True)

    q: queue.Queue = queue.Queue(maxsize=COMPUTE_WORKERS * 2)
    results: List[Optional[Path]] = [None] * n_tiles
    results_lock = threading.Lock()
    completed_count = [0]
    failed_count = [0]
    completed_lock = threading.Lock()

    def _compute_worker():
        while True:
            item = q.get()
            if item is None:
                break
            tile_idx, raw_ds, fetch_s, existing_path = item
            if existing_path is not None:
                with results_lock:
                    results[tile_idx] = existing_path
                with completed_lock:
                    completed_count[0] += 1
                    done = completed_count[0]
                logger.info(
                    "Tile %d/%d skipped (cached)  [%d%%]",
                    tile_idx + 1, n_tiles, round(100 * done / n_tiles),
                )
                continue
            if raw_ds is None:
                with completed_lock:
                    failed_count[0] += 1
                continue
            tile_path = scratch_dir / f"tile_{tile_idx:05d}.tif"
            t_compute = time.monotonic()
            try:
                nir  = raw_ds["nbart_nir"].astype(np.float32)
                red  = raw_ds["nbart_red"].astype(np.float32)
                ndvi = (nir - red) / (nir + red + 1e-10)
                ndvi = ndvi.clip(-1.0, 1.0)
                baseline_tile = ndvi.median(dim="time", skipna=True)
                baseline_tile = baseline_tile.rio.write_crs(config.TARGET_CRS)
                baseline_tile.rio.to_raster(
                    str(tile_path), driver="GTiff", dtype="float32", compress="deflate",
                )
                compute_s = round(time.monotonic() - t_compute, 1)
                with results_lock:
                    results[tile_idx] = tile_path
                with completed_lock:
                    completed_count[0] += 1
                    done = completed_count[0]
                logger.info(
                    "Tile %d/%d  fetch=%.1fs  compute=%.1fs  [%d%%]",
                    tile_idx + 1, n_tiles, fetch_s, compute_s,
                    round(100 * done / n_tiles),
                )
            except Exception as exc:
                with completed_lock:
                    failed_count[0] += 1
                logger.warning("Compute baseline tile %d failed: %s", tile_idx, exc)

    compute_threads = [
        threading.Thread(target=_compute_worker, daemon=True)
        for _ in range(COMPUTE_WORKERS)
    ]
    for t in compute_threads:
        t.start()

    def _fetch_tile(args):
        tile_idx, tile_bbox = args
        tile_path = scratch_dir / f"tile_{tile_idx:05d}.tif"
        if tile_path.exists() and tile_path.stat().st_size > 0:
            q.put((tile_idx, None, 0.0, tile_path))
            return
        import dask
        t_fetch = time.monotonic()
        try:
            ds = load_dea_landsat(
                bbox=tile_bbox,
                start=start,
                end=end,
                bands=DEA_BANDS,
                collection=config.DEA_COLLECTION,
                resolution=30,
                crs=config.TARGET_CRS,
            )
            with dask.config.set(scheduler="threads"):
                raw_ds = ds.compute()
            fetch_s = round(time.monotonic() - t_fetch, 1)
            q.put((tile_idx, raw_ds, fetch_s, None))
        except Exception as exc:
            fetch_s = round(time.monotonic() - t_fetch, 1)
            logger.warning("Fetch baseline tile %d failed (%.1fs): %s", tile_idx, fetch_s, exc)
            q.put((tile_idx, None, fetch_s, None))

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as fetch_pool:
        futures = [fetch_pool.submit(_fetch_tile, args) for args in enumerate(tile_bboxes)]
        for f in futures:
            f.result()

    for _ in range(COMPUTE_WORKERS):
        q.put(None)
    for t in compute_threads:
        t.join()

    valid_paths = [p for p in results if p is not None]
    if not valid_paths:
        raise RuntimeError("All baseline tiles failed — no output produced")

    elapsed_s = time.monotonic() - baseline_t0
    logger.info(
        "Baseline tiles done: %d/%d succeeded  %d failed  elapsed=%.0fs",
        len(valid_paths), n_tiles, failed_count[0], elapsed_s,
    )
    logger.info("Merging %d baseline tiles…", len(valid_paths))
    merged_path = scratch_dir / "baseline_merged.tif"
    merge_tile_rasters(valid_paths, merged_path, nodata=np.nan, crs=config.TARGET_CRS)

    baseline = xr.open_dataarray(str(merged_path)).load()
    if baseline.ndim == 3:
        baseline = baseline.squeeze()
    baseline = baseline.rio.write_crs(config.TARGET_CRS)

    for p in valid_paths:
        p.unlink(missing_ok=True)
    merged_path.unlink(missing_ok=True)
    try:
        scratch_dir.rmdir()
    except OSError:
        pass

    return baseline


def main() -> None:
    import config
    from utils.io import configure_logging, ensure_output_dirs, write_cog, read_raster
    from utils.quicklook import save_quicklook

    configure_logging()
    ensure_output_dirs(config.YEAR)

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON)
    bbox = list(catchment.to_crs("EPSG:4326").total_bounds)

    rebuild = os.environ.get("REBUILD_BASELINE", "false").lower() == "true"
    baseline_path = config.ndvi_baseline_path()

    if baseline_path.exists() and not rebuild:
        try:
            with rasterio.open(str(baseline_path)) as src:
                desc = src.tags().get("IMAGEDESCRIPTION", "")
            expected_desc = f"NDVI_BASELINE:{config.BASELINE_START_YEAR}-{config.YEAR - 1}"
            if desc != expected_desc:
                logger.info(
                    "Baseline cache tag mismatch ('%s' vs '%s') — rebuilding",
                    desc, expected_desc,
                )
                rebuild = True
            else:
                logger.info("Reusing cached baseline: %s", baseline_path)
        except Exception as exc:
            logger.warning("Could not read baseline cache metadata: %s — rebuilding", exc)
            rebuild = True

    if rebuild or not baseline_path.exists():
        baseline = _build_baseline(bbox, config)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        write_cog(baseline, baseline_path)
        with rasterio.open(str(baseline_path), "r+") as dst:
            dst.update_tags(
                IMAGEDESCRIPTION=f"NDVI_BASELINE:{config.BASELINE_START_YEAR}-{config.YEAR - 1}"
            )
        logger.info("Baseline written: %s", baseline_path)
    else:
        baseline = read_raster(baseline_path)
        if baseline.ndim == 3:
            baseline = baseline.squeeze()

    logger.info("── Step 02 anomaly  loading current-year NDVI ─────────────────────────")
    ndvi_path = config.ndvi_median_path(config.YEAR)
    if not ndvi_path.exists():
        raise FileNotFoundError(f"Step 01 output not found: {ndvi_path}")
    ndvi_current = read_raster(ndvi_path)
    if ndvi_current.ndim == 3:
        ndvi_current = ndvi_current.squeeze()

    logger.info("Reprojecting baseline to match current NDVI grid…")
    t_reproj = time.monotonic()
    baseline_reproj = baseline.rio.reproject_match(ndvi_current, resampling=RESAMPLING_METHOD)
    logger.info("Reprojection done  elapsed=%.1fs", time.monotonic() - t_reproj)

    logger.info("Computing anomaly (current − baseline)…")
    anomaly = ndvi_current - baseline_reproj
    anomaly = anomaly.clip(-1.0, 1.0)
    anomaly = anomaly.rio.write_crs(config.TARGET_CRS)

    out_path = config.ndvi_anomaly_path(config.YEAR)
    logger.info("Writing COG: %s", out_path)
    write_cog(anomaly, out_path)
    logger.info("── Step 02 complete  output: %s ────────────────────────────────────────", out_path)

    ql_path = Path(str(out_path).replace(".tif", "_quicklook.png"))
    save_quicklook(
        anomaly,
        ql_path,
        vmin=-0.3,
        vmax=0.3,
        cmap="RdBu",
        title=f"NDVI Anomaly {config.YEAR}",
    )


if __name__ == "__main__":
    main()
