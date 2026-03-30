"""
Step 02 — Long-term NDVI anomaly (DEA Landsat baseline).

Produces: ndvi_anomaly_{year}.tif  (COG, EPSG:7844, 30 m resampled to 10 m)
"""
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
import xarray as xr

# Script-level constants
DEA_BANDS = ["nbart_red", "nbart_nir"]
FC_PV_BAND = "pv"
RESAMPLING_METHOD = Resampling.bilinear
BASELINE_CACHE_FILENAME = "ndvi_baseline_median.tif"

# Baseline tiles can be larger: only 2 bands at 30 m
BASELINE_TILE_SIZE_PX = int(os.environ.get("TILE_SIZE_PX", "1024"))
BASELINE_TILE_WORKERS  = int(os.environ.get("TILE_WORKERS",  "32"))

logger = logging.getLogger(__name__)


def _build_baseline(bbox, config) -> xr.DataArray:
    """Download and compute the Landsat NDVI baseline median (1986 to year-1)."""
    from utils.stac import load_dea_landsat
    from utils.tiling import make_tile_bboxes, merge_tile_rasters

    start = f"{config.BASELINE_START_YEAR}-01-01"
    end   = f"{config.YEAR - 1}-12-31"
    logger.info("Building Landsat baseline %s → %s", start, end)

    # Set GDAL env vars before ThreadPoolExecutor so all worker threads inherit them
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    gdal_env = {
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "GDAL_HTTP_RETRY_ON_HTTP_ERROR": "429,500,502,503,504",
        "GDAL_HTTP_PERSISTENT": "YES",
        "CPL_VSIL_CURL_CACHE_SIZE": "67108864",  # 64 MB connection cache
        "CPL_VSIL_CURL_CHUNK_SIZE": "10485760",  # 10 MB
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }
    for k, v in gdal_env.items():
        os.environ.setdefault(k, v)

    if "PROJ_DATA" not in os.environ:
        try:
            import rasterio
            proj_data = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
            if os.path.isdir(proj_data):
                os.environ["PROJ_DATA"] = proj_data
        except Exception:
            pass
    if "PROJ_DATA" not in os.environ:
        try:
            from pyproj.datadir import get_data_dir
            os.environ["PROJ_DATA"] = get_data_dir()
        except Exception:
            pass
    try:
        from pyproj.datadir import set_data_dir
        set_data_dir(os.environ["PROJ_DATA"])
    except Exception:
        pass

    tile_bboxes = make_tile_bboxes(bbox, 30, BASELINE_TILE_SIZE_PX)
    n_tiles = len(tile_bboxes)
    logger.info(
        "Processing %d spatial tiles for baseline (%d px, %d concurrent)",
        n_tiles, BASELINE_TILE_SIZE_PX, BASELINE_TILE_WORKERS,
    )

    scratch_dir = Path(config.WORKING_DIR) / f"tiles_baseline_{config.YEAR}"
    scratch_dir.mkdir(exist_ok=True)

    def process_tile(args):
        tile_idx, tile_bbox = args
        tile_path = scratch_dir / f"tile_{tile_idx:05d}.tif"

        if tile_path.exists() and tile_path.stat().st_size > 0:
            logger.info("Tile %d/%d skipped (cached)", tile_idx + 1, n_tiles)
            return tile_path

        import dask
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
            nir = ds["nbart_nir"].astype(np.float32)
            red = ds["nbart_red"].astype(np.float32)
            ndvi = (nir - red) / (nir + red + 1e-10)
            ndvi = ndvi.clip(-1.0, 1.0)

            with dask.config.set(scheduler="threads"):
                baseline_tile = ndvi.median(dim="time", skipna=True).compute()

            baseline_tile = baseline_tile.rio.write_crs(config.TARGET_CRS)

            baseline_tile.rio.to_raster(
                str(tile_path), driver="GTiff", dtype="float32",
                compress="deflate",
            )
            logger.info("Tile %d/%d complete (%.1f%%)", tile_idx + 1, n_tiles,
                        100 * (tile_idx + 1) / n_tiles)
            return tile_path
        except Exception as exc:
            logger.warning("Baseline tile %d failed: %s", tile_idx, exc)
            return None

    with ThreadPoolExecutor(max_workers=BASELINE_TILE_WORKERS) as pool:
        tile_paths = list(pool.map(process_tile, enumerate(tile_bboxes)))

    valid_paths = [p for p in tile_paths if p is not None]
    if not valid_paths:
        raise RuntimeError("All baseline tiles failed — no output produced")

    # Merge tiles into a temporary path, then load as DataArray
    merged_path = scratch_dir / "baseline_merged.tif"
    merge_tile_rasters(valid_paths, merged_path, nodata=np.nan, crs=config.TARGET_CRS)

    baseline = xr.open_dataarray(str(merged_path)).load()
    if baseline.ndim == 3:
        baseline = baseline.squeeze()
    baseline = baseline.rio.write_crs(config.TARGET_CRS)

    # Clean up scratch
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

    # Check if cached baseline is still valid (check IMAGEDESCRIPTION tag)
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
        # Tag with year range
        with rasterio.open(str(baseline_path), "r+") as dst:
            dst.update_tags(
                IMAGEDESCRIPTION=f"NDVI_BASELINE:{config.BASELINE_START_YEAR}-{config.YEAR - 1}"
            )
        logger.info("Baseline written: %s", baseline_path)
    else:
        baseline = read_raster(baseline_path)
        if baseline.ndim == 3:
            baseline = baseline.squeeze()

    # Load current year NDVI median
    ndvi_path = config.ndvi_median_path(config.YEAR)
    if not ndvi_path.exists():
        raise FileNotFoundError(f"Step 01 output not found: {ndvi_path}")
    ndvi_current = read_raster(ndvi_path)
    if ndvi_current.ndim == 3:
        ndvi_current = ndvi_current.squeeze()

    # Reproject baseline to match current NDVI grid
    baseline_reproj = baseline.rio.reproject_match(ndvi_current, resampling=RESAMPLING_METHOD)

    # Compute anomaly
    anomaly = ndvi_current - baseline_reproj
    anomaly = anomaly.clip(-1.0, 1.0)
    anomaly = anomaly.rio.write_crs(config.TARGET_CRS)

    out_path = config.ndvi_anomaly_path(config.YEAR)
    write_cog(anomaly, out_path)
    logger.info("Written: %s", out_path)

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
