"""
Step 01 — Dry-season NDVI composite (Sentinel-2).

Produces: ndvi_median_{year}.tif  (COG, EPSG:7855, 10 m)
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np

# Script-level constants (algorithm details — not in config.py)
SCL_CLEAR_CLASSES = [4, 5, 6]          # vegetation, bare soil, water
S2CLOUDLESS_PROB_MAX = 0.4
DASK_CHUNK_SPATIAL = 1024

TILE_SIZE_PX = int(os.environ.get("TILE_SIZE_PX", "256"))
TILE_WORKERS  = int(os.environ.get("TILE_WORKERS",  "32"))

logger = logging.getLogger(__name__)


def main() -> None:
    import config
    from utils.io import configure_logging, ensure_output_dirs, write_cog
    from utils.stac import search_sentinel2, load_stackstac
    from utils.mask import apply_scl_mask
    from utils.quicklook import save_quicklook
    from utils.tiling import make_tile_bboxes, merge_tile_rasters

    configure_logging()

    ensure_output_dirs(config.YEAR)

    # Load catchment geometry and derive bounding box
    catchment = gpd.read_file(config.CATCHMENT_GEOJSON)
    bbox = list(catchment.to_crs("EPSG:4326").total_bounds)  # [minx, miny, maxx, maxy]

    composite_start = f"{config.YEAR}-{config.COMPOSITE_START}"
    composite_end   = f"{config.YEAR}-{config.COMPOSITE_END}"

    logger.info("Searching Sentinel-2 items: %s → %s", composite_start, composite_end)
    items = search_sentinel2(
        bbox=bbox,
        start=composite_start,
        end=composite_end,
        cloud_cover_max=config.CLOUD_COVER_MAX,
        endpoint=config.STAC_ENDPOINT_ELEMENT84,
        collection=config.S2_COLLECTION,
    )
    if not items:
        raise RuntimeError(f"No Sentinel-2 items found for {config.YEAR}")

    load_bands = config.COMPOSITE_BANDS + ["scl"]
    logger.info("Loading %d scenes, bands: %s", len(items), load_bands)

    # Set GDAL env vars before ThreadPoolExecutor so all worker threads inherit them
    gdal_env = {
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "0.5",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "CPL_VSIL_CURL_CHUNK_SIZE": "10485760",  # 10 MB
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
    }
    for k, v in gdal_env.items():
        os.environ.setdefault(k, v)

    tile_bboxes = make_tile_bboxes(bbox, config.TARGET_RESOLUTION, TILE_SIZE_PX)
    n_tiles = len(tile_bboxes)
    logger.info(
        "Processing %d spatial tiles (%d px, %d concurrent)",
        n_tiles, TILE_SIZE_PX, TILE_WORKERS,
    )

    scratch_dir = Path(config.WORKING_DIR) / f"tiles_ndvi_{config.YEAR}"
    scratch_dir.mkdir(exist_ok=True)

    def process_tile(args):
        tile_idx, tile_bbox = args
        tile_path = scratch_dir / f"tile_{tile_idx:05d}.tif"

        if tile_path.exists() and tile_path.stat().st_size > 0:
            logger.info("Tile %d/%d skipped (cached)", tile_idx + 1, n_tiles)
            return tile_path

        import dask
        try:
            stack = load_stackstac(
                items=items,
                bands=load_bands,
                resolution=config.TARGET_RESOLUTION,
                bbox=tile_bbox,
                crs=config.TARGET_CRS,
                chunk_spatial=DASK_CHUNK_SPATIAL,
            )

            scl   = stack.sel(band="scl")
            stack = stack.sel(band=config.COMPOSITE_BANDS)
            stack = apply_scl_mask(stack, scl)

            nir = stack.sel(band="nir").astype(np.float32)
            red = stack.sel(band="red").astype(np.float32)
            ndvi = (nir - red) / (nir + red + 1e-10)
            ndvi = ndvi.clip(-1.0, 1.0)

            with dask.config.set(scheduler="threads"):
                ndvi_median = ndvi.median(dim="time", skipna=True).compute()

            ndvi_median = ndvi_median.rio.write_crs(config.TARGET_CRS)

            ndvi_median.rio.to_raster(
                str(tile_path), driver="GTiff", dtype="float32",
                compress="deflate",
            )
            logger.info("Tile %d/%d complete (%.1f%%)", tile_idx + 1, n_tiles,
                        100 * (tile_idx + 1) / n_tiles)
            return tile_path
        except Exception as exc:
            logger.warning("Tile %d failed: %s", tile_idx, exc)
            return None

    with ThreadPoolExecutor(max_workers=TILE_WORKERS) as pool:
        tile_paths = list(pool.map(process_tile, enumerate(tile_bboxes)))

    valid_paths = [p for p in tile_paths if p is not None]
    if not valid_paths:
        raise RuntimeError("All tiles failed — no output produced")

    out_path = config.ndvi_median_path(config.YEAR)
    merge_tile_rasters(valid_paths, out_path, nodata=np.nan, crs=config.TARGET_CRS)
    logger.info("Written: %s", out_path)

    # Clean up scratch tiles
    for p in valid_paths:
        p.unlink(missing_ok=True)
    try:
        scratch_dir.rmdir()
    except OSError:
        pass

    import xarray as xr
    ndvi_median_full = xr.open_dataarray(str(out_path))
    ql_path = Path(str(out_path).replace(".tif", "_quicklook.png"))
    save_quicklook(
        ndvi_median_full,
        ql_path,
        vmin=-0.1,
        vmax=0.8,
        cmap="RdYlGn",
        title=f"NDVI Median Composite {config.YEAR}",
    )


if __name__ == "__main__":
    main()
