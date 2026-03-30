"""
Step 03 — Parkinsonia flowering spectral index (August–October composite).

Produces:
  flowering_index_{year}.tif  — green/NIR ratio (COG)
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np

# Script-level constants
FLOWERING_BANDS = ["green", "nir", "rededge1", "rededge2"]   # green, NIR, RE1, RE2
GREEN_NIR_RATIO_NODATA = -9999.0
NDRE_NODATA = -9999.0
DASK_CHUNK_SPATIAL = 1024

TILE_SIZE_PX = int(os.environ.get("TILE_SIZE_PX", "256"))
TILE_WORKERS  = int(os.environ.get("TILE_WORKERS",  "4"))

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

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON)
    bbox = list(catchment.to_crs("EPSG:4326").total_bounds)

    flower_start = f"{config.YEAR}-{config.FLOWERING_WINDOW_START}"
    flower_end   = f"{config.YEAR}-{config.FLOWERING_WINDOW_END}"

    logger.info("Searching Sentinel-2 for flowering window: %s → %s", flower_start, flower_end)
    items = search_sentinel2(
        bbox=bbox,
        start=flower_start,
        end=flower_end,
        cloud_cover_max=config.CLOUD_COVER_MAX,
        endpoint=config.STAC_ENDPOINT_ELEMENT84,
        collection=config.S2_COLLECTION,
    )
    if not items:
        raise RuntimeError(f"No Sentinel-2 items found for flowering window {config.YEAR}")

    load_bands = FLOWERING_BANDS + ["scl"]
    logger.info("Loading %d scenes for flowering index", len(items))

    # Set GDAL env vars before ThreadPoolExecutor so all worker threads inherit them
    gdal_env = {
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "0.5",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "CPL_VSIL_CURL_CHUNK_SIZE": "10485760",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
    }
    for k, v in gdal_env.items():
        os.environ.setdefault(k, v)

    tile_bboxes = make_tile_bboxes(bbox, config.TARGET_RESOLUTION, TILE_SIZE_PX)
    logger.info(
        "Processing %d spatial tiles (%d px, %d concurrent)",
        len(tile_bboxes), TILE_SIZE_PX, TILE_WORKERS,
    )

    scratch_dir = Path(config.WORKING_DIR) / f"tiles_flowering_{config.YEAR}"
    scratch_dir.mkdir(exist_ok=True)

    def process_tile(args):
        tile_idx, tile_bbox = args
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
            stack = stack.sel(band=FLOWERING_BANDS)
            stack = apply_scl_mask(stack, scl)

            green = stack.sel(band="green").astype(np.float32)
            nir   = stack.sel(band="nir").astype(np.float32)

            ratio = green / (nir + 1e-10)
            ratio = ratio.where(nir > 0)

            with dask.config.set(scheduler="threads"):
                flowering_index = ratio.median(dim="time", skipna=True).compute()

            flowering_index = flowering_index.rio.write_crs(config.TARGET_CRS)

            tile_path = scratch_dir / f"tile_{tile_idx:05d}.tif"
            flowering_index.rio.to_raster(
                str(tile_path), driver="GTiff", dtype="float32",
                compress="deflate",
            )
            return tile_path
        except Exception as exc:
            logger.warning("Tile %d failed: %s", tile_idx, exc)
            return None

    with ThreadPoolExecutor(max_workers=TILE_WORKERS) as pool:
        tile_paths = list(pool.map(process_tile, enumerate(tile_bboxes)))

    valid_paths = [p for p in tile_paths if p is not None]
    if not valid_paths:
        raise RuntimeError("All tiles failed — no output produced")

    out_path = config.flowering_index_path(config.YEAR)
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
    flowering_full = xr.open_dataarray(str(out_path))
    ql_path = Path(str(out_path).replace(".tif", "_quicklook.png"))
    save_quicklook(
        flowering_full,
        ql_path,
        vmin=0.0,
        vmax=1.5,
        cmap="YlOrRd",
        title=f"Flowering Index (Green/NIR) {config.YEAR}",
    )


if __name__ == "__main__":
    main()
