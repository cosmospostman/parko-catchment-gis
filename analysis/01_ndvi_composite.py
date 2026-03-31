"""
Step 01 — Dry-season NDVI composite (Sentinel-2).

Produces: ndvi_median_{year}.tif  (COG, EPSG:7855, 10 m)
"""
import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np

# Script-level constants (algorithm details — not in config.py)
SCL_CLEAR_CLASSES = [4, 5, 6]          # vegetation, bare soil, water
S2CLOUDLESS_PROB_MAX = 0.4
DASK_CHUNK_SPATIAL = 1024

TILE_SIZE_PX    = int(os.environ.get("TILE_SIZE_PX",    "512"))
FETCH_WORKERS   = int(os.environ.get("FETCH_WORKERS",   "4" if os.environ.get("LOCAL_S2_ROOT") else "32"))
COMPUTE_WORKERS = int(os.environ.get("COMPUTE_WORKERS", str(os.cpu_count() or 4)))

logger = logging.getLogger(__name__)


def main() -> None:
    import config
    from utils.io import configure_logging, ensure_output_dirs
    from utils.stac import search_sentinel2, load_stackstac, filter_items_by_bbox
    from utils.mask import apply_scl_mask
    from utils.quicklook import save_quicklook
    from utils.tiling import make_tile_bboxes, merge_tile_rasters
    from utils.pipeline import setup_gdal_env, setup_proj, run_tiled_pipeline

    configure_logging()
    ensure_output_dirs(config.YEAR)

    catchment = gpd.read_file(config.CATCHMENT_GEOJSON)
    bbox = list(catchment.to_crs("EPSG:4326").total_bounds)

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

    load_bands = ["red", "nir", "scl"]

    if local_root := os.environ.get("LOCAL_S2_ROOT"):
        from utils.stac import rewrite_hrefs_to_local
        items = rewrite_hrefs_to_local(items, local_root, load_bands)
        logger.info("LOCAL_S2_ROOT set — hrefs rewritten to local paths")

    logger.info("Loading %d scenes, bands: %s", len(items), load_bands)

    setup_gdal_env()
    setup_proj()

    tile_bboxes = make_tile_bboxes(bbox, config.TARGET_RESOLUTION, TILE_SIZE_PX)
    n_tiles = len(tile_bboxes)
    logger.info(
        "Processing %d spatial tiles (%d px, fetch=%d compute=%d)",
        n_tiles, TILE_SIZE_PX, FETCH_WORKERS, COMPUTE_WORKERS,
    )

    scratch_dir = Path(config.WORKING_DIR) / f"tiles_ndvi_{config.YEAR}"
    scratch_dir.mkdir(exist_ok=True)

    def fetch_fn(tile_idx, tile_bbox, tile_path):
        tile_items = filter_items_by_bbox(items, tile_bbox)
        if not tile_items:
            logger.warning("Tile %d: no items intersect bbox, skipping", tile_idx)
            return None
        stack = load_stackstac(
            items=tile_items,
            bands=load_bands,
            resolution=config.TARGET_RESOLUTION,
            bbox=tile_bbox,
            crs=config.TARGET_CRS,
            chunk_spatial=DASK_CHUNK_SPATIAL,
        )
        scl   = stack.sel(band="scl")
        stack = stack.sel(band=["red", "nir"])
        stack = apply_scl_mask(stack, scl)
        return stack.astype(np.float32).compute(scheduler="synchronous")

    def compute_fn(tile_idx, raw, tile_path):
        try:
            nir  = raw.sel(band="nir")
            red  = raw.sel(band="red")
            ndvi = (nir - red) / (nir + red + 1e-10)
            ndvi = ndvi.clip(-1.0, 1.0)
            ndvi_median = ndvi.median(dim="time", skipna=True)
            ndvi_median = ndvi_median.rio.write_crs(config.TARGET_CRS)
            ndvi_median.rio.to_raster(
                str(tile_path), driver="GTiff", dtype="float32", compress="deflate",
            )
            logger.info("Tile %d/%d complete (%.1f%%)", tile_idx + 1, n_tiles,
                        100 * (tile_idx + 1) / n_tiles)
            return tile_path
        except Exception as exc:
            logger.warning("Compute tile %d failed: %s", tile_idx, exc)
            return None

    out_path = config.ndvi_median_path(config.YEAR)
    run_tiled_pipeline(
        tile_bboxes=tile_bboxes,
        scratch_dir=scratch_dir,
        fetch_fn=fetch_fn,
        compute_fn=compute_fn,
        merge_fn=merge_tile_rasters,
        out_path=out_path,
        nodata=np.nan,
        crs=config.TARGET_CRS,
        fetch_workers=FETCH_WORKERS,
        compute_workers=COMPUTE_WORKERS,
    )

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
