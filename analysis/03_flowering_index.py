"""
Step 03 — Parkinsonia flowering spectral index (August–October composite).

Produces:
  flowering_index_{year}.tif  — green/NIR ratio (COG)
"""
import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np

# Script-level constants
FLOWERING_BANDS = ["green", "nir", "rededge1", "rededge2"]   # green, NIR, RE1, RE2
DASK_CHUNK_SPATIAL = 1024

TILE_SIZE_PX    = int(os.environ.get("TILE_SIZE_PX",    "512"))
FETCH_WORKERS   = int(os.environ.get("FETCH_WORKERS",   "4" if os.environ.get("LOCAL_S2_ROOT") else "16"))
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

    if local_root := os.environ.get("LOCAL_S2_ROOT"):
        from utils.stac import rewrite_hrefs_to_local
        items = rewrite_hrefs_to_local(items, local_root, load_bands)
        logger.info("LOCAL_S2_ROOT set — hrefs rewritten to local paths")

    # Strip root catalog links — forked workers must not re-fetch the STAC endpoint.
    for item in items:
        item.clear_links()

    logger.info("Loading %d scenes for flowering index", len(items))

    setup_gdal_env()
    setup_proj()

    tile_bboxes = make_tile_bboxes(bbox, config.TARGET_RESOLUTION, TILE_SIZE_PX)
    n_tiles = len(tile_bboxes)
    logger.info(
        "Processing %d spatial tiles (%d px, fetch=%d compute=%d)",
        n_tiles, TILE_SIZE_PX, FETCH_WORKERS, COMPUTE_WORKERS,
    )

    scratch_dir = Path(config.WORKING_DIR) / f"tiles_flowering_{config.YEAR}"
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
        stack = stack.sel(band=FLOWERING_BANDS)
        stack = apply_scl_mask(stack, scl)
        return stack.astype(np.float32).compute(scheduler="synchronous")

    def compute_fn(tile_idx, raw, tile_path):
        try:
            green = raw.sel(band="green")
            nir   = raw.sel(band="nir")
            ratio = green / (nir + 1e-10)
            ratio = ratio.where(nir > 0)
            flowering_index = ratio.median(dim="time", skipna=True)
            flowering_index = flowering_index.rio.write_crs(config.TARGET_CRS)
            flowering_index.rio.to_raster(
                str(tile_path), driver="GTiff", dtype="float32", compress="deflate",
            )
            logger.info("Tile %d/%d complete (%.1f%%)", tile_idx + 1, n_tiles,
                        100 * (tile_idx + 1) / n_tiles)
            return tile_path
        except Exception as exc:
            logger.warning("Compute tile %d failed: %s", tile_idx, exc)
            return None

    out_path = config.flowering_index_path(config.YEAR)
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

    ql_path = Path(str(out_path).replace(".tif", "_quicklook.png"))
    save_quicklook(
        out_path,
        ql_path,
        vmin=0.0,
        vmax=1.5,
        cmap="YlOrRd",
        title=f"Flowering Index (Green/NIR) {config.YEAR}",
    )


if __name__ == "__main__":
    main()
