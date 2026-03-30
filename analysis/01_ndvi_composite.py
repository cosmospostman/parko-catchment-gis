"""
Step 01 — Dry-season NDVI composite (Sentinel-2).

Produces: ndvi_median_{year}.tif  (COG, EPSG:7855, 10 m)
"""
import logging
from pathlib import Path

import os

import dask
import geopandas as gpd
import numpy as np
import xarray as xr

# Script-level constants (algorithm details — not in config.py)
SCL_CLEAR_CLASSES = [4, 5, 6]          # vegetation, bare soil, water
S2CLOUDLESS_PROB_MAX = 0.4
DASK_CHUNK_SPATIAL = 512

logger = logging.getLogger(__name__)

PIPELINE_RUN = os.environ.get("PIPELINE_RUN") == "1"


def main() -> None:
    import config
    from utils.io import configure_logging, ensure_output_dirs, write_cog
    from utils.stac import search_sentinel2, load_stackstac
    from utils.mask import apply_scl_mask
    from utils.quicklook import save_quicklook
    from utils.progress import LogProgressCallback

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

    # Load all bands + SCL
    load_bands = config.COMPOSITE_BANDS + ["scl"]
    logger.info("Loading %d scenes, bands: %s", len(items), load_bands)

    if PIPELINE_RUN:
        from dask.distributed import LocalCluster, Client
        cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="3GiB")
        client = Client(cluster)
        logger.info("Dask dashboard: %s", client.dashboard_link)
        dask_ctx = dask.config.set(scheduler="synchronous")  # client takes over
    else:
        client = None
        cluster = None
        dask_ctx = dask.config.set(scheduler="synchronous")

    # GDAL tuning for high-throughput S3 COG reads
    gdal_env = {
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "0.5",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "CPL_VSIL_CURL_CHUNK_SIZE": "10485760",  # 10 MB
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
    }

    try:
        import rasterio
        from rasterio.env import Env as RasterioEnv
        rio_env = RasterioEnv(**gdal_env)
        rio_env.__enter__()
    except Exception:
        rio_env = None

    try:
        with dask_ctx:
            stack = load_stackstac(
                items=items,
                bands=load_bands,
                resolution=config.TARGET_RESOLUTION,
                bbox=bbox,
                crs=config.TARGET_CRS,
                chunk_spatial=DASK_CHUNK_SPATIAL,
            )

            # Separate SCL from optical bands
            scl   = stack.sel(band="scl")
            stack = stack.sel(band=config.COMPOSITE_BANDS)

            # Apply cloud mask per scene
            stack = apply_scl_mask(stack, scl)

            # Compute NDVI per scene: (B08 - B04) / (B08 + B04)
            nir = stack.sel(band="nir").astype(np.float32)
            red = stack.sel(band="red").astype(np.float32)
            ndvi = (nir - red) / (nir + red + 1e-10)
            ndvi = ndvi.clip(-1.0, 1.0)

            # Median composite over time
            scheduler_label = "distributed" if client else "synchronous"
            logger.info("Computing median composite (scheduler=%s)...", scheduler_label)
            with LogProgressCallback(label="median composite", log_every=200):
                ndvi_median = ndvi.median(dim="time", skipna=True).compute()
    finally:
        if rio_env is not None:
            rio_env.__exit__(None, None, None)
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()

    # Reproject to target CRS if needed (should already be correct from stackstac)
    ndvi_median = ndvi_median.rio.write_crs(config.TARGET_CRS)

    out_path = config.ndvi_median_path(config.YEAR)
    write_cog(ndvi_median, out_path)
    logger.info("Written: %s", out_path)

    # Quicklook
    ql_path = Path(str(out_path).replace(".tif", "_quicklook.png"))
    save_quicklook(
        ndvi_median,
        ql_path,
        vmin=-0.1,
        vmax=0.8,
        cmap="RdYlGn",
        title=f"NDVI Median Composite {config.YEAR}",
    )


if __name__ == "__main__":
    main()
