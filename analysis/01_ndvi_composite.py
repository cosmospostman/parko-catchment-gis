"""
Step 01 — Dry-season NDVI composite (Sentinel-2).

Produces: ndvi_median_{year}.tif  (COG, EPSG:7855, 10 m)
"""
import logging
import sys
from pathlib import Path

import os
import threading
import time

import dask
import geopandas as gpd
import numpy as np
import xarray as xr

# Script-level constants (algorithm details — not in config.py)
SCL_CLEAR_CLASSES = [4, 5, 6]          # vegetation, bare soil, water
S2CLOUDLESS_PROB_MAX = 0.4
DASK_CHUNK_SPATIAL = 2048

logger = logging.getLogger(__name__)

PIPELINE_RUN = os.environ.get("PIPELINE_RUN") == "1"


def _log_progress(label: str, stop_event: threading.Event, interval: int = 60) -> None:
    """Log elapsed time periodically until stop_event is set."""
    start = time.monotonic()
    while not stop_event.wait(interval):
        elapsed = int(time.monotonic() - start)
        logger.info("%s — still running (%dm %02ds elapsed)", label, elapsed // 60, elapsed % 60)


def main() -> None:
    import config
    from utils.io import ensure_output_dirs, write_cog
    from utils.stac import search_sentinel2, load_stackstac
    from utils.mask import apply_scl_mask
    from utils.quicklook import save_quicklook

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

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

    dask_scheduler = "threads" if PIPELINE_RUN else "synchronous"

    with dask.config.set(scheduler=dask_scheduler):
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
        logger.info("Computing median composite (scheduler=%s)...", dask_scheduler)
        stop = threading.Event()
        progress_thread = threading.Thread(
            target=_log_progress, args=("median composite", stop), daemon=True
        )
        progress_thread.start()
        try:
            ndvi_median = ndvi.median(dim="time", skipna=True).compute()
        finally:
            stop.set()
            progress_thread.join()

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
