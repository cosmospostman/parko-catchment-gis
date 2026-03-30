"""
Step 03 — Parkinsonia flowering spectral index (August–October composite).

Produces:
  flowering_index_{year}.tif  — green/NIR ratio (COG)
"""
import logging
import sys
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import xarray as xr

# Script-level constants
FLOWERING_BANDS = ["green", "nir", "rededge1", "rededge2"]   # green, NIR, RE1, RE2
GREEN_NIR_RATIO_NODATA = -9999.0
NDRE_NODATA = -9999.0
DASK_CHUNK_SPATIAL = 2048

logger = logging.getLogger(__name__)


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

    with dask.config.set(scheduler="synchronous"):
        stack = load_stackstac(
            items=items,
            bands=load_bands,
            resolution=config.TARGET_RESOLUTION,
            bbox=bbox,
            crs=config.TARGET_CRS,
            chunk_spatial=DASK_CHUNK_SPATIAL,
        )

        scl   = stack.sel(band="scl")
        stack = stack.sel(band=FLOWERING_BANDS)
        stack = apply_scl_mask(stack, scl)

        green = stack.sel(band="green").astype(np.float32)
        nir   = stack.sel(band="nir").astype(np.float32)

        # Green/NIR ratio — elevated during Parkinsonia flowering
        ratio = green / (nir + 1e-10)
        ratio = ratio.where(nir > 0)

        # Median over flowering window
        flowering_index = ratio.median(dim="time", skipna=True).compute()

    flowering_index = flowering_index.rio.write_crs(config.TARGET_CRS)

    out_path = config.flowering_index_path(config.YEAR)
    write_cog(flowering_index, out_path, nodata=np.nan)
    logger.info("Written: %s", out_path)

    ql_path = Path(str(out_path).replace(".tif", "_quicklook.png"))
    save_quicklook(
        flowering_index,
        ql_path,
        vmin=0.0,
        vmax=1.5,
        cmap="YlOrRd",
        title=f"Flowering Index (Green/NIR) {config.YEAR}",
    )


if __name__ == "__main__":
    main()
