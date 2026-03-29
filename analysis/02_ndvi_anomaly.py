"""
Step 02 — Long-term NDVI anomaly (DEA Landsat baseline).

Produces: ndvi_anomaly_{year}.tif  (COG, EPSG:7844, 30 m resampled to 10 m)
"""
import logging
import os
import sys
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr

# Script-level constants
DEA_BANDS = ["nbart_red", "nbart_nir"]
FC_PV_BAND = "pv"
RESAMPLING_METHOD = "bilinear"
BASELINE_CACHE_FILENAME = "ndvi_baseline_median.tif"

logger = logging.getLogger(__name__)


def _build_baseline(bbox, config) -> xr.DataArray:
    """Download and compute the Landsat NDVI baseline median (1986 to year-1)."""
    from utils.stac import load_dea_landsat

    start = f"{config.BASELINE_START_YEAR}-01-01"
    end   = f"{config.YEAR - 1}-12-31"
    logger.info("Building Landsat baseline %s → %s", start, end)

    ds = load_dea_landsat(
        bbox=bbox,
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
    baseline = ndvi.median(dim="time", skipna=True).compute()
    baseline = baseline.rio.write_crs(config.TARGET_CRS)
    return baseline


def main() -> None:
    import config
    from utils.io import ensure_output_dirs, write_cog, read_raster
    from utils.quicklook import save_quicklook

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

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
        with dask.config.set(scheduler="synchronous"):
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
