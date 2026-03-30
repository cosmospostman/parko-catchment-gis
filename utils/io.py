"""utils/io.py — raster I/O helpers for the Parkinsonia GIS pipeline."""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging for pipeline scripts.

    - Standard timestamped format to stdout
    - Suppresses noisy third-party warnings that are not actionable
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )
    # rasterio emits repeated CPLE_NotSupported warnings about unsupported warp
    # options (SHARING, WARP_EXTRAS) — cosmetic only, not actionable.
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)


def ensure_output_dirs(year: int) -> None:
    """Create all required output directories for the given year."""
    import config
    dirs = [
        Path(config.CACHE_DIR),
        Path(config.WORKING_DIR),
        Path(config.OUTPUTS_DIR) / str(year),
        Path(config.LOG_DIR),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", d)


def write_cog(da: xr.DataArray, path: Path, nodata: float = np.nan) -> None:
    """Write an xarray DataArray as a Cloud-Optimised GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    da.rio.to_raster(
        str(path),
        driver="GTiff",
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        overviews="auto",
    )
    logger.info("Written COG: %s", path)


def read_raster(path: Path) -> xr.DataArray:
    """Read a GeoTIFF as an xarray DataArray with rioxarray."""
    path = Path(path)
    da = rioxarray.open_rasterio(str(path), masked=True)
    logger.debug("Read raster: %s  shape=%s  crs=%s", path, da.shape, da.rio.crs)
    return da
