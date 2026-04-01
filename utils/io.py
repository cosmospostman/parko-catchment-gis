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
    """Write an xarray DataArray as a Cloud-Optimised GeoTIFF.

    Uses windowed block writes so memory usage is bounded by block size
    regardless of raster dimensions.
    """
    import rasterio
    from rasterio.transform import from_bounds
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise to 2-D (y, x)
    arr = da
    if arr.ndim == 3:
        arr = arr.squeeze()

    crs = da.rio.crs
    left, bottom, right, top = da.rio.bounds()
    height, width = arr.shape[-2], arr.shape[-1]
    transform = from_bounds(left, bottom, right, top, width, height)
    _nodata = nodata

    BLOCK = 2048
    profile = dict(
        driver="GTiff",
        dtype="float32",
        width=width,
        height=height,
        count=1,
        crs=crs,
        transform=transform,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=BLOCK,
        blockysize=BLOCK,
        BIGTIFF="YES",
        nodata=_nodata,
    )

    with rasterio.open(str(path), "w", **profile) as dst:
        for row_off in range(0, height, BLOCK):
            row_end = min(row_off + BLOCK, height)
            for col_off in range(0, width, BLOCK):
                col_end = min(col_off + BLOCK, width)
                window = rasterio.windows.Window(col_off, row_off,
                                                 col_end - col_off,
                                                 row_end - row_off)
                chunk = arr[..., row_off:row_end, col_off:col_end]
                # Materialise dask chunk if needed
                if hasattr(chunk, "compute"):
                    chunk = chunk.compute()
                data = np.asarray(chunk, dtype="float32")
                if data.ndim == 2:
                    data = data[np.newaxis]
                dst.write(data, window=window)

    with rasterio.open(str(path), "r+") as dst:
        min_dim = min(dst.width, dst.height)
        levels = [lv for lv in [2, 4, 8, 16, 32] if lv < min_dim]
        if levels:
            dst.build_overviews(levels, rasterio.enums.Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")
    logger.info("Written COG: %s", path)


def read_raster(path: Path) -> xr.DataArray:
    """Read a GeoTIFF as an xarray DataArray with rioxarray."""
    path = Path(path)
    da = rioxarray.open_rasterio(str(path), masked=True)
    logger.debug("Read raster: %s  shape=%s  crs=%s", path, da.shape, da.rio.crs)
    return da
