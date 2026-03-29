"""utils/sar.py — SAR preprocessing wrapper.

Isolated so tests can mock preprocess_s1_scene() without importing sarsen.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def preprocess_s1_scene(
    item: Any,
    bbox: list,
    resolution: int = 10,
) -> xr.Dataset:
    """Preprocess a Sentinel-1 GRD scene to sigma-naught (linear scale).

    Attempts to use sarsen for terrain-corrected processing.
    Falls back to direct stackstac loading with a warning if sarsen is not available.
    """
    try:
        import sarsen
        _SARSEN_AVAILABLE = True
    except ImportError:
        _SARSEN_AVAILABLE = False
        logger.warning(
            "sarsen not available — falling back to non-terrain-corrected S1 loading. "
            "Flood mapping accuracy may be reduced in hilly terrain."
        )

    if _SARSEN_AVAILABLE:
        return _preprocess_with_sarsen(item, bbox, resolution)
    else:
        return _preprocess_stackstac_fallback(item, bbox, resolution)


def _preprocess_with_sarsen(item: Any, bbox: list, resolution: int) -> xr.Dataset:
    """Terrain-corrected preprocessing using sarsen."""
    import sarsen

    # sarsen terrain correction using SRTM DEM
    result = sarsen.process_sentinel1_grd(
        item,
        resolution=resolution,
        bbox_latlon=bbox,
        output_polarisations=["VV", "VH"],
        dem_urlpath="cop-dem-glo-30",
    )
    logger.info("sarsen terrain-corrected S1 scene: shape=%s", result["VV"].shape)
    return result


def _preprocess_stackstac_fallback(item: Any, bbox: list, resolution: int) -> xr.Dataset:
    """Basic S1 loading via stackstac without terrain correction."""
    import stackstac

    da = stackstac.stack(
        [item],
        assets=["VV", "VH"],
        resolution=resolution,
        bounds_latlon=bbox,
    )
    ds = da.to_dataset(dim="band")
    # Convert from dB to linear scale if needed
    for var in ds.data_vars:
        ds[var] = 10 ** (ds[var] / 10)
    logger.info("Stackstac S1 fallback loaded: %s", list(ds.data_vars))
    return ds
