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


_COP_DEM_VRT = (
    "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_mosaic_WGS84.vrt"
)


def _safe_root_from_item(item: Any) -> str:
    """Return the SAFE root URL/path from a pystac Item.

    Derives the root from the 'safe-manifest' asset href by stripping
    '/manifest.safe', falling back to the item's self href or assets directory.
    """
    manifest_asset = item.assets.get("safe-manifest")
    if manifest_asset:
        href = manifest_asset.href
        if href.endswith("/manifest.safe"):
            return href[: -len("/manifest.safe")]
    # Fallback: derive from vv asset href (strip measurement subpath)
    vv_asset = item.assets.get("vv")
    if vv_asset:
        href = vv_asset.href
        # …/measurement/iw-vv.tiff → strip two components
        return str(Path(href).parent.parent)
    raise ValueError(f"Cannot determine SAFE root for item {item.id}")


def _preprocess_with_sarsen(item: Any, bbox: list, resolution: int) -> xr.Dataset:
    """Terrain-corrected preprocessing using sarsen."""
    import sarsen

    safe_root = _safe_root_from_item(item)
    logger.debug("sarsen SAFE root: %s", safe_root)

    bands = {}
    for pol in ("VV", "VH"):
        product = sarsen.Sentinel1SarProduct(
            product_urlpath=safe_root,
            measurement_group=f"IW/{pol}",
        )
        da = sarsen.terrain_correction(
            product,
            dem_urlpath=_COP_DEM_VRT,
            output_urlpath=None,
            correct_radiometry="gamma_nearest",
        )
        bands[pol] = da

    ds = xr.Dataset(bands)
    logger.info("sarsen terrain-corrected S1 scene: shape=%s", ds["VV"].shape)
    return ds


def _preprocess_stackstac_fallback(item: Any, bbox: list, resolution: int) -> xr.Dataset:
    """Basic S1 loading via stackstac without terrain correction."""
    import stackstac

    da = stackstac.stack(
        [item],
        assets=["vv", "vh"],
        resolution=resolution,
        bounds_latlon=bbox,
    )
    ds = da.to_dataset(dim="band")
    # Normalise band names to uppercase to match downstream expectations
    ds = ds.rename({k: k.upper() for k in ds.data_vars if k in ("vv", "vh")})
    # Convert from dB to linear scale if needed
    for var in ds.data_vars:
        ds[var] = 10 ** (ds[var] / 10)
    logger.info("Stackstac S1 fallback loaded: %s", list(ds.data_vars))
    return ds
