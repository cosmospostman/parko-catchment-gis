"""utils/mask.py — cloud/shadow and habitat masking helpers."""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# SCL class values that represent clear observations
SCL_CLEAR_CLASSES = [4, 5, 6]  # vegetation, bare soil, water


def apply_scl_mask(da: xr.DataArray, scl: xr.DataArray) -> xr.DataArray:
    """Mask a DataArray using Sentinel-2 Scene Classification Layer (SCL).

    Pixels not in SCL_CLEAR_CLASSES are set to NaN.
    """
    clear_mask = xr.zeros_like(scl, dtype=bool)
    for cls in SCL_CLEAR_CLASSES:
        clear_mask = clear_mask | (scl == cls)
    masked = da.where(clear_mask)
    frac_masked = float((~clear_mask).mean())
    logger.debug("SCL mask: %.1f%% of pixels masked", frac_masked * 100)
    return masked


def apply_s2cloudless_mask(
    da: xr.DataArray,
    cloud_prob: xr.DataArray,
    threshold: float = 0.4,
) -> xr.DataArray:
    """Mask using s2cloudless cloud probability array."""
    clear_mask = cloud_prob < threshold
    masked = da.where(clear_mask)
    frac_masked = float((~clear_mask).mean())
    logger.debug(
        "s2cloudless mask (thresh=%.2f): %.1f%% masked",
        threshold,
        frac_masked * 100,
    )
    return masked


def apply_habitat_mask(da: xr.DataArray, habitat_mask: xr.DataArray) -> xr.DataArray:
    """Mask out pixels outside the habitat of interest."""
    return da.where(habitat_mask.astype(bool))
