"""utils/verification.py — assertion helpers shared by verify scripts and tests.

All functions raise AssertionError with descriptive messages on failure.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def check_ndvi_range(da: xr.DataArray, name: str = "NDVI") -> None:
    """Assert all non-NaN values are in [-1, 1]."""
    valid = da.values[~np.isnan(da.values)]
    if len(valid) == 0:
        raise AssertionError(f"{name}: no valid (non-NaN) pixels found")
    vmin, vmax = float(valid.min()), float(valid.max())
    if vmin < -1.0 or vmax > 1.0:
        raise AssertionError(
            f"{name}: values outside [-1, 1] range — min={vmin:.4f}, max={vmax:.4f}"
        )
    logger.debug("%s range check passed: min=%.4f max=%.4f", name, vmin, vmax)


def check_nan_fraction(da: xr.DataArray, max_fraction: float, name: str = "raster") -> None:
    """Assert NaN fraction does not exceed max_fraction."""
    total = da.size
    nan_count = int(np.isnan(da.values).sum())
    fraction = nan_count / total
    if fraction > max_fraction:
        raise AssertionError(
            f"{name}: NaN fraction {fraction:.3f} exceeds maximum {max_fraction:.3f} "
            f"({nan_count}/{total} pixels)"
        )
    logger.debug("%s NaN fraction check passed: %.3f <= %.3f", name, fraction, max_fraction)


def check_value_range(
    da: xr.DataArray,
    min_val: float,
    max_val: float,
    name: str = "raster",
) -> None:
    """Assert all non-NaN values are within [min_val, max_val]."""
    valid = da.values[~np.isnan(da.values)]
    if len(valid) == 0:
        raise AssertionError(f"{name}: no valid (non-NaN) pixels found")
    vmin, vmax = float(valid.min()), float(valid.max())
    if vmin < min_val or vmax > max_val:
        raise AssertionError(
            f"{name}: values outside [{min_val}, {max_val}] — min={vmin:.4f}, max={vmax:.4f}"
        )
    logger.debug("%s value range check passed: [%.4f, %.4f]", name, vmin, vmax)


def check_crs(da: xr.DataArray, expected_crs: str, name: str = "raster") -> None:
    """Assert the DataArray has the expected CRS."""
    actual = str(da.rio.crs)

    def _normalise(s: str) -> str:
        return s.upper().replace(" ", "")

    if _normalise(actual) != _normalise(expected_crs):
        raise AssertionError(
            f"{name}: CRS mismatch — expected {expected_crs}, got {actual}"
        )
    logger.debug("%s CRS check passed: %s", name, actual)


def check_geometry_validity(gdf: "gpd.GeoDataFrame", name: str = "geodataframe") -> None:  # noqa: F821
    """Assert all geometries in a GeoDataFrame are valid."""
    import geopandas as gpd

    invalid = gdf[~gdf.geometry.is_valid]
    if len(invalid) > 0:
        raise AssertionError(
            f"{name}: {len(invalid)} invalid geometries found out of {len(gdf)}"
        )
    logger.debug("%s geometry validity check passed: %d features", name, len(gdf))


def check_catchment_median(
    da: xr.DataArray,
    min_val: float,
    max_val: float,
    name: str = "NDVI",
) -> None:
    """Assert the catchment median value is within expected range."""
    valid = da.values[~np.isnan(da.values)]
    if len(valid) == 0:
        raise AssertionError(f"{name}: no valid pixels for median check")
    median = float(np.median(valid))
    if median < min_val or median > max_val:
        raise AssertionError(
            f"{name}: catchment median {median:.4f} outside expected range "
            f"[{min_val}, {max_val}]"
        )
    logger.debug(
        "%s catchment median check passed: %.4f in [%.4f, %.4f]",
        name,
        median,
        min_val,
        max_val,
    )
