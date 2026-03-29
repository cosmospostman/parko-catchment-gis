"""Tests for utils/verification.py."""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
import rioxarray  # noqa: F401

# sys.path is set by conftest.py


def _da(values, crs="EPSG:7844"):
    H, W = values.shape
    x = np.linspace(0, W - 1, W, dtype=np.float64)
    y = np.linspace(0, H - 1, H, dtype=np.float64)
    da = xr.DataArray(values, dims=["y", "x"], coords={"y": y, "x": x})
    da = da.rio.write_crs(crs)
    return da


def test_check_ndvi_range_passes():
    from utils.verification import check_ndvi_range
    da = _da(np.array([[0.5, 0.3], [0.1, -0.5]], dtype=np.float32))
    check_ndvi_range(da)  # should not raise


def test_check_ndvi_range_fails_high():
    from utils.verification import check_ndvi_range
    da = _da(np.array([[1.5, 0.3]], dtype=np.float32))
    with pytest.raises(AssertionError, match="outside"):
        check_ndvi_range(da)


def test_check_ndvi_range_fails_low():
    from utils.verification import check_ndvi_range
    da = _da(np.array([[-1.5, 0.3]], dtype=np.float32))
    with pytest.raises(AssertionError, match="outside"):
        check_ndvi_range(da)


def test_check_nan_fraction_passes():
    from utils.verification import check_nan_fraction
    values = np.ones((10, 10), dtype=np.float32)
    values[0, 0] = np.nan  # 1% NaN
    da = _da(values)
    check_nan_fraction(da, 0.20)


def test_check_nan_fraction_fails():
    from utils.verification import check_nan_fraction
    values = np.full((10, 10), np.nan, dtype=np.float32)
    values[0, 0] = 0.5  # 99% NaN
    da = _da(values)
    with pytest.raises(AssertionError, match="NaN fraction"):
        check_nan_fraction(da, 0.20)


def test_check_value_range_passes():
    from utils.verification import check_value_range
    da = _da(np.array([[0.5, 0.7], [0.2, 0.9]], dtype=np.float32))
    check_value_range(da, 0.0, 1.0)


def test_check_value_range_fails():
    from utils.verification import check_value_range
    da = _da(np.array([[0.5, 1.5]], dtype=np.float32))
    with pytest.raises(AssertionError, match="outside"):
        check_value_range(da, 0.0, 1.0)


def test_check_crs_passes():
    from utils.verification import check_crs
    da = _da(np.ones((3, 3), dtype=np.float32), crs="EPSG:7844")
    check_crs(da, "EPSG:7844")


def test_check_crs_fails():
    from utils.verification import check_crs
    da = _da(np.ones((3, 3), dtype=np.float32), crs="EPSG:4326")
    with pytest.raises(AssertionError, match="CRS mismatch"):
        check_crs(da, "EPSG:7844")


def test_check_geometry_validity_passes():
    from utils.verification import check_geometry_validity
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:7844")
    check_geometry_validity(gdf)


def test_check_geometry_validity_fails():
    from utils.verification import check_geometry_validity
    # Self-intersecting polygon (bowtie)
    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    gdf = gpd.GeoDataFrame(geometry=[invalid], crs="EPSG:7844")
    with pytest.raises(AssertionError, match="invalid geometries"):
        check_geometry_validity(gdf)


def test_check_catchment_median_passes():
    from utils.verification import check_catchment_median
    values = np.full((10, 10), 0.35, dtype=np.float32)
    da = _da(values)
    check_catchment_median(da, 0.15, 0.50)


def test_check_catchment_median_fails():
    from utils.verification import check_catchment_median
    values = np.full((10, 10), 0.01, dtype=np.float32)
    da = _da(values)
    with pytest.raises(AssertionError, match="median"):
        check_catchment_median(da, 0.15, 0.50)
