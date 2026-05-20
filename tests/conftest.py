"""Shared pytest fixtures for the Parkinsonia GIS pipeline test suite."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import json
from shapely.geometry import box, mapping
import rioxarray  # noqa: F401 — registers .rio accessor

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Env / directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dirs(tmp_path, monkeypatch):
    """Set all required env vars to tmp_path subdirs and return their paths."""
    dirs = {
        "BASE_DIR":     tmp_path / "base",
        "CACHE_DIR":    tmp_path / "cache",
        "WORKING_DIR":  tmp_path / "working",
        "OUTPUTS_DIR":  tmp_path / "outputs",
        "CODE_DIR":     tmp_path / "base",
        "LOG_DIR":      tmp_path / "outputs" / "logs",
    }
    for key, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(path))

    monkeypatch.setenv("YEAR", "2025")
    monkeypatch.setenv("COMPOSITE_START", "05-01")
    monkeypatch.setenv("COMPOSITE_END", "10-31")

    # Place a synthetic catchment GeoJSON
    catchment_path = tmp_path / "base" / "mitchell_catchment.geojson"
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "mitchell"},
            "geometry": mapping(box(141.0, -17.0, 143.0, -15.0)),
        }],
    }
    catchment_path.write_text(json.dumps(geojson))
    monkeypatch.setenv("CATCHMENT_GEOJSON", str(catchment_path))

    # Reload config with new env vars
    import importlib
    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])

    yield dirs

    # Teardown: nothing needed — tmp_path is automatically cleaned up


# ---------------------------------------------------------------------------
# Synthetic raster fixtures
# ---------------------------------------------------------------------------

def _make_raster(values: np.ndarray, crs: str = "EPSG:7844") -> xr.DataArray:
    """Wrap a numpy array in a georeferenced DataArray."""
    H, W = values.shape
    x = np.linspace(700000, 750000, W)
    y = np.linspace(-1600000, -1650000, H)  # negative = south
    da = xr.DataArray(
        values,
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )
    da = da.rio.write_crs(crs)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


@pytest.fixture()
def synthetic_ndvi_raster() -> xr.DataArray:
    """50×50 NDVI raster, values from N(0.35, 0.08), seeded with rng=42."""
    rng = np.random.default_rng(42)
    values = rng.normal(0.35, 0.08, size=(50, 50)).astype(np.float32)
    values = np.clip(values, -1.0, 1.0)
    return _make_raster(values)


@pytest.fixture()
def synthetic_anomaly_raster() -> xr.DataArray:
    """50×50 NDVI anomaly raster, values near zero with small std."""
    rng = np.random.default_rng(42)
    values = rng.normal(0.01, 0.08, size=(50, 50)).astype(np.float32)
    values = np.clip(values, -1.0, 1.0)
    return _make_raster(values)


@pytest.fixture()
def synthetic_probability_raster() -> xr.DataArray:
    """50×50 probability raster, values from Beta(2,5)."""
    rng = np.random.default_rng(42)
    values = rng.beta(2, 5, size=(50, 50)).astype(np.float32)
    return _make_raster(values)


