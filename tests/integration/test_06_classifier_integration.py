"""Integration tests for analysis/06_classifier.py — end-to-end main() execution.

All tests use fully synthetic raster and vector inputs derived from the conftest
catchment (box 141–143 lon, -17 to -15 lat, EPSG:4326), projected to EPSG:7855.
No network calls or real data are required.

The catchment projects to approximately:
  x: -146075 to 73938
  y:  8110627 to 8337782  (EPSG:7855 / GDA2020 MGA zone 55)
"""
import importlib.util
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import Point, box

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT = PROJECT_ROOT / "analysis" / "06_classifier.py"

# EPSG:7855 bounds of the conftest catchment (box 141–143, -17 to -15 in WGS84)
# Computed via: gpd.GeoDataFrame(geometry=[box(141,-17,143,-15)], crs=4326).to_crs(7855).total_bounds
CATCHMENT_BOUNDS_7855 = (-146075, 8110627, 73938, 8337782)


def _load_module(module_name: str = "classifier05_int"):
    spec = importlib.util.spec_from_file_location(module_name, str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_raster(path: Path, values: np.ndarray, bounds=None, crs: str = "EPSG:7855") -> None:
    """Write a float32 GeoTIFF aligned to the synthetic catchment bounds."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    if bounds is None:
        bounds = CATCHMENT_BOUNDS_7855
    H, W = values.shape
    transform = from_bounds(*bounds, W, H)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(path), "w", driver="GTiff", dtype="float32", count=1,
                       width=W, height=H, transform=transform,
                       crs=CRS.from_epsg(int(crs.split(":")[1]))) as dst:
        dst.write(values.astype(np.float32)[np.newaxis])


def _write_inputs(config, H=40, W=40):
    rng = np.random.default_rng(42)
    for path_fn in [config.ndvi_anomaly_path, config.flowering_index_path, config.ndvi_median_path]:
        _write_raster(path_fn(config.YEAR), rng.uniform(0.1, 0.8, (H, W)))


def _write_ala_cache(config, n=8):
    """Write ALA occurrence points inside the projected catchment extent."""
    ala_cache = Path(config.CACHE_DIR) / "ala_occurrences.gpkg"
    ala_cache.parent.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = CATCHMENT_BOUNDS_7855
    rng = np.random.default_rng(0)
    xs = rng.uniform(minx + 1000, maxx - 1000, n)
    ys = rng.uniform(miny + 1000, maxy - 1000, n)
    pts = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs="EPSG:7855",
    ).to_crs("EPSG:4326")
    pts.to_file(str(ala_cache), driver="GPKG")


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def setup(tmp_dirs):
    """Reload config and write all synthetic inputs."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config
    _write_inputs(config)
    _write_ala_cache(config)
    return config


# ── tests ─────────────────────────────────────────────────────────────────────

def test_output_raster_exists_and_is_2d(setup):
    """main() must produce a 2-D probability raster at the expected path."""
    config = setup
    mod = _load_module("int_shape")
    with patch("sklearn.model_selection.cross_val_score", return_value=np.array([0.9] * 5)), \
         patch("utils.quicklook.save_quicklook"):
        mod.main()

    out = config.probability_raster_path(config.YEAR)
    assert out.exists(), f"Expected output at {out}"
    result = xr.open_dataarray(str(out)).squeeze()
    assert result.ndim == 2


def test_output_values_in_zero_one(setup):
    """All non-NaN output values must be in [0, 1]."""
    config = setup
    mod = _load_module("int_range")
    with patch("sklearn.model_selection.cross_val_score", return_value=np.array([0.9] * 5)), \
         patch("utils.quicklook.save_quicklook"):
        mod.main()

    result = xr.open_dataarray(str(config.probability_raster_path(config.YEAR)))
    vals = result.values.ravel()
    vals = vals[~np.isnan(vals)]
    assert len(vals) > 0, "Output raster has no valid pixels"
    assert np.all(vals >= 0.0), f"Min value {vals.min():.4f} < 0"
    assert np.all(vals <= 1.0), f"Max value {vals.max():.4f} > 1"


def test_output_crs_is_target(setup):
    """Output raster must be in TARGET_CRS (EPSG:7855)."""
    config = setup
    mod = _load_module("int_crs")
    with patch("sklearn.model_selection.cross_val_score", return_value=np.array([0.9] * 5)), \
         patch("utils.quicklook.save_quicklook"):
        mod.main()

    result = xr.open_dataarray(str(config.probability_raster_path(config.YEAR)))
    assert result.rio.crs is not None
    assert result.rio.crs.to_epsg() == 7855


def test_model_cache_written(setup):
    """main() must write a model cache pickle with model, feature_names, and cv_scores."""
    config = setup
    mod = _load_module("int_cache")
    with patch("sklearn.model_selection.cross_val_score", return_value=np.array([0.9] * 5)), \
         patch("utils.quicklook.save_quicklook"):
        mod.main()

    cache_path = Path(config.CACHE_DIR) / f"rf_model_{config.YEAR}.pkl"
    assert cache_path.exists()
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    assert "model" in cached
    assert "feature_names" in cached
    assert "cv_scores" in cached


def test_no_ala_records_raises(tmp_dirs):
    """main() must raise RuntimeError when no ALA records fall within the raster extent."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    _write_inputs(config)

    # Points in Western Australia — outside the synthetic catchment
    ala_cache = Path(config.CACHE_DIR) / "ala_occurrences.gpkg"
    ala_cache.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        geometry=[Point(120.0, -30.0)],
        crs="EPSG:4326",
    ).to_file(str(ala_cache), driver="GPKG")

    mod = _load_module("int_norecords")
    with patch("sklearn.model_selection.cross_val_score", return_value=np.array([0.9] * 5)), \
         patch("utils.quicklook.save_quicklook"), \
         pytest.raises(RuntimeError, match="No ALA occurrence records found"):
        mod.main()


def test_output_shape_matches_input_raster(setup):
    """Output probability raster must have the same spatial shape as the input rasters."""
    config = setup
    H, W = 40, 40
    mod = _load_module("int_shape2")
    with patch("sklearn.model_selection.cross_val_score", return_value=np.array([0.9] * 5)), \
         patch("utils.quicklook.save_quicklook"):
        mod.main()

    result = xr.open_dataarray(str(config.probability_raster_path(config.YEAR))).squeeze()
    assert result.shape == (H, W), f"Expected ({H}, {W}), got {result.shape}"
