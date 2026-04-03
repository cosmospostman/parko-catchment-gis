"""Shared pytest fixtures for the Parkinsonia GIS pipeline test suite."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import box, Point, Polygon
import rioxarray  # noqa: F401 — registers .rio accessor

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Pipeline source files whose modification should trigger a test data refresh.
# If any of these files changed after the sentinel was written, test data is
# considered stale and the pipeline may not reflect current code.
# ---------------------------------------------------------------------------

_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
_SENTINEL_FILE = _FIXTURE_DIR / ".fixture_commit"

_PIPELINE_SOURCES = [
    "stage0/fetch.py",
    "stage0/chip_store.py",
    "analysis/constants.py",
    "analysis/timeseries/observation.py",
]


def _git_diff_names(since_commit: str) -> list[str]:
    """Return list of files changed since the given commit (relative paths)."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", since_commit, "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return [line.strip() for line in out.decode().splitlines() if line.strip()]
    except Exception:
        return []


def pytest_configure(config: pytest.Config) -> None:
    """Warn if fixture test data is missing or stale.

    Missing sentinel → warn (data not yet staged; tests that need real chips
    will fail with FileNotFoundError rather than a confusing collection error).

    Stale sentinel → warn (pipeline sources changed since load-testdata ran).

    We warn rather than exit so the full unit test suite (which uses synthetic
    fixtures) continues to run during normal development. Only tests that
    explicitly require staged chip data will fail when data is absent.

    To refresh: python pipelines/train.py load-testdata
    """
    if not _SENTINEL_FILE.exists():
        print(
            "\n[conftest] WARNING: fixture test data not staged.\n"
            "  Some tests require real chip data. Run:\n"
            "    python pipelines/train.py load-testdata\n",
            file=sys.stderr,
        )
        return

    recorded_commit = _SENTINEL_FILE.read_text().strip()
    if not recorded_commit or recorded_commit == "unknown":
        return

    changed = _git_diff_names(recorded_commit)
    stale = [f for f in changed if f in _PIPELINE_SOURCES]
    if stale:
        print(
            f"\n[conftest] WARNING: test data may be stale — pipeline sources "
            f"changed since load-testdata ran:\n"
            + "".join(f"  {f}\n" for f in stale)
            + "  Run: python pipelines/train.py load-testdata\n",
            file=sys.stderr,
        )


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
    catchment_gdf = gpd.GeoDataFrame(
        {"name": ["mitchell"]},
        geometry=[box(141.0, -17.0, 143.0, -15.0)],
        crs="EPSG:4326",
    )
    catchment_gdf.to_file(str(catchment_path), driver="GeoJSON")
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


@pytest.fixture()
def synthetic_flood_gdf() -> gpd.GeoDataFrame:
    """Small GeoDataFrame with a single flood polygon."""
    poly = Polygon([(700000, -1600000), (710000, -1600000),
                    (710000, -1610000), (700000, -1610000)])
    return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:7844")


@pytest.fixture()
def synthetic_patches_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame with two priority patches in different tiers."""
    poly_a = Polygon([(700000, -1600000), (700500, -1600000),
                      (700500, -1600500), (700000, -1600500)])
    poly_b = Polygon([(705000, -1600000), (706000, -1600000),
                      (706000, -1601000), (705000, -1601000)])
    return gpd.GeoDataFrame(
        {
            "tier": ["A", "B"],
            "area_ha": [0.25, 1.0],
            "prob_mean": [0.90, 0.78],
            "prob_max":  [0.95, 0.85],
            "dist_to_kowanyama_km": [100.0, 120.0],
            "seed_flux_score": [0.72, 0.50],
            "stream_order": [3, 2],
        },
        geometry=[poly_a, poly_b],
        crs="EPSG:7844",
    )
