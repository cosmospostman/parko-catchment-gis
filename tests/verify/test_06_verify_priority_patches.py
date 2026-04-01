"""Tests for verify/06_verify_priority_patches.py."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_patches(path: Path, gdf: gpd.GeoDataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(path), driver="GPKG")


def test_single_tier_fails(tmp_dirs):
    """Output with only one tier value must fail."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    poly = Polygon([(700000, -1600000), (700500, -1600000),
                    (700500, -1600500), (700000, -1600500)])
    gdf = gpd.GeoDataFrame({
        "tier": ["A"], "area_ha": [0.5], "prob_mean": [0.90], "prob_max": [0.95],
        "dist_to_kowanyama_km": [100.0], "seed_flux_score": [0.8], "stream_order": [3],
    }, geometry=[poly], crs="EPSG:7844")
    _write_patches(config.priority_patches_path(config.YEAR), gdf)

    script = PROJECT_ROOT / "verify" / "06_verify_priority_patches.py"
    mod = _load_module(script, "verify06_single_tier")
    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 2


def test_valid_patches_pass(tmp_dirs, synthetic_patches_gdf):
    """Valid two-tier patches should pass all checks."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    _write_patches(config.priority_patches_path(config.YEAR), synthetic_patches_gdf)

    script = PROJECT_ROOT / "verify" / "06_verify_priority_patches.py"
    mod = _load_module(script, "verify06_valid")
    mod.main()  # should not raise


def test_below_min_area_fails(tmp_dirs):
    """Patches with area_ha below MIN_PATCH_AREA_HA (0.25 ha) must fail."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    poly_a = Polygon([(700000, -1600000), (700500, -1600000),
                      (700500, -1600500), (700000, -1600500)])
    poly_b = Polygon([(701000, -1600000), (701500, -1600000),
                      (701500, -1600500), (701000, -1600500)])
    gdf = gpd.GeoDataFrame({
        "tier": ["A", "B"],
        "area_ha": [0.50, 0.10],  # second patch is below 0.25 ha threshold
        "prob_mean": [0.90, 0.75], "prob_max": [0.95, 0.82],
        "dist_to_kowanyama_km": [100.0, 110.0],
        "seed_flux_score": [0.8, 0.5], "stream_order": [3, 2],
    }, geometry=[poly_a, poly_b], crs="EPSG:7844")
    _write_patches(config.priority_patches_path(config.YEAR), gdf)

    script = PROJECT_ROOT / "verify" / "06_verify_priority_patches.py"
    mod = _load_module(script, "verify06_min_area")
    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 2


def test_wrong_crs_fails(tmp_dirs):
    """Patches with wrong CRS must fail."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    poly_a = Polygon([(141.0, -16.0), (141.1, -16.0), (141.1, -16.1), (141.0, -16.1)])
    poly_b = Polygon([(142.0, -16.0), (142.1, -16.0), (142.1, -16.1), (142.0, -16.1)])
    gdf = gpd.GeoDataFrame({
        "tier": ["A", "B"], "area_ha": [1.0, 2.0], "prob_mean": [0.9, 0.75],
        "prob_max": [0.95, 0.82], "dist_to_kowanyama_km": [50.0, 60.0],
        "seed_flux_score": [0.7, 0.5], "stream_order": [3, 2],
    }, geometry=[poly_a, poly_b], crs="EPSG:4326")
    _write_patches(config.priority_patches_path(config.YEAR), gdf)

    script = PROJECT_ROOT / "verify" / "06_verify_priority_patches.py"
    mod = _load_module(script, "verify06_wrong_crs")
    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 2
