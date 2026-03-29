"""Tests for analysis/07_change_detection.py — including Year 1 path."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import rioxarray  # noqa: F401

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_prob_raster(path: Path, values: np.ndarray, crs_epsg: int = 7844):
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    H, W = values.shape
    transform = from_bounds(700000, -1650000, 750000, -1600000, W, H)
    with rasterio.open(
        str(path), "w",
        driver="GTiff", dtype="float32", count=1,
        width=W, height=H, transform=transform,
        crs=CRS.from_epsg(crs_epsg),
    ) as dst:
        dst.write(values[np.newaxis])


def test_year1_run_no_prior_raster(tmp_dirs):
    """Year 1 run: no prior raster → exits 0, no output file created."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    prior_path = config.probability_raster_path(config.YEAR - 1)
    assert not prior_path.exists()

    cur_path = config.probability_raster_path(config.YEAR)
    cur_path.parent.mkdir(parents=True, exist_ok=True)
    _write_prob_raster(cur_path, np.full((10, 10), 0.5, dtype=np.float32))

    script = PROJECT_ROOT / "analysis" / "07_change_detection.py"
    mod = _load_module(script, "step07")
    mod.main()

    out_path = config.change_detection_path(config.YEAR)
    assert not out_path.exists()


def test_crs_mismatch_raises(tmp_dirs):
    """CRS mismatch between years must raise ValueError."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    cur_path = config.probability_raster_path(config.YEAR)
    cur_path.parent.mkdir(parents=True, exist_ok=True)
    _write_prob_raster(cur_path, np.full((10, 10), 0.5, dtype=np.float32), crs_epsg=7844)

    prior_path = config.probability_raster_path(config.YEAR - 1)
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    _write_prob_raster(prior_path, np.full((10, 10), 0.4, dtype=np.float32), crs_epsg=4326)

    script = PROJECT_ROOT / "analysis" / "07_change_detection.py"
    mod = _load_module(script, "step07_crs")
    with pytest.raises(ValueError, match="CRS mismatch"):
        mod.main()
