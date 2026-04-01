"""Unit tests for analysis/03_flowering_index.py — tiled processing path."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_stack(n_scenes=3, height=2, width=2, bands=None):
    """Tiny (n_scenes, n_bands, H, W) DataArray mimicking stackstac output."""
    if bands is None:
        bands = ["green", "nir", "rededge1", "rededge2", "scl"]

    rng = np.random.default_rng(1)
    data = rng.uniform(0.05, 0.5, size=(n_scenes, len(bands), height, width)).astype(np.float32)
    if "scl" in bands:
        data[:, bands.index("scl"), :, :] = 4.0  # clear

    times = np.array([
        np.datetime64("2025-08-01"),
        np.datetime64("2025-09-01"),
        np.datetime64("2025-10-01"),
    ])[:n_scenes]

    x = np.linspace(700000.0, 700020.0, width)
    y = np.linspace(-1600000.0, -1600020.0, height)

    da = xr.DataArray(
        data,
        dims=["time", "band", "y", "x"],
        coords={"time": times, "band": bands, "y": y, "x": x},
    )
    da = da.rio.write_crs("EPSG:7855")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


def _fake_load_stackstac(items, bands, resolution, bbox, crs, chunk_spatial=1024):
    return _make_synthetic_stack(bands=bands)


def test_output_has_correct_shape_and_crs(tmp_dirs, monkeypatch):
    """Output raster must be 2-D with TARGET_CRS after tiled processing."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    dummy_item = MagicMock()
    dummy_item.bbox = [141.0, -17.0, 143.0, -15.0]

    with patch("utils.stac.search_sentinel2", return_value=[dummy_item]), \
         patch("utils.stac.load_stackstac", _fake_load_stackstac), \
         patch("utils.tiling.make_tile_bboxes",
               return_value=[[141.0, -17.0, 143.0, -15.0]]), \
         patch("utils.mask.apply_scl_mask", side_effect=lambda stack, scl: stack), \
         patch("utils.quicklook.save_quicklook"), \
         patch("xarray.open_dataarray",
               return_value=_make_synthetic_stack(n_scenes=1, height=2, width=2,
                                                  bands=["nir"]).isel(time=0, band=0)):

        script = PROJECT_ROOT / "analysis" / "03_flowering_index.py"
        _load_module(script, "step03_shape").main()

    out_path = config.flowering_index_path(config.YEAR)
    assert out_path.exists(), f"Expected output at {out_path}"

    result = xr.open_dataarray(str(out_path))
    assert result.ndim == 2 or (result.ndim == 3 and result.shape[0] == 1)
    assert result.rio.crs is not None
    assert result.rio.crs.to_epsg() == 7855


def test_output_values_in_valid_range(tmp_dirs, monkeypatch):
    """Non-NaN output values must be ≥ 0 and the output must contain valid pixels."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    dummy_item = MagicMock()
    dummy_item.bbox = [141.0, -17.0, 143.0, -15.0]

    with patch("utils.stac.search_sentinel2", return_value=[dummy_item]), \
         patch("utils.stac.load_stackstac", _fake_load_stackstac), \
         patch("utils.tiling.make_tile_bboxes",
               return_value=[[141.0, -17.0, 143.0, -15.0]]), \
         patch("utils.mask.apply_scl_mask", side_effect=lambda stack, scl: stack), \
         patch("utils.quicklook.save_quicklook"), \
         patch("xarray.open_dataarray",
               return_value=_make_synthetic_stack(n_scenes=1, height=2, width=2,
                                                  bands=["nir"]).isel(time=0, band=0)):

        script = PROJECT_ROOT / "analysis" / "03_flowering_index.py"
        _load_module(script, "step03_values").main()

    out_path = config.flowering_index_path(config.YEAR)
    result = xr.open_dataarray(str(out_path))
    data = result.values
    valid = data[~np.isnan(data)]
    assert len(valid) > 0, "Output must contain at least one valid (non-NaN) pixel"
    assert np.all(valid >= 0.0), "Green/NIR ratio should be non-negative"
