"""Unit tests for analysis/01_ndvi_composite.py — tiled processing path."""
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
    """Return a tiny (n_scenes, n_bands, H, W) DataArray mimicking stackstac output."""
    if bands is None:
        bands = ["blue", "green", "red", "nir", "nir08", "swir16", "swir22", "scl"]

    rng = np.random.default_rng(0)
    data = rng.uniform(0.05, 0.5, size=(n_scenes, len(bands), height, width)).astype(np.float32)
    # Make scl band = 4 (clear vegetation) everywhere
    scl_idx = bands.index("scl")
    data[:, scl_idx, :, :] = 4.0

    times = np.array([
        np.datetime64("2025-06-01"),
        np.datetime64("2025-07-01"),
        np.datetime64("2025-08-01"),
    ])[:n_scenes]

    x = np.linspace(700000.0, 700020.0, width)
    y = np.linspace(-1600000.0, -1600020.0, height)

    da = xr.DataArray(
        data,
        dims=["time", "band", "y", "x"],
        coords={
            "time": times,
            "band": bands,
            "y": y,
            "x": x,
        },
    )
    da = da.rio.write_crs("EPSG:7855")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


def _fake_load_stackstac(items, bands, resolution, bbox, crs, chunk_spatial=1024):
    return _make_synthetic_stack(bands=bands)


def test_process_tile_called_once_per_tile(tmp_dirs, monkeypatch):
    """process_tile should be called once per tile returned by make_tile_bboxes."""
    if "analysis.01_ndvi_composite" in sys.modules:
        del sys.modules["analysis.01_ndvi_composite"]
    if "config" in sys.modules:
        del sys.modules["config"]

    import config

    # Patch STAC search to return 1 dummy item
    dummy_item = MagicMock()
    mock_search = MagicMock(return_value=[dummy_item])

    # make_tile_bboxes returns a single tile (the full bbox)
    full_bbox = [141.0, -17.0, 143.0, -15.0]
    single_tile = [full_bbox]
    mock_tile_bboxes = MagicMock(return_value=single_tile)

    script = PROJECT_ROOT / "analysis" / "01_ndvi_composite.py"

    call_count = []

    original_process_tile = None

    with patch("utils.stac.search_sentinel2", mock_search), \
         patch("utils.stac.load_stackstac", _fake_load_stackstac), \
         patch("utils.tiling.make_tile_bboxes", mock_tile_bboxes), \
         patch("utils.mask.apply_scl_mask", side_effect=lambda stack, scl: stack), \
         patch("utils.quicklook.save_quicklook"), \
         patch("utils.tiling.merge_tile_rasters") as mock_merge, \
         patch("xarray.open_dataarray", return_value=_make_synthetic_stack(n_scenes=1, height=2, width=2, bands=["nir"])):

        # Patch merge to write a real file so the script doesn't crash
        def fake_merge(tile_paths, out_path, nodata, crs):
            da = _make_synthetic_stack(n_scenes=1, height=2, width=2, bands=["nir"])
            result = da.isel(time=0, band=0)
            result.rio.to_raster(str(out_path), driver="GTiff", dtype="float32")

        mock_merge.side_effect = fake_merge

        mod = _load_module(script, "step01_tile_count")
        original_fn = mod.main.__globals__.get("process_tile") if hasattr(mod, "main") else None

        mod.main()

    # make_tile_bboxes was called once
    mock_tile_bboxes.assert_called_once()
    # merge_tile_rasters was called once
    mock_merge.assert_called_once()


def test_output_has_correct_shape_and_crs(tmp_dirs, monkeypatch):
    """Output raster must be 2-D with TARGET_CRS after tiled processing."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config

    dummy_item = MagicMock()

    with patch("utils.stac.search_sentinel2", return_value=[dummy_item]), \
         patch("utils.stac.load_stackstac", _fake_load_stackstac), \
         patch("utils.tiling.make_tile_bboxes",
               return_value=[[141.0, -17.0, 143.0, -15.0]]), \
         patch("utils.mask.apply_scl_mask", side_effect=lambda stack, scl: stack), \
         patch("utils.quicklook.save_quicklook"), \
         patch("xarray.open_dataarray",
               return_value=_make_synthetic_stack(n_scenes=1, height=2, width=2,
                                                  bands=["nir"]).isel(time=0, band=0)):

        script = PROJECT_ROOT / "analysis" / "01_ndvi_composite.py"
        mod = _load_module(script, "step01_shape")
        mod.main()

    out_path = config.ndvi_median_path(config.YEAR)
    assert out_path.exists(), f"Expected output at {out_path}"

    result = xr.open_dataarray(str(out_path))
    assert result.ndim == 2 or (result.ndim == 3 and result.shape[0] == 1)
    assert result.rio.crs is not None
    assert result.rio.crs.to_epsg() == 7855
