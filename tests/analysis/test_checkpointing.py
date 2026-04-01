"""Tests for mid-step tile checkpointing in analysis scripts 01, 02, and 03.

Each test verifies that:
- A tile with non-zero size on disk is skipped (load_stackstac / load_dea_landsat NOT called)
- A zero-byte tile on disk is treated as a cache miss (load function IS called)
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

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


def _make_synthetic_stack(n_scenes=1, height=2, width=2, bands=None):
    if bands is None:
        bands = ["blue", "green", "red", "nir", "nir08", "swir16", "swir22", "scl"]

    rng = np.random.default_rng(0)
    data = rng.uniform(0.05, 0.5, size=(n_scenes, len(bands), height, width)).astype(np.float32)
    if "scl" in bands:
        data[:, bands.index("scl"), :, :] = 4.0

    times = np.array([np.datetime64("2025-06-01")])[:n_scenes]
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


def _write_real_tile(path: Path):
    """Write a valid non-zero GeoTIFF to path."""
    da = _make_synthetic_stack(bands=["nir"]).isel(time=0, band=0)
    da.rio.to_raster(str(path), driver="GTiff", dtype="float32")


# ---------------------------------------------------------------------------
# Script 01 — 01_ndvi_composite.py
# ---------------------------------------------------------------------------

class TestCheckpointing01:
    SCRIPT = PROJECT_ROOT / "analysis" / "01_ndvi_composite.py"

    def _run_with_tile(self, tmp_dirs, tile_exists: bool, tile_size: int):
        """Run script 01 main() with a single tile, optionally pre-seeded on disk."""
        if "config" in sys.modules:
            del sys.modules["config"]
        import config

        working_dir = Path(tmp_dirs["WORKING_DIR"])
        scratch_dir = working_dir / f"tiles_ndvi_{config.YEAR}"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        tile_path = scratch_dir / "tile_00000.tif"
        if tile_exists:
            if tile_size > 0:
                _write_real_tile(tile_path)
            else:
                tile_path.write_bytes(b"")  # zero-byte file

        load_mock = MagicMock(side_effect=lambda **kw: _make_synthetic_stack(
            bands=kw.get("bands", ["nir", "red", "scl"])
        ))

        def fake_merge(tile_paths, out_path, nodata, crs):
            da = _make_synthetic_stack(bands=["nir"]).isel(time=0, band=0)
            da.rio.to_raster(str(out_path), driver="GTiff", dtype="float32")

        mock_item = MagicMock()
        mock_item.bbox = [140.0, -18.0, 144.0, -14.0]
        with patch("utils.stac.search_sentinel2", return_value=[mock_item]), \
             patch("utils.stac.load_stackstac", load_mock), \
             patch("utils.tiling.make_tile_bboxes",
                   return_value=[[141.0, -17.0, 143.0, -15.0]]), \
             patch("utils.mask.apply_scl_mask", side_effect=lambda stack, scl: stack), \
             patch("utils.quicklook.save_quicklook"), \
             patch("utils.tiling.merge_tile_rasters", side_effect=fake_merge), \
             patch("xarray.open_dataarray",
                   return_value=_make_synthetic_stack(bands=["nir"]).isel(time=0, band=0)), \
             patch.dict("os.environ", {"FETCH_WORKERS": "1"}):
            _load_module(self.SCRIPT, f"step01_ckpt_{tile_exists}_{tile_size}").main()

        return load_mock

    def test_cached_tile_skips_load(self, tmp_dirs):
        """Non-zero tile on disk → load_stackstac must NOT be called."""
        load_mock = self._run_with_tile(tmp_dirs, tile_exists=True, tile_size=1)
        load_mock.assert_not_called()

    def test_zero_byte_tile_triggers_load(self, tmp_dirs):
        """Zero-byte tile on disk → load_stackstac must be called (cache miss)."""
        load_mock = self._run_with_tile(tmp_dirs, tile_exists=True, tile_size=0)
        load_mock.assert_called_once()

    def test_missing_tile_triggers_load(self, tmp_dirs):
        """No tile on disk → load_stackstac must be called."""
        load_mock = self._run_with_tile(tmp_dirs, tile_exists=False, tile_size=0)
        load_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Script 03 — 03_flowering_index.py
# ---------------------------------------------------------------------------

class TestCheckpointing03:
    SCRIPT = PROJECT_ROOT / "analysis" / "03_flowering_index.py"

    def _run_with_tile(self, tmp_dirs, tile_exists: bool, tile_size: int):
        if "config" in sys.modules:
            del sys.modules["config"]
        import config

        working_dir = Path(tmp_dirs["WORKING_DIR"])
        scratch_dir = working_dir / f"tiles_flowering_{config.YEAR}"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        tile_path = scratch_dir / "tile_00000.tif"
        if tile_exists:
            if tile_size > 0:
                _write_real_tile(tile_path)
            else:
                tile_path.write_bytes(b"")

        flowering_bands = ["green", "nir", "rededge1", "rededge2", "scl"]
        load_mock = MagicMock(side_effect=lambda **kw: _make_synthetic_stack(
            bands=kw.get("bands", flowering_bands)
        ))

        def fake_merge(tile_paths, out_path, nodata, crs):
            da = _make_synthetic_stack(bands=["nir"]).isel(time=0, band=0)
            da.rio.to_raster(str(out_path), driver="GTiff", dtype="float32")

        mock_item = MagicMock()
        mock_item.bbox = [140.0, -18.0, 144.0, -14.0]
        with patch("utils.stac.search_sentinel2", return_value=[mock_item]), \
             patch("utils.stac.load_stackstac", load_mock), \
             patch("utils.tiling.make_tile_bboxes",
                   return_value=[[141.0, -17.0, 143.0, -15.0]]), \
             patch("utils.mask.apply_scl_mask", side_effect=lambda stack, scl: stack), \
             patch("utils.quicklook.save_quicklook"), \
             patch("utils.tiling.merge_tile_rasters", side_effect=fake_merge), \
             patch("xarray.open_dataarray",
                   return_value=_make_synthetic_stack(bands=["nir"]).isel(time=0, band=0)), \
             patch.dict("os.environ", {"FETCH_WORKERS": "1"}):
            _load_module(self.SCRIPT, f"step03_ckpt_{tile_exists}_{tile_size}").main()

        return load_mock

    def test_cached_tile_skips_load(self, tmp_dirs):
        """Non-zero tile on disk → load_stackstac must NOT be called."""
        load_mock = self._run_with_tile(tmp_dirs, tile_exists=True, tile_size=1)
        load_mock.assert_not_called()

    def test_zero_byte_tile_triggers_load(self, tmp_dirs):
        """Zero-byte tile on disk → load_stackstac must be called (cache miss)."""
        load_mock = self._run_with_tile(tmp_dirs, tile_exists=True, tile_size=0)
        load_mock.assert_called_once()

    def test_missing_tile_triggers_load(self, tmp_dirs):
        """No tile on disk → load_stackstac must be called."""
        load_mock = self._run_with_tile(tmp_dirs, tile_exists=False, tile_size=0)
        load_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Script 02 — 02_ndvi_anomaly.py (_build_baseline tile path)
# ---------------------------------------------------------------------------

class TestCheckpointing02:
    SCRIPT = PROJECT_ROOT / "analysis" / "02_ndvi_anomaly.py"

    def _run_with_tile(self, tmp_dirs, tile_exists: bool, tile_size: int):
        if "config" in sys.modules:
            del sys.modules["config"]
        import config

        working_dir = Path(tmp_dirs["WORKING_DIR"])
        scratch_dir = working_dir / f"tiles_baseline_{config.YEAR}"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        tile_path = scratch_dir / "tile_00000.tif"
        if tile_exists:
            if tile_size > 0:
                _write_real_tile(tile_path)
            else:
                tile_path.write_bytes(b"")

        # Synthetic single-band DataArray for baseline and current NDVI
        da_2d = _make_synthetic_stack(bands=["nir"]).isel(time=0, band=0)

        def fake_odc_load(items, **kw):
            data = np.random.default_rng(0).uniform(0.1, 0.4, size=(1, 2, 2)).astype(np.float32)
            times = np.array([np.datetime64("2000-01-01")])
            x = np.linspace(700000.0, 700020.0, 2)
            y = np.linspace(-1600000.0, -1600020.0, 2)
            nir = xr.DataArray(data, dims=["time", "y", "x"],
                               coords={"time": times, "y": y, "x": x})
            red = nir.copy()
            ds = xr.Dataset({"nbart_nir": nir, "nbart_red": red})
            ds = ds.rio.write_crs("EPSG:7855")
            # odc.stac.load returns a lazy Dataset; _build_baseline calls .compute()
            ds_mock = MagicMock()
            ds_mock.compute.return_value = ds
            return ds_mock

        load_mock = MagicMock(side_effect=fake_odc_load)

        def fake_merge(tile_paths, out_path, nodata, crs):
            da_2d.rio.to_raster(str(out_path), driver="GTiff", dtype="float32")

        # Write a fake current-year NDVI so step 02 doesn't fail at ndvi_median_path check
        ndvi_path = config.ndvi_median_path(config.YEAR)
        ndvi_path.parent.mkdir(parents=True, exist_ok=True)
        da_2d.rio.to_raster(str(ndvi_path), driver="GTiff", dtype="float32")

        fake_stac_item = MagicMock()
        fake_stac_item.datetime = "2000-07-15T00:00:00Z"
        fake_catalog = MagicMock()
        fake_catalog.search.return_value.items.return_value = iter([fake_stac_item])

        with patch("odc.stac.load", load_mock), \
             patch("utils.tiling.make_tile_bboxes",
                   return_value=[[141.0, -17.0, 143.0, -15.0]]), \
             patch("utils.tiling.merge_tile_rasters", side_effect=fake_merge), \
             patch("utils.quicklook.save_quicklook"), \
             patch("utils.io.write_cog", side_effect=lambda da, path: da.rio.to_raster(
                 str(path), driver="GTiff", dtype="float32")), \
             patch("rioxarray.raster_array.RasterArray.reproject_match",
                   return_value=da_2d), \
             patch("pystac_client.Client.open", return_value=fake_catalog):
            mod = _load_module(self.SCRIPT, f"step02_ckpt_{tile_exists}_{tile_size}")
            # Force baseline rebuild so _build_baseline is always invoked
            with patch.dict("os.environ", {"REBUILD_BASELINE": "true"}):
                mod.main()

        return load_mock

    def test_cached_tile_skips_load(self, tmp_dirs):
        """Non-zero baseline tile on disk → load_dea_landsat must NOT be called."""
        load_mock = self._run_with_tile(tmp_dirs, tile_exists=True, tile_size=1)
        load_mock.assert_not_called()

    def test_zero_byte_tile_triggers_load(self, tmp_dirs):
        """Zero-byte baseline tile → load_dea_landsat must be called (cache miss)."""
        load_mock = self._run_with_tile(tmp_dirs, tile_exists=True, tile_size=0)
        load_mock.assert_called_once()

    def test_missing_tile_triggers_load(self, tmp_dirs):
        """No baseline tile on disk → load_dea_landsat must be called."""
        load_mock = self._run_with_tile(tmp_dirs, tile_exists=False, tile_size=0)
        load_mock.assert_called_once()
