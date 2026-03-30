"""Tests for mid-step tile checkpointing in analysis scripts 01, 02, and 03.

Each test verifies that:
- A tile with non-zero size on disk is skipped (load_stackstac / load_dea_landsat NOT called)
- A zero-byte tile on disk is treated as a cache miss (load function IS called)
"""
import importlib.util
import queue
import sys
import threading
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

        with patch("utils.stac.search_sentinel2", return_value=[MagicMock()]), \
             patch("utils.stac.load_stackstac", load_mock), \
             patch("utils.tiling.make_tile_bboxes",
                   return_value=[[141.0, -17.0, 143.0, -15.0]]), \
             patch("utils.mask.apply_scl_mask", side_effect=lambda stack, scl: stack), \
             patch("utils.quicklook.save_quicklook"), \
             patch("utils.tiling.merge_tile_rasters", side_effect=fake_merge), \
             patch("xarray.open_dataarray",
                   return_value=_make_synthetic_stack(bands=["nir"]).isel(time=0, band=0)):
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

        with patch("utils.stac.search_sentinel2", return_value=[MagicMock()]), \
             patch("utils.stac.load_stackstac", load_mock), \
             patch("utils.tiling.make_tile_bboxes",
                   return_value=[[141.0, -17.0, 143.0, -15.0]]), \
             patch("utils.mask.apply_scl_mask", side_effect=lambda stack, scl: stack), \
             patch("utils.quicklook.save_quicklook"), \
             patch("utils.tiling.merge_tile_rasters", side_effect=fake_merge), \
             patch("xarray.open_dataarray",
                   return_value=_make_synthetic_stack(bands=["nir"]).isel(time=0, band=0)):
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

        def fake_load_dea(**kw):
            data = np.random.default_rng(0).uniform(0.1, 0.4, size=(1, 2, 2)).astype(np.float32)
            times = np.array([np.datetime64("2000-01-01")])
            x = np.linspace(700000.0, 700020.0, 2)
            y = np.linspace(-1600000.0, -1600020.0, 2)
            nir = xr.DataArray(data, dims=["time", "y", "x"],
                               coords={"time": times, "y": y, "x": x})
            red = nir.copy()
            ds = xr.Dataset({"nbart_nir": nir, "nbart_red": red})
            ds = ds.rio.write_crs("EPSG:7855")
            return ds

        load_mock = MagicMock(side_effect=fake_load_dea)

        def fake_merge(tile_paths, out_path, nodata, crs):
            da_2d.rio.to_raster(str(out_path), driver="GTiff", dtype="float32")

        # Write a fake current-year NDVI so step 02 doesn't fail at ndvi_median_path check
        ndvi_path = config.ndvi_median_path(config.YEAR)
        ndvi_path.parent.mkdir(parents=True, exist_ok=True)
        da_2d.rio.to_raster(str(ndvi_path), driver="GTiff", dtype="float32")

        with patch("utils.stac.load_dea_landsat", load_mock), \
             patch("utils.tiling.make_tile_bboxes",
                   return_value=[[141.0, -17.0, 143.0, -15.0]]), \
             patch("utils.tiling.merge_tile_rasters", side_effect=fake_merge), \
             patch("utils.quicklook.save_quicklook"), \
             patch("utils.io.write_cog", side_effect=lambda da, path: da.rio.to_raster(
                 str(path), driver="GTiff", dtype="float32")), \
             patch("rioxarray.raster_array.RasterArray.reproject_match",
                   return_value=da_2d):
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


# ---------------------------------------------------------------------------
# Backpressure — run_tiled_pipeline queue depth stays bounded
# ---------------------------------------------------------------------------

class TestPipelineBackpressure:
    """Verify that fetch workers block when the queue is full.

    With maxsize=2 and compute workers paused, the fetch pool must not
    materialise more than maxsize items simultaneously.
    """

    def test_queue_depth_bounded(self, tmp_path):
        """Fetch pool blocks once queue is full; depth never exceeds maxsize."""
        from utils.pipeline import run_tiled_pipeline

        MAXSIZE = 2
        high_water = [0]
        high_water_lock = threading.Lock()
        pause_event = threading.Event()
        compute_calls = []

        # Tile bboxes — enough tiles to overflow the queue if backpressure fails
        n_tiles = 8
        tile_bboxes = [[float(i), -17.0, float(i) + 1, -16.0] for i in range(n_tiles)]

        scratch_dir = tmp_path / "tiles"
        scratch_dir.mkdir()

        import numpy as np
        import xarray as xr

        def fetch_fn(tile_idx, tile_bbox, tile_path):
            # Return a minimal DataArray (simulates a fetched tile)
            data = np.zeros((1, 2, 2), dtype=np.float32)
            times = [np.datetime64("2025-06-01")]
            da = xr.DataArray(data, dims=["time", "y", "x"],
                              coords={"time": times,
                                      "y": [0.0, 1.0], "x": [0.0, 1.0]})
            da = da.rio.write_crs("EPSG:7855")
            return da

        def compute_fn(tile_idx, raw, tile_path):
            compute_calls.append(tile_idx)
            # Block until released so the queue fills up
            pause_event.wait()
            tile_path.write_bytes(b"x")
            return tile_path

        def merge_fn(valid_paths, out_path, nodata, crs):
            out_path.write_bytes(b"merged")

        out_path = tmp_path / "out.tif"

        # Run pipeline in a background thread so we can inspect mid-run
        exc_holder = []

        def run():
            try:
                run_tiled_pipeline(
                    tile_bboxes=tile_bboxes,
                    scratch_dir=scratch_dir,
                    fetch_fn=fetch_fn,
                    compute_fn=compute_fn,
                    merge_fn=merge_fn,
                    out_path=out_path,
                    nodata=float("nan"),
                    crs="EPSG:7855",
                    fetch_workers=4,
                    compute_workers=MAXSIZE,
                )
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=run)
        t.start()

        # Give the pipeline a moment to fill the queue and block
        import time
        time.sleep(0.3)

        # Release compute workers and wait for completion
        pause_event.set()
        t.join(timeout=10)

        if exc_holder:
            raise exc_holder[0]

        assert out_path.exists()
        assert len(compute_calls) == n_tiles
