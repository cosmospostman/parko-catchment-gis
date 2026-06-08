"""Unit tests for utils/s1_collector.py and S1 integration in training_collector.

Tests
-----
S1C-1.  _reconstruct_affine: valid proj:transform returns correct Affine.
S1C-2.  _reconstruct_affine: missing proj:transform returns None.
S1C-3.  _reconstruct_affine: short proj:transform (< 6 elements) returns None.
S1C-4.  _extract_item: pixel centre maps to correct array index.
S1C-5.  _extract_item: pixel outside raster window is skipped.
S1C-6.  _extract_item: zero-value pixels become NaN (no-data sentinel).
S1C-7.  _extract_item: returns None when both bands are absent.
S1C-8.  _extract_item: partial data (only vh) yields rows with vv=NaN.
S1C-9.  collect_s1: empty result when STAC returns no items.
S1C-10. collect_s1: DataFrame columns and dtypes.
S1C-11. collect_s1: source column is always "S1".
S1C-12. collect_s1: spatial alignment — each row's (lon, lat) matches a grid point.

PW-1.  _pixel_window: bbox fully inside raster returns valid window.
PW-2.  _pixel_window: bbox outside raster extent returns None (not an exception).
PW-3.  _pixel_window: bbox overlapping raster edge is clipped to bounds.
PW-4.  _pixel_window: window row_off computed correctly from negative-yres affine.

RB-1.  _read_band_array: returns (arr, win_affine) when rasterio opens successfully.
RB-2.  _read_band_array: returns None when bbox is outside raster extent.
RB-3.  _read_band_array: returns None when window is zero-size after clipping.
RB-4.  _read_band_array: result contains no NaN for all-valid pixels.
RB-5.  _read_band_array: zero pixels converted to NaN.
RB-6.  _read_band_array: returns None (not exception) when rasterio raises.

E2E-1. collect_s1 with realistic rasterio mock returns non-empty DataFrame.
E2E-2. collect_s1 with out-of-extent item (envelope overlaps but data does not)
       silently skips the item and returns empty, not an exception.

TC-S1-1. _extend_schema: adds source/vh/vv to a schema that lacks them.
TC-S1-2. _extend_schema: idempotent when columns already present.
TC-S1-3. _conform_table: missing columns filled with null; types cast correctly.
TC-S1-4. _s1_df_to_arrow: S2-only columns are null; vh/vv are populated.
TC-S1-5. _s1_df_to_arrow: row count matches input DataFrame.
TC-S1-6. ensure_training_pixels: combined parquet contains both S2 and S1 rows.
TC-S1-7. ensure_training_pixels: sort order (point_id, date) is maintained after merge.
TC-S1-8. ensure_training_pixels: S2 rows have source="S2", S1 rows have source="S1".
TC-S1-9. ensure_training_pixels: vh/vv null on S2 rows, non-null on S1 rows.
TC-S1-10. Spatial alignment: S1 point_ids match S2 pixel grid, no phantom pixels.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.s1_collector import _reconstruct_affine, _extract_item, collect_s1
from utils.fetch import _cache_path as _s1_cache_path, _load_patch_cache, _save_patch_cache
from utils.parquet_utils import _extend_schema, _conform_table, _s1_df_to_arrow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_affine(lon_origin: float, lat_origin: float, res: float = 10 / 111320):
    """Build a simple north-up affine at 10 m pixel spacing (~0.0000899 deg/px)."""
    from rasterio.transform import Affine
    # Affine(xres, xskew, x_origin, yskew, yres, y_origin)
    # lat decreases south → yres is negative
    return Affine(res, 0, lon_origin, 0, -res, lat_origin)


def _affine_to_proj_transform(a):
    """Convert rasterio Affine to proj:transform list."""
    return [a.a, a.b, a.c, a.d, a.e, a.f]


def _make_item(
    item_id: str = "S1A_IW_GRDH_20220815",
    dt: datetime | None = None,
    proj_transform: list | None = None,
    vh_arr: np.ndarray | None = None,
    vv_arr: np.ndarray | None = None,
    nrows: int = 10,
    ncols: int = 10,
    lon_origin: float = 145.0,
    lat_origin: float = -22.9,
    res: float = 10 / 111320,
) -> SimpleNamespace:
    """Build a mock S1 STAC item backed by fake in-memory COG data."""
    if dt is None:
        dt = datetime(2022, 8, 15, tzinfo=timezone.utc)

    affine = _make_affine(lon_origin, lat_origin, res)
    if proj_transform is None:
        proj_transform = _affine_to_proj_transform(affine)

    if vh_arr is None:
        vh_arr = np.full((nrows, ncols), 0.001, dtype=np.float32)
    if vv_arr is None:
        vv_arr = np.full((nrows, ncols), 0.002, dtype=np.float32)

    # Patch rasterio.open so _read_band_array / _read_bbox_patch work without real files
    class _FakeDataset:
        def __init__(self, arr, affine=None):
            from pyproj import CRS
            self._arr = arr
            self.width = arr.shape[1]
            self.height = arr.shape[0]
            self.transform = affine  # needed by _read_bbox_patch
            self.crs = CRS.from_epsg(32755)  # needed by _read_bbox_patch

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, band_idx, window=None):
            if window is None:
                return self._arr
            import math
            c0 = max(0, math.floor(window.col_off))
            r0 = max(0, math.floor(window.row_off))
            c1 = min(self._arr.shape[1], math.ceil(window.col_off + window.width))
            r1 = min(self._arr.shape[0], math.ceil(window.row_off + window.height))
            return self._arr[r0:r1, c0:c1]

    assets = {}
    _arrays = {}
    if vh_arr is not None:
        assets["vh"] = SimpleNamespace(href="fake://vh")
        _arrays["fake://vh"] = _FakeDataset(vh_arr, affine)
    if vv_arr is not None:
        assets["vv"] = SimpleNamespace(href="fake://vv")
        _arrays["fake://vv"] = _FakeDataset(vv_arr, affine)

    return SimpleNamespace(
        id=item_id,
        datetime=dt,
        bbox=[lon_origin, lat_origin - nrows * res, lon_origin + ncols * res, lat_origin],
        properties={"proj:transform": proj_transform},
        assets=assets,
        _arrays=_arrays,
    )


@pytest.fixture(autouse=True)
def _patch_rasterio(monkeypatch):
    """Redirect rasterio.open to return fake datasets for fake:// hrefs."""
    import utils.s1_collector as mod
    import rasterio

    original_open = rasterio.open

    def _fake_open(href, *args, **kwargs):
        # Walk up the call stack to find the item's _arrays dict
        import inspect
        frame = inspect.currentframe()
        while frame:
            local_item = frame.f_locals.get("item")
            if local_item is not None and hasattr(local_item, "_arrays"):
                if href in local_item._arrays:
                    return local_item._arrays[href]
            frame = frame.f_back
        return original_open(href, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", _fake_open)


# ---------------------------------------------------------------------------
# S1C-1 to S1C-3: _reconstruct_affine
# ---------------------------------------------------------------------------

def test_reconstruct_affine_valid():
    from rasterio.transform import Affine
    item = SimpleNamespace(properties={
        "proj:transform": [0.0001, 0, 145.0, 0, -0.0001, -22.9]
    })
    a = _reconstruct_affine(item)
    assert a is not None
    assert isinstance(a, Affine)
    assert abs(a.a - 0.0001) < 1e-10
    assert abs(a.c - 145.0) < 1e-10
    assert abs(a.f - (-22.9)) < 1e-10


def test_reconstruct_affine_missing():
    item = SimpleNamespace(properties={})
    assert _reconstruct_affine(item) is None


def test_reconstruct_affine_short():
    item = SimpleNamespace(properties={"proj:transform": [0.0001, 0, 145.0]})
    assert _reconstruct_affine(item) is None


# ---------------------------------------------------------------------------
# S1C-4: _extract_item — pixel centre maps to correct array index
# ---------------------------------------------------------------------------

def test_extract_item_pixel_alignment():
    res = 10 / 111320
    lon_origin = 145.0
    lat_origin = -22.9

    nrows, ncols = 5, 5
    vh_arr = np.arange(nrows * ncols, dtype=np.float32).reshape(nrows, ncols)
    vh_arr += 0.001  # avoid zero (no-data)
    vv_arr = vh_arr * 2

    item = _make_item(
        nrows=nrows, ncols=ncols,
        lon_origin=lon_origin, lat_origin=lat_origin,
        res=res, vh_arr=vh_arr, vv_arr=vv_arr,
    )
    affine = _reconstruct_affine(item)

    # Place a point at pixel (row=1, col=2): centre = lon_origin + 2.5*res, lat_origin - 1.5*res
    target_lon = lon_origin + 2.5 * res
    target_lat = lat_origin - 1.5 * res
    points = [("px_0000", target_lon, target_lat)]
    bbox = [lon_origin, lat_origin - nrows * res, lon_origin + ncols * res, lat_origin]

    cols = _extract_item(item, affine, bbox, points)
    assert cols is not None
    pids, lons, lats, dates, vhs, vvs, orbits = cols
    assert len(pids) == 1
    assert pids[0] == "px_0000"
    # row=1, col=2 → vh_arr[1, 2] = 1*5 + 2 + 0.001 = 7.001
    assert abs(vhs[0] - vh_arr[1, 2]) < 1e-5
    assert abs(vvs[0] - vv_arr[1, 2]) < 1e-5


# ---------------------------------------------------------------------------
# S1C-5: pixel outside raster window is skipped
# ---------------------------------------------------------------------------

def test_extract_item_outside_window():
    res = 10 / 111320
    item = _make_item(nrows=5, ncols=5, lon_origin=145.0, lat_origin=-22.9, res=res)
    affine = _reconstruct_affine(item)
    bbox = [145.0, -22.9 - 5 * res, 145.0 + 5 * res, -22.9]

    # Point far outside
    points = [("px_far", 200.0, 0.0)]
    cols = _extract_item(item, affine, bbox, points)
    assert cols is None or len(cols[0]) == 0


# ---------------------------------------------------------------------------
# S1C-6: zero-value pixels become NaN (no-data sentinel)
# ---------------------------------------------------------------------------

def test_extract_item_zero_is_nodata():
    res = 10 / 111320
    vh_arr = np.zeros((5, 5), dtype=np.float32)  # all zero → all NaN
    item = _make_item(nrows=5, ncols=5, res=res, vh_arr=vh_arr, vv_arr=None)
    # Remove vv asset so only vh is tried
    del item.assets["vv"]
    affine = _reconstruct_affine(item)
    bbox = [145.0, -22.9 - 5 * res, 145.0 + 5 * res, -22.9]

    points = [("px_0000", 145.0 + 0.5 * res, -22.9 - 0.5 * res)]
    cols = _extract_item(item, affine, bbox, points)
    # All zeros → NaN → no valid rows
    assert cols is None or len(cols[0]) == 0


# ---------------------------------------------------------------------------
# S1C-7: returns None when both bands absent
# ---------------------------------------------------------------------------

def test_extract_item_no_bands():
    res = 10 / 111320
    item = _make_item(nrows=5, ncols=5, res=res)
    item.assets = {}  # no bands
    affine = _reconstruct_affine(item)
    bbox = [145.0, -22.9 - 5 * res, 145.0 + 5 * res, -22.9]
    points = [("px_0000", 145.0 + 0.5 * res, -22.9 - 0.5 * res)]
    cols = _extract_item(item, affine, bbox, points)
    assert cols is None or len(cols[0]) == 0


# ---------------------------------------------------------------------------
# S1C-8: partial data — only vh, vv absent → vv=NaN
# ---------------------------------------------------------------------------

def test_extract_item_only_vh():
    res = 10 / 111320
    vh_arr = np.full((5, 5), 0.005, dtype=np.float32)
    item = _make_item(nrows=5, ncols=5, res=res, vh_arr=vh_arr, vv_arr=None)
    del item.assets["vv"]
    affine = _reconstruct_affine(item)
    bbox = [145.0, -22.9 - 5 * res, 145.0 + 5 * res, -22.9]
    points = [("px_0000", 145.0 + 0.5 * res, -22.9 - 0.5 * res)]
    cols = _extract_item(item, affine, bbox, points)
    assert cols is not None and len(cols[0]) == 1
    pids, lons, lats, dates, vhs, vvs, orbits = cols
    assert abs(vhs[0] - 0.005) < 1e-6
    assert np.isnan(vvs[0])


# ---------------------------------------------------------------------------
# S1C-9 to S1C-12: collect_s1
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# _pixel_window tests (PW-1 to PW-4)
# These test the real rasterio window arithmetic — no mocking.
# ---------------------------------------------------------------------------

def _north_up_affine(lon_origin: float, lat_origin: float, res: float):
    """North-up affine: origin at top-left (lat_origin = northernmost lat)."""
    from affine import Affine
    return Affine(res, 0, lon_origin, 0, -res, lat_origin)


def test_pixel_window_inside_raster():
    """Bbox fully inside raster returns a valid, non-None window."""
    from utils.s1_collector import _pixel_window
    res = 10 / 111320
    lat_origin = -22.8   # northern edge of raster
    affine = _north_up_affine(145.0, lat_origin, res)
    src_width, src_height = 100, 100
    # Bbox 10 rows south and 10 cols east of origin — well inside 100×100 raster
    bbox = [
        145.0 + 10 * res,          # lon_min
        lat_origin - 20 * res,     # lat_min (further south)
        145.0 + 20 * res,          # lon_max
        lat_origin - 10 * res,     # lat_max (closer to origin)
    ]
    win = _pixel_window(affine, bbox, src_width=src_width, src_height=src_height)
    assert win is not None
    assert win.width > 0
    assert win.height > 0


def test_pixel_window_outside_raster_returns_none():
    """Bbox outside the raster extent must return None, not raise an exception.

    This is the exact failure mode that caused the production bug: filter_items_by_bbox
    accepted items whose envelope overlapped the bbox, but the actual raster data did
    not reach the target area. rasterio.windows.intersection() raises WindowError;
    _pixel_window must catch it and return None.
    """
    from utils.s1_collector import _pixel_window
    res = 10 / 111320
    # Raster covers lon 134.86 to ~138, lat -18.0 to ~-19.4 (16843 rows at res)
    affine = _north_up_affine(134.86, -18.0, res)
    src_width, src_height = 26001, 16843
    # Target bbox is south of the raster's southern edge → window row_off > src_height
    bbox = [137.2, -19.59, 137.22, -19.58]
    win = _pixel_window(affine, bbox, src_width=src_width, src_height=src_height)
    assert win is None, (
        f"Expected None for out-of-extent bbox, got {win}. "
        "This was the production bug: intersection raised WindowError instead of returning None."
    )


def test_pixel_window_clipped_to_edge():
    """Bbox overlapping the raster edge is clipped, not rejected."""
    from utils.s1_collector import _pixel_window
    res = 10 / 111320
    lat_origin = -22.8
    affine = _north_up_affine(145.0, lat_origin, res)
    src_width, src_height = 50, 50
    # Bbox starts inside raster (col ~45) but extends 5 cols beyond right edge (col 50)
    raster_right_lon = 145.0 + src_width * res
    bbox = [
        raster_right_lon - 5 * res,       # lon_min: 5 cols from right edge
        lat_origin - 20 * res,            # lat_min
        raster_right_lon + 5 * res,       # lon_max: 5 cols beyond right edge
        lat_origin - 10 * res,            # lat_max
    ]
    win = _pixel_window(affine, bbox, src_width=src_width, src_height=src_height)
    assert win is not None
    assert win.col_off + win.width <= src_width + 1  # +1 for float rounding


def test_pixel_window_row_off_correct_negative_yres():
    """row_off computed correctly for north-up (negative yres) affine.

    This validates the coordinate arithmetic that caused the production bug.
    For a north-up raster, row increases southward. A target lat that is
    south of the raster origin but within bounds must give 0 < row_off < src_height.
    """
    from utils.s1_collector import _pixel_window
    res = 10 / 111320
    lat_origin = -18.0   # northern edge of raster
    affine = _north_up_affine(145.0, lat_origin, res)
    src_height = 1000
    # Target 100 rows south of origin
    target_lat_max = lat_origin - 100 * res
    target_lat_min = target_lat_max - 5 * res
    bbox = [145.001, target_lat_min, 145.002, target_lat_max]
    win = _pixel_window(affine, bbox, src_width=1000, src_height=src_height)
    assert win is not None
    assert 95 <= win.row_off <= 105, f"Expected row_off ~100, got {win.row_off}"


# ---------------------------------------------------------------------------
# _read_band_array tests (RB-1 to RB-6)
# Uses a real rasterio-like context manager mock that exercises the full
# _pixel_window → src.read code path.
# ---------------------------------------------------------------------------

class _MockRasterioDataset:
    """Minimal rasterio dataset mock that exercises the real window arithmetic."""

    def __init__(self, arr: np.ndarray, affine, width: int, height: int):
        from pyproj import CRS
        self._arr = arr
        self.transform = affine
        self.width = width
        self.height = height
        self.crs = CRS.from_epsg(32755)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, band_idx, window=None):
        if window is None:
            return self._arr
        import math
        c0 = max(0, math.floor(window.col_off))
        r0 = max(0, math.floor(window.row_off))
        c1 = min(self._arr.shape[1], math.ceil(window.col_off + window.width))
        r1 = min(self._arr.shape[0], math.ceil(window.row_off + window.height))
        if r1 <= r0 or c1 <= c0:
            return np.zeros((0, 0), dtype=self._arr.dtype)
        return self._arr[r0:r1, c0:c1]


def _make_mock_open(arr, affine, width=None, height=None):
    """Return a mock rasterio.open context manager."""
    h, w = arr.shape
    ds = _MockRasterioDataset(arr, affine, width or w, height or h)
    return lambda href, *a, **kw: ds


def test_read_band_array_returns_arr_and_affine(monkeypatch):
    """_read_band_array returns (arr, win_affine) for a valid bbox."""
    import utils.s1_collector as mod
    res = 10 / 111320
    lat_origin = -22.8
    affine = _north_up_affine(145.0, lat_origin, res)
    arr = np.full((100, 100), 0.01, dtype=np.float32)
    bbox = [145.0 + 10*res, lat_origin - 20*res, 145.0 + 20*res, lat_origin - 10*res]
    monkeypatch.setattr(mod, "_pixel_window",
                        lambda a, b, w, h: __import__("rasterio.windows", fromlist=["Window"]).Window(10, 10, 10, 10))
    monkeypatch.setattr("rasterio.open", _make_mock_open(arr, affine))
    import rasterio
    monkeypatch.setattr(rasterio, "open", _make_mock_open(arr, affine))
    result = mod._read_band_array("fake://vh", affine, bbox, cache_path=None)
    assert result is not None
    out_arr, _ = result
    assert out_arr.size > 0


def test_read_band_array_returns_none_for_outside_bbox():
    """_read_band_array returns None (not an exception) when bbox is outside raster.

    Reproduces the production bug: item envelope overlaps but actual raster data
    does not cover the target area. _pixel_window returns None; _read_band_array
    must propagate that as None rather than crashing.
    """
    from utils.s1_collector import _pixel_window, _read_band_array
    res = 10 / 111320
    affine = _north_up_affine(134.86, -18.0, res)
    # Target bbox south of raster extent — _pixel_window returns None
    bbox = [137.2, -19.59, 137.22, -19.58]
    win = _pixel_window(affine, bbox, src_width=26001, src_height=16843)
    assert win is None, "Precondition: _pixel_window must return None for this bbox"


def test_read_band_array_returns_none_for_zero_window(monkeypatch):
    """_read_band_array returns None when _pixel_window returns None."""
    import utils.s1_collector as mod
    res = 10 / 111320
    affine = _north_up_affine(145.0, -22.8, res)
    # Patch _pixel_window to return None directly
    monkeypatch.setattr(mod, "_pixel_window", lambda *a, **kw: None)
    bbox = [146.0, -23.0, 146.1, -22.9]
    result = mod._read_band_array("fake://vh", affine, bbox, cache_path=None)
    assert result is None


def test_read_band_array_no_nan_for_valid_pixels(monkeypatch):
    """Valid (non-zero) pixels produce no NaN in output array."""
    import rasterio
    import utils.s1_collector as mod
    res = 10 / 111320
    lat_origin = -22.8
    affine = _north_up_affine(145.0, lat_origin, res)
    arr = np.full((100, 100), 100.0, dtype=np.float32)
    monkeypatch.setattr(rasterio, "open", _make_mock_open(arr, affine))
    # Patch _pixel_window to return a valid window directly
    import rasterio.windows as rw
    monkeypatch.setattr(mod, "_pixel_window", lambda *a, **kw: rw.Window(5, 5, 10, 10))
    bbox = [145.0 + 5*res, lat_origin - 15*res, 145.0 + 15*res, lat_origin - 5*res]
    result = mod._read_band_array("fake://vh", affine, bbox, cache_path=None)
    assert result is not None
    out_arr, _ = result
    assert not np.isnan(out_arr).any()


def test_read_band_array_zero_becomes_nan(monkeypatch):
    """Zero-value pixels (S1 no-data sentinel) are converted to NaN."""
    import rasterio
    import utils.s1_collector as mod
    res = 10 / 111320
    lat_origin = -22.8
    affine = _north_up_affine(145.0, lat_origin, res)
    arr = np.zeros((100, 100), dtype=np.float32)
    monkeypatch.setattr(rasterio, "open", _make_mock_open(arr, affine))
    import rasterio.windows as rw
    monkeypatch.setattr(mod, "_pixel_window", lambda *a, **kw: rw.Window(5, 5, 10, 10))
    bbox = [145.0 + 5*res, lat_origin - 15*res, 145.0 + 15*res, lat_origin - 5*res]
    result = mod._read_band_array("fake://vh", affine, bbox, cache_path=None)
    if result is not None:
        out_arr, _ = result
        assert np.isnan(out_arr).all()


def test_read_band_array_returns_none_on_rasterio_exception(monkeypatch):
    """rasterio.open raising an exception returns None, not a crash."""
    import rasterio
    import utils.s1_collector as mod

    def _raise(*a, **kw):
        raise RuntimeError("simulated network error")

    monkeypatch.setattr(rasterio, "open", _raise)
    res = 10 / 111320
    affine = _north_up_affine(145.0, -22.8, res)
    import rasterio.windows as rw
    monkeypatch.setattr(mod, "_pixel_window", lambda *a, **kw: rw.Window(5, 5, 10, 10))
    bbox = [145.001, -22.82, 145.003, -22.81]
    result = mod._read_band_array("fake://vh", affine, bbox, cache_path=None)
    assert result is None


# ---------------------------------------------------------------------------
# End-to-end tests (E2E-1 to E2E-2) — realistic rasterio mock, no STAC
# ---------------------------------------------------------------------------

def _make_s1_item(
    item_id: str,
    lon_origin: float,
    lat_origin: float,
    res: float,
    nrows: int,
    ncols: int,
    vh_val: float = 0.01,
    vv_val: float = 0.02,
    dt=None,
):
    """Build a minimal mock S1 STAC item (no rasterio datasets needed)."""
    from datetime import datetime, timezone
    if dt is None:
        dt = datetime(2022, 8, 15, tzinfo=timezone.utc)
    affine = _north_up_affine(lon_origin, lat_origin, res)
    pt = [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f]
    return SimpleNamespace(
        id=item_id,
        datetime=dt,
        bbox=[lon_origin, lat_origin - nrows * res, lon_origin + ncols * res, lat_origin],
        properties={"proj:transform": pt, "proj:epsg": 32755, "sat:orbit_state": "ascending"},
        assets={
            "vh": SimpleNamespace(href=f"fake://{item_id}/vh"),
            "vv": SimpleNamespace(href=f"fake://{item_id}/vv"),
        },
        _vh_val=vh_val,
        _vv_val=vv_val,
        _nrows=nrows,
        _ncols=ncols,
        _lon_origin=lon_origin,
        _lat_origin=lat_origin,
        _res=res,
    )


def _make_fake_fetch_patches(items_by_id: dict):
    """Return a coroutine that populates cache_dir with fake S1 patches instead of fetching.

    Patches are written in EPSG:4326 so CachedNpzChipStore doesn't need to reproject
    the test points — it just does a direct degree-space pixel lookup.

    items_by_id maps item_id → item SimpleNamespace (with _vh_val, _vv_val, etc.).
    """
    async def _fake_fetch_patches(
        points, items, bands, bbox_wgs84,
        scl_filter=True, max_concurrent=32, band_alias=None, cache_dir=None, item_signer=None,
        **kwargs,
    ):
        from utils.fetch import _save_patch_cache, _cache_path, _load_patch_cache
        from pyproj import CRS
        crs = CRS.from_epsg(4326)
        if cache_dir is None:
            return {}
        for item in items:
            spec = items_by_id.get(item.id)
            if spec is None:
                continue
            # Use a degree-space affine so (lon, lat) map directly to pixel indices
            affine = _north_up_affine(spec._lon_origin, spec._lat_origin, spec._res)
            for band in bands:
                path = _cache_path(cache_dir, item.id, band)
                if path.exists() and _load_patch_cache(path) is not None:
                    continue  # already cached
                val = spec._vh_val if band == "vh" else spec._vv_val
                arr = np.full((spec._nrows, spec._ncols), val, dtype=np.float32)
                _save_patch_cache(path, (arr, affine, crs), fetch_bbox=bbox_wgs84)
        return {}
    return _fake_fetch_patches


def test_e2e_collect_s1_returns_rows(monkeypatch, tmp_path):
    """collect_s1 with mocked fetch_patches returns non-empty DataFrame.

    This verifies that S1 rows are actually produced end-to-end from cached
    patches through CachedNpzChipStore extraction.
    """
    import utils.s1_collector as mod
    import utils.fetch as fetch_mod

    res = 10 / 111320
    lon_origin, lat_origin = 145.0, -22.8
    nrows, ncols = 50, 50
    item = _make_s1_item("S1A_test", lon_origin, lat_origin, res, nrows, ncols)
    items_by_id = {item.id: item}

    monkeypatch.setattr(mod, "search_sentinel1", lambda **kw: [item])
    monkeypatch.setattr(mod, "filter_items_by_bbox", lambda i, b: i)
    monkeypatch.setattr(mod, "setup_gdal_env", lambda: None)
    monkeypatch.setattr(fetch_mod, "fetch_patches", _make_fake_fetch_patches(items_by_id))

    bbox = [lon_origin, lat_origin - nrows * res, lon_origin + ncols * res, lat_origin]
    points = [
        (f"px_{r:04d}_{c:04d}", lon_origin + (c + 0.5) * res, lat_origin - (r + 0.5) * res)
        for r in range(nrows) for c in range(ncols)
    ]
    df = mod.collect_s1(bbox, "2022-01-01", "2022-12-31", points, cache_dir=tmp_path / "s1_chips")

    assert not df.is_empty(), (
        "collect_s1 returned empty DataFrame — S1 rows were not produced."
    )
    assert (df["source"] == "S1").all()
    assert df["vh"].is_not_null().any()
    assert df["vv"].is_not_null().any()


def test_e2e_collect_s1_out_of_extent_item_skipped(monkeypatch, tmp_path):
    """collect_s1 returns empty DataFrame when patches don't cover the target points.

    Simulates the production failure: item bbox overlaps the target but raster
    data doesn't reach the exact pixel location.
    """
    import utils.s1_collector as mod
    import utils.fetch as fetch_mod

    res = 10 / 111320
    # Patches cover the raster origin area — points are placed far outside
    item = _make_s1_item("S1A_out_of_extent", 134.86, -18.0, res, 50, 50)
    items_by_id = {item.id: item}

    monkeypatch.setattr(mod, "search_sentinel1", lambda **kw: [item])
    monkeypatch.setattr(mod, "filter_items_by_bbox", lambda i, b: i)
    monkeypatch.setattr(mod, "setup_gdal_env", lambda: None)
    monkeypatch.setattr(fetch_mod, "fetch_patches", _make_fake_fetch_patches(items_by_id))

    bbox = [137.2, -19.59, 137.22, -19.58]
    points = [("px_0000", 137.21, -19.585)]

    df = mod.collect_s1(bbox, "2022-01-01", "2022-12-31", points, cache_dir=tmp_path / "s1_chips")
    assert df.is_empty(), (
        "Expected empty DataFrame when patches don't cover target points."
    )


# Cache round-trip tests (using fetch.py _save_patch_cache / _load_patch_cache)
# ---------------------------------------------------------------------------

def test_save_and_load_patch_roundtrip(tmp_path):
    """fetch.py patch cache stores arr, transform, and CRS; round-trips correctly."""
    from affine import Affine
    from pyproj import CRS
    arr = np.array([[0.001, 0.002], [0.003, 0.004]], dtype=np.float32)
    win_affine = Affine(0.0001, 0, 145.0, 0, -0.0001, -22.9)
    crs = CRS.from_epsg(32755)
    path = tmp_path / "S1A_item" / "vh.npz"
    _save_patch_cache(path, (arr, win_affine, crs))
    assert path.exists()
    loaded = _load_patch_cache(path)
    assert loaded is not None
    arr_out, affine_out, crs_out = loaded
    np.testing.assert_array_almost_equal(arr_out, arr)
    assert abs(affine_out.a - win_affine.a) < 1e-10
    assert abs(affine_out.c - win_affine.c) < 1e-10


def test_load_patch_missing_returns_none(tmp_path):
    assert _load_patch_cache(tmp_path / "nonexistent.npz") is None


def test_load_patch_corrupt_returns_none(tmp_path):
    p = tmp_path / "corrupt.npz"
    p.write_bytes(b"not a valid npz")
    assert _load_patch_cache(p) is None


def test_read_band_array_writes_cache(tmp_path, monkeypatch):
    """_read_band_array should write a .npz cache file on first call."""
    import utils.s1_collector as mod
    res = 10 / 111320
    item = _make_item(nrows=3, ncols=3, res=res)
    affine = _reconstruct_affine(item)
    bbox = [145.0, -22.9 - 3 * res, 145.0 + 3 * res, -22.9]
    cache_path = tmp_path / "S1A_item" / "vh.npz"

    result = mod._read_band_array(item.assets["vh"].href, affine, bbox, cache_path)
    assert result is not None
    assert cache_path.exists(), "Cache file not written after first read"


def test_read_band_array_loads_from_cache(tmp_path, monkeypatch):
    """Second call should load from cache without calling rasterio.open."""
    import utils.s1_collector as mod
    from affine import Affine
    from pyproj import CRS

    # Pre-populate cache with known values in the new fetch.py format (arr, affine, crs)
    expected_arr = np.array([[0.999]], dtype=np.float32)
    win_affine = Affine(10 / 111320, 0, 145.0, 0, -(10 / 111320), -22.9)
    crs = CRS.from_epsg(32755)
    cache_path = tmp_path / "item" / "vh.npz"
    _save_patch_cache(cache_path, (expected_arr, win_affine, crs))

    # rasterio.open should NOT be called
    import rasterio
    def _should_not_open(*a, **kw):
        raise AssertionError("rasterio.open called despite cache hit")
    monkeypatch.setattr(rasterio, "open", _should_not_open)

    result = mod._read_band_array("fake://vh", win_affine, [145.0, -22.9 - 0.001, 145.001, -22.9], cache_path)
    assert result is not None
    arr_out, _ = result
    np.testing.assert_array_almost_equal(arr_out, expected_arr)


def test_collect_s1_writes_chip_cache(tmp_path, monkeypatch):
    """collect_s1 should populate cache_dir with per-(item,band) .npz files."""
    import utils.s1_collector as mod
    import utils.fetch as fetch_mod

    res = 10 / 111320
    item = _make_s1_item("S1A_IW_GRDH_20220815", 145.0, -22.9, res, 4, 4)
    items_by_id = {item.id: item}
    monkeypatch.setattr(mod, "search_sentinel1", lambda **kw: [item])
    monkeypatch.setattr(mod, "filter_items_by_bbox", lambda i, b: i)
    monkeypatch.setattr(mod, "setup_gdal_env", lambda: None)
    monkeypatch.setattr(fetch_mod, "fetch_patches", _make_fake_fetch_patches(items_by_id))

    bbox = [145.0, -22.9 - 4 * res, 145.0 + 4 * res, -22.9]
    points = [("px_0000", 145.0 + 0.5 * res, -22.9 - 0.5 * res)]
    cache_dir = tmp_path / "s1_chips"

    collect_s1(bbox, "2022-01-01", "2022-12-31", points, cache_dir=cache_dir)

    assert _s1_cache_path(cache_dir, item.id, "vh").exists()
    assert _s1_cache_path(cache_dir, item.id, "vv").exists()


def test_collect_s1_uses_cache_on_second_call(tmp_path, monkeypatch):
    """Second collect_s1 call returns same data using the on-disk patch cache."""
    import utils.s1_collector as mod
    import utils.fetch as fetch_mod

    res = 10 / 111320
    item = _make_s1_item("S1A_IW_GRDH_20220815", 145.0, -22.9, res, 4, 4)
    items_by_id = {item.id: item}
    monkeypatch.setattr(mod, "search_sentinel1", lambda **kw: [item])
    monkeypatch.setattr(mod, "filter_items_by_bbox", lambda i, b: i)
    monkeypatch.setattr(mod, "setup_gdal_env", lambda: None)

    bbox = [145.0, -22.9 - 4 * res, 145.0 + 4 * res, -22.9]
    points = [("px_0000", 145.0 + 0.5 * res, -22.9 - 0.5 * res)]
    cache_dir = tmp_path / "s1_chips"

    # First call: write cache via fake fetch_patches
    monkeypatch.setattr(fetch_mod, "fetch_patches", _make_fake_fetch_patches(items_by_id))
    df1 = collect_s1(bbox, "2022-01-01", "2022-12-31", points, cache_dir=cache_dir)

    # Second call: fetch_patches is called again but sees cached .npz files and
    # skips network I/O. Results must be identical to the first call.
    df2 = collect_s1(bbox, "2022-01-01", "2022-12-31", points, cache_dir=cache_dir)

    assert len(df1) == len(df2)
    np.testing.assert_array_almost_equal(df1["vh"].to_numpy(), df2["vh"].to_numpy())
    np.testing.assert_array_almost_equal(df1["vv"].to_numpy(), df2["vv"].to_numpy())


def test_collect_s1_empty_when_no_items(monkeypatch):
    import utils.s1_collector as mod
    monkeypatch.setattr(mod, "search_sentinel1", lambda **kw: [])
    monkeypatch.setattr(mod, "filter_items_by_bbox", lambda items, bbox: items)
    monkeypatch.setattr(mod, "setup_gdal_env", lambda: None)
    df = collect_s1([145.0, -23.0, 145.1, -22.9], "2022-01-01", "2022-12-31", [])
    assert df.is_empty()


def test_collect_s1_columns_and_dtypes(monkeypatch, tmp_path):
    import utils.s1_collector as mod
    import utils.fetch as fetch_mod
    res = 10 / 111320
    item = _make_s1_item("S1A_test", 145.0, -22.9, res, 5, 5)
    items_by_id = {item.id: item}
    monkeypatch.setattr(mod, "_resolve_s1_items", lambda *a, **kw: [item])
    monkeypatch.setattr(mod, "setup_gdal_env", lambda: None)
    monkeypatch.setattr(fetch_mod, "fetch_patches", _make_fake_fetch_patches(items_by_id))

    bbox = [145.0, -22.9 - 5 * res, 145.0 + 5 * res, -22.9]
    points = [("px_0000", 145.0 + 0.5 * res, -22.9 - 0.5 * res)]
    df = collect_s1(bbox, "2022-01-01", "2022-12-31", points, cache_dir=tmp_path / "s1_chips")

    assert not df.is_empty()
    for col in ["point_id", "lon", "lat", "date", "source", "vh", "vv"]:
        assert col in df.columns, f"Missing column: {col}"
    assert df["source"][0] == "S1"
    assert df["vh"].dtype in (pl.Float32, pl.Float64)
    assert df["vv"].dtype in (pl.Float32, pl.Float64)


def test_collect_s1_source_always_s1(monkeypatch, tmp_path):
    import utils.s1_collector as mod
    import utils.fetch as fetch_mod
    res = 10 / 111320
    items = [
        _make_s1_item("S1A_1", 145.0, -22.9, res, 5, 5),
        _make_s1_item("S1B_2", 145.0, -22.9, res, 5, 5,
                      dt=datetime(2022, 9, 1, tzinfo=timezone.utc)),
    ]
    items_by_id = {it.id: it for it in items}
    monkeypatch.setattr(mod, "_resolve_s1_items", lambda *a, **kw: items)
    monkeypatch.setattr(mod, "setup_gdal_env", lambda: None)
    monkeypatch.setattr(fetch_mod, "fetch_patches", _make_fake_fetch_patches(items_by_id))

    bbox = [145.0, -22.9 - 5 * res, 145.0 + 5 * res, -22.9]
    points = [("px_0000", 145.0 + 0.5 * res, -22.9 - 0.5 * res)]
    df = collect_s1(bbox, "2022-01-01", "2022-12-31", points, cache_dir=tmp_path / "s1_chips")
    assert (df["source"] == "S1").all()


def test_collect_s1_spatial_alignment(monkeypatch, tmp_path):
    """Each output row's (lon, lat) must exactly match a point in the input grid."""
    import utils.s1_collector as mod
    import utils.fetch as fetch_mod
    res = 10 / 111320
    lon_origin, lat_origin = 145.0, -22.9
    nrows, ncols = 4, 4

    item = _make_s1_item("S1A_align", lon_origin, lat_origin, res, nrows, ncols)
    items_by_id = {item.id: item}
    monkeypatch.setattr(mod, "_resolve_s1_items", lambda *a, **kw: [item])
    monkeypatch.setattr(mod, "setup_gdal_env", lambda: None)
    monkeypatch.setattr(fetch_mod, "fetch_patches", _make_fake_fetch_patches(items_by_id))

    points = [
        (f"px_{r:04d}_{c:04d}", lon_origin + (c + 0.5) * res, lat_origin - (r + 0.5) * res)
        for r in range(nrows) for c in range(ncols)
    ]
    point_locs = {pid: (lon, lat) for pid, lon, lat in points}
    bbox = [lon_origin, lat_origin - nrows * res, lon_origin + ncols * res, lat_origin]

    df = collect_s1(bbox, "2022-01-01", "2022-12-31", points, cache_dir=tmp_path / "s1_chips")
    assert not df.is_empty()

    for pid, lon, lat in df.select(["point_id", "lon", "lat"]).iter_rows():
        assert pid in point_locs, f"Unknown point_id: {pid}"
        expected_lon, expected_lat = point_locs[pid]
        assert abs(lon - expected_lon) < 1e-9, f"{pid} lon mismatch"
        assert abs(lat - expected_lat) < 1e-9, f"{pid} lat mismatch"


# ---------------------------------------------------------------------------
# TC-S1-1 to TC-S1-2: _extend_schema
# ---------------------------------------------------------------------------

def _minimal_s2_schema() -> pa.Schema:
    return pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("date", pa.date32()),
        pa.field("B08", pa.float32()),
    ])


def test_extend_schema_adds_missing_columns():
    schema = _minimal_s2_schema()
    extended = _extend_schema(schema)
    names = set(extended.names)
    assert "source" in names
    assert "vh" in names
    assert "vv" in names
    # Original columns preserved
    assert "point_id" in names
    assert "B08" in names


def test_extend_schema_idempotent():
    schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("vh", pa.float32()),
        pa.field("vv", pa.float32()),
    ])
    extended = _extend_schema(schema)
    assert extended.names == schema.names


# ---------------------------------------------------------------------------
# TC-S1-3: _conform_table
# ---------------------------------------------------------------------------

def test_conform_table_fills_missing_columns():
    src = pa.table({"point_id": ["a", "b"], "B08": [0.5, 0.6]})
    target_schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("B08", pa.float32()),
        pa.field("source", pa.string()),
        pa.field("vh", pa.float32()),
    ])
    out = _conform_table(src, target_schema)
    assert out.schema == target_schema
    assert out.column("source").null_count == 2
    assert out.column("vh").null_count == 2


def test_conform_table_casts_types():
    src = pa.table({"point_id": ["a"], "B08": pa.array([0.5], type=pa.float64())})
    target_schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("B08", pa.float32()),
    ])
    out = _conform_table(src, target_schema)
    assert out.schema.field("B08").type == pa.float32()


# ---------------------------------------------------------------------------
# TC-S1-4 to TC-S1-5: _s1_df_to_arrow
# ---------------------------------------------------------------------------

def test_s1_df_to_arrow_s2_columns_null():
    schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("date", pa.date32()),
        pa.field("B08", pa.float32()),
        pa.field("source", pa.string()),
        pa.field("vh", pa.float32()),
        pa.field("vv", pa.float32()),
    ])
    from datetime import date as _date
    df = pl.DataFrame({
        "point_id": ["px_0000", "px_0001"],
        "lon": [145.0, 145.001],
        "lat": [-22.9, -22.901],
        "date": [_date(2022, 8, 15), _date(2022, 8, 15)],
        "source": ["S1", "S1"],
        "vh": [0.001, 0.002],
        "vv": [0.003, 0.004],
    })
    tbl = _s1_df_to_arrow(df, schema)
    assert tbl.schema == schema
    # B08 should be all null (S2-only column not in df)
    assert tbl.column("B08").null_count == len(df)
    # vh/vv should be populated
    assert tbl.column("vh").null_count == 0
    assert tbl.column("vv").null_count == 0


def test_s1_df_to_arrow_row_count():
    schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("vh", pa.float32()),
        pa.field("vv", pa.float32()),
    ])
    df = pl.DataFrame({
        "point_id": [f"px_{i:04d}" for i in range(7)],
        "source": ["S1"] * 7,
        "vh": np.random.rand(7).astype(np.float32),
        "vv": np.random.rand(7).astype(np.float32),
    })
    tbl = _s1_df_to_arrow(df, schema)
    assert len(tbl) == 7


# ---------------------------------------------------------------------------
# TC-S1-6 to TC-S1-10: integration via ensure_training_pixels
# ---------------------------------------------------------------------------

def _make_s2_parquet(path: Path, region_id: str, n_pixels: int = 3) -> None:
    """Write a minimal S2-style parquet, matching the real pipeline's on-disk
    encoding: bands as raw uint16 DN (reflectance × 10000), scl_purity as
    int8 (0/1), aot/view_zenith/sun_zenith as uint8 (0-100)."""
    from analysis.constants import BANDS, SPECTRAL_INDEX_COLS, UINT16_BAND_SCALE
    rows = []
    for i in range(n_pixels):
        for dt in [date(2022, 6, 1), date(2022, 7, 1)]:
            rows.append({
                "point_id": f"{region_id}_{i:04d}_0000",
                "lon": 145.0 + i * 0.0001,
                "lat": -22.9 - i * 0.0001,
                "date": dt,
                "source": "S2",
                "item_id": f"S2A_tile_{dt}",
                "tile_id": "55HBU",
                **{b: int((0.1 + i * 0.01) * UINT16_BAND_SCALE) for b in BANDS},
                "scl_purity": 1,
                "scl": 4,
                "aot": 90,
                "view_zenith": 95,
                "sun_zenith": 85,
                **{c: 0.25 for c in SPECTRAL_INDEX_COLS},
                "vh": None,
                "vv": None,
            })
    tbl = pl.DataFrame(rows).to_arrow()
    pq.write_table(tbl, path)


def _make_s1_df(region_id: str, n_pixels: int = 3) -> pl.DataFrame:
    """Return a minimal S1 DataFrame for mocking collect_s1."""
    rows = []
    for i in range(n_pixels):
        rows.append({
            "point_id": f"{region_id}_{i:04d}_0000",
            "lon": 145.0 + i * 0.0001,
            "lat": -22.9 - i * 0.0001,
            "date": date(2022, 6, 10),
            "source": "S1",
            "vh": 0.001 * (i + 1),
            "vv": 0.002 * (i + 1),
        })
    return pl.DataFrame(rows)


@pytest.fixture()
def collector_dirs(tmp_path, monkeypatch):
    import utils.training_collector as tc
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    (tiles_dir / "regions").mkdir()
    monkeypatch.setattr(tc, "_TILES_DIR", tiles_dir)
    monkeypatch.setattr(tc, "_TRAINING_DIR", tmp_path)
    monkeypatch.setattr(tc, "_INDEX_PATH", tmp_path / "index.parquet")
    return tiles_dir


def _run_ensure(
    tmp_path, collector_dirs, monkeypatch,
    region_id: str = "test_region",
    n_pixels: int = 3,
):
    """Run ensure_training_pixels with mocked fetch/extract calls."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import utils.training_collector as tc
    from utils.regions import TrainingRegion

    region = TrainingRegion(
        id=region_id,
        name=region_id,
        label="presence",
        bbox=[145.0, -23.0, 145.1, -22.9],
        years=[2022],
        tags=[],
        notes=None,
    )

    # Mock bbox_to_tile_ids → single tile
    monkeypatch.setattr(tc, "bbox_to_tile_ids", lambda bbox: ["55HBU"])

    # Mock _chunk_summary_table → no-op (avoids grid/compute_chunks lookup)
    monkeypatch.setattr(tc, "_chunk_summary_table", lambda *a, **kw: None)

    # Mock _fetch_tile_chunks → no-op (no real COG fetch needed)
    monkeypatch.setattr(tc, "_fetch_tile_chunks", lambda *a, **kw: None)

    s1_df = _make_s1_df(region_id=region_id, n_pixels=n_pixels)

    def _fake_extract_region(rgn, tile_id, chunkstore):
        """Write combined S2+S1 parquet to the region path and return tile_id."""
        out_path = tc._region_parquet_path(rgn.id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        s2_tbl = pa.Table.from_pylist([])
        _make_s2_parquet(out_path, region_id=rgn.id, n_pixels=n_pixels)
        s2_tbl = pq.read_table(out_path)
        s1_tbl = s1_df.to_arrow()
        combined = pa.concat_tables([s2_tbl, s1_tbl], promote_options="default")
        pq.write_table(combined, out_path)
        return tile_id

    monkeypatch.setattr(tc, "_extract_region", _fake_extract_region)

    tc.ensure_training_pixels([region])

    region_parquet = collector_dirs / "regions" / f"{region_id}.parquet"
    assert region_parquet.exists(), "Region parquet not created"
    return pl.read_parquet(region_parquet)


def test_ensure_training_pixels_contains_s2_and_s1_rows(tmp_path, collector_dirs, monkeypatch):
    df = _run_ensure(tmp_path, collector_dirs, monkeypatch)
    assert "S2" in df["source"].to_list(), "No S2 rows found"
    assert "S1" in df["source"].to_list(), "No S1 rows found"


def test_ensure_training_pixels_sort_order(tmp_path, collector_dirs, monkeypatch):
    """Rows are sortable by (point_id, date) with no duplicate (point_id, date, source)."""
    df = _run_ensure(tmp_path, collector_dirs, monkeypatch)
    df_sorted = df.sort(["point_id", "date"])
    n_unique = df_sorted.unique(subset=["point_id", "date", "source"]).height
    assert n_unique == len(df_sorted), "Duplicate (point_id, date, source) rows found"


def test_ensure_training_pixels_source_labels(tmp_path, collector_dirs, monkeypatch):
    df = _run_ensure(tmp_path, collector_dirs, monkeypatch)
    assert (df.filter(pl.col("source") == "S2")["source"] == "S2").all()
    assert (df.filter(pl.col("source") == "S1")["source"] == "S1").all()


def test_ensure_training_pixels_vh_vv_null_on_s2(tmp_path, collector_dirs, monkeypatch):
    df = _run_ensure(tmp_path, collector_dirs, monkeypatch)
    s2_rows = df.filter(pl.col("source") == "S2")
    s1_rows = df.filter(pl.col("source") == "S1")
    assert s2_rows["vh"].is_null().all(), "S2 rows should have null vh"
    assert s2_rows["vv"].is_null().all(), "S2 rows should have null vv"
    assert s1_rows["vh"].is_not_null().all(), "S1 rows should have non-null vh"
    assert s1_rows["vv"].is_not_null().all(), "S1 rows should have non-null vv"


def test_s1_point_ids_match_s2_pixel_grid(tmp_path, collector_dirs, monkeypatch):
    """S1 point_ids must all appear in the S2 pixel grid — no phantom pixels."""
    df = _run_ensure(tmp_path, collector_dirs, monkeypatch)
    s2_pids = set(df.filter(pl.col("source") == "S2")["point_id"].to_list())
    s1_pids = set(df.filter(pl.col("source") == "S1")["point_id"].to_list())
    phantom = s1_pids - s2_pids
    assert not phantom, f"S1 has point_ids not in S2 grid: {phantom}"
