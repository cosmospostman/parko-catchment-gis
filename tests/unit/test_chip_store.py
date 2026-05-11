"""Unit tests for DiskChipStore and CachedNpzChipStore.

Synthetic 5×5 GeoTIFF chips are written to a tmp_path fixture directory
so the tests are fully self-contained and require no pre-staged inputs.

Tests
-----
1. get() returns a 2-D array with the correct shape.
2. Missing chip raises FileNotFoundError with path context.
3. CachedNpzChipStore.get_all_points raises ValueError when points fall outside the patch.
   Root cause documented: a patch cached for training bbox A is silently reused at
   inference for bbox B (~10 km away).  np.clip maps every out-of-bounds pixel to the
   same patch edge, making all pixels return identical values — destroying spatial
   variation and producing flat model outputs.  The guard converts the silent corruption
   into a loud failure.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from utils.chip_store import DiskChipStore


# ---------------------------------------------------------------------------
# Fixture: a small inputs/ tree with one synthetic chip
# ---------------------------------------------------------------------------

ITEM_ID = "S2A_20220815T003"
BAND = "B03"
POINT_ID = "pt_001"
CHIP_SHAPE = (5, 5)


def _write_chip(path: Path, shape: tuple[int, int] = CHIP_SHAPE) -> np.ndarray:
    """Write a synthetic single-band GeoTIFF to path; return the array written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    data = rng.uniform(0.0, 0.3, size=shape).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, shape[1], shape[0])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return data


@pytest.fixture()
def chip_store(tmp_path) -> tuple[DiskChipStore, np.ndarray]:
    """Return a DiskChipStore pointed at tmp_path and the array that was written."""
    chip_path = tmp_path / ITEM_ID / f"{BAND}_{POINT_ID}.tif"
    written = _write_chip(chip_path)
    store = DiskChipStore(inputs_dir=tmp_path)
    return store, written


# ---------------------------------------------------------------------------
# Test 1: get() returns correct array shape
# ---------------------------------------------------------------------------

def test_get_returns_correct_shape(chip_store):
    store, written = chip_store
    arr = store.get(ITEM_ID, BAND, POINT_ID)
    assert arr.shape == CHIP_SHAPE


def test_get_returns_correct_values(chip_store):
    store, written = chip_store
    arr = store.get(ITEM_ID, BAND, POINT_ID)
    np.testing.assert_array_almost_equal(arr, written)


# ---------------------------------------------------------------------------
# Test 2: Missing chip raises FileNotFoundError with path context
# ---------------------------------------------------------------------------

def test_missing_chip_raises_file_not_found(tmp_path):
    store = DiskChipStore(inputs_dir=tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        store.get(ITEM_ID, BAND, "nonexistent_point")
    message = str(exc_info.value)
    # The error must name the expected path so the user knows what to look for
    assert "nonexistent_point" in message
    assert ITEM_ID in message
    assert BAND in message


def test_missing_chip_error_includes_full_path(tmp_path):
    store = DiskChipStore(inputs_dir=tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        store.get(ITEM_ID, BAND, "pt_missing")
    # Full path should be in the message
    assert str(tmp_path) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 3: CachedNpzChipStore raises when points fall outside the cached patch
#
# Regression for the "flat model output" bug:
#   - Training collect caches patches for bbox A into a shared tile chip dir.
#   - Inference collect for bbox B (~10 km away) finds those .npz files and reuses them.
#   - CachedNpzChipStore._pixel_coords clips out-of-bounds row/col to the patch edge,
#     so every pixel maps to the same corner → all band values identical → zero spatial
#     variation → model output is constant for the entire scene.
#   - The guard converts this into a ValueError so the failure is loud, not silent.
# ---------------------------------------------------------------------------

def _write_npz_patch(
    cache_dir: Path,
    item_id: str,
    band: str,
    patch_arr: np.ndarray,
    transform,
    crs,
) -> None:
    """Write a synthetic patch to the CachedNpzChipStore .npz layout."""
    from utils.fetch import _save_patch_cache, _cache_path
    path = _cache_path(cache_dir, item_id, band)
    _save_patch_cache(path, (patch_arr, transform, crs))


def test_cached_npz_store_raises_when_points_far_outside_patch(tmp_path):
    """CachedNpzChipStore must raise ValueError when points are >1 pixel outside the patch.

    Simulates the cross-bbox cache poisoning: a 10×10 pixel patch cached for
    training bbox A is loaded by an inference run whose points are 1° away.
    Before the fix, np.clip silently mapped every point to the patch edge,
    returning the same DN value for all pixels.  After the fix, a ValueError
    is raised immediately so the caller knows to re-fetch.

    A 1-pixel slop is allowed for tile-edge rounding (coarser bands like SCL
    at 20 m may be 1 pixel short of the bbox due to floating-point window
    rounding in rasterio — those pixels are clip-clamped, not errored).
    """
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from utils.chip_store import CachedNpzChipStore

    item_id = "S2A_54LWH_20250430_0_L2A"
    band = "B08"

    # Patch covers a small area anchored at UTM (543000, 8243000) — training bbox
    crs = CRS.from_epsg(32754)
    patch_arr = np.arange(100, dtype=np.float32).reshape(10, 10) * 100
    transform = from_bounds(543_000, 8_243_000, 543_100, 8_243_100, 10, 10)
    _write_npz_patch(tmp_path, item_id, band, patch_arr, transform, crs)

    # Inference pixels are 10 km east — >1 pixel outside in every direction
    scoring_lons = np.array([141.533, 141.534, 141.535], dtype=np.float64)
    scoring_lats = np.array([-15.808, -15.808, -15.808], dtype=np.float64)
    point_coords = {f"px_{i:04d}_0000": (float(scoring_lons[i]), float(scoring_lats[i]))
                    for i in range(len(scoring_lons))}

    store = CachedNpzChipStore(cache_dir=tmp_path, point_coords=point_coords, bands=[band])

    with pytest.raises(ValueError, match="outside the cached patch"):
        store.get_all_points(item_id, band)


def test_cached_npz_store_clips_single_pixel_edge_overhang(tmp_path):
    """1-pixel edge overhang is silently clamped, not raised as an error.

    Coarser-resolution bands (SCL at 20 m, AOT at 60 m) are fetched at a window
    computed from the same WGS84 bbox as the 10 m bands but rounded independently,
    so they can end up 1 pixel short.  Points that project to row/col -1 or h/w
    should be clamped to the nearest valid pixel, not raise ValueError.
    """
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from utils.chip_store import CachedNpzChipStore
    from pyproj import Transformer

    item_id = "S2A_54LWH_20250430_0_L2A"
    band = "SCL"

    crs = CRS.from_epsg(32754)
    # 5×5 patch; pixel centres at 0.5, 1.5, ... within the patch
    patch_arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    x0, y0, x1, y1 = 553_000, 8_249_000, 553_050, 8_249_050
    transform = from_bounds(x0, y0, x1, y1, 5, 5)  # 10 m/px
    _write_npz_patch(tmp_path, item_id, band, patch_arr, transform, crs)

    # Place one point exactly 1 pixel past each edge (row=-1, col=5)
    t_inv = Transformer.from_crs("EPSG:32754", "EPSG:4326", always_xy=True)
    lon_inside, lat_inside   = t_inv.transform(553_025, 8_249_025)   # centre
    lon_past_right, lat_past = t_inv.transform(553_051, 8_249_025)   # 1 px past right edge
    point_coords = {
        "px_inside":     (lon_inside, lat_inside),
        "px_past_right": (lon_past_right, lat_past),
    }

    store = CachedNpzChipStore(cache_dir=tmp_path, point_coords=point_coords, bands=[band])
    # Must not raise — the 1-pixel overhang is within the allowed slop
    result = store.get_all_points(item_id, band)
    assert result is not None
    assert len(result) == 2


def test_cached_npz_store_returns_values_when_points_inside_patch(tmp_path):
    """Sanity check: get_all_points succeeds and returns spatially distinct values
    when the points are correctly located within the cached patch.
    """
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from utils.chip_store import CachedNpzChipStore
    from pyproj import Transformer

    item_id = "S2A_54LWH_20250430_0_L2A"
    band = "B08"

    crs = CRS.from_epsg(32754)
    # 10×10 patch at 100 m/px: each column has a distinct value
    patch_arr = np.tile(np.arange(10, dtype=np.float32) * 100, (10, 1))
    x0, y0, x1, y1 = 543_000, 8_243_000, 544_000, 8_244_000
    transform = from_bounds(x0, y0, x1, y1, 10, 10)

    _write_npz_patch(tmp_path, item_id, band, patch_arr, transform, crs)

    # Project patch pixel centres back to WGS84 so points are guaranteed in-bounds
    t_inv = Transformer.from_crs("EPSG:32754", "EPSG:4326", always_xy=True)
    pixel_size = 100.0
    utm_xs = np.array([x0 + (i + 0.5) * pixel_size for i in range(5)])
    utm_ys = np.full(5, (y0 + y1) / 2)
    lons, lats = t_inv.transform(utm_xs, utm_ys)

    point_coords = {f"px_{i:04d}_0000": (float(lons[i]), float(lats[i])) for i in range(5)}
    store = CachedNpzChipStore(cache_dir=tmp_path, point_coords=point_coords, bands=[band])

    vals = store.get_all_points(item_id, band)

    assert vals is not None
    assert len(vals) == 5
    # Each column is distinct (0, 100, 200, ...) — no clamp flattening
    assert len(np.unique(vals)) == 5, (
        f"Expected 5 distinct values, got {np.unique(vals)} — "
        "coordinate clamping is collapsing spatial variation"
    )
