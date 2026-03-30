"""Unit tests for utils/tiling.py."""
import sys
from pathlib import Path

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.tiling import make_tile_bboxes, merge_tile_rasters


# ---------------------------------------------------------------------------
# make_tile_bboxes
# ---------------------------------------------------------------------------

# Small bbox in EPSG:4326 (roughly 50 km × 50 km over Queensland)
FULL_BBOX = [141.0, -17.0, 141.5, -16.5]
RESOLUTION_M = 10
TILE_SIZE_PX = 256  # 2560 m per tile


def _union_bbox(bboxes):
    """Return the bounding box of a list of bboxes."""
    minx = min(b[0] for b in bboxes)
    miny = min(b[1] for b in bboxes)
    maxx = max(b[2] for b in bboxes)
    maxy = max(b[3] for b in bboxes)
    return [minx, miny, maxx, maxy]


def test_make_tile_bboxes_covers_full_extent():
    """Union of all tile bboxes must cover the full bbox."""
    tiles = make_tile_bboxes(FULL_BBOX, RESOLUTION_M, TILE_SIZE_PX)
    assert len(tiles) > 0

    union = _union_bbox(tiles)
    # Allow tiny floating-point tolerance
    assert union[0] <= FULL_BBOX[0] + 1e-8
    assert union[1] <= FULL_BBOX[1] + 1e-8
    assert union[2] >= FULL_BBOX[2] - 1e-8
    assert union[3] >= FULL_BBOX[3] - 1e-8


def test_make_tile_bboxes_no_overlap():
    """Adjacent tiles must not overlap (beyond a shared edge)."""
    tiles = make_tile_bboxes(FULL_BBOX, RESOLUTION_M, TILE_SIZE_PX)

    for i, a in enumerate(tiles):
        for j, b in enumerate(tiles):
            if i >= j:
                continue
            # Overlap in x AND y simultaneously means real area overlap
            x_overlap = a[0] < b[2] and b[0] < a[2]
            y_overlap = a[1] < b[3] and b[1] < a[3]
            if x_overlap and y_overlap:
                # Check it's only a shared edge (zero-width overlap)
                x_overlap_width = min(a[2], b[2]) - max(a[0], b[0])
                y_overlap_height = min(a[3], b[3]) - max(a[1], b[1])
                area = x_overlap_width * y_overlap_height
                assert area < 1e-10, (
                    f"Tiles {i} and {j} overlap with area {area}: {a}, {b}"
                )


def test_make_tile_bboxes_single_tile():
    """A bbox smaller than one tile returns exactly one tile equal to full bbox."""
    # 100 m × 100 m bbox — much smaller than 256 px × 10 m = 2560 m
    tiny_bbox = [141.0, -17.0, 141.0009, -16.9991]  # ~100 m × 100 m
    tiles = make_tile_bboxes(tiny_bbox, RESOLUTION_M, TILE_SIZE_PX)
    assert len(tiles) == 1

    t = tiles[0]
    assert t[0] == pytest.approx(tiny_bbox[0], abs=1e-8)
    assert t[1] == pytest.approx(tiny_bbox[1], abs=1e-8)
    assert t[2] == pytest.approx(tiny_bbox[2], abs=1e-8)
    assert t[3] == pytest.approx(tiny_bbox[3], abs=1e-8)


def test_make_tile_bboxes_tiles_within_full_bbox():
    """Every tile bbox must lie within (or on the boundary of) the full bbox."""
    tiles = make_tile_bboxes(FULL_BBOX, RESOLUTION_M, TILE_SIZE_PX)
    for t in tiles:
        assert t[0] >= FULL_BBOX[0] - 1e-8
        assert t[1] >= FULL_BBOX[1] - 1e-8
        assert t[2] <= FULL_BBOX[2] + 1e-8
        assert t[3] <= FULL_BBOX[3] + 1e-8


# ---------------------------------------------------------------------------
# merge_tile_rasters
# ---------------------------------------------------------------------------

def _write_tile(path: Path, values: np.ndarray, x_start: float, x_end: float,
                y_start: float, y_end: float, crs: str = "EPSG:7855"):
    """Write a tiny synthetic tile GeoTIFF."""
    H, W = values.shape
    x = np.linspace(x_start, x_end, W)
    y = np.linspace(y_start, y_end, H)
    da = xr.DataArray(values, dims=["y", "x"], coords={"y": y, "x": x})
    da = da.rio.write_crs(crs)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da.rio.to_raster(str(path), driver="GTiff", dtype="float32")


def test_merge_tile_rasters_values(tmp_path):
    """Merge of two side-by-side tiles produces correct pixel values."""
    left = np.full((4, 4), 0.3, dtype=np.float32)
    right = np.full((4, 4), 0.7, dtype=np.float32)

    left_path = tmp_path / "tile_left.tif"
    right_path = tmp_path / "tile_right.tif"

    _write_tile(left_path,  left,  x_start=700000, x_end=710000,
                y_start=-1610000, y_end=-1600000)
    _write_tile(right_path, right, x_start=710000, x_end=720000,
                y_start=-1610000, y_end=-1600000)

    out_path = tmp_path / "merged.tif"
    merge_tile_rasters([left_path, right_path], out_path,
                       nodata=np.nan, crs="EPSG:7855")

    assert out_path.exists()
    result = xr.open_dataarray(str(out_path))
    data = result.values.squeeze()

    # Left half should be ~0.3, right half ~0.7
    assert np.nanmean(data[:, :data.shape[1] // 2]) == pytest.approx(0.3, abs=0.01)
    assert np.nanmean(data[:, data.shape[1] // 2:]) == pytest.approx(0.7, abs=0.01)


def test_merge_tile_rasters_crs_preserved(tmp_path):
    """Output CRS must match the CRS passed to merge_tile_rasters."""
    tile = np.full((4, 4), 0.5, dtype=np.float32)
    tile_path = tmp_path / "tile.tif"
    _write_tile(tile_path, tile, x_start=700000, x_end=710000,
                y_start=-1610000, y_end=-1600000)

    out_path = tmp_path / "merged.tif"
    merge_tile_rasters([tile_path], out_path, nodata=np.nan, crs="EPSG:7855")

    result = xr.open_dataarray(str(out_path))
    assert result.rio.crs is not None
    assert result.rio.crs.to_epsg() == 7855
