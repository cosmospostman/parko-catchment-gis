"""Unit tests for DiskChipStore.

Synthetic 5×5 GeoTIFF chips are written to a tmp_path fixture directory
so the tests are fully self-contained and require no pre-staged inputs.

Tests
-----
1. get() returns a 2-D array with the correct shape.
2. Missing chip raises FileNotFoundError with path context.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from stage0.chip_store import DiskChipStore


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
