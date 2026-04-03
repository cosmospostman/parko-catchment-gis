"""Unit tests for extract_observations().

All tests use synthetic chips written to tmp_path — no pre-staged data required.

Tests
-----
 1. Returns one Observation per clear acquisition × point.
 2. Wholly-clouded acquisitions (no clear SCL pixels) are excluded.
 3. Acquisitions with no SCL chip at all are excluded.
 4. Returned Observations have the expected band keys.
 5. Band values are in surface-reflectance range [0, 1], not raw DN, not NaN.
 6. Center pixel (index [2,2]) is extracted, not mean or other aggregate.
 7. scl_purity equals the fraction of clear pixels in the SCL chip.
 8. AOT quality is 1.0 - center-pixel AOT value.
 9. Missing AOT chip defaults to quality 1.0.
10. Missing VZA / SZA chips default to quality 1.0.
11. greenness_z is always 1.0 at this stage.
12. Observation.date strips tzinfo (naive UTC).
"""

import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from analysis.constants import AOT_BAND, SCL_BAND
from analysis.timeseries.extraction import extract_observations
from stage0.chip_store import DiskChipStore


# ---------------------------------------------------------------------------
# Chip writing helpers
# ---------------------------------------------------------------------------

def _write_chip(
    path: Path,
    data: np.ndarray,
    shape: tuple[int, int] = (5, 5),
) -> None:
    """Write a float32 single-band GeoTIFF to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
        dst.write(data.astype(np.float32), 1)


def _const_chip(value: float, shape: tuple[int, int] = (5, 5)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fake pystac.Item factory
# ---------------------------------------------------------------------------

def _make_item(
    item_id: str = "S2A_20220815T003",
    dt: datetime | None = None,
    tile_id: str = "55HBU",
) -> SimpleNamespace:
    """Return a minimal duck-typed pystac.Item substitute."""
    if dt is None:
        dt = datetime(2022, 8, 15, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=item_id,
        datetime=dt,
        properties={"s2:mgrs_tile": tile_id},
    )


# ---------------------------------------------------------------------------
# Fixture: a single clear item + one point, all required chips written
# ---------------------------------------------------------------------------

POINT_ID = "pt_001"
POINT = (POINT_ID, 143.5, -16.0)  # (point_id, lon, lat)
BANDS_3 = ["B02", "B03", "B04"]


def _write_clear_scene(inputs_dir: Path, item_id: str, point_id: str,
                        bands: list[str], band_value: float = 0.1) -> None:
    """Write SCL chip (all clear = 4) and spectral band chips for one (item, point)."""
    # SCL: all vegetation (class 4 = clear)
    scl_data = _const_chip(4.0)
    _write_chip(inputs_dir / item_id / f"{SCL_BAND}_{point_id}.tif", scl_data)
    for band in bands:
        _write_chip(inputs_dir / item_id / f"{band}_{point_id}.tif", _const_chip(band_value))


def _write_cloudy_scene(inputs_dir: Path, item_id: str, point_id: str) -> None:
    """Write SCL chip where all pixels are cloud (class 9 = not clear)."""
    scl_data = _const_chip(9.0)
    _write_chip(inputs_dir / item_id / f"{SCL_BAND}_{point_id}.tif", scl_data)


# ---------------------------------------------------------------------------
# Test 1: Returns one Observation per clear acquisition × point
# ---------------------------------------------------------------------------

def test_returns_one_obs_per_clear_acquisition_per_point(tmp_path):
    items = [_make_item("item_A"), _make_item("item_B"), _make_item("item_C_cloudy")]
    points = [("pt_001", 143.5, -16.0), ("pt_002", 143.6, -16.1)]

    for item_id in ("item_A", "item_B"):
        for point_id, *_ in points:
            _write_clear_scene(tmp_path, item_id, point_id, BANDS_3)
    for point_id, *_ in points:
        _write_cloudy_scene(tmp_path, "item_C_cloudy", point_id)

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations(items, points, store, bands=BANDS_3)

    assert len(result) == 4  # 2 clear items × 2 points


# ---------------------------------------------------------------------------
# Test 2: Wholly-clouded acquisitions are excluded
# ---------------------------------------------------------------------------

def test_cloudy_acquisition_excluded(tmp_path):
    item = _make_item("item_cloudy")
    _write_cloudy_scene(tmp_path, "item_cloudy", POINT_ID)

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=BANDS_3)

    assert result == []


# ---------------------------------------------------------------------------
# Test 3: Missing SCL chip → acquisition excluded
# ---------------------------------------------------------------------------

def test_missing_scl_chip_excluded(tmp_path):
    item = _make_item("item_no_scl")
    # Write band chips but no SCL chip
    for band in BANDS_3:
        _write_chip(
            tmp_path / "item_no_scl" / f"{band}_{POINT_ID}.tif",
            _const_chip(0.1),
        )

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=BANDS_3)

    assert result == []


# ---------------------------------------------------------------------------
# Test 4: Returned Observations have expected band keys
# ---------------------------------------------------------------------------

def test_observation_has_expected_band_keys(tmp_path):
    item = _make_item()
    _write_clear_scene(tmp_path, item.id, POINT_ID, BANDS_3)

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=BANDS_3)

    assert len(result) == 1
    assert set(result[0].bands.keys()) == set(BANDS_3)


# ---------------------------------------------------------------------------
# Test 5: Band values in [0, 1], not raw DN, not NaN
# ---------------------------------------------------------------------------

def test_band_values_in_surface_reflectance_range(tmp_path):
    item = _make_item()
    _write_clear_scene(tmp_path, item.id, POINT_ID, BANDS_3, band_value=0.15)

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=BANDS_3)

    assert len(result) == 1
    for band, value in result[0].bands.items():
        assert not math.isnan(value), f"NaN for band {band}"
        assert 0.0 <= value <= 1.0, f"Out-of-range value {value} for band {band}"


# ---------------------------------------------------------------------------
# Test 6: Center pixel [2, 2] is extracted (not mean or other aggregate)
# ---------------------------------------------------------------------------

def test_center_pixel_extracted(tmp_path):
    item = _make_item()
    # SCL: all clear
    _write_chip(tmp_path / item.id / f"{SCL_BAND}_{POINT_ID}.tif", _const_chip(4.0))
    # Band chip: all zeros except the center pixel
    data = np.zeros((5, 5), dtype=np.float32)
    data[2, 2] = 0.12345
    _write_chip(tmp_path / item.id / f"B03_{POINT_ID}.tif", data)

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=["B03"])

    assert len(result) == 1
    assert result[0].bands["B03"] == pytest.approx(0.12345)


# ---------------------------------------------------------------------------
# Test 7: scl_purity equals fraction of clear pixels in SCL chip
# ---------------------------------------------------------------------------

def test_scl_purity_is_fraction_of_clear_pixels(tmp_path):
    item = _make_item()
    # 5×5 = 25 pixels. Set 10 clear (value 4), 15 cloudy (value 9).
    scl_data = np.full((5, 5), 9, dtype=np.float32)
    # Fill first 10 pixels (row-major) with clear value 4
    flat = scl_data.ravel()
    flat[:10] = 4.0
    scl_data = flat.reshape(5, 5)
    _write_chip(tmp_path / item.id / f"{SCL_BAND}_{POINT_ID}.tif", scl_data)
    _write_chip(tmp_path / item.id / f"B03_{POINT_ID}.tif", _const_chip(0.1))

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=["B03"])

    assert len(result) == 1
    assert result[0].quality.scl_purity == pytest.approx(10 / 25)


# ---------------------------------------------------------------------------
# Test 8: AOT quality = 1.0 - center-pixel AOT value
# ---------------------------------------------------------------------------

def test_aot_quality_inverted_from_center_pixel(tmp_path):
    item = _make_item()
    _write_clear_scene(tmp_path, item.id, POINT_ID, ["B03"])
    # AOT chip: all 0.2
    _write_chip(tmp_path / item.id / f"{AOT_BAND}_{POINT_ID}.tif", _const_chip(0.2))

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=["B03"])

    assert len(result) == 1
    assert result[0].quality.aot == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Test 9: Missing AOT chip → aot quality defaults to 1.0
# ---------------------------------------------------------------------------

def test_missing_aot_defaults_to_one(tmp_path):
    item = _make_item()
    _write_clear_scene(tmp_path, item.id, POINT_ID, ["B03"])
    # Do NOT write an AOT chip

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=["B03"])

    assert len(result) == 1
    assert result[0].quality.aot == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 10: Missing VZA / SZA chips → quality defaults to 1.0
# ---------------------------------------------------------------------------

def test_missing_vza_and_sza_default_to_one(tmp_path):
    item = _make_item()
    _write_clear_scene(tmp_path, item.id, POINT_ID, ["B03"])
    # No VZA or SZA chips written

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=["B03"])

    assert len(result) == 1
    assert result[0].quality.view_zenith == pytest.approx(1.0)
    assert result[0].quality.sun_zenith == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 11: greenness_z is always 1.0 at this stage
# ---------------------------------------------------------------------------

def test_greenness_z_is_one(tmp_path):
    item = _make_item()
    _write_clear_scene(tmp_path, item.id, POINT_ID, ["B03"])

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=["B03"])

    assert len(result) == 1
    assert result[0].quality.greenness_z == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 12: Observation.date is naive UTC (tzinfo stripped)
# ---------------------------------------------------------------------------

def test_observation_date_is_naive_utc(tmp_path):
    item = _make_item(dt=datetime(2022, 8, 15, 1, 30, tzinfo=timezone.utc))
    _write_clear_scene(tmp_path, item.id, POINT_ID, ["B03"])

    store = DiskChipStore(inputs_dir=tmp_path)
    result = extract_observations([item], [POINT], store, bands=["B03"])

    assert len(result) == 1
    assert result[0].date == datetime(2022, 8, 15, 1, 30)
    assert result[0].date.tzinfo is None
