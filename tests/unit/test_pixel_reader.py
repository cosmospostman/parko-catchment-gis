"""tests/unit/test_pixel_reader.py — Integration tests for ChunkIndex.

Requires /mnt/external/mitchell to be mounted with 2025/54LWH chunk parquets.
All tests are skipped automatically if the mount is absent.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.compute as pc
import pytest

CHUNKS_ROOT = Path("/mnt/external/mitchell")
YEAR = 2025
TILE_ID = "54LWH"

pytestmark = pytest.mark.skipif(
    not (CHUNKS_ROOT / str(YEAR) / TILE_ID).exists(),
    reason="/mnt/external/mitchell not mounted",
)

# Real coordinates sampled from 54LWH_r03_c07.parquet RG2 row 500.
SAMPLE_LON = 141.7352752685547
SAMPLE_LAT = -15.743045806884766
SAMPLE_PID = "px_4188_6118"

# Lat that falls in the overlap band between RG2 and RG3 of 54LWH_r03_c07.parquet.
# RG2 lat_max = -15.741680, RG3 lat_min = -15.741813 — midpoint is in both.
BOUNDARY_LAT = -15.741745
BOUNDARY_LON = SAMPLE_LON  # same easting column


@pytest.fixture(scope="module")
def idx():
    from utils.pixel_reader import ChunkIndex
    return ChunkIndex(CHUNKS_ROOT, YEAR, TILE_ID)


def test_query_point(idx):
    tbl = idx.query_point(SAMPLE_LON, SAMPLE_LAT)
    assert tbl is not None
    assert tbl.num_rows > 0
    pids = set(tbl.column("point_id").to_pylist())
    assert pids == {SAMPLE_PID}, f"expected single point_id {SAMPLE_PID!r}, got {pids}"
    dates = tbl.column("date").to_pylist()
    assert len(dates) > 1, "expected multiple observations for a pixel"


def test_query_point_rg_boundary(idx):
    # A lat in the RG2/RG3 overlap band should still resolve to exactly one point_id.
    tbl = idx.query_point(BOUNDARY_LON, BOUNDARY_LAT)
    assert tbl is not None
    assert tbl.num_rows > 0
    pids = set(tbl.column("point_id").to_pylist())
    assert len(pids) == 1, f"expected 1 point_id at RG boundary, got {pids}"


def test_query_bbox(idx):
    # Small bbox (~500 m × 500 m) around the sample point.
    lon_min, lon_max = SAMPLE_LON - 0.002, SAMPLE_LON + 0.002
    lat_min, lat_max = SAMPLE_LAT - 0.002, SAMPLE_LAT + 0.002
    tbl = idx.query_bbox(lon_min, lat_min, lon_max, lat_max)
    assert tbl.num_rows > 0

    # All returned pixels must be within the bbox.
    assert pc.all(pc.greater_equal(tbl.column("lon"), lon_min)).as_py()
    assert pc.all(pc.less_equal(tbl.column("lon"), lon_max)).as_py()
    assert pc.all(pc.greater_equal(tbl.column("lat"), lat_min)).as_py()
    assert pc.all(pc.less_equal(tbl.column("lat"), lat_max)).as_py()

    pids = set(tbl.column("point_id").to_pylist())
    assert len(pids) > 1, "expected multiple pixels in a 500 m bbox"


def test_query_bbox_multi_chunk(idx):
    # Bbox straddling the boundary between c07 (lon_max ~141.76440) and c08 (lon_min ~141.76416).
    lon_min, lon_max = 141.762, 141.767
    lat_min, lat_max = -15.748, -15.740
    tbl = idx.query_bbox(lon_min, lat_min, lon_max, lat_max)
    assert tbl.num_rows > 0

    lons = tbl.column("lon").to_pylist()
    # Expect pixels on both sides of the chunk boundary.
    assert min(lons) < 141.764, "expected pixels from the western chunk"
    assert max(lons) > 141.764, "expected pixels from the eastern chunk"
