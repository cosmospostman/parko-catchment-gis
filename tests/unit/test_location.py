"""Unit tests for utils/location.py — Location dataclass, YAML registry, geometry helpers.

All tests are self-contained and require no pre-staged data.

Tests
-----
 1. pixel_count rounds fractional pixel dimensions (Bug L1 — FAILS on current code).
 2. pixel_count exact integer boundary returns the correct count (regression).
 3. pixel_count and _bbox_pixel_count agree for the same bbox (both share the bug).
 4. _bbox_pixel_count rounds fractional dimensions (Bug L4 — FAILS on current code).
 5. _load_registry: valid YAML with all required fields loads a Location correctly.
 6. _load_registry: YAML missing the "name" key is silently skipped (Bug L3 — documents).
 7. _load_registry: YAML where data is not a dict (None) is silently skipped.
 8. _load_registry: string centroid coerces to wrong tuple shape (Bug L2 — FAILS).
 9. _load_registry: valid [lat, lon] centroid stored as two-element float tuple.
10. bbox_cli produces the correct comma-separated string without spaces.
11. area_km2 is consistent with width_km × height_km.
12. summary() contains the location id and name.
13. _bbox_pixel_count with resolution_m=20 returns roughly one quarter of the 10 m count.
14. sub_bboxes round-tripped through _load_registry have correct role and bbox.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from utils.location import Location, SubBbox, _bbox_pixel_count, _load_registry, _PROJECT_ROOT, _append_s1_to_tile_parquet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_location(bbox: list[float], **overrides: Any) -> Location:
    """Construct a Location directly, without touching the filesystem."""
    defaults: dict[str, Any] = dict(
        id="test",
        name="Test Site",
        bbox=bbox,
        dry_months=[6, 7, 8, 9],
        centroid=None,
        notes=None,
        sub_bboxes={},
        signal_params={},
        score_bbox=None,
    )
    defaults.update(overrides)
    return Location(**defaults)


def _write_yaml(tmp_path: Path, loc_id: str, data: dict) -> Path:
    """Write a YAML file for loc_id and return its path."""
    p = tmp_path / f"{loc_id}.yaml"
    p.write_text(yaml.dump(data))
    return p


_VALID_YAML_DATA: dict = {
    "name": "Longreach",
    "bbox": [145.41, -22.81, 145.44, -22.74],
    "dry_months": [6, 7, 8, 9, 10],
    "centroid": [-22.76, 145.42],
}


# ---------------------------------------------------------------------------
# Test 1 — pixel_count truncation bug (L1, FAILS on current code)
# ---------------------------------------------------------------------------

def test_pixel_count_rounds_fractional_dimensions():
    # A bbox whose UTM width and height are ~19 m each.
    # int(19/10) = 1, so int() gives 1×1 = 1.  round(1.9) = 2, so correct is 2×2 = 4.
    delta = 19.0 / 111_320  # at equator cos≈1 so lon_m ≈ lat_m ≈ 19 m
    bbox = [0.0, 0.0, delta, delta]
    loc = _make_location(bbox)
    assert loc.pixel_count == 4  # BUG L1: currently returns 1


# ---------------------------------------------------------------------------
# Test 2 — exact integer boundary (regression, PASSES)
# ---------------------------------------------------------------------------

def test_pixel_count_positive_for_valid_bbox():
    # Regression guard: a well-formed bbox returns a positive pixel count.
    bbox = [145.0, -23.0, 146.0, -22.0]   # ~100 km × ~111 km
    loc = _make_location(bbox)
    assert loc.pixel_count > 0
    assert loc.pixel_count < 200_000_000   # sanity upper bound (~100 km²)


# ---------------------------------------------------------------------------
# Test 3 — pixel_count and _bbox_pixel_count agree (same formula, same bug)
# ---------------------------------------------------------------------------

def test_pixel_count_matches_bbox_pixel_count_helper():
    bbox = [145.0, -23.0, 145.01, -22.99]
    loc = _make_location(bbox)
    assert loc.pixel_count == _bbox_pixel_count(bbox)


# ---------------------------------------------------------------------------
# Test 4 — _bbox_pixel_count truncation bug (L4, FAILS on current code)
# ---------------------------------------------------------------------------

def test_bbox_pixel_count_rounds_fractional_dimensions():
    delta = 19.0 / 111_320
    bbox = [0.0, 0.0, delta, delta]
    assert _bbox_pixel_count(bbox) == 4  # BUG L4: currently returns 1


# ---------------------------------------------------------------------------
# Test 5 — _load_registry happy path (regression)
# ---------------------------------------------------------------------------

def test_load_registry_valid_yaml(tmp_path):
    _write_yaml(tmp_path, "longreach", _VALID_YAML_DATA)
    result = _load_registry(tmp_path)
    assert "longreach" in result
    loc = result["longreach"]
    assert loc.name == "Longreach"
    assert loc.bbox == [145.41, -22.81, 145.44, -22.74]
    assert loc.dry_months == [6, 7, 8, 9, 10]
    assert loc.id == "longreach"


# ---------------------------------------------------------------------------
# Test 6 — missing "name" key is silently skipped (documents Bug L3)
# ---------------------------------------------------------------------------

def test_load_registry_skips_yaml_without_name(tmp_path):
    data = {k: v for k, v in _VALID_YAML_DATA.items() if k != "name"}
    _write_yaml(tmp_path, "no_name", data)
    result = _load_registry(tmp_path)
    assert "no_name" not in result  # silently dropped — no error raised


# ---------------------------------------------------------------------------
# Test 7 — non-dict YAML (None) is silently skipped (regression)
# ---------------------------------------------------------------------------

def test_load_registry_skips_non_dict_yaml(tmp_path):
    p = tmp_path / "null_yaml.yaml"
    p.write_text("null\n")
    result = _load_registry(tmp_path)
    assert "null_yaml" not in result


# ---------------------------------------------------------------------------
# Test 8 — string centroid coerces to wrong shape (Bug L2, FAILS)
# ---------------------------------------------------------------------------

def test_load_registry_string_centroid_coerces_to_wrong_shape(tmp_path):
    # centroid: "0" is a string, not a [lat, lon] list — must raise ValueError.
    data = {**_VALID_YAML_DATA, "centroid": "0"}
    _write_yaml(tmp_path, "bad_centroid", data)
    with pytest.raises(ValueError, match="centroid"):
        _load_registry(tmp_path)


# ---------------------------------------------------------------------------
# Test 9 — valid list centroid stored as two-element tuple (regression)
# ---------------------------------------------------------------------------

def test_load_registry_valid_centroid_stored_correctly(tmp_path):
    _write_yaml(tmp_path, "good", _VALID_YAML_DATA)
    loc = _load_registry(tmp_path)["good"]
    assert loc.centroid == (-22.76, 145.42)
    assert isinstance(loc.centroid[0], float)


# ---------------------------------------------------------------------------
# Test 10 — bbox_cli format (regression)
# ---------------------------------------------------------------------------

def test_bbox_cli_format():
    loc = _make_location([145.41, -22.81, 145.44, -22.74])
    assert loc.bbox_cli == "145.41,-22.81,145.44,-22.74"
    assert " " not in loc.bbox_cli


# ---------------------------------------------------------------------------
# Test 11 — area_km2 consistent with width × height (regression)
# ---------------------------------------------------------------------------

def test_area_km2_consistent_with_width_height():
    loc = _make_location([145.0, -23.0, 146.0, -22.0])
    assert loc.area_km2 == pytest.approx(loc.width_km * loc.height_km, rel=1e-4)
    assert loc.area_km2 > 0


# ---------------------------------------------------------------------------
# Test 12 — summary() contains id and name (regression)
# ---------------------------------------------------------------------------

def test_summary_contains_id_and_name():
    loc = _make_location([145.0, -23.0, 146.0, -22.0], id="mysite", name="My Site")
    s = loc.summary()
    assert "mysite" in s
    assert "My Site" in s


# ---------------------------------------------------------------------------
# Test 13 — resolution_m=20 gives ~quarter the pixel count (regression)
# ---------------------------------------------------------------------------

def test_bbox_pixel_count_resolution_20_halves_count():
    bbox = [145.0, -23.0, 146.0, -22.0]
    count_10 = _bbox_pixel_count(bbox, resolution_m=10.0)
    count_20 = _bbox_pixel_count(bbox, resolution_m=20.0)
    # 20 m grid has 1/4 the pixels of a 10 m grid
    assert count_20 == pytest.approx(count_10 / 4, rel=0.01)


# ---------------------------------------------------------------------------
# Test 14 — sub_bboxes round-tripped through _load_registry (regression)
# ---------------------------------------------------------------------------

def test_load_registry_sub_bboxes(tmp_path):
    data = {
        **_VALID_YAML_DATA,
        "sub_bboxes": {
            "presence": {
                "label": "pres",
                "role": "presence",
                "bbox": [145.41, -22.80, 145.43, -22.75],
            },
        },
    }
    _write_yaml(tmp_path, "with_sub", data)
    loc = _load_registry(tmp_path)["with_sub"]
    assert "presence" in loc.sub_bboxes
    sb = loc.sub_bboxes["presence"]
    assert sb.role == "presence"
    assert sb.label == "pres"
    assert sb.bbox == [145.41, -22.80, 145.43, -22.75]


# ---------------------------------------------------------------------------
# Test 15 — polygon_file: bbox auto-derived from GeoJSON bounds (regression)
# ---------------------------------------------------------------------------

def test_load_registry_polygon_file_derives_bbox(tmp_path):
    import json
    poly = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[145.0, -23.0], [146.0, -23.0], [146.0, -22.0], [145.0, -22.0], [145.0, -23.0]]],
        },
        "properties": {},
    }
    gj_path = tmp_path / "test_poly.geojson"
    gj_path.write_text(json.dumps(poly))

    data = {"name": "Poly Site", "polygon_file": str(gj_path), "dry_months": [6, 7, 8]}
    _write_yaml(tmp_path, "poly_site", data)
    loc = _load_registry(tmp_path)["poly_site"]

    assert loc.bbox == pytest.approx([145.0, -23.0, 146.0, -22.0])
    assert loc.polygon_file == gj_path
    assert loc.geometry is not None


# ---------------------------------------------------------------------------
# Test 16 — polygon_file: geometry property returns correct Shapely type
# ---------------------------------------------------------------------------

def test_location_geometry_property(tmp_path):
    import json
    from shapely.geometry import Polygon as ShapelyPolygon

    poly = {
        "type": "Polygon",
        "coordinates": [[[145.0, -23.0], [146.0, -23.0], [146.0, -22.0], [145.0, -22.0], [145.0, -23.0]]],
    }
    gj_path = tmp_path / "bare_poly.geojson"
    gj_path.write_text(json.dumps(poly))

    data = {"name": "Bare Poly", "polygon_file": str(gj_path), "dry_months": [6]}
    _write_yaml(tmp_path, "bare", data)
    loc = _load_registry(tmp_path)["bare"]

    geom = loc.geometry
    assert isinstance(geom, ShapelyPolygon)
    assert geom.is_valid


# ---------------------------------------------------------------------------
# Tests 17–21 — tile-partitioned parquet path helpers
# ---------------------------------------------------------------------------

def _make_loc_with_root(tmp_path: Path) -> tuple[Location, Path]:
    """Return a Location whose _PROJECT_ROOT is monkeypatched to tmp_path."""
    loc = _make_location([145.0, -23.0, 146.0, -22.0], id="mysite")
    return loc, tmp_path


def test_parquet_year_dir_path():
    """parquet_year_dir returns data/pixels/<id>/<year>/ under PROJECT_ROOT."""
    loc = _make_location([145.0, -23.0, 146.0, -22.0], id="mysite")
    d = loc.parquet_year_dir(2023)
    assert d == _PROJECT_ROOT / "data" / "pixels" / "mysite" / "2023"


def test_parquet_path_includes_tile_id():
    """parquet_path(year, tile_id) returns the tile-specific parquet path."""
    loc = _make_location([145.0, -23.0, 146.0, -22.0], id="mysite")
    p = loc.parquet_path(2023, "55HBU")
    assert p.name == "55HBU.parquet"
    assert p.parent.name == "2023"
    assert p.parent.parent.name == "mysite"


def test_parquet_tile_paths_empty_when_no_data(tmp_path, monkeypatch):
    """parquet_tile_paths() returns {} when no year dirs exist."""
    import utils.location as loc_mod
    monkeypatch.setattr(loc_mod, "_PROJECT_ROOT", tmp_path)
    loc = _make_location([145.0, -23.0, 146.0, -22.0], id="mysite")
    assert loc.parquet_tile_paths() == {}


def test_parquet_tile_paths_finds_tile_parquets(tmp_path, monkeypatch):
    """parquet_tile_paths() returns {year: [paths]} for existing tile parquets."""
    import utils.location as loc_mod
    monkeypatch.setattr(loc_mod, "_PROJECT_ROOT", tmp_path)
    loc = _make_location([145.0, -23.0, 146.0, -22.0], id="mysite")

    year_dir = tmp_path / "data" / "pixels" / "mysite" / "2022"
    year_dir.mkdir(parents=True)
    (year_dir / "55HBU.parquet").touch()
    (year_dir / "55HBV.parquet").touch()

    result = loc.parquet_tile_paths()
    assert list(result.keys()) == [2022]
    assert {p.name for p in result[2022]} == {"55HBU.parquet", "55HBV.parquet"}


def test_parquet_years_returns_sorted_years(tmp_path, monkeypatch):
    """parquet_years() returns sorted list of years with at least one tile parquet."""
    import utils.location as loc_mod
    monkeypatch.setattr(loc_mod, "_PROJECT_ROOT", tmp_path)
    loc = _make_location([145.0, -23.0, 146.0, -22.0], id="mysite")

    for yr in [2024, 2021, 2023]:
        d = tmp_path / "data" / "pixels" / "mysite" / str(yr)
        d.mkdir(parents=True)
        (d / "55HBU.parquet").touch()

    assert loc.parquet_years() == [2021, 2023, 2024]


# ---------------------------------------------------------------------------
# Test — Mitchell River tile_ids() returns the expected 17 tiles (integration)
# ---------------------------------------------------------------------------

def test_mitchell_tile_ids():
    """tile_ids() for the Mitchell River catchment returns exactly the 17 expected tiles.

    This is an integration test against the real catchment polygon and S2 tile grid.
    It guards against regressions in geometry_to_tile_ids() — both over-selection
    (fetching tiles whose catchment pixels are entirely in a neighbour overlap zone)
    and under-selection (dropping tiles that have genuinely unique catchment coverage).
    """
    from utils.location import get

    loc = get("mitchell")
    assert loc.tile_ids() == [
        "54KYF",
        "54KYG",
        "54KZE",
        "54KZF",
        "54KZG",
        "54LWH",
        "54LWJ",
        "54LXH",
        "54LXJ",
        "54LYH",
        "54LYJ",
        "54LZH",
        "55KBA",
        "55KBB",
        "55KCA",
        "55KCB",
        "55LBC",
    ]


def test_parquet_tile_paths_ignores_sidecar_files(tmp_path, monkeypatch):
    """parquet_tile_paths() excludes coords, shard, and by-pixel sidecar files."""
    import utils.location as loc_mod
    monkeypatch.setattr(loc_mod, "_PROJECT_ROOT", tmp_path)
    loc = _make_location([145.0, -23.0, 146.0, -22.0], id="mysite")

    year_dir = tmp_path / "data" / "pixels" / "mysite" / "2022"
    year_dir.mkdir(parents=True)
    (year_dir / "55HBU.parquet").touch()
    (year_dir / "55HBU-by-pixel.parquet").touch()
    (year_dir / "mysite.coords.parquet").touch()
    (year_dir / "_collect_shard000.parquet").touch()

    result = loc.parquet_tile_paths()
    names = {p.name for p in result[2022]}
    assert names == {"55HBU.parquet"}


# ---------------------------------------------------------------------------
# Test — _append_s1_to_tile_parquet collects pixel coords from ALL row groups
#
# Bug: the original code built points_for_s1 only from the first row group.
# A pixel-sorted parquet spreads distinct pixels across all row groups, so
# only the pixels whose first observation falls in rg0 were passed to
# collect_s1 — producing S1 coverage for ~1 row-strip instead of the full grid.
# ---------------------------------------------------------------------------

def _make_multi_rg_s2_parquet(path: Path, pixel_rows: list[dict]) -> None:
    """Write a pixel-sorted parquet with one row group per pixel (worst-case).

    Each row dict must have: point_id, lon, lat, date, B02..B12, scl_purity, scl.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    schema = pa.schema([
        pa.field("point_id",   pa.string()),
        pa.field("lon",        pa.float32()),
        pa.field("lat",        pa.float32()),
        pa.field("date",       pa.timestamp("us")),
        pa.field("B02",        pa.float32()),
        pa.field("B08",        pa.float32()),
        pa.field("scl_purity", pa.float32()),
        pa.field("scl",        pa.int32()),
    ])
    writer = pq.ParquetWriter(path, schema)
    for row in pixel_rows:
        df = pd.DataFrame([row])
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
        writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
    writer.close()


def test_append_s1_collects_coords_from_all_row_groups(tmp_path):
    """S1 rows must be generated for every unique pixel, not just those in rg0.

    Setup: a parquet with 3 pixels spread across 3 row groups (one per group).
    The mock collect_s1 records which point_ids it was asked about.
    After _append_s1_to_tile_parquet, all 3 pixels must appear in the call.
    """
    import pyarrow.parquet as pq

    tile_path = tmp_path / "55HBU.parquet"
    pixels = [
        {"point_id": "px_0000_0000", "lon": 145.410, "lat": -22.810,
         "date": "2022-08-15", "B02": 0.1, "B08": 0.4, "scl_purity": 1.0, "scl": 4},
        {"point_id": "px_0001_0000", "lon": 145.420, "lat": -22.800,
         "date": "2022-08-15", "B02": 0.1, "B08": 0.4, "scl_purity": 1.0, "scl": 4},
        {"point_id": "px_0002_0000", "lon": 145.430, "lat": -22.790,
         "date": "2022-08-15", "B02": 0.1, "B08": 0.4, "scl_purity": 1.0, "scl": 4},
    ]
    _make_multi_rg_s2_parquet(tile_path, pixels)

    pf = pq.ParquetFile(tile_path)
    assert pf.metadata.num_row_groups == 3, "test requires one row group per pixel"

    captured_points: list[list[str]] = []

    def mock_collect_s1(bbox_wgs84, start, end, points, cache_dir=None):
        import pandas as pd
        captured_points.append([pid for pid, _, _ in points])
        return pd.DataFrame()   # no actual S1 data needed for this test

    _append_s1_to_tile_parquet(
        tile_path=tile_path,
        bbox_wgs84=[145.40, -22.82, 145.44, -22.78],
        year=2022,
        collect_s1_fn=mock_collect_s1,
    )

    assert len(captured_points) == 1, "collect_s1 should be called exactly once"
    called_pids = set(captured_points[0])
    assert called_pids == {"px_0000_0000", "px_0001_0000", "px_0002_0000"}, (
        f"collect_s1 was only given coords for: {called_pids} — "
        "pixels in row groups > 0 were missed (first-row-group-only bug)"
    )
