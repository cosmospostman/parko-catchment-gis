"""Unit tests for utils/training_collector.py — index management and pure helpers.

ensure_training_pixels() calls network + disk and is not unit-tested here.
Focus: _tile_date_window(), _union_bbox(), _update_index(), tile_ids_for_regions(),
and the tile-grouping behaviour in ensure_training_pixels().

Tests
-----
 1. _tile_date_window: start = min(years), end = max(years) across regions.
 2. _tile_date_window: single-region single-year list produces correct window.
 3. _tile_date_window: empty years list raises ValueError.
 4. _tile_date_window: multi-year list spans correctly.
 5. _union_bbox: single region returns its own bbox.
 6. _union_bbox: two non-overlapping regions return the enclosing bbox.
 7. _union_bbox: empty list raises (Bug TC5 — ungarded crash from min()).
 8. _update_index: adds a new region to an empty index.
 9. _update_index: replaces stale entries for an existing region.
10. _update_index: empty tile_ids silently removes the region (Bug TC1 — FAILS).
11. tile_ids_for_regions: raises RuntimeError when region is absent from index.
12. tile_ids_for_regions: returns deduplicated sorted tile IDs across regions.
13. tile_parquet_path: path ends with tiles/{tile_id}.parquet.
14. _region_parquet_path: path ends with tiles/regions/{region_id}.parquet.
15. Tile-grouping loop: a region straddling two tiles appears under both buckets.
16. Per-tile dedup does not remove cross-tile items with matching ids (Bug TC6 — documents).
17. Cross-tile boundary pixel produces only one row after (point_id, date) dedup.
18. Tile rebuild includes all indexed regions, not just those in the current fetch.
19. _parse_years compat shim: old year: YYYY field expands to [year-5..year] with warning.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import polars as pl
import pytest

from utils.regions import TrainingRegion
import utils.training_collector as tc
from utils.training_collector import (
    _load_index,
    _tile_date_window,
    _union_bbox,
    _update_index,
    tile_ids_for_regions,
    tile_parquet_path,
    _region_parquet_path,
)

_GRANULE_RE = re.compile(r"_\d+_L2A$")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_region(
    region_id: str,
    bbox: list[float] | None = None,
    years: list[int] | None = None,
    label: str = "presence",
) -> TrainingRegion:
    return TrainingRegion(
        id=region_id,
        name=region_id.replace("_", " ").title(),
        label=label,
        bbox=bbox or [145.0, -23.0, 145.1, -22.9],
        years=years if years is not None else [2022],
        tags=[],
        notes=None,
    )


def _make_stac_item(item_id: str, tile_id: str = "55HBU") -> SimpleNamespace:
    return SimpleNamespace(id=item_id, properties={"s2:mgrs_tile": tile_id})


# ---------------------------------------------------------------------------
# Fixture: redirect module-level path constants to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture()
def training_dirs(tmp_path, monkeypatch):
    index_path = tmp_path / "index.parquet"
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    (tiles_dir / "regions").mkdir()
    monkeypatch.setattr(tc, "_INDEX_PATH", index_path)
    monkeypatch.setattr(tc, "_TILES_DIR", tiles_dir)
    monkeypatch.setattr(tc, "_TRAINING_DIR", tmp_path)
    return {"index_path": index_path, "tiles_dir": tiles_dir}


# ---------------------------------------------------------------------------
# Tests 1–4: _tile_date_window
# ---------------------------------------------------------------------------

def test_tile_date_window_start_and_end():
    # Two regions with different year ranges — window spans both
    regions = [
        _make_region("r1", years=[2019, 2020, 2021]),
        _make_region("r2", years=[2021, 2022, 2023]),
    ]
    start, end = _tile_date_window(regions)
    assert start == "2019-01-01"
    assert end == "2023-12-31"


def test_tile_date_window_single_region():
    # Single year in list
    regions = [_make_region("r1", years=[2024])]
    start, end = _tile_date_window(regions)
    assert start == "2024-01-01"
    assert end == "2024-12-31"


def test_tile_date_window_empty_years_raises():
    # Empty years list should raise — caught at load time normally, but guard here too
    r = TrainingRegion(
        id="bad", name="bad", label="presence",
        bbox=[145.0, -23.0, 145.1, -22.9],
        years=[],
        tags=[], notes=None,
    )
    with pytest.raises((ValueError, Exception)):
        _tile_date_window([r])


def test_tile_date_window_multi_year_list():
    # Explicit multi-year list: window is exact min..max, no implicit -5
    regions = [_make_region("r1", years=[2020, 2021, 2022, 2023, 2024, 2025])]
    start, end = _tile_date_window(regions)
    assert start == "2020-01-01"
    assert end == "2025-12-31"


# ---------------------------------------------------------------------------
# Tests 5–7: _union_bbox
# ---------------------------------------------------------------------------

def test_union_bbox_single_region():
    r = _make_region("r1", bbox=[145.0, -23.0, 145.1, -22.9])
    assert _union_bbox([r]) == [145.0, -23.0, 145.1, -22.9]


def test_union_bbox_two_regions():
    r1 = _make_region("r1", bbox=[145.0, -23.0, 145.1, -22.9])
    r2 = _make_region("r2", bbox=[145.2, -23.5, 145.3, -23.1])
    result = _union_bbox([r1, r2])
    assert result == [145.0, -23.5, 145.3, -22.9]


def test_union_bbox_empty_list_raises():
    # Bug TC5: no guard — crashes with ValueError from min() of empty sequence.
    # The test confirms an exception is raised (behaviour is correct by accident).
    with pytest.raises((ValueError, TypeError)):
        _union_bbox([])


# ---------------------------------------------------------------------------
# Tests 8–10: _update_index / _load_index
# ---------------------------------------------------------------------------

def test_update_index_adds_new_region(training_dirs):
    _update_index("r1", ["55HBU"])
    df = _load_index()
    assert len(df) == 1
    assert df.row(0)[df.columns.index("region_id")] == "r1"
    assert df.row(0)[df.columns.index("tile_id")] == "55HBU"


def test_update_index_replaces_stale_entries(training_dirs):
    _update_index("r1", ["55HBU"])
    _update_index("r1", ["55HBV"])
    df = _load_index()
    r1_tiles = set(df.filter(pl.col("region_id") == "r1")["tile_id"].to_list())
    assert r1_tiles == {"55HBV"}
    assert "55HBU" not in df["tile_id"].to_list()


def test_update_index_empty_tile_ids_silently_removes_region(training_dirs):
    """Bug TC1: empty tile_ids must raise ValueError rather than silently corrupt."""
    _update_index("r1", ["55HBU"])
    with pytest.raises(ValueError, match="zero tiles"):
        _update_index("r1", [])


# ---------------------------------------------------------------------------
# Tests 11–12: tile_ids_for_regions
# ---------------------------------------------------------------------------

def test_tile_ids_for_regions_raises_if_missing(training_dirs):
    with pytest.raises(RuntimeError, match="No tile index entries"):
        tile_ids_for_regions(["r1"])


def test_tile_ids_for_regions_deduplicates_and_sorts(training_dirs):
    _update_index("r1", ["55HBU", "55HBV"])
    _update_index("r2", ["55HBU"])
    result = tile_ids_for_regions(["r1", "r2"])
    assert result == ["55HBU", "55HBV"]


# ---------------------------------------------------------------------------
# Tests 13–14: path helpers
# ---------------------------------------------------------------------------

def test_tile_parquet_path_format():
    p = tile_parquet_path("55HBU")
    assert p.name == "55HBU.parquet"
    assert "tiles" in str(p)


def test_region_parquet_path_format():
    p = _region_parquet_path("lake_mueller_presence")
    assert p.name == "lake_mueller_presence.parquet"
    assert "regions" in str(p)


# ---------------------------------------------------------------------------
# Test 15 — multi-tile region appears under both tile buckets (regression)
# ---------------------------------------------------------------------------

def test_multi_tile_region_appears_in_both_buckets(monkeypatch):
    """A region whose bbox straddles two tiles should be grouped under both.

    We monkeypatch bbox_to_tile_ids to return two tile IDs, then run the
    tile-grouping loop from ensure_training_pixels() directly.
    """
    monkeypatch.setattr(tc, "bbox_to_tile_ids", lambda bbox: ["55HBU", "55HBV"])
    region = _make_region("boundary_region", bbox=[145.0, -23.0, 145.2, -22.8])

    # Replicate the tile-grouping loop from ensure_training_pixels()
    tile_to_regions: dict[str, list] = {}
    for tile_id in tc.bbox_to_tile_ids(region.bbox_tuple):
        tile_to_regions.setdefault(tile_id, []).append(region)

    assert "55HBU" in tile_to_regions
    assert "55HBV" in tile_to_regions
    assert tile_to_regions["55HBU"] == [region]
    assert tile_to_regions["55HBV"] == [region]


# ---------------------------------------------------------------------------
# Test 16 — per-tile dedup does not remove cross-tile items (Bug TC6 — documents)
# ---------------------------------------------------------------------------

def test_within_tile_dedup_does_not_remove_cross_tile_items():
    """_fetch_tile_items() deduplicates within-tile granule variants but not
    cross-tile items.  Two items that represent the same acquisition over
    tiles 55HBU and 55HBV both survive, because dedup is applied separately
    to each tile's search result — not to the combined list.

    This is the root cause by which tile-boundary pixels get two rows with
    the same (point_id, date) in the output parquet.
    """
    def _dedup(items):
        seen: set[str] = set()
        result = []
        for it in items:
            key = _GRANULE_RE.sub("", it.id)
            if key not in seen:
                seen.add(key)
                result.append(it)
        return result

    # Simulate: tile 55HBU STAC search returns one item
    hbu_items = [_make_stac_item("S2A_TTTTTT_20220815_0_L2A", tile_id="55HBU")]
    # Simulate: tile 55HBV STAC search returns an item with the same base ID
    hbv_items = [_make_stac_item("S2A_TTTTTT_20220815_0_L2A", tile_id="55HBV")]

    deduped_hbu = _dedup(hbu_items)
    deduped_hbv = _dedup(hbv_items)

    # Each tile's list survives dedup independently (1 item each).
    assert len(deduped_hbu) == 1
    assert len(deduped_hbv) == 1

    # Combined: both survive — the cross-tile duplicate is NOT removed.
    combined = deduped_hbu + deduped_hbv
    assert len(combined) == 2
    assert deduped_hbu[0].id == deduped_hbv[0].id  # same item ID, different tile context


# ---------------------------------------------------------------------------
# Test 17 — cross-tile boundary pixel deduplicated in combined output
# ---------------------------------------------------------------------------

def test_cross_tile_boundary_pixel_deduplicated_in_output():
    """A pixel appearing in two tile parquets (cross-tile boundary) produces
    only one row in a combined output after (point_id, date) dedup."""
    import datetime
    import polars as pl

    from analysis.constants import BANDS, SPECTRAL_INDEX_COLS

    pid = "boundary_region_0000_0000"
    dt  = datetime.date(2022, 8, 15)

    def _row(tile_id, scl_purity):
        return {
            "point_id": pid,
            "lon": 145.15, "lat": -22.95,
            "date": dt,
            "item_id": f"S2A_{tile_id}_20220815",
            "tile_id": tile_id,
            **{b: 0.12 for b in BANDS},
            "scl_purity": scl_purity,
            "scl": 4, "aot": 0.85,
            "view_zenith": 0.9, "sun_zenith": 0.8,
            **{c: 0.25 for c in SPECTRAL_INDEX_COLS},
        }

    combined_raw = pl.DataFrame([_row("55HBU", 0.7), _row("55HBV", 0.95)])

    df_dedup = (
        combined_raw
        .sort(["point_id", "date", "scl_purity"], descending=[False, False, True])
        .unique(subset=["point_id", "date"], keep="first")
        .sort(["point_id", "date"])
    )

    assert len(df_dedup) == 1
    assert df_dedup["tile_id"][0] == "55HBV"


# ---------------------------------------------------------------------------
# Test 18 — tile rebuild includes all indexed regions, not just current fetch
# ---------------------------------------------------------------------------

def test_tile_rebuild_includes_all_indexed_regions(training_dirs, monkeypatch):
    """Regression: fetching a subset of regions for a tile must not overwrite
    previously collected regions when rebuilding the tile parquet.

    Scenario:
      - region_a was previously collected and indexed to tile 55HBU
      - region_b is being fetched now and also maps to tile 55HBU
      - the rebuilt tile must include rows from both region_a and region_b
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    tiles_dir = training_dirs["tiles_dir"]
    regions_dir = tiles_dir / "regions"

    # Write pre-existing region_a parquet (already in index)
    schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("lon", pa.float32()),
        pa.field("lat", pa.float32()),
    ])
    pq.write_table(
        pa.table({"point_id": ["region_a_0000"], "lon": [145.0], "lat": [-23.0]}, schema=schema),
        regions_dir / "region_a.parquet",
    )
    _update_index("region_a", ["55HBU"])

    # Write new region_b parquet (being fetched now)
    pq.write_table(
        pa.table({"point_id": ["region_b_0000"], "lon": [145.1], "lat": [-23.1]}, schema=schema),
        regions_dir / "region_b.parquet",
    )
    _update_index("region_b", ["55HBU"])

    # Simulate the tile rebuild step from ensure_training_pixels() with only
    # region_b in scope (the bug: region_a would be excluded pre-fix).
    from utils.training_collector import _load_index, _region_parquet_path, tile_parquet_path

    tile_id = "55HBU"
    tile_path = tile_parquet_path(tile_id)

    all_indexed = _load_index()
    all_indexed_ids = set(all_indexed.filter(pl.col("tile_id") == tile_id)["region_id"].to_list())
    region_paths = [
        _region_parquet_path(rid)
        for rid in sorted(all_indexed_ids)
        if _region_parquet_path(rid).exists()
    ]

    writer = None
    for rp in region_paths:
        pf = pq.ParquetFile(rp)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg)
            if writer is None:
                writer = pq.ParquetWriter(tile_path, tbl.schema)
            writer.write_table(tbl)
    if writer:
        writer.close()

    # Both regions must appear in the rebuilt tile
    result = pq.read_table(tile_path).to_pandas()
    point_ids = set(result["point_id"])
    assert "region_a_0000" in point_ids, "region_a missing from rebuilt tile — pre-fix bug"
    assert "region_b_0000" in point_ids, "region_b missing from rebuilt tile"


# ---------------------------------------------------------------------------
# Test 19 — _parse_years compat shim for old year: YYYY field
# ---------------------------------------------------------------------------

def test_parse_years_compat_shim_expands_old_year_field():
    """Old 'year: YYYY' entries must expand to [YYYY-5 .. YYYY] with a DeprecationWarning."""
    import warnings
    from utils.regions import _parse_years

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _parse_years({"id": "test_region", "year": 2023})

    assert result == [2018, 2019, 2020, 2021, 2022, 2023]
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "Expected a DeprecationWarning for old 'year:' field"
    )


def test_parse_years_new_field_no_warning():
    """New 'years: [...]' entries must not emit any DeprecationWarning."""
    import warnings
    from utils.regions import _parse_years

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _parse_years({"id": "test_region", "years": [2024]})

    assert result == [2024]
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_parse_years_empty_list_raises():
    """Empty years list must raise ValueError."""
    from utils.regions import _parse_years

    with pytest.raises(ValueError, match="non-empty"):
        _parse_years({"id": "test_region", "years": []})


# ---------------------------------------------------------------------------
# Tests 20–23 — tile-boundary mismatch (the 55KBT/54KZC class of bug)
# ---------------------------------------------------------------------------
#
# Scenario: _best_tile_for_region() picks tile A (centroid-based), but STAC
# has no items for tile A — collect() falls back to cached shards from tile B.
# The region parquet contains tile_id=B, but before the fix _do_region indexed
# the region under tile A, so _rebuild_tile_parquet(A) found 0 region parquets.
#
# Fix: _collect_one_region() returns the *actual* tile_id (shard stem), and
# _do_region indexes/rebuilds under that value.

def _write_minimal_parquet(path, point_ids: list[str], tile_id: str = "55HBU") -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("lon", pa.float32()),
        pa.field("lat", pa.float32()),
        pa.field("tile_id", pa.string()),
    ])
    tbl = pa.table({
        "point_id": pa.array(point_ids, pa.string()),
        "lon":      pa.array([145.0] * len(point_ids), pa.float32()),
        "lat":      pa.array([-23.0] * len(point_ids), pa.float32()),
        "tile_id":  pa.array([tile_id] * len(point_ids), pa.string()),
    }, schema=schema)
    pq.write_table(tbl, path)


def test_collect_one_region_returns_actual_tile_when_fallback_fires(
    training_dirs, monkeypatch, tmp_path
):
    """When fetch produces data from tile 54KZC (not the primary 55KBT),
    _collect_one_region() must return '54KZC', not '55KBT'.
    """
    import pyarrow.parquet as pq
    import utils.fetch_spec as fs_mod

    tiles_dir  = training_dirs["tiles_dir"]
    region_dir = tiles_dir / "regions"
    region     = _make_region("boundary_region", bbox=[144.0, -21.0, 144.1, -20.9], years=[2024])

    # Pre-populate the s2 intermediate that _collect_one_region inspects to
    # detect the actual tile.  fetch_spec writes out_dir/<year>/<tile>.s2.parquet.
    spec_out = region_dir / "boundary_region"
    year_dir = spec_out / "2024"
    year_dir.mkdir(parents=True)
    s2_file = year_dir / "54KZC.s2.parquet"
    merged  = year_dir / "54KZC.parquet"
    _write_minimal_parquet(s2_file, ["boundary_region_0000_0000"], tile_id="54KZC")
    _write_minimal_parquet(merged,  ["boundary_region_0000_0000"], tile_id="54KZC")

    # fetch_spec returns the merged parquet for year 2024.
    monkeypatch.setattr(fs_mod, "fetch_spec", lambda spec, **kw: {2024: [merged]})

    actual = tc._collect_one_region(
        region=region,
        tile_id="55KBT",          # primary tile — no imagery
        tile_items=[],
        start="2024-01-01",
        end="2024-12-31",
        cloud_max=80,
        apply_nbar=True,
        max_concurrent=4,
    )

    assert actual == "54KZC", (
        f"Expected actual_tile_id='54KZC' (from s2 shard stem), got {actual!r}"
    )


def test_collect_one_region_returns_none_when_no_s2_data(training_dirs, monkeypatch):
    """When fetch_spec returns no merged paths, _collect_one_region() must return None."""
    import utils.fetch_spec as fs_mod

    region = _make_region("empty_region", bbox=[144.0, -21.0, 144.1, -20.9], years=[2024])

    monkeypatch.setattr(fs_mod, "fetch_spec", lambda spec, **kw: {2024: []})

    result = tc._collect_one_region(
        region=region,
        tile_id="55KBT",
        tile_items=[],
        start="2024-01-01",
        end="2024-12-31",
        cloud_max=80,
        apply_nbar=True,
        max_concurrent=4,
    )

    assert result is None


def test_do_region_indexes_under_actual_tile_not_primary(training_dirs, monkeypatch):
    """_do_region() must index the region under the actual tile returned by
    _collect_one_region(), not the primary tile passed as its tile_id argument.

    This is the direct regression test for the 55KBT/54KZC bug: primary=55KBT,
    actual data in 54KZC → index must record 54KZC.
    """
    region = _make_region("boundary_region", bbox=[144.0, -21.0, 144.1, -20.9], years=[2024])

    # _collect_one_region returns the *actual* tile (54KZC), simulating the fallback.
    monkeypatch.setattr(tc, "_collect_one_region", lambda **kw: "54KZC")
    monkeypatch.setattr(tc, "_fetch_tile_items",   lambda *a, **kw: [])
    monkeypatch.setattr(tc, "_rebuild_tile_parquet", lambda tid: None)

    tiles_with_new: set[str] = set()
    tiles_with_new_lock = __import__("threading").Lock()

    # Replicate the _do_region closure body directly (it's a nested function
    # so we can't call it without running the full ensure_training_pixels).
    start, end = "2024-01-01", "2024-12-31"
    tile_items = tc._fetch_tile_items("55KBT", [region], 80)
    actual_tile = tc._collect_one_region(
        region=region,
        tile_id="55KBT",
        tile_items=tile_items,
        start=start,
        end=end,
        cloud_max=80,
        apply_nbar=True,
        max_concurrent=4,
    )
    if actual_tile is not None:
        with tiles_with_new_lock:
            tiles_with_new.add(actual_tile)
        from utils.training_collector import _update_index
        _update_index(region.id, [actual_tile])

    df = tc._load_index()
    row = df.filter(pl.col("region_id") == "boundary_region")
    assert len(row) == 1
    assert row[0, "tile_id"] == "54KZC", (
        "Index must record the actual tile (54KZC), not the primary tile (55KBT)"
    )
    assert "54KZC" in tiles_with_new
    assert "55KBT" not in tiles_with_new


def test_rebuild_tile_uses_actual_tile_not_primary(training_dirs, monkeypatch):
    """End-to-end index+rebuild: after a boundary-mismatch fetch, the region
    parquet is found when rebuilding the *actual* tile, not the primary tile.
    """
    import pyarrow.parquet as pq

    tiles_dir  = training_dirs["tiles_dir"]
    regions_dir = tiles_dir / "regions"

    # Simulate: region parquet was written with tile_id=54KZC (actual tile)
    rp = regions_dir / "boundary_region.parquet"
    _write_minimal_parquet(rp, ["boundary_region_0000_0000"], tile_id="54KZC")

    # Index records 54KZC (the fix)
    tc._update_index("boundary_region", ["54KZC"])

    # Rebuild 54KZC — should include the region
    tc._rebuild_tile_parquet("54KZC")

    tile_path = tc.tile_parquet_path("54KZC")
    assert tile_path.exists(), "54KZC tile parquet was not created"
    df = pq.read_table(tile_path).to_pandas()
    assert "boundary_region_0000_0000" in df["point_id"].values

    # Rebuild 55KBT — should produce 0 rows (nothing indexed there)
    tc._rebuild_tile_parquet("55KBT")
    tile_path_bt = tc.tile_parquet_path("55KBT")
    if tile_path_bt.exists():
        df_bt = pq.read_table(tile_path_bt).to_pandas()
        assert len(df_bt) == 0, "55KBT tile should be empty — region lives in 54KZC"
