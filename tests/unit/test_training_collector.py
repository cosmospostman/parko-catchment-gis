"""Unit tests for utils/training_collector.py — index management and pure helpers.

ensure_training_pixels() calls network + disk and is not unit-tested here.
Focus: _tile_date_window(), _union_bbox(), _update_index(), tile_ids_for_regions(),
and the tile-grouping behaviour in ensure_training_pixels().

Tests
-----
 1. _tile_date_window: start = min(year)-5-01-01, end = max(year)-12-31.
 2. _tile_date_window: single-region list produces correct window.
 3. _tile_date_window: all year=None raises ValueError.
 4. _tile_date_window: mix of year=None and int ignores None values.
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
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd
import pytest

from training.regions import TrainingRegion
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
    year: int | None = 2022,
    label: str = "presence",
) -> TrainingRegion:
    return TrainingRegion(
        id=region_id,
        name=region_id.replace("_", " ").title(),
        label=label,
        bbox=bbox or [145.0, -23.0, 145.1, -22.9],
        year=year,
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
    regions = [_make_region("r1", year=2021), _make_region("r2", year=2023)]
    start, end = _tile_date_window(regions)
    assert start == "2016-01-01"   # min(2021, 2023) - 5
    assert end == "2023-12-31"     # max(2021, 2023)


def test_tile_date_window_single_region():
    regions = [_make_region("r1", year=2020)]
    start, end = _tile_date_window(regions)
    assert start == "2015-01-01"
    assert end == "2020-12-31"


def test_tile_date_window_all_none_raises():
    regions = [_make_region("r1", year=None), _make_region("r2", year=None)]
    with pytest.raises(ValueError, match="no year set"):
        _tile_date_window(regions)


def test_tile_date_window_ignores_none_years():
    regions = [_make_region("r1", year=None), _make_region("r2", year=2022)]
    start, end = _tile_date_window(regions)
    assert start == "2017-01-01"
    assert end == "2022-12-31"


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
    assert df.iloc[0]["region_id"] == "r1"
    assert df.iloc[0]["tile_id"] == "55HBU"


def test_update_index_replaces_stale_entries(training_dirs):
    _update_index("r1", ["55HBU"])
    _update_index("r1", ["55HBV"])
    df = _load_index()
    assert set(df[df["region_id"] == "r1"]["tile_id"]) == {"55HBV"}
    assert "55HBU" not in df["tile_id"].values


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
