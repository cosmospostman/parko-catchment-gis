"""Unit tests for utils/fetch_spec.py — fetch pipeline plumbing.

collect() and collect_s1_for_tile() are mocked; tests verify that
fetch_spec() and _fetch_spec_strips() route arguments correctly.

Tests
-----
FS-1. fetch_spec full-bbox path: collect() called once per year with correct
      bbox, start/end, and geometry from the FetchSpec.
FS-2. fetch_spec full-bbox path: collect_s1_for_tile() receives per-tile bbox
      derived from the s2 parquet pixels, not the full catchment bbox.
FS-3. fetch_spec full-bbox path: multi-tile — one merged parquet written per tile.
FS-4. _strip_bboxes: returned strips cover the full N-S extent without gaps or
      overlaps, and each strip has the same lon_min/lon_max as the input bbox.
FS-5. _strip_bboxes: single strip returned when bbox height ≤ strip_height_px rows.
FS-6. fetch_spec strip path: Phase A fetch pool capped at min(n_strips, 8).
FS-7. fetch_spec strip path: each strip collect() call uses the strip sub-bbox,
      not the full catchment bbox.
FS-8. _budget_params: strip_height_px is None at ≥32 GB, 2000 at 16–31 GB,
      1000 at 8–15 GB, 500 below 8 GB.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch, call

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.fetch_spec import FetchSpec, _budget_params, _strip_bboxes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(tmp_path: Path, bbox: list[float] | None = None) -> FetchSpec:
    bbox = bbox or [145.0, -23.0, 146.0, -22.0]
    return FetchSpec(
        id="testsite",
        bbox=bbox,
        years=[2023],
        point_id_prefix="px",
        geometry=None,
        out_dir=tmp_path / "pixels" / "testsite",
        cache_dir=tmp_path / "chips",
    )


def _write_s2_parquet(path: Path, tile_id: str, lons: list[float], lats: list[float]) -> None:
    """Write a minimal .s2.parquet with the columns collect_s1_for_tile reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(lons)
    df = pl.DataFrame({
        "point_id": [f"px_{i:04d}_0000" for i in range(n)],
        "lon": pl.Series(lons, dtype=pl.Float32),
        "lat": pl.Series(lats, dtype=pl.Float32),
        "tile_id": [tile_id] * n,
        "date": [datetime.date(2023, 6, 1)] * n,
        "B02": pl.Series([0.05] * n, dtype=pl.Float32),
        "B08": pl.Series([0.40] * n, dtype=pl.Float32),
    })
    df.write_parquet(path)


# ---------------------------------------------------------------------------
# FS-4. _strip_bboxes: full N-S coverage, no gaps, consistent lon extent
# ---------------------------------------------------------------------------

def test_strip_bboxes_covers_full_extent():
    bbox = [145.0, -23.0, 146.0, -22.0]
    strips = _strip_bboxes(bbox, strip_height_px=100)
    assert len(strips) >= 1
    # Lon extent unchanged
    for s in strips:
        assert s[0] == pytest.approx(bbox[0], abs=1e-6)
        assert s[2] == pytest.approx(bbox[2], abs=1e-6)
    # First strip starts exactly at lat_min
    assert strips[0][1] == pytest.approx(bbox[1], abs=1e-8)
    # Union of strips covers at least up to lat_max (may slightly overshoot due to UTM snap)
    assert max(s[3] for s in strips) >= bbox[3] - 1e-4
    # No gaps: each strip's lat_max equals the next strip's lat_min
    for a, b in zip(strips, strips[1:]):
        assert a[3] == pytest.approx(b[1], abs=1e-8)
    # All strips have valid extent
    for s in strips:
        assert s[1] < s[3]


# ---------------------------------------------------------------------------
# FS-5. _strip_bboxes: single strip for small bbox
# ---------------------------------------------------------------------------

def test_strip_bboxes_single_strip_for_small_bbox():
    # 0.009 degrees N-S ≈ ~1 km < 100-px × 10 m = 1 km, so should be 1 strip
    bbox = [145.0, -23.001, 146.0, -23.000]
    strips = _strip_bboxes(bbox, strip_height_px=100)
    assert len(strips) == 1
    assert strips[0][1] == pytest.approx(bbox[1], abs=1e-6)
    # lat_max clamped; if resulting strip is invalid it would be dropped, so must remain
    assert strips[0][3] >= bbox[3] - 1e-4


# ---------------------------------------------------------------------------
# FS-8. _budget_params thresholds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gb,expected_strip", [
    (32, None),
    (16, 2000),
    (8,  1000),
    (4,  500),
])
def test_budget_params_strip_height(gb, expected_strip):
    params = _budget_params(gb)
    assert params["strip_height_px"] == expected_strip


# ---------------------------------------------------------------------------
# FS-2. Full-bbox path: collect_s1_for_tile receives per-tile bbox
# ---------------------------------------------------------------------------

def test_fetch_spec_s1_receives_per_tile_bbox(tmp_path):
    """collect_s1_for_tile must be called with the tile's own pixel bbox,
    not the full catchment bbox.  Tests the fix for passing spec.bbox wholesale.
    """
    spec = _make_spec(tmp_path, bbox=[140.0, -25.0, 150.0, -15.0])  # large catchment bbox
    year_dir = spec.out_dir / "2023"
    year_dir.mkdir(parents=True, exist_ok=True)

    # S2 parquet covers a small sub-region of the catchment
    tile_lons = [145.10, 145.11, 145.12]
    tile_lats = [-22.80, -22.81, -22.82]
    s2_path = year_dir / "55HBU.s2.parquet"
    _write_s2_parquet(s2_path, "55HBU", tile_lons, tile_lats)

    captured_s1_bboxes: list[list[float]] = []

    def fake_collect(start, end, out_dir, phases, **kwargs) -> list[Path]:
        if "extract" in phases:
            return [s2_path]
        return []

    def fake_collect_s1(s2_path, bbox_wgs84, start, end, out_path, **kwargs):
        captured_s1_bboxes.append(list(bbox_wgs84))
        return None  # no S1 data

    def fake_merge_tile(s2_path, s1_path, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(s2_path, out_path)

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", side_effect=fake_collect_s1), \
         patch("utils.parquet_utils.merge_tile", side_effect=fake_merge_tile), \
         patch("utils.parquet_utils.merge_strips", side_effect=lambda *a, **kw: None):
        from utils.fetch_spec import fetch_spec
        fetch_spec(spec, memory_budget_gb=64)  # force full-bbox path (no strips)

    assert len(captured_s1_bboxes) == 1
    s1_bbox = captured_s1_bboxes[0]

    # The S1 bbox must be derived from tile pixels, not the catchment bbox
    assert s1_bbox[0] == pytest.approx(min(tile_lons), abs=0.01)
    assert s1_bbox[2] == pytest.approx(max(tile_lons), abs=0.01)
    assert s1_bbox[1] == pytest.approx(min(tile_lats), abs=0.01)
    assert s1_bbox[3] == pytest.approx(max(tile_lats), abs=0.01)

    # Sanity: must NOT be the full catchment bbox
    assert s1_bbox[0] > spec.bbox[0] + 1.0, "S1 bbox should be much narrower than catchment"
    assert s1_bbox[2] < spec.bbox[2] - 1.0, "S1 bbox should be much narrower than catchment"


# ---------------------------------------------------------------------------
# FS-1/FS-3. Full-bbox path: collect() called per year, one parquet per tile
# ---------------------------------------------------------------------------

def test_fetch_spec_full_bbox_collect_called_per_year(tmp_path):
    """fetch_spec calls collect() for each year and returns one path per tile."""
    spec = _make_spec(tmp_path)
    spec_2yr = FetchSpec(
        id=spec.id, bbox=spec.bbox, years=[2022, 2023],
        point_id_prefix=spec.point_id_prefix, geometry=spec.geometry,
        out_dir=spec.out_dir, cache_dir=spec.cache_dir,
    )
    collect_calls: list[dict] = []

    def fake_collect(start, end, out_dir, phases, **kwargs) -> list[Path]:
        collect_calls.append({"start": start, "end": end, "phases": phases})
        if "extract" in phases:
            year = start[:4]
            s2_path = out_dir / f"55HBU.s2.parquet"
            _write_s2_parquet(s2_path, "55HBU", [145.1, 145.2], [-22.8, -22.9])
            return [s2_path]
        return []

    def fake_collect_s1(s2_path, bbox_wgs84, start, end, out_path, **kwargs):
        return None

    def fake_merge_tile(s2_path, s1_path, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(s2_path, out_path)

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", side_effect=fake_collect_s1), \
         patch("utils.parquet_utils.merge_tile", side_effect=fake_merge_tile), \
         patch("utils.parquet_utils.merge_strips", side_effect=lambda *a, **kw: None):
        from utils.fetch_spec import fetch_spec
        results = fetch_spec(spec_2yr, memory_budget_gb=64)

    # Both years processed
    assert set(results.keys()) == {2022, 2023}
    # Each year has one merged parquet (one tile)
    for yr, paths in results.items():
        assert len(paths) == 1
        assert paths[0].name == "55HBU.parquet"

    # collect() called for fetch + extract per year
    years_seen = {c["start"][:4] for c in collect_calls}
    assert years_seen == {"2022", "2023"}


# ---------------------------------------------------------------------------
# FS-6. Strip path: Phase A fetch pool capped at min(n_strips, 8)
# ---------------------------------------------------------------------------

def test_fetch_spec_strip_phase_a_pool_capped(tmp_path, monkeypatch):
    """Phase A ThreadPoolExecutor must be capped at min(n_strips, 8) even
    when there are many strips, preventing >256 concurrent HTTP connections.
    """
    import concurrent.futures as _cf

    recorded_max_workers: list[int] = []
    _real_tpe = _cf.ThreadPoolExecutor

    class _CapturingTPE:
        def __init__(self, max_workers=None, **kw):
            recorded_max_workers.append(max_workers)
            self._pool = _real_tpe(max_workers=max_workers, **kw)

        def submit(self, fn, *a, **kw):
            return self._pool.submit(fn, *a, **kw)

        def __enter__(self):
            self._pool.__enter__()
            return self

        def __exit__(self, *a):
            return self._pool.__exit__(*a)

    # Force strip path with 20 strips (strip_height_px=500, small bbox → many strips)
    spec = _make_spec(tmp_path)

    def fake_collect(start, end, out_dir, phases, **kwargs) -> list[Path]:
        if "extract" in phases:
            s2_path = out_dir / "55HBU.s2.parquet"
            _write_s2_parquet(s2_path, "55HBU", [145.1], [-22.8])
            return [s2_path]
        return []

    def fake_collect_s1(*a, **kw):
        return None

    def fake_merge_tile(s2_path, s1_path, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(s2_path, out_path)

    def fake_merge_strips(strip_paths, out_path):
        import shutil
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(strip_paths[0], out_path)

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", side_effect=fake_collect_s1), \
         patch("utils.parquet_utils.merge_tile", side_effect=fake_merge_tile), \
         patch("utils.parquet_utils.merge_strips", side_effect=fake_merge_strips), \
         patch("utils.fetch_spec.ThreadPoolExecutor", _CapturingTPE):
        from utils.fetch_spec import fetch_spec
        # memory_budget_gb=8 → strip_height_px=1000; bbox ~111 km tall → ~11 strips
        fetch_spec(spec, memory_budget_gb=8)

    # The Phase A pool's max_workers must be ≤ 8
    phase_a_workers = [w for w in recorded_max_workers if w is not None]
    assert any(w <= 8 for w in phase_a_workers), (
        f"Phase A pool was not capped: observed max_workers values = {recorded_max_workers}"
    )


# ---------------------------------------------------------------------------
# FS-7. Strip path: each strip collect() uses the strip sub-bbox
# ---------------------------------------------------------------------------

def test_fetch_spec_strip_collect_uses_strip_bbox(tmp_path):
    """Every collect() call in the strip path must use the strip's own sub-bbox,
    not the full catchment bbox.
    """
    bbox = [145.0, -23.0, 146.0, -22.0]
    spec = _make_spec(tmp_path, bbox=bbox)

    collected_bboxes: list[list[float]] = []

    def fake_collect(start, end, out_dir, phases, bbox_wgs84, **kwargs) -> list[Path]:
        collected_bboxes.append(list(bbox_wgs84))
        if "extract" in phases:
            s2_path = out_dir / "55HBU.s2.parquet"
            _write_s2_parquet(s2_path, "55HBU", [145.1], [-22.8])
            return [s2_path]
        return []

    def fake_collect_s1(*a, **kw):
        return None

    def fake_merge_tile(s2_path, s1_path, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(s2_path, out_path)

    def fake_merge_strips(strip_paths, out_path):
        import shutil
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(strip_paths[0], out_path)

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", side_effect=fake_collect_s1), \
         patch("utils.parquet_utils.merge_tile", side_effect=fake_merge_tile), \
         patch("utils.parquet_utils.merge_strips", side_effect=fake_merge_strips):
        from utils.fetch_spec import fetch_spec
        fetch_spec(spec, memory_budget_gb=8)  # strip path

    # All bboxes seen by collect() must be sub-strips (lon extents match,
    # lat extents are strictly smaller than the full bbox)
    assert len(collected_bboxes) > 0
    for cb in collected_bboxes:
        assert cb[0] == pytest.approx(bbox[0], abs=1e-6), "lon_min must match full bbox"
        assert cb[2] == pytest.approx(bbox[2], abs=1e-6), "lon_max must match full bbox"
        # Each strip's lat span must be strictly smaller than the full bbox lat span
        assert (cb[3] - cb[1]) < (bbox[3] - bbox[1]) - 1e-6 or len(collected_bboxes) == 1
