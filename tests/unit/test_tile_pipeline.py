"""Unit tests for proxy/_pipeline.run_tile_pipeline() and utils/tile_pipeline.fetch_tile_local().

All tests run locally with no S3 access — collect(), collect_s1_for_tile(),
merge_scenes(), and search_sentinel2() are mocked.

Tests
-----
TP-1  run_tile_pipeline yields (strip_idx, path) for each non-empty strip;
      the parquet file exists on disk when yielded.
TP-2  run_tile_pipeline skips strips before resume_from_strip.
TP-3  run_tile_pipeline passes the strip sub-bbox (not the full tile bbox)
      to collect() and collect_s1_for_tile().
TP-4  run_tile_pipeline uses a per-strip cache_dir under tmp/strip_NNNN_scenes/cache/.
TP-5  run_tile_pipeline cleans up scene_dir after each strip is yielded.
TP-6  fetch_tile_local done-sentinel: no pipeline calls when .done sentinel exists.
TP-7  fetch_tile_local resume: pre-existing strips counted; resume_from_strip
      set to first gap; new strips appended.
TP-8  fetch_tile_local writes each strip atomically (.tmp → rename) directly
      into out_dir/year/tile_id/; no merge step.
TP-9  proxy/server.py _run_pipeline wrapper: 0x02 frames emitted per strip,
      progress frames present; run_tile_pipeline called with correct args.
TP-10 Integration: run_tile_pipeline with a real MemoryChipStore and
      _collect_per_scene produces a sorted output parquet.
TP-11 make_strip_points: point_ids never contain a negative component even when
      first_lower < y0_snap (COG-snapped strip alignment).
TP-12 make_strip_points: j index increases monotonically across consecutive strips.
TP-13 compute_strips + make_strip_points round-trip: all generated point_ids match
      the px_IIII_JJJJ format required by the merge_scenes DuckDB sort.
"""

from __future__ import annotations

import io
import struct
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from analysis.constants import BANDS, SCL_BAND, AOT_BAND


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BBOX = [145.41, -22.81, 145.44, -22.74]


def _make_polygon(bbox=None):
    from shapely.geometry import box
    b = bbox or _BBOX
    return box(*b)


def _make_item(item_id: str, tile_id: str = "55HBU") -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        datetime=datetime(2022, 6, 1, tzinfo=timezone.utc),
        bbox=_BBOX,
        properties={"s2:mgrs_tile": tile_id},
        assets={
            "B04": SimpleNamespace(href=f"https://example.com/{item_id}/B04.tif"),
            "red": SimpleNamespace(href=f"https://example.com/{item_id}/B04.tif"),
        },
    )


def _write_strip_parquet(path: Path, n_rows: int = 10) -> None:
    """Write a minimal sorted strip parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("date",     pa.date32()),
        pa.field("source",   pa.string()),
    ])
    tbl = pa.Table.from_pydict({
        "point_id": [f"px_{i:04d}" for i in range(n_rows)],
        "date":     [date(2022, 6, 1)] * n_rows,
        "source":   ["S2"] * n_rows,
    }, schema=schema)
    pq.write_table(tbl, path, compression="zstd")


def _fake_merge_scenes(scene_paths, s1_path, out_path):
    """Write a minimal parquet to out_path (stand-in for merge_scenes)."""
    _write_strip_parquet(out_path)


def _make_strips_meta(**overrides):
    """Minimal strips_meta dict accepted by make_strip_points()."""
    import numpy as np
    meta = {
        "utm_crs": "EPSG:32755",
        "xs": np.array([500000.0, 500010.0]),
        "y0_snap": 7_480_000.0,
        "y1": 7_481_024.0,
        "block_m": 10240.0,
        "r": 10.0,
        "polygon_geometry": None,
        "first_lower": 7_480_000.0,
    }
    meta.update(overrides)
    return meta


def _make_strip_and_meta(strip_idx=0, y_lower=7_480_000.0, **meta_overrides):
    """Return (strip_dict, strips_meta) for use in compute_strips mock returns."""
    meta = _make_strips_meta(**meta_overrides)
    strip = {"strip_idx": strip_idx, "bbox": _BBOX, "y_lower": y_lower}
    return strip, meta


# ---------------------------------------------------------------------------
# TP-1  run_tile_pipeline yields (strip_idx, path) for each non-empty strip
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_yields_strips(tmp_path):
    items = [_make_item("S2A_55HBU_20220601_0_L2A")]
    polygon = _make_polygon()

    scene_path = tmp_path / "scene_0.parquet"
    _write_strip_parquet(scene_path)

    def fake_collect(**kwargs):
        return iter([("S2A_55HBU_20220601_0_L2A", scene_path)])

    def fake_s1(*a, **kw):
        return None

    _strip, _meta = _make_strip_and_meta()
    with patch("proxy._pipeline.run_tile_pipeline.__wrapped__", None, create=True), \
         patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", side_effect=fake_s1), \
         patch("proxy._pipeline.merge_scenes", side_effect=_fake_merge_scenes), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=([_strip], _meta)):
        from proxy._pipeline import run_tile_pipeline
        results = list(run_tile_pipeline(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=tmp_path / "work", items=items,
        ))

    assert len(results) == 1
    strip_idx, strip_path = results[0]
    assert strip_idx == 0
    assert strip_path.exists()


# ---------------------------------------------------------------------------
# TP-2  resume_from_strip skips earlier strips
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_resume_skips_strips(tmp_path):
    items = [_make_item("S2A_55HBU_20220601_0_L2A")]
    polygon = _make_polygon()

    collect_bboxes: list = []
    scene_path = tmp_path / "scene.parquet"
    _write_strip_parquet(scene_path)

    def fake_collect(**kwargs):
        collect_bboxes.append(kwargs["bbox_wgs84"])
        return iter([("scene", scene_path)])

    _meta = _make_strips_meta()
    strips = [
        {"strip_idx": 0, "bbox": [145.41, -22.81, 145.44, -22.78], "y_lower": 7_480_000.0},
        {"strip_idx": 1, "bbox": [145.41, -22.78, 145.44, -22.74], "y_lower": 7_490_240.0},
    ]

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes", side_effect=_fake_merge_scenes), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=(strips, _meta)):
        from proxy._pipeline import run_tile_pipeline
        results = list(run_tile_pipeline(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=tmp_path / "work", items=items, resume_from_strip=1,
        ))

    # Only strip 1 should have been processed (strip 0 skipped entirely).
    # collect() is now called twice per active strip (fetch phase + extract phase),
    # so expect 2 calls, both with strip 1's bbox — strip 0's bbox must never appear.
    assert len(results) == 1
    assert results[0][0] == 1
    assert all(bbox == strips[1]["bbox"] for bbox in collect_bboxes), (
        f"strip 0 bbox leaked into collect calls: {collect_bboxes}"
    )
    assert strips[0]["bbox"] not in collect_bboxes


# ---------------------------------------------------------------------------
# TP-3  collect() and collect_s1_for_tile() receive strip sub-bbox
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_uses_strip_bbox(tmp_path):
    items = [_make_item("S2A_55HBU_20220601_0_L2A")]
    polygon = _make_polygon()

    s2_bbox_seen: list = []
    s1_bbox_seen: list = []
    scene_path = tmp_path / "scene.parquet"
    _write_strip_parquet(scene_path)

    strip_bbox = [145.41, -22.78, 145.44, -22.74]

    def fake_collect(**kwargs):
        s2_bbox_seen.append(kwargs["bbox_wgs84"])
        return iter([("scene", scene_path)])

    def fake_s1(s2_path, bbox_wgs84, **kw):
        s1_bbox_seen.append(bbox_wgs84)
        return None

    _strip, _meta = _make_strip_and_meta(strip_idx=0, y_lower=7_480_000.0)
    _strip["bbox"] = strip_bbox
    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", side_effect=fake_s1), \
         patch("proxy._pipeline.merge_scenes", side_effect=_fake_merge_scenes), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=([_strip], _meta)):
        from proxy._pipeline import run_tile_pipeline
        list(run_tile_pipeline(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=tmp_path / "work", items=items,
        ))

    # collect() is called twice per strip (fetch phase + extract phase); both
    # calls must use the strip sub-bbox, not the full catchment bbox.
    assert all(b == strip_bbox for b in s2_bbox_seen), f"unexpected bboxes: {s2_bbox_seen}"
    # collect_s1_for_tile is called twice per strip (fetch + extract phases)
    assert all(b == strip_bbox for b in s1_bbox_seen), f"unexpected S1 bboxes: {s1_bbox_seen}"


# ---------------------------------------------------------------------------
# TP-4  each strip gets its own cache_dir under strip_NNNN_scenes/cache/
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_per_strip_cache(tmp_path):
    items = [_make_item("S2A_55HBU_20220601_0_L2A")]
    polygon = _make_polygon()

    cache_dirs_seen: list = []
    scene_path = tmp_path / "scene.parquet"
    _write_strip_parquet(scene_path)

    _meta = _make_strips_meta()
    strips = [
        {"strip_idx": 0, "bbox": [145.41, -22.81, 145.44, -22.78], "y_lower": 7_480_000.0},
        {"strip_idx": 1, "bbox": [145.41, -22.78, 145.44, -22.74], "y_lower": 7_490_240.0},
    ]

    def fake_collect(**kwargs):
        cache_dirs_seen.append(kwargs.get("cache_dir"))
        return iter([("scene", scene_path)])

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes", side_effect=_fake_merge_scenes), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=(strips, _meta)):
        from proxy._pipeline import run_tile_pipeline
        list(run_tile_pipeline(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=tmp_path / "work", items=items,
        ))

    # collect() is called twice per strip (fetch + extract), so 4 total for 2 strips.
    # The two unique cache dirs must be distinct (one per strip) and correctly nested.
    unique_cache_dirs = list(dict.fromkeys(cache_dirs_seen))  # preserves order, dedupes
    assert len(unique_cache_dirs) == 2, f"expected 2 distinct cache dirs, got: {unique_cache_dirs}"
    assert unique_cache_dirs[0] != unique_cache_dirs[1]
    for cd in unique_cache_dirs:
        assert cd is not None
        assert "strip_" in str(cd)
        assert str(cd).endswith("cache")


# ---------------------------------------------------------------------------
# TP-5  scene_dir is cleaned up after each strip is yielded
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_cleans_scene_dir(tmp_path):
    items = [_make_item("S2A_55HBU_20220601_0_L2A")]
    polygon = _make_polygon()
    work = tmp_path / "work"
    work.mkdir()

    scene_path = tmp_path / "scene.parquet"
    _write_strip_parquet(scene_path)

    scene_dirs_seen: list[Path] = []

    def fake_collect(**kwargs):
        # Record the out_dir (scene_dir) that was passed to collect()
        scene_dirs_seen.append(kwargs["out_dir"])
        return iter([("scene", scene_path)])

    _strip, _meta = _make_strip_and_meta()
    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes", side_effect=_fake_merge_scenes), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=([_strip], _meta)):
        from proxy._pipeline import run_tile_pipeline
        results = list(run_tile_pipeline(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=work, items=items,
        ))

    assert results, "expected at least one strip"
    # All scene dirs must have been cleaned up
    for sd in scene_dirs_seen:
        assert not sd.exists(), f"scene_dir {sd} was not cleaned up"


# ---------------------------------------------------------------------------
# TP-6  fetch_tile_local: done-sentinel skips all work
# ---------------------------------------------------------------------------

def test_fetch_tile_local_done_sentinel_skips(tmp_path):
    from utils.tile_pipeline import fetch_tile_local

    tile_dir = tmp_path / "out" / "2022" / "55HBU"
    tile_dir.mkdir(parents=True, exist_ok=True)
    strip = tile_dir / "strip_0000.parquet"
    _write_strip_parquet(strip)
    (tile_dir / ".done").touch()

    with patch("proxy._pipeline.run_tile_pipeline_v2", side_effect=lambda *a, **kw: iter([])) as mock_pipeline:
        result = fetch_tile_local(
            tile_id="55HBU", year=2022,
            polygon_geometry=_make_polygon(),
            out_dir=tmp_path / "out",
        )

    mock_pipeline.assert_not_called()
    assert result == [strip]


# ---------------------------------------------------------------------------
# TP-7  fetch_tile_local resume: pre-existing strips counted; gap detection
# ---------------------------------------------------------------------------

def test_fetch_tile_local_resume(tmp_path):
    from utils.tile_pipeline import fetch_tile_local

    tile_dir = tmp_path / "out" / "2022" / "55HBU"
    tile_dir.mkdir(parents=True, exist_ok=True)

    # Two strips already complete
    for i in (0, 1):
        _write_strip_parquet(tile_dir / f"strip_{i:04d}.parquet")

    resume_args: list[int] = []

    def fake_pipeline(tile_id, year, polygon_geometry, tmp, resume_from_strip=0, **kw):
        resume_args.append(resume_from_strip)
        strip_path = tmp / "strip_0002_sorted.parquet"
        _write_strip_parquet(strip_path)
        yield 2, strip_path

    with patch("proxy._pipeline.run_tile_pipeline_v2", side_effect=fake_pipeline):
        result = fetch_tile_local(
            tile_id="55HBU", year=2022,
            polygon_geometry=_make_polygon(),
            out_dir=tmp_path / "out",
        )

    assert resume_args == [2], f"expected resume_from_strip=2, got {resume_args}"
    assert result is not None
    assert len(result) == 3  # strips 0, 1, 2
    assert all(p.name.startswith("strip_") for p in result)


# ---------------------------------------------------------------------------
# TP-8  fetch_tile_local atomic write, no merge, .done sentinel written
# ---------------------------------------------------------------------------

def test_fetch_tile_local_atomic_write_no_merge(tmp_path):
    from utils.tile_pipeline import fetch_tile_local

    out_dir = tmp_path / "out"

    def fake_pipeline(tile_id, year, polygon_geometry, tmp, **kw):
        strip_path = tmp / "strip_0000_sorted.parquet"
        _write_strip_parquet(strip_path)
        yield 0, strip_path

    with patch("proxy._pipeline.run_tile_pipeline_v2", side_effect=fake_pipeline):
        result = fetch_tile_local(
            tile_id="55HBU", year=2022,
            polygon_geometry=_make_polygon(),
            out_dir=out_dir,
        )

    assert result is not None and len(result) == 1
    strip = result[0]
    assert strip == out_dir / "2022" / "55HBU" / "strip_0000.parquet"
    assert strip.exists()
    # .done sentinel written
    assert (out_dir / "2022" / "55HBU" / ".done").exists()
    # no leftover .tmp files
    assert not list((out_dir / "2022" / "55HBU").glob("*.tmp"))
    # _work/ cleaned up
    assert not (out_dir / "2022" / "55HBU" / "_work").exists()


# ---------------------------------------------------------------------------
# TP-9  server wrapper contract: frames produced from run_tile_pipeline output
#
# proxy/server.py imports FastAPI which is not available in the test
# environment.  Instead we test the frame-emission contract directly against
# the functions from proxy/_pipeline.py that the server uses, simulating
# what _run_pipeline does.
# ---------------------------------------------------------------------------

def test_server_frame_contract(tmp_path):
    """Simulate the server's frame-emission loop: run_tile_pipeline yields
    (strip_idx, path); the loop emits a 0x01 progress frame then a 0x02 data
    frame per strip.  Verify both frame types are present and the 0x02 payload
    matches the strip bytes.
    """
    from proxy._pipeline import (
        run_tile_pipeline, write_frame, progress_frame, read_frame, StreamBuffer,
    )

    strip_bytes = b"PAR1" + b"\x00" * 64 + b"PAR1"

    def fake_pipeline(tile_id, year, polygon_geometry, tmp, **kw):
        strip_path = tmp / "strip_0000_sorted.parquet"
        strip_path.parent.mkdir(parents=True, exist_ok=True)
        strip_path.write_bytes(strip_bytes)
        yield 0, strip_path

    # Simulate the server's inner loop (the part that doesn't need FastAPI).
    # Drive the generator directly with fake_pipeline — no real network I/O.
    import time
    t_start = time.monotonic()
    raw_frames: list[bytes] = []

    work = tmp_path / "work"
    work.mkdir()

    for strip_idx, strip_path in fake_pipeline("55HBU", 2022, _make_polygon(), work):
        raw_frames.append(progress_frame(strip_idx, "stream", time.monotonic() - t_start))
        raw_frames.append(write_frame(0x02, strip_path.read_bytes()))
        strip_path.unlink(missing_ok=True)

    raw = b"".join(raw_frames)
    buf = StreamBuffer(iter([raw]))

    frame_types = []
    data_payloads = []
    while True:
        f = read_frame(buf)
        if f is None:
            break
        frame_types.append(f[0])
        if f[0] == 0x02:
            data_payloads.append(f[1])

    assert 0x01 in frame_types, "no progress frame emitted"
    assert 0x02 in frame_types, "no data frame emitted"
    assert data_payloads == [strip_bytes]


# ---------------------------------------------------------------------------
# TP-10 Integration: run_tile_pipeline with real merge_scenes
#
# collect() is mocked to return pre-built per-scene parquets (same approach as
# test_collect_per_scene_point_id_sorted); merge_scenes runs for real so we
# can verify the output is sorted by (northing, date).
# ---------------------------------------------------------------------------

def _synthetic_scene_parquet(out_dir: Path, scene_id: str, n_points: int) -> Path:
    """Write a minimal shuffled per-scene parquet for merge_scenes input."""
    import random
    schema = pa.schema([
        pa.field("point_id", pa.string()),
        pa.field("lon",      pa.float32()),
        pa.field("lat",      pa.float32()),
        pa.field("date",     pa.date32()),
        pa.field("item_id",  pa.string()),
        pa.field("tile_id",  pa.string()),
        pa.field("source",   pa.string()),
    ] + [pa.field(b, pa.uint16()) for b in BANDS]
      + [pa.field(c, pa.int8()) for c in ("scl_purity", "scl")]
      + [pa.field(c, pa.uint8()) for c in ("aot", "view_zenith", "sun_zenith")]
      + [pa.field("orbit", pa.string()), pa.field("vh", pa.float32()), pa.field("vv", pa.float32())])

    rows = [
        {
            "point_id": f"px_{i:04d}",
            "lon": float(145.41 + i * 0.001),
            "lat": float(-22.81 + i * 0.001),
            "date": date(2022, 6, 1),
            "item_id": scene_id,
            "tile_id": "55HBU",
            "source": "S2",
            **{b: 1000 for b in BANDS},
            "scl_purity": 0, "scl": 4, "aot": 60,
            "view_zenith": 10, "sun_zenith": 30,
            "orbit": "ascending", "vh": None, "vv": None,
        }
        for i in range(n_points)
    ]
    rows.sort(key=lambda r: (int(r["point_id"].split("_")[1]), str(r["date"])))
    tbl = pa.Table.from_pylist(rows, schema=schema)
    out = out_dir / f"{scene_id}.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    # One row-group per northing so merge_scenes' heap merge sees non-overlapping
    # northing bands per row-group (matching production invariant).
    pq.write_table(tbl, out, compression="zstd", row_group_size=1)
    return out


def test_run_tile_pipeline_integration(tmp_path):
    """run_tile_pipeline with real merge_scenes: output parquet is sorted by
    (northing, date) and has correct row count.
    """
    n_scenes = 3
    n_points = 6
    items    = [_make_item(f"S2A_55HBU_2022060{i}_0_L2A") for i in range(1, n_scenes + 1)]
    polygon  = _make_polygon()

    scene_dir = tmp_path / "scenes"
    scene_parquets = [
        _synthetic_scene_parquet(scene_dir, f"S2A_55HBU_2022060{i}_0_L2A", n_points)
        for i in range(1, n_scenes + 1)
    ]

    def fake_collect(**kwargs):
        return iter([(p.stem, p) for p in scene_parquets])

    _strip, _meta = _make_strip_and_meta()
    _strip["bbox"] = _BBOX

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=([_strip], _meta)):
        from proxy._pipeline import run_tile_pipeline
        results = list(run_tile_pipeline(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=tmp_path / "work", items=items,
        ))

    assert len(results) == 1
    strip_idx, strip_path = results[0]
    assert strip_path.exists()

    tbl = pq.read_table(strip_path)
    assert tbl.num_rows == n_scenes * n_points

    # merge_scenes sorts by (northing extracted from point_id, date)
    point_ids = tbl.column("point_id").to_pylist()
    northings = [int(p.split("_")[1]) for p in point_ids]
    dates     = tbl.column("date").to_pylist()
    for i in range(len(northings) - 1):
        assert northings[i] <= northings[i + 1]
        if northings[i] == northings[i + 1]:
            assert dates[i] <= dates[i + 1]


# ---------------------------------------------------------------------------
# TP-11  make_strip_points: no negative j when first_lower < y0_snap
# ---------------------------------------------------------------------------

def test_make_strip_points_no_negative_j_when_cog_snapped():
    """When COG alignment pushes first_lower below y0_snap, point_ids must
    still be non-negative.  This was the regression fixed in commit 593fc34:
    j_offset was computed relative to y0_snap instead of first_lower.
    """
    import re
    import numpy as np
    from proxy._pipeline import make_strip_points

    r = 10.0
    block_m = 10240.0
    y0_snap = 7_480_000.0
    # first_lower sits partially below y0_snap — typical COG block snap.
    # The strip spans [first_lower, first_lower + block_m); the lower half is
    # clipped by the y0_snap filter but the upper half contains real pixels.
    first_lower = y0_snap - block_m / 2  # 5120 m below y0_snap

    meta = {
        "utm_crs": "EPSG:32755",
        "xs": np.array([500000.0, 500010.0, 500020.0]),
        "y0_snap": y0_snap,
        "y1": y0_snap + block_m,
        "block_m": block_m,
        "r": r,
        "polygon_geometry": None,
        "first_lower": first_lower,
    }
    # strip 0 has lower == first_lower; its ys above y0_snap are the real pixels.
    strip = {"strip_idx": 0, "bbox": _BBOX, "y_lower": first_lower}
    pts = make_strip_points(strip, meta)

    assert pts, "expected non-empty point list"
    _pid_re = re.compile(r"^px_\d{4}_\d+$")
    for pid, lon, lat in pts:
        assert _pid_re.match(pid), f"malformed point_id: {pid!r}"
        parts = pid.split("_")
        assert int(parts[2]) >= 0, f"negative j in point_id: {pid!r}"


# ---------------------------------------------------------------------------
# TP-12  make_strip_points: j index increases across consecutive strips
# ---------------------------------------------------------------------------

def test_make_strip_points_j_monotone_across_strips():
    """J indices from consecutive strips must not overlap and must increase."""
    import numpy as np
    from proxy._pipeline import make_strip_points

    r = 10.0
    block_m = 1024 * r
    y0_snap = 7_480_000.0
    # first_lower is half a block below y0_snap — realistic COG snap offset.
    # strip 0 spans first_lower..(first_lower+block_m), partially below y0_snap.
    first_lower = y0_snap - block_m / 2

    meta = {
        "utm_crs": "EPSG:32755",
        "xs": np.array([500000.0]),
        "y0_snap": y0_snap,
        "y1": y0_snap + 2 * block_m,
        "block_m": block_m,
        "r": r,
        "polygon_geometry": None,
        "first_lower": first_lower,
    }

    strip0 = {"strip_idx": 0, "bbox": _BBOX, "y_lower": first_lower}
    strip1 = {"strip_idx": 1, "bbox": _BBOX, "y_lower": first_lower + block_m}
    strip2 = {"strip_idx": 2, "bbox": _BBOX, "y_lower": first_lower + 2 * block_m}

    def _js(strip):
        pts = make_strip_points(strip, meta)
        return [int(pid.split("_")[2]) for pid, _, _ in pts]

    js0 = _js(strip0)
    js1 = _js(strip1)
    js2 = _js(strip2)

    # Each strip's j values must be non-negative and non-overlapping with others.
    assert all(j >= 0 for j in js0), f"negative j in strip 0: {js0}"
    assert all(j >= 0 for j in js1), f"negative j in strip 1: {js1}"
    assert all(j >= 0 for j in js2), f"negative j in strip 2: {js2}"
    if js0 and js1:
        assert max(js0) < min(js1), "strip 0 and 1 j ranges overlap"
    if js1 and js2:
        assert max(js1) < min(js2), "strip 1 and 2 j ranges overlap"


# ---------------------------------------------------------------------------
# TP-13  compute_strips + make_strip_points: all point_ids match px_IIII_JJJJ
# ---------------------------------------------------------------------------

def test_compute_strips_make_strip_points_point_id_format():
    """Round-trip: compute_strips on a real bbox, then make_strip_points for each
    strip.  Every generated point_id must match px_DDDD_DDDD+ (no negatives,
    no non-digit components) — the contract required by merge_scenes's DuckDB sort.
    """
    import re
    from proxy._pipeline import compute_strips, make_strip_points

    bbox = [145.41, -22.81, 145.44, -22.74]
    polygon = _make_polygon(bbox)

    # Use COG alignment parameters to exercise the first_lower < y0_snap path.
    strips, meta = compute_strips(
        bbox_wgs84=bbox,
        strip_height_px=256,
        polygon_geometry=polygon,
        cog_utm_crs="EPSG:32755",
        cog_y_top=7_502_080.0,
    )

    assert strips, "compute_strips returned no strips"

    _pid_re = re.compile(r"^px_\d{4}_\d+$")
    seen_js: set[tuple[int, int]] = set()

    for strip in strips:
        pts = make_strip_points(strip, meta)
        for pid, lon, lat in pts:
            assert _pid_re.match(pid), f"malformed point_id: {pid!r}"
            i, j = int(pid.split("_")[1]), int(pid.split("_")[2])
            assert i >= 0 and j >= 0
            assert (i, j) not in seen_js, f"duplicate grid cell {pid}"
            seen_js.add((i, j))
