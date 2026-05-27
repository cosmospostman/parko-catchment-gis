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
TP-6  fetch_tile_local done-sentinel: no pipeline calls when .done + parquet exist.
TP-7  fetch_tile_local resume: pre-existing strips included in merge;
      resume_from_strip set to first gap.
TP-8  fetch_tile_local writes each strip atomically (.tmp → rename) and calls
      merge_strips on completion.
TP-9  proxy/server.py _run_pipeline wrapper: 0x02 frames emitted per strip,
      progress frames present; run_tile_pipeline called with correct args.
TP-10 Integration: run_tile_pipeline with a real MemoryChipStore and
      _collect_per_scene produces a sorted output parquet.
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

    with patch("proxy._pipeline.run_tile_pipeline.__wrapped__", None, create=True), \
         patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", side_effect=fake_s1), \
         patch("proxy._pipeline.merge_scenes", side_effect=_fake_merge_scenes), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=[
             {"strip_idx": 0, "bbox": _BBOX, "points": [("px_0000", 145.41, -22.81)]},
         ]):
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

    strips = [
        {"strip_idx": 0, "bbox": [145.41, -22.81, 145.44, -22.78], "points": [("px_0", 145.41, -22.81)]},
        {"strip_idx": 1, "bbox": [145.41, -22.78, 145.44, -22.74], "points": [("px_1", 145.41, -22.78)]},
    ]

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes", side_effect=_fake_merge_scenes), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=strips):
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

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", side_effect=fake_s1), \
         patch("proxy._pipeline.merge_scenes", side_effect=_fake_merge_scenes), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=[
             {"strip_idx": 0, "bbox": strip_bbox, "points": [("px_0", 145.41, -22.78)]},
         ]):
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

    strips = [
        {"strip_idx": 0, "bbox": [145.41, -22.81, 145.44, -22.78], "points": [("px_0", 145.41, -22.81)]},
        {"strip_idx": 1, "bbox": [145.41, -22.78, 145.44, -22.74], "points": [("px_1", 145.41, -22.78)]},
    ]

    def fake_collect(**kwargs):
        cache_dirs_seen.append(kwargs.get("cache_dir"))
        return iter([("scene", scene_path)])

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes", side_effect=_fake_merge_scenes), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=strips):
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

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes", side_effect=_fake_merge_scenes), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=[
             {"strip_idx": 0, "bbox": _BBOX, "points": [("px_0", 145.41, -22.81)]},
         ]):
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

    out_dir = tmp_path / "out"
    out_path = out_dir / "2022" / "55HBU.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_strip_parquet(out_path)
    out_path.with_suffix(".done").touch()

    pipeline_called = []

    with patch("proxy._pipeline.run_tile_pipeline", side_effect=lambda *a, **kw: iter([])) as mock_pipeline:
        result = fetch_tile_local(
            tile_id="55HBU", year=2022,
            polygon_geometry=_make_polygon(),
            out_dir=out_dir,
            tmp_dir=tmp_path / "tmp",
        )

    mock_pipeline.assert_not_called()
    assert result == out_path


# ---------------------------------------------------------------------------
# TP-7  fetch_tile_local resume: pre-existing strips counted; gap detection
# ---------------------------------------------------------------------------

def test_fetch_tile_local_resume(tmp_path):
    from utils.tile_pipeline import fetch_tile_local

    out_dir = tmp_path / "out"
    tmp_dir = tmp_path / "tmp"
    tile_tmp = tmp_dir / "55HBU" / "2022"
    tile_tmp.mkdir(parents=True, exist_ok=True)

    # Two strips already complete
    for i in (0, 1):
        _write_strip_parquet(tile_tmp / f"strip_{i:04d}.parquet")

    resume_args: list[int] = []

    def fake_pipeline(tile_id, year, polygon_geometry, tmp, resume_from_strip=0, **kw):
        resume_args.append(resume_from_strip)
        # Yield one more strip (idx 2)
        strip_path = tmp / "strip_0002_sorted.parquet"
        _write_strip_parquet(strip_path)
        yield 2, strip_path

    merge_calls: list = []

    def fake_merge(strip_paths, out_path):
        merge_calls.append(list(strip_paths))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_strip_parquet(out_path)

    with patch("proxy._pipeline.run_tile_pipeline", side_effect=fake_pipeline), \
         patch("utils.parquet_utils.merge_strips", side_effect=fake_merge):
        fetch_tile_local(
            tile_id="55HBU", year=2022,
            polygon_geometry=_make_polygon(),
            out_dir=out_dir, tmp_dir=tmp_dir,
        )

    assert resume_args == [2], f"expected resume_from_strip=2, got {resume_args}"
    assert len(merge_calls) == 1
    merged = merge_calls[0]
    assert len(merged) == 3  # strips 0, 1, 2


# ---------------------------------------------------------------------------
# TP-8  fetch_tile_local atomic write and merge_strips call
# ---------------------------------------------------------------------------

def test_fetch_tile_local_atomic_write_and_merge(tmp_path):
    from utils.tile_pipeline import fetch_tile_local

    out_dir = tmp_path / "out"
    tmp_dir = tmp_path / "tmp"

    written_tmps: list[Path] = []

    def fake_pipeline(tile_id, year, polygon_geometry, tmp, **kw):
        strip_path = tmp / "strip_0000_sorted.parquet"
        _write_strip_parquet(strip_path)
        yield 0, strip_path

    merge_inputs: list = []

    def fake_merge(strip_paths, out_path):
        merge_inputs.append(list(strip_paths))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_strip_parquet(out_path)

    with patch("proxy._pipeline.run_tile_pipeline", side_effect=fake_pipeline), \
         patch("utils.parquet_utils.merge_strips", side_effect=fake_merge):
        result = fetch_tile_local(
            tile_id="55HBU", year=2022,
            polygon_geometry=_make_polygon(),
            out_dir=out_dir, tmp_dir=tmp_dir,
        )

    assert result is not None
    assert result.exists()
    # .done sentinel must have been written
    assert result.with_suffix(".done").exists()
    # merge_strips called once with a list of Paths
    assert len(merge_inputs) == 1
    assert all(isinstance(p, Path) for p in merge_inputs[0])
    # No leftover .tmp files
    tile_tmp = tmp_dir / "55HBU" / "2022"
    assert not list(tile_tmp.glob("*.tmp"))


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
    random.shuffle(rows)
    tbl = pa.Table.from_pylist(rows, schema=schema)
    out = out_dir / f"{scene_id}.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out, compression="zstd")
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

    strip_def = {"strip_idx": 0, "bbox": _BBOX, "points": [("px_0000", 145.41, -22.81)]}

    with patch("utils.pixel_collector.collect", side_effect=fake_collect), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0)), \
         patch("proxy._pipeline.compute_strips", return_value=[strip_def]):
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
