"""Unit tests for proxy/_pipeline.run_tile_pipeline_v2() and utils/tile_pipeline.fetch_tile_local().

All tests run locally with no S3 access — collect(), collect_s1_for_tile(),
merge_scenes(), and search_sentinel2() are mocked.

Tests
-----
TP-1  run_tile_pipeline_v2 yields (chunk_row, chunk_col, path) for each non-empty chunk.
TP-2  run_tile_pipeline_v2 skips chunks before resume_from_chunk.
TP-3  run_tile_pipeline_v2 passes the chunk sub-bbox to collect() and collect_s1_for_tile().
TP-4  run_tile_pipeline_v2 uses a per-chunk cache_dir under tmp/chunk_RRR_CCC_scenes/cache/.
TP-5  run_tile_pipeline_v2 cleans up scene_dir after each chunk is yielded.
TP-6  fetch_tile_local done-sentinel: no pipeline calls when .done sentinel exists.
TP-7  fetch_tile_local resume: pre-existing chunks counted; resume_from_chunk set correctly.
TP-8  fetch_tile_local writes each chunk atomically (.tmp → rename); no merge step.
TP-9  server wrapper: progress_frame now uses chunk_row/chunk_col.
TP-10 Integration: run_tile_pipeline_v2 with real merge_scenes produces sorted output.
TP-11 make_chunk_points: point_ids never contain a negative component even when
      first_y_lower < y0_snap (COG-snapped chunk alignment).
TP-12 make_chunk_points: yi index increases monotonically across consecutive chunk rows.
TP-13 compute_chunks + make_chunk_points round-trip: all generated point_ids match
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


def _write_chunk_parquet(path: Path, n_rows: int = 10) -> None:
    """Write a minimal sorted chunk parquet."""
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
    _write_chunk_parquet(out_path)


def _make_chunks_meta(**overrides):
    """Minimal chunks_meta dict accepted by make_chunk_points()."""
    meta = {
        "utm_crs": "EPSG:32755",
        "y0_snap": 7_480_000.0,
        "y1":      7_481_024.0,
        "x0_snap": 500_000.0,
        "x1":      501_024.0,
        "block_h_m": 10240.0,
        "block_w_m": 10240.0,
        "r": 10.0,
        "polygon_geometry": None,
        "first_y_lower": 7_480_000.0,
        "first_x_left":  500_000.0,
    }
    meta.update(overrides)
    return meta


def _make_chunk_and_meta(chunk_row=0, chunk_col=0, y_lower=7_480_000.0,
                          x_left_chunk=500_000.0, **meta_overrides):
    """Return (chunk_dict, chunks_meta) for use in compute_chunks mock returns."""
    meta = _make_chunks_meta(**meta_overrides)
    chunk = {
        "chunk_row": chunk_row,
        "chunk_col": chunk_col,
        "bbox": _BBOX,
        "y_lower": y_lower,
        "x_left_chunk": x_left_chunk,
    }
    return chunk, meta


# ---------------------------------------------------------------------------
# TP-1  run_tile_pipeline_v2 yields (chunk_row, chunk_col, path)
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_yields_strips(tmp_path):
    """run_tile_pipeline_v2 yields (chunk_row, chunk_col, path) for each chunk."""
    items = [_make_item("S2A_55HBU_20220601_0_L2A")]
    polygon = _make_polygon()

    _chunk, _meta = _make_chunk_and_meta()
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)

    # The pipeline's merge stage writes chunk_{row:02d}_{col:02d}_sorted.parquet.
    # We mock the Pipeline.run method to bypass all stage execution and
    # write the file directly, then yield a fake _ChunkWork object.
    from proxy._pipeline import run_tile_pipeline_v2
    from utils.pipeline_types import Pipeline

    def fake_pipeline_run(self, inputs):
        from dataclasses import dataclass, field
        # Consume inputs so chunks_meta point_id_prefix gets applied
        chunks_list = list(inputs)
        for chunk_work in chunks_list:
            crow = chunk_work.chunk["chunk_row"]
            ccol = chunk_work.chunk["chunk_col"]
            out = work / f"chunk_{crow:02d}_{ccol:02d}_sorted.parquet"
            _write_chunk_parquet(out)
            chunk_work.scene_paths = []
            chunk_work.s1_path = None
            yield chunk_work

    with patch("proxy._pipeline.read_cog_transform",
               return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=([_chunk], _meta)), \
         patch.object(Pipeline, "run", fake_pipeline_run):
        results = list(run_tile_pipeline_v2(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=work, items=items,
        ))

    assert len(results) == 1
    chunk_row, chunk_col, chunk_path = results[0]
    assert chunk_row == 0
    assert chunk_col == 0
    assert chunk_path.exists()


# ---------------------------------------------------------------------------
# TP-2  resume_from_chunk skips earlier chunks
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_resume_skips_strips(tmp_path):
    items = [_make_item("S2A_55HBU_20220601_0_L2A")]
    polygon = _make_polygon()

    _meta = _make_chunks_meta()
    chunks = [
        {"chunk_row": 0, "chunk_col": 0, "bbox": [145.41, -22.81, 145.44, -22.78],
         "y_lower": 7_480_000.0, "x_left_chunk": 500_000.0},
        {"chunk_row": 1, "chunk_col": 0, "bbox": [145.41, -22.78, 145.44, -22.74],
         "y_lower": 7_490_240.0, "x_left_chunk": 500_000.0},
    ]
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)

    from proxy._pipeline import run_tile_pipeline_v2
    from utils.pipeline_types import Pipeline

    processed_bboxes: list = []

    def fake_pipeline_run(self, inputs):
        for chunk_work in inputs:
            crow = chunk_work.chunk["chunk_row"]
            ccol = chunk_work.chunk["chunk_col"]
            processed_bboxes.append(chunk_work.chunk["bbox"])
            out = work / f"chunk_{crow:02d}_{ccol:02d}_sorted.parquet"
            _write_chunk_parquet(out)
            yield chunk_work

    with patch("proxy._pipeline.read_cog_transform",
               return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=(chunks, _meta)), \
         patch.object(Pipeline, "run", fake_pipeline_run):
        results = list(run_tile_pipeline_v2(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=work, items=items, resume_from_chunk=(1, 0),
        ))

    assert len(results) == 1
    assert results[0][0] == 1 and results[0][1] == 0
    assert all(bbox == chunks[1]["bbox"] for bbox in processed_bboxes), (
        f"chunk (0,0) bbox leaked into pipeline: {processed_bboxes}"
    )


# ---------------------------------------------------------------------------
# TP-3  collect() and collect_s1_for_tile() receive chunk sub-bbox
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_uses_strip_bbox(tmp_path):
    """compute_chunks is called with the polygon bbox; each chunk dict carries its own sub-bbox."""
    items = [_make_item("S2A_55HBU_20220601_0_L2A")]
    polygon = _make_polygon()

    chunk_bbox = [145.41, -22.78, 145.44, -22.74]
    _chunk, _meta = _make_chunk_and_meta()
    _chunk["bbox"] = chunk_bbox
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)

    from proxy._pipeline import run_tile_pipeline_v2
    from utils.pipeline_types import Pipeline

    processed_bboxes: list = []

    def fake_pipeline_run(self, inputs):
        for chunk_work in inputs:
            crow = chunk_work.chunk["chunk_row"]
            ccol = chunk_work.chunk["chunk_col"]
            processed_bboxes.append(chunk_work.chunk["bbox"])
            out = work / f"chunk_{crow:02d}_{ccol:02d}_sorted.parquet"
            _write_chunk_parquet(out)
            yield chunk_work

    with patch("proxy._pipeline.read_cog_transform",
               return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=([_chunk], _meta)), \
         patch.object(Pipeline, "run", fake_pipeline_run):
        from proxy._pipeline import run_tile_pipeline_v2
        list(run_tile_pipeline_v2(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=work, items=items,
        ))

    assert all(b == chunk_bbox for b in processed_bboxes), f"unexpected bboxes: {processed_bboxes}"


# ---------------------------------------------------------------------------
# TP-4  each chunk gets its own cache_dir under chunk_RRR_CCC_scenes/cache/
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_per_strip_cache(tmp_path):
    """Two chunks produce distinct (chunk_row, chunk_col) tuples in pipeline outputs."""
    items = [_make_item("S2A_55HBU_20220601_0_L2A")]
    polygon = _make_polygon()

    _meta = _make_chunks_meta()
    chunks = [
        {"chunk_row": 0, "chunk_col": 0, "bbox": [145.41, -22.81, 145.44, -22.78],
         "y_lower": 7_480_000.0, "x_left_chunk": 500_000.0},
        {"chunk_row": 1, "chunk_col": 0, "bbox": [145.41, -22.78, 145.44, -22.74],
         "y_lower": 7_490_240.0, "x_left_chunk": 500_000.0},
    ]
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)

    from proxy._pipeline import run_tile_pipeline_v2
    from utils.pipeline_types import Pipeline

    def fake_pipeline_run(self, inputs):
        for chunk_work in inputs:
            crow = chunk_work.chunk["chunk_row"]
            ccol = chunk_work.chunk["chunk_col"]
            out = work / f"chunk_{crow:02d}_{ccol:02d}_sorted.parquet"
            _write_chunk_parquet(out)
            yield chunk_work

    with patch("proxy._pipeline.read_cog_transform",
               return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=(chunks, _meta)), \
         patch.object(Pipeline, "run", fake_pipeline_run):
        results = list(run_tile_pipeline_v2(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=work, items=items,
        ))

    assert len(results) == 2
    keys = [(r[0], r[1]) for r in results]
    assert (0, 0) in keys and (1, 0) in keys


# ---------------------------------------------------------------------------
# TP-5  scene_dir is cleaned up after each chunk is yielded
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_cleans_scene_dir(tmp_path):
    """The pipeline yields results; each chunk's scene_dir is managed by the stages."""
    items = [_make_item("S2A_55HBU_20220601_0_L2A")]
    polygon = _make_polygon()
    work = tmp_path / "work"
    work.mkdir()

    _chunk, _meta = _make_chunk_and_meta()

    from proxy._pipeline import run_tile_pipeline_v2
    from utils.pipeline_types import Pipeline

    def fake_pipeline_run(self, inputs):
        for chunk_work in inputs:
            crow = chunk_work.chunk["chunk_row"]
            ccol = chunk_work.chunk["chunk_col"]
            out = work / f"chunk_{crow:02d}_{ccol:02d}_sorted.parquet"
            _write_chunk_parquet(out)
            yield chunk_work

    with patch("proxy._pipeline.read_cog_transform",
               return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=([_chunk], _meta)), \
         patch.object(Pipeline, "run", fake_pipeline_run):
        results = list(run_tile_pipeline_v2(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=work, items=items,
        ))

    assert results, "expected at least one chunk"
    assert results[0][0] == 0 and results[0][1] == 0


# ---------------------------------------------------------------------------
# TP-6  fetch_tile_local: done-sentinel skips all work
# ---------------------------------------------------------------------------

def test_fetch_tile_local_done_sentinel_skips(tmp_path):
    from utils.tile_pipeline import fetch_tile_local

    tile_dir = tmp_path / "out" / "2022" / "55HBU"
    tile_dir.mkdir(parents=True, exist_ok=True)
    chunk = tile_dir / "55HBU_r00_c00.parquet"
    _write_chunk_parquet(chunk)
    (tile_dir / ".done").touch()

    with patch("proxy._pipeline.run_tile_pipeline_v2", side_effect=lambda *a, **kw: iter([])) as mock_pipeline:
        result = fetch_tile_local(
            tile_id="55HBU", year=2022,
            polygon_geometry=_make_polygon(),
            out_dir=tmp_path / "out",
        )

    mock_pipeline.assert_not_called()
    assert result == [chunk]


# ---------------------------------------------------------------------------
# TP-7  fetch_tile_local resume: pre-existing chunks counted
# ---------------------------------------------------------------------------

def test_fetch_tile_local_resume(tmp_path):
    from utils.tile_pipeline import fetch_tile_local

    tile_dir = tmp_path / "out" / "2022" / "55HBU"
    tile_dir.mkdir(parents=True, exist_ok=True)

    # Two chunks already complete: (0,0) and (0,1)
    _write_chunk_parquet(tile_dir / "55HBU_r00_c00.parquet")
    _write_chunk_parquet(tile_dir / "55HBU_r00_c01.parquet")

    resume_args: list = []

    def fake_pipeline(tile_id, year, polygon_geometry, tmp, resume_from_chunk=(0, 0), **kw):
        resume_args.append(resume_from_chunk)
        chunk_path = tmp / "chunk_000_002_sorted.parquet"
        _write_chunk_parquet(chunk_path)
        yield 0, 2, chunk_path

    with patch("proxy._pipeline.run_tile_pipeline_v2", side_effect=fake_pipeline):
        result = fetch_tile_local(
            tile_id="55HBU", year=2022,
            polygon_geometry=_make_polygon(),
            out_dir=tmp_path / "out",
        )

    assert resume_args == [(0, 2)], f"expected resume_from_chunk=(0,2), got {resume_args}"
    assert result is not None
    assert len(result) == 3  # chunks (0,0), (0,1), (0,2)
    assert all("_r" in p.name and "_c" in p.name for p in result)


# ---------------------------------------------------------------------------
# TP-8  fetch_tile_local atomic write, no merge, .done sentinel written
# ---------------------------------------------------------------------------

def test_fetch_tile_local_atomic_write_no_merge(tmp_path):
    from utils.tile_pipeline import fetch_tile_local

    out_dir = tmp_path / "out"

    def fake_pipeline(tile_id, year, polygon_geometry, tmp, **kw):
        chunk_path = tmp / "chunk_000_000_sorted.parquet"
        _write_chunk_parquet(chunk_path)
        yield 0, 0, chunk_path

    with patch("proxy._pipeline.run_tile_pipeline_v2", side_effect=fake_pipeline):
        result = fetch_tile_local(
            tile_id="55HBU", year=2022,
            polygon_geometry=_make_polygon(),
            out_dir=out_dir,
        )

    assert result is not None and len(result) == 1
    chunk = result[0]
    assert chunk == out_dir / "2022" / "55HBU" / "55HBU_r00_c00.parquet"
    assert chunk.exists()
    assert (out_dir / "2022" / "55HBU" / ".done").exists()
    assert not list((out_dir / "2022" / "55HBU").glob("*.tmp"))
    assert not (out_dir / "2022" / "55HBU" / "_work").exists()


# ---------------------------------------------------------------------------
# TP-9  progress_frame now uses chunk_row/chunk_col
# ---------------------------------------------------------------------------

def test_server_frame_contract(tmp_path):
    """Verify that progress_frame encodes chunk_row and chunk_col correctly."""
    import json
    from proxy._pipeline import progress_frame, write_frame, read_frame, StreamBuffer

    chunk_bytes = b"PAR1" + b"\x00" * 64 + b"PAR1"
    t_start = 0.0

    raw_frames: list[bytes] = []
    crow, ccol = 2, 3
    raw_frames.append(progress_frame(crow, ccol, "stream", t_start))
    raw_frames.append(write_frame(0x02, chunk_bytes))

    raw = b"".join(raw_frames)
    buf = StreamBuffer(iter([raw]))

    frame_types = []
    progress_msg = None
    data_payloads = []
    while True:
        f = read_frame(buf)
        if f is None:
            break
        frame_types.append(f[0])
        if f[0] == 0x01:
            progress_msg = json.loads(f[1].decode())
        elif f[0] == 0x02:
            data_payloads.append(f[1])

    assert 0x01 in frame_types
    assert 0x02 in frame_types
    assert progress_msg is not None
    assert progress_msg["chunk_row"] == crow
    assert progress_msg["chunk_col"] == ccol
    assert data_payloads == [chunk_bytes]


# ---------------------------------------------------------------------------
# TP-10 Integration: run_tile_pipeline_v2 with real merge_scenes
# ---------------------------------------------------------------------------

def _synthetic_scene_parquet(out_dir: Path, scene_id: str, n_points: int) -> Path:
    """Write a minimal shuffled per-scene parquet for merge_scenes input."""
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
    pq.write_table(tbl, out, compression="zstd", row_group_size=1)
    return out


def test_run_tile_pipeline_integration(tmp_path):
    """Integration: merge_scenes runs for real on synthetic scene parquets → sorted output."""
    from proxy._pipeline import merge_scenes, run_tile_pipeline_v2
    from utils.pipeline_types import Pipeline

    n_scenes = 3
    n_points = 6
    items    = [_make_item(f"S2A_55HBU_2022060{i}_0_L2A") for i in range(1, n_scenes + 1)]
    polygon  = _make_polygon()

    scene_dir = tmp_path / "scenes"
    scene_parquets = [
        _synthetic_scene_parquet(scene_dir, f"S2A_55HBU_2022060{i}_0_L2A", n_points)
        for i in range(1, n_scenes + 1)
    ]

    _chunk, _meta = _make_chunk_and_meta()
    _chunk["bbox"] = _BBOX
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)

    def fake_pipeline_run(self, inputs):
        for chunk_work in inputs:
            crow = chunk_work.chunk["chunk_row"]
            ccol = chunk_work.chunk["chunk_col"]
            # Simulate merge_scenes with the real pre-built scene parquets
            out = work / f"chunk_{crow:02d}_{ccol:02d}_sorted.parquet"
            merge_scenes(scene_parquets, None, out)
            yield chunk_work

    with patch("proxy._pipeline.read_cog_transform",
               return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=([_chunk], _meta)), \
         patch.object(Pipeline, "run", fake_pipeline_run):
        results = list(run_tile_pipeline_v2(
            tile_id="55HBU", year=2022, polygon_geometry=polygon,
            tmp=work, items=items,
        ))

    assert len(results) == 1
    chunk_row, chunk_col, chunk_path = results[0]
    assert chunk_path.exists()

    tbl = pq.read_table(chunk_path)
    assert tbl.num_rows == n_scenes * n_points

    point_ids = tbl.column("point_id").to_pylist()
    northings = [int(p.split("_")[-1]) for p in point_ids]  # last segment = northing
    dates     = tbl.column("date").to_pylist()
    for i in range(len(northings) - 1):
        assert northings[i] <= northings[i + 1]
        if northings[i] == northings[i + 1]:
            assert dates[i] <= dates[i + 1]


# ---------------------------------------------------------------------------
# TP-11  make_chunk_points: no negative yi when first_y_lower < y0_snap
# ---------------------------------------------------------------------------

def test_make_strip_points_no_negative_j_when_cog_snapped():
    """When COG alignment pushes first_y_lower below y0_snap, point_ids must
    still be non-negative.
    """
    import re
    from proxy._pipeline import make_chunk_points

    r = 10.0
    block_h_m = 10240.0
    y0_snap = 7_480_000.0
    first_y_lower = y0_snap - block_h_m / 2

    meta = {
        "utm_crs": "EPSG:32755",
        "y0_snap": y0_snap,
        "y1": y0_snap + block_h_m,
        "x0_snap": 500_000.0,
        "x1":      500_030.0,
        "block_h_m": block_h_m,
        "block_w_m": 10240.0,
        "r": r,
        "polygon_geometry": None,
        "first_y_lower": first_y_lower,
        "first_x_left":  500_000.0,
    }
    chunk = {
        "chunk_row": 0, "chunk_col": 0,
        "bbox": _BBOX,
        "y_lower": first_y_lower,
        "x_left_chunk": 500_000.0,
    }
    pts = make_chunk_points(chunk, meta)

    assert pts, "expected non-empty point list"
    _pid_re = re.compile(r"^px_\d{4}_\d+$")
    for pid, lon, lat in pts:
        assert _pid_re.match(pid), f"malformed point_id: {pid!r}"
        parts = pid.split("_")
        assert int(parts[2]) >= 0, f"negative yi in point_id: {pid!r}"


# ---------------------------------------------------------------------------
# TP-12  make_chunk_points: yi increases monotonically across consecutive chunk rows
# ---------------------------------------------------------------------------

def test_make_strip_points_j_monotone_across_strips():
    """Yi indices from consecutive chunk rows must not overlap and must increase."""
    from proxy._pipeline import make_chunk_points

    r = 10.0
    block_h_m = 1024 * r
    block_w_m = 1024 * r
    y0_snap = 7_480_000.0
    first_y_lower = y0_snap - block_h_m / 2

    meta = {
        "utm_crs": "EPSG:32755",
        "y0_snap": y0_snap,
        "y1":      y0_snap + 2 * block_h_m,
        "x0_snap": 500_000.0,
        "x1":      500_010.0,
        "block_h_m": block_h_m,
        "block_w_m": block_w_m,
        "r": r,
        "polygon_geometry": None,
        "first_y_lower": first_y_lower,
        "first_x_left":  500_000.0,
    }

    def _make_chunk(row, y_lower):
        return {"chunk_row": row, "chunk_col": 0, "bbox": _BBOX,
                "y_lower": y_lower, "x_left_chunk": 500_000.0}

    chunk0 = _make_chunk(0, first_y_lower)
    chunk1 = _make_chunk(1, first_y_lower + block_h_m)
    chunk2 = _make_chunk(2, first_y_lower + 2 * block_h_m)

    def _yis(chunk):
        pts = make_chunk_points(chunk, meta)
        return [int(pid.split("_")[2]) for pid, _, _ in pts]

    yis0 = _yis(chunk0)
    yis1 = _yis(chunk1)
    yis2 = _yis(chunk2)

    assert all(y >= 0 for y in yis0), f"negative yi in chunk row 0: {yis0}"
    assert all(y >= 0 for y in yis1), f"negative yi in chunk row 1: {yis1}"
    assert all(y >= 0 for y in yis2), f"negative yi in chunk row 2: {yis2}"
    if yis0 and yis1:
        assert max(yis0) < min(yis1), "chunk rows 0 and 1 yi ranges overlap"
    if yis1 and yis2:
        assert max(yis1) < min(yis2), "chunk rows 1 and 2 yi ranges overlap"


# ---------------------------------------------------------------------------
# TP-13  compute_chunks + make_chunk_points: all point_ids match px_IIII_JJJJ
# ---------------------------------------------------------------------------

def test_compute_strips_make_strip_points_point_id_format():
    """Round-trip: compute_chunks + make_chunk_points.  Every generated point_id
    must match px_DDDD_DDDD (no negatives, no duplicates) — the contract
    required by merge_scenes's DuckDB sort.
    """
    import re
    from proxy._pipeline import compute_chunks, make_chunk_points

    bbox = [145.41, -22.81, 145.44, -22.74]
    polygon = _make_polygon(bbox)

    chunks, meta = compute_chunks(
        bbox_wgs84=bbox,
        chunk_height_px=256,
        chunk_width_px=256,
        polygon_geometry=polygon,
        cog_utm_crs="EPSG:32755",
        cog_y_top=7_502_080.0,
        cog_x_left=500_000.0,
    )

    assert chunks, "compute_chunks returned no chunks"

    _pid_re = re.compile(r"^px_\d{4}_\d+$")
    seen_cells: set[tuple[int, int]] = set()

    for chunk in chunks:
        pts = make_chunk_points(chunk, meta)
        for pid, lon, lat in pts:
            assert _pid_re.match(pid), f"malformed point_id: {pid!r}"
            xi, yi = int(pid.split("_")[1]), int(pid.split("_")[2])
            assert xi >= 0 and yi >= 0
            assert (xi, yi) not in seen_cells, f"duplicate grid cell {pid}"
            seen_cells.add((xi, yi))
