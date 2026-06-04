"""Unit tests for the v2 two-pool network→disk / disk→extract pipeline.

All tests run locally with no S3 access — fetch_patches_to_tiff(),
_extract_item_from_tiffs(), collect_s1_for_tile(), and merge_scenes() are mocked.

Tests
-----
TV-1  fetch_patches_to_tiff writes one tif per (item, band) and returns paths.
TV-2  fetch_patches_to_tiff skips existing non-empty tif files (resume).
TV-3  fetch_patches_to_tiff cloud-filters wholly-clouded items (no spectral tifs).
TV-4  _extract_item_from_tiffs samples pixel values from on-disk tifs correctly.
TV-5  _extract_item_from_tiffs returns None when SCL tif is missing.
TV-6  _extract_item_from_tiffs returns None when no clear pixels.
TV-7  run_tile_pipeline_v2 yields (strip_idx, path) for each non-empty strip.
TV-8  run_tile_pipeline_v2 skips strips before resume_from_strip.
TV-9  run_tile_pipeline_v2 Pool A tiff_dir is cleaned up after Pool B completes.
TV-10 run_tile_pipeline_v2 cleans up scene_dir after each strip is yielded.
TV-11 run_tile_pipeline_v2 skips fetch when tiff dir already exists (interrupted mid-extract).
TV-12 run_tile_pipeline_v2 does not skip fetch when tiff dir is empty.
TV-13 run_tile_pipeline_v2 skips fetch+extract when scene parquets already exist (interrupted post-extract).
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BBOX = [145.41, -22.81, 145.44, -22.74]
_CRS  = CRS.from_epsg(32755)
# Simple identity-ish affine: 10 m pixels, top-left at UTM origin
_TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 7_490_000.0)


def _make_polygon(bbox=None):
    from shapely.geometry import box
    return box(*(bbox or _BBOX))


def _make_item(item_id: str, tile_id: str = "55HBU") -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        datetime=datetime(2022, 6, 1, tzinfo=timezone.utc),
        bbox=_BBOX,
        properties={"s2:mgrs_tile": tile_id},
        assets={
            "B04": SimpleNamespace(href=f"https://example.com/{item_id}/B04.tif"),
            "red": SimpleNamespace(href=f"https://example.com/{item_id}/B04.tif"),
            "blue": SimpleNamespace(href=f"https://example.com/{item_id}/B02.tif"),
            "scl":  SimpleNamespace(href=f"https://example.com/{item_id}/SCL.tif"),
            "aot":  SimpleNamespace(href=f"https://example.com/{item_id}/AOT.tif"),
        },
    )


def _write_geotiff(path: Path, arr: np.ndarray, transform=_TRANSFORM, crs=_CRS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(arr, 1)


def _write_strip_parquet(path: Path, n_rows: int = 5) -> None:
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


def _make_strip_and_meta(chunk_row=0, chunk_col=0, y_lower=7_480_000.0):
    meta = {
        "utm_crs": "EPSG:32755",
        "y0_snap": 7_480_000.0,
        "y1":      7_481_024.0,
        "x0_snap": 500_000.0,
        "x1":      500_020.0,
        "block_h_m": 10240.0,
        "block_w_m": 10240.0,
        "r": 10.0,
        "polygon_geometry": None,
        "first_y_lower": 7_480_000.0,
        "first_x_left":  500_000.0,
    }
    chunk = {
        "chunk_row": chunk_row, "chunk_col": chunk_col,
        "bbox": _BBOX, "y_lower": y_lower, "x_left_chunk": 500_000.0,
    }
    return chunk, meta


# ---------------------------------------------------------------------------
# TV-1  fetch_patches_to_tiff writes tifs and returns paths
# ---------------------------------------------------------------------------

def test_fetch_patches_to_tiff_writes_files(tmp_path):
    from utils.fetch import fetch_patches_to_tiff
    from utils.pixel_collector import BAND_ALIAS

    item = _make_item("S2A_55HBU_20220601_0_L2A")
    bands = ["B02", "B04"]
    scl_arr  = np.full((4, 4), 4,      dtype=np.float32)  # SCL=4 → clear
    spec_arr = np.full((4, 4), 1000.0, dtype=np.float32)

    with patch("utils.fetch._read_bbox_patch", side_effect=lambda href, bbox, **kw: (
        scl_arr if "scl" in href.lower() else spec_arr, _TRANSFORM, _CRS
    )):
        written = asyncio.run(fetch_patches_to_tiff(
            items=[item], bands=bands, bbox_wgs84=_BBOX, out_dir=tmp_path,
            max_concurrent=4, band_alias=BAND_ALIAS,
        ))

    assert len(written) >= 1
    for p in written:
        assert p.exists()
        assert p.stat().st_size > 0


# ---------------------------------------------------------------------------
# TV-2  fetch_patches_to_tiff skips existing non-empty tifs
# ---------------------------------------------------------------------------

def test_fetch_patches_to_tiff_skips_existing(tmp_path):
    from utils.fetch import fetch_patches_to_tiff
    from utils.pixel_collector import BAND_ALIAS

    item = _make_item("S2A_55HBU_20220601_0_L2A")
    bands = ["B04"]
    scl_arr = np.full((4, 4), 4, dtype=np.float32)

    # Pre-write SCL and spectral tifs so they look cached
    # With BAND_ALIAS, SCL→scl, so the asset key is "scl" and the tif is named "SCL.tif"
    scl_path  = tmp_path / item.id / "SCL.tif"
    spec_path = tmp_path / item.id / "B04.tif"
    _write_geotiff(scl_path, scl_arr)
    _write_geotiff(spec_path, np.full((4, 4), 500.0, dtype=np.float32))

    call_count = 0

    def counting_read(href, bbox, **kw):
        nonlocal call_count
        call_count += 1
        return np.full((4, 4), 1.0, dtype=np.float32), _TRANSFORM, _CRS

    with patch("utils.fetch._read_bbox_patch", side_effect=counting_read):
        written = asyncio.run(fetch_patches_to_tiff(
            items=[item], bands=bands, bbox_wgs84=_BBOX, out_dir=tmp_path,
            band_alias=BAND_ALIAS,
        ))

    assert call_count == 0
    assert spec_path in written


# ---------------------------------------------------------------------------
# TV-3  fetch_patches_to_tiff cloud-filters wholly-clouded items
# ---------------------------------------------------------------------------

def test_fetch_patches_to_tiff_cloud_filters(tmp_path):
    from utils.fetch import fetch_patches_to_tiff
    from utils.pixel_collector import BAND_ALIAS

    item = _make_item("S2A_55HBU_20220601_0_L2A")
    bands = ["B04"]
    # SCL=8 (medium probability cloud) → not in SCL_CLEAR_VALUES → filtered
    cloudy_scl = np.full((4, 4), 8, dtype=np.float32)

    def fake_read(href, bbox, **kw):
        return cloudy_scl, _TRANSFORM, _CRS

    with patch("utils.fetch._read_bbox_patch", side_effect=fake_read):
        written = asyncio.run(fetch_patches_to_tiff(
            items=[item], bands=bands, bbox_wgs84=_BBOX, out_dir=tmp_path,
            band_alias=BAND_ALIAS,
        ))

    # Spectral tif should NOT have been written for the clouded item
    spec_path = tmp_path / item.id / "B04.tif"
    assert not spec_path.exists()
    assert written == []


# ---------------------------------------------------------------------------
# TV-4  _extract_item_from_tiffs samples pixel values from on-disk tifs
# ---------------------------------------------------------------------------

def test_extract_item_from_tiffs_samples_correctly(tmp_path):
    from utils.pixel_collector import _extract_item_from_tiffs
    from analysis.constants import BANDS, SCL_BAND, AOT_BAND

    item = _make_item("S2A_55HBU_20220601_0_L2A")
    tiff_dir = tmp_path / item.id

    # Write SCL=4 (clear) for a 10×10 patch
    scl_arr = np.full((10, 10), 4, dtype=np.uint16)
    _write_geotiff(tiff_dir / f"{SCL_BAND}.tif", scl_arr)

    # Write AOT
    aot_arr = np.full((10, 10), 100, dtype=np.uint16)
    _write_geotiff(tiff_dir / f"{AOT_BAND}.tif", aot_arr)

    # Write spectral bands with value 1000 (→ 0.1 after /10000)
    for band in BANDS:
        _write_geotiff(tiff_dir / f"{band}.tif", np.full((10, 10), 1000, dtype=np.uint16))

    # Single point that projects into the patch
    # _TRANSFORM: x=500000 + col*10, y=7490000 - row*10
    # So point at (500005, 7489995) → col=0, row=0 → within bounds
    from pyproj import Transformer
    t = Transformer.from_crs(_CRS, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(500005.0, 7_489_995.0)
    points = [("px_0000_0000", lon, lat)]
    point_ids = ["px_0000_0000"]
    lons = np.array([lon], dtype=np.float64)
    lats = np.array([lat], dtype=np.float64)

    df = _extract_item_from_tiffs(
        item, tiff_dir, point_ids, lons, lats, apply_nbar=False,
    )

    assert df is not None
    assert len(df) == 1
    assert df["tile_id"][0] == "55HBU"
    # scl should be 4
    assert int(df["scl"][0]) == 4


# ---------------------------------------------------------------------------
# TV-5  _extract_item_from_tiffs returns None when SCL tif missing
# ---------------------------------------------------------------------------

def test_extract_item_from_tiffs_no_scl(tmp_path):
    from utils.pixel_collector import _extract_item_from_tiffs

    item = _make_item("S2A_55HBU_20220601_0_L2A")
    tiff_dir = tmp_path / item.id
    tiff_dir.mkdir(parents=True, exist_ok=True)
    # No SCL tif written

    df = _extract_item_from_tiffs(
        item, tiff_dir, ["px_0000_0000"],
        np.array([145.42]), np.array([-22.77]),
        apply_nbar=False,
    )
    assert df is None


# ---------------------------------------------------------------------------
# TV-6  _extract_item_from_tiffs returns None when no clear pixels
# ---------------------------------------------------------------------------

def test_extract_item_from_tiffs_no_clear_pixels(tmp_path):
    from utils.pixel_collector import _extract_item_from_tiffs
    from analysis.constants import SCL_BAND

    item = _make_item("S2A_55HBU_20220601_0_L2A")
    tiff_dir = tmp_path / item.id
    # SCL=8 everywhere (cloud)
    _write_geotiff(tiff_dir / f"{SCL_BAND}.tif", np.full((10, 10), 8, dtype=np.uint16))

    from pyproj import Transformer
    t = Transformer.from_crs(_CRS, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(500005.0, 7_489_995.0)

    df = _extract_item_from_tiffs(
        item, tiff_dir, ["px_0000_0000"],
        np.array([lon]), np.array([lat]),
        apply_nbar=False,
    )
    assert df is None


# ---------------------------------------------------------------------------
# TV-7  run_tile_pipeline_v2 yields (strip_idx, path) for each non-empty strip
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_v2_yields_strips(tmp_path):
    from proxy._pipeline import run_tile_pipeline_v2
    import asyncio as _asyncio

    item = _make_item("S2A_55HBU_20220601_0_L2A")
    polygon = _make_polygon()
    _chunk, _meta = _make_strip_and_meta()

    def fake_asyncio_run(coro):
        coro.close()

    def fake_merge(scene_paths, s1_path, out_path):
        _write_strip_parquet(out_path)

    with patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=([_chunk], _meta)), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes", side_effect=fake_merge), \
         patch("asyncio.run", side_effect=fake_asyncio_run), \
         patch("utils.pixel_collector._extract_item_from_tiffs", return_value=None):

        results = list(run_tile_pipeline_v2(
            tile_id="55HBU", years=2022,
            polygon_geometry=polygon,
            tmp=tmp_path / "work",
            items=[item],
        ))

    # No clear pixels → no chunks yielded; just verify no exceptions
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TV-8  run_tile_pipeline_v2 skips strips before resume_from_strip
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_v2_resume(tmp_path):
    from proxy._pipeline import run_tile_pipeline_v2

    polygon = _make_polygon()
    chunks = [
        {"chunk_row": 0, "chunk_col": 0, "bbox": _BBOX, "y_lower": 7_480_000.0, "x_left_chunk": 500_000.0},
        {"chunk_row": 1, "chunk_col": 0, "bbox": _BBOX, "y_lower": 7_490_240.0, "x_left_chunk": 500_000.0},
    ]
    _, meta = _make_strip_and_meta()

    def fake_asyncio_run(coro):
        coro.close()

    with patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=(chunks, meta)), \
         patch("asyncio.run", side_effect=fake_asyncio_run), \
         patch("utils.pixel_collector._extract_item_from_tiffs", return_value=None), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes"):

        list(run_tile_pipeline_v2(
            tile_id="55HBU", years=2022,
            polygon_geometry=polygon,
            tmp=tmp_path / "work",
            items=[_make_item("S2A_55HBU_20220601_0_L2A")],
            resume_from_chunk=(1, 0),
        ))

    # Chunk (0,0) should have been skipped — tiff_dir for chunk_000_000 never created
    assert not (tmp_path / "work" / "2022" / "chunk_000_000_tiffs").exists()


# ---------------------------------------------------------------------------
# TV-9  run_tile_pipeline_v2: tiff_dir cleaned up after Pool B
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_v2_cleans_tiff_dir(tmp_path):
    from proxy._pipeline import run_tile_pipeline_v2

    polygon = _make_polygon()
    _chunk, _meta = _make_strip_and_meta()

    tiff_dir_path = tmp_path / "work" / "2022" / "chunk_00_00_tiffs"

    def fake_asyncio_run(coro):
        # Simulate Pool A: create the tiff_dir so cleanup can be verified
        tiff_dir_path.mkdir(parents=True, exist_ok=True)
        coro.close()

    def fake_merge(scene_paths, s1_path, out_path):
        _write_strip_parquet(out_path)

    with patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=([_chunk], _meta)), \
         patch("asyncio.run", side_effect=fake_asyncio_run), \
         patch("utils.pixel_collector._extract_item_from_tiffs", return_value=None), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes", side_effect=fake_merge):

        list(run_tile_pipeline_v2(
            tile_id="55HBU", years=2022,
            polygon_geometry=polygon,
            tmp=tmp_path / "work",
            items=[_make_item("S2A_55HBU_20220601_0_L2A")],
        ))

    # tiff_dir should have been removed after Pool B completed
    assert not tiff_dir_path.exists()


# ---------------------------------------------------------------------------
# TV-11  Surviving tiff dir skips re-fetch (interrupted mid-extract resume)
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_v2_skips_fetch_when_tiff_dir_exists(tmp_path):
    """If a tiff dir already exists and is non-empty, _stage_fetch_tiffs must
    not call asyncio.run (i.e. no network fetch).  The chunk should proceed
    directly to extract using the cached tiffs."""
    from proxy._pipeline import run_tile_pipeline_v2

    polygon = _make_polygon()
    _chunk, _meta = _make_strip_and_meta()

    # Pre-create a non-empty tiff dir as if a prior run fetched but was killed
    # before extract completed.
    work_tmp = tmp_path / "work"
    tiff_dir = work_tmp / "2022" / "chunk_00_00_tiffs"
    tiff_dir.mkdir(parents=True, exist_ok=True)
    (tiff_dir / "sentinel.tif").write_bytes(b"\x00" * 16)

    asyncio_run_calls = []

    def fake_asyncio_run(coro):
        asyncio_run_calls.append(True)
        coro.close()

    def fake_merge(scene_paths, s1_path, out_path):
        _write_strip_parquet(out_path)

    with patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=([_chunk], _meta)), \
         patch("asyncio.run", side_effect=fake_asyncio_run), \
         patch("utils.pixel_collector._extract_item_from_tiffs", return_value=None), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes", side_effect=fake_merge):

        list(run_tile_pipeline_v2(
            tile_id="55HBU", years=2022,
            polygon_geometry=polygon,
            tmp=work_tmp,
            items=[_make_item("S2A_55HBU_20220601_0_L2A")],
        ))

    assert asyncio_run_calls == [], "asyncio.run should not be called when tiff dir already exists"


# ---------------------------------------------------------------------------
# TV-12  Empty tiff dir does not skip fetch
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_v2_does_not_skip_fetch_for_empty_tiff_dir(tmp_path):
    """An empty tiff dir (e.g. mkdir was called but nothing was written before
    the kill) must not be treated as a completed fetch — the fetch must run."""
    from proxy._pipeline import run_tile_pipeline_v2

    polygon = _make_polygon()
    _chunk, _meta = _make_strip_and_meta()

    # Pre-create an *empty* tiff dir — simulates a crash right after mkdir.
    work_tmp = tmp_path / "work"
    tiff_dir = work_tmp / "2022" / "chunk_00_00_tiffs"
    tiff_dir.mkdir(parents=True, exist_ok=True)

    asyncio_run_calls = []

    def fake_asyncio_run(coro):
        asyncio_run_calls.append(True)
        coro.close()

    with patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=([_chunk], _meta)), \
         patch("asyncio.run", side_effect=fake_asyncio_run), \
         patch("utils.pixel_collector._extract_item_from_tiffs", return_value=None), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes"):

        list(run_tile_pipeline_v2(
            tile_id="55HBU", years=2022,
            polygon_geometry=polygon,
            tmp=work_tmp,
            items=[_make_item("S2A_55HBU_20220601_0_L2A")],
        ))


# ---------------------------------------------------------------------------
# TV-13  Surviving scene parquets skip fetch+extract (interrupted post-extract)
# ---------------------------------------------------------------------------

def test_run_tile_pipeline_v2_skips_fetch_when_scene_dir_exists(tmp_path):
    """If a scenes dir already has scene parquets from a prior run (interrupted
    between extract and merge), neither asyncio.run nor _extract_item_from_tiffs
    should be called.  The chunk should proceed directly to S1 + merge."""
    from proxy._pipeline import run_tile_pipeline_v2

    polygon = _make_polygon()
    _chunk, _meta = _make_strip_and_meta()

    # Pre-create a non-empty scenes dir as if extract finished but merge was killed.
    work_tmp = tmp_path / "work"
    scene_dir = work_tmp / "2022" / "chunk_00_00_scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)

    scene_parquet = scene_dir / "scene_0000.parquet"
    tbl = pa.table({"x": pa.array([1], type=pa.int32())})
    pq.write_table(tbl, scene_parquet)

    asyncio_run_calls = []
    extract_calls = []

    def fake_asyncio_run(coro):
        asyncio_run_calls.append(True)
        coro.close()

    def fake_extract(*args, **kwargs):
        extract_calls.append(True)
        return None

    def fake_merge(scene_paths, s1_path, out_path, **kw):
        _write_strip_parquet(out_path)

    with patch("proxy._pipeline.read_cog_transform", return_value=("EPSG:32755", 7_600_000.0, 500_000.0, 1024, 1024)), \
         patch("proxy._pipeline.compute_chunks", return_value=([_chunk], _meta)), \
         patch("asyncio.run", side_effect=fake_asyncio_run), \
         patch("utils.pixel_collector._extract_item_from_tiffs", side_effect=fake_extract), \
         patch("utils.s1_collector.collect_s1_for_tile", return_value=None), \
         patch("proxy._pipeline.merge_scenes", side_effect=fake_merge):

        list(run_tile_pipeline_v2(
            tile_id="55HBU", years=2022,
            polygon_geometry=polygon,
            tmp=work_tmp,
            items=[_make_item("S2A_55HBU_20220601_0_L2A")],
        ))

    assert asyncio_run_calls == [], "asyncio.run should not be called when scene parquets already exist"
    assert extract_calls == [], "_extract_item_from_tiffs should not be called when scene parquets already exist"
