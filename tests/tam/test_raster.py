"""Tests for utils/raster.py — PMTiles rasterisation of TAM scores."""
from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _make_small_scores_and_coords(
    n_xi: int = 2,
    n_yi: int = 2,
    lon0: float = 138.5,
    lat0: float = -34.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build tiny scores and coords DataFrames for a n_xi × n_yi UTM grid.

    point_id format: px_{xi:04d}_{yi:04d}
    lon/lat are approximate 10m-spaced positions near (lon0, lat0).
    """
    # Approximate degrees per 10m at this lat
    deg_per_10m_lon = 10.0 / (111_320.0 * math.cos(math.radians(lat0)))
    deg_per_10m_lat = 10.0 / 111_320.0

    rows_scores = []
    rows_coords = []
    val = 10
    for yi in range(n_yi):
        for xi in range(n_xi):
            pid = f"px_{xi:04d}_{yi:04d}"
            rows_scores.append({"point_id": pid, "prob_tam": val})
            rows_coords.append({
                "point_id": pid,
                "lon": lon0 + xi * deg_per_10m_lon,
                "lat": lat0 + yi * deg_per_10m_lat,
            })
            val = min(100, val + 10)

    scores_df = pl.DataFrame(rows_scores).with_columns(
        pl.col("prob_tam").cast(pl.UInt8)
    )
    coords_df = pl.DataFrame(rows_coords)
    return scores_df, coords_df


def test_warp_to_mercator_shape():
    """warp_to_mercator should return a non-empty uint8 array."""
    from utils.raster import _scores_to_grid_from_xi_yi, warp_to_mercator

    scores_df, coords_df = _make_small_scores_and_coords(n_xi=4, n_yi=4)
    grid, transform, crs = _scores_to_grid_from_xi_yi(scores_df, coords_df, "54HTG")

    merc_grid, merc_transform = warp_to_mercator(grid, transform, crs)

    assert merc_grid.dtype == np.uint8, f"Expected uint8, got {merc_grid.dtype}"
    assert merc_grid.ndim == 2
    assert merc_grid.shape[0] > 0 and merc_grid.shape[1] > 0, "Mercator grid must be non-empty"

    # Pixel size in mercator at lat -34° should be very close to 10m
    merc_pixel_x = abs(merc_transform.a)
    merc_pixel_y = abs(merc_transform.e)
    # Mercator at lat -34° stretches by 1/cos(-34°) ≈ 1.206
    # so pixel size should be around 10 * 1.2 ≈ 12m  (±5m tolerance for small grid)
    assert 5.0 < merc_pixel_x < 30.0, f"Unexpected mercator x pixel size: {merc_pixel_x}"
    assert 5.0 < merc_pixel_y < 30.0, f"Unexpected mercator y pixel size: {merc_pixel_y}"


def test_iter_tiles_yields_nonempty():
    """iter_tiles should yield valid PNGs for a non-trivial grid."""
    from utils.raster import iter_tiles
    from rasterio.transform import Affine

    # Build a 512×512 all-ones grid at ~zoom 13 scale near Broken Hill, NSW
    # Mercator coords for ~143.8°E, -31.9°S at 10m resolution
    # EPSG:3857: approx x=16_005_000, y=-3_760_000
    width = height = 512
    x0 = 16_005_000.0
    y0 = -3_700_000.0
    pixel_m = 10.0
    transform = Affine.translation(x0, y0) * Affine.scale(pixel_m, -pixel_m)

    grid = np.ones((height, width), dtype=np.uint8) * 50

    tiles = list(iter_tiles(grid, transform, zoom_min=12, zoom_max=13))

    assert len(tiles) > 0, "iter_tiles should yield at least one tile"

    for z, x, y, png_bytes in tiles:
        assert isinstance(png_bytes, bytes)
        assert png_bytes[:4] == b"\x89PNG", f"Expected PNG magic bytes, tile ({z},{x},{y})"
        assert len(png_bytes) > 0


def _utm_to_lonlat(easting, northing, epsg: int):
    """Inverse-project UTM easting/northing arrays to lon/lat (EPSG:4326)."""
    from pyproj import Transformer
    tr = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
    return tr.transform(easting, northing)


def _chunk_frames(tile_id, utm_epsg, origin_e, origin_n, xi_max, yi_max, prob_fill,
                  yi_base=0):
    """Build (scores_df, coords_df) for one chunk with ITS OWN xi/yi origin.

    Mirrors the production chunkstore reality: each chunk's point_id xi/yi are
    NOT a clean tile-global grid — yi numbering restarts per chunk-row band, so
    a chunk's pixels can sit kilometres from where its xi/yi would imply on a
    single shared grid. point_id STRINGS stay globally unique (verified against
    real 55KCB: r00_c00 has px_0000_10000+, r05_c01 has px_0013_0000) — we model
    that with yi_base so chunks don't reuse identical pid strings. Correct
    placement therefore must come from real lon/lat, not xi/yi.
    """
    xs, ys = np.meshgrid(np.arange(xi_max), np.arange(yi_max))
    xs = xs.ravel(); ys = ys.ravel()
    east = origin_e + xs * 10.0
    north = origin_n + ys * 10.0
    lon, lat = _utm_to_lonlat(east, north, utm_epsg)
    # pid yi carries yi_base so the two chunks have disjoint, globally-unique pids
    # while their REAL northings still differ per the origins.
    pids = [f"px_{int(x):04d}_{int(y) + yi_base:05d}" for x, y in zip(xs, ys)]
    scores = pl.DataFrame({"point_id": pids,
                           "prob_tam": np.full(len(pids), prob_fill, dtype=np.uint8)})
    coords = pl.DataFrame({"point_id": pids, "lon": lon, "lat": lat})
    return scores, coords


def test_chunked_tile_pixels_placed_at_correct_utm():
    """Regression: pixels from chunks with DIFFERENT xi/yi origins must land at
    their true geographic location, not be collapsed onto one chunk's origin.

    This is the 55KCB bug: r05's pixels (origin ~47km south of r00's) were
    placed by xi/yi on a single grid whose origin came from one chunk, so the
    real r05 ground rendered blank. The fix places by real lon/lat.
    """
    from utils.raster import _scores_to_grid_from_xi_yi

    epsg = 32655  # UTM zone 55S (matches 55KCB)
    # Two chunks with globally-unique pids (distinct yi_base) but origins ~53km
    # apart in northing — the production 55KCB shape.
    s_a, c_a = _chunk_frames("55KCB", epsg, 300_000.0, -1_910_000.0,
                             xi_max=50, yi_max=50, prob_fill=80, yi_base=0)
    s_b, c_b = _chunk_frames("55KCB", epsg, 300_000.0, -1_857_000.0,
                             xi_max=50, yi_max=50, prob_fill=40, yi_base=80000)
    scores = pl.concat([s_a, s_b])
    coords = pl.concat([c_a, c_b]).unique("point_id")

    grid, transform, crs = _scores_to_grid_from_xi_yi(scores, coords, "55KCB")

    # The grid must span the full ~47km northing extent (≈4700 rows at 10m),
    # NOT just one chunk's 50-row footprint.
    assert grid.shape[0] > 4000, (
        f"grid height {grid.shape[0]} too small — chunk B was collapsed onto "
        "chunk A's origin (the bug)")

    # Both chunks' values must be present and at DISTINCT rows.
    rows_a = np.where(grid == 80)[0]
    rows_b = np.where(grid == 40)[0]
    assert rows_a.size == 2500 and rows_b.size == 2500, (
        f"expected 2500 cells each, got A={rows_a.size} B={rows_b.size}")
    # Row 0 = northernmost. Chunk B has the HIGHER (less negative) northing, so
    # it must sit at SMALLER row indices than the southern chunk A — and the two
    # must not overlap (≈53km / 10m ≈ 5300 rows apart).
    assert rows_b.max() < rows_a.min(), (
        f"chunk regions misplaced: A rows {rows_a.min()}-{rows_a.max()}, "
        f"B rows {rows_b.min()}-{rows_b.max()}")
    assert rows_a.min() - rows_b.max() > 4000, "chunks not separated by true ~53km gap"


def test_drop_guard_raises_when_coords_missing(caplog):
    """If coords don't cover the scored pixels, the rasterizer must fail loud
    (not silently drop a region, which is what hid the 55KCB bug)."""
    from utils.raster import _scores_to_grid_from_xi_yi

    epsg = 32655
    s_a, c_a = _chunk_frames("55KCB", epsg, 300_000.0, -1_910_000.0,
                             xi_max=20, yi_max=20, prob_fill=80, yi_base=0)
    s_b, _c_b = _chunk_frames("55KCB", epsg, 300_000.0, -1_857_000.0,
                              xi_max=20, yi_max=20, prob_fill=40, yi_base=80000)
    scores = pl.concat([s_a, s_b])
    coords_missing_b = c_a  # only chunk A's coords — B (50%) has none

    with pytest.raises(ValueError, match="no coords"):
        _scores_to_grid_from_xi_yi(scores, coords_missing_b, "55KCB")


def test_drop_guard_warns_on_small_gap(caplog):
    """A small coords gap (<5%) should warn, not raise — partial coverage still
    renders, but the operator is told pixels were dropped."""
    import logging
    from utils.raster import _scores_to_grid_from_xi_yi

    epsg = 32655
    s, c = _chunk_frames("55KCB", epsg, 300_000.0, -1_910_000.0,
                         xi_max=50, yi_max=50, prob_fill=70)  # 2500 px
    # Drop coords for ~1% of pixels.
    drop = set(s["point_id"].to_list()[:25])
    c_partial = c.filter(~pl.col("point_id").is_in(drop))

    with caplog.at_level(logging.WARNING, logger="utils.raster"):
        grid, _, _ = _scores_to_grid_from_xi_yi(s, c_partial, "55KCB")
    assert any("would" in r.message and "dropped" in r.message for r in caplog.records), \
        "expected a drop-fraction warning"
    assert (grid > 0).sum() == 2475, "matched pixels should still be rendered"


def test_load_coords_dedups_and_filters(tmp_path):
    """_load_coords must dedup per-chunk observation rows and filter to wanted
    pids (the optimisation), returning one row per point_id."""
    from utils.raster import _load_coords

    # Chunk with one row PER OBSERVATION (point_id repeated 5×), 2 pixels.
    obs = pl.DataFrame({
        "point_id": ["px_0000_0000"] * 5 + ["px_0001_0000"] * 5,
        "lon": [138.5] * 5 + [138.6] * 5,
        "lat": [-34.0] * 5 + [-34.0] * 5,
        "date": list(range(5)) * 2,
    })
    p = tmp_path / "55KCB_r00_c00.parquet"
    obs.write_parquet(p)

    # No filter: both pixels, one row each.
    out = _load_coords([p])
    assert len(out) == 2
    assert set(out["point_id"]) == {"px_0000_0000", "px_0001_0000"}
    assert set(out.columns) == {"point_id", "lon", "lat"}

    # Filter to wanted pids: only the requested pixel survives.
    want = pl.Series(["px_0001_0000"])
    out2 = _load_coords([p], want_pids=want)
    assert out2["point_id"].to_list() == ["px_0001_0000"]


def test_rasterize_tile_to_pmtiles_multichunk(tmp_path):
    """End-to-end: rasterize_tile_to_pmtiles with a LIST of chunk coords writes
    a valid PMTiles file (the new multi-coords signature)."""
    from utils.raster import rasterize_tile_to_pmtiles

    epsg = 32655
    s_a, c_a = _chunk_frames("55KCB", epsg, 300_000.0, -1_910_000.0,
                             xi_max=30, yi_max=30, prob_fill=80, yi_base=0)
    s_b, c_b = _chunk_frames("55KCB", epsg, 300_000.0, -1_857_000.0,
                             xi_max=30, yi_max=30, prob_fill=40, yi_base=80000)
    scores_path = tmp_path / "55KCB.scores.parquet"
    pl.concat([s_a, s_b]).write_parquet(scores_path)
    ca = tmp_path / "55KCB_r00_c00.parquet"; c_a.write_parquet(ca)
    cb = tmp_path / "55KCB_r05_c00.parquet"; c_b.write_parquet(cb)

    out = tmp_path / "55KCB.pmtiles"
    rasterize_tile_to_pmtiles(scores_path, [ca, cb], "55KCB", out)

    assert out.exists() and out.stat().st_size > 0
    with open(out, "rb") as fh:
        assert fh.read(7) == b"PMTiles"


def test_coords_cache_written_then_reused(tmp_path):
    """First rasterise builds <scores>.coords.parquet from chunks; a second call
    reads the cache and does NOT need the chunk parquets (they can be gone)."""
    from utils.raster import rasterize_tile_to_pmtiles, _coords_cache_path

    epsg = 32655
    s_a, c_a = _chunk_frames("55KCB", epsg, 300_000.0, -1_910_000.0,
                             xi_max=20, yi_max=20, prob_fill=80, yi_base=0)
    s_b, c_b = _chunk_frames("55KCB", epsg, 300_000.0, -1_857_000.0,
                             xi_max=20, yi_max=20, prob_fill=40, yi_base=80000)
    scores_path = tmp_path / "55KCB.scores.parquet"
    pl.concat([s_a, s_b]).write_parquet(scores_path)
    ca = tmp_path / "55KCB_r00_c00.parquet"; c_a.write_parquet(ca)
    cb = tmp_path / "55KCB_r05_c00.parquet"; c_b.write_parquet(cb)

    cache = _coords_cache_path(scores_path)
    assert not cache.exists()

    rasterize_tile_to_pmtiles(scores_path, [ca, cb], "55KCB", tmp_path / "a.pmtiles")
    assert cache.exists(), "first call must write the coords cache"
    assert pl.read_parquet(cache).height == 800  # 2 chunks × 20×20 pixels

    # Delete the chunk parquets — second call must succeed purely from the cache.
    ca.unlink(); cb.unlink()
    rasterize_tile_to_pmtiles(scores_path, [], "55KCB", tmp_path / "b.pmtiles")
    assert (tmp_path / "b.pmtiles").exists()


def test_rasterize_tile_to_pmtiles_smoke(tmp_path):
    """Smoke test: rasterize_tile_to_pmtiles should write a valid PMTiles file."""
    from utils.raster import rasterize_tile_to_pmtiles

    # 3×3 grid of pixels
    scores_df, coords_df = _make_small_scores_and_coords(
        n_xi=3, n_yi=3,
        lon0=138.5, lat0=-34.0,
    )

    scores_path = tmp_path / "54LWH.scores.parquet"
    coords_path = tmp_path / "54LWH_2022.parquet"

    scores_df.write_parquet(scores_path)
    coords_df.write_parquet(coords_path)

    out_pmtiles = tmp_path / "54LWH.pmtiles"

    rasterize_tile_to_pmtiles(scores_path, coords_path, "54LWH", out_pmtiles)

    assert out_pmtiles.exists(), "PMTiles output file should exist"
    assert out_pmtiles.stat().st_size > 0, "PMTiles output file should be non-empty"

    # Verify PMTiles magic number: first 7 bytes should be b'PMTiles'
    with open(out_pmtiles, "rb") as fh:
        header_start = fh.read(7)
    assert header_start == b"PMTiles", f"Expected PMTiles magic, got {header_start!r}"
