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


def test_scores_to_grid_basic():
    """scores_to_grid should return correct shape, dtype, and values."""
    from utils.raster import scores_to_grid

    scores_df, coords_df = _make_small_scores_and_coords(n_xi=2, n_yi=2)

    grid, transform, crs = scores_to_grid(scores_df, coords_df)

    assert grid.dtype == np.uint8, f"Expected uint8, got {grid.dtype}"
    assert grid.shape == (2, 2), f"Expected (2, 2), got {grid.shape}"

    # All non-zero (every pixel has a score)
    assert grid.any(), "Grid should have non-zero values"

    # Check transform has ~10m resolution in both axes
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)
    assert 9.0 < pixel_size_x < 11.0, f"Expected ~10m x pixel size, got {pixel_size_x}"
    assert 9.0 < pixel_size_y < 11.0, f"Expected ~10m y pixel size, got {pixel_size_y}"

    # CRS should be a UTM zone
    assert crs is not None
    epsg = crs.to_epsg()
    assert epsg is not None
    # Australia zone 54 or 55 — both are valid UTM south
    assert 32700 <= epsg <= 32760, f"Expected southern UTM zone, got EPSG:{epsg}"


def test_scores_to_grid_values():
    """Check that specific xi/yi positions map to the correct grid row/col."""
    from utils.raster import scores_to_grid

    # Single pixel at (xi=0, yi=0) with prob_tam=42
    scores_df = pl.DataFrame([{"point_id": "px_0000_0000", "prob_tam": 42}]).with_columns(
        pl.col("prob_tam").cast(pl.UInt8)
    )
    coords_df = pl.DataFrame([{"point_id": "px_0000_0000", "lon": 138.5, "lat": -34.0}])

    grid, transform, crs = scores_to_grid(scores_df, coords_df)

    assert grid.shape == (1, 1)
    assert grid[0, 0] == 42


def test_warp_to_mercator_shape():
    """warp_to_mercator should return a non-empty uint8 array."""
    from utils.raster import scores_to_grid, warp_to_mercator

    scores_df, coords_df = _make_small_scores_and_coords(n_xi=4, n_yi=4)
    grid, transform, crs = scores_to_grid(scores_df, coords_df)

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
