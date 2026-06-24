"""utils/raster.py — Rasterise TAM scores to PMTiles."""
from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def warp_to_mercator(
    grid: np.ndarray,
    src_transform: object,
    src_crs: object,
) -> tuple[np.ndarray, object]:
    """Reproject a UTM uint8 grid to Web Mercator (EPSG:3857).

    Returns (dst_grid, dst_transform).
    """
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.crs import CRS

    height, width = grid.shape
    # Compute bounds from transform: transform maps (col, row) → (x, y)
    # top-left is (transform.c, transform.f); bottom-right adds width/height steps
    left = src_transform.c
    top = src_transform.f
    right = left + width * src_transform.a
    bottom = top + height * src_transform.e  # e is negative

    dst_crs = CRS.from_epsg(3857)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, width, height,
        left=left, bottom=bottom, right=right, top=top,
    )
    dst_grid = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=grid,
        destination=dst_grid,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return dst_grid, dst_transform


def _lon_to_tile_x(lon_deg: float, z: int) -> int:
    return int((lon_deg + 180.0) / 360.0 * (2 ** z))


def _lat_to_tile_y(lat_deg: float, z: int) -> int:
    lat_r = math.radians(lat_deg)
    return int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * (2 ** z))


# rdylgn colormap: 20 stops for score values 0–100 (evenly spaced at 5-unit intervals)
_RDYLGN: list[tuple[int, int, int]] = [
    (165,   0,  38), (189,  24,  29), (213,  48,  39), (230,  82,  52),
    (245, 115,  68), (252, 152,  86), (253, 185, 110), (254, 212, 139),
    (255, 235, 171), (255, 251, 204), (235, 248, 188), (209, 238, 161),
    (169, 220, 136), (120, 198, 112), ( 75, 176,  90), ( 35, 152,  72),
    (  0, 125,  62), (  0, 104,  55), (  0,  81,  46), (  0,  68,  27),
]
_RDYLGN_ARR = np.array(_RDYLGN, dtype=np.float32)  # shape (20, 3)


def _apply_colormap(score_arr: np.ndarray) -> np.ndarray:
    """Map uint8 score array (0–100, 0=no-data) → RGBA uint8 array."""
    h, w = score_arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask = score_arr > 0
    if not mask.any():
        return rgba
    vals = score_arr[mask].astype(np.float32)
    t = vals / 100.0 * (len(_RDYLGN) - 1)
    lo = np.floor(t).astype(np.int32).clip(0, len(_RDYLGN) - 2)
    hi = lo + 1
    f = (t - lo)[:, np.newaxis]
    rgb = (_RDYLGN_ARR[lo] * (1 - f) + _RDYLGN_ARR[hi] * f).round().astype(np.uint8)
    rgba[mask, :3] = rgb
    rgba[mask, 3] = 255
    return rgba


def iter_tiles(
    grid: np.ndarray,
    transform: object,
    zoom_min: int,
    zoom_max: int,
) -> Iterator[tuple[int, int, int, bytes]]:
    """Yield (z, x, y, png_bytes) for every 256x256 Web Mercator tile with data.

    Parameters
    ----------
    grid:
        uint8 array in Web Mercator projection (output of warp_to_mercator).
    transform:
        rasterio Affine transform for the grid (top-left corner, metres in EPSG:3857).
    zoom_min, zoom_max:
        Inclusive zoom range.
    """
    from PIL import Image
    from pyproj import Transformer

    # Web Mercator extent
    MERC_R = 6_378_137.0
    MERC_HALF = math.pi * MERC_R  # ~20037508.34

    height, width = grid.shape

    # Grid bounding box in mercator metres
    # transform: (a=dx, b=0, c=xmin, d=0, e=-dy, f=ymax)
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + width * transform.a
    ymin = ymax + height * transform.e  # e is negative

    # Convert mercator bbox corners to lon/lat for tile coord math
    merc_to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    def _merc_to_lon_lat(x: float, y: float) -> tuple[float, float]:
        return merc_to_ll.transform(x, y)

    # Inverse: given a tile pixel offset in the grid, return its mercator coords
    # transform.a is pixel width (positive), transform.e is pixel height (negative)
    pixel_w = transform.a   # metres per pixel in x
    pixel_h = -transform.e  # metres per pixel in y (positive)

    for z in range(zoom_min, zoom_max + 1):
        n = 2 ** z
        tile_size_merc_x = 2.0 * MERC_HALF / n  # mercator width of one 256-px tile
        tile_size_merc_y = 2.0 * MERC_HALF / n

        # Tile range covering the grid bbox
        nw_lon, nw_lat = _merc_to_lon_lat(xmin, ymax)
        se_lon, se_lat = _merc_to_lon_lat(xmax, ymin)

        # Clamp lat to valid slippy-map range (avoids atan/log domain errors)
        nw_lat = min(nw_lat, 85.05112878)
        se_lat = max(se_lat, -85.05112878)

        tx_min = _lon_to_tile_x(nw_lon, z)
        tx_max = _lon_to_tile_x(se_lon, z)
        ty_min = _lat_to_tile_y(nw_lat, z)
        ty_max = _lat_to_tile_y(se_lat, z)

        # Clamp to valid tile range
        tx_min = max(0, tx_min)
        tx_max = min(n - 1, tx_max)
        ty_min = max(0, ty_min)
        ty_max = min(n - 1, ty_max)

        for tx in range(tx_min, tx_max + 1):
            for ty in range(ty_min, ty_max + 1):
                # Mercator bbox for this tile
                t_xmin = tx * tile_size_merc_x - MERC_HALF
                t_xmax = t_xmin + tile_size_merc_x
                t_ymax = MERC_HALF - ty * tile_size_merc_y
                t_ymin = t_ymax - tile_size_merc_y

                # Build 256x256 tile array by sampling from the grid
                tile_arr = np.zeros((256, 256), dtype=np.uint8)

                # For each tile pixel, compute grid pixel index
                # tile pixel (px, py) → mercator (mx, my)
                mx_arr = t_xmin + (np.arange(256) + 0.5) * (tile_size_merc_x / 256.0)
                my_arr = t_ymax - (np.arange(256) + 0.5) * (tile_size_merc_y / 256.0)

                # Convert mercator to grid pixel coords
                # gx = (mx - xmin) / pixel_w
                # gy = (ymax - my) / pixel_h
                gx_arr = (mx_arr - xmin) / pixel_w   # shape (256,) — x coords for each tile col
                gy_arr = (ymax - my_arr) / pixel_h   # shape (256,) — y coords for each tile row

                # Filter to grid bounds
                gx_int = np.round(gx_arr).astype(np.int32)
                gy_int = np.round(gy_arr).astype(np.int32)

                valid_x = (gx_int >= 0) & (gx_int < width)
                valid_y = (gy_int >= 0) & (gy_int < height)

                for py in range(256):
                    if not valid_y[py]:
                        continue
                    gy_i = gy_int[py]
                    # Vectorised fill across x axis
                    vx_mask = valid_x
                    gx_vals = gx_int.copy()
                    # Clamp invalid to 0 (masked out below)
                    gx_vals = np.where(vx_mask, gx_vals, 0)
                    row_vals = grid[gy_i, gx_vals]
                    tile_arr[py, :] = np.where(vx_mask, row_vals, 0)

                if not tile_arr.any():
                    continue

                buf = io.BytesIO()
                img = Image.fromarray(_apply_colormap(tile_arr), mode="RGBA")
                img.save(buf, format="PNG")
                yield z, tx, ty, buf.getvalue()


def _mgrs_utm_epsg(tile_id: str) -> int:
    """Return UTM EPSG code for an MGRS tile_id (e.g. '54LWJ' → 32754)."""
    zone = int(tile_id[:2])
    band = tile_id[2].upper()
    south = band < 'N'
    return (32700 + zone) if south else (32600 + zone)


def _grid_origin_from_tile(tile_id: str, loc_geom=None) -> tuple[float, float, int]:
    """Return (origin_e, origin_n, utm_epsg) for a tile_id's pixel grid.

    Reproduces the bbox derivation in the scoring pipeline:
    intersect the S2 tile polygon with loc_geom (or use tile bounds directly),
    then run the same make_pixel_grid snap logic.
    """
    from pyproj import Transformer
    from utils.location import _tile_polygon

    tile_poly = _tile_polygon(tile_id)
    if tile_poly is not None and loc_geom is not None:
        fetch_geom = tile_poly.intersection(loc_geom)
    elif tile_poly is not None:
        fetch_geom = tile_poly
    else:
        raise ValueError(f"Cannot determine tile polygon for {tile_id}")

    lon_min, lat_min, lon_max, lat_max = fetch_geom.bounds

    lon_mean = (lon_min + lon_max) / 2
    lat_mean = (lat_min + lat_max) / 2
    zone = int((lon_mean + 180) / 6) + 1
    south = lat_mean < 0
    utm_epsg = (32700 + zone) if south else (32600 + zone)

    to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    x0, y0 = to_utm.transform(lon_min, lat_min)
    x1, _ = to_utm.transform(lon_max, lat_max)

    r = 10.0
    origin_e = float(np.floor(x0 / r) * r)
    origin_n = float(np.floor(y0 / r) * r)
    return origin_e, origin_n, utm_epsg


def _scores_to_grid_from_xi_yi(
    scores_df: pl.DataFrame,
    lon_lat_df: pl.DataFrame | None,
    tile_id: str,
    loc_geom=None,
) -> tuple[np.ndarray, object, object]:
    """Build UTM grid from scores parquet.

    If lon_lat_df is provided, derives origin from actual coordinates.
    Otherwise reconstructs origin from the tile geometry + loc_geom intersection,
    mirroring the scoring pipeline's bbox derivation exactly.
    """
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    from pyproj import Transformer

    utm_epsg = _mgrs_utm_epsg(tile_id)
    crs = CRS.from_epsg(utm_epsg)

    if lon_lat_df is not None:
        # Place each pixel by its REAL lon/lat, not by reconstructing position
        # from point_id xi/yi. A chunked tile's xi/yi are origin-relative *per
        # chunk* (different chunks reuse low xi/yi for different ground), so
        # xi/yi-on-one-grid placement misplaces every chunk but the origin one
        # by kilometres. Projecting actual coords to UTM and binning to a 10 m
        # grid is correct regardless of how point_ids were assigned.
        n_scored = len(scores_df)
        joined = scores_df.join(
            lon_lat_df.select(["point_id", "lon", "lat"]),
            on="point_id",
            how="inner",
        )
        n_matched = len(joined)
        if n_matched == 0:
            raise ValueError(
                f"Tile {tile_id}: no scored point_ids matched the coords parquet(s) — "
                "coords source does not cover this tile's pixels."
            )
        # Guard: silently dropping scored pixels here is exactly the bug that hid
        # whole regions (one chunk's coords used for the whole tile). Fail loud.
        drop_frac = 1.0 - n_matched / n_scored
        if drop_frac > 0.001:
            msg = (f"Tile {tile_id}: {n_scored - n_matched:,}/{n_scored:,} "
                   f"scored pixels ({100*drop_frac:.1f}%) have no coords and would "
                   "be dropped from the raster — coords source is incomplete "
                   "(pass ALL chunk parquets for this tile).")
            if drop_frac > 0.05:
                raise ValueError(msg)
            logger.warning(msg)

        lon_arr = joined["lon"].to_numpy()
        lat_arr = joined["lat"].to_numpy()
        prob_arr = joined["prob_tam"].to_numpy().astype(np.uint8)
        transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
        easting_arr, northing_arr = transformer.transform(lon_arr, lat_arr)

        # 10 m grid spanning the matched pixels' UTM extent.
        origin_e = float(np.floor(easting_arr.min()  / 10.0) * 10.0)
        origin_n = float(np.floor(northing_arr.min() / 10.0) * 10.0)
        col_arr = np.rint((easting_arr  - origin_e) / 10.0).astype(np.int64)
        row_from_bottom = np.rint((northing_arr - origin_n) / 10.0).astype(np.int64)
        width  = int(col_arr.max()) + 1
        height = int(row_from_bottom.max()) + 1

        grid = np.zeros((height, width), dtype=np.uint8)
        rows = height - 1 - row_from_bottom   # row 0 = northernmost
        # Last write wins on the rare event of two coords sharing a 10m cell;
        # take the max so a real detection isn't overwritten by a zero.
        np.maximum.at(grid, (rows, col_arr), prob_arr)
    else:
        pid_col = scores_df["point_id"].to_list()
        xi_arr = np.array([int(p.split("_")[1]) for p in pid_col], dtype=np.int32)
        yi_arr = np.array([int(p.split("_")[2]) for p in pid_col], dtype=np.int32)
        prob_arr = scores_df["prob_tam"].to_numpy().astype(np.uint8)
        width = int(xi_arr.max()) + 1
        height = int(yi_arr.max()) + 1
        origin_e, origin_n, utm_epsg = _grid_origin_from_tile(tile_id, loc_geom)
        crs = CRS.from_epsg(utm_epsg)
        grid = np.zeros((height, width), dtype=np.uint8)
        rows = height - 1 - yi_arr
        np.maximum.at(grid, (rows, xi_arr.astype(np.int64)), prob_arr)

    transform = Affine.translation(origin_e, origin_n + height * 10) * Affine.scale(10, -10)
    return grid, transform, crs


def _load_coords(
    coords_paths: list[Path],
    want_pids: "pl.Series | None" = None,
) -> pl.DataFrame:
    """Load deduped (point_id, lon, lat) from one or more pixel parquets.

    Pixel parquets hold one row per (point_id, date) observation, so a tile's
    point_ids are repeated ~50× per chunk. We dedup per-chunk (lazily, via
    scan_parquet group_by-first) before concat, collapsing that ~50× before any
    cross-chunk work. When want_pids (the scored pixels) is given we restrict to
    them via a lazy semi-join — a Polars hash join, NOT an is_in() against a
    multi-million-element Python set (which is pathologically slow at tile scale).
    """
    want_lf = (
        pl.LazyFrame({"point_id": want_pids}).unique("point_id")
        if want_pids is not None else None
    )
    frames: list[pl.DataFrame] = []
    for p in coords_paths:
        # group_by-first is cheaper than unique(): lon/lat are constant per
        # point_id, so we just need one row each.
        lf = (pl.scan_parquet(p)
              .select(["point_id", "lon", "lat"])
              .group_by("point_id").first())
        if want_lf is not None:
            lf = lf.join(want_lf, on="point_id", how="semi")
        frames.append(lf.collect())
    out = pl.concat(frames) if len(frames) > 1 else frames[0]
    # A point_id can appear in >1 chunk only at chunk seams; collapse those too.
    return out.unique("point_id") if len(frames) > 1 else out


def _coords_cache_path(scores_path: Path) -> Path:
    """Sidecar path holding deduped (point_id, lon, lat) for a scored tile."""
    return scores_path.with_suffix(".coords.parquet")


def rasterize_tile_to_pmtiles(
    scores_path: Path,
    coords_paths: Path | list[Path],
    tile_id: str,
    out_pmtiles: Path,
    coords_cache: Path | None = None,
) -> None:
    """Rasterise a tile's scores parquet to a PMTiles archive.

    Parameters
    ----------
    scores_path:
        Path to ``.scores.parquet`` with columns ``point_id`` and ``prob_tam``.
    coords_paths:
        Pixel parquet(s) containing ``point_id``, ``lon``, ``lat``. For a chunked
        tile this MUST be every chunk file: each chunk uses its own xi/yi origin,
        so pixels are placed by their real lon/lat and the coords for every
        scored pixel must be available. A single Path is accepted for the
        unchunked (flat) case.
    tile_id:
        Tile identifier string (e.g. ``"54LWH"``).
    out_pmtiles:
        Output path for the ``.pmtiles`` file.
    coords_cache:
        Optional sidecar parquet of deduped (point_id, lon, lat). Extracting
        coords means scanning every chunk's ~100M-row per-observation table
        (minutes for a big tile); the cache holds the ~1M-row result so repeat
        rasterisation reads MBs not GBs. Read if present; otherwise built from
        coords_paths and written here. Defaults to <scores>.coords.parquet.
    """
    from pmtiles.writer import Writer
    from pmtiles.tile import zxy_to_tileid, TileType, Compression

    if isinstance(coords_paths, (str, Path)):
        coords_paths = [Path(coords_paths)]

    scores_df = pl.read_parquet(scores_path, columns=["point_id", "prob_tam"])

    if coords_cache is None:
        coords_cache = _coords_cache_path(scores_path)
    if coords_cache.exists():
        logger.info("Loading coords cache %s", coords_cache.name)
        lon_lat_df = pl.read_parquet(coords_cache, columns=["point_id", "lon", "lat"])
    else:
        lon_lat_df = _load_coords(coords_paths, want_pids=scores_df["point_id"])
        # Atomic write so a crash mid-write can't leave a truncated cache.
        _tmp = coords_cache.with_suffix(".tmp.parquet")
        coords_cache.parent.mkdir(parents=True, exist_ok=True)
        lon_lat_df.write_parquet(_tmp)
        _tmp.rename(coords_cache)
        logger.info("Wrote coords cache %s (%d pixels)", coords_cache.name, len(lon_lat_df))

    grid, transform, crs = _scores_to_grid_from_xi_yi(scores_df, lon_lat_df, tile_id)
    merc_grid, merc_transform = warp_to_mercator(grid, transform, crs)

    header = {
        "tile_type": TileType.PNG,
        "tile_compression": Compression.NONE,
    }
    metadata: dict = {"name": tile_id}

    tiles = list(iter_tiles(merc_grid, merc_transform, zoom_min=8, zoom_max=16))
    if not tiles:
        logger.warning(
            "Tile %s has no above-zero scores — skipping PMTiles output (%s)",
            tile_id, out_pmtiles,
        )
        return

    out_pmtiles.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pmtiles, "wb") as f:
        writer = Writer(f)
        for z, x, y, png_bytes in tiles:
            tileid = zxy_to_tileid(z, x, y)
            writer.write_tile(tileid, png_bytes)
        writer.finalize(header, metadata)


def rasterize_scores_dir_to_pmtiles(
    scores_dir: Path,
    tile_id: str,
    out_pmtiles: Path,
    loc_geom=None,
    zoom_min: int = 8,
    zoom_max: int = 16,
) -> None:
    """Rasterise scores from a directory of per-year score parquets to PMTiles.

    Uses xi/yi from point_id + tile geometry (no coordinate parquet needed).
    All per-year parquets for tile_id in scores_dir are merged (max prob_tam).
    loc_geom: optional Shapely geometry for the scored location; used to derive
    the pixel grid origin via tile intersection (same as the scoring pipeline).
    """
    from pmtiles.writer import Writer
    from pmtiles.tile import zxy_to_tileid, TileType, Compression

    parquets = list(scores_dir.rglob(f"{tile_id}.scores.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No scores parquet for {tile_id} under {scores_dir}")

    frames = [pl.read_parquet(p, columns=["point_id", "prob_tam"]) for p in parquets]
    scores_df = (
        pl.concat(frames)
        .group_by("point_id")
        .agg(pl.col("prob_tam").max())
    )

    grid, transform, crs = _scores_to_grid_from_xi_yi(scores_df, None, tile_id, loc_geom=loc_geom)
    merc_grid, merc_transform = warp_to_mercator(grid, transform, crs)

    header = {"tile_type": TileType.PNG, "tile_compression": Compression.NONE}
    metadata: dict = {"name": tile_id}

    tiles = list(iter_tiles(merc_grid, merc_transform, zoom_min=zoom_min, zoom_max=zoom_max))
    if not tiles:
        logger.warning(
            "Tile %s has no above-zero scores — skipping PMTiles output (%s)",
            tile_id, out_pmtiles,
        )
        return

    out_pmtiles.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pmtiles, "wb") as f:
        writer = Writer(f)
        for z, x, y, png_bytes in tiles:
            tileid = zxy_to_tileid(z, x, y)
            writer.write_tile(tileid, png_bytes)
        writer.finalize(header, metadata)
