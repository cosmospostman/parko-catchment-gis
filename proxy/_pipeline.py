"""proxy/_pipeline.py — Pure pipeline logic importable without FastAPI.

Shared by proxy/server.py (VM), utils/tile_pipeline.py (local path), and
tests (workstation).  No FastAPI, uvicorn, or httpx imports here.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import struct
import time
from pathlib import Path

from typing import Iterator

logger = logging.getLogger("proxy.pipeline")


# ---------------------------------------------------------------------------
# Frame encode / decode
# ---------------------------------------------------------------------------

def write_frame(frame_type: int, payload: bytes) -> bytes:
    """Encode one frame: [TYPE 1B][LENGTH 4B big-endian][PAYLOAD]."""
    return struct.pack(">BI", frame_type, len(payload)) + payload


def read_frame(stream: io.RawIOBase) -> tuple[int, bytes] | None:
    """Read one frame from a binary stream.  Returns None on clean EOF."""
    header = stream.read(5)
    if not header:
        return None
    if len(header) < 5:
        raise EOFError(f"truncated frame header ({len(header)} bytes)")
    frame_type, length = struct.unpack(">BI", header)
    payload = b""
    while len(payload) < length:
        chunk = stream.read(length - len(payload))
        if not chunk:
            raise EOFError(f"truncated frame payload: got {len(payload)}, expected {length}")
        payload += chunk
    return frame_type, payload


class StreamBuffer(io.RawIOBase):
    """Wrap a bytes iterator into a RawIOBase for read_frame (no httpx dependency)."""

    def __init__(self, iter_bytes: Iterator[bytes]) -> None:
        self._iter = iter_bytes
        self._buf = b""

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            return b"".join(self._iter)
        while len(self._buf) < n:
            try:
                self._buf += next(self._iter)
            except StopIteration:
                break
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def readable(self) -> bool:
        return True


def progress_frame(strip_idx: int, stage: str, t: float) -> bytes:
    payload = json.dumps({"strip": strip_idx, "stage": stage, "t": round(t, 2)}).encode()
    return write_frame(0x01, payload)


# ---------------------------------------------------------------------------
# merge_scenes — DuckDB N-way sort of per-scene parquets + optional S1 parquet
# ---------------------------------------------------------------------------

def merge_scenes(
    scene_paths: list[Path],
    s1_path: Path | None,
    out_path: Path,
) -> None:
    """DuckDB N-way sort-merge of S2 per-scene parquets (+ optional S1) into out_path.

    Output is sorted by (northing, date), ZSTD compressed, dictionary encoded
    on point_id.  Uses COMBINED_PIXEL_SCHEMA — no live-file schema derivation.
    """
    import duckdb
    import pyarrow.parquet as pq
    from utils.parquet_utils import COMBINED_PIXEL_SCHEMA

    out_names = COMBINED_PIXEL_SCHEMA.names

    def _select(file_cols: set[str], source_override: str | None) -> str:
        parts = []
        for name in out_names:
            if name == "source" and source_override is not None:
                parts.append(f"'{source_override}' AS source")
            elif name in file_cols:
                parts.append(f'"{name}"')
            else:
                parts.append(f'NULL AS "{name}"')
        return ", ".join(parts)

    col_list = ", ".join(f'"{c}"' for c in out_names)

    union_parts = []
    for sp in scene_paths:
        s2_cols = set(pq.ParquetFile(sp).schema_arrow.names)
        sel = _select(s2_cols, "S2")
        union_parts.append(f"SELECT {sel} FROM read_parquet('{sp}')")

    if s1_path and s1_path.exists():
        s1_cols = set(pq.ParquetFile(s1_path).schema_arrow.names)
        sel = _select(s1_cols, None)
        union_parts.append(f"SELECT {sel} FROM read_parquet('{s1_path}')")

    union_sql = "\nUNION ALL\n".join(union_parts)
    dst = str(out_path)
    tmp_dir = str(out_path.parent)

    n_threads = max(1, (os.cpu_count() or 4) // 2)
    mem_gb = int(os.environ.get("PROXY_MERGE_MEM_GB", "2"))

    sql = f"""
        COPY (
            SELECT {col_list}
            FROM ({union_sql}) t
            ORDER BY regexp_extract(point_id, '_([0-9]+)$', 1)::INTEGER, date
        ) TO '{dst}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 5000000
        )
    """
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{mem_gb}GB'")
        con.execute(f"SET temp_directory = '{tmp_dir}'")
        con.execute(f"SET threads = {n_threads}")
        con.execute(sql)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Strip geometry helpers
# ---------------------------------------------------------------------------

def read_cog_transform(href: str) -> tuple[str, float]:
    """Read the UTM CRS EPSG string and top-edge northing from a COG href.

    Opens only the GeoTIFF header via a range request (~2 kB).  No pixel data
    is read.  Returns (crs_epsg_str, y_top) where y_top is src.transform.f —
    the UTM northing of pixel row 0 (the top edge of the raster).

    Used by compute_strips to snap strip boundaries to the COG block grid.
    """
    import rasterio
    with rasterio.open(href) as src:
        crs_str = src.crs.to_epsg()
        epsg = f"EPSG:{crs_str}" if crs_str else src.crs.to_string()
        y_top = src.transform.f
    return epsg, y_top


def compute_strips(
    bbox_wgs84: list[float],
    strip_height_px: int,
    polygon_geometry,
    cog_utm_crs: str | None = None,
    cog_y_top: float | None = None,
) -> list[dict]:
    """Divide bbox into horizontal strips of strip_height_px pixels.

    When cog_utm_crs and cog_y_top are supplied (read from a representative COG
    via read_cog_transform), strip boundaries are snapped to the COG's internal
    1024-pixel block grid in UTM northing space.  This ensures every strip's
    row_off into the COG is a multiple of strip_height_px, eliminating block
    over-fetch waste.

    Without those parameters the function falls back to the geographic-space
    approximation (preserved for callers that have no COG reference available).

    Returns list of dicts with keys: strip_idx, bbox, points.
    """
    from utils.pixel_collector import make_pixel_grid, _utm_crs_for_bbox
    from shapely.geometry import MultiPoint
    from pyproj import Transformer

    utm_crs = cog_utm_crs or _utm_crs_for_bbox(bbox_wgs84)
    all_points = make_pixel_grid(bbox_wgs84, utm_crs=utm_crs)

    if polygon_geometry is not None:
        mp = MultiPoint([(lon, lat) for _, lon, lat in all_points])
        all_points = [
            pt for pt, contained in zip(all_points, [polygon_geometry.contains(p) for p in mp.geoms])
            if contained
        ]

    if not all_points:
        return []

    if cog_utm_crs is not None and cog_y_top is not None:
        return _compute_strips_utm(all_points, strip_height_px, utm_crs, cog_y_top)
    else:
        return _compute_strips_geographic(all_points, strip_height_px)


def _compute_strips_utm(
    all_points: list[tuple[str, float, float]],
    strip_height_px: int,
    utm_crs: str,
    cog_y_top: float,
) -> list[dict]:
    """Split points into strips whose boundaries fall on COG block-row edges.

    COG block k (0-indexed from the top) spans UTM northings:
        [cog_y_top - (k+1)*block_m,  cog_y_top - k*block_m)

    Strip boundaries are therefore at cog_y_top - k*block_m for integer k ≥ 0.
    Because make_pixel_grid snaps to the 10 m UTM grid and S2 COG pixels are
    exactly 10 m, these boundaries align with rasterio window row_off = k *
    strip_height_px — zero block over-fetch.

    Points are sorted by UTM northing (ascending) and assigned to the block
    whose half-open interval [lower, upper) contains their northing.
    """
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    block_m = strip_height_px * 10.0  # metres per strip at 10 m/pixel

    # Annotate each point with its UTM northing (y coordinate).
    pts_utm: list[tuple[str, float, float, float]] = []
    for pid, lon, lat in all_points:
        _, y = to_utm.transform(lon, lat)
        pts_utm.append((pid, lon, lat, y))

    pts_utm.sort(key=lambda p: p[3])
    y_min = pts_utm[0][3]
    y_max = pts_utm[-1][3]

    # Find the index of the COG block that contains y_min.
    # Block k lower bound = cog_y_top - (k+1)*block_m
    # cog_y_top - (k+1)*block_m  ≤  y_min  →  k ≥ (cog_y_top - y_min)/block_m - 1
    # First block that contains y_min: k = ceil((cog_y_top - y_min) / block_m) - 1
    k_first = math.ceil((cog_y_top - y_min) / block_m) - 1
    # Lower bound of the first strip (northing increases upward, strips go bottom→top).
    first_lower = cog_y_top - (k_first + 1) * block_m

    strips = []
    strip_idx = 0
    current_lower = first_lower
    while current_lower <= y_max + 1e-3:  # +epsilon to capture the topmost point
        current_upper = current_lower + block_m
        strip_pts_utm = [p for p in pts_utm if current_lower <= p[3] < current_upper]
        if strip_pts_utm:
            strip_points = [(pid, lon, lat) for pid, lon, lat, _ in strip_pts_utm]
            p_lons = [p[1] for p in strip_points]
            p_lats = [p[2] for p in strip_points]
            strip_bbox = [min(p_lons), min(p_lats), max(p_lons), max(p_lats)]
            strips.append({"strip_idx": strip_idx, "bbox": strip_bbox, "points": strip_points})
            strip_idx += 1
        current_lower = current_upper

    return strips


def _compute_strips_geographic(
    all_points: list[tuple[str, float, float]],
    strip_height_px: int,
) -> list[dict]:
    """Fallback: divide by latitude, converting strip_height_px to degrees.

    This approximation is preserved for callers without a COG reference.
    Strip boundaries are not guaranteed to align with COG block rows.
    """
    all_points = sorted(all_points, key=lambda p: p[2])
    lats = [p[2] for p in all_points]
    lat_min = min(lats)
    lat_max = max(lats)

    deg_per_px = 10.0 / (111_320 * math.cos(math.radians((lat_min + lat_max) / 2)))
    strip_height_deg = strip_height_px * deg_per_px

    strips = []
    strip_idx = 0
    strip_lat_min = lat_min
    while strip_lat_min < lat_max:
        strip_lat_max = strip_lat_min + strip_height_deg
        strip_points = [p for p in all_points if strip_lat_min <= p[2] < strip_lat_max]
        if strip_points:
            p_lons = [p[1] for p in strip_points]
            p_lats = [p[2] for p in strip_points]
            strip_bbox = [min(p_lons), min(p_lats), max(p_lons), max(p_lats)]
            strips.append({"strip_idx": strip_idx, "bbox": strip_bbox, "points": strip_points})
            strip_idx += 1
        strip_lat_min = strip_lat_max

    return strips


# ---------------------------------------------------------------------------
# run_tile_pipeline — shared core, used by server.py and tile_pipeline.py
# ---------------------------------------------------------------------------

def run_tile_pipeline(
    tile_id: str,
    year: int,
    polygon_geometry,
    tmp: Path,
    cloud_max: int = 20,
    apply_nbar: bool = True,
    strip_height_px: int = 1024,
    max_concurrent: int = 32,
    n_workers: int | None = None,
    resume_from_strip: int = 0,
    items=None,
    calibration_out: Path | None = None,
) -> Iterator[tuple[int, Path]]:
    """Core tile pipeline: STAC search → strips → per-strip collect+S1+merge.

    Yields (strip_idx, sorted_strip_parquet_path) for each completed strip.
    The caller is responsible for consuming or moving the parquet before the
    next iteration — it lives inside *tmp* and may be overwritten.

    All utils/ imports are lazy so this module stays importable without the
    full project stack (e.g. on a VM that only has proxy/ installed).

    Parameters
    ----------
    tile_id:
        MGRS tile ID, e.g. "55HBU".  Used only for logging.
    year:
        Calendar year; fetch window is {year}-01-01 / {year}-12-31.
    polygon_geometry:
        Shapely geometry defining the area of interest.  Points outside it
        are excluded from the pixel grid.
    tmp:
        Scratch directory managed by the caller.  Scene parquets and S1
        shards land here and are cleaned up after each strip is yielded.
    items:
        Optional pre-fetched STAC item list.  When supplied the STAC search
        is skipped.  Used by the training pipeline to share one search result
        across multiple regions on the same tile.
    calibration_out:
        Optional path for NBAR calibration output, forwarded to collect().
    """
    import shutil
    from utils.stac import search_sentinel2
    from utils.pixel_collector import collect, STAC_ENDPOINT, S2_COLLECTION
    from utils.s1_collector import collect_s1_for_tile

    bbox_wgs84 = list(polygon_geometry.bounds)
    start_date = f"{year}-01-01"
    end_date   = f"{year}-12-31"

    if items is None:
        logger.info("[tile %s %d] STAC search ...", tile_id, year)
        items = search_sentinel2(
            bbox=bbox_wgs84,
            start=start_date,
            end=end_date,
            cloud_cover_max=cloud_max,
            endpoint=STAC_ENDPOINT,
            collection=S2_COLLECTION,
        )
    logger.info("[tile %s %d] %d STAC items", tile_id, year, len(items))

    if not items:
        return

    # Read one COG header (~2 kB range request) to snap strip boundaries to
    # the COG's 1024-px block grid, eliminating block over-fetch waste.
    cog_utm_crs: str | None = None
    cog_y_top: float | None = None
    for item in items:
        href_obj = item.assets.get("red") or item.assets.get("B04")
        if href_obj is not None:
            try:
                cog_utm_crs, cog_y_top = read_cog_transform(href_obj.href)
                logger.info("[tile %s %d] COG origin: crs=%s y_top=%.1f",
                            tile_id, year, cog_utm_crs, cog_y_top)
            except Exception as exc:
                logger.warning("[tile %s %d] Could not read COG transform (%s) — using geographic fallback",
                               tile_id, year, exc)
            break

    strips = compute_strips(
        bbox_wgs84, strip_height_px, polygon_geometry,
        cog_utm_crs=cog_utm_crs, cog_y_top=cog_y_top,
    )
    logger.info("[tile %s %d] %d strips of %d px", tile_id, year, len(strips), strip_height_px)

    if not strips:
        return

    for strip in strips:
        strip_idx  = strip["strip_idx"]
        strip_bbox = strip["bbox"]
        strip_pts  = strip["points"]

        if strip_idx < resume_from_strip:
            logger.info("[tile %s %d] [strip %04d] skipping (resume_from_strip=%d)",
                        tile_id, year, strip_idx, resume_from_strip)
            continue

        scene_dir = tmp / f"strip_{strip_idx:04d}_scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)
        strip_cache = scene_dir / "cache"

        logger.info("[tile %s %d] [strip %04d] collect %d pts, %d items ...",
                    tile_id, year, strip_idx, len(strip_pts), len(items))

        scene_paths = [
            path for _, path in list(collect(
                bbox_wgs84=strip_bbox,
                start=start_date,
                end=end_date,
                out_dir=scene_dir,
                cloud_max=cloud_max,
                apply_nbar=apply_nbar,
                max_concurrent=max_concurrent,
                items=items,
                geometry=polygon_geometry,
                n_workers=n_workers,
                per_scene=True,
                cache_dir=strip_cache,
                calibration_out=calibration_out,
            ))
        ]

        if not scene_paths:
            logger.info("[tile %s %d] [strip %04d] no scene data — skipping", tile_id, year, strip_idx)
            shutil.rmtree(scene_dir, ignore_errors=True)
            continue

        s1_path = collect_s1_for_tile(
            s2_path=None,
            bbox_wgs84=strip_bbox,
            start=start_date,
            end=end_date,
            out_path=scene_dir / "s1_strip.parquet",
            cache_dir=strip_cache,
            max_concurrent=max_concurrent,
            points=strip_pts,
        )

        strip_out = tmp / f"strip_{strip_idx:04d}_sorted.parquet"
        merge_scenes(scene_paths, s1_path, strip_out)

        shutil.rmtree(scene_dir, ignore_errors=True)

        logger.info("[tile %s %d] [strip %04d] ready → %s", tile_id, year, strip_idx, strip_out.name)
        yield strip_idx, strip_out
