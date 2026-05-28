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


def _system_memory_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 8.0


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
    # Default: 75% of system RAM so DuckDB can sort large strips without OOM.
    # It will spill to temp_directory when the limit is hit, so setting it high
    # just avoids unnecessary spill — it does not prevent spill when needed.
    # Override with PROXY_MERGE_MEM_GB for constrained environments.
    _default_mem_gb = max(4, int(_system_memory_gb() * 0.75))
    mem_gb = int(os.environ.get("PROXY_MERGE_MEM_GB", str(_default_mem_gb)))

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

    Generates points strip-by-strip in UTM space so that only one strip's worth
    of points (~1M) is in memory at a time.  The full tile grid (~121M points)
    is never materialised, keeping peak RAM well under 1 GB regardless of tile size.

    When cog_utm_crs and cog_y_top are supplied the strip boundaries are snapped
    to the COG's 1024-px block grid (zero block over-fetch).  Without them the
    function falls back to a geographic-space approximation.

    Returns list of dicts with keys: strip_idx, bbox, points.
    """
    import numpy as np
    from utils.pixel_collector import _utm_crs_for_bbox
    from pyproj import Transformer

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    utm_crs = cog_utm_crs or _utm_crs_for_bbox(bbox_wgs84)

    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    r = 10.0  # S2 pixel spacing in metres
    x0, y0 = to_utm.transform(lon_min, lat_min)
    x1, y1 = to_utm.transform(lon_max, lat_max)

    # Snap to S2 pixel grid origin.
    x0_snap = math.floor(x0 / r) * r
    y0_snap = math.floor(y0 / r) * r

    xs = np.arange(x0_snap, x1, r)

    block_m = strip_height_px * r

    # Determine strip y-boundaries.
    if cog_utm_crs is not None and cog_y_top is not None:
        # Snap first strip lower bound to COG block grid.
        k_first = math.ceil((cog_y_top - y0_snap) / block_m) - 1
        first_lower = cog_y_top - (k_first + 1) * block_m
    else:
        first_lower = y0_snap

    # Enumerate strip y-ranges.
    strip_lowers = []
    current = first_lower
    y_top_snap = math.ceil((y1 - first_lower) / block_m) * block_m + first_lower
    while current < y_top_snap:
        strip_lowers.append(current)
        current += block_m

    strips = []
    strip_idx = 0

    # Shapely vectorised contains for polygon masking — prepare once.
    if polygon_geometry is not None:
        from shapely import contains_xy as _shp_contains_xy
        _use_vectorised = True
    else:
        _use_vectorised = False

    for lower in strip_lowers:
        upper = lower + block_m
        ys = np.arange(lower, upper, r)
        # Filter ys to those within the bbox.
        ys = ys[(ys >= y0_snap) & (ys < y1)]
        if len(ys) == 0:
            continue

        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        lons_arr, lats_arr = to_wgs.transform(xx.ravel(), yy.ravel())
        lons_arr = np.asarray(lons_arr)
        lats_arr = np.asarray(lats_arr)

        # Polygon mask — vectorised, no Python loop over points.
        if _use_vectorised:
            mask = _shp_contains_xy(polygon_geometry, lons_arr, lats_arr)
            lons_arr = lons_arr[mask]
            lats_arr = lats_arr[mask]
            # Recover (i, j) grid indices for the kept points to build point_ids.
            ii, jj = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)), indexing="ij")
            ii_flat = ii.ravel()[mask]
            jj_flat = jj.ravel()[mask]
        else:
            ii, jj = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)), indexing="ij")
            ii_flat = ii.ravel()
            jj_flat = jj.ravel()

        if len(lons_arr) == 0:
            continue

        strip_bbox = [
            float(lons_arr.min()), float(lats_arr.min()),
            float(lons_arr.max()), float(lats_arr.max()),
        ]
        # Store y_lower so points can be regenerated on demand via make_strip_points().
        strips.append({"strip_idx": strip_idx, "bbox": strip_bbox, "y_lower": lower})
        strip_idx += 1

    # Stash grid params on the list object so make_strip_points() can regenerate
    # points for any strip without re-running the full bbox computation.
    strips_meta = {
        "utm_crs": utm_crs, "xs": xs, "y0_snap": y0_snap, "y1": y1,
        "block_m": block_m, "r": r, "polygon_geometry": polygon_geometry,
        "first_lower": first_lower,
    }
    return strips, strips_meta


def make_strip_points(strip: dict, meta: dict) -> list[tuple[str, float, float]]:
    """Generate points for one strip on demand.  Call just before the strip is needed;
    discard (del) after use so only one strip's points are in memory at a time.
    """
    import numpy as np
    from pyproj import Transformer

    utm_crs = meta["utm_crs"]
    xs = meta["xs"]
    y0_snap = meta["y0_snap"]
    y1 = meta["y1"]
    block_m = meta["block_m"]
    r = meta["r"]
    polygon_geometry = meta["polygon_geometry"]
    first_lower = meta["first_lower"]

    lower = strip["y_lower"]
    upper = lower + block_m
    ys = np.arange(lower, upper, r)
    ys = ys[(ys >= y0_snap) & (ys < y1)]

    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    lons_arr, lats_arr = to_wgs.transform(xx.ravel(), yy.ravel())
    lons_arr = np.asarray(lons_arr)
    lats_arr = np.asarray(lats_arr)

    ii, jj = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)), indexing="ij")
    ii_flat = ii.ravel()
    jj_flat = jj.ravel()

    if polygon_geometry is not None:
        from shapely import contains_xy as _shp_contains_xy
        mask = _shp_contains_xy(polygon_geometry, lons_arr, lats_arr)
        lons_arr = lons_arr[mask]
        lats_arr = lats_arr[mask]
        ii_flat = ii_flat[mask]
        jj_flat = jj_flat[mask]

    if len(lons_arr) == 0:
        return []

    j_offset = round((lower - first_lower) / r)
    pids = [f"px_{int(i):04d}_{int(j + j_offset):04d}" for i, j in zip(ii_flat, jj_flat)]
    return list(zip(pids, lons_arr.tolist(), lats_arr.tolist()))


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

    strips, strips_meta = compute_strips(
        bbox_wgs84, strip_height_px, polygon_geometry,
        cog_utm_crs=cog_utm_crs, cog_y_top=cog_y_top,
    )
    logger.info("[tile %s %d] %d strips of %d px", tile_id, year, len(strips), strip_height_px)

    if not strips:
        return

    # Prefetch pipeline: run each strip's network fetch in a background thread
    # so the next strip's .npz cache is warm before the current strip's
    # CPU-bound extract → S1 → merge_scenes finishes.  A depth-1 look-ahead
    # keeps at most one strip prefetching at a time (bounded cache growth).
    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _fetch_strip(strip: dict, pts: list) -> dict:
        """Run the fetch-only phase for one strip (S2 + S1); returns the strip dict."""
        s_dir = tmp / f"strip_{strip['strip_idx']:04d}_scenes"
        s_dir.mkdir(parents=True, exist_ok=True)
        s_cache = s_dir / "cache"
        logger.info("[tile %s %d] [strip %04d] prefetch %d items ...",
                    tile_id, year, strip["strip_idx"], len(items))
        list(collect(
            bbox_wgs84=strip["bbox"],
            start=start_date,
            end=end_date,
            out_dir=s_dir,
            cloud_max=cloud_max,
            apply_nbar=apply_nbar,
            max_concurrent=max_concurrent,
            items=items,
            geometry=polygon_geometry,
            n_workers=n_workers,
            per_scene=True,
            cache_dir=s_cache,
            phases={"fetch"},
            calibration_out=calibration_out,
        ))
        collect_s1_for_tile(
            s2_path=None,
            bbox_wgs84=strip["bbox"],
            start=start_date,
            end=end_date,
            out_path=s_dir / "s1_strip.parquet",
            cache_dir=s_cache,
            max_concurrent=max_concurrent,
            points=pts,
            phases={"fetch"},
        )
        return strip

    # Strips to actually process (after resume skip).
    active_strips = [s for s in strips if s["strip_idx"] >= resume_from_strip]

    for s in strips:
        if s["strip_idx"] < resume_from_strip:
            logger.info("[tile %s %d] [strip %04d] skipping (resume_from_strip=%d)",
                        tile_id, year, s["strip_idx"], resume_from_strip)

    if not active_strips:
        return

    from collections import deque as _deque

    with _TPE(max_workers=2) as prefetch_pool:
        # Depth-2 prefetch: submit fetches for the first two strips immediately so
        # the network is busy while the CPU works.  At steady state strip i+2 is
        # fetching while strip i is being extracted/merged and strip i+1 is ready.
        pts_queue: _deque = _deque()
        fetch_futs: _deque = _deque()

        for k in range(min(2, len(active_strips))):
            pts = make_strip_points(active_strips[k], strips_meta)
            pts_queue.append(pts)
            fetch_futs.append(prefetch_pool.submit(_fetch_strip, active_strips[k], pts))

        for i, strip in enumerate(active_strips):
            strip_idx = strip["strip_idx"]
            strip_bbox = strip["bbox"]
            strip_pts  = pts_queue.popleft()

            # Wait for this strip's fetch to complete.
            fetch_futs.popleft().result()
            logger.info("[tile %s %d] [strip %04d] fetch done; starting extract (%d pts, %d items)",
                        tile_id, year, strip_idx, len(strip_pts), len(items))

            # Submit fetch for strip i+2 immediately (before CPU-bound extract/merge)
            # so the network stays busy throughout extraction.
            next_k = i + 2
            if next_k < len(active_strips):
                pts = make_strip_points(active_strips[next_k], strips_meta)
                pts_queue.append(pts)
                fetch_futs.append(prefetch_pool.submit(_fetch_strip, active_strips[next_k], pts))

            scene_dir   = tmp / f"strip_{strip_idx:04d}_scenes"
            strip_cache = scene_dir / "cache"

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
                    phases={"extract"},
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
                phases={"extract"},
            )

            strip_out = tmp / f"strip_{strip_idx:04d}_sorted.parquet"
            merge_scenes(scene_paths, s1_path, strip_out)

            shutil.rmtree(scene_dir, ignore_errors=True)

            logger.info("[tile %s %d] [strip %04d] ready → %s", tile_id, year, strip_idx, strip_out.name)
            yield strip_idx, strip_out
