"""proxy/_pipeline.py — Pure pipeline logic importable without FastAPI.

Shared by proxy/server.py (VM) and tests (workstation).
No FastAPI, uvicorn, or httpx imports here.
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

def compute_strips(
    bbox_wgs84: list[float],
    strip_height_px: int,
    polygon_geometry,
) -> list[dict]:
    """Divide bbox into horizontal strips of strip_height_px pixels.

    Returns list of dicts with keys: strip_idx, bbox, points.
    """
    from utils.pixel_collector import make_pixel_grid, _utm_crs_for_bbox
    from shapely.geometry import MultiPoint

    utm_crs = _utm_crs_for_bbox(bbox_wgs84)
    all_points = make_pixel_grid(bbox_wgs84, utm_crs=utm_crs)

    if polygon_geometry is not None:
        mp = MultiPoint([(lon, lat) for _, lon, lat in all_points])
        all_points = [
            pt for pt, contained in zip(all_points, [polygon_geometry.contains(p) for p in mp.geoms])
            if contained
        ]

    if not all_points:
        return []

    all_points.sort(key=lambda p: p[2])
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
