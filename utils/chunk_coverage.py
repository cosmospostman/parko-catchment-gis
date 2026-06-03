"""utils/chunk_coverage.py — Emit a JSON manifest of chunk parquet extents.

Prints a JSON array to stdout:
    [{"year": 2025, "tile": "54LWH", "chunk_row": 5, "chunk_col": 10,
      "lon_min": ..., "lon_max": ..., "lat_min": ..., "lat_max": ...,
      "ring": [[lon, lat], ...5 points...]}, ...]

When the parquet file-level metadata contains cog_utm_crs / cog_y_top /
cog_x_left (written by the pipeline since the metadata-embedding fix),
chunk boundaries are computed exactly in UTM space:
    x_left  = cog_x_left + chunk_col * 10240
    y_lower = cog_y_top  - (chunk_row + 1) * 10240

For older files without that metadata, the ring is omitted and the server
falls back to a simple lon_min/max, lat_min/max bbox rectangle.

One entry per chunk file.

Usage:
    python utils/chunk_coverage.py [--root /mnt/external/mitchell]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

_CHUNK_RE = re.compile(r"_r(\d{2})_c(\d{2})$")
_BLOCK = 10240.0  # 1024 px × 10 m


def _exact_ring(
    cog_utm_crs: str,
    cog_y_top: float,
    cog_x_left: float,
    chunk_row: int,
    chunk_col: int,
) -> list[list[float]]:
    """Compute the 5-point WGS84 ring for a chunk using exact UTM boundaries."""
    from pyproj import Transformer
    to_wgs = Transformer.from_crs(cog_utm_crs, "EPSG:4326", always_xy=True)
    e0 = cog_x_left + chunk_col * _BLOCK
    e1 = e0 + _BLOCK
    n1 = cog_y_top - chunk_row * _BLOCK
    n0 = n1 - _BLOCK
    lons, lats = to_wgs.transform([e0, e1, e1, e0, e0], [n0, n0, n1, n1, n0])
    return [[round(lo, 7), round(la, 7)] for lo, la in zip(lons, lats)]


def build_manifest(root: Path) -> list[dict]:
    entries: list[dict] = []
    if not root.exists():
        return entries
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        for tile_dir in sorted(year_dir.iterdir()):
            if not tile_dir.is_dir():
                continue
            tile = tile_dir.name
            for path in sorted(tile_dir.glob("*.parquet")):
                m = _CHUNK_RE.search(path.stem)
                if not m:
                    continue
                chunk_row = int(m.group(1))
                chunk_col = int(m.group(2))
                try:
                    pf = pq.ParquetFile(path)
                    md = pf.metadata
                    schema_names = pf.schema_arrow.names
                    lon_i = schema_names.index("lon")
                    lat_i = schema_names.index("lat")
                    lon_mins, lon_maxs, lat_mins, lat_maxs = [], [], [], []
                    for i in range(md.num_row_groups):
                        rg = md.row_group(i)
                        lon_mins.append(rg.column(lon_i).statistics.min)
                        lon_maxs.append(rg.column(lon_i).statistics.max)
                        lat_mins.append(rg.column(lat_i).statistics.min)
                        lat_maxs.append(rg.column(lat_i).statistics.max)

                    entry: dict = {
                        "year": year,
                        "tile": tile,
                        "chunk_row": chunk_row,
                        "chunk_col": chunk_col,
                        "lon_min": min(lon_mins),
                        "lon_max": max(lon_maxs),
                        "lat_min": min(lat_mins),
                        "lat_max": max(lat_maxs),
                    }

                    # Use exact UTM boundaries when the pipeline embedded them.
                    file_meta = md.metadata or {}
                    cog_utm = file_meta.get(b"cog_utm_crs", b"").decode()
                    cog_y_top_s = file_meta.get(b"cog_y_top", b"").decode()
                    cog_x_left_s = file_meta.get(b"cog_x_left", b"").decode()
                    if cog_utm and cog_y_top_s and cog_x_left_s:
                        entry["ring"] = _exact_ring(
                            cog_utm,
                            float(cog_y_top_s),
                            float(cog_x_left_s),
                            chunk_row,
                            chunk_col,
                        )

                    entries.append(entry)
                except Exception as e:
                    print(f"Warning: skipping {path}: {e}", file=sys.stderr)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/external/mitchell")
    args = parser.parse_args()
    manifest = build_manifest(Path(args.root))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
