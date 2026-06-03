"""utils/chunk_coverage.py — Emit a JSON manifest of chunk parquet extents.

Prints a JSON array to stdout:
    [{"year": 2025, "tile": "54LWH", "lon_min": ..., "lon_max": ..., "lat_min": ..., "lat_max": ...}, ...]

One entry per chunk file (not per row-group) — the chunk-level envelope is
sufficient for bbox overlap checks in the server.

Usage:
    python utils/chunk_coverage.py [--root /mnt/external/mitchell]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


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
                    entries.append({
                        "year": year,
                        "tile": tile,
                        "lon_min": min(lon_mins),
                        "lon_max": max(lon_maxs),
                        "lat_min": min(lat_mins),
                        "lat_max": max(lat_maxs),
                    })
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
