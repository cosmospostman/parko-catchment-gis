"""ui/s1_tile_builder.py — Build a .bin grid cache from S1 parquet data.

Called by the Deno tile server as a subprocess:
    python ui/s1_tile_builder.py <location> <band> <date> <out_path>

Reads data/pixels/<location>/<year>/<tile>.parquet files, filters to the
requested band (vh|vv) and date, normalises values to [0,1] using fixed
per-band percentiles, and writes a .bin compatible with tile_renderer.ts.

.bin format (little-endian):
    f64 lonMin, f64 latMax, f64 resX, f64 resY, u32 width, u32 height  (40 bytes)
    [u32 key, f32 val] * N  (8 bytes each, sorted by key ascending)
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIXELS_DIR = PROJECT_ROOT / "data" / "pixels"

# Normalisation ranges (linear sigma-naught, p2–p98 across beaudesert)
BAND_NORM = {
    "vh": (0.002, 0.040),
    "vv": (0.014, 0.151),
}

HEADER_BYTES = 40


def build_bin(location: str, band: str, date: str, out_path: Path) -> None:
    loc_dir = PIXELS_DIR / location
    if not loc_dir.exists():
        raise FileNotFoundError(f"Location not found: {loc_dir}")

    frames: list[pd.DataFrame] = []
    for year_dir in sorted(loc_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for pq in sorted(year_dir.glob("*.parquet")):
            if "coords" in pq.name:
                continue
            df = pd.read_parquet(pq, columns=["point_id", "lon", "lat", "date", "source", band])
            df = df[(df["source"] == "S1") & (df["date"].astype(str) == date)]
            if len(df) > 0:
                frames.append(df)

    if not frames:
        raise ValueError(f"No S1 data found for location={location} band={band} date={date}")

    df = pd.concat(frames, ignore_index=True)

    # Extract xi, yi from point_id ("px_XXXX_YYYY")
    parts = df["point_id"].str.split("_", expand=True)
    df["xi"] = parts[1].astype(int)
    df["yi"] = parts[2].astype(int)

    xi_max = int(df["xi"].max())
    yi_max = int(df["yi"].max())
    width  = xi_max + 1
    height = yi_max + 1

    # Derive grid origin from NW corner pixel (xi=0, yi=yi_max)
    nw = df[(df["xi"] == 0) & (df["yi"] == yi_max)]
    ne = df[(df["xi"] == xi_max) & (df["yi"] == yi_max)]
    sw = df[(df["xi"] == 0) & (df["yi"] == 0)]

    if len(nw) == 0 or len(ne) == 0 or len(sw) == 0:
        # Fallback: derive from overall extents
        lon_min = float(df["lon"].min())
        lat_max = float(df["lat"].max())
        res_x   = (float(df["lon"].max()) - lon_min) / max(xi_max, 1)
        res_y   = (lat_max - float(df["lat"].min())) / max(yi_max, 1)
    else:
        lon_nw = float(nw["lon"].iloc[0])
        lat_nw = float(nw["lat"].iloc[0])
        lon_ne = float(ne["lon"].iloc[0])
        lat_sw = float(sw["lat"].iloc[0])
        lon_min = lon_nw
        lat_max = lat_nw
        res_x   = (lon_ne - lon_nw) / max(xi_max, 1)
        res_y   = (lat_nw - lat_sw) / max(yi_max, 1)

    # Normalise band values to [0, 1]
    lo, hi = BAND_NORM.get(band, (float(df[band].min()), float(df[band].max())))
    vals_raw = df[band].values.astype(np.float32)
    vals_norm = np.clip((vals_raw - lo) / (hi - lo), 0.0, 1.0)

    # key = (yi_max - yi) * width + xi  (yi=0 → southernmost, flip to north-first)
    yi_flipped = yi_max - df["yi"].values
    keys = (yi_flipped * width + df["xi"].values).astype(np.uint32)

    # Sort by key
    order = np.argsort(keys)
    keys_sorted = keys[order]
    vals_sorted = vals_norm[order]

    n = len(keys_sorted)
    buf = bytearray(HEADER_BYTES + n * 8)
    struct.pack_into("<ddddII", buf, 0, lon_min, lat_max, res_x, res_y, width, height)
    off = HEADER_BYTES
    for k, v in zip(keys_sorted.tolist(), vals_sorted.tolist()):
        struct.pack_into("<If", buf, off, k, v)
        off += 8

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(buf))
    print(f"Built {out_path}  ({n} pixels, {width}x{height})", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: s1_tile_builder.py <location> <band> <date> <out_path>", file=sys.stderr)
        sys.exit(1)
    _, location, band, date, out_path_str = sys.argv
    build_bin(location, band, date, Path(out_path_str))
