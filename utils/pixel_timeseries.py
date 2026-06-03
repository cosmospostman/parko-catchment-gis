"""utils/pixel_timeseries.py — Aggregate spectral index timeseries for a bbox.

For each date present in the bbox pixel data, computes mean + p25 + p75 of
NDVI, MAVI, and VH/VV (cross-pol ratio in dB) across all pixels in the bbox.

Usage::

    python utils/pixel_timeseries.py \\
        --root /mnt/external/chunkstore \\
        --year 2025 \\
        --tile 54LWH \\
        --bbox 141.493,−15.860,141.494,−15.859

Output: JSON to stdout with schema:
    {
      "year": 2025,
      "tile": "54LWH",
      "series": [
        {
          "date": "2025-01-04",
          "ndvi": 0.42, "ndvi_p25": 0.31, "ndvi_p75": 0.55,
          "mavi": 0.18, "mavi_p25": 0.10, "mavi_p75": 0.26,
          "vh_vv": -3.1, "vh_vv_p25": -4.2, "vh_vv_p75": -2.0
        },
        ...
      ]
    }

Null is emitted for any index that cannot be computed (e.g. missing bands).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.pixel_reader import ChunkIndex


def _safe_div(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.full(len(num), np.nan, dtype="float32")
    valid = denom != 0
    out[valid] = num[valid] / denom[valid]
    return out


def _percentile(arr: np.ndarray, q: float) -> float | None:
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return None
    return float(np.percentile(finite, q))


def _mean(arr: np.ndarray) -> float | None:
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return None
    return float(np.mean(finite))


def compute_timeseries(
    root: Path,
    year: int,
    tile_id: str,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
) -> list[dict]:
    chunk_idx = ChunkIndex(root, year, tile_id)
    tbl = chunk_idx.query_bbox(lon_min, lat_min, lon_max, lat_max)

    if tbl.num_rows == 0:
        return []

    # Cast bands to float32 numpy arrays.
    def col(name: str) -> np.ndarray:
        return tbl.column(name).to_numpy(zero_copy_only=False).astype("float32")

    b04 = col("B04")
    b08 = col("B08")
    b11 = col("B11")
    vh  = col("vh")
    vv  = col("vv")

    # Derived indices — NaN where invalid.
    ndvi = _safe_div(b08 - b04, b08 + b04)
    mavi = _safe_div(b08 - b04, b08 + b04 + b11)

    # VH/VV in dB: 10*log10(vh) - 10*log10(vv). Both must be > 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        vh_db = np.where(vh > 0, 10.0 * np.log10(vh.astype("float64")), np.nan).astype("float32")
        vv_db = np.where(vv > 0, 10.0 * np.log10(vv.astype("float64")), np.nan).astype("float32")
    vh_vv = vh_db - vv_db

    # Group by date — use pyarrow to get unique sorted dates, then filter.
    date_col = tbl.column("date")
    unique_dates = sorted(set(date_col.to_pylist()))

    series = []
    for d in unique_dates:
        mask = np.array(pc.equal(date_col, d).to_pylist())
        row: dict[str, object] = {"date": str(d)}
        for name, signal in [("ndvi", ndvi), ("mavi", mavi), ("vh_vv", vh_vv)]:
            g = signal[mask]
            row[name]          = _mean(g)
            row[f"{name}_p25"] = _percentile(g, 25)
            row[f"{name}_p75"] = _percentile(g, 75)
        series.append(row)

    return series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",  default="/mnt/external/chunkstore")
    ap.add_argument("--year",  type=int, required=True)
    ap.add_argument("--tile",  required=True)
    ap.add_argument("--bbox",  required=True,
                    help="lon_min,lat_min,lon_max,lat_max")
    args = ap.parse_args()

    parts = [float(x) for x in args.bbox.split(",")]
    if len(parts) != 4:
        print("ERROR: --bbox must have 4 comma-separated values", file=sys.stderr)
        sys.exit(1)
    lon_min, lat_min, lon_max, lat_max = parts

    series = compute_timeseries(
        Path(args.root), args.year, args.tile,
        lon_min, lat_min, lon_max, lat_max,
    )

    print(json.dumps({"year": args.year, "tile": args.tile, "series": series}))


if __name__ == "__main__":
    main()
