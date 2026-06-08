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

import os

import numpy as np
import polars as pl

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_CHUNKSTORE_DEFAULT = os.environ.get("CHUNKSTORE_DIR", "/mnt/external/chunkstore")

from analysis.constants import add_spectral_indices
from signals.base import Signal
from tam.core.dataset import lin_to_db
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

    df = pl.from_arrow(tbl)
    # Raw chunkstore rows store S2 with source=null (only S1 is explicitly
    # tagged); training backfills null -> "S2" before gating (see
    # utils/training_collector.py) — mirror that here so the inspector shows
    # the same pixels the model trains/scores on, not a superset that includes
    # cloud/shadow-contaminated observations the model never sees.
    if "source" in df.columns:
        df = df.with_columns(pl.col("source").fill_null("S2"))
    good = Signal.quality_mask(df)

    # --- Optical indices: computed via the canonical `add_spectral_indices`
    # primitive (the same one `prepare_s2_frame`/training/scoring use) on rows
    # passing the training quality gate — eliminates the hand-rolled NDVI/MAVI
    # transcription this function used to carry (see
    # docs/UNIFIED-PIXEL-PIPELINE.md "fourth (UI-side) reimplementation").
    opt = df.filter(good)
    if len(opt) > 0:
        opt = add_spectral_indices(opt)
        ndvi = opt["NDVI"].to_numpy().astype("float32")
        mavi = opt["MAVI"].to_numpy().astype("float32")
        opt_dates = opt["date"].to_numpy()
    else:
        ndvi = mavi = np.array([], dtype="float32")
        opt_dates = np.array([])

    # --- Radar (VH/VV): SCL purity does not apply to S1; use all rows with
    # valid radar bands, and convert via the canonical `lin_to_db` (the same
    # helper `prepare_s1_frame` uses for training/scoring) rather than a
    # hand-rolled 10*log10.
    if "vh" in df.columns and "vv" in df.columns:
        rad = df.filter(pl.col("vh").is_not_null() & pl.col("vv").is_not_null())
    else:
        rad = df.head(0)
    if len(rad) > 0:
        vh_lin = rad["vh"].cast(pl.Float32).to_numpy()
        vv_lin = rad["vv"].cast(pl.Float32).to_numpy()
        vh_vv = (lin_to_db(vh_lin) - lin_to_db(vv_lin)).astype("float32")
        rad_dates = rad["date"].to_numpy()
    else:
        vh_vv = np.array([], dtype="float32")
        rad_dates = np.array([])

    unique_dates = sorted(set(opt_dates.tolist()) | set(rad_dates.tolist()))

    series = []
    for d in unique_dates:
        row: dict[str, object] = {"date": str(d)}
        opt_g  = ndvi[opt_dates == d]
        mavi_g = mavi[opt_dates == d]
        rad_g  = vh_vv[rad_dates == d]
        for name, g in [("ndvi", opt_g), ("mavi", mavi_g), ("vh_vv", rad_g)]:
            row[name]          = _mean(g)
            row[f"{name}_p25"] = _percentile(g, 25)
            row[f"{name}_p75"] = _percentile(g, 75)
        series.append(row)

    return series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",  default=_CHUNKSTORE_DEFAULT)
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
