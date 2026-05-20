"""Explore VH vs dry-season NDVI for Corfield presence pixels.

For each presence pixel (collapsed across years) compute:
  - mean dry-season VH (dB)
  - mean dry-season NDVI

Then report:
  1. Per-pixel scatter statistics (percentiles of each metric)
  2. Fraction of pixels that would be DROPPED by the current VH-only filter
  3. Fraction dropped by a combined VH+NDVI filter at several candidate thresholds
  4. Per-region breakdown so we can see which regions are most affected

Dry season: DOY 121–304 (May–Oct), matching the existing filter definition.
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parent.parent

_DRY_DOY_MIN = 121
_DRY_DOY_MAX = 304

REGIONS = [f"corfield_presence_{i}" for i in range(1, 7)]

# Candidate thresholds to sweep
VH_THRESHOLDS   = [-22.0, -20.0, -18.0, -16.0]   # dB — keep if ABOVE
NDVI_THRESHOLDS = [0.15, 0.20, 0.25, 0.30]        # keep if ABOVE


def load_tile(tile_id: str, cols: list[str]) -> pl.DataFrame:
    path = _REPO / "data" / "training" / "tiles" / f"{tile_id}.parquet"
    pf = pq.ParquetFile(path)
    avail = [c for c in cols if c in pf.schema_arrow.names]
    chunks = [pl.from_arrow(pf.read_row_group(rg, columns=avail))
              for rg in range(pf.metadata.num_row_groups)]
    return pl.concat(chunks)


def per_pixel_dry_season(df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per point_id with mean dry-season VH (dB) and NDVI."""
    df = df.with_columns(
        pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy")
    )
    dry = df.filter(pl.col("doy").is_between(_DRY_DOY_MIN, _DRY_DOY_MAX))

    s1 = dry.filter(pl.col("source") == "S1")
    vh_lin = s1["vh"].to_numpy().astype(np.float64)
    vh_db  = np.where(vh_lin > 0, 10.0 * np.log10(vh_lin), np.nan)
    s1 = s1.with_columns(pl.Series("_vh_db", vh_db))
    mean_vh = s1.group_by("point_id").agg(pl.col("_vh_db").mean().alias("mean_vh_db"))

    s2 = dry.filter(pl.col("source") == "S2")
    mean_ndvi = s2.group_by("point_id").agg(pl.col("NDVI").mean().alias("mean_ndvi"))

    return mean_vh.join(mean_ndvi, on="point_id", how="full", coalesce=True)


def drop_fraction(px: pl.DataFrame, vh_thresh: float, ndvi_thresh: float | None) -> float:
    """Fraction of pixels that would be DROPPED (i.e. excluded from presence).

    VH-only:   drop if mean_vh_db < vh_thresh
    VH+NDVI:   drop if mean_vh_db < vh_thresh AND mean_ndvi < ndvi_thresh
    """
    valid = px.drop_nulls(subset=["mean_vh_db"])
    if valid.is_empty():
        return float("nan")
    vh = valid["mean_vh_db"].to_numpy()
    if ndvi_thresh is None:
        return float((vh < vh_thresh).mean())
    valid2 = valid.drop_nulls(subset=["mean_ndvi"])
    if valid2.is_empty():
        return float("nan")
    vh2   = valid2["mean_vh_db"].to_numpy()
    ndvi2 = valid2["mean_ndvi"].to_numpy()
    return float(((vh2 < vh_thresh) & (ndvi2 < ndvi_thresh)).mean())


def main() -> None:
    idx = pl.read_parquet(_REPO / "data" / "training" / "index.parquet")
    tile_id = idx.filter(pl.col("region_id") == "corfield_presence_1")["tile_id"][0]
    print(f"Loading tile {tile_id} ...", file=sys.stderr)

    tile_df = load_tile(tile_id, ["point_id", "date", "source", "vh", "NDVI"])

    prefix_set = tuple(r + "_" for r in REGIONS)
    region_df = tile_df.filter(
        pl.col("point_id").map_elements(lambda p: p.startswith(prefix_set), return_dtype=pl.Boolean)
    ).with_columns(
        pl.col("point_id").str.splitn("_", 3).struct.field("field_0").alias("region_id")
    )

    # ---- 1. Per-pixel dry-season means (collapsed across years) ----
    px_all = per_pixel_dry_season(region_df).with_columns(
        pl.col("point_id").str.splitn("_", 3).struct.field("field_0").alias("region_id")
    )

    print("\n=== Dry-season signal distribution (all Corfield presence pixels) ===\n")
    for col, label in [("mean_vh_db", "VH (dB)"), ("mean_ndvi", "NDVI")]:
        vals = px_all[col].drop_nulls().to_numpy()
        pcts = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95])
        print(f"  {label}  n={len(vals)}")
        print(f"    p5={pcts[0]:.3f}  p10={pcts[1]:.3f}  p25={pcts[2]:.3f}  "
              f"p50={pcts[3]:.3f}  p75={pcts[4]:.3f}  p90={pcts[5]:.3f}  p95={pcts[6]:.3f}")
    print()

    # ---- 2. Current filter impact ----
    current_vh = -18.0
    frac_dropped_current = drop_fraction(px_all, current_vh, None)
    print(f"=== Current filter: VH-only @ {current_vh} dB ===")
    print(f"  Pixels dropped: {frac_dropped_current:.1%}\n")

    # ---- 3. Threshold sweep: VH-only vs VH+NDVI ----
    print("=== Drop-fraction sweep (rows=VH threshold, cols=NDVI threshold) ===")
    print("    Format: VH-only  |  VH+NDVI@0.15  VH+NDVI@0.20  VH+NDVI@0.25  VH+NDVI@0.30\n")
    header = f"{'VH thresh':>10}  {'VH-only':>8}  " + "  ".join(f"NDVI≥{t:.2f}" for t in NDVI_THRESHOLDS)
    print(header)
    print("-" * len(header))
    for vh_t in VH_THRESHOLDS:
        vh_only = drop_fraction(px_all, vh_t, None)
        ndvi_vals = [drop_fraction(px_all, vh_t, nd_t) for nd_t in NDVI_THRESHOLDS]
        row = f"{vh_t:>10.1f}  {vh_only:>8.1%}  " + "  ".join(f"{v:>10.1%}" for v in ndvi_vals)
        print(row)
    print()

    # ---- 4. Per-region breakdown at current and a candidate combined threshold ----
    candidate_vh   = -18.0
    candidate_ndvi = 0.20
    print(f"=== Per-region breakdown: current (VH@{current_vh}) vs combined (VH@{candidate_vh}+NDVI@{candidate_ndvi}) ===\n")
    print(f"  {'Region':<30}  {'n_px':>5}  {'VH-only':>8}  {'VH+NDVI':>8}  {'saved':>7}")
    print("  " + "-" * 65)
    for region in REGIONS:
        px_r = px_all.filter(pl.col("region_id") == region)
        n = px_r["mean_vh_db"].drop_nulls().len()
        d_vh   = drop_fraction(px_r, current_vh, None)
        d_comb = drop_fraction(px_r, candidate_vh, candidate_ndvi)
        saved  = d_vh - d_comb if not (np.isnan(d_vh) or np.isnan(d_comb)) else float("nan")
        print(f"  {region:<30}  {n:>5}  {d_vh:>8.1%}  {d_comb:>8.1%}  {saved:>+7.1%}")
    print()

    # ---- 5. Pixels that fail VH but pass NDVI (the ones we'd save) ----
    valid  = px_all.drop_nulls(subset=["mean_vh_db", "mean_ndvi"])
    rescue = valid.filter(
        (pl.col("mean_vh_db") < candidate_vh) & (pl.col("mean_ndvi") >= candidate_ndvi)
    )
    print(f"=== Pixels failing VH@{candidate_vh} but rescued by NDVI≥{candidate_ndvi} ===")
    print(f"  Count: {len(rescue)} / {len(valid)} ({len(rescue)/max(len(valid),1):.1%} of all pixels)\n")
    if not rescue.is_empty():
        print("  Distribution of rescued pixels:")
        for col, label in [("mean_vh_db", "VH (dB)"), ("mean_ndvi", "NDVI")]:
            vals = rescue[col].to_numpy()
            pcts = np.percentile(vals, [5, 25, 50, 75, 95])
            print(f"    {label}: p5={pcts[0]:.3f}  p25={pcts[1]:.3f}  "
                  f"p50={pcts[2]:.3f}  p75={pcts[3]:.3f}  p95={pcts[4]:.3f}")
    print()


if __name__ == "__main__":
    main()
