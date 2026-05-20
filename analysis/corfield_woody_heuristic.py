"""Run the woody VH heuristic on Corfield presence_1–6 for each year 2018–2023.

Prints a table: region × year → fraction of pixels classified as woody (prob_woody=1.0).

Heuristic: dry-season mean VH ≥ -18 dB → woody.
Dry season: DOY 121–304 (May–Oct).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parent.parent

_VH_WOODY_FLOOR_DB = -18.0
_DRY_DOY_MIN = 121
_DRY_DOY_MAX = 304

REGIONS = [f"corfield_presence_{i}" for i in range(1, 7)]
YEARS   = list(range(2018, 2024))


def load_tile(tile_id: str) -> pl.DataFrame:
    path = _REPO / "data" / "training" / "tiles" / f"{tile_id}.parquet"
    pf = pq.ParquetFile(path)
    cols = [c for c in ["point_id", "date", "source", "vh"]
            if c in pf.schema_arrow.names]
    chunks = [pl.from_arrow(pf.read_row_group(rg, columns=cols))
              for rg in range(pf.metadata.num_row_groups)]
    return pl.concat(chunks)


def woody_fraction(s1: pl.DataFrame) -> float:
    """Return fraction of pixels classified as woody for the given S1 observations."""
    if s1.is_empty():
        return float("nan")
    s1 = s1.with_columns(
        pl.col("date").cast(pl.Date).dt.ordinal_day().alias("doy")
    )
    vh_lin = s1["vh"].to_numpy().astype(np.float64)
    vh_db  = np.where(vh_lin > 0, 10.0 * np.log10(vh_lin), np.nan)
    s1 = s1.with_columns(pl.Series("_vh_db", vh_db))
    dry = s1.filter(
        pl.col("_vh_db").is_not_null() & pl.col("_vh_db").is_not_nan() &
        pl.col("doy").is_between(_DRY_DOY_MIN, _DRY_DOY_MAX)
    )
    if dry.is_empty():
        return float("nan")
    mean_vh = dry.group_by("point_id").agg(pl.col("_vh_db").mean())["_vh_db"].to_numpy()
    return float((mean_vh >= _VH_WOODY_FLOOR_DB).mean())


def main() -> None:
    idx = pl.read_parquet(_REPO / "data" / "training" / "index.parquet")

    # All Corfield presence regions are on the same tile
    tile_id = idx.filter(pl.col("region_id") == "corfield_presence_1")["tile_id"][0]
    print(f"Loading tile {tile_id} ...", file=sys.stderr)
    tile_df = load_tile(tile_id)

    # Filter to S1 rows for all presence regions
    prefix_set = tuple(r + "_" for r in REGIONS)
    s1_df = tile_df.filter(
        (pl.col("source") == "S1") &
        pl.col("point_id").map_elements(lambda p: p.startswith(prefix_set), return_dtype=pl.Boolean)
    ).with_columns([
        pl.col("date").cast(pl.Date).dt.year().alias("year"),
        pl.col("point_id").str.splitn("_", 3).struct.field("field_0").alias("region_id"),
    ])

    # Build results table
    rows = []
    for region in REGIONS:
        for year in YEARS:
            subset = s1_df.filter(
                (pl.col("region_id") == region) & (pl.col("year") == year)
            )
            frac = woody_fraction(subset)
            rows.append({"region": region, "year": year, "woody_frac": frac})

    # Print pivot table
    print("\nWoody fraction (dry-season VH ≥ -18 dB) — Corfield presence sites\n")
    header = f"  {'':30}" + "".join(f"  {y:>6}" for y in YEARS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for region in REGIONS:
        region_rows = {r["year"]: r["woody_frac"] for r in rows if r["region"] == region}
        vals = "".join(
            f"  {region_rows[y]:>6.2f}" if not np.isnan(region_rows.get(y, float("nan"))) else f"  {'nan':>6}"
            for y in YEARS
        )
        print(f"  {region:<30}{vals}")
    print()


if __name__ == "__main__":
    main()
