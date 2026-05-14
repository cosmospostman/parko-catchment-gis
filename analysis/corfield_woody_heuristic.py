"""Run the woody VH heuristic on Corfield presence_1–6 for each year 2018–2023.

Prints a table: region × year → fraction of pixels classified as woody (prob_woody=1.0).

Heuristic: dry-season mean VH ≥ -18 dB → woody.
Dry season: DOY 121–304 (May–Oct).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parent.parent

_VH_WOODY_FLOOR_DB = -18.0
_DRY_DOY_MIN = 121
_DRY_DOY_MAX = 304

REGIONS = [f"corfield_presence_{i}" for i in range(1, 7)]
YEARS   = list(range(2018, 2024))


def load_tile(tile_id: str) -> pd.DataFrame:
    path = _REPO / "data" / "training" / "tiles" / f"{tile_id}.parquet"
    pf = pq.ParquetFile(path)
    cols = [c for c in ["point_id", "date", "source", "vh"]
            if c in pf.schema_arrow.names]
    chunks = [pf.read_row_group(rg, columns=cols).to_pandas()
              for rg in range(pf.metadata.num_row_groups)]
    return pd.concat(chunks, ignore_index=True)


def woody_fraction(s1: pd.DataFrame) -> float:
    """Return fraction of pixels classified as woody for the given S1 observations."""
    if s1.empty:
        return float("nan")
    df = s1.copy()
    df["doy"] = pd.to_datetime(df["date"]).dt.day_of_year
    vh_lin = df["vh"].values.astype(np.float64)
    df["_vh_db"] = np.where(vh_lin > 0, 10.0 * np.log10(vh_lin), np.nan)
    dry = df[df["_vh_db"].notna() & df["doy"].between(_DRY_DOY_MIN, _DRY_DOY_MAX)]
    if dry.empty:
        return float("nan")
    mean_vh = dry.groupby("point_id")["_vh_db"].mean()
    return (mean_vh >= _VH_WOODY_FLOOR_DB).mean()


def main() -> None:
    idx = pd.read_parquet(_REPO / "data" / "training" / "index.parquet")

    # All Corfield presence regions are on the same tile
    tile_id = idx.loc[idx["region_id"] == "corfield_presence_1", "tile_id"].iloc[0]
    print(f"Loading tile {tile_id} ...", file=sys.stderr)
    tile_df = load_tile(tile_id)

    # Filter to S1 rows for all presence regions
    prefix_set = {r + "_" for r in REGIONS}
    s1_df = tile_df[
        (tile_df["source"] == "S1") &
        tile_df["point_id"].str.startswith(tuple(prefix_set))
    ].copy()
    s1_df["year"] = pd.to_datetime(s1_df["date"]).dt.year
    s1_df["region_id"] = s1_df["point_id"].str.rsplit("_", n=2).str[0]

    # Build results table
    rows = []
    for region in REGIONS:
        for year in YEARS:
            subset = s1_df[(s1_df["region_id"] == region) & (s1_df["year"] == year)]
            frac = woody_fraction(subset)
            rows.append({"region": region, "year": year, "woody_frac": frac})

    results = pd.DataFrame(rows)
    pivot = results.pivot(index="region", columns="year", values="woody_frac")
    pivot.columns.name = None
    pivot.index.name = None

    print("\nWoody fraction (dry-season VH ≥ -18 dB) — Corfield presence sites\n")
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}" if not np.isnan(x) else " nan"))
    print()


if __name__ == "__main__":
    main()
