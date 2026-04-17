"""utils/tile_harmonisation.py — Inter-tile radiometric harmonisation.

Two public functions:

  calibrate(parquet_path, out_path, bands)
      Scan a pixel-sorted parquet for same-pixel same-day multi-tile
      observations, compute per-(tile, band, year) scale factors relative to
      a reference tile, write to a small correction table parquet, and return
      it as a pandas DataFrame.

  load_corrections(calibration_path)
      Load a correction table as a lookup dict keyed by (tile_id, band, year).
      Returns None if the file does not exist.

CLI
---
    python -m utils.tile_harmonisation --location kowanyama

Writes to data/calibration/kowanyama.parquet and prints the table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SCALE_MIN = 0.85
_SCALE_MAX = 1.15
_DEFAULT_BANDS = ["B04", "B05", "B07", "B08", "B11"]


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def calibrate(
    parquet_path: Path,
    out_path: Path,
    bands: list[str] = _DEFAULT_BANDS,
) -> pd.DataFrame:
    """Compute per-(tile, band, year) scale factors from overlap observations.

    Reads the parquet row-group by row-group.  For each row group, finds all
    (point_id, date_only) pairs that appear under more than one tile_id, pivots
    to wide form, and accumulates per-row band ratios (non-reference / reference).

    Reference tile: the tile with the most total observations across all years.
    Tie-breaker: alphabetically first tile id.

    Returns and writes a DataFrame with columns:
        tile_id, band, year, scale_factor
    where ``scale_factor`` multiplies the tile's band value to bring it onto the
    reference tile's radiometry.  Scale factors are clamped to [0.85, 1.15].

    Parameters
    ----------
    parquet_path:
        Path to a pixel-sorted parquet containing ``point_id``, ``date``,
        ``tile_id``, and the requested band columns.
    out_path:
        Destination for the correction table parquet.  Parent directory is
        created if needed.
    bands:
        Band columns to calibrate.  Default: B04, B05, B07, B08, B11.

    Returns
    -------
    pandas DataFrame with columns [tile_id, band, year, scale_factor].
    """
    import concurrent.futures
    import pyarrow.parquet as pq

    LOAD_COLS = ["point_id", "date", "item_id"] + bands

    # Benchmarked on this dataset: 4 workers gives ~4x speedup on both IO and
    # CPU phases; beyond 6 there is no further gain.  Use separate ParquetFile
    # handles per worker to avoid lock contention inside pyarrow.
    N_WORKERS = 6

    pf = pq.ParquetFile(parquet_path)
    n_rg = pf.metadata.num_row_groups
    print(f"  [calibrate] {parquet_path.name}: {n_rg} row groups, "
          f"{pf.metadata.num_rows:,} rows")

    # Shared ParquetFile handles — one per worker, indexed by rg_idx % N_WORKERS.
    _pfs = [pq.ParquetFile(parquet_path) for _ in range(N_WORKERS)]

    # ------------------------------------------------------------------ #
    # Phase 1: tile-count scan (tile_id column only, ~10 MB total).       #
    # ------------------------------------------------------------------ #
    print("  [calibrate] counting observations per tile ...")
    tile_counts: dict[str, int] = {}

    def _read_tile_col(rg_idx: int) -> pl.DataFrame:
        return pl.from_arrow(
            _pfs[rg_idx % N_WORKERS].read_row_groups([rg_idx], columns=["item_id"])
        ).with_columns(
            pl.col("item_id").str.split("_").list.get(1).alias("tile_id")
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for chunk in ex.map(_read_tile_col, range(n_rg)):
            for tile, count in chunk["tile_id"].value_counts().iter_rows():
                tile_counts[tile] = tile_counts.get(tile, 0) + count

    if len(tile_counts) < 2:
        print("  [calibrate] only one tile found — no corrections needed.")
        empty = pd.DataFrame(columns=["tile_id", "band", "year", "scale_factor"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(out_path, index=False)
        return empty

    ref_tile    = sorted(tile_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    other_tiles = sorted(t for t in tile_counts if t != ref_tile)
    relevant_tiles = [ref_tile] + other_tiles
    print(f"  [calibrate] reference tile: {ref_tile}  "
          f"(counts: {dict(sorted(tile_counts.items()))})")

    # ------------------------------------------------------------------ #
    # Phase 2: per-row-group median aggregation.                          #
    # Each worker returns a small DataFrame of per-(tile,band,year)       #
    # median ratios + counts.  No shared state, no temp file.             #
    # Final step: weighted median across row-group medians.               #
    # ------------------------------------------------------------------ #

    # RgResult: list of (tile_id, band, year, median_ratio, n_pairs)
    RgResult = list  # type alias for clarity

    def _process_rg(rg_idx: int) -> RgResult:
        chunk = pl.from_arrow(
            _pfs[rg_idx % N_WORKERS].read_row_groups([rg_idx], columns=LOAD_COLS)
        ).with_columns(
            pl.col("item_id").str.split("_").list.get(1).alias("tile_id")
        )

        chunk = chunk.filter(pl.col("tile_id").is_in(relevant_tiles))
        if chunk.is_empty():
            return []

        chunk = chunk.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.date().alias("date_only"),
        ])

        # Find (point_id, date_only) pairs seen under more than one tile
        pair_counts = (
            chunk
            .group_by(["point_id", "date_only"])
            .agg(pl.col("tile_id").n_unique().alias("n_tiles"))
            .filter(pl.col("n_tiles") > 1)
        )
        if pair_counts.is_empty():
            return []

        overlap = chunk.join(pair_counts.select(["point_id", "date_only"]),
                             on=["point_id", "date_only"], how="inner")

        results: RgResult = []
        for band in bands:
            pivot = (
                overlap
                .select(["point_id", "date_only", "year", "tile_id", band])
                .pivot(on="tile_id", index=["point_id", "date_only", "year"],
                       values=band, aggregate_function="mean")
            )
            if ref_tile not in pivot.columns:
                continue
            ref_col = pl.col(ref_tile)
            for other in other_tiles:
                if other not in pivot.columns:
                    continue
                valid = (
                    pivot
                    .filter(ref_col.is_not_null() & pl.col(other).is_not_null()
                            & (ref_col > 0))
                    .with_columns((pl.col(other) / ref_col).alias("ratio"))
                )
                if valid.is_empty():
                    continue
                # Aggregate to per-(year) median + count within this row group
                agg = (
                    valid
                    .group_by("year")
                    .agg([
                        pl.col("ratio").median().alias("median_ratio"),
                        pl.col("ratio").count().alias("n_pairs"),
                    ])
                )
                for year, median_ratio, n_pairs in agg.iter_rows():
                    results.append((other, band, int(year), median_ratio, int(n_pairs)))

        return results

    # Collect per-row-group results — workers return plain lists, no locking needed
    all_results: list[tuple] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_process_rg, i): i for i in range(n_rg)}
        done = 0
        report_every = max(1, n_rg // 10)
        for fut in concurrent.futures.as_completed(futs):
            all_results.extend(fut.result())  # re-raises worker exceptions
            done += 1
            if done % report_every == 0 or done == n_rg:
                pct = 100 * done // n_rg
                print(f"  [calibrate] {done}/{n_rg} row groups ({pct}%)", flush=True)

    if not all_results:
        print("  [calibrate] no overlap observations found — empty correction table.")
        empty = pd.DataFrame(columns=["tile_id", "band", "year", "scale_factor"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(out_path, index=False)
        return empty

    # Weighted median across row-group medians:
    # group by (tile, band, year), sort medians, find the median_ratio whose
    # cumulative weight crosses 50% of total weight.
    from collections import defaultdict
    groups: dict[tuple, list[tuple[float, int]]] = defaultdict(list)
    total_pairs = 0
    for tile, band, year, median_ratio, n_pairs in all_results:
        groups[(tile, band, year)].append((median_ratio, n_pairs))
        total_pairs += n_pairs

    n_overlap_pairs = total_pairs // (len(bands) * max(len(other_tiles), 1))
    print(f"  [calibrate] ~{n_overlap_pairs:,} overlap pairs; computing weighted medians …")

    rows = []
    for (tile, band, year), entries in groups.items():
        entries.sort(key=lambda x: x[0])  # sort by median_ratio
        total_w = sum(w for _, w in entries)
        cumulative = 0
        median_ratio = entries[-1][0]
        for val, w in entries:
            cumulative += w
            if cumulative >= total_w / 2:
                median_ratio = val
                break
        if median_ratio <= 0:
            continue
        scale_factor = float(np.clip(1.0 / median_ratio, _SCALE_MIN, _SCALE_MAX))
        rows.append({"tile_id": tile, "band": band, "year": year,
                     "scale_factor": scale_factor})

    result = (
        pd.DataFrame(rows)
        .sort_values(["tile_id", "band", "year"])
        .reset_index(drop=True)
    )
    result["year"] = result["year"].astype(int)

    # Print summary
    print(f"\n  [calibrate] correction table ({len(result)} rows):")
    print(result.to_string(index=False))
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"  [calibrate] written → {out_path}")
    return result


# ---------------------------------------------------------------------------
# load_corrections
# ---------------------------------------------------------------------------

def load_corrections(
    calibration_path: Path,
) -> dict[tuple[str, str, int], float] | None:
    """Load correction table as a lookup dict keyed by (tile_id, band, year).

    Returns None if the file does not exist (correction disabled — graceful
    fallback so existing pipelines continue to work without re-calibrating).
    """
    if not calibration_path.exists():
        return None

    df = pd.read_parquet(calibration_path)
    if df.empty:
        return {}

    return {
        (row["tile_id"], row["band"], int(row["year"])): float(row["scale_factor"])
        for _, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute inter-tile radiometric correction table for a location."
    )
    parser.add_argument("--location", required=True,
                        help="Location id (e.g. kowanyama)")
    parser.add_argument("--bands", nargs="+", default=_DEFAULT_BANDS,
                        help="Band columns to calibrate")
    args = parser.parse_args()

    from utils.location import get
    loc = get(args.location)
    parquet_path = loc.parquet_path()
    out_path = _PROJECT_ROOT / "data" / "calibration" / f"{args.location}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    print(f"Calibrating {args.location} → {out_path}")
    calibrate(parquet_path, out_path, bands=args.bands)


if __name__ == "__main__":
    _main()
