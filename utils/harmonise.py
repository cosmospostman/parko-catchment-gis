"""utils/harmonise.py — Apply inter-tile radiometric harmonisation to an existing pixel parquet.

Runs calibrate() on the parquet, then rewrites it in-place with corrected band
values and recomputed spectral indices.  A .pre-harmonise backup is kept until
the rewrite is verified, then deleted.

Usage
-----
    python -m utils.harmonise --location quaids
    python -m utils.harmonise --parquet data/pixels/quaids/quaids.parquet --cal-out data/calibration/quaids.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from signals._shared import _WRITE_OPTS, _optimise_schema
from utils.tile_harmonisation import calibrate, load_corrections

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CORRECT_BANDS = ["B04", "B05", "B07", "B08", "B11"]


def _make_corr_df(corrections: dict) -> "pl.DataFrame":
    import polars as pl
    return pl.DataFrame(
        [(t, b, y, s) for (t, b, y), s in corrections.items()],
        schema={"tile_id": pl.String, "band": pl.String, "year": pl.Int32, "scale": pl.Float32},
        orient="row",
    )


def _apply_corrections(tbl: pa.Table, corr_df: "pl.DataFrame") -> pa.Table:
    import polars as pl

    chunk = pl.from_arrow(tbl).with_columns([
        pl.col("item_id").str.split("_").list.get(1).alias("tile_id"),
        pl.col("date").dt.year().cast(pl.Int32).alias("year"),
    ])

    bands_present = [b for b in _CORRECT_BANDS if b in chunk.columns]
    for band in bands_present:
        band_corr = corr_df.filter(pl.col("band") == band).select(["tile_id", "year", "scale"])
        chunk = (
            chunk
            .join(band_corr, on=["tile_id", "year"], how="left")
            .with_columns(
                pl.when(pl.col("scale").is_not_null())
                  .then(pl.col(band) * pl.col("scale"))
                  .otherwise(pl.col(band))
                  .alias(band)
            )
            .drop("scale")
        )

    chunk = chunk.drop(["tile_id", "year"]).with_columns([
        ((pl.col("B08") - pl.col("B04")) / (pl.col("B08") + pl.col("B04"))).alias("NDVI"),
        ((pl.col("B03") - pl.col("B08")) / (pl.col("B03") + pl.col("B08"))).alias("NDWI"),
        (2.5 * (pl.col("B08") - pl.col("B04")) /
         (pl.col("B08") + 6.0 * pl.col("B04") - 7.5 * pl.col("B02") + 1.0)).alias("EVI"),
    ])

    return chunk.to_arrow()


def harmonise(parquet_path: Path, cal_out: Path) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    src_rows = pq.ParquetFile(parquet_path).metadata.num_rows

    print(f"Calibrating {parquet_path.name} ...")
    calibrate(parquet_path, cal_out)

    corrections = load_corrections(cal_out)
    if not corrections:
        print("No corrections computed (single tile or no overlap). Nothing to do.")
        return

    corr_df = _make_corr_df(corrections)

    pf = pq.ParquetFile(parquet_path)
    n_rg = pf.metadata.num_row_groups
    print(f"Rewriting {n_rg} row groups, {src_rows:,} rows ...")

    backup = parquet_path.with_suffix(".pre-harmonise.parquet")
    tmp = parquet_path.with_suffix(".harmonising.parquet")
    tmp.unlink(missing_ok=True)

    parquet_path.rename(backup)
    print(f"Backup → {backup.name}")

    writer: pq.ParquetWriter | None = None
    written = 0
    try:
        pf = pq.ParquetFile(backup)
        for rg_idx in range(n_rg):
            if rg_idx % max(1, n_rg // 10) == 0:
                print(f"  row group {rg_idx + 1}/{n_rg} ({100 * rg_idx // n_rg}%)", flush=True)
            tbl = pf.read_row_group(rg_idx)
            tbl = _apply_corrections(tbl, corr_df)
            tbl = _optimise_schema(tbl)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp), tbl.schema, **_WRITE_OPTS)
            writer.write_table(tbl)
            written += len(tbl)
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        tmp.unlink(missing_ok=True)
        backup.rename(parquet_path)
        print("ERROR — original restored.", file=sys.stderr)
        raise
    finally:
        if writer is not None:
            writer.close()

    written_check = pq.ParquetFile(tmp).metadata.num_rows
    if written_check != src_rows:
        tmp.unlink(missing_ok=True)
        backup.rename(parquet_path)
        raise RuntimeError(
            f"Row count mismatch: wrote {written_check}, expected {src_rows} — original restored."
        )

    tmp.rename(parquet_path)
    backup.unlink()
    print(f"Done. {written:,} rows written → {parquet_path}")
    print(f"Calibration table → {cal_out}")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Apply inter-tile harmonisation to a pixel parquet.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--location", help="Location id (e.g. quaids)")
    group.add_argument("--parquet", type=Path, help="Direct path to pixel parquet")
    parser.add_argument("--cal-out", type=Path, default=None,
                        help="Correction table output path (default: data/calibration/<id>.parquet)")
    args = parser.parse_args()

    if args.location:
        from utils.location import get
        loc = get(args.location)
        tile_paths = loc.parquet_tile_paths()
        if not tile_paths:
            raise FileNotFoundError(f"No tile parquets found for {args.location}")
        parquet_path = tile_paths[max(tile_paths)][0]
        cal_out = args.cal_out or (_PROJECT_ROOT / "data" / "calibration" / f"{args.location}.parquet")
    else:
        parquet_path = args.parquet
        cal_out = args.cal_out or (_PROJECT_ROOT / "data" / "calibration" / f"{parquet_path.stem}.parquet")

    cal_out.parent.mkdir(parents=True, exist_ok=True)
    harmonise(parquet_path, cal_out)


if __name__ == "__main__":
    _main()
