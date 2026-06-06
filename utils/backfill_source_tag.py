"""One-off backfill: set source="S2" for null-source rows, dedup multi-orbit S2 rows,
and drop S2 rows with null band values (detector gaps)."""
import sys
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGIONS_DIR = PROJECT_ROOT / "data" / "training" / "tiles" / "regions"

_BAND_COLS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def backfill(regions_dir: Path = REGIONS_DIR, dry_run: bool = False) -> None:
    files = sorted(regions_dir.glob("*.parquet"))
    patched = 0
    for path in files:
        s = pq.read_schema(path)
        if "source" not in s.names:
            continue
        tbl = pq.read_table(path)
        src = tbl.column("source")
        df = pl.from_arrow(tbl)

        needs_tag   = pc.any(pc.is_null(src)).as_py()
        s2_view     = df.filter(pl.col("source").is_in(["S2"]) | pl.col("source").is_null())
        needs_dedup = s2_view.select(["point_id", "date"]).is_duplicated().any()
        present_bands = [b for b in _BAND_COLS if b in df.columns]
        null_band_count = (
            s2_view.filter(pl.any_horizontal([pl.col(b).is_null() for b in present_bands])).height
            if present_bands else 0
        )
        needs_strip = null_band_count > 0

        if not needs_tag and not needs_dedup and not needs_strip:
            continue

        parts = []
        null_src_count = pc.sum(pc.is_null(src).cast(pa.int32())).as_py()
        if null_src_count:
            parts.append(f"{null_src_count} null→S2")

        # tag nulls
        filled = pc.if_else(pc.is_null(src), "S2", src).cast(pa.large_utf8())
        tbl = tbl.set_column(s.get_field_index("source"), "source", filled)
        df = pl.from_arrow(tbl)

        # dedup multi-orbit S2
        n_before = len(df)
        s2 = (
            df.filter(pl.col("source") == "S2")
            .sort(["point_id", "date", "scl_purity", "item_id"], descending=[False, False, True, False])
            .unique(subset=["point_id", "date"], keep="first")
        )
        s1 = df.filter(pl.col("source") != "S2")
        df = pl.concat([s2, s1], how="diagonal")
        n_dedup = n_before - len(df)
        if n_dedup:
            parts.append(f"{n_dedup} multi-orbit dupes dropped")

        # drop S2 rows with null bands
        if present_bands:
            s2 = df.filter(pl.col("source") == "S2")
            null_mask = pl.any_horizontal([pl.col(b).is_null() for b in present_bands])
            n_null = s2.filter(null_mask).height
            if n_null:
                s2 = s2.filter(~null_mask)
                s1 = df.filter(pl.col("source") != "S2")
                df = pl.concat([s2, s1], how="diagonal")
                parts.append(f"{n_null} null-band rows dropped")

        print(f"  {path.name}: {', '.join(parts)}", end="" if dry_run else "\n")
        if dry_run:
            print(" (dry run)")
            continue

        out = df.to_arrow().cast(tbl.schema)
        tmp = path.with_suffix(".backfill_tmp.parquet")
        pq.write_table(out, tmp, compression="zstd")
        tmp.replace(path)
        patched += 1

    if not dry_run:
        print(f"Done — patched {patched} file(s).")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    backfill(dry_run=dry_run)
